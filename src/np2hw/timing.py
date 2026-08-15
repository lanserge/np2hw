"""Timing as a TRACED property: stage depth from the expression DAG.

Every pointwise datapath np2hw emits is a PExpr DAG whose nodes carry
exact value ranges -- so the logic each node becomes is arithmetic the
generator can do: an add is a carry chain as wide as its widest
argument, a clip is a compare and a mux, a gather is a mux tree as
deep as its table demands, a mul is a DSP block. Summing a stage's
worst path is therefore a generation-time computation, not a
synthesis-time discovery: the budget check runs in microseconds and
names the expression, where a place-and-route run takes half an hour
and names a net.

The device is a DRIVER, not a table of numbers. `Device` owns both
halves of the mapping -- the calibration constants AND the hardware
each operation becomes -- because the two cannot be separated: a
family with a narrower LUT does not merely have different delays, it
builds a mux tree of a different depth, and a family without a DSP
column does not multiply the same way at all. A bag of constants can
only misdescribe such a device; a subclass can override the method.

Nothing outside this file may contain a device constant. The emitters
ask the device for a shape and for nanoseconds, and adding a family is
subclassing `Device` -- not editing np2hw.

The same arithmetic drives automatic pipelining: `assign_stages()`
walks the DAG in dependency order and starts a new pipeline stage the
moment the accumulated path would exceed the budget, so the emitters
can place the registers mechanically. The streams are elastic, so the
added latency is free by construction; the values are untouched, so
the twin stays bit-exact by construction. The one thing a register
cannot fix is a single operation deeper than the clock -- that is the
floor, and it refuses with the operation named.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import PExpr, _clog2


def _bits(node: PExpr) -> int:
    lo, hi = int(node.lo), int(node.hi)
    span = max(abs(lo), abs(hi))
    width = max(1, span.bit_length())
    return width + (1 if lo < 0 else 0)


@dataclass
class Depth:
    """Worst path of one expression, in STRUCTURAL units only.

    Deliberately knows nothing about nanoseconds: a count of logic
    levels, carry bits, DSPs and memories is what the generator can
    derive on its own. Turning it into time is the device's job, and
    so is deciding which of two paths is worse -- that ordering
    depends on the ratio between a DSP and a LUT level, which is a
    property of the silicon, not of the expression.
    """
    levels: int = 0
    carry_bits: int = 0
    # HOW MANY separate carry chains, as distinct from how many bits of
    # carry. Starting a chain costs a fixed amount -- the lookup table
    # that forms its inputs, and the routed net out of it into whatever
    # comes next -- and that cost is paid once per arithmetic operation,
    # not once per bit. Four narrow adds in series are far slower than
    # one wide one, and without this term the model cannot say so.
    carries: int = 0
    dsps: int = 0
    mems: int = 0

    def __add__(self, other: "Depth") -> "Depth":
        return Depth(self.levels + other.levels,
                     self.carry_bits + other.carry_bits,
                     self.carries + other.carries,
                     self.dsps + other.dsps, self.mems + other.mems)


class Device:
    """A device family: its calibration AND how operations map to it.

    Subclass, set the constants, and override any method whose shape
    differs. Every constant is declared None here on purpose -- a
    calibration number has no sensible default, and a family that
    silently inherited another family's memory delay would be wrong
    in a way the model could not see.

    level_ns is an EFFECTIVE LUT level: logic plus its average routed
    net. overhead_ns covers clock-to-out, the final setup, and the
    first/last nets. Conservative by intent -- a pass here should
    survive routing, not merely hope to.
    """

    name = None
    level_ns = None            # one LUT level, including its routed net
    carry_ns_per_bit = None    # a carry chain, per bit
    dsp_ns = None              # a hard multiplier, in to out
    overhead_ns = None         # clock-to-out + setup + the first/last nets
    mem_ns = None              # a memory's clock-to-out (registered read)
    carry_chain_ns = None      # starting one carry chain, and leaving it
    lut_inputs = None          # how many inputs one lookup table has

    _REQUIRED = ("name", "level_ns", "carry_ns_per_bit", "dsp_ns",
                 "overhead_ns", "mem_ns", "carry_chain_ns", "lut_inputs")

    def __init__(self):
        missing = [f for f in self._REQUIRED if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"{type(self).__name__} leaves {', '.join(missing)} unset. "
                "A calibration constant has no default: measure it on the "
                "device (a sweep of known depths, read from the timing "
                "report) rather than inheriting another family's number.")

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"

    # -- structure to time ------------------------------------------------
    def ns(self, d: Depth) -> float:
        """What a structural depth costs on THIS device."""
        return (self.overhead_ns + d.levels * self.level_ns
                + d.carry_bits * self.carry_ns_per_bit
                + d.carries * self.carry_chain_ns
                + d.dsps * self.dsp_ns + d.mems * self.mem_ns)

    def worst(self, a: Depth, b: Depth) -> Depth:
        """The slower of two sibling paths, by this device's own ratios."""
        return a if self.ns(a) >= self.ns(b) else b

    # -- the shapes -------------------------------------------------------
    def mux_arity(self) -> int:
        """Widest m:1 mux one lookup table implements: m data inputs plus
        log2(m) select lines must fit. A 6-input LUT gives 4:1; a
        4-input LUT gives only 2:1, which is why the tree's depth is a
        device property and not a constant."""
        m = 2
        while (2 * m) + _clog2(2 * m) <= self.lut_inputs:
            m *= 2
        return m

    def mux_levels(self, size: int) -> int:
        """Select-tree depth for a size-entry read -- the law behind a
        gather's register array and a distributed table alike."""
        per = _clog2(self.mux_arity())
        return max(1, -(-_clog2(max(size, 2)) // per))

    def reduce_levels(self, terms: int) -> int:
        """Depth of a tree that reduces `terms` signals to one -- the OR
        over the bits a constant compare has to test, for instance. Set
        by how many inputs a lookup table has, like every tree here."""
        per = _clog2(self.lut_inputs)
        return max(1, -(-_clog2(max(terms, 2)) // per))

    def wire(self) -> Depth:
        """Costs nothing: a register output, a constant, a mask, a
        constant shift -- wiring into the stage that uses it."""
        return Depth()

    def add(self, bits: int) -> Depth:
        return Depth(levels=1, carry_bits=bits, carries=1)

    def compare(self, bits: int) -> Depth:
        return Depth(levels=1, carry_bits=bits, carries=1)

    def select(self) -> Depth:
        return Depth(levels=1)

    def absolute(self, bits: int) -> Depth:
        # a negate is a subtract, and the select is on top of it
        return Depth(levels=1, carry_bits=bits, carries=1)

    def clip(self, bits: int) -> Depth:
        """A compare against each bound and the select between them.

        The bounds are CONSTANTS, so neither comparison is a subtract:
        the synthesiser resolves a constant compare in lookup tables.
        But it is not free of width either -- testing whether a wide
        value has left its range is a reduction over the bits above
        that range, and that tree deepens as the value widens. So:
        the two bounds' reductions, which run in parallel, and the
        select between three values on top.
        """
        return Depth(levels=2 + self.reduce_levels(bits), carry_bits=0)

    def mul(self) -> Depth:
        return Depth(dsps=1)

    def mem(self) -> Depth:
        """A registered memory read: it OWNS its stage, so what it costs
        the stage that follows is the memory's clock-to-out, and nothing
        in front of it can share the cycle."""
        return Depth(mems=1)

    def gather(self, size: int) -> Depth:
        """A read from a register array: a select tree, whose depth the
        table's size and this device's LUT width decide together."""
        return Depth(levels=self.mux_levels(size))

    def adder_tree(self, terms: int, bits: int) -> Depth:
        """A balanced tree summing `terms` values: log2 levels, each one
        a carry chain."""
        lv = max(terms - 1, 0).bit_length()
        return Depth(levels=lv, carry_bits=lv * bits, carries=lv)

    def adder_chain(self, terms: int, bits: int, muxed=False) -> Depth:
        """A linear accumulation of `terms` values, optionally behind a
        position select -- what a tap chain becomes before its post ops."""
        n = max(terms - 1, 0)
        return Depth(levels=(1 if muxed else 0) + n, carry_bits=bits * n,
                     carries=n)

    def window(self, has_line_buffers: bool) -> Depth:
        """The read cone of a stencil's window: a registered memory read
        plus the edge selects. A CONSTANT in the line's width -- that is
        the whole point of reading line buffers through a register, and
        the reason this is not a select tree that deepens with the
        picture."""
        return Depth(mems=1 if has_line_buffers else 0, levels=2)


class Series7(Device):
    """Xilinx 7-series, speed grade -1.

    These are FITTED, not solved. Measured on xc7z020-1 at 6.734 ns,
    worst path per module, from placed designs:

      white balance   +0.252   1 level    2.590 ns   (a DSP and a clip)
      gamma           +0.942   6 levels   5.483 ns
      black level     +1.266   6 levels   5.148 ns
      line-buffer     +0.165   7 levels   5.965 ns   (address, not data)
      adaptive green  -0.815  12 levels   7.497 ns   (4 chained adds)

    carry_chain_ns was solved from the last of those. It over-predicts
    an older uncut gamma anchor -- 13.5 against 11.4 routed -- and the
    direction of that error is deliberate: a stage the model calls too
    slow costs pipeline registers, which the elastic streams make
    free, while a stage it calls fast enough and is not costs a build.

    Two lessons are recorded here rather than in a commit. Constant
    bounds are not subtracts, and modelling a clip as one made a DSP
    plus a clip look like two stages for the wrong reason. And the
    line-buffer ADDRESS, not the data, was the closest thing to
    critical in a working design -- the model still has no term for
    a control path, and 0.165 ns is not much of a margin.

    Resolving this properly needs a sweep of known depths.
    """
    name = "7series-1"
    level_ns = 0.62
    carry_ns_per_bit = 0.03
    dsp_ns = 3.2
    overhead_ns = 1.6
    mem_ns = 2.1
    # MEASURED, on the path that exposed the need for the term: a
    # 13-bit add in a chain costs about 1.6 ns end to end, of which the
    # level and the per-bit carry account for about 1.0. Solved from one
    # path, so still fitted rather than swept.
    carry_chain_ns = 0.5
    lut_inputs = 6


SERIES7 = Series7()

DEVICES = {"7series": SERIES7}

# The default is a CHOICE OF PROFILE, not a constant: it selects which
# calibration to read, and every number still comes from the device.
# One place owns it, so a build for other silicon changes one name.
DEFAULT_DEVICE = "7series"


def device(spec) -> Device:
    """Accept a registered name or a Device instance -- so a family that
    np2hw has never heard of needs no entry here, only a subclass."""
    if isinstance(spec, Device):
        return spec
    if spec is None:
        spec = DEFAULT_DEVICE
    try:
        return DEVICES[spec]
    except KeyError:
        raise ValueError(
            f"unknown device {spec!r}: known names are "
            f"{sorted(DEVICES)}, or pass a Device instance") from None


# -- np2hw's vocabularies, mapped onto the device's shapes ----------------
# These carry NO numbers. They translate an op name into a request the
# device answers, so every hardware shape has exactly one owner.

def node_cost(node, dev: Device) -> Depth:
    """The hardware shape of ONE PExpr node: what it adds to any path
    through it. Leaves cost nothing -- they are registers, constants, or
    the wires of a register array."""
    op = node.op
    if op in ("acc", "const", "param", "tap", "row", "col"):
        # registers, constants, window taps, and the position counters
        # every core already keeps: wires into the stage
        return dev.wire()
    if op in ("shr", "mask"):
        return dev.wire()
    if op == "gather":
        parent, _index = node.args
        return dev.gather(int(parent.shape[0]))
    if op == "rom":
        return dev.mem()
    if op in ("add", "sub"):
        return dev.add(_bits(node))
    if op == "mul":
        return dev.mul()
    if op == "clip":
        return dev.clip(_bits(node))
    if op == "lt":
        return dev.compare(_bits(node.args[0])
                           if isinstance(node.args[0], PExpr) else 0)
    if op == "abs":
        return dev.absolute(_bits(node))
    if op == "sel":
        return dev.select()
    raise ValueError(f"no depth rule for PExpr op {op!r} -- a new op "
                     "needs a hardware shape here before it ships")


def post_op_cost(op, bits: int, dev: Device) -> Depth:
    """The same question for the POST-CHAIN vocabulary the tap emitters
    speak (shr/trunc/clip/addp/mulc/mulp). One device, one set of
    shapes: a clip costs here exactly what it costs above."""
    kind = op[0]
    if kind in ("shr", "trunc"):
        return dev.wire()
    if kind == "clip":
        return dev.clip(bits)
    if kind == "addp":
        return dev.add(bits)
    if kind in ("mulc", "mulp"):
        return dev.mul()
    raise ValueError(f"no depth rule for post op {kind!r} -- a new op "
                     "needs a hardware shape here before it ships")


def expr_depth(node, dev: Device) -> Depth:
    """Worst-path structural depth of a PExpr DAG: node_cost summed
    along the worst path, sibling branches compared by the device's own
    nanosecond ratios."""
    if not isinstance(node, PExpr):
        return Depth()
    below = Depth()
    for a in node.args:
        if isinstance(a, PExpr):
            below = dev.worst(below, expr_depth(a, dev))
    return below + node_cost(node, dev)


# param and const leaves FLOAT: a register port is stable across a
# pixel's flight through the pipeline and a literal is wiring, so both
# feed any stage directly and never ride a delay line. acc, tap, row
# and col are PINNED to stage 0 -- they describe THIS pixel (its input
# word, its position), so a later stage must see them through the
# delay line, not as the counter's current value.
FLOATING = ("const", "param")


def assign_stages(roots, clk_ns: float, dev, label: str = "stage"):
    """Cut a PExpr DAG into pipeline stages that each fit the budget.

    Walks in dependency order with ONE memo shared across all roots
    (lanes sharing subexpressions must agree on where the registers
    go). A node lands in the latest stage any of its arguments lives
    in; when the accumulated path there would exceed the budget, the
    node starts the next stage instead -- its arguments arrive through
    the boundary registers, which is exactly the greedy cut the depth
    arithmetic recommends, and it falls naturally around the DSPs.

    Returns ({id(node): stage}, n_stages). Raises when one operation
    ALONE exceeds the budget: that is the floor -- a pipeline register
    cannot land inside one operation.
    """
    fam = device(dev)
    stage_of = {}
    depth_in = {}

    def visit(node):
        key = id(node)
        if key in stage_of:
            return
        args = [a for a in node.args if isinstance(a, PExpr)]
        for a in args:
            visit(a)
        cost = node_cost(node, fam)
        if fam.ns(cost) > clk_ns:
            raise ValueError(
                f"{label}: operation {node.op!r} alone estimates "
                f"{fam.ns(cost):.1f} ns against the {clk_ns:.1f} ns budget "
                f"on {fam.name} -- a pipeline register cannot land inside "
                "one operation. Slow the clock or restructure the model.")
        staged = [a for a in args if a.op not in FLOATING]
        s = max((stage_of[id(a)] for a in staged), default=0)
        if node.op == "rom":
            # A memory read is registered by construction: its address
            # is captured at the end of the previous stage and its data
            # appears in this one. So it ALWAYS starts a stage, budget
            # or no budget -- that is what makes it a block RAM instead
            # of a mux tree as deep as the table.
            stage_of[key] = s + 1
            depth_in[key] = cost
            return
        below = Depth()
        for a in staged:
            if stage_of[id(a)] == s:
                below = fam.worst(below, depth_in[id(a)])
        d = below + cost
        if fam.ns(d) > clk_ns:
            s += 1            # cut here: every argument arrives registered
            d = cost
        stage_of[key] = s
        depth_in[key] = d

    for root in roots:
        visit(root)
    n_stages = max((stage_of[id(root)] for root in roots), default=0) + 1
    return stage_of, n_stages


def check(root, clk_ns: float, dev, label: str = "stage") -> Depth:
    """The budget verdict for one stage, named and in nanoseconds.

    Raises when the estimate exceeds the clock: at generation time,
    in microseconds, instead of after place-and-route, in despair.
    """
    fam = device(dev)
    depth = expr_depth(root, fam)
    est = fam.ns(depth)
    if est > clk_ns:
        raise ValueError(
            f"{label}: estimated {est:.1f} ns ({depth.levels} levels, "
            f"{depth.carry_bits} carry bits, {depth.dsps} DSP) exceeds the "
            f"{clk_ns:.1f} ns budget on {fam.name}. Split the stage (the "
            "streams are elastic; a register here is free) or slow the "
            "clock island.")
    return depth
