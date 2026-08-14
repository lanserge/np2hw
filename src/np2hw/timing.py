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

The nanosecond mapping is CALIBRATION, once per device family: a few
constants (effective LUT level, carry per bit, DSP through-delay,
register overhead) measured from real routed designs. The first
7-series table below is anchored to a bench-measured stage -- a tone
curve's gather+multiply+add at 11.4 ns routed -- and is deliberately
conservative; refine it from CI sweeps, never per block.

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


@dataclass(frozen=True)
class Family:
    """Per-device-family calibration constants, nanoseconds.

    level_ns is an EFFECTIVE LUT level: logic plus its average routed
    net. overhead_ns covers clock-to-out, the final setup, and the
    first/last nets. Conservative by intent -- a pass here should
    survive routing, not merely hope to.
    """
    name: str
    level_ns: float
    carry_ns_per_bit: float
    dsp_ns: float
    overhead_ns: float
    # A block RAM's clock-to-out: a DATASHEET number, not a function of
    # the table's depth. Costing a memory read this way (instead of by
    # analogy to a mux tree) is the whole reason reads are registered.
    mem_ns: float = 2.1


# Anchored to the bench: an 11-level, 28-carry-bit, one-DSP stage
# routed at ~11.4 ns on xc7z020-1. 1.6 + 11*0.62 + 28*0.03 + 3.2 = 12.5:
# conservative by about a nanosecond on the anchor, as intended.
SERIES7 = Family("7series-1", level_ns=0.62, carry_ns_per_bit=0.03,
                 dsp_ns=3.2, overhead_ns=1.6)

FAMILIES = {"7series": SERIES7}


@dataclass
class Depth:
    """Worst path of one expression, in structural units."""
    levels: int = 0
    carry_bits: int = 0
    dsps: int = 0
    mems: int = 0

    def ns(self, family: Family) -> float:
        return (family.overhead_ns + self.levels * family.level_ns
                + self.carry_bits * family.carry_ns_per_bit
                + self.dsps * family.dsp_ns + self.mems * family.mem_ns)

    def __add__(self, other: "Depth") -> "Depth":
        return Depth(self.levels + other.levels,
                     self.carry_bits + other.carry_bits,
                     self.dsps + other.dsps, self.mems + other.mems)

    def worst(self, other: "Depth") -> "Depth":
        # Along the worst PATH the components add per node; across
        # siblings the slower branch wins by its nanosecond total.
        return self if self._key() >= other._key() else other

    def _key(self):
        return (self.levels * SERIES7.level_ns
                + self.carry_bits * SERIES7.carry_ns_per_bit
                + self.dsps * SERIES7.dsp_ns + self.mems * SERIES7.mem_ns)


def expr_depth(node) -> Depth:
    """Worst-path structural depth of a PExpr DAG: node_cost summed
    along the worst path, sibling branches compared by nanosecond
    total. One cost table (node_cost) owns every op's hardware shape;
    this walk and the stage cutter both read it."""
    if not isinstance(node, PExpr):
        return Depth()
    below = Depth()
    for a in node.args:
        if isinstance(a, PExpr):
            below = below.worst(expr_depth(a))
    return below + node_cost(node)


def mux_levels(size: int) -> int:
    """LUT6 select-tree depth for a size-entry read -- the law behind a
    gather's register array and a line buffer's column read alike."""
    return max(1, (_clog2(max(size, 2)) + 1) // 2)


def node_cost(node) -> Depth:
    """The hardware shape of ONE node: what it adds to any path through
    it. Leaves cost nothing (they are registers, constants, or the
    wires of a register array); adds and subtracts one level plus a
    carry as wide as the result, a multiply one DSP, a clip a
    compare-and-select, a mask or constant shift nothing at all, a
    gather the mux tree its table size dictates."""
    if node.op in ("acc", "const", "param", "tap", "row", "col"):
        # registers, constants, window taps, and the position counters
        # every core already keeps: wires into the stage
        return Depth()
    if node.op == "gather":
        parent, _index = node.args
        return Depth(levels=mux_levels(int(parent.shape[0])))
    if node.op == "rom":
        # A registered memory read: it OWNS its stage, so what it costs
        # the following stage is the memory's clock-to-out, and nothing
        # in front of it can share the cycle.
        return Depth(mems=1)
    if node.op in ("add", "sub"):
        return Depth(levels=1, carry_bits=_bits(node))
    if node.op == "mul":
        return Depth(dsps=1)
    if node.op == "clip":
        return Depth(levels=2, carry_bits=_bits(node))
    if node.op in ("shr", "mask"):
        return Depth()
    if node.op == "lt":
        return Depth(levels=1, carry_bits=_bits(node.args[0])
                     if isinstance(node.args[0], PExpr) else 0)
    if node.op == "abs":
        return Depth(levels=1, carry_bits=_bits(node))
    if node.op == "sel":
        return Depth(levels=1)
    raise ValueError(f"no depth rule for PExpr op {node.op!r} -- a new op "
                     "needs a hardware shape here before it ships")


# param and const leaves FLOAT: a register port is stable across a
# pixel's flight through the pipeline and a literal is wiring, so both
# feed any stage directly and never ride a delay line. acc, tap, row
# and col are PINNED to stage 0 -- they describe THIS pixel (its input
# word, its position), so a later stage must see them through the
# delay line, not as the counter's current value.
FLOATING = ("const", "param")


def assign_stages(roots, clk_ns: float, family: str = "7series",
                  label: str = "stage"):
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
    fam = FAMILIES[family]
    stage_of = {}
    depth_in = {}

    def visit(node):
        key = id(node)
        if key in stage_of:
            return
        args = [a for a in node.args if isinstance(a, PExpr)]
        for a in args:
            visit(a)
        cost = node_cost(node)
        if cost.ns(fam) > clk_ns:
            raise ValueError(
                f"{label}: operation {node.op!r} alone estimates "
                f"{cost.ns(fam):.1f} ns against the {clk_ns:.1f} ns budget "
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
                below = below.worst(depth_in[id(a)])
        d = below + cost
        if d.ns(fam) > clk_ns:
            s += 1            # cut here: every argument arrives registered
            d = cost
        stage_of[key] = s
        depth_in[key] = d

    for root in roots:
        visit(root)
    n_stages = max((stage_of[id(root)] for root in roots), default=0) + 1
    return stage_of, n_stages


def check(root, clk_ns: float, family: str = "7series",
          label: str = "stage") -> Depth:
    """The budget verdict for one stage, named and in nanoseconds.

    Raises when the estimate exceeds the clock: at generation time,
    in microseconds, instead of after place-and-route, in despair.
    """
    fam = FAMILIES[family]
    depth = expr_depth(root)
    est = depth.ns(fam)
    if est > clk_ns:
        raise ValueError(
            f"{label}: estimated {est:.1f} ns ({depth.levels} levels, "
            f"{depth.carry_bits} carry bits, {depth.dsps} DSP) exceeds the "
            f"{clk_ns:.1f} ns budget on {fam.name}. Split the stage (the "
            "streams are elastic; a register here is free) or slow the "
            "clock island.")
    return depth
