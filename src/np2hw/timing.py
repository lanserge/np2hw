# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Timing as a TRACED property: stage depth from the lowered graph.

Every datapath np2hw builds is a graph whose nodes carry exact value
ranges -- so the logic each node becomes is arithmetic the generator can
do: an add is a carry chain as wide as its widest
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

Nothing outside this file may contain a device constant. The compiler
asks the device for a shape and for nanoseconds, and adding a family is
subclassing `Device` -- not editing np2hw.

The same arithmetic drives automatic pipelining: the compiler's schedule
(np2hw.core.lower) walks the graph in dependency order and starts a new
pipeline stage the moment the accumulated path would exceed the budget,
so the registers are placed mechanically. The streams are elastic, so the
added latency is free by construction; the values are untouched, so
the twin stays bit-exact by construction. The one thing a register
cannot fix is a single operation deeper than the clock -- that is the
floor, and it refuses with the operation named.
"""
from __future__ import annotations

from dataclasses import dataclass

def _clog2(n: int) -> int:
    """Bits to count n things: ceil(log2(n)), 0 for one."""
    return (n - 1).bit_length() if n > 1 else 0


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
    # The attribute that makes synthesis build a multiply as that hard
    # multiplier. The model times every multiply at dsp_ns; left to its
    # own heuristics a tool builds a narrow one from LUTs and carry
    # chains, several times slower -- a stage scheduled for one shape and
    # built as another. So the emitter writes what the model assumed.
    mul_attribute = None
    overhead_ns = None         # clock-to-out + setup + the first/last nets
    mem_ns = None              # a memory's clock-to-out (registered read)
    carry_chain_ns = None      # starting one carry chain, and leaving it
    lut_inputs = None          # how many inputs one lookup table has
    rom_logic_entries = None   # the most entries a constant table built as
                               # logic has (rom_in_logic)

    _REQUIRED = ("name", "level_ns", "carry_ns_per_bit", "dsp_ns",
                 "mul_attribute", "overhead_ns", "mem_ns", "carry_chain_ns",
                 "lut_inputs", "rom_logic_entries")

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

    def rom_in_logic(self, entries: int) -> bool:
        """Whether a constant table of `entries` is built as LOGIC -- its
        contents constants, read through a select tree in the stage that
        uses it -- or as a MEMORY, an initialised block read through a
        register, a stage of its own. A question about the device, not the
        model: the table computes the same values either way, and only
        what it costs differs. Width does not enter: each output bit is a
        table of its own, the same shape whatever the word's width."""
        return entries <= self.rom_logic_entries

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

    def gather_is_read(self, size: int) -> bool:
        """Is a table this big a memory READ rather than a mux?

        A `rom` is registered by construction, and the reason is written
        beside it: that is what makes it a block RAM instead of a mux
        tree as deep as the table. A `gather` is the same read with
        writable contents, so the same reasoning applies to it -- and
        only the contents differ, never the shape.

        The line is one LUT. A table a single lookup table can select
        from is a mux, and registering it would buy a level of logic at
        the price of a pipeline stage. Past that the tree grows, its
        entries spread out across the die because they have to live
        somewhere, and the read wants a cycle of its own.

        This bench measured the difference. Gamma's 33 knots, selected by
        the pixel, arrived as a 3-level tree reaching 33 registers spread
        over four slice columns: 2.199ns of logic and 3.719ns of route
        for one stage. With the knots baked they folded into a table the
        tools put in distributed RAM and the path did not appear at all.
        """
        return self.mux_levels(size) > 1

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
    mul_attribute = '(* use_dsp = "yes" *)'
    overhead_ns = 1.6
    mem_ns = 2.1
    # MEASURED, on the path that exposed the need for the term: a
    # 13-bit add in a chain costs about 1.6 ns end to end, of which the
    # level and the per-bit carry account for about 1.0. Solved from one
    # path, so still fitted rather than swept.
    carry_chain_ns = 0.5
    lut_inputs = 6
    # A slice's four 6-input LUTs and its F7/F8 multiplexers read one bit of
    # a 256-entry table in one level of logic: a slice per output bit, where
    # block RAM would take a tile and a stage of its own.
    rom_logic_entries = 256


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
