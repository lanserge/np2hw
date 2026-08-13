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

What this is NOT, yet: automatic pipelining. When a stage exceeds the
budget, `check()` names it and the depth arithmetic says where the
cut belongs; inserting the register mechanically is the follow-on --
the streams are elastic, so added latency is free by construction.
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

    def ns(self, family: Family) -> float:
        return (family.overhead_ns + self.levels * family.level_ns
                + self.carry_bits * family.carry_ns_per_bit
                + self.dsps * family.dsp_ns)

    def __add__(self, other: "Depth") -> "Depth":
        return Depth(self.levels + other.levels,
                     self.carry_bits + other.carry_bits,
                     self.dsps + other.dsps)

    def worst(self, other: "Depth") -> "Depth":
        # Along the worst PATH the components add per node; across
        # siblings the slower branch wins by its nanosecond total.
        return self if self._key() >= other._key() else other

    def _key(self):
        return (self.levels * SERIES7.level_ns
                + self.carry_bits * SERIES7.carry_ns_per_bit
                + self.dsps * SERIES7.dsp_ns)


def expr_depth(node) -> Depth:
    """Worst-path structural depth of a PExpr DAG.

    Leaves cost nothing (they are registers, constants, or the wires
    of a register array); each interior node adds what its hardware
    is: adds and subtracts one level plus a carry as wide as the
    result, a multiply one DSP, a clip a compare-and-select, a mask
    or constant shift nothing at all, a gather the mux tree its table
    size dictates.
    """
    if not isinstance(node, PExpr):
        return Depth()
    if node.op in ("acc", "const", "param", "tap"):
        # registers, constants, and window taps: wires into the stage
        return Depth()
    if node.op == "gather":
        parent, index = node.args
        size = int(parent.shape[0])
        mux_levels = max(1, (_clog2(max(size, 2)) + 1) // 2)  # LUT6 = 4:1
        return expr_depth(index) + Depth(levels=mux_levels)
    args = [a for a in node.args if isinstance(a, PExpr)]
    below = Depth()
    for a in args:
        below = below.worst(expr_depth(a))
    if node.op in ("add", "sub"):
        return below + Depth(levels=1, carry_bits=_bits(node))
    if node.op == "mul":
        return below + Depth(dsps=1)
    if node.op == "clip":
        return below + Depth(levels=2, carry_bits=_bits(node))
    if node.op in ("shr", "mask"):
        return below                       # wiring, when the amount is fixed
    if node.op == "lt":
        return below + Depth(levels=1, carry_bits=_bits(node.args[0])
                             if isinstance(node.args[0], PExpr) else 0)
    if node.op == "abs":
        return below + Depth(levels=1, carry_bits=_bits(node))
    if node.op == "sel":
        return below + Depth(levels=1)     # the compare rode in through args
    raise ValueError(f"no depth rule for PExpr op {node.op!r} -- a new op "
                     "needs a hardware shape here before it ships")


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
