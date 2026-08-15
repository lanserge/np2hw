"""Timing as a traced property, proven against the bench's own facts.

The estimator walks the same PExpr DAGs the emitter turns into
Verilog and answers, at generation time, the question a
place-and-route run answers half an hour later: does this stage fit
this clock. Its calibration is anchored to a routed, measured stage,
so the claims here are the bench's history restated as arithmetic:

  ANCHOR         a tone-curve stage (33-knot gather, DSP multiply,
                 interpolation add, clip) measured ~11.4 ns routed on
                 xc7z020-1. The estimate must fail a 10 ns budget --
                 the failure that sent it to a clock island -- and
                 pass the island's 15 ns.
  SHALLOW        a black-level stage (add, clip) must fit even the
                 148.5 MHz pixel constraint. Small math is small.
  MONOTONE       a deeper table gathers through a deeper mux tree:
                 depth must not decrease as tables grow.
  REFUSAL        an op with no hardware shape in the table is an
                 ERROR at estimate time, not a silent zero -- a new
                 op must state its cost before it ships.

Run:  python examples/timing_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.ir import PExpr
from np2hw.timing import SERIES7, Device, check, expr_depth


class _Table:
    """A register array as the gather sees it: a shape and a name."""
    def __init__(self, size):
        self.shape = (size,)
        self.name = "knots"
        self.signed = False


def acc(lo=0, hi=1023):
    return PExpr("acc", (), lo, hi)


def gamma_stage(knot_count=33, bits=10):
    """The tone curve's DAG, as the block builds it: segment index and
    fraction from the input, two knot reads, linear interpolation."""
    seg_bits = 5
    top = 1 << (bits + 1)
    table = _Table(knot_count)
    index = PExpr("shr", (acc(), bits - seg_bits), 0, knot_count - 2)
    k0 = PExpr("gather", (table, index), 0, top)
    k1 = PExpr("gather", (table,
                          PExpr("add", (index, PExpr("const", (), 1, 1)),
                                1, knot_count - 1)), 0, top)
    frac = PExpr("mask", (acc(), bits - seg_bits), 0,
                 (1 << (bits - seg_bits)) - 1)
    span = PExpr("sub", (k1, k0), -top, top)
    scaled = PExpr("mul", (span, frac), -(top << 5), top << 5)
    interp = PExpr("add", (PExpr("shr", (scaled, bits - seg_bits),
                                 -top, top), k0), -top, 2 * top)
    return PExpr("clip", (interp, 0, (1 << bits) - 1), 0, (1 << bits) - 1)


def blacklevel_stage(bits=10):
    offset = PExpr("const", (), -64, -64)
    summed = PExpr("add", (acc(), offset), -64, (1 << bits) - 1 - 64)
    return PExpr("clip", (summed, 0, (1 << bits) - 1), 0, (1 << bits) - 1)


def main():
    checks = []

    def result(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<56} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("timing_budget:")

    gm = expr_depth(gamma_stage(), SERIES7)
    est = SERIES7.ns(gm)
    try:
        check(gamma_stage(), 10.0, SERIES7, label="gamma")
        anchored = False
    except ValueError:
        anchored = True
    result("anchor: the gamma stage FAILS the 10 ns it failed on",
           anchored, f"est {est:.1f} ns, {gm.levels} levels, {gm.dsps} DSP")
    ok = True
    try:
        check(gamma_stage(), 15.0, SERIES7, label="gamma")
    except ValueError:
        ok = False
    result("anchor: the same stage fits the 15 ns island it runs on", ok)

    bl = expr_depth(blacklevel_stage(), SERIES7)
    ok = True
    try:
        check(blacklevel_stage(), 6.7, SERIES7, label="blacklevel")
    except ValueError:
        ok = False
    result("shallow: black level fits the 148.5 MHz pixel budget", ok,
           f"est {SERIES7.ns(bl):.1f} ns")

    small = expr_depth(gamma_stage(knot_count=9), SERIES7)
    big = expr_depth(gamma_stage(knot_count=257), SERIES7)
    result("monotone: a deeper table cannot get faster",
           SERIES7.ns(big) >= SERIES7.ns(small))

    try:
        expr_depth(PExpr("modulo", (acc(),), 0, 1), SERIES7)
        refused = False
    except ValueError:
        refused = True
    result("refusal: an op without a hardware shape is an error", refused)

    # The device is a DRIVER: it owns the shapes, not only the numbers.
    # A narrower lookup table does not merely run slower, it builds a
    # DEEPER select tree -- so if any LUT6 assumption still lurked in
    # np2hw, this claim could not move.
    class Narrow(SERIES7.__class__):
        name = "hypothetical-4LUT"
        lut_inputs = 4
    narrow = Narrow()
    result("device: a 6-input LUT selects 4:1, a 4-input LUT only 2:1",
           SERIES7.mux_arity() == 4 and narrow.mux_arity() == 2)
    result("device: the same table is a deeper tree on a narrower LUT",
           narrow.mux_levels(256) > SERIES7.mux_levels(256),
           f"{SERIES7.mux_levels(256)} vs {narrow.mux_levels(256)} levels")
    deep = expr_depth(gamma_stage(), narrow)
    result("device: that depth reaches the estimate, not just the helper",
           deep.levels > gm.levels)

    # A calibration number has no default: a family that forgets one
    # must refuse, not silently inherit another family's silicon.
    class Forgetful(Device):
        name = "no-numbers"
        level_ns = 0.5
    try:
        Forgetful()
        named = False
    except ValueError as e:
        named = "mem_ns" in str(e) and "lut_inputs" in str(e)
    result("device: an unset constant refuses, and names what is missing",
           named)

    ok = all(checks)
    print("\n" + ("TIMING_BUDGET PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
