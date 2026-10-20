# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Timing as a traced property, proven against the bench's own facts.

The estimator walks the same lowered graph the printer turns into Verilog
and answers, at generation time, the question a place-and-route run
answers half an hour later: does this stage fit this clock. Its
calibration is anchored to a routed, measured stage, so the claims here
are the bench's history restated as arithmetic:

  ANCHOR         a tone-curve stage (33-knot gather, DSP multiply,
                 interpolation add, clip) measured ~11.4 ns routed on
                 xc7z020-1. The estimate must fail a 10 ns budget --
                 the failure that sent it to a clock island -- and
                 pass the island's 15 ns.
  SHALLOW        a black-level stage (add, clip) must fit even the
                 148.5 MHz pixel constraint. Small math is small.
  MONOTONE       a deeper table gathers through a deeper select tree:
                 depth must not decrease as tables grow.
  REFUSAL        an operation with no hardware shape in the table is an
                 ERROR at estimate time, not a silent zero -- a new
                 kind must state its cost before it ships.

Run:  python examples/timing_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw import Image2D, Param, generate, to_ir
from np2hw.core.pipeline import _cost, stage_ns
from np2hw.timing import SERIES7, Device

BITS = 12


def tone(knot_count):
    """The gamma block's piecewise-linear curve over `knot_count` knots."""
    shift = BITS - ((knot_count - 1).bit_length() - 1)
    knots = Param("knots", bits=13, shape=(knot_count,))

    def curve(img, knots):
        value = img.astype(np.int32)
        seg, frac = value >> shift, value & ((1 << shift) - 1)
        base = knots[seg].astype(np.int32)
        step = knots[seg + 1].astype(np.int32) - base
        return (base + ((step * frac) >> shift)).clip(0, (1 << BITS) - 1
                                                      ).astype(np.uint16)
    return curve, knots


def blacklevel(img, offset):
    return (img.astype(np.int32) + offset).clip(0, (1 << BITS) - 1
                                                ).astype(np.uint16)


def built(fn, *params):
    """The model lowered unstaged: one stage, whose depth is the estimate."""
    return to_ir(fn, Image2D("img", 64, 8, bits=BITS), *params)[1]


def fits(mod, ns, dev=SERIES7):
    return stage_ns(mod, dev) <= ns


def main():
    checks = []

    def result(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<56} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("timing_budget:")

    gamma = built(*tone(33))
    est = stage_ns(gamma, SERIES7)
    result("anchor: the gamma stage FAILS the 10 ns it failed on",
           not fits(gamma, 10.0), f"est {est:.1f} ns, routed 11.4 ns")
    result("anchor: the same stage fits the 15 ns island it runs on",
           fits(gamma, 15.0))

    bl = built(blacklevel, Param("offset", bits=10, signed=True))
    result("shallow: black level fits the 148.5 MHz pixel budget",
           fits(bl, 1000.0 / 148.5), f"est {stage_ns(bl, SERIES7):.1f} ns")

    small, big = built(*tone(9)), built(*tone(257))
    result("monotone: a deeper table cannot get faster",
           stage_ns(big, SERIES7) >= stage_ns(small, SERIES7),
           f"{stage_ns(small, SERIES7):.1f} ns for 9 knots, "
           f"{stage_ns(big, SERIES7):.1f} ns for 257")

    class Unknown:
        kind = "hw.modulo"
    try:
        _cost(Unknown(), SERIES7)
        refused = False
    except ValueError as error:
        refused = "hw.modulo" in str(error)
    result("refusal: a kind without a hardware shape is an error", refused)

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
    # And the shape reaches the BUILD, not just the helper. A table deep
    # enough to stop being one lookup table stops being a select at all:
    # it becomes a registered read, which owns its stage. Where that line
    # falls is the device's to say, so the SAME four-entry table is a
    # select on a 6-input LUT and a read on a 4-input one -- different
    # hardware from one model, under the same generous clock.
    table = Param("t", bits=10, shape=(4,))

    def lookup(img, t):
        return t[img.astype(np.int32) >> 8].astype(np.uint16)
    _, mod = to_ir(lookup, Image2D("img", 16, 4, bits=10), table)
    wide = generate(mod, "lookup", clk_ns=20.0)["pipeline_stages"]
    thin = generate(mod, "lookup", clk_ns=20.0, dev=narrow)["pipeline_stages"]
    result("device: that shape reaches the build, not just the helper",
           wide == 1 and thin == 2,
           f"4-entry table: {wide} stage on {SERIES7.name}, {thin} on the "
           "4-input LUT (the read owns its stage)")

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
