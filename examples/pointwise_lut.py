"""Pointwise expression DAGs: fork, gather, recombine -- the LUT case.

The trailing-op chain says "one value, a line of ops". A lookup table cannot
be said that way: it derives a segment index and a fraction from the SAME
input, reads TWO table entries, and recombines -- a small DAG. This example
is that shape, written as ordinary NumPy (shift, mask, fancy-index, lerp),
traced into one combinational datapath:

    seg  = x >> S                # top bits: segment index (a bit-slice)
    frac = x & (2**S - 1)        # bottom bits: position within the segment
    y    = knot[seg] + (knot[seg+1] - knot[seg]) * frac >> S

Checks, each a way the emitter could be wrong:

  BIT-EXACT      four knot tables -- identity ramp, a gamma-like curve, a
                 random curve, a deliberately NON-monotone one (negative
                 steps: the subtract must go signed) -- against the same
                 NumPy function on real data.
  IDENTITY       the ramp table reproduces the input exactly: the classic
                 LUT off-by-one (top segment one short) shows here first.
  STRUCTURE      one register-array mux, shared by both reads -- not one
                 array per gather; and the gather index is proven in range
                 at TRACE time, so an unclippable index REFUSES to trace.

Run:  python examples/pointwise_lut.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

BITS, K = 12, 5
N = (1 << K) + 1                 # 33 knots -> 32 segments
SHIFT = BITS - K
FULL = (1 << BITS) - 1


def pwl(img, knots):
    """Piecewise-linear lookup, integer lerp, exactly as the hardware."""
    value = img.astype(np.int32)
    seg = value >> SHIFT
    frac = value & ((1 << SHIFT) - 1)
    base = knots[seg].astype(np.int32)
    step = knots[seg + 1].astype(np.int32) - base
    return (base + ((step * frac) >> SHIFT)).clip(0, FULL).astype(np.uint16)


def main():
    rng = np.random.default_rng(12)
    A = rng.integers(0, FULL + 1, (6, 8)).astype(np.uint16)
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 0, FULL, 1 << (BITS - 1), (1 << (BITS - 1)) - 1

    knots_param = [Param("knots", bits=13, shape=(N,),
                         description="PWL knots; the segment is the input's "
                                     "top bits, the fraction its bottom bits")]

    identity = {f"knots_{i}": i << SHIFT for i in range(N)}
    gamma_ish = {f"knots_{i}": int(round(FULL * (i / (N - 1)) ** 0.45))
                 for i in range(N)}
    random_curve = {f"knots_{i}": int(v) for i, v in enumerate(
        rng.integers(0, FULL + 1, N))}
    non_monotone = {f"knots_{i}": int(FULL if i % 2 else 0) for i in range(N)}

    print(f"{BITS}-bit piecewise-linear LUT, {N} knots; one core, four curves:")
    results = []
    for label, table in (("identity", identity), ("gammaish", gamma_ish),
                         ("random", random_curve), ("sawtooth", non_monotone)):
        results.append(check(f"pwl_{label.strip()}", pwl, A,
                             params=knots_param, param_values=table, bits=BITS))

    # Identity really is identity -- the top segment is where LUTs lose a bit.
    oracle = pwl(A, np.array([identity[f"knots_{i}"] for i in range(N)]))
    ident_ok = np.array_equal(oracle, A)
    print(f"\n  identity table reproduces the input exactly: {ident_ok}")

    from np2hw import Image2D, generate, to_ir

    _, line = to_ir(pwl, Image2D("img", 8, 6, bits=BITS), *knots_param)
    core = generate(line, module_name="pwl")
    luts = core.verilog.count("] knots_lut [")
    reads = core.verilog.count("knots_lut[e")
    shape_ok = luts == 1 and reads == 2
    print(f"  one register array ({luts}), shared by both gathers ({reads} reads)")

    # An index that can escape the table must refuse to trace.
    def runaway(img, knots):
        return knots[img.astype(np.int32) >> (SHIFT - 1)]   # reaches 2N-2
    try:
        to_ir(runaway, Image2D("img", 8, 6, bits=BITS), *knots_param)
        refused = False
    except ValueError as error:
        refused = "falls outside the table" in str(error)
    print(f"  out-of-range gather refused at trace time: {refused}")

    ok = all(results) and ident_ok and shape_ok and refused
    print("\n" + ("POINTWISE LUT PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
