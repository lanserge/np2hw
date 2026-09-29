# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The index algebra: one axis of a pixel read, built by arithmetic.

`Index(n)` counts 0 .. n-1; multiplying scales the step (and the start),
adding moves the start. Every claim below is checked against NumPy itself:
the positions an index names must be the positions the same slicing reads
from `np.arange`, for every slice, every composition of two slices, and
every value of a phase register.

Run:  python examples/index_algebra.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw import Index, Param


def check(label, ok):
    print(f"  {label:<60} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    r = []
    i = Index(8)
    print("the algebra:")
    r.append(check("Index(8) counts 0..7", i.positions() == list(range(8))))
    r.append(check("2 * i has step 2", (2 * i).step == 2
                   and (2 * i).positions() == list(range(0, 16, 2))))
    r.append(check("3 * (2 * i) has step 6", (3 * (2 * i)).step == 6))
    r.append(check("2 * i + 1 starts at 1", (2 * i + 1).positions()
                   == list(range(1, 16, 2))))
    r.append(check("3 * (2 * i + 1) is 6 * i + 3 (the start scales too)",
                   3 * (2 * i + 1) == 6 * i + 3))
    r.append(check("i + 2 - 2 is i", i + 2 - 2 == i))

    print("against NumPy, every slice of an axis of 12:")
    n = 12
    axis = np.arange(n)
    ok = True
    slices = [slice(a, b, c) for a in (None, 0, 1, 2, 5, -3)
              for b in (None, -1, -2, 7, 12) for c in (None, 1, 2, 3)]
    for s in slices:
        ok &= Index.from_slice(s, n).positions() == axis[s].tolist()
    r.append(check(f"{len(slices)} slices name what NumPy reads", ok))

    ok = True
    for s in slices:
        outer = Index.from_slice(s, n)
        inner_axis = axis[s]
        for t in slices:
            inner = Index.from_slice(t, len(inner_axis))
            ok &= outer.compose(inner).positions() == inner_axis[t].tolist()
    r.append(check(f"{len(slices) ** 2} compositions name x[s][t]", ok))

    print("a register in the start (the CFA phase):")
    py = Param("py", bits=1)
    from np2hw.ir import PhaseRef
    half = n // 2
    ok = True
    for v in (0, 1):
        ok &= (2 * Index(half) + PhaseRef(py)).positions({"py": v}) \
            == axis[v::2].tolist()
        ok &= (2 * Index(half) + (1 - PhaseRef(py))).positions({"py": v}) \
            == axis[1 - v::2].tolist()
    r.append(check("x[py::2] and x[1-py::2], for py = 0 and 1", ok))
    both = 3 * (2 * Index(half) + PhaseRef(py))
    r.append(check("a multiply scales the register's coefficient too",
                   both.regs == (("py", 3),) and both.step == 6))

    print("refusals:")
    for label, bad in [("step 0", lambda: 0 * i), ("negative step", lambda: -1 * i),
                       ("a negative count", lambda: Index(-1))]:
        try:
            bad()
            r.append(check(f"{label} refused", False))
        except ValueError:
            r.append(check(f"{label} refused", True))
    try:
        (2 * Index(4) + PhaseRef(py)).last
        r.append(check("a register-dependent position is not a number", False))
    except ValueError:
        r.append(check("a register-dependent position is not a number", True))

    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
