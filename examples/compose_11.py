"""[1,1] applied twice == [1,2,1].  Filter composition (slicing a result) in the
NumPy front-end, validated against the same function and against the direct kernel.
Run:  python examples/compose_11.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6
A = np.random.default_rng(2).integers(0, 256, (H, W), dtype=np.uint8)


def f11_then_11(img):
    x = img.astype(np.uint16)
    a = x[:-1, :] + x[1:, :]        # first  [1,1]
    b = a[:-1, :] + a[1:, :]        # second [1,1] on the RESULT
    return b                        # == [1,2,1]


def f121_direct(img):
    x = img.astype(np.uint16)
    return x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]


def main():
    print("[1,1] o [1,1] == [1,2,1]:")
    same = np.array_equal(f11_then_11(A), f121_direct(A))
    print(f"  composed == direct kernel (NumPy): {same}")
    ok = check("compose11", f11_then_11, A)
    return 0 if (ok and same) else 1


if __name__ == "__main__":
    sys.exit(main())
