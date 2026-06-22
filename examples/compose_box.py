"""A 2x2 box applied twice == 3x3 Gaussian, normalised by //16.
Run:  python examples/compose_box.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6
A = np.random.default_rng(3).integers(0, 256, (H, W), dtype=np.uint8)


def box2(x):
    return x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:]


def gaussian_from_box(img):
    x = img.astype(np.uint16)
    return (box2(box2(x)) // 16).astype(np.uint8)


def direct_gaussian(img):
    x = img.astype(np.uint16)
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def main():
    print("box2 o box2 == 3x3 Gaussian:")
    same = np.array_equal(gaussian_from_box(A), direct_gaussian(A))
    print(f"  composed == direct gaussian (NumPy): {same}")
    ok = check("gauss_box", gaussian_from_box, A)
    return 0 if (ok and same) else 1


if __name__ == "__main__":
    sys.exit(main())
