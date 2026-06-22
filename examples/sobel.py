"""Signed arithmetic: Sobel-X gradient with negative coefficients, validated
against the SAME NumPy function in int16 (results go negative).

    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]

Run:  python examples/sobel.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6
A = np.random.default_rng(7).integers(0, 256, (H, W), dtype=np.uint8)


def sobel_x(img):
    x = img.astype(np.int16)                 # signed; uint8 pixels widened
    return (-x[:-2, :-2] +   x[:-2, 2:]
            - 2*x[1:-1, :-2] + 2*x[1:-1, 2:]
            -   x[2:, :-2] +   x[2:, 2:])


def laplacian(img):
    x = img.astype(np.int16)
    return (              x[:-2, 1:-1]
            + x[1:-1, :-2] - np.int16(4)*x[1:-1, 1:-1] + x[1:-1, 2:]   # numpy scalar coeff
            +              x[2:, 1:-1])


def main():
    print("signed filters (DUT vs int16 NumPy, values go negative):")
    r1 = check("sobel_x", sobel_x, A, show=6)
    r2 = check("laplacian", laplacian, A, show=6)
    return 0 if (r1 and r2) else 1


if __name__ == "__main__":
    sys.exit(main())
