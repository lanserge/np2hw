"""NumPy -> tree -> Verilog, validated against the SAME NumPy function (real uint8).
Run:  python examples/numpy_to_ir.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H = 8, 6
A = np.random.default_rng(1).integers(0, 256, (H, W), dtype=np.uint8)


def avg_blur(img):
    x = img.astype(np.uint16)
    return ((x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:]) // 4).astype(np.uint8)


def gaussian(img):
    x = img.astype(np.uint16)
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def gain(img, g):
    return g * img.astype(np.uint16)


def main():
    print("NumPy -> Verilog (validated vs real-uint8 oracle):")
    r = [
        check("avg_blur", avg_blur, A),
        check("gaussian", gaussian, A),
        check("gain", gain, A, params=[Param("gain", np.uint8)], param_values={"gain": 3}),
    ]
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
