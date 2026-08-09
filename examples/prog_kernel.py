"""Programmable 3x3 kernel: the coefficients are REGISTERS, not constants.

Param("k", shape=(3,3)) is a collection of 9 scalar Params (k[i,j] -> k_i_j).
Used as per-tap coefficients, they become 9 register-fed multipliers (a real
MAC), wired to a 9-entry register file. Software writes the kernel over the bus,
then streams the frame; output is validated against the SAME NumPy function with
the written kernel. (Valid-interior; edges + programmable kernel is a follow-up.)

Run:  python examples/prog_kernel.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check_ctrl
from np2hw.ir import Param

W, H = 8, 6
A = np.random.default_rng(77).integers(0, 256, (H, W), dtype=np.uint8)


def conv3x3(img, k):
    """Programmable 3x3 convolution (valid region), //16, saturate to 8-bit.
    k is a 3x3 Param; k[i,j] are the 9 register coefficients."""
    x = img.astype(np.int32)
    acc = sum(k[i, j] * x[i:i + H - 2, j:j + W - 2]
              for i in range(3) for j in range(3))
    return (acc // 16).clip(0, 255)


def main():
    k = Param("k", np.uint8, shape=(3, 3))
    # a blur-ish kernel summing to 16 (so //16 ~= unity gain)
    coeffs = [[0, 2, 0], [2, 8, 2], [0, 2, 0]]
    kvals = {f"k_{i}_{j}": coeffs[i][j] for i in range(3) for j in range(3)}
    print("programmable 3x3 kernel: write 9 coeff registers, then stream")
    r = [
        check_ctrl("conv", conv3x3, A, [k], kvals, ctrl="axil"),
        check_ctrl("conv", conv3x3, A, [k], kvals, ctrl="umi"),
    ]
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
