"""Faithful dtype tracking: uint8 wraps; astype(uint16) widens to avoid it.
Both validated against the SAME function on real uint8 data (no int64 cast).
Run:  python examples/dtype_faithful.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6
A = np.full((H, W), 200, np.uint8)
A[::2, ::2] = 250


def blur_u8(img):                       # stays uint8 -> the 4-pixel sum WRAPS
    return (img[:-1, :-1] + img[:-1, 1:] + img[1:, :-1] + img[1:, 1:]) // 4


def blur_u16(img):                      # widened -> correct average
    x = img.astype(np.uint16)
    return ((x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:]) // 4).astype(np.uint8)


def main():
    print("faithful dtype (DUT vs real uint8 NumPy):")
    r1 = check("blur_u8", blur_u8, A, show=4)    # wraps -> small values
    r2 = check("blur_u16", blur_u16, A, show=4)  # correct ~206
    return 0 if (r1 and r2) else 1


if __name__ == "__main__":
    sys.exit(main())
