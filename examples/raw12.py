"""Non-8-bit pixels: 12-bit raw sensor data (0..4095) -- the common ISP case.

The compiler is parametric in pixel width (Image2D bits=N); this exercises it
with 12-bit input through the same NumPy->RTL->validate flow. Validated against
the SAME NumPy function, and through the AXI4-Stream Video adapter (TDATA
byte-aligned to 16 bits for 12-bit pixels).

Run:  python examples/raw12.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check, check_bp

W, H, BITS = 8, 6, 12
A = np.random.default_rng(12).integers(0, 1 << BITS, (H, W), dtype=np.uint16)


def blur12(img):
    """2x2 average, kept within 12 bits (widen to avoid wrap)."""
    x = img.astype(np.uint32)
    return ((x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:]) // 4).astype(np.uint16)


def gaussian12(img):
    """3x3 Gaussian /16, replicate edges, same-size, 12-bit out."""
    x = np.pad(img.astype(np.uint32), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint16)


def main():
    print(f"{BITS}-bit raw pixels (0..{(1<<BITS)-1}), DUT vs NumPy:")
    r = [
        check("blur12", blur12, A, bits=BITS),
        check("gauss12", gaussian12, A, bits=BITS),
        check_bp("gauss12", gaussian12, A, iface="axis", bits=BITS),
    ]
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
