# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Mixed widths: a NARROWER sub-expression that wraps, meeting a WIDER one.

`2 * img` on a uint8 wraps at 8 bits; adding a widened term to it does not
undo the wrap. Every value carries its own dtype, so the wrap is kept
exactly where NumPy keeps it and the sum is as wide as its range -- the
hardware is the model's arithmetic, wraps included. Widening first is a
different model, with no wrap, and builds as one.

Each is proven bit-exact on frames whose brightest pixels do wrap.

Run:  python examples/mixed_width.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6


def mixed(img):
    a = 2 * img[1:, :]                   # uint8: 2*pixel WRAPS at 8 bits
    b = img[:-1, :].astype(np.uint16)    # widened
    return a + b                         # the wrap stays where NumPy put it


def widened(img):
    x = img.astype(np.uint16)            # widen FIRST: nothing wraps
    return 2 * x[1:, :] + x[:-1, :]


def main():
    frame = np.random.default_rng(31).integers(0, 256, (H, W)).astype(np.uint8)
    frame[:, :3] = 255                   # columns where 2*pixel wraps
    wraps = bool((mixed(frame) != widened(frame)).any())
    print(f"the two models differ on this frame: {wraps}")
    results = [wraps,
               check("mixed", mixed, frame),
               check("widened", widened, frame)]
    ok = all(results)
    print("\n" + ("MIXED WIDTH PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
