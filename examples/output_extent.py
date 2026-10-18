# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A crop states the map from output position to input position, and the
core is built from that map.

A slice states it: `img[2:6, 1:7]` on an 8x8 reads input (r + 2, c + 1)
for output (r, c), over a 4x6 output. So does `make_coords`, which states
the OUTPUT raster, so indexing the input with those coordinates states the
map:

    y2, x2 = make_coords(img, 4, 4)
    return img[y2 + 2, x2 + 2]        # output (r,c) reads input (r+2,c+2)

The map is what a core needs. The offset becomes the tap, the extent
becomes the gate, and the framing marks the window's corners rather than
the frame's. The input rows and columns past the window are still taken
from the stream -- the next frame begins where this one ends -- and never
read. A shape alone could be inferred wrongly; a map cannot, so the core's
extent is the model's, checked here as the beats it emits.

Run:  python examples/output_extent.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw import Image2D, generate, make_coords, to_ir

W, H = 8, 8
A = np.random.default_rng(20260822).integers(0, 256, (H, W), dtype=np.uint8)


def declared(img):
    y2, x2 = make_coords(img, 4, 4)
    return img[y2 + 2, x2 + 2]


MODELS = [
    ("crop_declared", declared),
    ("crop_corner", lambda img: img[0:4, 0:4]),
    ("crop_mid", lambda img: img[2:6, 1:7]),
    ("lone_slice", lambda img: img[0:-2, 0:-2]),
    ("whole", lambda img: img[:, :] // 2),
    ("stencil", lambda img: (img[0:-2, 0:-2] + img[2:, 2:]) // 2),
]


def main():
    results = []
    for name, fn in MODELS:
        want = fn(A).shape
        _, out = to_ir(fn, Image2D("img", W, H, bits=8))
        core = generate(out, module_name=name)
        extent = (core["out_rows"], core["out_cols"]) == want
        print(f"  {name:<13} extent {want[0]}x{want[1]} "
              f"{'PASS' if extent else 'FAIL'}")
        results += [extent, check(name, fn, A)]
    assert declared(A)[0, 0] == A[2, 2]
    ok = all(results)
    print("\n" + ("OUTPUT EXTENT PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
