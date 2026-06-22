"""Known limitation: mixing a NARROWER sub-expression that would clip with a
WIDER one cannot be flattened faithfully, so the front-end refuses it with a
clear error (rather than silently producing a wrong result). The fix is the
idiomatic one: widen the operands first.

Run:  python examples/mixed_width.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw.ir import Image2D
from np2hw.frontend import to_ir


def mixed_bad(img):
    a = 2 * img[1:, :]                   # uint8: 2*pixel WRAPS at 8 bits
    b = img[:-1, :].astype(np.uint16)    # widened
    return a + b                         # narrow-clipping term meets wider -> refused


def mixed_ok(img):
    x = img.astype(np.uint16)            # widen FIRST -> no clip, flattenable
    return 2 * x[1:, :] + x[:-1, :]


def main():
    img = Image2D("img", 8, 6, bits=8)

    print("mixed_bad (narrow clip meets wider):")
    try:
        to_ir(mixed_bad, img)
        print("  ERROR: should have been refused")
        return 1
    except ValueError as e:
        print("  refused with clear error:")
        print("   ", str(e))

    print("\nmixed_ok (widen first):")
    to_ir(mixed_ok, img)
    print("  compiles fine (single dtype, flattenable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
