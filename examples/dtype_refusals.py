# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Two compilers, one model: where they would differ, np2hw refuses.

The new compiler computes in NumPy's dtypes, so it wraps where NumPy wraps.
The reference computes a traced expression exactly; where a result would
leave its dtype, it refuses and says to widen -- a model and its hardware
that differ silently are worse than a refusal. The same holds when an option
(a clock budget, EOF framing, a run-time width) hands a model the new
compiler took to the reference instead: the reference must build the same
thing, or the option is refused.

Run:  python examples/dtype_refusals.py   (needs iverilog)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw import Image2D, generate, to_ir

W, H = 8, 6
A = np.random.default_rng(11).integers(0, 256, (H, W), dtype=np.uint8)
IMG = Image2D("img", W, H, 8)


def refused(name, thunk, kinds=(NotImplementedError, ValueError)):
    try:
        thunk()
    except kinds as e:
        print(f"  {name:<34} refused: {str(e)[:60]}...  PASS")
        return True
    print(f"  {name:<34} BUILT -- should have been refused  FAIL")
    return False


def built(name, thunk):
    try:
        thunk()
    except Exception as e:                          # noqa: BLE001
        print(f"  {name:<34} refused: {e}  FAIL")
        return False
    print(f"  {name:<34} builds  PASS")
    return True


def reference(fn):
    """Lower with the reference compiler alone."""
    old = os.environ.get("NP2HW_CORE")
    os.environ["NP2HW_CORE"] = "off"
    try:
        return to_ir(fn, IMG)
    finally:
        if old is None:
            del os.environ["NP2HW_CORE"]
        else:
            os.environ["NP2HW_CORE"] = old


def overflows_add(img):
    return img + 100                    # uint8: NumPy wraps past 255


def overflows_sub(img):
    return img - 10                     # uint8: NumPy wraps below 0


def overflows_rsub(img):
    return 10 - img


def widened_add(img):
    return img.astype(np.uint16) + 100


def wraps_inside(img):
    return (img >> 1) + 200             # uint8 throughout: wraps past 255


def main():
    print("the reference refuses what NumPy wraps:")
    r = [
        refused("x + k past uint8 (reference)", lambda: reference(overflows_add)),
        refused("x - k below 0 (reference)", lambda: reference(overflows_sub)),
        refused("k - x below 0 (reference)", lambda: reference(overflows_rsub)),
        built("widened first (reference)", lambda: reference(widened_add)),
    ]
    # max_width is an option the new compiler hands to the reference
    _, core = to_ir(wraps_inside, IMG)
    _, wide = to_ir(widened_add, IMG)
    r += [
        check("wraps_inside", wraps_inside, A),   # the new compiler wraps as NumPy does
        refused("...max_width forcing the reference",
                lambda: generate(core, "wraps", max_width=16)),
        built("widened model, max_width", lambda: generate(wide, "wide", max_width=16)),
    ]
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
