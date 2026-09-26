# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Integer constants in the datapath: rounding, offsets, centring, inversion.

`x + 8` before a shift is how a divide rounds; `x - 128` centres a sample;
`255 - x` inverts it. Each is ordinary NumPy, and each must trace as written
-- a model that has to be rephrased to compile is a model that has stopped
being the specification. Validated against the SAME functions on real data.

Also here: `x >> k` and `x // 2**k` are one operation. Both floor, so they
must trace identically on every path -- including after a register multiply,
where the shift once took a different route and was refused.

A constant that carries the result out of the value's dtype is refused: NumPy
wraps there, and the traced expression would not. Widening first builds.

Run:  python examples/const_bias.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw import Image2D, Param, Rom, to_ir

W, H = 8, 6
IMG = Image2D("img", W, H, 8)
A = np.random.default_rng(7).integers(0, 256, (H, W), dtype=np.uint8)
GAIN = Param("gain", np.uint8, description="Q4.4 gain")


def plus_const(img):
    return (img.astype(np.uint16) + 8).astype(np.uint16)


def round_shift(img):
    """Divide by 16, rounding to nearest: the bias goes in before the shift."""
    return ((img.astype(np.uint16) + 8) // 16).astype(np.uint8)


def centre(img):
    """Signed centring, then back to unsigned by a clip."""
    return (img.astype(np.int16) - 128).clip(0, 255).astype(np.uint8)


def centre_signed(img):
    """Centring that KEEPS the sign: negative samples out."""
    return (img.astype(np.int16) - 128).astype(np.int16)


def invert(img):
    """int - traced: the reflected subtraction."""
    return (255 - img.astype(np.int16)).astype(np.uint8)


def const_first(img):
    """int + traced: the reflected addition, with a constant that is not 0."""
    return (8 + img.astype(np.uint16)).astype(np.uint16)


def gain_div(img, gain):
    return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)


def gain_shift(img, gain):
    """The same as gain_div, written with a shift."""
    return ((gain * img.astype(np.uint16)) >> 4).clip(0, 255).astype(np.uint8)


def gain_round(img, gain):
    """A register gain, rounded: arithmetic past the multiply."""
    return ((gain * img.astype(np.uint16) + 8) >> 4).clip(0, 255).astype(np.uint8)


KNOTS = Param("knots", np.uint8, shape=(16,), description="curve")


def lut_by_shift(img, knots):
    """A table indexed by a shift: >> and // must both reach a lookup."""
    return knots[img.astype(np.int32) >> 4].astype(np.uint8)


def lut_by_div(img, knots):
    return knots[img.astype(np.int32) // 16].astype(np.uint8)


CURVE = Rom([(i * i) & 0xFF for i in range(16)], name="curve")


def rom_by_shift(img):
    """A constant table indexed by a shift: the same law as the registers."""
    return CURVE[img.astype(np.int32) >> 4].astype(np.uint8)


def overflows_add(img):
    return img + 100                    # uint8: NumPy wraps past 255


def overflows_sub(img):
    return img - 10                     # uint8: NumPy wraps below 0


def overflows_rsub(img):
    return 10 - img


def refused(name, fn):
    try:
        to_ir(fn, IMG)
    except NotImplementedError as e:
        print(f"  {name:<12} refused: {str(e)[:56]}...  PASS")
        return True
    print(f"  {name:<12} BUILT -- NumPy wraps here  FAIL")
    return False


def main():
    print("integer constants and shifts (DUT vs NumPy):")
    pv = {"gain": 37}
    r = [
        check("plus_const", plus_const, A),
        check("round_shift", round_shift, A),
        check("centre", centre, A),
        check("centre_s", centre_signed, A),
        check("invert", invert, A),
        check("const_first", const_first, A),
        check("gain_div", gain_div, A, [GAIN], pv),
        check("gain_shift", gain_shift, A, [GAIN], pv),
        check("gain_round", gain_round, A, [GAIN], pv),
    ]
    kv = {f"knots_{i}": (i * 37 + 11) & 0xFF for i in range(16)}
    r += [
        check("lut_shift", lut_by_shift, A, [KNOTS], kv),
        check("lut_div", lut_by_div, A, [KNOTS], kv),
        check("rom_shift", rom_by_shift, A),
    ]
    print("constants past the dtype:")
    r += [
        refused("x + 100", overflows_add),
        refused("x - 10", overflows_sub),
        refused("10 - x", overflows_rsub),
    ]
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
