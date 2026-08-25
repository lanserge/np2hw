# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""`saturate(x, bits)`: bound a value to what fits in a width, not to a
number that happens to correspond to one.

`clip(lo, hi)` is the general operation and stays the right one for an
arbitrary bound. A DATAPATH bound is not arbitrary -- it is a width --
and saying so buys an exact traced range: a clip to 1020 and a clip to
1023 both report ten bits, so a width alone cannot say where full scale
sits, while a saturation stated in bits can only ever produce
2**bits - 1. The emitted width IS the scale, and a consumer downstream
can derive one from the other instead of being told separately.

Checked three ways: against the SAME function on real NumPy data through
simulation, for the emitted width, and for agreement with the clip it
replaces.

Run:  python examples/saturate.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.frontend import saturate, to_ir
from np2hw.ir import Image2D
from np2hw.verilog import generate

W, H = 8, 6
A = np.random.default_rng(20260816).integers(0, 256, (H, W), dtype=np.uint8)


def gain_sat(img):
    """A gain that overflows, bounded by the width it has to fit."""
    return saturate(img.astype(np.uint16) * 3, 8).astype(np.uint8)


def gain_clip(img):
    """The same bound, spelled as the value it corresponds to."""
    return (img.astype(np.uint16) * 3).clip(0, 255).astype(np.uint8)


def narrow(img):
    """Saturating into a NARROWER width than the input: 8 bits -> 6."""
    return saturate(img.astype(np.uint16) * 3, 6).astype(np.uint8)


def _emitted_bits(fn, in_bits=8):
    _s, out = to_ir(fn, Image2D("img", W, H, in_bits))
    return generate(out, module_name="probe")["out_bits"]


def main():
    print("saturate(x, bits) against the same function on real data:")
    r = [check("gain_sat", gain_sat, A),
         check("gain_clip", gain_clip, A),
         check("narrow", narrow, A)]

    # The oracle for the refactor: stating a bound as a width must not
    # change what the bound IS.
    same = np.array_equal(gain_sat(A), gain_clip(A))
    print(f"\nsaturate(_, 8) == clip(0, 255): {same}")
    r.append(same)

    # The property the width-based form exists for. An arbitrary bound
    # rounds UP to a width and loses which one it was; a width does not.
    for fn, bits, label in ((gain_sat, 8, "saturate(_, 8)"),
                            (narrow, 6, "saturate(_, 6)")):
        got = _emitted_bits(fn)
        ok = got == bits
        print(f"{label:<16} emits {got} bits, wanted {bits}: {ok}")
        r.append(ok)

    odd = _emitted_bits(lambda i: (i.astype(np.uint16) * 3).clip(0, 1020))
    print(f"clip(0, 1020)    emits {odd} bits -- a width cannot say 1020, "
          "which is the reason saturate exists")
    r.append(odd == 10)

    for bad in (0, -1):
        try:
            saturate(A, bad)
        except ValueError as exc:
            print(f"refuses {bad} bits: {exc}")
            r.append(True)
        else:
            print(f"FAILED to refuse {bad} bits")
            r.append(False)

    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
