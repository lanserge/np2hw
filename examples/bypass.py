# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Boolean block-enable via np.where(enable, A, B): a per-pixel 2:1 mux driven
by a 1-bit register. Both branches share one window, so they are latency-aligned
(a tap is delay-matched within the window) -- the bypass needs no separate delay
line. Validated at enable=1 and enable=0 against the SAME NumPy function.

Run:  python examples/bypass.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H = 8, 6
A = np.random.default_rng(91).integers(0, 256, (H, W), dtype=np.uint8)


def blur(img):
    x = img.astype(np.uint16)
    return ((x[:-2, :-2] + x[:-2, 1:-1] + x[:-2, 2:]
             + x[1:-1, :-2] + x[1:-1, 1:-1] + x[1:-1, 2:]
             + x[2:, :-2] + x[2:, 1:-1] + x[2:, 2:]) // 16).astype(np.uint8)


def bypass(img, en):
    """enable ? 3x3 blur : the original (center) pixel, delay-matched."""
    return np.where(en, blur(img), img[1:-1, 1:-1])


def pick_filter(img, en):
    """enable ? blur : a brighter 2x gain of the center -- two branches, mux."""
    bright = (2 * img[1:-1, 1:-1].astype(np.uint16)).clip(0, 255).astype(np.uint8)
    return np.where(en, blur(img), bright)


def main():
    en = Param("enable", bits=1)
    print("np.where(enable, ...) block mux:")
    r = [
        check("bypass_on", bypass, A, [en], {"enable": 1}),
        check("bypass_off", bypass, A, [en], {"enable": 0}),
        check("pick_on", pick_filter, A, [en], {"enable": 1}),
        check("pick_off", pick_filter, A, [en], {"enable": 0}),
    ]
    # A mux model is a module like any other: generate() hands back a Core
    # whose interface a composer reads, on this path as on every other.
    from np2hw import Image2D, generate, to_ir
    from np2hw.verilog import Core
    _, out = to_ir(bypass, Image2D("img", W, H, bits=8), en)
    core = generate(out, module_name="bypass_core")
    ports = core.interface if isinstance(core, Core) else {}
    typed = (isinstance(core, Core) and ports.get("output") is not None
             and [n for n, _ in core["registers"]] == ["enable"]
             and [p[0] for p in ports.get("params", ())] == ["enable"]
             and [f[0] for f in ports["input"]["ctx"]] == ["width", "height"])
    print(f"  generate() returns a Core with its interface   "
          f"{'PASS' if typed else 'FAIL'}")
    r.append(typed)
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
