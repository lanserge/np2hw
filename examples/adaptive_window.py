"""Window EXPRESSIONS: abs, comparisons, per-pixel select, lane taps.

The adaptive-filter vocabulary, in the two shapes a two-stage CFA
interpolator needs (Hamilton-Adams' shape -- the 1995 Kodak method, long
expired -- but the capability is generic to every adaptive filter):

  stage A: gradients with np.abs, a data-driven np.where between a
           horizontally-corrected, a vertically-corrected and an
           averaged estimate, phase-routed; the raw sample rides along
           as a BROADCAST channel of the output word.
  stage B: window taps over the LANES of that 2-channel word -- colour
           differences (raw - green) interpolated per site. Field
           extraction happens per tap; no model touches the word.

Every plane's DAG shares ONE wire namespace, so a gradient used by two
planes -- or two channels -- is computed once.

Run:  python examples/adaptive_window.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H, BITS = 12, 8, 8
FULL = (1 << BITS) - 1

A = np.random.default_rng(23).integers(0, FULL + 1, (H, W)).astype(np.uint8)
A[0, :6] = (0, FULL, FULL, 0, 0, FULL)                   # hard edges

PARAMS = [Param("py", bits=1, description="Row parity the first colour sits on"),
          Param("px", bits=1, description="Column parity the first colour sits on")]


def green_adaptive(img, py, px):
    """Stage A: direction-adaptive green, raw riding along. 5x5 window."""
    v = img.astype(np.int32)
    x = np.pad(v, 2, mode="edge")

    def at(r, c):
        return x[2 + r:2 + r + H, 2 + c:2 + c + W]

    gw, ge, gn, gs = at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    cc = at(0, 0)
    lh = 2 * cc - at(0, -2) - at(0, 2)
    lv = 2 * cc - at(-2, 0) - at(2, 0)
    dh = np.abs(gw - ge) + np.abs(lh)
    dv = np.abs(gn - gs) + np.abs(lv)
    gh = ((gw + ge) // 2 + lh // 4).clip(0, FULL)
    gv = ((gn + gs) // 2 + lv // 4).clip(0, FULL)
    ga = ((gw + ge + gn + gs) // 4 + (lh + lv) // 8).clip(0, FULL)
    est = np.where(dh < dv, gh, np.where(dv < dh, gv, ga))

    green = np.empty_like(v)
    green[py::2, px::2] = est[py::2, px::2]              # G missing at R
    green[py::2, 1 - px::2] = cc[py::2, 1 - px::2]       # measured
    green[1 - py::2, px::2] = cc[1 - py::2, px::2]       # measured
    green[1 - py::2, 1 - px::2] = est[1 - py::2, 1 - px::2]  # missing at B
    return np.stack([cc, green], axis=-1).astype(np.uint8)


def rb_from_differences(word, py, px):
    """Stage B: R/B from colour differences against the green lane. 3x3."""
    raw = np.pad(word[..., 0].astype(np.int32), 1, mode="edge")
    grn = np.pad(word[..., 1].astype(np.int32), 1, mode="edge")

    def R(r, c):
        return raw[1 + r:1 + r + H, 1 + c:1 + c + W]

    def G(r, c):
        return grn[1 + r:1 + r + H, 1 + c:1 + c + W]

    def d(r, c):
        return R(r, c) - G(r, c)

    gc = G(0, 0)
    cent = R(0, 0)
    horiz = (gc + (d(0, -1) + d(0, 1)) // 2).clip(0, FULL)
    vert = (gc + (d(-1, 0) + d(1, 0)) // 2).clip(0, FULL)
    diag = (gc + (d(-1, -1) + d(-1, 1) + d(1, -1) + d(1, 1)) // 4).clip(0, FULL)

    red = np.empty_like(cent)
    blue = np.empty_like(cent)
    sites = (((py, px), cent, diag),          # R site
             ((py, 1 - px), horiz, vert),     # green in R's row
             ((1 - py, px), vert, horiz),     # green in B's row
             ((1 - py, 1 - px), diag, cent))  # B site
    for (rows, cols), r_tap, b_tap in sites:
        red[rows::2, cols::2] = r_tap[rows::2, cols::2]
        blue[rows::2, cols::2] = b_tap[rows::2, cols::2]
    return np.stack([red, gc, blue], axis=-1).astype(np.uint8)


def main():
    print(f"stage A: adaptive select over a 5x5 window, {W}x{H}:")
    results = []
    for phase in range(4):
        values = {"py": (phase >> 1) & 1, "px": phase & 1}
        results.append(check(f"adapt_g{phase}", green_adaptive, A,
                             params=PARAMS, param_values=values, bits=BITS))

    print("\nstage B: lane-tap stencils over the 2-channel word:")
    word = green_adaptive(A, 0, 0)                       # a real stage-A frame
    for phase in (0, 3):
        values = {"py": (phase >> 1) & 1, "px": phase & 1}
        results.append(check(f"adapt_rb{phase}", rb_from_differences, word,
                             params=PARAMS, param_values=values,
                             bits=2 * BITS, channels=2))

    print("\nthe claim, stated as something that can fail:")
    from np2hw import Image2D, generate, to_ir

    _, out = to_ir(green_adaptive, Image2D("img", W, H, bits=BITS), *PARAMS)
    core = generate(out, module_name="adapt_g")
    # dh/dv feed the select in BOTH written planes: one namespace means the
    # gradients appear once, and the select is a comparator plus a mux.
    selects = core.verilog.count("!= 0) ?")
    _, out2 = to_ir(rb_from_differences, Image2D("word", W, H, bits=2 * BITS),
                    *PARAMS, channels=2)
    core2 = generate(out2, module_name="adapt_rb")
    lane_taps = core2.verilog.count(":8]")               # green-lane slices
    print(f"  A: {core.line_buffers} line buffers, {selects} selects, "
          f"out {core.out_bits}b; B: {core2.line_buffers} line buffers, "
          f"{lane_taps} high-lane taps, out {core2.out_bits}b")
    shape_ok = (core.line_buffers == 4 and core.out_bits == 2 * BITS
                and selects >= 2 and core2.line_buffers == 2
                and core2.out_bits == 3 * BITS and lane_taps >= 5)

    print("\n" + ("ADAPTIVE WINDOW PASS" if all(results) and shape_ok
                  else "FAIL"))
    return 0 if all(results) and shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
