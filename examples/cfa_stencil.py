"""Phase-selected STENCILS: `out[py::2, px::2] = taps[py::2, px::2]`.

The CFA interpolation case. cfa_phase.py showed per-plane COEFFICIENTS
(pointwise); here each plane applies a different TAP COMBINATION of one 3x3
window -- exactly what demosaic does: at a red site the green estimate is the
orthogonal average, at a green site it is the sample itself. Written in
NumPy the way anyone would: compute the combinations at full rate, then say
which positions each applies to.

Two things lower specially, and both are asserted below rather than claimed:

- The planes share ONE window. Four tap combinations do not cost four sets
  of line buffers; they cost one window and a positional mux (position
  parity XOR the phase registers -- the same programmable mapping as
  cfa_phase, so one bitstream serves every CFA order).
- `np.stack([r, g, b], axis=-1)` is how a model says "channels", and the
  channels ride the SAME window: C muxes, one select, one output word with
  channel 0 in the low bits. Models never shift-and-mask channels together;
  the word layout is the stream boundary's one statement.

Edges replicate via np.pad(mode="edge"), so output is full-size -- and
np.pad is the oracle too.

Run:  python examples/cfa_stencil.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H, BITS = 8, 6, 8
FULL = (1 << BITS) - 1

A = np.random.default_rng(11).integers(0, FULL + 1, (H, W)).astype(np.uint8)
A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 0, FULL, FULL, 0    # rails in the window


def interpolate(img, py, px):
    """Demosaic-shaped: five tap combinations routed to channels by CFA site.

    The SAME function is the hardware and the oracle: on real arrays the
    slicing, np.pad and np.stack are ordinary NumPy; traced, they become one
    window, a positional mux per channel, and one packed word.
    """
    value = img.astype(np.uint16)
    x = np.pad(value, 1, mode="edge")
    centre = x[1:-1, 1:-1]
    north, south = x[:-2, 1:-1], x[2:, 1:-1]
    west, east = x[1:-1, :-2], x[1:-1, 2:]
    cross = (north + south + west + east) // 4
    diag = (x[:-2, :-2] + x[:-2, 2:] + x[2:, :-2] + x[2:, 2:]) // 4
    horiz = (west + east) // 2
    vert = (north + south) // 2

    r = np.empty_like(value)
    g = np.empty_like(value)
    b = np.empty_like(value)
    sites = (
        ((py, px), centre, cross, diag),          # the measured-R site
        ((py, 1 - px), horiz, centre, vert),      # green in R's row
        ((1 - py, px), vert, centre, horiz),      # green in B's row
        ((1 - py, 1 - px), diag, cross, centre),  # the measured-B site
    )
    for (rows, cols), rt, gt, bt in sites:
        r[rows::2, cols::2] = rt[rows::2, cols::2]
        g[rows::2, cols::2] = gt[rows::2, cols::2]
        b[rows::2, cols::2] = bt[rows::2, cols::2]
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def green_recover(img, py, px):
    """The single-channel flavour: one canvas, no stack, still stencils."""
    value = img.astype(np.uint16)
    x = np.pad(value, 1, mode="edge")
    centre = x[1:-1, 1:-1]
    cross = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) // 4

    g = np.empty_like(value)
    g[py::2, px::2] = cross[py::2, px::2]              # G missing at R
    g[py::2, 1 - px::2] = centre[py::2, 1 - px::2]     # G measured
    g[1 - py::2, px::2] = centre[1 - py::2, px::2]     # G measured
    g[1 - py::2, 1 - px::2] = cross[1 - py::2, 1 - px::2]  # G missing at B
    return g.astype(np.uint8)


def main():
    params = [
        Param("py", bits=1, description="Row parity the first colour sits on"),
        Param("px", bits=1, description="Column parity the first colour sits on"),
    ]

    print(f"{BITS}-bit phase-selected stencils, {W}x{H}; ONE build, every phase:")
    results = []
    for phase in range(4):
        values = {"py": (phase >> 1) & 1, "px": phase & 1}
        results.append(check(f"cfa_stencil{phase}", interpolate, A,
                             params=params, param_values=values, bits=BITS))
    print("\nsingle-channel flavour (one canvas, no stack):")
    for phase in (0, 3):
        values = {"py": (phase >> 1) & 1, "px": phase & 1}
        results.append(check(f"cfa_green{phase}", green_recover, A,
                             params=params, param_values=values, bits=BITS))

    print("\nthe claim, stated as something that can fail:")
    from np2hw import Image2D, generate, to_ir

    _, out = to_ir(interpolate, Image2D("img", W, H, bits=BITS), *params)
    core = generate(out, module_name="cfa_stencil")
    # FOUR planes x THREE channels over one 3x3 window: exactly TWO line
    # buffers (one window, shared), three positional muxes (one per
    # channel), and the select follows the OUTPUT pixel's position through
    # the edge flushes, XORed with the phase registers.
    buffers = core.verilog.count("] mem")
    muxes = core.verilog.count("case ({sel_row, sel_col})")
    positional = "wire sel_row = o_row[0] ^ param_py" in core.verilog
    print(f"  {core['phases']} planes x {core['channels']} channels -> "
          f"{buffers} line buffer(s), {muxes} channel mux(es), "
          f"out_bits={core.out_bits}")
    print(f"  select follows the output pixel, phase-programmable: {positional}")
    shape_ok = (buffers == 2 and muxes == 3 and core.out_bits == 3 * BITS
                and positional and core.line_buffers == 2)

    print("\n" + ("CFA STENCIL PASS" if all(results) and shape_ok else "FAIL"))
    return 0 if all(results) and shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
