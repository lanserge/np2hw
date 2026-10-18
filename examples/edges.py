# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Edge handling: same-height output by replicating first/last line (np.pad).

A vertical [1,2,1] filter normally shrinks height by 2. Padding the rows with
np.pad(..., 'edge') replicates the top/bottom lines so the output keeps full
height. Validated against the SAME NumPy function (np.pad is the oracle too).

Run:  python examples/edges.py   (needs iverilog)
"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check

W, H = 8, 6
A = np.random.default_rng(21).integers(0, 256, (H, W), dtype=np.uint8)


def blur121_edge(img):
    """Vertical [1,2,1]/4, replicate edges -> SAME height as input."""
    x = np.pad(img.astype(np.uint16), ((1, 1), (0, 0)), mode="edge")
    return ((x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]) // 4).astype(np.uint8)


def blur121_zero(img):
    """Same, but zero-padded borders (np.pad constant)."""
    x = np.pad(img.astype(np.uint16), ((1, 1), (0, 0)), mode="constant")
    return ((x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]) // 4).astype(np.uint8)


def avg5_edge(img):
    """5-tap vertical [1,1,1,1,1], replicate edges, 2 line buffers each side."""
    x = np.pad(img.astype(np.uint16), ((2, 2), (0, 0)), mode="edge")
    return ((x[:-4, :] + x[1:-3, :] + x[2:-2, :] + x[3:-1, :] + x[4:, :]) // 4
            ).astype(np.uint8)


def hblur_edge(img):
    """Horizontal [1,2,1]/4, replicate left/right -> SAME width."""
    x = np.pad(img.astype(np.uint16), ((0, 0), (1, 1)), mode="edge")
    return ((x[:, :-2] + 2 * x[:, 1:-1] + x[:, 2:]) // 4).astype(np.uint8)


def gaussian2d_edge(img):
    """Full 3x3 Gaussian /16, replicate ALL borders -> SAME height AND width."""
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def hblur_zero(img):
    """Horizontal [1,2,1]/4, ZERO borders -> same width, dark edges."""
    x = np.pad(img.astype(np.uint16), ((0, 0), (1, 1)), mode="constant")
    return ((x[:, :-2] + 2 * x[:, 1:-1] + x[:, 2:]) // 4).astype(np.uint8)


def gaussian2d_zero(img):
    """Full 3x3 Gaussian /16, ZERO all borders -> same size."""
    x = np.pad(img.astype(np.uint16), 1, mode="constant")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def hdiff_zero_signed(img):
    """Horizontal difference, ZERO borders, SIGNED arithmetic: negative out.

    Signed taps read an unsigned pixel through a {1'b0, px} concatenation, so
    the left-edge zero mux sits INSIDE a concatenation -- where an unsized 0
    is illegal (indefinite width). The unsigned cases above never reach it."""
    x = np.pad(img.astype(np.int16), ((0, 0), (1, 1)), mode="constant")
    return (x[:, 2:] - x[:, :-2]).astype(np.int16)


def laplace8_zero_signed(img):
    """8-neighbour Laplacian, ZERO all borders, signed: both edge muxes."""
    x = np.pad(img.astype(np.int16), 1, mode="constant")
    return (8 * x[1:-1, 1:-1] - x[:-2, :-2] - x[:-2, 1:-1] - x[:-2, 2:]
            - x[1:-1, :-2] - x[1:-1, 2:] - x[2:, :-2] - x[2:, 1:-1] - x[2:, 2:]
            ).astype(np.int16)


def vblur_padded_all(img):
    """Vertical [1,2,1] on an image padded on ALL sides: the kernel reads
    column 1 of the padded image and nothing else, so its offsets start past
    zero. Normalised, with the column padding it never reads dropped."""
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((x[:-2, 1:-1] + 2 * x[1:-1, 1:-1] + x[2:, 1:-1]) // 4).astype(np.uint8)


def hblur_padded_all(img):
    """The horizontal twin, zero borders: row 1 of the padded image only."""
    x = np.pad(img.astype(np.uint16), 1, mode="constant")
    return ((x[1:-1, :-2] + 2 * x[1:-1, 1:-1] + x[1:-1, 2:]) // 4).astype(np.uint8)


def laplace4_zero_signed(img):
    """4-neighbour Laplacian: no corner tap, so no offset reaches (0, 0)."""
    x = np.pad(img.astype(np.int16), 1, mode="constant")
    return (4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1]
            - x[1:-1, :-2] - x[1:-1, 2:]).astype(np.int16)


def cross_in_wide_pad(img):
    """A 3x3 cross inside a pad of 2: reads only the middle of the padding."""
    x = np.pad(img.astype(np.uint16), 2, mode="edge")
    return ((x[1:-3, 2:-2] + x[3:-1, 2:-2] + x[2:-2, 1:-3] + x[2:-2, 3:-1])
            // 4).astype(np.uint8)


def forward_diff_padded(img):
    """A difference reaching forward on a pad of 1: its left padding unread."""
    x = np.pad(img.astype(np.int16), 1)
    return (x[1:-1, 2:] - x[1:-1, 1:-1]).clip(0, 255).astype(np.uint8)


def backward_diff_padded(img):
    """Its mirror reads only the TOP-LEFT padding, and its output is signed."""
    x = np.pad(img.astype(np.int16), 1)
    return x[1:-1, 1:-1] - x[1:-1, :-2]


def main():
    print("edge handling (full-size output, DUT vs np.pad oracle):")
    r = [
        check("blur121_edge", blur121_edge, A),     # vertical replicate
        check("blur121_zero", blur121_zero, A),     # vertical zero
        check("avg5_edge", avg5_edge, A),           # 5-tap vertical
        check("hblur_edge", hblur_edge, A),         # horizontal replicate
        check("gaussian2d_edge", gaussian2d_edge, A),  # full 2D replicate
        check("hblur_zero", hblur_zero, A),         # horizontal zero
        check("gaussian2d_zero", gaussian2d_zero, A),  # full 2D zero
        check("hdiff_zero_s", hdiff_zero_signed, A),   # signed, left zero
        check("lap8_zero_s", laplace8_zero_signed, A), # signed, all zero
        check("vblur_padall", vblur_padded_all, A),    # 1-D in a 2-D pad
        check("hblur_padall", hblur_padded_all, A),
        check("lap4_zero_s", laplace4_zero_signed, A), # no corner tap
        check("cross_pad2", cross_in_wide_pad, A),     # unread padding
        check("fwd_diff_pad", forward_diff_padded, A), # left padding unread
        check("bwd_diff_pad", backward_diff_padded, A), # left padding only
    ]
    oh, ow = gaussian2d_edge(A).shape
    print(f"\n2D gaussian output = {oh}x{ow} (input {H}x{W}) -> "
          + ("SAME SIZE" if (oh, ow) == (H, W) else "shrunk!"))

    # The line buffers must be read THROUGH A REGISTER, so they land in
    # block RAM instead of distributed RAM behind a select tree that
    # deepens with the line. Every check above stays green either way --
    # the behaviour is identical and only the hardware differs -- so the
    # emitted structure is what has to be asserted.
    from np2hw import Image2D, to_ir, generate
    _, out = to_ir(gaussian2d_edge, Image2D("img", H, W, bits=8))
    core = generate(out, module_name="edge_shape")
    src = core["verilog"]
    # The line buffers live behind the memory seam: a *_linebuf module
    # whose read port is registered. That is the block-RAM shape, stated
    # once per module and published with every memory.
    seam = {m["module"] for m in core["memories"]}
    registered = (len(core["memories"]) == core.line_buffers
                  and all(m["read"] == "sync" for m in core["memories"])
                  and all(f"module {name}" in src for name in seam)
                  and "if (re) rd <= mem[ra];" in src)
    # The read is issued a column ahead, so its cycle of latency costs no
    # register: the datapath keeps the window's shift registers, one row
    # of N per row of the window, and the output word -- nothing more.
    top = src[:min(src.index(f"module {name}") for name in seam)]
    data_regs = len(re.findall(r"^\s*reg \[7:0\]", top, re.M))
    one_ahead = data_regs == (core.line_buffers + 1) * core["N"] + 1
    # the async form is a CONTINUOUS read of an array, which the seam
    # never makes
    async_read = re.search(r"^\s*(wire|assign)\b.*\bmem\w*\[", src,
                           re.M) is not None
    print(f"  line buffers read through a register           "
          f"{'PASS' if registered else 'FAIL'}")
    print(f"  the address is one column AHEAD, no shift       "
          f"{'PASS' if one_ahead else 'FAIL'}")
    print(f"  no asynchronous read survives                   "
          f"{'PASS' if not async_read else 'FAIL'}")
    r += [registered, one_ahead, not async_read]
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
