"""Edge handling: same-height output by replicating first/last line (np.pad).

A vertical [1,2,1] filter normally shrinks height by 2. Padding the rows with
np.pad(..., 'edge') replicates the top/bottom lines so the output keeps full
height. Validated against the SAME NumPy function (np.pad is the oracle too).

Run:  python examples/edges.py   (needs iverilog)
"""
import sys, os
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
    src = generate(out, module_name="edge_shape")["verilog"]
    registered = "chain1_q <= mem1[rd_col]" in src
    one_ahead = "rd_col = in_sof" in src
    # a READ is "= mem1[...]"; a write is "mem1[ecol] <= ...", so look
    # for the read form anywhere, not merely before the first rd_col
    async_read = "= mem1[ecol]" in src
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
