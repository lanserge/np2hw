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
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
