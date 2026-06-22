"""Unsharp mask: 2*center - 3x3 blur, saturated, replicate edges (same size).
Kept as one spatial sum (divide at the end) so it stays a single cone."""
import numpy as np


def model(img):
    x = np.pad(img.astype(np.int16), 1, mode="edge")
    blur_sum = (    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
                + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
                +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:])
    return ((32 * x[1:-1, 1:-1] - blur_sum) // 16).clip(0, 255).astype(np.uint8)
