"""3x3 box blur, replicate edges (same size)."""
import numpy as np


def model(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((x[:-2, :-2] + x[:-2, 1:-1] + x[:-2, 2:]
             + x[1:-1, :-2] + x[1:-1, 1:-1] + x[1:-1, 2:]
             + x[2:, :-2] + x[2:, 1:-1] + x[2:, 2:]) // 8).astype(np.uint8)
