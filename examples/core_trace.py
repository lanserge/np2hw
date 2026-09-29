# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The new tracer: every model's graph, evaluated, equals the model.

Each model is traced into the graph, the graph is evaluated on NumPy arrays
-- each node the NumPy operation it records -- and the result must EQUAL
running the model on the same input, element for element, dtype included.
Equality proves the graph records everything the model computed.

Run:  python examples/core_trace.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw import Image2D, Param, Rom, coords, make_coords
from np2hw.ir import PhaseRef
from np2hw.core.evaluate import evaluate
from np2hw.core.graph import verify, walk
from np2hw.core.trace import trace

H, W = 6, 8
RNG = np.random.default_rng(5)
GAIN = Param("gain", np.uint8)
PY, PX = Param("py", bits=1), Param("px", bits=1)


def gain(img, g):
    return ((img.astype(np.uint16) * g) >> 4).clip(0, 255).astype(np.uint8)


def centre_wraps(img):
    return img - 128                          # uint8: NumPy wraps


def blur(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((x[:-2, :-2] + 2 * x[:-2, 1:-1] + x[:-2, 2:]
             + 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] + 2 * x[1:-1, 2:]
             + x[2:, :-2] + 2 * x[2:, 1:-1] + x[2:, 2:]) // 16).astype(np.uint8)


def two_pass(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    h = x[:, :-2] + 2 * x[:, 1:-1] + x[:, 2:]
    return (h[:-2] + 2 * h[1:-1] + h[2:]) // 16


def zero_diff(img):
    x = np.pad(img.astype(np.int16), ((0, 0), (1, 1)), mode="constant")
    return x[:, 2:] - x[:, :-2]


def plane(img, py, px):
    """A CFA plane: rows from the register py, columns from its complement.
    Run on NumPy, py and px are 0 or 1, and the slices are ordinary."""
    v = img.astype(np.int32)
    rows = PhaseRef(py) if isinstance(py, Param) else py
    cols = PhaseRef(px, invert=True) if isinstance(px, Param) else 1 - px
    return v[rows::2, cols::2] + 7


def select(img):
    v = img.astype(np.int16)
    return np.where(v > 100, v - 100, abs(v - 50))


def mask_pack(img):
    v = img.astype(np.int64)
    a, b = v & 0x0F, (v >> 4) & 0x0F
    return (b + a * 16).astype(np.uint8)


def one_row(img):
    return img[2, 1:-1].astype(np.int32) * 3


def shading(img):
    """Position used as DATA: a term from the row and column."""
    y, x = coords(img)
    return img.astype(np.int32) + (y * x) // 4


def crop(img):
    """A declared output raster, read through its coordinates plus offsets."""
    y2, x2 = make_coords(img, H - 2, W - 3)
    return img[y2 + 1, x2 + 2].astype(np.int16) * 2


KNOTS = Param("knots", np.uint8, shape=(16,))
CURVE = Rom([(i * i) % 251 for i in range(16)], name="curve")


def lookup(img, knots):
    return knots[img.astype(np.int32) >> 4] + 1


def rom_lookup(img):
    return CURVE[img.astype(np.int32) >> 4]


def _phase(p, invert=False):
    if isinstance(p, Param):
        return PhaseRef(p, invert=invert)
    return 1 - p if invert else p


def black_canvas(img, py, px):
    """A per-CFA-colour offset, as planes written into a canvas."""
    v = img.astype(np.int32)
    out = np.empty_like(v)
    for i, inv_r in enumerate((False, True)):
        for j, inv_c in enumerate((False, True)):
            rows, cols = _phase(py, inv_r), _phase(px, inv_c)
            out[rows::2, cols::2] = (v[rows::2, cols::2] + (10 * i + j)).clip(0, 255)
    return out.astype(np.uint16)


def black_coords(img, py, px):
    """The same stage, its phase computed from the pixel's position."""
    v = img.astype(np.int32)
    y, x = coords(img)
    i, j = (y + py) & 1, (x + px) & 1
    off = np.where(i == 0, np.where(j == 0, 0, 1), np.where(j == 0, 10, 11))
    return (v + off).clip(0, 255).astype(np.uint16)


def stacked(img):
    v = img.astype(np.uint16)
    return np.stack([v, v * 2, v >> 1], axis=-1)


def pair_sum(img):
    """A reshape that splits: pixel pairs."""
    p = img.astype(np.uint16).reshape(H, W // 2, 2)
    return p[:, :, 0] + p[:, :, 1]


def flat_diff(img):
    """A reshape that merges: the frame as one line."""
    f = img.astype(np.int32).reshape(-1)
    return f[1:] - f[:-1]


def three_channels(img):
    """A 3-channel word: pixel[..., c] is an ordinary integer index."""
    v = img.astype(np.uint16)
    return v[..., 0] + 2 * v[..., 2] - v[..., 1] // 2


MODELS = [("gain", gain, [GAIN], {"gain": 37}),
          ("centre_wraps", centre_wraps, [], {}),
          ("blur", blur, [], {}),
          ("two_pass", two_pass, [], {}),
          ("zero_diff", zero_diff, [], {}),
          ("plane_py0", plane, [PY, PX], {"py": 0, "px": 0}),
          ("plane_py1", plane, [PY, PX], {"py": 1, "px": 1}),
          ("select", select, [], {}),
          ("mask_pack", mask_pack, [], {}),
          ("one_row", one_row, [], {}),
          ("shading", shading, [], {}),
          ("crop", crop, [], {}),
          ("lookup", lookup, [KNOTS],
           {"knots": np.array([(7 * i + 3) % 256 for i in range(16)])}),
          ("rom_lookup", rom_lookup, [], {}),
          ("canvas_py0", black_canvas, [PY, PX], {"py": 0, "px": 1}),
          ("canvas_py1", black_canvas, [PY, PX], {"py": 1, "px": 0}),
          ("coords_py0", black_coords, [PY, PX], {"py": 0, "px": 1}),
          ("coords_py1", black_coords, [PY, PX], {"py": 1, "px": 0}),
          ("stacked", stacked, [], {}),
          ("pair_sum", pair_sum, [], {}),
          ("flat_diff", flat_diff, [], {}),
          ("3 channels", three_channels, [], {}, 3)]


def main():
    ok_all = True
    print("traced graph, evaluated, against the model itself:")
    for name, fn, params, values, *ch in MODELS:
        ch = ch[0] if ch else 1
        shape = (H, W) if ch == 1 else (H, W, ch)
        img = RNG.integers(0, 256, shape, dtype=np.uint8)
        t = trace(fn, Image2D("img", W, H, bits=8 * ch), *params, channels=ch)
        verify(t.g, [t.output], levels={"model"})
        (got,) = evaluate(t.g, [t.output], t.source, img, values)
        want = fn(img, *[values[p.name] for p in params])
        ok = (np.asarray(got).dtype == np.asarray(want).dtype
              and np.array_equal(got, want))
        nodes = sum(1 for _ in walk([t.output]))
        print(f"  {name:<14} {nodes:>3} nodes  {str(np.asarray(want).dtype):<6} "
              f"{'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    print("\n" + ("ALL PASS" if ok_all else "SOME FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
