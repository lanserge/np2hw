# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The larger and the smaller, and reductions within a pixel's word.

`np.maximum` and `np.minimum` are each a compare and a select, their
ranges exact -- `np.maximum(x - 100, 0)` cannot be negative, so it may be
windowed and stacked. A reduction over a word's axes -- a source's
channels, a stack's, what slices and reshapes make of them -- is each
pixel's own arithmetic: the lanes combined pairwise, in the container
NumPy reduces in; `argmin` and `argmax` keep the first of equal lanes, as
NumPy does. Over the frame's axes as well, the rest is a fold
(tests/test_core_folds.py).

Each model is held to NumPy three ways: its Verilog under random
backpressure, its C, and -- reduced over the frame too -- its fold.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from np2hw import Image2D
from np2hw.core.cemit import compile_c
from np2hw.core.trace import Unsupported
from np2hw.ir import Param, np_dtype

HERE = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


WINDOWS = _module("test_core_windows")
FOLDS = _module("test_core_folds")
W, H = WINDOWS.W, WINDOWS.H

needs_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None,
                                    reason="needs iverilog")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None
                              and shutil.which("gcc") is None,
                              reason="needs a C compiler")


def _lanes(frames, channels):
    """Packed words as their channels, 8 bits each, channel 0 lowest."""
    return np.stack([(frames >> (8 * i)) & 255 for i in range(channels)],
                    axis=-1).astype(np.uint8)


def _check(tmp_path, fn, *, channels=1, params=(), values=None, seed=0):
    """`fn` bit-exact with NumPy, as Verilog and as C."""
    bits = 8 * channels
    frames = WINDOWS._frames(seed, bits=bits)
    pixels = _lanes(frames, channels) if channels > 1 \
        else frames.astype(np.uint8)
    args = [np_dtype(p.bits, p.signed).type(values[p.name]) for p in params] \
        if values else []
    want = [np.asarray(fn(f, *args)) for f in pixels]
    core = WINDOWS._compile(fn, params, bits=bits, channels=channels,
                            name=fn.__name__)
    mask = (1 << core["out_bits"]) - 1
    got = WINDOWS._sim(tmp_path, core, frames, values, seed=seed)
    words = [w.astype(np.int64) for w in want]
    if words[0].ndim == 3:              # channels, packed as the core packs
        field = core["interface"]["output"]["field_bits"]
        words = [sum(w[..., i] << (field * i) for i in range(w.shape[-1]))
                 for w in words]
    assert got == WINDOWS._beats([w & mask for w in words])
    if shutil.which("cc") or shutil.which("gcc"):
        k = compile_c(fn, Image2D("img", W, H, bits=bits), *params,
                      channels=channels, build=tmp_path / "c")
        regs = {p.name: values[p.name] for p in params} if values else {}
        for f, w in zip(pixels, want):
            c = k(f, **regs)
            assert c.dtype == w.dtype and np.array_equal(c, w)
    return core


# --------------------------------------------------------------------------- #
# np.maximum, np.minimum
# --------------------------------------------------------------------------- #

def relu(img):
    return np.maximum(img.astype(np.int16) - 100, 0)


def dilate(img):
    """A 3 x 3 max filter: the larger of the window's reads."""
    x = np.pad(img, 1, mode="edge")
    m = np.maximum(np.maximum(x[:-2, 1:-1], x[2:, 1:-1]),
                   np.maximum(x[1:-1, :-2], x[1:-1, 2:]))
    return np.maximum(m, x[1:-1, 1:-1])


FLOOR = Param("floor", np.int8)


def floored(img, floor):
    """The smaller of a signed value and a register."""
    return np.minimum(img.astype(np.int16) - 128, floor)


def relu_blur(img):
    """A window over the larger of a value and zero: never negative, so a
    window may keep it."""
    x = np.pad(np.maximum(img.astype(np.int16) - 100, 0), 1, mode="edge")
    return (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) >> 2


@needs_iverilog
@pytest.mark.parametrize("fn", [relu, dilate, relu_blur])
def test_the_larger_of_two(tmp_path, fn):
    _check(tmp_path, fn)


@needs_iverilog
def test_the_smaller_of_a_value_and_a_register(tmp_path):
    _check(tmp_path, floored, params=(FLOOR,), values={"floor": -20})


def test_the_ranges_are_exact():
    """What NumPy can give, no wider: the larger of [-100, 155] and 0 is
    [0, 155]; the smaller of [0, 255] and [-20, -20] is -20."""
    from np2hw.core.trace import trace
    image = Image2D("img", W, H, bits=8)
    out = trace(relu, image).output.type
    assert (out.lo, out.hi) == (0, 155)
    out = trace(lambda img: np.minimum(img, np.int16(-20)), image).output.type
    assert (out.lo, out.hi) == (-20, -20)
    out = trace(lambda img: np.minimum(img, 300), image).output.type
    assert (out.lo, out.hi) == (0, 255)


def test_a_maximum_with_zero_is_never_negative():
    """Its range is exact, so it may be stacked into a word, whose
    channels are unsigned -- a select's range would not be."""
    def stacked(img):
        x = img.astype(np.int16)
        return np.stack([np.maximum(x - 100, 0), x], axis=-1)
    core = WINDOWS._compile(stacked, name="stacked")
    assert core["interface"]["output"]["channels"] == 2


# --------------------------------------------------------------------------- #
# Reductions within a pixel's word
# --------------------------------------------------------------------------- #

def lanes_max(img):
    x = img.astype(np.int16)
    return np.stack([x, x + 3, 200 - x, x >> 1], axis=-1).max(axis=-1)


def lanes_min_signed(img):
    x = img.astype(np.int16)
    return np.stack([x - 300, 50 - x, x >> 2], axis=-1).min(axis=-1)


def lanes_sum(img):
    x = img.astype(np.int16)
    return np.stack([x, 3 * x, x - 7], axis=-1).sum(axis=-1)


def wrapping_sum(img):
    """In uint8, as `dtype` says: NumPy wraps each step, and so must this."""
    return np.stack([img, img, img], axis=-1).sum(axis=-1, dtype=np.uint8)


def reduce_min(img):
    x = img.astype(np.int16)
    return np.minimum.reduce(np.stack([x, 255 - x, (x >> 1) + 9], axis=-1),
                             axis=-1)


def nearest(img):
    """The nearest of five levels: argmin over the distances."""
    x = img.astype(np.int16)
    d = np.stack([abs(x - c) for c in (0, 60, 128, 190, 255)], axis=-1)
    return d.argmin(axis=-1)


def tied(img):
    """Lanes 0 and 1 always equal: the first of them wins, as NumPy's."""
    x = img.astype(np.int16)
    d = np.stack([abs(x - 40), abs(x - 40), abs(x - 200)], axis=-1)
    return np.argmax(d, axis=-1)


def tied_min(img):
    x = img.astype(np.int16)
    d = np.stack([abs(x - 200), x >> 3, x >> 3], axis=-1)
    return np.argmin(d, axis=-1)


def window_spread(img):
    """The spread of a cross of window reads, stacked and reduced."""
    x = np.pad(img, 1, mode="edge")
    taps = np.stack([x[:-2, 1:-1], x[2:, 1:-1], x[1:-1, :-2], x[1:-1, 2:],
                     x[1:-1, 1:-1]], axis=-1)
    return taps.max(axis=-1) - taps.min(axis=-1)


@needs_iverilog
@pytest.mark.parametrize("fn", [lanes_max, lanes_min_signed, lanes_sum,
                                wrapping_sum, reduce_min, nearest, tied,
                                tied_min, window_spread])
def test_a_reduction_over_a_stack(tmp_path, fn):
    _check(tmp_path, fn)


def source_max(px):
    return px.max(axis=-1)


def source_argmin(px):
    return px.argmin(axis=-1)


def reshaped(px):
    """Four channels as two pairs: each pair summed, the larger kept."""
    s = px.astype(np.uint16).reshape(*px.shape[:2], 2, 2).sum(axis=-1)
    return s.max(axis=-1)


@needs_iverilog
@pytest.mark.parametrize("fn", [source_max, source_argmin, reshaped])
def test_a_reduction_over_the_source_channels(tmp_path, fn):
    _check(tmp_path, fn, channels=4)


# --------------------------------------------------------------------------- #
# Written as NumPy writes them: newaxis, constant arrays
# --------------------------------------------------------------------------- #

CENTRES = np.array([0, 60, 128, 190, 255], dtype=np.int16)


def nearest_centre(img):
    """The nearest class, broadcast against a constant array of centres."""
    return np.abs(img.astype(np.int16)[..., None] - CENTRES).argmin(axis=-1)


def check_node(img):
    """One min-sum check node over six messages: the smallest magnitude,
    the next smallest -- the smallest with the first one's lane put out of
    reach -- and the sign, the parity of the negative ones."""
    x = img.astype(np.int16)
    m = np.stack([x - 128, 100 - x, (x >> 1) - 40, 7 - (x >> 2), x - 60,
                  90 - x], axis=-1)
    a = np.abs(m)
    first = a.argmin(axis=-1)
    min2 = np.where(np.arange(6) == first[..., None], 255, a).min(axis=-1)
    sign = (m < 0).sum(axis=-1) & 1
    return np.stack([a.min(axis=-1), min2, sign], axis=-1)


@needs_iverilog
@pytest.mark.parametrize("fn", [nearest_centre, check_node])
def test_numpy_as_written(tmp_path, fn):
    _check(tmp_path, fn)


# --------------------------------------------------------------------------- #
# Within the word, then over the frame: a fold
# --------------------------------------------------------------------------- #

def lanes_then_frame(img):
    x = img.astype(np.int16)
    return np.stack([x, 255 - x, x >> 1], axis=-1).max()


def lanes_then_rows(img):
    x = img.astype(np.int16)
    return np.stack([x, x - 300], axis=-1).sum(axis=(1, 2))


def class_counts(img):
    """Each pixel's nearest class, counted."""
    x = img.astype(np.int16)
    cls = np.stack([abs(x - c) for c in (30, 120, 220)],
                   axis=-1).argmin(axis=-1)
    acc = np.zeros(3, np.int32, like=img)
    np.add.at(acc, cls, 1)
    return acc


@needs_iverilog
@pytest.mark.parametrize("fn", [lanes_then_frame, lanes_then_rows,
                                class_counts])
def test_a_reduction_within_the_word_then_over_the_frame(tmp_path, fn):
    FOLDS._check(tmp_path, fn)


# --------------------------------------------------------------------------- #
# What is refused
# --------------------------------------------------------------------------- #

def test_an_argmin_over_the_frame_is_refused():
    with pytest.raises(Unsupported, match="over an axis of the frame"):
        WINDOWS._compile(lambda img: img.argmin(axis=1))


def test_an_argmin_without_an_axis_is_refused():
    with pytest.raises(Unsupported, match="name the word's axis"):
        WINDOWS._compile(lambda img: np.stack([img, img], axis=-1).argmin())
