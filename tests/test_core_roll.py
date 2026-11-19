# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""`np.roll`: an axis read at `(i - shift) mod n`.

On a pixel's word the lanes come in another order -- wiring, a stack of
them -- and hardware and C both take it. Along the frame it is a read that
wraps around a row or the frame, which the C backend takes and hardware,
needing the whole row or frame stored first, refuses by name.

A quasi-cyclic min-sum iteration -- the rotations of a code's circulant
blocks, its check nodes' smallest magnitudes and signs, the messages
rolled back and summed -- is the case the C backend is for: held to NumPy
here, and written into its output in the layout its reshape reads.
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

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_reductions",
                                               HERE / "test_core_reductions.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

needs_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None,
                                    reason="needs iverilog")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None
                              and shutil.which("gcc") is None,
                              reason="needs a C compiler")


# --------------------------------------------------------------------------- #
# On a word: the lanes in another order
# --------------------------------------------------------------------------- #

def roll_channels(px):
    return np.roll(px, 1, axis=-1)


def roll_stack(img):
    x = img.astype(np.int16)
    s = np.stack([x, x + 1, x + 2, 255 - x, x >> 1], axis=-1)
    return np.roll(s, -2, axis=-1).argmax(axis=-1)


@needs_iverilog
def test_a_roll_of_the_source_channels(tmp_path):
    R._check(tmp_path, roll_channels, channels=4)


@needs_iverilog
def test_a_roll_of_a_stack(tmp_path):
    R._check(tmp_path, roll_stack)


# --------------------------------------------------------------------------- #
# Along the frame: C, and a refusal in hardware
# --------------------------------------------------------------------------- #

def roll_columns(img):
    return np.roll(img, 3, axis=1)


def roll_both(img):
    return np.roll(img.astype(np.int16), (-2, 5), axis=(0, 1)) - img


@needs_cc
@pytest.mark.parametrize("fn", [roll_columns, roll_both])
def test_a_roll_along_the_frame_in_c(tmp_path, fn):
    k = compile_c(fn, Image2D("img", 40, 7, bits=8), build=tmp_path)
    for seed in range(3):
        img = np.random.default_rng(seed).integers(0, 256, (7, 40)).astype(
            np.uint8)
        want = fn(img)
        got = k(img)
        assert got.dtype == want.dtype and np.array_equal(got, want)


def test_a_roll_along_the_frame_is_refused_in_hardware():
    with pytest.raises(NotImplementedError, match="window read"):
        R.WINDOWS._compile(roll_columns)


def test_a_roll_without_an_axis_is_refused():
    with pytest.raises(Unsupported, match="name the axes"):
        R.WINDOWS._compile(lambda img: np.roll(img, 1))


# --------------------------------------------------------------------------- #
# A quasi-cyclic min-sum iteration
# --------------------------------------------------------------------------- #

Z, NB, B = 32, 6, 24
#: each row block's circulants: (column block, shift)
BASE = [[(0, 1), (1, 5), (3, 2), (4, 7)],
        [(1, 3), (2, 0), (4, 6), (5, 4)],
        [(0, 6), (2, 2), (3, 5), (5, 1)]]


def min_sum(x, base=BASE, nb=NB):
    """Every check's message to every variable, from the channel LLRs,
    summed into the posteriors: for each row block its variables rotated
    into check order, the smallest and next smallest magnitude and the sign
    parity, and each message rotated back."""
    llr = x.astype(np.int16) - 128
    blocks = llr.reshape(llr.shape[0], nb, Z)
    into = [[] for _ in range(nb)]
    for row in base:
        v = np.stack([np.roll(blocks[:, c], -s, axis=-1) for c, s in row],
                     axis=-1)
        a = np.abs(v)
        first = a.argmin(axis=-1)
        min1 = a.min(axis=-1)
        min2 = np.where(np.arange(len(row)) == first[..., None], 32767,
                        a).min(axis=-1)
        parity = (v < 0).sum(axis=-1) & 1
        for e, (c, s) in enumerate(row):
            mag = np.where(first == e, min2, min1)
            negative = parity ^ (v[..., e] < 0)
            into[c].append(np.roll(np.where(negative == 1, -mag, mag), s,
                                   axis=-1))
    post = []
    for c in range(nb):
        p = blocks[:, c]
        for m in into[c]:
            p = p + m
        post.append(p)
    return np.stack(post, axis=1).reshape(llr.shape[0], nb * Z)


@needs_cc
def test_a_min_sum_iteration_in_c(tmp_path):
    k = compile_c(min_sum, Image2D("llr", NB * Z, B, bits=8), build=tmp_path)
    for seed in range(3):
        x = np.random.default_rng(seed).integers(0, 256, (B, NB * Z)).astype(
            np.uint8)
        want = min_sum(x)
        got = k(x)
        assert got.dtype == want.dtype and np.array_equal(got, want)
    # the reshape moves nothing: the posteriors are written block by block,
    # never through a division of the flat column
    [source] = tmp_path.glob("*.c")
    text = source.read_text()
    loops = [l for l in text.splitlines() if "for (int64_t" in l]
    assert not any(f"< {NB * Z}; ++" in l for l in loops), loops
    # and the rolls are cut where they wrap: no quotient by Z is left, each
    # segment's reads running along memory
    assert f"/ {Z})" not in text and f"% {Z}" not in text


def many_rolls(img):
    """Rolls by several shifts, both ways, along both axes, summed."""
    x = img.astype(np.int16)
    return (np.roll(x, 3, axis=1) - np.roll(x, -7, axis=1)
            + np.roll(x, (2, -1), axis=(0, 1)) + np.roll(x, 39, axis=1))


def too_many_wraps(img):
    """A remainder by 5 along a row wraps every fifth pixel: too many cuts,
    so the loop keeps its general body."""
    from np2hw import coords
    _, col = coords(img)
    return img.astype(np.int16) + col % 5 - (col // 5 & 1)


def quotient_per_segment(img):
    """A quotient constant within each segment, and arithmetic on that
    constant alone: a value of its segment, not shared between them."""
    from np2hw import coords
    _, col = coords(img)
    return img.astype(np.int16) + (col // 160 & 1) + (col // 160) * 3


@needs_cc
@pytest.mark.parametrize("fn", [many_rolls, too_many_wraps,
                                quotient_per_segment])
def test_a_loop_cut_where_it_wraps(tmp_path, fn):
    k = compile_c(fn, Image2D("img", 640, 9, bits=8), build=tmp_path)
    for seed in range(3):
        img = np.random.default_rng(seed).integers(0, 256, (9, 640)).astype(
            np.uint8)
        want = fn(img)
        got = k(img)
        assert got.dtype == want.dtype and np.array_equal(got, want)


#: more column blocks than an axis is unrolled for: the blocks are a loop
WIDE = 20
WIDE_BASE = [[(c, (7 * c + 3 * r) % Z) for c in range(WIDE) if (c + r) % 3]
             for r in range(3)]


def wide_min_sum(x):
    return min_sum(x, WIDE_BASE, WIDE)


@needs_cc
@pytest.mark.parametrize("segments", [True, False])
def test_a_loop_over_stacked_blocks_is_one_nest_a_block(tmp_path, monkeypatch,
                                                        segments):
    import re
    from np2hw.core import cemit
    # the blocks' bodies cut into segments come to about 100 000 lines
    monkeypatch.setattr(cemit, "SEGMENT_LINES_MAX",
                        1 << 20 if segments else 1 << 16)
    k = compile_c(wide_min_sum, Image2D("llr", WIDE * Z, B, bits=8),
                  build=tmp_path)
    for seed in range(3):
        x = np.random.default_rng(seed).integers(
            0, 256, (B, WIDE * Z)).astype(np.uint8)
        want = wide_min_sum(x)
        got = k(x)
        assert got.dtype == want.dtype and np.array_equal(got, want)
    [source] = tmp_path.glob("*.c")
    text = source.read_text()
    # each block computes its own posterior: no choice among the blocks
    # by a loop variable, and a loop over the batch for each block
    assert not re.search(r"\(\(int64_t\)i\d+\) == \d+ \?", text)
    outer = [l for l in text.splitlines() if "for (int64_t i0 = lo" in l]
    assert len(outer) == WIDE, len(outer)
    # cut where the rolls wrap, unless that is over the line budget
    assert (f"% {Z}" in text or f"/ {Z})" in text) != segments
