# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A block run as its model printed as C, in the model's place.

:meth:`np2hw.blocks.Block.compile_c` builds the trace :meth:`generate`
turns into Verilog as C for a CPU; :meth:`Block.run` given that kernel runs
it with the same registers and context, and a chain runs every block so
(:func:`np2hw.pipeline.run.run_model` with ``c_build``). Each must equal
the NumPy run: registers scalar, register arrays read at constant places
and at computed ones, bound inputs taking each frame's context, a run of
frames read as the stream's history.
"""
import shutil

import numpy as np
import pytest

from np2hw.blocks import StreamPort, block
from np2hw.blocks.params import Param
from np2hw.blocks.stream import StreamSpec
from np2hw.design.composite import assemble
from np2hw.pipeline.run import pixel_chain, run_model
from test_block_history import quadrature
from toylib.blocks import (VOCABULARY, balance, bias, gain, offset, pair,
                           placed, triple)

pytestmark = pytest.mark.skipif(shutil.which("cc") is None,
                                reason="needs a C compiler")

W, H, BITS = 24, 6, 10


@block(version=(1, 0), description="a curve of four knots, by the low bits",
       inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
       params=[Param("curve", bits=8, shape=(4,),
                     description="the value of each low-bit pattern")])
def curve(pixel, p, bit_depth):
    """A register array read at a place the pixel computes: a table."""
    return (p.curve[pixel & 3].astype(np.int32) + (pixel >> 2)).astype(
        np.uint16)


def _random_values(blk, rng):
    """Every scalar register at a random value in its range."""
    out = {}
    for name, p in ((r.name, r.param) for r in blk.params.registers):
        lo = -(1 << (p.bits - 1)) if p.signed else 0
        hi = (1 << (p.bits - 1 if p.signed else p.bits)) - 1
        out[name] = int(rng.integers(lo, hi + 1))
    return out


@pytest.mark.parametrize("blk, channels", [
    (gain, 1), (offset, 1), (pair, 1), (triple, 2), (bias, 1),
    (balance, 1), (curve, 1)])
def test_a_block_as_c_is_its_model(tmp_path, blk, channels):
    """Scalar registers, a signed register array read at constant places,
    a 2x2 one, one read at a computed place; one channel in and out, one
    into two, two into three."""
    rng = np.random.default_rng(11)
    kernel = blk.compile_c(StreamSpec(bit_depth=BITS, channels=channels),
                           W, H, build=tmp_path)
    shape = (H, W, channels) if channels > 1 else (H, W)
    for _ in range(3):
        frame = rng.integers(0, 1 << BITS, shape)
        values = _random_values(blk, rng)
        want = blk.run(frame, values, bit_depth=BITS)
        got = blk.run(frame, values, bit_depth=BITS, kernel=kernel)
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


def test_the_reset_values_are_the_models(tmp_path):
    """No values: the declared resets, as the model takes them."""
    kernel = balance.compile_c(StreamSpec(bit_depth=BITS), W, H,
                               build=tmp_path)
    frame = np.random.default_rng(2).integers(0, 1 << BITS, (H, W))
    np.testing.assert_array_equal(
        balance.run(frame, bit_depth=BITS, kernel=kernel),
        balance.run(frame, bit_depth=BITS))


def test_a_history_block_with_a_bound_place_takes_each_frames(tmp_path):
    """The quadrature over a run, its place bound to each frame's tick and
    its tags an array read at the place the tick computes: every output
    frame the model's at the context of the frame it ends at."""
    bound = quadrature.bind({"pos": "tick & 3"}, VOCABULARY)
    kernel = bound.compile_c(StreamSpec(bit_depth=8), 16, 4, build=tmp_path)
    rng = np.random.default_rng(5)
    video = rng.integers(0, 256, (9, 4, 16))
    ticks = [5, 2, 7, 0, 1, 1, 6, 3, 4]
    values = {f"tag_{i}": v for i, v in enumerate([2, 0, 3, 1])}
    want = bound.run(video, values, bit_depth=8, tick=ticks)
    got = bound.run(video, values, bit_depth=8, kernel=kernel, tick=ticks)
    assert got.shape == (6, 4, 16, 2)
    np.testing.assert_array_equal(got, want)
    other = bound.run(video, values, bit_depth=8, kernel=kernel, tick=1)
    assert (other != want).any(), "the ticks changed nothing: no test"


def test_a_chain_as_c_is_the_chain_of_models(tmp_path, monkeypatch):
    """A history block bound to the tick, then a block of one frame: the
    chain run as C over a run of frames equals the chain of models -- and
    every block of it ran as C."""
    from np2hw.blocks import Block
    ran = []
    real = Block.run

    def spy(self, *args, kernel=None, **kwargs):
        ran.append((self.name, kernel is not None))
        return real(self, *args, kernel=kernel, **kwargs)

    design = assemble(
        "t", {"p": placed.bind({"pos": "tick & 3"}, VOCABULARY),
              "o": offset},
        [("in", "p.in"), ("p.out", "o.in"), ("o.out", "out")],
        vocabulary=VOCABULARY,
        stream={"bits": 8, "max_width": 16, "max_height": 4,
                "context": ["tick"]})
    chain = pixel_chain(design)
    video = np.random.default_rng(4).integers(0, 200, (7, 4, 16))
    values = {"p": {}, "o": {"offset": 9}}
    context = {"tick": [3, 0, 1, 1, 2, 5, 6]}
    model = run_model(chain, video, values, context, 8)
    monkeypatch.setattr(Block, "run", spy)
    c = run_model(chain, video, values, context, 8, c_build=tmp_path)
    assert set(ran) == {("placed", True), ("offset", True)}
    assert c.shape == model.shape == (6, 4, 16)
    np.testing.assert_array_equal(c, model)
