# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Toy blocks -- a gain, an offset, a 2x2 balance, a bias that reads the phase, a channel splitter and
joiner, a statistics sink -- and the context they are declared with."""
import functools

import numpy as np

from np2hw import saturate
from np2hw.blocks import ContextBit, StreamPort, block, configblock
from np2hw.blocks.params import Context, Param, StatsWindow, Vocabulary

CONTEXT = (
    Context(name="width", bits=16, default=64, follows="width",
            description="active line length"),
    Context(name="height", bits=16, default=8, follows="height",
            description="active frame height"),
    Context(name="limit", bits=8, default=200, origin="register",
            description="a register-only fact"),
    Context(name="phase", bits=2, default=0,
            description="a fact a register or the stream may carry"),
    Context(name="tick", bits=4, default=0, origin="stream",
            description="a fact that changes every frame"),
)
COMMIT = Param(name="commit", bits=1, description="land a batch of writes")
ctxblock = configblock("ctxblock", version=(1, 0),
                       description="the toy pipeline's context",
                       params=[c.as_param() for c in CONTEXT
                               if c.origin != "stream"] + [COMMIT])
VOCABULARY = Vocabulary(CONTEXT, block=ctxblock, commit="commit")
toyblock = functools.partial(block, vocabulary=VOCABULARY)


@toyblock(version=(1, 0), description="multiply by a gain",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          params=[Param("gain", bits=4, default=2, description="the gain")])
def gain(pixel, p, ctx, bit_depth):
    return saturate(pixel.astype(np.int32) * p.gain, bit_depth)


@toyblock(version=(1, 0), description="add an offset",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          params=[Param("offset", bits=8, default=1,
                        description="what is added")])
def offset(pixel, p, ctx, bit_depth):
    return saturate(pixel.astype(np.int32) + p.offset, bit_depth)


@toyblock(version=(1, 0), description="one sample into two channels",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),))
def pair(pixel, p, ctx, bit_depth):
    value = pixel.astype(np.int32)
    return np.stack([saturate(value - 1, bit_depth),
                     saturate(value + 1, bit_depth)], axis=-1)


@toyblock(version=(1, 0), description="two channels into three",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),))
def triple(pixel, p, ctx, bit_depth):
    a, b = pixel[..., 0].astype(np.int32), pixel[..., 1].astype(np.int32)
    return np.stack([saturate(a - b, bit_depth), saturate(b - a, bit_depth),
                     saturate(a + b, bit_depth)], axis=-1)


@toyblock(version=(1, 0), description="a sink: watches a stream, emits none",
          inputs=(StreamPort("in"),), outputs=(),
          not_traceable="a reduction, which np2hw does not trace",
          params=[Param("level", bits=8, default=128,
                        description="what counts as bright")],
          stats=[StatsWindow("counts", words=4, layout=("bright", "dark"),
                             description="two records of two counts")])
def meter(pixel, p, ctx, bit_depth):
    return np.array([(pixel >= p.level).sum(), (pixel < p.level).sum()])


@toyblock(version=(1, 0), description="add two biases and the phase's low bit",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          consumes=("phase",),
          context=(ContextBit("odd", "phase", bit=0,
                              description="low bit of the phase"),),
          params=[Param("bias", bits=10, signed=True, frac=2, shape=(2,),
                        labels=("coarse", "fine"),
                        description="Signed bias ADDED to the sample. Write "
                                    "a quarter-step value: 4 is one level")])
def bias(pixel, p, ctx, bit_depth):
    value = pixel.astype(np.int32) * 4 + p.bias[0] + p.bias[1] + ctx.odd
    return saturate(value >> 2, bit_depth)


@toyblock(version=(1, 0), description="scale by a gain chosen per 2x2 cell",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          params=[Param("gain", bits=16, frac=8, shape=(2, 2),
                        labels=(("a", "b"), ("c", "d")), default_unity=True,
                        configurable=("bits", "frac"),
                        description="Gain per cell of the 2x2 tile; the "
                                    "reset is unity")])
def balance(pixel, p, ctx, bit_depth):
    value = pixel.astype(np.int32) * p.gain[0, 0]
    return saturate(value >> p.decl.gain.frac, bit_depth)
