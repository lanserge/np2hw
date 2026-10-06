# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Where a context's value may come from: register, either, or stream only.

A register-only signal is software's choice and never a stream fact; an
"either" signal may be both; a stream-only one changes every frame, so no
register could hold it, and a pipeline has it only where the stream
delivers it.
"""
import pytest

from np2hw.blocks import ContextBit, StreamPort
from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from toylib.blocks import gain, toyblock

STREAM = {"bits": 8, "max_width": 16, "max_height": 8}


@toyblock(version=(1, 0), description="reads the frame's tick",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          consumes=("tick",),
          context=(ContextBit("lo", "tick", bit=0,
                              description="low bit of the tick"),))
def ticked(pixel, p, ctx, bit_depth):
    return pixel + ctx.lo


def _ticked(context_from_stream):
    return assemble("t", {"t": ticked}, [("in", "t.in"), ("t.out", "out")],
                    stream=dict(STREAM, context=list(context_from_stream)))


def _offsets(design):
    owner = compose.vocabulary(design).block.name
    block = next(b for b in compose.register_map(design)["blocks"]
                 if b["path"] == owner)
    return {r["name"]: r["offset"] for r in block["registers"]}


def test_consuming_it_without_the_stream_is_refused():
    with pytest.raises(ValueError, match="must arrive with the stream"):
        compose.stages(_ticked(()))


def test_where_the_stream_delivers_it_the_pipeline_has_it():
    """A port, no register, and the context block's registers exactly where
    they are in a pipeline without the signal."""
    design = _ticked(("tick",))
    assert "tick" in {c.name for c in compose.contexts(design)}
    generated = compose.generate(design, control=True)
    assert "ctx_tick" in dict(generated.modules)[generated.top]
    without = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                       stream=STREAM)
    assert _offsets(design) == _offsets(without)
    assert "tick" not in _offsets(design)


def test_elsewhere_the_pipeline_does_not_have_it():
    design = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                      stream=STREAM)
    assert "tick" not in {c.name for c in compose.contexts(design)}
