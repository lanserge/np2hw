# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Where a context's value may come from: register, either, or stream only.

A register-only signal is software's choice and never a stream fact; an
"either" signal may be both; a stream-only one changes every frame, so no
register could hold it, and a pipeline has it only where the stream
delivers it.
"""
import pytest

from np2hw.blocks import StreamPort
from np2hw.blocks.params import Param
from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from toylib.blocks import VOCABULARY, gain, toyblock

STREAM = {"bits": 8, "max_width": 16, "max_height": 8}


@toyblock(version=(1, 0), description="adds a one-bit input",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          params=[Param("lo", bits=1, description="what is added")])
def ticked(pixel, p, bit_depth):
    return pixel + p.lo


def _ticked(context_from_stream):
    return assemble("t", {"t": ticked.bind({"lo": "tick & 1"}, VOCABULARY)},
                    [("in", "t.in"), ("t.out", "out")],
                    vocabulary=VOCABULARY, stream=dict(STREAM, context=list(context_from_stream)))


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
    assert compose.carried_layout(design) == (("tick", 4, ""),)
    generated = compose.generate(design, control=True)
    assert "input  wire [3:0]  in_ctx" in dict(generated.modules)[generated.top]
    without = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                       vocabulary=VOCABULARY, stream=STREAM)
    assert _offsets(design) == _offsets(without)
    assert "tick" not in _offsets(design)


def test_elsewhere_the_pipeline_does_not_have_it():
    design = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                      vocabulary=VOCABULARY, stream=STREAM)
    assert "tick" not in {c.name for c in compose.contexts(design)}


def test_a_layout_puts_the_size_first_whatever_the_vocabulary_order():
    """The CTX member's one order: the fact following the width, the one
    following the height, then the rest as the vocabulary declares them --
    so a module reading the size alone finds it in the low 32 bits, and
    no port's listing order changes the word."""
    from np2hw.blocks.params import Context, Vocabulary
    from np2hw.stream import pack_context, unpack_context
    late = Vocabulary((Context(name="seq", bits=8, default=0,
                               description="a count"),
                       Context(name="rows", bits=16, default=8,
                               follows="height", description="its rows"),
                       Context(name="cols", bits=16, default=8,
                               follows="width", description="its columns")))
    layout = late.layout(["seq", "rows", "cols"])
    assert [f[0] for f in layout] == ["cols", "rows", "seq"]
    assert late.layout(["rows", "seq", "cols"]) == layout
    word = pack_context(layout, {"cols": 640, "rows": 480, "seq": 7})
    assert word & 0xFFFF == 640 and (word >> 16) & 0xFFFF == 480
    assert unpack_context(layout, word) == {"cols": 640, "rows": 480,
                                            "seq": 7}
