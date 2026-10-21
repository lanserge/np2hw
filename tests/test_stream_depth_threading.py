# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Depth travels the chain the way the channel count does.

A block is told the depth of the word it is GIVEN, read off its driver's
trace, not the depth the pipeline was declared with. The two coincide for
a chain whose every stage saturates back, so the first test here is the
one that says threading changes nothing there. The rest say what it makes
possible.
"""
from __future__ import annotations

import numpy as np
import pytest

from np2hw import saturate

from np2hw.blocks import StreamPort
from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from toylib.blocks import VOCABULARY, gain, offset, pair, toyblock, triple

WIDTH, HEIGHT = 32, 16


def _chain(name, bit_depth, names_and_blocks):
    """A straight chain from the design's input to its output."""
    edges, source = [], "in"
    for path, block in names_and_blocks:
        edges.append((source, f"{path}.{block.ports.inputs[0]}"))
        source = f"{path}.{block.ports.outputs[0]}"
    edges.append((source, "out"))
    return assemble(name, dict(names_and_blocks), edges,
                    vocabulary=VOCABULARY, stream={"bits": bit_depth, "max_width": WIDTH,
                            "max_height": HEIGHT})


def _depths(design):
    """The (channels, bit_depth) each stage is told, in datapath order."""
    seen, built = {}, {}
    spec = compose.shape(design).spec
    for stage in compose.datapath(design):
        given = compose._incoming(design, stage, built)
        channels, depth = given.channels, given.bit_depth
        result = stage.block.generate(
            spec.with_stream(channels, depth), WIDTH, HEIGHT,
            module_name=f"probe_{stage.module_prefix}")
        built[stage.path] = result
        seen[stage.path] = (channels, depth)
    return seen, built


# --------------------------------------------------------------------------- #
# The no-op: a saturating chain is unchanged
# --------------------------------------------------------------------------- #

def test_a_saturating_chain_is_told_the_pipeline_depth_at_every_stage():
    """Every stage saturates back, so threading changes nothing."""
    p = _chain("saturating", 10, [("g", gain), ("o", offset), ("p", pair),
                                  ("t", triple)])
    told, _ = _depths(p)
    assert {path: depth for path, (_ch, depth) in told.items()} == {
        "g": 10, "o": 10, "p": 10, "t": 10}


def test_channels_still_thread_from_the_trace():
    """The channel count keeps arriving from the driver, not a declaration."""
    p = _chain("channels", 10, [("g", gain), ("p", pair), ("t", triple)])
    told, _ = _depths(p)
    assert {path: ch for path, (ch, _d) in told.items()} == {
        "g": 1, "p": 1, "t": 2}


# --------------------------------------------------------------------------- #
# What threading makes possible
# --------------------------------------------------------------------------- #

@toyblock(
    version=(1, 0),
    description="Adds a pixel to itself without bounding it: one bit wider.",
    inputs=(StreamPort("in", "Any single-component stream."),),
    outputs=(StreamPort("out", "The same value, one bit taller."),),
)
def widen(pixel, p, bit_depth: int):
    return (pixel.astype(np.int32) * 2).astype(np.uint32)


@toyblock(
    version=(1, 0),
    description="Saturates to eight bits, whatever it was given.",
    inputs=(StreamPort("in", "Any single-component stream."),),
    outputs=(StreamPort("out", "Eight bits."),),
)
def narrow(pixel, p, bit_depth: int):
    return saturate(pixel.astype(np.int32), 8).astype(np.uint16)


def test_a_widening_block_hands_on_the_wider_depth():
    """bit_depth + 1 out of one stage is bit_depth + 1 into the next."""
    p = _chain("widening", 10, [("a", widen), ("b", widen), ("c", widen)])
    told, _ = _depths(p)
    assert [depth for _ch, depth in
            (told[k] for k in ("a", "b", "c"))] == [10, 11, 12]


def test_a_narrowing_block_hands_on_the_narrower_depth():
    """The 'eight bits after gamma' the stream docstring describes."""
    p = _chain("narrowing", 12, [("a", widen), ("b", narrow), ("c", widen)])
    told, _ = _depths(p)
    assert [depth for _ch, depth in
            (told[k] for k in ("a", "b", "c"))] == [12, 13, 8]


def test_the_pipeline_output_port_carries_the_traced_width():
    """The exit boundary is the last block's word, not the declared spec."""
    p = _chain("widening_out", 10, [("a", widen), ("b", widen)])
    _told, built = _depths(p)
    assert compose._output_stream(p, built, "out").data_bits == 12
    # and the entry stays the one declared fact
    assert compose.shape(p).spec.data_bits == 10


def test_a_widening_chain_composes():
    """np2hw refuses mismatched widths, so this only passes if they agree."""
    p = _chain("widening_whole", 10, [("a", widen), ("b", widen)])
    generated = compose.generate(p, control=False)
    assert generated.modules


def test_the_composer_refuses_when_a_consumer_disagrees(monkeypatch):
    """The safety net under all of this: a width mismatch is not silent.

    Threading is what keeps producer and consumer agreeing. Break the
    threading and np2hw's own check must still catch it, so a future
    change here fails loudly rather than emitting a wrong bus.
    """
    p = _chain("mismatch", 10, [("a", widen), ("b", widen)])
    # pretend depth never threaded: every block told the pipeline's depth
    form = compose.shape(p)
    monkeypatch.setattr(compose, "_incoming",
                        lambda design, stage, built: compose.Handed(
                            form.spec.channels, form.spec.bit_depth,
                            form.width, form.height))
    with pytest.raises(Exception) as caught:
        compose.generate(p, control=False)
    assert "bit" in str(caught.value).lower()


def test_geometry_is_threaded_from_the_driver_not_broadcast():
    """A block is built for the frame it is GIVEN.

    Depth and channels are read from the driver's traced interface, and so
    is geometry: a fact broadcast from the pipeline would be right only
    while every block preserves geometry, and the first block that changes
    size would make the two owners disagree.
    """
    import io
    from contextlib import redirect_stdout

    pipeline = _chain("geo", 10, [("g", gain), ("p", pair), ("t", triple)])

    built, seen = {}, []
    for stage in compose.datapath(pipeline):
        given = compose._incoming(pipeline, stage, built)
        channels, depth = given.channels, given.bit_depth
        width, height = given.width, given.height
        with redirect_stdout(io.StringIO()):
            result = stage.block.generate(
                compose.shape(pipeline).spec.with_stream(channels, depth),
                width, height,
                module_name=f"t_{stage.path}")
        built[stage.path] = result
        output = result.core["interface"].get("output") or {}
        seen.append((stage.path, (width, height),
                     (output.get("cols"), output.get("rows"))))

    # every block is told a real geometry, and it is the one its driver
    # produced -- not the pipeline's, which merely coincides here
    for index, (path, given, produced) in enumerate(seen):
        assert given == (WIDTH, HEIGHT), f"{path} was built for {given}"
        assert produced == (WIDTH, HEIGHT), f"{path} produced {produced}"
        if index:
            assert given == seen[index - 1][2], (
                f"{path} was built for {given} but its driver produced "
                f"{seen[index - 1][2]}")


# --------------------------------------------------------------------------- #
# A run of the models threads depth the same way, from the same trace
# --------------------------------------------------------------------------- #

@toyblock(
    version=(1, 0),
    description="Reflects a sample about the top of its stream.",
    inputs=(StreamPort("in", "Any single-component stream."),),
    outputs=(StreamPort("out", "The same depth."),),
)
def reflect(pixel, p, bit_depth: int):
    return (((1 << bit_depth) - 1) - pixel.astype(np.int32)).astype(np.uint16)


def test_a_run_of_the_models_is_told_the_traced_depth():
    """After a block that narrows to eight bits, the next block reflects
    about 255, not about the pipeline's 1023 -- because the run reads the
    depth the narrowing block's trace published, as the composer does."""
    from np2hw.pipeline import run as runner

    p = _chain("run_narrowing", 10, [("n", narrow), ("r", reflect)])
    frame = np.random.default_rng(2).integers(0, 1 << 10, size=(HEIGHT, WIDTH))
    out = runner.run_model(runner.pixel_chain(p), frame, {"n": {}, "r": {}},
                           {}, 10)
    np.testing.assert_array_equal(out, 255 - np.minimum(frame, 255))


@pytest.mark.skipif(__import__("shutil").which("verilator") is None,
                    reason="verilator not on PATH")
def test_the_run_and_the_rtl_agree_across_a_narrowing():
    from np2hw.pipeline import run as runner

    p = _chain("twin_narrowing", 10, [("n", narrow), ("r", reflect)])
    chain = runner.pixel_chain(p)
    frame = np.random.default_rng(3).integers(0, 1 << 10, size=(HEIGHT, WIDTH))
    values = {"n": {}, "r": {}}
    np.testing.assert_array_equal(runner.run_model(chain, frame, values, {}, 10),
                                  runner.run_rtl(chain, frame, values, {}, 10))


@toyblock(
    version=(1, 0),
    description="Traces as a pass-through, runs as a crop: a model and a "
                "trace that disagree about the frame.",
    inputs=(StreamPort("in", "Any single-component stream."),),
    outputs=(StreamPort("out", "Not what the trace says."),),
)
def two_faced(pixel, p, bit_depth: int):
    if isinstance(pixel, np.ndarray):
        return pixel[1:, :]
    return saturate(pixel.astype(np.int32), bit_depth)


def test_a_model_handing_on_another_frame_than_its_trace_is_refused():
    """Geometry is traced like depth, and a run checks the model against
    it: a model that disagrees with its own hardware about the frame it
    emits must not hand that frame on as if nothing happened."""
    from np2hw.pipeline import run as runner

    p = _chain("two_faced_chain", 8, [("t", two_faced), ("n", narrow)])
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    with pytest.raises(ValueError, match=r"t: the model hands on a 32x15 "
                       r"frame of 1 channel\(s\); its trace says 32x16 of 1"):
        runner.run_model(runner.pixel_chain(p), frame, {"t": {}, "n": {}},
                         {}, 8)
