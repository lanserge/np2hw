# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A composite whose stream brings its history: frames side by side, from a
frame store upstream, stated on the input port where the stream enters.

The block that reads the stream's history keeps no frames of its own then,
and its input is as wide as the frames brought; how many is one fact of the
composite's, readable by a design as ``node.port.frames`` -- which is what a
design sets its frame store's depth from.
"""
import pytest

from np2hw import pipeline
from np2hw.design import composite
from np2hw.pipeline import compose


def _brought(frames):
    return {"brought": {
        "description": "a history the stream brings",
        "vocabulary": "toylib.blocks:VOCABULARY",
        "parameters": {"frames": {"default": frames,
                                  "description": "frames side by side"}},
        "inputs": [{"name": "in", "bits": 10, "max_width": 8,
                    "max_height": 4, "frames": "frames",
                    "context": ["width", "height", "tick"]}],
        "outputs": [{"name": "out"}],
        "nodes": {"p": {"block": "toylib.blocks:placed",
                        "bind": {"pos": "tick & 3"}}},
        "connections": [["in", "p.in"], ["p.out", "out"]]}}


def test_the_frames_brought_widen_the_input_and_keep_none(library):
    name = library({"pipes": _brought(2)})
    design = pipeline.load(f"{name}.pipes:brought")
    generated = compose.generate(design, control=False)
    ports = {p["name"]: p for p in generated.meta["ports"]}
    assert ports["in_data"]["width"] == 2 * 10
    assert not any(generated.meta["memories"].values())


def test_without_frames_brought_the_block_keeps_its_own(library):
    name = library({"pipes": _brought(0)})
    generated = compose.generate(pipeline.load(f"{name}.pipes:brought"),
                                 control=False)
    ports = {p["name"]: p for p in generated.meta["ports"]}
    assert ports["in_data"]["width"] == 10
    assert any(generated.meta["memories"].values())


def test_a_design_sets_its_frame_store_from_the_frames_a_stream_brings(
        library):
    """The store's depth and sample are the pipeline's input's facts, read
    by the design, and its output -- the frames side by side -- is exactly
    the word the pipeline takes."""
    def files(name):
        return {"pipes": _brought(2), "board": {"board": {
            "description": "a frame store feeding a pipeline",
            "inputs": [{"name": n} for n in (
                "clk", "rst", "in", "in_frame_width", "in_frame_height",
                "in_ctx")],
            "outputs": [{"name": n} for n in (
                "out", "out_frame_width", "out_frame_height", "out_ctx",
                "out_source", "cold", "overrun")],
            "nodes": {
                "store": {"block": "np2hw.hdl.frames:frame_store",
                          "build": {"FRAMES": "isp.in.frames",
                                    "BITS": "isp.in.bits",
                                    "PIXELS": 32}},
                "isp": {"block": f"{name}.pipes:brought"}},
            "connections": [
                ["clk", "store.clk"], ["rst", "store.rst"],
                ["in", "store.in"],
                ["in_frame_width", "store.in_frame_width"],
                ["in_frame_height", "store.in_frame_height"],
                ["in_ctx", "store.in_ctx"],
                ["store.out", "isp.in"], ["isp.out", "out"],
                ["store.out_frame_width", "out_frame_width"],
                ["store.out_frame_height", "out_frame_height"],
                ["store.out_ctx", "out_ctx"],
                ["store.out_source", "out_source"],
                ["store.cold", "cold"], ["store.overrun", "overrun"]]}}}
    name = library(files)
    design = composite.load(f"{name}.board:board")
    store = design.instances["store"]
    # the block's parameters, which its model reads, and the module's, which
    # its implementation states over them
    assert store.leaf["block_parameters"]["FRAMES"] == 2
    assert (store.build["FRAMES"], store.build["BITS"]) == (2, 10)
    assert ("store.out", "isp.in") in design.edges


def test_frames_on_an_output_are_refused(library):
    files = _brought(2)
    files["brought"]["outputs"] = [{"name": "out", "frames": 2}]
    name = library({"pipes": files})
    with pytest.raises(composite.CompositionError, match="output out states"):
        pipeline.load(f"{name}.pipes:brought")
