# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A design of a modelled leaf and a pipeline of model blocks is proved
from its models: a frame store handing two frames side by side to a
pipeline whose block reads the stream's history and takes its place in the
capture loop from each frame's context.

One random stimulus drives the models and the Verilog: frames sent in
bursts, a consumer that waits and stalls at random. The design's Verilog is
the one it is built as -- the store's own module, the pipeline built whole
behind its register file -- and its frames out, and what goes out beside
them, are the models', frame for frame.
"""
import shutil

import numpy as np
import pytest

from np2hw.design import composite
from np2hw.hdl.models import Stimulus, frame_store
from np2hw.pipeline import proof
from test_frames_brought import _brought

pytestmark = pytest.mark.skipif(shutil.which("verilator") is None,
                                reason="verilator not on PATH")

W, H, BITS, DEPTH, SLOTS = 8, 4, 10, 2, 8
# the store's latency, the pipeline's depth behind it, and some room for
# the harness's way of counting cycles
MARGIN = 4 + 16


def _design(library):
    def files(name):
        return {"pipes": _brought(DEPTH), "paced": {"paced": {
            "description": "a frame store feeding a pipeline that reads the "
                           "stream's history",
            "inputs": [{"name": n} for n in (
                "clk", "rst", "in", "in_frame_width", "in_frame_height",
                "ctx_tick", "s_axil", "s_axil_aclk", "s_axil_aresetn")],
            "outputs": [{"name": n} for n in (
                "out", "out_frame_width", "out_frame_height", "out_source",
                "cold", "overrun")],
            "nodes": {
                "store": {"block": "np2hw.hdl.frames:frame_store",
                          "build": {"FRAMES": "isp.in.frames",
                                    "BITS": "isp.in.bits", "SLOTS": SLOTS,
                                    "CTX_BITS": 4, "PIXELS": W * H}},
                "isp": {"block": f"{name}.pipes:brought"}},
            "connections": [
                ["clk", "store.clk"], ["rst", "store.rst"],
                ["clk", "isp.clk"], ["rst", "isp.rst"],
                ["s_axil", "isp.s_axil"], ["s_axil_aclk", "isp.s_axil_aclk"],
                ["s_axil_aresetn", "isp.s_axil_aresetn"],
                ["in", "store.in"],
                ["in_frame_width", "store.in_frame_width"],
                ["in_frame_height", "store.in_frame_height"],
                ["ctx_tick", "store.in_ctx"],
                ["store.out", "isp.in"],
                ["store.out_frame_width", "isp.ctx_width"],
                ["store.out_frame_height", "isp.ctx_height"],
                ["store.out_ctx", "isp.ctx_tick"],
                ["isp.out", "out"],
                ["isp.out_frame_width", "out_frame_width"],
                ["isp.out_frame_height", "out_frame_height"],
                ["store.out_source", "out_source"],
                ["store.cold", "cold"], ["store.overrun", "overrun"]]}}}
    return composite.load(f"{library(files)}.paced:paced")


def _draw(rng, frames, ports, outputs=10):
    """Bursts of four frames, and a consumer whose every wait is drawn again
    until the choice it leads to is clear of every event the store must
    tell apart from it."""
    params = {"FRAMES": DEPTH, "SLOTS": SLOTS}
    gaps = tuple(int(rng.integers(20, 40)) if k % 4 else
                 int(rng.integers(60, 400)) for k in range(len(frames)))
    stalls = tuple(tuple(sorted({(int(rng.integers(0, W * H)),
                                  int(rng.integers(1, 20)))
                                 for _ in range(int(rng.integers(0, 3)))}))
                   for _ in range(outputs))

    def timeline(first, waits, n):
        stim = Stimulus(gaps, first, tuple(waits), stalls)
        return frame_store(frames, ports, stim, params, outputs=n).timeline

    def clear(first, waits):
        t = timeline(first, waits, len(waits) + 2)
        return not t.close(MARGIN, upto=len(waits) + 2) and not t.overrun

    t0 = timeline(0, [], 1)
    for _ in range(200):
        first = t0.chosen[0] + MARGIN + 1 + int(rng.integers(0, 200))
        if clear(first, []):
            break
    waits = []
    for _ in range(outputs - 1):
        for _ in range(200):
            wait = int(rng.integers(MARGIN + 2, 1500))
            if clear(first, waits + [wait]):
                waits.append(wait)
                break
        else:
            raise AssertionError("no clear wait")
    return Stimulus(gaps, first, tuple(waits), stalls)


@pytest.mark.parametrize("seed", range(4))
def test_the_design_is_its_models_frame_for_frame(library, seed):
    design = _design(library)
    rng = np.random.default_rng(seed)
    count = 24
    frames = [rng.integers(0, 1 << BITS, (H, W)) for _ in range(count)]
    ports = [{"in_frame_width": W, "in_frame_height": H, "ctx_tick": k % 16}
             for k in range(count)]
    # the store's own ports, as the design wires them
    stim = _draw(rng, frames, [{"in_ctx": p["ctx_tick"]} for p in ports])
    model = proof.run_model(design, frames, ports, stim)
    rtl = proof.run_rtl(design, frames, ports, stim)
    assert len(rtl.frames) == len(model.frames) == len(stim.waits) + 1
    out_bits = BITS
    for i, (got, want) in enumerate(zip(rtl.frames, model.frames)):
        assert got.tolist() == proof.words(want, out_bits).tolist(), \
            f"frame out {i}"
    for i, (got, want) in enumerate(zip(rtl.ports, model.ports)):
        for name, value in want.items():
            assert got[name] == value, f"frame out {i}: {name}"
    # and the context moved the answer: the place changes which frame of
    # the two each frame out shows
    shown = {tuple(f.ravel()) for f in model.frames}
    assert len(shown) > 1


def test_a_value_read_around_the_store_is_refused(library):
    """The pipeline taking its frame's width from the design's input, not
    from the store: the width it would read is whichever frame is arriving,
    and no model of frames can say which."""
    design = _design(library)
    design.edges = [("in_frame_width", "isp.ctx_width") if sink ==
                    "isp.ctx_width" else (source, sink)
                    for source, sink in design.edges]
    frames = [np.zeros((H, W), dtype=int)] * 8
    ports = [{"in_frame_width": W, "in_frame_height": H, "ctx_tick": k}
             for k in range(8)]
    stim = Stimulus((0,) * 8, 400, (100,), ())
    with pytest.raises(proof.ProofError, match="does not ride with"):
        proof.run_model(design, frames, ports, stim)


def test_a_leaf_with_no_model_is_refused(library):
    """The store's library description without its model: the design is
    built as before, and refused a proof, by name."""
    design = _design(library)
    del design.instances["store"].leaf["model"]
    with pytest.raises(proof.ProofError, match="brings no model"):
        proof.stages(design)
