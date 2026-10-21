# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A block whose input is the stream's history.

The model is handed the recent frames, (T, H, W) newest last, and reads
past ones by slicing the first axis; the port says so (``history=True``),
because a channel axis looks the same. Nothing declares how many frames are
kept: the core's store is how far back the model reads. Each block here is
generated and its Verilog driven with a run of frames, against the block's
own NumPy run -- a bound input taking each frame's context as its header
would carry it.
"""
import shutil

import numpy as np
import pytest

from np2hw.blocks import HISTORY_TRACE, StreamPort, block
from np2hw.blocks.params import Param
from np2hw.blocks.stream import StreamSpec
from test_core_frames import BITS, FRAMES, H, W, _beats, _sim, _video
from toylib.blocks import VOCABULARY

requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None,
                                       reason="needs iverilog")
SPEC = StreamSpec(bit_depth=BITS, channels=1)


@block(version=(1, 0), description="the last four frames, averaged",
       inputs=(StreamPort("in", "the frames", history=True),),
       outputs=(StreamPort("out"),))
def average(video, p, bit_depth):
    v = video.astype(np.int32)
    return ((v[3:] + v[2:-1] + v[1:-2] + v[:-3]) >> 2).astype(np.uint16)


@block(version=(1, 0), description="a four-phase quadrature by place",
       inputs=(StreamPort("in", "one reference phase a frame",
                          history=True),),
       outputs=(StreamPort("out"),),
       params=[Param("pos", bits=2, description="the live frame's place"),
               Param("tag", bits=2, shape=(4,),
                     description="each place's phase")])
def quadrature(video, p, bit_depth):
    v = video.astype(np.int32)
    f = [v[3:], v[2:-1], v[1:-2], v[:-3]]               # f[k]: frame n - k
    phase_of = [p.tag[(p.pos + 4 - k) & 3] for k in range(4)]

    def phase(j):
        return np.where(phase_of[0] == j, f[0],
                        np.where(phase_of[1] == j, f[1],
                                 np.where(phase_of[2] == j, f[2], f[3])))
    i, q = phase(0) - phase(2), phase(1) - phase(3)
    return np.stack([i + 1024, q + 1024], axis=-1).astype(np.uint16)


def _packed(core, out):
    """Two-channel model frames as the core's output words."""
    field = core["interface"]["output"]["field_bits"]
    out = out.astype(np.int64)
    return out[..., 0] + (out[..., 1] << field)


def test_the_store_is_how_far_back_the_model_reads():
    core = average.generate(SPEC, W, H, module_name="hist_average").core
    [store] = core["memories"]
    assert store["bits"] == 3 * BITS and store["depth"] == W * H


@requires_iverilog
def test_a_history_block_is_its_model_frame_after_frame(tmp_path):
    core = average.generate(SPEC, W, H, module_name="hist_average").core
    video = _video(1)
    assert _sim(tmp_path, core, video) == \
        _beats(average.run(video, bit_depth=BITS))


@requires_iverilog
def test_a_bound_place_takes_each_frames_context(tmp_path):
    """`pos` bound to the stream's count: each frame's header sets it, the
    run takes one value a frame, and every output frame is the model's at
    the place of the frame it ends at."""
    bound = quadrature.bind({"pos": "(tick + 2) & 3"}, VOCABULARY)
    core = bound.generate(SPEC, W, H, module_name="hist_quad").core
    assert core["memories"][0]["bits"] == 3 * BITS
    order = (0, 2, 1, 3)
    tags = {f"tag_{i}": order[i] for i in range(4)}
    ticks = [(9 + n) % 16 for n in range(FRAMES)]       # wraps at 4 bits
    video = _video(5)
    model = bound.run(video, tags, bit_depth=BITS, tick=ticks)
    assert len(model) == FRAMES - 3
    got = _sim(tmp_path, core, video, seed=11,
               values=[dict(tags, tick=t) for t in ticks])
    assert got == _beats(_packed(core, model))
    # and the context moved the answer: one place for every frame is not it
    still = bound.run(video, tags, bit_depth=BITS, tick=ticks[-1])
    assert (still != model).any(), "the context changed nothing: no test"


def test_context_a_frame_is_one_value_for_every_frame():
    bound = quadrature.bind({"pos": "tick & 3"}, VOCABULARY)
    with pytest.raises(ValueError, match="one per frame"):
        bound.run(_video(2), bit_depth=BITS, tick=[0, 1, 2])


def test_a_read_past_the_traced_history_is_refused():
    @block(version=(1, 0), description="reads too far back",
           inputs=(StreamPort("in", history=True),),
           outputs=(StreamPort("out"),))
    def ancient(video, p, bit_depth):
        return video[HISTORY_TRACE:]

    with pytest.raises(NotImplementedError):
        ancient.generate(SPEC, W, H, module_name="ancient")


def _side_by_side(video, frames):
    """Each frame from the `frames`-th on, with those before it, as a frame
    store hands them on: one word a pixel, the frame k back in the k-th
    field."""
    v = video.astype(np.int64)
    return np.stack([sum(v[t - k] << (k * BITS) for k in range(frames))
                     for t in range(frames - 1, len(v))])


def test_a_history_the_stream_brings_keeps_no_frames():
    core = average.generate(StreamSpec(bit_depth=BITS, frames=4), W, H,
                            module_name="hist_brought").core
    assert core["memories"] == [] and core["in_bits"] == 4 * BITS


@requires_iverilog
def test_a_history_the_stream_brings_is_read_as_it_comes(tmp_path):
    """Four frames side by side, from a frame store: every window an output
    frame, the model's own for the frame the window ends at."""
    core = average.generate(StreamSpec(bit_depth=BITS, frames=4), W, H,
                            module_name="hist_brought").core
    video = _video(1)
    assert _sim(tmp_path, core, _side_by_side(video, 4)) == \
        _beats(average.run(video, bit_depth=BITS))


@requires_iverilog
def test_a_history_brought_takes_the_newest_frames_context(tmp_path):
    """The place bound to the count is the newest frame's: a frame store
    hands on the context of the frame its window ends at."""
    bound = quadrature.bind({"pos": "(tick + 2) & 3"}, VOCABULARY)
    core = bound.generate(StreamSpec(bit_depth=BITS, frames=4), W, H,
                          module_name="hist_quad_brought").core
    order = (0, 2, 1, 3)
    tags = {f"tag_{i}": order[i] for i in range(4)}
    ticks = [(9 + n) % 16 for n in range(FRAMES)]
    video = _video(5)
    model = bound.run(video, tags, bit_depth=BITS, tick=ticks)
    got = _sim(tmp_path, core, _side_by_side(video, 4), seed=11,
               values=[dict(tags, tick=t) for t in ticks[3:]])
    assert got == _beats(_packed(core, model))


def test_a_read_further_back_than_the_stream_brings_is_refused():
    with pytest.raises(NotImplementedError, match="brings 3 side by side"):
        average.generate(StreamSpec(bit_depth=BITS, frames=3), W, H,
                         module_name="short")


def test_frames_side_by_side_into_a_block_of_one_frame_are_refused():
    @block(version=(1, 0), description="one frame",
           inputs=(StreamPort("in"),), outputs=(StreamPort("out"),))
    def same(pixel, p, bit_depth):
        return pixel

    with pytest.raises(ValueError, match="reads no history"):
        same.generate(StreamSpec(bit_depth=BITS, frames=4), W, H,
                      module_name="same")


requires_verilator = pytest.mark.skipif(shutil.which("verilator") is None,
                                        reason="needs verilator")


@requires_verilator
def test_a_chain_runs_a_run_of_frames_through_its_twin():
    """A history block then a block of one frame: the chain's model runs
    the run, each frame's tick its own, and the twin -- the frames sent
    back to back, each one's context set as it starts -- puts out the same
    run, the frame the history fills left out."""
    from np2hw.design.composite import assemble
    from np2hw.pipeline.run import pixel_chain, run_model, run_rtl
    from toylib.blocks import offset, placed

    design = assemble(
        "t", {"p": placed.bind({"pos": "tick & 3"}, VOCABULARY),
              "o": offset},
        [("in", "p.in"), ("p.out", "o.in"), ("o.out", "out")],
        vocabulary=VOCABULARY,
        stream={"bits": 8, "max_width": W, "max_height": H,
                "context": ["tick"]})
    chain = pixel_chain(design)
    video = np.random.default_rng(4).integers(0, 200, (FRAMES, H, W))
    ticks = [3, 0, 1, 1, 2, 5, 6]
    values = {"p": {}, "o": {"offset": 9}}
    model = run_model(chain, video, values, {"tick": ticks}, 8)
    assert model.shape == (FRAMES - 1, H, W)
    np.testing.assert_array_equal(
        run_rtl(chain, video, values, {"tick": ticks}, 8), model)
    still = run_model(chain, video, values, {"tick": 1}, 8)
    assert (still != model).any(), "the ticks changed nothing: no test"


@block(version=(1, 0), description="deep arithmetic, then the place added",
       inputs=(StreamPort("in", history=True),),
       outputs=(StreamPort("out"),),
       params=[Param("pos", bits=4, description="the live frame's place")])
def late(video, p, bit_depth):
    v = video.astype(np.int64)
    x = v[1:] * 3 + v[:-1]
    for _ in range(5):
        x = (x * x + 7) & 65535
    return ((x + p.pos) & 255).astype(np.uint16)


@requires_verilator
def test_a_frames_context_stays_with_the_frame_through_the_stages():
    """A bound input read deep in a core cut into stages: the frames sent
    back to back, each one's tick set as it starts, while the frame before
    is still inside. Every pixel takes its own frame's tick -- the core
    holds it from the frame's first pixel and carries it with the beats."""
    from np2hw.design.composite import assemble
    from np2hw.pipeline.run import pixel_chain, run_model, run_rtl

    design = assemble(
        "t", {"l": late.bind({"pos": "tick"}, VOCABULARY)},
        [("in", "l.in"), ("l.out", "out")], vocabulary=VOCABULARY,
        stream={"bits": 8, "max_width": 16, "max_height": 8,
                "context": ["tick"]})
    chain = pixel_chain(design)
    video = np.random.default_rng(2).integers(0, 256, (6, 8, 16))
    ticks = [1, 4, 9, 2, 15, 7]
    model = run_model(chain, video, {"l": {}}, {"tick": ticks}, 8)
    rtl = run_rtl(chain, video, {"l": {}}, {"tick": ticks}, 8, clk_ns=5.0)
    np.testing.assert_array_equal(rtl, model)
    built = late.bind({"pos": "tick"}, VOCABULARY).generate(
        StreamSpec(bit_depth=8, channels=1), 16, 8, module_name="late",
        clk_ns=5.0)
    assert built.core["pipeline_stages"] > 2, "one stage: nothing in flight"


@requires_iverilog
def test_a_value_computed_from_the_history_reads_its_frames(tmp_path):
    """A value computed from the whole history and then sliced on time is
    that value at each frame read: the slices are the stream's frames at
    their offsets, kept as any history is -- not the live frame at every
    offset."""
    @block(version=(1, 0), description="computed, then sliced",
           inputs=(StreamPort("in", history=True),),
           outputs=(StreamPort("out"),))
    def late(video, p, bit_depth):
        v = video.astype(np.uint16) << 1
        return ((v[1:] >> 1) - (v[:-1] >> 1) + 1024).astype(np.uint16)

    core = late.generate(SPEC, W, H, module_name="late").core
    [store] = core["memories"]
    assert store["bits"] == BITS
    video = _video(3)
    model = late.run(video, bit_depth=BITS)
    assert (model != 1024).any(), "the frames differ: no test"
    assert _sim(tmp_path, core, video) == _beats(model)


def test_a_read_across_time_through_a_non_pointwise_value_is_refused():
    """Where the value cannot be pushed through to the stream -- a stack of
    two of it -- a slice on time is no read a store can serve, and it is
    refused, saying how to write it."""
    @block(version=(1, 0), description="stacked, then sliced",
           inputs=(StreamPort("in", history=True),),
           outputs=(StreamPort("out"),))
    def stacked(video, p, bit_depth):
        v = np.stack([video, video], axis=-1)
        return (v[1:, ..., 0] >> 1).astype(np.uint16)

    with pytest.raises(NotImplementedError):
        stacked.generate(SPEC, W, H, module_name="stacked")
