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

