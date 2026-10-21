# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A video's frames, read across time.

A model handed (T, H, W) reads past frames by slicing the frame axis, and
the store that keeps them is derived from those reads: its depth is the
spread of their starts (docs/one-graph.md, decision 6). Each model here is
compiled and its Verilog driven with a run of
frames through ONE elaboration, under random backpressure on both sides,
and compared with the model run on the same frames -- every output frame,
and no output at all for the frames the history is still filling.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, generate, to_ir
from np2hw.ir import Param

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")

W, H, BITS = 7, 5, 10
FRAMES = 7


def _compile(fn, params=(), frames=FRAMES, bits=BITS, channels=1, name="m"):
    _, mod = to_ir(fn, Image2D("v", W, H, bits=bits, frames=frames), *params,
                   channels=channels)
    return generate(mod, module_name=name)


def _sim(tmp_path, core, video, seed=0, values=None):
    """`video` through the core, frame after frame with a gap between, the
    input's valid and the output's ready both random. `values` gives each
    frame its registers -- a scalar input set between frames, as a frame's
    context arrives with it. Returns every output beat as
    (data, sof, eol, last)."""
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    h, w = video.shape[1:3]
    # each frame its size -- the frames streamed, which may be smaller than
    # the core was traced at -- and whatever registers the caller sets
    values = [dict({"frame_width": w, "frame_height": h}, **v)
              for v in (values or [{} for _ in video])]
    regs = "".join(f"    reg [{b - 1}:0] param_{n} = 0;\n"
                   for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n"
                    for n, _ in core["params"])
    sets = ["".join(f"            param_{n} = {int(v.get(n, core["param_defaults"][n])) & ((1 << b) - 1)};\n"
                    for n, b in core["params"]) for v in values]
    per_frame = "".join(f"            if (k == {k}) begin\n{sets[k]}            end\n"
                        for k in range(len(values)))
    words = [int(v) for v in video.ravel()]
    per = w * h
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in words))
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0, out_ready = 0;
    reg [{in_bits - 1}:0] in_data = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire [{out_bits - 1}:0] out_data;
{regs}    reg [{in_bits - 1}:0] img [0:{len(words) - 1}];
    integer f, i, k, seed = {seed};
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_sof(in_sof), .in_data(in_data),
{conns}        .out_valid(out_valid), .out_ready(out_ready), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_data(out_data));
    always @(posedge clk) if (out_valid && out_ready)
        $fdisplay(f, "%0d %0d %0d %0d", out_data, out_sof, out_eol, out_last);
    always @(negedge clk) out_ready <= ($random(seed) & 3) != 0;
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
        repeat (3) @(negedge clk); rst = 0;
        for (k = 0; k < {len(words) // per}; k = k + 1) begin
{per_frame}            for (i = 0; i < {per}; i = i + 1) begin
                while (($random(seed) & 3) == 0) begin
                    in_valid = 0; @(negedge clk);
                end
                in_data = img[k * {per} + i]; in_sof = (i == 0); in_valid = 1;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            in_valid = 0; in_sof = 0;
            repeat (6) @(negedge clk);
        end
        repeat (40) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "core.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=120)
    return [tuple(map(int, r.split()))
            for r in (tmp_path / "out.txt").read_text().splitlines() if r]


def _beats(out):
    """The model's output frames as the stream carries them."""
    beats = []
    for frame in out:
        flat = frame.ravel()
        cols = frame.shape[-1]
        beats += [(int(v), int(i == 0), int(i % cols == cols - 1),
                   int(i == flat.size - 1)) for i, v in enumerate(flat)]
    return beats


def _video(seed, frames=FRAMES, bits=BITS):
    return np.random.default_rng(seed).integers(0, 1 << bits, (frames, H, W))


def average(video):
    """The last four frames, averaged: three of them kept."""
    v = video.astype(np.int32)
    return ((v[3:] + v[2:-1] + v[1:-2] + v[:-3]) >> 2).astype(np.uint16)


def test_four_frames_averaged_keep_three(tmp_path):
    core = _compile(average, name="average")
    [m] = core["memories"]
    assert (m["bits"], m["depth"], m["read"]) == (3 * BITS, W * H, "sync")
    video = _video(1)
    assert _sim(tmp_path, core, video) == _beats(average(video))


def change(video):
    """This frame against the one two before it: one read reaching back
    two frames keeps two, though only the oldest is read."""
    v = video.astype(np.int32)
    return (v[2:] - v[:-2] + 1024).astype(np.uint16)


def test_a_read_two_frames_back_keeps_two(tmp_path):
    core = _compile(change, name="change")
    assert core["memories"][0]["bits"] == 2 * BITS
    video = _video(2)
    assert _sim(tmp_path, core, video, seed=5) == _beats(change(video))


def late(video):
    """Frames from the fourth on, as they are: nothing kept, the first
    three never shown."""
    return video[3:]


def test_frames_left_out_make_no_output_and_keep_nothing(tmp_path):
    core = _compile(late, name="late")
    assert core["memories"] == []
    video = _video(3)
    assert _sim(tmp_path, core, video, seed=7) == _beats(late(video))


def mixed(pixel):
    """Two 8-bit channels a word, read across time and across channels:
    the store keeps the whole word."""
    a0, b0 = (pixel[1:, ..., k].astype(np.int32) for k in range(2))
    a1, b1 = (pixel[:-1, ..., k].astype(np.int32) for k in range(2))
    return np.stack([(a0 + a1) >> 1, (b0 + b1 + a1) >> 2],
                    axis=-1).astype(np.uint16)


def test_a_channel_word_is_kept_whole(tmp_path):
    core = _compile(mixed, bits=16, channels=2, name="mixed")
    assert core["memories"][0]["bits"] == 16
    video = _video(4, bits=16)
    lanes = np.stack([video & 255, video >> 8], axis=-1)
    res = mixed(lanes).astype(np.int64)
    field = core["interface"]["output"]["field_bits"]
    packed = res[..., 0] + (res[..., 1] << field)
    assert _sim(tmp_path, core, video, seed=9) == _beats(packed)


def forward(video):
    v = video.astype(np.int32)
    return (v[:-1] - v[1:] + 1024).astype(np.uint16)


def test_the_live_frame_is_the_latest_read():
    """Reads are kept from the past, so the latest frame read is the live
    one, whatever the slices are called: this is the same store, kept."""
    core = _compile(forward, name="forward")
    assert core["memories"][0]["bits"] == BITS


# --------------------------------------------------------------------------- #
# the phase: a scalar input, frame by frame
# --------------------------------------------------------------------------- #

POS = Param("pos", bits=2, description="the live frame's capture-loop place")
TAG = Param("tag", bits=2, shape=(4,), description="each place's phase")


def quadrature(video, pos, tag):
    """A time-of-flight sensor's quadrature from the last four frames, one
    reference phase a frame: frame n-k sits k places back in the capture
    loop, its phase the tag of that place -- so any phase order is table
    contents. `pos` is a scalar input; where it comes from (the stream's
    frame phase, a register) is the design's to say, not this model's."""
    v = video.astype(np.int32)
    f = [v[3:], v[2:-1], v[1:-2], v[:-3]]               # f[k]: frame n - k
    phase_of = [tag[(pos - k) & 3] for k in range(4)]

    def phase(j):
        return np.where(phase_of[0] == j, f[0],
                        np.where(phase_of[1] == j, f[1],
                                 np.where(phase_of[2] == j, f[2], f[3])))
    i, q = phase(0) - phase(2), phase(1) - phase(3)
    return np.stack([i + 1024, q + 1024], axis=-1).astype(np.uint16)


def test_the_phase_routes_by_each_frames_place_in_the_loop(tmp_path):
    """The live frame's place in the capture loop changes every frame, set
    with the frame as its context would be, and the phases step in a
    scrambled order: every output frame is the model's at that frame's
    place -- and one store of three frames serves every alignment."""
    core = _compile(quadrature, (POS, TAG), name="quadrature")
    assert core["memories"][0]["bits"] == 3 * BITS
    order = (0, 2, 1, 3)                              # place -> phase
    start = 2                                         # frame 0's place
    video = _video(5)
    places = [(start + n) % 4 for n in range(FRAMES)]
    values = [dict(pos=p, **{f"tag_{i}": order[i] for i in range(4)})
              for p in places]
    field = core["interface"]["output"]["field_bits"]
    want = []
    for n in range(3, FRAMES):
        res = quadrature(video, places[n], np.array(order))[n - 3]
        want.append(res[..., 0].astype(np.int64)
                    + (res[..., 1].astype(np.int64) << field))
    assert _sim(tmp_path, core, video, seed=11, values=values) == \
        _beats(np.array(want))


def test_a_video_at_a_frame_smaller_than_it_was_traced_at(tmp_path):
    """The store is as deep as the traced frame, the most it holds; a
    smaller frame's positions run to its own last pixel, and the history
    is the same history."""
    core = _compile(average, name="average_small")
    video = _video(6)[:, :3, :4]
    assert _sim(tmp_path, core, video, seed=13) == _beats(average(video))


# --------------------------------------------------------------------------- #
# a frame chosen by a register: comparisons on a register are hardware
# --------------------------------------------------------------------------- #

def by_equality(video, pos):
    """The live frame where the place is 1, the one before elsewhere:
    `pos == 1` is a comparator while traced, not Python's identity."""
    v = video.astype(np.int32)
    return np.where(pos == 1, v[1:], v[:-1]).astype(np.uint16)


def by_parity(video, pos):
    """The same choice by the place's low bit: a register's remainder is a
    value the same at every pixel, not a position's lattice."""
    v = video.astype(np.int32)
    return np.where((pos & 1) == 1, v[1:], v[:-1]).astype(np.uint16)


@pytest.mark.parametrize("model", [by_equality, by_parity],
                         ids=["equality", "parity"])
def test_a_frame_chosen_by_a_register_compare(tmp_path, model):
    core = _compile(model, (POS,), name=model.__name__)
    places = [1, 2, 3, 1, 0, 1, 2]
    video = _video(8)
    want = [model(video[n - 1:n + 1], places[n])[0]
            for n in range(1, FRAMES)]
    got = _sim(tmp_path, core, video, seed=17,
               values=[{"pos": p} for p in places])
    assert got == _beats(np.array(want))
    assert len({model(video[:2], p).tobytes() for p in range(4)}) == 2


def test_a_register_outside_a_trace_is_equal_only_to_itself():
    other = Param("pos", bits=2)
    assert POS == POS and POS != other and not (POS == 1)
    assert len({POS, other}) == 2
