# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A datapath cut into stages to meet a clock.

Each model is compiled at a budget tight enough to need several stages, so
it is the schedule that meets it or the test fails, and its Verilog is driven with a run of frames through ONE
elaboration under random backpressure on both sides and compared with the
model: every value, and every value's pixel -- a position read in a late
stage must be the pixel's own, carried with it, not where the counters have
got to since.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, coords, generate, to_ir
from np2hw.ir import Param
from np2hw.verilog import context_word

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")

W, H, BITS = 9, 5, 10
FRAMES = 3


def _compile(fn, params=(), clk_ns=None, frames=0, bits=BITS, channels=1,
             name="m"):
    _, mod = to_ir(fn, Image2D("img", W, H, bits=bits, frames=frames),
                   *params, channels=channels)
    return generate(mod, module_name=name, clk_ns=clk_ns)


def _sim(tmp_path, core, frames, values, seed=0):
    """`frames` through one elaboration, valid and ready both random, a gap
    between frames; every output beat as (data, sof, eol, last)."""
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    regs = "".join(f"    reg [{b - 1}:0] param_{n};\n" for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n"
                    for n, _ in core["params"])
    sets = "".join(f"        param_{n} = {int(values.get(n, core['param_defaults'][n])) & ((1 << b) - 1)};\n"
                   for n, b in core["params"])
    words = [int(v) for v in np.asarray(frames).ravel()]
    per = W * H
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in words))
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0, out_ready = 0;
    reg [{in_bits - 1}:0] in_data = 0;
{regs}    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire [{out_bits - 1}:0] out_data;
    reg [{in_bits - 1}:0] img [0:{len(words) - 1}];
    integer f, i, k, seed = {seed};
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_sof(in_sof), .in_data(in_data),
        .in_ctx({context_word(core, values)}), .out_ctx(),
{conns}        .out_valid(out_valid), .out_ready(out_ready), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_data(out_data));
    always @(posedge clk) if (out_valid && out_ready)
        $fdisplay(f, "%0d %0d %0d %0d", out_data, out_sof, out_eol, out_last);
    always @(negedge clk) out_ready <= ($random(seed) & 3) != 0;
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
{sets}        repeat (3) @(negedge clk); rst = 0;
        for (k = 0; k < {len(words) // per}; k = k + 1) begin
            for (i = 0; i < {per}; i = i + 1) begin
                while (($random(seed) & 3) == 0) begin
                    in_valid = 0; @(negedge clk);
                end
                in_data = img[k * {per} + i]; in_sof = (i == 0); in_valid = 1;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            in_valid = 0; in_sof = 0;
            repeat (5) @(negedge clk);
        end
        repeat (60) @(negedge clk);
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


def _beats(frames):
    beats = []
    for frame in frames:
        flat = np.asarray(frame).ravel()
        beats += [(int(v), int(i == 0), int(i % W == W - 1),
                   int(i == flat.size - 1)) for i, v in enumerate(flat)]
    return beats


GAIN = Param("gain", bits=8, description="a gain")
BIAS = Param("bias", bits=10, signed=True, description="an offset")


def polynomial(img, gain, bias):
    """Multiplies and adds in a chain: deeper than one stage at a clock
    worth meeting."""
    x = img.astype(np.int32)
    y = ((x * gain) >> 3) + bias
    z = ((y * y) >> 9) + ((x * 3) >> 1)
    return (((z * gain) >> 6) - x).clip(0, 1023).astype(np.uint16)


def test_a_deep_chain_is_cut_into_stages_that_meet_the_clock(tmp_path):
    one = _compile(polynomial, (GAIN, BIAS), name="poly1")
    core = _compile(polynomial, (GAIN, BIAS), clk_ns=6.5, name="poly")
    assert one["pipeline_stages"] == 1 and core["pipeline_stages"] > 2
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 1 << BITS, (FRAMES, H, W))
    values = {"gain": 37, "bias": -100}
    want = [polynomial(f.astype(np.uint16), 37, -100) for f in frames]
    assert _sim(tmp_path, core, frames, values) == _beats(want)


def tinted(img, gain):
    """The pixel's position, used only at the end of a deep path: it must
    be carried with the pixel to the stage that reads it."""
    y, x = coords(img)
    v = img.astype(np.int32)
    deep = (((v * gain) >> 4) * gain) >> 5
    return ((deep + x * 7 + y * 5) & 1023).astype(np.uint16)


def test_a_position_read_late_is_its_own_pixels(tmp_path):
    core = _compile(tinted, (GAIN,), clk_ns=5.0, name="tinted")
    assert core["pipeline_stages"] > 1
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 1 << BITS, (FRAMES, H, W))
    want = [tinted(f.astype(np.uint16), 23) for f in frames]
    assert _sim(tmp_path, core, frames, {"gain": 23}, seed=3) == _beats(want)


def lanes(pixel, gain):
    """Three channels out of two, each a different depth: a shallow lane
    must wait for the deep one, so the word assembles from one pixel."""
    a, b = (pixel[..., k].astype(np.int32) for k in range(2))
    deep = (((a * gain) >> 3) * b) >> 8
    return np.stack([a, (deep & 255), (a + b) >> 1],
                    axis=-1).astype(np.uint16)


def test_lanes_of_different_depth_assemble_from_one_pixel(tmp_path):
    core = _compile(lanes, (GAIN,), clk_ns=6.0, bits=16, channels=2,
                    name="lanes")
    assert core["pipeline_stages"] > 1
    rng = np.random.default_rng(4)
    frames = rng.integers(0, 1 << 16, (FRAMES, H, W))
    field = core["interface"]["output"]["field_bits"]
    want = []
    for f in frames:
        res = lanes(np.stack([f & 255, f >> 8], axis=-1), 41).astype(np.int64)
        want.append(sum(res[..., k] << (field * k) for k in range(3)))
    assert _sim(tmp_path, core, frames, {"gain": 41}, seed=5) == _beats(want)


def history(video, gain):
    """A video's past, read at stage 0 with the pixel, through a deep path."""
    v = video.astype(np.int32)
    d = v[1:] - v[:-1]
    return (((d * gain) >> 3) * gain >> 4).clip(0, 1023).astype(np.uint16)


def test_a_frame_store_feeds_a_staged_datapath(tmp_path):
    core = _compile(history, (GAIN,), clk_ns=6.0, frames=4, name="history")
    assert core["pipeline_stages"] > 1 and core["memories"]
    rng = np.random.default_rng(6)
    video = rng.integers(0, 1 << BITS, (5, H, W))
    want = history(video, 29)
    assert _sim(tmp_path, core, video, {"gain": 29}, seed=7) == _beats(want)


def test_an_operation_alone_past_the_budget_is_refused():
    with pytest.raises(ValueError, match="alone estimates"):
        _compile(polynomial, (GAIN, BIAS), clk_ns=2.0)
