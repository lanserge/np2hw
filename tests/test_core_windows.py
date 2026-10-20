# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Windows: line buffers, the lag, the flush, the edges.

A model that reads a padded buffer, or the source at an offset, reads a
WINDOW, and the spread of its reads is storage: line buffers for the rows,
a shift register for the columns (docs/one-graph.md, the window rule). The
output trails the input by the lag the reads give and is finished, past the
input's end, in flush steps; the edges are the pad's. Each model is compiled
and its Verilog driven with a run of frames through ONE
elaboration under random backpressure on both sides, and compared with the
model on every pixel -- the corners, where the edge rule meets the flush,
above all.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, generate, make_coords, to_ir
from np2hw.ir import Param

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")

W, H, BITS = 10, 6, 8
FRAMES = 3


def _compile(fn, params=(), clk_ns=None, bits=BITS, channels=1, name="m"):
    _, mod = to_ir(fn, Image2D("img", W, H, bits=bits), *params,
                   channels=channels)
    return generate(mod, module_name=name, clk_ns=clk_ns)


def _sim(tmp_path, core, frames, values=None, seed=0, gap=True):
    """`frames` through one elaboration, valid and ready both random; with
    `gap`, idle cycles between frames, else the next frame waiting at once.
    Every output beat as (data, sof, eol, last)."""
    values = values or {}
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    regs = "".join(f"    reg [{b - 1}:0] param_{n};\n" for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n"
                    for n, _ in core["params"])
    sets = "".join(f"        param_{n} = {int(values.get(n, core["param_defaults"][n])) & ((1 << b) - 1)};\n"
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
            {"repeat (5) @(negedge clk);" if gap else ""}
        end
        repeat ({4 * (W + 4) * 4 + 80}) @(negedge clk);
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
        frame = np.asarray(frame)
        h, w = frame.shape
        flat = frame.ravel()
        beats += [(int(v), int(i == 0), int(i % w == w - 1),
                   int(i == flat.size - 1)) for i, v in enumerate(flat)]
    return beats


def _frames(seed, bits=BITS):
    return np.random.default_rng(seed).integers(0, 1 << bits, (FRAMES, H, W))


def _check(tmp_path, fn, params=(), values=None, model_args=(), seed=0,
           clk_ns=None, name="m", gap=True):
    core = _compile(fn, params, clk_ns=clk_ns, name=name)
    frames = _frames(seed)
    want = [fn(f.astype(np.uint16), *model_args) for f in frames]
    assert _sim(tmp_path, core, frames, values, seed=seed, gap=gap) == \
        _beats(want)
    return core


def blur(img):
    """The design note's 3x3 with replicated edges."""
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((x[:-2, :-2] + 2 * x[:-2, 1:-1] + x[:-2, 2:]
             + 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] + 2 * x[1:-1, 2:]
             + x[2:, :-2] + 2 * x[2:, 1:-1] + x[2:, 2:]) // 16).astype(np.uint8)


def test_a_3x3_with_replicated_edges(tmp_path):
    core = _check(tmp_path, blur, name="blur")
    assert [m["depth"] for m in core["memories"]] == [W, W]


def test_the_next_frame_waits_for_the_flush(tmp_path):
    """No idle between frames: the next frame's first pixel waits behind
    the last frame's flush, which a start of frame must not cut short."""
    _check(tmp_path, blur, name="blur_tight", seed=1, gap=False)


def sparse(img):
    """Five rows and columns of reach, three of the taps: line buffers for
    all four rows back, shift registers for all four columns."""
    x = np.pad(img.astype(np.int32), 2, mode="edge")

    def at(r, c):
        return x[2 + r:2 + r + H, 2 + c:2 + c + W]
    return ((2 * at(0, 0) + at(-2, 1) + at(2, -2)) >> 2).astype(np.uint16)


def test_a_5x5_reach_with_a_pad_of_two(tmp_path):
    core = _check(tmp_path, sparse, seed=2, name="sparse")
    assert len(core["memories"]) == 4


def cross(img):
    """An unpadded window: the output is the frame less its border."""
    x = img.astype(np.int32)
    return ((x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) >> 2
            ).astype(np.uint16)


def test_an_unpadded_window_is_the_frame_less_its_border(tmp_path):
    core = _check(tmp_path, cross, seed=3, name="cross_win")
    assert (core["out_rows"], core["out_cols"]) == (H - 2, W - 2)


def dark(img):
    """The 3x3 on a ZERO pad: every row and column off the frame reads 0."""
    x = np.pad(img.astype(np.uint16), 1)
    return ((x[:-2, :-2] + 2 * x[:-2, 1:-1] + x[:-2, 2:]
             + 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] + 2 * x[1:-1, 2:]
             + x[2:, :-2] + 2 * x[2:, 1:-1] + x[2:, 2:]) // 16).astype(np.uint8)


def back(img):
    """A zero pad read on its top-left only, a signed difference out."""
    x = np.pad(img.astype(np.int16), ((2, 0), (1, 0)))
    return (x[2:, 1:] - x[:-2, :-1] + 256).astype(np.uint16)


@pytest.mark.parametrize("fn", [dark, back])
def test_a_zero_pad_reads_zero_off_the_frame(tmp_path, fn):
    _check(tmp_path, fn, seed=11, name=fn.__name__)


def composed(img):
    """A 2x2 box applied to a 2x2 box's result: one 3x3 window over the
    source, the inner box computed at each of the outer one's taps."""
    x = img.astype(np.uint16)
    a = x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:]
    return ((a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]) >> 4
            ).astype(np.uint8)


def test_a_window_over_a_result_is_a_window_over_the_source(tmp_path):
    core = _check(tmp_path, composed, seed=12, name="composed")
    assert core.line_buffers == 2 and core["N"] == 2


def tail(img):
    """A window that stops short of the frame: rows and columns past it are
    taken from the stream and never read."""
    x = img.astype(np.int32)
    return ((x[1:4, 2:7] + x[2:5, 1:6]) >> 1).astype(np.uint16)


@pytest.mark.parametrize("gap", [True, False])
def test_a_window_short_of_the_frame_takes_the_rest_unread(tmp_path, gap):
    core = _check(tmp_path, tail, seed=9, name="tail", gap=gap)
    assert (core["out_rows"], core["out_cols"]) == (3, 5)


def declared(img):
    """The output's raster declared, the input read at offsets into it."""
    y, x = make_coords(img, 3, 4)
    return ((img[y + 1, x + 5].astype(np.int32) + img[y + 3, x + 2]) >> 1
            ).astype(np.uint16)


def test_a_declared_raster_reads_its_window(tmp_path):
    core = _check(tmp_path, declared, seed=10, name="declared")
    assert (core["out_rows"], core["out_cols"]) == (3, 4)


def rows_only(img):
    x = np.pad(img.astype(np.int32), ((1, 1), (0, 0)), mode="edge")
    return ((x[:-2] + 2 * x[1:-1] + x[2:]) >> 2).astype(np.uint16)


def cols_only(img):
    x = np.pad(img.astype(np.int32), ((0, 0), (2, 0)), mode="edge")
    return ((x[:, :-2] + x[:, 2:]) >> 1).astype(np.uint16)


def test_a_window_down_the_rows_only(tmp_path):
    _check(tmp_path, rows_only, seed=4, name="rows_only")


def test_a_window_along_the_columns_only(tmp_path):
    core = _check(tmp_path, cols_only, seed=5, name="cols_only")
    assert core["memories"] == []


PY, PX = Param("py", bits=1), Param("px", bits=1)


def planes(img, py, px):
    """CFA phase planes over a window: each plane's own arithmetic, chosen
    by the OUTPUT pixel's parity and the phase registers."""
    x = np.pad(img.astype(np.int32), 1, mode="edge")
    c, n, s = x[1:-1, 1:-1], x[:-2, 1:-1], x[2:, 1:-1]
    w, e = x[1:-1, :-2], x[1:-1, 2:]
    out = np.zeros_like(c)
    out[py::2, px::2] = c[py::2, px::2]
    out[py::2, 1 - px::2] = ((w + e) >> 1)[py::2, 1 - px::2]
    out[1 - py::2, px::2] = ((n + s) >> 1)[1 - py::2, px::2]
    out[1 - py::2, 1 - px::2] = ((n + s + w + e) >> 2)[1 - py::2, 1 - px::2]
    return out.astype(np.uint16)


@pytest.mark.parametrize("py, px", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_phase_planes_over_a_window(tmp_path, py, px):
    _check(tmp_path, planes, (PY, PX), {"py": py, "px": px}, (py, px),
           seed=6, name="planes")


def two_channels(pixel):
    """Two channels a word, each padded and windowed on its own, stacked."""
    a = np.pad(pixel[..., 0].astype(np.int32), 1, mode="edge")
    b = np.pad(pixel[..., 1].astype(np.int32), 1, mode="edge")
    left = (a[1:-1, :-2] + a[1:-1, 2:]) >> 1
    down = (b[:-2, 1:-1] + b[2:, 1:-1] + 2 * a[1:-1, 1:-1]) >> 2
    return np.stack([left, down], axis=-1).astype(np.uint16)


def test_two_channels_padded_apart_and_stacked(tmp_path):
    core = _compile(two_channels, bits=16, channels=2, name="two")
    frames = _frames(7, bits=16)
    field = core["interface"]["output"]["field_bits"]
    want = []
    for f in frames:
        res = two_channels(np.stack([f & 255, f >> 8], axis=-1)).astype(
            np.int64)
        want.append(res[..., 0] + (res[..., 1] << field))
    assert _sim(tmp_path, core, frames, seed=7) == _beats(want)


GAIN = Param("gain", bits=8)


def sharpen(img, gain):
    x = np.pad(img.astype(np.int32), 1, mode="edge")
    c = x[1:-1, 1:-1]
    hp = 4 * c - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]
    return (c + ((hp * gain) >> 5)).clip(0, 255).astype(np.uint16)


def test_a_window_cut_into_stages(tmp_path):
    core = _check(tmp_path, sharpen, (GAIN,), {"gain": 23}, (23,), seed=8,
                  clk_ns=5.0, name="sharpen")
    assert core["pipeline_stages"] > 1
