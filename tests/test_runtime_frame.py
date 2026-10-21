# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""One core, any frame up to the size it was traced at, each frame's size given at run time.

Each core is traced at a maximum size and then driven, in ONE elaboration,
with frames of several sizes -- the size changing between frames as a mode
change changes it -- and every pixel and every frame end is compared with
the model run at that frame's own size, its line ends with it. A pointwise
model, phase planes, a window and a kept window each meet the frame's edges
differently, so each has its model here.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, generate, to_ir
from np2hw.ir import Param
from np2hw.stream import context_bits, unpack_context
from np2hw.verilog import context_word

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")


BITS = 10
FULL = (1 << BITS) - 1
MAX_W, MAX_H = 24, 16
# the traced size, smaller ones, a wide-and-short one, and back to the
# traced size: every change of size is a change between frames
SIZES = [(24, 16), (12, 8), (16, 6), (24, 4), (8, 10), (24, 16)]


def phase_offset(img, py, px, offset):
    value = img.astype(np.int32)
    out = np.empty_like(value)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            out[rows::2, cols::2] = (value[rows::2, cols::2]
                                     + offset[i, j]).clip(0, FULL)
    return out.astype(np.uint16)


KNEE = 600


def curve(img):
    value = img.astype(np.int32)
    return np.where(value > KNEE, KNEE + ((value - KNEE) >> 2),
                    value).astype(np.uint16)


def mix(pixel):
    """Three channels in one word, three out: a colour matrix's shape."""
    r, g, b = (pixel[..., k].astype(np.int32) for k in range(3))
    lanes = [(2 * r - g + b) >> 1, (r + 2 * g + b) >> 2, (b * 3 - r) >> 1]
    return np.stack([lane.clip(0, 255) for lane in lanes],
                    axis=-1).astype(np.uint16)


def box(img):
    """A valid 3x3 window: the output is the input less the kernel."""
    x = img.astype(np.int32)
    return ((x[:-2, :-2] + x[:-2, 2:] + x[2:, :-2] + x[2:, 2:]
             + 4 * x[1:-1, 1:-1]) >> 3).astype(np.uint16)


def gauss(img):
    """An edge-padded 3x3 window: the output is the input's size."""
    x = np.pad(img.astype(np.int32), 1, mode="edge")
    return ((x[:-2, :-2] + 2 * x[:-2, 1:-1] + x[:-2, 2:]
             + 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] + 2 * x[1:-1, 2:]
             + x[2:, :-2] + 2 * x[2:, 1:-1] + x[2:, 2:]) >> 4).astype(np.uint16)


def pick(img, enable):
    """np.where on a register over a shared window: a mux."""
    x = img.astype(np.int32)
    return np.where(enable, x[1:, 1:], (x[:-1, :-1] + x[1:, 1:]) >> 1
                    ).astype(np.uint16)


def crop(img, y0, x0, h, w):
    """A window registers place and size: in NumPy an ordinary crop."""
    return img[y0:, x0:][:h, :w]


def cross(img, py, px):
    """A same-size 5x5 stencil, edge-padded, the CFA phase choosing between
    the centre and a cross two pixels out: a demosaic's shape."""
    x = np.pad(img.astype(np.int32), 2, mode="edge")
    c = x[2:-2, 2:-2]
    far = (x[:-4, 2:-2] + x[4:, 2:-2] + x[2:-2, :-4] + x[2:-2, 4:]) >> 2
    out = np.empty_like(c)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            src = c if (i + j) % 2 == 0 else far
            out[rows::2, cols::2] = src[rows::2, cols::2]
    return out.astype(np.uint16)


def _sim(tmp_path, core, frames, values, blank=None):
    """Drive `frames` through one elaboration of `core`, each frame with its
    own size in its context, honouring the handshake, with a vertical blank
    between frames long enough for a stencil's bottom flush. Returns each
    output beat as (data, eol, last), and the frame size the core hands on
    in its context at each frame's last beat."""
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    names = [n for n, _ in core["params"]]
    ports = "".join(f"    reg [{b-1}:0] param_{n};\n" for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n" for n in names)
    sets = "".join(f"        param_{n} = {int(values.get(n, 0))};\n"
                   for n in names)
    layout = core["interface"]["input"]["ctx"]
    bits = context_bits(layout)
    blank = core.get("vdrain", 8) + 8 if blank is None else blank
    feed, words = [], []
    for k, frame in enumerate(frames):
        h, w = frame.shape
        feed.append(f"        in_ctx = {context_word(core, {'width': w, 'height': h})};\n"
                    f"        send({len(words)}, {w * h});\n"
                    f"        repeat ({blank}) @(negedge clk);\n")
        words += [int(v) for v in frame.ravel()]
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in words))
    tb = f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0;
    reg [{in_bits-1}:0] in_data = 0;
{ports}    wire in_ready, out_valid, out_sof, out_eol, out_last;
    reg out_ready = 1;
    wire [{out_bits-1}:0] out_data;
    reg [{bits-1}:0] in_ctx = 0;
    wire [{bits-1}:0] out_ctx;
    reg [{in_bits-1}:0] img [0:{len(words)-1}];
    integer f, i;
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_sof(in_sof), .in_data(in_data), .in_ctx(in_ctx),
{conns}        .out_valid(out_valid), .out_ready(out_ready), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_data(out_data),
        .out_ctx(out_ctx));
    always @(posedge clk) if (out_valid && out_ready)
        $fdisplay(f, "%0d %0d %0d %0h", out_data, out_eol, out_last,
                  out_ctx);
    task send(input integer at, input integer n);
        begin
            for (i = 0; i < n; i = i + 1) begin
                in_data = img[at + i]; in_sof = (i == 0); in_valid = 1;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            in_valid = 0; in_sof = 0;
        end
    endtask
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
{sets}        in_ctx = {context_word(core, {'width': frames[0].shape[1], 'height': frames[0].shape[0]})};
        repeat (3) @(negedge clk); rst = 0;
        @(negedge clk);
{"".join(feed)}        repeat ({core.get("vdrain", 8) + 8}) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
"""
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(tb)
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "core.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=300)
    rows = (tmp_path / "out.txt").read_text().split("\n")
    rows = [r.split() for r in rows if r.strip()]
    beats = [tuple(map(int, r[:3])) for r in rows]
    handed = [unpack_context(layout, int(r[3], 16)) for r in rows if r[2] == "1"]
    return beats, [(c["width"], c["height"]) for c in handed]


def _check(tmp_path, fn, params, values, model_args, name, channels=1,
           sizes=SIZES, blank=None):
    bits = BITS if channels == 1 else 8 * channels
    _, line = to_ir(fn, Image2D("img", MAX_W, MAX_H, bits=bits), *params,
                    channels=channels)
    core = generate(line, module_name=name)
    assert [f[0] for f in core["interface"]["input"]["ctx"]][:2] == \
        ["width", "height"]
    rng = np.random.default_rng(7)
    top = FULL if channels == 1 else (1 << bits) - 1
    frames = [rng.integers(0, top + 1, (h, w)).astype(np.int64)
              for w, h in sizes]
    beats, sizes = _sim(tmp_path, core, frames, values, blank)
    want, shapes = [], []
    for frame in frames:
        if channels == 1:
            out = fn(frame.astype(np.uint16), *model_args).astype(np.int64)
        else:
            # the word unpacked as the model reads it, and its result packed
            # at the field the core says it packs
            lanes = np.stack([(frame >> (8 * k)) & 255 for k in range(channels)],
                             axis=-1)
            res = fn(lanes, *model_args).astype(np.int64)
            field = core["interface"]["output"]["field_bits"]
            out = sum(res[..., k] << (field * k) for k in range(res.shape[-1]))
        cols = out.shape[1]
        if out.size:
            shapes.append((cols, out.shape[0]))
        out = out.ravel()
        want += [(int(v), int(i % cols == cols - 1), int(i == out.size - 1))
                 for i, v in enumerate(out)]
    assert len(beats) == len(want), (len(beats), len(want))
    for i, (got, exp) in enumerate(zip(beats, want)):
        assert got == exp, f"beat {i}: core {got}, model {exp}"
    # the size handed to the next block is the frame this one put out
    assert sizes == shapes, (sizes, shapes)


def test_a_phase_core_follows_the_frame_it_is_given(tmp_path):
    params = [Param("py", bits=1), Param("px", bits=1),
              Param("offset", bits=12, signed=True, shape=(2, 2))]
    off = np.array([[-40, 200], [-200, 40]])
    values = {"py": 1, "px": 0, "offset_0_0": -40, "offset_0_1": 200,
              "offset_1_0": -200, "offset_1_1": 40}
    _check(tmp_path, phase_offset, params, values, (1, 0, off), "rt_phase")


def test_an_expression_core_follows_the_frame_it_is_given(tmp_path):
    _check(tmp_path, curve, [], {}, (), "rt_expr")


def test_a_three_channel_expression_core_follows_the_frame_it_is_given(
        tmp_path):
    _check(tmp_path, mix, [], {}, (), "rt_stack", channels=3)


def test_a_phase_stencil_follows_the_frame_it_is_given(tmp_path):
    params = [Param("py", bits=1), Param("px", bits=1)]
    _check(tmp_path, cross, params, {"py": 0, "px": 1}, (0, 1), "rt_stencil")


def test_a_window_core_follows_the_frame_it_is_given(tmp_path):
    _check(tmp_path, box, [], {}, (), "rt_window")


def test_a_padded_window_core_follows_the_frame_it_is_given(tmp_path):
    _check(tmp_path, gauss, [], {}, (), "rt_gauss")


def test_a_mux_core_follows_the_frame_it_is_given(tmp_path):
    _check(tmp_path, pick, [Param("enable", bits=1)], {"enable": 0}, (0,),
           "rt_mux")


WINDOW = {"y0": 3, "x0": 5, "h": 9, "w": 14}


def _window_params():
    return [Param(n, bits=16) for n in ("y0", "x0", "h", "w")]


def test_a_window_keeps_what_its_registers_place(tmp_path):
    """The source streams through and the window is kept: each frame's
    window, ends and all, as NumPy's own slicing has it -- a frame smaller
    than the window ending at its own edges."""
    args = tuple(WINDOW[n] for n in ("y0", "x0", "h", "w"))
    _check(tmp_path, crop, _window_params(), WINDOW, args, "rt_crop")


def test_a_window_passes_a_packed_word_whole(tmp_path):
    args = tuple(WINDOW[n] for n in ("y0", "x0", "h", "w"))
    _check(tmp_path, crop, _window_params(), WINDOW, args, "rt_crop3",
           channels=3)


def test_a_window_takes_frames_taller_than_it_was_traced_at(tmp_path):
    """A window keeps lines, not frames: the traced width is what its line
    buffers hold, and its height bounds nothing it builds."""
    _check(tmp_path, gauss, [], {}, (), "rt_tall",
           sizes=[(24, 40), (12, 25), (24, 16)])


def short(img):
    """Two reads at offsets that stop short of the frame: the rows and
    columns past them are taken and never read, and the output is the
    slice's own size, whatever the frame's."""
    x = img.astype(np.int32)
    return ((x[1:9, 2:14] + x[2:10, 3:15]) >> 1).astype(np.uint16)


def test_a_crop_short_of_the_frame_hands_on_its_own_size(tmp_path):
    _check(tmp_path, short, [], {}, (), "rt_short",
           sizes=[(24, 16), (20, 14), (16, 11), (24, 16)])


def test_a_new_size_waits_behind_the_last_frames_flush(tmp_path):
    """Frames of different sizes, back to back: each waits while the frame
    before it is flushed, and that flush runs at the size of the frame it
    finishes, not the size now waiting."""
    _check(tmp_path, gauss, [], {}, (), "rt_back",
           sizes=[(24, 16), (10, 6), (20, 12), (8, 4)], blank=0)


def unpadded(img):
    x = img.astype(np.int32)
    return ((x[:-2, 1:-1] + x[2:, 1:-1]) >> 1).astype(np.uint16)


@pytest.mark.parametrize("fn, want", [(gauss, (12, 8)), (unpadded, (10, 6))])
def test_out_of_reset_a_core_hands_on_the_size_its_context_says(
        tmp_path, fn, want):
    """Before any frame, a core's size follows its context: what it hands
    on is the size coming, less its border -- not the most it can hold,
    which a display placing the picture would refuse as too large."""
    _, line = to_ir(fn, Image2D("img", MAX_W, MAX_H, bits=BITS))
    core = generate(line, module_name=f"rst_{fn.__name__}")
    layout = core["interface"]["input"]["ctx"]
    bits = context_bits(layout)
    ports = "".join(f"    reg [{b-1}:0] param_{n} = 0;\n" for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n" for n, _ in core["params"])
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1;
{ports}    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire [{core["out_bits"]-1}:0] out_data;
    reg [{bits-1}:0] in_ctx = 0;
    wire [{bits-1}:0] out_ctx;
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(1'b0), .in_ready(in_ready),
        .in_sof(1'b0), .in_data({core["in_bits"]}'d0),
{conns}        .out_valid(out_valid), .out_ready(1'b1), .out_sof(out_sof),
        .in_ctx(in_ctx), .out_eol(out_eol), .out_last(out_last),
        .out_data(out_data), .out_ctx(out_ctx));
    initial begin
        in_ctx = {context_word(core, {"width": 12, "height": 8})};
        repeat (3) @(negedge clk); rst = 0;
        repeat (4) @(negedge clk);
        $display("%0h", out_ctx);
        $finish;
    end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "core.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    out = subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout.split()
    handed = unpack_context(layout, int(out[0], 16))
    assert (handed["width"], handed["height"]) == want


def test_an_axis_read_to_its_end_follows_the_frame(tmp_path):
    """`x[:, 1:-1]` reads to one before the end: its last input column is
    unread, and its output is the frame less two columns at every size,
    as a slice counted from the end says -- not the traced count."""
    _check(tmp_path, unpadded, [], {}, (), "rt_rel")
