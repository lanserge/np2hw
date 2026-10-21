# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A core whose stream carries its context: each frame's facts ride the
stream's CTX member with its first beat, and the core hands them on with
the frame.

The image states the layout -- here the frame's width and height, which
follow its size, and a count -- and the core takes no size port and no
port for a value a frame brings: its frame's size and the count it reads
are fields of `in_ctx`, latched with the frame's first beat. It puts out
`out_ctx`, the same fields, the size its own. Frames of two sizes go
through back to back, each with its count, and the consumer stalls at
random: every frame out is the model's at its own count, and the context
beside it is that frame's.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, Param, generate, to_ir

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="iverilog not on PATH")

W, H, BITS = 8, 6, 10
LAYOUT = (("width", 16, "width"), ("height", 16, "height"),
          ("frame_seq", 32, ""))


def rows_mean(pixel, seq):
    """A row and the one above it, averaged, and the count's low bit
    added: the output is a row shorter than the frame, and depends on the
    frame's own count."""
    v = pixel.astype(np.int32)
    return (((v[1:, :] + v[:-1, :]) >> 1) + (seq & 1)).astype(np.uint16)


def _core():
    img = Image2D("p", W, H, bits=BITS, context=LAYOUT)
    _, mod = to_ir(rows_mean, img, Param("frame_seq", bits=32, frame=True))
    return generate(mod, module_name="ctx_core")


def _pack(fields):
    word, lo = 0, 0
    for (name, bits, _), value in zip(LAYOUT, fields):
        word |= (int(value) & ((1 << bits) - 1)) << lo
        lo += bits
    return word


TB = """`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;
    reg in_valid = 0, in_sof = 0, out_ready = 0;
    reg [9:0] in_data = 0;
    reg [63:0] in_ctx = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire [63:0] out_ctx;
    wire [@OB@:0] out_data;
    ctx_core dut (.clk(clk), .rst(rst), .in_valid(in_valid),
        .in_ready(in_ready), .in_sof(in_sof), .in_data(in_data),
        .in_ctx(in_ctx), .out_ctx(out_ctx), .out_valid(out_valid),
        .out_ready(out_ready), .out_sof(out_sof), .out_eol(out_eol),
        .out_last(out_last), .out_data(out_data));
    reg [9:0] pix [0:@N@-1];
    reg [63:0] ctx [0:@F@-1];
    integer heights [0:@F@-1];
    integer f, k, i, at = 0, seed = 5, done = 0;
    always @(posedge clk) if (!rst && out_valid && out_ready) begin
        if (out_sof) $fdisplay(f, "O %0h", out_ctx);
        $fdisplay(f, "B %0d", out_data);
        if (out_last) begin $fdisplay(f, "E"); done = done + 1; end
    end
    always @(negedge clk) out_ready = ($random(seed) & 3) != 0;
    initial begin
        f = $fopen("out.txt", "w");
        $readmemh("pix.hex", pix);
        $readmemh("ctx.hex", ctx);
@HEIGHTS@
        repeat (3) @(negedge clk); rst = 0;
        for (k = 0; k < @F@; k = k + 1) begin
            for (i = 0; i < @W@ * heights[k]; i = i + 1) begin
                in_valid = 1; in_sof = i == 0; in_data = pix[at + i];
                in_ctx = ctx[k];
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            at = at + @W@ * heights[k];
            in_valid = 0;
            repeat (($random(seed) & 7)) @(negedge clk);
        end
        repeat (400) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
"""


def test_a_core_takes_and_hands_on_each_frames_context(tmp_path):
    core = _core()
    assert "param_frame_width" not in core.verilog
    assert "param_frame_seq" not in core.verilog
    rng = np.random.default_rng(3)
    heights = [6, 4, 6, 5]
    seqs = [7, 8, 1000, 3]
    frames = [rng.integers(0, 1 << BITS, (h, W)) for h in heights]
    (tmp_path / "ctx_core.v").write_text(core.verilog)
    (tmp_path / "pix.hex").write_text(
        "".join(f"{int(v):x}\n" for fr in frames for v in fr.ravel()))
    (tmp_path / "ctx.hex").write_text("".join(
        f"{_pack((W, h, s)):x}\n" for h, s in zip(heights, seqs)))
    tb = (TB.replace("@N@", str(sum(f.size for f in frames)))
          .replace("@F@", str(len(frames))).replace("@W@", str(W))
          .replace("@OB@", str(core["out_bits"] - 1))
          .replace("@HEIGHTS@", "".join(
              f"        heights[{k}] = {h};\n" for k, h in enumerate(heights))))
    (tmp_path / "tb.v").write_text(tb)
    subprocess.run(["iverilog", "-g2005", "-o", str(tmp_path / "sim"),
                    str(tmp_path / "tb.v"), str(tmp_path / "ctx_core.v")],
                   check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "-n", str(tmp_path / "sim")], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    got, beats, ctx = [], [], None
    for line in (tmp_path / "out.txt").read_text().splitlines():
        kind, *rest = line.split()
        if kind == "O":
            ctx = int(rest[0], 16)
        elif kind == "B":
            beats.append(int(rest[0]))
        else:
            got.append((ctx, beats))
            beats = []
    assert len(got) == len(frames)
    for (ctx, beats), frame, h, s in zip(got, frames, heights, seqs):
        want = rows_mean(frame, s)
        assert beats == [int(v) for v in want.ravel()], f"frame at {s}"
        # the context out: this frame's count, the size the core put out
        assert ctx == _pack((W, h - 1, s)), f"context of the frame at {s}"
