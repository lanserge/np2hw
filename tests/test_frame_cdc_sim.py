# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw_frame_cdc carries each frame's context across with its samples.

Frames of different sizes cross between two unrelated clocks, the writer
and the reader stalling at random, frames back to back so the next
frame's context crosses while the last one is still draining. Every beat
must arrive as it went in, and the context published with every beat --
the first one while it is merely presented, before it is taken -- must
be its own frame's: here its size and its count, as a stream's CTX
member packs them, the first field in the low bits.
"""
import shutil
import subprocess

import pytest

from np2hw import hdl

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="iverilog not on PATH")

# (width, height): tiny frames, so several are in the crossing at once
FRAMES = [(5, 3), (1, 1), (7, 2), (3, 4), (2, 1), (6, 3), (4, 4)] * 3

TB = """`timescale 1ns/1ps
module tb;
    reg wclk = 0, rclk = 0, wrst = 1, rrst = 1;
    always #5 wclk = ~wclk;
    always #7.3 rclk = ~rclk;
    reg in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg [11:0] in_data = 0;
    reg [39:0] in_ctx = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire [11:0] out_data;
    wire [39:0] out_ctx;
    reg out_ready = 0;
    np2hw_frame_cdc #(.DATA_BITS(12), .CTX_BITS(40), .A(3), .SA(2)) dut (
        .wclk(wclk), .wrst(wrst), .in_valid(in_valid), .in_ready(in_ready),
        .in_data(in_data), .in_sof(in_sof), .in_eol(in_eol),
        .in_last(in_last), .in_ctx(in_ctx),
        .rclk(rclk), .rrst(rrst), .out_valid(out_valid),
        .out_ready(out_ready), .out_data(out_data), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last),
        .out_ctx(out_ctx));
    integer f, k, n, i, seed = 11;
    reg [15:0] ws [0:63];
    reg [15:0] hs [0:63];
    // every beat the reader TAKES, and every first beat it is merely
    // shown, with the context published beside it
    always @(posedge rclk) if (!rrst && out_valid) begin
        if (out_ready)
            $fdisplay(f, "B %0d %0d %0d %0d %0h", out_data, out_sof,
                      out_eol, out_last, out_ctx);
        else if (out_sof)
            $fdisplay(f, "S %0h", out_ctx);
    end
    always @(negedge rclk) out_ready = ($random(seed) & 3) != 0;
    initial begin
        f = $fopen("out.txt", "w");
{sizes}        repeat (4) @(negedge wclk); wrst = 0; rrst = 0;
        @(negedge wclk);
        for (k = 0; k < {frames}; k = k + 1) begin
            n = ws[k] * hs[k];
            for (i = 0; i < n; i = i + 1) begin
                while (($random(seed) & 3) == 0) @(negedge wclk);
                in_valid = 1; in_data = (k * 97 + i) & 12'hfff;
                in_sof = (i == 0); in_eol = (i % ws[k] == ws[k] - 1);
                in_last = (i == n - 1);
                // the context is the writer's own only around its
                // frame's first beat; after it, it is already the next's
                in_ctx = (i == 0) ? {k[7:0], hs[k], ws[k]} : 40'hdeadbeef55;
                @(posedge wclk); while (!in_ready) @(posedge wclk);
                @(negedge wclk);
                in_valid = 0;
            end
        end
        repeat (400) @(negedge rclk);
        $fclose(f); $finish;
    end
endmodule
"""


def _run(tmp_path):
    sizes = "".join(f"        ws[{k}] = {w}; hs[{k}] = {h};\n"
                    for k, (w, h) in enumerate(FRAMES))
    tb = TB.replace("{sizes}", sizes).replace("{frames}", str(len(FRAMES)))
    (tmp_path / "tb.v").write_text(tb)
    (tmp_path / "dut.v").write_text(hdl.text("np2hw_frame_cdc")
                                    + hdl.text("np2hw_cdc_fifo"))
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "dut.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=120)
    return [r.split() for r in (tmp_path / "out.txt").read_text().splitlines()]


def _fields(word):
    """A context word's (width, height, count)."""
    word = int(word, 16)
    return word & 0xFFFF, (word >> 16) & 0xFFFF, word >> 32


def test_each_beat_crosses_with_its_own_frames_context(tmp_path):
    rows = _run(tmp_path)
    want = []
    for k, (w, h) in enumerate(FRAMES):
        n = w * h
        want += [((k * 97 + i) & 0xfff, int(i == 0), int(i % w == w - 1),
                  int(i == n - 1), (w, h, k)) for i in range(n)]
    beats = [(*map(int, r[1:5]), _fields(r[5])) for r in rows if r[0] == "B"]
    assert len(beats) == len(want), (len(beats), len(want))
    for i, (got, exp) in enumerate(zip(beats, want)):
        assert got == exp, f"beat {i}: crossed {got}, sent {exp}"


def test_a_first_beat_shows_its_context_before_it_is_taken(tmp_path):
    rows = _run(tmp_path)
    # the context shown with an untaken first beat is the frame it starts
    frame, shown = -1, []
    for r in rows:
        if r[0] == "S":
            shown.append((frame + 1, _fields(r[1])))
        elif r[2] == "1":
            frame += 1
    assert shown, "the reader never stalled on a first beat"
    for k, got in shown:
        assert got == (*FRAMES[k], k), (k, got, FRAMES[k])
