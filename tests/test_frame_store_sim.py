# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw_frame_store, on the block-RAM binding of its seam, hands its
consumer the newest frames whole, at the consumer's own pace.

The source sends frames in bursts -- four close together, then a long
gap, as a time-of-flight sensor does -- and the consumer takes a frame of
frames whenever it likes, stalling at random inside one. What is checked is
what the store's own contract says, from outside it:

- channel k of every frame out is input frame n - k, sample for sample,
  and the size and context published with it are frame n's;
- n is the newest frame whole when the frame out began: never older than
  the newest whole when the previous frame out ended, never newer than the
  newest whole at its own first beat, and never going backwards;
- nothing goes out before FRAMES frames are whole, and a change of size
  starts the history over;
- with too few slots for the burst, `overrun` says so.
"""
import shutil
import subprocess

import pytest

from np2hw import hdl

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="iverilog not on PATH")

BITS, FRAMES = 12, 4

TB = """`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;
    reg in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg [11:0] in_data = 0;
    reg [15:0] in_ctx = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last, cold, overrun;
    wire [{frames}*12-1:0] out_data;
    wire [15:0] out_ctx, out_source;
    reg out_ready = 0;
    wire wr_valid, wr_ready, wr_sof, wr_last, wr_done, rd_start, rd_valid,
         rd_ready, mem_done;
    wire [11:0] wr_data;
    wire [$clog2({slots})-1:0] wr_slot, wr_done_slot, mem_done_slot;
    // the binding's "whole", late by DELAY cycles: what external memory's
    // acknowledged writes look like to the store
    reg [{delay}:0] late = 0;
    reg [$clog2({slots})-1:0] late_slot [0:{delay}];
    integer d;
    always @(posedge clk) begin
        late <= {late[{delay}-1:0], mem_done};
        late_slot[0] <= mem_done_slot;
        for (d = 1; d <= {delay}; d = d + 1) late_slot[d] <= late_slot[d-1];
    end
    assign wr_done = {late} ? late[{delay}-1] : mem_done;
    assign wr_done_slot = {late} ? late_slot[{delay}-1] : mem_done_slot;
    wire [{frames}*$clog2({slots})-1:0] rd_slots;
    wire [31:0] rd_pixels;
    wire [{frames}*12-1:0] rd_data;
    np2hw_frame_store #(.BITS(12), .FRAMES({frames}), .SLOTS({slots}),
                        .CTX_BITS(16)) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .in_ctx(in_ctx),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ctx(out_ctx), .out_source(out_source), .cold(cold),
        .overrun(overrun),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_sof(wr_sof), .wr_last(wr_last), .wr_slot(wr_slot),
        .wr_done(wr_done), .wr_done_slot(wr_done_slot),
        .rd_start(rd_start), .rd_slots(rd_slots), .rd_pixels(rd_pixels),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data));
    np2hw_frame_mem_bram #(.BITS(12), .FRAMES({frames}), .SLOTS({slots}),
                           .PIXELS(64)) mem (
        .clk(clk), .rst(rst),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_sof(wr_sof), .wr_last(wr_last), .wr_slot(wr_slot),
        .wr_done(mem_done), .wr_done_slot(mem_done_slot),
        .rd_start(rd_start), .rd_slots(rd_slots), .rd_pixels(rd_pixels),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data));
    integer f, k, n, i, g, cyc = 0, seed = {seed};
    reg [15:0] ws [0:255];
    reg [15:0] hs [0:255];
    reg [15:0] gap [0:255];
    always @(posedge clk) cyc <= cyc + 1;
    // every frame the source finished, every beat the consumer took
    always @(posedge clk) if (!rst) begin
        if (in_valid && in_ready && in_last) $fdisplay(f, "I %0d", cyc);
        if (out_valid && out_ready) begin
            if (out_sof)
                $fdisplay(f, "O %0d %0d %0d", cyc, out_source, out_ctx);
            $fdisplay(f, "B %0h %0d %0d", out_data, out_eol, out_last);
            if (out_last) $fdisplay(f, "E %0d", cyc);
        end
    end
    // the consumer: stalls inside a frame, and waits between frames
    integer rest = 0;
    always @(negedge clk) begin
        if (rest > 0) begin rest = rest - 1; out_ready = 0; end
        else out_ready = ($random(seed) & 3) != 0;
        if (out_valid && out_ready && out_last)
            rest = {consumer_wait} + ($random(seed) & 63);
    end
    initial begin
        f = $fopen("out.txt", "w");
{tables}        repeat (4) @(negedge clk); rst = 0;
        @(negedge clk);
        for (k = 0; k < {count}; k = k + 1) begin
            n = ws[k] * hs[k];
            for (i = 0; i < n; i = i + 1) begin
                in_valid = 1; in_data = (k * 97 + i) & 12'hfff;
                in_sof = (i == 0); in_eol = (i % ws[k] == ws[k] - 1);
                in_last = (i == n - 1);
                in_ctx = k * 7 + 3;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
                in_valid = 0;
            end
            for (g = 0; g < gap[k]; g = g + 1) @(negedge clk);
        end
        repeat (3000) @(negedge clk);
        $fdisplay(f, "R %0d", overrun);
        $fclose(f); $finish;
    end
endmodule
"""


def simulate(tmp_path, sizes, gaps, slots, consumer_wait, seed=11, delay=0):
    tables = "".join(
        f"        ws[{k}] = {w}; hs[{k}] = {h}; gap[{k}] = {g};\n"
        for k, ((w, h), g) in enumerate(zip(sizes, gaps)))
    (tmp_path / "tb.v").write_text(
        TB.replace("{frames}", str(FRAMES)).replace("{slots}", str(slots))
        .replace("{seed}", str(seed)).replace("{tables}", tables)
        .replace("{count}", str(len(sizes)))
        .replace("{consumer_wait}", str(consumer_wait))
        .replace("{delay}", str(max(delay, 1)))
        .replace("{late}", "1" if delay else "0"))
    sources = [hdl.path("np2hw_frame_store"),
               hdl.path("np2hw_frame_mem_bram")]
    subprocess.run(["iverilog", "-g2005", "-o", str(tmp_path / "sim"),
                    str(tmp_path / "tb.v"), *map(str, sources)],
                   check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "-n", str(tmp_path / "sim")], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    return (tmp_path / "out.txt").read_text().splitlines()


def parse(lines):
    done, frames, overrun = [], [], None
    for line in lines:
        kind, *rest = line.split()
        if kind == "I":
            done.append(int(rest[0]))
        elif kind == "O":
            cyc, source, ctx = map(int, rest)
            frames.append({"start": cyc, "source": source, "ctx": ctx,
                           "beats": [], "end": None})
        elif kind == "B":
            frames[-1]["beats"].append((int(rest[0], 16), int(rest[1]),
                                        int(rest[2])))
        elif kind == "E":
            frames[-1]["end"] = int(rest[0])
        elif kind == "R":
            overrun = int(rest[0])
    for out in frames:
        # the size a frame out has is the one its beats draw
        eols = [i for i, (_, eol, _) in enumerate(out["beats"]) if eol]
        w = eols[0] + 1 if eols else 0
        out["size"] = (w, len(out["beats"]) // w if w else 0)
    return done, frames, overrun


def check(sizes, done, frames, delay=0):
    """Every frame out against the store's contract, from outside it."""
    def newest_whole(cyc):
        # a frame is whole the cycle after its last beat is written, and
        # the binding says so `delay` cycles later
        whole = [k for k, t in enumerate(done) if t + 1 + delay < cyc]
        return whole[-1] if whole else -1

    # the frame the simulation ended inside is no frame
    frames = [f for f in frames if f["end"] is not None]
    assert frames, "nothing came out"
    last, previous_end = -1, None
    for out in frames:
        n = out["source"]
        assert n >= FRAMES - 1, f"frame out from {n}, before history"
        assert n >= last, f"frame out went back from {last} to {n}"
        assert n <= newest_whole(out["start"]), \
            f"frame out from {n}, not yet whole at its first beat"
        if previous_end is not None:
            assert n >= newest_whole(previous_end - 2), \
                f"frame out from {n} when a newer one was whole"
        w, h = sizes[n]
        assert out["size"] == (w, h) and out["ctx"] == n * 7 + 3
        assert all(sizes[n - k] == (w, h) for k in range(FRAMES)), \
            "a history across a change of size"
        assert len(out["beats"]) == w * h
        for i, (word, eol, lst) in enumerate(out["beats"]):
            assert eol == (i % w == w - 1) and lst == (i == w * h - 1)
            for k in range(FRAMES):
                got = (word >> (k * BITS)) & ((1 << BITS) - 1)
                assert got == ((n - k) * 97 + i) & 0xFFF, \
                    f"frame out from {n}: channel {k} pixel {i}"
        last, previous_end = n, out["end"]


def bursts(count):
    """Four frames close together, then a long gap: the ToF's cadence."""
    return [8 if k % 4 != 3 else 400 for k in range(count)]


def test_frames_come_out_whole_at_the_consumers_pace(tmp_path):
    sizes = [(6, 4)] * 40
    lines = simulate(tmp_path, sizes, bursts(40), slots=8,
                     consumer_wait=150)
    done, frames, overrun = parse(lines)
    check(sizes, done, frames)
    assert overrun == 0
    sources = [f["source"] for f in frames]
    # a slow consumer skips frames, and with long gaps repeats some
    assert len(set(sources)) < 40 - FRAMES + 1
    assert any(a == b for a, b in zip(sources, sources[1:]))


@pytest.mark.parametrize("delay", [0, 30])
def test_a_change_of_size_starts_the_history_over(tmp_path, delay):
    """A consumer reading without pause sees every history there is; with
    the binding's "whole" arriving after the next frame has begun, a frame
    of the old size must still not count in the new history."""
    sizes = [(6, 4)] * 12 + [(5, 3)] * 12
    lines = simulate(tmp_path, sizes, [8] * 24, slots=16, consumer_wait=0,
                     delay=delay)
    done, frames, overrun = parse(lines)
    check(sizes, done, frames, delay)
    assert any(f["size"] == (5, 3) for f in frames)
    assert overrun == 0


def test_frames_coming_faster_than_the_slack_are_told(tmp_path):
    """One slot of slack, frames back to back, and a consumer that takes a
    frame of frames no faster than a frame comes in: the source reaches a
    slot being read, and `overrun` says so."""
    sizes = [(6, 4)] * 24
    lines = simulate(tmp_path, sizes, [2] * 24, slots=FRAMES + 1,
                     consumer_wait=0)
    assert parse(lines)[2] == 1


def test_frames_written_and_not_yet_whole_take_their_slots_too(tmp_path):
    """A binding slow to say "whole" keeps frames in flight that hold slots:
    with eight, a read can begin on a frame already being overwritten, and
    `overrun` says so."""
    sizes = [(5, 3)] * 24
    lines = simulate(tmp_path, sizes, [8] * 24, slots=8, consumer_wait=0,
                     delay=30)
    assert parse(lines)[2] == 1


def test_a_read_that_begins_on_frames_being_overwritten_is_told(tmp_path):
    """Ten frames begin before the binding has said any four are whole, then
    the source goes quiet: the first read begins on a frame whose slot the
    tenth already took, and no write begins during it -- only the read's own
    look at the ring can tell."""
    sizes = [(5, 3)] * 10
    lines = simulate(tmp_path, sizes, [8] * 9 + [2000], slots=8,
                     consumer_wait=0, delay=300)
    assert parse(lines)[2] == 1
