# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The frame store in external memory, its stream starting over while its
memory is busy.

A stream starts over whenever its source does -- a link lost and found --
and the store's `rst` comes with it, at any moment: mid-burst, an address
offered and not yet taken, read beats still to come. The memory's own
reset does not come. The memory here holds the master to AXI as it goes,
and after the reset the store must put out the history that began after
it, frame for frame, as if the one before had never been.
"""
import shutil
import subprocess

import numpy as np
import pytest

from test_frame_store_model import (FOUND, IMPLEMENTATIONS, MEMORY,
                                    MEMORY_PORTS, MEMORY_WIRES)
from np2hw.design import composite

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="iverilog not on PATH")

W, H, BITS, DEPTH, SLOTS, FRAMES = 16, 8, 12, 4, 8, 40

TB = """`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, mem_rst = 1;
    always #5 clk = ~clk;
    reg in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg [11:0] in_data = 0;
    reg [15:0] in_ctx = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last, cold, overrun;
    wire [@DEPTH@*12-1:0] out_data;
    wire [15:0] out_ctx, out_source;
    reg out_ready = 0;
@MEMORY_WIRES@
    np2hw_frame_store_axi #(.BITS(12), .FRAMES(@DEPTH@), .SLOTS(@SLOTS@),
                            .CTX_BITS(16), .PIXELS(@PIXELS@)) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .in_ctx(in_ctx),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ctx(out_ctx), .out_source(out_source), .cold(cold),
        .overrun(overrun)@MEMORY_PORTS@);
    integer f, k, i, cyc = 0, ended = 0, seed = @SEED@;
    always @(posedge clk) cyc <= cyc + 1;
    always @(posedge clk) if (in_valid && in_ready && in_last) ended <= ended + 1;
    // every frame begun, and every beat taken, outside the reset
    always @(posedge clk) if (!rst) begin
        if (in_valid && in_ready && in_sof) $fdisplay(f, "S %0d %0d", in_ctx, cyc);
        if (out_valid && out_ready) begin
            if (out_sof) $fdisplay(f, "O %0d %0d %0d", cyc, out_source, out_ctx);
            $fdisplay(f, "B %0h", out_data);
            if (out_last) $fdisplay(f, "E");
        end
    end
    always @(negedge clk) out_ready = ($random(seed) & 3) != 0;
    // the stream's reset, at a moment drawn for it; the memory's, never again
    initial begin
        f = $fopen("out.txt", "w");
        repeat (3) @(negedge clk); mem_rst = 0; rst = 0;
@TRIGGER@
        rst = 1; $fdisplay(f, "R %0d", cyc);
        repeat (@LEN@) @(negedge clk);
        rst = 0; $fdisplay(f, "U %0d", cyc);
    end
    // the source: frames back to back, a short gap between, never waiting
    // for the store but as its ready says
    initial begin
        repeat (6) @(negedge clk);
        for (k = 0; k < @FRAMES@; k = k + 1) begin
            for (i = 0; i < @W@ * @H@; i = i + 1) begin
                in_valid = 1; in_sof = i == 0; in_ctx = k;
                in_eol = i % @W@ == @W@ - 1; in_last = i == @W@ * @H@ - 1;
                in_data = (k * 37 + i * 5) & 12'hfff;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            in_valid = 0;
            repeat (20 + ($random(seed) & 31)) @(negedge clk);
        end
        repeat (4000) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
"""


def pixel(k, i):
    return (k * 37 + i * 5) & 0xFFF


def _run(tmp_path, seed, trigger, late=0, length=None):
    rng = np.random.default_rng(seed)
    length = int(rng.integers(1, 6)) if length is None else length
    tb = (TB.replace("@MEMORY_WIRES@", MEMORY_WIRES
                     .replace("#(.WORDS(", f"#(.BLATE({late}), .WORDS(")
                     .replace("@WORDS@", "8192").replace("@SEED@", str(seed + 1)))
          .replace("@TRIGGER@", trigger)
          .replace("@MEMORY_PORTS@",
                   MEMORY_PORTS.replace(".mem_rst(rst)", ".mem_rst(mem_rst)"))
          .replace("@DEPTH@", str(DEPTH)).replace("@SLOTS@", str(SLOTS))
          .replace("@PIXELS@", str(W * H)).replace("@SEED@", str(seed + 7))
          .replace("@LEN@", str(length))
          .replace("@FRAMES@", str(FRAMES)).replace("@W@", str(W))
          .replace("@H@", str(H)))
    (tmp_path / "tb.v").write_text(tb + MEMORY)
    sources = [str(composite.source_file(FOUND.folder, f))
               for f in IMPLEMENTATIONS["axi"]["sources"]]
    subprocess.run(["iverilog", "-g2005", "-o", str(tmp_path / "sim"),
                    str(tmp_path / "tb.v"), *sources],
                   check=True, capture_output=True, text=True)
    run = subprocess.run(["vvp", "-n", str(tmp_path / "sim")], cwd=tmp_path,
                         check=True, capture_output=True, text=True,
                         timeout=300)
    broken = [line for line in run.stdout.splitlines()
              if line.startswith("AXI:")]
    assert not broken, broken[:5]
    return (tmp_path / "out.txt").read_text().splitlines()


def check(lines):
    """The frames out after the reset: the history begun after it, each
    the newest four frames side by side, with the newest's context."""
    starts, outs, reset, released = [], [], None, None
    for line in lines:
        kind, *rest = line.split()
        if kind == "S":
            starts.append((int(rest[0]), int(rest[1])))
        elif kind == "R":
            reset = int(rest[0])
        elif kind == "U":
            released = int(rest[0])
        elif kind == "O":
            outs.append({"at": int(rest[0]), "source": int(rest[1]),
                         "ctx": int(rest[2]), "beats": []})
        elif kind == "B":
            outs[-1]["beats"].append(int(rest[0], 16))
        elif kind == "E":
            outs[-1]["whole"] = True
    assert reset is not None and released is not None
    # the history after the reset: the frames begun once it was released,
    # counted from the first
    after = [k for k, cycle in starts if cycle > released]
    first = after[0]
    late = [o for o in outs if o["at"] > released and o.get("whole")]
    assert len(late) >= 8, f"{len(late)} frames out after the reset"
    mask = (1 << BITS) - 1
    for o in late:
        newest = first + o["source"]
        assert o["source"] >= DEPTH - 1, "a frame out before the history"
        assert o["ctx"] == newest, f"frame out at {o['at']}: context"
        assert len(o["beats"]) == W * H
        for b, word in enumerate(o["beats"]):
            got = [(word >> (j * BITS)) & mask for j in range(DEPTH)]
            want = [pixel(newest - j, b) for j in range(DEPTH)]
            assert got == want, f"frame out at {o['at']}, beat {b}"


@pytest.mark.parametrize("seed", range(6))
def test_a_stream_starting_over_leaves_the_memory_whole(tmp_path, seed):
    """The reset at a moment drawn at random, the frames streaming and
    the consumer reading: bursts in flight, an address offered."""
    frame = W * H + 40                     # a frame and a gap, in cycles
    at = int(np.random.default_rng(seed).integers(6 * frame, 12 * frame))
    check(_run(tmp_path, seed, f"        while (cyc < {at}) @(negedge clk);"))


@pytest.mark.parametrize("wait", [0, 2, 9])
def test_a_frame_answered_after_the_reset_counts_in_no_history(tmp_path,
                                                               wait):
    """A frame's writes all issued as the reset comes, and answered long
    after the next history has begun: it was written to the ring's first
    slot, so a store that began the ring again there would count it as
    the new history's first frame whole."""
    # frame SLOTS is the ninth, in the first slot again
    trigger = (f"        while (ended < {SLOTS + 1}) @(negedge clk);\n"
               f"        repeat ({wait}) @(negedge clk);")
    check(_run(tmp_path, wait, trigger, late=200, length=2))
