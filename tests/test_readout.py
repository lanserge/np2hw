# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The readout leaf: a stream's frames, read by software over AXI4-Lite on
a clock of its own.

Frames stream in on one clock while a bus master on an unrelated clock
reads the count and every entry of the last frame published -- between
frames and while the next is being written -- and each read is the frame
it says it is: every value in its slot, sign-extended where signed, the
words past it zero. A write is answered SLVERR.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import hdl
from np2hw.readout import layout, readout

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")


def _sim(tmp_path, shape, frames, overlap: bool):
    """`frames` -- (n, entries, channels), each value in `bits` -- through
    the leaf; after each is published, every entry's words read. With
    `overlap` the next frame streams while the reads are made. Returns
    [(count before, count after, {word: value})] a read pass."""
    rows, cols, channels, bits, signed = shape
    lay = layout(rows, cols, channels, bits, signed)
    leaf = readout(rows, cols, channels, bits, signed)
    mask = (1 << bits) - 1
    words = [sum((int(v) & mask) << (bits * c) for c, v in enumerate(e))
             for f in frames for e in f]
    (tmp_path / "in.hex").write_text("".join(f"{w:x}\n" for w in words))
    reads = [0] + [(e + 1) * lay["slot_words"] + k
                   for e in range(lay["entries"])
                   for k in range(lay["slot_words"])]
    (tmp_path / "addr.hex").write_text("".join(f"{4 * a:x}\n" for a in reads))
    aw, db, n, per = lay["addr_bits"], channels * bits, len(frames), \
        lay["entries"]
    (tmp_path / "dut.v").write_text(hdl.text("np2hw_readout") +
                                    leaf["verilog"])
    pass_ = f"""
                first = 0;
                axi_read(0, first);
                for (a = 0; a < {len(reads)}; a = a + 1) begin
                    axi_read(at[a], data);
                    $fdisplay(f, "w %0d %0d", at[a] >> 2, data);
                end
                axi_read(0, again);
                $fdisplay(f, "p %0d %0d", first, again);"""
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, aclk = 0, aresetn = 0;
    reg in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg [{db - 1}:0] in_data = 0;
    wire in_ready;
    reg [{aw - 1}:0] s_axi_araddr = 0, s_axi_awaddr = 0;
    reg s_axi_arvalid = 0, s_axi_rready = 0, s_axi_awvalid = 0,
        s_axi_wvalid = 0, s_axi_bready = 0;
    reg [31:0] s_axi_wdata = 0;
    reg [3:0] s_axi_wstrb = 4'hf;
    wire s_axi_arready, s_axi_rvalid, s_axi_awready, s_axi_wready,
         s_axi_bvalid;
    wire [31:0] s_axi_rdata;
    wire [1:0] s_axi_rresp, s_axi_bresp;
    reg [{db - 1}:0] img [0:{len(words) - 1}];
    reg [{aw - 1}:0] at [0:{len(reads) - 1}];
    integer f, i, k, a, seed = 5;
    reg [31:0] data, first, again;
    always #5 clk = ~clk;
    always #3.7 aclk = ~aclk;            // the bus's clock, unrelated
    {leaf["module"]} dut (.clk(clk), .rst(rst), .in_valid(in_valid),
        .in_ready(in_ready), .in_data(in_data), .in_sof(in_sof),
        .in_eol(in_eol), .in_last(in_last), .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready), .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready), .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready));
    task axi_read(input [{aw - 1}:0] addr, output [31:0] value);
        begin
            @(negedge aclk); s_axi_araddr = addr; s_axi_arvalid = 1;
            @(posedge aclk); while (!s_axi_arready) @(posedge aclk);
            @(negedge aclk); s_axi_arvalid = 0; s_axi_rready = 1;
            @(posedge aclk); while (!s_axi_rvalid) @(posedge aclk);
            value = s_axi_rdata;
            @(negedge aclk); s_axi_rready = 0;
        end
    endtask
    task send(input integer fr);
      begin
        @(negedge clk);                 // on the stream's own clock
        for (i = 0; i < {per}; i = i + 1) begin
            // beside reads, a stream slower than they are: each read pass
            // of the last frame done within the next
            if ({int(overlap)}) begin
                in_valid = 0; repeat (60) @(negedge clk);
            end
            while (($random(seed) & 3) == 0) begin
                in_valid = 0; @(negedge clk);
            end
            in_data = img[fr * {per} + i]; in_valid = 1;
            in_sof = (i == 0); in_eol = (i % {cols} == {cols - 1});
            in_last = (i == {per - 1});
            @(negedge clk);
        end
        in_valid = 0;
      end
    endtask
    initial begin
        $readmemh("in.hex", img);
        $readmemh("addr.hex", at);
        f = $fopen("out.txt", "w");
        repeat (4) @(negedge clk); rst = 0; aresetn = 1;
        // a write: nothing here is software's to set
        @(negedge aclk); s_axi_awvalid = 1; s_axi_wvalid = 1; s_axi_bready = 1;
        @(posedge aclk); while (!s_axi_bvalid) @(posedge aclk);
        $fdisplay(f, "b %0d", s_axi_bresp);
        @(negedge aclk); s_axi_awvalid = 0; s_axi_wvalid = 0; s_axi_bready = 0;
        for (k = 0; k < {n}; k = k + 1) begin
            fork
                send(k);
                if ({int(overlap)} && k > 0) begin
{pass_}
                end
            join
            repeat (20) @(negedge aclk);
            if (!{int(overlap)} || k == {n - 1}) begin
{pass_}
            end
        end
        $fclose(f); $finish;
    end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "dut.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=300)
    passes, current, bresp = [], {}, None
    for line in (tmp_path / "out.txt").read_text().splitlines():
        kind, *vals = line.split()
        if kind == "b":
            bresp = int(vals[0])
        elif kind == "w":
            current[int(vals[0])] = int(vals[1])
        else:
            passes.append((int(vals[0]), int(vals[1]), current))
            current = {}
    return lay, passes, bresp


def _entries(lay, words):
    """The values a read pass holds, entry by entry, as the layout says to
    read them -- and the words past them, which must be zero."""
    vw, slot = lay["value_words"], lay["slot_words"]
    out, spare = [], []
    for e in range(lay["entries"]):
        base = (e + 1) * slot
        row = []
        for c in range(slot // vw):
            v = sum(words[base + c * vw + k] << (32 * k) for k in range(vw))
            if c >= lay["channels"]:
                spare.append(v)
                continue
            if lay["signed"] and v >> (32 * vw - 1):
                v -= 1 << (32 * vw)
            row.append(v)
        out.append(row)
    return out, spare


SHAPES = [(2, 4, 1, 6, False), (3, 5, 3, 10, False), (1, 4, 1, 15, True),
          (2, 2, 2, 33, False), (1, 1, 1, 32, True)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("overlap", [False, True])
def test_every_frame_is_read_as_it_was_written(tmp_path, shape, overlap):
    rows, cols, channels, bits, signed = shape
    lo, hi = ((-(1 << (bits - 1)), 1 << (bits - 1)) if signed
              else (0, 1 << bits))
    frames = np.random.default_rng(bits).integers(
        lo, hi, (4, rows * cols, channels), dtype=np.int64)
    lay, passes, bresp = _sim(tmp_path, shape, frames, overlap)
    assert bresp == 2                                   # SLVERR
    assert passes, "nothing was read"
    whole = 0
    for before, after, words in passes:
        if before != after:
            # a frame was published across the read: what it read may be
            # either frame's, and the changed count says to read again
            assert overlap, "a frame was published across a read made "  \
                            "between frames"
            continue
        whole += 1
        assert words[0] == before
        values, spare = _entries(lay, words)
        assert values == frames[before - 1].tolist(), before
        assert not any(spare)
    # every frame, whether its reads wait for the next or run beside it
    assert [p[0] for p in passes if p[0] == p[1]] == [1, 2, 3, 4]
