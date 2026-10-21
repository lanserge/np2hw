# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw_frame_store's Verilog does what its NumPy model says.

One random stimulus drives both: frames of random samples, sent in bursts
with random gaps, a change of size somewhere in them, and a consumer that
waits at random between output frames and stalls at random inside them.
The model computes, from that stimulus alone, every frame the store must
put out -- which input frame it is computed at, the frames side by side,
the context -- and the Verilog must put out exactly those, frame for frame.
A draw that puts two events the store must tell apart closer together than
its latency is drawn again: the model and the hardware would be entitled to
break that tie differently.

Each implementation is held to the one model: block RAM, and external
memory, where the testbench brings the memory -- an AXI slave of its own,
its readies at random and its answers late at random, as a real
interconnect's are.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw.design import composite
from np2hw.hdl.models import Stimulus, frame_store

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="iverilog not on PATH")

# The store as its library describes it: the parameters its Verilog and its
# model are both given, the implementation that builds it, and the latency
# that implementation states.
FOUND = composite.find("np2hw.hdl.frames:frame_store")
IMPLEMENTATIONS = FOUND.obj["implementations"]
PARAMS = {"BITS": 12, "FRAMES": 4, "SLOTS": 8, "CTX_BITS": 16, "PIXELS": 512}
BITS, DEPTH, SLOTS = PARAMS["BITS"], PARAMS["FRAMES"], PARAMS["SLOTS"]


def margin(implementation):
    """The tie margin: the latency the implementation states, and a few
    cycles for the testbench's own way of counting them."""
    return IMPLEMENTATIONS[implementation]["facts"]["latency"] + 4


# A memory for the external binding: 64-bit words, every ready at random,
# each write answered and each read begun some cycles late, at random -- and
# a write lands only when it is answered, which is all AXI promises: a
# binding that calls a frame whole before its writes are answered reads
# what was there before.
MEMORY = """
module tb_axi_mem #(parameter WORDS = 4096, parameter SEED = 1) (
    input  wire clk,
    input  wire [31:0] awaddr, input wire [7:0] awlen, input wire awvalid,
    output reg  awready = 0,
    input  wire [63:0] wdata, input wire wlast, input wire wvalid,
    output reg  wready = 0,
    output reg  bvalid = 0, input wire bready, output wire [1:0] bresp,
    input  wire [31:0] araddr, input wire [7:0] arlen, input wire arvalid,
    output reg  arready = 0,
    output reg  [63:0] rdata = 0, output reg rlast = 0, output reg rvalid = 0,
    input  wire rready, output wire [1:0] rresp);
    assign bresp = 2'b00; assign rresp = 2'b00;
    reg [63:0] mem [0:WORDS-1];
    integer seed = SEED, now = 0;
    reg [31:0] wa [0:15]; integer aw_w = 0, aw_r = 0, wbeat = 0;
    integer bdue [0:15]; integer b_w = 0, b_r = 0, k;
    reg [31:0] pend_a [0:15]; integer pend_n [0:15];
    reg [63:0] pend [0:255];
    reg [31:0] ra [0:15]; reg [7:0] rl [0:15]; integer rdue [0:15];
    integer ar_w = 0, ar_r = 0, rbeat = 0;
    always @(posedge clk) begin
        now = now + 1;
        if (awvalid && awready) begin wa[aw_w % 16] = awaddr; aw_w = aw_w + 1; end
        if (wvalid && wready) begin
            pend[(b_w % 16) * 16 + wbeat] = wdata;
            wbeat = wbeat + 1;
            if (wlast) begin
                pend_a[b_w % 16] = wa[aw_r % 16]; pend_n[b_w % 16] = wbeat;
                wbeat = 0; aw_r = aw_r + 1;
                bdue[b_w % 16] = now + 3 + ($random(seed) & 31); b_w = b_w + 1;
            end
        end
        if (bvalid && bready) begin
            for (k = 0; k < pend_n[b_r % 16]; k = k + 1)
                mem[(pend_a[b_r % 16] >> 3) + k] = pend[(b_r % 16) * 16 + k];
            b_r = b_r + 1;
        end
        if (arvalid && arready) begin
            ra[ar_w % 16] = araddr; rl[ar_w % 16] = arlen;
            rdue[ar_w % 16] = now + 10 + ($random(seed) & 63); ar_w = ar_w + 1;
        end
        if (rvalid && rready) begin
            if (rlast) begin rbeat = 0; ar_r = ar_r + 1; end
            else rbeat = rbeat + 1;
        end
    end
    always @(negedge clk) begin
        awready = ($random(seed) & 3) != 0;
        wready = aw_w != aw_r && ($random(seed) & 3) != 0;
        arready = ($random(seed) & 3) != 0;
        bvalid = b_w != b_r && now >= bdue[b_r % 16];
        rvalid = ar_w != ar_r && now >= rdue[ar_r % 16]
                 && ($random(seed) & 3) != 0;
        rdata = mem[(ra[ar_r % 16] >> 3) + rbeat];
        rlast = rbeat == rl[ar_r % 16];
    end
endmodule
"""

MEMORY_PORTS = """,
        .mem_awaddr(mem_awaddr), .mem_awlen(mem_awlen),
        .mem_awsize(), .mem_awburst(), .mem_awcache(), .mem_awprot(),
        .mem_awvalid(mem_awvalid), .mem_awready(mem_awready),
        .mem_wdata(mem_wdata), .mem_wstrb(), .mem_wlast(mem_wlast),
        .mem_wvalid(mem_wvalid), .mem_wready(mem_wready),
        .mem_bresp(mem_bresp), .mem_bvalid(mem_bvalid),
        .mem_bready(mem_bready),
        .mem_araddr(mem_araddr), .mem_arlen(mem_arlen),
        .mem_arsize(), .mem_arburst(), .mem_arcache(), .mem_arprot(),
        .mem_arvalid(mem_arvalid), .mem_arready(mem_arready),
        .mem_rdata(mem_rdata), .mem_rresp(mem_rresp),
        .mem_rlast(mem_rlast), .mem_rvalid(mem_rvalid),
        .mem_rready(mem_rready)"""

MEMORY_WIRES = """
    wire [31:0] mem_awaddr, mem_araddr;
    wire [7:0] mem_awlen, mem_arlen;
    wire [63:0] mem_wdata, mem_rdata;
    wire [1:0] mem_bresp, mem_rresp;
    wire mem_awvalid, mem_awready, mem_wlast, mem_wvalid, mem_wready,
         mem_bvalid, mem_bready, mem_arvalid, mem_arready, mem_rlast,
         mem_rvalid, mem_rready;
    tb_axi_mem #(.WORDS(@WORDS@), .SEED(@SEED@)) memory (
        .clk(clk), .awaddr(mem_awaddr), .awlen(mem_awlen),
        .awvalid(mem_awvalid), .awready(mem_awready), .wdata(mem_wdata),
        .wlast(mem_wlast), .wvalid(mem_wvalid), .wready(mem_wready),
        .bvalid(mem_bvalid), .bready(mem_bready), .bresp(mem_bresp),
        .araddr(mem_araddr), .arlen(mem_arlen), .arvalid(mem_arvalid),
        .arready(mem_arready), .rdata(mem_rdata), .rlast(mem_rlast),
        .rvalid(mem_rvalid), .rready(mem_rready), .rresp(mem_rresp));
"""

TB = """`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;
    reg in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg [11:0] in_data = 0;
    reg [15:0] in_fw = 0, in_fh = 0, in_ctx = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last, cold, overrun;
    wire [@DEPTH@*12-1:0] out_data;
    wire [15:0] out_fw, out_fh, out_ctx, out_source;
    reg out_ready = 0;
@MEMORY_WIRES@
    @MODULE@ #(@PARAMS@) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .in_frame_width(in_fw), .in_frame_height(in_fh), .in_ctx(in_ctx),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_frame_width(out_fw), .out_frame_height(out_fh),
        .out_ctx(out_ctx), .out_source(out_source), .cold(cold),
        .overrun(overrun)@MEMORY_PORTS@);

    // the stimulus, as the model was given it
    reg [11:0] pix [0:@PIXELS@-1];
    reg [15:0] fw [0:@NF@-1];
    reg [15:0] fh [0:@NF@-1];
    integer gap [0:@NF@-1];
    integer waits [0:@NO@];
    integer st_beat [0:@NS@];
    integer st_cyc [0:@NS@];
    integer st_from [0:@NO@];
    integer st_count [0:@NO@];
    // the source counts with k, i, n, at; the consumer with j, s, taken
    integer f, k, i, n, at, j, s, taken, out_done = 0;
    always @(posedge clk) if (!rst && out_valid && out_ready) begin
        if (out_sof)
            $fdisplay(f, "O %0d %0d %0d %0d", out_source, out_ctx, out_fw,
                      out_fh);
        $fdisplay(f, "B %0h", out_data);
        if (out_last) $fdisplay(f, "E");
    end
    // the source: each frame `gap` cycles after the last, a beat a cycle
    initial begin
        $readmemh("pix.hex", pix);
@TABLES@
        repeat (2) @(negedge clk); rst = 0;
        at = 0;
        for (k = 0; k < @NF@; k = k + 1) begin
            repeat (gap[k]) @(negedge clk);
            n = fw[k] * fh[k];
            for (i = 0; i < n; i = i + 1) begin
                in_valid = 1; in_data = pix[at + i];
                in_sof = (i == 0); in_last = (i == n - 1);
                in_eol = (i % fw[k] == fw[k] - 1);
                in_fw = fw[k]; in_fh = fh[k]; in_ctx = k * 7 + 3;
                @(negedge clk);
            end
            in_valid = 0;
            at = at + n;
        end
    end
"""

# the consumer: from cycle `first`, each frame `waits` after the last, holding
# off `st_cyc` cycles before beat `st_beat`
CONSUMER = """    // a run that has not finished in twice the cycles its stimulus takes
    // never will: what came out is what the test sees
    initial begin
        #(@BOUND@ * 10);
        $fdisplay(f, "T");
        $fclose(f); $finish;
    end
    initial begin
    f = $fopen("out.txt", "w");
    repeat (2) @(negedge clk);
    repeat (@FIRST@) @(negedge clk);
    for (j = 0; j < @NO@; j = j + 1) begin
        taken = 0; s = st_from[j]; out_done = 0;
        while (!out_done) begin
            if (s < st_from[j] + st_count[j] && st_beat[s] == taken) begin
                out_ready = 0;
                repeat (st_cyc[s]) @(negedge clk);
                s = s + 1;
            end else begin
                out_ready = 1;
                @(posedge clk);
                if (out_valid) begin
                    taken = taken + 1;
                    if (out_last) out_done = 1;
                end
                @(negedge clk);
            end
        end
        out_ready = 0;
        if (j < @NO@ - 1) repeat (waits[j] - 1) @(negedge clk);
    end
    repeat (20) @(negedge clk);
    $fdisplay(f, "R %0d", overrun);
    $fclose(f); $finish;
end
endmodule
"""


def make_testbench(frames, stim, outputs, implementation="bram", seed=1):
    nf, no = len(frames), outputs
    stalls = [stim.stall(i) for i in range(no)]
    flat = [s for frame in stalls for s in frame]
    tables = []
    for k, fr in enumerate(frames):
        h, w = fr.shape
        tables.append(f"        fw[{k}] = {w}; fh[{k}] = {h}; "
                      f"gap[{k}] = {stim.gaps[k]};")
    for i in range(no - 1):
        tables.append(f"        waits[{i}] = {stim.waits[i]};")
    at = 0
    for i, frame in enumerate(stalls):
        tables.append(f"        st_from[{i}] = {at}; st_count[{i}] = "
                      f"{len(frame)};")
        at += len(frame)
    for j, (beat, cyc) in enumerate(flat):
        tables.append(f"        st_beat[{j}] = {beat}; st_cyc[{j}] = {cyc};")
    text = TB + CONSUMER
    params = ", ".join(f".{k}({v})" for k, v in PARAMS.items())
    external = implementation == "axi"
    slot_words = -(-PARAMS["PIXELS"] * 2 // 4096) * 4096 // 8
    text = (text.replace("@MEMORY_PORTS@", MEMORY_PORTS if external else "")
            .replace("@MEMORY_WIRES@", MEMORY_WIRES if external else "")
            .replace("@WORDS@", str(SLOTS * slot_words))
            .replace("@SEED@", str(seed)))
    if external:
        text += MEMORY
    return (text.replace("@MODULE@",
                         IMPLEMENTATIONS[implementation]["vivado"]["module"])
            .replace("@PARAMS@", params)
            .replace("@DEPTH@", str(DEPTH)).replace("@SLOTS@", str(SLOTS))
            .replace("@NF@", str(nf)).replace("@NO@", str(no))
            .replace("@NS@", str(max(1, len(flat))))
            .replace("@PIXELS@", str(sum(fr.size for fr in frames)))
            .replace("@FIRST@", str(stim.first))
            .replace("@BOUND@", str(2 * (sum(stim.gaps) + sum(
                fr.size for fr in frames) + stim.first + sum(stim.waits)
                + sum(c for i in range(no) for _, c in stim.stall(i))
                + no * 64) + 10000))
            .replace("@TABLES@", "\n".join(tables)))


def draw(rng, outputs=14, overrun=False, implementation="bram",
         sizes=((4, 6), (3, 5)), stall=20):
    """A random stimulus: bursts of frames, a change of size, and a consumer
    whose every wait is drawn again until the choice it leads to is clear of
    every event the store must tell apart from it."""
    count = 28
    change = int(rng.integers(10, 20))
    sizes = [sizes[0] if k < change else sizes[1] for k in range(count)]
    frames = [rng.integers(0, 1 << BITS, size) for size in sizes]
    contexts = [k * 7 + 3 for k in range(count)]
    # frames in a burst as close as the store can tell apart, the bursts
    # further apart again
    m = margin(implementation)
    near = 0 if implementation == "bram" else 3 * m
    gaps = tuple(int(rng.integers(near, near + 12)) if k % 4 else
                 int(rng.integers(near + 60, near + 400))
                 for k in range(count))
    beats = min(h * w for h, w in sizes)
    stalls = tuple(tuple(sorted({(int(rng.integers(0, beats)),
                                  int(rng.integers(1, stall)))
                                 for _ in range(int(rng.integers(0, 3)))}))
                   for _ in range(outputs))

    def model(first, waits, n):
        stim = Stimulus(gaps, first, tuple(waits), stalls)
        run = frame_store(frames, [{"in_ctx": c} for c in contexts], stim,
                          PARAMS, outputs=n)
        return stim, (run, run.timeline)

    def clear(first, waits):
        """Whether the choice the last-drawn wait leads to -- the store
        chooses the next frame the cycle after the consumer took the last
        -- is clear of every event it must be told apart from."""
        n = len(waits) + 2
        _, (_, t) = model(first, waits, n)
        return not t.close(m, upto=n) and (overrun or not t.overrun)

    _, (_, t) = model(0, [], 1)
    for _ in range(200):
        first = t.chosen[0] + m + 1 + int(rng.integers(0, 16 * m + 1500))
        if clear(first, []):
            break
    else:
        raise AssertionError("no clear first frame out")
    waits = []
    for i in range(1, outputs):
        for _ in range(1000):
            # long enough, some of them, to outlast a history warming up
            wait = int(rng.integers(m + 2, 16 * m + 1500))
            if clear(first, waits + [wait]):
                waits.append(wait)
                break
        else:
            raise AssertionError(f"no clear wait for frame out {i}")
    stim, (run, t) = model(first, waits, outputs)
    assert len(run.frames) == outputs and not t.close(m)
    assert overrun or not t.overrun
    return frames, stim, run, t.overrun


def run_rtl(tmp_path, frames, stim, outputs, implementation="bram", seed=1):
    (tmp_path / "tb.v").write_text(
        make_testbench(frames, stim, outputs, implementation, seed))
    (tmp_path / "pix.hex").write_text(
        "\n".join(f"{int(v):x}" for fr in frames for v in fr.ravel()) + "\n")
    sources = [str(composite.source_file(FOUND.folder, f))
               for f in IMPLEMENTATIONS[implementation]["sources"]]
    subprocess.run(["iverilog", "-g2005", "-o", str(tmp_path / "sim"),
                    str(tmp_path / "tb.v"), *sources],
                   check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "-n", str(tmp_path / "sim")], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    got, overrun = [], None
    for line in (tmp_path / "out.txt").read_text().splitlines():
        kind, *rest = line.split()
        if kind == "O":
            got.append({"head": tuple(map(int, rest)), "beats": []})
        elif kind == "B":
            got[-1]["beats"].append(int(rest[0], 16))
        elif kind == "E":
            got[-1]["whole"] = True
        elif kind == "R":
            overrun = int(rest[0])
    return [g for g in got if g.get("whole")], overrun


CASES = [("bram", s) for s in range(8)] + [("axi", s) for s in range(4)]


@pytest.mark.parametrize("implementation, seed", CASES)
def test_the_verilog_puts_out_what_the_model_says(tmp_path, implementation,
                                                  seed):
    rng = np.random.default_rng(seed)
    frames, stim, want, _ = draw(rng, implementation=implementation)
    got, said = run_rtl(tmp_path, frames, stim, len(want.frames),
                        implementation, seed + 1)
    assert said == 0, "overrun where the model says none"
    assert len(got) == len(want.frames)
    mask = (1 << BITS) - 1
    for i, (g, data, told) in enumerate(zip(got, want.frames, want.ports)):
        head = (told["out_source"], told["out_ctx"],
                told["out_frame_width"], told["out_frame_height"])
        assert g["head"] == head, \
            f"frame out {i}: {g['head']} where the model says {head}"
        flat = data.reshape(-1, DEPTH)
        for b, word in enumerate(g["beats"]):
            assert [(word >> (k * BITS)) & mask for k in range(DEPTH)] == \
                [int(v) for v in flat[b]], f"frame out {i}, beat {b}"


@pytest.mark.parametrize("implementation, seed", CASES)
def test_the_verilog_says_overrun_where_the_model_does(tmp_path,
                                                       implementation, seed):
    """Stimuli drawn without avoiding overruns -- consumers that wait long
    enough for a burst to reach the frames they read -- and the Verilog's
    sticky `overrun` as the model predicts it, either way."""
    rng = np.random.default_rng(100 + seed)
    frames, stim, want, predicted = draw(rng, overrun=True,
                                         implementation=implementation)
    _, said = run_rtl(tmp_path, frames, stim, len(want.frames),
                      implementation, seed + 1)
    assert said == int(predicted)


@pytest.mark.parametrize("seed", range(3))
def test_a_frame_longer_than_a_read_queue_is_fetched_as_it_is_taken(
        tmp_path, seed):
    """Frames of 128 words a frame read, twice what each read queue holds,
    and a consumer that stops for hundreds of cycles inside a frame: the
    fetch runs ahead only as far as its queues have room."""
    rng = np.random.default_rng(200 + seed)
    frames, stim, want, _ = draw(rng, outputs=6, implementation="axi",
                                 sizes=((16, 32), (12, 32)), stall=400)
    got, said = run_rtl(tmp_path, frames, stim, len(want.frames), "axi",
                        seed + 1)
    assert said == 0 and len(got) == len(want.frames)
    mask = (1 << BITS) - 1
    for i, (g, data) in enumerate(zip(got, want.frames)):
        flat = data.reshape(-1, DEPTH)
        for b, word in enumerate(g["beats"]):
            assert [(word >> (k * BITS)) & mask for k in range(DEPTH)] == \
                [int(v) for v in flat[b]], f"frame out {i}, beat {b}"
