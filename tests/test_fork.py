# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A stream into two consumers: np2hw_fork, and a design that forks.

Each consumer takes every beat, in order, once; the source moves on when
all have. A design whose stream feeds a datapath and a fold beside it is
simulated whole -- each output stalling on its own -- against its models.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import hdl
from np2hw.blocks import StreamPort, block
from np2hw.blocks.params import Param
from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from np2hw.pipeline import run as runner
from toylib.blocks import VOCABULARY, gain

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")


def _run(tmp_path, verilog: str, tb: str) -> str:
    (tmp_path / "dut.v").write_text(verilog)
    (tmp_path / "tb.v").write_text(tb)
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "dut.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=120)
    return (tmp_path / "out.txt").read_text()


@pytest.mark.parametrize("n", [2, 3])
def test_every_branch_takes_every_beat_once_in_order(tmp_path, n):
    beats = 300
    out = _run(tmp_path, hdl.text("np2hw_fork"), f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, s_valid = 0;
    reg [15:0] s_data = 0;
    reg [{n - 1}:0] m_ready = 0;
    wire s_ready;
    wire [{n - 1}:0] m_valid;
    integer f, k, seed = 7;
    always #5 clk = ~clk;
    np2hw_fork #(.N({n})) dut (.clk(clk), .rst(rst), .s_valid(s_valid),
        .s_ready(s_ready), .m_valid(m_valid), .m_ready(m_ready));
    genvar b;
    generate for (b = 0; b < {n}; b = b + 1) begin : log
        always @(posedge clk) if (m_valid[b] && m_ready[b])
            $fdisplay(f, "%0d %0d", b, s_data);
    end endgenerate
    always @(negedge clk) m_ready <= $random(seed);
    initial begin
        f = $fopen("out.txt", "w");
        repeat (3) @(negedge clk); rst = 0;
        for (k = 0; k < {beats}; k = k + 1) begin
            while (($random(seed) & 3) == 0) begin
                s_valid = 0; @(negedge clk);
            end
            s_data = k; s_valid = 1;
            @(posedge clk); while (!s_ready) @(posedge clk);
            @(negedge clk);
        end
        s_valid = 0;
        repeat (10) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
""")
    taken = {b: [] for b in range(n)}
    for line in out.splitlines():
        b, v = map(int, line.split())
        taken[b].append(v)
    assert all(taken[b] == list(range(beats)) for b in range(n)), \
        {b: len(v) for b, v in taken.items()}


@block(version=(1, 0), description="bright and dark pixels counted",
       inputs=(StreamPort("in"),),
       outputs=(StreamPort("out", "pixels below the level, at or above"),),
       params=[Param("level", bits=8, default=128,
                     description="what counts as bright")])
def counts(pixel, p, bit_depth):
    acc = np.zeros(2, np.int64, like=pixel)
    np.add.at(acc, (pixel >= p.level).astype(np.int64), 1)
    return acc


W, H = 16, 8


def _design():
    """A gain whose stream goes on to the output and into a fold beside
    it: one stream, two consumers."""
    return assemble(
        "t", {"g": gain, "m": counts},
        [("in", "g.in"), ("g.out", "out"), ("g.out", "m.in"),
         ("m.out", "stats")],
        inputs=("in",), outputs=("out", "stats"), vocabulary=VOCABULARY,
        stream={"bits": 8, "max_width": W, "max_height": H, "context": []})


def test_a_design_forks_its_stream_into_a_datapath_and_a_fold(tmp_path):
    """Frames in back to back, each output stalling at random on its own:
    every frame out, and every frame's counts, are its models'."""
    d = _design()
    gen = compose.generate(d, control=False)
    assert [m for m, _ in gen.modules][-1] == "np2hw_fork"
    values = {"g": {"gain": 3}, "m": {"level": 100}}
    frames = np.random.default_rng(2).integers(0, 90, (4, H, W))
    want = {}
    for name, chain in runner.legs(d):
        port = "stats" if chain[-1].path == "m" else "out"
        want[port] = [np.asarray(runner.run_model(chain, f, values, {}, 8))
                      .ravel().tolist() for f in frames]
    flat = [int(v) for v in frames.ravel()]
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in flat))
    per = W * H
    out = _run(tmp_path, "\n".join(t for _, t in gen.modules), f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0, in_eol = 0, in_last = 0;
    reg out_ready = 0, stats_ready = 0;
    reg [7:0] in_data = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire stats_valid, stats_sof, stats_eol, stats_last;
    wire [7:0] out_data, stats_data;
    reg [7:0] img [0:{len(flat) - 1}];
    integer f, i, k, seed = 3;
    always #5 clk = ~clk;
    t dut (.clk(clk), .rst(rst),
        .param_ctxblock_width(16'd{W}), .param_ctxblock_height(16'd{H}),
        .cfg_arm_req(1'b0), .cfg_arm_ack(),
        .param_g_gain(4'd3), .param_m_level(8'd100),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last), .out_ctx(),
        .stats_valid(stats_valid), .stats_ready(stats_ready),
        .stats_data(stats_data), .stats_sof(stats_sof),
        .stats_eol(stats_eol), .stats_last(stats_last), .stats_ctx());
    always @(posedge clk) if (out_valid && out_ready)
        $fdisplay(f, "out %0d", out_data);
    always @(posedge clk) if (stats_valid && stats_ready)
        $fdisplay(f, "stats %0d", stats_data);
    always @(negedge clk) begin
        out_ready <= ($random(seed) & 3) != 0;
        stats_ready <= ($random(seed) & 1) != 0;
    end
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
        repeat (3) @(negedge clk); rst = 0;
        repeat (40) @(negedge clk);
        for (k = 0; k < {len(frames)}; k = k + 1)
            for (i = 0; i < {per}; i = i + 1) begin
                in_data = img[k * {per} + i]; in_valid = 1;
                in_sof = (i == 0); in_eol = (i % {W} == {W - 1});
                in_last = (i == {per - 1});
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
        in_valid = 0;
        repeat (2000) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
""")
    got = {"out": [], "stats": []}
    for line in out.splitlines():
        port, v = line.split()
        got[port].append(int(v))
    assert got["out"] == [v for f in want["out"] for v in f]
    assert got["stats"] == [v for f in want["stats"] for v in f]
