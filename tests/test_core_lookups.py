# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The new compiler's channel words and table reads.

Every model here is compiled with NP2HW_CORE=strict, so it is np2hw.core
that builds it or the test fails -- the reference cannot quietly take it
over -- and its Verilog is simulated against the model: channels of one
word stacked into another; a register array, and a constant table small
enough for the device to build as logic, read at an index computed from
the pixel; the order the register ports are listed in, which must not
depend on which compiler built the core; and the reads the compiler
refuses, each with its reason.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, generate, to_ir
from np2hw.ir import Param, Rom

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")

W, H = 12, 6


@pytest.fixture(autouse=True)
def strict(monkeypatch):
    monkeypatch.setenv("NP2HW_CORE", "strict")


def _compile(fn, params=(), bits=10, channels=1, name="m"):
    _, line = to_ir(fn, Image2D("img", W, H, bits=bits), *params,
                    channels=channels)
    return generate(line, module_name=name)


def _sim(tmp_path, core, frame, values):
    """One frame through the core, every register set; its output words."""
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    names = [n for n, _ in core["params"]]
    regs = "".join(f"    reg [{b - 1}:0] param_{n};\n" for n, b in core["params"])
    conns = "".join(f"        .param_{n}(param_{n}),\n" for n in names)
    sets = "".join(f"        param_{n} = {int(values[n]) & ((1 << b) - 1)};\n"
                   for n, b in core["params"])
    words = [int(v) for v in frame.ravel()]
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in words))
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0;
    reg [{in_bits - 1}:0] in_data = 0;
{regs}    wire in_ready, out_valid, out_sof, out_eol, out_last;
    reg out_ready = 1;
    wire [{out_bits - 1}:0] out_data;
    reg [{in_bits - 1}:0] img [0:{len(words) - 1}];
    integer f, i;
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_sof(in_sof), .in_data(in_data),
{conns}        .out_valid(out_valid), .out_ready(out_ready), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_data(out_data));
    always @(posedge clk) if (out_valid && out_ready)
        $fdisplay(f, "%0d", out_data);
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
{sets}        repeat (3) @(negedge clk); rst = 0;
        @(negedge clk);
        for (i = 0; i < {len(words)}; i = i + 1) begin
            in_data = img[i]; in_sof = (i == 0); in_valid = 1;
            @(posedge clk); while (!in_ready) @(posedge clk);
            @(negedge clk);
        end
        in_valid = 0;
        repeat (8) @(negedge clk);
        $fclose(f); $finish;
    end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "core.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=120)
    return [int(r) for r in (tmp_path / "out.txt").read_text().split()]


# --------------------------------------------------------------------------- #
# channels in a word, channels out
# --------------------------------------------------------------------------- #

def swap_and_sum(pixel):
    """Two 8-bit channels read as `pixel[..., k]`, three stacked out."""
    a, b = (pixel[..., k].astype(np.int32) for k in range(2))
    return np.stack([b, a, (a + b) >> 1], axis=-1).astype(np.uint16)


def test_channels_read_and_stacked_are_one_word_channel_0_lowest(tmp_path):
    core = _compile(swap_and_sum, bits=16, channels=2, name="swap")
    out = core["interface"]["output"]
    assert (out["channels"], out["field_bits"]) == (3, 8)
    frame = np.random.default_rng(1).integers(0, 1 << 16, (H, W))
    lanes = np.stack([frame & 255, frame >> 8], axis=-1)
    res = swap_and_sum(lanes).astype(np.int64)
    want = sum(res[..., k] << (8 * k) for k in range(3)).ravel().tolist()
    assert _sim(tmp_path, core, frame, {}) == want


def negative_lane(img):
    v = img.astype(np.int32)
    return np.stack([v, v - 512], axis=-1)


def test_a_channel_that_can_be_negative_is_refused():
    with pytest.raises(NotImplementedError, match="channel 1 can be negative"):
        _compile(negative_lane)


# --------------------------------------------------------------------------- #
# a register array and a constant table, read at a computed index
# --------------------------------------------------------------------------- #

KNOTS = Param("knot", bits=10, shape=(9,), default=0)


def knee(img, knot):
    """A 9-knot curve: the segment from the top bits, then a straight line
    between its two knots -- a tone curve's shape."""
    v = img.astype(np.int32)
    seg = v >> 7
    frac = v & 127
    lo, hi = knot[seg], knot[seg + 1]
    return (lo + (((hi - lo) * frac) >> 7)).clip(0, 1023).astype(np.uint16)


def test_a_register_array_read_at_a_computed_index(tmp_path):
    core = _compile(knee, (KNOTS,), name="knee")
    names = [n for n, _ in core["params"]]
    assert names == [f"knot_{i}" for i in range(9)]
    knots = np.array([0, 40, 150, 330, 520, 700, 850, 960, 1023])
    frame = np.random.default_rng(2).integers(0, 1 << 10, (H, W))
    want = knee(frame.astype(np.uint16), knots).astype(np.int64)
    got = _sim(tmp_path, core, frame,
               {f"knot_{i}": int(k) for i, k in enumerate(knots)})
    assert got == want.ravel().tolist()


ROOTS = Rom([1024 - 16 * i * i // 9 for i in range(17)], name="roots")


def droop(img):
    """A 17-entry constant table and linear interpolation, its segment a
    power of two wide."""
    v = img.astype(np.int32)
    seg = (v >> 6) & 15
    frac = v & 63
    lo, hi = ROOTS[seg], ROOTS[seg + 1]
    return (lo + (((hi - lo) * frac) >> 6)).astype(np.uint16)


def test_a_small_constant_table_is_logic(tmp_path):
    """Small enough for the device to build as logic: constants read in the
    stage that uses them -- no memory, no stage of its own, no register."""
    core = _compile(droop, name="droop")
    assert core["params"] == [] and core["memories"] == []
    assert core["pipeline_stages"] == 1
    frame = np.random.default_rng(3).integers(0, 1 << 10, (H, W))
    want = droop(frame.astype(np.uint16)).astype(np.int64)
    assert _sim(tmp_path, core, frame, {}) == want.ravel().tolist()


BIG = Rom([(i * 37) % 1000 for i in range(1024)], name="big")


def big_table(img):
    return BIG[img.astype(np.int32)].astype(np.uint16)


def test_a_constant_table_the_device_makes_a_memory_is_a_registered_read(
        tmp_path):
    """Past the device's bound a table is a memory: its contents inline,
    read through a register -- a stage of its own, budget or no budget."""
    from np2hw.timing import device
    assert not device(None).rom_in_logic(BIG.data.size)
    core = _compile(big_table, name="big")
    assert core["pipeline_stages"] == 2
    assert core["verilog"].count("rom_big[") >= BIG.data.size
    frame = np.random.default_rng(5).integers(0, 1 << 10, (H, W))
    want = big_table(frame.astype(np.uint16)).astype(np.int64)
    assert _sim(tmp_path, core, frame, {}) == want.ravel().tolist()


def test_the_device_decides_how_a_table_is_built():
    """The same model on a device whose bound the table passes: logic here,
    a memory read there -- the model and the Rom say nothing about it."""
    from np2hw.timing import Series7

    class Lean(Series7):
        name = "lean"
        rom_logic_entries = 8

    _, mod = to_ir(droop, Image2D("img", W, H, bits=10))
    here = generate(mod, module_name="d")
    there = generate(mod, module_name="d", dev=Lean())
    assert here["pipeline_stages"] == 1 and "rom_roots[" not in here["verilog"]
    assert there["pipeline_stages"] == 2 and "rom_roots[" in there["verilog"]


def signed_index(img, knot):
    v = img.astype(np.int32)
    return knot[(v >> 7) - 1].astype(np.uint16)


def test_an_index_that_can_be_negative_is_refused():
    with pytest.raises(NotImplementedError, match="can be negative"):
        _compile(signed_index, (KNOTS,))


# --------------------------------------------------------------------------- #
# the order of the register ports
# --------------------------------------------------------------------------- #

GAIN, FLOOR = Param("gain", bits=8), Param("floor", bits=10)


def gated(img, gain, floor):
    """`gain` is met first running the model, `floor` first walking the
    output's expression."""
    x = img.astype(np.int32)
    v = (x * gain) >> 4
    level = x + 1
    return np.where(level - floor >= 0, v, 0).clip(0, 1023).astype(np.uint16)


def test_register_ports_are_listed_in_the_order_of_the_output_walk(tmp_path):
    """The order the reference's expression generators list them in -- so a
    core's interface does not change with the compiler that built it --
    and not the order the model happened to run in. (Addresses are the
    block's declaration's, and no compiler's.)"""
    core = _compile(gated, (GAIN, FLOOR), name="gated")
    assert [n for n, _ in core["params"]] == ["floor", "gain"]
    frame = np.random.default_rng(4).integers(0, 1 << 10, (H, W))
    want = gated(frame.astype(np.uint16), 20, 300).astype(np.int64)
    assert _sim(tmp_path, core, frame, {"gain": 20, "floor": 300}) == \
        want.ravel().tolist()
