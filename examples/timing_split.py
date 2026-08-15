"""Auto-split: the traced timing budget places the pipeline registers.

timing_budget.py proves the estimator's arithmetic; this example proves
what the emitter DOES with it. When a pointwise stage is deeper than the
clock, generate(clk_ns=...) no longer refuses: assign_stages cuts the
expression DAG where the accumulated path would exceed the budget, and
the emitter turns each cut into stage-boundary registers, a valid/flag
delay line, and nothing else. Retiming, never resampling -- the values
are untouched by construction, and these claims are where that
construction would fail first:

  SPLIT          the tone curve that failed 10 ns generates at the
                 148.5 MHz pixel budget as a 3-stage pipeline, and its
                 output is BIT-EXACT against the same NumPy function.
  ELASTIC        the split core under RANDOMIZED ready/valid
                 backpressure: values AND framing exact. The delay
                 lines advance on !stall only; a stall that slips one
                 stage and not another shows here first.
  ONE WORD       a 3-lane np.stack where every lane splits: the output
                 word must still assemble from ONE input pixel, so
                 shallow subexpressions ride delay lines to the final
                 stage. A lane misaligned by one pixel shows here.
  UNTOUCHED      a stage that fits the budget generates EXACTLY as it
                 would with no budget at all: same text, no registers
                 spent on stages nobody asked for.
  FLOOR          a DSP multiply alone outweighs a 4 ns clock: no
                 register placement can fix that, so generation
                 refuses, naming the operation.
  STENCIL        the windowed shape has its own cut: a demosaic over a
                 1280-wide line pays a LUT-RAM column read PLUS the
                 tap arithmetic in one cone, and that sum misses the
                 pixel budget (the bench measured it at -1.8 ns WNS).
                 With the budget stated, the window is snapshotted
                 into registers -- counters, edge replay and
                 write-backs untouched -- and the result is BIT-EXACT
                 against np.pad-and-slice NumPy, edges included.
  S-UNTOUCHED    the same stencil under a clock it fits generates
                 byte-identical to the no-budget text.
  S-FLOOR        a budget below the window read's own depth refuses,
                 naming the cone -- one snapshot register is this
                 shape's whole repertoire.

Run:  python examples/timing_split.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import BUILD, check, check_bp
from np2hw import Image2D, generate, to_ir
from np2hw.ir import Param

BITS, K = 12, 5
N = (1 << K) + 1                 # 33 knots -> 32 segments
SHIFT = BITS - K
FULL = (1 << BITS) - 1
CLK_148M5 = 1000.0 / 148.5       # the pixel clock the ISP wants to ride


def tone(img, knots):
    """The gamma block's piecewise-linear curve -- the bench's anchor
    stage, the one that missed 10 ns and got a clock island for it."""
    value = img.astype(np.int32)
    seg = value >> SHIFT
    frac = value & ((1 << SHIFT) - 1)
    base = knots[seg].astype(np.int32)
    step = knots[seg + 1].astype(np.int32) - base
    return (base + ((step * frac) >> SHIFT)).clip(0, FULL).astype(np.uint16)


def tone3(img, knots):
    """The same curve on each lane of a 3-channel word."""
    value = img.astype(np.int64)
    outs = []
    for c in range(3):
        chan = value[..., c]
        seg = chan >> SHIFT
        frac = chan & ((1 << SHIFT) - 1)
        base = knots[seg]
        step = knots[seg + 1] - base
        outs.append((base + ((step * frac) >> SHIFT)).clip(0, FULL))
    return np.stack(outs, axis=-1)


def shallow(img):
    """One add and a clip: comfortably inside any pixel clock."""
    value = img.astype(np.int32)
    return ((value >> 2) + (value >> 4)).clip(0, FULL).astype(np.uint16)


def white_balance(img, py, px, gain):
    """The phase-canvas chain shape: position-muxed gain, multiply,
    truncate, saturate -- the white balance stage as the bench runs it."""
    value = img.astype(np.uint32)
    out = np.empty_like(value)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            out[rows::2, cols::2] = (
                (value[rows::2, cols::2] * gain[i, j]) // 256).clip(0, FULL)
    return out.astype(np.uint16)


def demosaic(img, py, px):
    """The bilinear-demosaic shape over a real line width: one shared
    3x3 window, tap combinations routed to channels by CFA site."""
    value = img.astype(np.uint16)
    x = np.pad(value, 1, mode="edge")
    centre = x[1:-1, 1:-1]
    north, south = x[:-2, 1:-1], x[2:, 1:-1]
    west, east = x[1:-1, :-2], x[1:-1, 2:]
    cross = (north + south + west + east) // 4
    diag = (x[:-2, :-2] + x[:-2, 2:] + x[2:, :-2] + x[2:, 2:]) // 4
    horiz = (west + east) // 2
    vert = (north + south) // 2
    r = np.empty_like(value)
    g = np.empty_like(value)
    b = np.empty_like(value)
    sites = (
        ((py, px), centre, cross, diag),
        ((py, 1 - px), horiz, centre, vert),
        ((1 - py, px), vert, centre, horiz),
        ((1 - py, 1 - px), diag, cross, centre),
    )
    for (rows, cols), rt, gt, bt in sites:
        r[rows::2, cols::2] = rt[rows::2, cols::2]
        g[rows::2, cols::2] = gt[rows::2, cols::2]
        b[rows::2, cols::2] = bt[rows::2, cols::2]
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def adaptive(img, py, px):
    """A DIRECTION-ADAPTIVE plane, the Hamilton-Adams shape: second
    differences, their magnitudes, a comparison and a select. Deep
    enough that one clock cannot hold it, so the emitter must cut the
    arithmetic into stages rather than refuse."""
    value = img.astype(np.int32)
    h, w = value.shape
    x = np.pad(value, 2, mode="edge")

    def at(r, c):
        return x[2 + r:2 + r + h, 2 + c:2 + c + w]

    c0 = at(0, 0)
    lh = 2 * c0 - at(0, -2) - at(0, 2)
    lv = 2 * c0 - at(-2, 0) - at(2, 0)
    dh = np.abs(at(0, -1) - at(0, 1)) + np.abs(lh)
    dv = np.abs(at(-1, 0) - at(1, 0)) + np.abs(lv)
    gh = ((at(0, -1) + at(0, 1)) // 2 + lh // 4).clip(0, 255)
    gv = ((at(-1, 0) + at(1, 0)) // 2 + lv // 4).clip(0, 255)
    est = np.where(dh < dv, gh, gv)
    out = np.empty_like(value)
    out[py::2, px::2] = est[py::2, px::2]
    out[py::2, 1 - px::2] = c0[py::2, 1 - px::2]
    out[1 - py::2, px::2] = c0[1 - py::2, px::2]
    out[1 - py::2, 1 - px::2] = est[1 - py::2, 1 - px::2]
    return out.astype(np.uint8)


def _two_frames(meta, W, H, frames, values):
    """Drive the core through consecutive frames, in_sof on each first
    pixel, edge-feed pacing; return the collected output words."""
    mod = meta["module"]
    hb, vd = meta["hblank"], meta["vdrain"]
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a("module tb;")
    a("    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0;")
    a(f"    reg [{meta['in_bits']-1}:0] in_data;")
    for name, bits in meta["params"]:
        a(f"    reg [{bits-1}:0] param_{name};")
    a("    wire out_valid, in_ready, out_sof, out_eol, out_last;")
    a(f"    wire [{meta['out_bits']-1}:0] out_data;")
    a("    integer i, fh; integer r, c, fr;")
    a(f"    reg [{meta['in_bits']-1}:0] img [0:{len(frames)*W*H-1}];")
    a("    always #5 clk = ~clk;")
    a(f"    {mod} #(.WIDTH({W}), .HEIGHT({H})) dut (")
    a("        .clk(clk), .rst(rst), .in_valid(in_valid),")
    a("        .in_ready(in_ready), .in_sof(in_sof), .in_data(in_data),")
    for name, _ in meta["params"]:
        a(f"        .param_{name}(param_{name}),")
    a("        .out_valid(out_valid), .out_ready(1'b1), .out_sof(out_sof),")
    a("        .out_eol(out_eol), .out_last(out_last), "
      ".out_data(out_data));")
    a("    always @(posedge clk) if (out_valid) "
      "$fdisplay(fh, \"%0d\", out_data);")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        fh = $fopen(\"out.txt\", \"w\");")
    for name, _ in meta["params"]:
        a(f"        param_{name} = {int(values.get(name, 0))};")
    a("        @(negedge clk); rst = 0;")
    a(f"        for (fr = 0; fr < {len(frames)}; fr = fr + 1) begin")
    a(f"            for (r = 0; r < {H}; r = r + 1) begin")
    a(f"                for (c = 0; c < {W}; c = c + 1) begin")
    a(f"                    in_data = img[fr*{W*H} + r*{W} + c];")
    a("                    in_valid = 1;")
    a("                    in_sof = (r == 0) && (c == 0);")
    a("                    @(negedge clk);")
    a("                end")
    a("                in_valid = 0; in_sof = 0;")
    a(f"                for (c = 0; c < {hb}; c = c + 1) @(negedge clk);")
    a("            end")
    a(f"            for (c = 0; c < {vd}; c = c + 1) @(negedge clk);")
    a("        end")
    a("        $fclose(fh); $finish;")
    a("    end")
    a("endmodule")

    with open(os.path.join(BUILD, f"{mod}.v"), "w") as fh:
        fh.write(meta["verilog"] + "\n")
    with open(os.path.join(BUILD, "tb2f.v"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    flat = np.concatenate([f.ravel() for f in frames])
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(p):02x}" for p in flat) + "\n")
    subprocess.run(["iverilog", "-o", "sim2f.vvp", f"{mod}.v", "tb2f.v"],
                   check=True, cwd=BUILD, capture_output=True)
    subprocess.run(["vvp", "sim2f.vvp"], check=True, cwd=BUILD,
                   capture_output=True)
    return np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)


def main():
    rng = np.random.default_rng(21)
    knots_param = [Param("knots", bits=13, shape=(N,),
                         description="PWL knots")]
    table = {f"knots_{i}": int(round(FULL * (i / (N - 1)) ** 0.45))
             for i in range(N)}
    A = rng.integers(0, FULL + 1, (12, 16)).astype(np.uint16)
    A[0, 0], A[0, 1] = 0, FULL

    checks = []

    def result(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<56} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("timing_split:")

    _, out = to_ir(tone, Image2D("img", 16, 12, bits=BITS), *knots_param)
    meta = generate(out, module_name="tone_148", clk_ns=CLK_148M5)
    result("split: the 10 ns failure generates at 6.73 ns",
           meta["pipeline_stages"] == 3,
           f"{meta['pipeline_stages']} stages")

    ok = check("tone_148", tone, A, params=knots_param, param_values=table,
               bits=BITS, clk_ns=CLK_148M5)
    result("split: ...and stays bit-exact at full rate", ok)

    ok = check_bp("tone_bp", tone, A, params=knots_param, param_values=table,
                  bits=BITS, clk_ns=CLK_148M5)
    result("elastic: bit-exact under randomized backpressure", ok)

    FIELD, WBITS = 12, 36
    A3 = rng.integers(0, FULL + 1, (8, 12, 3)).astype(np.uint16)
    _, out3 = to_ir(tone3, Image2D("img", 12, 8, bits=WBITS), *knots_param,
                    channels=3)
    meta3 = generate(out3, module_name="tone3_148", clk_ns=CLK_148M5)
    split3 = meta3["pipeline_stages"] >= 2
    ok = split3 and check("tone3_148", tone3, A3, params=knots_param,
                          param_values=table, bits=WBITS, channels=3,
                          clk_ns=CLK_148M5)
    result("one word: all lanes split once, word from one pixel", ok,
           f"{meta3['pipeline_stages']} stages")

    _, outb = to_ir(shallow, Image2D("img", 16, 12, bits=BITS))
    plain = generate(outb, module_name="bl")["verilog"]
    budgeted = generate(outb, module_name="bl", clk_ns=CLK_148M5)
    result("untouched: a fitting stage generates identically",
           budgeted["pipeline_stages"] == 1 and budgeted["verilog"] == plain)

    try:
        generate(out, module_name="tone_floor", clk_ns=4.0)
        refused, named = False, False
    except ValueError as error:
        refused, named = True, "'mul'" in str(error)
    result("floor: one op deeper than the clock refuses, named",
           refused and named)

    SW, SH, SBITS = 1280, 6, 8
    cfa = [Param("py", bits=1, description="Row parity of the first colour"),
           Param("px", bits=1, description="Column parity of the first "
                                           "colour")]
    As = rng.integers(0, (1 << SBITS), (SH, SW)).astype(np.uint8)
    _, outs = to_ir(demosaic, Image2D("img", SW, SH, bits=SBITS), *cfa)
    meta_s = generate(outs, module_name="dm_148", clk_ns=CLK_148M5)
    split_s = meta_s["pipeline_stages"] == 2
    ok = split_s and check("dm_148", demosaic, As, params=cfa,
                           param_values={"py": 1, "px": 0}, bits=SBITS,
                           clk_ns=CLK_148M5)
    result("stencil: window snapshot at 6.73 ns, bit-exact w/ edges", ok,
           f"{meta_s['pipeline_stages']} stages")

    plain_s = generate(outs, module_name="dm_easy")["verilog"]
    easy_s = generate(outs, module_name="dm_easy", clk_ns=15.0)
    result("stencil: a fitting clock generates identically",
           easy_s["pipeline_stages"] == 1 and easy_s["verilog"] == plain_s)

    try:
        generate(outs, module_name="dm_floor", clk_ns=4.0)
        refused, named = False, False
    except ValueError as error:
        refused, named = True, "window read" in str(error)
    result("stencil: below the window read's own depth refuses, named",
           refused and named)

    # Two frames, back to back, in_sof pulsed on each first pixel: the
    # split stencil's registered line-buffer write leaves one write
    # pending ACROSS the frame boundary, and this is where it would
    # corrupt. Both frames must be bit-exact, independently.
    A2 = rng.integers(0, (1 << SBITS), (SH, SW)).astype(np.uint8)
    two = _two_frames(meta_s, SW, SH, [As, A2], {"py": 1, "px": 0})
    exp = [np.stack([c.astype(np.int64) & 0xFF for c in
                     np.moveaxis(demosaic(f, 1, 0), -1, 0)], 0) for f in
           (As, A2)]
    packed = [sum(e[c] << (8 * c) for c in range(3)).ravel() for e in exp]
    ok = (two is not None and two.size == 2 * SH * SW
          and np.array_equal(two[:SH * SW], packed[0])
          and np.array_equal(two[SH * SW:], packed[1]))
    result("stencil: two frames across a SOF, both bit-exact", ok)

    # A line buffer must be read THROUGH A REGISTER: block RAM has no
    # asynchronous read port, so an async read forces distributed RAM
    # plus a select tree that deepens with the line -- which is why the
    # same stencil closed at 1280 and missed at 1920. Assert the shape,
    # because simulation cannot see which primitive the tools pick.
    src = generate(outs, module_name="dm_shape")["verilog"]
    registered = ("chain1_q <= mem1[rd_col]" in src
                  and "mem1[ecol]" not in src.split("rd_col")[0])
    one_ahead = "rd_col = in_sof" in src
    result("stencil: line buffers are read through a register", registered)
    result("stencil: the address is one column AHEAD, so no schedule "
           "shift", one_ahead)

    # A plane deeper than the clock used to be the floor here: the
    # window snapshot was this emitter's only cut. Behind that snapshot
    # the planes read REGISTERS, so the arithmetic is an ordinary
    # pointwise DAG and cuts like one -- and the flags and the phase
    # select must ride the same delay, or the pixel that reaches the
    # output is described by another pixel's position.
    _, outad = to_ir(adaptive, Image2D("img", SW, SH, bits=SBITS), *cfa)
    meta_ad = generate(outad, module_name="ha_148", clk_ns=CLK_148M5)
    depth = meta_ad["pipeline_stages"]
    ok = depth > 2 and check("ha_148", adaptive, As, params=cfa,
                             param_values={"py": 1, "px": 0}, bits=SBITS,
                             clk_ns=CLK_148M5)
    result("stencil: a plane deeper than the clock CUTS, bit-exact", ok,
           f"{depth} stages")

    # A deeper pipeline means frame 2's first pixel is in flight while
    # frame 1's tail is still draining: SOF must re-anchor the counters
    # without the two frames meeting inside the delay lines.
    A2a = rng.integers(0, (1 << SBITS), (SH, SW)).astype(np.uint8)
    twoa = _two_frames(meta_ad, SW, SH, [As, A2a], {"py": 1, "px": 0})
    expa = [adaptive(f, 1, 0).astype(np.int64).ravel() for f in (As, A2a)]
    ok = (twoa is not None and twoa.size == 2 * SH * SW
          and np.array_equal(twoa[:SH * SW], expa[0])
          and np.array_equal(twoa[SH * SW:], expa[1]))
    result("stencil: two frames across a SOF with the arithmetic cut", ok)

    slow = generate(outad, module_name="ha_slow", clk_ns=20.0)
    result("stencil: the same plane under a clock it fits stays whole",
           slow["pipeline_stages"] == 1)

    wbp = [Param("py", bits=1, description="Row parity"),
           Param("px", bits=1, description="Column parity"),
           Param("gain", bits=16, shape=(2, 2),
                 description="Q8.8 white balance gain by CFA colour")]
    gains = {"py": 1, "px": 0, "gain_0_0": 435, "gain_0_1": 256,
             "gain_1_0": 256, "gain_1_1": 486}
    Aw = rng.integers(0, FULL + 1, (6, 8)).astype(np.uint16)
    _, outw = to_ir(white_balance, Image2D("img", 8, 6, bits=BITS), *wbp)
    meta_w = generate(outw, module_name="wb_148", clk_ns=CLK_148M5)
    ok = meta_w["pipeline_stages"] == 2 and check(
        "wb_148", white_balance, Aw, params=wbp, param_values=gains,
        bits=BITS, clk_ns=CLK_148M5)
    result("chain: the gain chain cuts after its multiply, bit-exact", ok,
           f"{meta_w['pipeline_stages']} stages")

    ok = all(checks)
    print("\n" + ("TIMING_SPLIT PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
