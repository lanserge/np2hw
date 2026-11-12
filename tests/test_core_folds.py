# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Folds: every position of a frame into a result that is the frame's.

A model reduces its frame -- ``x.sum()``, ``x.max()``, ``x.min()`` -- or
folds each pixel into the entry of an accumulator its key names,
``np.add.at(acc, key, value)`` (``maximum``, ``minimum``) into ``acc =
np.zeros(n, dtype, like=pixel)``. The core is a sink: it takes the stream,
and its result is read back by entry once the frame is folded. Every model
here is generated and simulated, frame after frame, its result read after
each and compared with NumPy's -- and read while the next frame streams,
which must still be the last frame's.
"""
import shutil
import subprocess

import numpy as np
import pytest

from np2hw import Image2D, generate, to_ir
from np2hw.core.trace import Unsupported, coords
from np2hw.ir import Param, Params

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None,
                                reason="needs iverilog")

W, H, BITS = 8, 6, 10


class _P:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _compile(fn, params=(), channels=1, bits=BITS, name="m", clk_ns=None):
    _, mod = to_ir(fn, Image2D("img", W, H, bits=bits), *params,
                   channels=channels)
    return generate(mod, module_name=name, clk_ns=clk_ns)


def _sim(tmp_path, core, frames, values=None, seed=0, settle=True, gap=40,
         ready=3):
    """Each frame through the core, its registers set before it, `gap`
    idle cycles after it (0: the next at once); the result stream taken
    while `$random & ready` is non-zero -- every cycle for ready=-1. With
    `settle`, the first frame waits out the clearing after reset. Returns
    every result beat as (data, sof, eol, last)."""
    in_bits, out_bits = core["in_bits"], core["out_bits"]
    h, w = frames.shape[1:3]
    values = values or [{} for _ in frames]
    regs = "".join(f"    reg [{b - 1}:0] param_{p} = 0;\n"
                   for p, b in core["params"])
    conns = "".join(f"        .param_{p}(param_{p}),\n"
                    for p, _ in core["params"])
    from np2hw.stream import pack_context
    layout = core["interface"]["input"]["ctx"]
    ctx = pack_context(layout, {nm: {"width": w, "height": h}.get(f, 0)
                                for nm, _, f in layout})
    sets = ["".join(f"            param_{p} = "
                    f"{int(v.get(p, core['param_defaults'][p])) & ((1 << b) - 1)};\n"
                    for p, b in core["params"]) for v in values]
    per_frame = "".join(f"            if (fr == {i}) begin\n{s_}            end\n"
                        for i, s_ in enumerate(sets))
    words = [int(v) for v in frames.ravel()]
    per = w * h
    sgn = "signed " if core["signed"] else ""
    (tmp_path / "in.hex").write_text("".join(f"{v:x}\n" for v in words))
    (tmp_path / "core.v").write_text(core["verilog"])
    (tmp_path / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0, out_ready = 0;
    reg [{in_bits - 1}:0] in_data = 0;
    wire in_ready, out_valid, out_sof, out_eol, out_last;
    wire {sgn}[{out_bits - 1}:0] out_data;
    wire [{len(f"{ctx:b}") - 1}:0] out_ctx;
{regs}    reg [{in_bits - 1}:0] img [0:{len(words) - 1}];
    integer f, i, fr, lasts = 0, seed = {seed};
    always #5 clk = ~clk;
    {core["module"]} dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_sof(in_sof), .in_data(in_data), .in_ctx({len(f"{ctx:b}")}'d{ctx}),
{conns}        .out_ctx(out_ctx), .out_valid(out_valid), .out_ready(out_ready),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_data(out_data));
    always @(posedge clk) if (out_valid && out_ready) begin
        $fdisplay(f, "%0d %0d %0d %0d %0d", out_data, out_sof, out_eol,
                  out_last, out_ctx);
        if (out_last) lasts = lasts + 1;
    end
    always @(negedge clk) out_ready <= {"1" if ready < 0 else f"($random(seed) & {ready}) != 0"};
    initial begin
        $readmemh("in.hex", img);
        f = $fopen("out.txt", "w");
        repeat (3) @(negedge clk); rst = 0;
        // the clearing after reset, as long a time as a stream's blanking
        if ({int(settle)}) repeat (300) @(negedge clk);
        for (fr = 0; fr < {len(frames)}; fr = fr + 1) begin
{per_frame}            for (i = 0; i < {per}; i = i + 1) begin
                while (($random(seed) & 3) == 0) begin
                    in_valid = 0; @(negedge clk);
                end
                in_data = img[fr * {per} + i]; in_sof = (i == 0); in_valid = 1;
                @(posedge clk); while (!in_ready) @(posedge clk);
                @(negedge clk);
            end
            in_valid = 0; in_sof = 0;
            repeat ({gap}) @(negedge clk);
        end
        i = 0;
        while (lasts < {len(frames)} && i < 100000) begin
            @(negedge clk); i = i + 1;
        end
        $fclose(f); $finish;
    end
endmodule
""")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", "core.v", "tb.v"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=120)
    beats = []
    from np2hw.stream import unpack_context
    for line in (tmp_path / "out.txt").read_text().splitlines():
        *beat, ctx_out = map(int, line.split())
        beats.append(tuple(beat))
        if beat[1]:
            # the result's frame carries the result's size
            res = core["interface"]["result"]
            got = unpack_context(layout, ctx_out)
            sizes = {f: got[nm] for nm, _, f in layout if f}
            assert (sizes.get("width"), sizes.get("height")) == \
                (res["cols"], res["rows"]), sizes
    return beats


def _beats(core, results):
    """The result stream a core puts out for these results: a beat an
    entry, its values side by side, framed by the result's rows."""
    cols = core["interface"]["result"]["cols"]
    out = core["interface"]["output"]
    field = out.get("field_bits")
    beats = []
    for r in results:
        r = np.asarray(r)
        many = field is not None
        entries = r.reshape(-1, r.shape[-1]) if many else r.reshape(-1, 1)
        for i, e in enumerate(entries):
            v = sum(int(x) << (field * j) for j, x in enumerate(e)) if many \
                else int(e[0])
            beats.append((v, int(i == 0), int(i % cols == cols - 1),
                          int(i == len(entries) - 1)))
    return beats


def _check(tmp_path, fn, params=(), values=None, frames=4, seed=1, **kw):
    core = _compile(fn, params, name=fn.__name__, **kw)
    rng = np.random.default_rng(seed)
    video = rng.integers(0, 1 << BITS, (frames, H, W))
    vals = values or [{} for _ in range(frames)]
    beats = _sim(tmp_path, core, video, vals)
    want = []
    for k, frame in enumerate(video):
        p = _P(**{n: np.uint16(v) for n, v in vals[k].items()})
        want.append(fn(frame.astype(np.uint16), p) if params
                    else fn(frame.astype(np.uint16)))
    assert beats == _beats(core, want), fn.__name__
    return core, beats


# --------------------------------------------------------------------------- #
# Reductions whole: one entry, two registers
# --------------------------------------------------------------------------- #

def total(img):
    return img.sum()


def peak(img):
    return img.max()


def low(img):
    return np.min(img.astype(np.int16) - 300)


@pytest.mark.parametrize("fn", [total, peak, low])
def test_a_reduction_is_one_entry(tmp_path, fn):
    core, _ = _check(tmp_path, fn)
    assert core["interface"]["result"]["rows"] == \
        core["interface"]["result"]["cols"] == 1
    assert not core["memories"]                 # a register accumulates


# --------------------------------------------------------------------------- #
# Reductions over the image's axes: an entry for each position kept
# --------------------------------------------------------------------------- #

def row_sums(img):
    return img.sum(axis=1)


def column_peaks(img):
    return np.max(img, axis=0)


def row_lows(img):
    return (img.astype(np.int32) - 512).min(axis=-1)


def both_axes(img):
    return img.sum(axis=(0, 1))


@pytest.mark.parametrize("fn", [row_sums, column_peaks, row_lows, both_axes])
def test_a_reduction_over_the_image_axes(tmp_path, fn):
    """`x.sum(axis=...)` is the fold keyed by the axes it keeps -- the same
    result as `np.add.at(acc, y, x)` written out."""
    _check(tmp_path, fn)


def test_a_reduction_over_the_channels_is_no_fold():
    """Summing a word's channels leaves every pixel its own value: each
    pixel's arithmetic, not a fold (tests/test_core_reductions.py proves
    it bit-exact)."""
    core = _compile(lambda pixel: pixel.sum(axis=-1), channels=2,
                    bits=2 * BITS)
    assert "result" not in core["interface"]
    assert not core["memories"]


def test_an_axis_out_of_range_is_refused():
    with pytest.raises(Unsupported, match="axis 2"):
        _compile(lambda img: img.sum(axis=2))


# --------------------------------------------------------------------------- #
# Accumulators: entries by a key, three memories
# --------------------------------------------------------------------------- #

def hist(img):
    """A histogram: each pixel counts once, in the bin its top bits name."""
    acc = np.zeros(16, np.int64, like=img)
    np.add.at(acc, img >> 6, 1)
    return acc


def zones(img):
    """Sums over a grid of zones, the key from where the pixel is."""
    y, x = coords(img)
    acc = np.zeros((3, 2), np.uint32, like=img)
    np.add.at(acc, (y // 2, x // 4), img)
    return acc


def brightest(img, p):
    """The largest difference from a black level, per low-bit class, from
    a start below anything a pixel gives."""
    acc = np.full(4, -5, np.int32, like=img)
    np.maximum.at(acc, img & 3, img.astype(np.int32) - p.black)
    return acc


def narrow(img):
    """A sum in eight bits, which NumPy wraps -- and so must the core."""
    acc = np.zeros(3, np.uint8, like=img)
    np.add.at(acc, (img & 1) + 1, img)
    return acc


def side_by_side(img):
    """Two folds by one key, their entries' values side by side."""
    acc = np.zeros(4, np.int64, like=img)
    low_ = np.full(4, 1023, np.int64, like=img)
    np.add.at(acc, img & 3, img)
    np.minimum.at(low_, img & 3, img)
    return np.stack([acc, low_], axis=-1)


def clipped_peak(img):
    """A maximum into eight bits: NumPy casts each value into the
    accumulator first, so the largest is of the values as cast."""
    acc = np.zeros(2, np.uint8, like=img)
    np.maximum.at(acc, img & 1, img)
    return acc


def mixed_widths(img):
    """A sum NumPy wraps in eight bits beside a count it does not: the
    stack is the wider, and the narrow sum keeps its wrap in it."""
    total = np.zeros(2, np.uint8, like=img)
    count = np.zeros(2, np.int64, like=img)
    np.add.at(total, img & 1, img)
    np.add.at(count, img & 1, 1)
    return np.stack([total, count], axis=-1)


@pytest.mark.parametrize("fn", [clipped_peak, mixed_widths])
def test_a_value_is_cast_into_its_accumulator(tmp_path, fn):
    _check(tmp_path, fn)


@pytest.mark.parametrize("fn, level", [(peak, 0), (low, 1023), (total, 0)])
def test_a_reduction_of_a_flat_frame(tmp_path, fn, level):
    """Every pixel at the end of the range a reduction starts from: the
    start must be the operation's identity, or it shows."""
    core = _compile(fn, name=fn.__name__)
    video = np.full((2, H, W), level)
    beats = _sim(tmp_path, core, video)
    assert beats == _beats(core, [fn(f.astype(np.uint16)) for f in video])


def test_a_histogram(tmp_path):
    core, _ = _check(tmp_path, hist)
    assert [(m["module"].rsplit("_", 1)[1], m["depth"])
            for m in core["memories"]] == [("acc", 16)]


def test_zones_by_position(tmp_path):
    _check(tmp_path, zones)


def test_a_maximum_from_a_start_below_the_values(tmp_path):
    _check(tmp_path, brightest, (Params([Param("black", bits=10)]),),
           values=[{"black": b} for b in (0, 700, 1023, 300)])


def test_a_narrow_accumulator_wraps_as_numpy_does(tmp_path):
    _check(tmp_path, narrow)


def test_folds_side_by_side(tmp_path):
    core, _ = _check(tmp_path, side_by_side)
    assert core["interface"]["output"]["channels"] == 2


def test_an_entry_no_pixel_reaches_reads_as_the_start(tmp_path):
    """A frame of one value reaches one bin; every other is its start --
    though the frame before filled them: streaming a frame out clears it."""
    core = _compile(hist, name="hist")
    video = np.random.default_rng(3).integers(0, 1 << BITS, (5, H, W))
    video[2:] = 64 * 5                       # bin 5 only, from frame 2 on
    beats = _sim(tmp_path, core, video)
    assert beats == _beats(core, [hist(f) for f in video])
    assert [b[0] for b in beats[-16:]] == [0] * 5 + [W * H] + [0] * 10


@pytest.mark.parametrize("fn", [zones, side_by_side])
def test_a_result_streams_out_while_the_next_frame_folds(tmp_path, fn):
    """Frames close behind each other, the result taken whenever its
    consumer is ready: every result exact, in order."""
    core = _compile(fn, name=fn.__name__)
    video = np.random.default_rng(4).integers(0, 1 << BITS, (5, H, W))
    beats = _sim(tmp_path, core, video, gap=0, ready=1)
    assert beats == _beats(core, [fn(f) for f in video])


def wide(img):
    """More entries than a frame has pixels: sweeping the frame out takes
    longer than the next frame does to arrive, so it waits."""
    acc = np.zeros(256, np.int64, like=img)
    np.add.at(acc, img >> 2, 1)
    return acc


def test_a_frame_waits_for_the_last_to_be_streamed_out(tmp_path):
    """Frames back to back, each arriving before the last is streamed out:
    it waits, and is folded whole."""
    core = _compile(wide, name="wide")
    video = np.random.default_rng(5).integers(0, 1 << BITS, (5, H, W))
    beats = _sim(tmp_path, core, video, gap=0, ready=-1)
    assert beats == _beats(core, [wide(f) for f in video])


@pytest.mark.parametrize("settle", [True, False])
def test_a_frame_sent_during_the_clearing_after_reset_waits(tmp_path, settle):
    """The first frame, sent at once after reset or once the memory is
    clear: folded whole either way."""
    core = _compile(hist, name="hist")
    video = np.random.default_rng(6).integers(0, 1 << BITS, (2, H, W))
    beats = _sim(tmp_path, core, video, settle=settle)
    assert beats == _beats(core, [hist(f) for f in video])


def test_a_fold_cut_into_stages(tmp_path):
    """The pixel's datapath cut to a clock: the fold takes each word as it
    leaves the last stage."""
    def deep(img):
        v = img.astype(np.int64)
        x = v * v * 3 + v
        acc = np.zeros(8, np.int64, like=img)
        np.add.at(acc, img & 7, (x * x) & 65535)
        return acc
    core, _ = _check(tmp_path, deep, clk_ns=6.0)
    assert core["pipeline_stages"] > 1


def edges(img):
    """Edge strength through a padded window, binned by the pixel's level:
    the fold takes what the window computes, its steps flushed past the
    frame's last."""
    x = np.pad(img.astype(np.int32), 1, mode="edge")
    gx = np.abs(x[1:-1, 2:] - x[1:-1, :-2])
    gy = np.abs(x[2:, 1:-1] - x[:-2, 1:-1])
    acc = np.zeros(4, np.int64, like=img)
    np.add.at(acc, img >> 8, gx + gy)
    return acc


def test_a_fold_of_a_window(tmp_path):
    _check(tmp_path, edges)


# --------------------------------------------------------------------------- #
# The graphs before any hardware
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fn", [total, peak, low, hist, zones, narrow,
                                side_by_side, clipped_peak, mixed_widths,
                                edges, wide, row_sums, column_peaks,
                                row_lows, both_axes])
def test_the_graphs_are_numpy(fn):
    """The traced graph, and the canonical one lowering starts from, each
    evaluated on frames: exactly what NumPy computes -- the canonical in
    exact integers."""
    from np2hw.core.canonical import canonicalise
    from np2hw.core.evaluate import evaluate
    from np2hw.core.trace import trace
    t = trace(fn, Image2D("img", W, H, bits=BITS))
    c = canonicalise(t)
    for seed in range(3):
        img = np.random.default_rng(seed).integers(
            0, 1 << BITS, (H, W)).astype(np.uint16)
        want = np.asarray(fn(img))
        traced = np.asarray(evaluate(t.g, [t.output], t.source, img, {})[0])
        exact = np.asarray(evaluate(c.g, [c.output], c.source, img, {})[0])
        assert traced.dtype == want.dtype and np.array_equal(traced, want)
        assert np.array_equal(exact, want.astype(np.int64))


# --------------------------------------------------------------------------- #
# What is refused, by name
# --------------------------------------------------------------------------- #

def test_an_accumulator_the_trace_cannot_see_is_refused():
    def plain(img):
        acc = np.zeros(4, np.int64)
        np.add.at(acc, img & 3, img)
        return acc
    with pytest.raises(Unsupported, match="like=pixel"):
        _compile(plain)


def test_two_writes_into_one_accumulator_are_refused():
    def twice(img):
        acc = np.zeros(4, np.int64, like=img)
        np.add.at(acc, img & 3, img)
        np.add.at(acc, img & 3, 1)
        return acc
    with pytest.raises(Unsupported, match="two writes"):
        _compile(twice)


def test_a_fold_over_part_of_the_frame_is_refused():
    def rows(img):
        y, _ = coords(img)
        acc = np.zeros(3, np.int64, like=img)
        np.add.at(acc, y // 2, y)
        return acc
    with pytest.raises(Unsupported, match="each pixel folds once"):
        _compile(rows)
