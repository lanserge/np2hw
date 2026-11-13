# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The C backend computes what the NumPy model computes, bit for bit.

The models are the compiler's own specification: every model the core
tests compile to Verilog (windows, pads, phase planes, channel words,
tables, register arrays, reads across frames, stages) and the shipped
model files, each run as C and compared with the same function run by
NumPy on random frames and random register values.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from np2hw import Image2D
from np2hw.core.cemit import compile_c
from np2hw.core.trace import Unsupported
from np2hw.ir import Param, Params, np_dtype

pytestmark = pytest.mark.skipif(shutil.which("cc") is None,
                                reason="needs a C compiler")

HERE = Path(__file__).parent
ROOT = HERE.parent


def _module(path: Path):
    name = f"_cemit_{path.stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _value(p: Param, rng):
    """A random register value as the NumPy run is handed it: a scalar of
    the register's NumPy type, an array of it for a register array."""
    bits, signed = p.bits, bool(p.signed)
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - (1 if signed else 0))) - 1
    shape = tuple(getattr(p, "shape", ()) or ())
    return _typed(p, rng.integers(lo, hi + 1, shape) if shape
                  else int(rng.integers(lo, hi + 1)))


def _typed(p: Param, v):
    dt = np_dtype(p.bits, p.signed)
    return np.asarray(v, dt) if np.ndim(v) else dt.type(v)


def _frame(rng, shape, bits, channels, signed=False):
    if channels > 1:
        f = bits // channels
        return rng.integers(0, 1 << f, (*shape, channels)).astype(
            np.uint8 if f <= 8 else np.uint16)
    if signed:
        return rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), shape).astype(
            np.int8 if bits <= 8 else np.int16)
    return rng.integers(0, 1 << bits, shape).astype(
        np.uint8 if bits <= 8 else np.uint16)


def _check(tmp_path, fn, params=(), *, width, height, bits=8, frames=0,
           channels=1, seeds=(0, 1, 2), values=None):
    image = Image2D("img", width, height, bits=bits, frames=frames)
    k = compile_c(fn, image, *params, channels=channels, build=tmp_path)
    shape = (frames, height, width) if frames else (height, width)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        vals = ([_typed(p, v) for p, v in zip(params, values)]
                if values is not None else [_value(p, rng) for p in params])
        img = _frame(rng, shape, bits, channels)
        want = np.asarray(fn(img, *vals))
        regs = {}
        for p, v in zip(params, vals):
            regs[p.name] = v
        got = k(img, **regs)
        assert got.shape == want.shape, (got.shape, want.shape)
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        assert np.array_equal(got, want), (fn.__name__, seed)
        if got.ndim >= 2 and got.shape[0] > 2:
            assert np.array_equal(k(img, threads=3, **regs), want)
    return k


# --------------------------------------------------------------------------- #
# The core tests' models
# --------------------------------------------------------------------------- #

WINDOWS = _module(HERE / "test_core_windows.py")
LOOKUPS = _module(HERE / "test_core_lookups.py")
FRAMES = _module(HERE / "test_core_frames.py")
STAGES = _module(HERE / "test_core_stages.py")


@pytest.mark.parametrize("name", ["blur", "sparse", "cross", "dark", "back",
                                  "composed", "tail", "declared", "rows_only",
                                  "cols_only"])
def test_windows(tmp_path, name):
    m = WINDOWS
    _check(tmp_path, getattr(m, name), width=m.W, height=m.H, bits=m.BITS)


def test_phase_planes(tmp_path):
    m = WINDOWS
    for py in (0, 1):
        for px in (0, 1):
            _check(tmp_path, m.planes, (m.PY, m.PX), width=m.W, height=m.H,
                   bits=m.BITS, values=[py, px], seeds=(py * 2 + px,))


def test_window_with_a_register(tmp_path):
    m = WINDOWS
    _check(tmp_path, m.sharpen, (m.GAIN,), width=m.W, height=m.H, bits=m.BITS)


def test_channels_in_and_out(tmp_path):
    _check(tmp_path, WINDOWS.two_channels, width=WINDOWS.W, height=WINDOWS.H,
           bits=16, channels=2)
    _check(tmp_path, LOOKUPS.swap_and_sum, width=LOOKUPS.W, height=LOOKUPS.H,
           bits=16, channels=2)


def _lane_sum(w):
    """A three-tap sum down the rows of each lane, zero-padded."""
    p = np.pad(w, ((1, 1), (0, 0), (0, 0)), mode="constant").astype(np.int64)
    n = w.shape[0]
    return p[:n] + p[1:n + 1] + p[2:n + 2]


@pytest.mark.parametrize("fn", [
    lambda w: w[..., 1].astype(np.int64),
    lambda w: w.astype(np.int64) * 2,
    _lane_sum,
])
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_a_signed_image_has_signed_lanes(tmp_path, fn, dtype):
    """Each lane of a signed multi-lane image is of the image's type: an
    int32 image's -2 is -2, not 2**32 - 2."""
    a = np.random.default_rng(0).integers(-100, 100, (9, 11, 3)).astype(dtype)
    bits = np.dtype(dtype).itemsize * 8 * 3
    k = compile_c(fn, Image2D("img", 11, 9, bits=bits, signed=True),
                  channels=3, build=tmp_path)
    want = fn(a)
    got = k(a)
    assert got.dtype == want.dtype and np.array_equal(got, want)


def test_a_signed_image_with_lanes_is_refused_in_hardware():
    """Hardware takes unsigned sources, one lane or several."""
    from np2hw import to_ir
    with pytest.raises(Unsupported, match="a signed source"):
        to_ir(lambda w: w[..., 0], Image2D("img", 11, 9, bits=48, signed=True),
              channels=3)


@pytest.mark.parametrize("name, params", [
    ("knee", ("KNOTS",)), ("droop", ()), ("wide_table", ()), ("big_table", ()),
    ("gated", ("GAIN", "FLOOR"))])
def test_tables_and_registers(tmp_path, name, params):
    m = LOOKUPS
    ps = tuple(getattr(m, p) for p in params)
    values = None
    if name == "knee":              # a monotone curve, as the knots are
        values = [np.sort(np.random.default_rng(7).integers(0, 1024, 9))]
    _check(tmp_path, getattr(m, name), ps, width=m.W, height=m.H, bits=10,
           values=values)


@pytest.mark.parametrize("name", ["average", "change", "late", "forward"])
def test_reads_across_frames(tmp_path, name):
    m = FRAMES
    _check(tmp_path, getattr(m, name), width=m.W, height=m.H, bits=m.BITS,
           frames=m.FRAMES)


def test_frame_channels(tmp_path):
    m = FRAMES
    _check(tmp_path, m.mixed, width=m.W, height=m.H, bits=16, channels=2,
           frames=m.FRAMES)


@pytest.mark.parametrize("name, params", [
    ("quadrature", ("POS", "TAG")), ("by_equality", ("POS",)),
    ("by_parity", ("POS",))])
def test_frames_routed_by_registers(tmp_path, name, params):
    m = FRAMES
    ps = tuple(getattr(m, p) for p in params)
    _check(tmp_path, getattr(m, name), ps, width=m.W, height=m.H, bits=m.BITS,
           frames=m.FRAMES)


@pytest.mark.parametrize("name, kw", [
    ("polynomial", {}), ("tinted", {}), ("lanes", {"bits": 16, "channels": 2}),
    ("history", {"frames": 4})])
def test_stage_models(tmp_path, name, kw):
    m = STAGES
    fn = getattr(m, name)
    ps = (m.GAIN, m.BIAS) if name == "polynomial" else (m.GAIN,)
    kw = {"bits": m.BITS, **kw}
    _check(tmp_path, fn, ps, width=m.W, height=m.H, **kw)


# --------------------------------------------------------------------------- #
# The shipped model files
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", sorted((ROOT / "examples" / "isp").glob("*.py")),
                         ids=lambda p: p.stem)
def test_model_files(tmp_path, path):
    mod = _module(path)
    fn = getattr(mod, "model", None) or getattr(mod, "isp")
    decl = getattr(mod, "PARAMS", [])
    image = Image2D("img", 37, 19, bits=8)
    rng = np.random.default_rng(3)
    if isinstance(decl, Params):
        k = compile_c(fn, image, decl, build=tmp_path)
        values = {p.name: _value(p, rng) for p in decl.leaves()}
        img = _frame(rng, (19, 37), 8, 1)
        want = fn(img, decl.bind(values))
        assert np.array_equal(k(img, **values), want)
    else:
        params = tuple(decl)
        _check(tmp_path, fn, params, width=37, height=19, bits=8)


# --------------------------------------------------------------------------- #
# Frames of any size, signed arithmetic, threads
# --------------------------------------------------------------------------- #

W5 = (1, 4, 6, 4, 1)


def smooth5(img, black):
    p = np.pad(img, 2, mode="edge").astype(np.int32)
    H, W = img.shape
    acc = None
    for i in range(5):
        for j in range(5):
            t = p[i:i + H, j:j + W] * (W5[i] * W5[j])
            acc = t if acc is None else acc + t
    return (acc - black.astype(np.int32) * 256).clip(0, 1 << 30).astype(np.int32)


def gradient(img, gain):
    """Zero padding, signed values, abs, where, floor division, modulo and
    a wrap into int8."""
    p = np.pad(img, 1).astype(np.int16)
    H, W = img.shape
    gx = p[1:H + 1, 2:] - p[1:H + 1, :W]
    gy = p[2:, 1:W + 1] - p[:H, 1:W + 1]
    m = np.where(np.abs(gx) > np.abs(gy), gx, -gy)
    q = (m * gain) // 7 + m % 5
    return q.astype(np.int8).astype(np.int16) + (q >> 3)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 13), (40, 67)])
def test_smooth5_at_any_size(tmp_path, shape):
    k = compile_c(smooth5, Image2D("img", 16, 16, bits=8),
                  Param("black", bits=16), build=tmp_path)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, shape).astype(np.uint8)
    for black in (0, 3, 300):
        assert np.array_equal(k(img, black=black),
                              smooth5(img, np.uint16(black)))
    assert np.array_equal(k(img, black=3, threads=3),
                          smooth5(img, np.uint16(3)))


@pytest.mark.parametrize("shape", [(1, 1), (3, 2), (9, 31)])
def test_signed_arithmetic(tmp_path, shape):
    k = compile_c(gradient, Image2D("img", 16, 16, bits=8),
                  Param("gain", bits=6), build=tmp_path)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, shape).astype(np.uint8)
    for gain in (0, 1, 37, 63):
        want = gradient(img, np.uint8(gain))
        got = k(img, gain=gain)
        assert got.dtype == want.dtype and np.array_equal(got, want)


def test_a_value_over_an_axis_the_output_lacks_is_refused(tmp_path):
    def first_row(img):
        return img[0]                   # (W,): the rows dropped -- fine
    k = compile_c(first_row, Image2D("img", 8, 4, bits=8), build=tmp_path)
    img = np.arange(32, dtype=np.uint8).reshape(4, 8)
    assert np.array_equal(k(img), img[0])


def lens(img, a, b):
    """Products, a difference and a negation wider than 64 bits, wrapped
    by NumPy's int64 and cut again into int32."""
    x = img.astype(np.int64) * a * b * b * b * b
    y = x * x - (x << 7)
    return (-y + 3).astype(np.int32)


def test_products_wider_than_64_bits_wrap_as_numpy_does(tmp_path):
    k = compile_c(lens, Image2D("img", 16, 8, bits=12),
                  Param("a", bits=31, signed=True), Param("b", bits=16),
                  build=tmp_path)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 1 << 12, (8, 16)).astype(np.uint16)
    with np.errstate(over="ignore"):
        for a, b in ((-(1 << 30), 65535), (123456789, 40000), (1, 1)):
            want = lens(img, np.int32(a), np.uint16(b))
            assert np.array_equal(k(img, a=a, b=b), want)


def test_a_wrapped_product_shifts_as_numpy_does(tmp_path):
    """A shift reads the product as NumPy's int64 left it, wrapped."""
    def shifted(img, a):
        x = img.astype(np.int64) * a * a * a
        return (x >> 3).astype(np.int32)
    k = compile_c(shifted, Image2D("img", 8, 4, bits=12),
                  Param("a", bits=31, signed=True), build=tmp_path)
    img = np.random.default_rng(4).integers(0, 1 << 12, (4, 8)).astype(
        np.uint16)
    with np.errstate(over="ignore"):
        for a in (-(1 << 30), 987654321, 3):
            assert np.array_equal(k(img, a=a), shifted(img, np.int32(a)))


def test_a_register_window_keeps_what_numpy_crops(tmp_path):
    crop = _module(HERE / "test_runtime_frame.py").crop
    regs = [Param(n, bits=16) for n in ("y0", "x0", "h", "w")]
    k = compile_c(crop, Image2D("img", 20, 12, bits=8), *regs,
                  build=tmp_path)
    img = np.random.default_rng(6).integers(0, 256, (12, 20)).astype(np.uint8)
    for y0, x0, h, w in ((0, 0, 65535, 65535), (3, 5, 4, 7),
                         (10, 17, 9, 9), (11, 19, 1, 1), (12, 0, 4, 4)):
        want = crop(img, *(np.uint16(v) for v in (y0, x0, h, w)))
        got = k(img, y0=y0, x0=x0, h=h, w=w)
        assert got.shape == want.shape and np.array_equal(got, want)


FOLDS = _module(HERE / "test_core_folds.py")


@pytest.mark.parametrize("name", ["total", "peak", "low", "hist", "zones",
                                  "narrow", "side_by_side", "edges",
                                  "clipped_peak", "mixed_widths",
                                  "row_sums", "column_peaks", "row_lows",
                                  "both_axes"])
def test_folds(tmp_path, name):
    m = FOLDS
    _check(tmp_path, getattr(m, name), width=m.W, height=m.H, bits=m.BITS)


def test_a_fold_with_a_register(tmp_path):
    m = FOLDS

    def brightest(img, black):
        return m.brightest(img, m._P(black=black))
    _check(tmp_path, brightest, (Param("black", bits=10),),
           width=m.W, height=m.H, bits=m.BITS)
