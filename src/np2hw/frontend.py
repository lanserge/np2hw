"""np2hw NumPy front-end: trace NumPy-style code into the line IR (ir.py).

One representation: slicing/arithmetic/astype are flattened into a weighted tap
map (weights may be negative) and lowered to the line IR, so the delay-line
analysis applies to everything.

Faithful dtype, signedness from NumPy: each value carries (bits, signed) read
from a NumPy dtype; promotion on +/-/* uses np.result_type, so width AND sign
follow NumPy exactly (uint8 * -1 -> int16, uint8 + int8 -> int16, ...). The
spatial sum is clipped to min(natural, recorded width); a uint8 sum WRAPS unless
widened with .astype first. Subtraction / negative weights produce signed
results (Sobel, gradients). Oracle = the SAME function on real NumPy dtypes.

Idiom:  x = img.astype(np.int16); ... -x[...] + x[...] ...

Flattening is bit-exact only within one dtype; mixing a NARROWER clipping
sub-expression into a WIDER one is refused with a clear error (widen first).

Scope: single image; power-of-2 floor-div; slices with non-negative starts, unit
step; stencil min offset 0.
"""
from __future__ import annotations

import numpy as np

from .ir import ImageStreamer, Image2D, Indexer, ImageOp, Const, Param, Params


def _np(bits, signed):
    return np.dtype(f"{'int' if signed else 'uint'}{bits}")


def _promote(ab, asg, bb, bsg):
    dt = np.result_type(_np(ab, asg), _np(bb, bsg))
    return dt.itemsize * 8, dt.kind == "i"


def _promote_const(bits, signed, k):
    other = k.dtype if isinstance(k, np.integer) else k     # numpy scalar -> its dtype
    dt = np.result_type(_np(bits, signed), other)
    return dt.itemsize * 8, dt.kind == "i"


def _dtbits(dt):
    if isinstance(dt, int):
        return dt, False
    d = np.dtype(dt)
    return d.itemsize * 8, d.kind == "i"


def _norm_pad(pad_width):
    """Normalise np.pad's pad_width to ((top,bottom),(left,right)) for a 2-D image."""
    pw = np.asarray(pad_width)
    if pw.ndim == 0:                       # int -> all sides
        p = int(pw); return ((p, p), (p, p))
    if pw.shape == (2,):                   # (before, after) for all axes
        a, b = int(pw[0]), int(pw[1]); return ((a, b), (a, b))
    if pw.shape == (2, 2):                 # per-axis
        return ((int(pw[0][0]), int(pw[0][1])), (int(pw[1][0]), int(pw[1][1])))
    raise ValueError(f"unsupported pad_width {pad_width!r}")


def _term_range(coeff, ilo, ihi):
    """(lo, hi) of `coeff * input` over input in [ilo, ihi]. coeff is an int
    constant or a Param (then bounded by its register width)."""
    if isinstance(coeff, Param):
        if coeff.signed:
            clo, chi = -(1 << (coeff.bits - 1)), (1 << (coeff.bits - 1)) - 1
        else:
            clo, chi = 0, (1 << coeff.bits) - 1
    else:
        clo = chi = coeff
    corners = [clo * ilo, clo * ihi, chi * ilo, chi * ihi]
    return min(corners), max(corners)


def _axis(s, dim):
    if isinstance(s, slice):
        if s.step not in (None, 1):
            raise NotImplementedError("strided slice is out of scope")
        start, stop, _ = s.indices(dim)
        if start < 0:
            raise NotImplementedError("negative slice start is out of scope")
        return start, max(0, stop - start)
    if isinstance(s, int):
        if s < 0:
            raise NotImplementedError("negative index is out of scope")
        return s, 1
    raise TypeError(f"unsupported index {s!r}")


class Traced:
    def __init__(self, image, taps, shape, bits, signed,
                 spatial_bits=None, post=(), pad=(0, 0, 0, 0), mode="none"):
        self.image = image
        self.taps = dict(taps)               # {(row,col): weight}, weight may be < 0
        self.shape = tuple(shape)
        self.bits = bits
        self.signed = signed
        self.spatial_bits = bits if spatial_bits is None else spatial_bits
        self.post = list(post)
        self.pad = tuple(pad)                # (top, bottom, left, right) padding rows/cols
        self.mode = mode                     # 'none' | 'edge' (replicate) | 'zero'

    @classmethod
    def image_input(cls, image):
        return cls(image, {(0, 0): 1}, (image.height, image.width),
                   image.bits, image.signed)

    def _derive(self, **kw):
        """Build a Traced sharing this one's pad/mode unless overridden."""
        base = dict(image=self.image, taps=self.taps, shape=self.shape,
                    bits=self.bits, signed=self.signed,
                    spatial_bits=self.spatial_bits, post=self.post,
                    pad=self.pad, mode=self.mode)
        base.update(kw)
        return Traced(**base)

    def _spatial(self, taps, shape, bits, signed):
        return self._derive(taps=taps, shape=shape, bits=bits, signed=signed,
                            spatial_bits=bits, post=[])

    # -- np.pad / np.where interception (keeps spec = oracle) ---------------- #

    def __array_function__(self, func, types, args, kwargs):
        if func is np.pad:
            return self._pad(*args[1:], **kwargs)
        if func is np.where:
            return _make_mux(*args)
        return NotImplemented

    def _pad(self, pad_width, mode="constant", **kw):
        if self.taps != {(0, 0): 1} or self.post or self.mode != "none":
            raise ValueError("np.pad must be applied directly to the image input")
        ((pt, pb), (pl, pr)) = _norm_pad(pad_width)
        m = {"edge": "edge", "constant": "zero"}.get(mode)
        if m is None:
            raise NotImplementedError(f"np.pad mode {mode!r} not supported (edge/constant)")
        if m == "zero" and kw.get("constant_values", 0) != 0:
            raise NotImplementedError("only constant_values=0 supported")
        H, W = self.shape
        return self._derive(shape=(H + pt + pb, W + pl + pr),
                            pad=(pt, pb, pl, pr), mode=m)

    def natural(self):
        """Bits the true (unclipped) spatial value needs, incl. a sign bit."""
        inb = self.image.bits
        if self.image.signed:
            ilo, ihi = -(1 << (inb - 1)), (1 << (inb - 1)) - 1
        else:
            ilo, ihi = 0, (1 << inb) - 1
        lo = hi = 0
        for w in self.taps.values():
            tlo, thi = _term_range(w, ilo, ihi)
            lo += tlo
            hi += thi
        if lo < 0:
            return 1 + max(max(1, hi.bit_length()), (-lo).bit_length())
        return max(1, hi.bit_length())

    # -- slicing ------------------------------------------------------------- #

    def __getitem__(self, key):
        if self.post:
            raise ValueError("cannot slice after a pointwise op (v1)")
        if not isinstance(key, tuple):
            key = (key, slice(None))
        (dr, rows), (dc, cols) = _axis(key[0], self.shape[0]), _axis(key[1], self.shape[1])
        return self._spatial({(r + dr, c + dc): w for (r, c), w in self.taps.items()},
                             (rows, cols), self.bits, self.signed)

    # -- arithmetic ---------------------------------------------------------- #

    def __add__(self, o):
        if isinstance(o, (int, np.integer)) and o == 0:
            return self                                   # identity (enables sum())
        if isinstance(o, Param):                          # bias/offset register
            rb = max(self.bits, o.bits) + 1
            rs = self.signed or o.signed
            return self._derive(bits=rb, signed=rs,
                                post=self.post + [("addp", o.name, o.bits, o.signed, o.default)])
        if not isinstance(o, Traced):
            return NotImplemented
        if self.post or o.post:
            raise ValueError("additions must come before pointwise ops (v1)")
        if self.shape != o.shape:
            raise ValueError(f"shape mismatch: {self.shape} + {o.shape}")
        rb, rs = _promote(self.bits, self.signed, o.bits, o.signed)
        for op in (self, o):                               # mixed-width guard
            if op.bits < rb and op.signed == rs and op.natural() > op.bits:
                raise ValueError(
                    "mixed-width add cannot be flattened faithfully: the narrower "
                    f"operand needs {op.natural()}b but its dtype is {op.bits}b. "
                    "Widen it first (e.g. .astype(np.uint16) before scaling/summing).")
        t = dict(self.taps)
        for k, v in o.taps.items():
            if k in t:
                if isinstance(t[k], Param) or isinstance(v, Param):
                    raise ValueError(
                        "cannot sum two symbolic (Param) coefficients on one tap")
                t[k] = t[k] + v
            else:
                t[k] = v
        t = {k: v for k, v in t.items() if isinstance(v, Param) or v != 0}
        return self._spatial(t, self.shape, rb, rs)

    __radd__ = __add__

    def __sub__(self, o):
        return self.__add__(o * -1)

    def __neg__(self):
        return self * -1

    def __mul__(self, o):
        if isinstance(o, bool):
            return NotImplemented
        if isinstance(o, (int, np.integer)):
            rb, rs = _promote_const(self.bits, self.signed, o)
            k = int(o)
            if self.post:
                return self._derive(bits=rb, signed=rs, post=self.post + [("mulc", k)])
            return self._spatial({key: v * k for key, v in self.taps.items()},
                                 self.shape, rb, rs)
        if isinstance(o, Param):
            # Param x a bare single unit tap -> programmable tap coefficient
            # (stays in the spatial sum); Param x anything else -> trailing gain.
            if (not self.post and len(self.taps) == 1
                    and next(iter(self.taps.values())) == 1):
                rc = next(iter(self.taps))
                rb, rs = _promote(self.bits, self.signed, o.bits, o.signed)
                return self._spatial({rc: o}, self.shape, rb, rs)
            return self._derive(post=self.post + [("mulp", o.name, o.bits, o.signed, o.default)])
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, o):
        if not isinstance(o, int):
            return NotImplemented
        shift = o.bit_length() - 1
        if (1 << shift) != o:
            raise ValueError(f"floor-div by {o}: power-of-2 only in v1")
        return self._derive(post=self.post + [("shr", shift)])

    def clip(self, lo, hi):
        """Saturate to [lo, hi] (NumPy ndarray.clip semantics) -> comparator+clamp.
        Output range is exactly [lo, hi], so this narrows the width."""
        lo, hi = int(lo), int(hi)
        if lo < 0:
            b = 1 + max(max(1, hi.bit_length()), (-lo).bit_length()); sg = True
        else:
            b = max(1, hi.bit_length()); sg = False
        return self._derive(bits=b, signed=sg, post=self.post + [("clip", lo, hi)])

    def astype(self, dtype):
        b, sg = _dtbits(dtype)
        if self.post:
            return self._derive(bits=b, signed=sg, post=self.post + [("trunc", b)])
        if b > self.bits and self.signed == sg and self.natural() > self.bits:
            raise ValueError(
                "mixed-width: widening a value that would clip "
                f"(needs {self.natural()}b, dtype {self.bits}b). Widen operands first.")
        return self._derive(bits=b, signed=sg, spatial_bits=b, post=[])


# --------------------------------------------------------------------------- #
# Lowering to the line IR
# --------------------------------------------------------------------------- #

def lower(traced: Traced):
    image, taps, post, shape = traced.image, traced.taps, traced.post, traced.shape
    if not taps:
        raise ValueError("empty expression")
    if min(r for r, _ in taps) != 0 or min(c for _, c in taps) != 0:
        raise ValueError("v1 expects a stencil whose smallest offset is 0 "
                         "(write slices starting at 0, e.g. img[:-2], img[1:-1], img[2:])")

    s = ImageStreamer()
    row_lines = []
    for r in sorted({r for r, _ in taps}):
        cols = sorted((c, w) for (rr, c), w in taps.items() if rr == r)
        line = s.line(image, Indexer(r))
        idxs = [Indexer(c) for c, _ in cols]
        coeffs = [w if isinstance(w, Param) else Const(w) for _, w in cols]
        row_lines.append(s.horizontal(ImageOp("mac", *idxs, coeffs=coeffs), line))

    out = s.vertical(ImageOp("add"), row_lines)
    if traced.spatial_bits < out.bits:
        out = s.horizontal(ImageOp("trunc", Const(traced.spatial_bits)), out)
    for op in post:
        if op[0] == "shr":
            out = s.horizontal(ImageOp("div", Const(1 << op[1])), out)
        elif op[0] == "mulc":
            out = s.horizontal(ImageOp("mul", Const(op[1])), out)
        elif op[0] == "mulp":
            out = s.horizontal(ImageOp("mul", Param(op[1], bits=op[2], signed=op[3],
                                                    default=op[4])), out)
        elif op[0] == "addp":
            out = s.horizontal(ImageOp("addp", Param(op[1], bits=op[2], signed=op[3],
                                                     default=op[4])), out)
        elif op[0] == "clip":
            out = s.horizontal(ImageOp("clip", Const(op[1]), Const(op[2])), out)
        else:                                              # trunc (astype)
            out = s.horizontal(ImageOp("trunc", Const(op[1])), out)

    for ln in s._lines:
        ln.shape = shape

    # edge handling: padded spans map to replicate/zero borders (same-size out).
    pt, pb, pl, pr = traced.pad
    M = max(r for r, _ in taps)
    N = max(c for _, c in taps)
    if traced.mode != "none" and (pt or pb or pl or pr):
        if (pt or pb) and pt + pb != M:
            raise ValueError(
                f"row padding {(pt, pb)} must sum to the vertical span {M} "
                "for same-size output (pad 1 each side for a 3-tap filter)")
        if (pl or pr) and pl + pr != N:
            raise ValueError(
                f"column padding {(pl, pr)} must sum to the horizontal span {N} "
                "for same-size output")
        out.edge = (pt, pb, pl, pr, traced.mode, image.height, image.width)
    return s, out


# --------------------------------------------------------------------------- #
# Mux — np.where(scalar_bool_Param, A, B): a per-pixel 2:1 select between two
# pipeline branches, controlled by a 1-bit config register. Both branches share
# one window in codegen, so they are latency-aligned for free. (See verilog.py.)
# --------------------------------------------------------------------------- #

class Mux:
    """A 2:1 select between two same-shape Traced branches, driven by a scalar
    bool Param (a 1-bit register). This is a terminal node (mux at the output)."""
    def __init__(self, image, select, a, b):
        if not isinstance(select, Param):
            raise TypeError("np.where condition must be a scalar bool Param "
                            "(a 1-bit enable register)")
        if a.shape != b.shape:
            raise ValueError(f"mux branch shape mismatch: {a.shape} vs {b.shape}")
        self.image = image
        self.select = select
        self.a = a
        self.b = b
        self.shape = a.shape
        self.bits = max(a.bits, b.bits)
        self.signed = a.signed or b.signed

    # terminal output-declaration ops apply per branch (NumPy lets you chain
    # them after np.where; push them into both branches so tracing matches)
    def astype(self, dtype):
        return Mux(self.image, self.select, self.a.astype(dtype), self.b.astype(dtype))

    def clip(self, lo, hi):
        return Mux(self.image, self.select, self.a.clip(lo, hi), self.b.clip(lo, hi))


def _make_mux(cond, x, y):
    if not isinstance(x, Traced) or not isinstance(y, Traced):
        raise TypeError("np.where branches must both be traced expressions")
    return Mux(x.image, cond, x, y)


def to_ir(fn, image: Image2D, *params, out_bits=None):
    # a single Params namespace -> hand the model its Param-valued view
    if len(params) == 1 and isinstance(params[0], Params):
        params = (params[0].trace_view(),)
    traced = fn(Traced.image_input(image), *params)
    if isinstance(traced, Mux):
        return None, traced                              # generate() handles Mux
    if out_bits is not None:
        traced = traced.astype(out_bits)
    return lower(traced)
