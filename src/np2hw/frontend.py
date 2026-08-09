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

Scope: single image; power-of-2 floor-div; slices with non-negative starts;
stencil min offset 0. Stride-2 slices with a constant or REGISTER-valued start
(`x[py::2, px::2]`) select one interleaved plane; assigning the planes back into
np.empty_like() covers the image and lowers to one full-rate datapath with the
coefficient selected by pixel position -- see PhaseCanvas.
"""
from __future__ import annotations

import numpy as np

from .ir import (ImageStreamer, Image2D, Indexer, ImageOp, Const, Param,
                 Params, PExpr, ExprLine, PhaseRef)


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


def _phase_start(value):
    """Normalise a strided slice's start into a constant or a PhaseRef."""
    if isinstance(value, Param):
        if value.bits != 1:
            raise ValueError(
                f"phase register {value.name!r} is {value.bits} bits; a slice "
                "phase selects one of two halves and must be 1 bit (split a "
                "wider register into per-axis bits at the call site)")
        return PhaseRef(value)
    if isinstance(value, PhaseRef):
        return value
    if isinstance(value, (int, np.integer)):
        if value not in (0, 1):
            raise NotImplementedError(
                f"strided slice start {value} must be 0, 1, or a phase register")
        return int(value)
    raise TypeError(f"unsupported strided slice start {value!r}")


def _phase_axis(s, dim):
    """Parse `start::step` where start may be a register. Returns (start, step)."""
    if not isinstance(s, slice):
        raise TypeError(f"a strided axis needs a slice, got {s!r}")
    step = int(s.step)
    if step != 2:
        raise NotImplementedError(
            f"stride {step} is out of scope; 2 is supported (the CFA case)")
    if s.stop is not None:
        raise NotImplementedError("a strided slice may not set a stop")
    if dim % step:
        raise ValueError(
            f"axis of {dim} does not divide by stride {step}; a phase-sliced "
            "plane must cover the axis exactly")
    return _phase_start(0 if s.start is None else s.start), step


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
                 spatial_bits=None, post=(), pad=(0, 0, 0, 0), mode="none",
                 spatial_signed=None, phase=None, stride=(1, 1), expr=None):
        self.image = image
        # Pointwise expression DAG, engaged the first time the model does
        # something a straight chain cannot say: forks the value, combines
        # two derived values, or gathers from a register array. None means
        # the ordinary chain semantics apply.
        self.expr = expr
        self.taps = dict(taps)               # {(row,col): weight}, weight may be < 0
        self.shape = tuple(shape)
        self.bits = bits
        self.signed = signed
        self.spatial_bits = bits if spatial_bits is None else spatial_bits
        # Signedness DECLARED for the spatial accumulator. `bits`/`signed` track
        # the value after the pointwise chain, which is not what the accumulator
        # is: `x.astype(np.int32) + p.offset` has a signed accumulator even though
        # a trailing .clip(0, 4095) makes the result unsigned. Codegen needs the
        # accumulator's declaration, so it is kept separately alongside
        # spatial_bits rather than recovered from the range downstream.
        self.spatial_signed = signed if spatial_signed is None else spatial_signed
        # Which half of each axis this value refers to, when it was taken with
        # a strided slice: (row_start, col_start), each a constant or a PhaseRef.
        # None means the value covers the whole image.
        self.phase = phase
        self.stride = tuple(stride)
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
                    spatial_bits=self.spatial_bits,
                    spatial_signed=self.spatial_signed, post=self.post,
                    pad=self.pad, mode=self.mode,
                    phase=self.phase, stride=self.stride, expr=self.expr)
        base.update(kw)
        return Traced(**base)

    def _spatial(self, taps, shape, bits, signed):
        return self._derive(taps=taps, shape=shape, bits=bits, signed=signed,
                            spatial_bits=bits, spatial_signed=signed, post=[])

    # -- np.pad / np.where interception (keeps spec = oracle) ---------------- #

    def __array_function__(self, func, types, args, kwargs):
        if func is np.pad:
            return self._pad(*args[1:], **kwargs)
        if func is np.where:
            return _make_mux(*args)
        if func in (np.empty_like, np.zeros_like):
            return self._canvas(zeroed=func is np.zeros_like)
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
        steps = [k.step for k in key if isinstance(k, slice)]
        if any(step not in (None, 1) for step in steps):
            return self._phase_slice(key)
        (dr, rows), (dc, cols) = _axis(key[0], self.shape[0]), _axis(key[1], self.shape[1])
        return self._spatial({(r + dr, c + dc): w for (r, c), w in self.taps.items()},
                             (rows, cols), self.bits, self.signed)

    def _phase_slice(self, key):
        """`img[py::2, px::2]` -- one interleaved plane of the image.

        The plane is not decimated in hardware. Nothing is dropped and no rate
        changes: the phase travels with the value and, when the planes are
        written back into a full-size result, becomes the select on a mux over
        the per-plane coefficients. That is why a CFA-phase-dependent operation
        costs a mux rather than four quarter-rate datapaths.

        The start may be a REGISTER, so which physical half a plane refers to is
        a two-bit write rather than a rebuild -- one bitstream, every CFA order.
        """
        if self.taps != {(0, 0): 1} or self.mode != "none":
            raise NotImplementedError(
                "a strided slice must be taken directly on the image (optionally "
                "after astype); slicing a stencil or a padded value is out of scope")
        if self.phase is not None:
            raise NotImplementedError("a phase-sliced value cannot be sliced again")

        rows, row_step = _phase_axis(key[0], self.shape[0])
        cols, col_step = _phase_axis(key[1], self.shape[1])
        return self._derive(shape=(self.shape[0] // row_step,
                                   self.shape[1] // col_step),
                            phase=(rows, cols), stride=(row_step, col_step))

    # -- np.empty_like / np.zeros_like -> a canvas to write planes back into -- #

    def _canvas(self, zeroed: bool):
        return PhaseCanvas(self, zeroed=zeroed)

    # -- pointwise expression DAG -------------------------------------------- #
    #
    # The chain says "one value, a line of ops". A lookup table cannot be
    # said that way: it derives a segment and a fraction from the same
    # input, reads two table entries, and recombines -- a small DAG. These
    # helpers engage that mode; once a Traced carries an expr, all further
    # arithmetic stays in expr form, and lower() emits an ExprLine.

    def _as_expr(self):
        if self.expr is not None:
            return self.expr
        if (self.taps != {(0, 0): 1} or self.post or self.mode != "none"
                or self.phase is not None):
            raise NotImplementedError(
                "expression-DAG models are pointwise in this version: fork, "
                "gather and recombine apply to the plain pixel value (after "
                "astype), not to a stencil, a padded value or a phase plane")
        lo = -(1 << (self.image.bits - 1)) if self.image.signed else 0
        hi = (1 << (self.image.bits - (1 if self.image.signed else 0))) - 1
        return PExpr("acc", (), lo, hi)

    def _expr_derive(self, node):
        negative = (-node.lo - 1).bit_length() if node.lo < 0 else 0
        bits = max(node.hi.bit_length(), negative) + (1 if node.lo < 0 else 0)
        return self._derive(expr=node, bits=max(1, bits), signed=node.lo < 0,
                            post=[])

    def _gather_from(self, parent):
        """`table[traced_index]`: the whole register array, muxed by data.

        The index's RANGE must sit inside the table, proven at trace time: a
        lookup that can fall off the end is refused here, where the message
        can say so, rather than reading garbage in hardware.
        """
        if len(parent.shape) != 1:
            raise NotImplementedError(
                f"gather from {parent.name!r}: only 1-D register arrays are "
                "gatherable in this version")
        index = self._as_expr()
        size = parent.shape[0]
        if index.lo < 0 or index.hi >= size:
            raise ValueError(
                f"gather from {parent.name!r}[{size}]: the index can reach "
                f"[{index.lo}, {index.hi}], which falls outside the table. "
                "Mask or clip the index first; hardware has no IndexError.")
        lo = -(1 << (parent.bits - 1)) if parent.signed else 0
        hi = (1 << (parent.bits - (1 if parent.signed else 0))) - 1
        return self._expr_derive(PExpr("gather", (parent, index), lo, hi))

    def _expr_binary(self, o, op):
        if isinstance(o, (int, np.integer)):
            other = PExpr("const", (int(o),), int(o), int(o))
        elif isinstance(o, Param):
            # A register as a LEAF of the expression: a coefficient in a
            # matrix multiply, a threshold, a per-channel gain. Its range is
            # its declared width, and it becomes an ordinary input port.
            lo = -(1 << (o.bits - 1)) if o.signed else 0
            hi = (1 << (o.bits - (1 if o.signed else 0))) - 1
            other = PExpr("param", (o,), lo, hi)
        elif isinstance(o, Traced):
            if o.image is not self.image:
                raise ValueError("expression combines values of two images")
            other = o._as_expr()
        else:
            return NotImplemented
        a, b = self._as_expr(), other
        if op == "add":
            lo, hi = a.lo + b.lo, a.hi + b.hi
        elif op == "sub":
            lo, hi = a.lo - b.hi, a.hi - b.lo
        else:                                            # mul
            corners = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
            lo, hi = min(corners), max(corners)
        return self._expr_derive(PExpr(op, (a, b), lo, hi))

    def __rshift__(self, o):
        if not isinstance(o, (int, np.integer)) or o < 0:
            return NotImplemented
        node = self._as_expr()
        return self._expr_derive(PExpr("shr", (node, int(o)),
                                       node.lo >> int(o), node.hi >> int(o)))

    def __and__(self, o):
        if not isinstance(o, (int, np.integer)):
            return NotImplemented
        mask = int(o)
        if mask < 0 or (mask & (mask + 1)) != 0:
            raise NotImplementedError(
                f"& {mask:#x}: only low-bit masks (2**k - 1) are supported -- "
                "that is a bit-slice in hardware; a general AND is not")
        node = self._as_expr()
        return self._expr_derive(PExpr("mask", (node, mask.bit_length()),
                                       0, mask))

    # -- arithmetic ---------------------------------------------------------- #

    def __add__(self, o):
        if self.expr is not None or (isinstance(o, Traced) and o.expr is not None):
            return self._expr_binary(o, "add")
        if isinstance(o, (int, np.integer)) and o == 0:
            return self                                   # identity (enables sum())
        if isinstance(o, Param):                          # bias/offset register
            rb = max(self.bits, o.bits) + 1
            rs = self.signed or o.signed
            return self._derive(bits=rb, signed=rs,
                                post=self.post + [("addp", o.name, o.bits, o.signed, o.default,
                                                   o.description)])
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
        if self.expr is not None or (isinstance(o, Traced) and o.expr is not None):
            return self._expr_binary(o, "sub")
        return self.__add__(o * -1)

    def __neg__(self):
        return self * -1

    def __mul__(self, o):
        if self.expr is not None or (isinstance(o, Traced) and o.expr is not None):
            return self._expr_binary(o, "mul")
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
            # A PHASE-SLICED value is the exception: its planes are pointwise
            # by contract and their coefficients are selected by the position
            # mux, so the multiply must peel into `post` where that mux can
            # reach it -- as a tap coefficient it would demand a spatial
            # datapath per plane, which is exactly what phase lowering exists
            # to avoid.
            if (not self.post and self.phase is None and len(self.taps) == 1
                    and next(iter(self.taps.values())) == 1):
                rc = next(iter(self.taps))
                rb, rs = _promote(self.bits, self.signed, o.bits, o.signed)
                return self._spatial({rc: o}, self.shape, rb, rs)
            return self._derive(post=self.post + [
                ("mulp", o.name, o.bits, o.signed, o.default, o.description)])
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, o):
        if self.expr is not None:
            k = int(o)
            if k <= 0 or (k & (k - 1)):
                raise NotImplementedError("expression // needs a power of two")
            return self >> (k.bit_length() - 1)
        if not isinstance(o, int):
            return NotImplemented
        shift = o.bit_length() - 1
        if (1 << shift) != o:
            raise ValueError(f"floor-div by {o}: power-of-2 only in v1")
        return self._derive(post=self.post + [("shr", shift)])

    def clip(self, lo, hi):
        """Saturate to [lo, hi] (NumPy ndarray.clip semantics) -> comparator+clamp.
        Output range is exactly [lo, hi], so this narrows the width."""
        if self.expr is not None:
            node = self.expr
            clo, chi = max(node.lo, int(lo)), min(node.hi, int(hi))
            if clo > chi:
                clo = chi = int(lo)
            return self._expr_derive(PExpr("clip", (node, int(lo), int(hi)),
                                           clo, chi))
        lo, hi = int(lo), int(hi)
        if lo < 0:
            b = 1 + max(max(1, hi.bit_length()), (-lo).bit_length()); sg = True
        else:
            b = max(1, hi.bit_length()); sg = False
        return self._derive(bits=b, signed=sg, post=self.post + [("clip", lo, hi)])

    def astype(self, dtype):
        if self.expr is not None:
            return self          # ranges track the value; containers are free
        b, sg = _dtbits(dtype)
        if self.post:
            return self._derive(bits=b, signed=sg, post=self.post + [("trunc", b)])
        if b > self.bits and self.signed == sg and self.natural() > self.bits:
            raise ValueError(
                "mixed-width: widening a value that would clip "
                f"(needs {self.natural()}b, dtype {self.bits}b). Widen operands first.")
        return self._derive(bits=b, signed=sg, spatial_bits=b, spatial_signed=sg,
                            post=[])


# --------------------------------------------------------------------------- #
# Lowering to the line IR
# --------------------------------------------------------------------------- #

def lower(traced: Traced):
    if getattr(traced, "expr", None) is not None:
        # A pointwise expression DAG: collect the gathered register arrays in
        # first-use order and hand the whole thing to the expression emitter.
        parents, seen = [], set()

        def walk(node):
            if node.op == "gather":
                parent = node.args[0]
                if ("gather", parent.name) not in seen:
                    seen.add(("gather", parent.name))
                    parents.append(("gather", parent))
                walk(node.args[1])
            elif node.op == "param":
                leaf = node.args[0]
                if ("param", leaf.name) not in seen:
                    seen.add(("param", leaf.name))
                    parents.append(("param", leaf))
            elif node.op in ("add", "sub", "mul"):
                walk(node.args[0]); walk(node.args[1])
            elif node.op in ("shr", "mask", "clip"):
                walk(node.args[0])

        walk(traced.expr)
        root = traced.expr
        return None, ExprLine(image=traced.image, root=root,
                              params=tuple(parents),
                              out_bits=max(1, root.hi.bit_length()
                                           if root.lo >= 0 else traced.bits),
                              out_signed=root.lo < 0)

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
    # Hand the accumulator's DECLARED dtype to codegen. Without this the Verilog
    # emitter re-derives the accumulator from the input image's value range and
    # disagrees with the NumPy the model was written in -- e.g. a 12-bit unsigned
    # pixel plus a signed offset register would be emitted as `wire signed [11:0]`,
    # which cannot represent 4095. See Line2D.declared.
    out.declared = (traced.spatial_bits, traced.spatial_signed)
    if traced.spatial_bits < out.bits:
        out = s.horizontal(ImageOp("trunc", Const(traced.spatial_bits)), out)
    for op in post:
        if op[0] == "shr":
            out = s.horizontal(ImageOp("div", Const(1 << op[1])), out)
        elif op[0] == "mulc":
            out = s.horizontal(ImageOp("mul", Const(op[1])), out)
        elif op[0] == "mulp":
            out = s.horizontal(ImageOp("mul", Param(op[1], bits=op[2], signed=op[3],
                                                    default=op[4],
                                                    description=op[5])), out)
        elif op[0] == "addp":
            out = s.horizontal(ImageOp("addp", Param(op[1], bits=op[2], signed=op[3],
                                                     default=op[4],
                                                     description=op[5])), out)
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


class PhaseCanvas:
    """A full-size result assembled from phase-sliced planes.

    Produced by `np.empty_like(x)` / `np.zeros_like(x)` on a traced value, and
    filled in by assigning to strided slices::

        out = np.empty_like(x)
        out[p.py::2,   p.px::2]   = f(x[p.py::2,   p.px::2],   p.coeff[0, 0])
        out[p.py::2,   1-p.px::2] = f(x[p.py::2,   1-p.px::2], p.coeff[0, 1])
        out[1-p.py::2, p.px::2]   = f(x[1-p.py::2, p.px::2],   p.coeff[1, 0])
        out[1-p.py::2, 1-p.px::2] = f(x[1-p.py::2, 1-p.px::2], p.coeff[1, 1])

    That is ordinary NumPy: run it on an array with integer phases and it does
    exactly what it says. Traced, it lowers to ONE full-rate datapath with the
    coefficient selected by the pixel's position -- not four quarter-rate paths
    -- because the planes are disjoint and together cover every pixel.

    The planes must PARTITION the image: every phase written exactly once. A
    missing plane leaves pixels undefined and an overlapping one leaves them
    ambiguous, and neither is something to discover in silicon.
    """

    def __init__(self, source, zeroed: bool):
        self.image = source.image
        self.shape = source.shape
        self.zeroed = zeroed
        self.bits = source.bits
        self.signed = source.signed
        self.branches = []          # [((row_desc, col_desc), stride, Traced)]

    # -- assembly -------------------------------------------------------------- #

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "assign to a two-axis strided slice, e.g. out[py::2, px::2]")
        rows, row_step = _phase_axis(key[0], self.shape[0])
        cols, col_step = _phase_axis(key[1], self.shape[1])
        if not isinstance(value, Traced):
            raise TypeError(
                f"a phase plane must be a traced expression, got {type(value).__name__}")
        if value.phase is None:
            raise ValueError(
                "the value written into a phase plane must itself be phase-sliced: "
                "out[py::2, px::2] = f(x[py::2, px::2], ...)")
        if value.phase != (rows, cols) or value.stride != (row_step, col_step):
            raise ValueError(
                f"phase mismatch: writing plane {(rows, cols)} but the value was "
                f"taken from {value.phase}. Each plane must be computed from the "
                "SAME phase it is written to, or the output is a shuffle rather "
                "than a per-phase operation")
        self.branches.append(((rows, cols), (row_step, col_step), value))

    # terminal declarations apply per plane, as they do for a mux
    def astype(self, dtype):
        return self._map(lambda b: b.astype(dtype))

    def clip(self, lo, hi):
        return self._map(lambda b: b.clip(lo, hi))

    def _map(self, fn):
        out = PhaseCanvas.__new__(PhaseCanvas)
        out.image, out.shape, out.zeroed = self.image, self.shape, self.zeroed
        out.branches = [(phase, stride, fn(value))
                        for phase, stride, value in self.branches]
        out.bits = max(v.bits for _, _, v in out.branches)
        out.signed = any(v.signed for _, _, v in out.branches)
        return out

    # -- validation ------------------------------------------------------------ #

    @staticmethod
    def _key(desc):
        """A comparable identity for one axis phase."""
        if isinstance(desc, PhaseRef):
            return ("reg", desc.param.name, desc.invert)
        return ("const", int(desc))

    @staticmethod
    def _axis_pair(descs, axis):
        """The two complementary halves of one axis, or a clear error."""
        keys = {PhaseCanvas._key(d): d for d in descs}
        if len(keys) != 2:
            raise ValueError(
                f"the {axis} phases {sorted(str(d) for d in descs)} do not split the "
                f"axis in two; a stride-2 partition needs exactly two complementary "
                f"halves")
        (ka, a), (kb, b) = list(keys.items())
        if ka[0] != kb[0]:
            raise ValueError(
                f"the {axis} phases mix a constant with a register ({a}, {b}); "
                "both halves must be described the same way")
        if ka[0] == "reg":
            if ka[1] != kb[1] or ka[2] == kb[2]:
                raise ValueError(
                    f"the {axis} phases {a} and {b} are not complementary halves of "
                    "one register; write `p.x::2` and `1 - p.x::2`")
        elif {ka[1], kb[1]} != {0, 1}:
            raise ValueError(f"the {axis} phases must be 0 and 1, got {a} and {b}")
        return a, b

    def validate(self):
        """Check the planes partition the image, and return the axis phases."""
        if not self.branches:
            raise ValueError(
                "no phase planes were written into this result; assign to "
                "out[py::2, px::2] and the other three phases")
        strides = {stride for _, stride, _ in self.branches}
        if len(strides) != 1:
            raise ValueError(f"mixed strides {sorted(strides)} in one result")

        rows = self._axis_pair([p[0] for p, _, _ in self.branches], "row")
        cols = self._axis_pair([p[1] for p, _, _ in self.branches], "column")

        seen = {(self._key(p[0]), self._key(p[1])) for p, _, _ in self.branches}
        if len(seen) != len(self.branches):
            raise ValueError(
                "the same phase plane was written twice; each of the four planes "
                "must be written exactly once")
        expected = {(self._key(r), self._key(c)) for r in rows for c in cols}
        if seen != expected:
            missing = expected - seen
            raise ValueError(
                f"the phase planes do not cover the image: "
                f"{len(missing)} of {len(expected)} missing. Every pixel must be "
                "written exactly once, or the output is undefined where it is not")
        return rows, cols

    def plane(self, row_desc, col_desc):
        """The branch written to one phase."""
        want = (self._key(row_desc), self._key(col_desc))
        for phase, _, value in self.branches:
            if (self._key(phase[0]), self._key(phase[1])) == want:
                return value
        raise KeyError(f"no plane for phase {(row_desc, col_desc)}")


def _make_mux(cond, x, y):
    if not isinstance(x, Traced) or not isinstance(y, Traced):
        raise TypeError("np.where branches must both be traced expressions")
    return Mux(x.image, cond, x, y)


def to_ir(fn, image: Image2D, *params, out_bits=None):
    # a single Params namespace -> hand the model its Param-valued view
    if len(params) == 1 and isinstance(params[0], Params):
        params = (params[0].trace_view(),)
    traced = fn(Traced.image_input(image), *params)
    if isinstance(traced, (Mux, PhaseCanvas)):
        return None, traced                  # generate() handles these terminals
    if out_bits is not None:
        traced = traced.astype(out_bits)
    return lower(traced)
