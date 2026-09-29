# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The model level: what NumPy computed, as graph kinds.

A model-level node's type is what NumPy holds: its axes as NumPy lays them
out, its container by NumPy 2's rules, and its range -- the exact range of
the arithmetic, WRAPPED into the container where NumPy wraps it. So the
model level is faithful: `uint8 - 128` has the range `[0, 255]`, because
that is what NumPy gives. Canonicalising makes each such wrap an explicit
node for the hardware.

Broadcasting is NumPy's, positional (decision 3): dimensions align from the
right, a dimension of size 1 (`None`) takes the other's, and two variables
paired on one position are UNIFIED -- which is how `x[:-2] + x[1:-1]`
comes to range over one row variable.
"""
from __future__ import annotations

import numpy as np

from .graph import (BOOL, COMMUTATIVE, CONSTANT, PURE, READS_REGISTER, WEAK,
                    GraphError, Kind, Type, container_range, register)


# --------------------------------------------------------------------------- #
# Containers and broadcasting
# --------------------------------------------------------------------------- #

def _np(c):
    kind, bits = c
    if kind == "b":
        return np.dtype(bool)
    size = next(b for b in (8, 16, 32, 64) if b >= bits)
    return np.dtype(f"{'u' if kind == 'u' else 'i'}{size // 8}")


#: The container of canonical arithmetic: exact integers, no wrap. After
#: canonicalising, a wrap is an explicit node, and nothing else wraps.
EXACT = ("x", 0)


def promote(*cs):
    """NumPy's result container, asked of NumPy -- a weak int stands as 0.
    Exact arithmetic stays exact."""
    if EXACT in cs:
        return EXACT
    args = [0 if c == WEAK else _np(c) for c in cs]
    if all(isinstance(a, int) for a in args):
        return WEAK
    d = np.result_type(*args)
    if d.kind == "b":
        return BOOL
    return ("u" if d.kind == "u" else "i", d.itemsize * 8)


def fit(lo: int, hi: int, c) -> tuple:
    """The range NumPy holds `[lo, hi]` in: unchanged when the container
    holds it, the container's whole range when it wraps."""
    if c in (WEAK, BOOL, EXACT) or c[0] not in ("u", "i"):
        return lo, hi
    clo, chi = container_range(c)
    return (lo, hi) if clo <= lo and hi <= chi else (clo, chi)


def broadcast(g, *axes_list) -> tuple:
    """NumPy's positional broadcast, unifying variables paired on one
    position."""
    n = max((len(a) for a in axes_list), default=0)
    padded = [(None,) * (n - len(a)) + tuple(a) for a in axes_list]
    out = []
    for dims in zip(*padded):
        v = None
        for d in dims:
            if d is None:
                continue
            v = d if v is None else g.vars.unify(v, d)
        out.append(None if v is None else g.vars.find(v))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

def _var(g, ts, a):
    name = a["name"]
    return (Type((g.vars.find(name),), ("i", 64), 0,
                 g.vars.count(name) - 1),)


def _source(g, ts, a):
    return (Type(tuple(None if v is None else g.vars.find(v)
                       for v in a["axes"]), a["container"], a["lo"], a["hi"]),)


def _const(g, ts, a):
    v = a["value"]
    return (Type((), a["container"], v, v),)


def _param(g, ts, a):
    bits, signed = a["bits"], a["signed"]
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - (1 if signed else 0))) - 1
    return (Type((), WEAK, lo, hi),)


def _arith(fn):
    def infer(g, ts, a):
        x, y = ts
        axes = broadcast(g, x.axes, y.axes)
        c = promote(x.container, y.container)
        lo, hi = fn(x, y)
        return (Type(axes, c, *fit(lo, hi, c)),)
    return infer


def _add(x, y):
    return x.lo + y.lo, x.hi + y.hi


def _sub(x, y):
    return x.lo - y.hi, x.hi - y.lo


def _mul(x, y):
    e = (x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi)
    return min(e), max(e)


def _unary(fn):
    def infer(g, ts, a):
        (x,) = ts
        lo, hi = fn(x, a)
        return (Type(x.axes, x.container, *fit(lo, hi, x.container)),)
    return infer


def _shr(x, a):
    return x.lo >> a["amount"], x.hi >> a["amount"]


def _floordiv(x, a):
    return x.lo // a["divisor"], x.hi // a["divisor"]


def _mod(x, a):
    k = a["divisor"]
    if 0 <= x.lo and x.hi < k:
        return x.lo, x.hi
    return 0, k - 1                     # floor semantics: in [0, k)


def _neg(x, a):
    return -x.hi, -x.lo


def _abs(x, a):
    lo = 0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi))
    return lo, max(abs(x.lo), abs(x.hi))


def _clip(x, a):
    lo, hi = max(x.lo, a["lo"]), min(x.hi, a["hi"])
    return (lo, hi) if lo <= hi else (a["lo"], a["lo"])


def _compare(g, ts, a):
    x, y = ts
    return (Type(broadcast(g, x.axes, y.axes), BOOL, 0, 1),)


def _pow2floor(lo):
    return -(1 << (-lo - 1).bit_length())


def _and(x, y):
    if x.lo >= 0 and y.lo >= 0:
        return 0, min(x.hi, y.hi)
    if x.lo >= 0 or y.lo >= 0:
        return 0, x.hi if x.lo >= 0 else y.hi
    return _pow2floor(min(x.lo, y.lo)), max(x.hi, y.hi)


def _or(x, y):
    hi = (1 << max(x.hi, y.hi, 0).bit_length()) - 1
    if x.lo >= 0 and y.lo >= 0:
        return max(x.lo, y.lo), hi
    return _pow2floor(min(x.lo, y.lo)), max(hi, -1)


def _where(g, ts, a):
    c, x, y = ts
    axes = broadcast(g, c.axes, x.axes, y.axes)
    cont = promote(x.container, y.container)
    return (Type(axes, cont, *fit(min(x.lo, y.lo), max(x.hi, y.hi), cont)),)


def _astype(g, ts, a):
    (x,) = ts
    c = a["container"]
    return (Type(x.axes, c, *fit(x.lo, x.hi, c)),)


def _expand(g, ts, a):
    """`x[:, None]` -- a reshape inserting dimensions of size 1: the value's
    variables placed at stated positions, nothing moved."""
    (x,) = ts
    target = tuple(None if v is None else g.vars.find(v) for v in a["axes"])
    have = tuple(g.vars.find(v) for v in x.axes if v is not None)
    if tuple(v for v in target if v is not None) != have:
        raise GraphError(f"an expand of axes {x.axes} to {a['axes']}")
    return (Type(target, x.container, x.lo, x.hi),)


def _pad(g, ts, a):
    """`np.pad`: each axis extended by an edge rule. The padded buffer's axes
    are new variables, their counts the source's plus the pad."""
    (x,) = ts
    axes = tuple(g.vars.find(v) for v in a["axes"])
    for old, new, (before, after) in zip(x.axes, axes, a["widths"]):
        if old is None or g.vars.count(new) != g.vars.count(old) + before + after:
            raise GraphError(f"a pad of axis {old} to {new} by "
                             f"{(before, after)}")
    lo, hi = x.lo, x.hi
    if a["mode"] == "zero":
        lo, hi = min(lo, 0), max(hi, 0)
    return (Type(axes, x.container, lo, hi),)


def _canvas(g, ts, a):
    """Planes written into a template through their indexes. Operands: the
    template, then per plane its value and one index per template axis.
    Each plane is cast into the template's container, as NumPy's
    assignment casts it."""
    t, rest = ts[0], ts[1:]
    per = 1 + len(t.axes)
    if len(rest) != a["planes"] * per:
        raise GraphError(f"a canvas of {a['planes']} planes with "
                         f"{len(rest)} operands")
    lo, hi = (0, 0) if a["zeroed"] else (None, None)
    for i in range(a["planes"]):
        v = rest[i * per]
        plo, phi = fit(v.lo, v.hi, t.container)
        lo = plo if lo is None else min(lo, plo)
        hi = phi if hi is None else max(hi, phi)
    if lo is None:
        raise GraphError("a canvas with no planes")
    return (Type(tuple(g.vars.find(v) for v in t.axes), t.container, lo, hi),)


def _stack(g, ts, a):
    """Values along a new axis: the variable `var`, at position `axis`."""
    axes = broadcast(g, *(t.axes for t in ts))
    pos = a["axis"] % (len(axes) + 1)
    axes = axes[:pos] + (g.vars.find(a["var"]),) + axes[pos:]
    cont = promote(*(t.container for t in ts))
    return (Type(axes, cont, *fit(min(t.lo for t in ts),
                                  max(t.hi for t in ts), cont)),)


def _wrap(g, ts, a):
    """What a NumPy container does to a value: kept when it fits, the low
    bits read as the container's otherwise. Canonical: the only wrap."""
    (x,) = ts
    c = a["container"]
    return (Type(x.axes, EXACT, *fit(x.lo, x.hi, c)),)


def _bits(g, ts, a):
    """A field of a non-negative value: bits [lo, lo + width)."""
    (x,) = ts
    if x.lo < 0:
        raise GraphError("a bit access of a value that can be negative")
    lo_bit, w = a["lo"], a["width"]
    top = (1 << w) - 1
    lo, hi = x.lo >> lo_bit, x.hi >> lo_bit
    if hi > top:                        # the field is cut: any value in it
        lo, hi = 0, top
    return (Type(x.axes, EXACT, lo, hi),)


def _select(g, ts, a):
    """One of several values, by keys: `sizes` gives each key's count, the
    values follow in row-major order of the keys' values. Pointwise."""
    n = len(a["sizes"])
    keys, vals = ts[:n], ts[n:]
    count = 1
    for k in a["sizes"]:
        count *= k
    if len(vals) != count:
        raise GraphError(f"a select over {a['sizes']} with {len(vals)} values")
    for k, size in zip(keys, a["sizes"]):
        if k.lo < 0 or k.hi >= size:
            raise GraphError(f"a select key in [{k.lo}, {k.hi}] for {size} "
                             "values")
    axes = broadcast(g, *(t.axes for t in ts))
    return (Type(axes, EXACT, min(v.lo for v in vals),
                 max(v.hi for v in vals)),)


def _access(g, ts, a):
    """A buffer read through one index per axis. The result spans the
    variables the indexes use, in the order the buffer's axes list them;
    its values are the buffer's."""
    buf, idx = ts[0], ts[1:]
    if len(idx) != len(buf.axes):
        raise GraphError(f"an access of a {len(buf.axes)}-axis buffer with "
                         f"{len(idx)} index(es)")
    axes, seen = [], set()
    for t in idx:
        for v in t.axes:
            if v is None:
                continue
            r = g.vars.find(v)
            if r not in seen:
                seen.add(r)
                axes.append(r)
    return (Type(tuple(axes), buf.container, buf.lo, buf.hi),)


_P = frozenset({PURE})
for _k in (
    Kind("model.var", _P, 0, _var),
    Kind("model.source", _P, 0, _source),
    Kind("model.const", frozenset({PURE, CONSTANT}), 0, _const),
    Kind("model.param", frozenset({PURE, READS_REGISTER}), 0, _param),
    Kind("model.add", frozenset({PURE, COMMUTATIVE}), 2, _arith(_add)),
    Kind("model.sub", _P, 2, _arith(_sub)),
    Kind("model.mul", frozenset({PURE, COMMUTATIVE}), 2, _arith(_mul)),
    Kind("model.and", frozenset({PURE, COMMUTATIVE}), 2, _arith(_and)),
    Kind("model.or", frozenset({PURE, COMMUTATIVE}), 2, _arith(_or)),
    Kind("model.neg", _P, 1, _unary(_neg)),
    Kind("model.abs", _P, 1, _unary(_abs)),
    Kind("model.shr", _P, 1, _unary(_shr)),
    Kind("model.floordiv", _P, 1, _unary(_floordiv)),
    Kind("model.mod", _P, 1, _unary(_mod)),
    Kind("model.clip", _P, 1, _unary(_clip)),
    Kind("model.lt", _P, 2, _compare),
    Kind("model.gt", _P, 2, _compare),
    Kind("model.le", _P, 2, _compare),
    Kind("model.ge", _P, 2, _compare),
    Kind("model.eq", frozenset({PURE, COMMUTATIVE}), 2, _compare),
    Kind("model.ne", frozenset({PURE, COMMUTATIVE}), 2, _compare),
    Kind("model.where", _P, 3, _where),
    Kind("model.astype", _P, 1, _astype),
    Kind("model.access", _P, None, _access),
    Kind("model.expand", _P, 1, _expand),
    Kind("model.pad", _P, 1, _pad),
    Kind("model.table", frozenset({PURE, READS_REGISTER}), 0, _source),
    Kind("model.canvas", _P, None, _canvas),
    Kind("model.stack", _P, None, _stack),
    Kind("model.wrap", _P, 1, _wrap),
    Kind("model.bits", _P, 1, _bits),
    Kind("model.select", _P, None, _select),
):
    register(_k)
