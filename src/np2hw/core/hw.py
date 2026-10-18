# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The hardware level: combinational nodes on exact ranges, and named state.

A hardware value is one wire in one beat: its type has no axes, and its
width and sign follow from its exact range. State is NAMED, as storage is
(docs/one-graph.md, decision 7): a register is one `hw.reg` declaration --
its next value and its enable -- and every use is an `hw.regread` by name,
which is how a stream's cycle through its registers is said in a graph
that has none. A register's declaration in the graph's symbols gives its
range and its reset value, or None for a register that needs none.
"""
from __future__ import annotations

from .graph import (COMMUTATIVE, CONSTANT, PURE, READS_REGISTER, STATEFUL,
                    GraphError, Kind, Type, register)
from .model import EXACT

BOOL01 = (0, 1)


def _t(lo, hi):
    return (Type((), EXACT, lo, hi),)


def _port(g, ts, a):
    return _t(a["lo"], a["hi"])


def _const(g, ts, a):
    return _t(a["value"], a["value"])


def _param(g, ts, a):
    bits, signed = a["bits"], a["signed"]
    lo = -(1 << (bits - 1)) if signed else 0
    return _t(lo, (1 << (bits - (1 if signed else 0))) - 1)


def _bin(fn):
    return lambda g, ts, a: _t(*fn(ts[0], ts[1]))


def _un(fn):
    return lambda g, ts, a: _t(*fn(ts[0], a))


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


def _clip(x, a):
    lo, hi = max(x.lo, a["lo"]), min(x.hi, a["hi"])
    return (lo, hi) if lo <= hi else (a["lo"], a["lo"])


def _abs(x, a):
    lo = 0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi))
    return lo, max(abs(x.lo), abs(x.hi))


def _wrap(x, a):
    """A container's wrap: the low `bits`, read signed or unsigned. A value
    that fits is unchanged."""
    bits, signed = a["bits"], a["signed"]
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - (1 if signed else 0))) - 1
    return (x.lo, x.hi) if lo <= x.lo and x.hi <= hi else (lo, hi)


def _bits(x, a):
    """A field of a value's two's-complement representation, unsigned."""
    lo_bit, w = a["lo"], a["width"]
    top = (1 << w) - 1
    if x.lo >= 0:
        lo, hi = x.lo >> lo_bit, x.hi >> lo_bit
        if hi <= top:
            return lo, hi
    return 0, top


def _cat(g, ts, a):
    """Fields into a word, operand 0 in the low bits, each `field` bits:
    the exact range of the word they make."""
    f = a["field"]
    return _t(sum(t.lo << (f * i) for i, t in enumerate(ts)),
              sum(t.hi << (f * i) for i, t in enumerate(ts)))


def _verify_cat(g, n):
    f = n.attr("field")
    for i, o in enumerate(n.operands):
        if o.type.lo < 0 or o.type.hi >= 1 << f:
            raise GraphError(f"bit assembly: field {i} holds "
                             f"[{o.type.lo}, {o.type.hi}], not {f} unsigned "
                             "bits", n.loc)


def _mux(g, ts, a):
    """One of the choices by an index: operand 0 the index, the rest the
    choices in order -- the range the choices span."""
    choices = ts[1:]
    return _t(min(t.lo for t in choices), max(t.hi for t in choices))


def _verify_mux(g, n):
    idx, choices = n.operands[0], n.operands[1:]
    if idx.type.lo < 0 or idx.type.hi >= len(choices):
        raise GraphError(f"a mux of {len(choices)} indexed by "
                         f"[{idx.type.lo}, {idx.type.hi}]", n.loc)


def _sel(g, ts, a):
    c, x, y = ts
    return _t(min(x.lo, y.lo), max(x.hi, y.hi))


def _regread(g, ts, a):
    d = g.symbols.get(a["name"])
    if d is None:
        raise GraphError(f"a read of undeclared register {a['name']!r}")
    return _t(d["lo"], d["hi"])


def _verify_reg(g, n):
    d, en = n.operands
    decl = g.symbols.get(n.attr("name"))
    if decl is None:
        raise GraphError(f"register {n.attr('name')!r} is not declared", n.loc)
    if not (decl["lo"] <= d.type.lo and d.type.hi <= decl["hi"]):
        raise GraphError(f"register {n.attr('name')!r} holds "
                         f"[{decl['lo']}, {decl['hi']}], given "
                         f"[{d.type.lo}, {d.type.hi}]", n.loc)
    if not (0 <= en.type.lo and en.type.hi <= 1):
        raise GraphError(f"register {n.attr('name')!r}: enable wider than "
                         "one bit", n.loc)


_P = frozenset({PURE})
for _k in (
    Kind("hw.port", _P, 0, _port),
    Kind("hw.parameter", _P, 0, _port),
    Kind("hw.const", frozenset({PURE, CONSTANT}), 0, _const),
    Kind("hw.param", frozenset({PURE, READS_REGISTER}), 0, _param),
    Kind("hw.add", frozenset({PURE, COMMUTATIVE}), 2,
         _bin(lambda x, y: (x.lo + y.lo, x.hi + y.hi))),
    Kind("hw.sub", _P, 2, _bin(lambda x, y: (x.lo - y.hi, x.hi - y.lo))),
    Kind("hw.mul", frozenset({PURE, COMMUTATIVE}), 2, _bin(lambda x, y: (
        min(x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi),
        max(x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi)))),
    Kind("hw.neg", _P, 1, _un(lambda x, a: (-x.hi, -x.lo))),
    Kind("hw.shl", _P, 1, _un(lambda x, a: (x.lo << a["amount"],
                                           x.hi << a["amount"]))),
    Kind("hw.shr", _P, 1, _un(lambda x, a: (x.lo >> a["amount"],
                                           x.hi >> a["amount"]))),
    Kind("hw.clip", _P, 1, _un(_clip)),
    Kind("hw.abs", _P, 1, _un(_abs)),
    Kind("hw.lt", _P, 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.gt", _P, 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.le", _P, 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.ge", _P, 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.eq", frozenset({PURE, COMMUTATIVE}), 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.ne", frozenset({PURE, COMMUTATIVE}), 2, _bin(lambda x, y: BOOL01)),
    Kind("hw.and", frozenset({PURE, COMMUTATIVE}), 2, _bin(_and)),
    Kind("hw.or", frozenset({PURE, COMMUTATIVE}), 2, _bin(_or)),
    Kind("hw.not", _P, 1, _un(lambda x, a: BOOL01)),
    Kind("hw.sel", _P, 3, _sel),
    Kind("hw.wrap", _P, 1, _un(_wrap)),
    Kind("hw.bits", _P, 1, _un(_bits)),
    # bit assembly: fields into a word, operand 0 lowest
    Kind("hw.cat", _P, None, _cat, _verify_cat),
    # one of n choices by an index: the shape a lookup table selects in,
    # costed by the device's own mux arity (Device.mux_levels)
    Kind("hw.mux", _P, None, _mux, _verify_mux),
    # a constant table the device makes a memory: operand 0 its address,
    # `data` its contents; read through a register, a stage of its own
    Kind("hw.rom", _P, 1, lambda g, ts, a: _t(min(a["data"]), max(a["data"]))),
    # a table read the device makes a read of its own (Device.gather_is_read):
    # its value unchanged, a stage boundary after it when stages are cut
    Kind("hw.read", _P, 1, _un(lambda x, a: (x.lo, x.hi))),
    # the position of the pixel being taken along a driving variable: the
    # rewrite pass's reference, replaced by its counter in the control pass
    Kind("hw.position", _P, 0, lambda g, ts, a: _t(0, a["count"] - 1)),
    Kind("hw.regread", frozenset({STATEFUL}), 0, _regread),
    # a memory's read port, named: the word its registered read holds
    Kind("hw.memread", frozenset({STATEFUL}), 0,
         lambda g, ts, a: _t(a["lo"], a["hi"])),
    Kind("hw.reg", frozenset({STATEFUL}), 2, lambda g, ts, a: (), _verify_reg),
):
    register(_k)
