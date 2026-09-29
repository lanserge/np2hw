# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The one graph: every traced value is a node of it. See docs/one-graph.md.

A node has the same six fields whatever it is -- kind, operands, attrs,
results, regions, loc -- so walking, hashing, printing and verifying work on
any kind, including one registered later. What a kind MEANS lives in its
descriptor (`Kind`): its traits, and the hooks that infer its result ranges,
fold it and verify it. No pass here names a kind.

Nodes are immutable and hash-consed: building a node that already exists
returns the existing one, so shared logic is shared by construction. The one
exception is `loc`, which is provenance, not identity: when a second line of
the model builds a node that already exists, the node keeps both lines.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

_NP2HW = os.path.dirname(os.path.realpath(__file__))


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class FileLine:
    file: str
    line: int

    def __str__(self):
        return f'"{os.path.basename(self.file)}":{self.line}'


@dataclass(frozen=True, slots=True)
class Named:
    """A name -- a register, a model variable -- around a location."""
    name: str
    child: "Loc"

    def __str__(self):
        return f'"{self.name}"({self.child})'


@dataclass(frozen=True, slots=True)
class Fused:
    """Several locations one node came from: made when a merge unifies them."""
    locs: tuple

    def __str__(self):
        return "fused[" + ", ".join(str(l) for l in self.locs) + "]"


@dataclass(frozen=True, slots=True)
class Unknown:
    def __str__(self):
        return "?"


UNKNOWN = Unknown()
Loc = FileLine | Named | Fused | Unknown


_INSIDE: dict[str, bool] = {}


def here() -> Loc:
    """The innermost frame outside np2hw: the line of the model that is
    building a node."""
    f = sys._getframe(1)
    inside = _INSIDE
    while f is not None:
        name = f.f_code.co_filename
        flag = inside.get(name)
        if flag is None:              # once per file: paths may hold '..'
            flag = inside[name] = os.path.realpath(name).startswith(
                _NP2HW + os.sep)
        if not flag:
            return FileLine(name, f.f_lineno)
        f = f.f_back
    return UNKNOWN


def fuse(a: Loc, b: Loc) -> Loc:
    """Both locations, flattened and deduplicated; order of first sight."""
    if a == b or isinstance(b, Unknown):
        return a
    if isinstance(a, Unknown):
        return b
    parts = []
    for loc in (a, b):
        for p in (loc.locs if isinstance(loc, Fused) else (loc,)):
            if p not in parts:
                parts.append(p)
    return Fused(tuple(parts))


# --------------------------------------------------------------------------- #
# Types: a result's facts
# --------------------------------------------------------------------------- #

class Range(tuple):
    """The exact integer interval a value can take. Width and sign follow.

    A tuple underneath: built once per node, so it must cost what a tuple
    costs."""
    __slots__ = ()

    def __new__(cls, lo: int, hi: int):
        if lo > hi:
            raise ValueError(f"empty range [{lo}, {hi}]")
        return tuple.__new__(cls, (lo, hi))

    @property
    def lo(self) -> int:
        return self[0]

    @property
    def hi(self) -> int:
        return self[1]

    @property
    def signed(self) -> bool:
        return self[0] < 0

    @property
    def bits(self) -> int:
        lo, hi = self
        negative = (-lo - 1).bit_length() if lo < 0 else 0
        return max(1, max(hi.bit_length(), negative) + (1 if lo < 0 else 0))

    def __str__(self):
        return f"[{self[0]}, {self[1]}]"

    def __repr__(self):
        return f"Range({self[0]}, {self[1]})"


# --------------------------------------------------------------------------- #
# Kinds
# --------------------------------------------------------------------------- #

PURE = "pure"                     # no state, no effect: a function of operands
STATEFUL = "stateful"             # holds a value from one beat to the next
COMMUTATIVE = "commutative"       # operand order does not matter
CONSTANT = "constant"             # a value fixed at generation
READS_REGISTER = "reads_register" # a value that changes only at a commit


@dataclass(frozen=True)
class Kind:
    """What a node kind means. Registered once; passes consult it.

    infer(operand_types, attrs, symbols) -> tuple of result types
    fold(builder, operands, attrs) -> Value | int | None
        a simpler existing value, a constant, or None for "no change";
        it may not build nodes other than through `builder.const`
    verify(node, symbols) -> None, raising GraphError on a malformed node
    """
    name: str
    traits: frozenset = frozenset()
    arity: int | None = None                  # None: any
    infer: Callable | None = None
    fold: Callable | None = None
    verify: Callable | None = None

    @property
    def level(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def commutative(self) -> bool:
        return COMMUTATIVE in self.traits


_KINDS: dict[str, Kind] = {}


def register(kind: Kind) -> Kind:
    if kind.name in _KINDS:
        raise ValueError(f"kind {kind.name!r} registered twice")
    if "." not in kind.name:
        raise ValueError(f"kind {kind.name!r} needs a level prefix")
    _KINDS[kind.name] = kind
    return kind


def kind(name: str) -> Kind:
    try:
        return _KINDS[name]
    except KeyError:
        raise GraphError(f"unknown node kind {name!r}") from None


class GraphError(Exception):
    """A malformed graph, or an operation it cannot express. Carries the
    location of the node at fault, so the message names the model's line."""

    def __init__(self, message: str, loc: Loc = UNKNOWN):
        super().__init__(message if isinstance(loc, Unknown)
                         else f"{message} (at {loc})")
        self.loc = loc


# --------------------------------------------------------------------------- #
# Nodes and values
# --------------------------------------------------------------------------- #

_PLAIN = (int, str, bool, type(None))


def _plain(v) -> bool:
    return isinstance(v, _PLAIN) or (
        isinstance(v, tuple) and all(_plain(x) for x in v))


class Node:
    """One operation. Build through a `Builder`, never directly."""
    __slots__ = ("kind", "operands", "attrs", "results", "regions", "loc",
                 "seq", "_values", "__weakref__")

    def __init__(self, kind_, operands, attrs, results, regions, loc, seq):
        self.kind = kind_
        self.operands = operands              # tuple[Value]
        self.attrs = attrs                    # tuple of (key, value), sorted
        self.results = results                # tuple of result types
        self.regions = regions                # tuple, empty today
        self.loc = loc
        self.seq = seq                        # creation order: determinism
        self._values = None                   # made on first use

    def attr(self, key, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    @property
    def value(self) -> "Value":
        """The single result of a single-result node."""
        vs = self._values
        if vs is None:
            if len(self.results) != 1:
                raise GraphError(f"{self.kind} has {len(self.results)} "
                                 "results", self.loc)
            vs = self._values = (Value(self, 0),)
        return vs[0]

    def result(self, index: int) -> "Value":
        if self._values is None:
            self._values = tuple(Value(self, i)
                                 for i in range(len(self.results)))
        return self._values[index]

    def __repr__(self):
        return f"<{self.kind} #{self.seq}>"


class Value:
    """One result of one node. Operands are values, not nodes, so a node with
    several results needs no redesign. Its type is stored, not looked up:
    every construction reads its operands' types."""
    __slots__ = ("node", "index", "type")

    def __init__(self, node: Node, index: int):
        self.node = node
        self.index = index
        self.type = node.results[index]

    def __repr__(self):
        return f"{self.node!r}#{self.index}"


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

class Builder:
    """Makes nodes: infers their result types, folds them, hash-conses them.

    `fold=False` builds exactly what is asked, which a byte-identical port of
    existing code needs; folding is otherwise on.
    """

    def __init__(self, symbols: dict | None = None, fold: bool = True,
                 locations: bool = True, intern: bool = True):
        self.symbols = symbols if symbols is not None else {}
        self.fold = fold
        self.locations = locations
        # intern=False builds a node per request, in the order and with the
        # operand order asked: a faithful record of what a model did.
        self.intern = intern
        self._table: dict = {}
        self._seq = 0

    def op(self, kind_name: str, *operands: Value, loc: Loc | None = None,
           **attrs) -> Value:
        return self.node(kind_name, operands, attrs, loc=loc).value

    def node(self, kind_name: str, operands: Iterable[Value] = (),
             attrs: dict | None = None, regions: tuple = (),
             loc: Loc | None = None) -> Node:
        # The hot path: every traced operation comes through here, so it
        # does only what identity needs. Attribute plain-ness and arity are
        # checked by `verify`, not on every construction.
        k = _KINDS.get(kind_name) or kind(kind_name)
        operands = tuple(operands)
        attrs = attrs or {}
        if self.locations and loc is None:
            loc = here()
        elif loc is None:
            loc = UNKNOWN
        if self.fold and k.fold is not None:
            folded = k.fold(self, operands, attrs)
            if folded is not None:
                if isinstance(folded, Value):
                    if len(folded.node.results) != 1:
                        _refuse_multi(k, loc)
                    return folded.node
                if not isinstance(folded, bool):
                    return self.const(folded, loc=loc).node
        if self.intern and k.commutative and len(operands) == 2:
            x, y = operands
            if (y.node.seq, y.index) < (x.node.seq, x.index):
                operands = (y, x)
        if len(attrs) > 1:
            akey = tuple(sorted(attrs.items()))
        else:
            akey = tuple(attrs.items())
        key = (k.name, operands, akey, regions)
        hit = self._table.get(key) if self.intern else None
        if hit is not None:
            if loc is not UNKNOWN:
                hit.loc = fuse(hit.loc, loc)  # provenance, not identity
            return hit
        types = k.infer(operands and [v.type for v in operands], attrs,
                        self.symbols) if k.infer else ()
        n = Node(k.name, operands, akey, tuple(types), regions, loc,
                 self._seq)
        self._seq += 1
        if self.intern:
            self._table[key] = n
        return n

    # -- direct constructors for the hot kinds ------------------------------ #
    #
    # The same uniform nodes the generic path builds -- same kinds, same
    # hash-cons table, same range rules -- without its indirection, because
    # a tracer builds thousands of them. A kind without one here is no less
    # a kind; `node()` builds it.

    def _intern(self, name, operands, akey, result, loc):
        key = (name, operands, akey)
        hit = self._table.get(key)
        if hit is not None:
            if loc is not UNKNOWN:
                hit.loc = fuse(hit.loc, loc)
            return hit.value
        n = Node(name, operands, akey, (result,), (), loc, self._seq)
        self._seq += 1
        self._table[key] = n
        return n.value

    def _loc(self, loc):
        if loc is not None:
            return loc
        return here() if self.locations else UNKNOWN

    def add(self, x: Value, y: Value, loc: Loc | None = None) -> Value:
        loc = self._loc(loc)
        if self.fold:
            f = _fold_add(self, (x, y), None)
            if f is not None:
                return f if isinstance(f, Value) else self.const(f, loc)
        xn, yn = x.node, y.node
        if (yn.seq, y.index) < (xn.seq, x.index):
            x, y = y, x
        (a, b), (c, d) = x.type, y.type
        return self._intern("model.add", (x, y), (), _R(Range, (a + c, b + d)), loc)

    def sub(self, x: Value, y: Value, loc: Loc | None = None) -> Value:
        loc = self._loc(loc)
        if self.fold:
            f = _fold_sub(self, (x, y), None)
            if f is not None:
                return f if isinstance(f, Value) else self.const(f, loc)
        (a, b), (c, d) = x.type, y.type
        return self._intern("model.sub", (x, y), (), _R(Range, (a - d, b - c)), loc)

    def mul(self, x: Value, y: Value, loc: Loc | None = None) -> Value:
        loc = self._loc(loc)
        if self.fold:
            f = _fold_mul(self, (x, y), None)
            if f is not None:
                return f if isinstance(f, Value) else self.const(f, loc)
        if (y.node.seq, y.index) < (x.node.seq, x.index):
            x, y = y, x
        (a, b), (c, d) = x.type, y.type
        e = (a * c, a * d, b * c, b * d)
        return self._intern("model.mul", (x, y), (), _R(Range, (min(e), max(e))),
                            loc)

    def shr(self, x: Value, amount: int, loc: Loc | None = None) -> Value:
        loc = self._loc(loc)
        if self.fold:
            f = _fold_shr(self, (x,), {"amount": amount})
            if f is not None:
                return f if isinstance(f, Value) else self.const(f, loc)
        a, b = x.type
        return self._intern("model.shr", (x,), (("amount", amount),),
                            _R(Range, (a >> amount, b >> amount)), loc)

    def sel(self, c: Value, x: Value, y: Value,
            loc: Loc | None = None) -> Value:
        loc = self._loc(loc)
        (a, b), (e, f) = x.type, y.type
        return self._intern("model.sel", (c, x, y), (),
                            _R(Range, (min(a, e), max(b, f))), loc)

    def lt(self, x: Value, y: Value, loc: Loc | None = None) -> Value:
        return self._intern("model.lt", (x, y), (), _BOOL, self._loc(loc))

    def const(self, value: int, loc: Loc | None = None) -> Value:
        value = int(value)
        return self._intern("model.const", (), (("value", value),),
                            _R(Range, (value, value)), self._loc(loc))


def _refuse_multi(k, loc):
    raise GraphError(f"{k.name}: fold returned one result of a "
                     "multi-result node", loc)


# --------------------------------------------------------------------------- #
# Walking, verifying, printing
# --------------------------------------------------------------------------- #

def walk(roots: Iterable[Value]) -> Iterator[Node]:
    """Every node reachable from `roots`, operands before users, each once.
    Iterative: a deep datapath (a CORDIC is hundreds of levels) must not
    meet Python's recursion limit."""
    seen: set[int] = set()
    for root in roots:
        stack = [(root.node, False)]
        while stack:
            n, expanded = stack.pop()
            if id(n) in seen:
                continue
            if expanded:
                seen.add(id(n))
                yield n
                continue
            stack.append((n, True))
            for v in reversed(n.operands):
                if id(v.node) not in seen:
                    stack.append((v.node, False))


def verify(roots: Iterable[Value], symbols: dict | None = None,
           levels: set[str] | None = None) -> None:
    """Raise GraphError at the first malformed node. `levels`, when given,
    is the legality rule: every node's kind must be at one of them."""
    symbols = symbols or {}
    for n in walk(roots):
        k = kind(n.kind)
        if levels is not None and k.level not in levels:
            raise GraphError(f"{n.kind} is not legal at level "
                             f"{sorted(levels)}", n.loc)
        if k.arity is not None and len(n.operands) != k.arity:
            raise GraphError(f"{n.kind} takes {k.arity} operand(s)", n.loc)
        for key, val in n.attrs:
            if not _plain(val):
                raise GraphError(f"{n.kind}: attribute {key}={val!r} is not "
                                 "plain data (int, str, bool, None, tuple)",
                                 n.loc)
        for v in n.operands:
            if not 0 <= v.index < len(v.node.results):
                raise GraphError(f"{n.kind}: operand {v!r} has no such "
                                 "result", n.loc)
        if k.infer is not None:
            expected = tuple(k.infer([v.type for v in n.operands],
                                     dict(n.attrs), symbols))
            if expected != n.results:
                raise GraphError(
                    f"{n.kind}: result types {[str(t) for t in n.results]} "
                    f"disagree with the kind's rule "
                    f"{[str(t) for t in expected]}", n.loc)
        if k.verify is not None:
            k.verify(n, symbols)


def dump(roots: Iterable[Value], locations: bool = True) -> str:
    """Deterministic text of the graph: values numbered in walk order, so two
    dumps of the same graph are the same text and two versions diff."""
    roots = list(roots)
    names: dict[int, str] = {}
    lines = []
    for i, n in enumerate(walk(roots)):
        names[id(n)] = f"%{i}"
        lhs = f"%{i}" if len(n.results) == 1 else f"%{i}:{len(n.results)}"

        def ref(v):
            base = names[id(v.node)]
            return base if len(v.node.results) == 1 else f"{base}#{v.index}"

        ops = ", ".join(ref(v) for v in n.operands)
        attrs = (" {" + ", ".join(f"{k}={v!r}" for k, v in n.attrs) + "}"
                 if n.attrs else "")
        types = ", ".join(str(t) for t in n.results)
        loc = f" loc({n.loc})" if locations else ""
        lines.append(f"{lhs} = {n.kind}({ops}){attrs} : {types}{loc}")
    outs = ", ".join(names[id(r.node)] for r in roots)
    lines.append(f"return {outs}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The model level: kinds as traced from NumPy
# --------------------------------------------------------------------------- #

def _declared(ts, a, s):
    return (Range(a["lo"], a["hi"]),)


def _extent(ts, a, s):
    return (Range(0, a["extent"] - 1),)


def _add(ts, a, s):
    x, y = ts
    return (Range(x.lo + y.lo, x.hi + y.hi),)


def _sub(ts, a, s):
    x, y = ts
    return (Range(x.lo - y.hi, x.hi - y.lo),)


def _mul(ts, a, s):
    x, y = ts
    c = (x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi)
    return (Range(min(c), max(c)),)


def _shr(ts, a, s):
    (x,), k = ts, a["amount"]
    return (Range(x.lo >> k, x.hi >> k),)


def _mask(ts, a, s):
    return (Range(0, (1 << a["bits"]) - 1),)


def _clip(ts, a, s):
    (x,) = ts
    lo, hi = max(x.lo, a["lo"]), min(x.hi, a["hi"])
    if lo > hi:                       # the value always lands on one bound
        lo = hi = a["lo"]
    return (Range(lo, hi),)


def _abs(ts, a, s):
    (x,) = ts
    lo = 0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi))
    return (Range(lo, max(abs(x.lo), abs(x.hi))),)


_BOOL = Range(0, 1)
_R = tuple.__new__             # Range without the check: rules that cannot empty it


def _bool(ts, a, s):
    return (_BOOL,)


def _sel(ts, a, s):
    _, x, y = ts
    return (Range(min(x.lo, y.lo), max(x.hi, y.hi)),)


def _register(ts, a, s):
    bits, signed = a["bits"], a["signed"]
    if signed:
        return (Range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1),)
    return (Range(0, (1 << bits) - 1),)


def _table(ts, a, s):
    """A table read takes the table's declared range: the table is a symbol,
    stated once, and the read refers to it by name."""
    try:
        decl = s[a["table"]]
    except KeyError:
        raise GraphError(f"read of undeclared table {a['table']!r}") from None
    return (Range(decl["lo"], decl["hi"]),)


def _lin(ts, a, s):
    ws = a["weights"]
    lo = sum(w * (t.lo if w >= 0 else t.hi) for w, t in zip(ws, ts))
    hi = sum(w * (t.hi if w >= 0 else t.lo) for w, t in zip(ws, ts))
    return (Range(lo, hi),)


def _container(bits, signed):
    if signed:
        return Range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return Range(0, (1 << bits) - 1)


def _declare(ts, a, s):
    """A declared container (`astype` before the spatial sum): the value is
    unchanged when it fits, and wraps into the container when it does not."""
    (x,) = ts
    c = _container(a["bits"], a["signed"])
    return (x if c.lo <= x.lo and x.hi <= c.hi else c,)


def _trunc(ts, a, s):
    """A narrowing `astype` on the trailing chain: keeps the low `bits`."""
    (x,) = ts
    signed = x.lo < 0
    c = _container(a["bits"], signed)
    return (x if c.lo <= x.lo and x.hi <= c.hi else c,)


def _same(ts, a, s):
    return (ts[0],)


def _astype(ts, a, s):
    """astype: the value is kept when the container holds it; a narrower
    container keeps the low bits, which can land anywhere in it."""
    (x,) = ts
    bits, signed = a["bits"], a["signed"]
    c = _container(bits, signed)
    return (x if c.lo <= x.lo and x.hi <= c.hi else c,)


def _floordiv(ts, a, s):
    (x,), k = ts, a["divisor"]
    return (Range(x.lo // k, x.hi // k),)


def _neg(ts, a, s):
    (x,) = ts
    return (Range(-x.hi, -x.lo),)


def _floor_pow2(lo: int) -> int:
    """The most negative value of the smallest two's-complement container
    holding `lo` (lo < 0)."""
    return -(1 << (-lo - 1).bit_length())


def _and(ts, a, s):
    """Bitwise AND, two's complement. A never-negative operand bounds the
    result to [0, it]; two negative operands give a negative result no
    larger than either and no smaller than their container allows."""
    x, y = ts
    if x.lo >= 0 and y.lo >= 0:
        return (Range(0, min(x.hi, y.hi)),)
    if x.lo >= 0 or y.lo >= 0:
        return (Range(0, x.hi if x.lo >= 0 else y.hi),)
    return (Range(_floor_pow2(min(x.lo, y.lo)), max(x.hi, y.hi)),)


def _or(ts, a, s):
    """Bitwise OR, two's complement: never below the larger operand's low
    bound when both are never negative; bounded by the containers."""
    x, y = ts
    top = max(x.hi, y.hi, 0)
    hi = (1 << top.bit_length()) - 1
    if x.lo >= 0 and y.lo >= 0:
        return (Range(max(x.lo, y.lo), hi),)
    return (Range(_floor_pow2(min(x.lo, y.lo)), max(hi, -1)),)


def _coords(ts, a, s):
    """Two results: the row and the column of a raster, from zero."""
    return (Range(0, a["rows"] - 1), Range(0, a["cols"] - 1))


def _hull(ts, a, s):
    return (Range(min(x.lo for x in ts), max(x.hi for x in ts)),)


def _canvas(ts, a, s):
    """The planes together: the template only gives the shape."""
    planes = ts[1:]
    if not planes:
        return (Range(0, 0),)
    return _hull(planes, a, s)


def _verify_lin(n, s):
    if len(n.attr("weights", ())) != len(n.operands):
        raise GraphError("model.lin: one weight per operand", n.loc)


def _fold_add(b, ops, a):
    x, y = ops
    cx, cy = _const_of(x), _const_of(y)
    if cx is not None and cy is not None:
        return cx + cy
    if cx == 0:
        return y
    if cy == 0:
        return x
    return None


def _fold_sub(b, ops, a):
    x, y = ops
    cx, cy = _const_of(x), _const_of(y)
    if cx is not None and cy is not None:
        return cx - cy
    if cy == 0:
        return x
    return None


def _fold_mul(b, ops, a):
    x, y = ops
    cx, cy = _const_of(x), _const_of(y)
    if cx is not None and cy is not None:
        return cx * cy
    if cx == 1:
        return y
    if cy == 1:
        return x
    if cx == 0 or cy == 0:
        return 0
    return None


def _fold_shr(b, ops, a):
    (x,) = ops
    c = _const_of(x)
    if c is not None:
        return c >> a["amount"]
    if a["amount"] == 0:
        return x
    return None


def _const_of(v: Value):
    return v.node.attr("value") if v.node.kind == "model.const" else None


_P = frozenset({PURE})
for _k in (
    Kind("model.const", frozenset({PURE, CONSTANT}), 0,
         lambda ts, a, s: (Range(a["value"], a["value"]),)),
    Kind("model.input", _P, 0, _declared),         # the pixel, as streamed
    Kind("model.tap", _P, 0, _declared),           # a read: an Index per axis
    Kind("model.row", _P, 0, _extent),
    Kind("model.col", _P, 0, _extent),
    Kind("model.orow", _P, 0, _extent),
    Kind("model.ocol", _P, 0, _extent),
    Kind("model.param", frozenset({PURE, READS_REGISTER}), 0, _register),
    Kind("model.add", frozenset({PURE, COMMUTATIVE}), 2, _add, _fold_add),
    Kind("model.sub", _P, 2, _sub, _fold_sub),
    Kind("model.mul", frozenset({PURE, COMMUTATIVE}), 2, _mul, _fold_mul),
    Kind("model.shr", _P, 1, _shr, _fold_shr),
    Kind("model.mask", _P, 1, _mask),
    Kind("model.clip", _P, 1, _clip),
    Kind("model.abs", _P, 1, _abs),
    Kind("model.lt", _P, 2, _bool),
    Kind("model.sel", _P, 3, _sel),
    Kind("model.gather", frozenset({PURE, READS_REGISTER}), 1, _table),
    Kind("model.rom", _P, 1, _table),
    Kind("model.lin", _P, None, _lin, None, _verify_lin),
    Kind("model.declare", _P, 1, _declare),
    Kind("model.trunc", _P, 1, _trunc),
    # as a model writes them, recorded by np2hw.trace
    Kind("model.source", _P, 0, _declared),       # the streamed image
    Kind("model.pad", _P, 1, _same),               # np.pad: a larger source
    Kind("model.read", _P, 1, _same),              # a slice: an Index per axis
    Kind("model.astype", _P, 1, _astype),
    Kind("model.floordiv", _P, 1, _floordiv),
    Kind("model.neg", _P, 1, _neg),
    Kind("model.canvas", _P, None, _canvas),       # planes, by their indexes
    Kind("model.gt", _P, 2, _bool),
    Kind("model.le", _P, 2, _bool),
    Kind("model.ge", _P, 2, _bool),
    Kind("model.and", _P, 2, _and),
    Kind("model.or", _P, 2, _or),
    Kind("model.where", _P, 3, _sel),
    Kind("model.coords", _P, 1, _coords),          # the input's row, column
    Kind("model.make_coords", _P, 1, _coords),     # a declared raster's
    Kind("model.stack", _P, None, _hull),          # values along a channel axis
):
    register(_k)


# --------------------------------------------------------------------------- #
# PExpr: the expression DAG's nodes ARE graph nodes (stage 1, step 1)
# --------------------------------------------------------------------------- #

class PExpr(Node):
    """A graph node that also presents the old expression-DAG view.

    Built the old way -- `PExpr(op, args, lo, hi)` -- so every construction
    site and every reader stays as it is while the compiler moves onto the
    graph. The graph fields are filled in: `kind` is `model.<op>`, the child
    expressions are its operands, the plain arguments its attributes, a
    register or table object a named symbol. The ORIGINAL arguments are kept
    beside them, so `.args` reads exactly what it read before; that copy is
    the transition's, and goes when the readers move to the graph fields.

    Not hash-consed yet: merging identical nodes can change the emitted
    Verilog, so it is a step of its own, judged by the hardware it produces.
    """
    __slots__ = ("_args",)
    _next_seq = 0

    def __new__(cls, op: str, args: tuple = (), lo: int = 0, hi: int = 0):
        self = object.__new__(cls)
        args = tuple(args)
        operands, attrs = [], []
        for i, a in enumerate(args):
            if isinstance(a, PExpr):
                operands.append(a.value)
            elif _plain(a):
                attrs.append((f"a{i}", a))
            else:                          # a register or a table: a symbol
                attrs.append((f"a{i}", ("symbol", getattr(a, "name", "?"))))
        Node.__init__(self, f"model.{op}", tuple(operands), tuple(attrs),
                      (_R(Range, (lo, hi)),), (), UNKNOWN, PExpr._next_seq)
        PExpr._next_seq += 1
        self._args = args
        return self

    def __init__(self, *args, **kwargs):
        pass                               # built in __new__

    @property
    def op(self) -> str:
        return self.kind[6:]

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def lo(self) -> int:
        return self.results[0][0]

    @property
    def hi(self) -> int:
        return self.results[0][1]

    # The old node was a dataclass: equal by value, and so unhashable.
    def __eq__(self, other):
        if not isinstance(other, PExpr):
            return NotImplemented
        return (self.op, self._args, self.lo, self.hi) == \
            (other.op, other._args, other.lo, other.hi)

    __hash__ = None

    def __repr__(self):
        return (f"PExpr(op={self.op!r}, args={self._args!r}, lo={self.lo!r}, "
                f"hi={self.hi!r})")


# --------------------------------------------------------------------------- #
# From today's expression DAG (the transition; see "Getting there")
# --------------------------------------------------------------------------- #

def from_pexpr(root, builder: Builder | None = None) -> tuple[Value, Builder]:
    """Translate a PExpr DAG into the graph, node for node.

    Built with folding OFF: the translation must say exactly what the DAG
    says, so the two can be compared. Registers, register arrays and tables
    become named SYMBOLS, and the reads refer to them by name.
    """
    b = builder or Builder(fold=False)
    done: dict[int, Value] = {}
    order = []
    stack = [(root, False)]
    while stack:                               # iterative post-order
        p, expanded = stack.pop()
        if id(p) in done:
            continue
        if expanded:
            done[id(p)] = _translate(p, done, b)
            order.append(p)
            continue
        stack.append((p, True))
        for a in reversed(p.args):
            if _is_pexpr(a) and id(a) not in done:
                stack.append((a, False))
    return done[id(root)], b


def _is_pexpr(a) -> bool:
    return type(a).__name__ == "PExpr"


def _translate(p, done, b: Builder) -> Value:
    op, args = p.op, p.args
    ops = [done[id(a)] for a in args if _is_pexpr(a)]
    loc = UNKNOWN
    if op == "acc":
        return b.op("model.input", lo=p.lo, hi=p.hi, loc=loc)
    if op == "tap":
        # a window-expression tap is relative to the pixel being produced:
        # one position per axis, at its offset
        r, c, shift, fld = args
        return b.op("model.tap", row=(1, 1, r, ()), col=(1, 1, c, ()),
                    shift=shift, field=fld, lo=p.lo, hi=p.hi, loc=loc)
    if op == "const":
        # the value is the node's range; `args` repeats it, or is empty in
        # hand-built DAGs
        return b.const(p.lo, loc=loc)
    if op == "param":
        prm = args[0]
        b.symbols.setdefault(prm.name, {"kind": "register",
                                        "bits": prm.bits,
                                        "signed": prm.signed})
        return b.op("model.param", name=prm.name, bits=prm.bits,
                    signed=bool(prm.signed), loc=loc)
    if op in ("row", "col", "orow", "ocol"):
        return b.op(f"model.{op}", extent=p.hi + 1, loc=loc)
    if op in ("add", "sub", "mul", "lt", "sel", "abs"):
        return b.op(f"model.{op}", *ops, loc=loc)
    if op == "shr":
        return b.op("model.shr", ops[0], amount=args[1], loc=loc)
    if op == "mask":
        return b.op("model.mask", ops[0], bits=args[1], loc=loc)
    if op == "clip":
        return b.op("model.clip", ops[0], lo=args[1], hi=args[2], loc=loc)
    if op in ("gather", "rom"):
        table = args[0]
        b.symbols.setdefault(table.name, {
            "kind": "register_array" if op == "gather" else "rom",
            "lo": p.lo, "hi": p.hi})
        return b.op(f"model.{op}", ops[0], table=table.name, loc=loc)
    raise GraphError(f"no graph kind for PExpr op {op!r}")
