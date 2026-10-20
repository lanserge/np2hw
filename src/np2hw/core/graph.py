# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The graph: nodes, values, kinds, types, and what works on any of them.

Every node has the same six fields -- kind, operands, attrs, results,
regions, loc -- so walking, hashing, printing and verifying work on any kind,
including one registered later. What a kind MEANS lives in its `Kind`: its
traits, and the hooks that infer its results and verify it. Nothing here
names a kind.

A node's result TYPE holds three facts (docs/one-graph.md, decisions 2, 3
and 5): its axes -- one per NumPy dimension, each a variable's name or None
for a dimension of size 1 -- its NumPy container, and its exact range.

Nodes are immutable. A builder may hash-cons them, so the same operation on
the same operands is one node; its location is provenance, not identity,
and a node built from two lines of the model keeps both.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

_CORE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FileLine:
    file: str
    line: int

    def __str__(self):
        return f'"{os.path.basename(self.file)}":{self.line}'


@dataclass(frozen=True)
class Fused:
    """Several lines one node came from: a merge unified them."""
    locs: tuple

    def __str__(self):
        return "fused[" + ", ".join(str(l) for l in self.locs) + "]"


@dataclass(frozen=True)
class Unknown:
    def __str__(self):
        return "?"


UNKNOWN = Unknown()
_INSIDE: dict[str, bool] = {}


def here():
    """The innermost frame outside np2hw: the model's line."""
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_filename
        inside = _INSIDE.get(name)
        if inside is None:
            inside = _INSIDE[name] = os.path.realpath(name).startswith(
                _CORE + os.sep)
        if not inside:
            return FileLine(name, f.f_lineno)
        f = f.f_back
    return UNKNOWN


def fuse(a, b):
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
# Types
# --------------------------------------------------------------------------- #

#: Containers -- what NumPy holds a value in (decision 5).
WEAK = ("w", 0)          # a Python int or a register: adopts the other side
BOOL = ("b", 1)


def container_range(c) -> tuple:
    """The values a container holds."""
    kind, bits = c
    if kind == "u":
        return 0, (1 << bits) - 1
    if kind == "i":
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if kind == "b":
        return 0, 1
    raise ValueError(f"container {c} holds no fixed range")


@dataclass(frozen=True)
class Type:
    """One result's facts: axes, container, exact range."""
    axes: tuple                 # per NumPy dimension: a variable name, or None
    container: tuple
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"empty range [{self.lo}, {self.hi}]")

    @property
    def bits(self) -> int:
        """Width the exact range needs, a sign bit included when negative."""
        lo, hi = self.lo, self.hi
        neg = (-lo - 1).bit_length() if lo < 0 else 0
        return max(1, max(hi.bit_length(), neg) + (1 if lo < 0 else 0))

    def __str__(self):
        axes = "(" + ", ".join(a or "1" for a in self.axes) + ")"
        kind, bits = self.container
        cont = {"w": "weak", "b": "bool"}.get(kind, f"{kind}{bits}")
        return f"{axes} {cont} [{self.lo}, {self.hi}]"


# --------------------------------------------------------------------------- #
# Kinds
# --------------------------------------------------------------------------- #

PURE = "pure"
STATEFUL = "stateful"
COMMUTATIVE = "commutative"
CONSTANT = "constant"
READS_REGISTER = "reads_register"


@dataclass(frozen=True)
class Kind:
    """What a node kind means. Registered once; passes consult it.

    infer(graph, operand_types, attrs) -> tuple of result Types
    verify(graph, node) -> None, raising GraphError
    """
    name: str
    traits: frozenset = frozenset()
    arity: int | None = None
    infer: Callable | None = None
    verify: Callable | None = None

    @property
    def level(self) -> str:
        return self.name.split(".", 1)[0]


_KINDS: dict[str, Kind] = {}


def register(k: Kind) -> Kind:
    if k.name in _KINDS:
        raise ValueError(f"kind {k.name!r} registered twice")
    if "." not in k.name:
        raise ValueError(f"kind {k.name!r} needs a level prefix")
    _KINDS[k.name] = k
    return k


def kind(name: str) -> Kind:
    try:
        return _KINDS[name]
    except KeyError:
        raise GraphError(f"unknown node kind {name!r}") from None


class GraphError(Exception):
    """A malformed graph, or something the model does that cannot be
    expressed; carries the node's location, so the message names the line."""

    def __init__(self, message: str, loc=UNKNOWN):
        super().__init__(message if isinstance(loc, Unknown)
                         else f"{message} (at {loc})")
        self.loc = loc


# --------------------------------------------------------------------------- #
# Nodes and values
# --------------------------------------------------------------------------- #

def _plain(v) -> bool:
    return isinstance(v, (int, str, bool, type(None))) or (
        isinstance(v, tuple) and all(_plain(x) for x in v))


class Node:
    __slots__ = ("kind", "operands", "attrs", "results", "regions", "loc",
                 "seq", "_values")

    def __init__(self, kind_, operands, attrs, results, regions, loc, seq):
        self.kind = kind_
        self.operands = operands
        self.attrs = attrs              # tuple of (key, value), sorted
        self.results = results          # tuple of Type
        self.regions = regions
        self.loc = loc
        self.seq = seq                  # creation order: determinism
        self._values = tuple(Value(self, i) for i in range(len(results)))

    def attr(self, key, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    @property
    def value(self) -> "Value":
        if len(self._values) != 1:
            raise GraphError(f"{self.kind} has {len(self._values)} results",
                             self.loc)
        return self._values[0]

    def result(self, i: int) -> "Value":
        return self._values[i]

    def __repr__(self):
        return f"<{self.kind} #{self.seq}>"


class Value:
    """One result of one node: (node, result index)."""
    __slots__ = ("node", "index")

    def __init__(self, node, index):
        self.node = node
        self.index = index

    @property
    def type(self) -> Type:
        return self.node.results[self.index]

    def __repr__(self):
        return f"{self.node!r}#{self.index}"


# --------------------------------------------------------------------------- #
# Variables: named declarations, with unification
# --------------------------------------------------------------------------- #

class Variables:
    """Every variable of a graph: its count, its origin, its rank in axis
    order, and the unification NumPy's positional broadcasting implies --
    two variables paired on one position with equal counts are one."""

    def __init__(self):
        self._decl: dict[str, dict] = {}
        self._parent: dict[str, str] = {}
        self._n = 0

    def declare(self, count: int, origin: str, hint: str = "v",
                ends: str | None = None) -> str:
        """`ends` says how the axis's extent ends, where the model said: at
        the end of the axis it reads, whatever that axis's size ("end" --
        `x[1:-1]`), or after a fixed count ("fixed" -- `x[2:10]`, a declared
        raster). A count alone cannot tell them apart at the size traced,
        and a frame of another size can."""
        if count < 1:
            raise GraphError(f"a variable counting {count}")
        name = f"{hint}{self._n}"
        self._n += 1
        self._decl[name] = {"count": int(count), "origin": origin,
                            "rank": self._n, "ends": ends}
        self._parent[name] = name
        return name

    def find(self, name: str) -> str:
        root = name
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[name] != root:          # path compression
            self._parent[name], name = root, self._parent[name]
        return root

    def unify(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        ca, cb = self.count(ra), self.count(rb)
        if ca != cb:
            raise GraphError(f"axes of {ca} and {cb} paired on one position")
        keep, drop = sorted((ra, rb), key=lambda v: self._decl[v]["rank"])
        self._parent[drop] = keep
        # an axis read up to its end by any of the reads paired on it
        # follows the end; it is fixed only where every read said fixed
        said = {self._decl[keep]["ends"], self._decl[drop]["ends"]} - {None}
        self._decl[keep]["ends"] = ("end" if "end" in said else
                                    "fixed" if said else None)
        return keep

    def count(self, name: str) -> int:
        return self._decl[self.find(name)]["count"]

    def rank(self, name: str) -> int:
        return self._decl[self.find(name)]["rank"]

    def origin(self, name: str) -> str:
        return self._decl[self.find(name)]["origin"]

    def ends(self, name: str) -> str | None:
        """How the variable's extent ends: "end", "fixed", or None (not
        said). Where reads paired on it said differently, "end" wins: an
        axis read up to its end anywhere follows the end."""
        return self._decl[self.find(name)]["ends"]

    def names(self) -> list[str]:
        return list(self._decl)


# --------------------------------------------------------------------------- #
# The graph and its builder
# --------------------------------------------------------------------------- #

class Graph:
    """Nodes, the variables they range over, and named declarations:
    registers, tables, and (at the hardware level) state."""

    def __init__(self, intern: bool = True, locations: bool = True):
        self.vars = Variables()
        self.symbols: dict[str, dict] = {}
        self.intern = intern
        self.locations = locations
        self._table: dict = {}
        self._seq = 0

    def node(self, kind_name: str, operands: Iterable[Value] = (),
             attrs: dict | None = None, regions: tuple = (),
             loc=None) -> Node:
        k = kind(kind_name)
        operands = tuple(operands)
        attrs = tuple(sorted((attrs or {}).items()))
        if loc is None:
            loc = here() if self.locations else UNKNOWN
        if k.arity is not None and len(operands) != k.arity:
            raise GraphError(f"{k.name} takes {k.arity} operand(s), got "
                             f"{len(operands)}", loc)
        if self.intern and COMMUTATIVE in k.traits and len(operands) == 2:
            x, y = operands
            if self.skey(y) < self.skey(x):
                operands = (y, x)
        key = (k.name, tuple((id(v.node), v.index) for v in operands), attrs,
               tuple(id(r) for r in regions))
        if self.intern:
            hit = self._table.get(key)
            if hit is not None:
                hit.loc = fuse(hit.loc, loc)
                return hit
        types = tuple(k.infer(self, [v.type for v in operands],
                              dict(attrs))) if k.infer else ()
        n = Node(k.name, operands, attrs, types, regions, loc, self._seq)
        self._seq += 1
        if self.intern:
            self._table[key] = n
        return n

    def op(self, kind_name: str, *operands: Value, loc=None, **attrs) -> Value:
        return self.node(kind_name, operands, attrs, loc=loc).value

    def skey(self, v: Value) -> str:
        """A structural key for a value -- its kind, attributes and operands'
        keys, recursively -- the same in any graph. It orders the operands of
        a commutative operation, so a canonical form does not depend on the
        order nodes happened to be made in."""
        cache = self.__dict__.setdefault("_skeys", {})
        k = cache.get(id(v.node))
        if k is None:
            n = v.node
            # a fixed-size digest of the structure: nesting the operands'
            # keys as text would grow with the graph unfolded as a tree
            k = hashlib.blake2b(repr((n.kind, n.attrs, tuple(
                self.skey(o) for o in n.operands))).encode(),
                digest_size=16).hexdigest()
            cache[id(n)] = k
        return f"{k}#{v.index}"


# --------------------------------------------------------------------------- #
# Walking, users, verifying, printing
# --------------------------------------------------------------------------- #

def walk(roots: Iterable[Value]) -> Iterator[Node]:
    """Every node reachable from `roots`, operands first, each once."""
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


def users(roots: Iterable[Value]) -> dict[int, list[Node]]:
    """For each node, the nodes that use it -- an analysis over the
    immutable graph: every access through one index is that index's user."""
    out: dict[int, list[Node]] = {}
    for n in walk(roots):
        out.setdefault(id(n), [])
        for v in n.operands:
            lst = out.setdefault(id(v.node), [])
            if not lst or lst[-1] is not n:
                lst.append(n)
    return out


def verify(g: Graph, roots: Iterable[Value],
           levels: set[str] | None = None) -> None:
    """Raise GraphError at the first malformed node."""
    for n in walk(roots):
        k = kind(n.kind)
        if levels is not None and k.level not in levels:
            raise GraphError(f"{n.kind} is not legal at level "
                             f"{sorted(levels)}", n.loc)
        for key, val in n.attrs:
            if not _plain(val):
                raise GraphError(f"{n.kind}: attribute {key}={val!r} is not "
                                 "plain data", n.loc)
        if k.infer is not None:
            expect = tuple(k.infer(g, [v.type for v in n.operands],
                                   dict(n.attrs)))
            if _resolved(g, expect) != _resolved(g, n.results):
                raise GraphError(f"{n.kind}: results {n.results} disagree "
                                 f"with its rule {expect}", n.loc)
        if k.verify is not None:
            k.verify(g, n)


def _resolved(g: Graph, types) -> tuple:
    """Types with their axes resolved through unification: a variable
    unified after a node was built still names the same axis."""
    return tuple(Type(tuple(None if a is None else g.vars.find(a)
                            for a in t.axes), t.container, t.lo, t.hi)
                 for t in types)


def dump(g: Graph, roots: Iterable[Value], locations: bool = False) -> str:
    """The graph as text: its variables, then every node, deterministically."""
    roots = list(roots)
    lines = []
    for name in g.vars.names():
        rep = g.vars.find(name)
        if rep == name:
            lines.append(f"var {name} : {g.vars.count(name)} "
                         f"({g.vars.origin(name)})")
        else:
            lines.append(f"var {name} = {rep}")
    names: dict[int, str] = {}
    for i, n in enumerate(walk(roots)):
        names[id(n)] = f"%{i}"

        def ref(v):
            base = names[id(v.node)]
            return base if len(v.node.results) == 1 else f"{base}#{v.index}"

        ops = ", ".join(ref(v) for v in n.operands)
        attrs = (" {" + ", ".join(f"{k}={v!r}" for k, v in n.attrs) + "}"
                 if n.attrs else "")
        types = ", ".join(str(t) for t in n.results)
        where = f" loc({n.loc})" if locations else ""
        lhs = f"%{i}" if len(n.results) <= 1 else f"%{i}:{len(n.results)}"
        lines.append(f"{lhs} = {n.kind}({ops}){attrs}"
                     f"{' : ' + types if types else ''}{where}")
    lines.append("return " + ", ".join(
        names[id(r.node)] + ("" if len(r.node.results) == 1
                             else f"#{r.index}") for r in roots))
    return "\n".join(lines)
