# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Indexes are expressions; their affine form is derived (decision 1).

Index math is ordinary graph arithmetic on variables. This module reads an
index expression's AFFINE FORM -- `sum(step_i * v_i) + start`, the start a
constant plus register terms -- when it has one, and builds the one
CANONICAL node for a form, so that equal indexes are the same node and every
access through an index is one of its users.

`v // k` and `v % k` on a variable are SPLITS: `v = k * q + r`, two new
variables recorded once per (variable, k), and every form over `v` is
written in `q` and `r` when either part appears.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import GraphError, Graph, Value


@dataclass(frozen=True)
class Affine:
    """`sum(step * var) + sum(coef * register) + const`, normalised: terms
    sorted by the variables' rank, no zero coefficients."""
    terms: tuple = ()               # ((variable, step), ...)
    regs: tuple = ()                # ((register, coefficient), ...)
    const: int = 0

    @property
    def is_constant(self) -> bool:
        return not self.terms and not self.regs


def _norm(g: Graph, terms: dict, regs: dict, const: int) -> Affine:
    t = {}
    for v, s in terms.items():
        r = g.vars.find(v)
        t[r] = t.get(r, 0) + s
    t = sorted(((v, s) for v, s in t.items() if s), key=lambda vs:
               g.vars.rank(vs[0]))
    rg = tuple(sorted((n, c) for n, c in regs.items() if c))
    return Affine(tuple(t), rg, const)


def _scale(a: Affine, k: int) -> tuple:
    return ({v: s * k for v, s in a.terms}, {r: c * k for r, c in a.regs},
            a.const * k)


def _plus(g, a: Affine, b: Affine, sign: int = 1) -> Affine:
    terms, regs = dict(a.terms), dict(a.regs)
    for v, s in b.terms:
        terms[v] = terms.get(v, 0) + sign * s
    for r, c in b.regs:
        regs[r] = regs.get(r, 0) + sign * c
    return _norm(g, terms, regs, a.const + sign * b.const)


class Splits:
    """The splits of a graph's variables: (variable, k) -> (q, r)."""

    def __init__(self):
        self._by: dict = {}
        self._parent: dict = {}         # q or r -> (variable, k)

    def split(self, g: Graph, v: str, k: int) -> tuple:
        v = g.vars.find(v)
        key = (v, k)
        if key not in self._by:
            n = g.vars.count(v)
            if k < 1 or n % k:
                raise GraphError(f"a split of an axis of {n} by {k}: a ragged "
                                 "split has no lattice")
            q = g.vars.declare(n // k, f"{v} // {k}", "q")
            r = g.vars.declare(k, f"{v} % {k}", "r")
            self._by[key] = (q, r)
        return self._by[key]

    def defined(self, g: Graph) -> dict:
        """variable -> (k, q, r) for every split variable."""
        return {g.vars.find(v): (k, q, r) for (v, k), (q, r) in
                self._by.items()}


def splits(g: Graph) -> Splits:
    s = g.symbols.get("__splits__")
    if s is None:
        s = g.symbols["__splits__"] = {"object": Splits()}
    return s["object"]


def affine(g: Graph, v: Value) -> Affine | None:
    """The affine form of an index expression, or None when it is data.

    A value computed from registers alone -- `p.phase >> 1`, a register's
    low bit -- is one more register term where no finer form is found: it
    is the same at every position, so to the index math it is a register,
    held in the graph's symbols under its structural key."""
    form = _affine(g, v)
    if form is None and _of_registers(v):
        key = f"~{g.skey(v)}"
        g.symbols.setdefault(key, {"object": None, "read": v})
        form = Affine(regs=((key, 1),))
    return form


def _of_registers(v: Value) -> bool:
    """Whether `v` is the same at every position and reads registers: no
    axis, and nothing beneath it but registers and constants."""
    if any(a is not None for a in v.type.axes):
        return False
    seen, stack, reads = set(), [v.node], False
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.kind == "model.param":
            reads = True
        elif n.kind != "model.const":
            if not n.operands or n.kind in ("model.var", "model.table",
                                            "model.source", "model.access"):
                return False
            stack.extend(o.node for o in n.operands)
    return reads


def _affine(g: Graph, v: Value) -> Affine | None:
    n, k = v.node, v.node.kind
    if k == "model.var":
        return _norm(g, {n.attr("name"): 1}, {}, 0)
    if k == "model.const":
        return Affine(const=n.attr("value"))
    if k == "model.param":
        return Affine(regs=((n.attr("name"), 1),))
    if k in ("model.add", "model.sub"):
        a, b = (affine(g, o) for o in n.operands)
        if a is None or b is None:
            return None
        return _plus(g, a, b, 1 if k == "model.add" else -1)
    if k == "model.expand":            # placing an axis moves nothing
        return affine(g, n.operands[0])
    if k == "model.neg":
        a = affine(g, n.operands[0])
        return None if a is None else _norm(g, *_scale(a, -1))
    if k == "model.mul":
        a, b = (affine(g, o) for o in n.operands)
        if a is None or b is None:
            return None
        if b.is_constant:
            return _norm(g, *_scale(a, b.const))
        if a.is_constant:
            return _norm(g, *_scale(b, a.const))
        return None                    # a variable times a variable: data
    if k in ("model.floordiv", "model.mod"):
        a = affine(g, n.operands[0])
        d = n.attr("divisor")
        if a is None or a.regs or a.const or len(a.terms) != 1 \
                or a.terms[0][1] != 1:
            return None                # a split of a bare variable only
        q, r = splits(g).split(g, a.terms[0][0], d)
        return _norm(g, {q if k == "model.floordiv" else r: 1}, {}, 0)
    return None


def refine(g: Graph, a: Affine) -> Affine:
    """Write a form in its finest variables: a split variable is replaced by
    `k * q + r` wherever it appears, so that forms over a variable and over
    its parts compare equal."""
    defs = splits(g).defined(g)
    changed = True
    while changed:
        changed = False
        for v, s in a.terms:
            if v in defs:
                k, q, r = defs[v]
                rest = {w: t for w, t in a.terms if w != v}
                rest[q] = rest.get(q, 0) + s * k
                rest[r] = rest.get(r, 0) + s
                a = _norm(g, rest, dict(a.regs), a.const)
                changed = True
                break
    return a


def canonical(g: Graph, a: Affine) -> Value:
    """The one node for a form: variables by rank, then registers, then the
    constant -- built through the graph, so equal forms are one node."""
    a = refine(g, a)
    out = None

    def add(x):
        nonlocal out
        out = x if out is None else g.op("model.add", out, x)

    order = [v for v, _ in a.terms]        # its axes: variables by rank
    for v, s in a.terms:
        term = g.op("model.var", name=v)
        if len(order) > 1:                   # each variable on its own axis
            term = g.op("model.expand", term,
                        axes=tuple(v if w == v else None for w in order))
        if s != 1:
            term = g.op("model.mul", term, _const(g, s))
        add(term)
    for name, c in a.regs:
        term = g.symbols[name]["read"]
        if c != 1:
            term = g.op("model.mul", term, _const(g, c))
        add(term)
    if a.const or out is None:
        add(_const(g, a.const))
    return out


def _const(g, value):
    return g.op("model.const", value=int(value), container=("w", 0))
