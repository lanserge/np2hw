# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The C backend: the canonical model graph printed as a C function.

The same graph the Verilog comes from, printed for a CPU, kind by kind, as
`evaluate.py` runs it with NumPy. A value is computed AT A POSITION: an
environment binds each variable to an index expression, and a node's value
there is its operation on its operands' values there. That is all access
needs: a read of the source or a table is a load at the index; a read of a
pad is the padded buffer's value at the index moved inside by the pad's
edge rule; a read of a computed value is that value computed at the index
(the substitution canonicalising does for windows, done here for every
read); a stack is the stacked value its variable selects.

Loops come from the output's variables: one loop per variable, in axis
order, and a variable of a few positions (channels, a small stack) is
UNROLLED -- each position its own statements -- so the innermost loop runs
along a long axis and the compiler can vectorise it.

Exactness: every statement is typed by the narrowest C integer type that
holds its exact range, and its operands are converted to a type holding
the operands and the result before the operation, so C's integer
promotions and signed overflow never change a value -- the range proves no
overflow happens. Wraps are explicit nodes after canonicalising: nothing
wraps that NumPy does not.

Speed comes from the loop's shape, not from intrinsics:

- a statement goes to the outermost loop it depends on (a row's clamped
  index once a row, a register once a call);
- along the innermost loop, positions where no pad read can leave the
  padded buffer's interior run in a loop of their own with no clamps and no
  tests, between two short edge loops that keep them;
- the caller hands a range of the outermost loop, so threads take bands.

The function is plain C11 with fixed-width types and caller-owned buffers:

    void NAME(const SRC* src, OUT* dst, const int64_t* regs,
              int64_t lo, int64_t hi);

`src` and `dst` are dense C-ordered arrays of the source's and output's
shapes (constants in the generated header), `regs` the registers in the
model's register order, a register array one slot per entry, and
[lo, hi) the positions of the outermost loop to compute.

Refused, by name: a canvas the canonical form leaves (planes that do not
tile a lattice), a register window (`model.window`), and a value over a
variable the output does not have.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .canonical import Canonical, canonicalise
from .graph import Value, walk
from .index import splits
from .trace import Unsupported, trace

#: an output axis this short is unrolled when a longer one can be the loop
UNROLL_MAX = 16

_CTYPES = [(np.int8, "int8_t"), (np.uint8, "uint8_t"), (np.int16, "int16_t"),
           (np.uint16, "uint16_t"), (np.int32, "int32_t"),
           (np.uint32, "uint32_t"), (np.int64, "int64_t"),
           (np.uint64, "uint64_t")]
_CNAME = {np.dtype(d): n for d, n in _CTYPES}


def ctype_for(lo: int, hi: int) -> tuple:
    """(NumPy dtype, C type) of the narrowest integer holding [lo, hi]."""
    for dt, name in _CTYPES:
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dt), name
    raise Unsupported(f"a value in [{lo}, {hi}] needs more than 64 bits")


def _fits64(lo: int, hi: int) -> bool:
    """Whether one C integer type holds [lo, hi]: int64_t or uint64_t."""
    return (-(1 << 63) <= lo and hi < (1 << 63)) or (0 <= lo and hi < 1 << 64)


def _lit(v: int) -> str:
    if -(1 << 31) <= v < (1 << 31):
        return str(v)
    return f"INT64_C({v})" if v < (1 << 63) else f"UINT64_C({v})"


_BIN = {"model.add": "+", "model.sub": "-", "model.mul": "*",
        "model.and": "&", "model.or": "|", "model.xor": "^"}
#: the operations whose low 64 bits are those of their operands': a result
#: wider than 64 bits, which NumPy's 64-bit container wraps, is computed
#: modulo 2**64
_RING = {"model.add": "+", "model.sub": "-", "model.mul": "*",
         "model.and": "&", "model.or": "|", "model.xor": "^"}
_CMP = {"model.lt": "<", "model.gt": ">", "model.le": "<=", "model.ge": ">=",
        "model.eq": "==", "model.ne": "!="}


# --------------------------------------------------------------------------- #
# Values at a position
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Sym:
    """A value at a position, as C sees it: AFFINE in the loop variables --
    `const + sum(coef * loop)` -- or an opaque C expression (a statement's
    name). `deps` are the loop variables it depends on, [lo, hi] its range.
    A MODULAR value is wider than 64 bits and held modulo 2**64, its low
    bits exact: only the wrap into its 64-bit container reads it."""
    terms: tuple = ()           # ((loop variable, coefficient), ...), sorted
    const: int = 0
    expr: str | None = None     # set when opaque
    deps: frozenset = frozenset()
    lo: int = 0
    hi: int = 0
    ctype: str = "int64_t"
    modular: bool = False

    @property
    def affine(self) -> bool:
        return self.expr is None

    @property
    def literal(self) -> bool:
        return self.expr is None and not self.terms

    def key(self):
        return ("a", self.terms, self.const) if self.expr is None \
            else ("o", self.expr)

    def c(self, names: dict) -> str:
        """The C expression (int64 arithmetic for an affine one)."""
        if self.expr is not None:
            return self.expr
        parts = [(f"{names[v]}" if k == 1 else f"{_lit(k)} * {names[v]}")
                 for v, k in self.terms]
        if self.const or not parts:
            parts.append(_lit(self.const))
        return "(" + " + ".join(parts) + ")" if len(parts) > 1 else parts[0]


def _affine(terms: dict, const: int, lo: int, hi: int) -> Sym:
    t = tuple(sorted((v, k) for v, k in terms.items() if k))
    if not t:
        return _literal(const)
    return Sym(t, const, None, frozenset(v for v, _ in t), lo, hi)


def _literal(v: int) -> Sym:
    return Sym((), int(v), None, frozenset(), int(v), int(v))


# --------------------------------------------------------------------------- #
# One loop nest
# --------------------------------------------------------------------------- #

class _Nest:
    """Statements of one loop nest, each at the outermost loop it depends
    on. Along the innermost loop there are two bodies -- the EDGE body,
    every read kept inside its buffer, and the INTERIOR body, where reads
    the interior keeps inside are left alone -- and the statements that
    depend on the innermost variable are each body's own."""

    def __init__(self, loops: list):
        self.loops = loops                       # loop variables, outer first
        self.level = {v: i for i, v in enumerate(loops)}
        self.inner = loops[-1] if loops else None
        self.top: list[str] = []
        self.at: list[list[str]] = [[] for _ in loops]
        self.bodies = {"edge": [], "interior": []}
        self.mode = "edge"
        self.shared: dict = {}
        self.own = {"edge": {}, "interior": {}}
        self.interior = None                    # [xa, xb) once known
        self.bounds: list = []                  # (coef, const, n) per read

    def place(self, deps: frozenset, line: str):
        if not deps:
            self.top.append(line)
            return
        lvl = max(self.level[d] for d in deps)
        if lvl == len(self.loops) - 1:
            self.bodies[self.mode].append(line)
        else:
            self.at[lvl].append(line)

    def get(self, key):
        hit = self.shared.get(key)
        return hit if hit is not None else self.own[self.mode].get(key)

    def put(self, key, sym: Sym):
        if self.inner in sym.deps:
            self.own[self.mode][key] = sym
        else:
            self.shared[key] = sym


# --------------------------------------------------------------------------- #
# The printer
# --------------------------------------------------------------------------- #

class _Printer:
    def __init__(self, c: Canonical):
        from ..ir import Rom
        self.c, self.g = c, c.g
        self.find = c.g.vars.find
        self.count = c.g.vars.count
        self.k = 0
        self.decls: list[str] = []              # file scope: constant tables
        self.roms: dict[str, str] = {}
        # registers: one slot each, a register array one per entry
        self.regs: list[tuple] = []             # (name, size)
        self.reg_at: dict[str, int] = {}
        at = 0
        for name in c.registers:
            obj = c.g.symbols[name]["object"]
            if isinstance(obj, Rom):
                continue
            read = c.g.symbols[name]["read"]
            size = (self.count(read.type.axes[0])
                    if read.node.kind == "model.table" else 1)
            self.regs.append((name, size))
            self.reg_at[name] = at
            at += size
        self.names: dict[str, str] = {}         # loop variable -> C name

    # -- statements ---------------------------------------------------------- #

    def stmt(self, nest: _Nest, ctype: str, expr: str, deps, lo, hi,
             modular: bool = False) -> Sym:
        name = f"v{self.k}"
        self.k += 1
        nest.place(frozenset(deps), f"const {ctype} {name} = {expr};")
        return Sym((), 0, name, frozenset(deps), lo, hi, ctype, modular)

    def ref(self, s: Sym, ctype: str) -> str:
        return f"(({ctype}){s.c(self.names)})"

    @staticmethod
    def span(*syms) -> str:
        return ctype_for(min(s.lo for s in syms), max(s.hi for s in syms))[1]

    # -- variables ----------------------------------------------------------- #

    def lookup(self, nest: _Nest, env: dict, var: str) -> Sym:
        """A variable's value: bound, or derived through a split -- `q` and
        `r` of `v = k * q + r` from `v`, or `v` from its parts."""
        var = self.find(var)
        if var in env:
            return env[var]
        defs = splits(self.g).defined(self.g)
        for v, (k, q, r) in defs.items():
            if var in (self.find(q), self.find(r)) and v in env:
                s = env[v]
                if s.literal:
                    return _literal(s.const // k if var == self.find(q)
                                    else s.const % k)
                hi = self.count(var) - 1
                op = "/" if var == self.find(q) else "%"
                return self.stmt(nest, "int64_t",
                                 f"{s.c(self.names)} {op} {k}", s.deps, 0, hi)
            if var == v and self.find(q) in env and self.find(r) in env:
                qs, rs = env[self.find(q)], env[self.find(r)]
                return self.combine(nest, [(qs, k), (rs, 1)], 0,
                                    0, self.count(v) - 1)
        raise Unsupported(f"a value over the axis {var} "
                          f"({self.g.vars.origin(var)}), which the output "
                          "does not have")

    def combine(self, nest, parts, const, lo, hi) -> Sym:
        """sum(coef * sym) + const, affine where every part is."""
        if all(s.affine for s, _ in parts):
            terms: dict = {}
            c = const
            for s, k in parts:
                c += k * s.const
                for v, t in s.terms:
                    terms[v] = terms.get(v, 0) + k * t
            return _affine(terms, c, lo, hi)
        expr = " + ".join(f"{_lit(k)} * {self.ref(s, 'int64_t')}"
                          for s, k in parts)
        deps = frozenset().union(*(s.deps for s, _ in parts))
        return self.stmt(nest, "int64_t", f"{expr} + {_lit(const)}", deps,
                         lo, hi)

    # -- a value at a position ------------------------------------------------ #

    def val(self, nest: _Nest, v: Value, env: dict) -> Sym:
        axes = [self.find(a) for a in v.type.axes if a is not None]
        key = (id(v.node), v.index,
               tuple((a, self.lookup(nest, env, a).key()) for a in axes))
        hit = nest.get(key)
        if hit is not None:
            return hit
        out = self.node(nest, v, env)
        nest.put(key, out)
        return out

    def node(self, nest: _Nest, v: Value, env: dict) -> Sym:
        n, t = v.node, v.type
        k = n.kind
        ops = n.operands
        if k == "model.const":
            return _literal(n.attr("value"))
        if k == "model.var":
            s = self.lookup(nest, env, n.attr("name"))
            return s if s.affine else Sym((), 0, s.expr, s.deps, t.lo, t.hi,
                                          s.ctype)
        if k == "model.expand":
            return self.val(nest, ops[0], env)
        if k == "model.param":
            own = ctype_for(t.lo, t.hi)[1]
            return self.stmt(nest, own,
                             f"({own})regs[{self.reg_at[n.attr('name')]}]",
                             (), t.lo, t.hi)
        if k in ("model.source", "model.table", "model.pad"):
            idx = [self.lookup(nest, env, a) for a in t.axes]
            return self.read(nest, v, idx, env)
        if k == "model.access":
            idx = [self.val(nest, i, env) for i in ops[1:]]
            return self.read(nest, ops[0], idx, env)
        if k == "model.stack":
            sel = self.lookup(nest, env, n.attr("var"))
            if sel.literal:
                return self.val(nest, ops[sel.const], env)
            vals = [self.val(nest, o, env) for o in ops]
            return self.choose(nest, sel, vals, t)
        if k == "model.select":
            m = len(n.attr("sizes"))
            keys = [self.val(nest, o, env) for o in ops[:m]]
            parts, size = [], 1
            for s, sz in reversed(list(zip(keys, n.attr("sizes")))):
                parts.append((s, size))
                size *= sz
            flat = self.combine(nest, parts, 0, 0, size - 1)
            if flat.literal:
                return self.val(nest, ops[m + flat.const], env)
            vals = [self.val(nest, o, env) for o in ops[m:]]
            return self.choose(nest, flat, vals, t)
        if k == "model.canvas":
            raise Unsupported("a canvas whose planes do not tile a lattice "
                              "(the C backend reads the selects canonicalising "
                              "makes of one that does)")
        if k == "model.window":
            raise Unsupported("a register window (model.window) in the C "
                              "backend")
        vals = [self.val(nest, o, env) for o in ops]
        return self.arith(nest, n, t, vals)

    def choose(self, nest, sel: Sym, vals, t) -> Sym:
        own = ctype_for(t.lo, t.hi)[1]
        w = self.span(_literal(t.lo), _literal(t.hi), *vals)
        s = self.ref(sel, "int64_t")
        expr = self.ref(vals[-1], w)
        for j in range(len(vals) - 2, -1, -1):
            expr = f"({s} == {j} ? {self.ref(vals[j], w)} : {expr})"
        deps = sel.deps.union(*(x.deps for x in vals))
        return self.stmt(nest, own, f"({own}){expr}", deps, t.lo, t.hi)

    # -- reads ---------------------------------------------------------------- #

    def read(self, nest: _Nest, buf: Value, idx: list, env: dict) -> Sym:
        """`buf` read at the index values `idx`, one per axis."""
        b = buf.node
        t = buf.type
        if b.kind == "model.source":
            return self.load(nest, "src", t, [self.count(a) for a in t.axes],
                             idx)
        if b.kind == "model.table":
            name = b.attr("name")
            if name in self.reg_at:
                own = ctype_for(t.lo, t.hi)[1]
                i = self.ref(idx[0], "int64_t")
                return self.stmt(nest, own,
                                 f"({own})regs[{self.reg_at[name]} + {i}]",
                                 idx[0].deps, t.lo, t.hi)
            return self.load(nest, self.rom(name, t), t,
                             [self.count(t.axes[0])], idx)
        if b.kind == "model.pad":
            return self.read_pad(nest, buf, idx, env)
        if b.kind in ("model.param", "model.const"):
            return self.val(nest, buf, env)
        # a computed value read at an index is the value computed there
        inner = dict(env)
        for a, s in zip(t.axes, idx):
            if a is not None:
                inner[self.find(a)] = s
        return self.val(nest, buf, inner)

    def load(self, nest, array: str, t, shape: list, idx: list) -> Sym:
        off, deps = None, frozenset()
        for s, n in zip(idx, shape):
            term = s.c(self.names) if s.affine else f"(int64_t){s.expr}"
            off = term if off is None else f"({off}) * {n} + {term}"
            deps |= s.deps
        own = ctype_for(t.lo, t.hi)[1]
        return self.stmt(nest, own, f"{array}[{off or 0}]", deps, t.lo, t.hi)

    def rom(self, name: str, t) -> str:
        if name not in self.roms:
            data = np.asarray(self.g.symbols[name]["object"].data).ravel()
            own = ctype_for(int(data.min()), int(data.max()))[1]
            cname = f"rom_{len(self.roms)}"
            body = ", ".join(_lit(int(x)) for x in data)
            self.decls.append(f"static const {own} {cname}[{data.size}] = "
                              f"{{{body}}};")
            self.roms[name] = cname
        return self.roms[name]

    def read_pad(self, nest: _Nest, buf: Value, idx: list, env: dict) -> Sym:
        p = buf.node
        inner_v = p.operands[0]
        zero = p.attr("mode") == "zero"
        coords, inside = [], []
        for s, a, (before, _) in zip(idx, inner_v.type.axes, p.attr("widths")):
            n = self.count(a)
            q = self.combine(nest, [(s, 1)], -before, -before + s.lo,
                             -before + s.hi) if before else s
            if q.lo >= 0 and q.hi < n:
                coords.append(q)                 # inside by its range
                continue
            if q.affine and len(q.terms) == 1 \
                    and q.terms[0][0] == nest.inner and q.terms[0][1] > 0:
                if nest.mode == "edge":
                    nest.bounds.append((q.terms[0][1], q.const, n))
                elif nest.interior is not None:
                    coords.append(q)             # the interior keeps it in
                    continue
            c = q.c(self.names)
            clamped = self.stmt(nest, "int64_t",
                                f"{c} < 0 ? 0 : {c} > {n - 1} ? {n - 1} : {c}",
                                q.deps, 0, n - 1)
            coords.append(clamped)
            if zero:
                inside.append(self.stmt(nest, "bool",
                                        f"{c} >= 0 && {c} <= {n - 1}",
                                        q.deps, 0, 1))
        at = dict(env)
        for a, s in zip(inner_v.type.axes, coords):
            at[self.find(a)] = s
        if inner_v.node.kind in ("model.source", "model.table", "model.pad"):
            got = self.read(nest, inner_v, coords, at)
        else:
            got = self.val(nest, inner_v, at)
        if not inside:
            return got
        t = buf.type
        own = ctype_for(t.lo, t.hi)[1]
        cond = " && ".join(s.expr for s in inside)
        deps = got.deps.union(*(s.deps for s in inside))
        return self.stmt(nest, own, f"({cond}) ? ({own}){got.c(self.names)} "
                                    f": ({own})0", deps, t.lo, t.hi)

    # -- arithmetic ------------------------------------------------------------ #

    def arith(self, nest: _Nest, n, t, vals: list) -> Sym:
        k = n.kind
        lo, hi = t.lo, t.hi
        if k in ("model.add", "model.sub") and all(s.affine for s in vals):
            return self.combine(nest, [(vals[0], 1),
                                       (vals[1], 1 if k == "model.add" else -1)],
                                0, lo, hi)
        if k == "model.neg" and vals[0].affine:
            return self.combine(nest, [(vals[0], -1)], 0, lo, hi)
        if k == "model.mul" and all(s.affine for s in vals) \
                and (vals[0].literal or vals[1].literal):
            a, b = vals if vals[1].literal else vals[::-1]
            return self.combine(nest, [(a, b.const)], 0, lo, hi)
        if all(s.literal for s in vals) and k in _FOLD:
            return _literal(_FOLD[k](n, *[s.const for s in vals]))
        deps = frozenset().union(*(s.deps for s in vals))
        res = _literal(lo), _literal(hi)
        if k == "model.wrap" and vals[0].modular:
            return self.cut(nest, n, vals[0], deps, lo, hi)
        if any(s.modular for s in vals):
            raise Unsupported(f"{k} of a value wider than 64 bits")
        if not _fits64(lo, hi) and k in _RING:
            # NumPy wraps it in its 64-bit container: its low 64 bits are
            # all that is read, and uint64_t arithmetic is exact in them
            a, b = (self.ref(s, "uint64_t") for s in vals)
            return self.stmt(nest, "uint64_t", f"{a} {_RING[k]} {b}", deps,
                             lo, hi, modular=True)
        if not _fits64(lo, hi) and k == "model.neg":
            return self.stmt(nest, "uint64_t",
                             f"(uint64_t)0 - {self.ref(vals[0], 'uint64_t')}",
                             deps, lo, hi, modular=True)
        if k in _CMP:
            w = self.span(*vals)
            a, b = (self.ref(s, w) for s in vals)
            return self.stmt(nest, "bool", f"{a} {_CMP[k]} {b}", deps, 0, 1)
        own = ctype_for(lo, hi)[1]
        if k in _BIN:
            w = self.span(*res, *vals)
            a, b = (self.ref(s, w) for s in vals)
            return self.stmt(nest, own, f"({own})({a} {_BIN[k]} {b})", deps,
                             lo, hi)
        if k == "model.neg":
            w = self.span(*res, *vals)
            return self.stmt(nest, own, f"({own})(-{self.ref(vals[0], w)})",
                             deps, lo, hi)
        if k == "model.abs":
            w = self.span(*res, *vals)
            a = self.ref(vals[0], w)
            return self.stmt(nest, own, f"({own})({a} < 0 ? ({w})(-{a}) : {a})",
                             deps, lo, hi)
        if k == "model.shr":
            w = self.span(*res, *vals)
            return self.stmt(nest, own, f"({own})({self.ref(vals[0], w)} >> "
                                        f"{n.attr('amount')})", deps, lo, hi)
        if k in ("model.floordiv", "model.mod"):
            d = n.attr("divisor")
            if d <= 0:
                raise Unsupported("a division by a constant that is not "
                                  "positive")
            w = self.span(*res, *vals)
            a = self.ref(vals[0], w)
            if vals[0].lo >= 0:
                q = f"({a} / {d})"
            else:                       # floor: C truncates toward zero
                q = f"(({a} >= 0 ? {a} : ({w})({a} - {d - 1})) / {d})"
            expr = q if k == "model.floordiv" else f"({w})({a} - {q} * {d})"
            return self.stmt(nest, own, f"({own}){expr}", deps, lo, hi)
        if k == "model.clip":
            w = self.span(*res, *vals)
            a = self.ref(vals[0], w)
            clo = n.attr("lo")
            chi = n.attr("hi")
            clo = vals[0].lo if clo is None else max(clo, vals[0].lo)
            chi = vals[0].hi if chi is None else min(chi, vals[0].hi)
            return self.stmt(nest, own,
                             f"({own})({a} < ({w}){_lit(clo)} ? ({w}){_lit(clo)}"
                             f" : {a} > ({w}){_lit(chi)} ? ({w}){_lit(chi)} : "
                             f"{a})", deps, lo, hi)
        if k in ("model.max", "model.min"):
            w = self.span(*res, *vals)
            a, b = (self.ref(s, w) for s in vals)
            op = ">" if k == "model.max" else "<"
            return self.stmt(nest, own, f"({own})({a} {op} {b} ? {a} : {b})",
                             deps, lo, hi)
        if k == "model.where":
            w = self.span(*res, vals[1], vals[2])
            cond = vals[0].c(self.names)
            return self.stmt(nest, own, f"({own})({cond} ? "
                                        f"{self.ref(vals[1], w)} : "
                                        f"{self.ref(vals[2], w)})", deps, lo, hi)
        if k == "model.wrap":
            kind_, bits = n.attr("container")
            a = self.ref(vals[0], "int64_t")
            if bits >= 64:
                expr = a
            elif kind_ == "u":
                expr = f"({a} & {_lit((1 << bits) - 1)})"
            else:
                half = 1 << (bits - 1)
                expr = (f"((({a} + {_lit(half)}) & {_lit((1 << bits) - 1)}) "
                        f"- {_lit(half)})")
            return self.stmt(nest, own, f"({own}){expr}", deps, lo, hi)
        if k == "model.bits":
            w = self.span(*vals)
            return self.stmt(nest, own, f"({own})(({self.ref(vals[0], w)} >> "
                                        f"{n.attr('lo')}) & "
                                        f"{_lit((1 << n.attr('width')) - 1)})",
                             deps, lo, hi)
        raise Unsupported(f"{k} in the C backend")

    def cut(self, nest: _Nest, n, v: Sym, deps, lo, hi) -> Sym:
        """A modular value wrapped into its 64-bit container: its 64 bits,
        read as the container reads them."""
        kind_, bits = n.attr("container")
        if bits != 64:
            raise Unsupported(f"a value wider than 64 bits wrapped into "
                              f"{bits}")
        own = ctype_for(lo, hi)[1]
        a = self.ref(v, "uint64_t")
        # two's complement, as every compiler this builds with converts
        expr = a if kind_ == "u" else f"(int64_t){a}"
        return self.stmt(nest, own, f"({own}){expr}", deps, lo, hi)

    # -- the file --------------------------------------------------------------- #

    def fold_source(self, fname: str, out_ctype: str, folds) -> tuple:
        """A fold: every entry starts at the fold's start, every position of
        the frame folds its value into the entry its keys name, in raster
        order, as NumPy's `ufunc.at` does; then each entry is wrapped into
        the result's container. One thread: the entries are shared."""
        c = self.c
        f0 = folds[0][0]
        reach = [self.find(a) for a in f0.value.type.axes]       # entries
        frame = []
        for o in f0.operands:
            for a in o.type.axes:
                if a and self.find(a) not in frame:
                    frame.append(self.find(a))
        src_order = [self.find(a) for a in c.source.type.axes]
        frame.sort(key=lambda v: src_order.index(v) if v in src_order
                   else len(src_order))
        for i, v in enumerate(frame):
            self.names[v] = f"i{i}"
        entries = 1
        for a in reach:
            entries *= self.count(a)
        lanes = len(folds)
        lo = min(f.value.type.lo for f, _ in folds)
        hi = max(f.value.type.hi for f, _ in folds)
        acc_t = ctype_for(lo, hi)[1]
        ops_c = {"add": "+", "max": None, "min": None}

        def body(nest: _Nest):
            env = {v: _affine({v: 1}, 0, 0, self.count(v) - 1) for v in frame}
            for j, (f, _) in enumerate(folds):
                v, *keys = f.operands
                val = self.val(nest, v, env)
                flat = None
                for key, axis in zip(keys, f.attr("axes")):
                    k = self.val(nest, key, env)
                    n = self.count(self.find(axis))
                    flat = k.c(self.names) if flat is None else \
                        f"({flat}) * {n} + {k.c(self.names)}"
                at = f"(({flat or '0'}) * {lanes} + {j})"
                w = self.span(val, _literal(lo), _literal(hi))
                a, b = f"(({w})acc[{at}])", self.ref(val, w)
                op = f.attr("op")
                new = (f"{a} + {b}" if op == "add" else
                       f"({b} > {a} ? {b} : {a})" if op == "max" else
                       f"({b} < {a} ? {b} : {a})")
                # Each step is put back in the accumulator's container, as
                # NumPy's `at` puts it: where a step can leave it, the
                # entry's range is the container's and `acc_t` IS that
                # container, so the conversion is the wrap -- two's
                # complement, as every compiler this builds with converts.
                line = f"acc[{at}] = ({acc_t})({new});"
                if frame:
                    nest.bodies[nest.mode].append(line)
                else:
                    nest.top.append(line)

        nest = _Nest(frame)
        body(nest)
        if frame:
            xa, xb = 0, self.count(nest.inner)
            for coef, const, n in nest.bounds:
                xa = max(xa, -(const // coef))
                xb = min(xb, -((const - n) // coef))
            if nest.bounds and xa < xb:
                nest.interior = (xa, xb)
                nest.mode = "interior"
                body(nest)
        size = entries * lanes
        pre = [f"{acc_t} acc[{size}];",
               *(f"for (int64_t e = 0; e < {entries}; ++e) "
                 f"acc[e * {lanes} + {j}] = ({acc_t}){_lit(f.attr('init'))};"
                 for j, (f, _) in enumerate(folds))]
        post = []
        for j, (f, wrap) in enumerate(folds):
            x = f"(({acc_t})acc[e * {lanes} + {j}])"
            if wrap is not None:
                kind_, bits = wrap
                if bits < 64:
                    m = _lit((1 << bits) - 1)
                    x = (f"({x} & {m})" if kind_ == "u" else
                         f"((({x} + {_lit(1 << (bits - 1))}) & {m}) - "
                         f"{_lit(1 << (bits - 1))})")
            post.append(f"for (int64_t e = 0; e < {entries}; ++e) "
                        f"dst[e * {lanes} + {j}] = ({out_ctype}){x};")
        out_shape = [self.count(self.find(a)) if a else 1
                     for a in c.output.type.axes]
        src_shape = [self.count(a) for a in c.source.type.axes]
        text = self.assemble(nest, fname, out_ctype, src_shape, whole=True,
                             pre=pre, post=post)
        return text, out_shape, src_shape, ()

    def source(self, fname: str, out_ctype: str) -> tuple:
        c = self.c
        from .lower import _folds
        folds = _folds(c.output)
        if folds is not None:
            self.fold = True
            return self.fold_source(fname, out_ctype, folds)
        # A window registers place keeps the source's frame and drops what
        # lies outside it: the frame is computed whole, and the caller keeps
        # the window -- by the registers that place it, as NumPy's crop does.
        output, window = c.output, ()
        if output.node.kind == "model.window":
            output, *place = output.node.operands
            if any(v.node.kind != "model.param" for v in place):
                raise Unsupported("a window placed by computed values in the "
                                  "C backend")
            window = tuple(v.node.attr("name") for v in place)
        out_shape = [self.count(a) if a else 1 for a in
                     (self.find(a) if a else None for a in output.type.axes)]
        # a C-order reshape moves nothing: the value it reads is computed in
        # its own layout, which is the reshape's, without the index math
        output = self._unreshaped(output)
        out_axes = [self.find(a) if a else None for a in output.type.axes]
        real = [a for a in out_axes if a is not None]
        long_ = [a for a in real if self.count(a) > UNROLL_MAX]
        loops = [a for a in real if not long_ or self.count(a) > UNROLL_MAX]
        unrolled = [a for a in real if a not in loops]
        for i, v in enumerate(loops):
            self.names[v] = f"i{i}"
        buf_shape = [self.count(a) if a else 1 for a in out_axes]
        src_shape = [self.count(a) for a in c.source.type.axes]

        def body(nest: _Nest):
            combos = [{}]
            for u in unrolled:
                combos = [dict(e, **{u: _literal(i)}) for e in combos
                          for i in range(self.count(u))]
            for combo in combos:
                env = {v: _affine({v: 1}, 0, 0, self.count(v) - 1)
                       for v in loops}
                env.update(combo)
                got = self.val(nest, output, env)
                off = None
                for a, n in zip(out_axes, buf_shape):
                    term = "0" if a is None else env[a].c(self.names)
                    off = term if off is None else f"({off}) * {n} + {term}"
                line = (f"dst[{off}] = ({out_ctype})"
                        f"{got.c(self.names)};")
                if loops:
                    nest.bodies[nest.mode].append(line)
                else:
                    nest.top.append(line)

        nest = _Nest(loops)
        body(nest)
        if loops:
            n_in = self.count(nest.inner)
            xa, xb = 0, n_in
            for coef, const, n in nest.bounds:
                xa = max(xa, -(const // coef))           # ceil(-const / coef)
                xb = min(xb, -((const - n) // coef))     # ceil((n-const)/coef)
            if nest.bounds and xa < xb:
                nest.interior = (xa, xb)
                nest.mode = "interior"
                body(nest)
        text = self.assemble(nest, fname, out_ctype, src_shape)
        return text, out_shape, src_shape, window

    def _unreshaped(self, output: Value) -> Value:
        """The value an output reads through C-order reshapes -- an access
        whose index maps each position to the same flat offset of the value
        it reads, checked on every position -- or the output itself."""
        from .evaluate import _place, evaluate
        g = self.c.g
        while output.node.kind == "model.access":
            buf, *idx = output.node.operands
            if any(n.kind in ("model.param", "model.source", "model.table")
                   for i in idx for n in walk([i])):
                return output
            have = [self.find(a) for a in output.type.axes if a is not None]
            shape = [self.count(a) if a else 1 for a in buf.type.axes]
            size = int(np.prod([self.count(a) for a in have])) if have else 1
            if int(np.prod(shape)) != size or not have:
                return output
            vals = evaluate(g, idx, self.c.source, None, {})
            coords = [np.broadcast_to(
                _place(g, v, i.type.axes, tuple(have)) if i.type.axes
                else np.asarray(v), [self.count(a) for a in have])
                for v, i in zip(vals, idx)]
            flat = np.ravel_multi_index(tuple(np.asarray(c, np.int64)
                                              for c in coords), shape)
            if not np.array_equal(flat.ravel(), np.arange(size)):
                return output
            output = buf
        return output

    def assemble(self, nest: _Nest, fname, out_ctype, src_shape,
                 whole: bool = False, pre=(), post=()) -> str:
        """The function: `whole` runs every loop over its whole count --
        a fold, which no thread shares -- with `pre` before the loops and
        `post` after them."""
        src_t = ctype_for(self.c.source.type.lo, self.c.source.type.hi)[1]
        ind = "    "
        lines = [f"{ind}{s}" for s in [*pre, *nest.top]]
        loops = nest.loops

        def inner_loops(depth: int) -> list:
            pad = ind * (depth + 2)
            v = loops[-1]
            name = self.names[v]
            n = self.count(v)
            lo_, hi_ = ("lo", "hi") if len(loops) == 1 and not whole \
                else ("0", str(n))
            edge = [pad + ind + s for s in nest.bodies["edge"]]
            if nest.interior is None:
                return [f"{pad}for (int64_t {name} = {lo_}; {name} < {hi_}; "
                        f"++{name}) {{", *edge, f"{pad}}}"]
            xa, xb = nest.interior
            mid = [pad + ind + s for s in nest.bodies["interior"]]
            clip = (lambda e: f"(({e}) < lo ? lo : ({e}) > hi ? hi : ({e}))") \
                if len(loops) == 1 and not whole else (lambda e: e)
            a, b = clip(str(xa)), clip(str(xb))
            return [f"{pad}for (int64_t {name} = {lo_}; {name} < {a}; ++{name}) {{",
                    *edge, f"{pad}}}",
                    f"{pad}for (int64_t {name} = {a}; {name} < {b}; ++{name}) {{",
                    *mid, f"{pad}}}",
                    f"{pad}for (int64_t {name} = {b}; {name} < {hi_}; ++{name}) {{",
                    *edge, f"{pad}}}"]

        if loops:
            close = []
            for d, v in enumerate(loops[:-1]):
                pad = ind * (d + 1)
                name = self.names[v]
                lo_, hi_ = ("lo", "hi") if d == 0 and not whole \
                    else ("0", str(self.count(v)))
                lines.append(f"{pad}for (int64_t {name} = {lo_}; {name} < {hi_};"
                             f" ++{name}) {{")
                lines += [pad + ind + s for s in nest.at[d]]
                close.append(f"{pad}}}")
            lines += inner_loops(len(loops) - 2 + 0)
            lines += reversed(close)
        else:
            lines = [f"{ind}if (lo > 0) return;"] + lines
        if whole:
            lines = [f"{ind}if (lo > 0) return;"] + lines
        lines += [f"{ind}{s}" for s in post]
        body = "\n".join(lines)
        decls = "\n".join(self.decls)
        return f"""/* generated by np2hw (C backend); do not edit */
#include <stdbool.h>
#include <stdint.h>

{decls}

void {fname}(const {src_t}* restrict src, {out_ctype}* restrict dst,
        const int64_t* restrict regs, int64_t lo, int64_t hi) {{
    (void)regs; (void)lo; (void)hi; (void)src;
{body}
}}
"""


_FOLD = {
    "model.add": lambda n, a, b: a + b,
    "model.sub": lambda n, a, b: a - b,
    "model.mul": lambda n, a, b: a * b,
    "model.neg": lambda n, a: -a,
    "model.abs": lambda n, a: abs(a),
    "model.max": lambda n, a, b: max(a, b),
    "model.min": lambda n, a, b: min(a, b),
    "model.shr": lambda n, a: a >> n.attr("amount"),
    "model.floordiv": lambda n, a: a // n.attr("divisor"),
    "model.mod": lambda n, a: a % n.attr("divisor"),
}


# --------------------------------------------------------------------------- #
# Building and calling
# --------------------------------------------------------------------------- #

def default_flags() -> tuple:
    """-O3 for this machine; on x86 the compiler may use 512-bit vectors
    (measured 1.2-1.4 against 0.7-0.85 Gpx/s a core for a 5 x 5 cone)."""
    arch = platform.machine()
    if arch in ("x86_64", "AMD64"):
        return ("-O3", "-std=c11", "-march=native",
                "-mprefer-vector-width=512")
    return ("-O3", "-std=c11", "-mcpu=native")


def _out_dtype(container, lo, hi) -> np.dtype:
    kind, bits = container
    if kind == "b":
        return np.dtype(bool)
    if kind in ("u", "i"):
        size = next(b for b in (8, 16, 32, 64) if b >= bits)
        return np.dtype(f"{kind}{size // 8}")
    return ctype_for(lo, hi)[0]


@dataclass
class Kernel:
    """A model compiled for a CPU at one frame size: call it on a frame and
    its registers. A frame of another size compiles the model again."""
    fn: object
    params: tuple
    image: object
    channels: int
    flags: tuple
    build: Path
    lib: ctypes.CDLL = None
    fname: str = ""
    regs: list = field(default_factory=list)        # (name, size)
    defaults: dict = field(default_factory=dict)
    src_dtype: np.dtype = None
    src_shape: tuple = ()
    out_dtype: np.dtype = None
    out_shape: tuple = ()
    outer: int = 1
    c_path: Path = None
    window: tuple = ()          # registers: row, column start; height, width
    _others: dict = field(default_factory=dict)

    def _for(self, shape) -> "Kernel":
        if tuple(shape) == self.src_shape:
            return self
        if tuple(shape) not in self._others:
            from dataclasses import replace
            img = self.image
            dims = list(shape)
            if self.channels > 1:
                dims.pop()
            frames = dims.pop(0) if len(dims) == 3 else 0
            h, w = dims
            image = replace(img, width=w, height=h, frames=frames)
            self._others[tuple(shape)] = _build(
                self.fn, image, self.params, self.channels, self.flags,
                self.build)
        return self._others[tuple(shape)]

    def __call__(self, img, threads: int = 1, out=None, **regs):
        img = np.asarray(img)
        k = self._for(img.shape)
        img = np.ascontiguousarray(img, dtype=k.src_dtype)
        if out is None:
            out = np.empty(k.out_shape, k.out_dtype)
        P = []
        for name, size in k.regs:
            val = regs.get(name, k.defaults.get(name, 0))
            vals = np.broadcast_to(np.asarray(val, dtype=np.int64), (size,)) \
                if size > 1 else [int(np.asarray(val))]
            P.extend(int(x) for x in vals)
        P = np.array(P or [0], np.int64)
        dst = out.view(np.uint8) if out.dtype == bool else out
        f = getattr(k.lib, k.fname)
        args = (img.ctypes.data, dst.ctypes.data, P.ctypes.data)
        if threads <= 1 or k.outer <= 1:
            f(*args, 0, k.outer)
        else:
            from concurrent.futures import ThreadPoolExecutor
            cuts = np.linspace(0, k.outer, threads + 1).astype(int)
            with ThreadPoolExecutor(threads) as ex:   # ctypes drops the GIL
                list(ex.map(lambda i: f(*args, int(cuts[i]), int(cuts[i + 1])),
                            range(threads)))
        if k.window:
            y0, x0, h, w = (int(np.asarray(regs.get(n, k.defaults.get(n, 0))))
                            for n in k.window)
            frames = (slice(None),) if getattr(k.image, "frames", 0) else ()
            return out[(*frames, slice(y0, None), slice(x0, None))][
                (*frames, slice(None, h), slice(None, w))]
        return out


def emit(c: Canonical, fname: str, out_ctype: str) -> tuple:
    """(C text, registers [(name, size)], output shape, source shape,
    count of the outermost loop, the registers placing a window the output
    keeps -- its row and column starts, its height and width -- or ())."""
    p = _Printer(c)
    text, out_shape, src_shape, window = p.source(fname, out_ctype)
    loops = [a for a in c.output.type.axes if a is not None]
    outer = 1
    if p.names and not getattr(p, "fold", False):
        first = next(v for v, nm in p.names.items() if nm == "i0")
        outer = c.g.vars.count(first)
    return text, p.regs, out_shape, src_shape, outer, window


def _defaults(c: Canonical) -> dict:
    out = {}
    for name in c.registers:
        obj = c.g.symbols[name]["object"]
        d = getattr(obj, "default", None)
        if d is not None:
            out[name] = d
    return out


def _build(fn, image, params, channels, flags, build) -> Kernel:
    from ..ir import Params
    tparams = params
    if len(params) == 1 and isinstance(params[0], Params):
        tparams = (params[0].trace_view(),)
    traced = trace(fn, image, *tparams, channels=channels)
    c = canonicalise(traced)
    t = c.output.type
    out_dtype = _out_dtype(traced.output.type.container, t.lo, t.hi)
    out_c = "uint8_t" if out_dtype == bool else _CNAME[out_dtype]
    fname = f"np2hw_{getattr(fn, '__name__', 'model')}"
    text, regs, out_shape, src_shape, outer, window = emit(c, fname, out_c)
    key = hashlib.sha1((text + " ".join(flags)).encode()).hexdigest()[:12]
    d = Path(build)
    d.mkdir(parents=True, exist_ok=True)
    src, lib = d / f"{fname}_{key}.c", d / f"{fname}_{key}.so"
    src.write_text(text)
    if not lib.exists():
        cc = os.environ.get("CC", "cc")
        tmp = lib.with_suffix(f".{os.getpid()}.so")
        subprocess.run([cc, *flags, "-shared", "-fPIC", str(src), "-o",
                        str(tmp)], check=True)
        os.replace(tmp, lib)
    so = ctypes.CDLL(str(lib.resolve()))
    f = getattr(so, fname)
    f.restype = None
    f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2
    s = c.source.type
    return Kernel(fn, tuple(params), image, channels, tuple(flags), d, so,
                  fname, regs, _defaults(c), ctype_for(s.lo, s.hi)[0],
                  tuple(src_shape), out_dtype, tuple(out_shape), outer, src,
                  window)


def compile_c(fn, image, *params, channels: int = 1, build="build/c",
              flags=None) -> Kernel:
    """Trace `fn` on `image` (an Image2D) with its registers, print it as C,
    build a shared library and load it. The kernel computes what `fn`
    computes, bit for bit, into an array of the dtype NumPy gives."""
    flags = default_flags() if flags is None else tuple(flags)
    return _build(fn, image, params, channels, flags, build)
