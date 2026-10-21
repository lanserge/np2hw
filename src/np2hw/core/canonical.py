# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Canonicalising: the faithful record, rebuilt in one form for each thing.

Every rewrite here is exact node by node (docs/one-graph.md, "Lowering"):

- the graph is rebuilt INTERNED, so identical nodes become one;
- arithmetic becomes EXACT -- its container exact integers -- and each wrap
  NumPy's containers imposed becomes an explicit `wrap` node, placed
  exactly where the exact range leaves the container and nowhere else;
  `astype` disappears into those wraps; a canvas plane is wrapped into the
  template's container, as NumPy's assignment casts it;
- every access index is replaced by the canonical node of its affine form,
  so equal indexes are one node and its users are every access through it;
- a low-bit mask or a shift of non-negative DATA becomes a bit access; of
  an INDEX it stays index math -- `% 2**k`, `// 2**k` -- which is a split.
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph import COMMUTATIVE, Graph, Node, Value, kind, walk
from .index import affine, canonical
from .model import EXACT
from .trace import Traced

_LEAVES_EXACT = ("model.source", "model.const", "model.table")


@dataclass
class Canonical:
    g: Graph
    source: Value
    output: Value
    image: object
    # the registers and register arrays by name, in the order the module
    # lists their ports (registers_in_order)
    registers: tuple = ()


def registers_in_order(output: Value) -> tuple:
    """The model's registers and register arrays by name, in the order the
    generated module lists their ports, so a core's interface depends on
    what the model computes and not on the order its statements ran in: met
    walking the FAITHFUL record from the output, depth first, each node's operands in the order the model
    wrote them -- a table read by its index -- each name at its first
    meeting. Read from the record, not the canonical graph, which orders
    commutative operands for recognising. (A register's ADDRESS is not
    this: it is the block's own declaration's.)"""
    seen: dict = {}
    visited: set = set()
    stack = [output.node]
    while stack:
        n = stack.pop()
        if id(n) in visited:
            continue
        visited.add(id(n))
        if n.kind in ("model.param", "model.table"):
            seen.setdefault(n.attr("name"), None)
        stack.extend(reversed([o.node for o in n.operands]))
    return tuple(seen)


def _needs_wrap(v: Value, c) -> bool:
    kind, bits = c
    if kind not in ("u", "i"):
        return False
    lo = -(1 << (bits - 1)) if kind == "i" else 0
    hi = (1 << (bits - (1 if kind == "i" else 0))) - 1
    return not (lo <= v.type.lo and v.type.hi <= hi)


class _Rebuild:
    def __init__(self, t: Traced):
        self.t = t
        self.g = Graph(intern=True, locations=False)
        self.g.vars = t.g.vars                    # one set of variables
        self.g.symbols = {k: dict(v) for k, v in t.g.symbols.items()}
        self.map: dict[int, Value] = {}

    def v(self, old: Value) -> Value:
        return self.map[id(old.node)]

    def wrap(self, v: Value, c) -> Value:
        return self.g.op("model.wrap", v, container=c) if _needs_wrap(v, c) \
            else v

    def index(self, v: Value) -> Value:
        """The canonical node of an index -- when it ranges over the same
        variables. A merge read through `f // W` and `f % W` stays over `f`:
        writing it in the split parts would change what the access spans."""
        a = affine(self.g, v)
        if a is None:
            return v
        c = canonical(self.g, a)
        find = self.g.vars.find
        want = tuple(find(x) if x else None for x in v.type.axes)
        if {x for x in want if x} != {find(x) for x in c.type.axes if x}:
            return v
        have = tuple(find(x) if x else None for x in c.type.axes)
        if have != want:                   # keep the index's layout: (H, 1)
            c = self.g.op("model.expand", c, axes=want)
        return c

    def node(self, n: Node) -> Value:
        g, k = self.g, n.kind
        attrs = dict(n.attrs)
        ops = [self.v(o) for o in n.operands]
        if k in _LEAVES_EXACT:
            attrs["container"] = EXACT
            out = g.node(k, ops, attrs).value
            if k == "model.table":
                self.g.symbols[attrs["name"]]["read"] = out
            return out
        if k == "model.param":
            out = g.node(k, ops, attrs).value
            self.g.symbols[attrs["name"]]["read"] = out
            return out
        if k == "model.var":
            return g.node(k, ops, attrs).value
        if k == "model.astype":
            return self.wrap(ops[0], attrs["container"])
        if k == "model.access":
            return g.op("model.access", ops[0], *(self.index(i)
                                                  for i in ops[1:]))
        if k == "model.canvas":
            tc = n.operands[0].type.container
            per = 1 + len(n.operands[0].type.axes)
            rest = ops[1:]
            planes = []
            for i in range(attrs["planes"]):
                value, idx = rest[i * per], rest[i * per + 1:(i + 1) * per]
                planes += [self.wrap(value, tc), *(self.index(x) for x in idx)]
            return g.node(k, [ops[0], *planes], attrs).value
        if k == "model.and" and self._low_mask(n) is not None:
            return self._mask(ops, self._low_mask(n))
        if k == "model.shr" and ops[0].type.lo >= 0:
            return self._shift(ops[0], attrs["amount"])
        out = g.node(k, ops, attrs).value
        return self.wrap(out, n.value.type.container)

    # -- masks and shifts -----------------------------------------------------

    def _low_mask(self, n: Node):
        """k for `x & (2**k - 1)` with x known non-negative, else None."""
        for i in (0, 1):
            c, x = n.operands[i], n.operands[1 - i]
            if c.node.kind == "model.const" and x.type.lo >= 0:
                m = c.node.attr("value")
                if m > 0 and m & (m + 1) == 0:
                    return (1 - i, m.bit_length())
        return None

    def _mask(self, ops, found):
        which, width = found
        x = ops[which]
        if affine(self.g, x) is not None:          # an index: a remainder
            return self.index(self.g.op("model.mod", self.index(x),
                                        divisor=1 << width))
        return self.g.op("model.bits", x, lo=0, width=width)

    def _shift(self, x: Value, k: int) -> Value:
        if affine(self.g, x) is not None:          # an index: a split
            return self.index(self.g.op("model.floordiv", x, divisor=1 << k))
        width = max(1, x.type.hi.bit_length() - k)
        return self.g.op("model.bits", x, lo=k, width=width)


# --------------------------------------------------------------------------- #
# Selects by remainders
# --------------------------------------------------------------------------- #

def _key(g, v: Value):
    """(key, size) when `v` is a remainder key -- a POSITION mod k -- else
    None. A register's remainder is no position: the same at every pixel,
    a select on it is a select, not a lattice."""
    n = v.node
    if n.kind == "model.mod":
        form = affine(g, n.operands[0])
        if form is not None and form.terms:
            return v, n.attr("divisor")
    return None


def _const_value(v: Value):
    return v.node.attr("value") if v.node.kind == "model.const" else None


class _Selects:
    """Rules, each exact: a `where` on a remainder is a select (A); a canvas
    whose planes tile the lattice is a select (B); a select over the same
    operation is that operation over selects (C); an access through a
    buffer's own variables is the buffer; an access of a computed value is
    that value computed at the index (D)."""

    def __init__(self, c: Canonical):
        self.c, self.g = c, c.g
        self.map: dict[int, Value] = {}

    def run(self) -> Canonical:
        for n in walk([self.c.output]):
            self.map[id(n)] = self.node(n)
        return Canonical(self.g, self.map.get(id(self.c.source.node),
                                              self.c.source),
                         self.map[id(self.c.output.node)], self.c.image,
                         self.c.registers)

    def node(self, n: Node) -> Value:
        g = self.g
        ops = [self.map[id(o.node)] for o in n.operands]
        if n.kind == "model.access":
            return self.access(ops)
        if n.kind == "model.where":
            got = self.where(ops)
            if got is not None:
                return got
        if n.kind == "model.canvas":
            got = self.canvas(n, ops)
            if got is not None:
                return got
        return g.node(n.kind, ops, dict(n.attrs)).value

    def access(self, ops) -> Value:
        buf, idx = ops[0], ops[1:]
        vars_ = self.g.vars
        find = vars_.find
        if all(i.node.kind == "model.var" and a is not None
               and find(i.node.attr("name")) == find(a)
               for i, a in zip(idx, buf.type.axes)):
            return buf
        # Every axis read whole, in order, through a variable of its own
        # count that is none of the buffer's other axes: the same positions,
        # so the same variable -- two variables paired on one position with
        # equal counts are unified, as broadcasting pairs them (decision 3)
        # -- and the access is the buffer.
        axes = [find(a) if a is not None else None for a in buf.type.axes]
        names = [find(i.node.attr("name")) if i.node.kind == "model.var"
                 else None for i in idx]
        if (len(idx) == len(axes) and None not in axes and None not in names
                and len(set(names)) == len(names)
                and all(vars_.count(v) == vars_.count(a)
                        for v, a in zip(names, axes))
                and all(v == a or v not in axes for v, a in zip(names, axes))):
            for v, a in zip(names, axes):
                vars_.unify(v, a)
            return buf
        # D: a computed value read at an index is the value computed there
        # -- its own variables replaced by the index -- so a slice of a
        # result is a window over the source, as composing filters is
        if buf.node.kind not in _BUFFERS and None not in axes \
                and len(idx) == len(axes) and all(map(_offset, idx)):
            got = self._at(buf, dict(zip(axes, idx)))
            if got is not None:
                return got
        return self.g.op("model.access", buf, *idx)

    def _at(self, v: Value, sub: dict):
        """`v` with its variables replaced by the values in `sub`, through
        operations that act position by position -- or None where one does
        not (an operation whose attributes name an axis)."""
        memo: dict[int, Value] = {}
        g, find = self.g, self.g.vars.find

        def go(x: Value):
            n = x.node
            if id(n) in memo:
                return memo[id(n)]
            if n.kind == "model.const":
                out = x
            elif n.kind in _BUFFERS:
                # a buffer read whole is read at its own variables: replaced,
                # it is read at the index -- the source used as a value and
                # sliced on time is its frames at those offsets, not the live
                # one at every offset
                axes = [find(a) if a is not None else None
                        for a in x.type.axes]
                if any(a in sub for a in axes if a is not None) \
                        and None not in axes:
                    out = self.access([x, *(
                        sub[a] if a in sub else g.op("model.var", name=a)
                        for a in axes)])
                else:
                    out = x                  # re-indexed only by its access
            elif n.kind == "model.var":
                out = sub.get(find(n.attr("name")), x)
            elif n.kind == "model.access":
                # the buffer an access reads is read at the access's own
                # index, which is what is replaced: re-indexed as a value
                # too, it would be read at an index twice over
                buf, *idx = n.operands
                ops = [buf if buf.node.kind in _BUFFERS else go(buf),
                       *(go(o) for o in idx)]
                out = None if None in ops else self.access(ops)
            elif n.kind in _POINTWISE:
                ops = [go(o) for o in n.operands]
                out = None if None in ops else \
                    g.node(n.kind, ops, dict(n.attrs)).value
            else:
                out = None
            memo[id(n)] = out
            return out

        return go(v)

    # -- A: where on a remainder ---------------------------------------------

    def where(self, ops):
        cond, x, y = ops
        if cond.node.kind != "model.eq":
            return None
        a, b = cond.node.operands
        for key_side, const_side in ((a, b), (b, a)):
            k = _key(self.g, key_side)
            v = _const_value(const_side)
            if k and v is not None and k[1] == 2 and v in (0, 1):
                vals = [x, y] if v == 0 else [y, x]
                return self.select([k[0]], [2], vals)
        return None

    # -- building selects: flatten, then push through (C) ------------------

    def select(self, keys, sizes, vals) -> Value:
        # nested selects on one key set flatten into one
        inner = [v.node for v in vals]
        if all(n.kind == "model.select" for n in inner):
            ik = [tuple(id(o.node) for o in n.operands[:len(n.attr("sizes"))])
                  for n in inner]
            if len(set(ik)) == 1:
                n0 = inner[0]
                k0 = list(n0.operands[:len(n0.attr("sizes"))])
                flat = [o for n in inner
                        for o in n.operands[len(n.attr("sizes")):]]
                return self.select(list(keys) + k0,
                                   list(sizes) + list(n0.attr("sizes")), flat)
        return self.push(keys, sizes, vals)

    def push(self, keys, sizes, vals) -> Value:
        """A select over one operation with one set of attributes is that
        operation over selects of its operands: exact, the select being
        pointwise; it stops where the values are different operations."""
        g = self.g
        first = vals[0]
        if all(v.node is first.node and v.index == first.index for v in vals):
            return first
        nodes = [v.node for v in vals]
        k0 = nodes[0]
        same = all(n.kind == k0.kind and n.attrs == k0.attrs
                   and len(n.operands) == len(k0.operands) for n in nodes)
        leafy = k0.kind in ("model.const", "model.param", "model.table",
                            "model.source", "model.var")
        if same and not leafy and k0.operands:
            rows = [list(n.operands) for n in nodes]
            if COMMUTATIVE in kind(k0.kind).traits and len(k0.operands) == 2:
                # align each value's operands with the first's, so an operand
                # they share stands in one position -- hash-consing orders a
                # commutative operation's operands by structure, which puts a
                # shared pixel first in one plane and second in the next
                same_v = lambda a, b: a.node is b.node and a.index == b.index
                ref = rows[0]
                for r in rows[1:]:
                    if not (same_v(r[0], ref[0]) or same_v(r[1], ref[1])) and \
                            (same_v(r[1], ref[0]) or same_v(r[0], ref[1])):
                        r.reverse()
            ops = [self.push(keys, sizes, [r[i] for r in rows])
                   for i in range(len(k0.operands))]
            return g.node(k0.kind, ops, dict(k0.attrs)).value
        return g.node("model.select", [*keys, *vals],
                      {"sizes": tuple(sizes)}).value

    # -- B: a canvas whose planes tile the lattice ----------------------------

    def canvas(self, n: Node, ops):
        g, find = self.g, self.g.vars.find
        template, rest = ops[0], ops[1:]
        axes = [find(a) if a else None for a in template.type.axes]
        per = 1 + len(axes)
        planes = [(rest[i * per], rest[i * per + 1:(i + 1) * per])
                  for i in range(n.attr("planes"))]
        unset = object()
        steps, regs, cases = [None] * len(axes), [unset] * len(axes), []
        plane_vars = []
        for value, idx in planes:
            case, own = [], set()
            for j, i in enumerate(idx):
                form = affine(g, i)
                if form is None or len(form.terms) != 1:
                    return None
                (var, step), = form.terms
                if steps[j] not in (None, step):
                    return None
                steps[j] = step
                if len(form.regs) > 1 or (form.regs and step != 2):
                    return None
                reg = form.regs[0][0] if form.regs else None
                if regs[j] is not unset and regs[j] != reg:
                    return None          # every plane on an axis alike
                regs[j] = reg
                case.append(form.const % step)
                own.add(find(var))
            cases.append(tuple(case))
            plane_vars.append(own)
        sizes = tuple(steps)
        want = {c for c in _product(sizes)}
        if len(cases) != len(want) or set(cases) != want:
            return None                            # the planes do not tile
        # the keys: (axis + register) mod step, on the axis's own position
        keys = []
        for j, a in enumerate(axes):
            e = g.op("model.var", name=a)
            if regs[j] is not None:
                e = g.op("model.add", e, g.symbols[regs[j]]["read"])
            if len(axes) > 1:
                e = g.op("model.expand", e,
                         axes=tuple(a if k == j else None
                                    for k in range(len(axes))))
            keys.append(g.op("model.mod", e, divisor=sizes[j]))
        # each plane read through its own index is the plain position
        rewritten = {}
        for (value, idx), case, own in zip(planes, cases, plane_vars):
            sub = {id(i.node): g.op("model.var", name=axes[j])
                   for j, i in enumerate(idx)}
            new = self._substitute(value, sub, own)
            if new is None:
                return None
            rewritten[case] = new
        vals = [rewritten[c] for c in _product(sizes)]
        return self.select(keys, list(sizes), vals)

    def _substitute(self, v: Value, sub: dict, own: set):
        """`v` with each index node in `sub` replaced by the plain position --
        or None when the plane uses its own variables `own` anywhere else,
        which this rule does not rewrite."""
        memo: dict[int, Value] = {}
        g, find = self.g, self.g.vars.find

        def go(x: Value):
            n = x.node
            if id(n) in memo:
                return memo[id(n)]
            if n.kind == "model.access":
                buf = go(n.operands[0])
                idx = [sub[id(i.node)] if id(i.node) in sub else go(i)
                       for i in n.operands[1:]]
                out = None if buf is None or any(i is None for i in idx) \
                    else self.access([buf, *idx])
            elif n.kind == "model.var":
                out = None if find(n.attr("name")) in own else x
            else:
                ops = [go(o) for o in n.operands]
                out = None if any(o is None for o in ops) else \
                    g.node(n.kind, ops, dict(n.attrs)).value
            memo[id(n)] = out
            return out

        return go(v)


# what an access re-indexes rather than computes, and the operations that
# act position by position -- the ones rule D substitutes through
_BUFFERS = {"model.source", "model.pad", "model.table", "model.param"}
_POINTWISE = {"model.add", "model.sub", "model.mul", "model.floordiv",
              "model.mod", "model.shr", "model.and", "model.or", "model.neg",
              "model.abs", "model.eq", "model.ne", "model.lt", "model.le",
              "model.gt", "model.ge", "model.where", "model.clip",
              "model.astype", "model.bits", "model.wrap"}


def _offset(v: Value) -> bool:
    """An index that is a variable at a constant start: a shift, which
    rule D substitutes -- a stride is the canvas's to read."""
    n = v.node
    if n.kind == "model.var":
        return True
    if n.kind == "model.add":
        a, b = n.operands
        return (_offset(a) and b.node.kind == "model.const"
                or _offset(b) and a.node.kind == "model.const")
    return False


def _product(sizes):
    out = [()]
    for k in sizes:
        out = [c + (i,) for c in out for i in range(k)]
    return out


def canonicalise(t: Traced) -> Canonical:
    r = _Rebuild(t)
    for n in walk([t.output]):
        r.map[id(n)] = r.node(n)
    # a model may use only where its pixels are, never their values: its
    # source is still the stream it is given
    if id(t.source.node) not in r.map:
        r.map[id(t.source.node)] = r.node(t.source.node)
    c = Canonical(r.g, r.map[id(t.source.node)], r.map[id(t.output.node)],
                  t.image, registers_in_order(t.output))
    return _Selects(c).run()
