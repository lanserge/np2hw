# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Evaluate a model-level graph on NumPy arrays.

Each kind is the NumPy operation it records, run by NumPy -- so evaluating
a traced graph on the model's own input must give exactly what running the
model gives. That equality is how the tracer is tested: it proves the graph
records everything the model computed, before any hardware exists.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph, Value, walk


def _dtype(c):
    kind, bits = c
    if kind == "b":
        return np.dtype(bool)
    return np.dtype(f"{'u' if kind == 'u' else 'i'}{bits // 8}")


def _place(g: Graph, arr, have: tuple, want: tuple):
    """An array over the axes `have`, reshaped to broadcast into `want`: its
    size-1 dimensions dropped, its variables put in `want`'s order, a size-1
    dimension wherever `want` has a variable it lacks."""
    arr = np.asarray(arr)
    real = [g.vars.find(a) for a in have if a is not None]
    squeezed = arr.reshape([s for a, s in zip(have, arr.shape)
                            if a is not None])
    order = [real.index(w) for w in want if w in real]
    shape = [g.vars.count(w) if w in real else 1 for w in want]
    return np.transpose(squeezed, order).reshape(shape)


def _access(g, n, vals):
    buf, idx = vals[0], vals[1:]
    want = tuple(g.vars.find(a) for a in n.value.type.axes)
    placed = [_place(g, v, o.type.axes, want) if o.type.axes else v
              for v, o in zip(idx, n.operands[1:])]
    return np.asarray(buf)[tuple(placed)]


def _pad(g, n, vals):
    mode = "edge" if n.attr("mode") == "edge" else "constant"
    return np.pad(vals[0], n.attr("widths"), mode=mode)


def _expand(g, n, vals):
    return np.reshape(vals[0], [g.vars.count(a) if a else 1
                                for a in n.attr("axes")])


def _canvas(g, n, vals):
    template, rest = vals[0], vals[1:]
    tax = n.operands[0].type.axes
    per = 1 + len(tax)
    out = (np.zeros_like if n.attr("zeroed") else np.empty_like)(template)
    for i in range(n.attr("planes")):
        value = rest[i * per]
        idx_vals = rest[i * per + 1:(i + 1) * per]
        idx_ops = n.operands[1 + i * per + 1:1 + (i + 1) * per]
        want = tuple(g.vars.find(a) for a in n.operands[1 + i * per].type.axes)
        placed = [_place(g, v, o.type.axes, want) if o.type.axes else v
                  for v, o in zip(idx_vals, idx_ops)]
        out[tuple(placed)] = value
    return out


def _wrap(g, n, vals):
    """A NumPy container's wrap, on exact integers."""
    kind, bits = n.attr("container")
    x = np.asarray(vals[0], dtype=np.int64)
    m = 1 << bits
    return x % m if kind == "u" else (x + m // 2) % m - m // 2


def _bits(g, n, vals):
    x = np.asarray(vals[0], dtype=np.int64)
    return (x >> n.attr("lo")) & ((1 << n.attr("width")) - 1)


def _select(g, n, vals):
    sizes = n.attr("sizes")
    keys, choices = vals[:len(sizes)], vals[len(sizes):]
    flat = 0
    for k, size in zip(keys, sizes):
        flat = flat * size + np.asarray(k, dtype=np.int64)
    out = np.zeros(np.broadcast_shapes(np.shape(flat),
                                       *(np.shape(c) for c in choices)),
                   dtype=np.int64)
    for j, c in enumerate(choices):
        out = np.where(flat == j, c, out)
    return out


_FOLD = {"add": np.add, "max": np.maximum, "min": np.minimum}


def _scatter(g, n, vals):
    """`ufunc.at` into an accumulator starting at `init`, or the value
    reduced whole: the value and its keys broadcast as NumPy broadcasts
    them, every position folded into the entry its keys name."""
    c = n.attr("container")
    exact = c == ("x", 0)
    # the accumulator NumPy keeps, each step wrapped into it
    into = dict(n.attrs).get("into")
    dtype = _dtype(into or c) if into or not exact else np.dtype(np.int64)
    fold = _FOLD[n.attr("op")]
    ops = n.operands
    width = max(len(o.type.axes) for o in ops)
    want = []
    for dims in zip(*((None,) * (width - len(o.type.axes)) + tuple(o.type.axes)
                      for o in ops)):
        real = [g.vars.find(d) for d in dims if d is not None]
        want.append(real[0] if real else None)
    want = tuple(w for w in want if w is not None)
    shape = tuple(g.vars.count(w) for w in want)
    placed = [np.broadcast_to(_place(g, v, o.type.axes, want) if o.type.axes
                              else np.asarray(v), shape)
              for v, o in zip(vals, ops)]
    value = np.asarray(placed[0])
    acc_shape = tuple(g.vars.count(a) for a in n.attr("axes"))
    acc = np.full(acc_shape or (1,), n.attr("init"), dtype=dtype)
    keys = (tuple(np.asarray(k, dtype=np.int64) for k in placed[1:])
            if acc_shape else (np.zeros(value.shape, np.int64),))
    fold.at(acc, keys, value)
    out = acc if acc_shape else acc[0]
    return np.asarray(out).astype(np.int64) if exact else out


def _stack(g, n, vals):
    return np.stack(vals, axis=n.attr("axis"))


_EVAL = {
    "model.add": lambda g, n, v: v[0] + v[1],
    "model.sub": lambda g, n, v: v[0] - v[1],
    "model.mul": lambda g, n, v: v[0] * v[1],
    "model.and": lambda g, n, v: v[0] & v[1],
    "model.or": lambda g, n, v: v[0] | v[1],
    "model.xor": lambda g, n, v: v[0] ^ v[1],
    "model.max": lambda g, n, v: np.maximum(v[0], v[1]),
    "model.min": lambda g, n, v: np.minimum(v[0], v[1]),
    "model.neg": lambda g, n, v: -v[0],
    "model.abs": lambda g, n, v: np.absolute(v[0]),
    "model.shr": lambda g, n, v: v[0] >> n.attr("amount"),
    "model.floordiv": lambda g, n, v: v[0] // n.attr("divisor"),
    "model.mod": lambda g, n, v: v[0] % n.attr("divisor"),
    "model.clip": lambda g, n, v: v[0].clip(n.attr("lo"), n.attr("hi")),
    "model.lt": lambda g, n, v: v[0] < v[1],
    "model.gt": lambda g, n, v: v[0] > v[1],
    "model.le": lambda g, n, v: v[0] <= v[1],
    "model.ge": lambda g, n, v: v[0] >= v[1],
    "model.eq": lambda g, n, v: v[0] == v[1],
    "model.ne": lambda g, n, v: v[0] != v[1],
    "model.where": lambda g, n, v: np.where(v[0], v[1], v[2]),
    "model.astype": lambda g, n, v: np.asarray(v[0]).astype(
        _dtype(n.attr("container"))),
    "model.access": _access,
    "model.pad": _pad,
    "model.expand": _expand,
    "model.canvas": _canvas,
    "model.stack": _stack,
    "model.wrap": _wrap,
    "model.bits": _bits,
    "model.select": _select,
    "model.scatter": _scatter,
    "model.var": lambda g, n, v: np.arange(g.vars.count(n.attr("name"))),
}


def evaluate(g: Graph, outputs: list[Value], source: Value, image,
             params: dict) -> list:
    """The outputs' values, with `image` standing for the source and
    `params` giving each register's value."""
    values: dict[int, object] = {}
    for n in walk(outputs):
        exact = n.attr("container") == ("x", 0)
        if n is source.node:
            out = np.asarray(image).astype(np.int64) if exact else image
        elif n.kind == "model.const":
            c = n.attr("container")
            out = n.attr("value") if c in (("w", 0), ("x", 0)) else \
                _dtype(c).type(n.attr("value"))
        elif n.kind == "model.param":
            out = int(params[n.attr("name")])
        elif n.kind == "model.table":
            name = n.attr("name")
            out = np.asarray(params[name] if name in params
                             else g.symbols[name]["object"].data)
            if exact:
                out = out.astype(np.int64)
        else:
            out = _EVAL[n.kind](g, n, [values[id(v.node)][v.index]
                                       if len(v.node.results) > 1
                                       else values[id(v.node)]
                                       for v in n.operands])
        values[id(n)] = out
    return [values[id(v.node)] for v in outputs]
