# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The tracer: a NumPy model, recorded as the model level of the graph.

The model runs once, on traced arrays, and every operation adds one node --
recording what the model did, in the operand order written, nothing
decided. The graph is a FAITHFUL record: it merges nothing (canonicalising
builds the shared graph from it), and each node carries the model's line.

A slice is index math and an access: `x[1:-1]` declares a variable of the
slice's length and reads the buffer through `var + 1`; a stride and a
register start are more index math (`2 * var + py`). `np.pad` is a larger
buffer with new variables for its axes. Constants and registers are
buffers with no axes -- a Python int and a register WEAK, a NumPy scalar
with its own dtype, as NumPy 2 has them.
"""
from __future__ import annotations

import numpy as np

from ..ir import Image2D, Param, PhaseRef
from .graph import BOOL, WEAK, Graph, GraphError, Value

# importing registers the model kinds
from . import model as _model  # noqa: F401


class Unsupported(NotImplementedError):
    """The model does something the tracer does not record yet."""


def _container_of(dtype) -> tuple:
    d = np.dtype(dtype)
    if d.kind == "b":
        return BOOL
    if d.kind not in "ui":
        raise Unsupported(f"a {d} value: np2hw traces integers")
    return ("u" if d.kind == "u" else "i", d.itemsize * 8)


def image_container(bits: int, signed: bool) -> tuple:
    """The container a NumPy oracle holds an image of `bits` in."""
    size = next(b for b in (8, 16, 32, 64) if b >= bits)
    return ("i" if signed else "u", size)


class Tracer:
    """One trace: its graph, and its registers read once each."""

    def __init__(self):
        self.g = Graph(intern=False, locations=True)

    def const(self, value: int, container=WEAK) -> Value:
        return self.g.op("model.const", value=int(value), container=container)

    def register(self, p: Param) -> Value:
        """A register: one named declaration, one read."""
        sym = self.g.symbols.get(p.name)
        if sym is None:
            read = self.g.op("model.param", name=p.name, bits=p.bits,
                             signed=bool(p.signed))
            sym = self.g.symbols[p.name] = {"object": p, "read": read}
        elif (sym["object"].bits, bool(sym["object"].signed)) != \
                (p.bits, bool(p.signed)):
            raise Unsupported(f"two registers named {p.name!r}")
        return sym["read"]

    def table(self, name: str, obj, size: int, container, lo: int,
              hi: int) -> Value:
        """A register array or a constant table: one named buffer."""
        sym = self.g.symbols.get(name)
        if sym is None:
            axis = self.g.vars.declare(size, f"entries of {name}", "e")
            read = self.g.op("model.table", name=name, axes=(axis,),
                             container=container, lo=lo, hi=hi)
            sym = self.g.symbols[name] = {"object": obj, "read": read}
        return sym["read"]

    def operand(self, o) -> Value | None:
        if isinstance(o, Canvas):
            return o.array().v
        if isinstance(o, Array):
            return o.v
        if isinstance(o, Param):
            return self.register(o)
        if isinstance(o, np.integer):
            return self.const(int(o), _container_of(type(o)))
        if isinstance(o, int) and not isinstance(o, bool):
            return self.const(o)
        return None


class Array:
    """A traced NumPy array: a graph value. Its shape is its type's axes."""
    __slots__ = ("t", "v")
    __array_priority__ = 1000

    def __init__(self, t: Tracer, v: Value):
        self.t, self.v = t, v

    @property
    def shape(self) -> tuple:
        return tuple(1 if a is None else self.t.g.vars.count(a)
                     for a in self.v.type.axes)

    @property
    def ndim(self) -> int:
        return len(self.v.type.axes)

    def _op(self, kind, *vals, **attrs):
        return Array(self.t, self.t.g.op(kind, *vals, **attrs))

    def _binary(self, kind, o, reflected=False):
        v = self.t.operand(o)
        if v is None:
            return NotImplemented
        a, b = (v, self.v) if reflected else (self.v, v)
        return self._op(kind, a, b)

    # -- arithmetic, comparisons, masks: as written ---------------------------

    def __add__(self, o): return self._binary("model.add", o)
    def __radd__(self, o): return self._binary("model.add", o, True)
    def __sub__(self, o): return self._binary("model.sub", o)
    def __rsub__(self, o): return self._binary("model.sub", o, True)
    def __mul__(self, o): return self._binary("model.mul", o)
    def __rmul__(self, o): return self._binary("model.mul", o, True)
    def __and__(self, o): return self._binary("model.and", o)
    def __rand__(self, o): return self._binary("model.and", o, True)
    def __or__(self, o): return self._binary("model.or", o)
    def __ror__(self, o): return self._binary("model.or", o, True)
    def __lt__(self, o): return self._binary("model.lt", o)
    def __gt__(self, o): return self._binary("model.gt", o)
    def __le__(self, o): return self._binary("model.le", o)
    def __ge__(self, o): return self._binary("model.ge", o)
    def __eq__(self, o): return self._binary("model.eq", o)
    def __ne__(self, o): return self._binary("model.ne", o)

    __hash__ = None                  # compared by value, as NumPy arrays are

    def __neg__(self):
        return self._op("model.neg", self.v)

    def __abs__(self):
        return self._op("model.abs", self.v)

    def _by_constant(self, kind, o, key):
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
            return self._op(kind, self.v, **{key: int(o)})
        return NotImplemented

    def __floordiv__(self, o): return self._by_constant("model.floordiv", o, "divisor")
    def __mod__(self, o): return self._by_constant("model.mod", o, "divisor")
    def __rshift__(self, o): return self._by_constant("model.shr", o, "amount")

    def clip(self, lo, hi):
        return self._op("model.clip", self.v, lo=int(lo), hi=int(hi))

    def astype(self, dtype):
        return self._op("model.astype", self.v, container=_container_of(dtype))

    def __int__(self):
        raise Unsupported("a traced value used as a Python int")

    __index__ = __int__

    def __bool__(self):
        raise Unsupported("a traced value used as a condition")

    # -- slicing: index math, then an access ---------------------------------

    def __getitem__(self, key):
        key = _expand(key, self.ndim)
        g = self.t.g
        window = self._window_of(key)
        if window is not None:
            return window
        idx = []
        for k, axis in zip(key, self.v.type.axes):
            if axis is not None and isinstance(k, slice) and k == slice(None):
                # the whole axis, in order: its own variable, so `x[..., k]`
                # is still x's frame -- a variable of its own would be a
                # second lattice that only happens to have the same count
                idx.append(g.op("model.var", name=axis))
                continue
            dim = 1 if axis is None else g.vars.count(axis)
            idx.append(self._index(k, dim))
        return self._op("model.access", self.v, *idx)

    def _index(self, k, dim: int) -> Value:
        """The index expression one key reads an axis of `dim` through."""
        g, t = self.t.g, self.t
        if isinstance(k, Array):                   # an index expression
            return k.v
        if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
            k = int(k)
            if not -dim <= k < dim:
                raise IndexError(f"index {k} outside an axis of {dim}")
            return t.const(k % dim)                # an integer drops the axis
        if not isinstance(k, slice):
            raise Unsupported(f"indexing with {type(k).__name__}")
        step = 1 if k.step is None else int(k.step)
        if isinstance(k.start, (PhaseRef, Param)):
            if k.stop is not None or step < 1 or dim % step:
                raise Unsupported(f"a register start with {k}")
            count, start = dim // step, self._register_start(k.start)
        else:
            s0, s1, s2 = k.indices(dim)
            if s2 < 1:
                raise Unsupported("a slice that reads backwards")
            count, start = len(range(s0, s1, s2)), t.const(s0)
        if count < 1:
            raise Unsupported("an empty slice")
        v = g.op("model.var", name=g.vars.declare(count, f"slice {k}", "s"))
        if step != 1:
            v = g.op("model.mul", v, t.const(step))
        return g.op("model.add", v, start)

    def _window_of(self, key):
        """`img[p.y0:, p.x0:][:p.h, :p.w]` -- a window registers place and
        size -- as one node over the source it reads; None for any other
        key. The first slice reads the source from a register start; this
        one ends it at a register length. In NumPy the two are an ordinary
        crop; in hardware the source streams through and the window is
        kept, so the node keeps the source's own frame."""
        stops = [k for k in key[:2] if isinstance(k, slice)
                 and isinstance(k.stop, Param)]
        if not stops:
            return None
        if (len(stops) != 2
                or any(k.start is not None or k.step not in (None, 1)
                       for k in key[:2])
                or any(not (isinstance(k, slice) and k == slice(None))
                       for k in key[2:])):
            raise Unsupported("a window ends rows and columns at register "
                              "lengths, after a register start: "
                              "img[p.y0:, p.x0:][:p.h, :p.w]")
        n = self.v.node
        starts = []
        if n.kind == "model.access":
            for i in n.operands[1:3]:
                d = i.node
                if (d.kind == "model.add" and d.operands[0].node.kind
                        == "model.var" and d.operands[1].node.kind
                        == "model.param"):
                    starts.append(d.operands[1])
        if len(starts) != 2:
            raise Unsupported("a window's lengths need its starts: slice "
                              "the source from registers first, "
                              "img[p.y0:, p.x0:]")
        source = n.operands[0]
        h, w = (self.t.register(k.stop) for k in key[:2])
        return Array(self.t, self.t.g.op("model.window", source, starts[0],
                                  starts[1], h, w))

    def _register_start(self, s) -> Value:
        """`p.py` or `1 - p.py`: index math on a register."""
        t = self.t
        if isinstance(s, Param):
            return t.register(s)
        reg = t.register(s.param)
        return t.g.op("model.sub", t.const(1), reg) if s.invert else reg

    # -- position: the axes' variables, as values ------------------------------

    def _coords(self):
        """`coords(img)`: the variables of img's row and column axes, as
        values shaped (H, 1) and (1, W) -- the input raster's driving
        indexes."""
        g = self.t.g
        r, c = self.v.type.axes[0], self.v.type.axes[1]
        y = g.op("model.expand", g.op("model.var", name=r), axes=(r, None))
        x = g.op("model.expand", g.op("model.var", name=c), axes=(None, c))
        return Array(self.t, y), Array(self.t, x)

    def _make_coords(self, rows: int, cols: int):
        """`make_coords(img, rows, cols)`: a declared output raster's
        variables, shaped (rows, 1) and (1, cols)."""
        g = self.t.g
        r = g.vars.declare(rows, "output rows", "o")
        c = g.vars.declare(cols, "output cols", "o")
        y = g.op("model.expand", g.op("model.var", name=r), axes=(r, None))
        x = g.op("model.expand", g.op("model.var", name=c), axes=(None, c))
        return Array(self.t, y), Array(self.t, x)

    # -- lookups: a table read through this value ---------------------------

    def _gather_from(self, parent: Param):
        """`table[x]` for a register array: the table is a buffer of
        registers, and this is an access with a data index."""
        if len(parent.shape) != 1:
            raise Unsupported(f"a lookup in a {len(parent.shape)}-D array")
        lo = -(1 << (parent.bits - 1)) if parent.signed else 0
        hi = (1 << (parent.bits - (1 if parent.signed else 0))) - 1
        tab = self.t.table(parent.name, parent, parent.shape[0], ("i", 64),
                           lo, hi)
        return self._op("model.access", tab, self.v)

    def _rom_from(self, rom):
        """`rom[x]` for a constant table: the same, its entries constants."""
        data = np.asarray(rom.data)
        tab = self.t.table(rom.name, rom, data.size, _container_of(data.dtype),
                           int(rom.lo), int(rom.hi))
        return self._op("model.access", tab, self.v)

    # -- reshape: access through split and merged indexes ------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _reshape(self, tuple(int(n) for n in shape))

    # -- NumPy's protocols ---------------------------------------------------

    def __array_function__(self, func, types, args, kwargs):
        if func is np.pad:
            return self._pad(*args[1:], **kwargs)
        if func is np.where:
            return _where(self.t, *args, **kwargs)
        if func in (np.empty_like, np.zeros_like):
            if kwargs or len(args) != 1:
                raise Unsupported(f"np.{func.__name__} with options")
            return Canvas(self, zeroed=func is np.zeros_like)
        if func is np.stack:
            return _stack(self.t, *args, **kwargs)
        if func is np.reshape:
            return _reshape(args[0], tuple(int(n) for n in args[1]))
        raise Unsupported(f"np.{func.__name__}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise Unsupported(f"{ufunc.__name__}.{method}")
        if ufunc is np.absolute and len(inputs) == 1:
            return self._op("model.abs", self.v)
        binary = {np.add: "model.add", np.subtract: "model.sub",
                  np.multiply: "model.mul", np.less: "model.lt",
                  np.greater: "model.gt", np.less_equal: "model.le",
                  np.greater_equal: "model.ge", np.equal: "model.eq",
                  np.not_equal: "model.ne", np.bitwise_and: "model.and",
                  np.bitwise_or: "model.or"}
        if ufunc in binary and len(inputs) == 2:
            a, b = (self.t.operand(x) for x in inputs)
            if a is None or b is None:
                raise Unsupported(f"np.{ufunc.__name__} of those operands")
            return self._op(binary[ufunc], a, b)
        raise Unsupported(f"np.{ufunc.__name__}")

    def _pad(self, pad_width, mode="constant", **kw):
        edge = {"edge": "edge", "constant": "zero"}.get(mode)
        if edge is None:
            raise Unsupported(f"np.pad mode {mode!r}")
        if edge == "zero" and kw.get("constant_values", 0) != 0:
            raise Unsupported("np.pad with a non-zero constant")
        widths = np.broadcast_to(np.asarray(pad_width), (self.ndim, 2))
        widths = tuple((int(a), int(b)) for a, b in widths)
        g = self.t.g
        axes = tuple(g.vars.declare(n + a + b, "pad", "p")
                     for n, (a, b) in zip(self.shape, widths))
        return self._op("model.pad", self.v, axes=axes, widths=widths,
                        mode=edge)


def _expand(key, ndim: int) -> tuple:
    if not isinstance(key, tuple):
        key = (key,)
    if sum(k is Ellipsis for k in key) > 1:
        raise Unsupported("an index with two ellipses")
    if Ellipsis in key:
        i = key.index(Ellipsis)
        key = key[:i] + (slice(None),) * (ndim - len(key) + 1) + key[i + 1:]
    if len(key) > ndim:
        raise Unsupported(f"{len(key)} indexes into a {ndim}-D value")
    return key + (slice(None),) * (ndim - len(key))


class Canvas:
    """`np.empty_like(x)` followed by writes to planes. NumPy assembles it
    by mutation; the graph cannot, so the writes are collected and become
    ONE node when the canvas is used: the template, then each plane's value
    and the indexes it was written through."""

    def __init__(self, template: Array, zeroed: bool):
        self.template, self.zeroed = template, zeroed
        self.planes: list = []
        self._array = None

    @property
    def shape(self):
        return self.template.shape

    def __setitem__(self, key, value):
        if self._array is not None:
            raise Unsupported("a canvas written after it was used")
        t = self.template
        key = _expand(key, t.ndim)
        idx = [t._index(k, n) for k, n in zip(key, t.shape)]
        v = t.t.operand(value)
        if v is None:
            raise Unsupported(f"a canvas plane of {type(value).__name__}")
        # the plane's axes are the key's variables: NumPy pairs them by
        # position, so they are unified
        key_axes = [a for i in idx for a in i.type.axes if a is not None]
        val_axes = [a for a in v.type.axes if a is not None]
        if len(key_axes) == len(val_axes):
            for a, b in zip(key_axes, val_axes):
                t.t.g.vars.unify(a, b)
        self.planes.append((v, idx))

    def array(self) -> Array:
        if self._array is None:
            t = self.template
            ops = [t.v]
            for v, idx in self.planes:
                ops += [v, *idx]
            node = t.t.g.node("model.canvas", ops,
                              {"planes": len(self.planes),
                               "zeroed": self.zeroed})
            self._array = Array(t.t, node.value)
        return self._array

    def astype(self, dtype):
        return self.array().astype(dtype)

    def clip(self, lo, hi):
        return self.array().clip(lo, hi)

    def __array_function__(self, func, types, args, kwargs):
        return self.array().__array_function__(func, types, args, kwargs)


def _stack(t: Tracer, arrays, axis=0, **kw):
    if kw:
        raise Unsupported(f"np.stack options {sorted(kw)}")
    vals = [t.operand(a) for a in arrays]
    if any(v is None for v in vals):
        raise Unsupported("np.stack of those operands")
    var = t.g.vars.declare(len(vals), "stacked", "k")
    return Array(t, t.g.node("model.stack", vals,
                             {"var": var, "axis": int(axis)}).value)


def _groups(old: tuple, new: tuple) -> list:
    """C-order reshape: consecutive old and new dimensions whose products
    agree, as (old indices, new indices)."""
    out, i, j = [], 0, 0
    while i < len(old) or j < len(new):
        oi, nj = [i], [j]
        po, pn = old[i] if i < len(old) else 1, new[j] if j < len(new) else 1
        i, j = i + 1, j + 1
        while po != pn:
            if po < pn:
                po *= old[i]; oi.append(i); i += 1
            else:
                pn *= new[j]; nj.append(j); j += 1
        out.append(([k for k in oi if k < len(old)],
                    [k for k in nj if k < len(new)]))
    return out


def _reshape(a: Array, shape: tuple) -> Array:
    """A C-order reshape, as access: each new dimension a variable, each old
    axis read through index math on them -- a split `k * q + r`, a merge
    `(f // s) % d`. Nothing moves."""
    t, g = a.t, a.t.g
    old = a.shape
    if -1 in shape:
        known = int(np.prod([n for n in shape if n != -1]))
        shape = tuple(int(np.prod(old)) // known if n == -1 else n
                      for n in shape)
    if int(np.prod(old)) != int(np.prod(shape)):
        raise Unsupported(f"a reshape of {old} to {shape}")
    new_vars = [None if n == 1 else g.vars.declare(n, "reshape", "h")
                for n in shape]
    idx = [None] * len(old)

    def placed(j):
        """New dimension j's variable, on its own axis among the new axes."""
        v = g.op("model.var", name=new_vars[j])
        axes = tuple(new_vars[k] if k == j else None
                     for k in range(len(shape)) if new_vars[k] is not None)
        if len(axes) > 1:
            v = g.op("model.expand", v, axes=axes)
        return v

    for olds, news in _groups(old, shape):
        live = [j for j in news if new_vars[j] is not None]
        flat = None
        for j in live:
            stride = int(np.prod([shape[k] for k in news if k > j]))
            term = placed(j)
            if stride != 1:
                term = g.op("model.mul", term, t.const(stride))
            flat = term if flat is None else g.op("model.add", flat, term)
        for i in olds:
            if old[i] == 1 or flat is None:
                idx[i] = t.const(0)
                continue
            stride = int(np.prod([old[k] for k in olds if k > i]))
            e = flat
            if stride != 1:
                e = g.op("model.floordiv", e, divisor=stride)
            if i != olds[0]:
                e = g.op("model.mod", e, divisor=old[i])
            idx[i] = e
    return Array(t, g.op("model.access", a.v, *idx))


def _where(t: Tracer, cond, a, b, **kw):
    if kw:
        raise Unsupported(f"np.where options {sorted(kw)}")
    vals = [t.operand(x) for x in (cond, a, b)]
    if any(v is None for v in vals):
        raise Unsupported("np.where of those operands")
    return Array(t, t.g.op("model.where", *vals))


class Traced:
    """What `trace` returns: the graph, its source, its output."""

    def __init__(self, g: Graph, source: Value, output: Value, image):
        self.g, self.source, self.output, self.image = g, source, output, image


def trace(fn, image: Image2D, *params, channels: int = 1) -> Traced:
    """Run the model on a traced image; return the graph it built.

    With `channels`, the source is (H, W, C): each pixel's word carries C
    fields, channel 0 in the low bits -- how that axis is laid out at the
    stream boundary, a fact of the source."""
    t = Tracer()
    g = t.g
    rows = g.vars.declare(image.height, "rows of the source", "r")
    cols = g.vars.declare(image.width, "cols of the source", "c")
    axes = (rows, cols)
    bits, signed = image.bits, image.signed
    if channels > 1:
        if bits % channels:
            raise Unsupported(f"a {bits}-bit word in {channels} fields")
        bits, signed = bits // channels, False
        axes += (g.vars.declare(channels, "channels of the source", "ch"),)
    c = image_container(bits, signed)
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - (1 if signed else 0))) - 1
    src = g.op("model.source", axes=axes, container=c, lo=lo, hi=hi)
    out = fn(Array(t, src), *params)
    if isinstance(out, Canvas):
        out = out.array()
    if not isinstance(out, Array):
        raise Unsupported(f"a model returning {type(out).__name__}")
    return Traced(g, src, out.v, image)
