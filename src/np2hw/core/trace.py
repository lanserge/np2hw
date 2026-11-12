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
buffers with no axes -- a Python int WEAK, a NumPy scalar with its own
dtype, as NumPy 2 has them; a register is the NumPy scalar the oracle is
handed for it (np2hw.ir.np_dtype).
"""
from __future__ import annotations

import numpy as np

from ..ir import Image2D, Param, PhaseRef
from .graph import BOOL, WEAK, Graph, GraphError, Value, container_range
from .model import register_container

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


# The trace a model is running in, while it runs: what a register reads as
# when the model does arithmetic on it (np2hw.ir.Param's operators).
_ACTIVE: list = []


def active():
    """The tracer a model is running in, or None outside a trace."""
    return _ACTIVE[-1] if _ACTIVE else None


class Tracer:
    """One trace: its graph, and its registers read once each."""

    def __init__(self):
        self.g = Graph(intern=False, locations=True)
        # the axes inside a pixel's word -- a source's channels, a stack's,
        # and what slices and reshapes make of them -- as declared: a
        # reduction over them is arithmetic within one pixel, not a fold
        # over the frame
        self.word: set = set()

    def is_word(self, axis) -> bool:
        """Whether an axis lies within a pixel's word; a dimension of size
        one (None) is no axis of the frame either."""
        if axis is None:
            return True
        find = self.g.vars.find
        return any(find(w) == find(axis) for w in self.word)

    def const(self, value: int, container=WEAK) -> Value:
        return self.g.op("model.const", value=int(value), container=container)

    def register(self, p: Param, weak: bool = False) -> Value:
        """A register: one named declaration, one read -- in the container
        the NumPy run hands it in (np2hw.ir.np_dtype), or WEAK where the
        NumPy run reads it as a Python int (a bound expression's facts)."""
        sym = self.g.symbols.get(p.name)
        if sym is None:
            read = self.g.op("model.param", name=p.name, bits=p.bits,
                             signed=bool(p.signed))
            sym = self.g.symbols[p.name] = {"object": p, "read": read}
        elif (sym["object"].bits, bool(sym["object"].signed)) != \
                (p.bits, bool(p.signed)):
            raise Unsupported(f"two registers named {p.name!r}")
        if weak:
            if "weak_read" not in sym:
                sym["weak_read"] = self.g.op(
                    "model.param", name=p.name, bits=p.bits,
                    signed=bool(p.signed), container=WEAK)
            return sym["weak_read"]
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
        if isinstance(o, (Canvas, Accumulator)):
            return o.array().v
        if isinstance(o, Array):
            return o.v
        if isinstance(o, Param):
            return self.register(o)
        if isinstance(o, (bool, np.bool_)):
            return self.const(int(o), BOOL)
        if isinstance(o, np.integer):
            return self.const(int(o), _container_of(type(o)))
        if isinstance(o, int) and not isinstance(o, bool):
            return self.const(o)
        if isinstance(o, np.ndarray) and o.ndim == 1 and o.dtype.kind in "ui":
            # constants along an axis of their own -- a pixel's word, as a
            # stack's is -- which broadcasting pairs with the value's:
            # `abs(x[..., None] - centres)`
            c = _container_of(o.dtype)
            return _stack(self, [np.dtype(o.dtype).type(v) for v in o]).v \
                if o.size > 1 else self.const(int(o[0]), c)
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
    def __xor__(self, o): return self._binary("model.xor", o)
    def __rxor__(self, o): return self._binary("model.xor", o, True)
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

    def __lshift__(self, o):
        # a multiply by a power of two, in the value's own container, as
        # NumPy shifts: the bits shifted past it are gone
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool) \
                and o >= 0:
            return self._binary("model.mul", 1 << int(o))
        return NotImplemented

    def clip(self, lo, hi):
        return self._op("model.clip", self.v, lo=int(lo), hi=int(hi))

    def astype(self, dtype):
        return self._op("model.astype", self.v, container=_container_of(dtype))

    def view(self, dtype):
        """The same bits read as another integer of their width: the value's
        container wrapped to that one, which is what a view of the same
        size does. Hardware: wires."""
        have, want = self.v.type.container, _container_of(dtype)
        if have[0] not in "ui" or want[0] not in "ui" or have[1] != want[1]:
            raise Unsupported(f"a view of a {have} value as {np.dtype(dtype)}"
                              ": np2hw views an integer as another of its "
                              "own width")
        return self.astype(dtype)

    def __int__(self):
        raise Unsupported("a traced value used as a Python int")

    __index__ = __int__

    def __bool__(self):
        raise Unsupported("a traced value used as a condition")

    # -- slicing: index math, then an access ---------------------------------

    def __getitem__(self, key):
        if key is None or isinstance(key, tuple) and \
                any(k is None for k in key):
            return self._newaxis(key)
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
            idx.append(self._index(k, dim, axis is not None
                                   and self.t.is_word(axis)))
        return self._op("model.access", self.v, *idx)

    def _newaxis(self, key) -> "Array":
        """`x[..., None]`: the value indexed without the Nones, then placed
        among axes with a dimension of size one at each -- NumPy's newaxis,
        an expand."""
        key = key if isinstance(key, tuple) else (key,)
        real = sum(k is not None and k is not Ellipsis for k in key)
        if sum(k is Ellipsis for k in key) > 1:
            raise Unsupported("an index with two ellipses")
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - real) + key[i + 1:]
        else:
            key = key + (slice(None),) * (self.ndim - real)
        r = self[tuple(k for k in key if k is not None)]
        have = iter(r.v.type.axes)
        axes = []
        for k in key:
            if k is None:
                axes.append(None)
            elif not (isinstance(k, (int, np.integer))
                      and not isinstance(k, bool)):
                axes.append(next(have))
        return r._op("model.expand", r.v, axes=tuple(axes))

    def _index(self, k, dim: int, word: bool = False) -> Value:
        """The index expression one key reads an axis of `dim` through;
        `word`, an axis within the pixel's word, whose slice is one too."""
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
        if isinstance(k.start, (PhaseRef, Param)) or (
                isinstance(k.start, Array) and k.start.ndim == 0):
            if k.stop is not None or step < 1 or dim % step:
                raise Unsupported(f"a register start with {k}")
            count, start = dim // step, self._register_start(k.start, step)
        else:
            s0, s1, s2 = k.indices(dim)
            if s2 < 1:
                raise Unsupported("a slice that reads backwards")
            count, start = len(range(s0, s1, s2)), t.const(s0)
        if count < 1:
            raise Unsupported("an empty slice")
        # how the slice ends: at the axis's end -- no stop, a stop counted
        # back from the end, or one at or past it -- or at a fixed index
        stop = k.stop if isinstance(k.stop, (int, np.integer)) else None
        ends = "end" if stop is None or stop < 0 or stop >= dim else "fixed"
        name = g.vars.declare(count, f"slice {k}", "s", ends=ends)
        if word:
            t.word.add(name)
        v = g.op("model.var", name=name)
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

    def _register_start(self, s, step: int = 0) -> Value:
        """`p.py`, `1 - p.py`, or a value computed from registers -- `p.phase
        >> 1` -- whose exact range stays inside one stride: index math on
        registers."""
        t = self.t
        if isinstance(s, Array):
            lo, hi = s.v.type.lo, s.v.type.hi
            if lo < 0 or hi >= step:
                raise Unsupported(f"a slice start ranging over [{lo}, {hi}], "
                                  f"which a stride of {step} cannot hold")
            return s.v
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
        r = g.vars.declare(rows, "output rows", "o", ends="fixed")
        c = g.vars.declare(cols, "output cols", "o", ends="fixed")
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
        tab = self.t.table(parent.name, parent, parent.shape[0],
                           register_container(parent.bits, parent.signed),
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

    # -- reductions: every position into one value ---------------------------

    def sum(self, axis=None, dtype=None, **kw):
        """`x.sum()`, or over `axis`: in the container NumPy sums in -- a
        64-bit integer of the value's sign, or `dtype`."""
        _options(kw, "sum")
        have = self.v.type.container
        if dtype is not None:
            c = _container_of(dtype)
        elif have[0] == "u":
            c = ("u", 64)
        else:                                   # signed, bool, a Python int
            c = ("i", 64)
        return self._reduce("add", c, 0, axis)

    def max(self, axis=None, **kw):
        _options(kw, "max")
        return self._reduce("max", self._held(), self.v.type.lo, axis)

    def min(self, axis=None, **kw):
        _options(kw, "min")
        return self._reduce("min", self._held(), self.v.type.hi, axis)

    def argmax(self, axis=None, **kw):
        _options(kw, "argmax")
        return self._arg("max", axis)

    def argmin(self, axis=None, **kw):
        _options(kw, "argmin")
        return self._arg("min", axis)

    def _held(self):
        c = self.v.type.container
        return ("i", 64) if c == WEAK else c

    def _reduce(self, op, container, init, axis=None):
        """Every position of the value into one -- a scatter with no key --
        or, over `axis`, each position into the entry of the axes it keeps:
        a scatter keyed by their own variables, as `np.add.at(acc, x, v)`
        with `x` the position is. Its start is the operation's identity on
        the value's range."""
        g, ndim = self.t.g, self.ndim
        axes = tuple(range(ndim)) if axis is None else \
            tuple(int(a) for a in (axis if isinstance(axis, tuple)
                                   else (axis,)))
        if any(not -ndim <= a < ndim for a in axes):
            raise Unsupported(f"a reduction over axis {axis} of a "
                              f"{ndim}-D value")
        gone = {a % ndim for a in axes}
        if len(gone) != len(axes):
            raise Unsupported(f"a reduction over axis {axis}: an axis twice")
        own = self.v.type.axes
        word = sorted((a for a in gone if self.t.is_word(own[a])),
                      reverse=True)
        if word:
            # within each pixel's word: the lanes combined as NumPy combines
            # them -- then, over the frame's axes left, a fold as below
            x = self._lanes_reduced(op, container, word)
            rest = tuple(a - sum(w < a for w in word) for a in gone
                         if a not in word)
            if not rest:
                return x
            init = {"add": init, "max": x.v.type.lo,
                    "min": x.v.type.hi}[op]
            return x._reduce(op, container, init, rest)
        kept = [i for i in range(ndim) if i not in gone]
        if any(own[i] is None for i in kept):
            raise Unsupported("a reduction keeping an axis of size 1")
        keys, entries = [], []
        for i in kept:
            v = g.op("model.var", name=own[i])
            if ndim > 1:
                v = g.op("model.expand", v, axes=tuple(
                    own[i] if j == i else None for j in range(ndim)))
            keys.append(v)
            entries.append(g.vars.declare(g.vars.count(own[i]),
                                          "entries of a reduction", "b"))
        return self._op("model.scatter", self.v, *keys, op=op,
                        init=int(init), axes=tuple(entries),
                        container=container)

    def _lane(self, axis: int, i: int) -> "Array":
        """Lane `i` of a word axis: `x[..., i, ...]`."""
        return self[tuple(i if a == axis else slice(None)
                          for a in range(self.ndim))]

    def _lanes_reduced(self, op, container, axes) -> "Array":
        """The value reduced over word axes, last first: per axis its lanes
        combined pairwise, a balanced tree -- each step the operation in
        the container NumPy reduces in, which for a sum is where it wraps
        (modulo its width, as NumPy's sequential sum is)."""
        kind = {"add": "model.add", "max": "model.max",
                "min": "model.min"}[op]
        x = self
        for a in axes:
            n = x.shape[a]
            lanes = [x._lane(a, i) for i in range(n)]
            lanes = [l if l.v.type.container == container
                     else l.astype(_dtype_of(container)) for l in lanes]
            while len(lanes) > 1:
                pairs = [lanes[i:i + 2] for i in range(0, len(lanes), 2)]
                lanes = [p[0] if len(p) == 1 else p[0]._op(kind, p[0].v, p[1].v)
                         for p in pairs]
            x = lanes[0]
        return x

    def _arg(self, op, axis) -> "Array":
        """`argmax` / `argmin` over a word axis: the lanes paired in a
        balanced tree, each pair keeping the better value and its lane --
        the left one on a tie, so the first such lane wins, as NumPy's
        does. The index is NumPy's: a 64-bit integer."""
        if axis is None:
            if self.ndim != 1:
                raise Unsupported(f"np.arg{op} of a {self.ndim}-D value "
                                  "without an axis: name the word's axis")
            axis = 0
        if not isinstance(axis, (int, np.integer)) or \
                not -self.ndim <= axis < self.ndim:
            raise Unsupported(f"np.arg{op} over axis {axis}")
        axis = int(axis) % self.ndim
        if not self.t.is_word(self.v.type.axes[axis]):
            raise Unsupported(f"np.arg{op} over an axis of the frame: over a "
                              "pixel's word, its channels, only")
        better = "model.gt" if op == "max" else "model.lt"
        keep = "model.max" if op == "max" else "model.min"
        pairs = [(self._lane(axis, i), self.t.const(i, ("i", 64)))
                 for i in range(self.shape[axis])]
        g = self.t.g
        while len(pairs) > 1:
            nxt = []
            for i in range(0, len(pairs), 2):
                if i + 1 == len(pairs):
                    nxt.append(pairs[i])
                    continue
                (lv, li), (rv, ri) = pairs[i], pairs[i + 1]
                take = g.op(better, rv.v, lv.v)
                nxt.append((lv._op(keep, lv.v, rv.v),
                            g.op("model.where", take, ri, li)))
            pairs = nxt
        return Array(self.t, pairs[0][1])

    # -- NumPy's protocols ---------------------------------------------------

    def __array_function__(self, func, types, args, kwargs):
        if func in _REDUCTIONS:
            return getattr(args[0], _REDUCTIONS[func])(*args[1:], **kwargs)
        if func in _CREATIONS:
            return Accumulator.made(self.t, func, args, kwargs)
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
        if func is np.roll:
            return _roll(*args, **kwargs)
        raise Unsupported(f"np.{func.__name__}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "at" and not kwargs:
            if not isinstance(inputs[0], Accumulator):
                raise Unsupported(
                    f"np.{ufunc.__name__}.at into an array the model made "
                    "itself: make it like the frame -- np.zeros(n, dtype, "
                    "like=pixel) -- so the trace sees it")
            return inputs[0].at(ufunc, *inputs[1:])
        if method == "reduce" and ufunc in _UFUNC_REDUCE \
                and inputs[0] is self and len(inputs) == 1:
            axis = kwargs.pop("axis", 0)
            dtype = kwargs.pop("dtype", None)
            _options(kwargs, f"np.{ufunc.__name__}.reduce")
            if ufunc is np.add:
                return self.sum(axis=axis, dtype=dtype)
            if dtype is not None:
                raise Unsupported(f"np.{ufunc.__name__}.reduce with a dtype")
            return getattr(self, _UFUNC_REDUCE[ufunc])(axis=axis)
        if method != "__call__" or kwargs:
            raise Unsupported(f"{ufunc.__name__}.{method}")
        if ufunc is np.absolute and len(inputs) == 1:
            return self._op("model.abs", self.v)
        binary = {np.add: "model.add", np.subtract: "model.sub",
                  np.multiply: "model.mul", np.less: "model.lt",
                  np.greater: "model.gt", np.less_equal: "model.le",
                  np.greater_equal: "model.ge", np.equal: "model.eq",
                  np.not_equal: "model.ne", np.bitwise_and: "model.and",
                  np.bitwise_or: "model.or", np.bitwise_xor: "model.xor",
                  np.maximum: "model.max",
                  np.minimum: "model.min"}
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


def _options(kw, what):
    if kw:
        raise Unsupported(f"a {what} with options {sorted(kw)}")


#: NumPy's reductions, by the method each is
_REDUCTIONS = {np.sum: "sum", np.max: "max", np.amax: "max", np.min: "min",
               np.amin: "min", np.argmax: "argmax", np.argmin: "argmin"}
#: the ufuncs whose `reduce` is a reduction the tracer has
_UFUNC_REDUCE = {np.add: "sum", np.maximum: "max", np.minimum: "min"}


def _dtype_of(container):
    """The NumPy dtype of a container."""
    kind, bits = container
    if kind == "b":
        return np.bool_
    return np.dtype(f"{'u' if kind == 'u' else 'i'}{bits // 8}")
#: what an accumulator is made by, and what it starts at
_CREATIONS = {np.zeros: 0, np.ones: 1, np.full: None}
#: what each ufunc's `at` does to the entries it reaches
_AT = {np.add: "add", np.maximum: "max", np.minimum: "min"}


class Accumulator:
    """`np.zeros(shape, dtype, like=pixel)`, then one `np.add.at(acc, key,
    value)` -- or `maximum`, `minimum`: each position of the value folded
    into the entry its key names. NumPy builds it by mutation; the graph
    cannot, so the write is collected and becomes ONE node when the array
    is used: a scatter, the value and the keys its operands."""

    __array_priority__ = 1001

    def __init__(self, t: Tracer, shape: tuple, container, init: int):
        self.t, self.shape, self.container, self.init = t, shape, container, init
        self.write = None
        self._array = None

    @classmethod
    def made(cls, t, func, args, kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "like"}
        if kwargs.pop("device", None) not in (None, "cpu"):
            raise Unsupported("an accumulator on a device")
        order = kwargs.pop("order", "C")
        if order != "C":
            raise Unsupported(f"an accumulator in order {order!r}")
        names = ("shape", "fill_value", "dtype") if func is np.full \
            else ("shape", "dtype")
        given = dict(zip(names, args))
        for k, v in kwargs.items():
            if k not in names or k in given:
                raise Unsupported(f"np.{func.__name__} with {k}=")
            given[k] = v
        shape = given["shape"]
        shape = (int(shape),) if np.ndim(shape) == 0 else \
            tuple(int(n) for n in shape)
        init = _CREATIONS[func]
        if init is None:
            init = given["fill_value"]
            if not isinstance(init, (int, np.integer)) or isinstance(init,
                                                                      bool):
                raise Unsupported("np.full with a value the model computes: "
                                  "an accumulator starts at a constant")
        dtype = given.get("dtype")
        if dtype is None:
            raise Unsupported(f"np.{func.__name__} with no dtype: a float; "
                              "an accumulator is an integer -- say which")
        container = _container_of(dtype)
        lo, hi = container_range(container)
        if not lo <= int(init) <= hi:
            raise Unsupported(f"an accumulator of {np.dtype(dtype)} starting "
                              f"at {init}")
        return cls(t, shape, container, int(init))

    @property
    def ndim(self):
        return len(self.shape)

    def at(self, ufunc, key, value=None):
        if ufunc not in _AT or value is None:
            raise Unsupported(f"np.{ufunc.__name__}.at: an accumulator folds "
                              "by np.add, np.maximum or np.minimum, with a "
                              "value")
        if self._array is not None:
            raise Unsupported("an accumulator written after it was used")
        if self.write is not None:
            raise Unsupported(
                "two writes into one accumulator: each pixel would fold "
                "into it twice in a beat. Fold one value, its key telling "
                "the entries apart")
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) != self.ndim:
            raise Unsupported(f"{len(keys)} key(s) into a {self.ndim}-D "
                              "accumulator: one per axis")
        t = self.t
        vals = []
        for k, n in zip(keys, self.shape):
            if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
                if not -n <= int(k) < n:
                    raise IndexError(f"key {int(k)} outside an axis of {n}")
                vals.append(t.const(int(k) % n))
                continue
            v = t.operand(k)
            if v is None or v.type.container == BOOL:
                raise Unsupported(f"a key of {type(k).__name__}: a key is an "
                                  "integer the model computes, or a constant")
            vals.append(v)
        v = t.operand(value)
        if v is None:
            raise Unsupported(f"a value of {type(value).__name__} folded")
        self.write = (_AT[ufunc], vals, v)

    def array(self) -> "Array":
        if self._array is None:
            if self.write is None:
                raise Unsupported("an accumulator nothing was folded into")
            op, keys, v = self.write
            g = self.t.g
            axes = tuple(g.vars.declare(n, "entries of an accumulator", "b")
                         for n in self.shape)
            self._array = Array(self.t, g.op(
                "model.scatter", v, *keys, op=op, init=self.init, axes=axes,
                container=self.container))
        return self._array

    def __getattr__(self, name):
        # used as an array: the node, from here on
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.array(), name)

    def __getitem__(self, key):
        return self.array()[key]

    def __array_function__(self, func, types, args, kwargs):
        args = tuple(a.array() if a is self else a for a in args)
        return self.array().__array_function__(func, types, args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "at" and inputs and inputs[0] is self and not kwargs:
            return self.at(ufunc, *inputs[1:])
        inputs = tuple(a.array() if a is self else a for a in inputs)
        return self.array().__array_ufunc__(ufunc, method, *inputs, **kwargs)


def _delegate(name):
    def method(self, *args):
        return getattr(self.array(), name)(*args)
    method.__name__ = name
    return method


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__",
              "__rmul__", "__and__", "__rand__", "__or__", "__ror__",
              "__lt__", "__gt__", "__le__", "__ge__", "__eq__", "__ne__",
              "__neg__", "__abs__", "__floordiv__", "__mod__", "__rshift__",
              "__lshift__"):
    setattr(Accumulator, _name, _delegate(_name))
Accumulator.__hash__ = None


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
    t.word.add(var)
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
    groups = _groups(old, shape)
    # an axis the reshape keeps whole keeps its variable, as a whole-axis
    # slice does: the same positions, not a second lattice of equal count
    kept = {news[0]: a.v.type.axes[olds[0]] for olds, news in groups
            if len(olds) == len(news) == 1 and old[olds[0]] == shape[news[0]]
            and a.v.type.axes[olds[0]] is not None}
    new_vars = [kept.get(j) or (None if n == 1 else
                                g.vars.declare(n, "reshape", "h"))
                for j, n in enumerate(shape)]
    idx = [None] * len(old)

    def placed(j):
        """New dimension j's variable, on its own axis among the new axes."""
        v = g.op("model.var", name=new_vars[j])
        axes = tuple(new_vars[k] if k == j else None
                     for k in range(len(shape)) if new_vars[k] is not None)
        if len(axes) > 1:
            v = g.op("model.expand", v, axes=axes)
        return v

    for olds, news in groups:
        if len(news) == 1 and news[0] in kept:
            # read whole, in order: by its own variable, bare
            idx[olds[0]] = g.op("model.var", name=kept[news[0]])
            continue
        if all(old[i] == 1 or t.is_word(a.v.type.axes[i]) for i in olds):
            t.word.update(new_vars[j] for j in news
                          if new_vars[j] is not None)
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


def _roll(a: "Array", shift, axis=None) -> "Array":
    """`np.roll(a, shift, axis)`: each axis read at `(i - shift) mod n` --
    on a word's axis its lanes in another order, a stack of them; on the
    frame's a read that wraps around it, which the C backend takes and
    hardware, needing the whole row or frame first, refuses. Shifts are
    constants."""
    if axis is None:
        raise Unsupported("np.roll without an axis: name the axes it rolls")
    shifts = shift if isinstance(shift, (tuple, list)) else (shift,)
    axes = axis if isinstance(axis, (tuple, list)) else (axis,)
    if len(shifts) == 1 and len(axes) > 1:
        shifts = shifts * len(axes)
    if len(shifts) != len(axes):
        raise Unsupported(f"np.roll by {shift} over axes {axis}")
    t, g = a.t, a.t.g
    out = a
    for s, ax in zip(shifts, axes):
        if not isinstance(s, (int, np.integer)) or isinstance(s, bool):
            raise Unsupported("np.roll by a shift that is not a constant")
        if not -out.ndim <= int(ax) < out.ndim:
            raise Unsupported(f"np.roll over axis {ax} of a {out.ndim}-D value")
        ax = int(ax) % out.ndim
        own = out.v.type.axes[ax]
        n = out.shape[ax]
        s = int(s) % n
        if own is None or s == 0:
            continue
        if t.is_word(own):
            # the word's lanes in another order: wiring, a stack of them
            lanes = [out._lane(ax, (i - s) % n) for i in range(n)]
            out = _stack(t, lanes, axis=ax)
            continue
        idx = []
        for i, v in enumerate(out.v.type.axes):
            var = g.op("model.var", name=v) if v is not None else t.const(0)
            if i == ax:
                var = g.op("model.mod", g.op("model.add", var, t.const(n - s)),
                           divisor=n)
            idx.append(var)
        out = Array(t, g.op("model.access", out.v, *idx))
    return out


def _where(t: Tracer, cond, a, b, **kw):
    if kw:
        raise Unsupported(f"np.where options {sorted(kw)}")
    if isinstance(cond, (bool, np.bool_)):
        # a condition known when the model is traced -- a build parameter, a
        # bound constant: NumPy picks one operand, and so does the trace
        chosen = a if cond else b
        return chosen if isinstance(chosen, Array) else \
            Array(t, t.operand(chosen))
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
    # a video's frames are its outermost axis, as they arrive
    axes = ((g.vars.declare(image.frames, "frames of the source", "t"),)
            if image.frames else ())
    rows = g.vars.declare(image.height, "rows of the source", "r")
    cols = g.vars.declare(image.width, "cols of the source", "c")
    axes += (rows, cols)
    bits, signed = image.bits, image.signed
    if channels > 1:
        if bits % channels:
            raise Unsupported(f"a {bits}-bit word in {channels} fields")
        bits, signed = bits // channels, False
        axes += (g.vars.declare(channels, "channels of the source", "ch"),)
        t.word.add(axes[-1])
    c = image_container(bits, signed)
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - (1 if signed else 0))) - 1
    src = g.op("model.source", axes=axes, container=c, lo=lo, hi=hi)
    _ACTIVE.append(t)
    try:
        out = fn(Array(t, src), *params)
    finally:
        _ACTIVE.pop()
    if isinstance(out, (Canvas, Accumulator)):
        out = out.array()
    if not isinstance(out, Array):
        raise Unsupported(f"a model returning {type(out).__name__}")
    return Traced(g, src, out.v, image)


# --------------------------------------------------------------------------- #
# what a model calls: position, a declared raster, a saturation
# --------------------------------------------------------------------------- #

def coords(img):
    """Pixel POSITION as values: (row, col).

    On real arrays these are ordinary NumPy ramps, so a model that uses
    them stays runnable and stays its own oracle. Traced, they are the
    variables of the input raster's row and column axes -- the indexes the
    core already counts, because its framing depends on them -- so position
    costs no logic, and an overlay, a lens-shading profile or a windowed
    statistic can be written as plain arithmetic on where the pixel is.

    The ramps are BROADCAST-SHAPED -- ``(H, 1)`` and ``(1, W)`` rather than
    ``np.indices``' two full grids. Every elementwise use is identical by
    NumPy's broadcasting rules, and the shapes say what matters about
    position: a row-only value is constant along a line, and a column-only
    value repeats every line.
    """
    if hasattr(img, "_coords"):              # a value that traces itself
        return img._coords()
    height, width = np.asarray(img).shape[:2]
    return (np.arange(height, dtype=np.int64).reshape(height, 1),
            np.arange(width, dtype=np.int64).reshape(1, width))


def make_coords(img, rows, cols):
    """Declare an OUTPUT raster and return its position, as `coords` does
    for the input's.

    `coords(img)` says where a pixel is in the frame arriving. This says
    where a pixel is in the frame LEAVING, which is a different raster the
    moment a model stops emitting one pixel per pixel. Indexing the input
    with these coordinates states the map from output position to input
    position:

        y2, x2 = make_coords(img, H - 4, W - 4)
        return img[y2 + 2, x2 + 2]          # a crop, offset by (2, 2)

    On real arrays these are ordinary ramps of the declared size, so a
    model using them stays runnable and stays its own oracle -- the same
    bargain `coords` makes.
    """
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"make_coords: an output raster of {rows}x{cols} "
                         "has no pixels in it")
    if hasattr(img, "_make_coords"):         # a value that traces itself
        return img._make_coords(rows, cols)
    return (np.arange(rows, dtype=np.int64).reshape(rows, 1),
            np.arange(cols, dtype=np.int64).reshape(1, cols))


def saturate(value, bits: int):
    """Saturate to what fits in ``bits`` unsigned: the range [0, 2**bits - 1].

    ``clip`` takes a lo and a hi, which is the general operation and the
    honest one for an arbitrary bound. A DATAPATH bound is not arbitrary:
    it is a width, and writing it as ``clip(0, (1 << bits) - 1)`` states a
    width in the vocabulary of values. Stated as bits, the range is exact:
    a saturation in bits can only ever produce ``2**bits - 1``, so the
    width IS the scale, and a consumer downstream can derive one from the
    other without being told.

    Runs on real arrays as readily as on traced ones -- both have ``clip``
    with NumPy's semantics -- so a model using it stays its own oracle.

    Not for signed intermediates. A difference stream (sharpening, chroma)
    is bounded by a magnitude on both sides, which is a different
    statement; write that ``clip`` out.
    """
    if bits < 1:
        raise ValueError(f"saturate to {bits} bits: need at least 1")
    return value.clip(0, (1 << int(bits)) - 1)
