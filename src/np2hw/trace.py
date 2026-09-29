# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The tracer: a NumPy model, recorded as the one graph. See docs/one-graph.md.

Every operation the model performs adds ONE node, and records only what the
model did -- which operation, on which operands, with which constants. A
slice is a read through one `Index` per axis; `np.pad` is a larger source
with an edge mode; `astype` is a declaration. Nothing is decided here: no
forms, no fields, no choice of emitter. Deciding is lowering's job.

Evaluating the graph is generic (`evaluate`): each node kind is the ordinary
Python operation it records. Evaluated on NumPy arrays, the graph is the
model. Evaluated on the old tracer's values, it is lowered by the old
compiler -- the reference the new path is measured against until lowering
passes replace it.
"""
from __future__ import annotations

import operator

import numpy as np

from . import graph as G
from .ir import Image2D, Index, Param, PhaseRef


class Unsupported(NotImplementedError):
    """The model does something this tracer does not record yet."""


# --------------------------------------------------------------------------- #
# The traced array
# --------------------------------------------------------------------------- #

class Array:
    """A traced NumPy array: a graph value and the array's shape.

    The shape is the array's, as NumPy has it -- the extent of what the value
    covers -- and is the one fact kept beside the graph value, because
    slicing needs it before any node exists.
    """
    __slots__ = ("t", "v", "shape")
    __array_priority__ = 1000               # our operators before ndarray's

    def __init__(self, tracer: "Tracer", value: G.Value, shape: tuple):
        self.t = tracer
        self.v = value
        self.shape = tuple(shape)

    def _new(self, value, shape=None):
        return Array(self.t, value, self.shape if shape is None else shape)

    # -- operands ------------------------------------------------------------ #

    def _operand(self, o):
        """A graph value for an operand: an array, a register, or an int."""
        if isinstance(o, Array):
            if o.t is not self.t:
                raise Unsupported("an operation across two traces")
            return o.v
        if isinstance(o, Param):
            return self.t.param(o)
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
            return self.t.b.const(int(o))
        return None

    def _binary(self, kind, o, reflected=False):
        v = self._operand(o)
        if v is None:
            return NotImplemented
        a, b = (v, self.v) if reflected else (self.v, v)
        return self._new(self.t.b.op(kind, a, b))

    # -- arithmetic, as written ---------------------------------------------- #

    def __add__(self, o):
        return self._binary("model.add", o)

    def __radd__(self, o):
        return self._binary("model.add", o, reflected=True)

    def __sub__(self, o):
        return self._binary("model.sub", o)

    def __rsub__(self, o):
        return self._binary("model.sub", o, reflected=True)

    def __mul__(self, o):
        return self._binary("model.mul", o)

    def __rmul__(self, o):
        return self._binary("model.mul", o, reflected=True)

    def __neg__(self):
        return self._new(self.t.b.op("model.neg", self.v))

    def __floordiv__(self, o):
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
            return self._new(self.t.b.op("model.floordiv", self.v,
                                         divisor=int(o)))
        return NotImplemented

    def __rshift__(self, o):
        if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
            return self._new(self.t.b.op("model.shr", self.v, amount=int(o)))
        return NotImplemented

    def clip(self, lo, hi):
        return self._new(self.t.b.op("model.clip", self.v, lo=int(lo),
                                     hi=int(hi)))

    # -- comparisons, masks, predicates: as written --------------------------- #
    # `3 < x` reaches here as `x > 3`, and is recorded as the call Python made.

    def __lt__(self, o):
        return self._binary("model.lt", o)

    def __gt__(self, o):
        return self._binary("model.gt", o)

    def __le__(self, o):
        return self._binary("model.le", o)

    def __ge__(self, o):
        return self._binary("model.ge", o)

    def __and__(self, o):
        return self._binary("model.and", o)

    def __rand__(self, o):
        return self._binary("model.and", o, reflected=True)

    def __or__(self, o):
        return self._binary("model.or", o)

    def __ror__(self, o):
        return self._binary("model.or", o, reflected=True)

    # -- lookups: a table indexed by this value ------------------------------ #

    def _gather_from(self, parent: Param):
        """`table[x]` for a register array: the table is a symbol, named."""
        self.t.table(parent, parent.name)
        return self._new(self.t.b.op("model.gather", self.v,
                                     table=parent.name))

    def _rom_from(self, rom):
        """`rom[x]` for a constant table: the table is a symbol, named."""
        self.t.table(rom, rom.name)
        return self._new(self.t.b.op("model.rom", self.v, table=rom.name))

    # -- position: the driving indexes, handed to the model as values ------ #

    def _coords(self):
        """`coords(img)`: the input's row and column. One node, two
        results -- the two driving indexes of the input raster."""
        h, w = self.shape[0], self.shape[1]
        n = self.t.b.node("model.coords", [self.v], {"rows": h, "cols": w})
        return (Array(self.t, n.result(0), (h, 1)),
                Array(self.t, n.result(1), (1, w)))

    def _make_coords(self, rows, cols):
        """`make_coords(img, rows, cols)`: a declared OUTPUT raster's row and
        column -- its driving indexes."""
        n = self.t.b.node("model.make_coords", [self.v],
                          {"rows": rows, "cols": cols})
        return (Array(self.t, n.result(0), (rows, 1)),
                Array(self.t, n.result(1), (1, cols)))

    def _declared_read(self, key):
        """`img[y2 + dy, x2 + dx]`: a read through a declared raster's
        indexes. Each key is decoded to its Index -- `Index(rows) + dy` --
        and the keys stay operands, as the model wrote them."""
        (rk, ck), idx = key[:2], []
        for axis, k in enumerate((rk, ck)):
            if not isinstance(k, Array):
                raise Unsupported("a crop indexes both axes with output "
                                  "coordinates")
            idx.append(_decode_coordinate(k.v, axis))
        row, col = idx
        return Array(self.t, self.t.b.node(
            "model.read", [self.v, rk.v, ck.v],
            {"row": row.as_attr(), "col": col.as_attr(),
             "declared": True}).value, (row.count, col.count))

    # anything else a model asks of a traced array is not recorded yet
    def __int__(self):
        raise Unsupported("a traced value used as a Python int")

    __index__ = __int__

    def __bool__(self):
        raise Unsupported("a traced value used as a condition")

    def astype(self, dtype):
        if isinstance(dtype, (int, np.integer)):
            spec = ("bits", int(dtype))
            bits, signed = int(dtype), False
        else:
            d = np.dtype(dtype)
            spec = ("dtype", d.str)
            bits, signed = d.itemsize * 8, d.kind == "i"
        return self._new(self.t.b.op("model.astype", self.v, bits=bits,
                                     signed=signed, spec=spec))

    # -- slicing: a read through one Index per axis -------------------------- #

    def __getitem__(self, key):
        key = _expand(key, len(self.shape))
        if any(isinstance(k, Array) for k in key):
            return self._declared_read(key)
        if len(self.shape) == 3:
            return self._channel_read(key)
        if len(self.shape) != 2:
            raise Unsupported(f"indexing a {len(self.shape)}-D value")
        row = _index_of(key[0], self.shape[0], self.t)
        col = _index_of(key[1], self.shape[1], self.t)
        return self._new(self.t.b.op("model.read", self.v,
                                     row=row.as_attr(), col=col.as_attr()),
                         (row.count, col.count))

    def _channel_read(self, key):
        """A read of a multi-channel source: a third index, over channels.
        An integer channel drops the axis, as NumPy does."""
        row = _index_of(key[0], self.shape[0], self.t)
        col = _index_of(key[1], self.shape[1], self.t)
        k = key[2]
        integer = isinstance(k, (int, np.integer)) and not isinstance(k, bool)
        chan = _index_of(k, self.shape[2], self.t)
        shape = (row.count, col.count) + (() if integer else (chan.count,))
        whole = (row == Index(self.shape[0]) and col == Index(self.shape[1]))
        return self._new(self.t.b.op("model.read", self.v,
                                     row=row.as_attr(), col=col.as_attr(),
                                     chan=chan.as_attr(), drop=integer,
                                     whole=whole),
                         shape)

    # -- NumPy protocols ----------------------------------------------------- #

    def __array_function__(self, func, types, args, kwargs):
        if func is np.pad:
            return self._pad(*args[1:], **kwargs)
        if func in (np.empty_like, np.zeros_like):
            if kwargs or len(args) != 1:
                raise Unsupported(f"np.{func.__name__} with options")
            return Canvas(self, zeroed=func is np.zeros_like)
        if func is np.stack:
            return _stack(self.t, *args, **kwargs)
        if func is np.where:
            return _where(self.t, *args, **kwargs)
        raise Unsupported(f"np.{func.__name__}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise Unsupported(f"ufunc {ufunc.__name__}.{method}")
        if ufunc is np.absolute and len(inputs) == 1:
            return self._new(self.t.b.op("model.abs", self.v))
        binary = {np.add: ("__add__", "__radd__"),
                  np.subtract: ("__sub__", "__rsub__"),
                  np.multiply: ("__mul__", "__rmul__"),
                  np.less: ("__lt__", "__gt__"),
                  np.greater: ("__gt__", "__lt__")}
        if ufunc in binary and len(inputs) == 2:
            a, b = inputs
            name = binary[ufunc][0] if a is self else binary[ufunc][1]
            other = b if a is self else a
            return getattr(self, name)(other)
        raise Unsupported(f"np.{ufunc.__name__}")

    def _pad(self, pad_width, mode="constant", **kw):
        from .frontend import _norm_pad
        ((pt, pb), (pl, pr)) = _norm_pad(pad_width)
        edge = {"edge": "edge", "constant": "zero"}.get(mode)
        if edge is None:
            raise Unsupported(f"np.pad mode {mode!r}")
        if edge == "zero" and kw.get("constant_values", 0) != 0:
            raise Unsupported("np.pad with a non-zero constant")
        h, w = self.shape
        return self._new(self.t.b.op("model.pad", self.v,
                                     pad=(pt, pb, pl, pr), mode=edge),
                         (h + pt + pb, w + pl + pr))


class Canvas:
    """`out = np.empty_like(x)` followed by writes to strided planes.

    NumPy assembles it by mutation; the graph cannot, so the writes are
    collected here and become ONE node when the canvas is used: its operands
    are the template and the plane values, and each plane is stated by the
    two indexes it was written through -- a register start included.
    """

    def __init__(self, template: Array, zeroed: bool):
        self.template = template
        self.zeroed = zeroed
        self.shape = template.shape
        self.planes: list = []          # [(row Index, col Index, Array)]
        self._array = None

    def __setitem__(self, key, value):
        if self._array is not None:
            raise Unsupported("a canvas written after it was used")
        if not isinstance(key, tuple) or len(key) != 2:
            raise Unsupported(f"a canvas write at {key!r}")
        if not isinstance(value, Array):
            raise Unsupported(f"a canvas plane of {type(value).__name__}")
        row = _index_of(key[0], self.shape[0], self.template.t)
        col = _index_of(key[1], self.shape[1], self.template.t)
        self.planes.append((row, col, value))

    def array(self) -> Array:
        if self._array is None:
            t = self.template.t
            node = t.b.node(
                "model.canvas",
                [self.template.v] + [v.v for _, _, v in self.planes],
                {"planes": tuple((r.as_attr(), c.as_attr())
                                 for r, c, _ in self.planes),
                 "zeroed": self.zeroed})
            self._array = Array(t, node.value, self.shape)
        return self._array

    # what a model does with a finished canvas acts on its value
    def astype(self, dtype):
        return self.array().astype(dtype)

    def clip(self, lo, hi):
        return self.array().clip(lo, hi)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.stack:
            return _stack(self.template.t, *args, **kwargs)
        raise Unsupported(f"np.{func.__name__} of a canvas")


def _where(t, cond, a, b, **kwargs):
    """np.where: a select, the condition a register or a traced predicate."""
    if kwargs:
        raise Unsupported(f"np.where options {sorted(kwargs)}")
    ops, shape = [], None
    for x in (cond, a, b):
        x = x.array() if isinstance(x, Canvas) else x
        if isinstance(x, Array):
            ops.append(x.v)
            shape = shape or x.shape
        elif isinstance(x, Param):
            ops.append(t.param(x))
        elif isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            ops.append(t.b.const(int(x)))
        else:
            raise Unsupported(f"np.where operand {type(x).__name__}")
    return Array(t, t.b.op("model.where", *ops), shape)


def _stack(t, arrays, axis=0, **kwargs):
    """np.stack: values assembled along a new channel axis."""
    if kwargs:
        raise Unsupported(f"np.stack options {sorted(kwargs)}")
    vals = []
    for a in arrays:
        a = a.array() if isinstance(a, Canvas) else a
        if not isinstance(a, Array):
            raise Unsupported(f"np.stack of {type(a).__name__}")
        vals.append(a)
    shape = vals[0].shape
    node = t.b.node("model.stack", [a.v for a in vals], {"axis": int(axis)})
    ndim = len(shape) + 1
    ax = axis % ndim
    return Array(t, node.value, shape[:ax] + (len(vals),) + shape[ax:])


def _decode_coordinate(v: G.Value, axis: int) -> Index:
    """An output coordinate plus constants, as an Index: `y2 + 2` reads row
    r + 2 for every output row r. Anything else is refused by name."""
    offset = 0
    while v.node.kind == "model.add":
        a, b = v.node.operands
        if b.node.kind == "model.const":
            offset, v = offset + b.node.attr("value"), a
        elif a.node.kind == "model.const":
            offset, v = offset + a.node.attr("value"), b
        else:
            break
    if v.node.kind != "model.make_coords" or v.index != axis:
        raise Unsupported(f"an index on axis {axis} that is not an output "
                          "coordinate plus a constant")
    count = v.node.attr("rows" if axis == 0 else "cols")
    return Index(count) + offset


def _expand(key, ndim: int) -> tuple:
    """A key as one entry per axis: a lone key is the first axis, and an
    Ellipsis stands for as many full slices as it takes."""
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


def _index_of(k, dim: int, tracer) -> Index:
    """The Index a key reads from an axis of `dim` -- a constant slice or
    integer, or a stride-2 slice whose start may be a register."""
    if isinstance(k, slice) and k.step not in (None, 1):
        step = int(k.step)
        if k.stop is not None:
            raise Unsupported("a strided slice with a stop")
        base = Index(dim // step) * step
        start = 0 if k.start is None else k.start
        if isinstance(start, PhaseRef):
            tracer.param(start.param)
            return base + start
        if isinstance(start, Param):
            tracer.param(start)
            return base + start
        return base + int(start)
    try:
        return Index.from_slice(k, dim)
    except (TypeError, IndexError) as e:
        raise Unsupported(f"index {k!r}: {e}") from None


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #

class Tracer:
    """One trace: its graph builder, and the registers it read."""

    def __init__(self):
        # A faithful record: no folding, no merging, operand order as
        # written. Canonicalising is a pass, run on the graph afterwards.
        self.b = G.Builder(fold=False, locations=True, intern=False)
        self._params: dict[str, G.Value] = {}

    def table(self, obj, name: str):
        """A register array or constant table read by a traced index: one
        named declaration, stating the range its entries take."""
        if isinstance(obj, Param):
            lo = -(1 << (obj.bits - 1)) if obj.signed else 0
            hi = (1 << (obj.bits - (1 if obj.signed else 0))) - 1
        else:
            lo, hi = int(obj.lo), int(obj.hi)
        self.b.symbols.setdefault(name, {"object": obj, "lo": lo, "hi": hi})

    def param(self, p: Param) -> G.Value:
        """A register: one named declaration, one read node per register."""
        known = self.b.symbols.get(p.name)
        if known is not None:
            k = known["object"]
            if (k.bits, bool(k.signed)) != (p.bits, bool(p.signed)):
                raise Unsupported(f"two registers named {p.name!r}")
        self.b.symbols.setdefault(p.name, {"object": p})
        if p.name not in self._params:
            self._params[p.name] = self.b.op(
                "model.param", name=p.name, bits=p.bits,
                signed=bool(p.signed))
        return self._params[p.name]


class Traced:
    """What `trace` returns: the graph's outputs, and its symbols."""

    def __init__(self, image: Image2D, outputs: tuple, shapes: tuple,
                 symbols: dict, source: G.Value):
        self.image = image
        self.outputs = outputs
        self.shapes = shapes
        self.symbols = symbols
        self.source = source

    def dump(self) -> str:
        return G.dump(self.outputs)


def trace(fn, image: Image2D, *params, channels: int = 1) -> Traced:
    """Run the model on a traced image; return the graph it built.

    With `channels`, the source is (H, W, C): each pixel's word carries C
    fields, channel 0 in the low bits -- how that axis is laid out at the
    stream boundary, a fact of the source."""
    t = Tracer()
    if channels > 1:
        if image.bits % channels:
            raise Unsupported(f"a {image.bits}-bit word in {channels} fields")
        field = image.bits // channels
        lo, hi = 0, (1 << field) - 1
        shape = (image.height, image.width, channels)
    else:
        lo = -(1 << (image.bits - 1)) if image.signed else 0
        hi = ((1 << (image.bits - 1)) - 1) if image.signed else \
            (1 << image.bits) - 1
        shape = (image.height, image.width)
    src = t.b.op("model.source", image=image.name, lo=lo, hi=hi,
                 channels=channels)
    arg = Array(t, src, shape)
    # Registers are recorded when the model USES them, so the arguments are
    # handed over as given -- Params, a namespace's view, anything.
    out = fn(arg, *params)
    if isinstance(out, Canvas):
        out = out.array()
    if not isinstance(out, Array):
        raise Unsupported(f"a model returning {type(out).__name__}")
    return Traced(image, (out.v,), (out.shape,), t.b.symbols, src)


# --------------------------------------------------------------------------- #
# Evaluating
# --------------------------------------------------------------------------- #

def _slice_of(a: tuple, env: dict):
    """The slice that reads the Index `a` -- what the model wrote. A register
    in the start stands for whatever `env` evaluates that register to."""
    ix = Index.from_attr(a)
    if ix.step == 1:
        return slice(ix.start, ix.start + ix.count)
    if not ix.regs:
        return slice(ix.start, None, ix.step)
    ((name, coef),) = ix.regs
    p = env["params"][name]
    if coef == 1 and ix.start == 0:
        return slice(PhaseRef(p), None, ix.step)
    if coef == -1 and ix.start == 1:
        return slice(PhaseRef(p, invert=True), None, ix.step)
    raise Unsupported(f"a register start {ix}")


def _astype_arg(spec):
    kind, v = spec
    return v if kind == "bits" else np.dtype(v)


_EVAL = {
    "model.add": lambda n, a, s: a[0] + a[1],
    "model.sub": lambda n, a, s: a[0] - a[1],
    "model.mul": lambda n, a, s: a[0] * a[1],
    "model.neg": lambda n, a, s: -a[0],
    "model.floordiv": lambda n, a, s: a[0] // n.attr("divisor"),
    "model.shr": lambda n, a, s: a[0] >> n.attr("amount"),
    "model.clip": lambda n, a, s: a[0].clip(n.attr("lo"), n.attr("hi")),
    "model.astype": lambda n, a, s: a[0].astype(_astype_arg(n.attr("spec"))),
    "model.read": lambda n, a, s: _eval_read(n, a, s),
    "model.pad": lambda n, a, s: np.pad(
        a[0], ((n.attr("pad")[0], n.attr("pad")[1]),
               (n.attr("pad")[2], n.attr("pad")[3])),
        mode="edge" if n.attr("mode") == "edge" else "constant"),
    "model.canvas": lambda n, a, s: _eval_canvas(n, a, s),
    "model.lt": lambda n, a, s: a[0] < a[1],
    "model.gt": lambda n, a, s: a[0] > a[1],
    "model.le": lambda n, a, s: a[0] <= a[1],
    "model.ge": lambda n, a, s: a[0] >= a[1],
    "model.and": lambda n, a, s: a[0] & a[1],
    "model.or": lambda n, a, s: a[0] | a[1],
    "model.abs": lambda n, a, s: np.absolute(a[0]),
    "model.where": lambda n, a, s: np.where(a[0], a[1], a[2]),
    "model.gather": lambda n, a, s: s["params"][n.attr("table")][a[0]],
    "model.rom": lambda n, a, s: s["params"][n.attr("table")][a[0]],
    "model.stack": lambda n, a, s: np.stack(list(a), axis=n.attr("axis")),
    "model.coords": lambda n, a, s: _coords_of(a[0]),
    "model.make_coords": lambda n, a, s: _make_coords_of(
        a[0], n.attr("rows"), n.attr("cols")),
    "model.const": lambda n, a, s: n.attr("value"),
    "model.param": lambda n, a, s: s["params"][n.attr("name")],
}


def _eval_read(n, a, s):
    if n.attr("declared"):
        return a[0][a[1], a[2]]         # through the model's coordinates
    row, col = _slice_of(n.attr("row"), s), _slice_of(n.attr("col"), s)
    chan = n.attr("chan")
    if chan is None:
        return a[0][row, col]
    c = Index.from_attr(chan)
    ch = c.start if n.attr("drop") else slice(c.start, c.start + c.count)
    if n.attr("whole"):
        return a[0][..., ch]              # every row and column: pixel[..., c]
    return a[0][row, col, ch]


def _coords_of(img):
    from .frontend import coords
    return coords(img)


def _make_coords_of(img, rows, cols):
    from .frontend import make_coords
    return make_coords(img, rows, cols)


def _eval_canvas(n, a, s):
    template, planes = a[0], a[1:]
    make = np.zeros_like if n.attr("zeroed") else np.empty_like
    out = make(template)
    for (row, col), value in zip(n.attr("planes"), planes):
        out[_slice_of(row, s), _slice_of(col, s)] = value
    return out


def evaluate(traced: Traced, source, params: dict):
    """Evaluate the graph: `source` stands for the image, `params` maps each
    register's name to what stands for it. Returns the outputs' values."""
    env = {"params": params}
    values: dict[int, object] = {}
    for n in G.walk(traced.outputs):
        if n is traced.source.node:
            values[id(n)] = source
            continue
        fn = _EVAL.get(n.kind)
        if fn is None:
            raise Unsupported(f"no evaluation for {n.kind}")
        values[id(n)] = fn(n, [_value(values, v) for v in n.operands], env)
    return tuple(_value(values, v) for v in traced.outputs)


def _value(values: dict, v: G.Value):
    """What a value evaluated to: a multi-result node evaluates to a tuple,
    and the value is one element of it."""
    out = values[id(v.node)]
    return out[v.index] if len(v.node.results) > 1 else out


# --------------------------------------------------------------------------- #
# The new path, lowered by the reference compiler
# --------------------------------------------------------------------------- #

def to_ir(fn, image: Image2D, *params, out_bits=None, channels=1):
    """Trace a model and lower it. The new compiler (np2hw.core) takes it if
    it can; a model it does not reach yet goes to the reference. The
    variable NP2HW_CORE=off forces the reference, =strict refuses it, and
    NP2HW_CORE_LOG names a file recording the path each model took."""
    import os
    from .ir import Params
    if len(params) == 1 and isinstance(params[0], Params):
        params = (params[0].trace_view(),)   # the Param-valued view
    core = os.environ.get("NP2HW_CORE", "")
    if core != "off" and out_bits is None:
        from .core.pipeline import compile as core_compile
        try:
            mod = core_compile(fn, image, *params, channels=channels)
            _core_log(fn, "core")
            return None, mod
        except Exception as e:
            _core_log(fn, f"reference: {type(e).__name__}: {e}")
            if core == "strict":
                raise
    return reference(trace(fn, image, *params, channels=channels),
                     channels=channels, out_bits=out_bits)


def _core_log(fn, what):
    import os
    path = os.environ.get("NP2HW_CORE_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(f"{getattr(fn, '__name__', fn)}\t{what}\n")


def reference(traced: Traced, channels: int = 1, out_bits=None):
    """The reference lowering: the graph evaluated onto the older frontend's
    values, lowered by it -- the yardstick while the new compiler grows."""
    from . import frontend as F
    old_source = F.Traced.image_input(traced.image)
    if channels > 1:
        old_source = F.Channels(old_source, channels)
    (value,) = evaluate(traced, old_source,
                        {n: s["object"] for n, s in traced.symbols.items()})
    if isinstance(value, F.ChannelStack):
        value.in_channels = channels
    if isinstance(value, (F.Mux, F.PhaseCanvas, F.ChannelStack)):
        return None, value            # generate() handles these terminals
    if not isinstance(value, F.Traced):
        raise Unsupported(f"the reference lowers {type(value).__name__}")
    if out_bits is not None:
        value = value.astype(out_bits)
    return F.lower(value)
