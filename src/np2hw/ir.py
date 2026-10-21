# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""What a model is traced on and reads: the image it is handed (Image2D),
its registers (Param, Params), a register that names a CFA phase
(PhaseRef), and its constant tables (Rom). Plain descriptions -- the
compiler (np2hw.core) decides what each becomes in hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Leaf descriptors
# --------------------------------------------------------------------------- #

class Rom:
    """A CONSTANT lookup table: contents fixed at generation time.

    A `Param` array is one register per entry -- exactly right for the
    33 knots of a tone curve, and absurd for the 16384 pixels of a
    logo. A Rom is the same read with the contents baked, addressed by a
    data-derived index. How it is built is the device's question, not
    this object's (np2hw.timing.Device.rom_in_logic): a table small
    enough is logic, its contents constants read in the stage that uses
    them; a larger one is an initialised memory read through a register,
    a stage of its own, so the tools infer block RAM rather than a mux
    tree the size of the table. Either way it reads the same values.

    Indexed by a TRACED value it becomes hardware; indexed by an
    ordinary integer or array it is plain NumPy, so a model that reads
    a table stays runnable -- and stays its own oracle.
    """

    def __init__(self, data, name="rom", bits=None, signed=None):
        self.data = np.asarray(data)
        if self.data.ndim != 1:
            raise NotImplementedError(
                f"Rom {name!r}: only 1-D tables in this version; flatten "
                "the index (row*WIDTH + col) in the model, where the "
                "arithmetic is visible")
        self.name = name
        self.lo = int(self.data.min()) if self.data.size else 0
        self.hi = int(self.data.max()) if self.data.size else 0
        self.signed = (self.lo < 0) if signed is None else bool(signed)
        span = max(self.hi.bit_length(),
                   (-self.lo - 1).bit_length() if self.lo < 0 else 0)
        self.bits = bits if bits is not None else max(
            1, span + (1 if self.signed else 0))

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        # A traced index is a hardware read; anything else is the model
        # reading its own table.
        if hasattr(idx, "_rom_from"):
            return idx._rom_from(self)
        return self.data[idx]


@dataclass(frozen=True)
class Image2D:
    name: str
    width: int
    height: int
    bits: int = 8
    signed: bool = False
    # A HISTORY when set: the model is handed the stream's recent frames,
    # (T, H, W), `frames` long -- the length it is traced at, enough for its
    # reads across time. The stream has no end; only the reads' spread on
    # the frame axis is hardware.
    frames: int = 0


class Param:
    """A register-backed value (a config register). Declare its type with a
    NumPy dtype -- Param("gain", np.uint8) / Param("bias", np.int16) -- which
    sets width and signedness; or pass bits=/signed= for non-standard widths.

    A `shape` makes it a MATRIX of registers (e.g. a programmable 3x3 kernel):
    it holds no value itself; indexing kernel[i, j] returns the scalar Param
    `kernel_i_j`. The IR only ever sees scalar Params -- the matrix is just a
    naming/collection convenience at the front end."""
    def __init__(self, name, dtype=None, *, bits=8, signed=False, shape=(),
                 default=0, description="", labels=(), frame=False):
        self.name = name
        # A value each FRAME brings -- the frame's own context, set as it
        # starts -- rather than one software holds steady across frames. A
        # core takes it with the frame's first pixel and carries it with the
        # frame through its stages, so the frame before, still inside, keeps
        # its own.
        self.frame = bool(frame)
        # Carried into the generated Verilog as the port's comment. A register
        # whose meaning is only in the Python is a register the person reviewing
        # the RTL has to guess at.
        self.description = description
        # What each element of a SHAPED Param means, nested to match `shape`:
        # (("R", "Gr"), ("Gb", "B")) for a CFA array. An index says where a
        # register sits; a label says what it is, and that is what the person
        # reading the generated RTL needs. Without it every leaf repeats the
        # parent's description and the index is the only thing telling them
        # apart, which is no help at all.
        self.labels = tuple(labels)
        if dtype is not None:
            d = np.dtype(dtype)
            self.bits = d.itemsize * 8
            self.signed = d.kind == "i"
        else:
            self.bits = bits
            self.signed = signed
        self.shape = tuple(shape)
        self.default = int(default)          # register reset/power-on value

    def __getitem__(self, idx):
        if not self.shape:
            raise TypeError(f"Param {self.name!r} is scalar, not indexable")
        # A TRACED index is a runtime lookup: the whole register array becomes
        # a mux selected by a data-derived value -- the LUT case. Delegated to
        # the frontend, which owns traced values.
        if hasattr(idx, "_gather_from"):
            return idx._gather_from(self)
        idx = idx if isinstance(idx, tuple) else (idx,)
        flat = "_".join(str(int(i)) for i in idx)
        return Param(f"{self.name}_{flat}", bits=self.bits, signed=self.signed,
                     default=self.default,
                     description=self._leaf_description(idx))

    def _leaf_description(self, idx):
        """One element's description, prefixed with its label when there is one."""
        node = self.labels
        for step in idx:
            if not isinstance(node, (tuple, list)) or int(step) >= len(node):
                return self.description
            node = node[int(step)]
        return f"{node}: {self.description}" if self.description else str(node)

    # -- arithmetic ---------------------------------------------------------- #
    #
    # A register is a scalar input, and arithmetic on it is the model's math,
    # as on any value -- while np2hw.core traces the model, where it is a
    # value with a range like any other. Outside such a trace a Param has
    # no arithmetic but the COMPLEMENT a phase needs (`1 - py`, `py ^ 1`),
    # a PhaseRef, so a slice start written with it stays a phase.

    def _traced(self):
        """This register as a value of the trace running, or None."""
        from .core.trace import Array, active
        t = active()
        if t is None or self.shape:
            return None
        return Array(t, t.register(self))

    def _arith(op, reflected=False):
        def method(self, other):
            v = self._traced()
            if v is None:
                return NotImplemented
            return op(other, v) if reflected else op(v, other)
        return method

    __add__ = _arith(lambda a, b: a + b)
    __radd__ = _arith(lambda a, b: a + b, True)
    __sub__ = _arith(lambda a, b: a - b)
    __mul__ = _arith(lambda a, b: a * b)
    __rmul__ = _arith(lambda a, b: a * b, True)
    __and__ = _arith(lambda a, b: a & b)
    __rand__ = _arith(lambda a, b: a & b, True)
    __or__ = _arith(lambda a, b: a | b)
    __ror__ = _arith(lambda a, b: a | b, True)
    __rshift__ = _arith(lambda a, b: a >> b)
    __floordiv__ = _arith(lambda a, b: a // b)
    __mod__ = _arith(lambda a, b: a % b)
    del _arith

    def __neg__(self):
        v = self._traced()
        return NotImplemented if v is None else -v

    # Compared while a model is traced, a register is a value like any
    # other, `p.pos == 1` a comparator. Outside a trace a Param is a
    # declaration, equal only to itself -- and hashed as one.
    def __eq__(self, other):
        v = self._traced()
        return (self is other) if v is None else v == other

    def __ne__(self, other):
        v = self._traced()
        return (self is not other) if v is None else v != other

    def __lt__(self, other):
        v = self._traced()
        return NotImplemented if v is None else v < other

    def __le__(self, other):
        v = self._traced()
        return NotImplemented if v is None else v <= other

    def __gt__(self, other):
        v = self._traced()
        return NotImplemented if v is None else v > other

    def __ge__(self, other):
        v = self._traced()
        return NotImplemented if v is None else v >= other

    __hash__ = object.__hash__

    def __xor__(self, other):
        if other in (0, 1):
            return PhaseRef(self, bool(other))
        return NotImplemented

    __rxor__ = __xor__

    def __rsub__(self, other):
        if other == 1:
            return PhaseRef(self, invert=True)
        v = self._traced()
        return NotImplemented if v is None else other - v

    def __repr__(self):
        sh = f" {self.shape}" if self.shape else ""
        return f"Param({self.name!r}{sh}, {self.bits}b{'s' if self.signed else 'u'})"


@dataclass(frozen=True)
class PhaseRef:
    """A slice start that is a REGISTER rather than a constant.

    `img[p.py::2]` takes every other row starting at whichever row the register
    `py` selects, and `img[1 - p.py::2]` takes the other half. In hardware that
    is one comparison against the row counter's low bit, XORed with the register
    -- so which half of the image a plane refers to becomes a two-bit write
    rather than a rebuild.

    That is what lets ONE bitstream serve every CFA order: the phase moves, the
    gateware does not. `invert` is the complementary half, spelled either
    `1 - p.py` or `p.py ^ 1`.
    """

    param: "Param"
    invert: bool = False

    def __xor__(self, other):
        if other in (0, 1):
            return PhaseRef(self.param, self.invert ^ bool(other))
        return NotImplemented

    def __rsub__(self, other):
        if other == 1:
            return PhaseRef(self.param, not self.invert)
        return NotImplemented

    def expr(self) -> str:
        """Verilog for the phase this reference selects."""
        return (f"(~param_{self.param.name})" if self.invert
                else f"param_{self.param.name}")

    def __repr__(self):
        return f"{'1-' if self.invert else ''}{self.param.name}"


class Params:
    """A named collection of config registers, for ISPs with many parameters.

    Declare once and pass the SAME object to the model as its single param arg:

        P = Params([Param("gain", np.uint8, default=16),
                    Param("ccm",  np.int16, shape=(3, 3))])
        def model(img, p):                 # access registers BY NAME
            return (p.gain * (p.ccm[0, 0] * img)) // 16

    The tool feeds the model a Param-valued view when tracing (-> hardware
    registers) and a value-valued view when running NumPy (-> ints / ndarrays),
    so the one function is both spec and hardware. Each leaf is a scalar Param,
    so the register file / AXI-Lite / UMI / live sliders all enumerate leaves()."""

    def __init__(self, params):
        self._params = list(params)
        self._by_name = {p.name: p for p in self._params}

    def __iter__(self):
        return iter(self._params)

    def leaves(self):
        """Flatten to scalar Params: a shaped Param expands to its name_i_j."""
        out = []
        for p in self._params:
            if p.shape:
                for idx in np.ndindex(p.shape):
                    out.append(p[idx])
            else:
                out.append(p)
        return out

    def trace_view(self):
        """Attribute access returns the declared Param objects (for tracing)."""
        return _ParamsView(self._by_name, None)

    def descriptions(self):
        """{scalar register name: description} for every declared leaf."""
        return {p.name: p.description for p in self.leaves() if p.description}

    def bind(self, values):
        """Attribute access returns int / ndarray values (for the NumPy run).
        `values` is keyed by LEAF scalar name (e.g. 'gain', 'ccm_0_0')."""
        return _ParamsView(self._by_name, dict(values))


def np_dtype(bits, signed):
    """Smallest NumPy int dtype holding `bits` with the right signedness. Param
    values use this so the NumPy oracle promotes EXACTLY like the hardware
    (e.g. uint16 + int16 register -> int32), instead of hitting NEP-50 limits."""
    nb = next(w for w in (8, 16, 32, 64) if bits <= w)
    return np.dtype(f"{'int' if signed else 'uint'}{nb}")


class _ParamsView:
    """Bound view of a Params set. values=None -> trace (yields Param objects);
    values=dict -> NumPy (yields a typed scalar for a scalar Param, an ndarray
    for a shaped Param, assembled from its leaf values)."""

    def __init__(self, by_name, values):
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name):
        by_name = object.__getattribute__(self, "_by_name")
        values = object.__getattribute__(self, "_values")
        if name not in by_name:
            raise AttributeError(name)
        p = by_name[name]
        if values is None:                       # trace: hand back the Param
            return p
        dt = np_dtype(p.bits, p.signed)
        if not p.shape:                          # NumPy: typed scalar value
            return dt.type(values[p.name])
        arr = np.zeros(p.shape, dtype=dt)        # NumPy: assemble from leaves
        for idx in np.ndindex(p.shape):
            arr[idx] = values[p[idx].name]
        return arr
