"""np2hw IR — line-based streaming intermediate representation.

This is the layer that makes the hardware structure explicit (per idea.txt):
the pipeline is a graph of 2D *lines*. Every line is one of three concrete
kinds, matching the three ways idea.txt builds a line:

  SourceLine      a source (image OR another line) read at a vertical offset.
                  Pure addressing -- spends no compute and no line buffers;
                  it sets the line's vertical `lag`.
  HProcLine  an op applied to ONE line -> within-row taps
                  -> a SHIFT REGISTER of depth (max-min column offset).
  VProcLine    an op applied to a GROUP of lines -> across-row combine
                  -> LINE BUFFERS equal to the span of input lags.

Because horizontal vs vertical is now a type (not a string flag), the cost of
each line is a property of its class, and the delay-line totals fall out by
summation. This is the deliverable of step 1: read off how many line buffers
and how deep a shift register a design needs.

Vertical model (how delay lines are counted):
  Each line has `lag` = how many rows behind the input front its data sits.
    SourceLine(image, start=s)      -> lag = s
    SourceLine(line,  start=s)      -> lag = line.lag + s
    HProcLine                  -> lag unchanged
    VProcLine over lags L        -> line buffers = max(L) - min(L);
                                       output lag = min(L)
  Total line buffers = sum over VProcLines (exact for feed-forward
  cascades; an upper bound if buffers could be physically shared).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _clog2(n: int) -> int:
    return math.ceil(math.log2(n)) if n > 1 else 0


def _is_const(c):       # forward-friendly check (Const defined below)
    return type(c).__name__ == "Const"


def _term_bits(coeff, vbits: int) -> int:
    """Bitwidth of coeff * value, where value is `vbits` wide."""
    if _is_const(coeff):
        return max(1, (coeff.value * ((1 << vbits) - 1)).bit_length())
    return vbits + coeff.bits                       # Param: unknown value


def _mac_out_bits(terms) -> int:
    """Output bits of Σ coeffₖ·valueₖ for terms = [(coeff, value_bits), ...],
    with non-negative inputs in [0, 2^b-1]. Range-based, so negative Const coeffs
    (signed result) add a sign bit. A safe bound when any coeff is a Param."""
    if all(_is_const(c) for c, _ in terms):
        hi = sum(c.value * ((1 << b) - 1) for c, b in terms if c.value > 0)
        lo = sum(c.value * ((1 << b) - 1) for c, b in terms if c.value < 0)
        if lo < 0:                                       # signed range [lo, hi]
            return 1 + max(max(1, hi.bit_length()), (-lo).bit_length())
        return max(1, hi.bit_length())
    return max(_term_bits(c, b) for c, b in terms) + _clog2(len(terms))


def _coeff_str(c) -> str:
    return str(c.value) if _is_const(c) else c.name


# --------------------------------------------------------------------------- #
# Leaf descriptors
# --------------------------------------------------------------------------- #

@dataclass
class PExpr:
    """One node of a POINTWISE expression DAG.

    The trailing-op chain describes a straight line of operations on one
    value; a lookup table cannot be written that way -- it derives a segment
    index and a fraction from the same input, reads two table entries, and
    recombines. This is that shape: a small DAG whose leaves are the spatial
    result ('acc'), constants, and register-array reads ('gather'), and whose
    interior is ordinary arithmetic. Every node carries the exact value
    range it can take, because the emitter sizes every wire from the range,
    not from a container width.

    ops: 'acc' | 'const' | 'gather' | 'add' | 'sub' | 'mul' | 'shr'
         | 'mask' | 'clip'
    """

    op: str
    args: tuple
    lo: int
    hi: int


@dataclass
class ExprLine:
    """A pointwise model whose datapath is a PExpr DAG. No line buffers:
    v1 expression models are strictly pointwise -- a stencil that also
    gathers is a later composition."""

    image: "Image2D"
    root: PExpr
    params: tuple            # shaped parent Params read by gathers, in first-use order
    out_bits: int
    out_signed: bool


@dataclass(frozen=True)
class Indexer:
    """A one-axis access pattern: walk from `start` by `step`, `count` taps."""
    start: int = 0
    step: int = 1
    count: int = 1

    def combine(self, child: "Indexer") -> "Indexer":
        """Fold a child indexer expressed in THIS indexer's coordinates."""
        return Indexer(start=self.start + child.start * self.step,
                       step=self.step * child.step,
                       count=child.count)


@dataclass(frozen=True)
class Image2D:
    name: str
    width: int
    height: int
    bits: int = 8
    signed: bool = False


class Param:
    """A register-backed value (a config register). Declare its type with a
    NumPy dtype -- Param("gain", np.uint8) / Param("bias", np.int16) -- which
    sets width and signedness; or pass bits=/signed= for non-standard widths.

    A `shape` makes it a MATRIX of registers (e.g. a programmable 3x3 kernel):
    it holds no value itself; indexing kernel[i, j] returns the scalar Param
    `kernel_i_j`. The IR only ever sees scalar Params -- the matrix is just a
    naming/collection convenience at the front end."""
    def __init__(self, name, dtype=None, *, bits=8, signed=False, shape=(),
                 default=0, description="", labels=()):
        self.name = name
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

    # -- phase arithmetic ---------------------------------------------------- #
    #
    # Deliberately minimal: a Param may be used as a slice start, and its
    # COMPLEMENT may be taken. That is all a phase needs, and it is all that
    # lowers to a single XOR against a counter bit. General arithmetic on
    # registers is a different feature and would lower to real logic.

    def __xor__(self, other):
        if other in (0, 1):
            return PhaseRef(self, bool(other))
        return NotImplemented

    __rxor__ = __xor__

    def __rsub__(self, other):
        if other == 1:
            return PhaseRef(self, invert=True)
        return NotImplemented

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


@dataclass(frozen=True)
class Const:
    """A compile-time constant. Accepts a NumPy scalar (np.int16(5)) -- its
    value is taken; signedness follows the sign of the value."""
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            object.__setattr__(self, "value", int(self.value))


@dataclass(frozen=True)
class View2D:
    """image + row/col indexer; folds when derived from another view."""
    image: Image2D
    row: Indexer
    col: Indexer

    def derive(self, row: Indexer, col: Indexer) -> "View2D":
        return View2D(self.image, self.row.combine(row), self.col.combine(col))


class ImageOp:
    """An operation: name + args + optional coefficients.

      'add'        unit-weight sum of taps/lines
      'mac'        weighted sum: coeffs pair with indexers (horizontal) or with
                   the input lines (vertical), positionally
      'mul'        pointwise multiply by a single Const/Param coefficient
      'div'        pointwise divide by a power-of-2 Const (a right shift)

    Indexer args are taps on ONE axis. Const/Param args (or the `coeffs=` list)
    are coefficients; when `coeffs=` is given it pairs positionally with the
    indexers (or, for a vertical op, with the lines passed to vertical())."""
    def __init__(self, name, *args, coeffs=None):
        self.name = name
        self.indexers = [a for a in args if isinstance(a, Indexer)]
        bare = [a for a in args if isinstance(a, (Param, Const))]
        self.coeffs = list(coeffs) if coeffs is not None else bare

    def __repr__(self):
        return f"ImageOp({self.name}, taps={len(self.indexers)}, coeffs={len(self.coeffs)})"


# --------------------------------------------------------------------------- #
# Line2D — base class for the streaming unit
# --------------------------------------------------------------------------- #

class Line2D:
    """Abstract base. Every node in the pipeline IS a Line2D; the concrete
    subclass records how it was produced and what hardware it costs."""
    label = "line"

    # set by ImageStreamer when the line is registered
    id: int | None = None

    # defaults; subclasses override where relevant
    image: Image2D | None = None
    inputs: tuple = ()
    line_buffers: int = 0      # vertical delay resources
    shift_depth: int = 0       # horizontal delay resources
    shape: tuple | None = None  # NumPy data shape (rows, cols) of this line
    clip_bits: int | None = None  # declared bits to KEEP (astype); None = full

    # (bits, signed) the model DECLARED for the spatial accumulator via astype.
    # SIGNEDNESS OF THE DATAPATH COMES FROM HERE, not from the value range: the
    # NumPy dtype the model was written in is the specification, so `astype(
    # np.int32)` means the accumulator is signed even when every value in it
    # happens to be positive. Width still comes from the range (so a declared
    # int32 does not cost a 32-bit adder), but the width must be able to hold
    # that range IN this signedness -- see _range_bits_as in verilog.py.
    declared: tuple | None = None

    def __init__(self, bits: int, lag: int):
        self.bits = bits
        self.lag = lag

    def opname(self) -> str:
        return self.label

    def detail(self) -> str:
        return ""

    def __repr__(self):
        return f"Line#{self.id}<{self.label} lag={self.lag} {self.bits}b>"


class SourceLine(Line2D):
    """A source read at a vertical offset. `source` is the image (idea.txt's
    "2D line = image + vertical indexer") or another line (a vertically-offset
    view of it). Pure addressing: costs nothing; just sets `lag`."""
    label = "source"

    def __init__(self, source, vindexer: Indexer):
        self.source = source
        self.vindexer = vindexer
        if isinstance(source, Image2D):
            self.image = source
            self.src_line = None
            super().__init__(bits=source.bits, lag=vindexer.start)
        elif isinstance(source, Line2D):
            self.image = source.image
            self.src_line = source
            super().__init__(bits=source.bits, lag=source.lag + vindexer.start)
        else:
            raise TypeError(
                f"SourceLine source must be Image2D or Line2D, got {type(source).__name__}")

    @property
    def inputs(self):
        return (self.src_line,) if self.src_line is not None else ()

    def opname(self):
        return "read"

    def detail(self):
        if self.src_line is None:
            return f"img={self.source.name} voffset={self.vindexer.start}"
        return f"from #{self.source.id} voffset={self.vindexer.start}"


class HProcLine(Line2D):
    """An op applied to ONE line (idea.txt's "horizontal processing").

    Column indexers are taps -> a shift register of depth (max-min offset);
    a pointwise op is the degenerate case (shift_depth == 0). Per-op bits:
      'add'/'mac'  weighted sum of taps -> grows per the coefficients
      'mul'        pointwise scale by Const/Param -> grows
      'div'        pointwise divide by power-of-2 Const -> right shift"""
    label = "horizontal"

    def __init__(self, op: ImageOp, line: Line2D):
        self.op = op
        self.line = line
        self.shift = 0
        self.shift_depth = 0
        # per-tap (coeff, col_offset); empty for pure pointwise mul/div
        self.taps = []

        if op.name == "div":
            c = op.coeffs[0].value
            self.shift = c.bit_length() - 1
            if (1 << self.shift) != c:
                raise ValueError(f"div by {c}: only power-of-2 supported in v1")
            bits = max(1, line.bits - self.shift)
        elif op.name == "trunc":                       # astype: keep low N bits
            n = op.coeffs[0].value
            self.clip_bits = n
            bits = min(line.bits, n)                    # widening declaration = no-op
        elif op.name == "clip":                        # saturate to [lo, hi]
            self.clip_lo = op.coeffs[0].value
            self.clip_hi = op.coeffs[1].value
            if self.clip_lo < 0:
                bits = 1 + max(max(1, self.clip_hi.bit_length()),
                               (-self.clip_lo).bit_length())
            else:
                bits = max(1, self.clip_hi.bit_length())
        elif op.name == "mul":                         # pointwise scale
            self.coeff = op.coeffs[0]
            bits = _term_bits(self.coeff, line.bits)
        elif op.name == "addp":                        # pointwise add (bias reg)
            self.coeff = op.coeffs[0]
            bits = max(line.bits, self.coeff.bits) + 1
        else:                                          # 'add' / 'mac'
            idxs = op.indexers or [Indexer(0)]
            coeffs = op.coeffs if op.coeffs else [Const(1)] * len(idxs)
            if len(coeffs) != len(idxs):
                raise ValueError("coeffs must pair 1:1 with indexers")
            self.taps = list(zip(coeffs, [ix.start for ix in idxs]))
            starts = [s for _, s in self.taps]
            self.shift_depth = max(starts) - min(starts)
            bits = _mac_out_bits([(c, line.bits) for c, _ in self.taps])
        super().__init__(bits=bits, lag=line.lag)

    @property
    def inputs(self):
        return (self.line,)

    def opname(self):
        return self.op.name

    def detail(self):
        if self.op.name == "div":
            return f"const={self.op.coeffs[0].value} (>>{self.shift})"
        if self.op.name == "trunc":
            return f"astype keep {self.clip_bits}b"
        if self.op.name == "clip":
            return f"saturate [{self.clip_lo}, {self.clip_hi}]"
        if self.op.name == "mul":
            return f"x {_coeff_str(self.coeff)}"
        if self.op.name == "addp":
            return f"+ {_coeff_str(self.coeff)}"
        ws = ",".join(_coeff_str(c) for c, _ in self.taps)
        return f"shiftreg_depth={self.shift_depth} coeffs=[{ws}]"


class VProcLine(Line2D):
    """An op applied to a GROUP of lines (weighted sum across rows) -> line
    buffers equal to the span between newest and oldest line consumed. For a
    'mac' op the coeffs pair positionally with the input lines."""
    label = "vertical"

    def __init__(self, op: ImageOp, lines):
        self.op = op
        self.lines = tuple(lines)
        lags = [ln.lag for ln in self.lines]
        self.line_buffers = max(lags) - min(lags)
        coeffs = op.coeffs if op.coeffs else [Const(1)] * len(self.lines)
        if len(coeffs) != len(self.lines):
            raise ValueError("vertical coeffs must pair 1:1 with lines")
        self.coeffs = coeffs
        terms = [(coeffs[k], self.lines[k].bits) for k in range(len(self.lines))]
        super().__init__(bits=_mac_out_bits(terms), lag=min(lags))

    @property
    def inputs(self):
        return self.lines

    def opname(self):
        return self.op.name

    def detail(self):
        ws = ",".join(_coeff_str(c) for c in self.coeffs)
        return f"line_buffers+={self.line_buffers} coeffs=[{ws}]"


# --------------------------------------------------------------------------- #
# ImageStreamer — the builder + analysis
# --------------------------------------------------------------------------- #

class ImageStreamer:
    def __init__(self):
        self._next = 0
        self._lines: list[Line2D] = []

    def _reg(self, line: Line2D) -> Line2D:
        line.id = self._next
        self._next += 1
        self._lines.append(line)
        return line

    # -- construction (idea.txt's three line-creation methods) --------------- #

    def line(self, source, vindexer: Indexer) -> SourceLine:
        return self._reg(SourceLine(source, vindexer))

    def horizontal(self, op: ImageOp, line: Line2D) -> HProcLine:
        return self._reg(HProcLine(op, line))

    def vertical(self, op: ImageOp, lines) -> VProcLine:
        return self._reg(VProcLine(op, lines))

    # -- analysis ------------------------------------------------------------ #

    def _cone(self, out: Line2D):
        seen, order = set(), []

        def visit(n):
            if n.id in seen:
                return
            seen.add(n.id)
            for i in n.inputs:
                visit(i)
            order.append(n)

        visit(out)
        return order

    def analyze(self, out: Line2D) -> dict:
        cone = self._cone(out)
        shift_regs = [(n.id, n.shift_depth) for n in cone if n.shift_depth]
        images = {n.image.name: n.image
                  for n in cone if isinstance(n, SourceLine) and n.src_line is None}
        return {
            "line_buffers": sum(n.line_buffers for n in cone),
            "shift_registers": shift_regs,
            "max_shift_depth": max((s for _, s in shift_regs), default=0),
            "output_bits": out.bits,
            "output_lag": out.lag,
            "images": images,
            "nodes": cone,
        }

    def report(self, out: Line2D):
        a = self.analyze(out)
        print(f"pipeline for {out}  ({len(a['nodes'])} lines)")
        print("  data images   : "
              + ", ".join(f"{im.name}({im.width}x{im.height})@{im.bits}b"
                          for im in a["images"].values()))
        print(f"  LINE BUFFERS  : {a['line_buffers']}")
        print(f"  shift regs    : depth {a['max_shift_depth']} "
              f"({len(a['shift_registers'])} horizontal stage(s))")
        sh = f", shape {out.shape}" if out.shape else ""
        print(f"  output        : {a['output_bits']} bits, lag {a['output_lag']} rows{sh}")
        for n in a["nodes"]:
            ins = ",".join(f"#{i.id}" for i in n.inputs)
            shp = f" shape={n.shape}" if n.shape else ""
            print(f"   #{n.id:<2} {n.label:<10} {n.opname():<7} [{ins:<7}] "
                  f"lag={n.lag} {n.bits}b{shp}  {n.detail()}")
