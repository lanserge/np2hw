# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Parameter declarations and register-map allocation.

RULE 2 lives here. A parameter is declared ONCE, by the block that owns it, in
LOCAL terms only -- a block never knows its absolute address. A single ``Param``
declaration drives four things:

  1. the arithmetic in the block's NumPy model (via :meth:`Param.to_np2hw`),
  2. the CSR width and offset in the generated Verilog,
  3. the register-map documentation,
  4. the host-side accessor.

Absolute addresses are assigned at composition time, per block INSTANCE, by
the pipeline composer. That is what makes a stereo pipeline (every block
instantiated twice) and standalone unit tests both work: the block declaration
is instance-free.

Three kinds of state, deliberately kept distinct
------------------------------------------------

``Param``      BLOCK-OWNED CONFIG. CCM coefficients, gamma tables, WB gains.
               Lives in the owning block's register map. Written to a shadow
               register and committed as one batch when software writes
               the pipeline's commit register; the datapath takes the batch at the next FRAME
               BOUNDARY, so a frame is never processed with half-updated
               coefficients.

``Context``    PIPELINE CONTEXT. Width, height, active window, Bayer phase, bit
               depth. Declared by the pipeline's context block, an ordinary block
               at base 0, and fanned out to consumers as WIRES; a fact the
               design names in ``stream.context`` arrives with the stream
               instead and takes no register. Never duplicated as a per-block
               CSR -- one owner per fact, not one per block.

Fixed-point convention
----------------------

``bits`` is the register width; ``frac`` is how many of those bits are
fractional. A gain declared ``Param("gain", bits=16, frac=8)`` is Q8.8, so the
integer 256 means 1.0. The model does the arithmetic in integers and shifts by
``frac``; ``frac`` exists so the documentation and the host API can present a
human number without anyone re-deriving the scale factor. Nothing in
a block's model ever converts a parameter to float.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

# Address decode is a bit-slice compare, not a range comparator: every block
# instance starts on a 256-byte boundary, so the decoder is `addr[N-1:8] == k`.
# Cheaper in hardware, and far more readable in the generated Verilog.
BLOCK_ALIGN = 256

# Local offset 0 of every block is the ID-and-version word, so that the host can
# read back a block and confirm the loaded bitstream matches the software that
# is about to drive it. Parameters therefore start at local offset 4.
ID_VERSION_OFFSET = 0
FIRST_PARAM_OFFSET = 4

# Register-map bus width. Every CSR is one 32-bit word; a parameter wider than
# 32 bits is rejected at declaration rather than silently split.
REG_WIDTH = 32

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_name(name: str, what: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"{what} name {name!r} must be lower_snake_case matching {_NAME_RE.pattern} "
            "-- names become Verilog identifiers, JSON keys and Python attributes")
    return name


def fnv1a16(text: str) -> int:
    """Stable 16-bit FNV-1a of a block's type name, used as its block ID.

    Derived rather than hand-assigned on purpose: a central registry of magic
    numbers is a merge-conflict generator and gets out of step with the code.
    The ID is emitted into the JSON register map, so host and hardware always
    agree on it; it changes if the block is renamed, and the separately declared
    version changes if the register layout moves.
    """
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return ((h >> 16) ^ h) & 0xFFFF


# --------------------------------------------------------------------------- #
# Param -- block-owned configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Param:
    """One configuration register (or a rectangular array of them).

    Args:
        name: lower_snake_case; becomes the Verilog signal, JSON key and host
            attribute. Unique within the declaring block.
        bits: register width. Must hold ``default`` and fit in one bus word.
        frac: fractional bits. ``bits=16, frac=8`` is Q8.8; 256 means 1.0.
        default: reset / power-on value, as the raw integer the register holds.
        signed: two's complement if True.
        shape: ``()`` for a scalar, ``(2, 2)`` for a Bayer-phase array, ``(3, 3)``
            for a CCM. A shaped Param is a naming convenience: it allocates one
            register per element, named ``<name>_<i>_<j>``, in row-major order.
        labels: what each element of a shaped Param MEANS, nested to match
            ``shape`` -- ``(("R", "Gr"), ("Gb", "B"))`` for a CFA array. An index
            says where a register sits; a label says what it is, and that is what
            somebody reading the generated RTL or the register map needs. Without
            it a leaf is described as "[element [0, 1]]", which is just the index
            again.
        description: one sentence, carried through to the generated Verilog
            comment, the JSON map and the documentation. Required -- rule 4 says
            someone will read the output in review.
    """

    name: str
    bits: int
    frac: int = 0
    default: int = 0
    signed: bool = False
    shape: tuple[int, ...] = ()
    description: str = ""
    labels: tuple = ()
    # Attributes a DESIGN may override at composition time ("bits", "frac",
    # "default"). Empty means this declaration is structural and fixed. The
    # whitelist is the block author's statement of what varies between
    # variants of this block -- a Q format usually does, a CCM's shape never.
    configurable: tuple = ()
    # The reset value IS the scale (1 << frac), re-derived whenever frac
    # changes. For gain-like registers the unity reset is a semantic
    # invariant, not a number: overriding frac without this would quietly
    # turn "pass-through at reset" into "x16 at reset".
    default_unity: bool = False
    # Each LEAF of a 1-D shaped Param resets to its point on the identity
    # ramp: leaf i = i * ((1 << (bits - 1)) / (N - 1)). The LUT counterpart
    # of default_unity -- an unconfigured lookup table is a pass-through for
    # a (bits - 1)-bit datapath, whatever its knot count, and the ramp
    # re-derives when a design overrides bits or shape.
    default_ramp: bool = False
    # Each LEAF of a square 2-D shaped Param resets to the identity MATRIX:
    # unity (1 << frac) on the diagonal, zero elsewhere. The matrix
    # counterpart of default_unity -- an unconfigured colour matrix is a
    # pass-through -- and the diagonal re-derives when a design overrides
    # frac, because unity is a semantic invariant, not a number.
    default_identity: bool = False
    # Named values, for registers that SELECT rather than measure:
    # ((value, "label"), ...). This is the one presentation fact a control
    # surface cannot derive -- which raw values are meaningful and what
    # each one is called -- so it is the one that gets declared. Everything
    # else about how a UI presents a register (tick box, fixed-point box,
    # integer slider) derives from bits, frac and signedness; see `widget`.
    choices: tuple = ()
    # Reset values a DESIGN set, one per leaf in row-major order (one for a
    # scalar). They replace the declared or derived default -- unity, ramp,
    # identity -- because the design, the block being built, has the last
    # word over the libraries it uses. Values, not structure, so no
    # `configurable` whitelist gates them: they are the same kind of fact a
    # profile writes at run time, stated at reset instead.
    resets: tuple = ()

    def __post_init__(self) -> None:
        _check_name(self.name, "parameter")
        unknown = set(self.configurable) - {"bits", "frac", "default", "shape"}
        if unknown:
            raise ValueError(
                f"param {self.name!r}: configurable {sorted(unknown)} is not "
                "overridable; a design may adjust bits, frac, default or shape")
        if self.default_ramp:
            if self.default_unity:
                raise ValueError(
                    f"param {self.name!r}: default_ramp and default_unity "
                    "contradict each other")
            if len(self.shape) != 1 or self.shape[0] < 2:
                raise ValueError(
                    f"param {self.name!r}: default_ramp needs a 1-D shape of "
                    "at least two knots")
            span = 1 << (self.bits - 1)
            if span % (self.shape[0] - 1):
                raise ValueError(
                    f"param {self.name!r}: default_ramp needs the knot count "
                    f"minus one ({self.shape[0] - 1}) to divide the half range "
                    f"({span}); use 2**k + 1 knots")
            if self.default != 0:
                raise ValueError(
                    f"param {self.name!r}: default_ramp derives per-leaf "
                    "defaults; do not also declare one")
        if self.default_identity:
            if self.default_unity or self.default_ramp:
                raise ValueError(
                    f"param {self.name!r}: default_identity contradicts "
                    "default_unity/default_ramp -- pick the one that matches "
                    "the parameter's shape")
            if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
                raise ValueError(
                    f"param {self.name!r}: default_identity needs a square "
                    f"2-D shape, got {self.shape}")
            if self.default != 0:
                raise ValueError(
                    f"param {self.name!r}: default_identity derives per-leaf "
                    "defaults; do not also declare one")
            if (1 << self.frac) > self.limits[1]:
                raise ValueError(
                    f"param {self.name!r}: unity (1 << {self.frac}) does not "
                    f"fit {self.bits} {'signed' if self.signed else 'unsigned'} "
                    "bits, so the diagonal cannot reset to pass-through")
        if self.default_unity:
            if self.default not in (0, 1 << self.frac):
                raise ValueError(
                    f"param {self.name!r}: default_unity derives the default "
                    "from frac; do not also declare one")
            object.__setattr__(self, "default", 1 << self.frac)
            if "default" in self.configurable:
                raise ValueError(
                    f"param {self.name!r}: default_unity and a configurable "
                    "default contradict each other -- unity IS the default")
        if not 1 <= self.bits <= REG_WIDTH:
            raise ValueError(
                f"param {self.name!r}: bits={self.bits} outside 1..{REG_WIDTH}; a "
                "parameter must fit one bus word")
        if not 0 <= self.frac <= self.bits:
            raise ValueError(
                f"param {self.name!r}: frac={self.frac} must be within 0..{self.bits}")
        if not self.description:
            raise ValueError(
                f"param {self.name!r}: description is required -- it is carried into "
                "the generated Verilog and the register map documentation")
        if not all(int(d) > 0 for d in self.shape):
            raise ValueError(f"param {self.name!r}: shape {self.shape} must be positive")
        lo, hi = self.limits
        if self.resets:
            if self.default_unity or self.default_ramp or self.default_identity:
                raise ValueError(
                    f"param {self.name!r}: design reset values replace the "
                    "derived default; clear default_unity/ramp/identity")
            count = 1
            for d in self.shape:
                count *= int(d)
            if len(self.resets) != count:
                raise ValueError(
                    f"param {self.name!r}: {len(self.resets)} reset value(s) "
                    f"for {count} register(s)")
            for v in self.resets:
                if not lo <= int(v) <= hi:
                    raise ValueError(
                        f"param {self.name!r}: reset value {v} outside the "
                        f"representable range [{lo}, {hi}]")
            if not self.shape:
                object.__setattr__(self, "default", int(self.resets[0]))
        if not lo <= self.default <= hi:
            raise ValueError(
                f"param {self.name!r}: default {self.default} outside the representable "
                f"range [{lo}, {hi}] for {self.bits} {'signed' if self.signed else 'unsigned'} bits")
        if self.choices:
            if self.shape:
                raise ValueError(
                    f"param {self.name!r}: choices need a scalar register; an "
                    "array of selectors has no single thing being selected")
            if self.frac:
                raise ValueError(
                    f"param {self.name!r}: choices and fractional bits "
                    "contradict each other -- a named value is not a quantity")
            entries = []
            for entry in self.choices:
                if len(entry) != 2 or not str(entry[1]).strip():
                    raise ValueError(
                        f"param {self.name!r}: each choice is (value, label), "
                        "and the label is what a person selects -- it cannot "
                        "be empty")
                entries.append((int(entry[0]), str(entry[1])))
            if len(entries) < 2:
                raise ValueError(
                    f"param {self.name!r}: a choice needs at least two named "
                    "values; with one there is nothing to choose")
            values = [v for v, _ in entries]
            labels = [label for _, label in entries]
            if len(set(values)) != len(values) or len(set(labels)) != len(labels):
                raise ValueError(
                    f"param {self.name!r}: choice values and labels must each "
                    "be unique -- two names for one value (or one name for "
                    "two) is an ambiguity handed straight to a user")
            for v in values:
                if not lo <= v <= hi:
                    raise ValueError(
                        f"param {self.name!r}: choice value {v} outside the "
                        f"representable range [{lo}, {hi}]")
            if self.default not in values:
                raise ValueError(
                    f"param {self.name!r}: default {self.default} is not one "
                    "of the named choices -- reset would select something "
                    "the UI cannot display")
            object.__setattr__(self, "choices", tuple(entries))

    # -- fixed-point presentation ------------------------------------------- #

    @property
    def limits(self) -> tuple[int, int]:
        """Inclusive ``(lo, hi)`` range of the raw register value."""
        if self.signed:
            return -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        return 0, (1 << self.bits) - 1

    @property
    def q_format(self) -> str:
        """Human Q notation, e.g. ``Q8.8`` or ``u12.0``."""
        kind = "Q" if self.signed else "u"
        return f"{kind}{self.bits - self.frac}.{self.frac}"

    @property
    def widget(self) -> str:
        """How a control surface presents this register -- DERIVED, in this
        one place, never stated per parameter: named values are a choice,
        one bit is a tick box, fractional bits mean a real value typed into
        a box in the declared Q format, and anything else is an integer
        slider over the representable range. A block author who wants a
        different widget changes the declaration the widget derives from,
        because a register whose presentation disagrees with its arithmetic
        is a register the UI lies about."""
        if self.choices:
            return "choice"
        if self.bits == 1:
            return "checkbox"
        if self.frac:
            return "fixed"
        return "slider"

    @property
    def count(self) -> int:
        """Number of registers this declaration allocates."""
        return int(np.prod(self.shape)) if self.shape else 1

    def scale(self) -> int:
        """The integer that represents 1.0, i.e. ``1 << frac``."""
        return 1 << self.frac

    def quantise(self, value: float) -> int:
        """Round a human value to the raw register integer, saturating to range.

        Host- and control-side only. Never called from a block's model: the
        models take raw integers, because that is what the hardware has.
        """
        raw = int(np.round(float(value) * self.scale()))
        lo, hi = self.limits
        return max(lo, min(hi, raw))

    def dequantise(self, raw: int) -> float:
        """Inverse of :meth:`quantise`, for documentation and host read-back."""
        return int(raw) / self.scale()

    @property
    def dtype(self) -> np.dtype:
        """Smallest NumPy integer dtype that holds this register.

        Used so the NumPy model promotes exactly the way the hardware does.
        """
        width = next(w for w in (8, 16, 32, 64) if self.bits <= w)
        return np.dtype(f"{'int' if self.signed else 'uint'}{width}")

    # -- shaped params -------------------------------------------------------- #

    def indices(self) -> Iterator[tuple[int, ...]]:
        """Row-major indices of a shaped Param; a single ``()`` if scalar."""
        if not self.shape:
            yield ()
        else:
            yield from np.ndindex(self.shape)

    def leaf_name(self, idx: tuple[int, ...]) -> str:
        return self.name if not idx else self.name + "_" + "_".join(str(int(i)) for i in idx)

    def label_of(self, idx: tuple[int, ...]) -> str:
        """The declared meaning of one element, or its index if none was given."""
        if not idx:
            return ""
        node = self.labels
        for step in idx:
            if not isinstance(node, (tuple, list)) or step >= len(node):
                return f"element {list(idx)}"
            node = node[step]
        return str(node)

    def leaf(self, idx: tuple[int, ...]) -> "Param":
        """The scalar Param for one element of a shaped declaration."""
        if not idx:
            return self
        default = self.default
        if self.resets:
            flat = 0
            for step, size in zip(idx, self.shape):
                flat = flat * int(size) + int(step)
            default = int(self.resets[flat])
        if self.default_ramp:
            default = idx[0] * ((1 << (self.bits - 1)) // (self.shape[0] - 1))
        if self.default_identity:
            default = (1 << self.frac) if idx[0] == idx[1] else 0
        return replace(self, name=self.leaf_name(idx), shape=(), labels=(),
                       default=default, default_ramp=False,
                       default_identity=False, resets=(),
                       description=f"{self.label_of(idx)}: {self.description}")

    def leaves(self) -> list["Param"]:
        """Flatten to the scalar registers actually allocated."""
        return [self.leaf(idx) for idx in self.indices()]

    def values(self, array) -> dict[str, int]:
        """Shaped raw values -> register values keyed by leaf name.

        THE way a host turns an array into register writes. The shape, the
        leaf naming and the representable range all come from this
        declaration -- a helper that spells ``f"{name}_{i}_{j}"`` or loops
        over a literal 3 is restating facts declared here, and restated
        facts are how the two copies start to disagree.

        Args:
            array: integer values in this declaration's raw fixed point,
                shaped exactly as declared (a scalar for an unshaped Param).
                Quantisation is the caller's job -- a register value is an
                integer, and rounding policy belongs to whoever chose the
                curve, not to the naming.
        """
        grid = np.asarray(array)
        if grid.shape != self.shape:
            raise ValueError(
                f"param {self.name!r} is declared {self.shape or 'scalar'}, "
                f"got values shaped {grid.shape}")
        if not np.issubdtype(grid.dtype, np.integer):
            raise ValueError(
                f"param {self.name!r}: raw register values are integers; "
                f"got dtype {grid.dtype} -- quantise on the host first")
        out = {}
        for idx in self.indices():
            leaf = self.leaf(idx)
            value = int(grid[idx]) if idx else int(grid)
            lo, hi = leaf.limits
            if not lo <= value <= hi:
                raise ValueError(
                    f"{leaf.name} value {value} outside [{lo}, {hi}], the "
                    f"declared {leaf.bits}-bit "
                    f"{'signed' if leaf.signed else 'unsigned'} register")
            out[leaf.name] = value
        return out

    # -- np2hw bridge --------------------------------------------------------- #

    def to_np2hw(self):
        """The equivalent ``np2hw.Param``, for tracing the model into hardware.

        This Param carries ``frac`` and ``description``, which a traced Param has no
        use for -- np2hw only needs width, signedness, shape and reset value.
        This is the single point where the two representations meet, so there is
        still exactly one declaration.
        """
        from np2hw import Param as Np2hwParam

        return Np2hwParam(self.name, bits=self.bits, signed=self.signed,
                          shape=self.shape, default=self.default,
                          description=self.description, labels=self.labels)


# --------------------------------------------------------------------------- #
# Context -- pipeline-wide facts, fanned out as wires
# --------------------------------------------------------------------------- #

CONTEXT_ORIGINS = ("register", "either", "stream")


@dataclass(frozen=True)
class Context:
    """A pipeline-wide fact owned by the context block and consumed as a wire.

    Width, height, Bayer phase, bit depth, a frame count. A design binds a
    block's input to an expression over them; the pipeline composer connects
    the wire, from the stream where the design names the fact in
    ``stream.context`` and from the context block otherwise. It is emphatically not copied into the
    consuming block's register map -- one owner per fact, and every block reads
    it.
    """

    name: str
    bits: int
    default: int = 0
    signed: bool = False
    description: str = ""
    #: Where the value may come from, because the kinds are genuinely
    #: different. ``"register"``: always software's CHOICE -- a crop window --
    #: so never taken from a stream. ``"either"``: a register, or a fact a
    #: transport's header carries (width, Bayer phase). ``"stream"``: a fact
    #: that changes every frame, so no register could ever hold it -- software
    #: configures, it does not maintain -- and it exists in a pipeline only
    #: where the stream delivers it.
    origin: str = "either"
    #: The fact of a frame this signal's value follows when a model is run
    #: over one -- ``"width"``, ``"height"`` or ``"bit_depth"`` -- so a run
    #: sets it from the frame rather than from a list of names somewhere.
    #: Empty: the caller's value, or the declared default.
    follows: str = ""

    def __post_init__(self) -> None:
        _check_name(self.name, "context signal")
        if self.follows not in ("", "width", "height", "bit_depth"):
            raise ValueError(
                f"context {self.name!r}: follows {self.follows!r} is not "
                "'width', 'height' or 'bit_depth'")
        if not self.description:
            raise ValueError(f"context {self.name!r}: description is required")
        if self.origin not in CONTEXT_ORIGINS:
            raise ValueError(
                f"context {self.name!r}: origin {self.origin!r} is not one of "
                f"{CONTEXT_ORIGINS}")

    def as_param(self) -> Param:
        """The CSR backing this context signal inside the context block.

        The context block is an ordinary block: its context signals are ordinary Params in
        its own register map. The only thing that makes them context is that the
        pipeline routes them outward as wires instead of leaving them local.
        """
        return Param(name=self.name, bits=self.bits, default=self.default,
                     signed=self.signed, description=self.description)


class Vocabulary:
    """The pipeline context a design binds its blocks' inputs to, and the
    block that owns it.

    A design names it (a composite's ``vocabulary``, or ``assemble``'s);
    the blocks it places know nothing of it. ``signals`` are the context
    declarations, in order; ``block`` the configuration block that holds the
    register-backed ones and is placed in the pipeline; ``commit`` the name
    of the register in it that lands a batch of coefficient writes. A
    mapping from signal name to declaration."""

    def __init__(self, signals, block=None, commit: str = "commit"):
        self.signals = tuple(signals)
        self.block = block
        self.commit = commit
        self._by_name = {c.name: c for c in self.signals}

    def __getitem__(self, name: str) -> Context:
        return self._by_name[name]

    def __contains__(self, name) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def layout(self, names) -> tuple:
        """The context a stream carrying the facts `names` has in its CTX
        member: (name, bits, follows) each -- the frame's size first, the
        fact following its width then the one following its height, and
        the rest in this vocabulary's order: one order for every port,
        whatever order it lists them in, so a producer and a consumer agree
        without saying it twice, and a reader of the size alone finds it in
        the low bits."""
        wanted = set(names)
        for name in wanted:
            self.resolve(name)
        size = {"width": 0, "height": 1}
        chosen = [c for c in self.signals if c.name in wanted]
        chosen.sort(key=lambda c: size.get(c.follows, len(size)))
        return tuple((c.name, c.bits, c.follows) for c in chosen)

    def resolve(self, name: str) -> Context:
        """The declaration of one signal, or an error naming what exists."""
        try:
            return self._by_name[name]
        except KeyError:
            owner = f"the {self.block.name} block" if self.block else \
                "this vocabulary"
            raise KeyError(f"no pipeline context signal {name!r}; {owner} "
                           f"provides {list(self._by_name)}") from None


# --------------------------------------------------------------------------- #
# ParamSet -- one block's declaration, in local terms only
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Register:
    """An allocated scalar register: a Param plus its LOCAL byte offset."""

    param: Param
    offset: int

    @property
    def name(self) -> str:
        return self.param.name


class ParamSet:
    """Every configuration register one block owns, at local offsets only.

    A block declares this once, at module scope. It contains no absolute
    address and no instance identity, so the same declaration serves a unit test
    instantiating the block alone and a stereo pipeline instantiating it twice.

    Local layout::

        +0x00   id_version   (read-only) [31:16] block id, [15:8] major, [7:0] minor
        +0x04   first declared parameter
        ...     one 32-bit word per scalar register, declaration order

    Shaped Params expand row-major into consecutive words.
    """

    def __init__(self, block: str, version: tuple[int, int], params=(),
                 consumes: tuple[str, ...] = (), description: str = ""):
        self.block = _check_name(block, "block")
        if not (len(version) == 2 and all(0 <= v <= 255 for v in version)):
            raise ValueError(
                f"block {block!r}: version must be (major, minor), each 0..255")
        self.version = (int(version[0]), int(version[1]))
        self.description = description
        self.consumes = tuple(consumes)
        self.params = tuple(params)

        seen: set[str] = set()
        self._registers: list[Register] = []
        offset = FIRST_PARAM_OFFSET
        for param in self.params:
            for leaf in param.leaves():
                if leaf.name in seen:
                    raise ValueError(
                        f"block {block!r}: duplicate register name {leaf.name!r}")
                seen.add(leaf.name)
                self._registers.append(Register(leaf, offset))
                offset += 4
        self._by_name = {r.name: r for r in self._registers}

    # -- identity ------------------------------------------------------------- #

    @property
    def block_id(self) -> int:
        return fnv1a16(self.block)

    @property
    def id_version_word(self) -> int:
        """The read-only word at local offset 0."""
        return (self.block_id << 16) | (self.version[0] << 8) | self.version[1]

    # -- local layout --------------------------------------------------------- #

    @property
    def registers(self) -> list[Register]:
        return list(self._registers)

    def offset_of(self, name: str) -> int:
        """LOCAL byte offset of a scalar register. Never an absolute address."""
        try:
            return self._by_name[name].offset
        except KeyError:
            raise KeyError(
                f"block {self.block!r} has no register {name!r}; "
                f"it declares {sorted(self._by_name)}") from None

    def param(self, name: str) -> Param:
        return self._by_name[name].param

    def declaration(self, name: str) -> Param:
        """The DECLARED (possibly shaped) Param, as distinct from its leaves.

        ``param()`` answers for allocated scalar registers; this answers for
        the declaration a host reasons about -- the whole matrix, the whole
        table -- whose :meth:`Param.values` turns an array into writes.
        """
        for declared in self.params:
            if declared.name == name:
                return declared
        raise KeyError(
            f"block {self.block!r} declares no parameter {name!r}; "
            f"it declares {[p.name for p in self.params]}")

    @property
    def size_bytes(self) -> int:
        """Bytes of config space this block occupies, aligned up to BLOCK_ALIGN."""
        used = FIRST_PARAM_OFFSET + 4 * len(self._registers)
        return max(BLOCK_ALIGN, ((used + BLOCK_ALIGN - 1) // BLOCK_ALIGN) * BLOCK_ALIGN)

    def __iter__(self) -> Iterator[Param]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self._registers)

    def __repr__(self) -> str:
        return (f"ParamSet({self.block!r}, v{self.version[0]}.{self.version[1]}, "
                f"{len(self._registers)} registers)")

    # -- np2hw bridge --------------------------------------------------------- #

    def to_np2hw(self):
        """An ``np2hw.Params`` namespace over the same declarations."""
        from np2hw import Params as Np2hwParams

        return Np2hwParams([p.to_np2hw() for p in self.params])

    def defaults(self) -> dict[str, int]:
        """Reset values keyed by scalar register name."""
        return {r.name: r.param.default for r in self._registers}

    def bind(self, values: dict[str, int] | None = None):
        """A by-name view of parameter VALUES for running the NumPy model.

        Missing entries fall back to the declared default, so a test can override
        one register and leave the rest at reset. Values are handed back in the
        register's own dtype, and a shaped Param is reassembled into an ndarray,
        so the model promotes exactly as the hardware does.
        """
        merged = self.defaults()
        for key, value in (values or {}).items():
            if key not in merged:
                raise KeyError(
                    f"block {self.block!r} has no register {key!r}; "
                    f"it declares {sorted(merged)}")
            merged[key] = int(value)
        return _BoundParams(self, merged)


class Declarations:
    """Attribute access over a block's CONFIGURED Param declarations.

    What a model reads when its arithmetic depends on a declaration -- the
    shift of a fixed-point multiply is ``1 << p.decl.gain.frac``. This is the
    same object the register map and the host derive from, so the model's
    constant and the map's q_format cannot disagree: there is one Param, and
    everyone asks it.
    """

    __slots__ = ("_params",)

    def __init__(self, params):
        object.__setattr__(self, "_params", {p.name: p for p in params})

    def __getattr__(self, name: str) -> "Param":
        try:
            return object.__getattribute__(self, "_params")[name]
        except KeyError:
            raise AttributeError(
                f"no declared parameter {name!r}; this block declares "
                f"{sorted(object.__getattribute__(self, '_params'))}") from None


class _BoundParams:
    """Attribute access over one block's register values.

    Scalar Param -> a typed NumPy scalar. Shaped Param -> an ndarray assembled
    from its leaves. This is what a block model receives as its ``p`` argument
    when it runs as NumPy; when it is traced into hardware it receives np2hw's
    Param-valued view instead, and the same function body serves both.
    """

    def __init__(self, paramset: ParamSet, values: dict[str, int]):
        object.__setattr__(self, "_paramset", paramset)
        object.__setattr__(self, "_values", values)

    @property
    def decl(self) -> Declarations:
        """The declarations behind these values, for arithmetic that depends
        on them (a Q format's shift). One source: the same Params that emit
        the register map."""
        return Declarations(object.__getattribute__(self, "_paramset").params)

    def __getattr__(self, name: str):
        paramset: ParamSet = object.__getattribute__(self, "_paramset")
        values: dict[str, int] = object.__getattribute__(self, "_values")
        for declared in paramset.params:
            if declared.name != name:
                continue
            if not declared.shape:
                return declared.dtype.type(values[name])
            out = np.zeros(declared.shape, dtype=declared.dtype)
            for idx in declared.indices():
                out[idx] = values[declared.leaf_name(idx)]
            return out
        raise AttributeError(
            f"block {paramset.block!r} has no parameter {name!r}; "
            f"it declares {[p.name for p in paramset.params]}")

    def __repr__(self) -> str:
        paramset: ParamSet = object.__getattribute__(self, "_paramset")
        values = object.__getattribute__(self, "_values")
        return f"<bound params for {paramset.block!r}: {values}>"


# --------------------------------------------------------------------------- #
# Global allocation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BlockInstance:
    """One INSTANCE of a block within a pipeline, with its assigned base.

    The base is assigned here, at composition time, never by the block. A stereo
    pipeline holds two instances of the same ParamSet at two different bases;
    ``path`` ("left.blacklevel") is what distinguishes them and is what the host
    API mirrors.
    """

    path: str
    paramset: ParamSet
    base: int

    @property
    def instance(self) -> str:
        """Leaf name of the instance path, e.g. ``blacklevel``."""
        return self.path.rsplit(".", 1)[-1]

    def address_of(self, register: str) -> int:
        """Absolute byte address of one of this instance's registers."""
        return self.base + self.paramset.offset_of(register)

    @property
    def id_version_address(self) -> int:
        return self.base + ID_VERSION_OFFSET


class AddressAllocator:
    """Assigns each block INSTANCE a base address at composition time.

    Config blocks are packed from ``config_base`` upward, each aligned to
    :data:`BLOCK_ALIGN`, so decoding a block is ``addr[N-1:8] == constant``.

    Address space is free. Alignment that makes the decoder a bit-slice compare
    is not, and neither is a register map a reviewer can read.
    """

    def __init__(self, config_base: int = 0x0000):
        if config_base % BLOCK_ALIGN:
            raise ValueError(f"config_base 0x{config_base:x} must be {BLOCK_ALIGN}-aligned")
        self.config_base = config_base
        self._next_config = config_base
        self._instances: list[BlockInstance] = []
        self._paths: set[str] = set()

    def allocate(self, path: str, paramset: ParamSet) -> BlockInstance:
        """Give one instance of ``paramset`` a base address."""
        if path in self._paths:
            raise ValueError(f"duplicate block instance path {path!r}")
        self._paths.add(path)

        base = self._next_config
        self._next_config = base + paramset.size_bytes

        inst = BlockInstance(path=path, paramset=paramset, base=base)
        self._instances.append(inst)
        return inst

    @property
    def instances(self) -> list[BlockInstance]:
        return list(self._instances)

    @property
    def config_span(self) -> int:
        return self._next_config - self.config_base

    def address_bits(self) -> int:
        """Bits needed to address everything allocated so far."""
        return max(1, (self._next_config - 1).bit_length())
