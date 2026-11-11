# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Blocks: a NumPy model plus what its arithmetic cannot say.

A block is its model -- the decorated function, written at the hardware's
arithmetic: the specification and the golden reference -- and the
declarations nothing in that arithmetic states: its registers and scalar
inputs (:class:`np2hw.blocks.params.ParamSet`, LOCAL offsets only; a
composer assigns each INSTANCE its base) and the names of its stream ports.
What feeds a scalar input -- its register, or the pipeline context a design
binds it to (:meth:`Block.bind`) -- is the design's. Stream widths are not
declared: np2hw traces
them from the model, and a composer threads each block's traced output into
the next block's input.

    ``Block.run()``      runs the model over a frame, validated from the
                         declarations -- no per-block adapter.
    ``Block.generate()`` traces the model into Verilog.

Verilog traced from a model carries the model's licence: its copyright
and SPDX identifier are read from the model's own source file, which states
them. The content that writes blocks binds one convention, in its own
A block knows nothing of the pipeline it is placed in: what feeds its
inputs, and what pipeline context there is to feed them from, are the
design's. np2hw names no context signal and no licence.
"""
from __future__ import annotations

import importlib
import pkgutil
import textwrap
from dataclasses import dataclass, field
from types import ModuleType


@dataclass(frozen=True)
class StreamPort:
    """One stream port of a block: a name for the netlist to wire.

    The WIDTH is deliberately absent -- and so is everything else about the
    word. Channel count and field width are consequences of the model's
    arithmetic: np2hw derives them from the trace, and the composer threads
    each block's traced output into the next block's input. Declaring any
    of it here would be a second source of truth free to disagree with the
    first. What flows through a wire is whatever the math upstream computed;
    whether a connection makes SENSE is the author's judgment, not a check.

    ``history`` is part of the math's signature: the model is handed the
    stream's recent frames, ``(T, H, W)`` newest last, and reads past ones
    by slicing the first axis -- which a channel axis, ``(H, W, C)``, is
    indistinguishable from by shape. How many frames are kept is not
    declared: it is how far back the model reads, and the core's store is
    derived from that.
    """

    name: str
    description: str = ""
    history: bool = False


@dataclass(frozen=True)
class Ports:
    """The stream ports a block presents, by name.

    Declared rather than assumed, because "one in, one out" is only true of the
    simplest blocks. ``stats`` consumes a stream and produces none -- it observes
    the datapath rather than sitting in it, which a linear chain cannot express
    and which is exactly the distinction that decides whether a connection needs
    to carry backpressure.

    A block with no outputs is a SINK. A sink must never stall its source: it is
    watching a stream that has somewhere else to be, so it accepts a pixel every
    cycle and its ``ready`` is not wired back.
    """

    inputs: tuple[str, ...] = ("in",)
    outputs: tuple[str, ...] = ("out",)

    @property
    def is_sink(self) -> bool:
        return not self.outputs

    def has(self, port: str) -> bool:
        return port in self.inputs or port in self.outputs

    def direction(self, port: str) -> str:
        if port in self.inputs:
            return "in"
        if port in self.outputs:
            return "out"
        raise KeyError(
            f"no port {port!r}; this block has inputs {list(self.inputs)} and "
            f"outputs {list(self.outputs)}")


# The default for an ordinary inline block: consume a stream, produce a stream.
INLINE = Ports()

# A block that observes the datapath and produces no pixels.
MONITOR = Ports(inputs=("in",), outputs=())

# A block with no stream ports at all -- configuration only, like a context block.
NO_STREAM = Ports(inputs=(), outputs=())


@dataclass(frozen=True)
class Generated:
    """The Verilog one block generated, plus what a composer needs to wire it.

    Args:
        top: name of the module a pipeline should instantiate.
        modules: ``(module_name, verilog_text)`` in dependency order, so writing
            them out in sequence gives a file that elaborates.
        params: ``(scalar_register_name, bits)`` the top module expects on its
            ``param_*`` inputs, in register-map order.
        consumes: pipeline context signals the top module reads: fields of
            its stream's context, ``in_ctx``, never registers of its own --
            a Bayer phase a design binds is context, not configuration, one
            owner, fanned out with the frame.
        latency: pixels of delay from input to output, for pipeline alignment.
        meta: generator detail worth keeping (line buffers, accumulator width).
    """

    top: str
    modules: tuple[tuple[str, str], ...]
    params: tuple[tuple[str, int], ...] = ()
    consumes: tuple[str, ...] = ()
    # The np2hw result this block was generated from. It carries the module's
    # own interface -- which framing flags it accepts, which parameter ports it
    # has -- so the composer reads that from the generator rather than
    # keeping a second description of it that can drift.
    core: dict | None = None
    latency: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def verilog(self) -> str:
        """All modules concatenated, in elaboration order."""
        return "\n\n".join(text.rstrip() for _, text in self.modules) + "\n"

    def module_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.modules)


class _View:
    """Attribute access over a set of values, for the model's `p`."""

    __slots__ = ("_values", "_decl")

    def __init__(self, values):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name):
        if name == "decl":
            try:
                return object.__getattribute__(self, "_decl")
            except AttributeError:
                pass
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError:
            raise AttributeError(
                f"{name!r} is not available here; this view has "
                f"{sorted(object.__getattribute__(self, '_values'))}") from None


#: The frames a history input is traced at: the deepest read a block may
#: make is one fewer. The hardware is the same at any length that holds the
#: reads -- the store is the reads' spread -- so this bounds, and declares
#: nothing.
HISTORY_TRACE = 64


@dataclass(frozen=True)
class Binding:
    """One scalar input of a block, bound by a design to an expression.

    A block's model is only math: what feeds a scalar input -- the frame's
    place in a capture loop, a Bayer phase -- is the design's statement,
    not the block's. Bound, the input has no register: its value is the
    expression, over the pipeline context the DESIGN's vocabulary declares
    and integer constants, evaluated by the design's one expression walker --
    over integers when the model runs as NumPy, over traced values when it
    is traced, so the expression IS hardware, folded into the block's core.
    The core's ports are then the context facts the expression reads.
    """

    param: object             # the input's declaration (a blocks Param)
    text: str                 # the expression, as the design wrote it
    tree: object = field(compare=False, hash=False, repr=False)
    facts: tuple = ()         # the context signals it reads, in order
    signals: tuple = ()       # their declarations (Context), in that order

    @property
    def name(self) -> str:
        return self.param.name

    def value(self, lookup):
        """The expression's value, each fact read through ``lookup``."""
        from np2hw.design.composite import evaluate
        return evaluate(self.tree, lookup)

    def check(self, lo: int, hi: int, block: str) -> None:
        """Refuse a value range the input cannot hold: an expression that
        could leave its input's width says so here, rather than wrapping
        where nothing wrote a wrap."""
        p = self.param
        top = 1 << (p.bits - 1 if p.signed else p.bits)
        low = -top if p.signed else 0
        if lo < low or hi > top - 1:
            raise ValueError(
                f"block {block!r}: {p.name} = {self.text!r} ranges over "
                f"[{lo}, {hi}], which a {p.bits}-bit "
                f"{'signed' if p.signed else 'unsigned'} input cannot hold "
                f"([{low}, {top - 1}]); state the wrap in the expression")


@dataclass
class Block:
    """A block: one model function, plus what cannot be derived from it.

    Produced by :func:`block`. The model is the specification; everything
    here is a declaration the arithmetic cannot supply -- the register set, what
    the streams MEAN. Notably absent are the
    stream WIDTHS, which np2hw derives from the trace; declaring them here too
    would be a second source of truth free to disagree with the first.
    """

    model: object
    params: "ParamSet"
    inputs: tuple
    outputs: tuple
    # Register-attribute overrides this VARIANT was configured with, empty on
    # the base block. {"gain": {"frac": 12}} -- the same dict the design JSON
    # carries and describe() recovers, so a variant knows its own recipe.
    overrides: dict = field(default_factory=dict)
    # Reset values a design set, {register: (leaf values, row-major)}; empty
    # on the base block. Part of the variant, because the register file's
    # reset is part of the block's register TYPE.
    resets: dict = field(default_factory=dict)
    # The model has a consistent meaning only on a frame whose dimensions
    # are both even -- it indexes a 2x2 colour tile -- so running it over an
    # odd frame is refused. A fact about the block's own math.
    even_frame: bool = False
    # Why this block cannot be traced yet, if it cannot. Empty means it can.
    # Stated as a REASON rather than a flag, because "declared but not built" is
    # a fact about np2hw's current reach and the reader deserves to know which.
    not_traceable: str = ""
    # The scalar inputs a design bound to expressions (Binding), empty on
    # the base block. A bound input has left the register set: its value
    # is its expression, and what the expression reads is consumed.
    bindings: tuple = ()

    @property
    def name(self) -> str:
        return self.params.block

    @property
    def ports(self) -> Ports:
        return Ports(inputs=tuple(port.name for port in self.inputs),
                     outputs=tuple(port.name for port in self.outputs))

    def __call__(self, *args, **kwargs):
        """Run the model. A Block is the function it decorates."""
        if self.model is None:
            raise TypeError(
                f"block {self.name!r} is configuration only and has no model")
        return self.model(*args, **kwargs)

    @property
    def traceable(self) -> bool:
        return self.model is not None and not self.not_traceable

    @property
    def implemented(self) -> bool:
        """Whether this block's declaration reaches hardware.

        A block with a datapath reaches hardware only if it can be traced. A
        block with no stream ports has no datapath to trace -- a context block owns
        configuration and its registers reach the blocks as context wires -- so
        there is nothing left for it to be missing.

        The distinction is not cosmetic. Every declared register is decoded by
        the register file, because the register map and the decode must be the
        same thing; but a register belonging to a block that was never built
        reads back what was written and drives nothing. Software is told which
        is which rather than left to discover it from an image that did not
        change.
        """
        return self.traceable or not (self.inputs or self.outputs)

    def configure(self, registers: dict | None = None,
                  resets: dict | None = None) -> "Block":
        """The variant of this block with register attributes overridden.

        ``resets`` is ``{register_name: value}``, a value shaped as the
        register is (nested lists for an array): the reset values a design
        sets, replacing the declared or derived default. No whitelist gates
        these -- they are values, not structure.

        ``registers`` is ``{register_name: {attribute: value}}``, and only
        attributes the declaration lists as ``configurable`` are accepted --
        the whitelist is the block author's statement of what varies between
        variants, and everything else is structure.

        Variants are CACHED per canonical override set, so two instances
        configured identically share one Block object. That object identity
        is what downstream type grouping keys on: the register map and the
        SystemRDL emit one type per variant, never one per instance and
        never one type lying about two different Q formats.
        """
        import dataclasses

        if not registers and not resets:
            return self
        registers = registers or {}
        base = getattr(self, "_base", self)
        def hashable(value):
            return tuple(value) if isinstance(value, (list, tuple)) else value

        def flatten(value):
            if isinstance(value, (list, tuple)):
                return tuple(x for item in value for x in flatten(item))
            return (int(value),)

        flat_resets = {reg: flatten(v) for reg, v in (resets or {}).items()}
        key = (tuple(sorted(
            (reg, tuple(sorted((attr, hashable(v))
                               for attr, v in attrs.items())))
            for reg, attrs in registers.items())),
            tuple(sorted(flat_resets.items())))
        cache = base.__dict__.setdefault("_variants", {})
        if key in cache:
            return cache[key]

        from np2hw.blocks.params import ParamSet

        declared = {param.name: param for param in base.params.params}
        rebuilt = []
        for param in base.params.params:
            attrs = registers.get(param.name)
            if not attrs:
                rebuilt.append(_with_resets(param, flat_resets.get(param.name)))
                continue
            refused = set(attrs) - set(param.configurable)
            if refused:
                raise ValueError(
                    f"block {base.name!r}: register {param.name!r} does not "
                    f"allow overriding {sorted(refused)}; it allows "
                    f"{sorted(param.configurable) or 'nothing'}")
            changes = {k: (tuple(int(x) for x in v) if k == "shape"
                           else int(v))
                       for k, v in attrs.items()}
            if param.default_unity or (param.default_ramp
                                       and not attrs.keys() - {"bits", "shape"}):
                # replace() would carry an OLD resolved default forward;
                # zero it so __post_init__ / leaf() re-derive for the new
                # frac, bits or shape.
                changes.setdefault("default", 0)
            rebuilt.append(_with_resets(dataclasses.replace(param, **changes),
                                        flat_resets.get(param.name)))
        unknown = (set(registers) | set(flat_resets)) - set(declared)
        if unknown:
            raise KeyError(
                f"block {base.name!r} declares no register named "
                f"{sorted(unknown)}; it declares {sorted(declared)}")

        variant = dataclasses.replace(
            base,
            params=ParamSet(block=base.params.block,
                            version=base.params.version,
                            params=rebuilt, stats=list(base.params.stats),
                            consumes=base.params.consumes,
                            description=base.params.description),
            overrides={reg: dict(attrs) for reg, attrs in registers.items()},
            resets=flat_resets,
        )
        variant.__dict__["_base"] = base
        cache[key] = variant
        return variant

    def bind(self, bindings: dict | None = None,
             vocabulary=None) -> "Block":
        """The variant of this block with scalar inputs bound to expressions.

        ``bindings`` is ``{input: expression}``: an integer, or the text of
        an expression in the design's language over the context signals
        ``vocabulary`` declares -- the DESIGN's, which places the block
        (``"frame_seq & 3"``). Each bound
        input is a scalar register of the block, and leaves the register
        set -- a register a design never lets software write would be
        decoded, mapped and documented while driving nothing. The signals
        the expressions read are added to what the block consumes, so the
        composer wires them exactly as it wires any consumed context.

        Variants are cached per binding set, as :meth:`configure`'s are.
        """
        import dataclasses

        from np2hw.blocks.params import ParamSet
        from np2hw.design.composite import CompositionError, parse, references

        if not bindings:
            return self
        cache = self.__dict__.setdefault("_bound", {})
        key = (tuple(sorted((k, str(v)) for k, v in bindings.items())),
               id(vocabulary))
        if key in cache:
            return cache[key]
        declared = {param.name: param for param in self.params.params}
        made = []
        for name, raw in bindings.items():
            param = declared.get(name)
            if param is None:
                raise KeyError(
                    f"block {self.name!r} has no input {name!r} to bind; its "
                    f"scalar inputs are "
                    f"{sorted(p.name for p in self.params.params if not p.shape)}")
            if param.shape:
                raise ValueError(
                    f"block {self.name!r}: {name} is an array of "
                    f"{param.shape}; only a scalar input is bound to an "
                    "expression")
            text = str(raw)
            where = f"block {self.name!r}: bind {name}"
            try:
                tree = parse(text, where)
            except CompositionError as error:
                raise ValueError(str(error)) from None
            facts = tuple(dict.fromkeys(references(tree)))
            unknown = [f for f in facts if not vocabulary or f not in vocabulary]
            if unknown:
                raise ValueError(
                    f"{where} = {text!r} reads {unknown}, which the pipeline "
                    f"context does not declare ({sorted(vocabulary or ())})")
            made.append(Binding(param=param, text=text, tree=tree,
                                facts=facts,
                                signals=tuple(vocabulary[f] for f in facts)))
        bound = {b.name for b in made}
        kept = [p for p in self.params.params if p.name not in bound]
        facts = tuple(dict.fromkeys(f for b in made for f in b.facts))
        clash = sorted({p.name for p in kept} & set(facts))
        if clash:
            raise ValueError(
                f"block {self.name!r}: {clash} name both a context signal a "
                "binding reads and an input of the block; one name, one port")
        variant = dataclasses.replace(
            self,
            params=ParamSet(block=self.params.block,
                            version=self.params.version, params=kept,
                            stats=list(self.params.stats),
                            consumes=tuple(dict.fromkeys(
                                (*self.params.consumes, *facts))),
                            description=self.params.description),
            bindings=(*self.bindings, *made),
        )
        variant.__dict__["_base"] = self.__dict__.get("_base", self)
        cache[key] = variant
        return variant

    @property
    def variant_signature(self) -> str:
        """A stable, Verilog-safe suffix naming this variant's overrides.

        Empty for the base block. Used wherever a TYPE needs a distinct name
        -- the SystemRDL regfile, the shared-module cache -- because two
        variants merging into one type is the map lying about one of them.
        """
        if not self.overrides and not self.resets and not self.bindings:
            return ""
        parts = []
        for register in sorted(self.overrides):
            for attr, value in sorted(self.overrides[register].items()):
                flat = ("x".join(str(int(x)) for x in value)
                        if isinstance(value, (tuple, list)) else value)
                parts.append(f"{register}_{attr}{flat}")
        if self.resets:
            # A digest, not the values: 33 knots do not fit in a name, and
            # the name only has to tell two variants apart.
            import hashlib
            digest = hashlib.blake2b(repr(sorted(self.resets.items()))
                                     .encode(), digest_size=4).hexdigest()
            parts.append(f"reset{digest}")
        if self.bindings:
            import hashlib
            digest = hashlib.blake2b(repr([(b.name, b.text) for b in
                                           self.bindings]).encode(),
                                     digest_size=4).hexdigest()
            parts.append(f"bind{digest}")
        return "__" + "_".join(parts)

    # -- generation ----------------------------------------------------------- #

    def run(self, pixel, values: dict | None = None, bit_depth: int = 12,
            *, kernel=None, **context):
        """Run the model over a whole frame, validated from the DECLARATIONS.

        The declarations drive the validation -- a block whose math needs an
        even frame (``even_frame``) gets the even-dimension check, and every
        context signal a bound input reads its range check -- and a new block
        gets the whole adapter by existing.

        The model is handed the frame as the smallest unsigned NumPy integer
        that holds ``bit_depth`` bits -- the type the trace gives the same
        input, so the model's arithmetic promotes, and wraps, exactly as the
        generated hardware does. A frame with a value outside ``bit_depth``
        bits is refused rather than cut down to fit: it means the caller and
        the stream disagree about the depth.

        Args:
            pixel: the frame, ``(h, w)`` or ``(h, w, c)`` as the model's
                arithmetic expects, unsigned, ``bit_depth`` bits -- or, for
                a history input, the run of frames, ``(t, h, w[, c])``.
            values: register values for ``params.bind`` (defaults fill in).
            bit_depth: datapath width.
            kernel: the model compiled for a CPU (:meth:`compile_c`), run
                in its place with the same registers and context -- the
                same checks, the same frames, the C backend's result.
            **context: pipeline context by name, e.g. ``frame_seq=5``: what
                the block's bound inputs read. Everything it consumes
                defaults to 0; a name it does not consume is refused, because
                silently accepting it would be accepting a typo. For a
                history input a value may be one per frame, as each frame's
                header carries it: each output frame is then the model's
                with the context of the frame it ends at.
        """
        import numpy as np

        from np2hw.ir import np_dtype

        if self.model is None:
            raise TypeError(
                f"block {self.name!r} is configuration only and has no model")
        # No shape gate: a model speaks CHANNELS ((h, w) or (h, w, c)), its
        # arithmetic states its own requirements, and the caller is trusted
        # to know what they are feeding it -- the same contract NumPy itself
        # offers. Words exist on the wire only; StreamSpec owns that
        # translation.
        pixel = np.asarray(pixel)
        if pixel.size and (pixel.min() < 0 or int(pixel.max()) >> bit_depth):
            raise ValueError(
                f"block {self.name!r} was given values "
                f"{int(pixel.min())}..{int(pixel.max())}, outside the "
                f"{bit_depth}-bit stream it was told it is reading")
        pixel = pixel.astype(np_dtype(bit_depth, False))
        consumed = set(self.params.consumes)
        unknown = set(context) - consumed
        if unknown:
            raise TypeError(
                f"block {self.name!r} does not consume {sorted(unknown)}; "
                f"it consumes {sorted(consumed) or 'no context'}")
        history = bool(self.inputs) and self.inputs[0].history
        frames = len(pixel) if history else 1
        ctx_values = {}
        for name in consumed:
            declared = self.signal(name)
            raw = context.get(name, 0)
            each = np.asarray(raw).ravel().tolist() if np.ndim(raw) else None
            if each is not None and (not history or len(each) != frames):
                raise ValueError(
                    f"context {name!r} has {len(each)} values; one per frame "
                    f"is {frames} frames of a history input"
                    if history else
                    f"context {name!r} is one value a frame, and "
                    f"{self.name!r} takes one frame")
            for value in (each if each is not None else [raw]):
                if not 0 <= int(value) <= (1 << declared.bits) - 1:
                    raise ValueError(
                        f"context {name!r} = {value} outside its declared "
                        f"{declared.bits}-bit range")
            ctx_values[name] = [int(v) for v in each] if each is not None \
                else int(raw)
        if self.even_frame:
            height, width = pixel.shape[1:3] if history else pixel.shape[:2]
            if height % 2 or width % 2:
                raise ValueError(
                    f"frame {width}x{height} has an odd dimension; a "
                    "CFA-indexed block has no consistent Bayer phase unless "
                    "both dimensions are even")
        if kernel is None:
            def compute(ctx):
                return self.model(pixel, self.bound_values(values, ctx),
                                  bit_depth)
        else:
            # The kernel takes the registers as the trace took them -- each
            # scalar register, and a register array as its entries where it
            # is read at a constant place and whole where at a computed one
            # -- and the context facts, and works out the bound inputs.
            bound = self.params.bind(values)
            registers = {**self.params.defaults(), **(values or {}),
                         **{p.name: np.ravel(getattr(bound, p.name))
                            for p in self.params.params if p.shape}}

            def compute(ctx):
                self.bound_values(values, ctx)    # the model run's refusals
                return kernel(pixel, **registers, **ctx)
        if not any(isinstance(v, list) for v in ctx_values.values()):
            return compute(ctx_values)
        # Context a frame: each output frame is the model's at the context
        # of the frame it ends at -- the live one, the last it reads.
        def at(n):
            return {k: v[n] if isinstance(v, list) else v
                    for k, v in ctx_values.items()}
        first = compute(at(frames - 1))
        lag = frames - len(first)
        out = np.empty_like(first)
        for k in range(len(first)):
            out[k] = compute(at(k + lag))[k]
        return out

    def bound_values(self, values: dict | None, context: dict):
        """The model's `p` for a NumPy run: the registers' values, and each
        bound input's expression over the context's."""
        from np2hw.blocks.params import ParamSet

        if not self.bindings:
            return self.params.bind(values or {})
        values = dict(values or {})
        for binding in self.bindings:
            if binding.name in values:
                raise KeyError(
                    f"block {self.name!r}: {binding.name} is bound to "
                    f"{binding.text!r} and has no register to set")
            value = int(binding.value(lambda fact: int(context[fact])))
            binding.check(value, value, self.name)
            values[binding.name] = value
        every = ParamSet(block=self.params.block, version=self.params.version,
                         params=[*self.params.params,
                                 *(b.param for b in self.bindings)],
                         description=self.params.description)
        return every.bind(values)

    def signal(self, name: str):
        """The declaration of one context signal a bound input reads, as the
        design that bound it declared it."""
        for binding in self.bindings:
            for signal in binding.signals:
                if signal.name == name:
                    return signal
        raise KeyError(f"block {self.name!r} reads no context signal "
                       f"{name!r}; its bindings read "
                       f"{sorted(self.params.consumes) or 'nothing'}")

    def default_context(self) -> tuple:
        """The context a core of this block carries where no composer says
        what its stream does: its frame's width and height, then the facts
        its bindings read -- (name, bits, follows) each, in that order."""
        facts = list(dict.fromkeys(f for b in self.bindings for f in b.facts))
        return (("width", 16, "width"), ("height", 16, "height"),
                *((f, self.signal(f).bits, self.signal(f).follows)
                  for f in facts if f not in ("width", "height")))

    def _traceable(self, spec, width: int, height: int,
                   context: tuple | None = None):
        """What a backend traces: the model as a function of the frame's
        data word, its registers and the context facts its bindings read,
        the image it is traced on, and those registers and facts."""
        from np2hw import Image2D

        if not self.traceable:
            raise NotImplementedError(
                f"block {self.name!r} is declared but not built: "
                f"{self.not_traceable}")
        if len(self.inputs) != 1 or len(self.outputs) > 1:
            raise NotImplementedError(
                f"block {self.name!r} has {len(self.inputs)} inputs and "
                f"{len(self.outputs)} outputs; tracing handles one stream in and "
                "at most one out. A block that merges or splits streams needs "
                "np2hw to trace multiple images first.")

        # The traced image is the packed DATA WORD, not one component: a
        # 3-channel block sees channels side by side in one value and
        # unpacks them itself, exactly as the wire carries them.
        history = self.inputs[0].history
        if spec.frames and not history:
            raise ValueError(
                f"block {self.name!r} reads no history, and is fed "
                f"{spec.frames} frames side by side")
        from np2hw import Param as Np2hwParam

        registers = [param.to_np2hw() for param in self.params]
        # What the bindings read: fields of the stream's context, each a
        # context signal as its vocabulary declares it.
        facts = list(dict.fromkeys(f for b in self.bindings for f in b.facts))
        if context is None:
            context = self.default_context()
        image = Image2D("pixel", width, height,
                        bits=spec.bit_depth * spec.channels, signed=False,
                        frames=HISTORY_TRACE if history else 0,
                        delivered=spec.frames,
                        context=tuple(tuple(f) for f in context))
        # Each frame brings its own: the core takes it with the frame and
        # carries it with the frame's beats (np2hw.ir.Param's frame).
        read = [Np2hwParam(f, bits=self.signal(f).bits,
                           signed=self.signal(f).signed,
                           description=self.signal(f).description,
                           frame=True)
                for f in facts]

        def traced(pixel, *traced_args):
            from np2hw.blocks.params import Declarations

            given = dict(zip((param.name for param in self.params),
                             traced_args[:len(registers)]))
            by_fact = dict(zip(facts, traced_args[len(registers):]))
            for binding in self.bindings:
                # A signal bound as it is stays the register it is, which a
                # slice may start at; in an expression each signal is the
                # traced value it reads as, so the arithmetic is the trace's
                # -- exact, and refused where the trace cannot do it.
                lookup = (by_fact.__getitem__ if binding.facts == (binding.text,)
                          else lambda f: by_fact[f]._traced(weak=True))
                given[binding.name] = value = binding.value(lookup)
                binding.check(*_value_range(value), self.name)
            values = _View(given)
            # The same declaration access the NumPy run has: a model whose
            # arithmetic depends on a declared attribute (a Q format's shift)
            # reads it from the ONE configured Param, in both modes.
            object.__setattr__(values, "_decl", Declarations(
                [*self.params.params, *(b.param for b in self.bindings)]))
            return self.model(pixel, values, spec.bit_depth)

        return traced, image, registers, read

    def compile_c(self, spec, width: int, height: int, build,
                  context: tuple | None = None):
        """The model printed as C and built for a CPU: the same trace
        :meth:`generate` turns into Verilog, through np2hw's C backend
        (:func:`np2hw.core.cemit.compile_c`). Hand it to :meth:`run` as
        ``kernel`` to run it in the model's place."""
        from np2hw.core.cemit import compile_c

        traced, image, registers, read = self._traceable(spec, width, height,
                                                         context)
        traced.__name__ = self.name
        return compile_c(traced, image, *registers, *read,
                         channels=spec.channels, build=build)

    def generate(self, spec, width: int, height: int, module_name: str,
                 clk_ns: float | None = None,
                 sync_memory: bool = False,
                 context: tuple | None = None) -> "Generated":
        """Trace the model into Verilog. Generic: every block uses this one.

        There is no per-block generator and no hand-written Verilog anywhere in
        this package. A block is its model plus its declarations, and this turns
        the pair into a module.

        The core holds frames up to ``width`` x ``height`` and takes each
        frame's own size, and every value a frame brings that its bindings
        read, in the context its stream carries: ``context``, the stream's
        layout (np2hw.blocks.params.Vocabulary.layout), as a composer hands
        it -- or, for a block built on its own, its frame's width and height
        and the facts its bindings read. It hands the context on with the
        frame, the size its own.
        """
        from np2hw import to_ir, generate as np2hw_generate

        traced, image, registers, read = self._traceable(spec, width, height,
                                                         context)
        # The channel count is the SPEC's -- threaded forward by the composer
        # from whatever the driving block's trace published, starting from the
        # pipeline input. Nothing here re-declares it; if the arithmetic
        # cannot digest the stream it is given (a 3-channel unpack of a
        # 1-channel word), the trace itself is where that surfaces.
        _, result = to_ir(traced, image, *registers, *read,
                          channels=spec.channels)
        core = np2hw_generate(result, module_name=module_name,
                              clk_ns=clk_ns, label=self.name,
                              sync_memory=sync_memory)

        header = model_header(
            self.model, f"{self.name} -- {self.params.description} "
                        f"({spec.bit_depth}-bit datapath)")
        verilog = "\n".join([*header, "", core.verilog])
        return Generated(
            top=module_name,
            modules=((module_name, verilog),),
            params=tuple((r.name, r.param.bits) for r in self.params.registers),
            consumes=self.params.consumes,
            core=core,
            latency=1,
            meta={"line_buffers": core.line_buffers,
                  "shift_depth": core.shift_depth,
                  "out_bits": core.out_bits, "bit_depth": spec.bit_depth,
                  # what this block's memories are and whether each is
                  # behind np2hw's seam -- the fact an integrator binding
                  # macros needs, published rather than grepped for
                  "memories": list(core.get("memories", ()))},
        )

def _value_range(value) -> tuple:
    """The range a bound input's value covers: a constant's own, a
    context signal's declared width, a traced value's exact range."""
    from np2hw import Param as Np2hwParam

    if isinstance(value, Np2hwParam):
        top = 1 << (value.bits - 1 if value.signed else value.bits)
        return (-top, top - 1) if value.signed else (0, top - 1)
    if isinstance(value, int):
        return value, value
    return value.v.type.lo, value.v.type.hi


def _with_resets(param, flat):
    """``param`` with a design's reset values, which replace its derived
    default."""
    import dataclasses
    if flat is None:
        return param
    return dataclasses.replace(param, resets=tuple(flat), default=0,
                               default_unity=False, default_ramp=False,
                               default_identity=False)


def source_licence(function) -> list[str]:
    """The copyright and SPDX-License-Identifier lines a model's source file
    states in its header, as written there; empty where it states none."""
    import inspect
    try:
        path = inspect.getsourcefile(function)
        with open(path, encoding="utf-8") as fh:
            head = [next(fh, "") for _ in range(10)]
    except (TypeError, OSError):
        return []
    lines = []
    for line in head:
        text = line.strip().lstrip("#").strip()
        if text.startswith("Copyright") or \
                text.startswith("SPDX-License-Identifier:"):
            lines.append(text)
    return lines


def model_header(function, what: str) -> list[str]:
    """The header of Verilog traced from ``function``: the licence its source
    states -- a design derived from a model is under the model's terms -- and
    what it was generated from."""
    licence = [f"// {line}" for line in source_licence(function)]
    where = function.__module__.replace(".", "/") + ".py"
    return [
        *licence,
        *(["//"] if licence else []),
        f"// {what}",
        f"// Generated by np2hw from {function.__name__}() in {where}. Do not",
        "// edit by hand: the NumPy model is the specification, and the tests",
        "// hold this file bit-exactly equal to it.",
    ]


def configblock(name, *, version, description, params=(), stats=()):
    """A block with registers but no pixel stream: a pipeline's context
    block, say.

    Not a special case in the composer: it is allocated an address like any other
    block and simply has no ports to wire.
    """
    from np2hw.blocks.params import ParamSet

    return Block(
        model=None,
        params=ParamSet(block=name, version=version, params=list(params),
                        stats=list(stats), description=description),
        inputs=(), outputs=(),
        not_traceable="it owns configuration and has no datapath to trace",
    )


def block(*, version, description, name=None, params=(), stats=(),
          inputs=(), outputs=(), not_traceable="", even_frame=False):
    """Declare a block. The decorated function IS the block's model.

    A block is its model, its inputs and registers, and its stream ports;
    it knows nothing of the pipeline it is placed in. ``even_frame`` says its
    math indexes a 2x2 tile, so an odd frame is refused.

        @block(
            version=(1, 0),
            description="Per-CFA-colour black level offset with saturation.",
            inputs=(StreamPort("in"),),
            outputs=(StreamPort("out"),),
            params=[Param("offset", ...), Param("bayer_phase", bits=2)],
            even_frame=True,
        )
        def blacklevel(pixel, p, bit_depth):
            row, col = p.bayer_phase >> 1, p.bayer_phase & 1
            ...

    The block's name is the function's name. The model takes the pixel stream,
    its inputs (``p``) and the datapath width, and it is called
    with real arrays when it is the reference and with traced values when it is
    the hardware -- one function, both roles.

    Everything declared here is something the arithmetic cannot say. Nothing
    about the streams is declared beyond the port names: channel counts and
    widths are traced from the models and threaded edge to edge by the
    composer.
    """
    from np2hw.blocks.params import ParamSet

    def decorate(function):
        return Block(
            model=function,
            params=ParamSet(block=name or function.__name__, version=version,
                            params=list(params), stats=list(stats),
                            description=description),
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            not_traceable=not_traceable,
            even_frame=even_frame,
        )

    return decorate


def registry(package: str) -> dict[str, Block]:
    """Every :class:`Block` declared under ``package``, keyed by name.

    Discovered rather than listed, so a new block is available to a pipeline
    description as soon as it is declared -- there is no second place to register
    it and therefore no way for the two to disagree. Stub blocks are not declared
    as Blocks and so do not appear: a pipeline cannot name a block that has no
    model and no register interface.
    """
    root = importlib.import_module(package)
    found: dict[str, Block] = {}
    for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if isinstance(value, Block):
                found[value.name] = value
    return found


def resolve(package: str, name: str) -> Block:
    """The :class:`Block` named ``name`` under ``package``, or an error
    listing what exists."""
    available = registry(package)
    try:
        return available[name]
    except KeyError:
        raise KeyError(
            f"no block named {name!r}; declared blocks are {sorted(available)}. "
            "Blocks that are declared stubs have no model yet and cannot be "
            "composed.") from None


def comment(text: str, indent: str = "    ", width: int = 78) -> list[str]:
    """Wrap a parameter description into Verilog comment lines.

    Rule 4 says each parameter's description is carried through from its
    declaration into the generated Verilog. Carried through means ALL of it:
    truncating to the first clause drops precisely the part a reviewer needs
    (what to write, what the reset value means), which is the part that cannot
    be guessed from the signal name.
    """
    lines = textwrap.wrap(" ".join(text.split()), width=width - len(indent) - 3)
    return [f"{indent}// {line}" for line in lines] or [f"{indent}//"]
