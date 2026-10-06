# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Blocks: a NumPy model plus what its arithmetic cannot say.

A block is its model -- the decorated function, written at the hardware's
arithmetic: the specification and the golden reference -- and the
declarations nothing in that arithmetic states: its registers
(:class:`np2hw.blocks.params.ParamSet`, LOCAL offsets only; a composer
assigns each INSTANCE its base), the names of its stream ports, the
pipeline context it reads. Stream widths are not declared: np2hw traces
them from the model, and a composer threads each block's traced output into
the next block's input.

    ``Block.run()``      runs the model over a frame, validated from the
                         declarations -- no per-block adapter.
    ``Block.generate()`` traces the model into Verilog.

Verilog traced from a model carries the model's licence: its copyright
and SPDX identifier are read from the model's own source file, which states
them. The content that writes blocks binds one convention, in its own
decorator over :func:`block`: the pipeline context vocabulary its blocks
may consume (``vocabulary``). np2hw names no context signal and no licence.
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
    """

    name: str
    description: str = ""


@dataclass(frozen=True)
class ContextBit:
    """A single-bit slice of a pipeline context register, taken by the model.

    A traced model wants the Bayer phase as two one-bit values it can use as
    slice starts; the pipeline owns it as one two-bit context register. This
    says which bit feeds which model argument, so the composer can wire it and
    nobody writes a wrapper to split it.
    """

    name: str                 # the model argument, and the generated param port
    context: str              # the pipeline context signal it comes from
    bit: int
    description: str = ""

    @property
    def expression(self) -> str:
        return f"ctx_{self.context}[{self.bit}]"

    def to_np2hw(self):
        """The one-bit np2hw Param the traced model uses as a slice phase."""
        from np2hw import Param as Np2hwParam

        return Np2hwParam(self.name, bits=1, description=self.description)


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
        consumes: pipeline context signals the top module expects as wires.
        context_map: ``{param_name: verilog_expr}`` for generated ``param_*``
            inputs that are driven by pipeline CONTEXT rather than by a register.
            A Bayer phase is context, not configuration -- one owner, fanned out
            -- so the composer wires it straight from the context block instead of giving
            the block a CSR that would be a second copy of it.
        latency: pixels of delay from input to output, for pipeline alignment.
        meta: generator detail worth keeping (line buffers, accumulator width).
    """

    top: str
    modules: tuple[tuple[str, str], ...]
    params: tuple[tuple[str, int], ...] = ()
    consumes: tuple[str, ...] = ()
    context_map: dict = field(default_factory=dict)
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
    """Attribute access over a set of values, for the model's `p` and `ctx`."""

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


@dataclass
class Block:
    """A block: one model function, plus what cannot be derived from it.

    Produced by :func:`block`. The model is the specification; everything
    here is a declaration the arithmetic cannot supply -- the register set, what
    the streams MEAN, which context bits the model takes. Notably absent are the
    stream WIDTHS, which np2hw derives from the trace; declaring them here too
    would be a second source of truth free to disagree with the first.
    """

    model: object
    params: "ParamSet"
    inputs: tuple
    outputs: tuple
    context: tuple
    # Register-attribute overrides this VARIANT was configured with, empty on
    # the base block. {"gain": {"frac": 12}} -- the same dict the design JSON
    # carries and describe() recovers, so a variant knows its own recipe.
    overrides: dict = field(default_factory=dict)
    # Reset values a design set, {register: (leaf values, row-major)}; empty
    # on the base block. Part of the variant, because the register file's
    # reset is part of the block's register TYPE.
    resets: dict = field(default_factory=dict)
    # The pipeline context this block's content may consume: {name: Context},
    # bound by the content's decorator. A block reads each consumed signal's
    # declared width and properties from it, never from a list of its own.
    vocabulary: object = None
    # Why this block cannot be traced yet, if it cannot. Empty means it can.
    # Stated as a REASON rather than a flag, because "declared but not built" is
    # a fact about np2hw's current reach and the reader deserves to know which.
    not_traceable: str = ""

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

    @property
    def variant_signature(self) -> str:
        """A stable, Verilog-safe suffix naming this variant's overrides.

        Empty for the base block. Used wherever a TYPE needs a distinct name
        -- the SystemRDL regfile, the shared-module cache -- because two
        variants merging into one type is the map lying about one of them.
        """
        if not self.overrides and not self.resets:
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
        return "__" + "_".join(parts)

    # -- generation ----------------------------------------------------------- #

    def run(self, pixel, values: dict | None = None, bit_depth: int = 12,
            **context):
        """Run the model over a whole frame, validated from the DECLARATIONS.

        The per-block adapters this replaces were three copies of the same
        function, each re-stating facts the Block already declares: which
        context registers it consumes, and whether it is CFA-indexed. Now
        the declaration drives the validation -- a block that consumes
        ``bayer_phase`` gets the even-dimension check and the phase range
        check; one that does not, does not -- and a new block gets the whole
        adapter by existing.

        The model is handed the frame as the smallest unsigned NumPy integer
        that holds ``bit_depth`` bits -- the type the trace gives the same
        input, so the model's arithmetic promotes, and wraps, exactly as the
        generated hardware does. A frame with a value outside ``bit_depth``
        bits is refused rather than cut down to fit: it means the caller and
        the stream disagree about the depth.

        Args:
            pixel: the frame, ``(h, w)`` or ``(h, w, c)`` as the model's
                arithmetic expects, unsigned, ``bit_depth`` bits.
            values: register values for ``params.bind`` (defaults fill in).
            bit_depth: datapath width.
            **context: pipeline context by name, e.g. ``bayer_phase=2``.
                Everything the block consumes defaults to 0; a name the
                block does not consume is refused, because silently
                accepting it would be accepting a typo.
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
        ctx_values = {}
        for name in consumed:
            declared = self.context_signal(name)
            value = int(context.get(name, 0))
            lo, hi = 0, (1 << declared.bits) - 1
            if not lo <= value <= hi:
                raise ValueError(
                    f"context {name!r} = {value} outside its declared "
                    f"{declared.bits}-bit range")
            ctx_values[name] = value
        if any(self.context_signal(bit.context).even_frame
               for bit in self.context):
            height, width = pixel.shape[:2]
            if height % 2 or width % 2:
                raise ValueError(
                    f"frame {width}x{height} has an odd dimension; a "
                    "CFA-indexed block has no consistent Bayer phase unless "
                    "both dimensions are even")
        bound = self.params.bind(values or {})
        return self.model(pixel, bound, self.context_view(ctx_values),
                          bit_depth)

    def context_signal(self, name: str):
        """The declaration of one consumed context signal, from the
        vocabulary the block's content bound."""
        vocabulary = self.vocabulary() if callable(self.vocabulary) \
            else self.vocabulary
        if not vocabulary or name not in vocabulary:
            raise KeyError(
                f"block {self.name!r} consumes {name!r}, which its content's "
                f"context vocabulary does not declare "
                f"({sorted(vocabulary or ())})")
        return vocabulary[name]

    def context_view(self, values: dict):
        """The `ctx` argument for a NumPy run, from context register values."""
        return _View({bit.name: (int(values[bit.context]) >> bit.bit) & 1
                      for bit in self.context})

    def generate(self, spec, width: int, height: int, module_name: str,
                 clk_ns: float | None = None,
                 sync_memory: bool = False) -> "Generated":
        """Trace the model into Verilog. Generic: every block uses this one.

        There is no per-block generator and no hand-written Verilog anywhere in
        this package. A block is its model plus its declarations, and this turns
        the pair into a module.
        """
        from np2hw import Image2D, to_ir, generate as np2hw_generate

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
        image = Image2D("pixel", width, height, bits=spec.data_bits, signed=False)
        context = [bit.to_np2hw() for bit in self.context]
        registers = [param.to_np2hw() for param in self.params]

        def traced(pixel, *traced_args):
            from np2hw.blocks.params import Declarations

            ctx = _View(dict(zip((bit.name for bit in self.context),
                                 traced_args[:len(context)])))
            values = _View(dict(zip((param.name for param in self.params),
                                    traced_args[len(context):])))
            # The same declaration access the NumPy run has: a model whose
            # arithmetic depends on a declared attribute (a Q format's shift)
            # reads it from the ONE configured Param, in both modes.
            object.__setattr__(values, "_decl",
                               Declarations(self.params.params))
            return self.model(pixel, values, ctx, spec.bit_depth)

        # The channel count is the SPEC's -- threaded forward by the composer
        # from whatever the driving block's trace published, starting from the
        # pipeline input. Nothing here re-declares it; if the arithmetic
        # cannot digest the stream it is given (a 3-channel unpack of a
        # 1-channel word), the trace itself is where that surfaces.
        _, result = to_ir(traced, image, *context, *registers,
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
            context_map={bit.name: bit.expression for bit in self.context},
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
        inputs=(), outputs=(), context=(),
        not_traceable="it owns configuration and has no datapath to trace",
    )


def block(*, version, description, name=None, params=(), stats=(),
          consumes=(), inputs=(), outputs=(), context=(), not_traceable="",
          vocabulary=None):
    """Declare a block. The decorated function IS the block's model.

    Content binds the pipeline context its blocks may consume once, in its
    own decorator over this one (``functools.partial(block,
    vocabulary=...)``).

        @block(
            version=(1, 0),
            description="Per-CFA-colour black level offset with saturation.",
            inputs=(StreamPort("in"),),
            outputs=(StreamPort("out"),),
            consumes=("bayer_phase",),
            context=(ContextBit("phase_row", "bayer_phase", bit=1), ...),
            params=[Param("offset", ...)],
        )
        def blacklevel(pixel, p, ctx, bit_depth):
            ...

    The block's name is the function's name. The model takes the pixel stream,
    a parameter view, a context view and the datapath width, and it is called
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
                            consumes=tuple(consumes), description=description),
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            context=tuple(context),
            not_traceable=not_traceable,
            vocabulary=vocabulary,
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
