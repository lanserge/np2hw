# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Composition, address allocation, and the machine-readable register map.

This is where blocks stop being declarations and become hardware. Every step
here is a function of a loaded design (:func:`np2hw.pipeline.load`) -- its
block instances in declaration order, the edges between them, the stream
entering it -- and holds no copy of it. Three jobs:

  1. Compose blocks as a netlist: each stream from the port that drives it
     to every port it feeds.
  2. Assign each block INSTANCE a base address. Blocks declare local offsets
     only; the same block added twice gets two different bases, which is what
     makes a stereo pipeline work and what makes a block independently testable.
  3. Emit the register map as JSON. The host API and the documentation are both
     generated from that file. Nothing anywhere hardcodes an address.

That last point is the one that decays quietly if it is not enforced, so it is
worth stating what it rules out: no constant in a host script, no ``#define`` in
a C header checked into this repo, no address in a docstring that a reader might
copy. If software needs an address it reads the JSON, and if the JSON is stale
the ID-and-version word at each block's local offset 0 catches it before the
first pixel moves.

Layout
------

    0x0000  pipe            pipeline context, fanned out to blocks as wires
    0x0100  first block instance
    0x0200  second block instance
    ...
    0x8000  statistics region -- RAM windows, double-buffered, bulk read

Every config block starts on a 256-byte boundary, so the address decoder is a
bit-slice compare (``addr[15:8] == 8'h01``) rather than a pair of magnitude
comparators. Address space is free; decode logic and readable output are not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from np2hw.blocks import Block, Generated, comment, source_licence
from np2hw.blocks.params import (
    BLOCK_ALIGN,
    ID_VERSION_OFFSET,
    REG_WIDTH,
    AddressAllocator,
    BlockInstance,
    ParamSet,
)
from np2hw.blocks.stream import StreamSpec

# Bumped when the STRUCTURE of the emitted JSON changes, so a host can refuse a
# map it does not understand instead of misreading it.
#
#   2  added `control` (the bus, and what the commit and the error response mean);
#      `commit` on a register became "frame_boundary" -- the copy happened
#      automatically as the frame in flight ended.
#   3  the contract changed and for a while the map DID NOT SAY SO, which
#      is the exact failure this file exists to prevent: software reading
#      the map would have written coefficients and waited forever for an
#      automatic commit that no longer exists. `commit` on a register is
#      now "armed": nothing takes effect until software writes 1 to
#      the context block's commit; while it is set, coefficient writes are REFUSED
#      (SLVERR); the datapath clears it when the values are taken. The
#      bump is what lets a version-2 host refuse this map instead of
#      misreading it.
#   4  every register carries its presentation: `min`/`max` (the raw
#      representable range, so no consumer re-derives two's complement),
#      `widget` (choice | checkbox | fixed | slider, derived from the
#      declaration by one rule), and `choices` when the register selects
#      among named values rather than measuring a quantity.
#   5  `composites`: every composite instance the design was built from,
#      by path and reference ("" for the design itself), so a profile
#      keyed relative to a composite can be anchored wherever it sits.
MAP_FORMAT_VERSION = 5


# One end of a connection. np2hw's: the netlist is np2hw's structure, and a
# second Endpoint here would be the graph vocabulary described twice.
from np2hw.netlist import Endpoint, Netlist, Node  # noqa: E402


@dataclass(frozen=True)
class Stage:
    """One block instance of a design, with everything needed to wire it."""

    path: str
    block: "Block"
    instance: BlockInstance

    @property
    def paramset(self) -> ParamSet:
        return self.instance.paramset

    @property
    def ports(self):
        """The block's declared stream ports."""
        return self.block.ports

    @property
    def module_prefix(self) -> str:
        """Verilog-safe prefix derived from the instance path.

        ``left.blacklevel`` becomes ``left_blacklevel``, so two instances of one
        block produce two distinctly named modules and two distinct signal
        groups.
        """
        return self.path.replace(".", "_")


@dataclass(frozen=True)
class Subsystem:
    """A reusable sub-graph, composed once and instantiated by name.

    A stereo camera is two eyes running the same chain. Describing that as a
    composite instantiated twice emits ONE module per block, instantiated per
    instance, rather than the same graph copied into the top level.

    Addresses are still allocated per INSTANCE: a `blacklevel` inside instance
    `left` is the instance `left.blacklevel`, with its own registers. So the
    register map is byte-identical whether a design shares a composite or
    spells the graph out, and only the emitted structure differs. There is a
    test for that, because it is the property that makes the change safe.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    nodes: tuple                    # ((inner_instance, Block), ...)
    edges: tuple                    # ((source, sink), ...) as raw strings


@dataclass(frozen=True)
class Shape:
    """What a design is built around: the stream entering it.

    ``context`` names what arrives WITH THE DATA rather than being configured
    -- geometry and CFA phase carried in a stream's header, say. Those become
    input ports on the control top and get no register, because a register
    holding a second copy of a header fact is a second answer to one
    question, and software could set it to disagree with the picture
    arriving. What stays a register is what is genuinely a CHOICE: a crop
    window is software's to pick, a line length is not."""

    spec: StreamSpec
    width: int
    height: int
    context: frozenset


# A design is built by the steps below, each a function of the design alone:
# its block instances in declaration order, the edges between them, and the
# stream entering it. Nothing here holds a second copy of the graph -- the
# netlist is the design's edges, handed to np2hw for its laws -- and every
# answer is recomputed from structure, which is what makes allocation a pure
# function of it.

def _check_blocks(design) -> None:
    """A design generated by these steps is made of model blocks."""
    from np2hw.design.composite import CompositionError
    others = [f"{i.path} ({i.kind})" for i in design.instances.values()
              if i.kind != "block"]
    if others:
        raise CompositionError(
            f"{design.ref}: {', '.join(others)} -- what is generated and "
            "proved from its models is made of @ispblocks; everything else "
            "is built around it")
    if design.aliases:
        raise CompositionError(
            f"{design.ref}: {sorted(design.aliases)} follow other registers; "
            "two blocks sharing one register is not built yet")
    if not design.inputs:
        raise CompositionError(f"{design.ref}: no inputs, so no stream")


def shape(design) -> Shape:
    """The stream a design is built around, from where it enters.

    Every block is built around ONE geometry and its depth traced from ONE
    input depth, so every input must state the same facts; carrying
    geometry block by block, through a block that changes it, belongs to
    the compiler that traces shapes."""
    from np2hw.design.composite import CompositionError
    cached = design.__dict__.get("_shape")
    if cached is not None:
        return cached
    _check_blocks(design)
    facts = None
    for port in design.inputs:
        stream = design.streams.get(port, {})
        missing = [f for f in ("bits", "max_width", "max_height")
                   if f not in stream]
        if missing:
            raise CompositionError(
                f"{design.ref}: the stream entering {port} states no "
                f"{', '.join(missing)}; say it on the input port where the "
                "stream enters")
        here = {f: stream[f] for f in ("bits", "max_width", "max_height")}
        here["channels"] = stream.get("channels", 1)
        here["context"] = tuple(stream.get("context", ()))
        if facts is not None and here != facts[1]:
            raise CompositionError(
                f"{design.ref}: the streams entering {facts[0]} and {port} "
                f"differ ({facts[1]} vs {here}); one design is built around "
                "one stream shape today")
        facts = (port, here)
    stream = facts[1]
    for name in stream["context"]:
        if vocabulary(design).resolve(name).origin == "register":
            raise ValueError(
                f"{name!r} cannot arrive with the stream: it is software's "
                "choice, not a fact a transport carries, so it stays a "
                "register")
    result = Shape(spec=StreamSpec(bit_depth=int(stream["bits"]),
                                   channels=int(stream.get("channels", 1))),
                   width=int(stream["max_width"]),
                   height=int(stream["max_height"]),
                   context=frozenset(stream["context"]))
    design.__dict__["_shape"] = result
    return result


def vocabulary(design):
    """The pipeline context of a design's blocks: the one vocabulary every
    block that has one was declared with (np2hw.blocks.params.Vocabulary),
    which names the context block placed in the pipeline."""
    found = {}
    for inst in design.instances.values():
        leaf = inst.leaf
        if isinstance(leaf, Block) and leaf.vocabulary is not None:
            found.setdefault(id(leaf.vocabulary), (leaf.vocabulary, inst.path))
    if len(found) != 1:
        from np2hw.design.composite import CompositionError
        where = sorted(path for _, path in found.values())
        raise CompositionError(
            f"{design.ref}: its blocks bring {len(found)} context "
            f"vocabularies{' (' + ', '.join(where) + ')' if where else ''}; "
            "a pipeline is built around one, which names its context block")
    vocab = next(iter(found.values()))[0]
    if vocab.block is None:
        from np2hw.design.composite import CompositionError
        raise CompositionError(f"{design.ref}: its blocks' context vocabulary "
                               "names no context block")
    return vocab


def licence(design) -> str:
    """The SPDX licence expression of a design's generated Verilog: the one
    its models' sources state. A design derived from models is under their
    terms; models under different terms are refused rather than guessed
    between."""
    lines = header_licence(design)
    ids = [line.split(":", 1)[1].strip() for line in lines
           if line.startswith("SPDX-License-Identifier:")]
    return ids[0] if ids else ""


def header_licence(design) -> list[str]:
    """The copyright and SPDX lines every traced model's source states --
    all the same, or the design is refused."""
    seen = {}
    for inst in design.instances.values():
        leaf = inst.leaf
        if isinstance(leaf, Block) and leaf.model is not None:
            seen.setdefault(tuple(source_licence(leaf.model)), inst.path)
    if len(seen) > 1:
        from np2hw.design.composite import CompositionError
        raise CompositionError(
            f"{design.ref}: its models' sources state different licences "
            f"({', '.join(sorted(seen.values()))}); the Verilog generated "
            "from them would have no one licence to carry")
    return list(next(iter(seen), ()))


def header(design, what: str, source: str) -> list[str]:
    """The header of a file generated from a design: the licence its
    models' sources state, and what it was generated from."""
    lines = [f"// {line}" for line in header_licence(design)]
    return [
        *lines,
        *(["//"] if lines else []),
        f"// {what}",
        f"// Generated by np2hw from {source}. Do not edit by hand:",
        "// the NumPy models are the specification, and the tests hold this",
        "// file bit-exactly equal to them.",
    ]


def module_prefix(design) -> str:
    """The prefix of the modules generated for a design's blocks: the package
    their models come from, so the same blocks make the same modules whatever
    library the design that places them lives in."""
    roots = {leaf.model.__module__.partition(".")[0]
             for leaf in (i.leaf for i in design.instances.values())
             if isinstance(leaf, Block) and leaf.model is not None}
    return roots.pop() if len(roots) == 1 else "np2hw"


def _available(design) -> tuple:
    """The context signals a design COULD have: a register-backed one, and
    a stream-only one where the stream delivers it."""
    carried = shape(design).context
    return tuple(c for c in vocabulary(design).signals
                 if c.origin != "stream" or c.name in carried)


def contexts(design) -> tuple:
    """The context signals a design HAS: every one its stream delivers, and
    a register-backed one only where something reads it -- a block that
    consumes it, or the cores' run-time frame, which reads the frame's
    width and height. A register nothing reads is not made: the vocabulary
    may declare it, and the first block that consumes it brings it back.
    One answer, read by the ports, the binding, the pass-through and the
    register map alike."""
    carried = shape(design).context
    read = {name for i in design.instances.values()
            if isinstance(i.leaf, Block) for name in i.leaf.params.consumes}
    if any(isinstance(i.leaf, Block) and i.leaf.traceable
           for i in design.instances.values()):
        read.update(frame_signals(design) or ())
    return tuple(c for c in _available(design)
                 if c.name in carried or c.name in read)


def _unmade(design) -> set:
    """The context block's registers the hardware does not have: a signal
    the stream owns, or one nothing reads. The decode and the map both
    leave them out -- the map must say what the decode does, or it is
    documentation of a design that was not built."""
    have = {c.name for c in contexts(design)} - set(shape(design).context)
    return {c.name for c in vocabulary(design).signals} - have


def frame_signals(design):
    """The context signals that ARE the frame's width and height -- the one
    signal following each that can arrive with a frame -- or None where the
    design has no such pair. With them, every core is built for any frame up
    to the design's size and told the frame's size at run time; the build's
    size stays what a line buffer holds."""
    have = {c.name for c in _available(design)}
    pair = []
    for dim in ("width", "height"):
        found = [c.name for c in vocabulary(design).signals
                 if c.follows == dim and c.origin != "register"
                 and c.name in have]
        if len(found) != 1:
            return None
        pair.append(found[0])
    return tuple(pair)


def _frame_at(signal: str, less: int) -> str:
    """A context signal less what the stages before took off it."""
    return (f"ctx_{signal}" if less == 0 else
            f"(ctx_{signal} - 16'd{less})" if less > 0 else
            f"(ctx_{signal} + 16'd{-less})")


def stages(design) -> list[Stage]:
    """Every block instance with its base address: the context block first,
    at base 0, then the design's instances in DECLARATION order -- the order
    addresses are assigned in, so the same design is the same map however
    its connections are written.

    The context block has no stream ports, so it is no node of the graph,
    but it is allocated like everything else: the moment it becomes a
    special case every future change has to remember it exists."""
    cached = design.__dict__.get("_stages")
    if cached is not None:
        return cached
    form = shape(design)
    carried = form.context
    allocator = AddressAllocator(config_base=0x0000, stats_base=0x8000)
    out = []
    vocab = vocabulary(design)
    # A register that follows the frame's width or height resets to THIS
    # design's: a bitstream nobody configures processes the frame it was
    # built for, whatever default the vocabulary was written with.
    sizes = {"width": form.width, "height": form.height}
    declared = {p.name for p in vocab.block.params.params}
    frame_resets = {c.name: sizes[c.follows] for c in vocab.signals
                    if c.follows in sizes and c.name in declared}
    context_block = vocab.block.configure(resets=frame_resets)
    for path, block in [(vocab.block.name, context_block)] + [
            (i.path, i.leaf) for i in design.instances.values()]:
        for signal in block.params.consumes:
            context = vocab.resolve(signal)    # raises with the names
            if context.origin == "stream" and signal not in carried:
                raise ValueError(
                    f"{path!r} consumes {signal!r}, which changes every frame "
                    "and so has no register: it must arrive with the stream. "
                    "List it in the stream's context where it enters")
        out.append(Stage(path=path, block=block,
                         instance=allocator.allocate(path, block.params)))
    design.__dict__["_stages"] = out
    design.__dict__["_allocator"] = allocator
    return out


def allocator(design) -> AddressAllocator:
    """The allocation `stages` made: regions and the address width."""
    stages(design)
    return design.__dict__["_allocator"]


def stage(design, path: str) -> Stage:
    for s in stages(design):
        if s.path == path:
            return s
    raise KeyError(f"no block instance {path!r}; this design has "
                   f"{[s.path for s in stages(design)]}")


def address_of(design, path: str, register: str) -> int:
    """Absolute address of one register of one instance."""
    return stage(design, path).instance.address_of(register)


def netlist(design) -> Netlist:
    """The design's graph, as np2hw holds it, checked by np2hw's laws: the
    four validations (undriven input, double driver, cycle, unbuffered fork)
    are rules about np2hw's streaming handshake, so np2hw performs them.
    What this adds is only the mapping from nodes to STAGES."""
    cached = design.__dict__.get("_netlist")
    if cached is not None:
        return cached
    graph = Netlist(design.name, external_inputs=design.inputs,
                    external_outputs=design.outputs)
    for s in stages(design):
        if s.ports.inputs or s.ports.outputs:
            graph.add(Node(s.path, inputs=tuple(s.ports.inputs),
                           outputs=tuple(s.ports.outputs)))
    for source, sink in design.edges:
        graph.connect(source, sink)
    graph.validate()
    design.__dict__["_netlist"] = graph
    return graph


def datapath(design) -> list[Stage]:
    """Nodes that carry pixels, in topological order (the context block
    excluded).

    Generation order. NOT the order addresses were allocated in -- see
    :func:`stages` for that, and note the difference matters: a description
    recovered in the wrong order rebuilds to different addresses."""
    owner = vocabulary(design).block.name
    return [stage(design, node.name) for node in netlist(design).order()
            if node.name != owner]


# -- register map ----------------------------------------------------------- #

def register_map(design) -> dict:
    """The whole address map as a JSON-able dict.

    This is the single source of truth that the host API and the register-map
    documentation are generated from."""
    form = shape(design)
    alloc = allocator(design)
    owner = vocabulary(design).block.name
    commit = f"{owner}.{vocabulary(design).commit}"
    unmade = _unmade(design)
    blocks = []
    for s in stages(design):
        paramset, inst = s.paramset, s.instance
        registers = []
        for reg in paramset.registers:
            param = reg.param
            if s.path == owner and param.name in unmade:
                continue        # no register for it; see _unmade
            low, high = param.limits
            registers.append({
                "name": param.name,
                "offset": reg.offset,
                "address": inst.base + reg.offset,
                "bits": param.bits,
                "signed": param.signed,
                "frac": param.frac,
                "q_format": param.q_format,
                "default": param.default,
                # Presentation, derived from the declaration by ONE rule
                # (Param.widget) and stated here because this map is
                # generated output -- a UI consumes it, it never re-derives
                # two's-complement limits in JavaScript.
                "min": low,
                "max": high,
                "widget": param.widget,
                **({"choices": [{"value": v, "label": label}
                                for v, label in param.choices]}
                   if param.choices else {}),
                "access": "rw",
                "commit": "armed",
                "description": param.description,
            })
        windows = []
        for window in paramset.stats:
            windows.append({
                "name": window.name,
                "base": inst.stats_bases[window.name],
                "buffer_bytes": window.buffer_bytes,
                "size_bytes": window.size_bytes,
                "words": window.words,
                "records": window.records,
                "record_words": window.record_words,
                "layout": list(window.layout),
                "access": "ro",
                "buffering": "double",
                "description": window.description,
            })
        blocks.append({
            "path": s.path,
            "block": paramset.block,
            "id": paramset.block_id,
            "version": f"{paramset.version[0]}.{paramset.version[1]}",
            "base": inst.base,
            "size": paramset.size_bytes,
            "description": paramset.description,
            "consumes": list(paramset.consumes),
            # Build-time register-attribute overrides this instance was
            # configured with; {} for the base declaration. Informative:
            # every derived fact (q_format, defaults) already reflects
            # them, this records WHY.
            "register_overrides": s.block.overrides,
            # False means the registers below are decoded and read back but
            # reach no datapath: the block is declared and not built yet.
            # Software is told, rather than finding out from an image that
            # did not change.
            "implemented": s.block.implemented,
            "not_implemented_reason": ("" if s.block.implemented
                                       else s.block.not_traceable),
            "id_version": {
                "offset": ID_VERSION_OFFSET,
                "address": inst.id_version_address,
                "value": paramset.id_version_word,
                "access": "ro",
                "description": "Block ID in the high half, major.minor version "
                               "in the low half. Read it before anything else: "
                               "it is how software proves the bitstream is the "
                               "one it was built against",
            },
            "registers": registers,
            "statistics": windows,
        })

    return {
        "map_format_version": MAP_FORMAT_VERSION,
        "name": design.name,
        "composites": [{"path": path, "block": ref}
                       for path, ref in design.composites.items()],
        "generator": "np2hw",
        "licence": licence(design),
        "geometry": {"width": form.width, "height": form.height},
        "stream": {
            "bit_depth": form.spec.bit_depth,
            "channels": form.spec.channels,
            "data_bits": form.spec.data_bits,
        },
        "address_bits": alloc.address_bits(),
        "block_align": BLOCK_ALIGN,
        "control": {
            "bus": "axi4-lite",
            "data_bits": REG_WIDTH,
            "address_bits": alloc.address_bits(),
            "commit": "armed",
            "commit_register": commit,
            "commit_description":
                "Writes land in the registers but reach the datapath only "
                f"when software writes 1 to {commit} -- a host that never "
                "commits sees its writes read back and change nothing. While "
                "commit is set, coefficient writes are refused (SLVERR) so "
                "the batch is frozen; the datapath takes the whole batch at "
                "a frame boundary and clears the bit, so poll it to know the "
                "update landed. One commit therefore applies its batch "
                "atomically: never half a colour matrix",
            "commit_caveat":
                "A link reset (cable unplug) reloads the datapath's copies "
                "from the registers as they stand, UNCOMMITTED writes "
                "included -- commit gates when an update lands, not whether "
                "it eventually becomes visible. Do not stage writes you are "
                "not prepared to see applied",
            "error_response":
                "A write to a read-only word, any access to an unmapped "
                "address, and a coefficient write while commit is set, are "
                "answered SLVERR. On Linux /dev/mem that is a SIGBUS, not "
                "an error code",
        },
        "regions": {
            "config": {
                "base": alloc.config_base,
                "size": alloc.config_span,
                "structure": "one 32-bit CSR per word, shadowed and committed "
                             "at the frame boundary",
            },
            "statistics": {
                "base": alloc.stats_base,
                "structure": "double-buffered RAM windows, read-only, read in "
                             "bulk",
            },
        },
        "blocks": blocks,
    }


def write_register_map(design, path: str | Path) -> Path:
    """Write the register map JSON next to the generated Verilog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register_map(design), indent=2) + "\n")
    return path


# -- control plane ------------------------------------------------------------ #

def address_map(design):
    """The whole register map as an ``np2hw.AddrMap``: types, then instances.

    This is the same allocation :func:`register_map` publishes, handed to
    np2hw so it can emit the decode and render the SystemRDL. the pipeline decides
    WHERE everything lives -- one 256-byte-aligned block per instance, an
    identity word at each base -- and np2hw decides what an AXI4-Lite slave
    and a ``.rdl`` file look like. Neither holds a copy of the other's half,
    which is the only way the JSON map and the hardware cannot drift apart.

    The hierarchy is kept: one ``RegBlock`` per block TYPE, one
    ``RegInstance`` per stage. A stereo design therefore hands np2hw ONE
    ``blacklevel`` layout instantiated twice, exactly as the emitted Verilog
    has one module instantiated twice -- flattening happens inside np2hw, at
    the emitter, the one place the flat list is genuinely needed.

    Q format and fractional bits travel in ``properties``: the map's
    concepts, carried opaquely by np2hw into renderings that can hold them.
    """
    from np2hw import AddrMap, Reg, RegBlock, RegInstance

    owner = vocabulary(design).block.name
    unmade = _unmade(design)
    types: dict[tuple, RegBlock] = {}
    instances = []
    for s in stages(design):
        paramset = s.paramset
        # Type identity includes the VARIANT: two whitebalance instances with
        # different Q formats are two types, and merging them would make the
        # map lie about one. Identically-configured instances share a
        # variant object (configure() caches), so they still collapse to one
        # type.
        type_name = paramset.block + s.block.variant_signature
        block = types.get(type_name)
        if block is None:
            # A declared-but-unbuilt block still gets its registers decoded,
            # so the map and the hardware stay the same document. Saying so
            # in the RTL and the map is the alternative to a register that
            # quietly does nothing. A property of the TYPE, like the layout.
            unbuilt = ("" if s.block.implemented else
                       f" NOTE: {paramset.block} is declared but not built "
                       f"yet ({s.block.not_traceable}), so this register "
                       "reads back what was written and drives nothing.")
            regs = [Reg(
                name="id_version", bits=32, offset=ID_VERSION_OFFSET,
                access="ro", value=paramset.id_version_word,
                description=f"block ID 0x{paramset.block_id:04x} in the high "
                            f"half, version {paramset.version[0]}."
                            f"{paramset.version[1]} in the low half. Read it "
                            "before anything else: it is how software proves "
                            "the bitstream is the one it was built against.")]
            for register in paramset.registers:
                param = register.param
                if s.path == owner and param.name in unmade:
                    continue        # the stream owns it, or nothing reads it
                regs.append(Reg(
                    name=param.name, bits=param.bits, offset=register.offset,
                    signed=param.signed, reset=param.default,
                    description=f"({param.q_format}). "
                                f"{param.description}.{unbuilt}",
                    properties={"q_format": param.q_format,
                                "frac": param.frac}))
            block = types[type_name] = RegBlock(
                name=type_name, regs=tuple(regs),
                size=paramset.size_bytes, description=paramset.description)
        elif block.size != paramset.size_bytes or (
                s.path != owner
                and len(block.regs) != 1 + len(paramset.registers)):
            raise ValueError(
                f"two different layouts both call themselves "
                f"{paramset.block!r}; a block TYPE has one layout")
        instances.append(RegInstance(path=s.path, block=block,
                                     base=s.instance.base))
    return AddrMap(name=design.name, instances=tuple(instances),
                   data_bits=REG_WIDTH,
                   description=f"{design.name} -- generated by np2hw; the "
                               "emitted JSON register map is the "
                               "authoritative companion")


def registers(design) -> list:
    """The flat ``np2hw.Reg`` list, in address order: :func:`address_map`
    flattened, a projection of the one allocation, not a second rendering."""
    return address_map(design).flatten()


def write_systemrdl(design, path: str | Path) -> Path:
    """Write the register map as SystemRDL 2.0, next to the Verilog.

    SystemRDL is what an integration or verification team feeds their own
    tooling: the PeakRDL exporters turn this one file into a UVM register
    model, C headers, IP-XACT and browsable documentation. Those are tools
    the INTEGRATOR runs; np2hw emits the standard format and stops."""
    from np2hw import systemrdl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(systemrdl(
        address_map(design),
        header=header(
            design, what=f"{design.name} -- register map as SystemRDL 2.0",
            source="the same allocation as the emitted JSON map")))
    return path


def control_bind(design, built: dict) -> dict:
    """What drives each parameter port of the composed datapath top.

    Three kinds of port, and the distinction is the whole of rule 2: a
    block's own configuration comes from its own register; pipeline context
    comes from the context block's registers, ONE copy fanned out to every
    consumer;
    and a context BIT is a slice of one of those."""
    carried = shape(design).context
    owner = vocabulary(design).block.name
    bind = {f"ctx_{ctx.name}":
            (f"ctx_{ctx.name}" if ctx.name in carried
             else f"param_{owner}_{ctx.name}")
            for ctx in contexts(design)}
    for path, result in built.items():
        s = stage(design, path)
        for name, _ in result.params:
            bind[f"param_{s.module_prefix}_{name}"] = \
                f"param_{s.module_prefix}_{name}"
    return bind


# -- generation --------------------------------------------------------------- #

def generate(design, control: bool = True, clk_ns: float | None = None,
             sync_memory: bool = False) -> Generated:
    """Emit every block's Verilog, and hand the netlist to np2hw to compose.

    This emits no Verilog. It describes: which instances exist, how they
    connect, what drives their parameter ports, and what every register
    means. np2hw generates the block datapaths from the models and the top
    level from the netlist, because it is np2hw that decides what a
    generated core's ports are and what the valid/ready contract means.

    Args:
        control: wrap the datapath in the AXI4-Lite register file, which is
            what a design needs to be configurable on hardware. ``False``
            stops at the datapath top, whose parameters are flat input
            ports -- what a bit-exact testbench wants, because it drives the
            coefficients directly instead of writing them over a bus.
    """
    from np2hw import compose
    from np2hw.design.composite import _shared_subsystems

    form = shape(design)
    netlist(design)                      # np2hw's laws, before anything is made
    shared = _shared_subsystems(design)
    signals = frame_signals(design)

    modules: list[tuple[str, str]] = []
    built: dict[str, Generated] = {}
    # A block inside a shared composite is generated ONCE, under a name that
    # does not mention which instance it ended up in -- that is the point.
    seen: dict[str, Generated] = {}
    for s in datapath(design):
        if not s.block.traceable:
            continue                    # declared, not yet built (e.g. stats)
        owner = s.path.rpartition(".")[0]
        sub = shared.get(owner)
        prefix = module_prefix(design)
        name = (f"{prefix}_{sub.name}_{s.path.rpartition('.')[2]}"
                if sub else f"{prefix}_{s.module_prefix}")
        if name in seen:
            built[s.path] = seen[name]
            continue
        # Each block sees the stream its DRIVER's trace produced, threaded
        # forward edge by edge from the design's input: a chroma stage eats
        # the 2-channel word its green stage emitted, ccm eats the 3-channel
        # word the chroma stage emitted. Depth travels the same way -- a
        # stage that saturates back hands on the depth it was given, one
        # that narrows after a tone curve hands on less, one that widens
        # hands on more. Nothing declares any of it anywhere: np2hw
        # publishes what each model's arithmetic packed, and that IS the next
        # block's input. The only declared stream fact in a whole design is
        # the input's own.
        given = _incoming(design, s, built)
        # The frame as it reaches this block: the design's, less what the
        # blocks before it took off. Every core that follows a run-time
        # frame keeps its input's size or loses a constant span (a valid
        # window's), so the traced difference holds at every size.
        frame = None
        if signals:
            frame = (_frame_at(signals[0], form.width - given.width),
                     _frame_at(signals[1], form.height - given.height))
        result = s.block.generate(
            form.spec.with_stream(given.channels, given.bit_depth),
            given.width, given.height, module_name=name, clk_ns=clk_ns,
            sync_memory=sync_memory, frame=frame)
        modules.extend(result.modules)
        built[s.path] = seen[name] = result

    top = compose(
        module_name=design.name,
        instances=_instances(design, built),
        connections=_connections(design, built),
        ports=_ports(design, built),
        header=header(
            design, what=f"{design.name} -- composed ISP pipeline "
                 f"({len(built)} generated block(s), {form.spec.bit_depth}-bit)",
            source="the design's JSON netlist",
        ),
        notes=_address_notes(design),
        # A block reads its own copy of its own coefficients, kept beside
        # it, and takes an update as the frame boundary reaches it. The
        # register file stays the one place software writes; it stops being
        # the place the datapath reads from, which is what put a knot
        # register on one side of the chip and the arithmetic that selects
        # it on the other.
        commit=True,
        arm=("cfg_arm_req", "cfg_arm_ack"),
        # Configuration only. `ctx_*` is context that arrived with a frame,
        # and holding it to a commit would make a frame's own geometry
        # arrive one frame late.
        commit_ports={f"param_{stage(design, path).module_prefix}_{name}"
                      for path, result in built.items()
                      for name, _ in result.params},
    )
    modules.append((design.name, top["verilog"]))

    meta = {"nodes": [s.path for s in datapath(design)],
            "generated": sorted(built), "nets": top["nets"],
            # What the core's boundaries actually carry. The INPUT is the
            # one declared stream fact -- the shape IS that declaration.
            # Every OUTPUT is traced: a three-channel word is three fields
            # wide because the models packed it that way, not because
            # anyone wrote the number down. Published so an integrator reads
            # the width instead of restating it, which is how a 30-bit slice
            # survives into a 36-bit build.
            "boundary": {
                "inputs": {n: {"data_bits": form.spec.data_bits}
                           for n in design.inputs},
                "outputs": {n: _output_boundary(design, built, n)
                            for n in design.outputs},
            },
            # Every memory the design infers, per stage, and whether it is
            # behind np2hw's seam. An integrator binding macros reads this;
            # sync_memory=True is the promise that every entry here says
            # "sync".
            "memories": {path: g.meta["memories"]
                         for path, g in built.items()
                         if g.meta.get("memories")}}
    name = design.name
    # every port of the top, as np2hw published it: what an integrator --
    # a board, a testbench -- wires, never read back from the Verilog
    meta["ports"] = top["interface"]["ports"]
    if control:
        wrapper = _control_top(design, top, built)
        modules.append((wrapper["module"], wrapper["verilog"]))
        name = wrapper["module"]
        meta["control"] = {"bus": "axi4-lite", "addr_bits": wrapper["addr_bits"],
                           "commit": wrapper["commit"],
                           "regfile": wrapper["reg"]["module"]}
        meta["ports"] = wrapper["interface"]["ports"]
        meta["facts"] = wrapper["facts"]
        meta["constraints"] = wrapper["constraints"]

    return Generated(
        top=name,
        modules=tuple(modules),
        consumes=tuple(c.name for c in contexts(design)),
        latency=sum(r.latency for r in built.values()),
        meta=meta,
    )


def _control_top(design, top: dict, built: dict) -> dict:
    """The register file, in front of the composed datapath.

    Everything structural here is np2hw's: the pipeline hands it the register list
    it allocated and the binding it decided, and gets back the decode, the
    shadow registers and the commit. That the datapath top's parameter ports
    and the register names line up is not a coincidence to be maintained --
    both are built from :attr:`Stage.module_prefix` in one place.
    """
    from np2hw import control_wrap

    carried = shape(design).context
    return control_wrap(
        top, address_map(design), bind=control_bind(design, built),
        passthrough=[(f"ctx_{c.name}", c.bits) for c in contexts(design)
                     if c.name in carried],
        module_name=f"{design.name}_ctrl",
        addr_bits=allocator(design).address_bits(),
        # ONE bank here, not two. A shadow-and-live pair in the register
        # file existed to freeze values while a frame boundary crossed a
        # full pipeline; the datapath now keeps each block's copy beside the
        # block and loads them all at the instant the pipe is empty, so
        # there is nothing left to freeze. Keeping the second bank would be
        # a third register for one number, and two of them would be
        # describing the same fact.
        frame_sync=False,
        bus_clock=True,
        # (register name in the map, then the two datapath ports)
        arm=(f"{vocabulary(design).block.name}_{vocabulary(design).commit}",
             "cfg_arm_req", "cfg_arm_ack"),
        header=header(
            design, what=f"{design.name} -- AXI4-Lite control register file "
                         f"and the {design.name} datapath",
            source="the register map np2hw.pipeline allocated",
        ),
        notes=_control_notes(),
    )


def _control_notes() -> list[str]:
    """What a reviewer needs to know before reading the decode."""
    return [
        "// Every register below sits where the allocator put it -- one",
        f"// {BLOCK_ALIGN}-byte-aligned region per block INSTANCE, an identity word at",
        "// each region's base -- so address decode is a bit-slice compare and",
        "// the emitted JSON register map describes exactly this file.",
        "//",
        "// Writes land in a shadow register and are committed to the live value",
        "// at the frame boundary, so a frame is processed with one coherent set",
        "// of coefficients. A write during frame N therefore takes effect on",
        "// frame N+1; software that needs it sooner has to wait for the boundary,",
        "// not race it.",
        "//",
        "// A write to a read-only word, or to an unmapped address, is answered",
        "// SLVERR rather than silently dropped.",
    ]


class Handed(NamedTuple):
    """The stream a block hands on, as its trace published it."""

    channels: int
    bit_depth: int
    width: int
    height: int


def handed_on(result) -> Handed:
    """The stream a generated block hands on: the channels and depth its
    model's arithmetic packed and the frame it emits, as its trace published
    them. The one reading of a block's output, for the composer and for a
    run of the models alike.

    np2hw packs a multi-channel word as equal fields, so the per-component
    depth is the word divided by its components -- and it is exact, because
    a datapath bound stated in bits can only produce ``2**bits - 1``."""
    out = result.core["interface"].get("output") or {}
    if "cols" not in out or "rows" not in out:
        raise ValueError(
            f"{result.top} publishes no output geometry; what it hands on "
            "is the next block's frame, and it is not guessed")
    channels = int(out.get("channels", 1))
    return Handed(channels, int(result.core.out_bits) // max(1, channels),
                  int(out["cols"]), int(out["rows"]))


def _incoming(design, s: Stage, built: dict) -> Handed:
    """The stream arriving at ``s``'s input.

    Read from the DRIVER's trace -- what its model's arithmetic packed, and
    the frame it emits -- or from the design's input when the driver is the
    outside world. The datapath walk is topological, so a driver is always
    generated before its consumer asks about it.

    A block is told the stream it is GIVEN, not the design's input: a stage
    that saturates back hands on what it received, a stage that narrows
    after a tone curve hands on less, a stage that widens hands on more, and
    a stage that changes size hands on its own frame."""
    form = shape(design)
    entering = Handed(form.spec.channels, form.spec.bit_depth,
                      form.width, form.height)
    if not s.block.ports.inputs:
        return entering
    source = netlist(design).driver(Endpoint(s.path, s.block.ports.inputs[0]))
    if source is None or source.node is None:
        return entering
    result = built.get(source.node)
    if result is None:
        return entering
    return handed_on(result)


def _address_notes(design) -> list[str]:
    """The address map and netlist, as comments a reviewer reads first."""
    graph = netlist(design)
    lines = ["// Address map (see the emitted JSON for the authoritative version):"]
    for s in stages(design):
        lines.append(f"//   0x{s.instance.base:04x}  {s.path:<24} "
                     f"{s.paramset.block} v{s.paramset.version[0]}."
                     f"{s.paramset.version[1]}")
    lines.append("//")
    lines.append("// Netlist:")
    for source, sink in graph.edges:
        tap = "  (tap: sink never stalls)" if graph.is_tap(sink) else ""
        lines.append(f"//   {str(source):<28} -> {str(sink)}{tap}")
    return lines


def _ports(design, built) -> list:
    """Top-level ports: context, per-instance CSRs, then the pixel streams."""
    from np2hw import Port, StreamType

    ports = [
        Port(f"ctx_{ctx.name}", "in", width=ctx.bits,
             comment=f"Pipeline context, owned by "
                     f"`{vocabulary(design).block.name}` at base 0 and fanned "
                     f"out as a wire -- never duplicated per block. "
                     f"{ctx.description}")
        for ctx in contexts(design)
    ]
    # The coefficients are written from another clock -- see the context
    # block's commit register. These two carry the whole crossing: the file says its
    # values are frozen, the datapath says it has taken them. One bit each
    # way; the values themselves never cross.
    ports.append(Port("cfg_arm_req", "in",
                      comment="the register file's values are frozen"))
    ports.append(Port("cfg_arm_ack", "out",
                      comment="the copy has been taken"))
    for path, result in built.items():
        s = stage(design, path)
        for name, bits in result.params:
            param = s.paramset.param(name)
            ports.append(Port(
                f"param_{s.module_prefix}_{name}", "in", width=bits,
                signed=param.signed,
                comment=f"{path}.{name} @ "
                        f"0x{s.instance.address_of(name):04x} "
                        f"({param.q_format}). {param.description}"))
    # The INPUT boundary is the one declared stream fact in a design: its
    # width is the shape's, because the shape IS the input declaration and
    # the first block was traced against it -- one owner, two readers. The
    # OUTPUT boundary is the opposite: entirely traced.
    entry = StreamType(shape(design).spec.data_bits, ("sof", "eol", "last"))
    ports += [Port(name, "in", stream=entry) for name in design.inputs]
    ports += [Port(name, "out", stream=_output_stream(design, built, name))
              for name in design.outputs]
    return ports


def _output_stream(design, built, name):
    """The stream an OUTPUT port carries: the driving block's traced word.

    Width comes from what the driver's model actually packed -- a 3-channel
    word is three fields wide because the trace says so, not because
    anything declared it."""
    from np2hw import StreamType

    for source, sink in netlist(design).edges:
        if sink.node is None and sink.port == name and source.node:
            result = built.get(source.node)
            if result is not None:
                return StreamType(int(result.core.out_bits),
                                  ("sof", "eol", "last"))
    return StreamType(shape(design).spec.data_bits, ("sof", "eol", "last"))


def _output_boundary(design, built, name) -> dict:
    """What an output port carries, in the terms an integrator needs.

    The total width alone is not enough to slice: a 36-bit word is three
    12-bit channels here, but that is a TRACED fact, not a law that the width
    can be divided by three to recover. So the channel count is published
    beside the width, and whoever unpacks the word reads both rather than
    dividing and hoping."""
    channels = 1
    for source, sink in netlist(design).edges:
        if sink.node is None and sink.port == name and source.node:
            result = built.get(source.node)
            if result is not None:
                channels = int((result.core["interface"].get("output") or {})
                               .get("channels", 1))
            break
    data_bits = _output_stream(design, built, name).data_bits
    out = {"data_bits": data_bits, "channels": channels}
    if channels and data_bits % channels == 0:
        out["channel_bits"] = data_bits // channels
    return out


def _instances(design, built) -> list:
    """One np2hw Instance per generated block, with its parameter bindings."""
    from np2hw import Instance

    instances = []
    for path, result in built.items():
        s = stage(design, path)
        declared = [name for name, _, _ in result.core.interface["params"]]
        bind = {}
        for name in declared:
            if name in result.context_map:
                bind[name] = (result.context_map[name],
                              "pipeline context, not a register")
            else:
                bind[name] = f"param_{s.module_prefix}_{name}"
        instances.append(Instance(
            name=s.module_prefix, core=result.core, bind=bind,
            comment=f"{path} (base 0x{s.instance.base:04x})"))
    return instances


def _connections(design, built) -> list:
    """Netlist edges, restricted to instances that were actually generated.

    A block that is declared but not yet built -- `stats`, whose accumulation
    np2hw cannot trace -- simply has no instance to wire to, so its edges are
    dropped. Its source's `ready` then ties high on its own, which is the
    right answer anyway: it was a tap."""
    from np2hw import Connection

    def endpoint(reference):
        if reference.node is None:
            return reference.port
        if reference.node not in built:
            return None
        return f"{stage(design, reference.node).module_prefix}.{reference.port}"

    edges = []
    for source, sink in netlist(design).edges:
        names = (endpoint(source), endpoint(sink))
        if None in names:
            continue
        edges.append(Connection(names[0], names[1]))
    return edges


# --------------------------------------------------------------------------- #
# Documentation, generated from the same map the host API reads
# --------------------------------------------------------------------------- #

def register_map_markdown(mapping: dict) -> str:
    """Render a register map as documentation.

    Generated from the emitted JSON rather than written alongside it, for the
    same reason the host API is: a hand-written register map is correct on the
    day it is written and slowly stops being correct afterwards, and the errors
    it acquires are exactly the ones nobody notices until hardware misbehaves.
    """
    out: list[str] = []
    a = out.append

    a(f"# {mapping['name']} register map")
    a("")
    a("Generated from the emitted register map JSON by "
      "`np2hw.pipeline.compose.register_map_markdown`. Do not edit by hand.")
    a("")
    geometry, stream = mapping["geometry"], mapping["stream"]
    a(f"- Built for {geometry['width']}x{geometry['height']}, "
      f"{stream['bit_depth']}-bit, {stream['channels']}-channel "
      f"({stream['data_bits']}-bit data bus)")
    a(f"- {mapping['address_bits']}-bit address space, "
      f"blocks aligned to {mapping['block_align']} bytes")
    a(f"- Licence: {mapping['licence']}")
    a("")
    a("Every block starts on a power-of-two boundary, so address decode is a "
      "bit-slice compare rather than a range comparator. Nothing outside this "
      "file should hold an address.")
    a("")

    control = mapping.get("control")
    if control:
        a("## Control interface")
        a("")
        a(f"- {control['bus']}, {control['data_bits']}-bit data, "
          f"{control['address_bits']}-bit address")
        a(f"- Commit: {control['commit']}, via {control['commit_register']}. "
          f"{control['commit_description']}.")
        a(f"- Caveat: {control['commit_caveat']}.")
        a(f"- {control['error_response']}.")
        a("- Read the `id_version` word of a block before writing anything to it: "
          "it is how software proves the bitstream matches this map.")
        a("")

    a("## Layout")
    a("")
    a("| Base | Instance | Block | Version | Size |")
    a("| --- | --- | --- | --- | --- |")
    for block in mapping["blocks"]:
        a(f"| `0x{block['base']:04x}` | `{block['path']}` | {block['block']} "
          f"| {block['version']} | {block['size']} B |")
    a("")

    for block in mapping["blocks"]:
        a(f"## `{block['path']}` — {block['block']} v{block['version']}")
        a("")
        if block["description"]:
            a(block["description"])
            a("")
        a(f"Base `0x{block['base']:04x}`, block ID `0x{block['id']:04x}`.")
        a("")
        if not block.get("implemented", True):
            a(f"> **Declared, not built yet:** {block['not_implemented_reason']}. "
              "The registers below are decoded and read back, and reach no "
              "datapath. Writing them has no effect on the image.")
            a("")
        if block["consumes"]:
            a(f"Consumes pipeline context, as wires from `pipe`: "
              + ", ".join(f"`{c}`" for c in block["consumes"]) + ".")
            a("")

        identity = block["id_version"]
        a(f"| Address | Offset | Register | Format | Reset | Access | Description |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        a(f"| `0x{identity['address']:04x}` | `+0x{identity['offset']:02x}` "
          f"| `id_version` | u32.0 | `0x{identity['value']:08x}` | ro "
          f"| {identity['description']} |")
        for register in block["registers"]:
            a(f"| `0x{register['address']:04x}` | `+0x{register['offset']:02x}` "
              f"| `{register['name']}` | {register['q_format']} "
              f"| {register['default']} | {register['access']} "
              f"| {register['description']} |")
        a("")

        for window in block["statistics"]:
            a(f"### `{window['name']}` statistics window")
            a("")
            a(window["description"])
            a("")
            a(f"- Base `0x{window['base']:04x}`, {window['size_bytes']} bytes "
              f"({window['buffering']}-buffered, {window['buffer_bytes']} bytes "
              f"per buffer)")
            a(f"- {window['records']} records of {window['record_words']} "
              f"word(s): " + ", ".join(f"`{f}`" for f in window["layout"]))
            a(f"- Access: {window['access']}. Read in bulk; the completed frame "
              "is in buffer 0 while the next accumulates in buffer 1.")
            a("")

    return "\n".join(out) + "\n"
