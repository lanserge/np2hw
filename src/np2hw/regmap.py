"""Hierarchical register maps: block types, instances, and their renderings.

The flat form -- a list of :class:`Reg` at absolute offsets -- is what a register
file EMITTER needs, and it is all the emitters in verilog.py consume. But it is
not what a design IS. A design has block TYPES ("blacklevel": these registers at
these local offsets) and INSTANCES of them at bases ("left.blacklevel" at
0x0100), and flattening early throws that away: two instances of one block read
as unrelated names, and every downstream artefact that wants the structure back
-- a UVM register model, a SystemRDL description, generated documentation -- has
to reconstruct it by guessing at name prefixes.

So the hierarchy is kept, in the same shape compose() already keeps for modules:
a type defined once, instantiated N times. :meth:`AddrMap.flatten` produces the
flat list exactly where an emitter needs it, and nowhere earlier.

Who owns what
-------------

The APPLICATION owns the content: which blocks exist, which registers they have,
where instances sit, and what any of it means. np2hw owns the mechanics: the
dataclasses, the flattening, the validation, and the renderers. Application
concepts that np2hw has no business understanding -- a fixed-point Q format, a
commit policy -- travel in ``properties``, which np2hw carries into its outputs
and never interprets. That is the same arrangement as StreamType.domain, and it
exists for the same reason: the alternative is np2hw growing one application's
vocabulary.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Reg:
    """One control register, at an offset its CALLER chose.

    In a flat list the offset is absolute; inside a :class:`RegBlock` it is
    LOCAL to the block, and :meth:`AddrMap.flatten` adds the instance base. The
    bare `(name, bits)` tuple form the emitters also accept packs registers at
    word offset `i*4`, which is fine for one core and wrong for a design:
    whoever allocates an address owns it, so it is passed in rather than
    re-derived here.

    Args:
        name: register name. A read/write register drives an output port
            `param_<name>`; a read-only one has no port and only reads back.
        bits: width. Must fit one 32-bit bus word.
        offset: BYTE offset. Word-aligned and unique within its container.
            None packs it after the previous register in a flat list.
        signed: two's complement. The output port is declared signed and
            read-back sign-extends to 32 bits, so software reading the word as a
            signed int gets the value it wrote.
        reset: power-on value, as the raw integer the register holds.
        access: ``"rw"`` for configuration, ``"ro"`` for a constant the hardware
            reports -- an ID and version word, say. A write to a read-only
            address is answered SLVERR rather than silently dropped.
        value: the constant a read-only register reads back. Ignored for ``rw``.
        description: carried into the generated Verilog as a comment, because
            somebody reads the RTL in review and a bare `param_offset_0_0` does
            not tell them what to write.
        properties: application-defined facts about this register -- a Q format,
            a commit policy. Opaque to np2hw: carried into renderings that can
            hold them (SystemRDL user-defined properties), never interpreted.
    """

    name: str
    bits: int
    offset: int | None = None
    signed: bool = False
    reset: int = 0
    access: str = "rw"
    value: int = 0
    description: str = ""
    properties: dict = field(default_factory=dict)

    @property
    def writable(self) -> bool:
        return self.access == "rw"

    @property
    def word(self) -> int:
        """Word index this register decodes at."""
        return self.offset >> 2


@dataclass(frozen=True)
class RegBlock:
    """A block TYPE: named registers at LOCAL offsets, declared once.

    The type carries no base address and no instance identity, which is what
    lets one declaration serve every instance -- the same property that makes a
    module reusable. Offsets are local; the instance supplies the base.
    """

    name: str
    regs: tuple = ()
    size: int | None = None          # bytes this block spans; None = derived
    description: str = ""
    properties: dict = field(default_factory=dict)

    def __post_init__(self):
        seen = {}
        for reg in self.regs:
            if reg.offset is None:
                raise ValueError(
                    f"block {self.name!r}: register {reg.name!r} has no offset; "
                    "inside a RegBlock every offset is explicit, because the "
                    "block's layout IS the declaration")
            if reg.offset in seen:
                raise ValueError(
                    f"block {self.name!r}: registers {seen[reg.offset]!r} and "
                    f"{reg.name!r} both claim local offset 0x{reg.offset:x}")
            seen[reg.offset] = reg.name

    @property
    def span(self) -> int:
        """Bytes from the block base to the end of its last register."""
        if self.size is not None:
            return self.size
        return max((r.offset + 4 for r in self.regs), default=0)


@dataclass(frozen=True)
class RegInstance:
    """One instance of a block type, at a base address.

    ``path`` is the instance's identity ("left.blacklevel"): it distinguishes
    the two instances of a stereo pipeline's shared type, keys the flattened
    register names, and is what a host API mirrors.
    """

    path: str
    block: RegBlock
    base: int

    @property
    def flat(self) -> str:
        """The path as a Verilog-safe identifier prefix."""
        return self.path.replace(".", "_")


@dataclass(frozen=True)
class AddrMap:
    """A whole design's register map: block types instantiated at bases.

    This is the structure the emitters flatten and the renderers keep. It is
    deliberately the same shape compose() gives modules -- types defined once,
    instances placed -- so the register side of a design finally matches its
    structural side.
    """

    name: str
    instances: tuple = ()
    data_bits: int = 32
    description: str = ""

    def __post_init__(self):
        placed = []
        for inst in sorted(self.instances, key=lambda i: i.base):
            for prev in placed:
                if inst.base < prev.base + prev.block.span:
                    raise ValueError(
                        f"{self.name}: instance {inst.path!r} at 0x{inst.base:x} "
                        f"overlaps {prev.path!r} "
                        f"(0x{prev.base:x}..0x{prev.base + prev.block.span - 1:x})")
            placed.append(inst)
        paths = [i.path for i in self.instances]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{self.name}: duplicate instance paths")

    def addr_bits(self) -> int:
        """Bits needed to address the highest register."""
        top = max((i.base + i.block.span for i in self.instances), default=1)
        return max(1, (top - 1).bit_length())

    def flatten(self) -> list[Reg]:
        """The flat register list an emitter consumes.

        Names are prefixed with the instance path (`left.blacklevel` +
        `offset_0_0` -> `left_blacklevel_offset_0_0`), offsets become absolute,
        and descriptions gain the instance path -- the flat form is where the
        hierarchy is spent, so it is spent completely and in exactly one place.
        """
        out = []
        for inst in self.instances:
            for reg in inst.block.regs:
                out.append(replace(
                    reg,
                    name=f"{inst.flat}_{reg.name}",
                    offset=inst.base + reg.offset,
                    description=(f"{inst.path}.{reg.name} {reg.description}"
                                 if reg.description else f"{inst.path}.{reg.name}"),
                ))
        return out


# --------------------------------------------------------------------------- #
# SystemRDL rendering
# --------------------------------------------------------------------------- #

def _rdl_string(text: str) -> str:
    """One SystemRDL string literal, whitespace-normalised."""
    return '"' + " ".join(str(text).split()).replace('"', '\\"') + '"'


def _property_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "longint unsigned"
    return "string"


def _property_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return _rdl_string(value)


def systemrdl(addrmap: AddrMap, header=()) -> str:
    """Render an :class:`AddrMap` as SystemRDL 2.0 source.

    SystemRDL is the Accellera standard a verification or SoC-integration team
    already has tooling for: from this one file the PeakRDL exporters produce a
    UVM register model, C headers, IP-XACT and browsable documentation. Those
    exporters are separate tools the USER runs -- they are not dependencies of
    np2hw, and their licences attach to them, not to this output.

    The rendering keeps the hierarchy: one `regfile` component per block TYPE,
    instantiated per instance at its base -- the same one-definition-N-uses
    shape the emitted Verilog has. Application-defined register ``properties``
    become SystemRDL user-defined properties, declared once and assigned per
    field, so facts np2hw does not understand (a Q format, say) survive into a
    format whose consumers might.

    Read-only registers render as constants (`sw = r; hw = na;` with a reset),
    matching the emitted decode, where they are literals with no storage.
    """
    lines = list(header)
    a = lines.append
    if header:
        a("")

    # User-defined properties: collected from every register, declared once.
    # Their MEANING belongs to whoever set them; this only carries them.
    declared: dict[str, str] = {}
    for inst in addrmap.instances:
        for reg in inst.block.regs:
            for key, value in reg.properties.items():
                kind = _property_type(value)
                if declared.setdefault(key, kind) != kind:
                    raise ValueError(
                        f"property {key!r} is used with more than one type "
                        f"({declared[key]} and {kind}); a SystemRDL property "
                        "has exactly one")
    for key, kind in sorted(declared.items()):
        a(f"property {key} {{ type = {kind}; component = field; }};")
    if declared:
        a("")

    a(f"addrmap {addrmap.name} {{")
    if addrmap.description:
        a(f"    desc = {_rdl_string(addrmap.description)};")
    a(f"    default regwidth = {addrmap.data_bits};")
    a("")

    # One regfile per TYPE, in first-use order -- exactly one definition, so a
    # reader (or a diff) sees immediately that two instances share a layout.
    types: list[RegBlock] = []
    for inst in addrmap.instances:
        if inst.block not in types:
            types.append(inst.block)

    for block in types:
        a(f"    regfile {block.name}_rf {{")
        if block.description:
            a(f"        desc = {_rdl_string(block.description)};")
        for reg in block.regs:
            a("        reg {")
            if reg.description:
                for line in textwrap.wrap(reg.description, width=68):
                    a(f"            // {line}")
                a(f"            desc = {_rdl_string(reg.description)};")
            mask = (1 << reg.bits) - 1
            if reg.writable:
                access = "sw = rw; hw = r;"
                reset = reg.reset & mask
            else:
                # A constant wired into the read decode: no storage, no write
                # path. `hw = na` says the hardware neither reads nor writes it.
                access = "sw = r; hw = na;"
                reset = reg.value & mask
            a("            field {")
            a(f"                {access}")
            a(f"                reset = 0x{reset:X};")
            for key, value in reg.properties.items():
                a(f"                {key} = {_property_literal(value)};")
            a(f"            }} value[{reg.bits - 1}:0];")
            a(f"        }} {reg.name} @ 0x{reg.offset:04X};")
        a("    };")
        a("")

    for inst in addrmap.instances:
        a(f"    {inst.block.name}_rf {inst.flat} @ 0x{inst.base:04X};")
    a("};")
    return "\n".join(lines) + "\n"
