# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A register bus: the AXI4-Lite decoder a design places, and where its
slaves sit.

The decoding is a fixed module, ``np2hw/hdl/np2hw_axil_decoder.v``, over
packed ports. What varies with a design is how many slaves it has and
how big each one's window is, so that is all :func:`axil_decoder`
generates: a wrapper giving each slave its own named interface
(``m00_axi``, ``m01_axi``, ...) and the core its windows, placed by
:func:`allocate` in the region the processor's bus reaches. The placement
is published as the decoder's fact ``windows`` -- {its port: [base,
size]}, the form any leaf states where the slave on one of its ports sits
-- and a toolchain and a build's address map read it from there; nothing
else states an address.
"""
from __future__ import annotations

import zlib

from np2hw.design.interfaces import expand
from np2hw.ports import Pin, header, published

AXIMM = "xilinx.com:interface:aximm:1.0"
AW = 32


def _bus(bundle: str, side: str) -> list[Pin]:
    pins = []
    for name, (direction, width) in expand(
            {"name": bundle, "bus": "axilite"}, side, AW).items():
        signal = name[len(bundle) + 1:]
        pins.append(Pin(name, direction, width,
                        bus=(AXIMM, bundle, signal.upper())))
    return pins


def windows(bases, sizes) -> list[tuple[int, int]]:
    """``[(base, size)]``, each a power of two aligned to itself and no two
    overlapping -- what a bit-slice decode can serve, refused otherwise."""
    if len(bases) != len(sizes) or not bases:
        raise ValueError(f"{len(bases)} bases and {len(sizes)} sizes; a "
                         "decoder needs one window per slave, and a slave")
    out = []
    for base, size in zip(bases, sizes):
        base, size = int(base), int(size)
        if size <= 0 or size & (size - 1):
            raise ValueError(f"window size {size:#x} is not a power of two")
        if base % size:
            raise ValueError(f"window {base:#x} is not aligned to its "
                             f"size {size:#x}")
        if base + size > 1 << AW:
            raise ValueError(f"window {base:#x}+{size:#x} is past {AW} bits")
        out.append((base, size))
    spans = sorted(out)
    for (a, sa), (b, _) in zip(spans, spans[1:]):
        if a + sa > b:
            raise ValueError(f"windows {a:#x}+{sa:#x} and {b:#x} overlap")
    return out


def allocate(sizes, base: int, span: int) -> list[int]:
    """Each window's base in the region ``[base, base + span)``: the
    largest placed first, each aligned to its own size, so every window is
    a bit-slice decode and none overlaps. Deterministic: the same sizes in
    the same region always land at the same addresses."""
    order = sorted(range(len(sizes)), key=lambda i: (-int(sizes[i]), i))
    bases, cursor = [0] * len(sizes), int(base)
    for i in order:
        size = int(sizes[i])
        cursor = (cursor + size - 1) // size * size
        bases[i] = cursor
        cursor += size
    if cursor > base + span:
        raise ValueError(f"{sum(map(int, sizes)):#x} bytes of windows do "
                         f"not fit the {span:#x}-byte region at {base:#x}")
    return bases


def axil_decoder(sizes, base, span) -> dict:
    """An AXI4-Lite decoder, one master onto a slave per window.

    A leaf a design names (``np2hw.bus:axil_decoder``) with each slave's
    window size, in the order its ``mNN_axi`` ports take them, and the
    region they are placed in (``base``, ``span``): the processor's bus
    reach, a fact of the processor. A size of 0 is a port with no slave --
    a design's optional one, absent in this build -- and is not made, so
    the other ports keep their numbers. An address in no window is
    answered DECERR. Every port is published, and where each window landed
    is published as the fact ``windows``, ``{port: [base, size]}``.
    """
    sizes = [int(s) for s in sizes]
    present = [i for i, s in enumerate(sizes) if s]
    placed = dict(zip(present, allocate([sizes[i] for i in present],
                                        base, span)))
    wins = windows([placed[i] for i in present], [sizes[i] for i in present])
    n = len(wins)
    tag = zlib.crc32(repr((present, wins)).encode()) & 0xFFFFFF
    module = f"np2hw_axil_decode_{n}_{tag:06x}"
    slaves = [f"m{i:02d}_axi" for i in present]
    clock = Pin("aclk", "input", attribute=(
        '(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", '
        'X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF '
        + ":".join(["s_axi"] + slaves) + ', ASSOCIATED_RESET aresetn" *)'))
    reset = Pin("aresetn", "input", attribute=(
        '(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", '
        'X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)'))
    pins = [clock, reset, *_bus("s_axi", "inputs")]
    for slave in slaves:
        pins += _bus(slave, "outputs")

    L = ["// Copyright 2026 Serge Rabyking",
         "// SPDX-License-Identifier: MIT",
         "//",
         f"// {module} -- np2hw_axil_decoder, its slaves by name.",
         "// Generated by np2hw.bus.axil_decoder; the windows:"]
    L += [f"//   {s}  {b:#010x} + {sz:#x}"
          for s, (b, sz) in zip(slaves, wins)]
    L += header(module, pins)
    signals = list(expand({"name": "m", "bus": "axilite"}, "outputs", AW))
    L.append("    np2hw_axil_decoder #(")
    L.append(f"        .N({n}), .AW({AW}),")
    L.append("        .BASE({" + ", ".join(
        f"32'h{b:08x}" for b, _ in reversed(wins)) + "}),")
    L.append("        .MASK({" + ", ".join(
        f"32'h{(~(sz - 1)) & 0xFFFFFFFF:08x}" for _, sz in reversed(wins))
        + "})")
    L.append("    ) core (")
    L.append("        .aclk(aclk), .aresetn(aresetn),")
    for name in expand({"name": "s", "bus": "axilite"}, "inputs", AW):
        L.append(f"        .{name}(s_axi_{name[2:]}),")
    for k, name in enumerate(signals):
        signal = name[2:]
        # slave i at [i*W +: W]: the concatenation lists the last first
        parts = ", ".join(f"{s}_{signal}" for s in reversed(slaves))
        end = ")" if k == len(signals) - 1 else "),"
        L.append(f"        .m_{signal}({{{parts}}}{end}")
    L.append("    );")
    L.append("endmodule")
    return {"verilog": "\n".join(L) + "\n", "module": module,
            "requires": ("np2hw_axil_decoder",),
            "interface": {"ports": published(pins)},
            "facts": {"windows": {f"m{i:02d}_axi": [placed[i], sizes[i]]
                                  for i in present},
                      "clocks": {bus: ["aclk", "aresetn"]
                                 for bus in ["s_axi"] + slaves}}}


axil_decoder.leaf = True


def hops(design) -> list:
    """Every window a leaf publishes, walked to the slave it reaches:
    ``[(base, size, [((space, port), (slave, port)), ...], decoded)]`` --
    each hop a slave port in the address space of the master port before
    it, and whether an np2hw decoder published the window. A
    block that passes an access on unchanged says so (its fact
    ``forwards``, {slave port: master port}), and the walk goes through it
    to what is behind; the last hop is the slave that decodes the window.
    """
    def far(endpoint):
        ends = [b if a == endpoint else a for a, b in design.edges
                if endpoint in (a, b)]
        if len(ends) != 1:
            raise ValueError(f"{endpoint} has a window and {len(ends)} "
                             "slaves on it; a window reaches one")
        return ends[0].rpartition(".")[::2]

    def facts(path):
        leaf = design.instances[path].leaf
        return leaf.get("facts", {}) if isinstance(leaf, dict) else {}

    # the decoders' windows first: where two masters' walks share a hop,
    # the register bus's placement is the one recorded
    def decoder(path):
        leaf = design.instances[path].leaf
        made = leaf.get("made_by", {}) if isinstance(leaf, dict) else {}
        return made.get("generator") == "np2hw.bus:axil_decoder"

    out = []
    for path in sorted(design.instances, key=lambda p: not decoder(p)):
        for port, (base, size) in facts(path).get("windows", {}).items():
            chain, space = [], (path, port)
            while True:
                slave = far(f"{space[0]}.{space[1]}")
                chain.append((space, slave))
                onward = facts(slave[0]).get("forwards", {}).get(slave[1])
                if onward is None:
                    break
                space = (slave[0], onward)
            out.append((base, size, chain, decoder(path)))
    return out


def address_map(design) -> dict:
    """Where every slave on a register bus sits, as the host reads it:
    ``{slave path: {"base", "size", "port"}}`` -- the slave that decodes
    the window (through any block that only forwards), its window, and its
    port. Read from the windows the design's decoders published and the
    design's own edges; it states nothing of its own. A second master's
    reach (a cable's bridge into one register file, from 0) is its own
    address space, not the host's, and is not in it."""
    out = {}
    for base, size, chain, decoded in hops(design):
        if not decoded:
            continue
        slave, port = chain[-1][1]
        out[slave] = {"base": base, "size": size, "port": port}
    return dict(sorted(out.items(), key=lambda item: item[1]["base"]))
