# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Every wire of a loaded design, checked against the leaves' own ports.

A leaf's ports are stated in the hierarchy: a generated leaf publishes the
list its module was written from, a hand-written one declares it in its
library entry (and is held to its Verilog by the tests). So a design is
checked here, once, for every toolchain -- never by reading HDL back, and
never in an emitter, where a check is a property of one tool's output.

What is refused is what a block design would quietly get wrong:

  - a stream wired to a bus, or a bus to a bus of another kind;
  - a stream, or an AXI4-Stream bus, whose data widths differ at its two
    ends -- a block design connects the low bits and calls it a warning,
    so the symptom is a picture rather than an error;
  - a net whose pins are of different widths, the same failure one wire
    at a time;
  - a net two of our leaves drive.

A leaf with no port facts -- vendor IP, whose ports are its tool's, or an
ISP block, whose streams np2hw's composer checks -- is not checked here.
That a stream or a bus is wired on its own side, and that a port exists,
is the loader's, at the connection that names it.
"""
from __future__ import annotations

from np2hw.design.composite import CompositionError
from np2hw.design.interfaces import buses, expand


class WiringError(CompositionError):
    """A design whose wires disagree with the leaves they join."""


_KINDS = {"np2hw_stream": "stream", ":axis:": "axis", ":aximm:": "aximm",
          ":vid_io:": "vid_io"}
# a declared bus's kind, in the same words as a published one's
_DECLARED = {"axilite": "aximm", "axis": "axis", "tmds": "tmds",
             "video": "vid_io", "axi3": "aximm", "axi4": "aximm",
             "pins": "pins"}


def _kind(vlnv: str) -> str:
    for marker, kind in _KINDS.items():
        if marker in vlnv:
            return kind
    return vlnv


def role(bundle: str, member: str) -> str:
    """A declared bundle member's part in its bus: its signal, as the bus's
    tables and a published port's logical name both say it, in lower case.
    Two bundles are wired member to member by this, never by name."""
    return (member[len(bundle) + 1:] if member.startswith(bundle + "_")
            else member).lower()


def facts(inst) -> dict | None:
    """``{"signals": {name: (direction, width)}, "bundles": {name: {kind,
    side, width, members, roles}}}`` for one leaf, at this instance's
    parameters -- a bundle's members its signals, by name, and each one's
    part in the bus (:func:`role`) -- or None
    where it states no ports."""
    obj = inst.leaf if isinstance(inst.leaf, dict) else None
    if obj is None:
        return None
    if "ports" in obj:
        from np2hw.ports import resolve
        params = obj.get("module_parameters", {})
        signals = {p["name"]: (p["direction"], resolve(p["width"], params))
                   for p in obj["ports"]}
        bundles = {}
        for name, bus in buses(obj["ports"]).items():
            kind = _kind(bus["ports"][0]["bus"]["vlnv"])
            data = (signals.get(f"{name}_data") if kind == "stream" else
                    signals.get(f"{name}_tdata") if kind == "axis" else None)
            bundles[name] = {"kind": kind, "side": bus["side"],
                             "width": data[1] if data else None,
                             "members": [p["name"] for p in bus["ports"]],
                             "roles": {p["name"]: p["bus"]["logical"].lower()
                                       for p in bus["ports"]}}
        return {"signals": signals, "bundles": bundles}
    if not ("inputs" in obj or "outputs" in obj):
        return None
    from np2hw.design.composite import evaluate, parse
    signals, bundles = {}, {}
    for side in ("inputs", "outputs"):
        for port in obj.get(side, []):
            raw = port.get("width", 1)
            if isinstance(raw, int):
                width = raw
            else:
                # a block filled by an implementation reads the block's
                # parameters, spent on the implementation, not its build
                params = {**obj.get("block_parameters", {}), **inst.build}

                def lookup(name, port=port, params=params):
                    if name not in params:
                        raise WiringError(
                            f"{inst.path}: port {port['name']}'s width reads "
                            f"{name}, which this instance never sets")
                    return params[name]
                width = evaluate(parse(raw, f"{inst.path}.{port['name']}"),
                                 lookup)
            members = expand(port, side, width)
            signals.update(members)
            roles = {m: role(port["name"], m) for m in members}
            if "stream" in port:
                bundles[port["name"]] = {"kind": "stream", "side": side,
                                         "width": width,
                                         "members": list(members),
                                         "roles": roles}
            elif "bus" in port:
                axis = port["bus"] == "axis"
                bundles[port["name"]] = {
                    "kind": _DECLARED[port["bus"]], "side": side,
                    "width": width if axis else None,
                    "members": list(members), "roles": roles}
    return {"signals": signals, "bundles": bundles}


def check(design) -> None:
    """Refuse a design whose wires disagree with the ports they join, read
    from its netlist (np2hw.design.netlist): a bundle wired to one of
    another kind or width, or with an input the other has no partner to
    drive; a net whose pins differ in width (an address excepted: each end
    uses the bits it decodes); a net two leaves drive."""
    netlist = getattr(design, "netlist", None)
    if netlist is None:
        from np2hw.design.netlist import build
        netlist = build(design)
    instances = netlist["instances"]

    def bundle(pin):
        path, _, port = pin.rpartition(".")
        if pin.startswith("pin:") or path not in instances:
            return None
        return (instances[path]["bundles"] or {}).get(port)

    problems = []
    for b in netlist["bundles"]:
        bs, bd = bundle(b["from"]), bundle(b["to"])
        if bs and bd:
            if bs["kind"] != bd["kind"]:
                problems.append(f"{b['from']} ({bs['kind']}) -> {b['to']} "
                                f"({bd['kind']}): not one kind of bus")
            elif (bs["width"] is not None and bd["width"] is not None
                  and bs["width"] != bd["width"]):
                problems.append(f"{b['from']} ({bs['width']} bits) -> "
                                f"{b['to']} ({bd['width']} bits): WIDTH "
                                "MISMATCH")
            elif "unpaired" in b:
                problems.append(
                    f"{b['from']} -> {b['to']}: inputs nothing drives -- "
                    + "; ".join(f"{', '.join(roles)} at {b[end]}"
                                for end, roles in b["unpaired"].items()
                                if roles))
    for net in netlist["nets"]:
        widths = net["widths"]
        if len(set(widths.values())) > 1 and not net.get("ranged"):
            problems.append("WIDTH MISMATCH on one net: " + ", ".join(
                f"{p} ({w} bits)" for p, w in sorted(widths.items())))
        if len(net["drivers"]) > 1:
            problems.append(f"one net driven by {sorted(net['drivers'])}")
    if problems:
        raise WiringError(f"{netlist['design']}: {len(problems)} problem(s):"
                          "\n  " + "\n  ".join(problems))
