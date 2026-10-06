# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A loaded design as a netlist: the product of elaboration.

Every leaf with its implementation, its settings and its ports as this
build resolves them; every connection, either a bundle -- a stream or a
bus, joined by its name -- or a net of pins; the board's external pins.
It is built once, from the design the loader made, and everything after
it reads it: the wiring check, the toolchain's script, a testbench, a
tool that only wants to look. Written out as ``netlist.json``.

A leaf whose ports nothing states -- a procedure without a description,
or a block the loader did not build here -- is in the netlist with
``ports`` None: its connections are still there, unchecked.
"""
from __future__ import annotations

from np2hw.design.wiring import facts


def _split(endpoint: str):
    if endpoint.startswith("pin:") or "." not in endpoint:
        return None, endpoint.removeprefix("pin:")
    path, _, port = endpoint.rpartition(".")
    return path, port


def _implementation(leaf) -> dict:
    """What fills a leaf, as a toolchain needs to know it."""
    if not isinstance(leaf, dict):
        return {"model": True}
    out = {k: v for k, v in leaf.get("vivado", {}).items()
           if k in ("ip", "module", "tcl")}
    if "filled_by" in leaf:
        out["implementation"] = leaf["filled_by"]["implementation"]
        if leaf.get("port_map"):
            out["port_map"] = dict(leaf["port_map"])
        if leaf.get("tie"):
            out["tie"] = dict(leaf["tie"])
    sources = list(leaf.get("sources", [])) + list(leaf.get("generated", []))
    if sources:
        out["sources"] = sources
    return out


def build(design) -> dict:
    """The design as a netlist: ``{"design", "parameters", "instances",
    "bundles", "nets", "pins"}``."""
    known = {path: facts(inst) for path, inst in design.instances.items()}
    instances = {}
    for path, inst in design.instances.items():
        leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
        f = known[path]
        instances[path] = {
            "block": inst.ref,
            "kind": inst.kind,
            "implementation": _implementation(inst.leaf),
            "parameters": {**leaf.get("module_parameters", {}), **inst.build},
            "config": dict(inst.config),
            "ports": ({name: list(v) for name, v in f["signals"].items()}
                      if f else None),
            "bundles": (dict(f["bundles"]) if f else None),
        }

    bundles, nets, member = [], [], {}

    def pin(path, port):
        return f"{path}.{port}" if path is not None else f"pin:{port}"

    def join(a, b):
        ia, ib = member.get(a), member.get(b)
        if ia is None and ib is None:
            member[a] = member[b] = len(nets)
            nets.append([a, b])
        elif ib is None:
            nets[ia].append(b)
            member[b] = ia
        elif ia is None:
            nets[ib].append(a)
            member[a] = ib
        elif ia != ib:
            nets[ia] += [x for x in nets[ib] if x not in nets[ia]]
            for x in nets[ib]:
                member[x] = ia
            nets[ib] = []

    for source, sink in design.edges:
        (sp, sport), (dp, dport) = _split(source), _split(sink)
        fs, fd = known.get(sp), known.get(dp)
        if ((fs and sport in fs["bundles"]) or (fd and dport in fd["bundles"])):
            bundles.append({"from": pin(sp, sport), "to": pin(dp, dport)})
            continue
        join(pin(sp, sport), pin(dp, dport))

    out_nets = []
    for members in nets:
        if not members:
            continue
        widths, drivers = {}, []
        for p in members:
            path, port = _split(p)
            f = known.get(path)
            if f is None or port not in f["signals"]:
                continue                 # a top-level pin, or unstated
            direction, width = f["signals"][port]
            widths[p] = width
            if direction == "output":
                drivers.append(p)
        out_nets.append({"pins": members, "widths": widths,
                         "drivers": drivers})
    return {"design": design.ref, "parameters": dict(design.parameters),
            "instances": instances, "bundles": bundles, "nets": out_nets,
            "pins": dict(design.pins)}
