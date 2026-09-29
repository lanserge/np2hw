# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Running passes: each takes a unit and returns one, verified after it.

A UNIT is what lowering carries from pass to pass: the graph, the values
that matter (its roots), and what the passes have established about it.
The runner verifies the graph after every pass -- a malformed graph is
caught at the pass that made it, not three passes later in Verilog -- and
can dump it after any pass, so each step can be read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .graph import Graph, Value, dump, verify


@dataclass
class Unit:
    g: Graph
    roots: dict                       # name -> Value
    facts: dict = field(default_factory=dict)
    regs: list = field(default_factory=list)     # hw.reg declarations

    def all_roots(self) -> list[Value]:
        return list(self.roots.values()) + [v for r in self.regs
                                             for v in r.operands]


@dataclass
class Pass:
    name: str
    run: callable                     # Unit -> Unit
    levels: set | None = None         # the legality rule after the pass


def run(unit: Unit, passes: list[Pass], dump_after: set | None = None) -> Unit:
    """Run `passes` in order, verifying after each. NP2HW_DUMP_AFTER, a
    comma-separated list of pass names, prints the graph after those."""
    dump_after = dump_after or set(
        filter(None, os.environ.get("NP2HW_DUMP_AFTER", "").split(",")))
    for p in passes:
        unit = p.run(unit)
        verify(unit.g, unit.all_roots(), p.levels)
        if p.name in dump_after:
            print(f"--- after {p.name} ---")
            print(dump_unit(unit))
    return unit


def dump_unit(unit: Unit) -> str:
    """A unit as text: its nodes, then each register by name -- next value,
    enable, reset -- then each root by name."""
    text = dump(unit.g, unit.all_roots()).splitlines()
    # value numbers as the dump assigned them, in walk order
    from .graph import walk
    num = {id(n): f"%{i}" for i, n in enumerate(walk(unit.all_roots()))}
    ref = lambda v: num[id(v.node)] + ("" if len(v.node.results) == 1
                                        else f"#{v.index}")
    out = [l for l in text if not l.startswith("return ")]
    for r in unit.regs:
        d, en = r.operands
        init = unit.g.symbols[r.attr("name")]["init"]
        out.append(f"reg {r.attr('name')} <- {ref(d)} if {ref(en)}"
                   f"  (reset {'none' if init is None else init})")
    out += [f"out {name} = {ref(v)}" for name, v in unit.roots.items()]
    return "\n".join(out)
