# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Where a design's bus clocks and resets come from, read from its netlist.

The invariant, learned one power cycle at a time on a board:

    ANYTHING A HOST CAN REACH MUST ANSWER, and that means every clock
    AND every reset of every memory-mapped bus port belongs to the BOARD
    -- never to a cable, whose clock and resets come and go with it.

A slave on a clock recovered from a cable hangs the processor that reads
it; an interconnect held in a reset the cable drives hangs it the same
way. So every end of every memory-mapped bus is traced, its clock and its
reset, back through the design's nets to what generates them.

Nothing here is matched by name. Each leaf states its own part:

``clocks``    ``{bus port: [clock port, reset port]}`` -- what a bus port
              is timed and reset by;
``ancestry``  ``{output: "board" | "cable" | [inputs]}`` -- an output the
              leaf generates on the board (a processor's clock, a
              constant, a register software writes), one it recovers from
              a cable, or one that follows inputs (a clock multiplier, an
              inverter, a reset synchroniser asserted by any of its
              sources): of the board only if every input is.

Both are a leaf's ``facts``. Whatever is not stated is UNKNOWN and fails:
an audit that shrugs at what it cannot classify passes the next new
failure.
"""
from __future__ import annotations

import sys

BOARD, CABLE, UNKNOWN = "board", "CABLE", "UNKNOWN"


def _facts(design, path) -> dict:
    inst = design.instances.get(path)
    leaf = inst.leaf if inst is not None else None
    return leaf.get("facts", {}) if isinstance(leaf, dict) else {}


def audit(design) -> list[str]:
    """Every problem with the design's bus clocks and resets, as text;
    an empty list where every one is the board's."""
    netlist = design.netlist
    drivers = {}
    for net in netlist["nets"]:
        for pin in net["pins"]:
            drivers[pin] = net["drivers"]

    def origin(pin, seen=frozenset()):
        """BOARD, CABLE or UNKNOWN, for the signal at ``path.port``."""
        if pin in seen:
            return UNKNOWN
        found = drivers.get(pin, [])
        if len(found) != 1:
            return UNKNOWN               # undriven, or a pin of the FPGA
        path, _, port = found[0].rpartition(".")
        stated = _facts(design, path).get("ancestry", {}).get(port)
        if stated == "board":
            return BOARD
        if stated == "cable":
            return CABLE
        if isinstance(stated, list) and stated:
            kids = {origin(f"{path}.{i}", seen | {pin}) for i in stated}
            if CABLE in kids:
                return CABLE
            return BOARD if kids == {BOARD} else UNKNOWN
        return UNKNOWN

    def kind(path, port):
        bundles = (netlist["instances"].get(path, {}).get("bundles") or {})
        return bundles.get(port, {}).get("kind")

    ends = []
    for b in netlist["bundles"]:
        a, z = (e.rpartition(".")[::2] for e in (b["from"], b["to"]))
        if "aximm" in (kind(*a), kind(*z)):
            ends += [a, z]

    problems, seen = [], set()
    for path, port in ends:
        if (path, port) in seen:
            continue
        seen.add((path, port))
        stated = _facts(design, path).get("clocks", {}).get(port)
        if stated is None:
            problems.append(f"{path}.{port}: a bus port whose clock and "
                            "reset its leaf does not state (facts.clocks)")
            continue
        for role, signal in zip(("clock", "reset"), stated):
            lineage = origin(f"{path}.{signal}")
            if lineage != BOARD:
                problems.append(
                    f"{path}.{port}: its {role} {signal} is {lineage} -- a "
                    "cable event would stop it answering")
    return sorted(problems)


def main(argv=None) -> int:
    """``python -m np2hw.design.ancestry <design ref> [name=value ...]``"""
    from np2hw.design.composite import load
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m np2hw.design.ancestry <design> "
              "[parameter=value ...]")
        return 1
    build = dict(a.split("=", 1) for a in args[1:])
    design = load(args[0], build={k: int(v) for k, v in build.items()})
    problems = audit(design)
    for problem in problems:
        print(f"  {problem}")
    print(f"  {design.ref}: bus clocks and resets "
          + ("all the board's" if not problems
             else f"-- {len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
