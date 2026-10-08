# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Hold every described vendor block to what its tool built.

A block filled by a vendor implementation is wired by the ports its
description states, renamed to the implementation's. Nothing in the build
reads the tool's own port list -- the description is the statement -- so
this is the test that the statement is true: the block design, once built,
dumps every such cell's pins (`vendor_ports.txt`, written by the script
`np2hw.toolchain.vivado` emits), and each is compared with what the
description expects (`vendor_expected.json`, written by the block-design
step):

- every pin the description names exists, in the same direction and width;
- every input of the implementation is reached by a port of the block or
  held by the implementation's `tie`;
- an output the block has no port for is left open, which is allowed.

    python3 -m np2hw.toolchain.vivado.vendor_check <build dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from np2hw.design.composite import evaluate, parse
from np2hw.design.interfaces import expand

_DIRECTION = {"input": "I", "output": "O", "inout": "IO"}


def vendor(leaf) -> bool:
    """A leaf a vendor implementation fills: its tool makes the cell, from
    an IP or a procedure."""
    v = leaf.get("vivado", {}) if isinstance(leaf, dict) else {}
    return "filled_by" in leaf and ("ip" in v or "tcl" in v)


def vendor_pins(leaf, where: str = "") -> dict:
    """``{signal: (pin, direction, width)}`` for a leaf a vendor
    implementation fills: each signal of the block's ports -- named as the
    design wires it -- by the implementation's name for it (its
    ``ports`` map, its ``pin_names`` patterns), at the block's parameters.
    The one statement of that mapping: the check here and a writer of the
    design as Verilog both read it."""
    params = leaf.get("block_parameters", {})
    renamed = leaf.get("port_map", {})
    patterns = leaf.get("filled_by", {}).get("chosen", {}).get(
        "pin_names", {})
    out = {}
    for side in ("inputs", "outputs"):
        for port in leaf.get(side, []):
            raw = port.get("width", 1)
            width = raw if isinstance(raw, int) else evaluate(
                parse(raw, f"{where}: {port['name']}"), params.__getitem__)
            for signal, (direction, w) in expand(port, side, width).items():
                pin = renamed.get(signal)
                suffix = signal[len(port["name"]) + 1:]
                if pin is None and port["name"] in patterns:
                    pin = patterns[port["name"]].format(
                        signal=suffix, SIGNAL=suffix.upper())
                elif pin is None and port["name"] in renamed \
                        and signal != port["name"]:
                    pin = f"{renamed[port['name']]}_{suffix}"
                out[signal] = (pin or signal, direction, w)
    return out


def pin_name(leaf, signal: str) -> str:
    """A vendor implementation's name for one signal of its block."""
    return vendor_pins(leaf)[signal][0]


def expected(design, cell: dict) -> dict:
    """``{cell: {"pins": {pin: [dir, width]}, "tie": [...]}}`` for every
    instance a vendor implementation fills, by its cell name."""
    out = {}
    for path, inst in design.instances.items():
        leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
        if not vendor(leaf):
            continue
        pins = {pin: [_DIRECTION[d], w]
                for pin, d, w in vendor_pins(leaf, path).values()}
        out[cell[path]] = {"block": inst.ref, "pins": pins,
                           "tie": sorted(leaf.get("tie", {}))}
    return out


def dumped(text: str) -> dict:
    """The tool's pins per cell, from the block design's dump."""
    cells, here = {}, None
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "CELL":
            here = cells.setdefault(fields[1], {})
        elif fields[0] == "PIN":
            name, direction = fields[1], fields[2]
            left = fields[3] if len(fields) > 3 and fields[3].isdigit() else ""
            right = fields[4] if left and len(fields) > 4 else ""
            width = int(left) - int(right) + 1 if left else 1
            here[name] = [direction, width]
    return cells


def compare(want: dict, got: dict) -> list[str]:
    """Every way a description disagrees with what its tool built."""
    problems = []
    for cell, spec in sorted(want.items()):
        built = got.get(cell)
        where = f"{cell} ({spec['block']})"
        if built is None:
            problems.append(f"{where}: not in the built design")
            continue
        for pin, (direction, width) in sorted(spec["pins"].items()):
            if pin not in built:
                problems.append(f"{where}: describes {pin}, which the "
                                f"implementation does not have")
            elif built[pin] != [direction, width]:
                problems.append(f"{where}: describes {pin} as {direction} "
                                f"{width}, and it is {built[pin][0]} "
                                f"{built[pin][1]}")
        for pin in spec["tie"]:
            if pin in built and built[pin][0] != "I":
                problems.append(f"{where}: ties {pin}, which is not an "
                                "input")
        for pin, (direction, _) in sorted(built.items()):
            if direction in ("I", "IO") and pin not in spec["pins"] \
                    and pin not in spec["tie"]:
                problems.append(f"{where}: {pin} is an input nothing in "
                                "the description reaches; map it to a "
                                "port of the block, or tie it")
    return problems


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    want = json.loads((out / "generated" / "vendor_expected.json").read_text())
    got = dumped((out / "generated" / "vendor_ports.txt").read_text())
    problems = compare(want, got)
    for p in problems:
        print(f"vendor_check: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"vendor_check: {len(want)} described vendor blocks agree with "
          "what the tool built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
