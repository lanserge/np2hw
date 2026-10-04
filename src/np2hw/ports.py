# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A module's ports, declared once: the header is written from the list,
and the list is published beside the Verilog.

A generator's result carries ``interface``, which a composer reads instead
of knowing np2hw's conventions. It named the streams, the clock, the reset
and the parameter ports, and nothing else -- so a receiver's video pins, a
status word or an AXI4-Lite port could only be found by reading the
Verilog back. Whoever writes the ports owns the port list: here it is the
same list that writes them, so the two cannot describe the module
differently.

    pins = [Pin("clk", "input", attribute=clock_attribute(...)),
            Pin("in_data", "input", "W"),
            Pin("status", "output", 16, reg=True, note="sticky")]
    L += header("skid", pins, parameters=[("W", 8)])
    interface["ports"] = published(pins)

A width is a number of bits, or an expression over the module's own
parameters (``"W"``, ``"W + 3"``), which is how a module whose parameters
an instantiating design overrides states what it becomes.

`elaborated_ports` asks a tool for the ports a module REALLY has, at given
parameters -- Verilator's elaborated tree, never a pattern over the text.
It is how a published list, or a hand-written module's declared one, is
held to the Verilog.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DIRECTIONS = ("input", "output", "inout")


@dataclass(frozen=True)
class Pin:
    """One port of a generated module."""
    name: str
    direction: str              # "input" | "output" | "inout"
    width: int | str = 1        # bits, or an expression over the parameters
    reg: bool = False           # declared `reg` (an output driven in always)
    signed: bool = False
    attribute: str | tuple | None = None  # attribute line(s) binding to it
    bus: tuple | None = None    # (vlnv, bundle, logical): the bus it is in
    note: str | tuple = ""      # a trailing comment; a tuple runs on below
    before: tuple = ()          # comment lines emitted above it, without //

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.name}: direction {self.direction!r} is "
                             f"not one of {DIRECTIONS}")
        if isinstance(self.width, int) and self.width < 1:
            raise ValueError(f"{self.name}: width {self.width} must be at "
                             "least 1")

    def range(self) -> str:
        """The declaration's range: empty for one bit."""
        if isinstance(self.width, int):
            return "" if self.width == 1 else f"[{self.width - 1}:0]"
        return f"[{self.width}-1:0]"


def header(module: str, pins, parameters=(), parameter_lines=()) -> list[str]:
    """The module's opening, from `module` to `);`, one port per line.

    `parameters` are ``(name, default)`` pairs; `parameter_lines`, for a
    module whose parameter list is its own layout (typed, grouped,
    commented), the lines between ``#(`` and ``)`` as written. Ranges are
    padded to one column so the names line up, which is the form every
    generator here wrote by hand."""
    pins = list(pins)
    if not pins:
        raise ValueError(f"{module}: a module with no ports")
    names = [p.name for p in pins]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"{module}: ports declared twice: {twice}")
    specs = [(("signed " if p.signed else "") + p.range()).strip()
             for p in pins]
    col = max(len(s) for s in specs)
    L = []
    if parameters and parameter_lines:
        raise ValueError(f"{module}: parameters or parameter_lines, not both")
    if parameter_lines:
        L += [f"module {module} #(", *parameter_lines, ") ("]
    elif parameters:
        L.append(f"module {module} #(")
        for i, (name, default) in enumerate(parameters):
            comma = "," if i < len(parameters) - 1 else ""
            L.append(f"    parameter {name} = {default}{comma}")
        L.append(") (")
    else:
        L.append(f"module {module} (")
    decls = []
    for i, (pin, spec) in enumerate(zip(pins, specs)):
        kind = "reg " if pin.reg else "wire"
        decl = f"    {pin.direction:<6} {kind} "
        decl += f"{spec:<{col}} {pin.name}" if col else pin.name
        decls.append(decl + ("," if i < len(pins) - 1 else ""))
    # Trailing comments line up across each run of neighbouring ports
    # that carry one, the way a person aligns them.
    notes = [(p.note,) if isinstance(p.note, str) else tuple(p.note)
             for p in pins]
    at = [0] * len(pins)
    i = 0
    while i < len(pins):
        if not notes[i][0]:
            i += 1
            continue
        j = i
        while j < len(pins) and notes[j][0]:
            j += 1
        column = max(len(d) for d in decls[i:j]) + 2
        for k in range(i, j):
            at[k] = column
        i = j
    for pin, decl, note, column in zip(pins, decls, notes, at):
        for line in pin.before:
            L.append(f"    // {line}" if line else "    //")
        attrs = ((pin.attribute,) if isinstance(pin.attribute, str)
                 else tuple(pin.attribute or ()))
        L.extend(f"    {line}" for line in attrs)
        if pin.bus:
            L.append(f"    {bus_attribute(*pin.bus)}")
        if note[0]:
            L.append(f"{decl:<{column}}// {note[0]}")
            L.extend(" " * column + f"// {more}" for more in note[1:])
        else:
            L.append(decl)
    L.append(");")
    return L


def published(pins) -> list[dict]:
    """The port list as a result publishes it: name, direction, width --
    a width an expression where the module takes it from a parameter --
    and, for a port that is part of a bus, which bus and which signal."""
    out = []
    for p in pins:
        entry = {"name": p.name, "direction": p.direction, "width": p.width}
        if p.bus:
            vlnv, bundle, logical = p.bus
            entry["bus"] = {"vlnv": vlnv, "bundle": bundle,
                            "logical": logical}
        out.append(entry)
    return out


def bus_attribute(vlnv: str, bundle: str, logical: str) -> str:
    """The attribute binding a port to its bus: which interface definition,
    which bundle of this module, and which of that bundle's signals."""
    return f'(* X_INTERFACE_INFO = "{vlnv} {bundle} {logical}" *)'


def resolve(width, parameters: dict) -> int:
    """A published width at given parameter values. The expression is the
    module's own arithmetic over its parameters: integers, + - * // and
    parentheses, evaluated with nothing else in scope."""
    if isinstance(width, int):
        return width
    import ast
    tree = ast.parse(width, mode="eval").body
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b}

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in parameters:
                raise KeyError(f"width {width!r} reads {node.id}, which is "
                               f"not a parameter ({sorted(parameters)})")
            return int(parameters[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"width {width!r}: {type(node).__name__} is not "
                         "integer arithmetic over parameters")
    return walk(tree)


def elaborated_ports(sources, top: str, parameters: dict | None = None
                     ) -> dict[str, tuple[str, int]]:
    """``{port: (direction, width)}`` of module `top` as Verilator elaborates
    it, at `parameters` (overrides of its defaults).

    `sources` is Verilog text, a path, or a list of either -- every module
    `top` instantiates must be among them. The answer is read from the
    elaborated tree Verilator writes as JSON, so a width is what the
    parameters made it, not what a declaration looks like."""
    if shutil.which("verilator") is None:
        raise RuntimeError("elaborated_ports needs verilator on the PATH")
    if isinstance(sources, (str, Path)):
        sources = [sources]
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, src in enumerate(sources):
            if isinstance(src, Path) or (isinstance(src, str)
                                         and "\n" not in src
                                         and src.endswith((".v", ".sv"))):
                files.append(str(Path(src).resolve()))
            else:
                f = Path(tmp) / f"src{i}.v"
                f.write_text(src)
                files.append(str(f))
        cmd = ["verilator", "--json-only", "-Wno-fatal", "-Wno-lint",
               "-Wno-style", "--top-module", top, "-Mdir", tmp]
        cmd += [f"-G{k}={int(v)}" for k, v in (parameters or {}).items()]
        run = subprocess.run(cmd + files, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"verilator could not elaborate {top}:\n"
                               + run.stderr.strip())
        tree = json.loads((Path(tmp) / f"V{top}.tree.json").read_text())

    types = {}

    def collect(node):
        if isinstance(node, dict):
            if str(node.get("type", "")).endswith("DTYPE") and "addr" in node:
                types[node["addr"]] = node
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)
    collect(tree)

    module = next((m for m in tree["modulesp"]
                   if m.get("origName", m.get("name")) == top), None)
    if module is None:
        raise RuntimeError(f"verilator's tree has no module {top}")
    ports = {}
    for node in module.get("stmtsp", []):
        if node.get("type") != "VAR":
            continue
        direction = str(node.get("direction", "NONE")).lower()
        if direction not in DIRECTIONS:
            continue
        dtype = types.get(node.get("dtypep"), {})
        span = dtype.get("range")
        if span:
            hi, lo = (int(x) for x in span.split(":"))
            width = abs(hi - lo) + 1
        else:
            width = 1
        ports[node["name"]] = (direction, width)
    return ports
