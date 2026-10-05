# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw's fixed modules, as Verilog files.

A block whose structure never varies -- a register slice, a tee, a
dual-clock FIFO -- is a parameterised Verilog file here, read and reviewed
as one, not printed by Python. What varies between uses is a parameter
(a width, a depth), set where it is instantiated. Generated Verilog is for
what genuinely varies: a traced model, a composition.

A generator result that instantiates one of these lists it in
``requires``; :func:`closure` is the result's own text with everything it
requires, for a simulator handed one string.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def path(module: str) -> Path:
    """Where the file defining `module` lives."""
    found = files(__package__).joinpath(f"{module}.v")
    if not found.is_file():
        raise KeyError(f"np2hw has no fixed module {module!r}")
    return Path(str(found))


def text(module: str) -> str:
    return path(module).read_text()


def closure(result: dict) -> str:
    """A result's Verilog with every fixed module it requires, each once."""
    seen, parts = set(), []
    for module in result.get("requires", ()):
        if module not in seen:
            seen.add(module)
            parts.append(text(module))
    return "\n".join(parts + [result["verilog"]])
