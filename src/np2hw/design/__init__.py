# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Designs as JSON: composites of blocks, nested to any depth.

A design is a NETLIST of named block instances and the connections between
their ports -- written by hand, or by a builder, since a graph of named
ports is exactly what a GUI edits -- and a composite like any other block,
so designs are built of designs. :mod:`np2hw.design.composite` loads one
(blocks.schema.json is the format), :mod:`np2hw.design.netlist` is what a
loaded design is to everything that reads it, :mod:`np2hw.design.wiring`
checks it, and :mod:`np2hw.design.profile` holds values for one.

A design is an INPUT and cannot express an address: blocks declare local
offsets, and a composer assigns each instance its base. If a design could
pin an address, a builder could produce a map that silently disagreed with
the hardware it generated.
"""
from __future__ import annotations

import re
from pathlib import Path


def is_reference(text) -> bool:
    """Whether ``text`` is the shape of a reference,
    ``package.module[:block]``, and names no file."""
    return not Path(str(text)).exists() and re.fullmatch(
        r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+(:[A-Za-z_]\w*)?", str(text)) is not None
