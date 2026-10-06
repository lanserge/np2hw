# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Pipelines: a composite of model blocks, built into hardware.

:mod:`np2hw.pipeline.compose` composes a loaded design's blocks, assigns
each instance its registers' base, emits the register map and builds the
Verilog; :mod:`np2hw.pipeline.run` runs it -- the models, and the RTL twin
beside them. A pipeline's blocks bring their content's context vocabulary,
which names the context block placed at base 0 in every pipeline of them.

    from np2hw import pipeline
    from np2hw.pipeline import compose
    design = pipeline.load("acme.pipelines.isp:isp")
    open("build/isp.v", "w").write(compose.generate(design).verilog)
"""
from __future__ import annotations


def load(ref: str):
    """Load the composite ``ref`` (``package.module:block``) as a pipeline
    to build: model blocks, a stream entering it, and a graph np2hw's laws
    accept -- each checked here, so a design that cannot be built is refused
    when it is named."""
    from np2hw.design import composite, is_reference
    from np2hw.pipeline import compose
    if not is_reference(ref):
        raise ValueError(
            f"{ref!r} is not a design reference: a design is a composite, "
            "named package.module:block (a library folder goes on the import "
            "path with --library)")
    design = composite.load(str(ref))
    compose.shape(design)
    compose.netlist(design)
    return design
