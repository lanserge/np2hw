# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Designs for the tests, spelled in Python and loaded as JSON.

The same format and the same loader as a design on disk: :func:`build`
writes a description to a library on the import path and loads it by
reference. Blocks are named as toylib registers them.
"""
import json
import sys
import tempfile
from pathlib import Path

from np2hw import pipeline as pipelines
from np2hw.blocks import resolve


def block_ref(name: str) -> str:
    """A toylib block's reference, from its registered name."""
    return f"{resolve('toylib', name).model.__module__}:{name}"


def chain(*blocks: str, prefix: str | None = None,
          source: str = "in", sink: str = "out") -> dict:
    """A linear run of blocks, with sinks TAPPED -- to hand to describe().

    A block with no outputs does not advance the datapath: it taps whatever
    the last real block produced and the stream carries on past it. With a
    ``prefix`` the run is its own composite, instantiated under that name.
    """
    nodes, connections = {}, []
    current = "in" if prefix else source
    for name in blocks:
        ports = resolve("toylib", name).ports
        nodes[name] = {"block": block_ref(name)}
        connections.append([current, f"{name}.{ports.inputs[0]}"])
        if ports.outputs:
            current = f"{name}.{ports.outputs[0]}"
    connections.append([current, "out" if prefix else sink])
    return {"prefix": prefix, "source": source, "sink": sink,
            "nodes": nodes, "connections": connections}


def describe(name: str, *chains: dict, bit_depth: int = 8,
             width: int = 64, height: int = 32,
             inputs: tuple = ("in",), outputs: tuple = ("out",)) -> dict:
    """A library holding the composite ``name``, built for a ``bit_depth``
    stream of up to ``width`` x ``height`` on every input."""
    stream = {"bits": bit_depth, "max_width": width, "max_height": height}
    top = {"inputs": [{"name": n, **stream} for n in inputs],
           "outputs": [{"name": n} for n in outputs],
           "nodes": {}, "connections": []}
    blocks = {name: top}
    for run in chains:
        if run["prefix"] is None:
            top["nodes"].update(run["nodes"])
            top["connections"] += run["connections"]
            continue
        prefix = run["prefix"]
        blocks[prefix] = {"inputs": [{"name": "in"}],
                          "outputs": [{"name": "out"}],
                          "nodes": run["nodes"],
                          "connections": run["connections"]}
        top["nodes"][prefix] = {"block": "{lib}:" + prefix}
        top["connections"] += [[run["source"], f"{prefix}.in"],
                               [f"{prefix}.out", run["sink"]]]
    return {"blocks": blocks}


_ROOT = []


def library(description: dict) -> str:
    """Write ``description`` as a new library package on the import path;
    return the reference of its first block. ``{lib}`` in it stands for the
    library itself, so its blocks can name each other."""
    if not _ROOT:
        _ROOT.append(Path(tempfile.mkdtemp(prefix="np2hw-tests-")))
        sys.path.insert(0, str(_ROOT[0]))
    package = f"testlib_{len(_ROOT)}"
    folder = _ROOT[0] / package
    folder.mkdir()
    _ROOT.append(folder)
    text = json.dumps(description).replace("{lib}", f"{package}.design")
    (folder / "design.json").write_text(text)
    return f"{package}.design:{next(iter(description['blocks']))}"


def build(description: dict):
    """The pipeline ``description`` describes, through the composite loader."""
    return pipelines.load(library(description))
