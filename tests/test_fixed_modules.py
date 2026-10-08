# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Every fixed module np2hw describes has exactly the ports its
description states.

A design is wired by a library description, never by reading the
Verilog, so the description has to be the module: each is elaborated by
Verilator, at parameter values chosen here rather than its own defaults,
and every port it has must be declared, with the direction and width its
expression gives.
"""
import shutil

import pytest

from np2hw.design import composite
from np2hw.design.interfaces import expand
from np2hw.ports import elaborated_ports

LIBRARIES = ("np2hw.hdl.bus", "np2hw.hdl.glue", "np2hw.hdl.stream")

# Values no module defaults to, so a width that ignores its parameter shows.
VALUES = {"WIDTH": 7, "VALUE": 5, "IN": 13, "HI": 9, "LO": 4, "STAGES": 4,
          "DATA_BITS": 11, "A": 3}


def _described():
    import json
    from importlib import resources
    out = []
    for library in LIBRARIES:
        module = library.rpartition(".")[2]
        blocks = json.loads(resources.files("np2hw.hdl")
                            .joinpath(f"{module}.json").read_text())["blocks"]
        for name in blocks:
            out.append(composite.find(f"{library}:{name}"))
    return out


BLOCKS = _described()


def _declared(obj, values):
    want = {}
    for side in ("inputs", "outputs"):
        for port in obj.get(side, []):
            raw = port.get("width", 1)
            width = raw if isinstance(raw, int) else composite.evaluate(
                composite.parse(raw, port["name"]), lambda n: values[n])
            want.update(expand(port, side, width))
    return want


def test_every_library_describes_something():
    assert len(BLOCKS) >= 7, [f.ref for f in BLOCKS]


@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not on PATH")
@pytest.mark.parametrize("found", BLOCKS, ids=[f.ref for f in BLOCKS])
def test_a_fixed_module_has_exactly_the_ports_it_declares(found):
    obj = found.obj
    declared = obj.get("parameters", {})
    params = {k: v for k, v in VALUES.items() if k in declared}
    sources = [composite.source_file(found.folder, f) for f in obj["sources"]]
    got = elaborated_ports(sources, obj["vivado"]["module"], params)
    want = _declared(obj, {**{k: v.get("default") for k, v in
                              declared.items()}, **params})
    assert want == got, {
        "undeclared": sorted(set(got) - set(want)),
        "not in the module": sorted(set(want) - set(got)),
        "differ": {k: (want[k], got[k]) for k in set(want) & set(got)
                   if want[k] != got[k]}}
