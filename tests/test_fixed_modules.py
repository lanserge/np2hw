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

LIBRARIES = ("np2hw.hdl.bus", "np2hw.hdl.glue", "np2hw.hdl.stream",
             "np2hw.hdl.frames")

# Values no module defaults to, so a width that ignores its parameter shows.
VALUES = {"WIDTH": 7, "VALUE": 5, "IN": 13, "HI": 9, "LO": 4, "STAGES": 4,
          "DATA_BITS": 11, "A": 3, "BITS": 11, "FRAMES": 3, "SLOTS": 5,
          "CTX_BITS": 9, "PIXELS": 37}


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


def _modules(found):
    """(id, found, its module, its sources, port renames) for each way the
    block is built: itself, or each of its implementations, which take the
    block's own ports under the names their `ports` map gives."""
    obj = found.obj
    if "implementations" not in obj:
        return [(found.ref, found, obj["vivado"]["module"], obj["sources"],
                 {})]
    return [(f"{found.ref}[{name}]", found, impl["vivado"]["module"],
             impl["sources"], impl.get("ports", {}))
            for name, impl in obj["implementations"].items()
            if "module" in impl.get("vivado", {})]


BLOCKS = _described()
MODULES = [m for found in BLOCKS for m in _modules(found)]


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
@pytest.mark.parametrize("ref, found, module, files, renames", MODULES,
                         ids=[m[0] for m in MODULES])
def test_a_fixed_module_has_exactly_the_ports_it_declares(ref, found, module,
                                                          files, renames):
    obj = found.obj
    declared = {k: v for k, v in obj.get("parameters", {}).items()
                if k != "implementation"}
    params = {k: v for k, v in VALUES.items() if k in declared}
    sources = [composite.source_file(found.folder, f) for f in files]
    got = elaborated_ports(sources, module, params)
    want = _declared(obj, {**{k: v.get("default") for k, v in
                              declared.items()}, **params})
    for block_port, own in renames.items():
        for name in [n for n in want if n == block_port or
                     n.startswith(block_port + "_")]:
            want[own + name[len(block_port):]] = want.pop(name)
    assert want == got, {
        "undeclared": sorted(set(got) - set(want)),
        "not in the module": sorted(set(want) - set(got)),
        "differ": {k: (want[k], got[k]) for k in set(want) & set(got)
                   if want[k] != got[k]}}
