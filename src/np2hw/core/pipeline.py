# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The new compiler end to end: trace, canonicalise, lower -- or Unsupported.

A model the new compiler does not reach yet goes to the reference, which
serves it and is the yardstick while the new one grows.
"""
from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonicalise
from .lower import lower
from .passes import Unit
from .trace import trace


@dataclass
class CoreModule:
    """A model lowered by the new compiler, ready to print. `recipe` is what
    the reference needs to lower it instead, for an option this one does
    not honour yet. Lowered for the device `dev` and the clock budget
    `clk_ns`; for others, it is lowered again from the same canonical graph
    (for_build)."""
    unit: Unit
    image: object
    recipe: tuple
    canonical: object = None
    dev: object = None
    clk_ns: float | None = None

    def for_build(self, dev, clk_ns=None) -> "CoreModule":
        """This model lowered for `dev` with its stages fitting `clk_ns`:
        itself, if it already is."""
        from ..timing import device
        if self.canonical is None or (device(dev) is device(self.dev)
                                      and clk_ns == self.clk_ns):
            return self
        return CoreModule(lower(self.canonical, dev, clk_ns), self.image,
                          self.recipe, self.canonical, dev, clk_ns)


def compile(fn, image, *params, channels: int = 1, dev=None) -> CoreModule:
    c = canonicalise(trace(fn, image, *params, channels=channels))
    return CoreModule(lower(c, dev), image, (fn, image, params, channels),
                      c, dev)


def stage_ns(mod: CoreModule, dev) -> float:
    """The datapath's worst path into the output stage, in nanoseconds, asked
    of the device's cost model -- the table lowering keeps."""
    from ..timing import Depth, device
    fam = device(dev)
    reg = next(r for r in mod.unit.regs if r.attr("name") == "out_data")
    d = reg.operands[0]
    depth: dict = {}
    from .graph import walk
    for n in walk([d]):
        below = Depth()
        for v in n.operands:
            below = fam.worst(below, depth[id(v.node)])
        depth[id(n)] = below + _cost(n, fam)
    return fam.ns(depth[id(d.node)])


def _cost(n, dev):
    from ..timing import Depth
    k = n.kind
    if k in ("hw.port", "hw.parameter", "hw.const", "hw.param", "hw.regread",
             "hw.shr", "hw.shl", "hw.wrap", "hw.bits", "hw.cat", "hw.not",
             "hw.read"):
        return dev.wire()
    if k in ("hw.add", "hw.sub", "hw.neg"):
        return dev.add(n.value.type.bits)
    if k == "hw.mul":
        return dev.mul()
    if k == "hw.clip":
        return dev.clip(n.value.type.bits)
    if k in ("hw.lt", "hw.gt", "hw.le", "hw.ge", "hw.eq", "hw.ne"):
        return dev.compare(n.operands[0].type.bits)
    if k == "hw.abs":
        return dev.absolute(n.value.type.bits)
    if k in ("hw.sel", "hw.and", "hw.or"):
        return dev.select()
    if k == "hw.memread":
        return dev.mem()
    if k == "hw.mux":
        return Depth(levels=dev.mux_levels(len(n.operands) - 1))
    raise ValueError(f"no cost for {k} -- a new kind needs its hardware "
                     "shape here before it ships")
