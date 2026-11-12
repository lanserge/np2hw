# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The compiler end to end: trace, canonicalise, lower -- or Unsupported."""
from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonicalise
from .lower import lower
from .passes import Unit
from .trace import trace


@dataclass
class CoreModule:
    """A model lowered by the compiler, ready to print: lowered for the
    device `dev` and the clock budget `clk_ns`; for others, it is lowered
    again from the same canonical graph (for_build)."""
    unit: Unit
    image: object
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
                          self.canonical, dev, clk_ns)


def compile(fn, image, *params, channels: int = 1, dev=None) -> CoreModule:
    c = canonicalise(trace(fn, image, *params, channels=channels))
    return CoreModule(lower(c, dev), image, c, dev)


def to_ir(fn, image, *params, channels: int = 1):
    """Trace the NumPy model `fn` on `image` with its register `params`, and
    lower it: `(None, module)`, the module ready for `generate`. A model the
    compiler does not reach is refused here, by name (Unsupported), and one
    that cannot be built -- an index that can leave its table, say -- by its
    reason (ValueError)."""
    from ..ir import Params
    if len(params) == 1 and isinstance(params[0], Params):
        params = (params[0].trace_view(),)   # the Param-valued view
    return None, compile(fn, image, *params, channels=channels)


def _depth_ns(values, fam) -> dict:
    """Each value's worst path from the registers and ports it starts at, in
    nanoseconds, by the device's cost model."""
    from ..timing import Depth
    from .graph import walk
    depth: dict = {}
    for n in walk(values):
        below = Depth()
        for v in n.operands:
            below = fam.worst(below, depth[id(v.node)])
        depth[id(n)] = below + _cost(n, fam)
    return {id(v.node): fam.ns(depth[id(v.node)]) for v in values}


def port_ns(unit, fam) -> dict:
    """{name: ns} for every path the unit's control ends at: a memory's
    ports -- an address, a word written, an enable -- and each register's
    next value and enable, all worked out whole in the beat they are
    used, where no stage register can land."""
    ends = {k: v for k, v in unit.roots.items() if k.startswith("mem_")}
    for r in unit.regs:
        nxt, en = r.operands
        ends[f"register {r.attr('name')}"] = nxt
        ends[f"register {r.attr('name')} enable"] = en
    ns = _depth_ns(list(ends.values()), fam)
    return {k: ns[id(v.node)] for k, v in ends.items()}


def stage_ns(mod: CoreModule, dev) -> float:
    """The datapath's worst path into the output stage, in nanoseconds,
    asked of the device's cost model -- the table lowering keeps."""
    from ..timing import device
    fam = device(dev)
    reg = next(r for r in mod.unit.regs if r.attr("name") == "out_data")
    d = reg.operands[0]
    return _depth_ns([d], fam)[id(d.node)]


def control_ns(mod: CoreModule, dev) -> tuple:
    """The control's worst path, (name, ns): reported, not gated. The cost
    model is calibrated on datapath arithmetic and reads compare-and-select
    control pessimistically -- a line buffer's read address it put at
    12.7 ns routed at 9.9 on xc7z020-1, a row counter it put at 6.7 closes
    at 6.45 -- so it says where to look, and the real build decides."""
    from ..timing import device
    paths = {k: v for k, v in port_ns(mod.unit, device(dev)).items()
             if k not in ("register out_data",)
             and not k.startswith(("register out_valid", "register out_sof",
                                   "register out_eol", "register out_last"))}
    return max(paths.items(), key=lambda kv: kv[1]) if paths else ("", 0.0)


def _cost(n, dev):
    from ..timing import Depth
    k = n.kind
    if k in ("hw.port", "hw.const", "hw.param", "hw.frameparam", "hw.regread",
             "hw.shr", "hw.shl", "hw.wrap", "hw.bits", "hw.cat", "hw.not",
             "hw.read"):
        return dev.wire()
    if k in ("hw.add", "hw.sub", "hw.neg"):
        return dev.add(n.value.type.bits)
    if k == "hw.mul":
        return dev.mul()
    if k == "hw.clip":
        return dev.clip(n.value.type.bits)
    if k in ("hw.max", "hw.min"):
        return dev.compare(max(o.type.bits for o in n.operands)) + \
            dev.select()
    if k in ("hw.lt", "hw.gt", "hw.le", "hw.ge", "hw.eq", "hw.ne"):
        return dev.compare(n.operands[0].type.bits)
    if k == "hw.abs":
        return dev.absolute(n.value.type.bits)
    if k in ("hw.sel", "hw.and", "hw.or"):
        return dev.select()
    if k in ("hw.memread", "hw.rom"):
        return dev.mem()
    if k == "hw.mux":
        return Depth(levels=dev.mux_levels(len(n.operands) - 1))
    raise ValueError(f"no cost for {k} -- a new kind needs its hardware "
                     "shape here before it ships")
