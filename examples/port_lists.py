# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Every generator publishes its ports, and the list is the module's.

A composer instantiating a generated module reads ``interface["ports"]``:
every port with its direction and width -- the video pins, the status
words, the bus ports, not only the streams. The header is written from
that same list, so the two cannot drift; this holds them to the Verilog
anyway, by asking Verilator what the module elaborates to. A width that
is an expression over the module's parameters is checked at the
generated defaults AND at an override, since a design that overrides a
parameter gets what the expression says or nothing.

Run:  python examples/port_lists.py   (needs verilator)
"""
import sys

from np2hw import axil_regfile
from np2hw.frame_history import frame_history
from np2hw.i2c import ddc_slave
from np2hw.ports import elaborated_ports, resolve
from np2hw.stream import cdc_fifo, skid, stream_cdc, tee
from np2hw.video_mem import framebuffer_read
from np2hw.video_out import scanout

CASES = {
    "skid": lambda: skid(10),
    "cdc_fifo": lambda: cdc_fifo(26, 5),
    "tee": lambda: tee(27),
    "stream_cdc": lambda: stream_cdc(12),
    "scanout": lambda: scanout("1080p30"),
    "scanout_genlock": lambda: scanout("1080p30", genlock=True),
    "ddc_slave": lambda: ddc_slave(),
    "framebuffer_read": lambda: framebuffer_read(),
    "frame_history": lambda: frame_history(3, 13, 16, 8, loop=4),
    "axil_regfile": lambda: axil_regfile(
        [("a", 16), ("b", 1), ("arm", 1)], arm="arm", frame_sync=False),
    "compose": lambda: _composed(),
    "control_wrap": lambda: _wrapped(bus_clock=False),
    "control_wrap_bus_clock": lambda: _wrapped(bus_clock=True),
}


def _composed() -> dict:
    """Two traced cores in a line, composed, with a scalar and a signed
    port beside the streams. The cores are companion modules."""
    import numpy as np
    from np2hw import (Connection, Image2D, Instance, Param, Port, StreamType,
                       compose, generate, to_ir)

    def gain(img, g):
        return ((g * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)

    cores = []
    for name in ("a", "b"):
        _, out = to_ir(gain, Image2D("img", 8, 4, bits=8),
                       Param("g", bits=8, default=16))
        cores.append(generate(out, module_name=f"gain_{name}"))
    top = compose(
        "pair",
        [Instance("a", cores[0], bind={"g": "g_a"}),
         Instance("b", cores[1], bind={"g": "g_b"})],
        [Connection("in", "a.in"), Connection("a.out", "b.in"),
         Connection("b.out", "out")],
        [Port("g_a", "in", width=8), Port("g_b", "in", width=8),
         Port("trim", "in", width=6, signed=True, comment="unused, signed"),
         Port("in", "in", stream=StreamType(8, ("sof", "eol", "last"))),
         Port("out", "out", stream=StreamType(8, ("sof", "eol", "last")))])
    return dict(top, verilog="\n".join([cores[0]["verilog"], cores[1]["verilog"],
                                       top["verilog"]]))


def _wrapped(bus_clock: bool) -> dict:
    """A traced core behind its register file, with a context port passed
    through. The core is a companion module the wrapper instantiates."""
    import numpy as np
    from np2hw import Image2D, Param, Reg, control_wrap, generate, to_ir

    def gain(img, g):
        return ((g * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)

    _, out = to_ir(gain, Image2D("img", 8, 4, bits=8),
                   Param("g", bits=8, default=16))
    core = generate(out, module_name="gain_core")
    wrap = control_wrap(core, [Reg("g", 8, reset=16)], bind={"g": "param_g"},
                        module_name="gain_ctrl", addr_bits=8,
                        commit="out_valid && out_ready && out_last",
                        bus_clock=bus_clock, passthrough=[("ctx_width", 16)])
    return dict(wrap, verilog=core["verilog"] + "\n" + wrap["verilog"])

# The generators whose streams carry their bundle attributes; the others
# (frame_history) declare streams without them.
STREAMS_DECLARED = ("skid", "cdc_fifo", "tee", "stream_cdc", "compose",
                    "control_wrap", "control_wrap_bus_clock")

# One override per parameterised generator: a width that reads the
# parameter must follow it.
OVERRIDES = {"skid": {"W": 7}, "cdc_fifo": {"W": 9}, "tee": {"W": 5},
             "ddc_slave": {"AW": 12}}


def agrees(name, result, override=None) -> bool:
    iface = result["interface"]
    params = dict(iface.get("parameters", {}))
    params.update(override or {})
    want = {p["name"]: (p["direction"], resolve(p["width"], params))
            for p in iface["ports"]}
    from np2hw.hdl import closure
    # a fixed module is elaborated at the values this use sets on it, a
    # generated one at the defaults it was written with
    at = {**result.get("parameters", {}), **(override or {})}
    got = elaborated_ports(closure(result), result["module"], at or None)
    label = name + (f" {override}" if override else "")
    if want == got:
        print(f"  {label:<34} {len(got):>3} ports  PASS")
        return True
    missing = sorted(set(got) - set(want))
    extra = sorted(set(want) - set(got))
    differ = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    print(f"  {label:<34} FAIL  unpublished {missing}  not in the module "
          f"{extra}  differ {[(k, want[k], got[k]) for k in differ]}")
    return False


def streams_are_buses(name, result) -> bool:
    """Every stream the interface lists is one bus of the published ports,
    holding exactly that stream's signals -- so a composer can wire it by
    its name without knowing np2hw's suffixes."""
    from np2hw.stream import bundle_name
    iface = result["interface"]
    bad = []
    for stream in iface.get("streams", []):
        prefix = stream["prefix"]
        want = {f"{prefix}_{s}" for s in ("valid", "ready", "data",
                                          *stream["flags"])}
        got = {p["name"] for p in iface["ports"]
               if p.get("bus", {}).get("bundle") == bundle_name(prefix)}
        if want != got:
            bad.append((prefix, sorted(want ^ got)))
    print(f"  {name + ' streams':<34} {'PASS' if not bad else f'FAIL {bad}'}")
    return not bad


def main() -> int:
    print("published port lists against the elaborated module:")
    r = []
    for name, make in CASES.items():
        result = make()
        r.append(agrees(name, result))
        if name in STREAMS_DECLARED:
            r.append(streams_are_buses(name, result))
        if name in OVERRIDES:
            r.append(agrees(name, result, OVERRIDES[name]))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
