# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The stream, declared in a form an integration tool can read.

An AXI4-Lite port is understood everywhere because a standard bus
definition describes it. np2hw's elastic stream is np2hw's own, so
without a definition it reaches an integrator as six loose wires per
stream -- and a block-design writer groups them by matching port names
with a regex, which is inference in our own tooling.

`np2hw.ipxact` renders the bus definition and the abstraction definition
from `np2hw.stream.STREAM_SIGNALS`, and the Verilog emitters attach the
matching attributes reading the SAME declaration. This asserts they
agree, which is the whole claim: a tool's view of the protocol and the
RTL's cannot drift, because there is one declaration behind both.

It also guards the AXI4-Lite attributes. Those are what let a packager
see one bus instead of seventeen pins, nothing else checks they are
there, and an emitter edit that dropped one would degrade the packaging
silently.

Run:  python examples/ipxact_stream.py   (no simulator needed)
"""
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from np2hw import (Image2D, Param, control_wrap, generate, ipxact,
                   saturate, to_ir)
from np2hw.regmap import Reg
from np2hw.stream import STREAM_FLAGS, STREAM_REQUIRED, STREAM_SIGNALS

NS = {"s": ipxact.NS}


def main():
    results = []
    print("IP-XACT for the elastic stream:")

    # ---- the two documents parse, and say what they should ----
    bus = ET.fromstring(ipxact.bus_definition())
    abst = ET.fromstring(ipxact.abstraction_definition())
    point_to_point = (bus.find("s:maxMasters", NS).text == "1"
                      and bus.find("s:maxSlaves", NS).text == "1"
                      and bus.find("s:isAddressable", NS).text == "false")
    print(f"  bus definition: point to point, not addressable   "
          f"{'PASS' if point_to_point else 'FAIL'}")
    results.append(point_to_point)

    # the abstraction names the bus it abstracts, by VLNV
    bt = abst.find("s:busType", NS)
    key = "{http://schemas.xilinx.com/XMLSchema}"  # attrs are spirit-namespaced
    names = {k.split("}")[-1]: v for k, v in bt.attrib.items()}
    binds = (names["vendor"], names["library"], names["name"],
             names["version"]) == ipxact.STREAM_VLNV
    print(f"  abstraction binds to that bus by VLNV             "
          f"{'PASS' if binds else 'FAIL'}")
    results.append(binds)

    # ---- every declared signal appears, with the right role ----
    ports = abst.findall(".//s:port", NS)
    seen = {}
    for p in ports:
        logical = p.find("s:logicalName", NS).text
        wire = p.find("s:wire", NS)
        seen[logical] = (
            wire.find("s:onMaster/s:direction", NS).text,
            wire.find("s:onSlave/s:direction", NS).text,
            wire.find("s:onMaster/s:presence", NS).text)
    complete = len(seen) == len(STREAM_SIGNALS)
    # READY runs against the data; every other signal runs with it. The
    # master is the SOURCE, so this is the handshake's whole asymmetry.
    roles_ok = all(
        seen[logical] == (("out", "in", presence) if from_source
                          else ("in", "out", presence))
        for _s, logical, _w, from_source, presence, _d in STREAM_SIGNALS)
    print(f"  all {len(STREAM_SIGNALS)} signals present, master = the source     "
          f"{'PASS' if complete and roles_ok else 'FAIL'}")
    results += [complete, roles_ok]

    # the flags are OPTIONAL and that is load-bearing: a core takes SOF to
    # re-anchor and regenerates the rest, so an input carries fewer flags
    # than the output it feeds. Required flags would force all six wires.
    optional = all(seen[l][2] == "optional"
                   for s, l, *_ in STREAM_SIGNALS if s in STREAM_FLAGS)
    required = all(seen[l][2] == "required"
                   for s, l, *_ in STREAM_SIGNALS if s in STREAM_REQUIRED)
    print(f"  handshake required, framing flags optional        "
          f"{'PASS' if optional and required else 'FAIL'}")
    results += [optional, required]

    # ---- the RTL agrees with the documents ----
    def gain(pixel, g):
        return saturate(pixel.astype(np.int32) * g // 256, 8)

    _, out = to_ir(gain, Image2D("img", 8, 4, bits=8), Param("g", bits=8))
    core = generate(out, module_name="ipx_core")
    wrapped = control_wrap(core, [Reg("g", bits=8, offset=0)],
                           {"g": "param_g"}, module_name="ipx_top",
                           frame_sync=False)
    src = wrapped["verilog"]

    vlnv = ipxact.vlnv_string(ipxact.STREAM_RTL_VLNV)
    declared = {}
    for line in src.splitlines():
        if vlnv in line:
            bundle, logical = line.split(vlnv)[1].split('"')[0].split()
            declared.setdefault(bundle, set()).add(logical)
    two_bundles = set(declared) == {"in", "out"}
    print(f"  the wrapper presents two stream interfaces        "
          f"{'PASS' if two_bundles else 'FAIL'}")
    results.append(two_bundles)

    # every port the RTL emits for a bundle is a signal the abstraction
    # knows, and every logical name it uses is one the abstraction defines
    known = all(name in seen for names in declared.values() for name in names)
    handshake = all(
        {"VALID", "READY", "DATA"} <= names for names in declared.values())
    print(f"  every declared port is a signal the bus defines   "
          f"{'PASS' if known and handshake else 'FAIL'}")
    results += [known, handshake]

    # ---- the AXI4-Lite side, which nothing else guards ----
    # count on the PACKAGED boundary -- the top's own port list. The
    # register file inside declares its bundle too, so counting the whole
    # text would pass for the wrong reason.
    head = src.split("module ipx_top (")[1].split("\n);")[0]
    axi = "xilinx.com:interface:aximm:1.0 s_axil"
    n_axi = head.count(axi)
    clock_names_bus = 'ASSOCIATED_BUSIF s_axil' in head
    polarity = 'POLARITY ACTIVE_' in head
    print(f"  AXI4-Lite: {n_axi:2d} ports declare their bundle         "
          f"{'PASS' if n_axi == 17 else 'FAIL'}")
    print(f"  ...its clock names the bus, its reset the polarity "
          f"{'PASS' if clock_names_bus and polarity else 'FAIL'}")
    results += [n_axi == 17, clock_names_bus, polarity]

    # ---- every module a board design instantiates by hand ----
    # The ISP's own blocks are wired by np2hw's composer, which emitted
    # both ends and infers nothing. These are the ones a human or a Tcl
    # writer connects, so these are the ones that must declare.
    from np2hw import bayerlink_in, cdc_fifo, skid, tee
    import re

    expected = {"skid": {"s", "m"}, "tee": {"in", "a", "b"},
                "cdc_fifo": {"in", "out"}, "bayerlink_in": {"out"}}
    built = {"skid": skid(12), "tee": tee(12), "cdc_fifo": cdc_fifo(12, 8),
             "bayerlink_in": bayerlink_in(bits=12)}
    plumbing = True
    for name, want in expected.items():
        text = built[name]["verilog"]
        # the module's OWN ports, not those of anything it bundles
        own = text.split(f"module {name}")[1].split("endmodule")[0]
        got = set(re.findall(r"np2hw_stream_rtl:1\.0 (\w+) ", own))
        plumbing &= got == want
    print(f"  the plumbing declares its streams too            "
          f"{'PASS' if plumbing else 'FAIL'}")
    results.append(plumbing)

    # a crossing whose two faces run on different clocks has to say which
    # clock carries which face, or the one fact it exists for is inferred
    cdc = built["cdc_fifo"]["verilog"]
    faces = ("ASSOCIATED_BUSIF in" in cdc and "ASSOCIATED_BUSIF out" in cdc)
    print(f"  the CDC FIFO's clocks name the face each carries  "
          f"{'PASS' if faces else 'FAIL'}")
    results.append(faces)

    ok = all(results)
    print("\n" + ("IPXACT STREAM PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
