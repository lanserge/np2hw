# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.design.verilog: one Verilog module per composite.

The design here is real modules -- np2hw's decoder and arbiter, a vendor
register file as a black box -- with a composite inside a composite whose
ports are whole buses, and an FPGA pin driven from inside it. What is
written is linted by Verilator with every module it instances.
"""
import json
import shutil
import subprocess

import pytest

from np2hw import hdl
from np2hw.design.composite import load, write_generated
from np2hw.design.verilog import modules
from np2hw.toolchain.vivado.vendor_check import vendor_pins

IP = "toylib.ip"

PAIR = {"inputs": [{"name": "clk"}, {"name": "rst_n"}, {"name": "cpu"},
                   {"name": "side"}],
        "outputs": [{"name": "regs"}],
        "pins": [{"name": "busy", "dir": "O"}],
        "nodes": {"arb": {"block": "np2hw.hdl.bus:axil_arbiter"}},
        "connections": [["clk", "arb.aclk"], ["rst_n", "arb.aresetn"],
                        ["cpu", "arb.s0_axi"], ["side", "arb.s1_axi"],
                        ["arb.m_axi", "regs"],
                        ["arb.s0_axi_awready", "busy"]]}


def _design(library):
    lib = library(lambda name: {"lib": {"pair": PAIR, "top": {
        "inputs": [], "outputs": [],
        "pins": [{"name": "clk", "dir": "I"}, {"name": "rst_n", "dir": "I"}],
        "nodes": {
            "dec": {"block": "np2hw.bus:axil_decoder", "build": {
                "sizes": ["regs.S_AXI.window", "4096"],
                "base": 0x4000_0000, "span": 0x4000_0000}},
            "p": {"block": f"{name}.lib:pair"},
            "regs": {"block": f"{IP}:regs"}},
        "connections": [["clk", "dec.aclk"], ["rst_n", "dec.aresetn"],
                        ["clk", "p.clk"], ["rst_n", "p.rst_n"],
                        ["dec.m00_axi", "p.cpu"], ["dec.m01_axi", "p.side"],
                        ["p.regs", "regs.S_AXI"],
                        ["clk", "regs.s_axi_aclk"],
                        ["rst_n", "regs.s_axi_aresetn"]]}}})
    return load(f"{lib}.lib:top")


def test_each_composite_is_a_module_and_the_top_comes_last(library):
    design = _design(library)
    names = [n for n, _ in modules(design)]
    assert names == ["pair", "top"]


def test_a_composite_port_is_the_bus_it_meets(library):
    text = dict(modules(_design(library)))["pair"]
    # the bus from outside is a slave bundle, named by the composite's port
    assert "input  wire [31:0] cpu_awaddr" in text
    assert "output wire cpu_awready" in text
    assert "output wire [31:0] regs_awaddr" in text
    # and the pin rises through it
    assert "output wire busy" in text


@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not on PATH")
def test_what_is_written_lints_with_what_it_instances(library, tmp_path):
    design = _design(library)
    files = []
    for path in write_generated(design, tmp_path / "gen"):
        files.append(str(path))
    files.append(str(hdl.path("np2hw_axil_decoder")))
    files.append(str(hdl.path("np2hw_axil_arbiter")))
    # the vendor register file is the toolchain's to make; a stub with its
    # implementation's pins stands in for it here
    leaf = design.instances["regs"].leaf
    pins = vendor_pins(leaf)
    stub = ["module regs ("] + [
        f"    {d} wire [{w - 1}:0] {pin}" + ("," if i < len(pins) - 1 else "")
        for i, (pin, d, w) in enumerate(pins.values())] + [");", "endmodule"]
    (tmp_path / "regs.v").write_text("\n".join(stub) + "\n")
    files.append(str(tmp_path / "regs.v"))
    for name, text in modules(design):
        (tmp_path / f"{name}.v").write_text(text)
        files.append(str(tmp_path / f"{name}.v"))
    done = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "-Wno-UNUSEDSIGNAL",
         "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY", "-Wno-UNDRIVEN",
         "--top-module", "top", *files], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_wires_join_what_the_design_joins(library):
    """Through the composite boundary, signal by signal: the decoder's
    port into the pair's bus, the pair's bus out into the register file
    (its address the low bits the slave decodes), the pin up to the top."""
    text = dict(modules(_design(library)))
    top, pair = text["top"], text["pair"]
    assert ".cpu_awaddr(dec_m00_axi_awaddr)" in top
    assert ".side_wdata(dec_m01_axi_wdata)" in top
    assert ".S_AXI_awaddr(p_regs_awaddr[8:0])" in top
    assert ".busy(busy)" in top
    assert ".s0_axi_awaddr(cpu_awaddr)" in pair
    assert ".m_axi_awaddr(regs_awaddr)" in pair
    # the ready the bus returns is also the pin: one net, two ports
    assert ".s0_axi_awready(cpu_awready)" in pair
    assert "assign busy = cpu_awready;" in pair


def _grouped(library):
    lib = library(lambda name: {"lib": {"pair": PAIR, "top": {
        "inputs": [], "outputs": [],
        "pins": [{"name": "clk", "dir": "I"}, {"name": "rst_n", "dir": "I"}],
        "nodes": {
            "dec": {"block": "np2hw.bus:axil_decoder", "build": {
                "sizes": ["regs.S_AXI.window", "4096"],
                "base": 0x4000_0000, "span": 0x4000_0000}},
            "p": {"block": f"{name}.lib:pair"},
            "regs": {"block": f"{IP}:regs"},
            "k": {"block": f"{IP}:constant"},
            "inv": {"block": "np2hw.hdl.glue:not"}},
        "connections": [["clk", "dec.aclk"], ["rst_n", "dec.aresetn"],
                        ["clk", "p.clk"], ["rst_n", "p.rst_n"],
                        ["dec.m00_axi", "p.cpu"], ["dec.m01_axi", "p.side"],
                        ["p.regs", "regs.S_AXI"],
                        ["clk", "regs.s_axi_aclk"],
                        ["rst_n", "regs.s_axi_aresetn"],
                        ["k.dout", "inv.a"]]}}})
    return load(f"{lib}.lib:top")


def test_leaves_a_toolchain_makes_together_are_one_black_box(library):
    """A group is one cell: its members' ports that meet the rest are its
    ports, named <cell>_<port> -- as the member states the port, or the
    mirror of the far end where it states none."""
    from np2hw.design.verilog import group_ports
    design = _grouped(library)
    groups = {"bd": ["regs", "k"]}
    ports = group_ports(design, groups)["bd"]
    assert ports["regs_S_AXI"]["member"] == ("regs", "S_AXI")
    assert ports["regs_S_AXI"]["spec"][0] == "bundle"
    assert ports["k_dout"]["spec"] == ("signal", "output", 1)   # mirror
    assert ports["regs_s_axi_aclk"]["spec"] == ("signal", "input", 1)
    top = dict(modules(design, groups))["top"]
    assert "    bd bd (" in top
    assert ".regs_S_AXI_awaddr(p_regs_awaddr[8:0])" in top
    assert ".k_dout(bd_k_dout)" in top and ".a(bd_k_dout)" in top
    assert " regs (" not in top and " k (" not in top


@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not on PATH")
def test_a_group_lints_against_a_module_of_its_stated_ports(library,
                                                            tmp_path):
    from np2hw.design.verilog import group_ports
    design = _grouped(library)
    groups = {"bd": ["regs", "k"]}
    decls = []
    for port, rec in group_ports(design, groups)["bd"].items():
        spec = rec["spec"]
        if spec[0] == "signal":
            decls.append((spec[1], spec[2], port))
        else:
            decls += [(d, w, f"{port}_{s}") for s, (d, w) in spec[2].items()]
    stub = ["module bd ("] + [
        f"    {d} wire [{w - 1}:0] {n}" + ("," if i < len(decls) - 1 else "")
        for i, (d, w, n) in enumerate(decls)] + [");", "endmodule"]
    files = [str(p) for p in write_generated(design, tmp_path / "gen")]
    files += [str(hdl.path(m)) for m in ("np2hw_axil_decoder",
                                         "np2hw_axil_arbiter", "np2hw_not")]
    (tmp_path / "bd.v").write_text("\n".join(stub) + "\n")
    files.append(str(tmp_path / "bd.v"))
    for name, text in modules(design, groups):
        (tmp_path / f"{name}.v").write_text(text)
        files.append(str(tmp_path / f"{name}.v"))
    done = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "-Wno-UNUSEDSIGNAL",
         "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY", "-Wno-UNDRIVEN",
         "--top-module", "top", *files], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_an_instance_named_like_a_port_is_renamed(library):
    """A block design keeps cells and ports apart; a Verilog module has one
    namespace, so an instance sharing a port's name takes another."""
    lib = library(lambda name: {"lib": {"top": {
        "pins": [{"name": "flag", "dir": "O"}],
        "nodes": {"flag": {"block": "np2hw.hdl.glue:const"}},
        "connections": [["flag.q", "flag"]]}}})
    text = dict(modules(load(f"{lib}.lib:top")))["top"]
    assert "output wire flag" in text
    assert "np2hw_const #(.WIDTH(1), .VALUE(0)) flag_i (" in text
    assert ".q(flag)" in text


def test_a_bidirectional_pin_rises_as_inout(library):
    lib = library(lambda name: {"lib": {"top": {
        "pins": [{"name": "sda", "dir": "IO"}],
        "nodes": {"phy": {"block": f"{name}.lib:phy"}},
        "connections": [["sda", "phy.io"]]},
        "phy": {"inputs": [], "outputs": [{"name": "io", "inout": True}],
                "vivado": {"module": "phy"}, "sources": ["phy.v"]}}})
    text = dict(modules(load(f"{lib}.lib:top")))["top"]
    assert "inout  wire sda" in text and ".io(sda)" in text


RASTER_GEN = '''
def raster(bits: int) -> dict:
    """Video out on a bundle whose members do not carry its name."""
    from np2hw.ports import Pin, header, published
    vid = "xilinx.com:interface:vid_io:1.0"
    pins = [Pin(f"vid_{s}", "output", w, bus=(vid, "vid_io", s.upper()))
            for s, w in (("active_video", 1), ("data", bits), ("hsync", 1),
                         ("vsync", 1))]
    text = "\\n".join(header("raster", pins)) + "\\nendmodule\\n"
    return {"verilog": text, "module": "raster",
            "interface": {"ports": published(pins)}}


raster.leaf = True
'''


def test_bundles_named_apart_meet_member_to_member(library):
    """A raster's vid_* members, in a bundle called vid_io, through a
    composite to a transmitter's rgb_*: every member wired, by its part in
    the bus -- and the composite's port named by those parts."""
    lib = library(lambda name: {"lib": {
        "tx": {"inputs": [{"name": "rgb", "bus": "video", "width": 24}],
               "vivado": {"module": "tx"}, "sources": ["tx.v"]},
        "src": {"outputs": [{"name": "raster"}],
                "nodes": {"r": {"block": f"{name}.gen:raster",
                                "build": {"bits": 24}}},
                "connections": [["r.vid_io", "raster"]]},
        "top": {"nodes": {"s": {"block": f"{name}.lib:src"},
                          "t": {"block": f"{name}.lib:tx"}},
                "connections": [["s.raster", "t.rgb"]]}}},
        python={"gen": RASTER_GEN})
    text = dict(modules(load(f"{lib}.lib:top")))
    for signal in ("active_video", "data", "hsync", "vsync"):
        assert f".vid_{signal}(raster_{signal})" in text["src"], signal
        assert f".raster_{signal}(s_raster_{signal})" in text["top"], signal
        assert f".rgb_{signal}(s_raster_{signal})" in text["top"], signal
