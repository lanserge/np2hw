# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A design built under Vivado from its own Verilog.

np2hw.toolchain.vivado.rtl writes the design as Verilog and the Tcl that
makes what only Vivado can: each vendor IP by ``create_ip`` under its cell
name, and the leaves Vivado makes only in a block design together in one,
whose wrapper the Verilog instances. The toy here has both kinds, an AXI3
master leaving the block design and an AXI4 slave entering it.
"""
import re

from np2hw.design.composite import load
from np2hw.toolchain.vivado.rtl import emit_rtl, groups

IP = "toylib.ip"

CPU = {"inputs": [{"name": "M_AXI_ACLK"}],
       "outputs": [{"name": "clk0"},
                   {"name": "M_AXI", "bus": "axi3", "width": 32,
                    "id_width": 12}],
       "vivado": {"ip": "vendor.com:ip:cpu", "integrator": True}}
MEM = {"inputs": [{"name": "aclk"}, {"name": "aresetn"},
                  {"name": "S_AXI", "bus": "axi4", "width": 32}],
       "vivado": {"ip": "vendor.com:ip:mem", "integrator": True}}
WRITER = {"inputs": [{"name": "clk"}],
          "outputs": [{"name": "m_axi", "bus": "axi4", "width": 32}],
          "vivado": {"module": "writer"}, "sources": ["writer.v"]}


def _design(library, tmp_path):
    lib = library(lambda name: {"lib": {
        "cpu": CPU, "mem": MEM, "writer": WRITER, "top": {
            "pins": [{"name": "rst_n", "dir": "I"}],
            "nodes": {"cpu": {"block": f"{name}.lib:cpu"},
                      "mem": {"block": f"{name}.lib:mem"},
                      "br": {"block": "np2hw.hdl.bus:axi3_to_axil"},
                      "regs": {"block": f"{IP}:regs"},
                      "wr": {"block": f"{name}.lib:writer"}},
            "connections": [
                ["cpu.clk0", "cpu.M_AXI_ACLK"], ["cpu.clk0", "br.aclk"],
                ["cpu.clk0", "regs.s_axi_aclk"], ["cpu.clk0", "mem.aclk"],
                ["cpu.clk0", "wr.clk"],
                ["rst_n", "br.aresetn"], ["rst_n", "regs.s_axi_aresetn"],
                ["rst_n", "mem.aresetn"],
                ["cpu.M_AXI", "br.s_axi"], ["br.m_axi", "regs.S_AXI"],
                ["wr.m_axi", "mem.S_AXI"]]}}})
    (tmp_path / lib / "writer.v").write_text("module writer; endmodule\n")
    return load(f"{lib}.lib:top")


def _emit(library, tmp_path):
    design = _design(library, tmp_path)
    return emit_rtl(design, out=tmp_path / "out", part="xc7z020clg400-1",
                    board_part="tul.com.tw:pynq-z2:part0:1.0",
                    board_files=tmp_path, ip_repos=[tmp_path])


def _config(tcl, port):
    m = re.search(r"set_property -dict \[list (.*)\] "
                  rf"\[get_bd_intf_ports {port}\]", tcl)
    return dict(re.findall(r"CONFIG\.(\w+) \{?([^ }]*)\}?", m.group(1)))


def test_leaves_vivado_makes_only_in_a_block_design_are_one(library,
                                                           tmp_path):
    assert groups(_design(library, tmp_path), "top") == {"top_bd_wrapper":
                                               ["cpu", "mem"]}


def test_the_verilog_instances_the_ip_and_the_block_design(library,
                                                          tmp_path):
    tcl, files, facts = _emit(library, tmp_path)
    top = files["top.v"]
    assert re.search(r"^    top_bd_wrapper \w+ \($", top, re.M)
    assert re.search(r"^    regs regs \($", top, re.M)
    assert "create_ip -vendor vendor.com -library ip -name regs " \
           "-module_name regs" in tcl
    assert "create_bd_design top_bd" in tcl
    assert "create_bd_cell -type ip -vlnv vendor.com:ip:cpu cpu" in tcl
    assert "create_bd_cell -type ip -vlnv vendor.com:ip:mem mem" in tcl
    assert facts == {"top": "top"}


def test_a_bus_entering_is_shaped_as_its_far_end_states(library, tmp_path):
    """The slave port a block design takes in has the shape of the master
    outside; the master port it gives out has the member's shape, which
    Vivado derives -- all but its protocol, which must match the member's
    before Vivado connects the two."""
    tcl, _, _ = _emit(library, tmp_path)
    assert "create_bd_intf_port -mode Slave " in tcl
    slave = _config(tcl, "mem_S_AXI")
    assert slave["PROTOCOL"] == "AXI4" and slave["DATA_WIDTH"] == "32"
    assert slave["READ_WRITE_MODE"] == "READ_WRITE"
    assert _config(tcl, "cpu_M_AXI") == {"PROTOCOL": "AXI3"}
    # each takes its clock from the member's clock pin naming it
    assert "np2hw_associate mem_S_AXI mem/S_AXI" in tcl
    assert "np2hw_associate cpu_M_AXI cpu/M_AXI" in tcl
    # and an input that is the member's reset, its clock
    assert "np2hw_reset mem_aresetn mem/aresetn" in tcl


def test_the_block_design_is_synthesised_with_the_design(library, tmp_path):
    """One process: the block design's sources are made here and read
    whole by the design's synthesis -- no out-of-context runs."""
    tcl, _, _ = _emit(library, tmp_path)
    lines = tcl.splitlines()
    wrap = lines.index("make_wrapper -files [get_files top_bd.bd] "
                       "-top -import")
    assert lines[wrap + 1:wrap + 3] == [
        "set_property synth_checkpoint_mode None [get_files top_bd.bd]",
        "generate_target all [get_files top_bd.bd]"]


def _clocked(library, tmp_path, khz):
    lib = library(lambda name: {"lib": {"writer": WRITER, "top": {
        "parameters": {"rate": {"default": 50000}},
        "pins": [{"name": "refclk", "dir": "I", "clock": {"khz": khz}}],
        "nodes": {"wr": {"block": f"{name}.lib:writer"}},
        "connections": [["refclk", "wr.clk"]]}}})
    (tmp_path / lib / "writer.v").write_text("module writer; endmodule\n")
    return load(f"{lib}.lib:top")


def test_a_pin_is_constrained_at_the_clock_the_design_states(library,
                                                            tmp_path):
    """The rate is the design's -- here an expression of its parameter --
    and the constraint is written from it, ahead of every other."""
    design = _clocked(library, tmp_path, "rate * 2")
    assert design.pins["refclk"]["clock"] == {"khz": 100000}
    tcl, files, _ = emit_rtl(design, out=tmp_path / "out",
                             part="xc7z020clg400-1",
                             board_part="tul.com.tw:pynq-z2:part0:1.0",
                             board_files=tmp_path, ip_repos=[tmp_path])
    assert "create_clock -period 10.000 -name refclk_clk " \
           "[get_ports refclk]" in files["top_pin_clocks.xdc"]
    constrs = [line for line in tcl.splitlines()
               if line.startswith("add_files -fileset constrs_1")]
    assert constrs and constrs[0].endswith("top_pin_clocks.xdc}")


def test_a_pin_clock_of_no_rate_is_refused(library, tmp_path):
    from np2hw.design.composite import CompositionError
    import pytest
    with pytest.raises(CompositionError, match="refclk states a clock of 0"):
        _clocked(library, tmp_path, "rate - 50000")
