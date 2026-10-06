# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The memory-mapped bus tables, against pins a vendor's IP really has."""
from np2hw.design.interfaces import expand


def test_an_axi3_master_has_the_zynq_general_purpose_ports_signals():
    got = expand({"name": "gp", "bus": "axi3", "id_width": 12}, "outputs", 32)
    # what Vivado built for processing_system7's M_AXI_GP0, less its clock
    assert len(got) == 38
    assert got["gp_awlen"] == ("output", 4)
    assert got["gp_awlock"] == ("output", 2)
    assert got["gp_wid"] == ("output", 12)
    assert got["gp_rdata"] == ("input", 32)
    assert got["gp_wstrb"] == ("output", 4)


def test_an_axi4_write_master_without_ids_is_a_vdmas():
    got = expand({"name": "m", "bus": "axi4", "channels": ["write"],
                  "omit": ["awid", "awlock", "awqos", "bid"]}, "outputs", 64)
    assert sorted(got) == sorted(f"m_{s}" for s in (
        "awaddr", "awburst", "awcache", "awlen", "awprot", "awready",
        "awsize", "awvalid", "bready", "bresp", "bvalid", "wdata", "wlast",
        "wready", "wstrb", "wvalid"))
    assert got["m_awlen"] == ("output", 8)
    assert got["m_awready"] == ("input", 1)


def test_a_pins_bundle_keeps_its_members_names():
    got = expand({"name": "FIXED_IO", "bus": "pins",
                  "members": [{"name": "MIO", "width": 54, "inout": True},
                              {"name": "PS_CLK", "inout": True}]},
                 "outputs", 1)
    assert got == {"MIO": ("inout", 54), "PS_CLK": ("inout", 1)}
