# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.toolchain.vivado.checkhwh: the built design's clock and reset
ancestry, read from a hardware handoff."""
import pytest

from np2hw.toolchain.vivado.checkhwh import BOARD, CABLE, Design, audit


def _hwh(*modules):
    return "<EDKSYSTEM><MODULES>" + "".join(modules) + "</MODULES></EDKSYSTEM>"


def _module(name, typ, ports, extra=""):
    body = "".join(f'<PORT DIR="{d}" NAME="{p}" SIGNAME="{s}"/>'
                   for p, (d, s) in ports.items())
    return (f'<MODULE FULLNAME="/{name}" MODTYPE="{typ}" {extra}>'
            f"<PORTS>{body}</PORTS></MODULE>")


PS7 = _module("ps7", "processing_system7", {
    "FCLK_CLK0": ("O", "ps7_FCLK_CLK0"),
    "FCLK_RESET0_N": ("O", "ps7_FCLK_RESET0_N")})
ONE = _module("one", "xlconstant", {"dout": ("O", "one_dout")})
LINK = _module("link_rst", "link_reset", {"rst_pix": ("O", "link_rst_pix")})


def _sync(name, rst_n_in, locked="one_dout"):
    return _module(name, "np2hw_reset_sync", {
        "rst_n_in": ("I", rst_n_in), "locked": ("I", locked),
        "rst_n": ("O", f"{name}_rst_n")})


def test_a_reset_synchroniser_inherits_both_its_sources():
    d = Design(_hwh(PS7, ONE, LINK, _sync("rst_bus", "ps7_FCLK_RESET0_N"),
                    _sync("rst_isp", "link_rst_pix")))
    assert d.lineage("rst_bus_rst_n") is BOARD
    assert d.lineage("rst_isp_rst_n") is CABLE


@pytest.mark.parametrize("reset, ok", [("rst_bus_rst_n", True),
                                       ("rst_isp_rst_n", False)])
def test_an_np2hw_bus_module_on_the_links_reset_fails(tmp_path, reset, ok):
    bus = _module("regs", "np2hw_axil_decode_2_000000", {
        "aclk": ("I", "ps7_FCLK_CLK0"), "aresetn": ("I", reset)})
    path = tmp_path / "rx.hwh"
    path.write_text(_hwh(PS7, ONE, LINK,
                         _sync("rst_bus", "ps7_FCLK_RESET0_N"),
                         _sync("rst_isp", "link_rst_pix"), bus))
    assert (audit(path) == 0) is ok


def test_np2hw_glue_passes_its_inputs_ancestry_on():
    d = Design(_hwh(
        PS7, LINK,
        _module("k", "np2hw_const", {"q": ("O", "k_q")}),
        _module("inv", "np2hw_not", {"a": ("I", "link_rst_pix"),
                                     "y": ("O", "inv_y")}),
        _module("s", "np2hw_slice", {"d": ("I", "ps7_FCLK_RESET0_N"),
                                     "q": ("O", "s_q")}),
        _module("c", "np2hw_concat_1_1", {"in0": ("I", "k_q"),
                                          "in1": ("I", "s_q"),
                                          "q": ("O", "c_q")}),
        _module("c2", "np2hw_concat_1_1", {"in0": ("I", "k_q"),
                                           "in1": ("I", "inv_y"),
                                           "q": ("O", "c2_q")})))
    assert d.lineage("k_q") is BOARD
    assert d.lineage("inv_y") is CABLE
    assert d.lineage("s_q") is BOARD
    assert d.lineage("c_q") is BOARD
    assert d.lineage("c2_q") is CABLE
