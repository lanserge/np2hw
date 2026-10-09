# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A bus's clocks and resets, traced through the netlist to what makes them.

The toy: a processor's clock and reset, a receiver whose clock is the
cable's, np2hw's reset synchroniser between, and a register slave behind
np2hw's bridge. Each leaf states what its bus ports run on (``clocks``) and
where its outputs come from (``ancestry``); the audit reads only that.
"""
import pytest

from np2hw.design.ancestry import audit
from np2hw.design.composite import load

CPU = {"inputs": [], "outputs": [{"name": "clk"}, {"name": "rst_n"},
                                 {"name": "M_AXI", "bus": "axi3",
                                  "width": 32, "id_width": 12}],
       "vivado": {"module": "cpu"}, "sources": ["cpu.v"],
       "facts": {"clocks": {"M_AXI": ["clk"]},
                 "ancestry": {"clk": "board", "rst_n": "board"}}}
RX = {"outputs": [{"name": "pclk"}, {"name": "locked"}],
      "vivado": {"module": "rx"}, "sources": ["rx.v"],
      "facts": {"ancestry": {"pclk": "cable", "locked": "cable"}}}
SLAVE = {"inputs": [{"name": "aclk"}, {"name": "aresetn"},
                    {"name": "S_AXI", "bus": "axilite", "width": 32}],
         "vivado": {"module": "slave"}, "sources": ["slave.v"],
         "facts": {"clocks": {"S_AXI": ["aclk", "aresetn"]}}}


def _design(library, *wiring, slave=SLAVE):
    lib = library(lambda name: {"lib": {
        "cpu": CPU, "rx": RX, "slave": slave, "top": {
            "nodes": {"cpu": {"block": f"{name}.lib:cpu"},
                      "rx": {"block": f"{name}.lib:rx"},
                      "rst": {"block": "np2hw.hdl.bus:reset_sync"},
                      "one": {"block": "np2hw.hdl.glue:const",
                              "build": {"VALUE": 1}},
                      "br": {"block": "np2hw.hdl.bus:axi3_to_axil"},
                      "s": {"block": f"{name}.lib:slave"}},
            "connections": [
                ["cpu.M_AXI", "br.s_axi"], ["br.m_axi", "s.S_AXI"],
                ["cpu.clk", "br.aclk"], ["rst.rst_n", "br.aresetn"],
                *[list(w) for w in wiring]]}}})
    return load(f"{lib}.lib:top")


BOARD = (("cpu.clk", "rst.clk"), ("cpu.rst_n", "rst.rst_n_in"),
         ("one.q", "rst.locked"), ("cpu.clk", "s.aclk"),
         ("rst.rst_n", "s.aresetn"))


def test_a_bus_on_the_boards_clock_and_reset_passes(library):
    assert audit(_design(library, *BOARD)) == []


def test_a_slave_on_the_cables_clock_is_named(library):
    wiring = [w for w in BOARD if w[1] != "s.aclk"] + [("rx.pclk", "s.aclk")]
    assert audit(_design(library, *wiring)) == [
        "s.S_AXI: its clock aclk is CABLE -- a cable event would stop it "
        "answering"]


def test_a_reset_the_cable_can_assert_reaches_every_port_it_holds(library):
    """The synchroniser is asserted by either source: with the receiver's
    lock as one, the bridge and the slave both answer to the cable."""
    wiring = [w for w in BOARD if w[1] != "rst.locked"] + \
        [("rx.locked", "rst.locked")]
    assert audit(_design(library, *wiring)) == [
        "br.m_axi: its reset aresetn is CABLE -- a cable event would stop "
        "it answering",
        "br.s_axi: its reset aresetn is CABLE -- a cable event would stop it "
        "answering",
        "s.S_AXI: its reset aresetn is CABLE -- a cable event would stop it "
        "answering"]


def test_what_nothing_states_fails_as_unknown(library):
    unstated = {k: v for k, v in SLAVE.items() if k != "facts"}
    problems = audit(_design(library, *BOARD, slave=unstated))
    assert problems == ["s.S_AXI: a bus port whose clock and reset its leaf "
                        "does not state (facts.clocks)"]
    # an input nothing drives has no ancestry, and is not the board's
    wiring = [w for w in BOARD if w[1] != "s.aresetn"]
    assert audit(_design(library, *wiring)) == [
        "s.S_AXI: its reset aresetn is UNKNOWN -- a cable event would stop "
        "it answering"]
    # and what follows an unknown input is unknown, though the rest is the
    # board's: the synchroniser with its lock left undriven
    wiring = [w for w in BOARD if w[1] != "rst.locked"]
    assert audit(_design(library, *wiring)) == [
        f"{end}: its reset aresetn is UNKNOWN -- a cable event would stop "
        "it answering" for end in ("br.m_axi", "br.s_axi", "s.S_AXI")]
