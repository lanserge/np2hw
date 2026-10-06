# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A design's block design, rendered for Vivado from its JSON.

The rendering is read back with the same stubs np2hw.toolchain.vivado.bdtrace
uses, so what is checked is the design the script builds -- its cells, their
settings, its nets -- not the text of the script.
"""
import itertools
import json

import pytest

from np2hw.toolchain import vivado
from np2hw.toolchain.vivado import bdtrace
from np2hw.design.composite import load

IP = "toylib.ip"
_count = itertools.count()


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(blocks, files=None) -> str:
        """``blocks`` is a dict, or a function of the package name
        returning one, so a block can reference its own library; ``files``
        are written beside it."""
        name = f"emit_lib_{next(_count)}"
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        if callable(blocks):
            blocks = blocks(name)
        (package / "lib.json").write_text(json.dumps({"blocks": blocks}))
        for file, text in (files or {}).items():
            (package / file).write_text(text)
        return name
    return make


def render(design, tmp_path):
    out = tmp_path / "out"
    (out / "hdl").mkdir(parents=True, exist_ok=True)
    tcl = vivado.emit(design, out=out, part="xc7z020clg400-1",
                      board_part="tul.com.tw:pynq-z2:part0:1.0",
                      board_files=tmp_path, ip_repos=[tmp_path])
    script = tmp_path / "bd.tcl"
    script.write_text(tcl)
    return tcl, bdtrace.structure(bdtrace.trace(script, [str(out)]))


def test_cells_settings_pins_and_nets(library, tmp_path):
    lib = library({"top": {
        "parameters": {"n": {"default": 2}},
        "pins": [{"name": "led", "dir": "O"}],
        "nodes": {"one": {"block": f"{IP}:constant",
                          "config": {"CONST_VAL": "1"}},
                  "cat": {"block": f"{IP}:concat", "build": {"NUM_PORTS": "n"}}},
        "connections": [["one.dout", "cat.In0"], ["one.dout", "cat.In1"],
                        ["cat.dout", "led"]]}})
    tcl, s = render(load(f"{lib}.lib:top"), tmp_path)
    assert s["cells"]["one"] == {"type": "ip", "of": "vendor.com:ip:constant",
                                 "config": {"CONFIG.CONST_VAL": "1"}}
    assert s["cells"]["cat"]["config"] == {"CONFIG.NUM_PORTS": "2"}
    assert s["ports"] == {"led": "O"}
    assert sorted(s["nets"]) == [["cat/In0", "cat/In1", "one/dout"],
                                 ["cat/dout", "led"]]
    # one net, one command, however many connections made it
    assert tcl.count("connect_bd_net [get_bd_pins one/dout]") == 1


TX = {"parameters": {"implementation": {"default": "'a'"}},
      "outputs": [{"name": "LINK", "bus": "tmds"}],
      "implementations": {"a": {"vivado": {"ip": "vendor.com:ip:tx"}}}}
PACK = {"inputs": [{"name": "clk"}],
        "outputs": [{"name": "m_axis", "width": 32, "bus": "axis",
                     "signals": ["tlast"]}],
        "vivado": {"module": "pack"}, "sources": ["pack.v"]}
DMA = {"parameters": {"implementation": {"default": "'a'"}},
       "inputs": [{"name": "stream", "bus": "axis", "width": 32,
                   "signals": ["tlast"]}],
       "implementations": {"a": {"vivado": {"ip": "vendor.com:ip:dma"},
                                 "ports": {"stream": "S_AXIS"}}}}


def test_interfaces_and_external_interface_pins(library, tmp_path):
    lib = library(lambda name: {
        "tx": TX, "pack": PACK, "dma": DMA,
        "top": {"pins": [{"name": "LINK", "interface": True}],
                "nodes": {"tx": {"block": f"{name}.lib:tx"},
                          "pack": {"block": f"{name}.lib:pack"},
                          "dma": {"block": f"{name}.lib:dma"}},
                "connections": [["tx.LINK", "LINK"],
                                ["pack.m_axis", "dma.stream"]]}})
    tcl, s = render(load(f"{lib}.lib:top"), tmp_path)
    # one interface net, by the implementation's name for the block's port
    assert s["intf_nets"] == [["dma/S_AXIS", "pack/m_axis"]]
    assert s["ports"] == {"LINK": "external tx/LINK"}


PS = {"parameters": {"implementation": {"default": "'proc'"},
                     "fast": {"default": 0}},
      "outputs": [{"name": "CLK"}],
      "implementations": {"proc": {"vivado": {"tcl": "toy_ps",
                                              "source": "ps.tcl"},
                                   "config": {"FAST": "fast"}}}}
BUS = {"vivado": {"tcl": "toy_bus", "source": "bus.tcl", "phase": "late"}}
PS_TCL = """
proc toy_ps {name config} {
    create_bd_cell -type ip -vlnv vendor.com:ip:ps $name
    set props {}
    dict for {k v} $config { lappend props CONFIG.$k $v }
    set_property -dict $props [get_bd_cells $name]
}
"""
BUS_TCL = """
proc toy_bus {name spec} {
    foreach step [dict get $spec steps] {}
}
"""


def test_a_procedure_runs_in_place_and_a_late_one_after_the_nets(library,
                                                                  tmp_path):
    lib = library(lambda name: {
        "ps": PS, "bus": BUS,
        "top": {"nodes": {"ps": {"block": f"{name}.lib:ps",
                                 "build": {"fast": 1},
                                 "config": {"MHZ": "100"}},
                          "k": {"block": f"{IP}:constant"},
                          "bus": {"block": f"{name}.lib:bus",
                                  "config": {"steps": []}}},
                "connections": [["ps.CLK", "k.x"]]}},
        files={"ps.tcl": PS_TCL, "bus.tcl": BUS_TCL})
    tcl, s = render(load(f"{lib}.lib:top"), tmp_path)
    lines = tcl.splitlines()
    ps = next(i for i, l in enumerate(lines) if l.startswith("toy_ps ps"))
    net = next(i for i, l in enumerate(lines) if l.startswith("connect_bd_net"))
    bus = next(i for i, l in enumerate(lines) if l.startswith("toy_bus bus"))
    assert ps < net < bus
    # the block's settings, from its parameters, and the node's own
    assert s["cells"]["ps"]["config"] == {"CONFIG.FAST": "1",
                                          "CONFIG.MHZ": "100"}


def test_two_leaves_that_would_be_one_cell_are_refused(library, tmp_path):
    lib = library(lambda lib: {
        "jack": {"nodes": {"k": {"block": f"{IP}:constant"}}, "connections": []},
        "top": {"nodes": {"a": {"block": f"{lib}.lib:jack"},
                          "b": {"block": f"{lib}.lib:jack"}}, "connections": []}})
    with pytest.raises(vivado.EmitError, match="both be the Vivado cell 'k'"):
        render(load(f"{lib}.lib:top"), tmp_path)


STREAM = ["sof", "eol", "last"]
# a source and a sink of one stream, as their library entries state them
SOURCE = {"vivado": {"module": "src"}, "sources": ["src.v"],
          "inputs": [{"name": "clk"}],
          "outputs": [{"name": "out", "width": 8, "stream": STREAM}]}
SINK = {"vivado": {"module": "snk"}, "sources": ["snk.v"],
        "inputs": [{"name": "clk"},
                   {"name": "in", "width": 8, "stream": STREAM}]}


def _streams(tap):
    def blocks(lib):
        top = {"nodes": {"s": {"block": f"{lib}.lib:src"},
                         "d": {"block": f"{lib}.lib:snk"}},
               "connections": [["s.out", "d.in"]]}
        if tap:
            top["nodes"]["t"] = {"block": f"{IP}:concat"}
            top["connections"].append(["s.out_valid", "t.In0"])
        return {"src": SOURCE, "snk": SINK, "top": top}
    return blocks


def test_a_stream_is_one_interface_net_unless_a_pin_of_it_is_tapped(
        library, tmp_path):
    """A stream between two declared interfaces is one interface net. But
    Vivado lets a net on one pin of an interface override the interface's
    connection of it, so a stream with a tapped pin is joined signal by
    signal -- the tap on the same net as the sink, not instead of it."""
    _, s = render(load(f"{library(_streams(False))}.lib:top"), tmp_path)
    assert s["intf_nets"] == [["d/in_stream", "s/out_stream"]]
    assert not any("s/out_valid" in n for n in s["nets"])
    _, s = render(load(f"{library(_streams(True))}.lib:top"), tmp_path)
    assert s["intf_nets"] == []
    assert ["d/in_valid", "s/out_valid", "t/In0"] in s["nets"]
    assert ["d/in_data", "s/out_data"] in s["nets"]


def test_a_block_is_emitted_by_its_implementations_names(library, tmp_path):
    lib = library(lambda name: {
        "rx": {"parameters": {"implementation": {"default": "'a'"}},
               "outputs": [{"name": "q"}],
               "implementations": {"a": {
                   "vivado": {"ip": "vendor.com:ip:a"},
                   "ports": {"q": "Q_A"},
                   "config": {"MODE": "'fast'"}}}},
        "top": {"pins": [{"name": "led", "dir": "O"}],
                "nodes": {"rx": {"block": f"{name}.lib:rx"}},
                "connections": [["rx.q", "led"]]}})
    tcl, _ = render(load(f"{lib}.lib:top"), tmp_path)
    assert "create_bd_cell -type ip -vlnv vendor.com:ip:a rx" in tcl
    assert "CONFIG.MODE {fast}" in tcl
    assert "[get_bd_pins rx/Q_A]" in tcl
    assert "rx/q]" not in tcl
