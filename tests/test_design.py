# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.design: composites loaded, flattened into a netlist and checked."""
import json

import pytest

from np2hw.design import composite, is_reference
from np2hw.design.composite import SCHEMA_PATH, CompositionError
from np2hw.design.wiring import WiringError


def test_a_composite_flattens_to_its_leaves():
    design = composite.load("toylib.pipelines")
    assert list(design.instances) == ["g", "o"]
    assert ("g.out", "o.in") in design.edges
    assert design.instances["o"].values == {"offset": 5}


def test_the_enclosing_composite_sets_what_it_holds():
    design = composite.load("toylib.pipelines:wide")
    assert design.streams["in"]["bits"] == 10
    assert list(design.instances) == ["p.g", "p.o"]


def test_a_loaded_design_is_a_netlist():
    netlist = composite.load("toylib.pipelines").netlist
    assert set(netlist["instances"]) == {"g", "o"}
    assert netlist["instances"]["g"]["block"] == "toylib.blocks:gain"
    assert all(len(net["drivers"]) <= 1 for net in netlist["nets"])


def test_the_schema_names_context_and_states_no_vocabulary():
    """A stream's context is the names of its blocks' content's vocabulary;
    the format lists none of them."""
    schema = json.loads(SCHEMA_PATH.read_text())
    items = schema["$defs"]["ports"]["items"]["properties"]["context"]["items"]
    assert "enum" not in items and items["type"] == "string"


def test_a_setting_of_nothing_is_refused(tmp_path, monkeypatch):
    lib = tmp_path / "badlib"
    lib.mkdir()
    (lib / "top.json").write_text(json.dumps({"blocks": {"top": {
        "inputs": [{"name": "in"}], "outputs": [{"name": "out"}],
        "nodes": {"g": {"block": "toylib.blocks:gain",
                        "values": {"no_such": 1}}},
        "connections": [["in", "g.in"], ["g.out", "out"]]}}}))
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(CompositionError, match="no_such"):
        composite.load("badlib.top")


def test_a_net_of_two_widths_is_refused(tmp_path, monkeypatch):
    leaf = {"parameters": {"W": {}},
            "outputs": [{"name": "q", "width": "W"}],
            "vivado": {"module": "src"}}
    sink = {"inputs": [{"name": "d", "width": 4}], "vivado": {"module": "snk"}}
    lib = tmp_path / "wlib"
    lib.mkdir()
    (lib / "top.json").write_text(json.dumps({"blocks": {
        "src": leaf, "snk": sink,
        "top": {"nodes": {"a": {"block": "wlib.top:src", "build": {"W": 8}},
                          "b": {"block": "wlib.top:snk"}},
                "connections": [["a.q", "b.d"]]}}}))
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(WiringError, match="WIDTH MISMATCH"):
        composite.load("wlib.top:top")


def test_a_reference_is_a_dotted_name_not_a_file(tmp_path, monkeypatch):
    assert is_reference("toylib.pipelines:wide")
    (tmp_path / "pipelines.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert not is_reference("pipelines.json")
