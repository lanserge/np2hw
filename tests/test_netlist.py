# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A design is a netlist, and a netlist must be a buildable graph.

Each design here is a composite, written to a library and loaded by
reference -- the one way a pipeline is described. What is checked is the
graph: every block input driven exactly once, no cycle, a tap allowed and a
fork between two real datapaths refused, a port that exists.
"""
import pytest

from designs import block_ref, build, chain, describe
from np2hw.pipeline import compose

STREAM = {"bits": 8, "max_width": 64, "max_height": 32}


def _netlist(name, nodes, connections, context=None):
    """A design of ``nodes`` ({instance: block name}) wired by
    ``connections``, taking an 8-bit stream of up to 64x32 at ``in``."""
    port = {"name": "in", **STREAM}
    if context is not None:
        port["context"] = context
    return {"blocks": {name: {
        "inputs": [port], "outputs": [{"name": "out"}],
        "nodes": {inst: {"block": block_ref(b)} for inst, b in nodes.items()},
        "connections": [list(c) for c in connections]}}}


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def test_build_creates_the_declared_instances():
    """Declaration order for addresses, the context block first; a tap sits
    in the datapath where it watches."""
    pipeline = build(describe("ordered",
                              chain("gain", "meter", "offset")))
    assert [s.path for s in compose.datapath(pipeline)] == \
        ["gain", "meter", "offset"]
    assert compose.stages(pipeline)[0].path == "ctxblock"


def test_prefixed_chains_are_independent_graphs():
    pipeline = build(describe(
        "eyes",
        chain("gain", "meter", prefix="left", source="left_in",
              sink="left_out"),
        chain("gain", "meter", prefix="right", source="right_in",
              sink="right_out"),
        inputs=("left_in", "right_in"), outputs=("left_out", "right_out")))
    assert {s.path for s in compose.datapath(pipeline)} == {
        "left.gain", "left.meter", "right.gain", "right.meter"}
    # Separate instances at separate bases...
    assert (compose.stage(pipeline, "left.gain").instance.base
            != compose.stage(pipeline, "right.gain").instance.base)
    # ...and the two graphs genuinely never touch.
    for source, sink in compose.netlist(pipeline).edges:
        eyes = {e.split("_")[0].split(".")[0] for e in (str(source), str(sink))}
        assert len(eyes) == 1, f"{source} -> {sink} crosses between eyes"


def test_two_instances_of_one_block_in_one_graph():
    """Two of a thing in one datapath, distinguished by instance name."""
    pipeline = build(_netlist(
        "twice", {"g_a": "gain", "g_b": "gain"},
        [("in", "g_a.in"), ("g_a.out", "g_b.in"), ("g_b.out", "out")]))
    assert len({s.instance.base for s in compose.datapath(pipeline)}) == 2


# --------------------------------------------------------------------------- #
# Stream context
# --------------------------------------------------------------------------- #

def test_stream_context_becomes_ports_not_registers():
    """A header-carried fact gets a port and loses its register.

    When the stream itself carries a context fact, it must stop existing
    as a register entirely -- a register copy of a header fact is a second
    answer software could set to disagree with the picture arriving. What
    stays is what is genuinely software's choice.
    """
    carried = ["height", "phase", "width"]
    pipeline = build(_netlist("ctx_carried", {"g": "gain"},
                              [("in", "g.in"), ("g.out", "out")],
                              context=carried))
    owner = next(b for b in compose.register_map(pipeline)["blocks"]
                 if b["path"] == "ctxblock")
    names = {r["name"] for r in owner["registers"]}
    assert not names & set(carried), \
        f"{sorted(names & set(carried))} kept registers"
    assert "ctx_width" in compose.generate(pipeline, control=True).verilog
    # and what stays a register is what something reads: `limit` is
    # software's choice, but no block here reads it, so none is made
    assert "limit" not in names


def test_a_register_context_exists_where_a_block_reads_it():
    """A register-backed context is made where a block consumes it, and
    not where none does: the vocabulary declares, the reader brings it."""
    read = build(_netlist("ctx_read", {"b": "bias"},
                          [("in", "b.in"), ("b.out", "out")]))
    unread = build(_netlist("ctx_unread", {"g": "gain"},
                            [("in", "g.in"), ("g.out", "out")]))

    def owner(p):
        return {r["name"] for b in compose.register_map(p)["blocks"]
                if b["path"] == "ctxblock" for r in b["registers"]}
    assert "phase" in owner(read) and "phase" not in owner(unread)
    assert "limit" not in owner(read) | owner(unread)
    assert "param_ctxblock_phase" in compose.generate(
        read, control=True).verilog


def test_an_unknown_context_name_is_refused():
    """The vocabulary is the content's, closed by its blocks: a stream may
    carry only the context signals the pipeline's blocks were declared
    with."""
    with pytest.raises(KeyError, match="no pipeline context signal 'depth'"):
        build(_netlist("ctx_bad", {"g": "gain"},
                       [("in", "g.in"), ("g.out", "out")],
                       context=["depth"]))


# --------------------------------------------------------------------------- #
# The netlist must be a well-formed graph
# --------------------------------------------------------------------------- #

def test_an_undriven_input_is_refused():
    """A block input with no driver is a stream that never arrives."""
    with pytest.raises(ValueError, match="not connected"):
        build(_netlist("orphan", {"g": "gain"}, [("g.out", "out")]))


def test_two_drivers_on_one_input_are_refused():
    """Two sources on one input is a short."""
    with pytest.raises(ValueError, match="already driven"):
        build(_netlist(
            "shorted", {"a": "gain", "b": "gain"},
            [("in", "a.in"), ("in", "b.in"), ("a.out", "b.in"),
             ("b.out", "out")]))


def test_an_unconnected_pipeline_output_is_refused():
    with pytest.raises(ValueError, match="output out is driven by nothing"):
        build(_netlist("dangling", {"g": "gain"}, [("in", "g.in")]))


def test_a_cycle_is_refused():
    """A streaming pipeline must be feed-forward."""
    with pytest.raises(ValueError, match="cycle"):
        # a and b feed each other; the pipeline stream bypasses them, so every
        # input still has exactly one driver and only the cycle is wrong.
        build(_netlist(
            "loop", {"a": "gain", "b": "gain"},
            [("in", "out"), ("a.out", "b.in"), ("b.out", "a.in")]))


def test_a_tap_to_a_sink_is_allowed():
    """Statistics observe the datapath; the stream carries on past them."""
    pipeline = build(_netlist(
        "tapped", {"g": "gain", "m": "meter"},
        [("in", "g.in"), ("g.out", "out"), ("g.out", "m.in")]))
    assert "meter" in {s.paramset.block for s in compose.datapath(pipeline)}
    assert "tap: sink never stalls" in compose.generate(pipeline).verilog


def test_a_fork_between_two_real_datapaths_is_refused():
    """Both would apply backpressure, and there is no buffering fork element.

    Refusing is better than emitting a fork that deadlocks the first time one
    branch stalls -- which is a bug that only appears under load.
    """
    with pytest.raises(ValueError, match="buffering fork element"):
        # Only b reaches the output; c is a second real consumer of a.out, and
        # that alone is the problem -- both would apply backpressure.
        build(_netlist(
            "forked", {"a": "gain", "b": "gain", "c": "gain"},
            [("in", "a.in"), ("a.out", "b.in"), ("a.out", "c.in"),
             ("b.out", "out")]))


def test_an_unknown_port_lists_what_the_block_has():
    with pytest.raises(ValueError, match=r"has no output 'chroma' \(it has \['out'\]\)"):
        build(_netlist("badport", {"g": "gain"},
                       [("in", "g.in"), ("g.chroma", "out")]))


# --------------------------------------------------------------------------- #
# One composite, instantiated twice
# --------------------------------------------------------------------------- #

def _two_eyes(shared: bool) -> dict:
    """Two eyes of gain -> offset: one composite instantiated
    twice, or the same run spelled out as two."""
    description = describe(
        "twin",
        chain("gain", "offset", prefix="left",
              source="left_in", sink="left_out"),
        chain("gain", "offset", prefix="right",
              source="right_in", sink="right_out"),
        inputs=("left_in", "right_in"), outputs=("left_out", "right_out"))
    if shared:
        blocks = description["blocks"]
        blocks["eye"] = blocks.pop("left")
        del blocks["right"]
        for eye in ("left", "right"):
            blocks["twin"]["nodes"][eye]["block"] = "{lib}:eye"
    return description


def test_a_shared_composite_keeps_the_spelled_out_register_map():
    """Addresses are allocated per INSTANCE, so sharing the description
    changes nothing a host can see."""
    shared, spelled = build(_two_eyes(True)), build(_two_eyes(False))
    a, b = compose.register_map(shared), compose.register_map(spelled)
    a.pop("composites"), b.pop("composites")
    assert a == b


def test_a_shared_composite_emits_one_module_per_block():
    shared = compose.generate(build(_two_eyes(True))).module_names()
    spelled = compose.generate(build(_two_eyes(False))).module_names()
    assert "toylib_eye_gain" in shared
    assert "toylib_left_gain" in spelled and \
        "toylib_right_gain" in spelled
    assert len(shared) < len(spelled)


# --------------------------------------------------------------------------- #
# Allocation is a pure function of structure
# --------------------------------------------------------------------------- #

def test_allocation_is_a_pure_function_of_the_design():
    """Built twice, a design is the same hardware at the same addresses; and
    the order its connections are written in moves nothing -- only the
    declaration order of its nodes assigns addresses. A builder GUI writing
    the same graph in another order must get the same map."""
    import copy
    import random

    description = describe("pure", chain("gain", "meter", "offset"))
    def addresses(pipeline):
        # the map less the record of which library each build was read
        # from -- each build here is its own
        regmap = compose.register_map(pipeline)
        regmap.pop("composites")
        return regmap

    first, again = build(description), build(copy.deepcopy(description))
    assert addresses(first) == addresses(again)
    assert compose.generate(first).verilog == compose.generate(again).verilog

    shuffled = copy.deepcopy(description)
    random.Random(7).shuffle(shuffled["blocks"]["pure"]["connections"])
    assert shuffled != description
    assert addresses(build(shuffled)) == addresses(first)
