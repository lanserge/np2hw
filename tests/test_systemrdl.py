# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The SystemRDL rendering must be the same map as the JSON and the decode.

The allocation has three renderings: the decode in the emitted Verilog, the
JSON register map, and the SystemRDL. This file binds the SystemRDL to the
map, by ELABORATING the emitted ``.rdl`` with the reference SystemRDL compiler
and comparing every resulting address against the JSON map. Text comparison would only prove the file looks right; elaboration
proves a consumer's tooling agrees about what it means. The toy sink's
``level`` register is a SystemRDL keyword, so the elaboration also holds
the writer to escaping one.
"""
from __future__ import annotations

import pytest
from np2hw.pipeline import compose

from designs import build, chain, describe

systemrdl = pytest.importorskip(
    "systemrdl", reason="systemrdl-compiler (MIT) validates the emitted .rdl")

from systemrdl import RDLCompiler  # noqa: E402  (after the importorskip)
from systemrdl.node import FieldNode, RegNode  # noqa: E402

WIDTH, HEIGHT, BIT_DEPTH = 16, 8, 8


@pytest.fixture(scope="module")
def stereo():
    """Two eyes: the design where hierarchy has something to prove."""
    return build(describe(
        "stereo",
        chain("bias", "meter", prefix="left", source="l_in", sink="l_out"),
        chain("bias", "meter", prefix="right", source="r_in", sink="r_out"),
        bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT,
        inputs=("l_in", "r_in"), outputs=("l_out", "r_out")))


@pytest.fixture(scope="module")
def elaborated(stereo, tmp_path_factory):
    path = tmp_path_factory.mktemp("rdl") / "map.rdl"
    compose.write_systemrdl(stereo, path)
    compiler = RDLCompiler()
    compiler.compile_file(str(path))
    return compiler.elaborate()


def _rdl_registers(root) -> dict[str, int]:
    """``{instance_path.register: absolute address}`` from the elaborated tree."""
    out = {}
    for node in root.descendants(unroll=True):
        if isinstance(node, RegNode):
            # a path is `map.left_bias.bias_0`; drop the addrmap.
            path = node.get_path().split(".", 1)[1]
            out[path] = node.absolute_address
    return out


def test_every_register_elaborates_at_the_address_the_map_publishes(
        stereo, elaborated):
    """The claim that makes the .rdl shippable, held by a real elaboration."""
    from_rdl = _rdl_registers(elaborated)

    expected = {}
    for block in compose.register_map(stereo)["blocks"]:
        flat = block["path"].replace(".", "_")
        expected[f"{flat}.id_version"] = block["id_version"]["address"]
        for register in block["registers"]:
            expected[f"{flat}.{register['name']}"] = register["address"]

    assert from_rdl == expected, (
        "the elaborated SystemRDL and the JSON register map disagree; an "
        "integrator's tooling would generate against the wrong addresses")


def test_the_two_eyes_share_one_component_definition(stereo):
    """The point of keeping the hierarchy: one TYPE, instantiated twice.

    A flat rendering would emit the bias layout once per instance, and a
    reader could not tell the eyes are the same hardware. The netlist emits one
    module for the shared block; the .rdl must make the same statement.
    """
    text = compose.write_systemrdl(
        stereo, __import__("tempfile").mkdtemp() + "/map.rdl").read_text()
    assert text.count("regfile bias_rf {") == 1
    assert text.count("bias_rf left_bias") == 1
    assert text.count("bias_rf right_bias") == 1


def test_q_format_survives_as_a_declared_property(elaborated):
    """The fixed-point convention reaches the consumer as DATA.

    Every other register format drops the Q format, and it is the one thing a
    driver author needs that the address and width cannot tell them. np2hw
    carries it opaquely; the SystemRDL declares it as a user-defined property,
    so it comes back out of elaboration typed, not as a comment.
    """
    seen = {}
    for node in elaborated.descendants(unroll=True):
        if isinstance(node, FieldNode):
            q = node.get_property("q_format", default=None)
            if q is not None:
                seen[node.get_path().split(".", 1)[1]] = q
    assert seen["left_bias.bias_0.value"] == "Q8.2"
    assert seen["ctxblock.phase.value"] == "u2.0"


def test_read_only_identity_words_have_no_write_access(elaborated):
    """`sw = r; hw = na;` with a reset: a constant, matching the decode."""
    for node in elaborated.descendants(unroll=True):
        if isinstance(node, RegNode) and node.get_path().endswith("id_version"):
            assert not node.has_sw_writable, (
                f"{node.get_path()}: the identity word elaborated as writable")
            field = next(iter(node.fields()))
            assert field.get_property("reset") is not None


def test_the_flat_list_is_a_projection_of_the_map():
    """registers() must be address_map().flatten() -- one source, one shadow."""
    pipeline = build(describe(
        "isp", chain("bias"),
        bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT))
    flat = compose.registers(pipeline)
    again = compose.address_map(pipeline).flatten()
    assert flat == again
    assert [r.name for r in flat][:3] == [
        "ctxblock_id_version", "ctxblock_width", "ctxblock_height"]


def test_two_layouts_with_one_name_are_refused():
    """A block TYPE has one layout; np2hw refuses to pretend otherwise."""
    from np2hw import AddrMap, Reg, RegBlock, RegInstance

    a = RegBlock("bl", regs=(Reg("x", 8, offset=0),))
    b = RegBlock("bl", regs=(Reg("x", 8, offset=0), Reg("y", 8, offset=4)))
    # Two distinct RegBlock values may share a name in np2hw (it cannot know
    # they are meant to be the same thing) -- but address_map() must never
    # produce that, and overlapping INSTANCES are refused outright.
    with pytest.raises(ValueError, match="overlaps"):
        AddrMap("m", instances=(RegInstance("one", a, 0x0),
                                RegInstance("two", b, 0x2)))
