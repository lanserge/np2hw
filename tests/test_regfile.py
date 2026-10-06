# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The control register file: the emitted decode must be the emitted map.

A register map is only worth anything if the hardware decodes exactly what the
JSON says. The two are generated from one allocation, and these tests hold that
claim as something that can fail, by reading the addresses back out of the
generated Verilog and comparing them with the map.
"""
from __future__ import annotations

import re

import shutil

import numpy as np
import pytest

from designs import build, chain, describe
from np2hw.blocks.params import BLOCK_ALIGN
from np2hw.pipeline import compose

WIDTH, HEIGHT, BIT_DEPTH = 16, 8, 12

# Inside the context block's 256-byte region but past its last register:
# allocated to a block, decoded by nothing. Padding must not answer, or a typo
# in a host script would look like a successful write.
UNMAPPED_ADDRESS = 0x00FC

# Two negative biases, in quarter levels, so a frame processed with them
# differs from one processed at reset.
OFFSETS = {"bias_0": -40, "bias_1": -24}
PHASE = 0b11


@pytest.fixture
def pipeline():
    return build(describe("isp", chain("bias"),
                          bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT))


@pytest.fixture
def generated(pipeline):
    return compose.generate(pipeline)


def _flat(path: str) -> str:
    return path.replace(".", "_")


def _decoded_writes(verilog: str) -> dict[str, int]:
    """``{register name: word index}`` from the register file's write decode.

    The prefix is whatever the emitter uses for the bank software writes,
    which depends on whether the file keeps a second one. What this test
    is about is the ADDRESS each name decodes at, so it should not fail
    when that decision changes.
    """
    return {name: int(word) for word, name
            in re.findall(
                r"^\s+(\d+): (?:if \([^)]*\) )?begin (?:shadow|reg)_(\w+) <=",
                verilog, re.M)}


def _decoded_constants(verilog: str) -> dict[int, int]:
    """``{word index: constant}`` for the read-only words in the read decode."""
    return {int(word): int(value, 16) for word, value
            in re.findall(r"^\s+(\d+): begin s_axil_rdata <= 32'h([0-9a-f]{8});",
                          verilog, re.M)}


# --------------------------------------------------------------------------- #
# The decode is the map
# --------------------------------------------------------------------------- #

def test_every_register_decodes_at_the_address_the_map_publishes(pipeline, generated):
    """The claim the whole control plane rests on, stated so it can fail."""
    decoded = _decoded_writes(generated.verilog)
    expected = {}
    for block in compose.register_map(pipeline)["blocks"]:
        for register in block["registers"]:
            expected[f"{_flat(block['path'])}_{register['name']}"] = \
                register["address"] // 4

    assert decoded == expected, (
        "the emitted decode and the emitted register map disagree; software "
        "reading the map would write to the wrong register")


def test_identity_words_are_read_only_constants(pipeline, generated):
    """No storage, no write path -- the value is wired into the decode."""
    constants = _decoded_constants(generated.verilog)
    decoded = _decoded_writes(generated.verilog)

    for block in compose.register_map(pipeline)["blocks"]:
        identity = block["id_version"]
        word = identity["address"] // 4
        assert constants.get(word) == identity["value"], (
            f"{block['path']}: identity word at 0x{identity['address']:04x} is "
            f"not emitted as the constant 0x{identity['value']:08x}")
        assert f"{_flat(block['path'])}_id_version" not in decoded, (
            f"{block['path']}: the identity word has a write path; it is "
            "read-only")


def test_each_block_lands_on_its_own_aligned_region(pipeline):
    """What makes address decode a bit-slice compare rather than a comparator."""
    bases = [block["base"] for block in compose.register_map(pipeline)["blocks"]]
    assert len(bases) == len(set(bases))
    for base in bases:
        assert base % BLOCK_ALIGN == 0, f"0x{base:04x} is not {BLOCK_ALIGN}-aligned"


def test_the_unmapped_probe_address_really_is_unmapped(pipeline):
    """Keeps the simulation test honest if the allocation ever grows into it."""
    used = {register["address"]
            for block in compose.register_map(pipeline)["blocks"]
            for register in block["registers"]}
    used |= {block["id_version"]["address"]
             for block in compose.register_map(pipeline)["blocks"]}
    assert UNMAPPED_ADDRESS not in used


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_the_control_top_wraps_the_datapath_top(pipeline, generated):
    names = generated.module_names()
    assert generated.top == f"{pipeline.name}_ctrl"
    assert pipeline.name in names, "the datapath top must still be emitted"
    assert generated.meta["control"]["bus"] == "axi4-lite"


def test_the_datapath_can_still_be_generated_without_a_register_file(pipeline):
    """A bit-exact testbench drives coefficients directly; it wants the datapath."""
    plain = compose.generate(pipeline, control=False)
    assert plain.top == pipeline.name
    assert "s_axil_awaddr" not in plain.verilog
    assert "control" not in plain.meta


def test_context_reaches_the_datapath_from_its_block_not_from_a_copy(
        pipeline, generated):
    """Rule 2: one width register in a pipeline, fanned out.

    The context ports of the datapath top are driven by the context block's
    registers, so a resolution change is one write. A per-block copy would
    show up here as a context port driven by a register belonging to the
    consuming block.
    """
    bind = compose.control_bind(pipeline, {})
    assert bind["ctx_width"] == "param_ctxblock_width"
    assert bind["ctx_phase"] == "param_ctxblock_phase"
    for port, driver in bind.items():
        if port.startswith("ctx_"):
            assert driver.startswith("param_ctxblock_"), (
                f"{port} is driven by {driver}, which is not the context "
                "block; context has one owner")


def test_the_map_describes_the_commit_contract_the_rtl_implements(pipeline, generated):
    """The map and the decode must be one document -- for BEHAVIOUR too.

    The commit contract changed once (automatic at the frame boundary ->
    armed by software) and for a while the map kept describing the old
    one: a host reading it would write coefficients and wait forever for
    a commit that no longer happens by itself. Addresses were already
    held together by the tests above; this holds the SEMANTICS together,
    by checking the map's claims against the emitted RTL rather than
    against anyone's memory.
    """
    control = compose.register_map(pipeline)["control"]
    assert control["commit"] == "armed"

    # The register the map points at must exist at the address it says.
    block_path, reg_name = control["commit_register"].split(".")
    regs = {f"{b['path']}.{r['name']}": r
            for b in compose.register_map(pipeline)["blocks"] for r in b["registers"]}
    assert control["commit_register"] in regs

    # And the RTL must actually implement the two halves of the claim:
    # coefficient writes gated on that register, and the acknowledgement
    # clearing it. Grepping the generated Verilog is deliberate -- it is
    # the artifact a host talks to, not the intent.
    flat = f"reg_{block_path}_{reg_name}"
    assert f"if (!{flat}) begin" in generated.verilog, (
        "the map says writes are refused while committed, but no write "
        "in the decode is gated on the commit register")
    assert f"if (ack_s1) {flat} <= 1'b0;" in generated.verilog, (
        "the map says the datapath clears the commit, but nothing in "
        "the RTL clears it")

    # The caveat is part of the contract: reset publishes the bank,
    # uncommitted writes included. If the emission ever changes to make
    # that false, the map must stop warning about it -- so its presence
    # is tied to the rst-load in the RTL.
    assert "commit_caveat" in control
    assert "if (rst || " in generated.verilog


def test_a_signed_register_reads_back_sign_extended(generated):
    """A host reading a negative bias must not see a large positive number."""
    assert re.search(
        r"s_axil_rdata <= \{\{22\{(?:shadow|reg)_bias_bias_0\[9\]\}\}",
        generated.verilog), (
        "the signed bias register is not sign-extended on read-back")


def test_a_declared_but_unbuilt_block_says_so_in_both_artefacts():
    """`meter` has registers and no datapath; neither the map nor the RTL hides it.

    The registers are still decoded, because the map and the decode have to be
    one document. What must not happen is software writing a coefficient, seeing
    it read back, and having no way to know it reached nothing.
    """
    pipeline = build(describe("isp", chain("bias", "meter"),
                              bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT))
    blocks = {block["path"]: block for block in compose.register_map(pipeline)["blocks"]}
    assert blocks["bias"]["implemented"] is True
    assert blocks["ctxblock"]["implemented"] is True, \
        "the context block has no datapath to build"
    assert blocks["meter"]["implemented"] is False
    assert blocks["meter"]["not_implemented_reason"]

    verilog = compose.generate(pipeline).verilog
    # read as prose: a comment wraps wherever its line runs out
    prose = re.sub(r"\n\s*// ", " ", verilog)
    assert "meter is declared but not built yet" in prose
    # And it is still decoded, at the address the map publishes.
    decoded = _decoded_writes(verilog)
    for register in blocks["meter"]["registers"]:
        assert decoded[f"meter_{register['name']}"] == register["address"] // 4


def test_a_register_description_reaches_the_register_file(generated):
    """Rule 4, at the control plane: the description is written once."""
    assert "// bias.bias_0 (Q8.2). coarse: Signed bias ADDED" in generated.verilog
    assert "4 is one level" in generated.verilog


# --------------------------------------------------------------------------- #
# np2hw refuses an allocation it cannot decode
# --------------------------------------------------------------------------- #

def test_two_registers_at_one_address_are_refused():
    from np2hw import Reg, axil_regfile

    with pytest.raises(ValueError, match="both claim offset"):
        axil_regfile([Reg("a", 8, offset=0x10), Reg("b", 8, offset=0x10)],
                     addr_bits=8)


def test_an_unaligned_offset_is_refused():
    from np2hw import Reg, axil_regfile

    with pytest.raises(ValueError, match="not word-aligned"):
        axil_regfile([Reg("a", 8, offset=0x02)], addr_bits=8)


def test_an_offset_outside_the_decoded_space_is_refused():
    from np2hw import Reg, axil_regfile

    with pytest.raises(ValueError, match="outside the"):
        axil_regfile([Reg("a", 8, offset=0x400)], addr_bits=8)


def test_an_unbound_parameter_port_is_refused(pipeline):
    """Every port needs a driver, and the error says what is on offer."""
    from np2hw import control_wrap

    with pytest.raises(KeyError, match="not bound"):
        control_wrap({"module": "m",
                      "interface": {"clock": "clk", "reset": "rst",
                                    "param_prefix": "", "streams": [],
                                    "params": [("param_x", 8, False)]}},
                     compose.registers(pipeline), bind={}, frame_sync=False)


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not on PATH")
def test_registers_written_over_axi_reach_the_datapath(tmp_path, generated,
                                                       pipeline):
    """The end-to-end claim: a host writes the map, the image changes to match.

    Everything the testbench needs is in the emitted register map, because that
    is all a host will have.
    """
    pytest.importorskip("cocotb")
    from pathlib import Path

    from np2hw.testing import run_cocotb
    frame = np.random.default_rng(20260808).integers(
        0, 1 << BIT_DEPTH, size=(HEIGHT, WIDTH))
    case = {
        "width": WIDTH, "height": HEIGHT, "bit_depth": BIT_DEPTH,
        "map": compose.register_map(pipeline),
        "unmapped_address": UNMAPPED_ADDRESS,
        "offsets": OFFSETS,
        "phase": PHASE,
        "frame": frame.tolist(),
        "seed": 20260808,
    }
    run_cocotb(tmp_path, generated.verilog, generated.top, "tb_regfile", case,
               test_dir=Path(__file__).parent)


# --------------------------------------------------------------------------- #
# Presentation: what a control surface reads off the map
# --------------------------------------------------------------------------- #

def test_widget_is_derived_from_the_declaration():
    """One rule, in one place: named values are a choice, one bit is a tick
    box, fractional bits are a typed value, the rest is an integer slider."""
    from np2hw.blocks.params import Param

    assert Param("en", bits=1, description="x").widget == "checkbox"
    assert Param("gain", bits=16, frac=8, description="x").widget == "fixed"
    assert Param("rows", bits=12, description="x").widget == "slider"
    assert Param("mode", bits=2, default=0,
                 choices=((0, "off"), (1, "bars"), (2, "ramp")),
                 description="x").widget == "choice"


def test_choices_that_lie_are_refused():
    """The declaration is what the UI shows a person; contradictions and
    ambiguities must fail at declaration, not on someone's screen."""
    from np2hw.blocks.params import Param

    with pytest.raises(ValueError, match="not a quantity"):
        Param("q", bits=8, frac=4, default=0,
              choices=((0, "a"), (16, "b")), description="x")
    with pytest.raises(ValueError, match="nothing to choose"):
        Param("one", bits=2, default=0, choices=((0, "only"),),
              description="x")
    with pytest.raises(ValueError, match="unique"):
        Param("dup", bits=2, default=0,
              choices=((0, "same"), (1, "same")), description="x")
    with pytest.raises(ValueError, match="reset would select"):
        Param("stray", bits=2, default=3,
              choices=((0, "a"), (1, "b")), description="x")
    with pytest.raises(ValueError, match="representable"):
        Param("big", bits=2, default=0,
              choices=((0, "a"), (9, "b")), description="x")


def test_the_map_states_every_register_presentation():
    """min/max/widget ride the map, so a UI never re-derives two's
    complement in JavaScript -- the map is generated output and states
    what one rule derived."""
    built = build(describe(
        "present", chain("bias", "balance"),
        bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT))
    blocks = {b["path"]: {r["name"]: r for r in b["registers"]}
              for b in compose.register_map(built)["blocks"]}

    gain = blocks["balance"]["gain_0_0"]
    assert (gain["widget"], gain["min"], gain["max"]) == ("fixed", 0, 65535)

    bias = blocks["bias"]["bias_0"]
    assert (bias["widget"], bias["min"], bias["max"]) == ("fixed", -512, 511)

    width = blocks["ctxblock"]["width"]
    assert (width["widget"], width["min"], width["max"]) == (
        "slider", 0, 65535)

    commit = blocks["ctxblock"]["commit"]
    assert commit["widget"] == "checkbox"
