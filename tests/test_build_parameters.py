# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Build-time register overrides: one declaration, many fixed-point variants.

The principle under test: a block's Q format and widths live on its Param
declarations and NOWHERE else -- the model reads its shift from the configured
declaration, the register map derives q_format from it, the host quantises
through it. A design overriding `frac` therefore changes the model, the RTL,
the map and the docs together, or not at all. These tests hold each limb of
that claim, plus the two failure modes designed against from the start: two
variants merging into one type, and the round trip losing the overrides.
"""
import shutil

import numpy as np
import pytest

from designs import build, chain, describe
from np2hw.blocks import resolve
from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from toylib.blocks import VOCABULARY, balance

WIDTH, HEIGHT, BIT_DEPTH = 16, 8, 12

requires_verilator = pytest.mark.skipif(shutil.which("verilator") is None,
                                        reason="verilator not on PATH")


def _design_with_override(frac=12, name="isp"):
    description = describe(name, chain("bias", "balance"),
                           bit_depth=BIT_DEPTH, width=WIDTH, height=HEIGHT)
    node = description["blocks"][name]["nodes"]["balance"]
    node["build"] = {"gain.frac": frac}
    return description


# --------------------------------------------------------------------------- #
# The variant mechanism
# --------------------------------------------------------------------------- #

def test_identical_overrides_share_one_variant_object():
    """Object identity IS type identity downstream; the cache makes it hold."""
    a = balance.configure({"gain": {"frac": 12}})
    b = balance.configure({"gain": {"frac": 12}})
    c = balance.configure({"gain": {"frac": 10}})
    assert a is b and a is not c
    assert balance.configure(None) is balance


def test_unity_default_moves_with_the_overridden_frac():
    """Reset must stay a pass-through in EVERY variant, or an unconfigured
    Q4.12 pipeline would silently apply x16."""
    variant = balance.configure({"gain": {"frac": 12}})
    gain = next(p for p in variant.params if p.name == "gain")
    assert (gain.q_format, gain.default) == ("u4.12", 4096)

    frame = np.full((4, 4), 777, dtype=np.uint16)
    out = variant(frame, variant.params.bind({}),
                  BIT_DEPTH)
    np.testing.assert_array_equal(out, frame)


def test_the_model_reads_the_declaration_not_a_module_constant():
    """x1.5 encodes differently per variant; the result must not."""
    frame = np.full((2, 2), 100, dtype=np.uint16)
    for frac in (8, 10, 12):
        variant = balance.configure({"gain": {"frac": frac}})
        one_and_a_half = (1 << frac) + (1 << (frac - 1))
        bound = variant.params.bind(variant.params.declaration("gain")
                                    .values(np.full((2, 2), one_and_a_half)))
        out = variant(frame, bound, BIT_DEPTH)
        assert out[0, 0] == 150, f"frac={frac}"


def test_refused_attributes_name_what_is_allowed():
    with pytest.raises(ValueError, match="allows \\['bits', 'frac'\\]"):
        balance.configure({"gain": {"signed": 1}})
    with pytest.raises(KeyError, match="declares no register"):
        balance.configure({"volume": {"bits": 8}})


# --------------------------------------------------------------------------- #
# Through the design JSON
# --------------------------------------------------------------------------- #

@requires_verilator
def test_a_design_override_reaches_the_map_the_rtl_and_the_docs():
    rng = np.random.default_rng(5)
    pipeline = build(_design_with_override(frac=12))
    blocks = {b["path"]: b for b in compose.register_map(pipeline)["blocks"]}
    wb = blocks["balance"]
    assert wb["register_overrides"] == {"gain": {"frac": 12}}
    formats = {r["name"]: r["q_format"] for r in wb["registers"]}
    assert formats["gain_0_0"] == "u4.12"
    assert {r["name"]: r["default"] for r in wb["registers"]}["gain_0_0"] == 4096

    # The RTL, checked by what it computes rather than by how a compiler
    # happens to spell it: a gain of 1.5 in the overridden u4.12 must give
    # exactly the model's output, which it does only if the shift of 12
    # reached the hardware.
    from np2hw.pipeline.run import pixel_chain, run_model, run_rtl
    stages = pixel_chain(pipeline)
    one_and_a_half = (1 << 12) + (1 << 11)
    gains = {r["name"]: one_and_a_half for r in wb["registers"]
             if r["name"].startswith("gain_")}
    values = {path: {r["name"]: r["default"] for r in block["registers"]}
              for path, block in blocks.items()}
    values["balance"].update(gains)
    context = {"phase": 0}
    frame = rng.integers(0, 1 << BIT_DEPTH, size=(HEIGHT, WIDTH))
    model = run_model(stages, frame, values, context, BIT_DEPTH)
    rtl = run_rtl(stages, frame, values, context, BIT_DEPTH)
    np.testing.assert_array_equal(rtl, model)
    assert (model != frame).any(), "a gain of 1.5 changed nothing: no test"


def test_two_variants_of_one_block_are_two_types_not_one():
    """The map must never say two different Q formats are the same hardware."""
    # Two parallel balance instances, distinct fracs, built by hand.
    pipeline = assemble(
        "two",
        {"wb_a": resolve("toylib", "balance").configure({"gain": {"frac": 12}}),
         "wb_b": resolve("toylib", "balance")},
        [("a_in", "wb_a.in"), ("wb_a.out", "a_out"),
         ("b_in", "wb_b.in"), ("wb_b.out", "b_out")],
        inputs=("a_in", "b_in"), outputs=("a_out", "b_out"),
        vocabulary=VOCABULARY, stream={"bits": BIT_DEPTH, "max_width": WIDTH, "max_height": HEIGHT})
    compose.netlist(pipeline)

    addrmap = compose.address_map(pipeline)
    names = {inst.path: inst.block.name for inst in addrmap.instances}
    assert names["wb_a"] == "balance__gain_frac12"
    assert names["wb_b"] == "balance"
    assert names["wb_a"] != names["wb_b"], "variants merged into one type"

    formats = {}
    for block in compose.register_map(pipeline)["blocks"]:
        for register in block["registers"]:
            formats[(block["path"], register["name"])] = register["q_format"]
    assert formats[("wb_a", "gain_0_0")] == "u4.12"
    assert formats[("wb_b", "gain_0_0")] == "u8.8"


def test_an_undeclared_override_fails_at_build_naming_the_rules():
    description = _design_with_override()
    description["blocks"]["isp"]["nodes"]["balance"]["build"] = \
        {"gain.default": 999}
    with pytest.raises(ValueError, match="allows \\['bits', 'frac'\\]"):
        build(description)
