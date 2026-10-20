# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A block's scalar input, bound by a design to an expression.

The model is only math; what feeds an input is the design's statement. A
bound input has no register: its value is its expression, over the
pipeline context and the build parameters in scope, evaluated by the
design's one walker -- over integers as NumPy, over traced values as
hardware, so the expression is folded into the block's core and the core's
port is the context it reads.
"""
import shutil

import numpy as np
import pytest

from designs import build, chain, describe
from np2hw import saturate
from np2hw.blocks import StreamPort
from np2hw.blocks.params import Param
from np2hw.blocks.stream import StreamSpec
from np2hw.design.composite import CompositionError, assemble
from np2hw.pipeline import compose
from toylib.blocks import VOCABULARY, balance, offset, toyblock

WIDTH, HEIGHT, BITS = 16, 8, 8
SPEC = StreamSpec(bit_depth=BITS, channels=1)

requires_verilator = pytest.mark.skipif(shutil.which("verilator") is None,
                                        reason="verilator not on PATH")


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #

def test_a_bound_input_leaves_the_registers_and_its_context_is_consumed():
    bound = offset.bind({"offset": "(tick + 1) & 3"}, VOCABULARY)
    assert [p.name for p in bound.params.params] == []
    assert bound.params.consumes == ("tick",)
    assert offset.bind({"offset": "(tick + 1) & 3"}, VOCABULARY) is bound
    assert bound.variant_signature and bound.variant_signature != \
        offset.bind({"offset": "tick & 3"}, VOCABULARY).variant_signature


def test_the_numpy_run_evaluates_the_expression_over_the_context():
    frame = np.arange(WIDTH * HEIGHT, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    bound = offset.bind({"offset": "(tick + 1) & 3"}, VOCABULARY)
    for tick in range(16):
        np.testing.assert_array_equal(
            bound.run(frame, bit_depth=BITS, tick=tick),
            offset.run(frame, {"offset": (tick + 1) & 3}, bit_depth=BITS))


def test_the_core_reads_the_context_not_a_register():
    built = offset.bind({"offset": "(tick + 1) & 3"}, VOCABULARY).generate(
        SPEC, WIDTH, HEIGHT, module_name="bound_offset")
    ports = {name for name, _, _ in built.core.interface["params"]}
    assert "tick" in ports and "offset" not in ports
    assert built.context_map["tick"] == "ctx_tick"
    assert built.params == ()


def test_a_constant_folds_away_and_a_bare_signal_is_its_register():
    const = offset.bind({"offset": 5}, VOCABULARY).generate(SPEC, WIDTH, HEIGHT,
                                                module_name="const_offset")
    assert {n for n, _, _ in const.core.interface["params"]} == \
        {"frame_width", "frame_height"}
    bare = offset.bind({"offset": "tick"}, VOCABULARY).generate(SPEC, WIDTH, HEIGHT,
                                                    module_name="bare_offset")
    assert ("tick", 4, False) in bare.core.interface["params"]


def test_an_expression_the_input_cannot_hold_is_refused_by_its_range():
    with pytest.raises(ValueError, match=r"ranges over \[0, 480\]"):
        offset.bind({"offset": "tick * 32"}, VOCABULARY).generate(
            SPEC, WIDTH, HEIGHT, module_name="wide_offset")
    with pytest.raises(ValueError, match=r"ranges over \[300, 300\]"):
        offset.bind({"offset": 300}, VOCABULARY).run(np.zeros((2, 2), np.uint8),
                                         bit_depth=BITS)


def test_what_cannot_be_bound_is_refused_with_what_can():
    with pytest.raises(ValueError, match="does not declare"):
        offset.bind({"offset": "tock & 3"}, VOCABULARY)
    with pytest.raises(KeyError, match="no input 'gain'"):
        offset.bind({"gain": 1}, VOCABULARY)
    with pytest.raises(ValueError, match="only a scalar input"):
        balance.bind({"gain": 1}, VOCABULARY)
    with pytest.raises(KeyError, match="no register to set"):
        offset.bind({"offset": "tick & 3"}, VOCABULARY).run(
            np.zeros((2, 2), np.uint8), {"offset": 1}, bit_depth=BITS, tick=0)


# --------------------------------------------------------------------------- #
# The design
# --------------------------------------------------------------------------- #

def _bound(expression, context=("tick",), parameters=None, **node):
    description = describe("isp", chain("offset"), bit_depth=BITS,
                           width=WIDTH, height=HEIGHT)
    top = description["blocks"]["isp"]
    top["inputs"][0]["context"] = list(context)
    top["nodes"]["offset"]["bind"] = {"offset": expression}
    top["nodes"]["offset"].update(node)
    if parameters:
        top["parameters"] = parameters
    return description


def test_a_design_binds_an_input_and_it_has_no_register():
    pipeline = build(_bound("(tick + 1) & 3"))
    blocks = {b["path"]: b for b in compose.register_map(pipeline)["blocks"]}
    assert blocks["offset"]["registers"] == []
    assert "tick" in blocks["offset"]["consumes"]


def test_a_build_parameter_in_a_binding_is_the_integer_it_is():
    pipeline = build(_bound("(tick + step) & 3",
                            parameters={"step": {"default": 2}}))
    stage = next(s for s in compose.stages(pipeline) if s.path == "offset")
    assert [b.text for b in stage.block.bindings] == ["tick + 2 & 3"]


@requires_verilator
def test_the_hardware_is_the_model_for_every_value_of_the_context():
    from np2hw.pipeline.run import pixel_chain, run_model, run_rtl

    pipeline = build(_bound("(tick + 1) & 3"))
    stages = pixel_chain(pipeline)
    frame = np.random.default_rng(3).integers(0, 1 << BITS,
                                              size=(HEIGHT, WIDTH))
    seen = set()
    for tick in (0, 1, 2, 3, 14, 15):
        model = run_model(stages, frame, {"offset": {}}, {"tick": tick}, BITS)
        np.testing.assert_array_equal(
            run_rtl(stages, frame, {"offset": {}}, {"tick": tick}, BITS), model)
        seen.add(model.tobytes())
    assert len(seen) == 4, "the context moved nothing: no test"


def test_a_stream_fact_bound_where_the_stream_does_not_carry_it_is_refused():
    with pytest.raises(ValueError, match="must arrive with the stream"):
        compose.stages(build(_bound("tick & 3", context=())))


def test_a_bound_input_set_a_value_too_is_refused():
    with pytest.raises(CompositionError, match="no register to hold one"):
        build(_bound("tick & 3", values={"offset": 2}))


def test_a_build_parameter_is_not_bound():
    description = describe("outer", chain("offset", prefix="inner"),
                           bit_depth=BITS, width=WIDTH, height=HEIGHT)
    description["blocks"]["inner"]["parameters"] = {"step": {"default": 2}}
    description["blocks"]["outer"]["nodes"]["inner"]["bind"] = {
        "step": "tick & 3"}
    with pytest.raises(CompositionError, match="what is bound is a block"):
        build(description)


# --------------------------------------------------------------------------- #
# A phase read as bits of one input
# --------------------------------------------------------------------------- #

@toyblock(version=(1, 0), description="an offset per colour of a 2x2 tile",
          inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
          params=[Param("phase", bits=2, default=0,
                        description="bit 1 the row, bit 0 the column of the "
                                    "tile's first colour"),
                  Param("off", bits=8, signed=True, shape=(2, 2),
                        description="each colour's offset")])
def tiled(pixel, p, bit_depth):
    value = pixel.astype(np.int32)
    out = np.empty_like(value)
    row, col = p.phase >> 1, p.phase & 1
    for i, rows in enumerate((row, 1 - row)):
        for j, cols in enumerate((col, 1 - col)):
            out[rows::2, cols::2] = saturate(
                value[rows::2, cols::2] + p.off[i, j], bit_depth)
    return out.astype(np.uint16)


@requires_verilator
@pytest.mark.parametrize("bound", [False, True], ids=["register", "bound"])
def test_a_slice_starts_at_bits_of_one_input(bound):
    """`p.phase >> 1` is index math on a register: a CFA block takes the
    phase as the one value it is, whether software writes it or the design
    binds it, and the planes move with it."""
    from np2hw.pipeline.run import pixel_chain, run_model, run_rtl

    block = tiled.bind({"phase": "phase"}, VOCABULARY) if bound else tiled
    design = assemble("t", {"c": block}, [("in", "c.in"), ("c.out", "out")],
                      vocabulary=VOCABULARY, stream={"bits": BITS, "max_width": 8, "max_height": 4,
                              "context": ["phase"] if bound else []})
    stages = pixel_chain(design)
    frame = np.random.default_rng(1).integers(0, 1 << BITS, size=(4, 8))
    offsets = dict(zip(("off_0_0", "off_0_1", "off_1_0", "off_1_1"),
                       (-40, 3, 17, 90)))
    seen = set()
    for phase in range(4):
        values = {"c": dict(offsets, **({} if bound else {"phase": phase}))}
        context = {"phase": phase} if bound else {}
        model = run_model(stages, frame, values, context, BITS)
        np.testing.assert_array_equal(
            run_rtl(stages, frame, values, context, BITS), model)
        seen.add(model.tobytes())
    assert len(seen) == 4, "the phase moved nothing: no test"


def test_a_slice_start_past_its_stride_is_refused():
    @toyblock(version=(1, 0), description="a start of up to 3 on a stride 2",
              inputs=(StreamPort("in"),), outputs=(StreamPort("out"),),
              params=[Param("phase", bits=2, default=0, description="any")])
    def astray(pixel, p, bit_depth):
        out = np.zeros_like(pixel)
        out[(p.phase + 1) & 3::2, :] = pixel[(p.phase + 1) & 3::2, :]
        return out

    with pytest.raises(NotImplementedError, match="cannot hold"):
        astray.generate(SPEC, 8, 4, module_name="astray")
