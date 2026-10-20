# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.pipeline: a composite of model blocks, composed and allocated."""
import pytest

from designs import build, chain, describe

from np2hw import pipeline
from np2hw.blocks import StreamPort, block
from np2hw.blocks.params import Param, Vocabulary
from np2hw.design.composite import CompositionError, assemble
from np2hw.pipeline import compose


def test_the_context_block_comes_first_from_the_vocabulary():
    design = pipeline.load("toylib.pipelines")
    assert [s.path for s in compose.stages(design)] == ["ctxblock", "g", "o"]
    assert compose.stages(design)[0].instance.base == 0


def test_the_register_map_says_who_made_it_and_under_what_terms():
    regmap = compose.register_map(pipeline.load("toylib.pipelines"))
    assert regmap["generator"] == "np2hw"
    assert regmap["licence"] == "MIT"        # what the models' source states
    assert regmap["control"]["commit_register"] == "ctxblock.commit"


def test_modules_are_named_for_the_models_package():
    generated = compose.generate(pipeline.load("toylib.pipelines"))
    assert {"toylib_g", "toylib_o"} <= set(generated.module_names())
    head = generated.verilog.splitlines()[:2]
    assert head == ["// Copyright 2026 Serge Rabyking",
                    "// SPDX-License-Identifier: MIT"]


def test_a_register_only_context_cannot_arrive_with_the_stream():
    from toylib.blocks import VOCABULARY, gain
    design = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                      vocabulary=VOCABULARY,
                      stream={"bits": 8, "max_width": 64, "max_height": 8,
                              "context": ["limit"]})
    with pytest.raises(ValueError, match="software's choice"):
        compose.shape(design)


#: another family's pipeline context, for a design that names two
OTHER = Vocabulary((), block=None)


def test_a_design_naming_two_vocabularies_is_refused():
    description = describe("outer", chain("gain", prefix="inner"))
    description["blocks"]["inner"]["vocabulary"] = "test_pipeline:OTHER"
    with pytest.raises(CompositionError, match="has one context"):
        build(description)


def test_a_design_naming_no_vocabulary_has_no_pipeline():
    from toylib.blocks import gain
    design = assemble("t", {"g": gain}, [("in", "g.in"), ("g.out", "out")],
                      stream={"bits": 8, "max_width": 64, "max_height": 8})
    with pytest.raises(CompositionError, match="names no pipeline context"):
        compose.stages(design)