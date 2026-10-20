# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Rule 2: params are declared once, per block, and allocated globally.

The test of that rule is the stereo pipeline. Every block is instantiated twice,
from ONE declaration, with no parameter passed to the block and nothing
duplicated in its source -- and the two instances must land at different base
addresses with identical internal layouts. If that is awkward, the allocation is
wrong, so it is tested directly rather than assumed.
"""
from __future__ import annotations

import json

import pytest

from np2hw.blocks.params import (
    BLOCK_ALIGN,
    FIRST_PARAM_OFFSET,
    ID_VERSION_OFFSET,
    AddressAllocator,
    Param,
    ParamSet,
)
from designs import build, chain, describe

from np2hw.design.composite import assemble
from np2hw.pipeline import compose
from toylib.blocks import VOCABULARY, bias, toyblock

# bias reads the pipeline's phase: one owner, every bias bound to it
PHASE = {"bias": {"phase": "phase"}}


@pytest.fixture
def stereo():
    return build(describe(
        "stereo",
        chain("bias", "meter", prefix="left", bind=PHASE,
              source="left_in", sink="left_out"),
        chain("bias", "meter", prefix="right", bind=PHASE,
              source="right_in", sink="right_out"),
        inputs=("left_in", "right_in"), outputs=("left_out", "right_out")))


# --------------------------------------------------------------------------- #
# Instance-based allocation
# --------------------------------------------------------------------------- #

def test_the_same_block_twice_gets_two_bases(stereo):
    """The heart of rule 2: instances get addresses, blocks do not."""
    left = compose.stage(stereo, "left.bias")
    right = compose.stage(stereo, "right.bias")

    assert left.instance.base != right.instance.base
    # One declaration, shared by both instances -- not a copy.
    assert left.paramset is right.paramset is \
        bias.bind(PHASE["bias"], VOCABULARY).params


def test_both_instances_have_identical_local_layouts(stereo):
    """A block's internal layout cannot depend on where it was placed."""
    left = compose.stage(stereo, "left.bias").instance
    right = compose.stage(stereo, "right.bias").instance

    bound = bias.bind(PHASE["bias"], VOCABULARY)
    for register in bound.params.registers:
        local = bound.params.offset_of(register.name)
        assert left.address_of(register.name) == left.base + local
        assert right.address_of(register.name) == right.base + local
        assert (right.address_of(register.name)
                - left.address_of(register.name)) == right.base - left.base


def test_block_declarations_hold_no_absolute_address():
    """A block must be describable without knowing where it will live."""
    for register in bias.params.registers:
        assert register.offset < bias.params.size_bytes
    assert bias.params.offset_of("bias_0") == FIRST_PARAM_OFFSET


def test_a_block_can_be_allocated_alone_for_a_unit_test():
    """Independent instantiability, the other half of rule 2."""
    allocator = AddressAllocator()
    instance = allocator.allocate("bias", bias.params)
    assert instance.base == 0
    assert instance.address_of("bias_1") == FIRST_PARAM_OFFSET + 4


def test_duplicate_instance_paths_are_rejected():
    """Instances, not blocks, get addresses -- so one path is one instance.
    A composite cannot spell two nodes of one name; the allocator refuses
    them anyway, being the place the rule is."""
    allocator = AddressAllocator()
    allocator.allocate("left.bias", bias.params)
    with pytest.raises(ValueError, match="duplicate block instance path"):
        allocator.allocate("left.bias", bias.params)


# --------------------------------------------------------------------------- #
# Layout invariants
# --------------------------------------------------------------------------- #

def test_pipe_is_first_and_at_base_zero(stereo):
    """Convention, not a special case: the context block is allocated like
    everything else."""
    first = compose.stages(stereo)[0]
    assert first.path == "ctxblock"
    assert first.instance.base == 0
    assert first.paramset.block == "ctxblock"


def test_every_base_is_power_of_two_aligned(stereo):
    """Alignment is what makes address decode a bit-slice compare."""
    for stage in compose.stages(stereo):
        assert stage.instance.base % BLOCK_ALIGN == 0, (
            f"{stage.path} at 0x{stage.instance.base:04x} is not "
            f"{BLOCK_ALIGN}-byte aligned; decode would need a range comparator")


def test_no_two_blocks_overlap(stereo):
    spans = sorted((s.instance.base, s.instance.base + s.paramset.size_bytes, s.path)
                   for s in compose.stages(stereo))
    for (_, prev_end, prev_path), (start, _, path) in zip(spans, spans[1:]):
        assert prev_end <= start, (
            f"{prev_path} (ends 0x{prev_end:04x}) overlaps {path} "
            f"(starts 0x{start:04x})")


def test_statistics_live_in_a_separate_region(stereo):
    """Statistics are structurally different and are addressed separately."""
    config_top = max(s.instance.base + s.paramset.size_bytes for s in compose.stages(stereo))
    windows = [(base, window)
               for s in compose.stages(stereo)
               for window in s.paramset.stats
               for base in [s.instance.stats_bases[window.name]]]

    assert windows, "the stereo pipeline should contain statistics windows"
    for base, window in windows:
        assert base >= compose.allocator(stereo).stats_base > config_top
        # Naturally aligned, so the window decode is a bit-slice compare too.
        assert base % window.size_bytes == 0


def test_statistics_windows_are_double_buffered(stereo):
    """Host reads frame N while frame N+1 accumulates: two buffers, not one."""
    for stage in compose.stages(stereo):
        for window in stage.paramset.stats:
            assert window.size_bytes == window.buffer_bytes * 2
            assert window.buffer_bytes * 8 >= window.words * 4


def test_statistics_windows_do_not_overlap(stereo):
    spans = sorted(
        (s.instance.stats_bases[w.name], s.instance.stats_bases[w.name] + w.size_bytes,
         f"{s.path}.{w.name}")
        for s in compose.stages(stereo) for w in s.paramset.stats)
    for (_, prev_end, prev), (start, _, current) in zip(spans, spans[1:]):
        assert prev_end <= start, f"{prev} overlaps {current}"


# --------------------------------------------------------------------------- #
# The emitted register map
# --------------------------------------------------------------------------- #

def test_every_address_in_the_map_is_base_plus_local_offset(stereo):
    for block in compose.register_map(stereo)["blocks"]:
        for register in block["registers"]:
            assert register["address"] == block["base"] + register["offset"]
        assert block["id_version"]["address"] == block["base"] + ID_VERSION_OFFSET


def test_no_address_is_used_twice(stereo):
    seen: dict[int, str] = {}
    for block in compose.register_map(stereo)["blocks"]:
        entries = [(block["id_version"]["address"], "id_version")]
        entries += [(r["address"], r["name"]) for r in block["registers"]]
        for address, name in entries:
            where = f"{block['path']}.{name}"
            assert address not in seen, (
                f"0x{address:04x} is claimed by both {seen[address]} and {where}")
            seen[address] = where


def test_id_version_identifies_the_block_not_the_instance(stereo):
    """Software reads this to prove the bitstream matches what it was built for."""
    blocks = {b["path"]: b for b in compose.register_map(stereo)["blocks"]}
    left, right = blocks["left.bias"], blocks["right.bias"]

    assert left["id"] == right["id"]
    assert left["id_version"]["value"] == right["id_version"]["value"]
    # Distinct block types must not collide on the derived ID.
    ids = {b["block"]: b["id"] for b in blocks.values()}
    assert len(set(ids.values())) == len(ids), f"block ID collision among {ids}"
    # The word packs id in the high half and major.minor in the low half.
    assert left["id_version"]["value"] == (left["id"] << 16) | 0x0100


def test_map_carries_every_declared_description(stereo):
    """Rule 4's paper trail: descriptions reach the map, and so the docs."""
    for block in compose.register_map(stereo)["blocks"]:
        for register in block["registers"]:
            assert register["description"].strip(), (
                f"{block['path']}.{register['name']} has an empty description")


def test_map_round_trips_through_json(stereo, tmp_path):
    path = compose.write_register_map(stereo, tmp_path / "regmap.json")
    assert json.loads(path.read_text()) == compose.register_map(stereo)


def test_map_declares_enough_address_bits(stereo):
    mapping = compose.register_map(stereo)
    top = max(
        max(r["address"] for b in mapping["blocks"] for r in b["registers"]),
        max((w["base"] + w["size_bytes"]
             for b in mapping["blocks"] for w in b["statistics"]), default=0),
    )
    assert top <= (1 << mapping["address_bits"])


# --------------------------------------------------------------------------- #
# Generation follows the allocation
# --------------------------------------------------------------------------- #

def test_each_instance_generates_its_own_modules(stereo):
    """Two eyes must not accidentally share a module and therefore a register."""
    names = compose.generate(stereo).module_names()
    assert "toylib_left_bias" in names
    assert "toylib_right_bias" in names
    assert len(names) == len(set(names)), f"duplicate module names in {names}"


def test_generated_top_carries_parameter_descriptions():
    """Rule 4: the description written once must appear in the output."""
    pipeline = build(describe("isp", chain("bias", bind=PHASE)))
    verilog = compose.generate(pipeline).verilog
    # Each register port carries its declared description as a comment,
    # tagged with what that element of a shaped register means -- which the
    # index alone does not say -- and in full, including what to write.
    assert "// coarse: Signed bias ADDED to the sample" in verilog
    assert "// fine: Signed bias ADDED to the sample" in verilog
    assert "4 is one level" in verilog
    # The pipeline context keeps its description at the top level...
    assert "a fact a register or the stream may carry" in verilog
    # ...and the address map is stated where a reviewer will see it.
    assert "0x0000  ctxblock" in verilog


def test_context_is_fanned_out_not_duplicated():
    """One phase register per pipeline, not one per block: the design binds
    each block's input to it."""
    pipeline = build(describe(
        "stereo",
        chain("bias", "meter", prefix="left", bind=PHASE,
              source="left_in", sink="left_out"),
        chain("bias", "meter", prefix="right", bind=PHASE,
              source="right_in", sink="right_out"),
        inputs=("left_in", "right_in"), outputs=("left_out", "right_out")))
    mapping = compose.register_map(pipeline)

    owners = [b["path"] for b in mapping["blocks"]
              if any(r["name"] == "phase" for r in b["registers"])]
    assert owners == ["ctxblock"], (
        f"phase must exist only in the context block, found in {owners}")

    consumers = [b["path"] for b in mapping["blocks"]
                 if "phase" in b["consumes"]]
    assert "left.bias" in consumers and "right.bias" in consumers


def test_unknown_context_signal_is_rejected():
    """A block's input cannot be bound to context the vocabulary does not
    declare."""
    from np2hw.blocks import StreamPort

    @toyblock(version=(1, 0), description="placeholder",
              inputs=(StreamPort("in"),),
              outputs=(StreamPort("out"),),
              params=[Param("x", bits=8, description="placeholder")])
    def broken(pixel, p, bit_depth):
        return pixel

    with pytest.raises(ValueError, match="no_such_signal"):
        broken.bind({"x": "no_such_signal"}, VOCABULARY)