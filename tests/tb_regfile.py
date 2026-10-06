# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""cocotb testbench for the control register file in front of a pipeline.

Not collected by pytest -- ``tests/test_regfile.py`` builds the design and
runs this through :func:`np2hw.testing.run_cocotb`.

Everything here is addressed the way software will address it: the testbench is
handed the emitted register map JSON and reads every address out of it. Nothing
is hardcoded, which is not a stylistic preference -- a test holding its own copy
of the addresses would keep passing after the allocator started emitting
different ones, and the map is precisely the artefact that must not be able to
drift from the hardware.

Three claims are checked, each of which is a way real hardware fails:

  * the identity word at each block's base reads back the ID and version the map
    declares, so software can prove the bitstream is the one it was built for;
  * a write is answered, lands in the shadow, and reaches the datapath only at
    the FRAME BOUNDARY -- the frame in flight when it was written finishes with
    the values it started with;
  * once committed, the datapath is bit-exact with the NumPy model, so a
    coefficient that travelled over AXI is the same coefficient the model used.
"""
from __future__ import annotations

import random

import cocotb
import numpy as np
from cocotb.clock import Clock

from np2hw.blocks.stream import StreamSpec, frame_to_beats
from np2hw.testing import (OKAY, SLVERR, AxiLiteMaster, check_framing,
                           load_case, reset_stream, run_frame)
from toylib.blocks import bias

# The wrapper consumes the full framing, unlike a bare core: its commit pulse
# is derived from in_last, so the flags are load-bearing at this boundary.
OFFER = dict(drive=("sof", "eol", "last"))


def _check_frame(collected, expected, width: int, label: str) -> None:
    assert len(collected) == expected.size, (
        f"{label}: DUT produced {len(collected)} pixels, model produced "
        f"{expected.size}")
    flat = expected.ravel()
    for i, beat in enumerate(collected):
        if beat.data != int(flat[i]):
            row, col = divmod(i, width)
            raise AssertionError(
                f"{label}: pixel {i} (row {row}, col {col}) -- DUT {beat.data}, "
                f"model {int(flat[i])}")
    # The framing law is np2hw's; asserting it through the owner keeps this
    # file from carrying a second statement of it.
    check_framing(collected, width, expected.size // width)


# --------------------------------------------------------------------------- #
# The tests
# --------------------------------------------------------------------------- #

def _blocks(case) -> dict:
    return {block["path"]: block for block in case["map"]["blocks"]}


@cocotb.test()
async def identity_words_match_the_register_map(dut):
    """Every block reports the ID and version the map says it has.

    This is the check software performs before it writes anything: a bitstream
    built from a different revision has different registers at these addresses,
    and finding that out from a wrong image is much worse than finding it out
    from a mismatched word.
    """
    case = load_case()
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    # The register file runs on its OWN clock now, and deliberately not a
    # multiple of the pixel clock: it must be a clock that keeps running
    # when the link's does not, so the testbench must not quietly make
    # them related and hide the crossing.
    cocotb.start_soon(Clock(dut.s_axil_aclk, 7, unit="ns").start())
    # ...and reset it, on its own clock. Holding aresetn high from time
    # zero looks harmless and is not: the file initialises its write
    # state machine only in the reset branch, so it would never accept a
    # write at all.
    dut.s_axil_aresetn.value = 0
    await reset_stream(dut)
    dut.s_axil_aresetn.value = 1
    axi = AxiLiteMaster(dut)
    await axi.idle()

    for path, block in _blocks(case).items():
        identity = block["id_version"]
        data, response = await axi.read(identity["address"])
        assert response == OKAY, (
            f"{path}: reading the identity word at "
            f"0x{identity['address']:04x} was answered {response:#04b}")
        assert data == identity["value"], (
            f"{path}: identity word at 0x{identity['address']:04x} reads "
            f"0x{data:08x}, map says 0x{identity['value']:08x}")
        dut._log.info(f"{path}: id_version 0x{data:08x} at "
                      f"0x{identity['address']:04x}")


@cocotb.test()
async def read_only_and_unmapped_addresses_are_refused(dut):
    """A dropped write is a bug that presents as a configuration that had no
    effect, hours later and somewhere else. The bus says so instead."""
    case = load_case()
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    # The register file runs on its OWN clock now, and deliberately not a
    # multiple of the pixel clock: it must be a clock that keeps running
    # when the link's does not, so the testbench must not quietly make
    # them related and hide the crossing.
    cocotb.start_soon(Clock(dut.s_axil_aclk, 7, unit="ns").start())
    # ...and reset it, on its own clock. Holding aresetn high from time
    # zero looks harmless and is not: the file initialises its write
    # state machine only in the reset branch, so it would never accept a
    # write at all.
    dut.s_axil_aresetn.value = 0
    await reset_stream(dut)
    dut.s_axil_aresetn.value = 1
    axi = AxiLiteMaster(dut)
    await axi.idle()

    identity = _blocks(case)["bias"]["id_version"]
    response = await axi.write(identity["address"], 0xDEADBEEF)
    assert response == SLVERR, (
        f"writing the read-only identity word at 0x{identity['address']:04x} "
        f"was answered {response:#04b}, expected SLVERR")
    data, _ = await axi.read(identity["address"])
    assert data == identity["value"], (
        "the read-only identity word changed after a refused write: "
        f"0x{data:08x}")

    unmapped = case["unmapped_address"]
    response = await axi.write(unmapped, 1)
    assert response == SLVERR, (
        f"writing unmapped 0x{unmapped:04x} was answered {response:#04b}")
    _, response = await axi.read(unmapped)
    assert response == SLVERR, (
        f"reading unmapped 0x{unmapped:04x} was answered {response:#04b}")


@cocotb.test()
async def configuration_commits_at_the_frame_boundary(dut):
    """Written over AXI, committed at the boundary, then bit-exact.

    Two frames are streamed with identical pixels and one configuration write in
    between. The first must come out at the RESET configuration -- the write
    reached the shadow, not the datapath -- and the second at the written one.
    Both are compared against the NumPy model rather than against each other, so
    "nothing changed" cannot pass for "committed correctly".
    """
    case = load_case()
    width, height, bit_depth = case["width"], case["height"], case["bit_depth"]
    spec = StreamSpec(bit_depth=bit_depth)
    blocks = _blocks(case)

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    # The register file runs on its OWN clock now, and deliberately not a
    # multiple of the pixel clock: it must be a clock that keeps running
    # when the link's does not, so the testbench must not quietly make
    # them related and hide the crossing.
    cocotb.start_soon(Clock(dut.s_axil_aclk, 7, unit="ns").start())
    # ...and reset it, on its own clock. Holding aresetn high from time
    # zero looks harmless and is not: the file initialises its write
    # state machine only in the reset branch, so it would never accept a
    # write at all.
    dut.s_axil_aresetn.value = 0
    await reset_stream(dut)
    dut.s_axil_aresetn.value = 1
    axi = AxiLiteMaster(dut)
    await axi.idle()

    offsets = case["offsets"]
    phase = case["phase"]

    # Address every register through the map, exactly as a host would.
    addresses = {r["name"]: r["address"] for r in blocks["bias"]["registers"]}
    for name, value in offsets.items():
        response = await axi.write(addresses[name], value & 0xFFFFFFFF)
        assert response == OKAY, f"writing {name} was answered {response:#04b}"
    phase_address = {r["name"]: r["address"]
                     for r in blocks["ctxblock"]["registers"]}["phase"]
    assert await axi.write(phase_address, phase) == OKAY

    # Read-back returns what was written, sign-extended to the bus width, so a
    # negative bias does not come back as a large positive number.
    for name, value in offsets.items():
        data, response = await axi.read(addresses[name])
        assert response == OKAY
        signed = data - (1 << 32) if data >> 31 else data
        assert signed == value, (
            f"{name} read back {signed}, wrote {value} "
            f"(raw 0x{data:08x} at 0x{addresses[name]:04x})")

    # ARM: say the batch is complete. The coefficients live on the
    # processor's clock and the datapath reads them from the pixel clock,
    # so they are only safe to copy while they provably cannot move --
    # which is what this bit buys. Until it is set, the datapath keeps
    # the values it has, which is why a host that forgets it sees its
    # writes read back correctly and change nothing.
    pipe_regs = {r["name"]: r["address"]
                 for r in blocks["ctxblock"]["registers"]}
    assert await axi.write(pipe_regs["commit"], 1) == OKAY

    # And while armed, a coefficient write is REFUSED rather than
    # silently dropped -- so a driver that writes into the window learns.
    refused = await axi.write(addresses["bias_0"], 0)
    assert refused == SLVERR, (
        f"a coefficient write while armed was answered {refused:#04b}; it "
        "must be refused, or a host cannot tell its write was ignored")

    # And it must CLEAR once the datapath has taken the values -- that
    # is how software knows the update landed, and it is the half of the
    # handshake that lives on the return path. Left unchecked, the ack
    # wire was simply never connected: the datapath copied correctly
    # from its own state, every frame was right, and the arm stayed set
    # forever so every later write was refused. A frame test alone
    # cannot see that.
    assert (await axi.read(pipe_regs["commit"]))[0] & 1, (
        "the arm reads back 0 immediately after being set; software has "
        "no way to tell a pending update from a finished one")

    frame = np.array(case["frame"], dtype=np.uint16).reshape(height, width)
    beats = frame_to_beats(frame, spec)

    # Frame 1: the write has not been committed, so the datapath is still at its
    # reset configuration. Compared against the model at THOSE values.
    before = bias.run(frame, phase=0, bit_depth=bit_depth)
    collected = await run_frame(dut, beats, before.size,
                                random.Random(case["seed"]), **OFFER)
    _check_frame(collected, before, width,
                 "frame 1 (written, not yet committed)")

    # A frame has gone by, so the datapath has reached a boundary and
    # acknowledged: the arm must be back to 0 and writes must flow.
    # POLL, as software does: the acknowledgement crosses two clock
    # domains, so "cleared" is a state to wait for, not to assume.
    # Bounded, because an unbounded wait on this bench has outlived the
    # hardware before.
    for _ in range(64):
        armed, _ = await axi.read(pipe_regs["commit"])
        if not (armed & 1):
            break
    assert not (armed & 1), (
        "the arm is still set after a whole frame; the datapath's "
        "acknowledgement never reached the register file, so software "
        "would poll forever and every coefficient write would be refused")

    # Frame 2: the boundary at the end of frame 1 committed the shadow.
    after = bias.run(frame, offsets, phase=phase, bit_depth=bit_depth)
    assert not np.array_equal(before, after), (
        "the test configuration does not change the output, so it cannot tell a "
        "committed write from a dropped one -- pick different biases")
    collected = await run_frame(dut, beats, after.size,
                                random.Random(case["seed"] + 1), **OFFER)
    _check_frame(collected, after, width, "frame 2 (committed)")

    dut._log.info(
        f"{after.size} pixels bit-exact after an AXI4-Lite write, "
        f"committed at the frame boundary")
