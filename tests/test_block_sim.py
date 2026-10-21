# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw.testing.block: one cocotb testbench, every block, bit-exact.

Each block here is generated, simulated under Verilator with randomised
backpressure, and compared word for word and flag for flag with its own
model -- the testbench reads what to drive from the block's declarations.
"""
import shutil

import numpy as np
import pytest

from np2hw.blocks.stream import StreamSpec
from toylib.blocks import VOCABULARY, balance, bias, pair, placed, triple

pytestmark = pytest.mark.skipif(shutil.which("verilator") is None,
                                reason="verilator not on PATH")

WIDTH, HEIGHT, BIT_DEPTH = 16, 8, 10


def _simulate(tmp_path, block, ref, trials, registers=None, in_channels=1,
              bind=None):
    pytest.importorskip("cocotb")
    from np2hw.testing import run_cocotb
    variant = block.configure(registers).bind(bind, VOCABULARY)
    generated = variant.generate(
        StreamSpec(bit_depth=BIT_DEPTH, channels=in_channels), WIDTH, HEIGHT,
        module_name=f"sim_{block.name}")
    run_cocotb(tmp_path, generated.verilog, generated.top,
               "np2hw.testing.block",
               {"block": ref, "width": WIDTH, "height": HEIGHT,
                "bit_depth": BIT_DEPTH, "registers": registers, "bind": bind,
                "vocabulary": "toylib.blocks:VOCABULARY",
                "in_channels": in_channels, "trials": trials})


def _frame(seed, channels=1):
    rng = np.random.default_rng(seed)
    if channels == 1:
        return rng.integers(0, 1 << BIT_DEPTH, size=WIDTH * HEIGHT).tolist()
    spec = StreamSpec(bit_depth=BIT_DEPTH, channels=channels)
    samples = rng.integers(0, 1 << BIT_DEPTH, size=(WIDTH * HEIGHT, channels))
    return [spec.pack([int(v) for v in px]) for px in samples]


def test_a_block_with_registers(tmp_path):
    trials = [{"seed": s, "values": {"bias_0": b0, "bias_1": b1,
                                     "phase": phase}, "frame": _frame(s)}
              for s, phase, b0, b1 in ((1, 0, -40, 12), (2, 3, 100, -7))]
    _simulate(tmp_path, bias, "toylib.blocks:bias", trials)


def test_a_block_with_an_input_bound_to_context(tmp_path):
    trials = [{"seed": s, "context": {"phase": phase},
               "values": {"bias_0": b0, "bias_1": b1}, "frame": _frame(s)}
              for s, phase, b0, b1 in ((1, 0, -40, 12), (2, 3, 100, -7))]
    _simulate(tmp_path, bias, "toylib.blocks:bias", trials,
              bind={"phase": "phase"})


def test_a_variant_against_its_own_model(tmp_path):
    gains = {"gain_0_0": 6144}            # x1.5 in u4.12; the cell it reads
    _simulate(tmp_path, balance, "toylib.blocks:balance",
              [{"seed": 3, "values": gains, "frame": _frame(3)}],
              registers={"gain": {"frac": 12}})


def test_channels_in_and_out(tmp_path):
    _simulate(tmp_path / "pair", pair, "toylib.blocks:pair",
              [{"seed": 4, "frame": _frame(4)}])
    _simulate(tmp_path / "triple", triple, "toylib.blocks:triple",
              [{"seed": 5, "frame": _frame(5, channels=2)}], in_channels=2)


def test_a_history_run_frame_after_frame(tmp_path):
    """Trials of a history block are one run of frames: the first, which
    the history is still filling, owed nothing; every one after, the model
    at the place its own header set."""
    ticks = (5, 2, 7, 7, 0, 3)
    trials = [{"seed": 20 + n, "context": {"tick": t}, "frame": _frame(20 + n)}
              for n, t in enumerate(ticks)]
    _simulate(tmp_path, placed, "toylib.blocks:placed", trials,
              bind={"pos": "tick & 3"})
