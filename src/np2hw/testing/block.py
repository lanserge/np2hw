# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""THE cocotb testbench for stream blocks -- one testbench, every block.

A test builds the design, chooses stimulus, and runs this through
:func:`np2hw.testing.run_cocotb` with ``test_module="np2hw.testing.block"``.

Everything a block's testbench needs is a fact the block DECLARES -- which
registers exist and how wide they are (``params``), which pipeline context
its bound inputs read (``consumes``, against the case's ``vocabulary``), what output to expect (``Block.run``)
-- so this file reads the declarations and no block has a testbench of its
own.

The case names the block by reference (``package.module:name``); overrides
reconfigure it and bindings bind it, so a variant's RTL is checked against
the variant's own model -- one configured declaration on both sides.

What travels on ``data`` is the PACKED word (the model's own input format),
whatever the channel count. Framing is asserted by ``np2hw.testing.
check_framing`` -- the contract's owner states the law; this file does not
restate it.
"""
from __future__ import annotations

import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from np2hw.blocks.stream import StreamSpec
from np2hw.design.composite import find, find_vocabulary
from np2hw.testing import (check_framing, frame_to_beats, load_case,
                           reset_stream, run_frame)


def _diagnose(block, case, trial, i, got, want, frame):
    """Name the failing pixel in the block's own terms, from the declarations."""
    width = case["width"]
    bit_depth = case["bit_depth"]
    row, col = divmod(i, width)
    parts = [f"pixel {i} (row {row}, col {col}) -- DUT {got}, model {want}, "
             f"input {int(frame[row, col])}"]
    channels = case.get("channels", 1)
    if channels > 1:
        top = (1 << bit_depth) - 1
        unpack = lambda w: [(w >> (k * bit_depth)) & top for k in range(channels)]
        parts.append(f"per channel DUT {unpack(got)}, model {unpack(want)}")
    context = trial.get("context", {})
    if context:
        parts.append(f"context {context}")
    return "; ".join(parts)


def _model_frame(case, trial):
    """A trial's frame as the model takes it. Models speak channels ((h, w,
    c), the NumPy way); the wire speaks words, and the packing law's owner
    translates. The case states the channel count of the words it packed,
    right next to the spec the DUT was generated with -- one writer, both
    facts."""
    frame = np.array(trial["frame"], dtype=object).reshape(case["height"],
                                                           case["width"])
    in_channels = case.get("in_channels", 1)
    if in_channels == 1:
        return frame
    spec_in = StreamSpec(bit_depth=case["bit_depth"], channels=in_channels)
    return np.array([[spec_in.unpack(int(word)) for word in row]
                     for row in frame], dtype=np.int64)


@cocotb.test()
async def bit_exact_against_model(dut):
    """The generated Verilog must produce EXACTLY what the NumPy model produces.

    Not close. Identical, pixel for pixel and flag for flag, for every trial
    against ONE elaboration -- registers and context are runtime state, so a
    single bitstream must be right for all of them -- under randomised
    backpressure on both sides.
    """
    case = load_case()
    width, height = case["width"], case["height"]
    bit_depth = case["bit_depth"]
    block = find(case["block"]).obj.configure(case.get("registers")).bind(
        case.get("bind"), find_vocabulary(case["vocabulary"])
        if case.get("vocabulary") else None)
    # Raw two's complement at each register's DECLARED width -- the testbench
    # holds no mask of its own to go stale when a design overrides bits.
    reg_bits = {r.name: r.param.bits for r in block.params.registers}
    # A channel of the output word is as wide as the trace packed it -- a
    # block that narrows to a display's 8 bits packs 8-bit fields -- which
    # the core states; the input's depth is no statement of it.
    probe = block.generate(StreamSpec(bit_depth=bit_depth,
                                      channels=case.get("in_channels", 1)),
                           width, height, module_name="tb_probe")
    out_field = probe.core["interface"]["output"].get("field_bits", bit_depth)

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_stream(dut)

    # A block reading the stream's history takes the trials as ONE run of
    # frames, each trial the next: the model runs over the run, a bound
    # input's context one value a frame, and the frames the history is
    # still filling are owed no output at all.
    history = block.inputs[0].history
    expected_run = lag = None
    if history:
        trials = case["trials"]
        values0 = trials[0].get("values", {})
        if any(t.get("values", {}) != values0 for t in trials):
            raise AssertionError("a history run sets its registers once: "
                                 "every trial's values must be the same")
        names = {n for t in trials for n in t.get("context", {})}
        run_context = {n: [int(t.get("context", {}).get(n, 0))
                           for t in trials] for n in names}
        video = np.stack([_model_frame(case, t) for t in trials])
        expected_run = block.run(video, values0, bit_depth=bit_depth,
                                 **run_context)
        lag = len(trials) - len(expected_run)

    for trial_number, trial in enumerate(case["trials"]):
        rnd = random.Random(trial["seed"])
        # Python ints, not a 64-bit numpy word: a word of eight 16-bit
        # channels is 128 bits, and uint64 would not even hold it.
        frame = np.array(trial["frame"], dtype=object).reshape(height, width)
        values = trial.get("values", {})
        context = trial.get("context", {})

        for name, value in values.items():
            getattr(dut, f"param_{name}").value = (
                int(value) & ((1 << reg_bits[name]) - 1))
        # Each frame's context rides its stream: its own size, and the
        # facts a bound input's expression reads, packed as the block's
        # core carries them; the core latches it as the frame's first pixel
        # enters, and the expression itself is inside the core.
        from np2hw.stream import pack_context
        layout = block.default_context()
        dut.in_ctx.value = pack_context(layout, {
            **{name: int(context.get(name, 0)) for name, _, _ in layout},
            "width": width, "height": height})
        await RisingEdge(dut.clk)

        # Models speak channels ((h, w, c), the NumPy way); the wire speaks
        # words. The packing law's owner translates at BOTH boundaries.
        if not history:
            expected = block.run(_model_frame(case, trial), values,
                                 bit_depth=bit_depth, **context)
        elif trial_number >= lag:
            expected = expected_run[trial_number - lag]
        else:
            expected = expected_run[:0].reshape(0, *expected_run.shape[2:])
        if expected.ndim == 3:
            spec_out = StreamSpec(bit_depth=out_field,
                                  channels=expected.shape[-1])
            expected = np.array(
                [[spec_out.pack(px) for px in row]
                 for row in expected.astype(np.int64).tolist()], dtype=object)

        # The wire carries the model's own input words; the framing they get
        # is np2hw's statement, made where the flags' meaning lives.
        beats = frame_to_beats(frame.tolist())
        collected = await run_frame(dut, beats, expected.size, rnd,
                                    drive=("sof",))

        label = f"trial {trial_number} ({trial.get('label', '')}: " \
                f"context={context})"
        assert len(collected) == expected.size, (
            f"{label}: DUT produced {len(collected)} pixels, model produced "
            f"{expected.size}")
        flat = expected.ravel()
        for i, beat in enumerate(collected):
            if beat.data != int(flat[i]):
                raise AssertionError(
                    f"{label}: " + _diagnose(block, case, trial, i,
                                             beat.data, int(flat[i]), frame))
        # framed as the model's output is: a block that keeps a window
        # frames the window, not the frame it was given
        if expected.size:
            check_framing(collected, expected.shape[1], expected.shape[0])

        dut._log.info(f"{label}: {expected.size} pixels bit-exact, "
                      "framing exact")
