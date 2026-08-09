"""Phase-sliced models: `out[py::2, px::2] = ...` with REGISTER-valued phase.

The CFA case. A Bayer sensor interleaves four colour planes, and an operation
that differs per colour is written in NumPy the way anyone would write it -- take
every other row and column, starting where that colour starts:

    out[py::2,   px::2]   = f(x[py::2,   px::2],   coeff[0, 0])
    out[py::2,   1-px::2] = f(x[py::2,   1-px::2], coeff[0, 1])
    out[1-py::2, px::2]   = f(x[1-py::2, px::2],   coeff[1, 0])
    out[1-py::2, 1-px::2] = f(x[1-py::2, 1-px::2], coeff[1, 1])

The slice STARTS are registers, so which physical half of the image a plane
refers to is a two-bit write rather than a rebuild: one bitstream serves every
CFA order. That is the whole point -- generating a different pipeline per sensor
turns a verification matrix into sensors x modes and it does not stay green.

np2hw does NOT take the slicing literally. The four planes are disjoint and
together cover every pixel, so they lower to ONE full-rate datapath whose
coefficient is selected by the pixel's position -- an adder and a 4:1 mux, not
four quarter-rate paths and a recombiner.

Run:  python examples/cfa_phase.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H, BITS = 8, 6, 12
FULL = (1 << BITS) - 1

# Deliberately include the awkward values: zero, full scale, and either side of
# the midpoint, where a datapath that is signed by one bit too few starts reading
# positive numbers as negative.
A = np.random.default_rng(4).integers(0, FULL + 1, (H, W)).astype(np.uint16)
A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 0, FULL, 1 << (BITS - 1), (1 << (BITS - 1)) - 1


def black_level(img, py, px, offset):
    """Per-CFA-colour offset with saturation -- the black level stage.

    The SAME function is the hardware and the oracle: called with integer phases
    and a NumPy array it is ordinary NumPy, and traced it becomes the RTL.
    """
    value = img.astype(np.int32)
    out = np.empty_like(value)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            out[rows::2, cols::2] = (value[rows::2, cols::2]
                                     + offset[i, j]).clip(0, FULL)
    return out.astype(np.uint16)


def white_balance(img, py, px, gain):
    """Per-CFA-colour GAIN with truncation and saturation -- the multiply twin.

    Exercises the other coefficient kind on phase planes: a Param that
    multiplies. On an unsliced value `Param * pixel` may become a programmable
    TAP coefficient; on a phase plane it must peel into the post chain where
    the position mux can select it -- multiply, shift, clip, one datapath.
    """
    value = img.astype(np.uint32)
    out = np.empty_like(value)
    for i, rows in enumerate((py, 1 - py)):
        for j, cols in enumerate((px, 1 - px)):
            out[rows::2, cols::2] = (
                (value[rows::2, cols::2] * gain[i, j]) // 256).clip(0, FULL)
    return out.astype(np.uint16)


def main():
    params = [
        Param("py", bits=1, description="Row parity the first colour sits on"),
        Param("px", bits=1, description="Column parity the first colour sits on"),
        Param("offset", bits=16, signed=True, shape=(2, 2),
              labels=(("R", "Gr"), ("Gb", "B")),
              description="Signed offset added to the pixel before saturation"),
    ]
    offsets = {"offset_0_0": -64, "offset_0_1": 300,
               "offset_1_0": -300, "offset_1_1": 64}

    print(f"{BITS}-bit CFA phase slicing, {W}x{H}; ONE core, every phase:")
    results = []
    for phase in range(4):
        values = {"py": (phase >> 1) & 1, "px": phase & 1, **offsets}
        results.append(check(f"cfa_phase{phase}", black_level, A,
                             params=params, param_values=values, bits=BITS))

    gain_params = [
        Param("py", bits=1, description="Row parity the first colour sits on"),
        Param("px", bits=1, description="Column parity the first colour sits on"),
        Param("gain", bits=16, shape=(2, 2),
              labels=(("R", "Gr"), ("Gb", "B")), default=256,
              description="Q8.8 gain, truncated then saturated"),
    ]
    gains = {"gain_0_0": 512, "gain_0_1": 256, "gain_1_0": 199, "gain_1_1": 90}
    print("\nand the multiply twin (gain, shift, clip) through the same mux:")
    for phase in (0, 3):
        values = {"py": (phase >> 1) & 1, "px": phase & 1, **gains}
        results.append(check(f"cfa_gain{phase}", white_balance, A,
                             params=gain_params, param_values=values, bits=BITS))

    print("\nthe four planes lower to one datapath, not four:")
    from np2hw import Image2D, generate, to_ir

    _, canvas = to_ir(black_level, Image2D("img", W, H, bits=BITS), *params)
    core = generate(canvas, module_name="cfa_phase")

    # The claim, stated as something that can fail: FOUR planes, but ONE
    # accumulator and ONE coefficient mux -- not four datapaths, and no line
    # buffers, because nothing was actually decimated.
    accumulators = sum(1 for line in core.verilog.splitlines() if " acc = " in line)
    muxes = core.verilog.count("case ({sel_row, sel_col})")
    print(f"  {core['phases']} planes -> {accumulators} accumulator(s), "
          f"{muxes} coefficient mux, {core.line_buffers} line buffers")
    print(f"  coefficient selected by pixel position: "
          f"{'sel_row = erow[0] ^ param_py' in core.verilog}")
    shape_ok = (accumulators == 1 and muxes == 1 and core.line_buffers == 0
                and "sel_row = erow[0] ^ param_py" in core.verilog)

    print("\n" + ("CFA PHASE PASS" if all(results) and shape_ok else "FAIL"))
    return 0 if all(results) and shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
