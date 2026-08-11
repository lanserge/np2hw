"""Channels the NumPy way, in BOTH directions: pixel[..., c] in, np.stack out.

The colour-matrix case. A multi-channel stream carries one data word per
pixel; a model should not know that. It receives an (H, W, C) frame,
indexes channels as anyone would, and returns np.stack([...], axis=-1) --
ordinary NumPy when it is the oracle. Traced, the channel view hands the
model each field's lane (the shift-and-mask no model writes any more), the
per-channel expression DAGs share one wire namespace, and the results
concatenate into the output word: channel 0 in the low bits, each field
the input's sample width, the layout stated exactly once at this boundary.

Run:  python examples/word_channels.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

W, H = 8, 6
FIELD = 8                      # per-channel sample width
BITS = 3 * FIELD               # the word the wire carries
FULL = (1 << FIELD) - 1

A = np.random.default_rng(17).integers(0, FULL + 1, (H, W, 3)).astype(np.uint8)
A[0, 0], A[0, 1] = (0, 0, 0), (FULL, FULL, FULL)         # rails
A[0, 2], A[0, 3] = (FULL, 0, 0), (0, 0, FULL)


def colour_mix(img, m):
    """3x3 matrix, Q?.4 coefficients, rounded, clipped -- a colour matrix.

    The SAME function is the hardware and the oracle: on an (H, W, 3) array
    the indexing, arithmetic and np.stack are ordinary NumPy; traced, each
    channel is a field lane and each output a datapath into one word.
    """
    value = img.astype(np.int64)
    channels = [value[..., k] for k in range(3)]
    outs = []
    for row in range(3):
        acc = (channels[0] * m[row, 0]
               + channels[1] * m[row, 1]
               + channels[2] * m[row, 2])
        outs.append(((acc + 8) // 16).clip(0, FULL))
    return np.stack(outs, axis=-1)


def main():
    matrix = Param("m", bits=10, signed=True, shape=(3, 3),
                   description="Colour matrix coefficient, Q?.4: row = "
                               "output channel, column = input channel")
    cases = (
        ("identity", [[16, 0, 0], [0, 16, 0], [0, 0, 16]]),
        ("swap 0 and 2", [[0, 0, 16], [0, 16, 0], [16, 0, 0]]),
        ("mix, negative off-diagonals",
         [[22, -4, -2], [-3, 24, -5], [1, -8, 23]]),
        ("saturating", [[32, 16, 0], [-16, 32, 8], [0, -32, 32]]),
    )

    print(f"{FIELD}-bit x3 channel word ({BITS}b), {W}x{H}; "
          "pixel[..., c] in, np.stack out:")
    results = []
    for label, m in cases:
        values = {f"m_{i}_{j}": m[i][j] for i in range(3) for j in range(3)}
        results.append(check(f"mix_{label.split()[0].rstrip(',')}",
                             colour_mix, A, params=[matrix],
                             param_values=values, bits=BITS, channels=3))

    print("\nthe claim, stated as something that can fail:")
    from np2hw import Image2D, generate, to_ir

    _, out = to_ir(colour_mix, Image2D("img", W, H, bits=BITS), matrix,
                   channels=3)
    core = generate(out, module_name="word_channels")
    # One wire namespace across the three channel DAGs: each input lane
    # is a FIELD TAP extracted exactly once, shared by every output
    # channel that reads it -- not re-derived per use.
    lanes = core.verilog.count(", in_data[")
    iface = core["interface"]["output"]
    print(f"  channels={core['channels']} out_bits={core.out_bits} "
          f"field_bits={iface['field_bits']} lane-taps={lanes} "
          f"expr_nodes={core['expr_nodes']}")
    shape_ok = (core["channels"] == 3 and core.out_bits == BITS
                and iface["field_bits"] == FIELD and iface["channels"] == 3
                and lanes == 3 and core.line_buffers == 0)

    print("\n" + ("WORD CHANNELS PASS" if all(results) and shape_ok else "FAIL"))
    return 0 if all(results) and shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
