"""Registers as expression leaves -- the channel-mix (matrix multiply) case.

The LUT example reaches registers through a GATHER: a data-derived index
into an array. A colour matrix reaches them the other way -- each register
is a plain factor in the arithmetic:

    a' = (a*m00 + b*m01) >> F        # m signed Q-format registers
    b' = (a*m10 + b*m11) >> F

so a register must be allowed as a LEAF of the expression DAG, multiplied
and added like any traced value. This example packs two 8-bit channels into
one 16-bit word, unpacks them with shift-and-mask, mixes through a signed
Q8.8 matrix, clips each channel and repacks -- the exact shape of a colour
correction matrix, two channels instead of three.

Checks, each a way the emitter could be wrong:

  BIT-EXACT      four matrices -- identity, channel SWAP, crosstalk with
                 NEGATIVE off-diagonal terms (the subtract path must stay
                 signed against unsigned channel data), and a saturating
                 one -- against the same NumPy function on real data.
  IDENTITY       the identity matrix reproduces the packed input exactly:
                 unpack, mix, clip and repack cancel to the bit.
  STRUCTURE      four scalar register ports and NO register array -- a
                 coefficient is a port, not a one-entry LUT.

Run:  python examples/pointwise_mix.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

BD = 8                            # bits per channel
FULL = (1 << BD) - 1
FRAC = 8                          # Q8.8 coefficients
ONE = 1 << FRAC


def mix(img, m):
    """Unpack two channels, apply a signed matrix, clip, repack."""
    value = img.astype(np.int64)
    a = value & FULL
    b = (value >> BD) & FULL
    aa = ((a * m[0, 0] + b * m[0, 1]) >> FRAC).clip(0, FULL)
    bb = ((a * m[1, 0] + b * m[1, 1]) >> FRAC).clip(0, FULL)
    return (aa + bb * (1 << BD)).astype(np.uint16)


def main():
    rng = np.random.default_rng(21)
    A = rng.integers(0, (1 << (2 * BD)), (6, 8)).astype(np.uint16)
    A[0, 0], A[0, 1] = 0, (1 << (2 * BD)) - 1          # both channels at rails

    m_param = [Param("m", bits=16, signed=True, shape=(2, 2),
                     description="Q8.8 mix matrix, row = output channel")]

    def matrix(m00, m01, m10, m11):
        return {"m_0_0": m00, "m_0_1": m01, "m_1_0": m10, "m_1_1": m11}

    identity = matrix(ONE, 0, 0, ONE)
    swap = matrix(0, ONE, ONE, 0)
    crosstalk = matrix(320, -64, -64, 320)             # negative off-diagonals
    saturating = matrix(2 * ONE, ONE // 2, ONE // 2, 2 * ONE)

    print(f"two {BD}-bit channels, one packed word, Q8.{FRAC} matrix; "
          f"one core, four matrices:")
    results = []
    for label, table in (("identity", identity), ("swap", swap),
                         ("crosstalk", crosstalk), ("saturating", saturating)):
        results.append(check(f"mix_{label}", mix, A,
                             params=m_param, param_values=table, bits=2 * BD))

    # Identity really is identity: unpack/mix/clip/repack cancel to the bit.
    oracle = mix(A, np.array([[ONE, 0], [0, ONE]]))
    ident_ok = np.array_equal(oracle, A)
    print(f"\n  identity matrix reproduces the packed input exactly: {ident_ok}")

    from np2hw import Image2D, generate, to_ir

    _, line = to_ir(mix, Image2D("img", 8, 6, bits=2 * BD), *m_param)
    core = generate(line, module_name="mix2")
    ports = [name for name, _ in core["params"]]
    arrays = core.verilog.count("_lut [")
    shape_ok = (ports == ["m_0_0", "m_0_1", "m_1_0", "m_1_1"] and arrays == 0)
    print(f"  scalar register ports {ports}, register arrays: {arrays}")

    ok = all(results) and ident_ok and shape_ok
    print("\n" + ("POINTWISE MIX PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
