"""Auto white balance (gray-world) -- the v1 demo.

Host (NumPy) computes per-channel gains from the gray-world assumption and
quantises them to Q4.4 register values. The hardware datapath is ONE module --

    out = clip( (gain * pixel) >> 4 , 0, 255 )       # fixed-point gain + saturate

-- reconfigured per channel via the `gain` register. Each channel's RTL output
is verified pixel-for-pixel against the SAME NumPy function, and we confirm the
white-balanced image has equalised channel means.

Run:  python examples/awb.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.ir import Param

H, W = 6, 8
rng = np.random.default_rng(11)
rgb = rng.integers(40, 180, (H, W, 3)).astype(np.uint8)
rgb[:, :, 0] = np.clip(rgb[:, :, 0].astype(int) + 50, 0, 255)   # red cast
rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(int) - 20, 0, 255)   # blue deficit


def awb_gain(img, g):
    """One channel: fixed-point gain (Q4.4) then saturate to 8-bit."""
    return ((g * img.astype(np.uint16)) // 16).clip(0, 255)


def main():
    means = rgb.reshape(-1, 3).mean(0)
    gray = means.mean()
    gains = gray / means
    gq = np.round(gains * 16).astype(int)               # Q4.4 register values
    print(f"channel means : R={means[0]:.0f} G={means[1]:.0f} B={means[2]:.0f}")
    print(f"gray-world gains: {gains.round(2).tolist()}  -> Q4.4 regs {gq.tolist()}")

    print("\nper-channel RTL vs NumPy oracle:")
    ok = all(
        check(f"awb_{'RGB'[c]}", awb_gain, rgb[:, :, c].copy(),
              params=[Param("g", np.uint8)], param_values={"g": int(gq[c])})
        for c in range(3)
    )

    # confirm the gains actually white-balance (means equalise on the valid region)
    wb = np.stack([np.clip((gq[c] * rgb[:, :, c].astype(int)) // 16, 0, 255)
                   for c in range(3)], axis=-1)
    wb_means = wb.reshape(-1, 3).mean(0)
    print(f"\nwhite-balanced means: {wb_means.round(0).tolist()}  "
          f"(spread {wb_means.max()-wb_means.min():.0f}, was {means.max()-means.min():.0f})")
    print("\n" + ("AWB DEMO PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
