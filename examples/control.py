"""Control plane: program the config registers (Param) over a memory-mapped bus,
then stream a frame. Two bus variants:
  AXI4-Lite -- pairs with the AXI4-Stream Video data path (real silicon / standard)
  UMI       -- the Switchboard/ZeroAsic memory-mapped control (simplified model)

The test WRITES the gain register over the bus, then streams the image, and
checks the output equals the NumPy oracle for that gain -- i.e. the control
interface really sets what the datapath uses.

Run:  python examples/control.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check_ctrl
from np2hw.ir import Param

W, H = 8, 6
A = np.random.default_rng(44).integers(0, 256, (H, W), dtype=np.uint8)


def awb(img, g):
    """Pointwise gain (Q4.4) + saturate -- the AWB stage, software-programmable."""
    return ((g * img.astype(np.uint16)) // 16).clip(0, 255)


def gain_bias(img, g, b):
    """gain register THEN bias register (multiplicative + additive Param)."""
    return (((g * img.astype(np.uint16)) // 16) + b).clip(0, 255)


def main():
    g = Param("gain", np.uint8)
    b = Param("bias", np.uint8)
    print("program config over the control bus, then stream:")
    r = [
        check_ctrl("awb", awb, A, [g], {"gain": 24}, ctrl="axil"),
        check_ctrl("awb", awb, A, [g], {"gain": 24}, ctrl="umi"),
        # 2-entry register file: gain (mul) @0x00 + bias (add) @0x04
        check_ctrl("gb", gain_bias, A, [g, b], {"gain": 20, "bias": 12}, ctrl="axil"),
        check_ctrl("gb", gain_bias, A, [g, b], {"gain": 20, "bias": 12}, ctrl="umi"),
    ]
    print("\nframe-synced shadow registers (update at frame boundary):")
    r += [
        check_ctrl("awb", awb, A, [g], {"gain": 24}, ctrl="axil", frame_sync=True),
        check_ctrl("awb", awb, A, [g], {"gain": 24}, ctrl="umi", frame_sync=True),
    ]
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
