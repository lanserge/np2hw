"""Switchboard integration: the generic ready/valid core, and the SB adapter.

Two checks per filter:
  (bp) drive the GENERIC core with RANDOMIZED ready/valid backpressure
       (gaps in in_valid, downstream deasserting out_ready) -> proves the
       handshake; the edge flushes now ride the handshake, no blanking needed.
  (SB) wrap the SAME core in the Switchboard adapter and drive the SB ports.

Both validated against the SAME NumPy function. The core never mentions
Switchboard; the adapter is a separate module that just instances it.

Run:  python examples/switchboard.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check_bp, check_sb_packed
from np2hw.ir import Param

W, H = 8, 6
A = np.random.default_rng(31).integers(0, 256, (H, W), dtype=np.uint8)


def awb_gain(img, g):
    return ((g * img.astype(np.uint16)) // 16).clip(0, 255)


def gaussian2d(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def main():
    g = Param("g", np.uint8)
    print("generic core under random backpressure:")
    r = [
        check_bp("awb", awb_gain, A, params=[g], param_values={"g": 20}),
        check_bp("gauss", gaussian2d, A),
    ]
    print("\nSwitchboard adapter (SB ports, last frames the frame):")
    r += [
        check_bp("awb", awb_gain, A, params=[g], param_values={"g": 20}, iface="sb"),
        check_bp("gauss", gaussian2d, A, iface="sb"),
    ]
    print("\nAXI4-Stream Video adapter (s/m_axis, tuser=SOF, tlast=EOL):")
    r += [
        check_bp("awb", awb_gain, A, params=[g], param_values={"g": 20}, iface="axis"),
        check_bp("gauss", gaussian2d, A, iface="axis"),
    ]
    print("\nSwitchboard PACKED adapter (gearbox, many pixels per packet):")
    r += [
        check_sb_packed("awb", awb_gain, A, params=[g], param_values={"g": 20}),
        check_sb_packed("gauss", gaussian2d, A),
    ]
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
