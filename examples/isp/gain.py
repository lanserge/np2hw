"""Per-pixel gain (Q4.4 register) + saturate -- the AWB stage.
Run: np2hw run examples/isp/gain.py in.png out.png --param gain=24"""
import numpy as np
from np2hw import Param

PARAMS = [Param("gain", np.uint8, default=16)]   # 16 = unity (Q4.4) at power-on


def model(img, gain):
    return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)
