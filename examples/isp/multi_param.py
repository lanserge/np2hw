"""Many-register ISP via a Params namespace -- the scalable form for real
pipelines. The model takes ONE params arg `p` and reads registers BY NAME
(order-independent), instead of a long positional signature.

  np2hw run  examples/isp/multi_param.py in.png out.png --param gain=24 --param bias=8 --param en=1
  np2hw view examples/isp/multi_param.py --source camera   # sliders + an 'en' checkbox
"""
import numpy as np
from np2hw import Param, Params

# Declare the register set once; this IS the control/address map.
PARAMS = Params([
    Param("en",   bits=1,     default=1),     # 1-bit -> bypass checkbox in `view`
    Param("gain", np.uint8,   default=16),    # Q4.4, 16 = unity
    Param("bias", np.int8,    default=0),     # signed offset
])


def model(img, p):
    bright = ((p.gain * img.astype(np.uint16)) // 16 + p.bias).clip(0, 255)
    return np.where(p.en, bright, img).astype(np.uint8)   # en=0 -> passthrough
