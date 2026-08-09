# Parameters (config registers)

A `Param` is a **runtime config register**: software writes it over a control bus
(AXI-Lite/UMI) or it's driven directly, and the datapath uses its value. In the model
it's just a function argument; the tool passes a real `Param` object when tracing
(→ a register + multiplier/adder) and a NumPy-typed value when running the reference.

```python
from np2hw import Param, Params
import numpy as np
```

## Declaring registers

```python
Param("gain", np.uint8)               # 8-bit unsigned register
Param("bias", np.int16)               # 16-bit signed
Param("en",   bits=1)                 # 1-bit (bool) — see "bypass" below
Param("k",    np.int16, shape=(3, 3)) # a 3x3 matrix of registers (programmable kernel)
Param("gain", np.uint8, default=16)   # reset/power-on value (see "defaults")
```

`dtype=` sets width + signedness; or pass `bits=`/`signed=` for non-standard widths.

## Two ways to pass them to the model

### List form (a few registers) — positional args

```python
PARAMS = [Param("gain", np.uint8)]
def model(img, gain):
    return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)
```

### `Params` namespace (many registers) — one arg, by name

For a real ISP with dozens of registers, a long positional signature is unmanageable.
Declare a `Params` set and access registers **by name**:

```python
PARAMS = Params([
    Param("en",   bits=1,   default=1),    # bool
    Param("gain", np.uint8, default=16),
    Param("bias", np.int8,  default=0),
    Param("ccm",  np.int16, shape=(3, 3)),
])

def model(img, p):                          # ONE params arg
    bright = ((p.gain * img.astype(np.uint16)) // 16 + p.bias).clip(0, 255)
    return np.where(p.en, bright, img).astype(np.uint8)
```

`p.gain` is a scalar register; `p.ccm[i, j]` indexes the matrix's leaf registers.
The tool feeds a Param-valued view when tracing and a value-valued view (NumPy-typed
scalars / arrays) when running the reference, so the function stays pure. Order is
irrelevant — add/remove registers freely.

> The type of `PARAMS` selects the calling convention: a `list` → positional args,
> a `Params` → the single namespace arg. Both are fully supported.

## Defaults (reset / power-on value)

`Param(..., default=N)` sets the register's **reset value**. Without it, registers
reset to **0** — often useless at boot (`gain=0` → black, `active_width=0` → no
output). With a default the IP comes up in a sane state before any software write.
The default also seeds the live-view slider and the register-file reset
([interfaces.md](interfaces.md)).

## Matrix params = programmable kernels

A `Param(shape=(3,3))` is a *collection* of scalar registers (`k_0_0 … k_2_2`).
Indexing returns a scalar `Param`, so a programmable convolution is just:

```python
PARAMS = [Param("k", np.int16, shape=(3, 3))]
def model(img, k):
    x = img.astype(np.int16)
    return sum(k[i, j] * x[i:i+H-2, j:j+W-2] for i in range(3) for j in range(3))
```
Each tap gets a real multiplier driven by its register. (See `examples/prog_kernel.py`.)

## Bool params & bypass (`np.where`)

A **1-bit `Param`** is a boolean control. `np.where(p.en, A, B)` builds a **2:1 mux**
between two branches that share the same window — so `en=0` is a true passthrough:

```python
return np.where(p.en, processed, img).astype(np.uint8)   # en=0 -> input unchanged
```
In the live viewer a 1-bit register renders as a **checkbox** (multi-bit → slider).
(See `examples/isp/multi_param.py`, `examples/bypass.py`.)

## Live control

`np2hw view` builds a control per register and updates it **per frame** — for both the
NumPy and cxxrtl backends — so you can dial registers and watch the output react,
including through the actual RTL. See [view.md](view.md).

## Driving registers in hardware

Generate a register file so software sets these over a bus, with frame-synchronized
updates (shadow → live at the frame boundary). See [interfaces.md](interfaces.md)
(`axil_regfile`, `umi_regfile`, `control_top`). The geometry register `active_width`
for dynamic resolution rides the same mechanism
([framing-and-resolution.md](framing-and-resolution.md)).

## A note on faithfulness

Register values are passed to the NumPy reference as **NumPy-typed scalars** matching
the register dtype, so promotion matches the hardware exactly (e.g. a signed `bias`
register added to a `uint16` datapath promotes to `int32` — not a Python-int overflow).
That keeps the reference and the RTL bit-identical even for signed/negative registers.
