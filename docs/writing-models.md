# Writing models

A **model** is a plain Python file containing a NumPy function. The same function is
compiled to hardware *and* run as the validation reference, so it must be ordinary,
traceable NumPy.

## The model-file convention

```python
# my_isp.py
import numpy as np
from np2hw import Param, Params          # only if the model has config registers

PARAMS = [Param("gain", np.uint8)]        # optional; omit if no registers

def model(img, gain):                     # `img` first, then one arg per register
    return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)
```

- The function is named **`model`** or **`isp`** (or pick any with `file.py:func`).
- **`PARAMS`** declares the config registers — a `list` of `Param`, or a `Params`
  namespace (see [parameters.md](parameters.md)). Omit it entirely if there are none.
- There are **no built-in model names**; the CLI always takes a file. Ready-made
  examples live in [`examples/isp/`](../examples/isp) (gaussian, blur, sharpen, gain,
  multi_param) — each is a complete, self-contained model file.

## How tracing works (mental model)

`np2hw` runs your function once on a traced stand-in for `img`. NumPy operators and
`__array_function__`/`__array_ufunc__` record the operations into a graph, which is
flattened into a **weighted tap map** `{(row, col): coeff}` and lowered to a line IR.
From there: horizontal extent → a shift register, vertical extent → line buffers, and
the delay lines are counted automatically. (Details in
[`design/02`](../design/02-stream-array-tracing.md) and
[`design/03`](../design/03-hash-cons-delay-lines.md).)

The practical consequence: **think in terms of one output pixel as a weighted sum of a
neighborhood**, optionally followed by pointwise post-processing.

## What you can use

| Construct | Notes |
|---|---|
| Slicing `img[a:b, c:d]` | the neighborhood window (relative offsets become taps) |
| `+ - *`, integer `//` (power-of-two → shift) | element-wise / scale |
| `np.pad(x, k, mode="edge"\|"constant")` | same-size output with edge handling ([streaming-and-bitwidths.md](streaming-and-bitwidths.md)) |
| `.astype(dtype)` | declares the signal width (see below) |
| `.clip(lo, hi)` | saturating narrow |
| `np.where(cond_param, A, B)` | a 2:1 bypass mux on a bool `Param` ([parameters.md](parameters.md)) |
| `Param` / `Params` | config registers (scalar, matrix, bool) |

Signedness and width follow NumPy via `np.result_type`, so `uint8 * -1 → int16`, etc.

For the **complete, authoritative list** — every traced operator/method/`np.*`
function, and explicitly what is **not** supported yet (reductions, `@`/matmul,
convolve, LUT, …) — see [supported-ops.md](supported-ops.md).

## Gotchas (the trace is stricter than NumPy)

The NumPy reference is permissive; the RTL path is not. The compiler will raise a clear
error, but it helps to know the rules:

- **One spatial cone, then pointwise.** A spatial (neighbourhood) sum can be followed by
  pointwise ops, but you can't do a spatial op *after* a pointwise narrow. E.g. an
  unsharp mask must be written as a **single** weighted sum then divide:
  ```python
  # GOOD: one cone, divide at the end
  (32*center - blur_sum) // 16
  # BAD: pointwise (//16) then another spatial add  -> "add after pointwise"
  ```
  (See `examples/isp/sharpen.py`.)
- **`np.where(...)` is terminal-ish**: it builds a mux. You may chain `.astype`/`.clip`
  after it (they push into both branches), but treat it as the output stage.
- **Mixed-width tree** (combining sub-results of different declared widths) is rejected
  with a clear error rather than silently guessing — widen with `.astype` to unify.
- **`default` for registers**: `Param("gain", np.uint8, default=16)` makes the hardware
  boot at a sane value (registers reset to 0 otherwise → black/no-op). See
  [parameters.md](parameters.md).

If the NumPy runs but the RTL path errors, it's almost always one of the above — the
error message names the offending op.

## Validate it

```bash
np2hw run my_isp.py in.png out.png                              # NumPy reference
np2hw run my_isp.py in.png out.png --backend sim --sim cxxrtl  # the generated RTL
```
Identical output means the trace, lowering, and codegen agree with NumPy. This is the
core guarantee: **one function, spec = hardware = reference.**
