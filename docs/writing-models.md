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

`np2hw` runs your function once on a traced stand-in for `img`. NumPy's operators and
`__array_function__`/`__array_ufunc__` record every operation into ONE graph -- what
the model computed, nothing decided -- each value with its NumPy container and its
exact range. Lowering then reads the hardware off the graph: a read of the frame at
an offset is a window (line buffers for the rows, shift registers for the columns,
sized by the spread of the reads), a read across frames is a frame store, a fold is
an accumulator, everything else is the datapath. The whole story is
[one-graph.md](one-graph.md).

The practical consequence: **write the NumPy you mean.** A filter applied to a
filter's result is one window over the source; a value computed once and read at
offsets is stored, recomputed at every tap where you recompute it. The circuit is
the one the model states, and it is bit-exact because NumPy's own semantics are
followed operation by operation.

## What you can use

| Construct | Notes |
|---|---|
| Slicing `img[a:b, c:d]`, steps, a register start | windows, CFA planes |
| `+ - *`, `& \|`, comparisons, `abs`, `>>`/`<<` by a constant | the datapath |
| `//`, `%` by a power of two | shifts, low bits |
| `np.pad(x, k, mode="edge"\|"constant")` | same-size output with edge handling |
| `.astype(dtype)`, `.view(dtype)`, `.clip(lo, hi)`, `saturate(x, bits)` | containers and narrowing, as NumPy has them |
| `np.where(cond, a, b)` | a mux on any condition the model computes |
| `np.maximum`, `np.minimum` | a compare and a select, the range exact |
| `x.max(axis=-1)`, `x.sum(axis=-1)`, `x.argmin(axis=-1)` over a word's channels | each pixel's own arithmetic over its lanes |
| `table[x]` | a lookup in a register array or a `Rom` |
| `np.stack([...], axis=-1)` | channels of one word |
| `coords(img)` | the pixel's position, as values |
| `x.sum()`, `np.add.at(acc, key, v)` into `np.zeros(n, dtype, like=img)` | folds: the frame into a result of its own |
| `Param` / `Params` | config registers (scalar, matrix, bool) |

Signedness and width follow NumPy 2 (`uint8 * -1 -> int16`, a `uint8` sum wraps).

For the **complete, authoritative list** -- and what is refused -- see
[supported-ops.md](supported-ops.md).

## Gotchas (the trace is stricter than NumPy)

The NumPy reference is permissive; the RTL path is not. A refusal names what it
refuses:

- **A window over a value that can be negative is refused.** Offset it into an
  unsigned range first (`x + 128`), window it, and take the offset off after.
- **Division and remainder are by powers of two** in hardware; shifts are by
  constants. `x * k >> s` is the usual way to scale.
- **A lookup's index must provably stay inside its table** -- mask or clip it:
  hardware has no IndexError.
- **Channels stacked into a word are unsigned** -- clip them first. Folds
  stacked side by side too; a single fold, or a reduction, may be negative.
- **`default` for registers**: `Param("gain", np.uint8, default=16)` makes the
  hardware boot at a sane value (registers reset to 0 otherwise).

If the NumPy runs but the RTL path errors, the message names the operation and why.

## Validate it

```bash
np2hw run my_isp.py in.png out.png                              # NumPy reference
np2hw run my_isp.py in.png out.png --backend sim --sim cxxrtl  # the generated RTL
```
Identical output means the trace, lowering, and codegen agree with NumPy. This is the
core guarantee: **one function, spec = hardware = reference.**
