# Supported operations

The authoritative list of what the tracer (`frontend.py`) currently turns into
hardware. The trace is **stricter than NumPy**: anything not listed here is either
rejected with a clear error, or (for an unhandled `np.*` function) raises `TypeError`
— it never silently falls back to a NumPy result. If your model runs as NumPy but the
RTL path errors, it's using something below the line.

## Operators

| Operator | Forms | Hardware |
|---|---|---|
| `+` | `a + b`, `a + int`, `Param + a` (and `int + a`) | adder; a `Param` adds a **bias register** |
| `-` | `a - b`, `-a` | subtract / negate (result is signed) |
| `*` | `a * b`, `int * a`, `Param * a` (and `int * a`) | multiply; `int` → scale, `Param` → register multiplier / per-tap coeff |
| `//` | `a // 2**k` | arithmetic right shift (**power-of-two only**) |
| `[]` | `a[r0:r1, c0:c1]` | neighbourhood window → taps (shift register + line buffers) |

## Methods

| Method | Hardware |
|---|---|
| `.astype(dtype)` | width **declaration** — zero/sign-extend (widen), truncate (narrow), or assert (equal) |
| `.clip(lo, hi)` | saturate into `[lo, hi]` |

## NumPy functions

| Function | Support |
|---|---|
| `np.pad(x, width, mode=...)` | `mode="edge"` (replicate) or `"constant"` (zero, `constant_values=0` only); must be applied **directly to the image input** |
| `np.where(cond, A, B)` | `cond` must be a scalar **bool `Param`** (1-bit); `A`/`B` are same-shape branches → a 2:1 mux (bypass) |

**Any other `np.*` call on a traced value raises `TypeError`** (the handler returns
`NotImplemented`). So `np.sum`, `np.matmul`, `np.clip(...)` (the function), etc. are not
traced — see "Not yet supported" below.

## Operands

- the streamed image input (`Image2D`)
- `Param` / `Params` — config registers (scalar, matrix, 1-bit bool); see [parameters.md](parameters.md)
- `Const` / Python `int` — fold in as literals

## Dtype & signedness

Width and sign follow NumPy via `np.result_type` (`uint8 * -1 → int16`, etc.); the
spatial sum is clipped to `min(natural width, declared width)`, so `uint8` **wraps**
unless widened with `.astype`. Full rules:
[streaming-and-bitwidths.md](streaming-and-bitwidths.md).

## Not yet supported (roadmap)

These appear in the design pattern table
([design/05](../design/05-isp-operation-library.md)) as the *intended* library,
but are **not implemented in the tracer yet**:

- **Reductions**: `np.sum`, `np.mean`, `np.max`, `np.maximum` / `np.minimum`
- **`@` / `np.matmul`** (systolic array), **`np.convolve`**
- **`np.clip` as a function** — use the `.clip(lo, hi)` **method**
- **`.saturate(N)` / `.truncate(N)` / `.round_to(N)`** — use `.clip`, `//`, `.astype`
- **LUT / `np.take`** lookup, **multi-channel / CCM**, **floating point**

Until then, express these with the supported ops — e.g. a fixed convolution is just an
explicit **weighted sum of slices** (`sum(w[i,j] * x[i:.., j:..] ...)`), which is exactly
what the [`examples/isp/`](../examples/isp) models do. A genuinely needed op is a
candidate for the next pattern to add to the tracer.

## See also

- [writing-models.md](writing-models.md) — the model-file convention + gotchas
- [design/05-isp-operation-library.md](../design/05-isp-operation-library.md) — the aspirational pattern library (roadmap)
