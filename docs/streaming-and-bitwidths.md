# Streaming model, edges & bitwidths

## The streaming model

Hardware processes **one pixel per clock**, in raster order. A neighbourhood operation
needs context around the current pixel, which the generated RTL provides with:

- **Horizontal extent** (columns within a row) → a **shift register** of depth
  `max-col - min-col`.
- **Vertical extent** (rows of context) → **line buffers**; a `V`-row window needs
  `V-1` line buffers, each `WIDTH` deep (one full row).

The compiler derives these from the traced tap map and counts the delay lines
automatically; shared sub-expressions are hash-consed so two ops over the same window
share hardware. (Rationale: [`design/01`](../design/01-streaming-model.md),
[`design/03`](../design/03-hash-cons-delay-lines.md).)

Output is registered and ready/valid with backpressure: `stall = out_valid &&
!out_ready` freezes the datapath, so the core composes with FIFOs and the SB/AXI
adapters.

## Edge handling (same-size output)

By default a 3×3-style op shrinks the image (valid-interior). To get **same-size**
output, pad in NumPy and the compiler builds the matching edge hardware:

```python
x = np.pad(img.astype(np.uint16), 1, mode="edge")      # replicate borders
# ... 3x3 weighted sum over x ...
```

| `np.pad` mode | Hardware |
|---|---|
| `mode="edge"` (replicate) | top: broadcast first row into line buffers; bottom: a **flush** phase recirculates the last row; left: broadcast col 0 into the shift register; right: latch & replicate the last column |
| `mode="constant"` (zero) | borders fed as 0 (vertical and horizontal) |

Edge flushes run **during blanking** (HBLANK between rows, VBLANK between frames), so no
backpressure is needed as long as the blanking is at least the kernel's padding depth.
(See `examples/edges.py`; the generated testbench inserts the right blanking.)

## Bitwidths & dtypes (faithful semantics)

`np2hw` tracks two things per signal: the NumPy-style **dtype** (what you see) and the
hardware-actual **bitwidth** (for codegen). The rule that keeps RTL == NumPy:

> **The effective width is `min(naturally-grown width, the declared dtype width)`** —
> and a `uint8` value **wraps** (mod 256) unless you widen it first with `.astype`.

So this **wraps at 8 bits**, exactly like NumPy on `uint8`:
```python
def model(img):                       # img is uint8
    return img + img                  # uint8 + uint8 -> wraps
```
and this **keeps full range** because you declared 16 bits:
```python
def model(img):
    x = img.astype(np.uint16)         # declare 16-bit
    return x + x                      # no wrap
```

### Growth rules (for reference)

| op | result width |
|---|---|
| `add` | `max(N, M) + 1` |
| `multiply` | `N + M` |
| sum of `K` values | `N + ceil(log2 K)` |
| `matmul` (K×K of N-bit) | `2N + ceil(log2 K)` |

### Declaration vs narrowing

- **`astype(dtype)` is a *declaration*** of "this signal is N bits" — mostly at the
  input/output boundary. It's a free zero/sign-extension when widening, NumPy-style
  truncation when narrowing, a no-op assertion when equal.
- **Mid-pipeline narrowing is explicit**: `.clip(lo, hi)` (saturate to a range),
  power-of-two `//` (arithmetic shift), and the boundary `astype` (truncate).

Signedness follows NumPy: subtraction / negative weights produce signed results
(Sobel, gradients work out of the box — see `examples/sobel.py`). Register values are
passed to the reference as NumPy-typed scalars so signed registers promote identically
([parameters.md](parameters.md)).

Full rationale and the astype-vs-narrowing discussion:
[`design/04-bitwidth-tracking.md`](../design/04-bitwidth-tracking.md).
