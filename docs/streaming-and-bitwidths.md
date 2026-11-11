# Streaming model, edges & bitwidths

## The streaming model

Hardware processes **one pixel per clock**, in raster order. A model that reads
the frame at offsets needs the pixels around the current one, which the core
keeps:

- **columns** (offsets along a line) in a **shift register**, as deep as the
  spread of the column offsets read;
- **rows** in **line buffers**: a read spanning `V` rows keeps `V - 1` lines,
  each the traced width deep. Where those arrays land -- block RAM,
  distributed RAM, or on silicon a macro bound to the emitted `*_linebuf`
  module -- is [memories.md](memories.md);
- **frames** (a video's past) in a **frame store**, as deep as the spread of
  the frame offsets read.

Nothing declares any of it: the storage is the spread of the reads the model
makes, and the output trails the input by the most any read reaches past the
output's own pixel ([one-graph.md](one-graph.md), the window rule).

The output is registered and ready/valid with backpressure: one `ready` holds
every stage while the output stalls, so a core composes with FIFOs, forks and
the SB/AXI adapters.

## Edge handling (same-size output)

A window over the frame shrinks it (`x[1:-1]` is the frame less its border).
To keep its size, pad in NumPy and the core builds the matching edge rule:

```python
x = np.pad(img.astype(np.uint16), 1, mode="edge")      # replicate borders
# ... 3x3 weighted sum over x ...
```

| `np.pad` mode | Hardware |
|---|---|
| `mode="edge"` (replicate) | a row or column before the frame's first reads the first -- a select on the step's position; one past the last reads the last, fed back by the flush |
| `mode="constant"` (zero) | the same selects, with zero in place of the edge |

The rows and columns past the frame's last are **flush steps**: steps that take
no input pixel. Once a frame's last pixel is in, the core runs them in idle
input cycles -- blanking, where a source has it -- and when the next frame
arrives first, it holds that frame back through `in_ready` until they are
done. No blanking is required; it only saves the wait.

## Bitwidths & dtypes (faithful semantics)

Every value carries two facts: its **NumPy container** -- the dtype NumPy holds
it in, by NumPy 2's rules -- and its **exact integer range**. The container says
where NumPy wraps, and the hardware wraps exactly there and nowhere else; the
range sizes every wire.

So this **wraps at 8 bits**, exactly like NumPy on `uint8`:

```python
def model(img):                       # img is uint8
    return img + img                  # uint8 + uint8 -> wraps
```

and this keeps the full range, because NumPy holds it in 16 bits:

```python
def model(img):
    x = img.astype(np.uint16)
    return x + x                      # no wrap
```

A Python int is weak and takes the other operand's dtype; a register is the
NumPy scalar the NumPy run is handed for it, the smallest integer holding its
width, so `uint8` data plus an `int8` register promotes to `int16`, as NumPy
promotes it.

- **`astype` / `view`** change the container: a wrap where the value leaves it,
  nothing where it fits.
- **`.clip(lo, hi)` and `saturate(x, bits)`** saturate, where they are written.
- **`>>` and `//` by a power of two** floor, as NumPy does, on signed values too.

A wire is exactly as wide as its value's range needs -- not the container's
width, which only says where a wrap happens. Signedness follows the range:
subtraction and negative weights give signed values (gradients work out of the
box), and register values reach the reference as NumPy-typed scalars, so signed
registers promote identically ([parameters.md](parameters.md)).
