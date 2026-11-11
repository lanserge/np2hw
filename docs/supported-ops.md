# Supported operations

What the tracer (`np2hw.core.trace`) records and the lowering builds. The
trace is **stricter than NumPy**: anything not listed is refused by name
(`Unsupported`, a `NotImplementedError`) or by reason (`ValueError`) -- it
never falls back to a NumPy result. Every value carries NumPy 2's container
and its exact range, so the hardware wraps exactly where NumPy does
([one-graph.md](one-graph.md), decision 5).

## Operators

| Operator | Forms | Hardware |
|---|---|---|
| `+ - *` | value with value, constant, or register | adder, subtractor, multiplier -- by `+-2**k` a shift |
| `-x`, `abs(x)` / `np.abs` | | negate, absolute value |
| `< > <= >= == !=` | | comparisons, a 1-bit result |
| `&`, `\|` | value with value, constant, or register | bitwise; `x & (2**k - 1)` of a non-negative value is its low bits |
| `>> k`, `<< k` | `k` a constant | wiring |
| `// k`, `% k` | `k` a constant power of two in hardware (the C backend takes any) | a shift, low bits |
| `[a:b, c:d]` | constant slices, steps, a register start (`x[p.py::2]`, `x[1 - p.py::2]`) | a window: line buffers and shift registers, sized from the spread of the reads |
| `img[p.y0:, p.x0:][:p.h, :p.w]` | starts and lengths registers | a crop placed at run time |
| `table[x]` | a 1-D register array (`Param(shape=(n,))`) or a constant table (`Rom`) read at a value the model computes | a mux, or a ROM; an index that can leave the table is refused |
| `out[py::2, px::2] = ...` | on `np.empty_like` / `np.zeros_like` | planes assembled into one datapath, each pixel its plane's arithmetic |

## Methods and functions

| | Support |
|---|---|
| `.astype(dtype)`, `.view(dtype)` | a change of container: a wrap where the value leaves it |
| `.clip(lo, hi)`, `saturate(x, bits)` | constant bounds |
| `.reshape(...)`, `np.reshape` | access through split and merged indexes; nothing moves |
| `np.pad(x, w, mode="edge" \| "constant")` | any value; zero only for `"constant"` |
| `np.where(cond, a, b)` | a condition the model computes, or a register; a Python bool picks at trace time |
| `np.stack([...], axis=-1)` | channels of one word -- each unsigned |
| `coords(img)`, `make_coords(img, rows, cols)` | the input's, or a declared output's, position as values |
| `np.add`, `np.subtract`, `np.multiply`, comparisons, `np.bitwise_and/or`, `np.absolute` | as their operators |
| a video, `Image2D(..., frames=n)`: `video[1:]`, `video[:-1]` | reads across frames; the store derived from the reads |

## Reductions and folds

A frame into a result that is the frame's -- statistics, a histogram, zone
sums -- written as NumPy writes it:

| Model | Support |
|---|---|
| `x.sum()`, `x.max()`, `x.min()` (and `np.sum`, `np.max`, `np.min`) | the whole value, or over `axis=` of the image's axes -- an entry for each position kept, the fold keyed by it -- in the container NumPy reduces in (`sum` of an unsigned value is `uint64`, of a signed one `int64`; `dtype=` takes another) |
| `acc = np.zeros(n, dtype, like=pixel)` (or `np.full`, `np.ones`) | an accumulator: `like=` is NumPy's own routing, which puts it in the trace; a dtype is required |
| `np.add.at(acc, key, value)`, `np.maximum.at`, `np.minimum.at` | each pixel's value folded into the entry its key names -- one key per axis of `acc`, a key the pixel computes or a constant; each step wrapped into `acc` as NumPy's `at` wraps it |
| `np.stack([acc_a, acc_b], axis=-1)` | folds side by side: an entry's values together |

The value and its keys range over the whole frame -- each pixel folds once --
and may be anything the pixel's datapath computes, windows included. One
write per accumulator. The result leaves as an ordinary output stream once
its frame is folded: a beat an entry, the folds' values side by side as a
stack's channels, framed by the result's rows and columns (a histogram one
row, a reduction whole one beat), carrying the folded frame's context with
the result's size. Where it goes -- software's reading, another block --
is the design's wiring -- beside the datapath it measures, through a fork.
Each fold is an accumulator memory, cleared as the result streams out of
it; a frame arriving before that is done waits, as for any core not ready.

## What is refused

- a window over a value that can be negative -- offset it first;
- `//` and `%` by a constant that is not a power of two, and a shift by a
  register (the C backend takes the first);
- a source declared signed;
- folds side by side (`np.stack` of accumulators) where one can be
  negative -- a word's channels are unsigned; keep that fold an output
  of its own, where it may be signed, or offset it first;
- a reduction over a word's channels (`x.sum(axis=-1)` of a multi-channel
  value), `np.mean` (a divide);
- `np.minimum` / `np.maximum` as functions, `np.clip` as a function -- use
  `.clip`, `np.where`;
- `@` / `np.matmul`, `np.convolve`, floating point.

A fixed convolution is a weighted sum of slices, which is what the
[`examples/isp/`](../examples/isp) models do.

## See also

- [writing-models.md](writing-models.md) -- the model-file convention
- [one-graph.md](one-graph.md) -- how a model becomes hardware
