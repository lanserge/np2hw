# 01 — Streaming model: horizontal / vertical decomposition


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## Core architecture

All image-processing operations decompose into two dimensions:

- **Horizontal (H)**: pixels-within-a-row that the op needs
- **Vertical (V)**: rows the op needs

For a typical streaming ISP pipeline:
- One pixel streams in per clock
- Output pixels stream out per clock (with fixed latency)
- Throughput: 1 pixel/clock
- No frame buffer; only line buffers for vertical context

## Line buffers

For an operation requiring V rows of context:
- V - 1 line buffers needed (newest row streams in directly; V-1 past rows stored)
- Each line buffer = image_width × pixel_bitwidth bits (typically BRAM on FPGA, SRAM macro on ASIC)
- The newest row becomes the next "past row" as pixels stream in

For 1920-pixel-wide 8-bit Bayer:
- Each line buffer = 15 360 bits ≈ 2 BRAM18 blocks
- 3×3 filter (V=3) needs 2 line buffers = 4 BRAMs
- 5×5 filter (V=5) needs 4 line buffers = 8 BRAMs

## Line buffer sharing

When multiple operations need overlapping row contexts, they share line buffers.

Example:
- `sobel_x_3x3` needs rows {y-1, y, y+1}
- `sobel_y_3x3` needs rows {y-1, y, y+1}
- `gaussian_5x5` needs rows {y-2, y-1, y, y+1, y+2}

Union of row demands: {y-2, y-1, y, y+1, y+2} = 5 rows total

Total line buffers needed (storing past 4 rows; newest streams in): 4 line buffers

The 3×3 ops read from a subset of the 5×5's buffers.

## Pipeline alignment

Different operations have different vertical centres → operate at different pipeline positions.

A 3×3 op produces output for row `y` after seeing pixel through `y+1` (latency = W + 1 cycles where W is image width).
A 5×5 op produces output for row `y` after seeing pixel through `y+2` (latency = 2W + 2 cycles).

When chaining ops with different latencies → need explicit alignment / skew FIFOs at consumer side.

## Edge handling

When the window extends beyond the image (top/bottom rows, left/right columns):

| Mode | Behaviour |
|---|---|
| `replicate` | Copy nearest valid pixel |
| `zero` | Treat out-of-bound as 0 |
| `reflect` | Mirror around the boundary |
| `wrap` | Wrap around to other side (rare for ISP) |

Configurable per op. Default: `replicate` for ISP (standard convention).

## Operation classification

| Class | H | V | Examples |
|---|---|---|---|
| Element-wise | 1 | 1 | `*`, `+`, `clip`, LUT |
| 1D horizontal | W | 1 | 1D convolve along row, horizontal blur |
| 1D vertical | 1 | H | 1D convolve along column, vertical blur (less common in streaming) |
| 2D spatial | W | H | 2D convolution, morphology, bilateral |
| Reductions | full image | full image | `sum`, `mean`, `histogram` (need full frame; not pure streaming) |

Reductions and full-image operations need special handling (accumulate over the frame; output at end-of-frame).

## What this enables

- Fixed, predictable memory cost: line buffers only, no frame buffer
- Constant throughput: 1 pixel per clock (or N per clock if parallelised)
- Bounded latency: V × image_width pixels for the deepest op chain
- Plays well with Switchboard streaming model
- Maps naturally to ZeroAsic's emulation stack
