# 05 — ISP operation pattern library


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## The pattern library IS the core IP

For each NumPy operation we support, we have a hardware pattern that maps it to streaming Verilog. The library is what makes np2hw different from generic Python HLS tools — it captures Serge's 12 years of production ISP architecture in code.

## Pattern entry shape

```python
@dataclass
class StreamingPattern:
    H: int                  # horizontal window width (pixels)
    V: int                  # vertical window height (rows)
    pixel_kernel: str       # name of compute pattern (mac_9, mux, clip, ...)
    kernel_params: dict     # coefficients, lut data, gain values, etc.
    bitwidth_growth: callable  # (input_bws, params) → output_bw
    line_buffers_needed: int   # = V - 1 normally
    edge_policy: str = "replicate"
```

## Tier 1 patterns (v1 prototype scope)

### Element-wise (H=1, V=1, 0 line buffers)

| Pattern | Description |
|---|---|
| `add_const` | x + c |
| `multiply_const` | x * c |
| `add_pixel` | a + b |
| `multiply_pixel` | a * b |
| `subtract` | a - b |
| `clip` | clamp to [lo, hi] |
| `lut` | table lookup |
| `maximum` | max(a, b) |
| `minimum` | min(a, b) |
| `negate` | -x |
| `abs` | |x| |
| `compare_lt` | a < b → 0/1 |
| `mux` | s ? a : b |

### 1D horizontal (H=W, V=1, 0 line buffers)

| Pattern | Description |
|---|---|
| `horiz_blur_3` | [1, 2, 1] / 4 |
| `horiz_blur_5` | [1, 4, 6, 4, 1] / 16 |
| `horiz_gradient` | [-1, 0, 1] |

### 2D spatial (H>1, V>1)

| Pattern | H | V | LBs |
|---|---|---|---|
| `gaussian_3x3` | 3 | 3 | 2 |
| `gaussian_5x5` | 5 | 5 | 4 |
| `box_filter_3x3` | 3 | 3 | 2 |
| `sobel_x_3x3` | 3 | 3 | 2 |
| `sobel_y_3x3` | 3 | 3 | 2 |
| `laplacian_3x3` | 3 | 3 | 2 |
| `median_3x3` | 3 | 3 | 2 |
| `erode_3x3` | 3 | 3 | 2 |
| `dilate_3x3` | 3 | 3 | 2 |
| `conv2d_3x3` | 3 | 3 | 2 |
| `conv2d_5x5` | 5 | 5 | 4 |

### Bayer / colour

| Pattern | Description |
|---|---|
| `bayer_demosaic_3x3` | Bayer → RGB (3 line buffers) |
| `ccm_3x3` | Per-pixel 3×3 colour matrix |
| `awb_gain_per_channel` | Per-channel gain multiplier |

### Reductions (need full frame; output at end)

| Pattern | Description |
|---|---|
| `roi_sum` | Σ over ROI |
| `roi_mean` | Σ/N over ROI |
| `roi_histogram` | Histogram of pixel values in ROI |
| `frame_max` | max over frame |

## Tier 2 patterns (later)

- Bilateral filter (3×3 to 7×7)
- Guided filter
- Optical flow (Lucas-Kanade)
- Corner detection (Harris)
- Tonemap operators
- HDR fusion

## Pixel-kernel patterns (the per-pixel compute)

Each top-level pattern uses one of these inner compute structures:

| Kernel | Description |
|---|---|
| `mac_N` | N-term multiply-accumulate (FIR / convolution tap) |
| `add_tree_N` | Adder tree, N inputs (for reductions) |
| `mux` | 2-to-1 mux |
| `lut_addr` | Address generation for LUT lookup |
| `comparator` | < > == returning 0/1 |
| `clip` | Clamp to range |
| `sort_network_N` | Sort N values (for median) |
| `partial_products` | For multiplier construction |

A pattern is a (H, V) window structure plus a pixel-kernel that operates on the current window.

## Verilog emission per pattern

Each pattern has a Verilog template. For example, `gaussian_3x3`:

```verilog
module gaussian_3x3 #(
    parameter WIDTH = 1920,
    parameter PIXEL_BITS = 8
) (
    input  clk, rst,
    input  [PIXEL_BITS-1:0] pixel_in,
    input  valid_in,
    output [PIXEL_BITS-1:0] pixel_out,
    output valid_out
);
    // Line buffers (BRAM-inferred):
    reg [PIXEL_BITS-1:0] line_buffer_0 [0:WIDTH-1];
    reg [PIXEL_BITS-1:0] line_buffer_1 [0:WIDTH-1];
    
    // Shift register for 3-pixel window in current row:
    reg [PIXEL_BITS-1:0] w00, w01, w02;     // top row (oldest)
    reg [PIXEL_BITS-1:0] w10, w11, w12;     // middle row
    reg [PIXEL_BITS-1:0] w20, w21, w22;     // bottom row (newest)
    
    // ... shift logic that pulls from line buffers and feeds them ...
    
    // The Gaussian kernel: [[1,2,1],[2,4,2],[1,2,1]] / 16
    wire [PIXEL_BITS+3:0] sum =
          w00 + 2*w01 + w02
        + 2*w10 + 4*w11 + 2*w12
        +   w20 + 2*w21 + w22;
    
    assign pixel_out = sum >> 4;     // divide by 16 (shift)
endmodule
```

This template gets instantiated by the Verilog emitter, parameterised on WIDTH (image width) and bitwidth.

## Adding new patterns

To add a new pattern (e.g., for an unusual filter Serge knows from Apical):

1. Decide H, V, line buffer count
2. Decide pixel kernel (often `mac_N` with specific coefficients)
3. Write Verilog template
4. Register in the pattern dispatch table
5. Map relevant NumPy ops to it via `__array_function__` handler

The dispatch table grows as the library grows. v1 ships with 20-30 patterns; v2 adds whatever the user needs.

## Why this is Serge's IP

Patterns aren't just "convolution" — they're convolution-the-way-it-actually-ships-in-silicon:
- Bayer pattern handling (RGGB vs GBRG vs BGGR vs GRBG)
- Edge replication with the right pixel buffer arrangement
- Saturating arithmetic at the right points
- Coefficient quantisation tricks (e.g., approximating 1/3 as `(x + (x >> 2)) >> 1`)
- Anti-aliased downsample patterns

These are choices that production ISP designers know cold and academic papers don't cover well. Codifying them is what makes np2hw worth more than a generic Python HLS.
