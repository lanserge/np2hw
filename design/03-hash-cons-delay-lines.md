# 03 — Hash-cons and delay-line management


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## The pattern

Every node in the pipeline graph is uniquely identified. Cache: `(op, input_ids, params) → output_id`.

Two parts of the user's NumPy code computing the same operation on the same inputs **automatically share the same hardware**. No explicit user opt-in needed.

This is structurally the same algorithm as AIG structural hashing in logic synthesis — applied at the streaming pipeline level rather than the gate level.

## Example: unsharp mask

```python
def unsharp_mask(img):
    blurred = gaussian_3x3(img)
    sharpened = img + (img - blurred)      # img used twice
    return np.clip(sharpened, 0, 255)
```

Graph construction:
- `img` → id 0 (input)
- `gaussian_3x3(img)` → id 1
- `img - blurred` → id 2 (this needs `img` ALIGNED with `blurred`'s latency; an explicit `delay(img, by=gauss_latency)` is inserted → id 3 — then `subtract(id 3, id 1)` is id 2)
- `img + above` → id 4 (reuses id 3 for the delayed `img` — automatic share)
- `clip(...)` → id 5

The delayed `img` (id 3) is materialised ONCE; consumed by id 2 and id 4.

## Multi-scale example (where sharing shines)

```python
def multi_scale_demo(img):
    blur_3 = gaussian_3x3(img)
    blur_5 = gaussian_5x5(img)
    
    # Multiple branches, each reuses one of the blurs
    edge_x_3 = sobel_x(blur_3)
    edge_y_3 = sobel_y(blur_3)             # blur_3 shared (free!)
    
    detail = img.aligned_to(blur_5) - blur_5
    smooth_5 = blur_5 * 0.8                # blur_5 shared (free!)
    
    return ...
```

Without hash-cons: 2 copies of `gaussian_3x3` and 2 copies of `gaussian_5x5` materialise. Each Gaussian has multiple line buffers + MAC arrays. ~50% area waste on the blur work alone.

With hash-cons: each Gaussian materialised once, fanned out to multiple consumers. Yosys/OpenROAD handle the fanout in routing.

## Delay-line IDs and alignment

Each `StreamArray` has a `row_offset` and `col_offset` tracking its "pipeline position" relative to the input stream.

```
input image streaming in → row_offset = 0
gaussian_3x3 output → row_offset = 1 (one row behind input by the time it's valid)
gaussian_5x5 output → row_offset = 2 (two rows behind input)
```

When two values feed into the same op, they MUST be at the same `row_offset` and `col_offset`. If not, the system inserts an explicit `delay` op to align the earlier one to the later position.

Delay ops are themselves hash-consed: two consumers needing the same delay of the same source share one delay line.

## What gets stored, what gets recomputed

A delay line of length L over an N-bit signal needs L flip-flops. For longer delays:
- L < ~10 flip-flops: shift register
- L < ~image_width: BRAM-shaped FIFO
- L = image_width: line buffer (this is the "vertical" case from streaming model)
- L = K × image_width: K line buffers chained

When two ops need the same row context, line buffers share the underlying BRAM.

## Hash-cons across the boundary with line buffers

Two ops requiring "row at offset 1" share the same line buffer storage. The op IDs differ (they're different operations) but the LINE BUFFER ID is shared.

This is why line buffers also have their own ID space, separate from operation IDs.

## Lifetime / dead code elimination

After the graph is built but before Verilog emission:
1. Trace from declared outputs backward through `producer_inputs`
2. Mark all reachable nodes as `live`
3. Skip everything else when emitting Verilog

Same pass as DCE in classical compilers. Cheap to implement once the graph is hash-consed.

## Connection to Serge's other work

This same structural-sharing pattern shows up in:
- AIG synthesis (the DSD work being discussed for ZeroAsic)
- LLVM's GVN (Global Value Numbering) pass
- Halide's CSE pass
- TVM's structural equality
- Bazel/Nix derivation deduplication

It's the same algorithmic primitive at different IR levels. Important to recognise — the techniques transfer.
