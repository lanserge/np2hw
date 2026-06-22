# 04 — Bitwidth tracking: astype (declaration) vs mode (narrowing)


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## The clean separation

| Concern | Mechanism | Where it applies |
|---|---|---|
| **Declare input bitwidth/type** | `astype` (or input spec in decorator) | At function inputs, once per input |
| **Track natural bitwidth growth** | Automatic via op overloads | Throughout the pipeline, no user action |
| **Control output narrowing** | `mode` (truncate / saturate / round) | At function outputs (or via explicit narrowing ops) |

**astype is a DECLARATION**: "this signal is N bits."  
**mode is a NARROWING SEMANTIC CHOICE**: "when narrowing from M bits to N bits, do it this way."

The two concerns don't mix in the middle of the pipeline.

## StreamArray fields

```python
class StreamArray:
    dtype: np.dtype       # NumPy-equivalent dtype (what user sees)
    bitwidth: int         # actual hardware bitwidth (what hardware uses)
    signed: bool
    fractional_bits: int  # for fixed-point support later
```

**Key invariant**: `bitwidth` is the source of truth for Verilog emission. `dtype` is a user-facing label that may diverge in the middle (e.g., uint8 × uint8 keeps dtype=uint8 but bitwidth grows to 16).

## Natural bitwidth growth rules

| Operation | Output bitwidth |
|---|---|
| `a (Nbit) + b (Mbit)` | max(N, M) + 1 |
| `a (Nbit) - b (Mbit)` | max(N, M) + 1 (signed) |
| `a (Nbit) * b (Mbit)` | N + M |
| `a (Nbit) << k` | N + k |
| `sum of K values, each Nbit` | N + ceil(log2(K)) |
| `max/min(a, b)` | max(N, M) |
| `lut[a]` where lut output is W bits | W |
| Matmul (K×K of N-bit elements) | 2N + ceil(log2(K)) |

## astype semantics

```python
def astype(self, dtype, bitwidth=None):
    declared_bw = bitwidth or self._bits_for_dtype(dtype)
    if declared_bw == self.bitwidth:
        return self._relabel(dtype=dtype)         # pure assertion, free
    elif declared_bw > self.bitwidth:
        return StreamArray._build('extend', (self,),     # zero/sign-extend
                                  params=(declared_bw,),
                                  bitwidth_out=declared_bw, dtype_out=dtype)
    else:
        self._warn(f"astype narrowing from {self.bitwidth} to {declared_bw}; "
                   f"using truncate semantics; use .saturate({declared_bw}) for clamping")
        return StreamArray._build('truncate', (self,),    # NumPy-compatible
                                  params=(declared_bw,),
                                  bitwidth_out=declared_bw, dtype_out=dtype)
```

If user calls `astype` mid-pipeline expecting it to narrow with non-default semantics, they get a warning recommending explicit `.saturate()` / `.clip()` / `.round_to()`.

## Explicit narrowing ops (mid-pipeline)

```python
narrow = wide.saturate(8)              # clamp to 8-bit range
narrow = wide.saturate(8, signed=True) # signed clamp (-128 to 127)
narrow = wide.truncate(8)              # take lower 8 bits (NumPy default)
narrow = wide.round_to(8)              # round-and-truncate
narrow = wide.clip(lo, hi)             # clamp to specific range
```

Each generates its own hardware:

```verilog
// truncate: just take lower bits
assign narrow = wide[7:0];

// saturate (unsigned 8-bit):
assign narrow = (|wide[15:8]) ? 8'hFF : wide[7:0];

// saturate with signed:
assign narrow = (wide[15] && !&wide[14:7]) ? 8'h80      // clamp to -128
              : (!wide[15] && |wide[14:7]) ? 8'h7F     // clamp to +127
              :                              wide[7:0];

// round-and-truncate (unsigned, half-LSB up):
wire [16:0] r_temp = wide + 17'h80;
assign narrow = r_temp[15:8];

// round-and-saturate:
wire [16:0] rs_temp = wide + 17'h80;
assign narrow = (rs_temp[16] || |rs_temp[15:8]) ? 8'hFF : rs_temp[15:8];
```

## Output narrowing handled at compile time

```python
@hls_compile(
    inputs={...},
    outputs={
        'pixel_out': dict(shape=(...), dtype=np.uint8, mode='saturate'),  # saturate at output
    }
)
def isp_pipeline(...): ...
```

The compile pass walks backward from each output. If the natural bitwidth at the output point exceeds the declared output bitwidth:
- Inserts the narrowing op specified by `mode`
- The narrowing op is hash-consed (so shared with any other narrowing on the same source)

## ISP-specific defaults

For image processing, **saturate** is almost always the right narrowing semantic:
- Pixel wraparound (200 + 100 = 44) is jarringly visible — bright pixels turn dark
- Clamping at 255 produces a smooth highlight clip — visually acceptable

So our default for `mode` on output specs declared as 8-bit unsigned: **saturate**.

Override to `truncate` only when the user wants NumPy-strict compatibility for verification.

## Why this matters for output area

Without astype tracking + explicit narrowing:
- A pipeline of 5 multiplies on 8-bit inputs grows the bitwidth to 40+ unnecessarily
- Final 32-bit multiplier where 16-bit would suffice
- 4× area bloat on the multipliers

With proper bitwidth tracking + explicit narrowing at the right points:
- Each stage gets the natural bitwidth from inputs
- Explicit narrowing at known "should be N bits here" points reins it in
- Hardware area matches what an HDL designer would write by hand

This is the difference between 1× and 4× silicon area for the same algorithm.
