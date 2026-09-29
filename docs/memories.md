# Memories: the seam, and what binds to it

A stencil with vertical extent needs line buffers, one full row deep
each (see [streaming-and-bitwidths.md](streaming-and-bitwidths.md)).
Each is a memory array, and where that array ends up depends on the
target more than anything else np2hw emits:

| target | `reg [W-1:0] mem [0:D-1]` becomes |
| --- | --- |
| FPGA, read through a register | block RAM |
| FPGA, read combinationally | distributed RAM behind a select tree |
| silicon, either way | **D × W flip-flops** — 23,040 for one 1920-wide 12-bit line |

So the emitters that read their line buffers through a register put the
array behind a module of its own, and that module is **the memory seam**:
the one place an integrator targeting silicon replaces, by name, with a
memory macro. Nothing else in the generated design knows or cares which
implementation sits there.

## The module

One per emitting module, instantiated once per line buffer:

```verilog
module <module>_linebuf #(
    parameter W  = 12,      // word width, as traced
    parameter D  = 1920,    // depth: the line, as built
    parameter AW = 16       // address width
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] wa,
    input  wire [W-1:0]  wd,
    input  wire          re,
    input  wire [AW-1:0] ra,
    output reg  [W-1:0]  rd
);
    reg [W-1:0] mem [0:D-1];
    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        if (re) rd <= mem[ra];
    end
endmodule
```

Left as it is, every FPGA toolchain infers block RAM from it. The
parameters are the traced facts (`W` from the model's arithmetic, `D`
from the geometry the core was built for), so a macro is bound per
distinct `(W, D)` — which is how memory compilers work anyway.

## The contract a binding must honour

Four things, and the comment on every emitted seam module says the same:

1. **Write.** `wd` lands at `mem[wa]` on the `clk` edge where `we` is
   high.
2. **Read.** `rd <= mem[ra]` on the `clk` edge where `re` is high — one
   cycle of latency — and **`rd` holds while `re` is low.** The datapath
   pauses under backpressure with the read register holding the last
   word; a macro whose output floats or re-samples under `re = 0` needs
   `W` flops of hold in its wrapper.
3. **Collision.** `we` and `re` in the same cycle at the same address
   read the **old** word (read-before-write). This is not a corner case
   to wave through: a line delay writes column `c` while reading last
   line's column `c`, every beat of every line. A macro whose collision
   behaviour is undefined or read-new needs a bypass in its wrapper.
4. **No reset.** `rd` is data, not state; nothing here clears it and
   nothing downstream depends on its value before the first read.

Hold and bypass are both cheap, both the integrator's, and both outside
the seam's contract, which is why the module stays as small as it is.

## Which emitters offer the seam

| model shape | emitter | line-buffer read | under `sync_memory=True` |
| --- | --- | --- | --- |
| padded — `np.pad(..., mode="edge" \| "constant")` | edge | **registered** | accepted, behind the seam |
| CFA phases — `out[py::2, px::2] = ...` | phase stencil | **registered** | accepted, behind the seam |
| unpadded window | general core | combinational | **refused** |
| `np.where(enable, A, B)` over a window | mux | combinational | **refused** |
| a stream's recent past | `frame_history` | **registered** | behind its own seam, `<module>_mem`, same four rules; reported in its `memories` |

The two combinational readers are not a defect. On an FPGA a
combinational read is distributed RAM, and for a window that fits it is
the cheaper choice — which is exactly why an unflagged generate keeps
emitting it. What `sync_memory=True` states is a fact about the target:
no SRAM macro implements a combinational read, so a design bound for
silicon must not contain one. The refusal names the module, the count,
and the fix (pad the model, or write it as phases).

```python
generate(out, module_name="blur", sync_memory=True)
# ValueError: blur: 2 line buffers here read ASYNCHRONOUSLY -- ...
```

## What every core publishes

Whichever way the flag is set, a generated core reports the memories it
inferred, so an integrator reads the fact instead of grepping for it:

```python
core["memories"]
# [{"instance": "u_linebuf1", "module": "blur_linebuf",
#   "bits": 12, "depth": "WIDTH", "read": "sync"}, ...]
```

`read` is `"sync"` (behind the seam) or `"async"` (a combinational read;
`module` is then `None`, because there is nothing to bind to). A
composition can gather these into its build report; revela does.

## What this does not cover

- **Constant tables** (`np.take` on a literal array — a tone curve's
  knots) are read through a register and are small; they synthesise as
  logic or as a ROM at the integrator's choice and are not behind the
  seam.
- **The clock-crossing FIFO** (`cdc_fifo`) is a dual-clock memory, which
  is a different macro and a different contract. Not yet behind a seam.
- **Verification on silicon.** Everything here is proven bit-identical
  in simulation and on an FPGA. That a given foundry macro honours the
  contract is the integrator's to verify, and this document is the
  statement of what to verify.
