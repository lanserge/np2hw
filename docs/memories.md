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

So every line buffer is read through a register, and its
array sits behind a module of its own. That module is **the memory seam**:
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
   line's column `c`, every beat of every line, and a fold streaming its
   result out reads each entry while writing its start back. A macro whose collision
   behaviour is undefined or read-new needs a bypass in its wrapper.
4. **No reset.** `rd` is data, not state; nothing here clears it and
   nothing downstream depends on its value before the first read.

Hold and bypass are both cheap, both the integrator's, and both outside
the seam's contract, which is why the module stays as small as it is.

## Which memories sit behind a seam

| what the model reads | line-buffer read | under `sync_memory=True` |
| --- | --- | --- |
| any window -- padded (`mode="edge"` or zero), CFA phases, unpadded, a crop, a filter of a filter's result, `np.where` over a window | **registered**, read a column ahead | accepted, behind the seam, `<module>_linebuf` |
| a video's past frames | **registered**, read a position ahead | accepted, behind its own seam, `<module>_mem`, same four rules |
| a stream's recent past (`frame_history`) | **registered** | accepted, behind its own seam, `<module>_mem`, same four rules |
| a fold's entries (`np.add.at`, a reduction over an axis) | **registered**, read beside a write of the beat before, which is forwarded | accepted, behind its own seam, `<module>_acc`, same four rules |

What `sync_memory=True` states is a fact about the target: no SRAM macro
implements a combinational read, so a design bound for silicon must not
contain one. Every memory above keeps it, so the flag changes nothing that
is emitted; it is the promise, checked, that a core bound for silicon
makes, and it refuses -- naming the module, the count and the fix -- any
core that would break it.

```python
generate(out, module_name="blur", sync_memory=True)
```

## What every core publishes

A generated core reports the memories it inferred, so an integrator reads
the fact instead of grepping for it:

```python
core["memories"]
# [{"instance": "u_lb0_1", "module": "blur_linebuf",
#   "bits": 12, "depth": 1920, "read": "sync"}, ...]
```

`read` is `"sync"`: every memory is behind its seam, its `module` the one to
bind. A composition can gather these into its build report; revela does.

## What this does not cover

- **Constant tables** (`np2hw.Rom`) are built as the device decides
  (`Device.rom_in_logic`): a small one as logic, its contents constants
  read in the stage that uses them, and a larger one as an initialised
  memory read through a register, which the tools build as a ROM. Either
  way they are not behind the seam.
- **The clock-crossing FIFO** (`cdc_fifo`) is a dual-clock memory, which
  is a different macro and a different contract; its seam is a separate
  one (Next).
- **Verification on silicon.** Everything here is proven bit-identical
  in simulation and on an FPGA. That a given foundry macro honours the
  contract is the integrator's to verify, and this document is the
  statement of what to verify.
