# Clock crossings: what an integrator must constrain

`cdc_fifo` is the one crossing every design this generator feeds
contains — a clock recovered from a cable meeting the board's own — and
it is the piece a static timing tool cannot reason about without being
told. This is what to tell it.

Nothing here is a suggestion about how to design the crossing; that is
settled inside the module. It is the intent a signoff flow needs, stated
by the module that owns it rather than rediscovered by whoever
integrates it.

## The crossing, precisely

Two clock domains, `wclk` and `rclk`, unrelated. **The payload never
crosses.** It is written into memory in the write domain and read out in
the read domain, and it is only read once the reader's own arithmetic
says the slot is safely full. What crosses is two pointers, gray-coded,
each through a two-flop synchroniser:

| signal | from | to | width |
| --- | --- | --- | --- |
| `rgray` → `wq_rgray0` → `wq_rgray` | `rclk` | `wclk` | `A+1` |
| `wgray` → `rq_wgray0` → `rq_wgray` | `wclk` | `rclk` | `A+1` |

The first stage of each pair carries `(* ASYNC_REG = "TRUE" *)`, which
asks a placer to keep the two flops of a synchroniser adjacent. That
attribute is understood by FPGA tools; an ASIC flow ignores it and wants
the equivalent stated in its own constraints.

**Why gray.** A binary pointer sampled mid-increment can be misread as
any value. A gray-coded one changes a single bit per step, so a sample
caught mid-transition is wrong by at most one slot — and one slot of
pessimism costs a slot of headroom and nothing else. That is the whole
argument, and it is why the constraint below is safe.

## What to constrain

**The two pointer paths are asynchronous.** They are synchronised by
construction, so a tool must not try to meet setup or hold across them.

In Xilinx XDC terms, the two clocks declared asynchronous to each other:

```tcl
set_clock_groups -asynchronous \
    -group [get_clocks <write domain clock>] \
    -group [get_clocks <read domain clock>]
```

A flow that prefers per-path constraints wants a **max-delay on the
pointer paths, datapath-only**, so the synchroniser's first stage is not
asked to meet a launch-clock relationship it cannot have:

```tcl
set_max_delay -datapath_only -from [get_cells {*rgray_reg[*]}] \
              -to [get_cells {*wq_rgray0_reg[*]}] <one rclk period>
set_max_delay -datapath_only -from [get_cells {*wgray_reg[*]}] \
              -to [get_cells {*rq_wgray0_reg[*]}] <one wclk period>
```

The bound matters and should not be relaxed to infinity: gray coding
tolerates a sample caught mid-transition on **one** bit. If skew between
the pointer's bits exceeds a destination clock period, two bits can
change as seen by the sampler and the guarantee is gone. Constraining
the path to a period preserves it.

**Nothing else crosses.** The memory is written in one domain and read
in the other, but never in the same cycle at the same address by
construction — the reader is held off until the writer's pointer, safely
synchronised, says otherwise. A CDC checker that flags the memory is
seeing the structure and not the protection; the pointer synchronisers
are the protection, and they are what to point it at.

## Reset

**Both sides carry their own reset and a link reset must assert both** —
the write side directly, the read side through its own domain's
synchroniser. Pointers reset on one side only disagree forever after:
one side believes the FIFO holds `2**A` entries and the other believes
it is empty, and no traffic corrects it.

This is the board wiring's contract, not the module's, because only the
integrator knows what event constitutes a link loss. The module states
it; it cannot enforce it.

## For an ASIC flow

Three differences from an FPGA target:

- **`ASYNC_REG` means nothing.** Ask the place-and-route flow for
  synchroniser cell placement in its own terms, and prefer library
  synchroniser cells for the two-flop pairs if the library has them.
- **The memory is behind a seam.** `cdc_fifo`'s storage is an inferred
  array, not yet behind the module boundary `docs/memories.md`
  describes for line buffers. A dual-clock memory is a different macro
  and a different contract; binding one here is the integrator's, and
  the read/write discipline above is what it must preserve.
- **A CDC signoff tool will want waivers**, and the two pointer paths
  are the only ones that need them. Anything else it flags in this
  module is worth reading carefully rather than waiving.

## What this document is not

It is not a verification of the crossing on silicon. The discipline is
proven in simulation and on an FPGA, and the constraints above are what
that discipline requires a tool to be told. Whether a given flow's
synchroniser cells and a given library's setup/hold satisfy it is the
integrator's to establish — this states what must hold, not that it
holds in your process.
