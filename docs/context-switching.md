# Design note: time-multiplexed context switching

**Status: design note, nothing implemented.** This records the decisions that
make a context-switching design *possible later without reworking np2hw*, so
that code written in the meantime does not quietly close the door. Read it
before adding anything stateful to the emitters or anything global to the
stream contract.

## The pattern

N sensors stream through ONE physical datapath, interleaved in time, with the
per-sensor context — configuration registers, statistics, window state —
switched as the stream switches. One ISP at N× pixel rate instead of N ISPs:
a classic area trade, and a real product architecture (multi-camera ISPs have
shipped this way for decades).

Nobody is building this now. The point of the note is that the cheapest moment
to keep it buildable is before any code exists that assumes otherwise.

## The one contract decision: the context ID travels IN-BAND

With an elastic `valid/ready` stream and any buffering at all, **"which sensor
does this beat belong to" cannot be derived from time**. A beat sitting in a
skid buffer across a switch boundary belongs to the *old* context; a global
"current channel" register is wrong the moment backpressure exists, and wrong
in a way that only appears under load.

So, when the capability is built, the channel ID is a **sideband travelling
with the data** — the same idea as AXI-Stream's `TID` — and every stateful
element indexes its state by the ID *at the point of use*. Everything else in
this note is mere banking; this is the contract.

The extension points already exist and are additive:

- `StreamType(data_bits, flags, domain)` grows an optional `id_bits`;
- `compose()` propagates the sideband exactly as it propagates flags;
- `testing.Beat` grows an optional `ctx` tag, and `run_frame` a channel
  argument, rather than a second driver function.

Blocks that hold no state ignore the sideband entirely.

## Why the trace layer never changes

Context is orthogonal to dataflow. `to_ir()` and `lower()` describe *what is
computed per pixel*; whose pixel it is touches only *emission*. All per-context
state in a generated core lives in exactly three named structures:

    line buffers            (vertical delay)
    per-row shift registers (horizontal delay)
    row / col counters      (position)

Banking any of them is a multiplicative depth plus an ID-prefixed address on a
structure the emitter already names. That is the whole reason "no dramatic
change" is true, and it is worth protecting: **new emitter state goes into one
of those structures, or gets its own named structure — never scattered.**
Scattered state is what turns banking into a rewrite.

## What already aligns, deliberately

- **Register maps are types + instances** (`RegBlock` / `RegInstance` /
  `AddrMap`). Five contexts of one pipeline are the same `RegBlock`s
  instantiated five times — the software-visible face of a context-switching
  design is expressible today with zero new concepts, and a banked register
  file is *derivable* from an AddrMap whose instances share types: "N
  instances of one RegBlock" is exactly the information the bank mux needs.
- **The shadow→live commit generalises.** A context switch is "select a
  different live bank at a boundary" — the same mechanism the frame-boundary
  commit already uses, with N live sets instead of one. `axil_regfile` today
  has exactly one `live_` bank; `banks=N` plus a select input is an additive
  parameter.
- **Geometry is runtime, not build-time.** `active_width` registers, EOF
  (height-agnostic) framing, phase registers: one bitstream already serves
  every width, height and CFA order it was sized for. Per-context geometry is
  the same machinery, selected per ID. This is the standing rule — per-sensor
  facts are registers, never trace-time constants — and context switching is
  the second independent reason for it.

## Cost scales with interleave granularity

| Granularity | Banked state | Cost |
| --- | --- | --- |
| Frame-interleaved | none — `sof` re-anchors the counters and the line buffers refill before `out_valid` gates on | nearly buildable today: N-bank regfile + input arbiter |
| Line-interleaved | line buffers, shift registers, counters | moderate, additive to the emitters |
| Beat-interleaved | everything, including in-flight stages | deepest; viable only *because* the ID is in-band |

## The missing component is a stream ELEMENT, not a block

The input scheduler (N streams in, one tagged stream out, per-channel `ready`)
is not traced from a model and is not application logic. It belongs to the
same family as the buffered fork the netlist checks currently refuse for want
of: **stream elements** — structural, handshake-owning components np2hw emits
directly, like the adapters and the register file. When either is built, build
the category, not a one-off.

## A sibling pattern: mixed-resolution capture

Same note, second pattern, because it lands on the same contract. Run ONE
sensor at alternating modes — a full-resolution key frame, then several
binned/scaled frames — and reconstruct detail in the cheap frames from the key
frame at upscale time. Sensor readout, link and ISP clock all scale with pixel
count, so the average power drops roughly with the binning ratio; the field
has since validated the family (temporal and reference-guided
super-resolution). Never implemented at the time; kept buildable here.

What it needs, in np2hw terms:

- **Per-frame mode switching: already in.** Geometry is runtime registers
  (`active_width`, EOF framing), commits happen at frame boundaries, and the
  host control loop already writes sensor registers per frame. A key/binned
  cadence is software policy over existing mechanism.
- **Frame TYPE is in-band, or it is wrong.** "Is this beat from a key frame or
  a binned one" is the SAME decision as the channel ID above, at per-frame
  granularity — one tag definition serves both patterns. Deriving frame type
  from a mode register races the stream the moment anything buffers.
- **Multi-image tracing** — the reconstruction block consumes two streams (the
  live binned frame and the stored key frame). `to_ir()` traces one Image2D
  today; this is the second customer for multi-input tracing, after the HDR
  merge. The guided-upscale arithmetic itself is stencils + pointwise + mux —
  traceable already, once two inputs exist.
- **A frame store is a stream ELEMENT.** The key frame must persist across
  frames, which means stream-to-memory and memory-to-stream elements with an
  explicit external-memory port. That extends the "no DRAM round-trip" stance
  honestly: no HIDDEN round-trip — memory becomes a declared element in the
  netlist, visible to the same validation as everything else, never an
  assumption inside a block.
- **Reductions** (already on the roadmap for statistics) are what any motion
  or match metric will need; a static-scene or globally-aligned first version
  needs none.

So the pattern adds no new contract — it adds two entries to the stream-element
category (frame store in, frame store out) and seconds two roadmap items that
other blocks already want (multi-input tracing, reductions).

## Rules in force now (they cost nothing)

1. Never derive context from time. No "current channel" or "current frame
   type" global in anything new; identity rides the stream or does not exist
   yet.
2. Per-sensor facts are registers. No trace-time capture of anything a second
   sensor — or a second sensor MODE — would need different.
3. Emitter state stays in named structures.
4. The verification models take a channel tag when they grow, rather than
   forking into per-topology variants.
5. Memory is an explicit element. No block quietly assumes a frame of storage;
   a frame store appears in the netlist or it does not exist.
