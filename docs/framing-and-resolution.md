# Framing & resolution

How frames are delimited, and how one generated core handles frames of
different sizes. The rule: **the build fixes capacity, the stream states the
facts.** A core is generated at the largest frame it must hold; each frame's
own size arrives with the frame.

## Frame framing: SOF / EOL / EOF

The core carries frame markers on its streaming interface:

- **Inputs**: `in_sof` (Start Of Frame — pulse on the first pixel) re-anchors the
  internal counters to (0, 0). Tie it to `0` to free-run by counting. The AXI4-Stream
  Video adapter wires `s_axis_tuser → in_sof`, so the IP is a real video slave that
  locks to **TUSER** rather than only counting pixels.
- **Outputs**: `out_sof`, `out_eol` (End Of Line), `out_last` (End Of Frame). These are
  **derived from the (effective) input position** and registered alongside `out_data`,
  so they re-anchor with SOF — there's no separate output counter to drift.
  A crop keeps this: its framing is the input position gated to a window. A block
  that changes RATE could not, which is the invariant
  [`design/08`](../design/08-output-rasters.md) says has to give before an
  upscaler exists.

This means power-up mid-stream, or a master that frames with TUSER, both work: the core
locks to the real frame. (See `examples/axi_video.py` — a multi-frame TUSER/HBLANK/VBLANK
stream validated against NumPy.)

## Continuous (multi-frame) operation

The cores **re-arm** at the end of a frame (counter wraps to 0) rather than latching a
`done` flag, so they free-run across frames — continuous video, one reset. The cxxrtl
backend exercises this (60-frame GIF == NumPy with a single reset).

## A frame's size

Every core takes its frame's size at run time, in its stream's context, and
hands on the size of the frame it puts out. A stream's context is its `ctx`
member (`CTX` in `np2hw.stream.STREAM_SIGNALS`): each frame's facts, held with
its first beat, packed in the order the design's vocabulary declares them, the
first in the low bits -- the frame's `width` and `height` first, 16 bits each,
where the stream declares them. An image's `context` is the layout; by default
it is the size alone.

| port | |
|---|---|
| `in_ctx` | the arriving frame's context; it holds from the frame's first pixel (`in_sof`), and the core keeps it for the rest of the frame |
| `out_ctx` | the context of the frame this core puts out: the same fields, the size the frame out has -- the input's, or less a window's border, or a crop's own -- and every other field as the frame brought it |

So one generated core serves a sensor whose mode changes live: a new size lands
at the next frame's start. A frame is at least two pixels wide and two lines
high, the context's contract: at a frame's first pixel nothing a core decides
depends on the size, so the core compares against the size it latched as that
pixel was taken, and the context reaches only the latch. A window still flushing
one frame finishes it at that frame's size while the next waits. The size a crop hands on follows from what it reads:
`img[1:-1, 1:-1]` puts out the frame less its border, whatever the frame;
`img[2:10, 3:15]` always puts out 8×12, and needs a frame that reaches row 10
and column 15.

### Capacity: the size it is generated at

What a core HOLDS is fixed when it is generated, from the size its model is
traced at:

- a **window** keeps lines, so the traced width is the depth of its line
  buffers, and frames may be any width up to it. Its height bounds nothing: a
  window core takes a frame of any height.
- a **video's past frames** are kept whole, so the traced width and height are
  the frame store's depth, and frames may be up to that size.
- a model that reads its **position** (`coords`, a crop's window registers, a
  shading profile) has that arithmetic sized for the traced frame, so it is
  exact for frames up to the traced size.

The generated module has no size parameters to override: a different capacity
is a different generation, so nothing in the RTL can claim a size the
arithmetic was not built for.

### Who states the size

- **In a pipeline**, the context travels with the stream: each block's is the
  one the block before it hands on, and the first block's is the frame's own
  where the stream carries it (a bayerlink header's width and height, declared
  on the pipeline's input port).
- **A design whose stream carries no size** has the context block's registers
  supply it, joined with what the stream does carry into the one layout every
  core of the pipeline takes; their defaults are the size it is built for.
- **A wrapper of one core** (`control_top`, the Switchboard and AXI4-Stream
  wrappers) offers each context field as a register port like any other, its
  default the traced size, and packs them into `in_ctx`
  (`np2hw.verilog.offered_params`), so a register file nobody configures runs
  the frame the model was traced at, and software sets a new size the way it
  sets any register, landing at a frame boundary.
- **A testbench** drives the context of the frames it streams;
  `np2hw.verilog.testbench` and `np2hw.testing.block` do, and
  `np2hw.verilog.context_word` gives the word for a hand-written one.
