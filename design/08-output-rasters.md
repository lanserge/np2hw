# 08 — Output rasters: crop, and what upscale and downscale still need


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). Written 2026-08-22, when `make_coords`
> landed and the crop case closed; the rest is what the same idea needs next.

## The idea, in one line

A block that changes geometry must state the **map from output position to
input position**. Not its output shape — the map. A shape says how big the
output is and not where it came from, and the difference is the whole
problem.

## What was wrong before

Output geometry had two owners. The tracer carried a `shape`; the emitter
built its valid region as `[M, HEIGHT) x [N, WIDTH)` from the kernel margin
and the frame parameters. They were computed in adjacent lines of
`generate()` and never compared, and they agreed only because every model
was a stencil, for which `shape == (H-M, W-N)` genuinely holds.

`img[0:4, 0:4]` on an 8x8 reported a 4x4 output and emitted all 64 beats.
Silently — which is the expensive part, because a core emitting four times
the pixels its model describes is found downstream as a picture that is
wrong for no visible reason.

Both halves are fixed: a model whose extent the region cannot express is
refused with both numbers, and a model that states its map is built from
it.

## What crop cost, and why it was cheap

Almost nothing new. A crop's arithmetic was always a single tap, so the
datapath is the machinery that was already there; only the gate and the
framing changed. `make_coords(img, rows, cols)` declares the output raster
and returns its position, exactly as `coords(img)` does for the input. The
leaf's own range carries the extent (`hi + 1`), so nothing had to be stored
beside it.

One existing rule learned a distinction. `lower()` normalises a stencil's
smallest tap offset to zero, which is right: for a stencil that offset only
shifts the whole window and the margin already expresses it. For a crop the
offset is the window's POSITION and means something, so it survives.

## Downscale — the next cheapest

Two forms, and only one needs anything new.

**Nearest** is `img[y2 * 2, x2 * 2]`: output beats are a subset of input
beats in time, so `in_ready` never changes and the output is gated. What is
missing is that `_decode_output_index` refuses a multiply — it accepts
`out + k` only. A ratio needs the decoder to return `(extent, offset,
stride)` and the emitter to advance its output position by the stride.

**Box** is a stencil followed by decimation: compute the 2x2 average at
every position — traceable today, geometry-preserving — then emit one
sample in four. It needs no reduction over a variable extent; four taps and
a shift is ordinary arithmetic. This is worth saying plainly because
"scaling needs reductions" is the wrong conclusion and was drawn once
already: **reductions are what the statistics block needs** (accumulate a
zone), and scaling is not that.

## Upscale — the one that breaks an invariant

The emitter can already emit more beats than it consumes. `_generate_edge`
and `_generate_phase_stencil` do it during their flush: the datapath
advances, `out_valid` asserts, `in_ready` is withheld, and the framing is
computed from flush counters rather than input counters. That is precisely
an upscaler's mechanism, already written and tested.

What upscale adds beyond it:

- an output raster counter that advances independently of `erow`/`ecol`,
  rather than a constant offset from them;
- input hold, which the `in_active` pattern already expresses;
- input-row replay — the one place a line buffer appears that a 1:1
  pipeline would not have had;
- framing derived from the output counter, which **contradicts a documented
  invariant**: `docs/framing-and-resolution.md` says there is no separate
  output counter to drift. That sentence has to change or the feature does.

There is also a budget assumption to unpick. The flush was designed to fit
inside blanking (`verilog.py`, `_generate_phase_stencil`'s `hblank`/`vdrain`
metadata). A 2x upscaler needs three extra beats per input beat, which no
blanking budget covers, so the input is genuinely back-pressured and `skid`
becomes mandatory rather than advisory at a scaler boundary.

## The canvas (pad to a larger raster)

Same shape as crop but outward: an offset with no ratio, a declared extent
larger than the input, and a fill value outside the window. It needs the
flush generalised to arbitrary hold counts. `scanout` in `video_out.py`
already proves the RTL shape — `fits`, `in_win`, `w_x0` and a `FILL`
constant — but does it outside the traced world.

## What the emitter can already derive, and should

Bounded history is not a new analysis. `SourceLine.lag` and
`VProcLine.line_buffers = max(lags) - min(lags)` already compute exactly
"how far back must this reach". For `img[2*y2, 2*x2]` that is zero; for a
2x upscale it is one row replayed. Both are constants derivable from the
map.

The refusal pattern exists too. Every `PExpr` carries an exact `(lo, hi)`,
and `_gather_from` already refuses an out-of-range index at trace time
because *hardware has no IndexError*. The same interval check over a
coordinate map gives "this map reaches K rows back, which exceeds the line
buffers" or "this map is not monotone in the output raster, so it cannot be
served from bounded history". Refuse at generation time; never emit a core
that silently reads the wrong pixel.

## Conflicts to settle before building more

1. **`StreamType` has no word for geometry** (`compose.py`: data bits, flags,
   domain). A scaler is the first block whose output differs from its input
   in a way the type system cannot say, so a downstream core parameterised
   for the pre-scale width would elaborate happily and be wrong. This is the
   class of error `StreamType` exists to prevent.

2. **One geometry per pipeline** is a stated rule downstream in revela
   (`blocks/pipe.py`, `docs/design-rules.md`): one width register read by
   everyone, because a width register per block makes a resolution change N
   writes that must all land in the same frame. A scaler makes it false for
   blocks after it. The rule may survive — post-scale geometry is DERIVED
   from the context by a fixed map, not separately declared, so there is
   still one owner — but that is a decision to take deliberately.

3. **Two rasters mean two frame boundaries.** `frame_sync` commits registers
   on `out_last`. Which boundary commits a register downstream of a scaler
   is an unforced choice that will be made accidentally if it is not made
   on purpose.

4. **`coords()` currently implies pointwise, no line buffers**
   (`ir.py`'s `ExprLine`). A gather that reads a window *and* uses
   coordinates needs the pointwise and windowed emitters joined;
   `_generate_phase_stencil` is where they most nearly meet already.

## The open question

Should the model state only the extent and let the emitter recover the
ratio by pattern-matching the index arithmetic, or should a block declare
its rate?

Stating the extent is the elegant answer and matches how ranges already
carry extents — but the emitter must then recognise the affine forms a
model might be written in, and a model written in an unrecognised form gets
refused for reasons that look arbitrary. Declaring the rate is unambiguous
and duplicates a fact the map already states, which is the failure mode
this project names most often.

Crop was implemented the first way and it reads well. That is one data
point, at the easiest ratio there is: 1:1.
