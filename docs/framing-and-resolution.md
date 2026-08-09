# Framing & resolution

How frames are delimited, and how one generated core can handle different image sizes —
from per-synthesis parameters up to fully runtime-variable resolution.

## Frame framing: SOF / EOL / EOF

The core carries frame markers on its streaming interface:

- **Inputs**: `in_sof` (Start Of Frame — pulse on the first pixel) re-anchors the
  internal counters to (0, 0). Tie it to `0` to free-run by counting. The AXI4-Stream
  Video adapter wires `s_axis_tuser → in_sof`, so the IP is a real video slave that
  locks to **TUSER** rather than only counting pixels.
- **Outputs**: `out_sof`, `out_eol` (End Of Line), `out_last` (End Of Frame). These are
  **derived from the (effective) input position** and registered alongside `out_data`,
  so they re-anchor with SOF — there's no separate output counter to drift.

This means power-up mid-stream, or a master that frames with TUSER, both work: the core
locks to the real frame. (See `examples/axi_video.py` — a multi-frame TUSER/HBLANK/VBLANK
stream validated against NumPy.)

## Continuous (multi-frame) operation

The cores **re-arm** at the end of a frame (counter wraps to 0) rather than latching a
`done` flag, so they free-run across frames — continuous video, one reset. The cxxrtl
backend exercises this (60-frame GIF == NumPy with a single reset).

## EOF-driven, height-agnostic framing

`generate(out, name, framing="eof")` makes the core **not count to HEIGHT**. Instead an
`in_eof` input (the sensor's frame-end / VSYNC, pulsed on the last input pixel) triggers
the bottom flush; the flush drains the line buffers and **`out_last` falls out of the
flush ending** — no HEIGHT anywhere. The same core then processes **any number of
lines**:

```python
core = generate(out, "gcore", framing="eof")     # edge core; needs a vertical edge
```
Only `WIDTH` is fixed (it sizes the line buffers); height comes from the stream.
(See `examples/eof_framing.py` — a core built for HEIGHT=4 runs 6/12/20-row frames,
each bit-exact.)

## Resolution

### 1. Per-synthesis parameters (default)

`WIDTH`/`HEIGHT` are emitted as **Verilog `parameter`s** (not macros, not registers).
Every size-dependent expression rides them (line-buffer depth `mem[0:WIDTH-1]`, the
bounds, the framing thresholds); only kernel constants (`M`, `N`, coefficients) are
baked. So one generated source handles any resolution **by overriding the parameter at
instantiation**:

```verilog
gcore #(.WIDTH(1280), .HEIGHT(720))  hd  ( ... );
gcore #(.WIDTH(3840), .HEIGHT(2160)) uhd ( ... );
```
Resolution is chosen at **elaboration/synthesis** — one RTL, many builds. (Verified: a
core generated at 8×6 runs correct at 16×12 via parameter override.)

### 2. Runtime-variable resolution (dynamic)

For a **reprogrammable sensor** (resolution changes live, one bitstream),
`generate(..., framing="eof", max_width=N)` splits WIDTH's two roles:

| role | becomes |
|---|---|
| line-buffer depth (must be fixed at synth) | **`MAX_WIDTH`** parameter (`mem[0:MAX_WIDTH-1]`) |
| where each row ends (wrap / EOL / right-edge) | **`active_width`** register input |
| frame height | **`in_eof`** / VSYNC (no HEIGHT counter) |

So one synthesized core handles **any W×H ≤ MAX_WIDTH**, set at runtime:

```python
core = generate(out, "gcore", framing="eof", max_width=1920)
```
(See `examples/dynamic_res.py` — one core, MAX_WIDTH=64, runs 10×12, 20×6, 7×9, 64×3.)

`active_width` is a normal config register, so it rides the AXI-Lite/UMI register file
like any tuning register — frame-synced so a resolution change lands cleanly at the
frame boundary. The `control_top` generator wires `param_active_width → core.active_width`
for you; see [interfaces.md](interfaces.md) and `examples/active_width.py`.

### Choosing

| You want… | Use |
|---|---|
| fixed resolution per build, smallest hardware | parameter override (default) |
| variable **height** only (line-based pipeline) | `framing="eof"` |
| variable **width and height** at runtime | `framing="eof", max_width=N` + `active_width` register |
