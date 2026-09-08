# Changelog

Notable changes to np2hw. The claim this project makes is that generated
Verilog is bit-exactly equal to the NumPy model it came from; anything
here that changes an *emitted width* is called out, because a width that
moves silently is how a downstream integrator reads the wrong bits.

Pre-1.0, so the API can still move. Anything that moves is named here.

## Unreleased

### Added

- **The memory seam.** Every line buffer a registered-read stencil emits
  (the edge and CFA-phase emitters) now lives in its own module,
  `<module>_linebuf` — one write port, one read port, both synchronous,
  read-before-write on collision, no reset — instantiated once per
  buffer. On an FPGA it infers block RAM exactly as the inline array did;
  on silicon it is the module an integrator binds a memory macro to, by
  name. The contract is on the module and in `docs/memories.md`.
  Behaviour is bit-identical: the array moved, its enables did not.
- **`generate(..., sync_memory=True)`** refuses a core whose line
  buffers read combinationally — the general window and the mux — by
  module name, with the count and the fix. No SRAM macro implements a
  combinational read; without the flag those emitters are unchanged,
  because on an FPGA distributed RAM is the right answer for them.
- **`core["memories"]`** — every generated core publishes the memories it
  inferred: instance, seam module (or `None`), width, depth, and whether
  the read is `sync` or `async`.
- **The elastic stream has an IP-XACT bus definition.**
  `np2hw.ipxact` renders the bus/abstraction definition pair, and every
  stream port now carries the matching `X_INTERFACE_INFO`. Both are
  generated from `np2hw.stream.STREAM_SIGNALS`, so a tool's view of the
  protocol and the RTL's cannot disagree. With the definitions in an IP
  catalogue, a packager sees each stream as one interface with its
  master/slave role inferred, instead of six loose wires. Flags are
  declared `optional` because a core takes SOF and regenerates the rest;
  see `docs/ipxact.md`. Declared on the composed top and on every module
  a board design instantiates by hand -- `skid`, `tee`, `cdc_fifo`,
  `bayerlink_in` -- and `cdc_fifo`'s two clocks each name the face they
  carry, which is the one fact a dual-clock crossing exists for. A
  generated core's internal blocks are wired by the composer, which
  emitted both ends and infers nothing, so they are left alone.

### Changed

- **Every AXI4-Lite slave port declares its bundle.** `axil_regfile`,
  `control_top` and `control_wrap` carry `X_INTERFACE_INFO` on every
  bus port, clocks name the bus they drive and its reset, and resets
  state their polarity — as `ddc_slave` and `framebuffer_read` already
  did for their masters. Verilog attributes only; simulation and
  synthesis ignore them, and an IP packager reads exactly these.
- The two-stage phase stencil's registered write address is
  `POS` bits wide, not 32. Same values, sized by construction.

## 0.5.1

### Fixed

- **`control_wrap` declares the arm handshake it wires.** The `arm_req` /
  `arm_ack` pair runs between the register file and the datapath, so it is
  neither a port of the wrapper nor bound by a caller — and having been
  dropped from the port list it was then declared nowhere, leaving both
  signals implicit.

  Legal Verilog, one bit wide by default, and correct here only because
  they happen to be one bit. Verilator refuses them and is right to: the
  same omission on a multi-bit signal is silent truncation rather than a
  warning. Synthesis accepted it, which is why a board build never
  complained.

## 0.5.0

### Added to the public API

- **`saturate(value, bits)`** — saturate to a width, so the emitted width
  is the scale. The trace pins a range, and pinning it explicitly is how
  a model states the width it means rather than leaving it inferred.
- **`make_coords`** — a crop states its map, and the core is built from
  that.
- **Stream plumbing: `skid`, `tee`, `cdc_fifo`, `combinational_outputs`,
  `boundary_report`.** A core's timing claim is about its own registers
  and holds only while nothing else is on the path; a combinational
  output hands its depth to every consumer, so two cores that each meet
  timing alone can miss it together. `skid` is how a boundary stops being
  a place where that happens, and the reports are how a composition says
  where it still is one. `tee` faces one stream at two consumers and
  never stalls — a consumer that falls behind tears its own branch and
  says so. `cdc_fifo` is the one crossing every design this generator
  feeds contains: a recovered clock meeting the board's, owned so it can
  follow the design to another family or an ASIC, and so its discipline
  is checked rather than trusted.
- **`framebuffer_read`** — a framebuffer reader that cannot be out of
  phase, and which says why it is idle rather than sitting silent.
- **`ddc_slave`** — the display cable becomes the control plane: an I2C
  slave on the DDC pins, so parameters reach an FPGA over the video cable
  it is already attached to.

### Changed

- **A packed lane is as wide as its values, not as wide as it arrived.**
  A stacked word's fields were sized from the *input* sample width, and a
  channel needing more was refused. That was right while every model
  carried its samples through unchanged, and wrong the moment one
  narrowed them — a display curve, which is exactly where a wide linear
  signal becomes display values, was made to ship 8-bit results in 12-bit
  lanes and leave the truncation to whoever unpacked the word.

  The field is now the traced element width, the widest lane sets it for
  all of them, and the published `field_bits` follows the width the word
  was actually packed at. **If you consume a stacked word's channels, read
  `field_bits` rather than dividing the total by the channel count.**

- **The window's arithmetic in `scanout` gets a clock of its own.** Where
  the picture sits is frame-constant but was recomputed beneath every
  pixel. While placement was a build-time constant that cost nothing,
  because a constant folds away; the first build that put it on real
  registers missed setup by 124 ps, with nineteen levels of logic and
  seven carry chains under the pixel path at 74.25 MHz. Readback now
  follows the copy the raster is *using*, so it reports where the picture
  is rather than where it is about to be.

- **The receiver aligns depth on the way in** and counts what a link
  really does — settling the depth, and reporting which camera and which
  frame.

- **A stream's geometry is stated where its width already is**, and the
  output extent has one owner: a crop is refused rather than emitted
  whole.

### Internal

- Every source file carries its copyright and SPDX identifier, checked by
  `scripts/checklicence.py`. A script rather than a test, because this
  project's `examples/` **are** its suite — each runs a NumPy function
  through the compiler and compares the simulation bit-for-bit — and a
  licence check demonstrates nothing about the compiler. The identifier
  is read from `pyproject.toml`, never restated.
