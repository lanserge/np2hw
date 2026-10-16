# Changelog

Notable changes to np2hw. The claim this project makes is that generated
Verilog is bit-exactly equal to the NumPy model it came from; anything
here that changes an *emitted width* is called out, because a width that
moves silently is how a downstream integrator reads the wrong bits.

Pre-1.0, so the API can still move. Anything that moves is named here.

## Unreleased

### Changed

- **A window's bounds are registered.** Its last row and column -- the
  registers that place it, added -- are worked out once into a register,
  frame-constant as they are, and compared against; the sum and the
  compare in one cycle were a carry chain too long for the clock.
- **An implementation that is a module takes its settings as parameters.**
  A block filled by a Verilog implementation gets its `config`, evaluated
  over the block's parameters, as the module's parameters -- as any module
  leaf takes its build -- where it used to be refused as vendor IP
  properties.

- **A context register exists only where something reads it.** A
  register-backed context signal is made -- in the decode, the published
  map, the ports and the binding -- only where a block consumes it, or the
  cores' run-time frame reads it as the frame's width or height; stream-
  carried signals keep their ports as before. A vocabulary may declare a
  signal nothing in a design reads, and that design no longer carries a
  register for it; the first block that consumes it brings it back. No
  remaining register moves: the allocation is unchanged.

- **A composed pipeline follows the frame it is given.** Where a design's
  context vocabulary has a signal following `width` and one following
  `height` that a frame can bring (not a register-only choice), every core
  is built with `frame="runtime"` and bound to them: one core processes
  any frame up to the design's `max_width` x `max_height`, the frame's
  size latched as each frame enters. It used to count to the build's size
  whatever arrived, so a smaller frame ran its lines together and never
  ended -- and a batch committed at a frame end was never taken. A stage
  sees the frame less what the stages before it took off (a valid
  window's span). A register that follows `width` or `height` now resets
  to the design's size, not the vocabulary's default.
- **`generate(..., frame="runtime")`**, on both compilers and every
  generator: the core takes `frame_width` and `frame_height` parameter
  ports (defaulting to the traced size, so a testbench driving each port
  its default runs the traced frame) and keeps `WIDTH`/`HEIGHT` as the
  most a line buffer holds. A window cropped to a stated extent refuses
  it; it does not combine with `max_width` or `framing="eof"`, which
  follow the frame by other means. `frame="build"` (the default) emits
  what it always did.

- **Fixed modules are Verilog files.** `skid`, `tee` and `cdc_fifo` were
  Verilog printed by Python, the same structure every time. They are
  files in the package now -- `np2hw/hdl/np2hw_skid.v`, `np2hw_tee.v`,
  `np2hw_cdc_fifo.v` -- sized by module parameters. The functions return
  the file with the `parameters` a use sets; a result that instantiates
  one lists it in `requires`, and `np2hw.hdl.closure(result)` is the text
  with everything it requires. **`module_name` is gone from all three**:
  a fixed module has one name.
- **`Block.run` hands the model the type the trace gives its input.** The
  frame is cast to the smallest unsigned integer holding `bit_depth` bits
  (`uint8` for an 8-bit stream), so a model that adds before widening wraps
  in NumPy exactly as in the generated RTL. It used to keep the caller's
  type (often `uint16` or `int64`), and such a model disagreed with its own
  hardware; only the twin caught it. A frame with a value outside
  `bit_depth` bits is now refused instead of being passed through.
- **A run of the models reads each stage's depth from the trace.**
  `np2hw.pipeline.run.run_model` tells each block the depth of the word it
  is given as the composer does, from its driver's traced output
  (`compose.handed_on`). **`out_depth` is gone from `block`**: it declared
  a block's output depth beside the trace that already published it, and
  only the model run read it.
- **An uncalled leaf generator's build is not evaluated** (a load with
  `generate=False`): it would be spent making the module, and may read
  facts only the generated design has.
- **A primitive's declared default is its instance's value.** An instance
  carries every parameter its block declares a default for, so the wiring
  check, the toolchain and the facts read one value; the wiring check used
  to refuse a width reading a defaulted parameter as never set.
- **Geometry is read from the trace too, by both.** `compose.handed_on`
  returns the channels, depth, width and height a block hands on
  (`compose.Handed`). The composer no longer falls back to the design's
  geometry for a block that states none -- it refuses -- and a model run
  builds each block at the frame it is given and refuses a model whose
  frame comes out other than its trace says. A window's trace is a bound
  (`compose.keeps_window`): its model may hand on any frame inside it.
- **The RTL twin ends a frame where the hardware does**, at `last`, and
  reads its size from the `<output>_frame_width`/`_height` the top
  publishes, so a chain that keeps a window is proven against its model's
  window. It used to wait for as many words as went in.
- **`np2hw.testing.block` checks the framing of the model's frame**, not
  the frame it drove, so a block that keeps a window is testable.

### Added

- **A frame's size travels with it.** A core built for a run-time frame
  publishes `out_frame_width`/`out_frame_height`: the frame it hands on --
  the one it was given, less a valid window's or an unpadded stencil's
  margin, or the part of a window the frame holds. The composer binds each
  block's `frame_width`/`frame_height` to its driver's, the first block's
  to the context, and the top publishes `<output>_frame_width`/`_height`
  beside each output stream.
- **A window that registers place**, on the one-graph compiler:
  `img[p.y0:, p.x0:][:p.h, :p.w]`, NumPy's own slice with register bounds.
  The source streams through and only the beats inside the window go on,
  its lines and frame ending at its edges or the frame's, whichever comes
  first -- as the slice has it -- and a packed word passes whole. The
  reference compiler refuses it.
- **`np2hw_frame_cdc`** (`np2hw.hdl.stream:frame_cdc`): `stream_cdc` for a
  stream that carries its frame's size. Each frame's width and height
  cross beside its samples, published from the moment its first beat is
  presented until the next frame's is, so a consumer that places a frame
  has its size before it takes a pixel, and a frame still draining keeps
  its own.

- **One address map per master** (`np2hw.bus.address_maps`): each
  decoder's windows under the master that drives it, walked back through
  blocks that only forward, so a cable's bus beside a processor's is two
  spaces, not one. `address_map(design, master)` names which; with
  several spaces and none named it refuses rather than mixing them.

- **`np2hw.blocks`** -- the block framework, moved from revela and
  relicensed MIT by its author: `block` (the decorator), `Block`,
  `StreamPort`, `ContextBit`, `configblock`, discovery (`registry`,
  `resolve`); `np2hw.blocks.params` (register declarations, address
  allocation) and `np2hw.blocks.stream` (the stream spec and beat helpers).
  The content that writes blocks binds its pipeline context vocabulary in
  its own decorator; Verilog traced from a model carries the licence the
  model's source file states. `tests/` holds the framework's tests, apart
  from the examples.
- **`np2hw.design` and `np2hw.pipeline`** -- moved from revela, relicensed
  MIT by its author. `np2hw.design`: the JSON composite format (its schema,
  `composite.load`), the netlist a loaded design is, the wiring check,
  interfaces and profile values. `np2hw.pipeline`: `load`, the composer
  (`compose`: allocation, the register map, SystemRDL, Verilog) and the RTL
  twin (`run`). Two of revela's conventions left with the move: the context
  block (a pipeline's blocks bring a `Vocabulary` naming it, and its commit
  register) and the licence of generated files and of the map (the one the
  models' sources state). The schema states no context vocabulary, and
  generated modules are prefixed with the models' package. `jsonschema` is
  a dependency.
- **`np2hw.toolchain.vivado`** -- moved from revela, relicensed MIT by its
  author: `emit` (the block design and implementation scripts; `name=`
  names the block design apart from the design), `bdtrace`, and the two
  checks of what Vivado built, `vendor_check` and `checkhwh`.
- **`np2hw.pipeline.pack` and `np2hw pipeline run|generate`** -- moved from
  revela (its `revela.fusesoc` and `revela` command), relicensed MIT by its
  author. `pack.emit` proves a pipeline's Verilog against its models and
  writes the design pack; the FuseSoC generator builds one from a `design`
  parameter. The twin's context comes from the vocabulary
  (`run.synthetic_context`), and `--context SIGNAL=VALUE` sets one signal
  for a run. The default VLNV is `:<models' package>:<pipeline>:0`.
  `python -m np2hw` is the command.
- **`np2hw.testing.run_cocotb` and `np2hw.testing.block`** -- moved from
  revela's tests, relicensed MIT by its author: the Verilator + cocotb
  runner, and the one testbench for every block, which reads what to drive
  from the block's declarations and names the block by reference.
  `np2hw.testing` is a package now; what it exported is unchanged. CI
  installs Verilator, and the `test` extra brings cocotb (Python 3.13 at
  most; the simulations skip above it).
- **A register bus of fixed modules** (`np2hw/hdl`), each simulated against
  strict bus models with random stalls in `tests/`:
  `np2hw_axi3_to_axil` (a processor's AXI3 port onto AXI4-Lite: IDs
  echoed, a burst one access per beat, a write burst answering its worst
  response), `np2hw_axil_decoder` (one master onto N slaves by
  power-of-two windows, DECERR for an address in none), `np2hw_axil_arbiter`
  (two masters onto one slave, alternating), `np2hw_reset_sync`
  (asynchronous assertion, release on the domain's clock). Their library
  descriptions are `np2hw.hdl.bus:axi3_to_axil`, `:axil_arbiter`,
  `:reset_sync`.
- **`np2hw.bus.axil_decoder(sizes, base, span)`** -- the decoder a design
  places: a leaf giving each slave its own `mNN_axi` interface, its
  windows placed by `np2hw.bus.allocate` (largest first, each aligned to
  its size) and published as the leaf's fact `windows`, `{port: [base,
  size]}`. A size of 0 is an optional slave absent from the build: its
  port is not made, and the others keep their numbers.
  `np2hw.bus.address_map(design)` is where every slave sits, for the host,
  read from those windows and the design's edges.
- **The Vivado emitter sets every published window** as the slave's
  segment in the address space of the port it hangs off -- at every hop,
  through a block that only forwards -- then leaves the processor's own
  map to Vivado's assignment. A description may state `facts`; the
  arbiter's `forwards` ({slave port: master port}) is how a window is
  walked through it (`np2hw.bus.hops`), to the slave the host maps.
- **Glue a design places** -- fixed modules `np2hw_const`, `np2hw_slice`,
  `np2hw_not` (described as `np2hw.hdl.glue:const`, `:slice`, `:not`) and
  the leaf generator `np2hw.glue:concat(widths)`, so a design's constants,
  bit picks and concatenations are portable Verilog rather than a vendor's
  cells. checkhwh traces their ancestry like the vendor's.
  `concat(widths, interrupt=1)` marks its output as interrupt lines, level
  high, its width stated -- Vivado sizes a processor's interrupt input
  only from such a driver, and refuses to have it set.
- **`np2hw.toolchain.vivado.rtl.emit_rtl`** -- a design built by Vivado
  from its own Verilog: the composites written as modules, each vendor
  leaf an IP by its cell's name (`create_ip`, its implementation's
  settings), and the leaves that exist only in IP integrator (a procedure,
  or an IP marked `integrator`) one block design, its ports to the rest
  named as the Verilog instances them -- an AXI interface port shaped from
  the signals it meets, any other bundle signal by signal. The described
  vendor blocks' pins are dumped for vendor_check: a block design's from
  it, the rest from the elaborated design.
- **`np2hw.design.verilog.modules(design)`** -- a loaded design as Verilog,
  one module per composite: children instanced by name, connections as
  wires signal by signal, a composite's own ports what they meet inside (a
  bus port a bundle named `<port>_<signal>`), external pins rising to the
  top, every pin of an instance written (an empty one where nothing is
  wired). A vendor leaf is a black box by its cell's name and its
  implementation's pins (`vendor_check.vendor_pins`, the one statement of
  that mapping). A composite instanced twice is one module where both
  write alike. The loader keeps each composite as written
  (`Design.hierarchy`), and a bundle's facts list its members.
  `modules(design, groups={name: [leaves]})` folds leaves a toolchain
  makes together (one block design) into one black box, its ports what
  its members meet outside -- as a member states the port, or the far
  end's mirror -- named `<cell>_<port>`; `group_ports()` gives the
  toolchain the same list. A parameter past 32 bits is written sized.
- **A second master's reach is recorded, and kept out of the host's
  map.** `ddc_slave` publishes its master's window (`[0, 2**AW)`); the
  Vivado emitter sets it where it starts and sets a hop two masters' walks
  share once, keeping the register bus's placement; `address_map` holds
  the decoders' windows only. The block design no longer leaves the
  bridge's address space unassigned (`[BD 41-1630]`).
- **An implementation setting may be conditional**: `{"value", "when"}`
  is set only where `when` holds, for a property a tool refuses unless
  another enables it. `np2hw.toolchain.vivado.hex_word` writes a word the
  way Vivado's IP takes one.
- **`np2hw/hdl/np2hw_reset_sync.xdc`** -- the reset synchroniser's timing
  intent, scoped to the module and named in its description: its
  asynchronous clear and preset are not timing paths (assertion may land
  at any moment; release is what the chain resynchronises). Without it a
  link's reset into another domain's synchroniser failed recovery by
  whatever the two clocks' phase happened to be.
- **checkhwh knows np2hw's reset synchroniser** (its ancestry is both its
  sources') and audits np2hw's bus modules as interconnects.
- **`node.port.window`** -- the address window an AXI4-Lite slave port
  needs (two to its address width, at least a page), read from what the
  slave states or publishes; reading it builds the slave first. A leaf
  generator may publish `facts`.
- **A list in a design may hold expressions**, each evaluated like any
  other.
- **`run_cocotb(..., parameters=)`** sets the top module's parameters.
- **`np2hw.video_out.pixel_khz(mode)`** -- a raster's pixel clock in kHz,
  callable from a design's expressions.
- **`np2hw.hdl.stream:stream_cdc`** -- the stream crossing's library
  description (ports, sources, its scoped constraints), shipped with the
  module instead of in the content that places it.
- **`stream_cdc`** -- `np2hw/hdl/np2hw_stream_cdc.v`: a stream with its
  `sof`/`eol`/`last` across two unrelated clocks, on `cdc_fifo`, so a
  crossing sits between two stream ports with nothing to pack.
  `examples/stream_cdc.py` proves the flags stay with their samples.
- **`np2hw/hdl/np2hw_cdc_fifo.xdc`** -- the FIFO's timing intent as a file,
  scoped to the module: each gray-pointer path datapath-only, bounded by
  the sampling clock's period. `docs/clock-crossings.md` gave the
  constraint; now the module ships it.

### Fixed

- **The SystemRDL map escapes SystemRDL's keywords.** A register, design or
  instance named one -- a threshold called `level` -- was written bare, and
  the reference compiler refused the file. It is written as an escaped
  identifier (`\level`), which names the same component.

### Changed (resources)

- **`cdc_fifo`'s memory is block RAM at every depth.** A shallow FIFO
  used to infer LUT RAM, whose read is combinational from the write
  clock's stores -- a path from one domain into the other that a timer
  sees. A block RAM's read port is the reader's, so only the two pointers
  cross. A 16-deep crossing now costs one 18 Kb block RAM.
- **`frame_cdc` is two crossings**, the samples' and the sizes', and so
  costs one more 18 Kb block RAM than `stream_cdc`.

### Removed

- **`scanout`, `mode_timing` and `MODES`** (`np2hw.video_out`), and the
  scanout example. **Breaking** for anyone importing them. np2hw compiles
  NumPy to Verilog; a raster is a fixed design, not a NumPy model, and a
  table of display modes is a fact of the links that carry them. The
  raster is a Verilog file in the design that drives a display, and the
  rasters are `bayerlink.rasters`.

- **The bayerlink receiver moved to bayerlink-fpga.** `bayerlink_in` and
  `np2hw.video_in` are gone, with the `bayerlink` extra: the receiver is
  `bayerlink_fpga/hdl/bayerlink_in.v` and `bayerlink_fpga.receiver` now,
  beside the protocol it implements, proven there against the reference
  codec. np2hw no longer depends on bayerlink for anything.

## 0.6.1

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

### Documentation

- **`docs/clock-crossings.md`** — the intent a timing and CDC flow needs
  for `cdc_fifo`, stated by the module that owns it. Which two pointer
  paths cross and in which direction, why gray coding makes a *bounded*
  max-delay the right constraint rather than a waiver to infinity, the
  reset contract (both sides, or the pointers disagree forever), and the
  three things an ASIC flow needs that an FPGA one does not. Nothing
  about the crossing changed; it was undocumented, and a crossing a tool
  cannot be told about is one that fails somebody's signoff.

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
