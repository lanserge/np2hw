# Changelog

Notable changes to np2hw. The claim this project makes is that generated
Verilog is bit-exactly equal to the NumPy model it came from; anything
here that changes an *emitted width* is called out, because a width that
moves silently is how a downstream integrator reads the wrong bits.

Pre-1.0, so the API can still move. Anything that moves is named here.

## Unreleased

### Added

- **The tee's branch A may hold its stream** (`HOLD_A`, `tee(...,
  hold_a=True)`): for a stream that can wait -- a frame store's, paced by
  its consumer, never a sensor's -- A joins every frame it is enabled for
  at its start and holds the stream while it is not ready, so it is never
  torn, and B, still a disposable tap, sees only the beats taken. Off by
  default: the tee still never stalls.

- **`np.where` on a condition the trace already knows picks its operand,**
  as NumPy does: a bound constant or a build parameter choosing between two
  values is the one chosen, in the core as in the run -- wires, where a
  register's would be a select.
- **The frame store keeps its frames in external memory**
  (`implementation: "axi"`): an AXI4 master, its port `mem`, slots from
  `BASE` a frame apart rounded up to 4K. A frame is whole once its writes
  are answered, bursts never cross a frame's end or a page, and each frame
  read is fetched ahead into a queue of its own only as far as the queue
  has room, so a read is never paused. Bursts are sixteen beats at most, as
  a Zynq's AXI3 ports take them. It is held to the store's one model, under
  a memory that answers late, lands a write only when it answers it, and
  holds the master to the protocol. Its master has a reset of its own,
  `mem_rst`, the memory's, which the port states as its reset: the store's
  `rst` is the stream's, and when the stream starts over mid-burst every
  address out is still taken, every burst promised sent whole, every
  answer owed taken, and the ring goes on from the slot it had reached, so
  a frame answered whole after the reset counts in no history. Every
  request is made from registers -- each frame read's next address and
  the words it still has to ask for, the slots' bases worked out once a
  frame, the frame's samples counted as they come rather than its size
  multiplied -- so the store meets a 7 ns clock on a Zynq-7020 out of
  context, where it missed by 4.7 ns. Every beat comes in through a skid,
  so the store's logic starts at a flop whatever drives it -- a board's
  entry decides its framing combinationally from its counters -- and the
  block-RAM binding's latency fact is 5. Its `overrun` has one driver; it
  had two, which a design that wired the flag out would have refused.
- **A design holding modelled leaves is proved from its models**
  (`np2hw.pipeline.proof`). The stream is followed from the design's input
  to its output: each composite of model blocks built whole runs its
  chain, each leaf its model; each frame's context rides its stream, and
  what else rides beside the frames is followed net by net. One stimulus drives the models and the
  design's own Verilog under Verilator: the frames, the input's gaps, the
  consumer's waits and stalls, every register at its reset value. A leaf
  with no model, a context the stream does not bring, and a value read
  around a stage that chose which frames go on, are refused by name. Every leaf model has one signature: frames
  and their ports in, frames and their ports out (`LeafRun`); **the frame
  store's model moved to it**.
- **A loaded design keeps each pipeline it built whole**
  (`Design.pipelines`), as its build made it. `np2hw pipeline generate
  DESIGN --node PATH` proves and packs the one at `PATH` -- the part of a
  design holding a frame store, say, made of models -- as the design sets
  it, and takes the design's own parameters as `NAME=VALUE`; `pack.emit`
  takes such a loaded design as well as a reference.
- **A stream may bring its history.** An input port's `frames` says the
  stream entering carries that many frames side by side -- a frame store
  upstream keeps them -- and a block reading the stream's history then
  reads them as fields of its input word and keeps no frames of its own.
  A read further back than the stream brings is refused, and so is a
  block of one frame fed several. A design reads it as `node.port.frames`,
  which is what it sets its frame store's depth from. `StreamSpec` carries
  `frames`; **the input word of such a block is `frames` samples wide**.
- **A frame store: a stream's recent frames, for a consumer at its own
  pace** (`np2hw.hdl.frames:frame_store`). Every frame that comes in is
  kept, in a ring of `SLOTS`, and the consumer is handed the newest
  `FRAMES` whole ones side by side, as often as it asks, with the size and
  context of the newest; the input never waits for the output. A frame cut
  short, or a frame of a new size, starts the history over, and a source
  that reaches a frame being read is told, sticky (`overrun`). The frames
  live behind a sequential seam; `bram` keeps them in block RAM.
- **A leaf brings its NumPy model** (`"model": "module:function"` in its
  library description), called with the frames, the stimulus that drives
  them and the node's parameters by the description's names. A timed
  leaf's implementation states the one fact of its hardware the model
  needs, its `latency`. The frame store's model computes, from frames and
  timing alone, every frame the Verilog must put out; random stimuli hold
  the two to it frame for frame.
- **A leaf's implementations are checked against its ports**, each module
  elaborated as the block's ports say.
- **A model reads a value's bits as another integer, and shifts left, as
  NumPy does.** `x.view(np.int16)` on a 16-bit unsigned value is the same
  bits read signed -- the container's wrap, which np2hw already models --
  and is refused for a dtype of another width; `x << k` is a multiply by
  `2**k` in the value's own container, so the bits shifted past it are
  gone, exactly as in NumPy. Both are wires in the hardware. A sensor's
  signed sample in a wider word is then the model's own statement --
  `(pixel << 5).view(np.int16) >> 1` for eleven bits of two's complement in
  twelve -- not a conversion upstream.
- **Scalar inputs bound to expressions.** A design binds a block's scalar
  input under a node's `bind` (`{"quad.pos": "frame_seq & 3"}`), and
  `Block.bind()` does the same in Python. The input leaves the block's
  registers; its value is the expression, over the pipeline context the
  block's vocabulary declares, integers, and the build parameters in scope
  (integers at load). The design's one expression walker evaluates it --
  over integers in a NumPy run, over traced values when the model is
  traced -- so the expression is folded into the block's core, which reads
  the context it needs from its stream's `ctx`, with each frame. A range the
  input cannot hold is refused, with the range. **Register maps change**
  where a design binds an input: the bound register is gone.
- **A design asks whether a stream carries a fact.** `in` and `not in`
  in the expression language, and an input port's `context` among the
  facts a design reads (`isp.isp_in.context`), so a connection exists only
  where its port does: `"when": "'frame_seq' in isp.isp_in.context"`.
- **A pack is proved over a run of frames where its chain reads a
  history**, each frame's stream context its own (`synthetic_context`'s
  `frames`), and every pack's twin at the clock it is cut for, the stages
  the hardware has.
- **A block may read the stream's history.** `StreamPort("in",
  history=True)` hands the model the recent frames, `(T, H, W)` newest
  last, and it reads past ones by slicing the first axis. How many frames
  are kept is declared nowhere: the core's store is how far back the model
  reads (traced at `HISTORY_TRACE`, 64 frames, the deepest read 63). A run
  takes a bound input's context one value a frame, and each output frame is
  the model's at the context of the frame it ends at.
- **A chain runs a run of frames.** Where a block reads the history,
  `run_model` and `run_rtl` take `(t, h, w[, c])` and context one value a
  frame; the twin sends the frames back to back, each frame's context set
  as it starts, and returns every frame the hardware put out.
  `run_rtl(clk_ns=)` builds the twin cut into the stages a design at that
  clock has. `np2hw.testing.block` takes a history block's trials as one
  run.
- **A slice may start at a value computed from registers** -- `p.phase >>
  1`, `1 - (p.phase & 1)` -- whose exact range fits the stride: a CFA block
  takes its phase as the one input it is.

### Removed

- **`ContextBit`, and `consumes=`, `context=` and `vocabulary=` on
  `block()`.** A block knows nothing of the pipeline it is placed in: it
  takes what it reads as inputs, and a design binds them (above). What a
  block consumes is what its bindings read. The pipeline context is the
  DESIGN's: a composite names it (`"vocabulary": "package.module:NAME"`,
  every composite of one design the same one), `assemble()` takes
  `vocabulary=`, and `Block.bind()` takes it from the design. A design that
  names none has no pipeline. `even_frame` moves from `Context` to
  `block()`: the block's own statement that its math tiles 2x2.
- **A model's `ctx` argument.** A model is `(pixel, p, bit_depth)`: its
  stream, its inputs and registers, and the stream's depth. **Every model
  signature changes.**

### Changed

- **One compiler.** `np2hw.core` builds every model; the earlier compiler,
  its tracer (`np2hw.frontend`, `np2hw.trace`) and its emitters are gone,
  with the `NP2HW_CORE` switch. `to_ir(fn, image, *params, channels=1)`
  returns `(None, module)` or refuses, by name (`NotImplementedError`) or by
  reason (`ValueError`); `generate(module, module_name, clk_ns, label, dev,
  sync_memory)` builds it, or refuses -- it no longer hands a model to
  another compiler when its own build misses the clock.
- **Every core takes each frame's size at run time** -- one kind of core
  -- **in its stream's context.** A stream carries each frame's context in
  a new optional member, `ctx` (`CTX` in `np2hw.stream.STREAM_SIGNALS`):
  the facts the stream declares, packed in its vocabulary's order, the
  first in the low bits, the frame's `width` and `height` first where it
  declares them -- `context_bits`, `pack_context` and `unpack_context` read
  and write it. A core latches `in_ctx` with a frame's first beat and hands
  `out_ctx` on with the frame: the same fields, the size the frame it puts
  out has, every other field as the frame brought it, each in a register.
  An image's `context` is its layout, by default the size alone. The size a
  model is traced at is the most the core holds (a line buffer's depth, a
  frame store's), and a generated module has no `WIDTH`/`HEIGHT` parameters
  left to override. A frame is at least 2x2. **Ports change on every
  core**: `in_ctx` and `out_ctx` are new, and a value a frame brings
  (`Param(frame=True)`, a bound input's context) is a field of `in_ctx`,
  not a port. `core["registers"]` lists the model's own registers,
  `core["params"]` every register port.
- **A frame's context travels with its stream.** A composed module wires
  each stream's `ctx` with its other members, and a sink taking a context
  is refused one of another layout. A pipeline's input stream carries the
  facts its port declares; the context block's registers supply the rest,
  joined into the one layout every core of the pipeline carries
  (`compose.context_layout`, `compose.carried_layout`), and a fact the
  stream carries has no register. A design reads a stream's layout off a
  node's port, an output's as well as an input's -- `context`,
  `context_bits` and `context_at`, each fact of the vocabulary's low bit
  in it or -1 -- so a module packing or unpacking it is told where each
  fact goes. The RTL twin reads each output's size from its `out_ctx`. A wrapper of one core (`control_top`, the
  Switchboard and AXI4-Stream wrappers, the testbenches) offers each
  context field as a register port, the size among them, and packs them
  into `in_ctx` (`verilog.offered_params`, `verilog.context_word`).
- **Fixed modules carry the context they do not read.** `frame_store`
  and `frame_cdc` carry their stream's `ctx` whole (`CTX_BITS`, which a
  design reads as `node.port.context_bits`), and the frame store's size
  is the one each frame's own beats draw: its width to the first `eol`,
  its rows to its `last`. **Their ports change**: the separate size ports
  are gone.
- **A window takes frames of any height**: its row steps count at the
  16-bit geometry width (its column steps stay sized from the traced width,
  which its line buffers hold). **Emitted widths change** for those step
  registers.
- **A video takes frames smaller than it was traced at**: its store's
  address wraps at the frame's last pixel.
- **An index that can leave a table is a model error** (`ValueError`,
  "falls outside the table"), for register arrays and constant tables alike.

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

- **Windows on the compiler**: line buffers behind the `<module>_linebuf`
  seam, read a column ahead; replicate and zero edges; crops by slice or
  `make_coords`, the rows and columns past them taken and never read; a
  filter of a filter's result, as one window over the source.
- **A window that registers place**:
  `img[p.y0:, p.x0:][:p.h, :p.w]`, NumPy's own slice with register bounds.
  The source streams through and only the beats inside the window go on,
  its lines and frame ending at its edges or the frame's, whichever comes
  first -- as the slice has it -- and a packed word passes whole.
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

- **`cdc_fifo` reads through a register after the RAM.** Its one output
  register was loaded from the RAM, and a block RAM's read is clocked, so
  that register was the RAM's own read latch: the RAM's slow clock-to-out
  on the port and at the head of whatever the reader built on it -- on the
  PYNQ-Z2 the board's worst path, the ISP's first stage. The read latch
  now feeds a register that is the port (a tool may fold it into the RAM's
  output register). **One more read clock of latency.**
- **`cdc_fifo`'s memory is block RAM at every depth.** A shallow FIFO
  used to infer LUT RAM, whose read is combinational from the write
  clock's stores -- a path from one domain into the other that a timer
  sees. A block RAM's read port is the reader's, so only the two pointers
  cross. A 16-deep crossing now costs one 18 Kb block RAM.
- **`frame_cdc` is two crossings**, the samples' and the sizes', and so
  costs one more 18 Kb block RAM than `stream_cdc`.

### Removed

- `generate(framing=, max_width=)`: a run-time size is every core's; the
  end-of-frame framing and `MAX_WIDTH`/`active_width` cores, and the
  `eof_framing`, `dynamic_res` and `active_width` examples, go with them.
- `to_ir(out_bits=)`, `Const`, `Index` and `Indexer` (the earlier
  compiler's IR), and the `ir_blur`, `verilog_blur`, `index_algebra` and
  `dtype_refusals` examples.

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
