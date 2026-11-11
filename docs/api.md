# Python API

For driving the compiler directly (the CLI is a thin wrapper over this). Public exports:

```python
from np2hw import Param, Params, Image2D, PhaseRef, Rom, to_ir, generate
from np2hw import coords, make_coords, saturate                    # in a model
from np2hw import compose, Instance, Connection, Port, StreamType   # composition
from np2hw import Netlist, Node, Endpoint                          # graph, pre-emission
from np2hw import Reg, RegBlock, RegInstance, AddrMap, systemrdl   # register maps
from np2hw import axil_regfile, control_wrap                       # control plane
from np2hw import skid, tee, cdc_fifo, frame_history               # plumbing
from np2hw.blocks import block, configblock, StreamPort, Binding     # blocks
from np2hw.blocks.params import Param, ParamSet, Context, StatsWindow
from np2hw.blocks.stream import StreamSpec
from np2hw.design import composite                                 # designs
from np2hw import pipeline                                         # pipelines
from np2hw.pipeline import compose, run
from np2hw.core.cemit import compile_c                             # a model as C
from np2hw.testing import (Beat, frame_to_beats, beats_to_words,   # verification
                           check_framing, reset_stream, run_frame,
                           AxiLiteMaster, axis_video_map)
from np2hw.verilog import (switchboard_wrap, axis_video_wrap,
                           umi_regfile, control_top)
```

## Declaring inputs & registers

```python
Image2D(name, width, height, bits=8, signed=False)     # the streamed image input
Param(name, dtype=None, *, bits=8, signed=False, shape=(), default=0)
Params([Param(...), ...])                               # named register set (namespace)
Rom(data, name="rom")                                   # a constant table
```
`Param` with `shape=` is a matrix of registers; indexing returns scalar `Param`s. See
[parameters.md](parameters.md).

## Trace: `to_ir`

```python
_, module = to_ir(fn, image, *params, channels=1)
```
- `fn` — the model function. `to_ir(fn, image, *param_objs)` (list form) or
  `to_ir(fn, image, params_namespace)` (single `Params`).
- `image` is the largest frame the core must hold (see
  [framing-and-resolution.md](framing-and-resolution.md)).
- Returns `(None, module)`: the model lowered by the compiler (`np2hw.core`).
  Pass the second element to `generate`. A model the compiler does not reach is
  refused here by name (`NotImplementedError`), one that cannot be built -- an
  index that can leave its table -- by its reason (`ValueError`).

```python
_, out = to_ir(model, Image2D("img", 1920, 1080, 8), Param("gain", np.uint8))
```

## Generate RTL: `generate`

```python
core = generate(module, module_name="np2hw_top", clk_ns=None, label=None,
                dev=None, sync_memory=False)
```

| arg | meaning |
|---|---|
| `clk_ns` | the clock budget: the datapath is cut into stages that meet it, or refused, naming the operation deeper than the clock |
| `label` | the name a refusal uses (the module name when None) |
| `dev` | the device family the core is timed and built for (`np2hw.timing.Device`; 7-series by default) |
| `sync_memory` | the promise a core bound for silicon makes: every memory reads through a register ([memories.md](memories.md)) |

Every core takes each frame's size in its stream's context, `in_ctx`, and hands
its own on in `out_ctx`
([framing-and-resolution.md](framing-and-resolution.md)).

Returns a `Core`. It is a `dict` subclass, so the detail keys np2hw's own
wrappers read are still there, but the **public surface is the typed one**:

| attribute | meaning |
|---|---|
| `core.verilog` | the module text |
| `core.module` | its name |
| `core.interface` | its ports — see below |
| `core.line_buffers` / `core.shift_depth` | delay-line cost (`M` / `N` are the emitter's private spelling) |
| `core.out_bits`, `core.params` | output width; `(name, bits)` per register port, the frame's size among them |
| `core.registers` | the model's own registers: `core.params` less the frame's size |
| `core["result"]` | a fold's: the result's `shape`, and the `rows` and `cols` of the frame it streams out; None for any other core |

### `core.interface` — a module describing itself

Whoever writes the ports owns the port list, so a composer never has to know the
conventions:

```python
{"clock": "clk", "reset": "rst", "param_prefix": "param_",
 "input":  {"prefix": "in",  "data_bits": 12, "flags": ("sof",)},
 "output": {"prefix": "out", "data_bits": 12, "flags": ("sof", "eol", "last")},
 "params": [("gain", 8, False)]}
```

`input.flags` is what the module ACCEPTS (a core self-frames from its
frame's size and only needs `sof` to re-anchor); `output.flags` is what it
regenerates. `output` is `None` for a SINK — a module that consumes a stream and
produces none, and therefore never stalls its source.

The fixed modules -- `skid`, `cdc_fifo`, `stream_cdc` (a stream with its
flags across clocks, on `cdc_fifo`), `frame_cdc` (the same, with each
frame's context, its `ctx` member), `tee` -- are Verilog files, `np2hw/hdl/np2hw_skid.v`,
`np2hw_cdc_fifo.v`, `np2hw_stream_cdc.v`, `np2hw_frame_cdc.v`,
`np2hw_tee.v`, read and reviewed as files: their structure never varies, only a width and a depth,
which are module parameters set where they are instantiated. The
Python functions return the file with the values a use sets (`"parameters"`)
and its interface; a result that instantiates one lists it in `"requires"`,
and `np2hw.hdl.closure(result)` is the text with everything it requires.
Generated Verilog is for what genuinely varies: a traced model, a
composition.

They, the generators -- `ddc_slave`, `framebuffer_read`,
`frame_history`, `axil_regfile` -- and `compose()`, `control_wrap()`
publish `interface["ports"]`: EVERY port, video pins and status words and
bus ports as much as streams, as `{"name", "direction", "width"}`, plus
`"bus": {"vlnv", "bundle", "logical"}` for a port that is part of a bus. A width
the module takes from a parameter is published as its expression (`"W"`), with
the defaults under `interface["parameters"]`. The header is written from the
same list (`np2hw.ports`: `Pin`, `header`, `published`), so the two cannot
differ; `np2hw.ports.elaborated_ports` asks Verilator what a module really
elaborates to, which is how `examples/port_lists.py` holds every list to its
module.

## Blocks (`np2hw.blocks`)

A block is a NumPy model plus what its arithmetic cannot say: its registers
and scalar inputs (`np2hw.blocks.params`: `Param`, `ParamSet`,
`StatsWindow`, LOCAL offsets only, a composer assigning each instance its
base) and the names of its stream ports (`StreamPort`; widths are traced,
never declared). It knows nothing of the pipeline it is placed in: what
feeds an input is the design's -- its register, or the pipeline context the
design binds it to.

```python
@block(version=(1, 0), description="...", inputs=(StreamPort("in"),),
       outputs=(StreamPort("out"),),
       params=[Param("gain", bits=16, frac=8, description="..."),
               Param("phase", bits=2, description="...")],
       even_frame=True)                         # its math tiles 2x2
def gain(pixel, p, bit_depth):
    row, col = p.phase >> 1, p.phase & 1        # slice starts, at run time
    ...
```

The decorated function IS the block: `Block.run(frame, values, bit_depth,
**context)` runs it validated from the declarations -- each context signal
a binding reads in its declared range, a frame with an odd dimension
refused where the block says `even_frame` -- and `Block.generate(spec,
width, height, module_name)` traces it into Verilog. `Block.compile_c(spec,
width, height, build)` prints the same trace as C (`np2hw.core.cemit`), and
`Block.run(..., kernel=k)` runs it in the model's place, with the same checks. `Block.configure()`
gives the cached variant with register attributes overridden;
`Block.bind({input: expression}, vocabulary)` the variant with scalar
inputs bound, against the design's pipeline context -- each leaves the
registers, its value the expression over the pipeline context (`"(tick + 1)
& 3"`) or an integer, evaluated by the design's one walker over integers in
a run and over traced values in a trace, so it is folded into the core and
the core's ports are the context it reads. A value range the input cannot
hold is refused. A stream input declared `StreamPort("in", history=True)`
hands the model the stream's recent frames, `(T, H, W)` newest last, traced
at `HISTORY_TRACE` frames; the core keeps as many as the model reads back,
and a run takes the context one value a frame. `registry(package)` and
`resolve(package, name)` discover the blocks declared under a package.

A design names its pipeline context (`np2hw.blocks.params.Vocabulary`): a
composite's `"vocabulary": "package.module:NAME"`, or `assemble`'s
`vocabulary=`; every composite of one design that names one names the same.
np2hw names no context signal. Verilog traced from a model carries the copyright and
`SPDX-License-Identifier` its source file states (`model_header`): a design
derived from a model is under the model's terms.

## Designs (`np2hw.design`)

A design is JSON: a library file `{"blocks": {name: {...}}}`
(`np2hw/design/blocks.schema.json`) of leaves and composites, a composite
holding nodes -- each a block by reference, `package.module:block` from the
top of the import path -- their connections, and settings by path (`build`
for what shapes the hardware, `values` for what it holds, `bind` for what
feeds a block's scalar input in place of a register -- over the pipeline
context and the build parameters in scope; outermost wins; a string is
always an expression). `composite.load(ref, build=None)`
flattens one into leaves at their full paths; its `.netlist`
(`np2hw.design.netlist`) is the design as every tool reads it -- each leaf
with its implementation, settings and resolved ports, every bundle and net
-- and `np2hw.design.wiring` checks it at load. A block may be described
apart from its implementations, chosen by its `implementation` parameter.
`np2hw.design.profile` holds a set of register values for a block.

## Pipelines (`np2hw.pipeline`)

`pipeline.load(ref)` loads a composite of model blocks as a pipeline;
`np2hw.pipeline.compose` composes it: `stages` (every instance with its
base address, the context block first at base 0), `register_map` (the JSON
map the host reads), `write_systemrdl`, `generate` (the Verilog, with the
AXI4-Lite register file with `control=True`). A pipeline's blocks bring
their content's `Vocabulary`, which names the context block and its commit
register; blocks of two vocabularies are refused. The map's `licence` and
the generated files' headers are what the models' sources state.
`np2hw.pipeline.run` runs a chain -- the models, the models printed as C
(`run_model(..., c_build=dir)`), and the RTL twin under Verilator beside
them. `np2hw.pipeline.pack.emit(design, out, vlnv=None,
control=True, clock_mhz=None, verify=True, sync_memory=False)` writes the
design pack once the twin agrees with the models on every input's chain;
`np2hw pipeline run|generate` (docs/cli.md) and the FuseSoC generator's
`design` parameter call it.

## Toolchains (`np2hw.toolchain`)

A toolchain renders a loaded design for the tool that builds it and decides
nothing about it. `np2hw.toolchain.vivado.emit(design, out, name=None)`
writes the block design (`bd.tcl`, named `name` or the design's name) and
the implementation scripts; `np2hw.toolchain.vivado.bdtrace` traces a built
block design. Two checks hold Vivado to the design:
`python -m np2hw.toolchain.vivado.vendor_check OUT` compares every described
vendor cell's pins, dumped once the block design validates, with its
description, and `python -m np2hw.toolchain.vivado.checkhwh HWH` audits the
hardware handoff as built.

## Compose: `compose`

Instantiate generated cores and wire them as a netlist. See
[`examples/hierarchy.py`](../examples/hierarchy.py).

```python
top = compose(module_name, instances, connections, ports, header=(), notes=())
```

| type | meaning |
|---|---|
| `Instance(name, core, bind, domains, comment)` | one module. `bind` maps its parameter ports to expressions; `domains` tags its streams |
| `Connection(source, sink, comment)` | an edge. An endpoint is `instance.port` or a top-level port name |
| `Port(name, direction, width, signed, comment, stream)` | a top-level port. `stream=StreamType(...)` makes it a pixel stream |
| `StreamType(data_bits, flags, domain)` | what a net carries. `domain` is an **opaque tag** np2hw compares and never interprets |

`compose()` returns an `interface` of its own, so **a composed module is
instantiable inside another compose()** — build a subsystem once and instantiate
it per sensor. Single-stream subsystems nest; a multi-stream one reports
`nestable: false`.

Checked before a line is emitted: every net's width **and domain** must match
(a 12-bit Bayer stream and a 12-bit luma stream are identical to a compiler and
nonsense to connect), and an output may feed one consumer that applies
backpressure plus any number of sinks. Two blocking consumers need a buffering
fork; that case is refused (Next: buffered forks).

Returns a **meta dict** consumed by the adapters and testbenches:

```python
meta["verilog"]         # the module source
meta["module"]          # module name
meta["in_bits"], meta["out_bits"], meta["signed"]
meta["params"]          # [(name, bits)] — config register ports
meta["param_defaults"]  # {name: reset_value}
meta["out_rows"], meta["out_cols"]   # the output at the traced size
meta["memories"]        # every memory, behind its seam
```

## Adapters (from `np2hw.verilog`)

All take the `meta`/`core` dict and return a dict with `"verilog"` + `"module"`:

```python
axis_video_wrap(meta, W, H, module_name=None)                  # AXI4-Stream Video
switchboard_wrap(meta, W, H, module_name=None, dest=0,
                 pack=False, native=False)                     # Switchboard SB
axil_regfile(params, module_name=..., addr_bits=8,
             frame_sync=False, defaults=None)                  # AXI-Lite registers
umi_regfile(params, ..., write_op=1, read_op=2,
            frame_sync=False, defaults=None)                   # UMI registers
control_top(core, module_name=None, ctrl="axil",
            frame_sync=True, addr_bits=8)                      # regfile + core wired
control_wrap(core, registers, bind, module_name=None,
             addr_bits=16, frame_sync=True, commit=None)       # regfile + ANY module
```
See [interfaces.md](interfaces.md) for each. `params`/`defaults` come from
`meta["params"]` / `meta["param_defaults"]`.

`params` for the register files is either `(name, bits)` tuples — packed at word
offset `i*4` — or `Reg` values, which carry their own address:

```python
Reg(name, bits, offset=None, signed=False, reset=0, access="rw", value=0,
    description="")
```
Whoever allocates the address owns it. An application with its own map (aligned
regions, an identity word at each base) passes `Reg`s and keeps that decision;
np2hw keeps the decode. `access="ro"` emits a constant with no storage and no
write path. See `examples/regfile_map.py`.

The hierarchical form keeps what flattening loses — block TYPES and INSTANCES:

```python
RegBlock(name, regs, size=None, description="", properties={})   # a layout, once
RegInstance(path, block, base)                                    # that layout, placed
AddrMap(name, instances, data_bits=32)                            # the whole design
systemrdl(addrmap, header=()) -> str                              # SystemRDL 2.0
```

`axil_regfile`/`umi_regfile`/`control_wrap` accept an `AddrMap` directly and
flatten internally — the flat list exists only where an emitter needs it.
`systemrdl()` renders one `regfile` per type, instantiated per instance, with
caller `properties` (a Q format, say) declared as user-defined properties; the
PeakRDL exporters then produce UVM/IP-XACT/C headers from it — tools the USER
runs, not dependencies. See `examples/systemrdl_map.py`.

## The netlist, before emission (`np2hw.netlist`)

```python
net = Netlist("design", external_inputs=("in",), external_outputs=("out",))
net.add(Node("a"))                       # inputs=("in",), outputs=("out",)
net.add(Node("stats", outputs=()))       # a SINK: observes, never stalls
net.connect("in", "a.in"); net.connect("a.out", "out")
net.validate()                           # undriven / double-driver / cycle / fork
net.order()                              # deterministic topological order
```

The four validations are laws of the streaming handshake, enforced by its
owner: an output may feed one blocking consumer plus any number of sinks, and
a fork to two blocking consumers is refused rather than emitted as a deadlock.
See `examples/netlist_rules.py`.

## A stream's recent past (`np2hw.frame_history`)

```python
frame_history(frames, bits, width, height, loop=None,
              module_name="frame_history") -> dict
```
Each pixel delivered with its `frames` predecessors at the same position,
as channels of one word, with a tag per frame -- for a model that computes
across frames, a time-of-flight phase set say. The store sits behind the
`<module_name>_mem` seam, under the same contract as the line buffers
([memories.md](memories.md)), and is reported in the core's `memories`.

## Verification models (`np2hw.testing`)

The Python model of np2hw's own contracts, so applications stop re-deriving
them: `Beat`/`frame_to_beats`/`beats_to_words`/`check_framing` (pure Python,
no simulator), and the cocotb BFMs — `reset_stream`, `run_frame` (randomised
backpressure on both sides), and `AxiLiteMaster`, whose handshake timing
matches the register file np2hw emits. `axis_video_map` states the AXI4-Stream
Video signal mapping, owned here because `axis_video_wrap` writes it.

## A register bus (`np2hw.bus`, `np2hw.hdl.bus`)

Fixed modules a design places by reference: `np2hw.hdl.bus:axi3_to_axil`
(a processor's AXI3 port onto AXI4-Lite), `:axil_arbiter` (two masters onto
one slave) and `:reset_sync` (a reset per clock domain). The decoder is
`np2hw.bus:axil_decoder`, built with each slave's window size -- usually
the slave's own fact, `node.port.window` -- and the region the processor
reaches (`base`, `span`); `np2hw.bus.allocate` places the windows, and the
decoder publishes where they landed as its fact `windows` (`{port: [base,
size]}`; a size of 0 makes no port). The Vivado emitter sets each window as
the slave's segment, and `np2hw.bus.address_map(design)` gives the host the
same windows by slave.

## Simulations (`np2hw.testing`)

`run_cocotb(workdir, verilog, toplevel, test_module, case, test_dir=None)`
builds a design under Verilator and runs a cocotb testbench over it; the
testbench reads `case` with `load_case()`. `np2hw.testing.block` is the
testbench for any block: `case = {"block": "package.module:name", "width",
"height", "bit_depth", "registers" (overrides), "in_channels", "trials":
[{"seed", "frame", "values", "context"}]}`, each trial compared word for
word and flag for flag with the block's model under randomised backpressure.

## Minimal end-to-end

```python
import numpy as np
from np2hw import to_ir, generate, Image2D, Param
from np2hw.verilog import axis_video_wrap

def model(img, gain):
    return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)

_, out = to_ir(model, Image2D("img", 1920, 1080, 8), Param("gain", np.uint8))
core = generate(out, "gain_core")
wrap = axis_video_wrap(core, 1920, 1080, module_name="gain_axis")
open("gain.v", "w").write(core["verilog"] + "\n" + wrap["verilog"])
```

## Testbench generators

For self-checking RTL sims (used by the examples and the CLI sim backends):
`testbench`, `testbench_handshake(iface="core"|"sb"|"axis")`, `testbench_ctrl`,
`testbench_sb_packed` — all in `np2hw.verilog`. See `examples/_harness.py` for usage.
