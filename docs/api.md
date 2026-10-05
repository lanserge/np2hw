# Python API

For driving the compiler directly (the CLI is a thin wrapper over this). Public exports:

```python
from np2hw import Param, Params, Const, Image2D, Indexer, PhaseRef, to_ir, generate
from np2hw import compose, Instance, Connection, Port, StreamType   # composition
from np2hw import Netlist, Node, Endpoint                          # graph, pre-emission
from np2hw import Reg, RegBlock, RegInstance, AddrMap, systemrdl   # register maps
from np2hw import axil_regfile, control_wrap                       # control plane
from np2hw import skid, tee, cdc_fifo, frame_history               # plumbing
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
Const(value)                                            # compile-time constant
```
`Param` with `shape=` is a matrix of registers; indexing returns scalar `Param`s. See
[parameters.md](parameters.md).

## Trace: `to_ir`

```python
_, out = to_ir(fn, image, *params, out_bits=None, channels=1)
```
- `fn` — the model function. `to_ir(fn, image, *param_objs)` (list form) or
  `to_ir(fn, image, params_namespace)` (single `Params`).
- Returns `(None, module)` when the new compiler (`np2hw.core`) takes the model
  -- the pointwise family today -- `(None, Mux | PhaseCanvas | ChannelStack)` for
  those shapes, else `(stream, out_line)`. Pass the second element to `generate`.

```python
_, out = to_ir(model, Image2D("img", 1920, 1080, 8), Param("gain", np.uint8))
```

## Generate RTL: `generate`

```python
core = generate(out_line, module_name="np2hw_top", framing="height",
                max_width=None, clk_ns=None, label=None, dev=None,
                sync_memory=False)
```

| arg | meaning |
|---|---|
| `framing="height"` | default: self-frame by counting to HEIGHT |
| `framing="eof"` | height-agnostic: `in_eof`/VSYNC drives the flush ([framing-and-resolution.md](framing-and-resolution.md)); edge core only |
| `max_width=N` | runtime-variable width: `MAX_WIDTH` param + `active_width` register |

Returns a `Core`. It is a `dict` subclass, so the detail keys np2hw's own
wrappers read are still there, but the **public surface is the typed one**:

| attribute | meaning |
|---|---|
| `core.verilog` | the module text |
| `core.module` | its name |
| `core.interface` | its ports — see below |
| `core.line_buffers` / `core.shift_depth` | delay-line cost (`M` / `N` are the emitter's private spelling) |
| `core.out_bits`, `core.params` | output width; `(name, bits)` per parameter port |

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
WIDTH/HEIGHT and only needs `sof` to re-anchor); `output.flags` is what it
regenerates. `output` is `None` for a SINK — a module that consumes a stream and
produces none, and therefore never stalls its source.

The fixed modules -- `skid`, `cdc_fifo`, `stream_cdc` (a stream with its
flags across clocks, on `cdc_fifo`), `tee` -- are Verilog files,
`np2hw/hdl/np2hw_skid.v`, `np2hw_cdc_fifo.v`, `np2hw_stream_cdc.v`,
`np2hw_tee.v`, read and reviewed as files: their structure never varies, only a width and a depth,
which are module parameters set where they are instantiated. The
Python functions return the file with the values a use sets (`"parameters"`)
and its interface; a result that instantiates one lists it in `"requires"`,
and `np2hw.hdl.closure(result)` is the text with everything it requires.
Generated Verilog is for what genuinely varies: a traced model, a
composition.

They, the generators -- `scanout`, `ddc_slave`, `framebuffer_read`,
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
meta["out_rows"], meta["out_cols"]
# edge cores also: meta["edge"], meta["eof"], meta["dynamic"], meta["max_width"], ...
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
