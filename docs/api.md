# Python API

For driving the compiler directly (the CLI is a thin wrapper over this). Public exports:

```python
from np2hw import Param, Params, Const, Image2D, Indexer, to_ir, generate
from np2hw.verilog import (switchboard_wrap, axis_video_wrap,
                           axil_regfile, umi_regfile, control_top)
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
out_line = to_ir(fn, image, *params, out_bits=None)     # returns (stream, out_line)
```
- `fn` — the model function. `to_ir(fn, image, *param_objs)` (list form) or
  `to_ir(fn, image, params_namespace)` (single `Params`).
- Returns `(None, Mux)` for an `np.where` model, else `(stream, out_line)`. Pass the
  second element to `generate`.

```python
_, out = to_ir(model, Image2D("img", 1920, 1080, 8), Param("gain", np.uint8))
```

## Generate RTL: `generate`

```python
meta = generate(out_line, module_name="np2hw_top",
                framing="height", max_width=None)
```
| arg | meaning |
|---|---|
| `framing="height"` | default: self-frame by counting to HEIGHT |
| `framing="eof"` | height-agnostic: `in_eof`/VSYNC drives the flush ([framing-and-resolution.md](framing-and-resolution.md)); edge core only |
| `max_width=N` | runtime-variable width: `MAX_WIDTH` param + `active_width` register |

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
```
See [interfaces.md](interfaces.md) for each. `params`/`defaults` come from
`meta["params"]` / `meta["param_defaults"]`.

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
