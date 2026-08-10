# np2hw

[![CI](https://github.com/lanserge/np2hw/actions/workflows/ci.yml/badge.svg)](https://github.com/lanserge/np2hw/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/np2hw)](https://pypi.org/project/np2hw/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**NumPy → streaming hardware (Verilog) compiler for image and 2-D pipelines.**

Write streaming image-processing code as ordinary NumPy. `np2hw` *traces* it (like JAX/TVM)
into a streaming line-based IR and emits synthesizable Verilog — with line buffers,
shift registers, edge handling, config registers, and ZeroAsic-stack interfaces
generated for you. **The same NumPy function is the spec, the hardware, and the
validation reference.**

```python
# my_isp.py  — a complete model file
import numpy as np
from np2hw import Param, Params

PARAMS = Params([Param("gain", np.uint8, default=16)])   # a config register

def model(img, p):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")    # 3x3, replicate edges
    blur = (x[:-2,:-2] + 2*x[:-2,1:-1] + x[:-2,2:]
            + 2*x[1:-1,:-2] + 4*x[1:-1,1:-1] + 2*x[1:-1,2:]
            + x[2:,:-2] + 2*x[2:,1:-1] + x[2:,2:]) // 16
    return ((p.gain * blur) // 16).clip(0, 255).astype(np.uint8)
```

Save that as `my_isp.py` and run it — or use a shipped model from
[`examples/isp/`](examples/isp):

```bash
np2hw run  examples/isp/gain.py in.png out.png --param gain=24            # NumPy reference
np2hw run  examples/isp/gain.py in.png out.png --param gain=24 --backend sim --sim cxxrtl   # the generated RTL
np2hw view examples/isp/gain.py --source camera                          # live, with a gain slider
```

## Why

For image processing and DSP, **NumPy is the natural language** — but mainstream HLS
targets C/C++ (Bambu, Vitis HLS, Catapult), and the Python accelerator DSLs
(HeteroCL, Allo, PyLog) use their own APIs, not plain NumPy, and target CPU/GPU/ML
rather than streaming ISP RTL. `np2hw` traces **standard, unmodified NumPy** into
**streaming image-processing RTL**, leaning on a hardware **pattern library** drawn
from production ISP experience. See [`design/07-prior-art.md`](design/07-prior-art.md)
for the competitive map.

The engine traces general streaming NumPy — stencils, pointwise, edge handling,
mux (`np.where`), and **phase-sliced planes** (`out[py::2, px::2] = ...` with a
register-valued phase, which is how a per-CFA-colour operation is written).
Generated cores can be wired together with `compose()`, a composed module nests
inside another, and `control_wrap()` puts an AXI-Lite register file in front of
either. The hardest exerciser is a full ISP
built on np2hw end to end — models, netlist, control plane and verification.
(Next on the tracing roadmap: reductions, `@`/matmul, convolve — the
authoritative list of what traces today is
[docs/supported-ops.md](docs/supported-ops.md).)

## Install

```bash
uv venv
uv pip install -e ".[media]"               # core + image/video IO (np2hw run/view)
uv pip install -e ".[media,camera]"        # + webcam for `view --source camera` (opencv-python)
uv pip install -e ".[media,switchboard]"   # + run through the ZeroAsic Switchboard stack
```

`uv` installs the Python package and its extras. **RTL simulation uses external
tools installed separately** — Icarus (`iverilog`), Yosys (`yosys`), and Verilator
(`verilator`), e.g. from Homebrew, your package manager, or the OSS CAD Suite. A
non-`uv` setup (`PYTHONPATH=src python3 …`) also works — see
[`docs/install.md`](docs/install.md).

## Documentation

| Doc | Covers |
|---|---|
| [docs/install.md](docs/install.md) | Install, uv, optional extras, system tools |
| [docs/cli.md](docs/cli.md) | `np2hw run` / `np2hw view` — full reference |
| [docs/writing-models.md](docs/writing-models.md) | The model-file convention; which NumPy is traceable; gotchas |
| [docs/supported-ops.md](docs/supported-ops.md) | Exact list of traced operators/methods/`np.*` functions (and what isn't yet) |
| [docs/parameters.md](docs/parameters.md) | Config registers: `Param`, `Params` namespace, defaults, matrix kernels, bool/bypass, live control |
| [docs/streaming-and-bitwidths.md](docs/streaming-and-bitwidths.md) | Streaming model, line buffers, edge handling, dtype/bitwidth semantics |
| [docs/framing-and-resolution.md](docs/framing-and-resolution.md) | SOF/EOF framing, continuous frames, parametric & dynamic resolution |
| [docs/interfaces.md](docs/interfaces.md) | AXI4-Stream Video, Switchboard, AXI-Lite/UMI register files, `control_top` |
| [docs/backends.md](docs/backends.md) | numpy / iverilog / cxxrtl / switchboard (verilator, icarus) |
| [docs/view.md](docs/view.md) | Live viewer: camera/screen sources, sliders, FPS |
| [docs/zeroasic.md](docs/zeroasic.md) | Switchboard stack, Logik/FPGA, Platypus, cloud emulation |
| [docs/api.md](docs/api.md) | Python API: `to_ir`, `generate`, adapter generators |

Design rationale (the "why" behind each mechanism) lives in
[`design/`](design/) (`01`–`07`).

## What it can do (built and validated)

- **Trace NumPy → Verilog**: slicing/arithmetic/`astype`/`clip`/`np.pad`/`np.where`,
  flattened to a weighted tap map and lowered to a line IR (line buffers + shift
  registers, automatic delay-line counting, hash-consed sharing).
- **Faithful dtypes**: `uint8` wraps unless widened with `astype`; signed via NumPy
  types; `.clip`/`.saturate`/`.truncate` for narrowing. The oracle is the same
  function on real NumPy dtypes.
- **Config registers**: `Param` (scalar/matrix/bool) with reset `default=`, a
  `Params` namespace for many-register ISPs, programmable kernels, `np.where` bypass.
- **Edges**: same-size output via replicate/zero padding (top/bottom flush during
  blanking, left/right replicate).
- **Framing**: honors AXI-Video `TUSER` (SOF); EOF-driven *height-agnostic* framing;
  continuous multi-frame.
- **Resolution**: WIDTH/HEIGHT as Verilog parameters (per-synthesis), or full
  **runtime** resolution (`MAX_WIDTH` buffers + `active_width` register + VSYNC).
- **Interfaces**: AXI4-Stream Video, Switchboard (SB + packed gearbox), AXI-Lite and
  UMI register files (with frame-synced shadow registers), and a `control_top` that
  wires geometry/config registers to the core.
- **Control plane**: registers at addresses the CALLER allocated (`Reg`), including
  read-only identity words and SLVERR on an unmapped access, and `control_wrap()`,
  which puts that register file in front of any self-describing module — a single
  core or a whole composed design.
- **Run anywhere**: NumPy reference, Icarus, cxxrtl (fast compiled C++), and the real
  **Switchboard** stack (Verilator/Icarus + `PySbTx`/`PySbRx` queues) — every path
  validated bit-exact against the NumPy reference.
- **Tooling**: a generic CLI (`run` file-to-file, `view` live camera/screen with
  sliders + FPS); models are plain `.py` files (no built-in names).
- **FuseSoC**: ships as a generator (`np2hw.core`, command `np2hw-fusesoc`) --
  a consuming core names a model file and geometry, and the Verilog core is
  traced during that project's own build, byte-identical on every run.

## Status

The compiler, interfaces, register files, simulation backends and CLI are
implemented and validated (NumPy == RTL, bit-exact, across the example suite).
Pointwise expression DAGs cover LUT-with-gather and matrix arithmetic with
register coefficients, so multi-channel packed streams (a CCM's shape) trace
today. Next: reductions, >1 pixel/clock, and the FPGA bitstream flow via Logik
— see [docs/zeroasic.md](docs/zeroasic.md) for the path to FPGA/Platypus.

## Funding

Developed independently; recurring support via
[github.com/sponsors/lanserge](https://github.com/sponsors/lanserge), or write
first: **s.rabykin@gmail.com**. Sponsorable capability targets -- each a generic
NumPy-level feature that unblocks everyone downstream, ISP or otherwise:

- **Reductions** ([#1](https://github.com/lanserge/np2hw/issues/1)) -- per-zone
  accumulate/min/max over a frame, which is what any statistics or metering
  block needs.
- **Buffered forks** ([#2](https://github.com/lanserge/np2hw/issues/2)) -- one
  stream feeding consumers that accept at different rates, with the elasticity
  stated and verified.
- **Status registers** ([#3](https://github.com/lanserge/np2hw/issues/3)) --
  live core state (sticky flags, counters) readable back through the generated
  register file, alongside the config it already carries.
- **Multi-stream tracing** ([#4](https://github.com/lanserge/np2hw/issues/4)) --
  blocks that merge or split streams, which today refuse to trace.
- **Bypass-aware power gating** ([#5](https://github.com/lanserge/np2hw/issues/5))
  -- for every register-driven mux
  (`np.where(enable, ...)`), compute the exclusive logic cone it disables --
  exact graph reachability on the traced IR, not a netlist heuristic -- and
  emit operand isolation plus flop clock-enables for it, with the savings
  proven by toggle counts in simulation, not asserted.
- **Portable C emitter** ([#6](https://github.com/lanserge/np2hw/issues/6)) --
  the same pipeline as a plain C function, bit-exact against the NumPy that
  generated the Verilog: a golden model that runs beside the hardware, from
  the one source of truth.
- **Register context banks** ([#7](https://github.com/lanserge/np2hw/issues/7)) --
  N shadow copies of a core's full configuration, swapped atomically at frame
  boundaries: one pipeline serving several streams, context isolation proven
  bit-exact, not asserted.

Scope is agreed in writing before work starts; sponsored work lands in the
open tree immediately, MIT like everything else -- sponsorship buys ordering
and named credit, not exclusivity.

## Author

Serge Rabyking — 12 years at Apical (acquired by Arm) on production ISP/image-processing
IP; patent inventor (WO2016063023A1 / US10063787B2).
