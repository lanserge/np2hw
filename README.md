# np2hw

**NumPy → streaming ISP hardware (Verilog) compiler.**

Write image-processing (ISP) code as ordinary NumPy. `np2hw` *traces* it (like JAX/TVM)
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

The focus is **ISP** (the domain where the pattern library is strongest and the
integration with the open-source ZeroAsic stack is most natural); the engine itself
traces general streaming NumPy — stencils, pointwise, edge handling, and mux
(`np.where`). (Reductions, `@`/matmul, and convolve are on the roadmap, not yet
traced — see [docs/supported-ops.md](docs/supported-ops.md).)

## Install

```bash
uv venv && uv pip install -e ".[media]"     # core + image/video IO
```
See [`docs/install.md`](docs/install.md) for the optional extras (`camera`,
`switchboard`) and the system tools (Icarus, Yosys, Verilator).

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
- **Run anywhere**: NumPy reference, Icarus, cxxrtl (fast compiled C++), and the real
  **Switchboard** stack (Verilator/Icarus + `PySbTx`/`PySbRx` queues) — every path
  validated bit-exact against the NumPy reference.
- **Tooling**: a generic CLI (`run` file-to-file, `view` live camera/screen with
  sliders + FPS); models are plain `.py` files (no built-in names).

## Status

Working prototype. The compiler, interfaces, register files, simulation backends,
and CLI are implemented and validated (NumPy == RTL, bit-exact, across the example
suite). Not yet: FPGA bitstream flow via Logik, multi-channel/CCM, >1 pixel/clock.
See [docs/zeroasic.md](docs/zeroasic.md) for the path to FPGA/Platypus.

## Author

Serge Rabyking — 12 years at Apical (acquired by Arm) on production ISP/image-processing
IP; patent inventor (WO2016063023A1 / US10063787B2); sole architect of ChipFlow's
open-source RTL-to-GDSII backend.
