# Install

`np2hw` is a Python package. The core only needs **NumPy**; image/video IO, the live
viewer, and the Switchboard backend are **optional extras**. RTL simulation uses
**system tools** (Icarus / Yosys / Verilator) invoked as subprocesses.

## With uv (recommended)

```bash
cd np2hw
uv venv
uv pip install -e ".[media]"        # core + Pillow (image/GIF IO)
uv run np2hw run examples/isp/gaussian.py in.png out.png
```

`uv run np2hw …` puts the `np2hw` command on the path with no `PYTHONPATH`. An
editable install (`-e`) means code edits take effect immediately.

## Optional extras

| Extra | Pulls in | Enables |
|---|---|---|
| `media` | `pillow` | `np2hw run` on PNG/JPG/BMP and animated GIF "video" |
| `camera` | `opencv-python` | `np2hw view --source camera` (webcam); screen/test sources need no extra |
| `switchboard` | `switchboard-hw`, `umi` (pull in `siliconcompiler`) | `np2hw run --backend switchboard` |

```bash
uv pip install -e ".[media,camera]"             # add webcam
uv pip install -e ".[media,switchboard]"        # add the Switchboard stack
```

The core stays NumPy-only on purpose — none of these are required to compile NumPy
to Verilog; they only power the optional run/view/integration paths.

## System tools (for the simulation backends)

These are external binaries, not pip packages. Install whichever backend you use:

| Tool | Used by | Notes |
|---|---|---|
| `iverilog` / `vvp` | `--sim iverilog`, all example testbenches | Icarus Verilog |
| `yosys` + `yosys-config`, a C++ compiler | `--sim cxxrtl` | Yosys `write_cxxrtl` + compiled C++ model (fast) |
| `verilator` | `--backend switchboard --sb-tool verilator` | fast Switchboard engine |

(`iverilog` is also the Switchboard `--sb-tool icarus` engine.) On macOS these come
from Homebrew; on Linux from your package manager or the
[OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build).

## Without uv

```bash
PYTHONPATH=src python3 -m np2hw.cli run examples/isp/gaussian.py in.png out.png
PYTHONPATH=src python3 examples/sobel.py        # run an example/test
```

## Verifying the install

```bash
uv run np2hw run examples/isp/gaussian.py in.png out.png                 # numpy
uv run np2hw run examples/isp/gaussian.py in.png out.png --backend sim --sim cxxrtl
```
If both produce identical output, the compile + cxxrtl path work end to end.
