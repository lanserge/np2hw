# CLI reference

The CLI has two commands for a model: **`run`** (file-to-file) and **`view`**
(live). The model is always a **`.py` file** — there are no built-in model names.
A pipeline has its own: **`pipeline run`** and **`pipeline generate`**, below. See
[writing-models.md](writing-models.md) for the model-file convention.

```bash
np2hw run  MODEL INPUT OUTPUT [options]
np2hw view MODEL [options]
```

`MODEL` is `path/to/file.py` (uses the function `model` or `isp`) or
`path/to/file.py:func` to pick a function by name.

---

## `np2hw run` — apply a model to an image/video

```bash
np2hw run examples/isp/gaussian.py in.png  out.png
np2hw run examples/isp/gain.py     in.gif  out.gif --param gain=24   # animated GIF = "video"
np2hw run examples/isp/sharpen.py  in.png  out.png
# pick a function by name in a multi-function file:  np2hw run your_model.py:funcname …
```

| Option | Values | Meaning |
|---|---|---|
| `--backend` | `numpy` (default), `sim`, `rtl`, `switchboard`/`sb` | how to run — see [backends.md](backends.md) |
| `--sim` | `iverilog` (default), `cxxrtl` | engine for `--backend sim` (`rtl` = alias for `sim`) |
| `--sb-tool` | `verilator` (default), `icarus` | engine for `--backend switchboard` |
| `--param` | `name=value` (repeatable) | a config register value; required for each `Param` the model declares |

- **Media**: still images via Pillow (PNG/JPG/BMP/…); "video" via animated GIF
  (multi-frame, no ffmpeg). Frames are processed as **8-bit grayscale**.
- **`--param`** is needed once per declared register, e.g. `--param gain=24 --param bias=8`.
  Matrix/namespace registers expose one leaf per element (`--param k_0_0=…`).

Examples:
```bash
np2hw run examples/isp/gain.py     in.png out.png --param gain=24
np2hw run examples/isp/gaussian.py in.gif out.gif --backend sim --sim cxxrtl
np2hw run examples/isp/gain.py     in.png out.png --param gain=24 --backend switchboard
```

---

## `np2hw view` — live capture → model → display

Opens a window showing **input | output** side by side, a **slider per `Param`**
(checkbox for 1-bit registers), and **end-to-end FPS** in the title bar. See
[view.md](view.md).

```bash
np2hw view examples/isp/gain.py --source camera
np2hw view examples/isp/gaussian.py --source screen --backend sim --sim cxxrtl
```

| Option | Values | Meaning |
|---|---|---|
| `--source` | `screen` (default), `camera`, `test` | frame source; `camera` needs the `camera` extra |
| `--backend` | `numpy` (default), `sim`, `switchboard`/`sb` | `sim` runs the RTL (cxxrtl) live; `switchboard` runs each frame through the SB stack (built once, slow). With config registers it adds an AXI-Lite control interface so **sliders work live** over the SB control plane |
| `--sim` | `cxxrtl` | engine for `--backend sim` |
| `--sb-tool` | `verilator` (default), `icarus` | engine for `--backend switchboard` |
| `--size` | `WxH` (default `160x120`) | processing size (smaller = faster, esp. for `sim`) |
| `--camera` | int (default 0) | camera device index |
| `--param` | `name=value` | *initial* register value (sliders change it live; not required) |
| `--snapshot` | path | headless: write an input\|output montage file instead of opening a window |
| `--frames` | int (default 8) | frames to capture for `--snapshot` |

- `--source test` is a synthetic moving gradient — useful for headless/CI checks.
- `--snapshot` needs no display or camera, so it works in CI:
  ```bash
  np2hw view examples/isp/gaussian.py --source test --frames 4 --snapshot montage.png
  ```

---

## `np2hw pipeline` — a composite of blocks

A pipeline is a composite of model blocks, named by reference
(`package.module:name`, found on the import path or under `--library DIR`).

```bash
np2hw pipeline run acme.isp:core in.npy out.png
np2hw pipeline run acme.isp:core --from gamma.in mid.npy out.png   # a sub-chain
np2hw pipeline run acme.isp:core --rtl in.npy out.npy              # the RTL twin too
np2hw pipeline run acme.isp:core --c --rtl in.npy out.npy          # C and RTL both
np2hw pipeline generate acme.isp:core --out build --clock-mhz 148.5
```

`run` applies the models over one frame (`.npy` raw integers, `.png` an 8-bit
view), the whole pipeline or the blocks between `--from` and `--to`;
`--profile` layers a block's register values, `--set INSTANCE.REG=VALUE` one
register, `--context SIGNAL=VALUE` one context signal, and `--explain` prints
where every value came from. `--c` runs every block as its model printed as
C (docs/backends.md) and `--rtl` the generated Verilog under Verilator,
each beside the models, and each fails on any difference.

`generate` proves the pipeline's Verilog bit-exact against its models under
Verilator -- every leg, each input to each output it reaches, a fork's every
branch among them -- then writes the design pack (`np2hw.pipeline.pack`): the
Verilog, the register map, SystemRDL, the register documentation,
`<name>_build.json` (top, latency, geometry, boundary widths, memories)
and a FuseSoC `.core`.
`--no-control` stops at the datapath, `--sync-memory` refuses a line buffer
that reads combinationally, `--vlnv` names the core.

---

## Exit / errors

- A bare name (not a file) errors: *"model file not found … pass a .py file"*.
- A missing `--param` errors: *"model needs --param NAME=<value>"*.
- `--backend switchboard` without the extra errors with the install hint.
