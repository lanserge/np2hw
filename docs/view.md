# Live viewer (`np2hw view`)

Capture frames → run the model → show **input | output** side by side in a live window,
with a **control per config register** and **end-to-end FPS** in the title bar. The
model in the middle is pure NumPy (the same one that compiles to RTL); capture and
display are runtime harness only — not NumPy, not hardware.

```bash
np2hw view examples/isp/gain.py --source camera                # webcam + gain slider
np2hw view examples/isp/gaussian.py --source screen --backend sim --sim cxxrtl
np2hw view examples/isp/multi_param.py --source test --snapshot m.png    # headless
```

## Sources

| `--source` | Backend lib | Notes |
|---|---|---|
| `screen` (default) | Pillow `ImageGrab` | grab the screen — no extra dep; feed it any video on screen |
| `camera` | OpenCV (`[camera]` extra) | webcam; `--camera N` selects the device |
| `test` | — | synthetic moving gradient; for headless / CI |

All sources yield 8-bit grayscale frames, resized to `--size WxH` (default `160x120`).

## Controls (live registers)

Each `Param` becomes a control, updated **every frame** (for both numpy and cxxrtl
backends), so you can dial registers and watch the output react — including through the
actual RTL:

- **multi-bit register** → a slider over its range (e.g. `gain [0..255]`).
- **1-bit register** → a **checkbox** (e.g. an `en`/bypass toggle).
- `--param name=value` sets the *initial* value (defaults to the register's `default`,
  else mid-range); it's not required for `view`.

## FPS

The title bar shows end-to-end frames/sec over a 30-frame window:
```
np2hw: gaussian.py [numpy] 160x120 — 58.3 fps
np2hw: gaussian.py [sim:cxxrtl] 128x96 — 11.4 fps
```
This is the *whole loop* (capture → model → display), so flipping `--backend sim --sim
cxxrtl` lets you literally watch the cost of running through the generated hardware.
For the `sim` backend, drop `--size` if it feels slow.

## Backends

- `--backend numpy` (default) — smooth real-time.
- `--backend sim --sim cxxrtl` — runs the **actual generated RTL** per frame (compiled
  once). Lower FPS — that's the hardware cost, made visible.
- `--backend switchboard [--sb-tool verilator|icarus]` — runs each captured frame
  **through ZeroAsic's Switchboard stack** live (SB-wrapped core + `PySbTx`/`PySbRx`
  pixel queues), built **once** at startup (~10 s). The headline demo: *live camera
  through the SB stack on screen.* If the model has config registers, the build adds
  an **AXI-Lite control interface**, so the **sliders work** — each change is written
  as a register over the SB control plane (`AxiLiteTxRx`) while pixels stream. Caveat:
  it's slow (low FPS, sim — keep `--size` small).

```bash
np2hw view examples/isp/gain.py --source camera --backend switchboard --param gain=24 --size 96x72
# drag the gain slider -> AXI-Lite register write over SB -> output changes live
```

## Headless / CI: `--snapshot`

Writes an input|output montage image instead of opening a window — no display or camera
needed:
```bash
np2hw view examples/isp/gaussian.py --source test --frames 4 --snapshot montage.png
```
Useful to validate the capture→model→compose pipeline in CI. (The live Tkinter window
and webcam need a real display/device.)

## Demo recipes

```bash
# live gain you can turn, on the webcam:
np2hw view examples/isp/gain.py --source camera
# the same, but through the generated hardware (watch FPS drop):
np2hw view examples/isp/gain.py --source camera --backend sim --sim cxxrtl
# a model with a bypass checkbox + sliders:
np2hw view examples/isp/multi_param.py --source screen
```
