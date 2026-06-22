# Interfaces & adapters

The core has a plain ready/valid streaming interface plus `param_*` register inputs and
SOF/EOL/last framing. **Adapters** wrap it for specific ecosystems — each is a separate,
generated module, so the core stays reusable. Import them from `np2hw.verilog`.

```python
from np2hw import to_ir, generate, Image2D
from np2hw.verilog import (switchboard_wrap, axis_video_wrap,
                           axil_regfile, umi_regfile, control_top)

_, out = to_ir(model, Image2D("img", W, H, 8), *params)
core = generate(out, "gcore")               # dict: {"verilog", "module", "params", ...}
```

## AXI4-Stream Video

```python
wrap = axis_video_wrap(core, W, H, module_name="gcore_axis")
```
UG934-style: `s_axis`/`m_axis` with `tvalid`/`tready`/`tdata`, **`tlast = End Of Line`**,
**`tuser[0] = Start Of Frame`**, byte-aligned `tdata`, active-low `aresetn`. The slave
honors incoming `s_axis_tuser` (→ core `in_sof`) to anchor frames; the master drives
`m_axis_tuser`/`tlast` from the core's framing. (See `examples/axi_video.py`.)

## Switchboard (ZeroAsic)

```python
wrap = switchboard_wrap(core, W, H, module_name="gcore_sb")              # 416-bit SB ports
wrap = switchboard_wrap(core, W, H, native=True)   # SB_CONNECT bundle (1-bit `last`)
wrap = switchboard_wrap(core, W, H, pack=True)     # pixel-packing gearbox (throughput)
```
- **pass-through** (default): one pixel in the low bits of the 416-bit payload; `flags`
  carries `last` in bit 0. Simple; fine for validation.
- **`native=True`**: exact Switchboard SB bundle (`_data/_dest/_last/_valid/_ready`,
  1-bit `last`) so it drops into `SbDut` autowrap / the `SB_CONNECT` macro. Used by the
  `--backend switchboard` runner.
- **`pack=True`**: a gearbox that unpacks `P_in = 416//in_bits` pixels/packet → core →
  repacks `P_out`/packet, with `last` framing each output frame — for real throughput.
  This is what the `--backend switchboard` runner uses by default (`native=True,
  pack=True`); `PySbTx`/`PySbRx` pack/unpack to match.

**`switchboard_control_wrap(meta, W, H)`** — a combined top: the packed SB datapath
**plus an AXI-Lite register file** driving the core's params, exposed as an SB pixel
interface (`sb_in`/`sb_out`) and an AXI-Lite control interface (`ctrl_*`). This is
what lets config registers be written at **runtime over the SB control plane**
(`AxiLiteTxRx`) while pixels stream — used by `view --backend switchboard` so the
live sliders work.

To actually *run* through the Switchboard stack, use `--backend switchboard`
([backends.md](backends.md)). The bigger picture: [zeroasic.md](zeroasic.md).

## Control register files (AXI-Lite / UMI)

Expose the model's `Param` registers (and geometry registers) to software over a bus.

```python
reg = axil_regfile(core["params"], frame_sync=True, defaults=core["param_defaults"])
reg = umi_regfile(core["params"],  frame_sync=True, defaults=core["param_defaults"])
```
- One register per `Param` at word offset `i*4`; `param_<name>` wires drive the core.
- **`defaults=`** sets each register's **reset value** (from `Param(default=)`), so the
  IP boots configured instead of all-zero.
- **`frame_sync=True`** adds **shadow registers** + an `update` input: software writes
  land in a shadow, and the live value copies from it on an `update` pulse. Wire
  `update` to the core's `out_last` and writes take effect **at the frame boundary** —
  glitch-free, no mid-frame tearing. (See `examples/control.py`.)

The UMI variant models write/read register semantics for the Switchboard stack; for
production, connect `param_*` to Switchboard's `umi_endpoint`/`umi_regif`.

## `control_top` — register file + core, wired

```python
top = control_top(core, ctrl="axil", frame_sync=True)   # or ctrl="umi"
```
Emits a top module that instances the register file **and** the core and connects them:
each `Param` → its register, and for a **dynamic-resolution** core
([framing-and-resolution.md](framing-and-resolution.md)) it adds an **`active_width`**
register and wires `param_active_width → core.active_width`. So software sets the line
length over AXI-Lite/UMI like any tuning register. Returns the top Verilog (regfile +
top), the register-file `offsets`, etc. (See `examples/active_width.py`.)

> Note: with `frame_sync=True` a written value goes live at the **next frame boundary**,
> so the first frame uses the register's `default`. Use `frame_sync=False` for immediate
> initial config; `frame_sync=True` for glitch-free runtime changes.

## What the adapters share

All take the `core` dict from `generate(...)` and pass the `param_*` ports through, so a
model with config registers works through every adapter unchanged. The core itself is
never modified — adapters are pure wrappers.
