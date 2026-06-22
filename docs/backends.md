# Backends (how a model is run)

Every backend runs the **same model** and (for the hardware ones) is validated **bit-exact
against the NumPy reference**. That's the central guarantee — the NumPy function is the
oracle, and two independent simulators plus the Switchboard stack all agree with it.

| `--backend` | `--sim` / `--sb-tool` | What runs | Speed | Needs |
|---|---|---|---|---|
| `numpy` (default) | — | the Python function | fastest | core only |
| `sim` (= `rtl`) | `iverilog` | generated Verilog in Icarus, per frame | slow (per-frame vvp) | `iverilog` |
| `sim` | `cxxrtl` | generated Verilog compiled to C++ (Yosys cxxrtl), all frames one process | **fast** | `yosys`, C++ compiler |
| `switchboard` (= `sb`) | `verilator` (default) | SB-wrapped core in Verilator + SB queues | medium | `[switchboard]` extra, `verilator` |
| `switchboard` | `icarus` | SB-wrapped core in Icarus + SB queues | slow | `[switchboard]` extra, `iverilog` |

```bash
np2hw run m.py in.png out.png                                  # numpy
np2hw run m.py in.gif out.gif --backend sim --sim cxxrtl      # fast RTL (video)
np2hw run m.py in.png out.png --backend switchboard           # ZeroAsic stack (verilator)
np2hw run m.py in.png out.png --backend switchboard --sb-tool icarus
```

## numpy — the reference

Runs your function directly. It's the **spec/oracle** the others are checked against.
Register values are passed as NumPy-typed scalars so it promotes exactly like the
hardware ([parameters.md](parameters.md)).

## sim / iverilog

Compiles the core to Verilog once, then streams each frame through a generated Icarus
testbench (the same path the example suite uses). Robust and dependency-light; slow
because each frame is a fresh `vvp` run. Good for a single image or a quick check.

## sim / cxxrtl — the fast one

Yosys `write_cxxrtl` turns the core into a C++ class; a generated driver streams **all
frames in one process**. Wins decisively for video (e.g. a 60-frame GIF). Config
registers are read at runtime via argv, so it's also what the live viewer uses for the
`--backend sim` path. This is the engine to use for anything multi-frame.

## switchboard — the ZeroAsic stack

Runs frames through **ZeroAsic's real Switchboard framework**: the core is SB-wrapped
(`switchboard_wrap(native=True)`), built with `SbDut` autowrap, and frames are streamed
through **shared-memory SB queues** with `PySbTx`/`PySbRx` — the closed loop from the
project's integration goal. Two engines:

- **`--sb-tool verilator`** (default) — fast; the recommended Switchboard engine.
- **`--sb-tool icarus`** — simpler build, slower.

It uses the **packed gearbox** by default — `p_in = 416 // in_bits` pixels per packet
(52 for 8-bit, 34 for 12-bit, …) in, `p_out` per output packet — so the 416-bit SB
payload is filled rather than carrying one pixel. Still Icarus/Verilator simulation
(not silicon-rate), so keep demo images modest, but far fewer packets than one-per-
pixel. Needs the `[switchboard]` extra; see [install.md](install.md) and
[zeroasic.md](zeroasic.md).

### Implementation notes (Switchboard backend)

Worked around three gaps in the installed `switchboard-hw` 0.3.3 (Verilator path):
the C++ main and the DPI source aren't auto-added (we add `verilator/testbench.cc` +
`dpi/switchboard_dpi.cc`), and sim-module lint warnings are waived. Icarus needs none of
this. Streaming uses a single-threaded **non-blocking send/recv interleave** (the SB
queues are bounded, so sending a whole frame before reading would deadlock).

## Which to use

- **Developing a model / quick check** → `numpy`, or `sim --sim iverilog` for one image.
- **Video / performance / the live viewer** → `sim --sim cxxrtl`.
- **Proving it inside the ZeroAsic stack** → `switchboard --sb-tool verilator`.
