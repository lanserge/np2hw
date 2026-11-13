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

Beside these, the model runs as **C** on the CPU (`compile_c`, `np2hw pipeline
run --c`; below): not a simulation of the hardware, the same graph printed as a
C function, bit-exact against NumPy.

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

The generated core+wrapper is registered as a siliconcompiler `Design` with an `rtl`
fileset plus `icarus`/`verilator` filesets that depend on it; `SbDut` then selects the
fileset matching its `tool`, which pulls in the C++ testbench + DPI automatically (the
recommended setup per [zeroasiccorp/switchboard#309](https://github.com/zeroasiccorp/switchboard/issues/309)
— so we don't add those sources by hand). A few Verilator lint warnings are waived,
including `PINMISSING` for an upstream reset-pin gap in the SB sim shims and the usual
width/sign notes from the generated bit-width arithmetic. Streaming uses a
single-threaded **non-blocking send/recv interleave** (the SB queues are bounded, so
sending a whole frame before reading would deadlock).

## Which to use

- **Developing a model / quick check** → `numpy`, or `sim --sim iverilog` for one image.
- **Video / performance / the live viewer** → `sim --sim cxxrtl`.
- **Proving it inside the ZeroAsic stack** → `switchboard --sb-tool verilator`.
- **A golden model beside the hardware, at CPU speed** → C (`compile_c`, `pipeline run --c`).

## C -- the model on a CPU (`np2hw.core.cemit`)

Not a simulation of the hardware: the canonical model graph -- the one the
Verilog is lowered from -- printed as a plain C11 function, computing what
the model computes, bit for bit, at CPU speed.

```python
from np2hw import Image2D
from np2hw.core.cemit import compile_c

k = compile_c(model, Image2D("img", 640, 480, bits=8), *params)
out = k(frame, threads=4, gain=24)      # registers by name
```

Each node is printed as `evaluate.py` runs it, at a position: a read of the
source or a table is a load, a read of a pad its edge rule, a read of a
computed value that value computed at the index. Loops are the output's
variables (an axis of up to 16 positions -- channels -- is unrolled), and
an output read through a C-order reshape is the value under it, written in
its own loops -- the same memory, without the reshape's index math; each
statement sits in the outermost loop it depends on; along the innermost
loop, the positions no pad read can leave run with no clamps, between two
edge loops. Every statement is typed by the narrowest integer holding its
exact range, its operands converted first, so C's promotions never change a
value -- except a sum, difference, product or negation wider than 64 bits,
which NumPy wraps in its 64-bit container: that one is computed modulo
2**64 in `uint64_t`, exact in the low 64 bits the wrap keeps. The function's ABI is caller-owned -- `src`, `dst`, the registers in
order, and a range of the outermost loop, which is how threads take bands
-- and its file has no dependency beyond `<stdint.h>`.

A kernel is built for one frame size and builds itself again for another.
A register window at the output (`img[p.y0:, p.x0:][:p.h, :p.w]`) is
computed over the whole frame and kept by its registers, as NumPy's crop
keeps it. A `np.roll` along the frame, which hardware refuses, is a read
at a wrapped index here, and a signed source, which hardware refuses too,
is read as its type -- each lane of a multi-lane image signed where the
image is. Refused by name: a canvas whose planes do not
tile a lattice, a register window placed by computed values. A block is printed the same way
through `Block.compile_c` and run in its model's place by `Block.run(...,
kernel=...)`; `np2hw pipeline run --c` runs a whole chain so. Checked by `tests/test_cemit.py`: every model the core tests compile
to Verilog, and the shipped model files, against NumPy; and by
`tests/test_block_c.py`, blocks and chains against their models.
