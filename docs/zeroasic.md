# ZeroAsic integration (Switchboard → Logik → Platypus)

`np2hw` is designed to plug into ZeroAsic's open silicon stack. The goal is a **closed
loop**: a NumPy ISP algorithm becomes SB-wrapped Verilog, runs through Switchboard,
Logik, and Platypus emulation, and is validated against the **same NumPy** at every
stage. This doc maps what's done, what's next, and the relevant ZeroAsic pieces.

The stack:
- **Switchboard** — distributed simulation/emulation transport (open source). Uniform
  packet queues (shared-memory / TCP / PCIe) so blocks in sim, emulation, and silicon
  interoperate. *Why it exists*: glue heterogeneous components behind one streaming
  interface so the same design + test move across the sim→emulation→silicon continuum.
- **Logik** — open FPGA toolchain (RTL → bitstream) on SiliconCompiler.
- **Platypus** — ZeroAsic's open-standard embedded FPGA (eFPGA) IP (commercial).

## Done: run the IP through Switchboard

`np2hw run model.py in.png out.png --backend switchboard` runs the generated IP through
the **real Switchboard framework**: SB-wrap → `SbDut` (Verilator or Icarus) →
shared-memory SB queues → `PySbTx`/`PySbRx` → validated bit-exact vs NumPy. See
[backends.md](backends.md) and [interfaces.md](interfaces.md). This is the
software-stack half of the closed loop, working today.

## Next: bitstream via Logik

To run on an FPGA, the SB-wrapped (synthesizable, yosys-clean) RTL goes through **Logik**
(SiliconCompiler FPGA flow) with a target FPGA part, producing a bitstream. Not yet
wired in `np2hw`; the natural addition is an `np2hw fpga model.py --part <fpga>`
scaffold. The RTL is ready (it's the same module the sim backends compile).

## Then: FPGA / Platypus emulation

Load the bitstream on an FPGA (or the Platypus eFPGA) and connect the host over
Switchboard's hardware transport — typically **PCIe** (`PySbTxPcie`/`PySbRxPcie`); the
FPGA side instantiates a UMI/SB endpoint around the core. The host test is the **same**
`PySbTx`/`PySbRx` code as the sim backend — only the transport changes. The usual
de-risking is to prove it on a stock FPGA dev board (Logik + Switchboard PCIe) before
the commercial eFPGA.

### ZeroAsic cloud emulation

ZeroAsic runs a cloud emulation platform (`emulation.zeroasic.com`) and a Logik-tied
"digital twin" cloud simulation. As of writing, **custom-RTL upload to the public
emulation site is not yet available** (it's demo/chiplet-driven), so today the path for
a custom np2hw core is **Logik bitstream → cloud digital-twin simulation**, or
self-hosted Switchboard emulation on your own cloud/FPGA. Confirm custom-RTL access and
the target part directly with ZeroAsic.

## Status summary

| Stage | Status |
|---|---|
| NumPy → SB-wrapped Verilog | ✅ done |
| Run through Switchboard (Verilator/Icarus + queues), validated vs NumPy | ✅ done |
| AXI-Lite / UMI register files, frame-synced, `control_top` | ✅ done |
| Logik bitstream flow | ⛔ not wired (RTL is ready) |
| FPGA / Platypus on hardware via PCIe | ⛔ needs board + ZeroAsic specifics |
| Custom-RTL on ZeroAsic cloud emulation | ⛔ upload not yet public |

## Why this matters

Nobody else's HLS has the property that the **same NumPy source is the spec, the
hardware, and the host-side validation reference** across this whole continuum — an
accessible NumPy→ISP-RTL path that drops natively into the open silicon stack.
(Design rationale: [`design/06-zeroasic-integration.md`](../design/06-zeroasic-integration.md).)
