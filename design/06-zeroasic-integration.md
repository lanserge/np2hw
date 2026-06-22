# 06 — ZeroAsic stack integration


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## The three pieces

ZeroAsic's "emulation platform" is a coordinated stack:

### Switchboard (open source — zeroasiccorp/switchboard)

Distributed simulation/emulation framework. Connects heterogeneous components:
- RTL simulators (Verilator, etc.)
- Real FPGAs
- C++ models
- Python host code

All talk via shared-memory queues (~hundreds of ns latency). Designs expose **SB ports** carrying packets:
- 32-bit destination
- 32-bit flags (including 'last' bit for transaction framing)
- 416-bit data payload

Has built-in SiliconCompiler integration via `SbDut`. Python host bindings: `PySbTx`, `PySbRx`, `UmiTxRx`.

### Logik (open source — siliconcompiler/logik)

FPGA toolchain converting RTL → bitstream. Pipeline:
- **Yosys** → synthesis
- **VPR** → placement and routing
- **FASM** → bitstream generation

Orchestrated by SiliconCompiler. Supports inputs: SystemVerilog, Verilog, VHDL, C, Chisel, Python, Bluespec.

### Platypus eFPGA (commercial — IP cores)

Their embedded FPGA IP. Not a dev board you buy off the shelf — it's licensable IP for chip designs. ZeroAsic has internal Platypus-based hardware for emulation/dev work.

## Our integration path

`np2hw` emits Verilog with **Switchboard-compatible ports** (streaming I/O), then hands off to existing ZeroAsic tooling for the rest:

```
NumPy ISP algorithm
    ↓ np2hw (our work)
Switchboard-wrapped Verilog
    ↓ Logik (existing — Yosys + VPR + FASM)
FPGA bitstream
    ↓ load to Platypus instance
Hardware running ISP algorithm
    ↑ Switchboard streams I/O
Host-side Python driver (the SAME NumPy code serves as validation reference)
```

## Switchboard port wrapping

For a NumPy function:
```python
@hls_compile(inputs={'pixel_in': ...}, outputs={'pixel_out': ...})
def some_isp(pixel_in):
    ...
```

The generated Verilog wraps the core in Switchboard input/output ports:

```verilog
module some_isp_top (
    input  wire clk, rst,
    
    // Switchboard input port (streams pixels in)
    output wire        sb_in_ready,
    input  wire        sb_in_valid,
    input  wire [415:0] sb_in_data,
    input  wire [31:0]  sb_in_dest,
    input  wire [31:0]  sb_in_flags,
    
    // Switchboard output port (streams pixels out)
    input  wire        sb_out_ready,
    output wire        sb_out_valid,
    output wire [415:0] sb_out_data,
    output wire [31:0]  sb_out_dest,
    output wire [31:0]  sb_out_flags
);
    // Unpack input packets to pixel stream
    // ... pixel_in_stream from sb_in_data[7:0] ...
    
    // Instance the np2hw-generated core
    some_isp_core core_inst (
        .clk(clk), .rst(rst),
        .pixel_in(pixel_in_stream),
        .pixel_out(pixel_out_stream)
    );
    
    // Pack output stream to Switchboard packets
    // ... sb_out_data <= {408'b0, pixel_out_stream} ...
endmodule
```

## Host-side validation

The killer feature: the SAME NumPy code serves as the validation reference.

```python
import switchboard as sb
import numpy as np

# Reference NumPy implementation (the one passed to @hls_compile):
def reference_isp(img):
    return my_isp_pipeline(img)   # plain NumPy

# Connect to running emulation:
tx = sb.PySbTx("/tmp/sb_in")
rx = sb.PySbRx("/tmp/sb_out")

# Stream a test image in:
test_img = load_test_image()
for px in test_img.flatten():
    tx.send(sb.Packet(data=px, ...))

# Receive hardware output:
hw_result = np.zeros_like(test_img)
for i in range(test_img.size):
    hw_result.flat[i] = rx.recv().data & 0xFF

# Compare against reference:
ref_result = reference_isp(test_img)
assert np.allclose(hw_result, ref_result, atol=1)
```

The same Python code defines the hardware AND validates the hardware. End-to-end closed-loop testing for free.

## Why the fit is natural

1. **No new infrastructure required** — the IP fits the existing open stack rather than needing changes to it
2. **Switchboard is already designed for this** — heterogeneous co-simulation with Python host drivers is its core use case
3. **Logik already supports Python frontends** — the generated Verilog plugs into the existing path
4. **The closed-loop validation property is unique** — no other HLS tool has "same source serves as spec + validator"

## Open questions

- Access to Platypus emulation infrastructure (internal hardware vs cloud)
- Whether Switchboard's SB-port spec is stable
- Whether to keep `np2hw` standalone or upstream into an open org

These are integration questions, not engineering blockers.
