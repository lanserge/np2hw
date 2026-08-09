# 07 — Prior art and positioning


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

Competitive landscape for np2hw. Based on a
deep-research pass (2026-06-20): 22 sources, 25 adversarially-verified claims,
0 refuted. Findings below are cited; read the caveats at the end before quoting.

## Bottom line

To our knowledge, **no existing tool occupies the niche np2hw targets: tracing
*plain, unmodified* NumPy (via the `__array_ufunc__` / `__array_function__`
dispatch protocols) directly into streaming image-processing/ISP RTL.**

Every adjacent Python→FPGA tool surveyed differs on at least one of three axes:

1. **Frontend** — uses its own embedded DSL or a NumPy-*resembling* subset that
   needs decorators / type annotations, not standard NumPy.
2. **Technique** — lowers via AST parsing / custom IR / MLIR, not NumPy dispatch
   tracing.
3. **Output** — emits HLS C/C++ for a downstream vendor synthesizer
   (Vivado / Vitis / Intel OpenCL), not Verilog/VHDL directly.

np2hw is the only point that unites: plain-NumPy dispatch tracing **+** direct
streaming RTL **+** ISP domain focus.

## Closest existing work

| Tool | Input | Output | ISP-focused | Why it is not np2hw |
|---|---|---|---|---|
| **PyLog** (Cornell) | `@pylog` Python subset; NumPy only at the boundary for shape/type | HLS C (Vivado downstream) | no | own decorator + map/dot operators; AST→PLIR compiler, not dispatch tracing; supports only a curated NumPy subset |
| **Allo** (Cornell, PLDI'24) | Python-embedded MLIR ADL; explicit type annotations (`float32[K,N]`) | HLS C++ / bitstream via Vitis | no (LLM / PolyBench examples) | own ADL; **supersedes HeteroCL**; ML-oriented |
| **HeteroCL** (Cornell) | `hcl.compute` / `hcl.placeholder` DSL | HLS | partial | superseded by Allo per its own README |
| **DaCe** (ETH Zurich) | NumPy-like Python subset via decorators (SDFG) | vendor HLS (Vivado / Intel) | no | not plain NumPy; its RTL backend is for hand-written SystemVerilog tasklets, not auto-generated Verilog |
| **Halide-to-Hardware** (Stanford AHA) | Halide DSL embedded in C++ | hardware via CoreIR | **yes** | Halide DSL, not NumPy |
| **HeteroHalide** | existing Halide C++ programs | FPGA via HeteroCL IR | **yes** | Halide input |
| **SODA** (UCLA-VAST) | own stencil DSL (`kernel`, `burst width`, `unroll`, …) | HLS | **yes** | custom stencil DSL |
| **Stream-HLS / Prometheus** (2025) | PyTorch or C/C++ via Torch-MLIR / Polygeist | HLS via MLIR | no | no NumPy path |

Two structural facts hold across the whole set (both unanimously verified):

- **None use NumPy dispatch tracing** (`__array_ufunc__`/`__array_function__`)
  to target RTL — that JAX/CuPy-style technique is distinct prior-art territory
  for np2hw.
- **Most emit HLS C, not Verilog** — they rely on Vivado/Vitis/Intel as the
  actual RTL generator. np2hw's direct streaming-RTL emission is a real
  difference.

The strongest "this already exists" objections are the image-processing tools
(Halide-to-Hardware, HeteroHalide, SODA) — but **all take Halide or a stencil
DSL, never NumPy.**

## Genuine differentiators

1. **Zero new syntax.** Your existing NumPy *is* the program — no DSL, no
   decorators, no type annotations. Halide/HeteroCL/Allo/SODA all impose an API.
2. **Spec = oracle, bit-exact.** The same NumPy function validates the generated
   hardware pixel-for-pixel (faithful dtype + signed semantics; see
   `04-bitwidth-tracking.md`). Most surveyed tools validate against a separate
   reference.
3. **ISP production pattern library.** Apical-era knowledge of which filter
   variants actually ship in silicon (`05-isp-operation-library.md`) — domain IP
   none of the academic tools encode.
4. **Open-silicon stack fit.** Switchboard / Logik / Platypus integration
   (`06-zeroasic-integration.md`). Halide-HW targets specific CGRAs; the Python
   tools target AMD/Intel via proprietary HLS.

## How np2hw differs (one-paragraph summary)

> Existing image-processing HLS (Halide-to-Hardware, HeteroHalide, SODA)
> requires authoring in a dedicated DSL; existing Python-FPGA tools (PyLog,
> Allo, DaCe) require a NumPy-resembling subset with annotations and emit HLS C
> for a proprietary downstream synthesizer. np2hw is, to our knowledge, the
> first to trace **unmodified NumPy** via its dispatch protocols directly into
> **streaming ISP RTL** — combining a zero-new-syntax frontend, an ISP-specific
> pattern library, and open-stack (Switchboard/Logik) integration.

## Caveats (do not overstate)

- **Not tool-specifically verified**: Hipacc, Darkroom/Rigel, PolyMage-HLS, and
  the ML-to-FPGA frontends hls4ml, FINN, TVM/VTA. Conclusions about these rest
  on the general pattern, not direct evidence.
- The "niche unoccupied" claim is **absence-of-evidence** across the surveyed
  set — phrase it as *"to our knowledge"*, never *"no tool exists."* It cannot
  rule out an obscure or very recent project.
- Project maintenance/activity (commit recency, release cadence) was not
  verified beyond HeteroCL's self-declared supersession by Allo.

## Open follow-ups

- Input language, RTL-emission strategy, and maintenance of **Hipacc**,
  **Darkroom/Rigel**, **PolyMage-HLS**.
- Whether any ML-to-FPGA frontend (**hls4ml**, **FINN**, **TVM/VTA**) is
  dispatch-traced vs strictly ONNX/Keras/PyTorch-model driven.
- Whether any tool emits Verilog/VHDL *directly* from a NumPy-like frontend, and
  how its line-buffer/shift-register generation compares to ours.

## Key sources

- PyLog — https://sitaohuang.com/publications/2021_pylog.pdf
- Allo (PLDI'24) — https://www.csl.cornell.edu/~zhiruz/pdfs/allo-pldi2024.pdf ,
  https://github.com/cornell-zhang/allo
- HeteroCL — https://github.com/cornell-zhang/heterocl
- DaCe (Python FPGA, ETH Zurich) — https://arxiv.org/pdf/2212.13768
- Halide-to-Hardware — https://github.com/StanfordAHA/Halide-to-Hardware ,
  https://arxiv.org/pdf/2105.12858
- HeteroHalide — https://dl.acm.org/doi/10.1145/3373087.3375320
- SODA — https://github.com/UCLA-VAST/soda
- Prometheus / Stream-HLS (2025) — https://arxiv.org/pdf/2501.09242
</content>
