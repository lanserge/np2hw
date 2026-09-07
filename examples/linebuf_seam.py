# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The memory seam, and the refusal that guards it.

A line buffer is `reg [W-1:0] mem [0:D-1]`. Inferred on an FPGA that is
block RAM; on silicon it is D*W flip-flops, and one 1920-wide 12-bit
line is 23,040 of them. So the stencil emitters that read their line
buffers through a register put the array behind a module of its own --
`<module>_linebuf`, one write port, one read port, both synchronous --
which an integrator binds a memory macro to by name. The contract the
binding must honour is written on the module (docs/memories.md).

Not every emitter can offer that. The general window reads its line
buffers COMBINATIONALLY -- distributed RAM on an FPGA, and no SRAM macro
implements a combinational read. `sync_memory=True` refuses such a model
rather than emit it; the same model generates as before without the
flag, because on an FPGA the asynchronous read is the right choice.

Three things are proven here, each as something that can fail:

  1. a padded model, generated under sync_memory=True, is bit-exact and
     its line buffers are seam instances (the shape, asserted);
  2. the same padded model generates IDENTICALLY with and without the
     flag -- the flag never changes what is emitted, only what is refused;
  3. an unpadded model is refused under the flag, by name and with the
     reason, and still generates and simulates bit-exact without it.

Run:  python examples/linebuf_seam.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw import Image2D, generate, to_ir

W, H = 8, 6
A = np.random.default_rng(29).integers(0, 256, (H, W), dtype=np.uint8)


def blur121_edge(img):
    """Vertical [1,2,1]/4, replicate edges: same height, sync line buffers."""
    x = np.pad(img.astype(np.uint16), ((1, 1), (0, 0)), mode="edge")
    return ((x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]) // 4).astype(np.uint8)


def blur121(img):
    """The same filter unpadded: shrinks by two rows, reads asynchronously."""
    x = img.astype(np.uint16)
    return ((x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]) // 4).astype(np.uint8)


def main():
    results = []
    print("the memory seam:")

    # 1. behind the seam, under the flag, bit-exact
    results.append(check("seam_edge", blur121_edge, A, sync_memory=True))
    _, out = to_ir(blur121_edge, Image2D("img", W, H, bits=8))
    core = generate(out, module_name="seam_edge", sync_memory=True)
    src = core["verilog"]
    seam = ("module seam_edge_linebuf #(" in src
            and src.count(" u_linebuf") == 2
            and "if (re) rd <= mem[ra];" in src
            and ".rd(chain1_q)" in src)
    published = ([m["read"] for m in core["memories"]] == ["sync", "sync"]
                 and all(m["module"] == "seam_edge_linebuf"
                         for m in core["memories"]))
    print(f"  two line buffers are seam instances               "
          f"{'PASS' if seam else 'FAIL'}")
    print(f"  published as memories, read=sync, by module name  "
          f"{'PASS' if published else 'FAIL'}")
    results += [seam, published]

    # 2. the flag changes nothing that is emitted
    plain = generate(out, module_name="seam_edge")["verilog"]
    same = plain == src
    print(f"  with and without the flag, byte-identical         "
          f"{'PASS' if same else 'FAIL'}")
    results.append(same)

    # 3. the refusal, named, and the unrefused path still bit-exact
    _, out_async = to_ir(blur121, Image2D("img", W, H, bits=8))
    try:
        generate(out_async, module_name="seam_async", sync_memory=True)
        refused, named = False, False
    except ValueError as error:
        text = str(error)
        refused = True
        named = ("seam_async" in text and "ASYNCHRONOUSLY" in text
                 and "np.pad" in text)
    print(f"  an asynchronous line buffer is refused            "
          f"{'PASS' if refused else 'FAIL'}")
    print(f"  ...by module name, with the reason and the fix    "
          f"{'PASS' if named else 'FAIL'}")
    async_core = generate(out_async, module_name="seam_async")
    async_ok = ([m["read"] for m in async_core["memories"]] == ["async"] * 2
                and "chain1 = mem1[" in async_core["verilog"])
    print(f"  without the flag it emits, published read=async   "
          f"{'PASS' if async_ok else 'FAIL'}")
    results += [refused, named, async_ok]
    results.append(check("seam_async", blur121, A))

    ok = all(results)
    print("\n" + ("LINEBUF SEAM PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
