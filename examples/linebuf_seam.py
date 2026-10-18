# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The memory seam, and the promise it keeps.

A line buffer is `reg [W-1:0] mem [0:D-1]`. Inferred on an FPGA that is
block RAM; on silicon it is D*W flip-flops, and one 1920-wide 12-bit
line is 23,040 of them. So every window reads its line buffers through
a register, a column ahead, and puts the array behind a module of its
own -- `<module>_linebuf`, one write port, one read port, both
synchronous -- which an integrator binds a memory macro to by name. The
contract the binding must honour is written on the module
(docs/memories.md). `sync_memory=True` is the promise a design bound for
silicon makes -- no memory reads combinationally -- and every window
keeps it, padded or not.

Three things are proven here, each as something that can fail:

  1. a padded model, generated under sync_memory=True, is bit-exact and
     its line buffers are seam instances (the shape, asserted);
  2. the same padded model generates IDENTICALLY with and without the
     flag -- the flag never changes what is emitted, only what is refused;
  3. an unpadded model, under the flag, is a seam too, and bit-exact.

Run:  python examples/linebuf_seam.py   (needs iverilog)
"""
import os
import re
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
    """The same filter unpadded: shrinks by two rows."""
    x = img.astype(np.uint16)
    return ((x[:-2, :] + 2 * x[1:-1, :] + x[2:, :]) // 4).astype(np.uint8)


def _seam(core, name):
    """Two line buffers, each an instance of the seam, its read
    registered -- and no array read anywhere else."""
    src = "\n".join(line for line in core["verilog"].splitlines()
                    if not line.lstrip().startswith("//"))
    return (f"module {name}_linebuf #(" in src
            and len(re.findall(rf"^\s+{name}_linebuf #\(", src, re.M)) == 2
            and "if (re) rd <= mem[ra];" in src
            and src.count(" mem[") == 2)


def main():
    results = []
    print("the memory seam:")

    # 1. behind the seam, under the flag, bit-exact
    results.append(check("seam_edge", blur121_edge, A, sync_memory=True))
    _, out = to_ir(blur121_edge, Image2D("img", W, H, bits=8))
    core = generate(out, module_name="seam_edge", sync_memory=True)
    src = core["verilog"]
    seam = _seam(core, "seam_edge")
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

    # 3. an unpadded window keeps the promise too
    _, out_unpadded = to_ir(blur121, Image2D("img", W, H, bits=8))
    unpadded = generate(out_unpadded, module_name="seam_crop",
                        sync_memory=True)
    kept = (_seam(unpadded, "seam_crop")
            and [m["read"] for m in unpadded["memories"]] == ["sync"] * 2)
    print(f"  an unpadded window is a seam under the flag too   "
          f"{'PASS' if kept else 'FAIL'}")
    results.append(kept)
    results.append(check("seam_crop", blur121, A, sync_memory=True))

    ok = all(results)
    print("\n" + ("LINEBUF SEAM PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
