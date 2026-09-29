# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Canonicalising: one form for each thing, and every rewrite exact.

Every model of core_trace.py is traced, canonicalised, and the canonical
graph evaluated -- exact arithmetic, explicit wraps, bit access, canonical
indexes -- must still EQUAL the model. Then the forms themselves: equal
indexes are one node; a wrap stands exactly where NumPy wraps and nowhere
else; masks and shifts of data are bit access.

Run:  python examples/core_canonical.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw import Image2D, Param
from np2hw.ir import PhaseRef
from np2hw.core.canonical import canonicalise
from np2hw.core.evaluate import evaluate
from np2hw.core.graph import dump, users, verify, walk
from np2hw.core.trace import trace
from core_trace import (MODELS, H, W, RNG, GAIN, PY, PX, blur, gain,
                        centre_wraps, mask_pack, black_canvas, black_coords)


def check(label, ok):
    print(f"  {label:<60} {'PASS' if ok else 'FAIL'}")
    return ok


def kinds(c):
    out = {}
    for n in walk([c.output]):
        out[n.kind] = out.get(n.kind, 0) + 1
    return out


def canon(fn, *params):
    return canonicalise(trace(fn, Image2D("img", W, H, bits=8), *params))


def main():
    r = []
    print("canonical graphs, evaluated, against the models:")
    for name, fn, params, values, *ch in MODELS:
        ch = ch[0] if ch else 1
        shape = (H, W) if ch == 1 else (H, W, ch)
        img = RNG.integers(0, 256, shape, dtype=np.uint8)
        t = trace(fn, Image2D("img", W, H, bits=8 * ch), *params, channels=ch)
        c = canonicalise(t)
        verify(c.g, [c.output], levels={"model"})
        (got,) = evaluate(c.g, [c.output], c.source, img, values)
        want = fn(img, *[values[p.name] for p in params])
        before = sum(1 for _ in walk([t.output]))
        after = sum(1 for _ in walk([c.output]))
        r.append(check(f"{name:<14} {before:>3} -> {after:>3} nodes",
                       np.array_equal(got, want)))

    print("the forms:")
    c = canon(blur)
    accesses = [n for n in walk([c.output]) if n.kind == "model.access"]
    rows = {id(n.operands[1].node) for n in accesses}
    cols = {id(n.operands[2].node) for n in accesses}
    r.append(check("the 3x3 stencil's nine accesses use 3 row and 3 col indexes",
                   len(accesses) == 9 and len(rows) == 3 and len(cols) == 3))
    u = users([c.output])
    idx = accesses[0].operands[1].node
    r.append(check("...and each index's users are its three accesses",
                   sum(1 for m in u[id(idx)] if m.kind == "model.access") == 3))
    k = kinds(canon(gain, GAIN))
    r.append(check("a gain whose values fit its containers has no wrap",
                   "model.wrap" not in k and "model.astype" not in k))
    r.append(check("uint8 - 128 has exactly one wrap, where NumPy wraps",
                   kinds(canon(centre_wraps)).get("model.wrap") == 1))
    k = kinds(canon(mask_pack))
    r.append(check("masks and shifts of data are bit access, not AND/shift",
                   k.get("model.bits", 0) >= 2 and "model.and" not in k
                   and "model.shr" not in k))
    print("selects by remainders:")
    a = canon(black_canvas, PY, PX)
    b = canon(black_coords, PY, PX)
    body = lambda c: [l for l in dump(c.g, [c.output]).splitlines()
                      if not l.startswith("var ")]
    r.append(check("the CFA stage as a canvas and as coords arithmetic: "
                   "one graph", body(a) == body(b)))
    ka = kinds(a)
    r.append(check("...one select over the remainders, one datapath",
                   ka.get("model.select") == 1 and ka.get("model.add") == 3
                   and ka.get("model.clip") == 1
                   and "model.canvas" not in ka and "model.where" not in ka))
    OFF = Param("offset", np.int16, shape=(2, 2))

    def black_regs(img, py, px, off):
        """The CFA stage with REGISTER offsets: hash-consing orders each
        plane's `pixel + offset` by structure, differently per register."""
        v = img.astype(np.int32)
        out = np.empty_like(v)
        for i, inv_r in enumerate((False, True)):
            for j, inv_c in enumerate((False, True)):
                rows = PhaseRef(py, invert=inv_r)
                cols = PhaseRef(px, invert=inv_c)
                out[rows::2, cols::2] = (v[rows::2, cols::2] + off[i, j]).clip(0, 4095)
        return out.astype(np.uint16)

    kr = kinds(canon(black_regs, PY, PX, OFF))
    r.append(check("with register offsets too: one select, one datapath",
                   kr.get("model.select") == 1 and kr.get("model.clip") == 1))
    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
