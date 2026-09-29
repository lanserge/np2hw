# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The new compiler's graph core: what docs/one-graph.md promises, checked.

Variables unify where NumPy pairs them; indexes are expressions whose
canonical form is ONE node, so every access through an index is found among
its users; splits relate a variable to its parts; a container wraps where
NumPy wraps; the dump is deterministic and the verifier accepts what the
builder made.

Run:  python examples/core_graph.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.core import model  # noqa: F401  (registers the model kinds)
from np2hw.core.graph import Graph, GraphError, dump, users, verify
from np2hw.core.index import affine, canonical, refine


def check(label, ok):
    print(f"  {label:<62} {'PASS' if ok else 'FAIL'}")
    return ok


def refused(fn):
    try:
        fn()
        return False
    except GraphError:
        return True


def main():
    r = []
    H, W = 6, 8

    print("variables:")
    g = Graph(locations=False)
    a, b = g.vars.declare(8, "test"), g.vars.declare(8, "test")
    r.append(check("two variables of one count unify to one",
                   g.vars.unify(a, b) == g.vars.find(a) == g.vars.find(b)))
    c = g.vars.declare(5, "test")
    r.append(check("variables of different counts are refused",
                   refused(lambda: g.vars.unify(a, c))))

    print("slices share a variable, as NumPy pairs them:")
    g = Graph(locations=False)
    P, Q = g.vars.declare(H + 2, "padded rows"), g.vars.declare(W + 2, "padded cols")
    buf = g.op("model.source", axes=(P, Q), container=("u", 16), lo=0, hi=255)
    r1, r2 = g.vars.declare(H, "slice"), g.vars.declare(H, "slice")
    c1 = g.vars.declare(W, "slice")
    w = lambda n: g.op("model.const", value=n, container=("w", 0))
    var = lambda v: g.op("model.var", name=v)
    rows_a = var(r1)                                  # x[:-2]  -> r + 0
    rows_b = g.op("model.add", var(r2), w(1))         # x[1:-1] -> r + 1
    cols = g.op("model.add", var(c1), w(1))
    xa = g.op("model.access", buf, rows_a, cols)
    xb = g.op("model.access", buf, rows_b, cols)
    s = g.op("model.add", xa, xb)
    r.append(check("x[:-2] + x[1:-1]: the two row variables are one",
                   g.vars.find(r1) == g.vars.find(r2)
                   and s.type.axes == (g.vars.find(r1), g.vars.find(c1))))

    print("indexes are expressions; the canonical form is one node:")
    fa, fb = affine(g, rows_b), affine(g, g.op("model.add", w(1), var(r1)))
    r.append(check("r2 + 1 and 1 + r1 have one affine form", fa == fb))
    r.append(check("...and one canonical node",
                   canonical(g, fa) is not None
                   and canonical(g, fa).node is canonical(g, fb).node))
    idx = canonical(g, fa)
    x1 = g.op("model.access", buf, idx, cols)
    x2 = g.op("model.access", buf, idx, var(c1))
    u = users([x1, x2]).get(id(idx.node), [])
    r.append(check("both accesses through that index are among its users",
                   x1.node in u and x2.node in u))
    r.append(check("(H,) with (W,) is refused, as NumPy refuses it",
                   refused(lambda: g.op("model.mul", var(r1), var(c1)))))
    ycol = g.op("model.source", axes=(r1, None), container=("i", 64),
                lo=0, hi=H - 1)                    # coords: (H, 1)
    xrow = g.op("model.source", axes=(None, c1), container=("i", 64),
                lo=0, hi=W - 1)                    # coords: (1, W)
    prod = g.op("model.mul", ycol, xrow)
    r.append(check("(H, 1) with (1, W) broadcasts to (H, W)",
                   prod.type.axes == (g.vars.find(r1), g.vars.find(c1))))
    r.append(check("a variable times a variable is data, not an index",
                   affine(g, g.op("model.mul", var(r1), var(r1))) is None))

    print("splits:")
    v = g.vars.declare(8, "test")
    q = affine(g, g.op("model.floordiv", var(v), divisor=2))
    rem = affine(g, g.op("model.mod", var(v), divisor=2))
    whole = refine(g, affine(g, var(v)))
    r.append(check("v // 2 and v % 2 are the parts q, r of v = 2q + r",
                   whole.terms == ((q.terms[0][0], 2), (rem.terms[0][0], 1))))
    qn, rn = q.terms[0][0], rem.terms[0][0]
    qcol = g.op("model.expand", var(qn), axes=(qn, None))      # q[:, None]
    rrow = g.op("model.expand", var(rn), axes=(None, rn))      # r[None, :]
    two_q_r = g.op("model.add", g.op("model.mul", qcol, w(2)), rrow)
    r.append(check("2q + r over distinct variables is (q, r), as NumPy "
                   "broadcasts q[:, None] with r[None, :]",
                   two_q_r.type.axes == (qn, rn)))
    r.append(check("v and 2q + r are one canonical node",
                   canonical(g, affine(g, var(v))).node
                   is canonical(g, affine(g, two_q_r)).node))
    odd = g.vars.declare(7, "test")
    r.append(check("a ragged split (7 by 2) is refused",
                   refused(lambda: affine(g, g.op("model.floordiv", var(odd),
                                                  divisor=2)))))

    print("containers wrap where NumPy wraps:")
    g = Graph(locations=False)
    R, C = g.vars.declare(H, "rows"), g.vars.declare(W, "cols")
    img = g.op("model.source", axes=(R, C), container=("u", 8), lo=0, hi=255)
    k = g.op("model.const", value=128, container=("w", 0))
    wrapped = g.op("model.sub", img, k)
    signed = g.op("model.sub", g.op("model.astype", img, container=("i", 16)), k)
    r.append(check("uint8 - 128 stays uint8 and wraps: [0, 255]",
                   wrapped.type.container == ("u", 8)
                   and (wrapped.type.lo, wrapped.type.hi) == (0, 255)))
    r.append(check("int16 - 128 does not wrap: [-128, 127]",
                   (signed.type.lo, signed.type.hi) == (-128, 127)))

    print("the dump and the verifier:")
    roots = [wrapped, signed]
    r.append(check("the dump is deterministic",
                   dump(g, roots) == dump(g, roots)))
    try:
        verify(g, roots, levels={"model"})
        r.append(check("the verifier accepts what the builder made", True))
    except GraphError as e:
        r.append(check(f"the verifier accepts what the builder made ({e})",
                       False))
    print()
    print(dump(g, roots))

    print("\n" + ("ALL PASS" if all(r) else "SOME FAILED"))
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
