"""Step 3+coeffs: generate Verilog for weighted filters, simulate, check vs NumPy.

Covers: unit-coefficient averaging blur, Const-weighted Gaussian (/16), and a
Param (register) gain. Run:  python examples/verilog_blur.py   (needs iverilog)
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.ir import ImageStreamer, Image2D, Indexer, ImageOp, Const, Param
from np2hw.verilog import generate, testbench, reference

W, H = 8, 6
BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
os.makedirs(BUILD, exist_ok=True)
RNG = np.random.default_rng(0)


# ---- pipelines -------------------------------------------------------------

def averaging_blur():
    s = ImageStreamer()
    img = Image2D("img", W, H, bits=8)
    add = ImageOp("add", Indexer(0), Indexer(1))
    rows = [s.horizontal(add, s.line(img, Indexer(k))) for k in (0, 1)]
    summed = s.vertical(add, rows)
    return s, s.horizontal(ImageOp("div", Const(4)), summed)


def gaussian_3x3():
    s = ImageStreamer()
    img = Image2D("img", W, H, bits=8)
    h = ImageOp("mac", Indexer(0), Indexer(1), Indexer(2),
                coeffs=[Const(1), Const(2), Const(1)])
    rows = [s.horizontal(h, s.line(img, Indexer(k))) for k in (0, 1, 2)]
    v = ImageOp("mac", coeffs=[Const(1), Const(2), Const(1)])
    summed = s.vertical(v, rows)
    return s, s.horizontal(ImageOp("div", Const(16)), summed)


def param_gain():
    s = ImageStreamer()
    img = Image2D("img", W, H, bits=8)
    line = s.line(img, Indexer(0))
    return s, s.horizontal(ImageOp("mul", Param("gain", bits=4)), line)


# ---- harness ---------------------------------------------------------------

def run(name, build_fn, param_values=None):
    print("\n" + "=" * 60 + f"\n{name}\n" + "=" * 60)
    param_values = param_values or {}
    s, out = build_fn()
    meta = generate(out, module_name=name)
    print(f"weighted taps = {dict(sorted(meta['weighted'].items()))}")
    print(f"post = {meta['post']}  params = {meta['params']}  "
          f"out_bits = {meta['out_bits']}")

    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(meta["verilog"] + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(testbench(meta, W, H, param_values) + "\n")

    A = RNG.integers(0, 256, size=(H, W), dtype=np.uint8)
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{p:02x}" for p in A.flatten()) + "\n")

    sim = os.path.join(BUILD, "sim.vvp")
    subprocess.run(["iverilog", "-o", sim, f"{name}.v", "tb.v"], check=True, cwd=BUILD,
                   capture_output=True)
    subprocess.run(["vvp", sim], check=True, cwd=BUILD, capture_output=True)

    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)
    ref = reference(meta["weighted"], A, meta["post"], param_values)
    expected = ref.flatten()
    ok = got.size == expected.size and np.array_equal(got, expected)
    print(f"valid region {ref.shape} = {expected.size} px  ->  "
          + ("PASS" if ok else "FAIL"))
    if not ok:
        print("DUT  :", got.tolist())
        print("NumPy:", expected.tolist())
    return ok


def main():
    results = [
        run("avgblur", averaging_blur),
        run("gaussian", gaussian_3x3),
        run("gain", param_gain, param_values={"gain": 3}),
    ]
    print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
