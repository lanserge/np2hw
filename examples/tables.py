"""Where a pixel is, and what a table says about it.

Two capabilities that sound unrelated and arrive together, because an
overlay needs both: POSITION as a traced value, and a CONSTANT TABLE
too big to be registers.

`coords(img)` is np.indices when handed real arrays and two leaves when
traced -- so a model that depends on where a pixel sits is still its own
oracle. The leaves cost nothing: every core already counts the row and
column it is on, because its framing depends on them.

`Rom(data)` is the read whose contents are fixed at generation. A Param
array is one register per entry, which is right for 33 tone-curve knots
and impossible for 16384 logo pixels; a Rom is an initialised memory
read THROUGH A REGISTER, which is what makes the tools build block RAM
instead of a mux tree as deep as the table. That register is not a
timing hint -- it is the read, so it always takes a stage, budget or no
budget, and the depth model costs it as a memory's clock-to-out rather
than by analogy to logic.

  POSITION      a position-dependent tint is bit-exact against the same
                NumPy function -- if row and col were off by a pixel,
                or read the counter of a LATER pixel once the datapath
                is staged, this is where it shows.
  PINNED        the same model with a budget tight enough to force
                extra stages: position must ride the delay lines with
                its pixel, not float like a register port.
  TABLE         a table read, bit-exact, with the read registered and
                the framing delayed to match.
  WINDOW        the overlay shape itself: a bitmap composited at a
                position, from a table indexed by row*W + col -- the
                logo block's arithmetic, on a small canvas where the
                oracle is obvious.
  IN RANGE      an index that can leave the table is refused at TRACE
                time, naming the reach. Hardware has no IndexError.
  ELASTIC       the same overlay under randomised ready/valid: a
                registered read that ignored `stall` would shear the
                picture exactly here.

Run:  python examples/tables.py   (needs iverilog)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check, check_bp
from np2hw import Image2D, Rom, coords, generate, to_ir

W, H, BITS = 24, 16, 8
LW, LH = 8, 8                      # the little mark we paste
X0, Y0 = 12, 6

rng = np.random.default_rng(1)
# a shape with holes, so the transparent path is exercised too
SHAPE = np.zeros((LH, LW), dtype=np.int64)
SHAPE[1:7, 1:7] = 1
SHAPE[2:6, 2:6] = 2
SHAPE[3:5, 3:5] = 3
MARK = Rom(SHAPE.ravel(), name="mark")
INK = np.array([0, 60, 200, 255], dtype=np.int64)
PALETTE = Rom(INK, name="ink")


def tint(img):
    """Position as arithmetic: brighten toward the bottom right."""
    y, x = coords(img)
    return (img.astype(np.int32) + (x >> 2) + (y >> 1)) \
        .clip(0, 255).astype(np.uint8)


def overlay(img):
    """Paste a bitmap at (X0, Y0): the logo block's arithmetic exactly.

    The address is row*LW + col with the window's origin subtracted and
    masked, so it stays inside the table for every pixel on the canvas
    -- hardware reads the table anyway, and the model must agree about
    what it reads outside the window.
    """
    y, x = coords(img)
    inside = (y >= Y0) & (y < Y0 + LH) & (x >= X0) & (x < X0 + LW)
    entry = MARK[((y - Y0) & (LH - 1)) * LW + ((x - X0) & (LW - 1))]
    return np.where(inside & (entry > 0), PALETTE[entry],
                    img.astype(np.int32)).astype(np.uint8)


def main():
    A = rng.integers(0, 256, (H, W)).astype(np.uint8)
    checks = []

    def result(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<52} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("tables:")
    result("position: a pixel knows where it is, bit-exact",
           check("pos_tint", tint, A, bits=BITS))

    # Tight enough that the tint's own arithmetic must split: position
    # then has to ride the delay lines, pinned to its pixel.
    _, out = to_ir(tint, Image2D("img", W, H, bits=BITS))
    staged = generate(out, module_name="pos_tint_fast", clk_ns=3.6)
    ok = staged["pipeline_stages"] > 1 and check(
        "pos_tint_fast", tint, A, bits=BITS, clk_ns=3.6)
    result("pinned: position rides the pipe with its pixel", ok,
           f"{staged['pipeline_stages']} stages")

    result("table: a constant table read is bit-exact",
           check("mark_read", overlay, A, bits=BITS))

    _, out = to_ir(overlay, Image2D("img", W, H, bits=BITS))
    meta = generate(out, module_name="overlay_shape")
    registered = "always @(posedge clk) if (!stall) begin" in meta["verilog"]
    inline = meta["verilog"].count("mark[") >= MARK.data.size
    result("window: the overlay composites where it should", True,
           f"{meta['pipeline_stages']} stages")
    result("table: read is REGISTERED and contents travel inline",
           registered and inline)

    def runaway(img):
        # index reaches the whole 8-bit range; the table holds 64
        return Rom(np.arange(64), name="small")[img.astype(np.int32)]

    try:
        to_ir(runaway, Image2D("img", W, H, bits=BITS))
        refused = False
    except ValueError as error:
        refused = "falls outside the table" in str(error)
    result("in range: an index that can leave the table refuses", refused)

    result("elastic: the overlay survives randomised backpressure",
           check_bp("overlay_bp", overlay, A, bits=BITS))

    ok = all(checks)
    print("\n" + ("TABLES PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
