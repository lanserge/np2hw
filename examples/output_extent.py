"""A crop states the map from output position to input position, and the
core is built from that map.

A generated core's valid region is [M, HEIGHT) x [N, WIDTH), written in
the WIDTH and HEIGHT parameters so one synthesised core serves any
resolution. That is the right region exactly when the model's output is
the input minus the kernel's margin -- which is what a stencil produces,
and why the relation held for as long as every model was a stencil.

A model that crops FURTHER has a smaller extent than that, and a shape
alone cannot say where the window sits: `img[0:4, 0:4]` on an 8x8 yielded
a 4x4 and threw the relationship away, so the emitter fell back on the
input's own raster and produced all 64 beats -- two owners of one fact,
disagreeing in silence.

`make_coords` closes it by stating the OUTPUT raster, so indexing the
input with those coordinates states the map:

    y2, x2 = make_coords(img, 4, 4)
    return img[y2 + 2, x2 + 2]        # output (r,c) reads input (r+2,c+2)

The map is what a core needs. The offset becomes the tap, the declared
extent becomes the gate, and the framing marks the window's corners
rather than the frame's. A shape can be inferred wrongly; a map cannot.

Run:  python examples/output_extent.py   (needs iverilog)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import check
from np2hw.frontend import make_coords, to_ir
from np2hw.ir import Image2D
from np2hw.verilog import generate

W, H = 8, 8


def build(fn):
    ir = to_ir(fn, Image2D("img", W, H, 8, False))
    return ir[-1] if isinstance(ir, tuple) else ir


def emits(fn, name):
    result = generate(build(fn), module_name=name)
    verilog = result["verilog"] if isinstance(result, dict) else result.verilog
    assert "endmodule" in verilog, name
    return verilog


# The crop, checked bit-exact against the same function on real data.
A = np.random.default_rng(20260822).integers(0, 256, (H, W), dtype=np.uint8)


def crop(img):
    y2, x2 = make_coords(img, 4, 4)
    return img[y2 + 2, x2 + 2]


assert crop(A).shape == (4, 4) and crop(A)[0, 0] == A[2, 2]
check("crop_declared", crop, A)
print("a declared crop generates and matches its model, pixel for pixel")

# The refusal: a shape with no map behind it.
try:
    emits(lambda img: img[0:4, 0:4], "crop")
except ValueError as error:
    message = str(error)
    assert "4x4" in message and "8x8" in message, message
    print("refused, and said why:")
    print("   ", message.split(" -- ")[0])
else:
    raise SystemExit("a crop was emitted as a whole frame -- the extent "
                     "has two owners again")

# A lone slice is the same defect wearing a different face -- it drops two
# rows and the region would still emit all eight.
try:
    emits(lambda img: img[0:-2, 0:-2], "lone_slice")
except ValueError as error:
    assert "6x6" in str(error), str(error)
    print("a lone slice is refused too:", str(error).split(" -- ")[0])
else:
    raise SystemExit("a lone slice was emitted whole")

# ...and everything the region CAN describe still builds.
emits(lambda img: img[:, :] // 2, "whole")           # no margin at all
emits(lambda img: (img[0:-2, 0:-2] + img[2:, 2:]) // 2, "stencil")
print("whole frame and stencil still generate")
