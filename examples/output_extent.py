"""The output extent has ONE owner, and a model the gate cannot express is
refused rather than emitted wrong.

A generated core's valid region is [M, HEIGHT) x [N, WIDTH), written in
the WIDTH and HEIGHT parameters so one synthesised core serves any
resolution. That is the right region exactly when the model's output is
the input minus the kernel's margin -- which is what a stencil produces,
and why the relation held for as long as every model was a stencil.

A model that crops FURTHER has a smaller extent than that. The tracer
knew: `img[0:4, 0:4]` on an 8x8 reported a 4x4 shape. The emitter did
not: it gated on HEIGHT and WIDTH and produced all 64 beats. Two owners
of one fact, disagreeing in silence -- and silence is the expensive part,
because a core that emits four times the pixels its model describes is
discovered downstream, as a picture that is wrong for no visible reason.

A fixed crop is also not resolution-agnostic, so it cannot be written in
those parameters at all: it needs an output raster of its own, which this
emitter does not have. Until it does, the honest answer is to refuse.

Run:  python examples/output_extent.py   (no simulator needed)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from np2hw.frontend import to_ir
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


# The refusal: a crop the region cannot describe.
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
