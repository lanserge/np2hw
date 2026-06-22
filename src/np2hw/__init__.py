"""np2hw — NumPy → streaming ISP hardware (Verilog) compiler.

Public API for writing model files and driving the compiler:
    from np2hw import Param, Image2D, to_ir, generate
"""
from .ir import Param, Params, Const, Image2D, Indexer
from .frontend import to_ir
from .verilog import generate

__all__ = [
    "Param", "Params", "Const", "Image2D", "Indexer",
    "to_ir", "generate",
]
