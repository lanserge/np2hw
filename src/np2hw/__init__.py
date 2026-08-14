"""np2hw — NumPy → streaming hardware (Verilog) compiler for image and 2-D pipelines.

Public API for writing model files and driving the compiler:
    from np2hw import Param, Image2D, to_ir, generate
"""
from .ir import Param, Params, Const, Image2D, Indexer, PhaseRef, Rom
from .frontend import coords, to_ir
from .verilog import axil_regfile, control_wrap, generate
from .regmap import AddrMap, Reg, RegBlock, RegInstance, systemrdl
from .compose import Connection, Instance, Port, StreamType, compose
from .netlist import Endpoint, Netlist, Node
from .video_in import bayerlink_in
from .video_out import MODES, mode_timing, scanout

__all__ = [
    "Param", "Params", "Const", "Image2D", "Indexer", "PhaseRef",
    "Rom",
    "to_ir", "generate", "coords",
    # Structural composition: instantiate generated cores and wire them.
    "compose", "Instance", "Connection", "Port", "StreamType",
    # The netlist as a validated structure, before emission: the graph checks
    # (undriven input, double driver, cycle, unbuffered fork) live with the
    # handshake they are laws of.
    "Netlist", "Node", "Endpoint",
    # Receivers for external links: parallel video in, elastic stream out.
    "bayerlink_in",
    # Transmitters: an elastic stream becomes a timed raster. The mode
    # table is the one owner of raster geometry -- read it, do not copy.
    "scanout", "mode_timing", "MODES",
    # Control plane: a register map the caller allocated -- block types,
    # instances at bases -- with the regfile emitters and renderings over it.
    "Reg", "RegBlock", "RegInstance", "AddrMap", "systemrdl",
    "axil_regfile", "control_wrap",
]
