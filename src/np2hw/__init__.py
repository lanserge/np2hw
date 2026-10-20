# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""np2hw — NumPy → streaming hardware (Verilog) compiler for image and 2-D pipelines.

Public API for writing model files and driving the compiler:
    from np2hw import Param, Image2D, to_ir, generate
"""
from .ir import Param, Params, Image2D, PhaseRef, Rom
from .core.pipeline import to_ir
from .core.trace import coords, make_coords, saturate
from .verilog import axil_regfile, control_wrap, generate
from .regmap import AddrMap, Reg, RegBlock, RegInstance, systemrdl
from .compose import Connection, Instance, Port, StreamType, compose
from .netlist import Endpoint, Netlist, Node
from .stream import (STREAM_SIGNALS, boundary_report, cdc_fifo,
                     combinational_outputs, skid, tee)
from . import ipxact
from .i2c import ddc_slave
from .video_mem import framebuffer_read
from .frame_history import frame_history

__all__ = [
    "Param", "Params", "Image2D", "PhaseRef", "Rom",
    "to_ir", "generate", "coords", "make_coords", "saturate",
    # Structural composition: instantiate generated cores and wire them.
    "compose", "Instance", "Connection", "Port", "StreamType",
    # The netlist as a validated structure, before emission: the graph checks
    # (undriven input, double driver, cycle, unbuffered fork) live with the
    # handshake they are laws of.
    "Netlist", "Node", "Endpoint",
    # Stream plumbing. A core's timing claim is about its own registers
    # and holds only while nothing else is on the path; a combinational
    # output hands its depth to every consumer, so two cores that each
    # meet timing alone can miss it together. `skid` is how a boundary
    # stops being a place where that happens, and the other two are how
    # a composition says where it still does.
    "skid", "combinational_outputs", "boundary_report",
    # The one crossing every design this generator feeds contains:
    # a recovered clock meeting the board's. Owned, so it can follow
    # the design to another family or an ASIC, and so its discipline
    # is checked rather than trusted.
    "cdc_fifo",
    # One stream, two switchable consumers, off means DISCARD: what
    # makes a preview branch and a store branch legal at once, and
    # what makes standby "outputs off" instead of "pipeline stalled".
    "tee",
    # The display cable as the control plane: EDID says who the sink
    # is, a second address on the same DDC wires reads and writes the
    # register file -- refusal (SLVERR) becomes NACK, and no CPU or
    # network is needed on the board.
    "ddc_slave",
    # Receivers for external links: parallel video in, elastic stream out.
    # Memory: a frame of DDR fetched back at a raster's pace, by an
    # engine that owns its addressing and therefore its phase.
    "framebuffer_read",
    # A stream's recent past delivered with its present: channel k of each
    # pixel is the same pixel k frames ago -- the time axis a traced model
    # reads as pixel[..., k] and a trace cannot hold itself.
    "frame_history",
    # Control plane: a register map the caller allocated -- block types,
    # instances at bases -- with the regfile emitters and renderings over it.
    "Reg", "RegBlock", "RegInstance", "AddrMap", "systemrdl",
    "axil_regfile", "control_wrap",
    "ipxact", "STREAM_SIGNALS",
]
