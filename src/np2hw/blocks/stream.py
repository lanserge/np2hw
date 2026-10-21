# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The video stream interface, and its AXI4-Stream Video adapter.

Blocks pass pixels as streams -- a block declares its stream ports,
and a sink, statistics say, consumes one and produces none. The interface is
deliberately small: pixel data, two framing flags, and a ready/valid
handshake at every block boundary.

    clk, rst
    valid   -> source has a pixel this cycle
    ready   <- sink can accept one this cycle
    data    -> `channels` components of `bit_depth` bits, channel 0 in the low bits
    sof     -> this pixel is the first of a frame
    eol     -> this pixel is the last of its line
    last    -> this pixel is the last of the frame (sof of the next frame follows)

A transfer happens on a cycle where ``valid && ready``. ``sof``, ``eol`` and
``last`` are qualified by ``valid`` and ride alongside the pixel they describe --
they are not separate pulses to be counted against a pixel stream.

Why these flags and not a coordinate bus
----------------------------------------

Framing that travels with the data is self-describing: a block never needs to
know the frame size to know where it is, so one bitstream serves any resolution
and the pipeline context registers (width, height) are needed only where the
geometry genuinely matters -- line buffers and windowing. This is what lets the
same generated core process any line length up to its buffer size.

``last`` is redundant with ``eol`` plus a line count, and ``sof`` is redundant
with the previous ``last``. Both are carried anyway because recovering them costs
every consumer a counter, and because they are what AXI4-Stream Video's TUSER and
TLAST mean -- carrying them makes the adapter a rename rather than logic.

Relationship to np2hw
---------------------

The cores np2hw generates already present exactly this interface
(``in_valid/in_ready/in_sof/in_data`` -> ``out_valid/out_ready/out_sof/out_eol/
out_last/out_data``). This module names it, parametrises it on bit depth and
channel count, and gives the testbenches a way to turn NumPy arrays into beats
and back. It does not re-implement it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

# Framing flags carried alongside every pixel, in the order they appear in the
# generated port list and in the testbench beat tuple.
FLAGS = ("sof", "eol", "last")


@dataclass(frozen=True)
class StreamSpec:
    """The parametrisation of one stream: how wide a pixel is, and how many.

    Args:
        bit_depth: bits per component. 10 or 12 for raw sensor data, 8 after
            gamma. This is the datapath width, and it is one of only two things
            taken from a sensor description at build time.
        channels: components per pixel. 1 in the Bayer domain (one colour per
            pixel, its identity given by position and the Bayer phase register),
            3 after demosaic.
        name: prefix for the generated signal names.
        signed: two's complement components. False everywhere in the pixel
            datapath; available because intermediate difference streams
            (sharpening, chroma) are signed.
        frames: frames the stream brings side by side -- each beat the
            pixel of the newest `frames` frames, the frame k before the
            newest in the k-th group of `channels` components -- for a
            block that reads the stream's history; 0 for one frame.
    """

    bit_depth: int
    channels: int = 1
    name: str = "video"
    signed: bool = False
    frames: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.bit_depth <= 32:
            raise ValueError(f"bit_depth {self.bit_depth} outside 1..32")
        if self.channels < 1:
            raise ValueError(f"channels {self.channels} must be >= 1")
        if self.frames < 0:
            raise ValueError(f"frames {self.frames} must be >= 0")

    # -- widths ---------------------------------------------------------------- #

    @property
    def data_bits(self) -> int:
        """Width of the packed ``data`` bus."""
        return self.bit_depth * self.channels * max(1, self.frames)

    @property
    def max_value(self) -> int:
        """Largest value one component can carry."""
        return (1 << (self.bit_depth - 1)) - 1 if self.signed else (1 << self.bit_depth) - 1

    @property
    def min_value(self) -> int:
        return -(1 << (self.bit_depth - 1)) if self.signed else 0

    @property
    def dtype(self) -> np.dtype:
        """Smallest NumPy integer dtype holding one component."""
        width = next(w for w in (8, 16, 32, 64) if self.bit_depth <= w)
        return np.dtype(f"{'int' if self.signed else 'uint'}{width}")

    def with_depth(self, bit_depth: int) -> "StreamSpec":
        """This stream at a different component width (a block that grows bits)."""
        return StreamSpec(bit_depth=bit_depth, channels=self.channels,
                          name=self.name, signed=self.signed,
                          frames=self.frames)

    def with_stream(self, channels: int, bit_depth: int,
                    frames: int = 0) -> "StreamSpec":
        """This stream as some block's model actually packed it.

        Both numbers are consequences of arithmetic, not declarations: a
        demosaic packs three components where it read one, and a stage
        that widens rather than saturating packs a taller one. The
        composer reads them off the driver's trace and hands them to the
        consumer, so a block is told the stream it is really given.
        """
        return StreamSpec(bit_depth=bit_depth, channels=channels,
                          name=self.name, signed=self.signed, frames=frames)

    # -- packing --------------------------------------------------------------- #

    def pack(self, components: Iterable[int]) -> int:
        """Pack per-channel values into one ``data`` word, channel 0 in the low bits."""
        comps = list(components)
        if len(comps) != self.channels:
            raise ValueError(f"expected {self.channels} components, got {len(comps)}")
        mask = (1 << self.bit_depth) - 1
        word = 0
        for i, value in enumerate(comps):
            word |= (int(value) & mask) << (i * self.bit_depth)
        return word

    def unpack(self, word: int) -> list[int]:
        """Inverse of :meth:`pack`, sign-extending each component if signed."""
        mask = (1 << self.bit_depth) - 1
        out = []
        for i in range(self.channels):
            value = (int(word) >> (i * self.bit_depth)) & mask
            if self.signed and value >> (self.bit_depth - 1):
                value -= 1 << self.bit_depth
            out.append(value)
        return out

    # -- port declarations ------------------------------------------------------ #

    def ports(self, direction: str, prefix: str | None = None) -> list[tuple[str, str, int]]:
        """Port list as ``(name, direction, width)`` triples.

        ``direction`` is the direction of the DATA: ``"in"`` for a sink port
        (data and framing are inputs, ready is an output), ``"out"`` for a source.
        """
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")
        p = prefix if prefix is not None else direction
        fwd, back = ("input", "output") if direction == "in" else ("output", "input")
        return [
            (f"{p}_valid", fwd, 1),
            (f"{p}_ready", back, 1),
            (f"{p}_data", fwd, self.data_bits),
            *[(f"{p}_{flag}", fwd, 1) for flag in FLAGS],
        ]

    def verilog_ports(self, direction: str, prefix: str | None = None,
                      indent: str = "    ") -> list[str]:
        """The same port list rendered as Verilog declarations, with comments."""
        why = {
            "valid": "source presents a pixel",
            "ready": "sink can accept a pixel",
            "data": f"{self.channels} x {self.bit_depth}b component(s), channel 0 in the low bits",
            "sof": "first pixel of the frame",
            "eol": "last pixel of this line",
            "last": "last pixel of the frame",
        }
        lines = []
        for name, io, width in self.ports(direction, prefix):
            rng = "" if width == 1 else f"[{width - 1}:0] "
            kind = f"{io:<6} wire {rng}"
            lines.append(f"{indent}{kind}{name},".ljust(56)
                         + f"// {why[name.rsplit('_', 1)[-1]]}")
        return lines


# --------------------------------------------------------------------------- #
# Beats -- the testbench view of a stream
# --------------------------------------------------------------------------- #
#
# The beat model and the framing rules are np2hw's: it defines the handshake,
# so it ships the Python model of it (np2hw.testing). What this adds here is
# only the CHANNEL layer -- packing an (h, w, c) pixel frame into the stream's
# data words and back -- because how components share a word is the
# application's convention, and the framing is not.

from np2hw.testing import Beat, check_framing  # noqa: F401  (re-exported)
from np2hw import testing as _np2hw_testing


def frame_to_beats(frame: np.ndarray, spec: StreamSpec) -> list[Beat]:
    """Raster-scan a frame into beats, packing channels per ``spec``.

    ``frame`` is ``(height, width)`` for a single-channel stream or
    ``(height, width, channels)`` otherwise. This is how a testbench turns the
    NumPy model's input into something to drive at the DUT.
    """
    array = np.asarray(frame)
    if spec.channels == 1:
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D frame for a 1-channel stream, got {array.shape}")
        array = array[:, :, None]
    elif array.ndim != 3 or array.shape[2] != spec.channels:
        raise ValueError(
            f"expected (h, w, {spec.channels}) for a {spec.channels}-channel stream, "
            f"got {array.shape}")
    words = [[spec.pack(array[y, x].tolist()) for x in range(array.shape[1])]
             for y in range(array.shape[0])]
    return _np2hw_testing.frame_to_beats(words)


def beats_to_frame(beats: Iterable[Beat], spec: StreamSpec,
                   width: int | None = None) -> np.ndarray:
    """Reassemble beats into a frame, taking the geometry from ``eol``/``last``.

    The width is recovered from the framing rather than passed in, which is the
    point of carrying it: if the DUT's idea of a line differs from the model's,
    this raises instead of silently reshaping into a plausible-looking image.
    """
    rows = _np2hw_testing.beats_to_words(beats)
    if width is not None and len(rows[0]) != width:
        raise ValueError(f"expected {width}-pixel lines, stream framed {len(rows[0])}")
    array = np.array([[spec.unpack(word) for word in row] for row in rows],
                     dtype=spec.dtype)
    return array[:, :, 0] if spec.channels == 1 else array


# --------------------------------------------------------------------------- #
# AXI4-Stream Video adapter
# --------------------------------------------------------------------------- #

def byte_align(bits: int) -> int:
    """Round a width up to a whole number of bytes, as AXI4-Stream TDATA requires."""
    return ((bits + 7) // 8) * 8


@dataclass(frozen=True)
class AxiStreamVideo:
    """The mapping between the block stream and AXI4-Stream Video.

    The AXI4-Stream Video protocol (as used by Xilinx VDMA and the Video IP
    suite) carries framing in exactly the two places this stream does, so the adapter
    is a rename plus a byte-alignment of the data bus -- there is no state
    machine, and nothing to get wrong at the boundary:

        TVALID   = valid
        TREADY   = ready
        TDATA    = data, zero-extended to a byte multiple
        TUSER[0] = sof      (Start Of Frame, per the video protocol)
        TLAST    = eol      (End Of Line -- NOT end of frame; the frame ends at
                             the TLAST that coincides with the next TUSER)

    The one thing worth stating plainly, because it is the usual source of bugs
    at this boundary: AXI4-Stream Video's TLAST is END OF LINE, while plain
    AXI4-Stream uses TLAST for end of packet. The stream's ``last`` (end of frame)
    has no AXI4-Stream Video equivalent and is dropped by the adapter; a
    downstream consumer recovers it from the next TUSER.
    """

    spec: StreamSpec

    @property
    def tdata_bits(self) -> int:
        return byte_align(self.spec.data_bits)

    @property
    def tkeep_bits(self) -> int:
        return self.tdata_bits // 8

    def signal_map(self, direction: str) -> dict[str, str]:
        """AXI signal -> stream signal, for one side of the adapter.

        Taken from np2hw, which WRITES the adapter this mapping describes -- a
        copy kept here would be a copy free to disagree with it.
        """
        from np2hw.testing import axis_video_map

        return axis_video_map(direction)

    def wrap(self, core: dict, width: int, height: int, module_name: str | None = None) -> dict:
        """Wrap an np2hw core in the AXI4-Stream Video adapter.

        Delegates to np2hw, which already emits this adapter; this supplies the
        naming and the geometry. Kept here so blocks have one place to ask for a
        bus-attached version of themselves.
        """
        from np2hw.verilog import axis_video_wrap

        return axis_video_wrap(core, width, height, module_name=module_name)
