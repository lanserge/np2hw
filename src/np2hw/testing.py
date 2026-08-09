"""Verification models for np2hw's own contracts.

np2hw defines the streaming handshake -- `valid/ready/data` with `sof/eol/last`
framing, a transfer where valid && ready -- and it emits the AXI4-Lite register
file with its exact response timing. Until now the Python-side models of both
lived in the applications: every project built on np2hw re-derived "what beats
does a frame become" and re-learned the register file's handshake in its own
testbench, and the same driver loop got written once per testbench. A contract's
owner ships its model, so they live here.

Two layers, deliberately separable:

  PURE PYTHON   :class:`Beat`, :func:`frame_to_beats`, :func:`beats_to_words`,
                :func:`check_framing`. No simulator, no cocotb -- usable from
                any test, including one that only checks a model.

  COCOTB BFMS   :func:`reset_stream`, :func:`run_frame`,
                :class:`AxiLiteMaster`. Imported lazily, so this module works
                without cocotb installed until a BFM is actually used.

The framing rules, in one place because this file is now their reference:
`sof` marks the first beat of a frame, `eol` the last beat of every line,
`last` the final beat of the frame. A generated core consumes only `sof` (it
re-frames from its own geometry) and regenerates all three on its output; a
composed design's wrapper may consume all three, because its commit pulse is
derived from `last`.
"""
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# The stream, as data
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Beat:
    """One transfer: a data word and its framing flags."""

    data: int
    sof: bool = False
    eol: bool = False
    last: bool = False

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.data, int(self.sof), int(self.eol), int(self.last))


def frame_to_beats(words) -> list[Beat]:
    """Raster-scan a 2-D array of data words into beats with framing.

    ``words`` is ``(height, width)`` of already-packed integers -- how
    multi-component pixels pack into a word is the application's business; what
    the framing flags mean on the resulting stream is np2hw's, and is encoded
    here exactly once.
    """
    rows = [[int(value) for value in row] for row in words]
    if not rows or not rows[0]:
        raise ValueError("an empty frame has no framing")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged input: every row must have the same width")

    height = len(rows)
    beats = []
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            beats.append(Beat(
                data=value,
                sof=(y == 0 and x == 0),
                eol=(x == width - 1),
                last=(y == height - 1 and x == width - 1),
            ))
    return beats


def beats_to_words(beats) -> list[list[int]]:
    """Reassemble beats into rows of words, taking geometry from the FRAMING.

    The width is recovered from ``eol`` rather than passed in, which is the
    point of carrying it: if a DUT's idea of a line differs from the model's,
    this raises instead of silently reshaping into a plausible-looking frame.
    """
    beats = list(beats)
    if not beats:
        raise ValueError("no beats to reassemble")

    rows: list[list[int]] = []
    current: list[int] = []
    for beat in beats:
        current.append(int(beat.data))
        if beat.eol:
            rows.append(current)
            current = []
    if current:
        raise ValueError(
            f"stream ended mid-line: {len(current)} word(s) after the last eol")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged frame: line widths {sorted(widths)}")
    return rows


def check_framing(beats, width: int, height: int) -> None:
    """Assert the flags describe exactly a ``width`` x ``height`` frame.

    Framing is part of what has to be exact: a block that produces the right
    pixels with the wrong ``eol`` breaks every block downstream, so testbenches
    check the flags alongside the data.
    """
    beats = list(beats)
    expected = width * height
    if len(beats) != expected:
        raise AssertionError(f"expected {expected} beats, got {len(beats)}")
    for i, beat in enumerate(beats):
        want = (i == 0, (i + 1) % width == 0, i == expected - 1)
        got = (bool(beat.sof), bool(beat.eol), bool(beat.last))
        if got != want:
            raise AssertionError(
                f"beat {i} (row {i // width}, col {i % width}): framing "
                f"sof/eol/last = {int(got[0])}/{int(got[1])}/{int(got[2])}, "
                f"expected {int(want[0])}/{int(want[1])}/{int(want[2])}")


def axis_video_map(direction: str, prefix: str | None = None) -> dict[str, str]:
    """AXI4-Stream Video signal -> stream signal, for one side of the adapter.

    Owned here because :func:`np2hw.verilog.axis_video_wrap` WRITES this
    mapping; a copy kept by an application is a copy that can disagree with the
    adapter it describes. The one thing worth stating plainly, because it is
    the usual source of bugs at this boundary: AXI4-Stream Video's TLAST is END
    OF LINE, not end of packet, and `last` (end of frame) has no equivalent --
    the adapter drops it and a consumer recovers it from the next TUSER.
    """
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    p = prefix or ("s_axis" if direction == "in" else "m_axis")
    q = direction
    return {
        f"{p}_tvalid": f"{q}_valid",
        f"{p}_tready": f"{q}_ready",
        f"{p}_tdata": f"{q}_data",
        f"{p}_tuser": f"{q}_sof",
        f"{p}_tlast": f"{q}_eol",
    }


# --------------------------------------------------------------------------- #
# cocotb bus-functional models
# --------------------------------------------------------------------------- #

async def reset_stream(dut, cycles: int = 4, flags=("sof", "eol", "last")):
    """Hold `rst`, quiesce the stream inputs, release, and settle one edge.

    Only flags the DUT actually has are driven, so the same reset serves a bare
    core (which takes only `sof`) and a wrapped design (which takes all three).
    """
    from cocotb.triggers import RisingEdge

    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.in_data.value = 0
    for flag in flags:
        signal = getattr(dut, f"in_{flag}", None)
        if signal is not None:
            signal.value = 0
    dut.out_ready.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def run_frame(dut, beats, expected_count: int, rnd,
                    offer: float = 0.75, accept: float = 0.7,
                    drive=("sof",), budget_per_beat: int = 40) -> list[Beat]:
    """Drive one frame and collect the output, both under random backpressure.

    The source withholds `valid` and the sink withholds `ready` at random --
    a block that only works when the sink never stalls is not finished, and the
    elastic handshake is what lets blocks compose at all. Source and sink run
    in one loop so the ordering within a cycle is explicit: drive at the top,
    sample in ReadOnly once combinational logic settles, advance on the edge.

    Args:
        drive: which framing flags to PRESENT at the input. A generated core
            self-frames and takes only `sof`; a control-wrapped design also
            needs `eol`/`last`, because its commit pulse derives from `last`.
        rnd: a seeded ``random.Random`` -- the caller owns reproducibility.

    Returns:
        The collected output beats, framing flags included. Compare them with
        :func:`check_framing` and the model's pixels, not just the data.
    """
    from cocotb.triggers import ReadOnly, RisingEdge

    collected: list[Beat] = []
    index = 0
    # Generous bound: every beat needs at least a cycle, and backpressure adds
    # bubbles. A hang becomes a clear failure rather than a stalled CI job.
    budget = 200 + budget_per_beat * (len(beats) + expected_count)

    for _ in range(budget):
        offering = index < len(beats) and rnd.random() < offer
        dut.in_valid.value = int(offering)
        beat = beats[index] if offering else Beat(0)
        if offering:
            dut.in_data.value = beat.data
        for flag in drive:
            getattr(dut, f"in_{flag}").value = int(getattr(beat, flag)) if offering else 0

        accepting = rnd.random() < accept
        dut.out_ready.value = int(accepting)

        await ReadOnly()
        consumed = offering and int(dut.in_ready.value) == 1
        if accepting and int(dut.out_valid.value) == 1:
            collected.append(Beat(
                data=int(dut.out_data.value),
                sof=bool(int(dut.out_sof.value)),
                eol=bool(int(dut.out_eol.value)),
                last=bool(int(dut.out_last.value)),
            ))

        await RisingEdge(dut.clk)
        if consumed:
            index += 1
        if len(collected) >= expected_count:
            break

    dut.in_valid.value = 0
    for flag in drive:
        getattr(dut, f"in_{flag}").value = 0
    dut.out_ready.value = 0
    return collected


OKAY = 0b00
SLVERR = 0b10


class AxiLiteMaster:
    """A software-shaped master for the AXI4-Lite register file np2hw emits.

    The handshake details live HERE, next to the RTL whose timing they encode:
    address and data are presented together and held until the response,
    because the slave registers `awready` for one cycle and samples the write
    on the cycle after -- dropping the valids early loses the transfer. That
    is exactly the mistake a hand-written testbench makes once per project.

    Responses are returned, not asserted: refusing a write (SLVERR on a
    read-only or unmapped word) is behaviour under test, not an error.
    """

    def __init__(self, dut, prefix: str = "s_axil", limit: int = 64):
        self.dut = dut
        self.prefix = prefix
        self.limit = limit

    def _signal(self, name: str):
        return getattr(self.dut, f"{self.prefix}_{name}")

    async def idle(self) -> None:
        for name in ("awvalid", "wvalid", "bready", "arvalid", "rready"):
            self._signal(name).value = 0
        self._signal("wstrb").value = 0xF

    async def write(self, address: int, value: int) -> int:
        """Write one word. Returns the response code (OKAY / SLVERR)."""
        from cocotb.triggers import ReadOnly, RisingEdge

        clk = self.dut.clk
        self._signal("awaddr").value = address
        self._signal("awvalid").value = 1
        self._signal("wdata").value = int(value) & 0xFFFFFFFF
        self._signal("wstrb").value = 0xF
        self._signal("wvalid").value = 1
        self._signal("bready").value = 1

        response = None
        for _ in range(self.limit):
            await RisingEdge(clk)
            await ReadOnly()
            if int(self._signal("bvalid").value):
                response = int(self._signal("bresp").value)
                break
        await RisingEdge(clk)
        self._signal("awvalid").value = 0
        self._signal("wvalid").value = 0
        self._signal("bready").value = 0
        await RisingEdge(clk)
        assert response is not None, (
            f"write to {address:#06x} was never answered within "
            f"{self.limit} cycles")
        return response

    async def read(self, address: int) -> tuple[int, int]:
        """Read one word. Returns ``(data, response)``."""
        from cocotb.triggers import ReadOnly, RisingEdge

        clk = self.dut.clk
        self._signal("araddr").value = address
        self._signal("arvalid").value = 1
        self._signal("rready").value = 1

        result = None
        for _ in range(self.limit):
            await RisingEdge(clk)
            await ReadOnly()
            if int(self._signal("rvalid").value):
                result = (int(self._signal("rdata").value),
                          int(self._signal("rresp").value))
                break
        await RisingEdge(clk)
        self._signal("arvalid").value = 0
        self._signal("rready").value = 0
        await RisingEdge(clk)
        assert result is not None, (
            f"read of {address:#06x} was never answered within "
            f"{self.limit} cycles")
        return result
