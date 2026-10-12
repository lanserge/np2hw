# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Stream plumbing: the pieces that join generated cores to each other.

A core's timing claim is made about ITS OWN registers. That claim stays
true only while nothing else is on the path -- and a design is a
composition, so something always is. If a core's output is combinational,
every consumer inherits that depth before doing any work of its own, and
two modules that each meet timing alone can miss it together. This bench
saw exactly that twice, both times from a link receiver's output (a 4:1
mux over its FIFO banks) into the first arithmetic of an ISP: -1.424ns on
one build and -1.792ns on another, from a receiver and a pipeline that
each passed on their own.

Nothing in a per-module model can catch it, and here the composing is
done by a block design np2hw never sees. So the fix is not a better
model: it is making the claim TRUE by construction, with a register at
the boundary.

`skid()` is that register. Registering a valid/ready interface naively
either drops backpressure or halves the throughput; a skid buffer holds
one extra beat so `ready` can be registered too, and the interface runs
at full rate with every signal leaving a flop.
"""
from __future__ import annotations

from . import hdl
from .ports import Pin, bus_attribute, published

# ---------------------------------------------------------------------------
# The elastic stream, declared ONCE.
#
# Every generated core, every plumbing block here, and every composed top
# speaks this: valid/ready with the data, and flags that ride the same
# handshake. That shape was implied by port names and re-derived wherever
# it was needed -- a block design's netlist writer matched `(.+)_(valid|
# ready|data|sof|eol|last)$` with a regex and decided a prefix was a
# stream if it carried valid AND ready. Inference, from names, in our own
# tooling: the same guess IP Integrator was making about AXI4-Lite before
# the ports declared themselves.
#
# So it is stated here, once, in the module the plumbing already lives in.
# `np2hw.ipxact` renders it as an IP-XACT bus definition and the Verilog
# emitters attach the matching interface attributes, both reading THIS --
# so a tool's view of the protocol and the RTL's cannot disagree.
# The vendor field matches the FuseSoC VLNVs this project already
# publishes (`lanserge:np2hw:gen:0`, `lanserge:revela:...`), so one
# identity names this work in both packaging worlds. IP-XACT more
# often carries a reverse domain; a VLNV is an identity a consumer
# references, so it is chosen once and then left alone.
STREAM_VLNV = ("lanserge", "interface", "np2hw_stream", "1.0")
STREAM_RTL_VLNV = ("lanserge", "interface", "np2hw_stream_rtl", "1.0")

# (signal suffix, IP-XACT logical name, width or None for the stream's own,
#  driven by the SOURCE, presence, what it means)
STREAM_SIGNALS = (
    ("valid", "VALID", 1, True, "required",
     "the source offers a beat; it may not be withdrawn before READY"),
    ("ready", "READY", 1, False, "required",
     "the sink accepts; a beat transfers where VALID and READY are both high"),
    ("data", "DATA", None, True, "required",
     "one sample, as wide as the stream was traced at"),
    ("sof", "SOF", 1, True, "optional",
     "start of frame: this beat is the frame's first, and re-anchors a core"),
    ("eol", "EOL", 1, True, "optional",
     "end of line: this beat is a row's last"),
    ("last", "LAST", 1, True, "optional",
     "end of frame: this beat is the frame's last"),
)

# The three that make a stream a stream. A prefix carrying these is one;
# the flags are per-stream and stated by each module's interface.
STREAM_REQUIRED = tuple(s for s, _, _, _, p, _ in STREAM_SIGNALS
                        if p == "required")
STREAM_FLAGS = tuple(s for s, _, _, _, p, _ in STREAM_SIGNALS
                     if p == "optional")


# A bus interface named after an HDL reserved word is a critical warning
# in a block design -- `in` and `out` are the two a stream prefix collides
# with, and they are the two most natural names for a stream's ends. The
# INTERFACE is renamed, never the ports: the ports are the module's and
# renaming them would ripple through every design that wires them.
_RESERVED = frozenset({
    "in", "out", "inout", "buffer", "signal", "entity", "block", "process",
    "port", "map", "type", "of", "is", "end", "begin", "for", "when",
})


def bundle_name(prefix: str) -> str:
    """The interface name for a stream whose ports are ``<prefix>_*``.

    The prefix where it is safe, so a block design shows the name the
    ports already carry; suffixed where it is a reserved word. Whoever
    reads these attributes back groups by the PORT name rather than by
    this label, so the two need not agree and only this one has to be
    legal.
    """
    return f"{prefix}_stream" if prefix in _RESERVED else prefix


def stream_port_attribute(prefix: str, suffix: str) -> str:
    """The attribute one stream port carries, naming its bundle and role.

    Same form the AXI4-Lite ports use, pointing at OUR definition instead
    of a vendor's. `prefix` is the bundle -- `in`, `out`, `isp_in` -- so
    two streams on one module stay two interfaces. Returned unindented:
    an attribute binds to the declaration that FOLLOWS it, so in a port
    list written with leading commas the comma has to come first and only
    the caller knows which style it is emitting.
    """
    logical = next(logical for s, logical, *_ in STREAM_SIGNALS if s == suffix)
    return bus_attribute(":".join(STREAM_RTL_VLNV), bundle_name(prefix),
                         logical)


def clock_attribute(name: str, busifs, reset: str | None = None) -> str:
    """The attribute a clock port carries: the buses it drives, and their
    reset.

    A stream declares its bundle; which clock runs it is not in any of
    its signals, and an integrator told nothing has to guess -- IP
    Integrator does, and says so, and then takes a reset it cannot tie to
    a clock for an asynchronous one. `busifs` is a bundle name or a list
    of them (`bundle_name` of each stream's prefix, `s_axil`), everything
    this clock drives. Returned unindented, like stream_port_attribute.
    """
    names = [busifs] if isinstance(busifs, str) else list(busifs)
    param = f"ASSOCIATED_BUSIF {':'.join(names)}"
    if reset:
        param += f", ASSOCIATED_RESET {reset}"
    return (f'(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 {name} CLK",'
            f' X_INTERFACE_PARAMETER = "{param}" *)')


def reset_attribute(name: str, active_low: bool = False) -> str:
    """The attribute a reset port carries: its polarity, stated rather
    than implied by a name."""
    pol = "ACTIVE_LOW" if active_low else "ACTIVE_HIGH"
    return (f'(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 {name} RST",'
            f' X_INTERFACE_PARAMETER = "POLARITY {pol}" *)')


def stream_signal(suffix: str) -> tuple:
    """The declaration for one signal, by its port suffix."""
    for entry in STREAM_SIGNALS:
        if entry[0] == suffix:
            return entry
    raise KeyError(f"{suffix!r} is not a signal of the np2hw stream; "
                   f"it carries {[s[0] for s in STREAM_SIGNALS]}")


def stream_pins(prefix: str, direction: str, data_bits, flags=(),
                order=None, before=(), signed=False) -> list:
    """The ports of one stream, each carrying its bundle attribute.

    `direction` is the stream's: ``"in"`` for a sink, ``"out"`` for a
    source, and each signal faces the way STREAM_SIGNALS says its source
    drives it. `data_bits` is a width or an expression over the module's
    parameters; `flags` the optional signals this stream carries; `order`
    the order the declarations appear in, which is the module's own;
    `before` the comment lines above the first of them."""
    carried = ["valid", "ready", "data", *flags]
    unknown = [f for f in flags if f not in ("sof", "eol", "last")]
    if unknown:
        raise KeyError(f"{prefix}: {unknown} are not stream flags")
    order = tuple(order or carried)
    if sorted(order) != sorted(carried):
        raise ValueError(f"{prefix}: order {order} must name exactly "
                         f"{carried}")
    pins = []
    for suffix in order:
        _, _, width, from_source, *_ = stream_signal(suffix)
        out = (direction == "out") == from_source
        pins.append(Pin(f"{prefix}_{suffix}", "output" if out else "input",
                        data_bits if suffix == "data" else 1,
                        signed=signed and suffix == "data",
                        bus=(":".join(STREAM_RTL_VLNV), bundle_name(prefix),
                             stream_signal(suffix)[1]),
                        before=tuple(before) if not pins else ()))
    return pins


def skid(data_bits: int) -> dict:
    """A two-deep elastic register slice: every port leaves a flop.

    Args:
        data_bits: width of the payload carried through. Flags ride in
            it -- pack them alongside the data rather than adding ports,
            so one instance serves any stream shape.

    Returns:
        ``{"verilog", "module", "interface", "latency"}``. Latency is one
        cycle; throughput is one beat per cycle, unimpaired.
    """
    if data_bits < 1:
        raise ValueError(f"data_bits {data_bits} must be at least 1")

    pins = [
        Pin("clk", "input", attribute=clock_attribute(
            "clk", [bundle_name("s"), bundle_name("m")], "rst")),
        Pin("rst", "input", attribute=reset_attribute("rst")),
        *stream_pins("s", "in", "W", order=("data", "valid", "ready")),
        *stream_pins("m", "out", "W", order=("data", "valid", "ready")),
    ]

    return {
        "verilog": hdl.text("np2hw_skid"),
        "module": "np2hw_skid",
        "parameters": {"W": data_bits},
        "latency": 1,
        "interface": {
            "clock": "clk", "reset": "rst", "param_prefix": "param_",
            "ports": published(pins),
            "parameters": {"W": data_bits},
            "input": {"prefix": "s", "data_bits": data_bits, "flags": ()},
            "output": {"prefix": "m", "data_bits": data_bits, "flags": ()},
            "streams": [
                {"prefix": "s", "direction": "in", "data_bits": data_bits,
                 "flags": (), "domain": ""},
                {"prefix": "m", "direction": "out", "data_bits": data_bits,
                 "flags": (), "domain": ""},
            ],
            "params": [],
            # The claim this module exists to make. Empty is the good case.
            "combinational_outputs": (),
        },
    }


def combinational_outputs(interface) -> tuple:
    """Which of a module's output ports do NOT leave a flop.

    Empty means the module's timing is its own under any composition:
    whatever it is wired to starts from a register. Anything listed is
    depth this module hands to its neighbour, to be spent before that
    neighbour reaches its own first register.

    Stated as ONE fact rather than a boolean plus a list of exceptions,
    because a boolean would be derivable from the list and two places to
    read the same thing is two places for them to disagree.

    Takes either a generator's whole result or its `interface`: not every
    generator has an interface to put it in (a video sink is not an np2hw
    stream), and a caller should not have to
    know which kind it is holding to ask a question about boundaries.
    """
    if interface is None:
        return ()
    if "combinational_outputs" in interface:
        return tuple(interface["combinational_outputs"])
    inner = interface.get("interface")
    if isinstance(inner, dict):
        return tuple(inner.get("combinational_outputs", ()))
    return ()


def boundary_report(instances) -> list:
    """Where a composition's combinational paths cross module boundaries.

    Args:
        instances: ``(name, interface)`` pairs, in any order -- the
            report is per module, not per connection, because a
            combinational output costs whatever it is wired to and the
            wiring is not always np2hw's to see.

    Returns:
        A list of ``(name, ports)`` for modules that hand depth on, IN
        THE ORDER GIVEN. Empty means every boundary in the set is a
        register, and each module's timing claim is its own.

        The order is the caller's because the caller knows the topology
        and this function does not. Sorting by how many ports a module
        hands on would look like ranking by severity and would not be:
        four ports on one boundary is one boundary, while one port
        repeated down a chain of seven is a path seven deep. Given the
        modules in pipeline order, the output reads as the path.

    This REPORTS rather than refuses. A refusal belongs to a law that can
    be obeyed, and today a traced core's `in_ready` is combinational by
    construction, so refusing would only mean refusing everything. What
    it can do is stop the fact being invisible.
    """
    found = []
    for name, iface in instances:
        ports = combinational_outputs(iface)
        if ports:
            found.append((name, tuple(ports)))
    return found


def cdc_fifo(data_bits: int, addr_bits: int = 4) -> dict:
    """A stream crossing between two clocks that owe each other nothing.

    The classic dual-clock FIFO: gray-coded pointers cross through two
    flops, the payload never crosses at all -- it sits in the memory
    until the read side's own arithmetic says a slot is safely full.
    That discipline is the entire design: a binary pointer crossing
    mid-increment can be misread as ANY value, a gray-coded one is
    wrong by at most one slot, and one slot of pessimism is invisible.

    Why np2hw carries its own instead of pointing at a vendor part:
    the receiver's clock is recovered from a cable and the pipeline's
    is the board's, so every design this generator feeds has exactly
    this crossing in it -- and a vendor converter is the one piece of
    such a design that cannot follow it to another family, or to an
    ASIC. One component, proven once, owned.

    Args:
        data_bits: payload width. Framing flags ride IN the payload,
            packed by the caller, exactly as `skid` takes them -- one
            layout decision, made where the stream is assembled.
        addr_bits: depth is ``2**addr_bits``. The consumer being faster
            on average than the producer (1920 samples in a 2200-clock
            line), the FIFO only ever absorbs jitter, so the default 16
            is already generous. It is NOT a rate adapter: a producer
            faster than its consumer needs a frame store, not a buffer.

    Both sides carry their own reset, and a link reset must assert
    BOTH (the pixel side directly, the read side through its domain's
    synchroniser): pointers that reset on one side only disagree
    forever after. The board wiring owns that contract; the module
    states it.

    A timing or CDC flow has to be TOLD about the two pointer paths --
    they are synchronised by construction and no tool can see that --
    and `docs/clock-crossings.md` says exactly what to tell it: which
    signals cross, why gray coding makes a bounded max-delay safe
    rather than merely conventional, and what an ASIC flow needs that
    an FPGA one does not.
    """
    if data_bits < 1:
        raise ValueError(f"data_bits {data_bits} must be at least 1")
    if addr_bits < 2:
        raise ValueError(f"addr_bits {addr_bits}: gray-window arithmetic "
                         "needs at least 2")
    W, A = data_bits, addr_bits

    pins = [
        Pin("wclk", "input", attribute=clock_attribute(
            "wclk", bundle_name("in"), "wrst")),
        Pin("wrst", "input", attribute=reset_attribute("wrst")),
        *stream_pins("in", "in", "W", order=("data", "valid", "ready")),
        Pin("rclk", "input", attribute=clock_attribute(
            "rclk", bundle_name("out"), "rrst")),
        Pin("rrst", "input", attribute=reset_attribute("rrst")),
        *stream_pins("out", "out", "W", order=("data", "valid", "ready")),
    ]

    return {
        "verilog": hdl.text("np2hw_cdc_fifo"),
        "module": "np2hw_cdc_fifo",
        "parameters": {"W": W, "A": A},
        "depth": 1 << A,
        "interface": {
            "clock": "rclk", "reset": "rrst", "param_prefix": "param_",
            "ports": published(pins),
            "parameters": {"W": W, "A": A},
            "input": {"prefix": "in", "data_bits": W, "flags": ()},
            "output": {"prefix": "out", "data_bits": W, "flags": ()},
            "streams": [
                {"prefix": "in", "direction": "in", "data_bits": W,
                 "flags": (), "domain": ""},
                {"prefix": "out", "direction": "out", "data_bits": W,
                 "flags": (), "domain": ""},
            ],
            "params": [],
            "combinational_outputs": (),
        },
    }


def stream_cdc(data_bits: int, addr_bits: int = 4) -> dict:
    """A stream with its framing flags crossing between two clocks.

    `cdc_fifo` with a stream's ports on each face: samples and their
    `sof`/`eol`/`last` packed into the FIFO's payload and out again, so a
    crossing can sit between two stream ports without a caller packing
    anything. The discipline, the reset contract and the timing intent are
    `cdc_fifo`'s (`np2hw/hdl/np2hw_cdc_fifo.xdc`, scoped to that module).
    """
    if data_bits < 1:
        raise ValueError(f"data_bits {data_bits} must be at least 1")
    if addr_bits < 2:
        raise ValueError(f"addr_bits {addr_bits}: gray-window arithmetic "
                         "needs at least 2")
    flags = ("sof", "eol", "last")
    pins = [
        Pin("wclk", "input", attribute=clock_attribute(
            "wclk", bundle_name("in"), "wrst")),
        Pin("wrst", "input", attribute=reset_attribute("wrst")),
        *stream_pins("in", "in", "DATA_BITS", flags=flags),
        Pin("rclk", "input", attribute=clock_attribute(
            "rclk", bundle_name("out"), "rrst")),
        Pin("rrst", "input", attribute=reset_attribute("rrst")),
        *stream_pins("out", "out", "DATA_BITS", flags=flags),
    ]
    parameters = {"DATA_BITS": data_bits, "A": addr_bits}
    return {
        "verilog": hdl.text("np2hw_stream_cdc"),
        "module": "np2hw_stream_cdc",
        "parameters": parameters,
        "requires": ("np2hw_cdc_fifo",),
        "depth": 1 << addr_bits,
        "interface": {
            "clock": "rclk", "reset": "rrst", "param_prefix": "param_",
            "ports": published(pins),
            "parameters": parameters,
            "input": {"prefix": "in", "data_bits": data_bits,
                      "flags": flags},
            "output": {"prefix": "out", "data_bits": data_bits,
                       "flags": flags},
            "streams": [
                {"prefix": "in", "direction": "in", "data_bits": data_bits,
                 "flags": flags, "domain": ""},
                {"prefix": "out", "direction": "out",
                 "data_bits": data_bits, "flags": flags, "domain": ""},
            ],
            "params": [],
            "combinational_outputs": (),
        },
    }


def tee(data_bits: int) -> dict:
    """One stream, two DISPOSABLE consumers -- and the tee NEVER stalls.

    The law: a branch's consumer may be owned by software, and
    software dies -- mid-frame, mid-burst, kill -9. A fork that
    advances its branches in lockstep hands every consumer a veto
    over the stream: one death freezes the fork at a frame boundary
    that can never arrive, and a display shares the coffin with the
    instrument that failed. So this tee is a pure parallel tap:
    upstream is ALWAYS accepted, each enabled branch receives what it
    is ready for, and a branch that cannot keep up loses its own
    beats -- never its sibling's, never the pipeline's.

    Honesty replaces losslessness: a branch that misses a beat
    MID-FRAME drops out for the rest of that frame (a partial tail
    would be the torn-frame class this element exists to prevent)
    and a sticky `torn_*` output says so, until reset. A branch not
    ready at its own SOF simply sits that frame out. Lossless-while-
    keeping-up is the contract each consumer signs for itself: the
    display's elastic keeps up by construction; a grabber that
    falls behind re-grabs.

    Discard-when-off stays: with both branches off the input drains
    at full rate, so the pipeline upstream keeps running -- standby
    is "outputs off", never "pipeline stalled". SWITCHING IS
    FRAME-ATOMIC: enables are sampled at each frame's SOF and held,
    so a branch joins at a frame start -- never receives a tail.

    Args:
        data_bits: payload width; flags ride in it, packed by the
            caller, except SOF which the tee must see to keep its
            frame-atomic promise -- so it is a port of its own.
    """
    if data_bits < 1:
        raise ValueError(f"data_bits {data_bits} must be at least 1")
    W = data_bits

    pins = [
        Pin("clk", "input", attribute=clock_attribute(
            "clk", [bundle_name(b) for b in ("in", "a", "b")], "rst")),
        Pin("rst", "input", attribute=reset_attribute("rst")),
        *stream_pins("in", "in", "W", flags=("sof",),
                     order=("valid", "ready", "data", "sof")),
        Pin("en_a", "input"),
        Pin("en_b", "input"),
        *stream_pins("a", "out", "W", order=("valid", "ready", "data")),
        *stream_pins("b", "out", "W", order=("valid", "ready", "data")),
        Pin("torn_a", "output", reg=True,
            before=("sticky until reset: that branch lost mid-frame beats",)),
        Pin("torn_b", "output", reg=True),
    ]

    return {
        "verilog": hdl.text("np2hw_tee"),
        "module": "np2hw_tee",
        "parameters": {"W": W},
        "interface": {
            "clock": "clk", "reset": "rst", "param_prefix": "",
            "ports": published(pins),
            "parameters": {"W": W},
            "input": {"prefix": "in", "data_bits": W, "flags": ("sof",)},
            "output": None,      # two outputs: `streams` is the truth
            "streams": [
                {"prefix": "in", "direction": "in", "data_bits": W,
                 "flags": ("sof",), "domain": ""},
                {"prefix": "a", "direction": "out", "data_bits": W,
                 "flags": (), "domain": ""},
                {"prefix": "b", "direction": "out", "data_bits": W,
                 "flags": (), "domain": ""},
            ],
            "params": [("en_a", 1, False), ("en_b", 1, False)],
            # valid fans out combinationally; in_ready is constant 1.
            "combinational_outputs": ("in_ready", "a_valid", "b_valid"),
        },
    }
