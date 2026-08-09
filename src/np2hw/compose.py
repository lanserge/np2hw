"""Structural composition: instantiate generated cores and wire them as a netlist.

The rest of np2hw turns one NumPy function into one module. This turns several
of those into a design: it declares the nets, instantiates the cores, and wires
the streaming handshake between them.

Why this belongs here rather than in the application
-----------------------------------------------------

np2hw defines what a generated core's ports are and what the valid/ready
contract means. An application that composes cores has to know both -- which
flags a core accepts, which it regenerates, how a fan-out's `ready` is resolved
-- and any copy of that knowledge outside np2hw is a copy that can disagree with
the emitter. It is not a hypothetical: the two failures that motivated this were
both an application modelling a port list that np2hw had already decided.

So a core describes itself (`meta["interface"]`) and the composer reads it. The
caller supplies what it alone knows: which instances exist, how they connect,
and what to bind to their parameter ports.

This is structural emission, which np2hw already does -- switchboard_wrap,
axis_video_wrap, axil_regfile, control_top and the testbenches are all
non-traced Verilog. Composition is the same category with N modules instead of
one.

The streaming contract, in one place
------------------------------------

Every net carries `valid / ready / data` plus the framing flags. A transfer
happens where `valid && ready`.

An output may feed ONE consumer that applies backpressure, plus any number of
SINKS -- modules that consume a stream and produce none, and therefore never
stall. That is what makes a statistics tap free. Two consumers that both apply
backpressure need an element that buffers one while the other stalls; there
isn't one yet, so that case is refused rather than emitted as a fork that
deadlocks the first time a branch stalls.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamType:
    """What a net carries. Checked where a source meets a sink.

    A composer that does not check this connects a 10-bit Bayer stream to a
    block built for 12-bit RGB and emits Verilog that elaborates, which is the
    worst kind of wrong.
    """

    data_bits: int
    flags: tuple[str, ...] = ("sof", "eol", "last")
    # An OPAQUE tag the caller supplies and np2hw only compares. np2hw has no
    # idea what "bayer" or "rgb" mean and should not: what it knows is that two
    # streams the application called different things must not be wired
    # together. Width and framing cannot catch that -- a 12-bit Bayer stream and
    # a 12-bit luma stream are identical to a compiler and nonsense to connect.
    domain: str = ""

    def compatible(self, other: "StreamType") -> str | None:
        """None if a source of this type may drive a sink of `other`."""
        if self.data_bits != other.data_bits:
            return (f"{self.data_bits}-bit source into a {other.data_bits}-bit "
                    "sink")
        if self.domain and other.domain and self.domain != other.domain:
            return f"{self.domain!r} source into a {other.domain!r} sink"
        missing = set(other.flags) - set(self.flags)
        if missing:
            return (f"sink needs framing {sorted(missing)} that the source does "
                    "not carry")
        return None


@dataclass(frozen=True)
class Port:
    """A top-level port of the composed module."""

    name: str
    direction: str                 # 'in' | 'out'
    width: int = 1
    signed: bool = False
    comment: str = ""
    stream: StreamType | None = None      # set for pixel streams


@dataclass(frozen=True)
class Instance:
    """One generated core, instantiated.

    Args:
        name: Verilog instance name.
        core: the dict `generate()` returned. Its `interface` says what the
            module's ports are, so the caller does not restate them.
        bind: `{param_name: expression}` for the module's parameter ports, or
            `{param_name: (expression, comment)}`. This
            is the caller's knowledge, not a duplicate of np2hw's: which
            register or which context bit drives a coefficient is a decision.
        comment: a note emitted above the instantiation.
    """

    name: str
    core: dict
    bind: dict = field(default_factory=dict)
    # {port_name: domain} -- an opaque tag the caller attaches to this module's
    # streams. np2hw never interprets it; it only refuses to wire two streams
    # the caller called different things. Widths cannot catch that.
    domains: dict = field(default_factory=dict)
    comment: str = ""

    @property
    def interface(self) -> dict:
        """What the module presents. Every Core and every composed module has one."""
        return self.core["interface"]

    @property
    def module(self) -> str:
        return self.core["module"]

    @property
    def is_sink(self) -> bool:
        """A module that consumes a stream and produces none never stalls."""
        return self.interface["output"] is None

    def _stream(self, role: str) -> StreamType | None:
        spec = self.interface[role]
        if spec is None:
            return None
        # An explicit binding wins; otherwise a nested subsystem's own declared
        # domain carries through, so a hierarchy does not lose its types.
        domain = self.domains.get(spec["prefix"], spec.get("domain", ""))
        return StreamType(spec["data_bits"], tuple(spec["flags"]), domain)

    def input_stream(self) -> StreamType | None:
        return self._stream("input")

    def output_stream(self) -> StreamType | None:
        return self._stream("output")


@dataclass(frozen=True)
class Connection:
    """A directed edge. Endpoints are `instance.port` or a top-level port name."""

    source: str
    sink: str
    comment: str = ""


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #

def _net(endpoint: str) -> str:
    return "net_" + endpoint.replace(".", "_")


def compose(module_name: str, instances, connections, ports,
            header=(), notes=()) -> dict:
    """Emit a module that instantiates `instances` and wires `connections`.

    Args:
        module_name: name of the composed module.
        instances: :class:`Instance` values, in the order they should appear.
        connections: :class:`Connection` values. An endpoint is `instance.port`
            or the name of a top-level :class:`Port`.
        ports: :class:`Port` values, in declaration order. Stream ports carry a
            :class:`StreamType`; everything else is a plain scalar or vector.
        header: comment lines emitted before the module (licence, provenance).
        notes: comment lines emitted after the header (an address map, say).

    Returns:
        ``{"verilog": str, "module": str, "nets": [...]}``.
    """
    by_name = {inst.name: inst for inst in instances}
    stream_ports = {p.name: p for p in ports if p.stream is not None}

    def endpoint(text, role):
        instance, _, port = text.rpartition(".")
        if not instance:
            if text not in stream_ports:
                raise KeyError(
                    f"{role} {text!r} is neither a top-level stream port "
                    f"({sorted(stream_ports)}) nor an `instance.port` reference")
            return None, text
        if instance not in by_name:
            raise KeyError(
                f"{role} {text!r}: no instance {instance!r}; this design has "
                f"{sorted(by_name)}")
        return by_name[instance], port

    parsed = [(endpoint(c.source, "source"), endpoint(c.sink, "sink"), c)
              for c in connections]

    # -- type check, before a line is emitted --------------------------------- #
    for (src_inst, src_port), (dst_inst, dst_port), edge in parsed:
        produced = (src_inst.output_stream() if src_inst
                    else stream_ports[src_port].stream)
        consumed = (dst_inst.input_stream() if dst_inst
                    else stream_ports[dst_port].stream)
        if produced is None:
            raise ValueError(f"{edge.source} produces no stream")
        if consumed is None:
            raise ValueError(f"{edge.sink} consumes no stream")
        problem = produced.compatible(consumed)
        if problem:
            raise ValueError(
                f"{edge.source} -> {edge.sink}: {problem}. A composer that lets "
                "this through emits Verilog that elaborates and is wrong.")

    # -- one net per source, shared by a fan-out ------------------------------ #
    sources: dict[str, StreamType] = {}
    for (src_inst, src_port), _, edge in parsed:
        key = edge.source
        sources.setdefault(key, src_inst.output_stream() if src_inst
                           else stream_ports[src_port].stream)

    def consumers(source_key):
        return [(dst, edge) for _, dst, edge in parsed if edge.source == source_key]

    L = list(header)
    a = L.append
    if notes:
        a("//")
        L.extend(notes)
    a("")
    a(f"module {module_name} (")
    a("    input  wire        clk,")
    a("    input  wire        rst,")
    for index, port in enumerate(ports):
        last = index == len(ports) - 1
        if port.comment:
            # Wrapped, not truncated: a description carried into the RTL is only
            # useful in full, and an 800-column comment is not readable.
            for line in textwrap.wrap(" ".join(port.comment.split()), width=72):
                a(f"    // {line}")
        if port.stream is None:
            rng = "" if port.width == 1 else f"[{port.width - 1}:0] "
            sign = "signed " if port.signed else ""
            direction = "input  wire" if port.direction == "in" else "output wire"
            a(f"    {direction} {sign}{rng}{port.name}" + ("" if last else ","))
            continue
        bits = port.stream.data_bits
        if port.direction == "in":
            a(f"    input  wire        {port.name}_valid,")
            a(f"    output wire        {port.name}_ready,")
            a(f"    input  wire [{bits - 1}:0] {port.name}_data,")
            flags = [f"    input  wire        {port.name}_{f}"
                     for f in port.stream.flags]
        else:
            a(f"    output wire        {port.name}_valid,")
            a(f"    input  wire        {port.name}_ready,")
            a(f"    output wire [{bits - 1}:0] {port.name}_data,")
            flags = [f"    output wire        {port.name}_{f}"
                     for f in port.stream.flags]
        for i, line in enumerate(flags):
            tail = "" if (last and i == len(flags) - 1) else ","
            a(line + tail)
    a(");")
    a("")

    a("    // Stream nets. Every boundary carries the full ready/valid handshake,")
    a("    // so any block may stall its upstream all the way back to the source.")
    for key, stream in sources.items():
        net = _net(key)
        a(f"    wire        {net}_valid;")
        a(f"    wire        {net}_ready;")
        a(f"    wire [{stream.data_bits - 1}:0] {net}_data;")
        for flag in stream.flags:
            a(f"    wire        {net}_{flag};")
    a("")

    for key, stream in sources.items():
        if "." in key:
            continue                        # driven by an instance below
        net = _net(key)
        a(f"    assign {net}_valid = {key}_valid;")
        a(f"    assign {net}_data  = {key}_data;")
        for flag in stream.flags:
            a(f"    assign {net}_{flag} = {key}_{flag};")
        a(f"    assign {key}_ready = {net}_ready;")
    a("")

    def blocking(source_key):
        """Consumers that can stall this net. Sinks never can."""
        out = []
        for dst, edge in consumers(source_key):
            dst_inst, dst_port = dst
            if dst_inst is not None and dst_inst.is_sink:
                continue
            out.append((dst_inst, dst_port, edge))
        return out

    for inst in instances:
        if inst.comment:
            a(f"    // ---- {inst.comment} ----")
        a(f"    {inst.module} u_{inst.name} (" if not inst.name.startswith("u_")
          else f"    {inst.module} {inst.name} (")
        interface = inst.interface
        a(f"        .{interface['clock']:<10} (clk),")
        a(f"        .{interface['reset']:<10} (rst),")

        declared = {name for name, _, _ in interface["params"]}
        unknown = set(inst.bind) - declared
        if unknown:
            raise KeyError(
                f"instance {inst.name!r} binds {sorted(unknown)}, which module "
                f"{inst.module!r} does not declare; it has {sorted(declared)}")
        for name, _, _ in interface["params"]:
            if name not in inst.bind:
                raise KeyError(
                    f"instance {inst.name!r}: parameter port {name!r} is not "
                    "bound; every parameter needs a driver")
            bound = inst.bind[name]
            expression, note = bound if isinstance(bound, tuple) else (bound, "")
            port = f"{interface.get('param_prefix', 'param_')}{name}"
            line = f"        .{port:<20} ({expression}),"
            a(line.ljust(52) + (f"// {note}" if note else ""))

        incoming = [(edge, dst_port) for (_, _), (dst_inst, dst_port), edge in parsed
                    if dst_inst is inst]
        for edge, port in incoming:
            net = _net(edge.source)
            spec = interface["input"]
            ready = (f"{net}_ready"
                     if blocking(edge.source) and blocking(edge.source)[0][0] is inst
                     else f"{inst.name}_{port}_ready_unused")
            a(f"        .{spec['prefix']}_valid  ({net}_valid),")
            a(f"        .{spec['prefix']}_ready  ({ready}),")
            a(f"        .{spec['prefix']}_data   ({net}_data),")
            for flag in spec["flags"]:
                a(f"        .{spec['prefix']}_{flag:<6} ({net}_{flag}),")

        spec = interface["output"]
        if spec is None:
            L[-1] = L[-1].rstrip(",")
        else:
            net = _net(f"{inst.name}.{spec['prefix']}")
            flags = list(spec["flags"])
            a(f"        .{spec['prefix']}_valid ({net}_valid),")
            a(f"        .{spec['prefix']}_ready ({net}_ready),")
            a(f"        .{spec['prefix']}_data  ({net}_data),")
            for i, flag in enumerate(flags):
                tail = "" if i == len(flags) - 1 else ","
                a(f"        .{spec['prefix']}_{flag:<5} ({net}_{flag}){tail}")
        a("    );")
        a("")

    for key, stream in sources.items():
        net = _net(key)
        stalls = blocking(key)
        if not stalls:
            a(f"    // {key}: only sinks consume this, and a sink never stalls.")
            a(f"    assign {net}_ready = 1'b1;")
            a("")
            continue
        dst_inst, dst_port, edge = stalls[0]
        if dst_inst is None:                 # a top-level output
            a(f"    assign {dst_port}_valid = {net}_valid;")
            a(f"    assign {dst_port}_data  = {net}_data;")
            for flag in stream_ports[dst_port].stream.flags:
                a(f"    assign {dst_port}_{flag} = {net}_{flag};")
            a(f"    assign {net}_ready = {dst_port}_ready;")
            a("")

    dangling = []
    for inst in instances:
        for (_, _), (dst_inst, dst_port), edge in parsed:
            if dst_inst is not inst:
                continue
            stalls = blocking(edge.source)
            if not stalls or stalls[0][0] is not inst:
                dangling.append(
                    f"    wire {inst.name}_{dst_port}_ready_unused;"
                    f"   // sink: does not stall {edge.source}")
    if dangling:
        a("    // A sink's ready is not fed back: it never stalls its source.")
        L.extend(dangling)
        a("")

    a("endmodule")
    return {"verilog": "\n".join(L), "module": module_name,
            "nets": sorted(sources),
            "interface": _composed_interface(module_name, ports)}


def _composed_interface(module_name: str, ports) -> dict:
    """The interface a COMPOSED module presents, so it can be composed again.

    This is what makes the hierarchy real rather than two flat levels. A
    composed module describes itself exactly as a generated core does, so it can
    be an :class:`Instance` inside another :func:`compose` -- a reusable front
    end built once and instantiated per sensor, rather than its graph copied
    into every top level.

    Only single-stream subsystems nest today: the interface shape carries one
    input and one output, because that is what an Instance knows how to wire.
    A multi-stream subsystem is composable at the top but not nestable, and
    says so rather than silently exposing its first stream.
    """
    streams_in = [p for p in ports if p.stream is not None and p.direction == "in"]
    streams_out = [p for p in ports if p.stream is not None and p.direction == "out"]
    scalars = [p for p in ports if p.stream is None and p.direction == "in"]

    def one(group, role):
        if len(group) > 1:
            return None            # not nestable; see the docstring
        if not group:
            return None
        port = group[0]
        return {"prefix": port.name, "data_bits": port.stream.data_bits,
                "flags": tuple(port.stream.flags), "domain": port.stream.domain}

    return {
        "clock": "clk",
        "reset": "rst",
        # A composed module's scalar ports are named literally by its caller,
        # with no prefix to add.
        "param_prefix": "",
        "input": one(streams_in, "input"),
        "output": one(streams_out, "output"),
        # Every stream, in declaration order. `input`/`output` go None as soon as
        # there are two of either, because an Instance can only wire one pair --
        # but a wrapper that passes streams through (a register-file top, say)
        # needs them all, and a stereo pipeline has four.
        "streams": [{"prefix": p.name, "direction": p.direction,
                     "data_bits": p.stream.data_bits,
                     "flags": tuple(p.stream.flags), "domain": p.stream.domain}
                    for p in ports if p.stream is not None],
        "params": [(p.name, p.width, p.signed) for p in scalars],
        "nestable": len(streams_in) <= 1 and len(streams_out) <= 1,
        "module": module_name,
    }
