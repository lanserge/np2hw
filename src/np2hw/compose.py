# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
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
from dataclasses import dataclass, field, replace as dc_replace

from .stream import context_bits


def arm_capture(req: str) -> str:
    """The datapath's first synchroniser flop on the arm request: the
    one register the request crosses into, named here once -- the
    emitter declares it, and the crossing's constraint names it."""
    return f"{req}_s0"


def arm_answer(ack: str) -> str:
    """The datapath's register that raises the acknowledgement: where
    the answer leaves its clock, named here once."""
    return f"{ack}_r"


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
    # The frame's context the stream carries in its CTX member: (name, bits,
    # follows) a field, in order. A sink that takes a context takes exactly
    # this one: two layouts are two different words.
    ctx: tuple = ()

    def compatible(self, other: "StreamType") -> str | None:
        """None if a source of this type may drive a sink of `other`."""
        if other.ctx and tuple(map(tuple, self.ctx)) != tuple(
                map(tuple, other.ctx)):
            have = [f[0] for f in self.ctx] or "none"
            want = [f[0] for f in other.ctx]
            return (f"a sink taking the context {want} from a source "
                    f"carrying {have}")
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
    # What this instance's input context is, where it is not the stream's
    # own: an expression -- the stream's context with the fields a register
    # supplies joined to it, where a pipeline's stream enters
    ctx_in: str | None = None
    # {port_name: domain} -- an opaque tag the caller attaches to this module's
    # streams. np2hw never interprets it; it only refuses to wire two streams
    # the caller called different things. Widths cannot catch that.
    domains: dict = field(default_factory=dict)
    # {port_name: (wire, bits)} -- scalar outputs of the module beyond its
    # stream, each driving a wire of that name the composer declares, for
    # another instance's binding or a top-level assign to read
    outputs: dict = field(default_factory=dict)
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
        return StreamType(spec["data_bits"], tuple(spec["flags"]), domain,
                          tuple(tuple(f) for f in spec.get("ctx", ())))

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
            header=(), notes=(), commit=None, commit_ports=(),
            arm=None, assigns=None) -> dict:
    """Emit a module that instantiates `instances` and wires `connections`.

    Args:
        module_name: name of the composed module.
        instances: :class:`Instance` values, in the order they should appear.
        connections: :class:`Connection` values. An endpoint is `instance.port`
            or the name of a top-level :class:`Port`.
        ports: :class:`Port` values, in declaration order. Stream ports carry a
            :class:`StreamType`; everything else is a plain scalar or vector.
        header: comment lines emitted before the module (licence, provenance).
        assigns: `{output_port: expression}` for scalar top-level outputs
            driven from inside -- an instance's output wire, say.
        notes: comment lines emitted after the header (an address map, say).
        commit: `True`, or a Verilog expression. Either way, every
            binding listed in `commit_ports` gets a per-instance copy,
            kept beside the block that reads it instead of read across
            the chip from a central bank. `True` loads them all on one
            pulse the moment a frame LEAVES the pipeline, when the
            pipe is provably empty; an expression says when to load
            instead, for a design whose output is not framed.
        arm: ``(request_port, acknowledge_port)`` when the registers live
            in another clock domain. The caller declares both ports. The
            copies are then taken only while the register file says its
            values are frozen, and the file is told when they have been.
            Nothing here waits for the bus: if the request never comes,
            the copies simply keep the values they have, so a slow or
            silent crossing delays an update and cannot corrupt one.
        commit_ports: which composed-module ports are configuration, and
            so belong in those local copies. Stated by the caller rather
            than guessed from a name, because a scalar input can equally
            be CONTEXT -- geometry that arrived in a stream's header --
            and latching that on a frame commit would hold a frame's own
            facts one frame too long.

    Returns:
        ``{"verilog": str, "module": str, "nets": [...]}``.

    A frame's size travels with its stream, in its context: the stream's
    CTX member, wired with the stream like its other members.
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
        if dst_inst is not None and dst_inst.ctx_in is not None:
            # its context is given, not the stream's: the rest is checked
            consumed = dc_replace(consumed, ctx=())
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
    from .ports import Pin, header as port_header, published
    from .stream import stream_pins
    pins = [Pin("clk", "input"), Pin("rst", "input")]
    for port in ports:
        # Wrapped, not truncated: a description carried into the RTL is only
        # useful in full, and an 800-column comment is not readable.
        about = tuple(textwrap.wrap(" ".join(port.comment.split()), width=72)) \
            if port.comment else ()
        if port.stream is None:
            pins.append(Pin(port.name,
                            "input" if port.direction == "in" else "output",
                            port.width, signed=port.signed, before=about))
            continue
        pins += stream_pins(port.name, port.direction, port.stream.data_bits,
                            flags=port.stream.flags, before=about,
                            ctx_bits=context_bits(port.stream.ctx))
    L += port_header(module_name, pins)
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
        if stream.ctx:
            a(f"    wire [{context_bits(stream.ctx) - 1}:0] {net}_ctx;")
    a("")
    scalars = [(w, b) for inst in instances for w, b in inst.outputs.values()]
    if scalars:
        a("    // Scalar nets an instance drives beside its stream.")
        for wire, bits in scalars:
            a(f"    wire {f'[{bits - 1}:0] ' if bits > 1 else ''}{wire};")
        a("")

    for key, stream in sources.items():
        if "." in key:
            continue                        # driven by an instance below
        net = _net(key)
        a(f"    assign {net}_valid = {key}_valid;")
        a(f"    assign {net}_data  = {key}_data;")
        for flag in stream.flags:
            a(f"    assign {net}_{flag} = {key}_{flag};")
        if stream.ctx:
            a(f"    assign {net}_ctx  = {key}_ctx;")
        a(f"    assign {key}_ready = {net}_ready;")
    a("")

    # ---- each instance's own copy of the parameters it reads ----------
    #
    # A central register bank is the right place to WRITE parameters --
    # it is the address map, and software's view of the design. It is the
    # wrong place to READ them from. The value has to reach the logic that
    # uses it, and that logic sits wherever its block was placed, so a
    # central bank means a long wire with the block's first arithmetic on
    # the far end of it, all inside one clock period.
    #
    # This bench measured that: a gamma knot register in the register
    # file feeding a 64:1 select and a subtract in the gamma block --
    # 1.889ns of logic against 4.133ns of route, 69% of the path spent
    # travelling. Pipelining the arithmetic cannot help, because the
    # distance is not the arithmetic.
    #
    # WHEN they load is decided by vertical blanking, and that makes it
    # simple. Blanking is 45 lines at 1080p60, about 99,000 clocks,
    # against a pipeline tens deep: by the time it is half over the pipe
    # has drained thousands of times. Load there and the questions that
    # look hard never arise -- there are no pixels in flight to be caught
    # with old coefficients in the blocks behind them and new ones ahead,
    # and one instant serves every block, so there is nothing to hold
    # atomic and no second bank needed to freeze it. A write lands either
    # before the pulse or after it.
    #
    # The pulse waits for the input to be IDLE rather than counting off a
    # known latency. A count would have to be kept in step with a
    # pipeline that changes every time a block is added; an idle input is
    # the same fact, observed instead of predicted.
    live_of = {}
    if commit:
        wanted = set(commit_ports)
        for inst in instances:
            widths = {n: (b, s) for n, b, s in inst.interface["params"]}
            for name, bound in sorted(inst.bind.items()):
                expr = bound[0] if isinstance(bound, tuple) else bound
                if expr not in wanted or name not in widths:
                    continue
                bits, signed = widths[name]
                live = f"{inst.name}_live_{name}"
                live_of[(inst.name, name)] = (live, expr)
                rng = f"[{bits - 1}:0] " if bits > 1 else ""
                a(f"    reg {'signed ' if signed else ''}{rng}{live};")

    if live_of:
        # WHEN they load: as a WAVE, travelling down the chain with the
        # frame boundary itself. Each instance takes its copy at its own
        # incoming `last` beat -- the moment the outgoing frame's final
        # pixel enters it -- so the next frame's first pixel, one beat
        # later at the earliest, computes with the new values. Upstream
        # blocks load first, downstream later, exactly as the new frame
        # advances: every block has its parameters "in time, before the
        # pixels arrive at the block".
        #
        # Ordering does the work that pipe-EMPTINESS used to do. The old
        # form was one global pulse gated on the pipe having drained --
        # also one frame of loop latency, to be fair to it, but resting
        # on a drained-detector being right and on the source leaving a
        # gap as long as the WHOLE pipe. The wave rests on ordering,
        # which the stream itself guarantees, and shrinks the gap each
        # block needs to that block's own flush -- not the pipeline's.
        #
        # THE REQUIREMENT, stated rather than enjoyed silently: a block
        # that lags its input (a stencil flushing buffered lines) is
        # still finishing frame N when its copy loads, so the gap
        # between frames at its input must exceed its own flush latency
        # -- two line-times for a 5-line stencil, against an inter-frame
        # gap of tens of lines on any real raster. A source that packs
        # frames tighter than its deepest block flushes would tear that
        # block's last rows, and no load instant fixes a machine still
        # computing two frames at once.
        #
        # A commit arriving MID-WAVE must not split the chain -- blocks
        # one and two on the old set, the rest on the new. The armed
        # flag is latched ONCE, at the wave's head, and the rest of the
        # chain obeys the latch.
        per_inst = {}
        for (inst_name, name), (live, expr) in live_of.items():
            per_inst.setdefault(inst_name, []).append((live, expr))

        def in_boundary(inst):
            edges = [e for (_src), (dst, _p), e in parsed if dst is inst]
            if not edges:
                return None
            key = edges[0].source
            if "last" not in tuple(sources[key].flags):
                return None
            n = _net(key)
            return f"{n}_valid && {n}_ready && {n}_last"

        boundaries = {}
        for inst in instances:
            if inst.name in per_inst:
                b = in_boundary(inst)
                if b is None and commit is True:
                    raise ValueError(
                        f"instance {inst.name!r} has live parameters but no "
                        "framed input to time their load by. Pass "
                        "commit='<expression>' to say when a copy loads.")
                boundaries[inst.name] = b
        ordered = [i.name for i in instances if i.name in per_inst]
        head = boundaries[ordered[0]] if commit is True else None

        gate = ""
        if arm:
            req, ack = arm
            a("")
            a("    // The register file freezes its values when software arms")
            a("    // it (writes refused until acknowledged), so the wave reads")
            a("    // a bank that provably cannot move under it. One bit each")
            a("    // way; the values themselves never cross anything.")
            a("    //")
            a("    // And nothing here WAITS. An unarmed boundary loads no")
            a("    // copies and every pixel stays correct on the old set; a")
            a("    // late commit lands at the next frame, never in a wrong")
            a("    // pixel. Timing decides WHEN an update takes effect --")
            a("    // never whether the result is right.")
            a(f"    (* ASYNC_REG = \"TRUE\" *) reg {arm_capture(req)}, "
              f"{req}_s1;")
            a(f"    reg {arm_answer(ack)};")
            a("    // Latched at the wave's head, so a commit that arrives")
            a("    // mid-wave waits for the next frame instead of splitting")
            a("    // the chain between two parameter sets.")
            a("    reg wave_armed;")
            a(f"    assign {ack} = {arm_answer(ack)};")
            a("    always @(posedge clk) begin")
            a("        if (rst) begin")
            a(f"            {arm_capture(req)} <= 1'b0; {req}_s1 <= 1'b0;")
            a(f"            {arm_answer(ack)} <= 1'b0; wave_armed <= 1'b0;")
            a("        end else begin")
            a(f"            {arm_capture(req)} <= {req}; "
              f"{req}_s1 <= {arm_capture(req)};")
            if head:
                a(f"            if ({head}) wave_armed <= {req}_s1;")
            out = next((pt.name for pt in ports if pt.stream is not None
                        and pt.direction == "out"
                        and "last" in tuple(pt.stream.flags)), None)
            done = (f"{out}_valid && {out}_ready && {out}_last"
                    if out else (head or "1'b0"))
            a("            // Acknowledged when the wave has passed the whole")
            a("            // chain: the frame that carried it leaves the pipe.")
            a(f"            if ({done} && wave_armed && !{arm_answer(ack)}) "
              f"{arm_answer(ack)} <= 1'b1;")
            a(f"            else if (!{req}_s1) {arm_answer(ack)} <= 1'b0;")
            a("        end")
            a("    end")

        first = ordered[0] if ordered else None
        for inst_name in ordered:
            copies = per_inst[inst_name]
            if commit is True:
                b = boundaries[inst_name]
                armed_term = ""
                if arm:
                    armed_term = (f" && {arm[0]}_s1" if inst_name == first
                                  else " && wave_armed")
                cond = f"({b}{armed_term})"
            else:
                cond = f"({commit})"
            a(f"    // {inst_name}: loads as the frame boundary reaches it")
            a(f"    always @(posedge clk) if (rst || {cond}) begin")
            for live, expr in copies:
                a(f"        {live} <= {expr};")
            a("    end")
        a("")
        live_of = {k: v[0] for k, v in live_of.items()}

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
            if (inst.name, name) in live_of:
                expression = live_of[(inst.name, name)]
                note = note or "this instance's own live copy"
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
            if spec.get("ctx"):
                a(f"        .{spec['prefix']}_ctx    "
                  f"({inst.ctx_in or net + '_ctx'}),")

        for port, (wire, _bits) in inst.outputs.items():
            a(f"        .{port:<20} ({wire}),")
        spec = interface["output"]
        if spec is None:
            L[-1] = L[-1].rstrip(",")
        else:
            net = _net(f"{inst.name}.{spec['prefix']}")
            flags = list(spec["flags"])
            a(f"        .{spec['prefix']}_valid ({net}_valid),")
            a(f"        .{spec['prefix']}_ready ({net}_ready),")
            a(f"        .{spec['prefix']}_data  ({net}_data),")
            if spec.get("ctx"):
                a(f"        .{spec['prefix']}_ctx   ({net}_ctx),")
            for i, flag in enumerate(flags):
                tail = "" if i == len(flags) - 1 else ","
                a(f"        .{spec['prefix']}_{flag:<5} ({net}_{flag}){tail}")
        a("    );")
        a("")

    for port, expr in (assigns or {}).items():
        a(f"    assign {port} = {expr};")
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
            if stream_ports[dst_port].stream.ctx:
                a(f"    assign {dst_port}_ctx = {net}_ctx;")
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
    interface = _composed_interface(module_name, ports)
    interface["ports"] = published(pins)
    return {"verilog": "\n".join(L), "module": module_name,
            "nets": sorted(sources), "interface": interface}


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
        return stream_of(port)

    def stream_of(port):
        out = {"prefix": port.name, "data_bits": port.stream.data_bits,
               "flags": tuple(port.stream.flags), "domain": port.stream.domain}
        if port.stream.ctx:
            out["ctx"] = [list(f) for f in port.stream.ctx]
        return out

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
        "streams": [dict(stream_of(p), direction=p.direction)
                    for p in ports if p.stream is not None],
        "params": [(p.name, p.width, p.signed) for p in scalars],
        "nestable": len(streams_in) <= 1 and len(streams_out) <= 1,
        "module": module_name,
    }
