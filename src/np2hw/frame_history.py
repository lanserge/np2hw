# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A stream's recent past, delivered with its present: a frame history.

Some arithmetic needs the same pixel from several frames at once -- a
time-of-flight sensor reads out one reference phase per frame and depth needs
all four, a sequential HDR sensor alternates exposures, a temporal filter
averages. A NumPy model says that with a time axis: ``pixel[..., k]`` is the
pixel from frame ``n - k``. A trace cannot say it, because a traced model sees
one frame; the time axis is storage, not arithmetic. So the storage is this
generator's, exactly as the line buffers behind a stencil are the emitter's,
and the model keeps only the arithmetic.

What leaves the core
--------------------

One word per pixel with ``2 * (frames + 1)`` channels of ``bits`` each,
channel 0 in the low bits, as every multi-channel stream in np2hw is packed:

    channel k               the same pixel from frame n - k, k = 0 .. frames;
                            channel 0 is the live pixel, passed through
    channel frames + 1 + k  the TAG of frame n - k

So a model downstream is written ``pixel[..., k]`` and never learns there was
memory involved. Framing is regenerated from the core's own counters, as a
traced core's is: ``out_sof``, ``out_eol``, ``out_last``.

What each frame IS rides with it
--------------------------------

A sensor that captures differently frame by frame -- a time-of-flight
reference phase, an HDR exposure -- makes a frame's pixels meaningless without
knowing which capture they came from. So the history carries that too: one
TAG register per position in the sensor's capture loop, ``param_tag_0 ..
param_tag_{loop-1}``, written by software with whatever the consumer needs --
the phase, a gain, an exposure ratio -- and delivered beside each frame's
pixels. The live frame's loop position arrives as context bits
(``param_pos_*``, from the receiver, which derives it from its frame
counter); frame ``n - k`` sits ``k`` positions back, so its tag is
``tag[(pos - k) mod loop]``. The selection is a small mux on values constant
for the whole frame, done HERE, where it is plain structure -- a traced model
cannot select a register by context, but it can multiply a pixel by a tag it
is handed. ``live_tag`` reports the live frame's tag for software.

The tags are channels, not a side channel: a model reads them as data, the
wires carry them inside the fabric, and the memory never stores them -- they
are added at the output. A loop order the arithmetic does not assume (a
sensor stepping phases 0, 2, 1, 3; a multi-frequency sequence; AE moving the
exposure every frame) is table contents, not a rebuild.

How the past is kept
--------------------

One memory word per pixel position holds that position's ``frames`` past
samples, newest in the low bits. Each accepted pixel reads its word, emits it
beside the live sample, and writes it back SHIFTED -- the live sample in, the
oldest out. There is no ring pointer and no slot arithmetic: the address is
the pixel's position in the frame, and the same position recurs only a frame
later, so the write that follows a read can never meet a read it disturbs.

The read is REGISTERED, because that is the only kind a memory macro
implements, and it costs no schedule for the reason the stencil emitter's line
buffers give: the address is a counter. The core presents the position it
will want NEXT, from the counter's own next state, and the word lands exactly
when that pixel arrives. While the stream stalls, the read is not enabled and
the word holds.

The memory lives behind a seam module, ``<module>_mem`` -- one write port, one
read port, both synchronous -- carrying the same four-rule contract as the
line buffers', so an integrator binds a macro to it by name. This is the
resident binding: inferred block RAM. A binding onto external memory presents
the same ports and the same stream; what changes is only where the words
live and how their latency is hidden.

History is not trustworthy until it exists
------------------------------------------

Reset leaves the memory holding whatever it holds, so for the first
``frames`` frames the past channels are not a past. A frame start arriving
where the counter did not expect one breaks the history the same way: the
word prefetched for the predicted position belongs to the wrong one. Neither
is corrected here -- a store cannot know what its consumer should do with
untrustworthy history -- and both are the events a pipeline's `cold` signal
exists to report.
"""
from __future__ import annotations

from .ports import Pin, header, published

POS = 16            # counter width for column and row, as every core uses


def _clog2(n: int) -> int:
    return max(1, (n - 1).bit_length())


def frame_history(frames: int, bits: int, width: int, height: int,
                  loop: int | None = None,
                  module_name: str = "frame_history") -> dict:
    """Emit a core that delivers each pixel with its ``frames`` predecessors.

    Args:
        frames: how many past frames to deliver, >= 1. Channel k of the
            output is frame n - k; channel 0 is the live frame.
        bits: width of one sample, input and every output channel alike.
        width, height: the frame the memory is built for. The ``WIDTH`` and
            ``HEIGHT`` parameters default to these and may be set smaller at
            instantiation; the memory and its address are sized here.
        loop: length of the sensor's capture loop -- how many tag registers
            there are. Defaults to the window, ``frames + 1``; it may differ
            (an 8-frame multi-frequency loop read through a 4-frame window).
        module_name: the emitted module's name; the memory seam is
            ``<module_name>_mem``.

    Returns:
        ``{"verilog", "module", "interface", "memories", "frames", "loop",
        "channels"}``. The interface is the one a traced core publishes,
        with the output's channel count stated, so a composer threads it
        exactly as it threads a block's.
    """
    if frames < 1:
        raise ValueError(f"frames {frames}: a history needs at least one "
                         "past frame")
    loop = frames + 1 if loop is None else loop
    if loop < 1:
        raise ValueError(f"loop {loop}: a capture loop has at least one frame")
    if bits < 1:
        raise ValueError(f"bits {bits} must be at least 1")
    if width * height < 2:
        raise ValueError(
            f"geometry {width}x{height}: the read runs one position ahead of "
            "the write, so a frame needs at least two pixels -- with one, both "
            "would land on the same word in the same cycle")

    depth = width * height
    aw = _clog2(depth)
    pb = max(1, (loop - 1).bit_length())        # bits of a loop position
    past_bits = frames * bits
    out_bits = 2 * (frames + 1) * bits
    mem = f"{module_name}_mem"

    L = []
    a = L.append
    a(f"// generated by np2hw -- frame history: each pixel with its "
      f"{frames} predecessor{'s' if frames > 1 else ''}, {bits}-bit samples.")
    a("// out_data channel 0 (low bits) is the live pixel; channel k is the")
    a("// same pixel from k frames ago. One memory word per position holds")
    a("// the past, read one pixel ahead and written back shifted.")
    pins = [
        Pin("clk", "input"),
        Pin("rst", "input"),
        Pin("in_valid", "input"),
        Pin("in_ready", "output"),
        Pin("in_sof", "input"),
        Pin("in_data", "input", bits),
        Pin("out_valid", "output", reg=True),
        Pin("out_ready", "input"),
        Pin("out_sof", "output", reg=True),
        Pin("out_eol", "output", reg=True),
        Pin("out_last", "output", reg=True),
        Pin("out_data", "output", out_bits, reg=True),
    ]
    pins += [Pin(f"param_pos_{i}", "input", before=(
        "the live frame's position in the capture loop, bit by bit, as",
        "context: constant for the frame") if i == 0 else ())
        for i in range(pb)]
    pins += [Pin(f"param_tag_{j}", "input", bits, before=(
        "the tag of each loop position, written by software",)
        if j == 0 else ()) for j in range(loop)]
    pins.append(Pin("live_tag", "output", bits,
                    before=("the live frame's tag, for software to read",)))
    L += header(module_name, pins,
                parameters=[("WIDTH", width), ("HEIGHT", height)])
    a("    localparam DEPTH = WIDTH * HEIGHT;")
    a(f"    reg [{POS - 1}:0] col;")
    a(f"    reg [{POS - 1}:0] row;")
    a(f"    reg [{aw - 1}:0] addr;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a("    wire take = in_valid && !stall;")
    a("    // A frame start re-anchors the position to (0, 0), as in every core.")
    a(f"    wire [{POS - 1}:0] ecol = in_sof ? {POS}'d0 : col;")
    a(f"    wire [{POS - 1}:0] erow = in_sof ? {POS}'d0 : row;")
    a(f"    wire [{aw - 1}:0] eaddr = in_sof ? {aw}'d0 : addr;")
    a("    // The position this core will read NEXT, from the counter's own")
    a("    // next state: the address a cycle early, the word on time.")
    a(f"    wire [{aw - 1}:0] next_addr = in_sof ? {aw}'d1")
    a(f"                     : ((addr == DEPTH - 1) ? {aw}'d0 : addr + 1'b1);")
    a(f"    wire [{past_bits - 1}:0] past;      // this pixel's {frames} past samples")
    pos_bits = ", ".join(f"param_pos_{i}" for i in reversed(range(pb)))
    a(f"    wire [{pb - 1}:0] pos = {{{pos_bits}}};")
    a("    // Frame n-k sits k positions back in the loop, so its tag is")
    a("    // tag[(pos - k) mod loop]: a small mux on frame-constant values.")
    for k in range(frames + 1):
        a(f"    reg [{bits - 1}:0] tag{k};")
        a("    always @* begin")
        a("        case (pos)")
        for j in range(loop):
            a(f"            {pb}'d{j}: tag{k} = param_tag_{(j - k) % loop};")
        a(f"            default: tag{k} = {bits}'d0;")
        a("        endcase")
        a("    end")
    a("    assign live_tag = tag0;")
    a("    // Written back shifted: the live sample becomes frame n-1, and the")
    a("    // oldest falls off the top.")
    if frames > 1:
        a(f"    wire [{past_bits - 1}:0] shifted = "
          f"{{past[{(frames - 1) * bits - 1}:0], in_data}};")
    else:
        a(f"    wire [{past_bits - 1}:0] shifted = in_data;")
    a(f"    {mem} #(.W({past_bits}), .D({depth}), .AW({aw})) u_mem (")
    a("        .clk(clk), .we(take), .wa(eaddr), .wd(shifted),")
    a("        .re(take), .ra(next_addr), .rd(past));")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            col <= 0; row <= 0; addr <= 0;")
    a("            out_valid <= 1'b0; out_sof <= 1'b0; out_eol <= 1'b0;")
    a("            out_last <= 1'b0;")
    a("        end else if (!stall) begin")
    a("            if (in_valid) begin")
    a("                out_valid <= 1'b1;")
    tag_list = ", ".join(f"tag{k}" for k in reversed(range(frames + 1)))
    a(f"                out_data  <= {{{tag_list}, past, in_data}};")
    a("                out_sof   <= (erow == 0) && (ecol == 0);")
    a("                out_eol   <= (ecol == WIDTH - 1);")
    a("                out_last  <= (erow == HEIGHT - 1) && (ecol == WIDTH - 1);")
    a("                addr <= next_addr;")
    a("                if (in_sof) begin col <= 1; row <= 0; end")
    a("                else if (col == WIDTH - 1) begin")
    a("                    col <= 0; row <= (row == HEIGHT - 1) ? 0 : row + 1;")
    a("                end else col <= col + 1;")
    a("            end else begin")
    a("                out_valid <= 1'b0; out_sof <= 1'b0; out_eol <= 1'b0;")
    a("                out_last <= 1'b0;")
    a("            end")
    a("        end")
    a("    end")
    a("endmodule")
    a("")
    L.extend(_memory_module(mem, past_bits, depth, aw))

    interface = {
        "clock": "clk",
        "reset": "rst",
        "param_prefix": "param_",
        "ports": published(pins),
        "input": {"prefix": "in", "data_bits": bits, "flags": ("sof",)},
        "output": {"prefix": "out", "data_bits": out_bits, "signed": False,
                   "flags": ("sof", "eol", "last"),
                   "channels": 2 * (frames + 1), "channel_bits": bits},
        "params": ([(f"pos_{i}", 1, False) for i in range(pb)]
                   + [(f"tag_{j}", bits, False) for j in range(loop)]),
        # Same backward path as a traced core: in_ready is !stall, and stall
        # reads out_ready.
        "combinational_outputs": ("in_ready",),
    }
    interface["streams"] = [dict(interface["input"], direction="in"),
                            dict(interface["output"], direction="out")]
    return {
        "verilog": "\n".join(L),
        "module": module_name,
        "interface": interface,
        "memories": [{"instance": "u_mem", "module": mem, "bits": past_bits,
                      "depth": depth, "read": "sync"}],
        "frames": frames,
        "loop": loop,
        "channels": 2 * (frames + 1),
    }


def _memory_module(name: str, width: int, depth: int, aw: int) -> list[str]:
    """The history's memory as its own module: the seam a macro binds to."""
    return [
        f"// generated by np2hw -- frame history memory for {name[:-4]}: one",
        "// write port, one read port, both synchronous. THE MEMORY SEAM: bind",
        "// a memory macro here, by module name, and honour four things.",
        "//   1. write: wd lands at mem[wa] on the clk edge where we is high.",
        "//   2. read: rd <= mem[ra] on the clk edge where re is high -- one",
        "//      cycle of latency -- and rd HOLDS while re is low.",
        "//   3. collision: we and re in one cycle at one address read the OLD",
        "//      word. The history never does this -- its read runs one",
        "//      position ahead of its write -- but a binding must not rely",
        "//      on that silently.",
        "//   4. no reset: rd is data, not state. Nothing here clears it.",
        f"module {name} #(",
        f"    parameter W  = {width},",
        f"    parameter D  = {depth},",
        f"    parameter AW = {aw}",
        ") (",
        "    input  wire          clk,",
        "    input  wire          we,",
        "    input  wire [AW-1:0] wa,",
        "    input  wire [W-1:0]  wd,",
        "    input  wire          re,",
        "    input  wire [AW-1:0] ra,",
        "    output reg  [W-1:0]  rd",
        ");",
        "    reg [W-1:0] mem [0:D-1];",
        "    always @(posedge clk) begin",
        "        if (we) mem[wa] <= wd;",
        "        if (re) rd <= mem[ra];",
        "    end",
        "endmodule",
    ]
