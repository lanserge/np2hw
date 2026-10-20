# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""What every generated core shares: `generate`, and around it the module's
published facts (Core), its interface, its testbenches, its register files
and the wrappers that put it on a bus.

`generate` prints a model the compiler lowered (np2hw.core); the rest is
fixed structure, parameterised by the facts a core publishes, so it serves
every core alike.
"""
from __future__ import annotations

import textwrap
from dataclasses import replace as dc_replace

from .regmap import AddrMap, Reg


# Video geometry is 16 bits wherever it is stated -- the protocol
# header says so, and the register map repeats it. The position
# counters used to be 32-bit integers doing an 11-bit job, which put a
# 32-bit carry chain on a block RAM's address port and made the column
# counter, not the arithmetic, a stencil's critical path.
POS = 16


def _mul_hint(op, dev) -> str:
    """What a wire computing `op` carries so it is built as the model
    timed it: a multiply, the device's hard multiplier (its
    mul_attribute); anything else, nothing."""
    if op != "mul":
        return ""
    from .timing import device
    return device(dev).mul_attribute + " "


class Core(dict):
    """What `generate()` produces: a module, and what a composer needs to use it.

    A dict subclass rather than a bare dict, so the PUBLIC surface is documented
    and typed while np2hw's own wrappers -- switchboard, AXI-Stream video, the
    register files, the testbenches -- keep reading the detail keys they already
    read. The public surface is:

        verilog     the module text
        module      its name
        interface   its ports, so a composer never has to know the conventions
        line_buffers / shift_depth  the delay-line cost, named rather than
                    abbreviated: `M` and `N` are the compiler's private spelling
                    and nothing outside it should have to learn them

    Everything else in here (`image`, `M`, `N`, ...) is working detail of
    the compiler. It stays reachable because np2hw's own
    wrappers use it, but it is not a contract and an application reading it is
    reaching through the interface rather than using it.
    """

    @property
    def verilog(self) -> str:
        return self["verilog"]

    @property
    def module(self) -> str:
        return self["module"]

    @property
    def interface(self) -> dict:
        return self["interface"]

    @property
    def line_buffers(self) -> int:
        """Rows of storage this module needs -- the expensive resource."""
        return self["M"]

    @property
    def shift_depth(self) -> int:
        """Depth of the within-row shift register."""
        return self["N"]

    @property
    def out_bits(self) -> int:
        return self["out_bits"]

    @property
    def params(self) -> list:
        """(name, bits) per parameter port, in declaration order."""
        return self["params"]

    def __repr__(self) -> str:
        return (f"Core({self['module']!r}, {self['out_bits']}b out, "
                f"{self['M']} line buffer(s), shift depth {self['N']})")


def _interface(in_bits, out_bits, out_signed, params, in_flags=("sof",),
               out_flags=("sof", "eol", "last"), sink=False) -> dict:
    """What a generated module presents to whoever instantiates it.

    Emitted alongside the Verilog so that a composer does not have to KNOW the
    conventions -- which flags a core takes, which it regenerates, what its
    parameter ports are called. That knowledge belongs to whoever wrote the
    ports, and anywhere else it is a copy waiting to disagree.

    A core self-frames from its frame size -- its frame_width/frame_height
    ports, latched as each frame enters (_frame_params) -- so it takes only
    `sof` to re-anchor and regenerates the rest. `sink` marks a module that consumes a
    stream and produces none: it never stalls its source, so a composer may fan
    any number of them off one output for free.
    """
    stream_in = {"prefix": "in", "data_bits": in_bits, "flags": tuple(in_flags)}
    stream_out = (None if sink else
                  {"prefix": "out", "data_bits": out_bits,
                   "signed": out_signed, "flags": tuple(out_flags)})
    return {
        "clock": "clk",
        "reset": "rst",
        # How this module names its parameter ports. Stated rather than assumed:
        # a generated core prefixes them `param_`, a COMPOSED module names them
        # whatever its caller called them, and a composer that assumes one
        # convention cannot instantiate the other.
        "param_prefix": "param_",
        "input": stream_in,
        "output": stream_out,
        # EVERY stream, which `input`/`output` cannot say once there is more than
        # one of either. A wrapper that has to pass the streams straight through
        # needs the full list; a composer that nests this module needs the single
        # pair above. Both come from here, so they cannot describe it differently.
        "streams": ([dict(stream_in, direction="in")]
                    + ([] if stream_out is None
                       else [dict(stream_out, direction="out")])),
        "params": [(name, bits, signed) for name, bits, signed, _, _ in params],
        # What this module costs the modules around it. `out_*` leave the
        # last pipeline register, so the FORWARD boundary is clean. The
        # backward one is not: `in_ready` is `!stall` and `stall` reads
        # `out_ready`, so a chain of these passes `ready` straight
        # through, combinationally, from the sink to the source -- and it
        # gets one gate deeper with every block added to the pipeline.
        # Said out loud because it is invisible from any single module,
        # which is exactly why it survived this long.
        "combinational_outputs": () if sink else ("in_ready",),
    }


def _frame_params(image) -> list:
    """The frame-size ports every core declares: the frame's width and
    height, latched as each frame enters. The traced size is the most the
    core holds (a line buffer's depth, a frame store's) and the ports'
    default, so a testbench or a register file that leaves them at their
    defaults runs the frame the model was traced at. They are ordinary
    parameter ports, so a composer binds them as it binds every other."""
    return [("frame_width", POS, False, image.width,
             f"The frame's width in pixels, at most {image.width} (what the "
             "core was generated for); latched as each frame enters"),
            ("frame_height", POS, False, image.height,
             "The frame's height in lines; latched as each frame enters")]


def _frame_out_ports() -> list:
    """The frame size a core hands to the next: two output ports."""
    return ["    // the frame this core puts out, as it has it in force: what",
            "    // the next core latches as its own frame starts",
            f"    output wire [{POS-1}:0] out_frame_width,",
            f"    output wire [{POS-1}:0] out_frame_height,"]


# --------------------------------------------------------------------------- #
# Verilog module emission
# --------------------------------------------------------------------------- #


def _stamp_stream_geometry(meta, out_line):
    """Put the traced extents where a consumer reads a stream's facts:
    beside `data_bits` in the interface, which is where a composer looks,
    so it threads geometry the way it threads depth and channels instead of
    assuming it. Stated once, from the metadata the printer produces."""
    interface = meta.get("interface") if isinstance(meta, dict) else None
    if not interface:
        return meta
    image = getattr(out_line, "image", None)
    if image is not None and interface.get("input") is not None:
        interface["input"]["rows"] = int(image.height)
        interface["input"]["cols"] = int(image.width)
    out = interface.get("output")
    if out is not None and meta.get("out_rows") is not None:
        out["rows"] = int(meta["out_rows"])
        out["cols"] = int(meta["out_cols"])
    return meta


def generate(mod, module_name="np2hw_top", clk_ns=None, label=None,
             dev=None, sync_memory=False) -> dict:
    """Emit a core, with the traced geometry stated in its interface.

    `mod` is what `to_ir` returns: a model lowered by np2hw.core, lowered
    again here for the device `dev` with its stages cut to meet `clk_ns`.
    A budget no cut can meet -- one operation deeper than the clock -- is
    refused, naming `label` (the module name when None).

    sync_memory=True is the promise a core bound for silicon makes: every
    memory it infers reads through a register, the only kind a memory macro
    implements. Every memory np2hw.core builds does, so the promise is
    checked against the memories the core publishes, and a core that broke
    it would be refused rather than emitted.

    Every core takes its frame's size at run time, on its frame_width and
    frame_height ports, and hands the size it puts out on: the size the
    model was traced at is the most it holds (_frame_params).
    """
    from .core import emit
    from .core.pipeline import CoreModule, stage_ns
    if not isinstance(mod, CoreModule):
        raise TypeError(f"{module_name}: generate() takes what to_ir returns, "
                        f"not {type(mod).__name__}")
    name = label or module_name
    try:
        built = mod.for_build(dev, clk_ns)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    if clk_ns is not None and stage_ns(built, dev) > clk_ns:
        raise ValueError(
            f"{name}: its output stage estimates {stage_ns(built, dev):.2f} ns "
            f"against the {clk_ns:.2f} ns budget, past what the stages were "
            "cut to -- the schedule and the cost model disagree, which is "
            "np2hw's defect, not the model's")
    meta = emit.generate(built.unit, module_name, dev)
    from .core.pipeline import control_ns
    # the control's worst path, by the model: reported, not gated
    meta["control_path"] = control_ns(built, dev)
    if sync_memory:
        loose = [m["instance"] for m in meta["memories"] if m["read"] != "sync"]
        if loose:
            raise ValueError(f"{name}: memories {loose} read combinationally; "
                             "sync_memory=True promises none does")
    return _stamp_stream_geometry(meta, built)


# --------------------------------------------------------------------------- #
# Testbench emission
# --------------------------------------------------------------------------- #

def _bench_value(meta, name, param_values, W, H) -> int:
    """The value a bench drives a register port with: the caller's, else
    -- for the frame's size -- the size of the frame the bench streams,
    else the port's own default."""
    if name in param_values:
        return int(param_values[name])
    frame = {"frame_width": W, "frame_height": H}
    if name in frame:
        return frame[name]
    return int(meta.get("param_defaults", {}).get(name, 0))


def testbench(meta, W, H, param_values=None, tb_name="tb") -> str:
    param_values = param_values or {}
    in_bits, out_bits = meta["in_bits"], meta["out_bits"]
    mod = meta["module"]
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a(f"module {tb_name};")
    a("    reg clk = 0, rst = 1, in_valid = 0, in_sof = 0;")
    a(f"    reg  [{in_bits-1}:0] in_data;")
    for name, bits in meta["params"]:
        a(f"    reg  [{bits-1}:0] param_{name};")
    a("    wire out_valid; wire in_ready; wire out_last; reg out_ready = 1'b1;")
    a(f"    wire {'signed ' if meta.get('signed') else ''}[{out_bits-1}:0] out_data;")
    a("    integer i, f;")
    a(f"    reg [{in_bits-1}:0] img [0:{W*H-1}];")
    a("    always #5 clk = ~clk;")
    conns = ["        .clk(clk), .rst(rst),",
             "        .in_valid(in_valid), .in_ready(in_ready), .in_sof(in_sof), .in_data(in_data),"]
    for name, _ in meta["params"]:
        conns.append(f"        .param_{name}(param_{name}),")
    conns.append("        .out_valid(out_valid), .out_ready(out_ready), "
                 ".out_last(out_last), .out_data(out_data));")
    a(f"    {mod} dut (")
    L.extend(conns)
    a("    always @(posedge clk) if (out_valid) $fdisplay(f, \"%0d\", out_data);")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        f = $fopen(\"out.txt\", \"w\");")
    for name, _ in meta["params"]:
        a(f"        param_{name} = "
          f"{_bench_value(meta, name, param_values, W, H)};")
    a("        @(negedge clk); rst = 0;")
    # the frame back to back, each pixel held until the core takes it (a
    # window holds its input while it flushes a line's end), then the
    # drain the core publishes: what it needs, past the last pixel, to
    # finish the frame
    a(f"        for (i = 0; i < {W*H}; i = i + 1) begin")
    a("            in_data = img[i]; in_sof = (i == 0); in_valid = 1;")
    a("            @(posedge clk); while (!in_ready) @(posedge clk); @(negedge clk);")
    a("        end")
    drain = max(meta.get("flush_cycles", 2), meta.get("vdrain", 0))
    a(f"        in_valid = 0; in_sof = 0; for (i = 0; i < {drain}; i = i + 1) @(negedge clk);")
    a("        $fclose(f); $finish;")
    a("    end")
    a("endmodule")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Switchboard adapter (separate, optional) — instances the generic core
# --------------------------------------------------------------------------- #

def switchboard_wrap(meta, W, H, module_name=None, dest=0, pack=False,
                     native=False) -> dict:
    """Wrap the generic ready/valid core in Switchboard SB ports.

    pack=False (default): thin pass-through, one pixel in the low bits of the
    416-bit payload. Simple; fine for small validation images.

    pack=True: a gearbox for real video throughput -- unpack P_in pixels/packet
    -> feed the core one/clock -> repack P_out pixels/packet. Backpressure both
    ways; 'last' frames each output FRAME (final packet, padded). The core is
    untouched in both cases (separate adapter).

    native=True: use Switchboard's exact SB port bundle (<p>_data/_dest/_last/
    _valid/_ready, 1-bit `last`) so the module drops straight into SbDut autowrap
    / `SB_CONNECT`. native=False keeps the 32-bit `_flags` form (last in bit 0)."""
    if pack:
        return _switchboard_wrap_pack(meta, W, H, module_name, dest, native=native)
    core = meta["module"]
    top = module_name or core + "_sb"
    ib, ob = meta["in_bits"], meta["out_bits"]
    params = meta["params"]
    DW = 416
    pports = "".join(f"\n    , input wire [{b-1}:0] param_{n}" for n, b in params)
    pconns = "".join(f"\n        .param_{n}(param_{n})," for n, _ in params)
    pad = f"{{{DW - ob}{{1'b0}}}}"                        # zero-extend pixel to payload
    if native:                                            # 1-bit `last` (SB_CONNECT)
        fin = "    input  wire        sb_in_last,"
        fout = "    output wire        sb_out_last"
        fassign = "    assign sb_out_last = core_last;   // 'last' frames the output"
    else:                                                 # 32-bit `flags` (last in [0])
        fin = "    input  wire [31:0] sb_in_flags,"
        fout = "    output wire [31:0] sb_out_flags"
        fassign = "    assign sb_out_flags = {31'b0, core_last};   // 'last' frames the output"
    verilog = f"""// generated by np2hw -- Switchboard adapter for {core} (separate module)
module {top} (
    input  wire clk,
    input  wire rst,
    // Switchboard input port
    output wire        sb_in_ready,
    input  wire        sb_in_valid,
    input  wire [{DW-1}:0] sb_in_data,
    input  wire [31:0] sb_in_dest,
{fin}
    // Switchboard output port
    input  wire        sb_out_ready,
    output wire        sb_out_valid,
    output wire [{DW-1}:0] sb_out_data,
    output wire [31:0] sb_out_dest,
{fout}{pports}
);
    wire [{ob-1}:0] core_out;
    wire core_last;
    {core} u_core (
        .clk(clk), .rst(rst),
        .in_valid(sb_in_valid), .in_ready(sb_in_ready), .in_sof(1'b0),
        .in_data(sb_in_data[{ib-1}:0]),{pconns}
        .out_valid(sb_out_valid), .out_ready(sb_out_ready),
        .out_last(core_last), .out_data(core_out)
    );
    assign sb_out_data  = {{{pad}, core_out}};
    assign sb_out_dest  = 32'd{int(dest)};
{fassign}
endmodule
"""
    return {
        "verilog": verilog,
        "module": top,
        "core": core,
        "in_bits": ib, "out_bits": ob,
        "params": params,
        "signed": meta.get("signed"),
        "native": native,
    }


def _switchboard_wrap_pack(meta, W, H, module_name, dest, native=False) -> dict:
    """Switchboard adapter with pixel packing (gearbox). See switchboard_wrap."""
    core = meta["module"]
    top = module_name or core + "_sb"
    ib, ob = meta["in_bits"], meta["out_bits"]
    params = meta["params"]
    DW = 416
    p_in = DW // ib
    p_out = DW // ob
    frame_in = W * H
    frame_out = meta["out_rows"] * meta["out_cols"]
    L = []
    a = L.append
    a(f"// generated by np2hw -- Switchboard PACKED adapter for {core} "
      f"({p_in} px/in-packet, {p_out} px/out-packet)")
    a(f"module {top} (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    output wire        sb_in_ready,")
    a("    input  wire        sb_in_valid,")
    a(f"    input  wire [{DW-1}:0] sb_in_data,")
    a("    input  wire [31:0] sb_in_dest,")
    a("    input  wire        sb_in_last," if native else "    input  wire [31:0] sb_in_flags,")
    a("    input  wire        sb_out_ready,")
    a("    output wire        sb_out_valid,")
    a(f"    output wire [{DW-1}:0] sb_out_data,")
    a("    output wire [31:0] sb_out_dest,")
    a("    output wire        sb_out_last" if native else "    output wire [31:0] sb_out_flags")
    for name, bits in params:
        a(f"    , input wire [{bits-1}:0] param_{name}")
    a(");")
    # -- input depacketizer: hold one packet, feed core one pixel/clock --------
    a(f"    reg [{DW-1}:0] inbuf; integer ipos; reg in_loaded; integer iframe;")
    a(f"    wire [{ib-1}:0] in_pix = inbuf[ipos*{ib} +: {ib}];")
    a("    wire core_iv = in_loaded;")
    a("    wire core_ir;")                              # core in_ready
    a("    assign sb_in_ready = !in_loaded;")
    # -- output packetizer: collect core pixels, emit a packet -----------------
    a(f"    reg [{DW-1}:0] outbuf; integer opos; reg out_full; reg out_lastp;")
    a(f"    wire core_ov; wire [{ob-1}:0] core_od; wire core_ol;")
    a("    wire core_or = !out_full;")                  # core out_ready
    a("    assign sb_out_valid = out_full;")
    a("    assign sb_out_data  = outbuf;")
    a(f"    assign sb_out_dest  = 32'd{int(dest)};")
    a("    assign sb_out_last  = out_lastp;" if native
      else "    assign sb_out_flags = {31'b0, out_lastp};")
    a(f"    {core} u_core (")
    a("        .clk(clk), .rst(rst),")
    a("        .in_valid(core_iv), .in_ready(core_ir), .in_sof(1'b0), .in_data(in_pix),")
    for name, _ in params:
        a(f"        .param_{name}(param_{name}),")
    a("        .out_valid(core_ov), .out_ready(core_or),")
    a("        .out_sof(), .out_eol(), .out_last(core_ol), .out_data(core_od)")
    a("    );")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            ipos<=0; in_loaded<=1'b0; iframe<=0;")
    a("            opos<=0; out_full<=1'b0; out_lastp<=1'b0; outbuf<=0;")
    a("        end else begin")
    # input: load a packet, or consume a pixel into the core
    a("            if (sb_in_valid && sb_in_ready) begin")
    a("                inbuf <= sb_in_data; ipos <= 0; in_loaded <= 1'b1;")
    a("            end else if (core_iv && core_ir) begin")
    a(f"                if (iframe == {frame_in - 1}) begin")
    a("                    iframe <= 0; in_loaded <= 1'b0;")  # frame done -> drop tail
    a("                end else begin")
    a("                    iframe <= iframe + 1;")
    a(f"                    if (ipos == {p_in - 1}) in_loaded <= 1'b0;")  # packet drained
    a("                    else ipos <= ipos + 1;")
    a("                end")
    a("            end")
    # output: send the held packet, and/or pack the next core pixel
    a("            if (out_full && sb_out_ready) out_full <= 1'b0;")
    a("            if (core_ov && core_or) begin")
    a(f"                outbuf[opos*{ob} +: {ob}] <= core_od;")
    a(f"                if (core_ol || opos == {p_out - 1}) begin")
    a("                    out_full <= 1'b1; out_lastp <= core_ol; opos <= 0;")
    a("                end else opos <= opos + 1;")
    a("            end")
    a("        end")
    a("    end")
    a("endmodule")
    return {
        "verilog": "\n".join(L),
        "module": top,
        "core": core,
        "in_bits": ib, "out_bits": ob,
        "params": params,
        "signed": meta.get("signed"),
        "pack": True, "p_in": p_in, "p_out": p_out, "native": native,
        "out_rows": meta.get("out_rows"), "out_cols": meta.get("out_cols"),
    }


def switchboard_control_wrap(meta, W, H, module_name=None, dest=0, addr_bits=8) -> dict:
    """Combined SB datapath (packed gearbox) + AXI-Lite control register file, so
    config registers can be set/changed at RUNTIME over a separate SB control
    interface (PySbTx/PySbRx for pixels, AxiLiteTxRx for registers) -- e.g. live
    slider tuning while pixels stream. Emits three modules (regfile + gearbox +
    top) wired together; ports match SbDut autowrap's SB_CONNECT (sb_in/sb_out)
    and SB_AXIL_CONNECT (ctrl)."""
    core = meta["module"]
    top = module_name or core + "_sbc"
    params = meta["params"]
    DW, A = 416, addr_bits
    reg = axil_regfile(params, module_name=top + "_rf", addr_bits=A,
                       defaults=meta.get("param_defaults"))
    gb = _switchboard_wrap_pack(meta, W, H, module_name=top + "_gb",
                                dest=dest, native=True)
    pdecl = "".join(f"    wire [{b-1}:0] param_{n};\n" for n, b in params)
    rf_p = "".join(f"\n        , .param_{n}(param_{n})" for n, _ in params)
    gb_p = "".join(f"\n        , .param_{n}(param_{n})" for n, _ in params)
    L = []
    a = L.append
    a(reg["verilog"]); a(""); a(gb["verilog"]); a("")
    a("// generated by np2hw -- SB datapath + AXI-Lite control (live registers)")
    a(f"module {top} (")
    a("    input  wire clk, input wire rst,")
    a("    output wire        sb_in_ready, input wire sb_in_valid,")
    a(f"    input  wire [{DW-1}:0] sb_in_data, input wire [31:0] sb_in_dest, input wire sb_in_last,")
    a("    input  wire        sb_out_ready, output wire sb_out_valid,")
    a(f"    output wire [{DW-1}:0] sb_out_data, output wire [31:0] sb_out_dest, output wire sb_out_last,")
    a(f"    input  wire [{A-1}:0] ctrl_awaddr, input wire [2:0] ctrl_awprot, input wire ctrl_awvalid, output wire ctrl_awready,")
    a("    input  wire [31:0] ctrl_wdata, input wire [3:0] ctrl_wstrb, input wire ctrl_wvalid, output wire ctrl_wready,")
    a("    output wire [1:0] ctrl_bresp, output wire ctrl_bvalid, input wire ctrl_bready,")
    a(f"    input  wire [{A-1}:0] ctrl_araddr, input wire [2:0] ctrl_arprot, input wire ctrl_arvalid, output wire ctrl_arready,")
    a("    output wire [31:0] ctrl_rdata, output wire [1:0] ctrl_rresp, output wire ctrl_rvalid, input wire ctrl_rready")
    a(");")
    if pdecl:
        a(pdecl.rstrip("\n"))
    a(f"    {reg['module']} u_rf (")          # AXI-Lite ctrl -> param_* (awprot/arprot unused)
    a("        .aclk(clk), .aresetn(!rst),")
    a("        .s_axil_awaddr(ctrl_awaddr), .s_axil_awvalid(ctrl_awvalid), .s_axil_awready(ctrl_awready),")
    a("        .s_axil_wdata(ctrl_wdata), .s_axil_wstrb(ctrl_wstrb), .s_axil_wvalid(ctrl_wvalid), .s_axil_wready(ctrl_wready),")
    a("        .s_axil_bresp(ctrl_bresp), .s_axil_bvalid(ctrl_bvalid), .s_axil_bready(ctrl_bready),")
    a("        .s_axil_araddr(ctrl_araddr), .s_axil_arvalid(ctrl_arvalid), .s_axil_arready(ctrl_arready),")
    a("        .s_axil_rdata(ctrl_rdata), .s_axil_rresp(ctrl_rresp), .s_axil_rvalid(ctrl_rvalid), .s_axil_rready(ctrl_rready)"
      + rf_p)
    a("    );")
    a(f"    {gb['module']} u_gb (")           # packed SB datapath, params from regfile
    a("        .clk(clk), .rst(rst),")
    a("        .sb_in_ready(sb_in_ready), .sb_in_valid(sb_in_valid), .sb_in_data(sb_in_data),")
    a("        .sb_in_dest(sb_in_dest), .sb_in_last(sb_in_last),")
    a("        .sb_out_ready(sb_out_ready), .sb_out_valid(sb_out_valid), .sb_out_data(sb_out_data),")
    a("        .sb_out_dest(sb_out_dest), .sb_out_last(sb_out_last)" + gb_p)
    a("    );")
    a("endmodule")
    return {
        "verilog": "\n".join(L), "module": top, "core": core,
        "in_bits": meta["in_bits"], "out_bits": meta["out_bits"],
        "params": params, "p_in": gb["p_in"], "p_out": gb["p_out"],
        "out_rows": meta["out_rows"], "out_cols": meta["out_cols"],
        "offsets": reg["offsets"], "addr_bits": A, "signed": meta.get("signed"),
    }


# --------------------------------------------------------------------------- #
# Control-plane register files — drive the core's param_* config registers.
# Each register maps to a word address; software writes it over a bus.
# --------------------------------------------------------------------------- #

def _normalise_regs(params, defaults=None, addr_bits=8) -> list[Reg]:
    """Accept `(name, bits)` tuples or :class:`Reg` values; assign any gaps.

    Both forms exist so that the single-core path (`control_top`, the examples)
    keeps its terse tuples while a caller with its own address map passes Regs.
    An :class:`AddrMap` -- the hierarchical form -- flattens here, at the one
    point an emitter genuinely needs the flat list. They all end up as the same
    list, so there is one emitter and not three.
    """
    if isinstance(params, AddrMap):
        params = params.flatten()
    dv = defaults or {}
    out: list[Reg] = []
    cursor = 0
    for entry in params:
        reg = entry if isinstance(entry, Reg) else Reg(entry[0], entry[1])
        if reg.name in dv:
            reg = dc_replace(reg, reset=int(dv[reg.name]))
        if reg.offset is None:
            reg = dc_replace(reg, offset=cursor)
        cursor = reg.offset + 4
        out.append(reg)

    seen: dict[int, str] = {}
    top = 1 << addr_bits
    for reg in out:
        if not 1 <= reg.bits <= 32:
            raise ValueError(
                f"register {reg.name!r}: {reg.bits} bits does not fit one 32-bit "
                "bus word")
        if reg.access not in ("rw", "ro"):
            raise ValueError(
                f"register {reg.name!r}: access {reg.access!r} must be 'rw' or 'ro'")
        if reg.offset % 4:
            raise ValueError(
                f"register {reg.name!r}: offset 0x{reg.offset:x} is not word-aligned; "
                "this bus addresses 32-bit words")
        if reg.offset in seen:
            raise ValueError(
                f"registers {seen[reg.offset]!r} and {reg.name!r} both claim offset "
                f"0x{reg.offset:x}")
        if not 0 <= reg.offset < top:
            raise ValueError(
                f"register {reg.name!r} at 0x{reg.offset:x} is outside the "
                f"{addr_bits}-bit address space this file decodes (0x0..0x{top-1:x})")
        seen[reg.offset] = reg.name
    return out


def _read_expr(reg: Reg, source: str) -> str:
    """The 32-bit read-back value for one register.

    Sign-extended when the register is signed: a host that reads a negative
    black-level offset back as a large positive number has been told a different
    thing from what it wrote, and it is the register file that decided the width.
    """
    if reg.bits == 32:
        return source
    if reg.signed:
        return f"{{{{{32 - reg.bits}{{{source}[{reg.bits-1}]}}}}, {source}}}"
    return f"{{{32 - reg.bits}'b0, {source}}}"


def _clock_port(name, busifs, reset_name=None) -> list[str]:
    """A clock port that names the buses it drives, and their reset."""
    from .stream import clock_attribute
    return [f"    {clock_attribute(name, busifs, reset_name)}",
            f"    input  wire {name},"]


def _reset_port(name, active_low) -> list[str]:
    """A reset port that states its polarity rather than implying it."""
    from .stream import reset_attribute
    return [f"    {reset_attribute(name, active_low)}",
            f"    input  wire {name},"]


def _axil_slave_ports(addr_bits, storage="wire", bundle="s_axil",
                      prefix="s_axil", trailing_comma=False) -> list[str]:
    """The AXI4-Lite slave port list, every port DECLARING its bundle.

    An integrator told nothing has to GUESS the bus from the port names.
    IP Integrator guesses well -- until one name is not what it expects,
    and then it offers seventeen loose pins where an interface should
    be. That failure reads as a wiring mistake rather than as a missing
    declaration, which is what makes it expensive. The attributes say it
    outright: the same lesson as reset polarity, stated at the boundary
    that owns the fact instead of at whatever glue consumes it.

    These are Verilog attributes. Simulators ignore them, and so does
    every synthesiser that is not packaging IP; the tools that read them
    stop guessing. An IP-XACT packager reads them too, so declaring here
    is the prerequisite for that packaging rather than a detour from it.
    """
    out = []
    pins = axil_slave_pins(addr_bits, storage, bundle, prefix)
    from .ports import bus_attribute
    for i, pin in enumerate(pins):
        out.append(f"    {bus_attribute(*pin.bus)}")
        span = pin.range()
        tail = "," if trailing_comma or i < len(pins) - 1 else ""
        out.append(f"    {pin.direction:<6} {'reg' if pin.reg else 'wire'} "
                   f"{span + ' ' if span else ''}{pin.name}{tail}")
    return out


def axil_slave_pins(addr_bits, storage="wire", bundle="s_axil",
                    prefix="s_axil") -> list:
    """The AXI4-Lite slave's ports as :class:`~np2hw.ports.Pin` values,
    each declaring its bundle -- the one list both the port declarations
    and a published port list are made from."""
    from .ports import Pin
    A = addr_bits
    reg = storage == "reg"
    table = [
        ("AWADDR", "input", A, False), ("AWVALID", "input", 1, False),
        ("AWREADY", "output", 1, reg), ("WDATA", "input", 32, False),
        ("WSTRB", "input", 4, False), ("WVALID", "input", 1, False),
        ("WREADY", "output", 1, reg), ("BRESP", "output", 2, reg),
        ("BVALID", "output", 1, reg), ("BREADY", "input", 1, False),
        ("ARADDR", "input", A, False), ("ARVALID", "input", 1, False),
        ("ARREADY", "output", 1, reg), ("RDATA", "output", 32, reg),
        ("RRESP", "output", 2, reg), ("RVALID", "output", 1, reg),
        ("RREADY", "input", 1, False),
    ]
    return [Pin(f"{prefix}_{sig.lower()}", direction, width, reg=r,
                bus=("xilinx.com:interface:aximm:1.0", bundle, sig))
            for sig, direction, width, r in table]


def axil_regfile(params, module_name="np2hw_axil", addr_bits=8,
                 frame_sync=False, defaults=None, arm=None) -> dict:
    """AXI4-Lite slave exposing one register per entry of `params`.

    `params` is a list of `(name, bits)` -- packed at word offset i*4 -- or of
    :class:`Reg` values, which carry their own offset, signedness, reset, access
    and description. Output `param_<name>` wires drive the core.

    frame_sync=True adds shadow registers + an `update` input: writes land in a
    shadow, and the live value (driving the core) copies from the shadow on an
    `update` pulse -- wire `update` to the core's out_last/SOF for glitch-free,
    frame-aligned coefficient updates (no mid-frame tearing).

    arm=<register name> makes that register an ARM, for a reader in
    ANOTHER clock domain. Software writes the coefficients, then writes
    1 to the arm; from that instant this file REFUSES coefficient writes
    (SLVERR, answered at once) and raises `arm_req`. The reader takes
    its copy whenever it likes and raises `arm_ack`, which clears the
    arm and lets writes through again. Reading the arm returns whether
    it is still set, so software polls rather than guesses.

    Why refuse rather than stall: delaying `awready` until the reader is
    ready would hold the bus for up to a frame, and a processor waiting
    on a bus that long is a processor that has hung. SLVERR answers
    immediately and says why.

    What this buys, and it is the whole point: while armed, the values
    provably cannot change, so the reader may sample them from any clock
    at any moment. Only one bit crosses in each direction -- the
    register values never cross -- so there is no wide crossing to get
    wrong. And a slow crossing makes the update land a frame later,
    never a torn value: timing changes WHEN, not WHETHER.

    It also makes a multi-register update atomic. Writing five of nine
    matrix coefficients either side of a frame boundary otherwise
    applies half a matrix."""
    A = addr_bits
    regs = _normalise_regs(params, defaults, addr_bits=A)
    rw = [r for r in regs if r.writable]
    wreg = "shadow_" if frame_sync else REGFILE_WRITTEN   # software-written reg
    dreg = "live_" if frame_sync else REGFILE_WRITTEN     # drives param_*
    L = []
    a = L.append
    a("// generated by np2hw -- AXI4-Lite control register file"
      + (" (frame-synced shadow)" if frame_sync else ""))
    from .ports import Pin, header
    from .stream import clock_attribute, reset_attribute
    pins = [
        Pin("aclk", "input",
            attribute=clock_attribute("aclk", "s_axil", "aresetn")),
        Pin("aresetn", "input",
            attribute=reset_attribute("aresetn", active_low=True)),
    ]
    if arm:
        pins += [
            Pin("arm_req", "output", before=(
                "The reader lives in another clock domain. While `arm_req`",
                "is high no write commits, so the values it samples cannot",
                "change underneath it; `arm_ack` says it has taken them.",
                "One bit each way. The values never cross.")),
            Pin("arm_ack", "input"),
        ]
    if frame_sync:
        pins.append(Pin("update", "input",
                        note="pulse (e.g. frame boundary): shadow -> live"))
    pins += axil_slave_pins(A, storage="reg")
    for reg in rw:
        about = (tuple(textwrap.wrap(" ".join(str(reg.description).split()),
                                     72)) if reg.description else ())
        pins.append(Pin(f"param_{reg.name}", "output", reg.bits,
                        signed=reg.signed, before=about))
    L += header(module_name, pins)
    for reg in regs:
        if not reg.writable:
            a(f"    // @ 0x{reg.offset:04x}  {reg.name} = 32'h{reg.value & 0xFFFFFFFF:08x}"
              " (read-only)")
            continue
        a(f"    reg {'signed ' if reg.signed else ''}[{reg.bits-1}:0] "
          f"{wreg}{reg.name};   // @ 0x{reg.offset:04x}")
        if frame_sync:
            a(f"    reg {'signed ' if reg.signed else ''}[{reg.bits-1}:0] "
              f"{dreg}{reg.name};")
        a(f"    assign param_{reg.name} = {dreg}{reg.name};")
    if arm:
        # After the register declarations: the arm IS its register, so
        # it has to exist before this reads it.
        a("    // The arm IS its register: software sets it, the reader's")
        a("    // acknowledgement clears it, and it reads back so software")
        a("    // can poll. Two flops on the ack because it was launched by")
        a("    // a clock that has no relation to this one.")
        a(f"    assign arm_req = {wreg}{arm};")
        a(f"    (* ASYNC_REG = \"TRUE\" *) reg {REGFILE_ACK_SYNC}, ack_s1;")
    a(f"    reg [{A-1}:0] awaddr_q; reg [{A-1}:0] araddr_q; reg aw_en;")
    # write channel
    a("    always @(posedge aclk) begin")
    a("        if (!aresetn) begin")
    a("            s_axil_awready<=0; s_axil_wready<=0; s_axil_bvalid<=0;")
    a("            s_axil_bresp<=0; aw_en<=1;")
    if arm:
        a(f"            {REGFILE_ACK_SYNC}<=0; ack_s1<=0;")
    for reg in rw:
        a(f"            {wreg}{reg.name}<={int(reg.reset)};")
    a("        end else begin")
    if arm:
        a(f"            {REGFILE_ACK_SYNC} <= arm_ack; "
          f"ack_s1 <= {REGFILE_ACK_SYNC};")
        a("            // The reader has its copy: let writes through again.")
        a("            // Placed BEFORE the decode so the decode, being")
        a("            // later, wins -- an arm written in the very cycle an")
        a("            // ack lands still takes, rather than being dropped.")
        a(f"            if (ack_s1) {wreg}{arm} <= 1'b0;")
    a("            if (!s_axil_awready && s_axil_awvalid && s_axil_wvalid && aw_en) begin")
    a("                s_axil_awready<=1; awaddr_q<=s_axil_awaddr; aw_en<=0;")
    a("            end else if (s_axil_bready && s_axil_bvalid) begin")
    a("                aw_en<=1; s_axil_awready<=0;")
    a("            end else s_axil_awready<=0;")
    a("            if (!s_axil_wready && s_axil_wvalid && s_axil_awvalid && aw_en)")
    a("                s_axil_wready<=1; else s_axil_wready<=0;")
    a("            if (s_axil_awready && s_axil_awvalid && s_axil_wready && s_axil_wvalid) begin")
    a("                s_axil_bvalid<=1; s_axil_bresp<=2'b10;   // SLVERR unless decoded")
    a(f"                case (awaddr_q[{A-1}:2])")
    for reg in rw:
        # The arm itself is always writable -- refusing it would leave
        # software no way to say it had finished.
        guard = (f"if (!{wreg}{arm}) " if (arm and reg.name != arm) else "")
        tail = ("" if not arm else
                "   // refused while armed: see arm_req")
        a(f"                    {reg.word}: {guard}begin {wreg}{reg.name} <= "
          f"s_axil_wdata[{reg.bits-1}:0]; s_axil_bresp<=2'b00; end{tail}")
    a("                    default: ;   // unmapped, or a write to a read-only word")
    a("                endcase")
    a("            end else if (s_axil_bready && s_axil_bvalid) s_axil_bvalid<=0;")
    a("        end")
    a("    end")
    if frame_sync:                                        # shadow -> live at update
        a("    always @(posedge aclk) begin")
        a("        if (!aresetn) begin")
        for reg in rw:
            a(f"            {dreg}{reg.name}<={int(reg.reset)};")
        a("        end else if (update) begin")
        for reg in rw:
            a(f"            {dreg}{reg.name} <= {wreg}{reg.name};")
        a("        end")
        a("    end")
    # read channel (reads the written/shadow value)
    a("    always @(posedge aclk) begin")
    a("        if (!aresetn) begin s_axil_arready<=0; s_axil_rvalid<=0; s_axil_rresp<=0; araddr_q<=0; s_axil_rdata<=0; end")
    a("        else begin")
    a("            if (!s_axil_arready && s_axil_arvalid) begin s_axil_arready<=1; araddr_q<=s_axil_araddr; end")
    a("            else s_axil_arready<=0;")
    a("            if (s_axil_arready && s_axil_arvalid && !s_axil_rvalid) begin")
    a("                s_axil_rvalid<=1; s_axil_rresp<=2'b10;   // SLVERR unless decoded")
    a(f"                case (araddr_q[{A-1}:2])")
    for reg in regs:
        if not reg.writable:
            value = f"32'h{reg.value & 0xFFFFFFFF:08x}"
        else:
            value = _read_expr(reg, f"{wreg}{reg.name}")
        a(f"                    {reg.word}: begin s_axil_rdata <= {value}; "
          "s_axil_rresp<=2'b00; end")
    a("                    default: s_axil_rdata <= 0;")
    a("                endcase")
    a("            end else if (s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;")
    a("        end")
    a("    end")
    a("endmodule")
    from .ports import published
    return {"verilog": "\n".join(L), "module": module_name,
            "interface": {"clock": "aclk", "reset": "aresetn",
                          "ports": published(pins)},
            "params": [(r.name, r.bits) for r in rw], "regs": regs,
            "addr_bits": A, "frame_sync": frame_sync,
            "offsets": {r.name: r.offset for r in regs}}


def umi_regfile(params, module_name="np2hw_umi", addr_bits=8,
                write_op=1, read_op=2, frame_sync=False, defaults=None) -> dict:
    """SIMPLIFIED UMI register slave for the Switchboard stack: a request channel
    (valid/ready + opcode + addr + 32-bit data) writes/reads one register per
    register; reads produce a response. opcode = udev_req_cmd[4:0].
    frame_sync=True adds shadow registers + an `update` input (see axil_regfile).

    NOTE: this models UMI write/read register semantics but is NOT the full UMI
    command encoding. For production, use switchboard's umi_endpoint/umi_regif and
    connect the param_* outputs to it. `params` takes the same two forms
    axil_regfile does -- `(name, bits)` tuples packed at i*4, or Reg values with
    their own offsets -- because the address map is the caller's either way, and
    two emitters that assign offsets differently is exactly the disagreement Reg
    exists to prevent."""
    A = addr_bits
    regs = _normalise_regs(params, defaults, addr_bits=A)
    rw = [r for r in regs if r.writable]
    wreg = "shadow_" if frame_sync else "reg_"
    dreg = "live_" if frame_sync else "reg_"
    L = []
    a = L.append
    a("// generated by np2hw -- simplified UMI register file (see docstring)"
      + (" (frame-synced shadow)" if frame_sync else ""))
    a(f"module {module_name} (")
    a("    input  wire clk, input wire nreset,")
    if frame_sync:
        a("    input  wire update,   // pulse (frame boundary): shadow -> live")
    a("    input  wire        udev_req_valid,")
    a("    output wire        udev_req_ready,")
    a("    input  wire [4:0]  udev_req_cmd,")          # opcode (simplified)
    a(f"    input  wire [{A-1}:0] udev_req_addr,")
    a("    input  wire [31:0] udev_req_data,")
    a("    output reg         udev_resp_valid,")
    a("    input  wire        udev_resp_ready,")
    a("    output reg  [31:0] udev_resp_data")
    for reg in rw:
        if reg.description:
            for line in textwrap.wrap(" ".join(str(reg.description).split()), 72):
                a(f"    // {line}")
        a(f"    , output wire {'signed ' if reg.signed else ''}"
          f"[{reg.bits-1}:0] param_{reg.name}")
    a(");")
    for reg in regs:
        if not reg.writable:
            a(f"    // @ 0x{reg.offset:04x}  {reg.name} = 32'h{reg.value & 0xFFFFFFFF:08x}"
              " (read-only)")
            continue
        a(f"    reg {'signed ' if reg.signed else ''}[{reg.bits-1}:0] "
          f"{wreg}{reg.name};   // @ 0x{reg.offset:04x}")
        if frame_sync:
            a(f"    reg {'signed ' if reg.signed else ''}[{reg.bits-1}:0] "
              f"{dreg}{reg.name};")
        a(f"    assign param_{reg.name} = {dreg}{reg.name};")
    a("    assign udev_req_ready = !udev_resp_valid;")
    a(f"    wire is_write = udev_req_cmd[4:0] == 5'd{write_op};")
    a(f"    wire is_read  = udev_req_cmd[4:0] == 5'd{read_op};")
    a("    always @(posedge clk) begin")
    a("        if (!nreset) begin")
    a("            udev_resp_valid<=0; udev_resp_data<=0;")
    for reg in rw:
        a(f"            {wreg}{reg.name}<={int(reg.reset)};")
    a("        end else begin")
    a("            if (udev_req_valid && udev_req_ready) begin")
    a("                if (is_write) begin")
    a(f"                    case (udev_req_addr[{A-1}:2])")
    for reg in rw:
        a(f"                        {reg.word}: {wreg}{reg.name} <= "
          f"udev_req_data[{reg.bits-1}:0];")
    a("                        default: ;   // unmapped, or read-only")
    a("                    endcase")
    a("                end else if (is_read) begin")
    a("                    udev_resp_valid <= 1'b1;")
    a(f"                    case (udev_req_addr[{A-1}:2])")
    for reg in regs:
        rd = (f"32'h{reg.value & 0xFFFFFFFF:08x}" if not reg.writable
              else _read_expr(reg, f"{wreg}{reg.name}"))
        a(f"                        {reg.word}: udev_resp_data <= {rd};")
    a("                        default: udev_resp_data <= 0;")
    a("                    endcase")
    a("                end")
    a("            end else if (udev_resp_valid && udev_resp_ready) udev_resp_valid<=0;")
    a("        end")
    a("    end")
    if frame_sync:
        a("    always @(posedge clk) begin")
        a("        if (!nreset) begin")
        for reg in rw:
            a(f"            {dreg}{reg.name}<={int(reg.reset)};")
        a("        end else if (update) begin")
        for reg in rw:
            a(f"            {dreg}{reg.name} <= {wreg}{reg.name};")
        a("        end")
        a("    end")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": module_name,
            "params": [(r.name, r.bits) for r in rw], "regs": regs,
            "addr_bits": A, "frame_sync": frame_sync,
            "write_op": write_op, "read_op": read_op,
            "offsets": {r.name: r.offset for r in regs}}


def control_top(core, module_name=None, ctrl="axil", frame_sync=True,
                addr_bits=8) -> dict:
    """Top module = control register file + the core: every register port of
    the core gets a register, the frame's size among them -- so software sets
    the frame's width and height over AXI-Lite / UMI like any other config
    register (frame-synced when frame_sync=True), and a register file left at
    its defaults runs the frame the core was traced at. Returns {verilog
    (regfile + top), module, reg}."""
    cm = core["module"]
    top = module_name or cm + "_ctrl"
    ib, ob = core["in_bits"], core["out_bits"]
    regs = list(core["params"])
    defaults = dict(core.get("param_defaults", {}))
    mk = axil_regfile if ctrl == "axil" else umi_regfile
    reg = mk(regs, module_name=f"{top}_rf", addr_bits=addr_bits,
             frame_sync=frame_sync, defaults=defaults)
    A = addr_bits

    L = []
    a = L.append
    a(reg["verilog"]); a("")
    a(f"// generated by np2hw -- control top: {ctrl} regfile + {cm}")
    a(f"module {top} (")
    if ctrl == "axil":
        # One clock here: the bus rides the datapath's. It is named as the
        # bus's clock so an integrator is not left to infer which of the
        # two drives it -- there being only one is a fact of THIS wrapper,
        # not of every wrapper (control_wrap can give the bus its own).
        L.extend(_clock_port("clk", "s_axil", "rst"))
        L.extend(_reset_port("rst", active_low=False))
        L.extend(_axil_slave_ports(A, trailing_comma=True))
    else:
        a("    input  wire clk, input wire rst,")
        a("    input  wire udev_req_valid, output wire udev_req_ready, input wire [4:0] udev_req_cmd,")
        a(f"    input  wire [{A-1}:0] udev_req_addr, input wire [31:0] udev_req_data,")
        a("    output wire udev_resp_valid, input wire udev_resp_ready, output wire [31:0] udev_resp_data,")
    a("    input  wire in_valid, output wire in_ready, input wire in_sof,")
    a(f"    input  wire [{ib-1}:0] in_data,")
    a("    output wire out_valid, input wire out_ready,")
    a("    output wire out_sof, output wire out_eol, output wire out_last,")
    a(f"    output wire {'signed ' if core.get('signed') else ''}[{ob-1}:0] out_data")
    a(");")
    for name, bits in regs:
        a(f"    wire [{bits-1}:0] param_{name};")
    a("    wire core_last;")
    a("    assign out_last = core_last;")
    # register file
    a(f"    {reg['module']} u_rf (")
    if ctrl == "axil":
        a("        .aclk(clk), .aresetn(!rst),")
        if frame_sync:
            a("        .update(core_last),")            # frame boundary: shadow->live
        a("        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),")
        a("        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),")
        a("        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),")
        a("        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),")
        a("        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready)")
    else:
        a("        .clk(clk), .nreset(!rst),")
        if frame_sync:
            a("        .update(core_last),")
        a("        .udev_req_valid(udev_req_valid), .udev_req_ready(udev_req_ready),")
        a("        .udev_req_cmd(udev_req_cmd), .udev_req_addr(udev_req_addr), .udev_req_data(udev_req_data),")
        a("        .udev_resp_valid(udev_resp_valid), .udev_resp_ready(udev_resp_ready), .udev_resp_data(udev_resp_data)")
    for name, _ in regs:
        a(f"        , .param_{name}(param_{name})")
    a("    );")
    # core
    a(f"    {cm} u_core (")
    a("        .clk(clk), .rst(rst),")
    a("        .in_valid(in_valid), .in_ready(in_ready), .in_sof(in_sof),")
    a("        .in_data(in_data),")
    for name, _ in core["params"]:
        a(f"        .param_{name}(param_{name}),")
    a("        .out_valid(out_valid), .out_ready(out_ready),")
    a("        .out_sof(out_sof), .out_eol(out_eol), .out_last(core_last), .out_data(out_data)")
    a("    );")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": top, "reg": reg,
            "regs": regs, "ctrl": ctrl, "frame_sync": frame_sync,
            "offsets": reg["offsets"], "addr_bits": A,
            "in_bits": ib, "out_bits": ob, "signed": core.get("signed")}


# The register port's clock and reset, named once: its attributes and its
# facts both read them.
AXIL_CLOCK, AXIL_RESET = "s_axil_aclk", "s_axil_aresetn"

# The register file's first synchroniser flop on the arm acknowledgement,
# and the prefix of every register software writes (without a shadow bank):
# named once, for the file that declares them and the constraint on the
# crossing that names them. The wrapper's two instances likewise.
REGFILE_ACK_SYNC = "ack_s0"
REGFILE_WRITTEN = "reg_"
WRAP_CSR, WRAP_DATAPATH = "u_csr", "u_datapath"


def control_wrap(core, registers, bind, module_name=None, addr_bits=16,
                 frame_sync=True, commit=None, header=(), notes=(),
                 passthrough=(), bus_clock=False, arm=None,
                 outputs=()) -> dict:
    """Put an AXI4-Lite register file in front of ANY self-describing module.

    `control_top` does this for one generated core and folds in the geometry
    register, which means it also knows that core's port names and its
    `active_width` convention. This one knows nothing: it reads `core[
    "interface"]` for the clock, the reset, the streams and the parameter ports,
    so it wraps a composed multi-block design exactly as it wraps a single core.

    The register ADDRESSES come from the caller, as :class:`Reg` values. An
    application that allocates its own map -- one aligned block per instance, an
    identity word at each base -- keeps that decision, and np2hw keeps the
    emission. Neither has to hold a copy of the other's half.

    Args:
        core: any dict with an `interface` and a `module`, i.e. a generated
            :class:`Core` or the result of :func:`np2hw.compose`.
        registers: :class:`Reg` values, in address order.
        bind: `{parameter_port_name: expression}` for every parameter port the
            wrapped module declares. Which register drives which port is the
            caller's decision -- one register may fan out to several ports, and
            some ports are context rather than configuration -- so it is stated
            rather than matched by name.
        commit: Verilog expression pulsing the shadow-to-live copy. Defaults to
            the frame boundary of the first input stream that carries `last`,
            which is the point at which a frame has finished and the next has
            not started, so no frame is ever processed with half its
            coefficients updated.
        outputs: ``[(name, bits), ...]`` extra OUTPUTS the datapath drives
            beside its streams, passed straight out -- a frame size, say.
        passthrough: ``[(name, bits), ...]`` extra INPUTS the wrapper
            declares and does nothing with except make them available to
            `bind`. This is how a port that is context rather than
            configuration keeps its own source: geometry that arrives in a
            stream's header belongs to the stream, and a register holding a
            second copy of it would be a second answer to one question.
        header, notes: comment lines emitted before the module.

    Returns:
        ``{"verilog", "module", "reg", "offsets", "addr_bits"}``. The Verilog
        holds the register file and the wrapper, in elaboration order; the
        wrapped module is not included, since the caller already has it.
    """
    interface = core["interface"]
    inner = core["module"]
    top = module_name or inner + "_ctrl"
    A = addr_bits
    reg = axil_regfile(registers, module_name=f"{top}_rf", addr_bits=A,
                       frame_sync=frame_sync,
                       arm=arm[0] if arm else None)
    rw = [r for r in reg["regs"] if r.writable]

    streams = interface.get("streams")
    if streams is None:
        raise KeyError(
            f"module {inner!r} does not describe its streams; control_wrap needs "
            "interface['streams'] to pass them through")
    # A frame's size travels with its stream: the size ports of a stream the
    # wrapper passes through pass through beside it -- context, so no
    # register holds them -- unless the caller binds them.
    bind, passthrough, outputs = dict(bind), list(passthrough), list(outputs)
    for stream in streams:
        size = stream.get("size")
        if not size:
            continue
        beside = (f"{stream['prefix']}_frame_width",
                  f"{stream['prefix']}_frame_height")
        if stream["direction"] == "in":
            for port, name in zip(size, beside):
                if port not in bind:
                    bind[port] = name
                    passthrough.append((name, 16))
        else:
            outputs += [(port, 16) for port in size
                        if port not in {n for n, _ in outputs}]

    declared = [name for name, _, _ in interface["params"]]
    # The arm handshake is PLUMBING between the register file and the
    # datapath, not configuration, so no bind supplies it.
    arm_ports = set(arm[1:]) if arm else set()
    declared = [name for name in declared if name not in arm_ports]
    missing = [name for name in declared if name not in bind]
    if missing:
        raise KeyError(
            f"wrapping {inner!r}: parameter port(s) {missing} are not bound; every "
            f"port needs a driver. The register file offers "
            f"{[f'param_{r.name}' for r in rw]}")
    unknown = set(bind) - set(declared)
    if unknown:
        raise KeyError(
            f"wrapping {inner!r}: bind names {sorted(unknown)}, which the module "
            f"does not declare; it has {declared}")

    if commit is None and frame_sync:
        for stream in streams:
            if stream["direction"] == "in" and "last" in stream["flags"]:
                p = stream["prefix"]
                commit = f"{p}_valid && {p}_ready && {p}_last"
                break
        if commit is None:
            raise ValueError(
                f"wrapping {inner!r}: frame_sync needs a commit pulse, and no input "
                "stream carries `last` to derive one from. Pass commit= explicitly.")

    L = list(header)
    a = L.append
    if notes:
        a("//")
        L.extend(notes)
    a("")
    a(reg["verilog"])
    a("")
    a(f"// generated by np2hw -- AXI4-Lite control register file + {inner}.")
    a("// Register addresses are the caller's; this module only decodes them.")
    # Every stream rides the datapath's clock; without a separate bus
    # clock the control interface does too, and `clk` names them all.
    # Which clock drives a bus is not guessable from names.
    from .ports import Pin, header, published
    from .stream import bundle_name, clock_attribute, reset_attribute, stream_pins
    on_clk = [bundle_name(stream["prefix"]) for stream in streams]
    if not bus_clock:
        on_clk.append("s_axil")
    pins = [
        Pin("clk", "input", attribute=(
            clock_attribute("clk", on_clk, "rst") if on_clk else None)),
        Pin("rst", "input", attribute=(
            reset_attribute("rst") if on_clk else None)),
    ]
    axil_note = ("AXI4-Lite control interface. Every configuration register, and every",
                 "read-only identity word, is behind this port.")
    if bus_clock:
        pins += [
            Pin(AXIL_CLOCK, "input", attribute=clock_attribute(
                AXIL_CLOCK, "s_axil", AXIL_RESET), before=axil_note + (
                "ON ITS OWN CLOCK, and that is the whole point. `clk` above",
                "is the datapath's, and a datapath clock can STOP -- one",
                "recovered from an input link has no edges whenever the",
                "link is down, which includes just after the fabric is",
                "reprogrammed. AXI has no timeout, so a slave whose clock",
                "is absent never answers and the master waits for it",
                "forever. That hung a board here and needed the power",
                "pulled. A control interface may never depend on a clock",
                "that can stop.")),
            Pin(AXIL_RESET, "input", attribute=reset_attribute(
                AXIL_RESET, active_low=True)),
        ]
    axil = axil_slave_pins(A)
    if not bus_clock:
        axil[0] = Pin(axil[0].name, axil[0].direction, axil[0].width,
                      bus=axil[0].bus, before=axil_note)
    pins += axil
    for n, (name, bits) in enumerate(passthrough):
        pins.append(Pin(name, "input", bits, before=(
            "context, not configuration: this arrives with the data",)))
    for name, bits in outputs:
        pins.append(Pin(name, "output", bits, before=(
            "what the datapath hands on beside its stream",)))
    for stream in streams:
        flags = [f for f in ("sof", "eol", "last") if f in stream["flags"]]
        pins += stream_pins(stream["prefix"], stream["direction"],
                            stream["data_bits"], flags=flags,
                            signed=bool(stream.get("signed")))
    L += header(top, pins)
    a("")
    for r in rw:
        a(f"    wire {'signed ' if r.signed else ''}[{r.bits-1}:0] param_{r.name};"
          f"   // @ 0x{r.offset:04x}")
    a("")
    if arm:
        # The arm handshake runs BETWEEN the register file and the datapath,
        # so it is neither a port of this wrapper nor bound by a caller --
        # which is why it was dropped from `declared` above. Dropping it
        # there and declaring it nowhere left both signals implicit: legal
        # Verilog, one bit wide by default, and correct here only because
        # they happen to be one bit. Verilator refuses them, and it is
        # right to -- the same omission on a multi-bit signal is silent
        # truncation.
        a("    // The arm handshake, internal to this wrapper: the register")
        a("    // file raises `req` and the datapath answers `ack` when it has")
        a("    // taken the batch.")
        a(f"    wire {arm[1]};")
        a(f"    wire {arm[2]};")
        a("")
    if frame_sync:
        a("    // Shadow -> live at the frame boundary: software may write a register")
        a("    // at any time, and the frame that follows sees ALL of the writes or")
        a("    // none of them. A coefficient changing mid-frame would tear the image.")
        a(f"    wire csr_commit = {commit};")
        a("")
    a(f"    {reg['module']} {WRAP_CSR} (")
    if bus_clock:
        a(f"        .aclk({AXIL_CLOCK}), .aresetn({AXIL_RESET}),")
    else:
        a("        .aclk(clk), .aresetn(!rst),")
    if arm:
        a(f"        .arm_req({arm[1]}), .arm_ack({arm[2]}),")
    if frame_sync:
        a("        .update(csr_commit),")
    a("        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid),"
      " .s_axil_awready(s_axil_awready),")
    a("        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),"
      " .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),")
    a("        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid),"
      " .s_axil_bready(s_axil_bready),")
    a("        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid),"
      " .s_axil_arready(s_axil_arready),")
    a("        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),"
      " .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready)")
    for r in rw:
        a(f"        , .param_{r.name}(param_{r.name})")
    a("    );")
    a("")
    a(f"    {inner} {WRAP_DATAPATH} (")
    a(f"        .{interface['clock']}(clk),")
    a(f"        .{interface['reset']}(rst),")
    prefix = interface.get("param_prefix", "param_")
    connected = set()
    for name, _, _ in interface["params"]:
        # The handshake wires reach the register file, not a bind.
        a(f"        .{prefix}{name}"
          f"({name if name in arm_ports else bind[name]}),")
        connected.add(name)
    # An OUTPUT handshake port is not in `params` -- that list is input
    # scalars only -- so it has to be connected explicitly. Left to the
    # loop above it silently became an undriven wire: the datapath still
    # took its copy, using its own internal state, and only the path BACK
    # to the register file was missing, so the arm never cleared and
    # every later write was refused. Nothing in simulation noticed,
    # because nothing asserted that the arm clears.
    for name in sorted(arm_ports - connected):
        a(f"        .{prefix}{name}({name}),")
    lines = [f"        .{name}({name})" for name, _ in outputs]
    for stream in streams:
        p = stream["prefix"]
        lines.append(f"        .{p}_valid({p}_valid)")
        lines.append(f"        .{p}_ready({p}_ready)")
        lines.append(f"        .{p}_data({p}_data)")
        lines.extend(f"        .{p}_{flag}({p}_{flag})" for flag in stream["flags"])
    for i, line in enumerate(lines):
        a(line + ("" if i == len(lines) - 1 else ","))
    a("    );")
    a("endmodule")

    return {"verilog": "\n".join(L), "module": top, "reg": reg,
            "offsets": reg["offsets"], "addr_bits": A,
            "frame_sync": frame_sync, "commit": commit,
            "interface": {"clock": "clk", "reset": "rst",
                          "ports": published(pins),
                          "streams": [dict(st) for st in streams]},
            # what its register port is timed and reset by, as its clock
            # attribute says it (np2hw.design.ancestry)
            "facts": {"clocks": {"s_axil": [AXIL_CLOCK, AXIL_RESET]}},
            "constraints": _crossing(top, arm) if (arm and bus_clock)
            else []}


def arm_crossing(module: str, *, regfile: str, capture: str, answer: str,
                 apart) -> dict:
    """The constraint on an armed register file read from another clock:
    an np2hw register file at instance `regfile` of `module`, its values
    and its arm request sampled by `capture` (the reader's first
    synchroniser flop) and its acknowledgement raised by `answer` -- both
    flops by their names in `module`. Software writes the values before
    the commit; they stay frozen while it is armed, and the reader copies
    them only after the request has crossed two flops, so each crossing is
    bounded by one period of the clock that samples it.

    Returned as a generated constraint -- ``{"file", "text", "scoped_to",
    "apart"}`` -- applied by a build only where the module's two ports in
    `apart` are on different nets: on one clock nothing crosses, and a
    datapath-only bound would drop the hold checks of ordinary paths."""
    values = (f"[get_cells -filter {{IS_SEQUENTIAL}} "
              f"{{{regfile}/{REGFILE_WRITTEN}*}}]")
    ack = f"{regfile}/{REGFILE_ACK_SYNC}_reg"
    sampled = f"[get_clocks -of_objects [get_cells {{{capture}_reg}}]]"
    text = "\n".join([
        f"# GENERATED by np2hw.verilog.arm_crossing -- {module}'s register",
        f"# file ({regfile}) read from another clock, scoped to the module.",
        f"# A build applies it only where {apart[0]} and {apart[1]} are apart.",
        "#",
        "# The values are written before the commit, frozen while it is",
        "# armed, and copied only after the request has crossed two flops:",
        "# each is bounded by one period of the clock that samples it.",
        "set_max_delay -datapath_only \\",
        f"    -from {values} \\",
        f"    -to {sampled} \\",
        f"    [get_property -min PERIOD {sampled}]",
        "# The acknowledgement, back into the register file's clock.",
        "set_max_delay -datapath_only \\",
        f"    -from [get_cells {{{answer}_reg}}] -to [get_cells {{{ack}}}] \\",
        f"    [get_property -min PERIOD [get_clocks -of_objects "
        f"[get_cells {{{ack}}}]]]",
        ""])
    return {"file": f"{module}_crossing.xdc", "text": text,
            "scoped_to": module, "apart": list(apart)}


def _crossing(top: str, arm) -> list:
    """The wrapper's own crossing, by the names its two halves declare."""
    from .compose import arm_answer, arm_capture
    return [arm_crossing(
        top, regfile=WRAP_CSR,
        capture=f"{WRAP_DATAPATH}/{arm_capture(arm[1])}",
        answer=f"{WRAP_DATAPATH}/{arm_answer(arm[2])}",
        apart=["clk", AXIL_CLOCK])]


def testbench_ctrl(core, reg, W, H, n_out, param_values, ctrl="axil",
                   frame_sync=False, n_frames=1, tb_name="tb") -> str:
    """Wire a register file + the (plain, non-edge) core together: write each
    param over the control bus (AXI4-Lite or UMI), THEN stream the frame(s) and
    log outputs. Proves the control interface sets the datapath's config.

    frame_sync: connect rf.update = core out_last, stream n_frames frames; the
    written values take effect only at the frame boundary (shadow -> live)."""
    ib, ob = core["in_bits"], core["out_bits"]
    A = reg["addr_bits"]
    off = reg["offsets"]
    total = W * H
    feed_n = n_frames * total
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a(f"module {tb_name};")
    a("    reg clk = 0, rst = 1;")
    a("    reg in_valid; wire in_ready; reg out_ready;")
    a("    wire out_valid, out_last;")
    a(f"    wire {'signed ' if core.get('signed') else ''}[{ob-1}:0] out_pix;")
    a(f"    reg [{ib-1}:0] img [0:{total-1}];")
    a("    integer fed = 0, got = 0, f, cyc = 0;")
    for name, bits in core["params"]:
        a(f"    wire [{bits-1}:0] param_{name};")
    a(f"    wire [{ib-1}:0] cur = img[fed % {total}];")
    if frame_sync:
        # pause feeding at the frame boundary (VBLANK) so the shadow->live update
        # lands while no pixel is being computed -- no mid-stream glitch.
        a("    reg started2; integer bcnt;")
        a(f"    wire in_v = in_valid && (fed < {feed_n}) && (fed < {total} || started2);")
    else:
        a(f"    wire in_v = in_valid && (fed < {feed_n});")
    # control bus signals
    if ctrl == "axil":
        a(f"    reg [{A-1}:0] s_axil_awaddr; reg s_axil_awvalid; wire s_axil_awready;")
        a("    reg [31:0] s_axil_wdata; reg [3:0] s_axil_wstrb; reg s_axil_wvalid; wire s_axil_wready;")
        a("    wire [1:0] s_axil_bresp; wire s_axil_bvalid; reg s_axil_bready;")
        a(f"    reg [{A-1}:0] s_axil_araddr; reg s_axil_arvalid; wire s_axil_arready;")
        a("    wire [31:0] s_axil_rdata; wire [1:0] s_axil_rresp; wire s_axil_rvalid; reg s_axil_rready;")
        a(f"    {reg['module']} rf (")
        a("        .aclk(clk), .aresetn(!rst),")
        if frame_sync:
            a("        .update(out_last),")
        a("        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),")
        a("        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),")
        a("        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),")
        a("        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),")
        a("        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready)")
        for name, _ in core["params"]:
            a(f"        , .param_{name}(param_{name})")
        a("    );")
    else:                                                # umi
        a("    reg udev_req_valid; wire udev_req_ready; reg [4:0] udev_req_cmd;")
        a(f"    reg [{A-1}:0] udev_req_addr; reg [31:0] udev_req_data;")
        a("    wire udev_resp_valid; reg udev_resp_ready; wire [31:0] udev_resp_data;")
        a(f"    {reg['module']} rf (")
        a("        .clk(clk), .nreset(!rst),")
        if frame_sync:
            a("        .update(out_last),")
        a("        .udev_req_valid(udev_req_valid), .udev_req_ready(udev_req_ready),")
        a("        .udev_req_cmd(udev_req_cmd), .udev_req_addr(udev_req_addr), .udev_req_data(udev_req_data),")
        a("        .udev_resp_valid(udev_resp_valid), .udev_resp_ready(udev_resp_ready), .udev_resp_data(udev_resp_data)")
        for name, _ in core["params"]:
            a(f"        , .param_{name}(param_{name})")
        a("    );")
    # core
    a(f"    {core['module']} dut (")
    a("        .clk(clk), .rst(rst),")
    a("        .in_valid(in_v), .in_ready(in_ready), .in_sof(1'b0), .in_data(cur),")
    for name, _ in core["params"]:
        a(f"        .param_{name}(param_{name}),")
    a("        .out_valid(out_valid), .out_ready(out_ready),")
    a("        .out_sof(), .out_eol(), .out_last(out_last), .out_data(out_pix)")
    a("    );")
    a("    always #5 clk = ~clk;")
    a("    always @(posedge clk) if (!rst) begin")
    a("        cyc <= cyc + 1;")
    a("        if (in_v && in_ready) fed <= fed + 1;")
    if frame_sync:
        a(f"        if (fed == {total} && !started2) begin")  # VBLANK between frames
        a("            if (bcnt == 8) started2 <= 1'b1; else bcnt <= bcnt + 1;")
        a("        end")
    a("        if (out_valid && out_ready) begin")
    a("            $fdisplay(f, \"%0d\", out_pix); got <= got + 1;")
    a(f"            if (got + 1 == {n_out}) begin $fclose(f); $finish; end")
    a("        end")
    a(f"        if (cyc > {feed_n * 8 + 2000}) begin $fclose(f); $finish; end")
    a("    end")
    # control write tasks
    if ctrl == "axil":
        a("    task axil_w(input [31:0] ad, input [31:0] d); begin")
        a("        @(negedge clk); s_axil_awaddr=ad; s_axil_awvalid=1; s_axil_wdata=d;")
        a("        s_axil_wstrb=4'hf; s_axil_wvalid=1; s_axil_bready=1;")
        a("        @(posedge clk); while(!(s_axil_awready && s_axil_wready)) @(posedge clk);")
        a("        @(negedge clk); s_axil_awvalid=0; s_axil_wvalid=0;")
        a("        while(!s_axil_bvalid) @(posedge clk); @(negedge clk); s_axil_bready=0;")
        a("    end endtask")
    else:
        a("    task umi_w(input [31:0] ad, input [31:0] d); begin")
        a(f"        @(negedge clk); udev_req_valid=1; udev_req_cmd={reg['write_op']};")
        a("        udev_req_addr=ad; udev_req_data=d;")
        a("        @(posedge clk); while(!udev_req_ready) @(posedge clk);")
        a("        @(negedge clk); udev_req_valid=0;")
        a("    end endtask")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        f = $fopen(\"out.txt\", \"w\");")
    a("        in_valid=0; out_ready=1;")
    if frame_sync:
        a("        started2=0; bcnt=0;")
    if ctrl == "axil":
        a("        s_axil_awvalid=0; s_axil_wvalid=0; s_axil_bready=0; s_axil_arvalid=0; s_axil_rready=0;")
    else:
        a("        udev_req_valid=0; udev_resp_ready=1;")
    a("        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);")
    a("        // --- program config registers over the control bus ---")
    for name, _ in core["params"]:
        val = _bench_value(core, name, param_values, W, H)
        wr = "axil_w" if ctrl == "axil" else "umi_w"
        a(f"        {wr}({off[name]}, {val});")
    a("        @(negedge clk);")
    a("        // --- then stream the frame ---")
    a("        in_valid = 1;")
    a("    end")
    a("endmodule")
    return "\n".join(L)


def testbench_sb_packed(wrap, W, H, n_out, param_values=None, tb_name="tb") -> str:
    """TB for the packed SB wrapper: assembles input packets of p_in pixels,
    drives SB ports with randomized valid/ready, unpacks output packets, writes
    the frame's pixels to out.txt, and checks 'last' frames the final packet."""
    param_values = param_values or {}
    ib, ob = wrap["in_bits"], wrap["out_bits"]
    p_in, p_out = wrap["p_in"], wrap["p_out"]
    mod = wrap["module"]
    frame_in = W * H
    np_in = (frame_in + p_in - 1) // p_in
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a(f"module {tb_name};")
    a("    reg clk = 0, rst = 1; reg in_offer; reg sb_out_ready;")
    a("    wire sb_in_ready, sb_out_valid;")
    a(f"    wire [415:0] sb_out_data; wire [31:0] sb_out_flags;")
    for name, bits in wrap["params"]:
        a(f"    reg [{bits-1}:0] param_{name};")
    a(f"    reg [{ib-1}:0] img [0:{frame_in-1}];")
    a("    integer ipkt = 0, got = 0, f, cyc = 0, j, fail = 0;")
    a(f"    wire sb_in_valid = in_offer && (ipkt < {np_in});")
    # assemble the current input packet from img
    a("    reg [415:0] inpkt;")
    a("    always @* begin")
    a("        inpkt = 0;")
    a(f"        for (j = 0; j < {p_in}; j = j + 1)")
    a(f"            if (ipkt*{p_in} + j < {frame_in})")
    a(f"                inpkt[j*{ib} +: {ib}] = img[ipkt*{p_in} + j];")
    a("    end")
    a(f"    {mod} dut (")
    a("        .clk(clk), .rst(rst),")
    a("        .sb_in_valid(sb_in_valid), .sb_in_ready(sb_in_ready),")
    a("        .sb_in_data(inpkt), .sb_in_dest(32'b0), .sb_in_flags(32'b0),")
    a("        .sb_out_valid(sb_out_valid), .sb_out_ready(sb_out_ready),")
    a("        .sb_out_data(sb_out_data), .sb_out_dest(), .sb_out_flags(sb_out_flags)")
    for name, _ in wrap["params"]:
        a(f"        , .param_{name}(param_{name})")
    a("    );")
    a("    always #5 clk = ~clk;")
    a("    always @(negedge clk) begin")
    a("        in_offer    <= ($random % 4 != 0);")
    a("        sb_out_ready<= ($random % 4 != 0);")
    a("    end")
    a("    always @(posedge clk) if (!rst) begin")
    a("        cyc <= cyc + 1;")
    a("        if (sb_in_valid && sb_in_ready) ipkt <= ipkt + 1;")
    pix_expr = (f"$signed(sb_out_data[j*{ob} +: {ob}])" if wrap.get("signed")
                else f"sb_out_data[j*{ob} +: {ob}]")
    a("        if (sb_out_valid && sb_out_ready) begin")
    a(f"            for (j = 0; j < {p_out}; j = j + 1) begin")
    a(f"                if (got + j < {n_out})")
    a(f"                    $fdisplay(f, \"%0d\", {pix_expr});")
    a("            end")
    # framing: 'last' must be set iff this packet completes the frame
    a(f"            if ((got + {p_out} >= {n_out}) != (sb_out_flags[0] == 1'b1)) fail <= 1;")
    a(f"            got <= got + {p_out};")
    a(f"            if (got + {p_out} >= {n_out}) begin")
    a("                if (fail) $display(\"FRAMING FAIL\");")
    a("                $fclose(f); $finish;")
    a("            end")
    a("        end")
    a(f"        if (cyc > {(frame_in + n_out) * 16 + 2000}) begin $fclose(f); $finish; end")
    a("    end")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        f = $fopen(\"out.txt\", \"w\");")
    for name, _ in wrap["params"]:
        a(f"        param_{name} = {_bench_value(wrap, name, param_values, W, H)};")
    a("        in_offer = 0; sb_out_ready = 0;")
    a("        @(negedge clk); @(negedge clk); rst = 0;")
    a("    end")
    a("endmodule")
    return "\n".join(L)


def _byte_align(bits):
    return ((bits + 7) // 8) * 8


def axis_video_wrap(meta, W, H, module_name=None) -> dict:
    """Wrap the generic core in an AXI4-Stream Video interface (UG934 convention):
    s_axis/m_axis with tvalid/tready/tdata, tlast = End Of Line, tuser[0] = Start
    Of Frame. tdata is byte-aligned. Active-low aresetn. The slave honors the
    incoming s_axis_tuser (SOF) to anchor frames (-> core in_sof); s_axis_tlast is
    accepted but unused. Separate adapter — the core is untouched and reusable."""
    core = meta["module"]
    top = module_name or core + "_axis"
    ib, ob = meta["in_bits"], meta["out_bits"]
    params = meta["params"]
    signed = meta.get("signed")
    tin, tout = _byte_align(ib), _byte_align(ob)
    pports = "".join(f"\n    , input wire [{b-1}:0] param_{n}" for n, b in params)
    pconns = "".join(f"\n        .param_{n}(param_{n})," for n, _ in params)
    if tout > ob:
        ext = f"{{{tout - ob}{{core_out[{ob-1}]}}}}" if signed else f"{tout - ob}'b0"
        tdata = f"{{{ext}, core_out}}"
    else:
        tdata = "core_out"
    verilog = f"""// generated by np2hw -- AXI4-Stream Video adapter for {core} (separate module)
module {top} (
    input  wire aclk,
    input  wire aresetn,
    // AXI4-Stream Video slave (pixel input)
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire [{tin-1}:0] s_axis_tdata,
    input  wire s_axis_tuser,   // Start Of Frame -> anchors the core
    input  wire s_axis_tlast,   // End Of Line (accepted, unused)
    // AXI4-Stream Video master (pixel output)
    output wire m_axis_tvalid,
    input  wire m_axis_tready,
    output wire [{tout-1}:0] m_axis_tdata,
    output wire m_axis_tuser,   // Start Of Frame
    output wire m_axis_tlast    // End Of Line{pports}
);
    wire rst = !aresetn;        // AXI reset is active-low
    wire [{ob-1}:0] core_out; wire core_sof, core_eol;
    {core} u_core (
        .clk(aclk), .rst(rst),
        .in_valid(s_axis_tvalid), .in_ready(s_axis_tready), .in_sof(s_axis_tuser),
        .in_data(s_axis_tdata[{ib-1}:0]),{pconns}
        .out_valid(m_axis_tvalid), .out_ready(m_axis_tready),
        .out_sof(core_sof), .out_eol(core_eol), .out_last(),
        .out_data(core_out)
    );
    assign m_axis_tdata = {tdata};
    assign m_axis_tuser = core_sof;   // Start Of Frame
    assign m_axis_tlast = core_eol;   // End Of Line
endmodule
"""
    return {
        "verilog": verilog,
        "module": top,
        "core": core,
        "in_bits": ib, "out_bits": ob,
        "tdata_in": tin, "tdata_out": tout,
        "params": params,
        "signed": signed,
        "out_rows": meta.get("out_rows"), "out_cols": meta.get("out_cols"),
    }


# --------------------------------------------------------------------------- #
# Backpressure testbench — randomized ready/valid handshake on either the
# generic core or the SB wrapper. Proves the handshake without needing blanking.
# --------------------------------------------------------------------------- #

def testbench_handshake(meta, W, H, n_out, param_values=None, iface="core",
                        tb_name="tb") -> str:
    """Randomized ready/valid handshake TB. iface: 'core' (generic), 'sb'
    (Switchboard wrapper), 'axis' (AXI4-Stream Video wrapper). Logs
    'pixel f0 f1' per accepted output; (f0,f1) = (last,0) for core/sb,
    (sof,eol) for axis -- so the harness can verify framing."""
    param_values = param_values or {}
    ib, ob = meta["in_bits"], meta["out_bits"]
    mod = meta["module"]
    total = W * H
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a(f"module {tb_name};")
    a("    reg clk = 0, rst = 1;")
    a("    reg in_offer; reg out_ready;")
    a("    wire in_ready, out_valid, f0, f1;")
    a(f"    wire {'signed ' if meta.get('signed') else ''}[{ob-1}:0] out_pix;")
    for name, bits in meta["params"]:
        a(f"    reg  [{bits-1}:0] param_{name};")
    a(f"    reg  [{ib-1}:0] img [0:{total-1}];")
    a("    integer fed = 0, got = 0, f, cyc = 0;")
    a("    always #5 clk = ~clk;")
    a(f"    wire [{ib-1}:0] cur = img[fed];")
    a(f"    wire in_valid = in_offer && (fed < {total});")
    if iface == "sb":
        a("    wire [415:0] sb_out_bus; wire [31:0] sb_out_flags_w;")
        a(f"    assign out_pix = sb_out_bus[{ob-1}:0];")
        a("    assign f0 = sb_out_flags_w[0];")             # 'last'
        a("    assign f1 = 1'b0;")
    a(f"    {mod} dut (")
    if iface == "sb":
        a("        .clk(clk), .rst(rst),")
        a("        .sb_in_valid(in_valid), .sb_in_ready(in_ready),")
        a(f"        .sb_in_data({{{{{416 - ib}{{1'b0}}}}, cur}}), .sb_in_dest(32'b0), .sb_in_flags(32'b0),")
        a("        .sb_out_valid(out_valid), .sb_out_ready(out_ready),")
        a("        .sb_out_data(sb_out_bus), .sb_out_dest(), .sb_out_flags(sb_out_flags_w)")
    elif iface == "axis":
        tin = _byte_align(ib)
        a("        .aclk(clk), .aresetn(!rst),")
        a("        .s_axis_tvalid(in_valid), .s_axis_tready(in_ready),")
        a(f"        .s_axis_tdata({{{{{tin - ib}{{1'b0}}}}, cur}}),"
          if tin > ib else "        .s_axis_tdata(cur),")
        a("        .s_axis_tuser(1'b0), .s_axis_tlast(1'b0),")
        a("        .m_axis_tvalid(out_valid), .m_axis_tready(out_ready),")
        a(f"        .m_axis_tdata(out_pix), .m_axis_tuser(f0), .m_axis_tlast(f1)")
    else:                                                  # generic core
        a("        .clk(clk), .rst(rst),")
        a("        .in_valid(in_valid), .in_ready(in_ready), .in_sof(1'b0), .in_data(cur),")
        a("        .out_valid(out_valid), .out_ready(out_ready),")
        a("        .out_last(f0), .out_data(out_pix)")
        a("        , .out_sof(), .out_eol()")
    for name, _ in meta["params"]:
        a(f"        , .param_{name}(param_{name})")
    a("    );")
    if iface != "axis" and iface != "sb":
        a("    assign f1 = 1'b0;")
    a("    always @(negedge clk) begin")
    a("        in_offer  <= ($random % 4 != 0);")          # ~75% offer
    a("        out_ready <= ($random % 4 != 0);")          # ~75% ready
    a("    end")
    a("    always @(posedge clk) if (!rst) begin")
    a("        cyc <= cyc + 1;")
    a("        if (in_valid && in_ready) fed <= fed + 1;")
    a("        if (out_valid && out_ready) begin")
    a("            $fdisplay(f, \"%0d %0d %0d\", out_pix, f0, f1); got <= got + 1;")
    a(f"            if (got + 1 == {n_out}) begin $fclose(f); $finish; end")
    a("        end")
    a(f"        if (cyc > {(total + n_out) * 12 + 1000}) begin $fclose(f); $finish; end")
    a("    end")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        f = $fopen(\"out.txt\", \"w\");")
    for name, _ in meta["params"]:
        a(f"        param_{name} = {_bench_value(meta, name, param_values, W, H)};")
    a("        in_offer = 0; out_ready = 0;")
    a("        @(negedge clk); @(negedge clk); rst = 0;")
    a("    end")
    a("endmodule")
    return "\n".join(L)
