# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The printer: a lowered hardware unit as Verilog, one rule per kind.

Nothing is written that is not a node of the graph: the datapath, the
counters, the flags and the registers all come from the passes. The ports
and their meaning are np2hw's stream contract (docs/interfaces.md).

Every wire is sized from its exact range. The one rule kept for Verilog's
sake: an expression is signed only if EVERY operand is, so in a signed
operation each unsigned operand is widened by a zero bit and read as
signed -- here, once, for every kind.
"""
from __future__ import annotations

from .graph import Type, Value, walk
from .hw import cat_fields
from .passes import Unit


def _literal(value: int) -> str:
    """A constant as Verilog reads it: an unsized decimal is a 32-bit
    integer, so one that does not fit says its width -- signed, with room
    for its sign -- and a tool takes it whole instead of truncating it (or,
    as Vivado does, refusing it)."""
    if -(1 << 31) <= value < 1 << 31:
        return str(value)
    bits = abs(value).bit_length() + 1
    return f"{bits}'sd{value}" if value >= 0 else f"-{bits}'sd{-value}"


def _signed(v: Value) -> bool:
    return v.type.lo < 0


def _decl(v: Value) -> str:
    w = v.type.bits
    if _signed(v):
        return f"signed [{w - 1}:0] "
    return f"[{w - 1}:0] " if w > 1 else ""


_BIN = {"hw.add": "+", "hw.sub": "-", "hw.mul": "*", "hw.and": "&",
        "hw.or": "|"}
_CMP = {"hw.lt": "<", "hw.gt": ">", "hw.le": "<=", "hw.ge": ">=",
        "hw.eq": "==", "hw.ne": "!="}


class _Printer:
    def __init__(self, dev=None):
        self.dev = dev
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
        self.roms: dict[str, list[str]] = {}
        self.k = 0

    def name(self, v: Value) -> str:
        return self.names[id(v.node)]

    def ref(self, v: Value, signed: bool) -> str:
        n = self.name(v)
        return f"$signed({{1'b0, {n}}})" if signed and not _signed(v) else n

    def wire(self, v: Value, expr: str, hint="n", attribute="") -> str:
        name = f"{hint}{self.k}"
        self.k += 1
        self.lines.append(f"    {attribute}wire {_decl(v)}{name} = {expr};")
        return name

    def literal(self, value: int, signed: bool) -> str:
        w = Type((), ("x", 0), min(value, 0), max(value, 0)).bits
        w += 1 if signed and value >= 0 else 0
        name = f"c{self.k}"
        self.k += 1
        self.lines.append(f"    wire {'signed ' if signed else ''}"
                          f"[{max(w, 1) - 1}:0] {name} = {value};")
        return name

    def node(self, n) -> str:
        k = n.kind
        v = n.value if n.results else None
        ops = n.operands
        if k == "hw.port":
            return n.attr("name")
        if k == "hw.param":
            return f"param_{n.attr('name')}"
        if k == "hw.const":
            return self.wire(v, _literal(int(n.attr("value"))), "c")
        if k == "hw.regread":
            return f"r_{n.attr('name')}"
        if k == "hw.memread":
            return f"mem_{n.attr('name')}_rd"
        dom = _signed(v) or any(_signed(o) for o in ops)
        if k in _BIN:
            a, b = (self.ref(o, dom) for o in ops)
            # a multiply, built as the shape the device timed it as
            from ..verilog import _mul_hint
            return self.wire(v, f"{a} {_BIN[k]} {b}", attribute=_mul_hint(
                "mul" if k == "hw.mul" else k, self.dev))
        if k in _CMP:
            dom = any(_signed(o) for o in ops)
            a, b = (self.ref(o, dom) for o in ops)
            return self.wire(v, f"({a} {_CMP[k]} {b})")
        if k == "hw.not":
            return self.wire(v, f"!{self.name(ops[0])}")
        if k == "hw.neg":
            return self.wire(v, f"-{self.ref(ops[0], True)}")
        if k == "hw.shl":
            op = "<<<" if dom else "<<"
            return self.wire(v, f"{self.ref(ops[0], dom)} {op} "
                                f"{n.attr('amount')}")
        if k == "hw.shr":
            op = ">>>" if dom else ">>"
            return self.wire(v, f"{self.ref(ops[0], dom)} {op} "
                                f"{n.attr('amount')}")
        if k == "hw.abs":
            a = self.name(ops[0])
            if not _signed(ops[0]):
                return self.wire(v, a)
            return self.wire(v, f"({a} < 0) ? -{a} : {a}")
        if k == "hw.clip":
            x = ops[0]
            dom = _signed(x) or n.attr("lo") < 0
            lo = self.literal(n.attr("lo"), dom)
            hi = self.literal(n.attr("hi"), dom)
            a = self.ref(x, dom)
            return self.wire(v, f"({a} < {lo}) ? {lo} : "
                                f"(({a} > {hi}) ? {hi} : {a})")
        if k == "hw.sel":
            c, a, b = ops
            dom = _signed(v) or _signed(a) or _signed(b)
            return self.wire(v, f"({self.name(c)} != 0) ? "
                                f"{self.ref(a, dom)} : {self.ref(b, dom)}")
        if k in ("hw.wrap", "hw.bits"):
            x = ops[0]
            lo_bit = n.attr("lo") if k == "hw.bits" else 0
            width = n.attr("width") if k == "hw.bits" else n.attr("bits")
            need = max(x.type.bits, lo_bit + width)
            ext = f"x{self.k}"
            self.k += 1
            sgn = "signed " if _signed(x) else ""
            self.lines.append(f"    wire {sgn}[{need - 1}:0] {ext} = "
                              f"{self.name(x)};")
            field = f"{ext}[{lo_bit + width - 1}:{lo_bit}]"
            if k == "hw.wrap" and n.attr("signed") and _signed(v):
                return self.wire(v, f"$signed({field})")
            return self.wire(v, field)
        if k == "hw.mux":
            idx, choices = ops[0], ops[1:]
            dom = _signed(v) or any(_signed(o) for o in choices)
            i = self.name(idx)
            # a balanced tree on the index's bits, which the tools map onto
            # their lookup tables and wide multiplexers
            def tree(base, bit):
                if bit < 0 or base + (1 << bit) >= len(choices):
                    return (self.ref(choices[base], dom) if bit < 0 else
                            tree(base, bit - 1))
                b = i if idx.type.bits == 1 else f"{i}[{bit}]"
                return (f"({b} ? {tree(base + (1 << bit), bit - 1)} "
                        f": {tree(base, bit - 1)})")
            top = max(1, (len(choices) - 1).bit_length())
            return self.wire(v, tree(0, top - 1))
        if k == "hw.rom":
            # contents inline; the address it reads is the stage register
            # before it, so the tools build the table as a ROM
            name = f"rom_{n.attr('name')}"
            data = n.attr("data")
            if name not in self.roms:
                lo, hi = min(data), max(data)
                t = Type((), ("x", 0), min(lo, 0), max(hi, 0))
                self.roms[name] = [
                    f"    reg {'signed ' if lo < 0 else ''}"
                    f"[{t.bits - 1}:0] {name} [0:{len(data) - 1}];",
                    "    initial begin",
                    *(f"        {name}[{i}] = {d};" for i, d in enumerate(data)),
                    "    end"]
            return self.wire(v, f"{name}[{self.name(ops[0])}]")
        if k == "hw.read":
            return self.name(ops[0])        # the value; its register is a stage's
        if k == "hw.cat":
            parts = []
            for o, f in reversed(list(zip(ops, cat_fields(n, len(ops))))):
                w = o.type.bits
                parts.append(self.name(o) if w == f else
                             f"{{{f - w}'b0, {self.name(o)}}}")
            return self.wire(v, f"{{{', '.join(parts)}}}")
        raise NotImplementedError(f"no printing for {k}")


_SEAM_WHAT = {"linebuf": "a window's line buffer",
              "mem": "a frame store"}


def _seam(name: str, kind: str, width: int, depth: int, aw: int) -> list:
    """A memory behind its seam (docs/memories.md): one write port, one
    read port, both synchronous -- the module an integrator binds a memory
    macro to, by name."""
    return [
        f"// generated by np2hw -- {_SEAM_WHAT[kind]}: one write port, one",
        "// read port, both synchronous. THE MEMORY SEAM: bind a memory macro",
        "// here, by module name, and honour four things.",
        "//   1. write: wd lands at mem[wa] on the clk edge where we is high.",
        "//   2. read: rd <= mem[ra] on the clk edge where re is high -- one",
        "//      cycle of latency -- and rd HOLDS while re is low.",
        "//   3. collision: we and re in one cycle at one address read the OLD",
        "//      word. The reads here run a position ahead of the writes, but",
        "//      a binding must not rely on that silently.",
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
        "endmodule"]


def generate(unit: Unit, module_name: str, dev=None) -> dict:
    """Print the unit; return its Core, as every generated module does.
    `dev` is the device family it is timed against (the default when
    None), whose shapes the printing honours. The raster's extents are the
    frame registers, which the size fields of `in_ctx` load as each frame
    enters; the size the module hands on in `out_ctx` is the unit's own, as
    lowering made it from the output's extent."""
    from ..verilog import Core, _interface
    f = unit.facts
    img = f["image"]
    mems = f.get("memories", [])
    p = _Printer(dev)
    for n in walk(unit.all_roots()):
        p.names[id(n)] = p.node(n)
    # the stream carries its context: the core takes its frame's size and
    # every value a frame brings in it, and hands it on -- no port of either
    carried = {name for name, _, _ in img.context}
    params = [p for p in f["params"] if p[0] not in carried]
    head = [f"// generated by np2hw -- lowered from the graph "
            f"({len(params)} register port(s))",
            "// Every wire is sized from its node's exact value range; the",
            "// counters, flags and registers are nodes like the rest.",
            f"// The most it holds: a frame {img.width} wide"
            + (f" and {img.height} high" if mems and any(
                m["seam"] == "mem" for m in mems) else "")
            + "; each frame's own size arrives in its context.",
            f"module {module_name} (",
            "    input  wire clk,",
            "    input  wire rst,",
            "    input  wire in_valid,",
            "    output wire in_ready,",
            "    input  wire in_sof,",
            f"    input  wire [{img.word_bits - 1}:0] in_data,"]
    head.append("    // each frame's context, with its first beat: "
                + ", ".join(f"{n} {b}" for n, b, _ in img.context)
                + ", the first in the low bits")
    head.append(f"    input  wire [{img.context_bits - 1}:0] in_ctx,")
    for name, bits, signed, _, doc in params:
        if doc:
            head.append(f"    // {doc}")
        head.append(f"    input  wire {'signed ' if signed else ''}"
                    f"[{bits - 1}:0] param_{name},")
    sgn = "signed " if f["out_signed"] else ""
    head += ["    // the context it hands on: the same fields, the size its own",
             f"    output wire [{img.context_bits - 1}:0] out_ctx,",
             "    output wire out_valid,",
             "    input  wire out_ready,",
             "    output wire out_sof,",
             "    output wire out_eol,",
             "    output wire out_last,",
             f"    output wire {sgn}[{f['out_bits'] - 1}:0] out_data",
             ");"]
    decls, body = [], []
    for m in mems:
        # a memory behind its seam: the module an integrator binds by name
        n, w, aw = m["name"], m["bits"], m["aw"]
        decls += [f"    wire mem_{n}_we, mem_{n}_re;",
                  f"    wire [{aw - 1}:0] mem_{n}_wa, mem_{n}_ra;",
                  f"    wire [{w - 1}:0] mem_{n}_wd, mem_{n}_rd;",
                  f"    {module_name}_{m['seam']} #(.W({w}), .D({m['depth']}), "
                  f".AW({aw})) {m['instance']} (",
                  f"        .clk(clk), .we(mem_{n}_we), .wa(mem_{n}_wa), "
                  f".wd(mem_{n}_wd),",
                  f"        .re(mem_{n}_re), .ra(mem_{n}_ra), "
                  f".rd(mem_{n}_rd));"]
    for r in unit.regs:
        name = r.attr("name")
        d = unit.g.symbols[name]
        t = Type((), ("x", 0), d["lo"], d["hi"])
        rs = "signed " if d["lo"] < 0 else ""
        decls.append(f"    reg {rs}[{t.bits - 1}:0] r_{name};" if t.bits > 1
                     or rs else f"    reg r_{name};")
        nxt, en = r.operands
        # A next value that is a chain of selects prints as a priority chain
        # inside the register's block: the same logic, in the form synthesis
        # maps onto a flip-flop's own set and reset pins where a branch loads
        # a constant -- as an explicit mux tree it spends LUT levels on it.
        chain, rest = [], nxt
        while rest.node.kind == "hw.sel":
            c, a_, b_ = rest.node.operands
            chain.append((p.name(c), p.name(a_)))
            rest = b_
        if chain:
            load = [f"            {'if' if i == 0 else 'else if'} ({c} != 0) "
                    f"r_{name} <= {v};" for i, (c, v) in enumerate(chain)]
            load.append(f"            else r_{name} <= {p.name(rest)};")
        else:
            load = [f"            r_{name} <= {p.name(nxt)};"]
        body.append("    always @(posedge clk)")
        if d["init"] is None:                 # no reset: nothing needs one
            body.append(f"        if ({p.name(en)}) begin")
        else:
            body += [f"        if (rst) r_{name} <= {d['init']};",
                     f"        else if ({p.name(en)}) begin"]
        body += load + ["        end"]
    assigns = [f"    assign {port} = {p.name(v)};"
               for port, v in unit.roots.items()]
    # one seam module of each kind used, its parameters set per instance
    seams = []
    for kind in dict.fromkeys(m["seam"] for m in mems):
        m = next(x for x in mems if x["seam"] == kind)
        seams += [""] + _seam(f"{module_name}_{kind}", kind, m["bits"],
                              m["depth"], m["aw"])
    decls += [line for rom in p.roms.values() for line in rom]
    verilog = "\n".join(head + decls + p.lines + body + assigns +
                        ["endmodule"] + seams)
    interface = _interface(img.word_bits, f["out_bits"], f["out_signed"],
                           params)
    for stream in [interface["input"], interface["output"],
                   *interface["streams"]]:
        if stream is not None:
            # the stream's CTX member, its fields in order
            stream["ctx"] = [list(field) for field in img.context]
    if f.get("window") is not None:
        # the frame it hands on is the window's, set at run time: the traced
        # size bounds it, and is not it
        interface["output"]["window"] = True
    if f.get("channels", 1) > 1:
        # a window passes the packed word through: its channels as they came
        interface["output"]["channels"] = f["channels"]
        interface["output"]["field_bits"] = img.bits // f["channels"]
    if f.get("out_channels", 1) > 1:
        # a stack: its channels in one word, channel 0 lowest
        interface["output"]["channels"] = f["out_channels"]
        interface["output"]["field_bits"] = f["out_field"]
    return Core({
        "verilog": verilog,
        "interface": interface,
        "channels": f.get("out_channels", f.get("channels", 1)),
        "in_bits": img.word_bits,
        "out_bits": f["out_bits"],
        "signed": f["out_signed"],
        # every register port; and the model's own registers, which are
        # what a block declares
        "params": [(n, b) for n, b, _, _, _ in params],
        "registers": [(n, b) for n, b, _, _, _ in f["params"]],
        # and each context field's, for a wrapper that offers it as a
        # register: the traced size where it follows the frame's
        "param_defaults": {**{n: d for n, _, _, d, _ in params},
                           **{n: {"width": img.width,
                                  "height": img.height}.get(follows, 0)
                              for n, _, follows in img.context}},
        # the window's reach: rows of line buffer, columns of shift
        "M": max((k["K"][0] - 1 for k in f.get("stencil", {}).get(
            "sources", [])), default=0),
        "N": max((k["K"][1] - 1 for k in f.get("stencil", {}).get(
            "sources", [])), default=0),
        "out_rows": f.get("out_size", (img.height, img.width))[0],
        "out_cols": f.get("out_size", (img.height, img.width))[1],
        "module": module_name,
        "image": img,
        "memories": [{"instance": m["instance"], "module":
                      f"{module_name}_{m['seam']}", "bits": m["bits"],
                      "depth": m["depth"], "read": "sync"} for m in mems],
        "pipeline_stages": f.get("stages", 1),
        "flush_cycles": f.get("stages", 1) + 1,
        # what a testbench or a composer waits, past the last pixel, for
        # the frame's output to finish
        "vdrain": f.get("drain", 8) + f.get("stages", 1),
    })
