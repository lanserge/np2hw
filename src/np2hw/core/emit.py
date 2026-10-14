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
from .passes import Unit


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
        self.frame: dict[str, str] = {}       # parameter -> run-time register
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
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
        if k == "hw.parameter":
            name = n.attr("name")
            return self.wire(v, self.frame.get(name, name), "p")
        if k == "hw.param":
            return f"param_{n.attr('name')}"
        if k == "hw.const":
            return self.wire(v, str(n.attr("value")), "c")
        if k == "hw.regread":
            return f"r_{n.attr('name')}"
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
        raise NotImplementedError(f"no printing for {k}")


def generate(unit: Unit, module_name: str, dev=None,
             frame: str = "build") -> dict:
    """Print the unit; return its Core, as every generated module does.
    `dev` is the device family it is timed against (the default when
    None), whose shapes the printing honours. frame="runtime" reads the
    raster's extents from the frame registers (np2hw.verilog.FRAMES), which
    the frame_width and frame_height ports load as each frame enters, where
    "build" reads the WIDTH and HEIGHT parameters: the same nodes, printed
    against a run-time number instead of a constant."""
    from ..verilog import Core, _interface, _frame_params
    f = unit.facts
    img = f["image"]
    p = _Printer(dev)
    p.frame = {"WIDTH": "fw", "HEIGHT": "fh"} if frame == "runtime" else {}
    for n in walk(unit.all_roots()):
        p.names[id(n)] = p.node(n)
    params = list(f["params"]) + _frame_params(img, frame)
    head = [f"// generated by np2hw -- lowered from the graph "
            f"({len(params)} register port(s))",
            "// Every wire is sized from its node's exact value range; the",
            "// counters, flags and registers are nodes like the rest.",
            f"module {module_name} #(",
            f"    parameter WIDTH  = {img.width},",
            f"    parameter HEIGHT = {img.height}",
            ") (",
            "    input  wire clk,",
            "    input  wire rst,",
            "    input  wire in_valid,",
            "    output wire in_ready,",
            "    input  wire in_sof,",
            f"    input  wire [{img.bits - 1}:0] in_data,"]
    for name, bits, signed, _, doc in params:
        if doc:
            head.append(f"    // {doc}")
        head.append(f"    input  wire {'signed ' if signed else ''}"
                    f"[{bits - 1}:0] param_{name},")
    sgn = "signed " if f["out_signed"] else ""
    head += ["    output wire out_valid,",
             "    input  wire out_ready,",
             "    output wire out_sof,",
             "    output wire out_eol,",
             "    output wire out_last,",
             f"    output wire {sgn}[{f['out_bits'] - 1}:0] out_data",
             ");"]
    if frame == "runtime":
        # loaded at reset and as a frame's first beat is taken, so a size
        # that changes mid-stream lands at a frame's start
        head += [f"    reg [15:0] fw = {img.width}, fh = {img.height};",
                 "    always @(posedge clk)",
                 "        if (rst || (in_valid && in_sof && in_ready)) begin",
                 "            fw <= param_frame_width; fh <= param_frame_height;",
                 "        end"]
    decls, body = [], []
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
    verilog = "\n".join(head + decls + p.lines + body + assigns +
                        ["endmodule"])
    return Core({
        "verilog": verilog,
        "interface": _interface(img.bits, f["out_bits"], f["out_signed"],
                                params),
        "weighted": {(0, 0): 1},
        "in_bits": img.bits,
        "out_bits": f["out_bits"],
        "signed": f["out_signed"],
        "post": [],
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": 0, "N": 0,
        "out_rows": img.height, "out_cols": img.width,
        "module": module_name,
        "image": img,
        "memories": [],
        "pipeline_stages": f.get("stages", 1),
        "flush_cycles": f.get("stages", 1) + 1,
    })
