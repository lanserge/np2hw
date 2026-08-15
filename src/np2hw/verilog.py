"""np2hw Verilog generator (v1) for the line-based IR.

A pipeline of weighted sums (add / mac) flattens to a *weighted* map of image
taps: {(row_off, col_off): weight}. Each tap reads image[r+row_off, c+col_off]
for output pixel (r, c); the weight is the product of the Const coefficients
along its path. That is a streaming window filter, emitted as the classic
datapath:

    line buffers (one per vertical row of delay)
        -> per-row column shift registers (horizontal delay)
        -> Σ weight * pixel
        -> trailing pointwise ops (>> for div, * for mul/gain)

Coefficients:
  * Const on spatial taps  -> integer weights folded into the tap map (exact).
  * Param/Const pointwise  -> trailing 'mul' (and 'div') ops, peeled off the
    spatial cone. A Param becomes a CONFIG REGISTER input port on the module.

Scope of v1: non-negative weights (signed taps, e.g. Sobel, are a follow-up);
Param coefficients only as trailing pointwise ops (not per spatial tap).

Validity: out_valid is gated on (row >= M) && (col >= N) -- the interior region
where the whole window is in-bounds, the same region reference() computes.
"""
from __future__ import annotations

import math

import textwrap
from dataclasses import dataclass, replace as dc_replace

import numpy as np

from .ir import (SourceLine, HProcLine, VProcLine, Image2D, Const, Param,
                 ExprLine, PExpr, PhaseRef)
from .regmap import AddrMap, Reg


# Video geometry is 16 bits wherever it is stated -- the protocol
# header says so, and the register map repeats it. The position
# counters used to be 32-bit integers doing an 11-bit job, which put a
# 32-bit carry chain on a block RAM's address port and made the column
# counter, not the arithmetic, a stencil's critical path.
POS = 16


def _clog2(n: int) -> int:
    return math.ceil(math.log2(n)) if n > 1 else 0


def _bitlen(v: int) -> int:
    return max(1, int(v).bit_length())


def _coeff_range(coeff):
    """(lo, hi) value range of a tap coefficient (int const or Param register)."""
    if isinstance(coeff, Param):
        if coeff.signed:
            return -(1 << (coeff.bits - 1)), (1 << (coeff.bits - 1)) - 1
        return 0, (1 << coeff.bits) - 1
    return coeff, coeff


def _acc_range(weighted, in_lo, in_hi):
    lo = hi = 0
    for coeff in weighted.values():
        clo, chi = _coeff_range(coeff)
        corners = [clo * in_lo, clo * in_hi, chi * in_lo, chi * in_hi]
        hi += max(corners)
        lo += min(corners)
    return lo, hi


def _mul_coeff(a, b):
    """Fold two tap coefficients (each int or Param) into one. Only foldable when
    at most one is a Param and any constant factor is 1 (a programmable kernel
    keeps unit folding factors)."""
    ap, bp = isinstance(a, Param), isinstance(b, Param)
    if not ap and not bp:
        return a * b
    if ap and not bp:
        if b == 1:
            return a
        raise ValueError("cannot scale a Param tap coefficient by a constant != 1")
    if bp and not ap:
        if a == 1:
            return b
        raise ValueError("cannot scale a Param tap coefficient by a constant != 1")
    raise ValueError("cannot multiply two Param tap coefficients")


def _add_coeff(existing, new):
    if isinstance(existing, Param) or isinstance(new, Param):
        raise ValueError("cannot sum onto a Param tap coefficient (same tap twice)")
    return existing + new


def _tap_params(weighted, already):
    """(name, bits, signed, default, description) for each distinct Param tap
    coefficient not already listed (e.g. as a trailing mulp/addp)."""
    seen = {p[0] for p in already}
    out = []
    for coeff in weighted.values():
        if isinstance(coeff, Param) and coeff.name not in seen:
            out.append((coeff.name, coeff.bits, coeff.signed, coeff.default,
                        coeff.description))
            seen.add(coeff.name)
    return out


def _tap_term(px, coeff, signed, in_signed):
    """Verilog for one tap: const folds to `w*px`; a Param becomes a register
    multiplier `param_x * px`, with $signed casts when the datapath is signed."""
    if signed:
        px = f"$signed({px})" if in_signed else f"$signed({{1'b0, {px}}})"
    if isinstance(coeff, Param):
        if signed:
            pe = (f"$signed(param_{coeff.name})" if coeff.signed
                  else f"$signed({{1'b0, param_{coeff.name}}})")
        else:
            pe = f"param_{coeff.name}"
        return f"{pe} * {px}"
    return px if coeff == 1 else f"{coeff}*{px}"


def _range_bits(lo, hi):
    """(bits, signed) needed to hold integers in [lo, hi]."""
    if lo < 0:
        return 1 + max(_bitlen(hi if hi > 0 else 0), _bitlen(-lo)), True
    return _bitlen(hi), False


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
                    abbreviated: `M` and `N` are this emitter's private spelling
                    and nothing outside it should have to learn them

    Everything else in here (`weighted`, `post`, `image`, `M`, `N`, ...) is
    working detail of the emitter. It stays reachable because np2hw's own
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

    A core self-frames from its WIDTH/HEIGHT parameters, so it takes only `sof`
    to re-anchor and regenerates the rest. `sink` marks a module that consumes a
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
    }


def _param_port(name, bits, signed, description, indent="    "):
    """A register's port declaration, with its description as a comment.

    The description comes from the Param that declared it, so the person reading
    the generated RTL in review sees what the register means without having to
    find the Python. A generated file that is correct but unreadable has failed.
    """
    lines = []
    if description:
        body = " ".join(str(description).split())
        for line in textwrap.wrap(body, width=72):
            lines.append(f"{indent}// {line}")
    lines.append(f"{indent}input  wire {'signed ' if signed else ''}"
                 f"[{bits-1}:0] param_{name},")
    return lines


def _range_bits_as(lo, hi, signed):
    """Bits to hold [lo, hi] on a datapath of the GIVEN signedness.

    Width comes from the value range; signedness comes from the dtype the model
    declared (Line2D.declared). The two are not independent: a range that happens
    to be non-negative still needs a sign bit once it is carried on a signed
    datapath, or its top reads as negative. A 12-bit pixel in [0, 4095] on a
    signed datapath is 13 bits, not 12 -- emitting `wire signed [11:0]` there
    turns 4095 into -1, and a following clip(0, ...) floors it to zero.

    Use _range_bits when nothing has forced the signedness yet; use this once it
    is known.
    """
    if not signed:
        if lo < 0:
            raise ValueError(f"range [{lo}, {hi}] cannot be held unsigned")
        return _bitlen(hi)
    return 1 + max(_bitlen(hi) if hi > 0 else 1, _bitlen(-lo) if lo < 0 else 1)


def _datapath_signed(spatial_line, range_signed, params):
    """Whether the accumulator is signed: declared dtype, range, or a register.

    The model's declared dtype comes FIRST -- it is the specification. The range
    and the coefficient registers can only add signedness, never remove it.
    """
    declared = getattr(spatial_line, "declared", None)
    declared_signed = bool(declared[1]) if declared else False
    return declared_signed or range_signed or any(ps for _, _, ps, _, _ in params)


def _emit_post(emit, post, acc_bits, acc_signed, prefix="", signal=None,
               clk_ns=None, base=None, no_cut_before=(), staging=None,
               label="chain", dev=None):
    """Emit the trailing pointwise stages; return (result_wire, bits, signed).

    `emit(line)` appends one Verilog line. `prefix` namespaces the wires so
    multiple datapaths (mux branches) don't collide. `signal` maps a Param name
    to the signal that carries it, defaulting to the input port `param_<name>`;
    a phase-sliced datapath passes a mapping so its operands come from muxes
    rather than straight from ports.

    With clk_ns, the chain is CUT where the traced depth would exceed
    the budget: the running value is registered (!stall-enabled) and
    the rest of the chain reads the register -- the linear-chain twin
    of the DAG cutter. `base` is the depth already spent in front of
    the chain (tap adds, a coefficient mux); ops in `no_cut_before`
    (position-muxed operands, timed at ingest) refuse a cut in front
    of them; each cut appends to `staging`, whose length is the number
    of boundaries the caller must delay its framing by."""
    if signal is None:
        def signal(name):
            return f"param_{name}"
    budget_fam = None
    if clk_ns is not None:
        from .timing import Depth, device, post_op_cost
        budget_fam = device(dev)
        spent = base if base is not None else Depth()

        def op_cost(op, bits):
            return post_op_cost(op, bits, budget_fam)
    prev, cur_bits, cur_signed = f"{prefix}acc", acc_bits, acc_signed
    for i, op in enumerate(post):
        if budget_fam is not None:
            cost = op_cost(op, cur_bits)
            if budget_fam.ns(cost) > clk_ns:
                raise ValueError(
                    f"{label}: operation {op[0]!r} alone estimates "
                    f"{budget_fam.ns(cost):.1f} ns against the {clk_ns:.1f} "
                    f"ns budget -- a register cannot land inside one "
                    "operation. Slow the clock or restructure the model.")
            if budget_fam.ns(spent + cost) > clk_ns:
                if i in no_cut_before:
                    raise ValueError(
                        f"{label}: the budget asks for a register in front "
                        f"of {op[0]!r}, but its coefficient is selected by "
                        "the pixel's position at ingest; this version cuts "
                        "only after the position-muxed operations. Slow "
                        "the clock or restructure the model.")
                emit(f"    reg {'signed ' if cur_signed else ''}"
                     f"[{cur_bits-1}:0] {prev}_q;")
                emit(f"    always @(posedge clk) if (!stall) "
                     f"{prev}_q <= {prev};")
                prev = f"{prev}_q"
                if staging is not None:
                    staging.append(i)
                spent = cost
            else:
                spent = spent + cost
        if op[0] == "shr":
            cur_bits = max(1, cur_bits - op[1])
            rhs = f"{prev} >>> {op[1]}" if cur_signed else f"{prev} >> {op[1]}"
        elif op[0] == "mulc":
            k = op[1]
            cur_signed = cur_signed or k < 0
            cur_bits = _bitlen(abs(k) * ((1 << cur_bits) - 1)) + (1 if cur_signed else 0)
            rhs = f"{prev} * {k}"
        elif op[0] == "trunc":
            if op[1] >= cur_bits:                        # widening declaration: no-op
                continue
            cur_bits = op[1]
            rhs = f"$signed({prev}[{cur_bits-1}:0])" if cur_signed else f"{prev}[{cur_bits-1}:0]"
        elif op[0] == "clip":
            lo, hi = op[1], op[2]
            rhs = f"({prev} > {hi}) ? {hi} : (({prev} < {lo}) ? {lo} : {prev})"
            cur_bits, cur_signed = _range_bits(lo, hi)   # output range is [lo, hi]
        elif op[0] == "addp":                            # add a bias register
            name, pbits, psg = op[1], op[2], op[3]
            res_signed = cur_signed or psg
            wire = signal(name)
            pexpr = (f"$signed({wire})" if psg else
                     (f"$signed({{1'b0, {wire}}})" if res_signed else wire))
            prevx = (f"$signed({{1'b0, {prev}}})" if res_signed and not cur_signed else prev)
            cur_bits, cur_signed = max(cur_bits, pbits) + 1, res_signed
            rhs = f"{prevx} + {pexpr}"
        else:                                            # mulp
            name, pbits, psg = op[1], op[2], op[3]
            res_signed = cur_signed or psg
            wire = signal(name)
            pexpr = (f"$signed({wire})" if psg else
                     (f"$signed({{1'b0, {wire}}})" if res_signed else wire))
            prevx = (f"$signed({{1'b0, {prev}}})" if res_signed and not cur_signed else prev)
            cur_bits, cur_signed = cur_bits + pbits, res_signed
            rhs = f"{prevx} * {pexpr}"
        emit(f"    wire {'signed ' if cur_signed else ''}[{cur_bits-1}:0] {prefix}stage{i} = {rhs};")
        prev = f"{prefix}stage{i}"
    return prev, cur_bits, cur_signed


# --------------------------------------------------------------------------- #
# Flatten the spatial cone to a weighted tap map
# --------------------------------------------------------------------------- #

def flatten_weighted(line) -> dict:
    """Return {(row_off, col_off): coeff} for `line`, where coeff is an int (a
    folded constant) OR a Param (a programmable, unfolded tap coefficient).
    Pointwise mul/div/addp must be peeled off first (see peel_pointwise)."""
    def merge(out, key, term):
        out[key] = _add_coeff(out[key], term) if key in out else term

    if isinstance(line, SourceLine):
        if line.src_line is None:                       # reads the image
            return {(line.vindexer.start, 0): 1}
        dr = line.vindexer.start                        # reads a computed line
        return {(r + dr, c): w for (r, c), w in flatten_weighted(line.src_line).items()}

    if isinstance(line, HProcLine):
        if line.op.name in ("div", "mul", "addp"):
            raise ValueError("pointwise op in spatial cone; peel it first")
        base = flatten_weighted(line.line)
        out: dict = {}
        for coeff, off in line.taps:
            cval = coeff.value if isinstance(coeff, Const) else coeff
            for (r, c), w in base.items():
                merge(out, (r, c + off), _mul_coeff(w, cval))
        return out

    if isinstance(line, VProcLine):
        out = {}
        for coeff, inp in zip(line.coeffs, line.lines):
            cval = coeff.value if isinstance(coeff, Const) else coeff
            for key, w in flatten_weighted(inp).items():
                merge(out, key, _mul_coeff(w, cval))
        return out

    raise TypeError(f"cannot flatten {type(line).__name__}")


def find_image(line) -> Image2D:
    if isinstance(line, SourceLine) and line.src_line is None:
        return line.source
    for inp in line.inputs:
        im = find_image(inp)
        if im is not None:
            return im
    return None


def peel_pointwise(line):
    """Peel trailing pointwise ops off the spatial cone. Returns
    (spatial_line, post_ops) where post_ops apply in order to the accumulator:
      ('shr', amount)            div by 2**amount
      ('mulc', value)            multiply by Const value
      ('mulp', name, bits)       multiply by Param register `name`
      ('trunc', nbits)           astype: keep the low nbits"""
    post = []
    while isinstance(line, HProcLine) and line.op.name in ("div", "mul", "trunc", "clip", "addp"):
        if line.op.name == "div":
            post.append(("shr", line.shift))
        elif line.op.name == "trunc":
            post.append(("trunc", line.clip_bits))
        elif line.op.name == "clip":
            post.append(("clip", line.clip_lo, line.clip_hi))
        elif line.op.name == "addp":
            c = line.coeff
            post.append(("addp", c.name, c.bits, c.signed, c.default,
                         c.description))
        else:
            c = line.coeff
            post.append(("mulc", c.value) if isinstance(c, Const)
                        else ("mulp", c.name, c.bits, c.signed, c.default,
                              c.description))
        line = line.line
    post.reverse()
    return line, post


# --------------------------------------------------------------------------- #
# Host-side oracle
# --------------------------------------------------------------------------- #

def reference(weighted: dict, A: np.ndarray, post=(), params=None) -> np.ndarray:
    """Weighted window sum over the valid interior, then the pointwise ops."""
    params = params or {}
    M = max(r for r, _ in weighted)
    N = max(c for _, c in weighted)
    H, W = A.shape
    acc = np.zeros((H - M, W - N), dtype=np.int64)
    for (dr, dc), w in weighted.items():
        acc += w * A[dr:dr + H - M, dc:dc + W - N].astype(np.int64)
    for op in post:
        if op[0] == "shr":
            acc = acc >> op[1]
        elif op[0] == "mulc":
            acc = acc * op[1]
        elif op[0] == "trunc":
            acc = acc & ((1 << op[1]) - 1)              # keep low N bits
        elif op[0] == "clip":
            acc = np.clip(acc, op[1], op[2])
        else:                                           # mulp
            acc = acc * int(params[op[1]])
    return acc


# --------------------------------------------------------------------------- #
# Verilog module emission
# --------------------------------------------------------------------------- #

def _phase_order(pair):
    """(base, complement) for one axis, base first so it selects 0."""
    a, b = pair
    inverted = (lambda d: d.invert if isinstance(d, PhaseRef) else bool(d))
    return (b, a) if inverted(a) else (a, b)


def _phase_expr(desc):
    """Verilog XOR term that turns a position bit into a plane select."""
    if isinstance(desc, PhaseRef):
        return f"param_{desc.param.name}"
    return None                       # constant phase: the position bit IS the select


def _post_shape(post):
    """A post chain's structure, ignoring which registers it names.

    Two planes with the same shape differ only in their coefficients, which means
    one datapath with a mux on the coefficient does the job of four datapaths.
    That is the whole area argument for lowering a phase-sliced model this way.
    """
    shape = []
    for op in post:
        if op[0] in ("addp", "mulp"):
            shape.append((op[0], op[2], op[3]))       # kind, bits, signed
        else:
            shape.append(tuple(op))
    return tuple(shape)


def _common_name(names):
    """A readable name for a coefficient selected across planes."""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix.rstrip("_")
    return f"{prefix or 'coeff'}_by_phase"




def _emit_expr_dag(a, roots, image, plan=None):
    """Emit the shared wire namespace of one or more expression roots,
    optionally cut into pipeline stages.

    Without a plan this is the classic single-stage combinational DAG.
    With one -- (stage_of, n_stages) from timing.assign_stages -- each
    node's wire is emitted in its stage's group, and any value read
    across a boundary rides a !stall-enabled delay register per stage
    crossed, so every pixel's operands are always from the same input
    word: retiming, never resampling. param and const leaves float
    (a register port is stable across a pixel's flight, a literal is
    wiring); acc and tap pin to stage 0 with the input word.

    Returns (per-root references as read at the FINAL stage, per-root
    widths, node count).
    """
    stage_of, n_stages = plan if plan else ({}, 1)
    final = n_stages - 1
    emitted = {}                 # id(node) -> (text, stage-or-None, width)
    counter = [0]
    bucket = [[] for _ in range(n_stages)]
    pipes = {}                   # wire name -> [width, max stages crossed]
    roms = {}                    # id(table) -> table, in first-use order
    reads = []                   # (reg name, width, table name, address)

    def width(node):
        negative = (-node.lo - 1).bit_length() if node.lo < 0 else 0
        return max(node.hi.bit_length(), negative) + 1     # +1: signed carrier

    def ref(node, use_stage):
        text, born, w = emitted[id(node)]
        if born is None:                       # const literal / floating param
            return text
        delta = use_stage - born
        if delta <= 0:
            return text
        have = pipes.setdefault(text, [w, 0])
        have[1] = max(have[1], delta)
        return f"{text}_q{delta}"

    def emit(node):
        key = id(node)
        if key in emitted:
            return
        for child in node.args:
            if isinstance(child, PExpr):
                emit(child)
        s = stage_of.get(key, 0)
        w = width(node)
        floating = False
        if node.op == "acc":
            expr = ("$signed(in_data)" if image.signed
                    else "$signed({1'b0, in_data})")
        elif node.op in ("row", "col"):
            # The position counters the core already keeps for its own
            # framing: free to read, and pinned to this pixel (they ride
            # the delay lines like any other stage-0 value).
            wire = "erow" if node.op == "row" else "ecol"
            expr = f"$signed({{1'b0, {wire}[{max(1, w - 1) - 1}:0]}})"
        elif node.op == "tap":
            r, c, shift, fld = node.args
            if (r, c) != (0, 0):
                raise NotImplementedError(
                    "a window tap in a pointwise datapath: this model has "
                    "no line buffers; pad the image and use the windowed "
                    "emitters")
            base = ("in_data" if fld is None
                    else f"in_data[{shift + fld - 1}:{shift}]")
            expr = (f"$signed({base})" if image.signed and fld is None
                    else f"$signed({{1'b0, {base}}})")
        elif node.op == "abs":
            inner = ref(node.args[0], s)
            expr = f"({inner} < 0) ? -{inner} : {inner}"
        elif node.op == "lt":
            expr = (f"({ref(node.args[0], s)} < {ref(node.args[1], s)}) "
                    "? 2'sd1 : 2'sd0")
        elif node.op == "sel":
            expr = (f"({ref(node.args[0], s)} != 0) ? "
                    f"{ref(node.args[1], s)} : {ref(node.args[2], s)}")
        elif node.op == "const":
            emitted[key] = (str(node.args[0]), None, w)
            return
        elif node.op == "param":
            leaf = node.args[0]
            expr = (f"$signed(param_{leaf.name})" if leaf.signed
                    else f"$signed({{1'b0, param_{leaf.name}}})")
            floating = True
        elif node.op == "gather":
            parent, index = node.args
            idx = ref(index, s)
            kb = max(1, (parent.shape[0] - 1).bit_length())
            read = f"{parent.name}_lut[{idx}[{kb-1}:0]]"
            expr = (f"$signed({read})" if parent.signed
                    else f"$signed({{1'b0, {read}}})")
        elif node.op == "rom":
            # The address is captured at the END of the previous stage;
            # the data lands here. Emitted as a REGISTER, which is what
            # makes the tools build a block RAM.
            table, index = node.args
            roms.setdefault(id(table), table)
            addr = ref(index, s - 1)
            kb = max(1, (len(table) - 1).bit_length())
            name = f"e{counter[0]}"
            counter[0] += 1
            reads.append((name, w, table.name, f"{addr}[{kb-1}:0]"))
            emitted[key] = (name, s, w)
            return
        elif node.op in ("add", "sub", "mul"):
            op = {"add": "+", "sub": "-", "mul": "*"}[node.op]
            expr = f"{ref(node.args[0], s)} {op} {ref(node.args[1], s)}"
        elif node.op == "shr":
            expr = f"{ref(node.args[0], s)} >>> {node.args[1]}"
        elif node.op == "mask":
            expr = (f"$signed({{1'b0, "
                    f"{ref(node.args[0], s)}[{node.args[1]-1}:0]}})")
        elif node.op == "clip":
            source, lo, hi = node.args
            inner = ref(source, s)
            expr = (f"({inner} > {hi}) ? {hi} : "
                    f"(({inner} < {lo}) ? {lo} : {inner})")
        else:
            raise NotImplementedError(f"expression op {node.op!r}")
        name = f"e{counter[0]}"
        counter[0] += 1
        bucket[s].append(f"    wire signed [{w-1}:0] {name} = {expr};")
        emitted[key] = (name, None if floating else s, w)

    for root in roots:
        emit(root)
    lanes = [ref(root, final) for root in roots]
    widths = [width(root) for root in roots]

    for table in roms.values():
        size = len(table)
        a(f"    // {table.name}: {size}-entry constant table, read through a")
        a("    // register so the tools infer block RAM. The contents travel")
        a("    // INSIDE this file: a generated design that needs a loose")
        a("    // .mem beside it is a design that breaks when it moves.")
        a(f"    reg [{table.bits-1}:0] {table.name} [0:{size-1}];")
        a("    initial begin")
        digits = (table.bits + 3) // 4
        for i, value in enumerate(table.data.tolist()):
            v = int(value) & ((1 << table.bits) - 1)
            a(f"        {table.name}[{i}] = {table.bits}'h{v:0{digits}x};")
        a("    end")
        a("")
    if reads:
        for name, w, _t, _addr in reads:
            a(f"    reg signed [{w-1}:0] {name};")
        a("")
    if pipes:
        a("    // stage-boundary delay lines: one register per boundary a")
        a("    // value crosses, advancing with the pipe (!stall), so every")
        a("    // consumer sees operands from the same input word")
        for name, (w, deep) in pipes.items():
            regs = ", ".join(f"{name}_q{k}" for k in range(1, deep + 1))
            a(f"    reg signed [{w-1}:0] {regs};")
        a("")
    for s in range(n_stages):
        if n_stages > 1:
            a(f"    // ---- pipeline stage {s} ----")
        for line_text in bucket[s]:
            a(line_text)
    if pipes or reads:
        a("    always @(posedge clk) if (!stall) begin")
        for name, _w, table_name, addr in reads:
            a(f"        {name} <= {table_name}[{addr}];")
        for name, (_w, deep) in pipes.items():
            for k in range(1, deep + 1):
                src = name if k == 1 else f"{name}_q{k-1}"
                a(f"        {name}_q{k} <= {src};")
        a("    end")
    return lanes, widths, counter[0]


def _emit_expr_framing(a, gate, plan, data_line):
    """The registered framing of a pointwise core: counters at ingest,
    flags riding delay lines beside the data, outputs at the far end.
    Without a plan this is the classic one-register framing; with one,
    the valid/sof/eol/last quartet is computed at stage 0 and shifted
    n_stages-1 times, and the counters stay at ingest untouched."""
    stages = (plan[1] if plan else 1) - 1
    if stages == 0:
        a("    always @(posedge clk) begin")
        a("        if (rst) begin")
        a("            col <= 0; row <= 0; out_valid <= 1'b0;")
        a("            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;")
        a("        end else begin")
        a("            if (!stall) begin")
        a("            if (in_valid) begin")
        a(f"                out_valid <= {gate};")
        a(f"                out_sof <= ({gate}) && (erow == 0) && (ecol == 0);")
        a(f"                out_eol <= ({gate}) && (ecol == WIDTH-1);")
        a(f"                out_last <= ({gate}) && (erow == HEIGHT-1) && "
          f"(ecol == WIDTH-1);")
        a(f"                {data_line}")
        a("                if (in_sof) begin col <= 1; row <= 0; end")
        a("                else if (col == WIDTH-1) begin")
        a("                    col <= 0; row <= (row == HEIGHT-1) ? 0 : row + 1;")
        a("                end else col <= col + 1;")
        a("            end else begin out_valid <= 1'b0; out_sof <= 1'b0;")
        a("                out_eol <= 1'b0; out_last <= 1'b0; end")
        a("            end")
        a("        end")
        a("    end")
        return
    names = [f"{f}_q{k}" for k in range(1, stages + 1)
             for f in ("vld", "sof", "eol", "lst")]
    clears = " ".join(n + " <= 1'b0;" for n in names)
    a(f"    reg {', '.join(names)};")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a(f"            {clears}")
    a("        end else if (!stall) begin")
    a(f"            vld_q1 <= in_valid && ({gate});")
    a(f"            sof_q1 <= in_valid && ({gate}) && (erow == 0) && "
      f"(ecol == 0);")
    a(f"            eol_q1 <= in_valid && ({gate}) && (ecol == WIDTH-1);")
    a(f"            lst_q1 <= in_valid && ({gate}) && (erow == HEIGHT-1) && "
      f"(ecol == WIDTH-1);")
    for k in range(2, stages + 1):
        a(f"            vld_q{k} <= vld_q{k-1}; sof_q{k} <= sof_q{k-1}; "
          f"eol_q{k} <= eol_q{k-1}; lst_q{k} <= lst_q{k-1};")
    a("        end")
    a("    end")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin col <= 0; row <= 0; end")
    a("        else if (!stall && in_valid) begin")
    a("            if (in_sof) begin col <= 1; row <= 0; end")
    a("            else if (col == WIDTH-1) begin")
    a("                col <= 0; row <= (row == HEIGHT-1) ? 0 : row + 1;")
    a("            end else col <= col + 1;")
    a("        end")
    a("    end")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            out_valid <= 1'b0; out_sof <= 1'b0;")
    a("            out_eol <= 1'b0; out_last <= 1'b0;")
    a("        end else if (!stall) begin")
    a(f"            out_valid <= vld_q{stages}; out_sof <= sof_q{stages};")
    a(f"            out_eol <= eol_q{stages}; out_last <= lst_q{stages};")
    a(f"            {data_line}")
    a("        end")
    a("    end")


def _generate_expr(line, module_name, plan=None) -> dict:
    """Emit a pointwise expression-DAG model: LUTs, forks, recombination.

    One full-rate datapath with the standard registered framing, like
    the phase cores -- combinational in one stage, or cut into pipeline
    stages by a timing plan. Every internal wire is SIGNED and sized
    from its node's exact value range -- uniform signedness costs one bit on
    some wires and removes the whole class of "signed by one bit too few"
    defects; the range does the narrowing that matters. A gather becomes a
    register array read: the registers assembled into an indexed wire array,
    the index proven in range AT TRACE TIME, so the mux needs no guard.
    """
    image, root = line.image, line.root
    in_bits = image.bits

    params = []
    arrays = []
    for kind, declared in line.params:
        if kind == "gather":
            arrays.append(declared)
            for index in range(declared.shape[0]):
                leaf = declared[index]
                params.append((leaf.name, leaf.bits, leaf.signed,
                               leaf.default, leaf.description))
        else:                                   # scalar register leaf
            params.append((declared.name, declared.bits, declared.signed,
                           declared.default, declared.description))

    L = []
    a = L.append
    a(f"// generated by np2hw -- pointwise expression datapath "
      f"({len(arrays)} register array(s), "
      f"{len(params)} register port(s))")
    a("// Wires are signed and sized from each node's exact value range;")
    a("// lookups are register arrays muxed by a data-derived index whose")
    a("// range was proven at trace time.")
    if plan:
        a(f"// timing: cut into {plan[1]} pipeline stages by the traced "
          f"depth model")
    a(f"module {module_name} #(")
    a(f"    parameter WIDTH  = {image.width},")
    a(f"    parameter HEIGHT = {image.height}")
    a(") (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    input  wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, doc in params:
        L.extend(_param_port(name, bits, psg, doc))
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")
    a("    output reg  out_eol,")
    a("    output reg  out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a(f"    reg [{POS-1}:0] col;")
    a(f"    reg [{POS-1}:0] row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    a("")

    for parent in arrays:
        size = parent.shape[0]
        a(f"    // {parent.name}: {size}-entry register array, read by index")
        a(f"    wire [{parent.bits-1}:0] {parent.name}_lut [0:{size-1}];")
        for index in range(size):
            a(f"    assign {parent.name}_lut[{index}] = "
              f"param_{parent.name}_{index};")
        a("")

    (result,), _widths, nodes = _emit_expr_dag(a, [root], image, plan)
    out_signed = root.lo < 0
    magnitude = max(root.hi, -root.lo - 1 if root.lo < 0 else 0)
    out_bits = max(1, magnitude.bit_length() + (1 if out_signed else 0))
    a("")
    gate = "(erow < HEIGHT) && (ecol < WIDTH)"
    _emit_expr_framing(a, gate, plan,
                       f"out_data <= {result}[OUT_BITS-1:0];")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if out_signed else ""))
    return Core({
        "verilog": verilog,
        "interface": _interface(in_bits, out_bits, out_signed, params),
        "weighted": {(0, 0): 1},
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": [],
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": 0, "N": 0,
        "out_rows": image.height, "out_cols": image.width,
        "module": module_name,
        "image": image,
        "expr_nodes": nodes,
        "pipeline_stages": plan[1] if plan else 1,
        "flush_cycles": (plan[1] if plan else 1) + 1,
    })

def _walk_expr_params(roots):
    """Register leaves and gathered arrays across expression roots, in
    first-use order, deduplicated -- the multi-root twin of lower()'s walk.
    Recursion is generic over PExpr arguments, so every operator -- present
    and future -- is covered by construction."""
    from .ir import PExpr

    parents, seen = [], set()

    def walk(node):
        if node.op == "gather":
            parent = node.args[0]
            if ("gather", parent.name) not in seen:
                seen.add(("gather", parent.name))
                parents.append(("gather", parent))
            walk(node.args[1])
            return
        if node.op == "param":
            leaf = node.args[0]
            if ("param", leaf.name) not in seen:
                seen.add(("param", leaf.name))
                parents.append(("param", leaf))
            return
        for arg in node.args:
            if isinstance(arg, PExpr):
                walk(arg)

    for root in roots:
        walk(root)
    return parents


def _wexpr_taps(root):
    """Window positions a windowed expression reads."""
    from .ir import PExpr

    taps = set()

    def walk(node):
        if node.op == "tap":
            taps.add((node.args[0], node.args[1]))
        for arg in node.args:
            if isinstance(arg, PExpr):
                walk(arg)

    walk(root)
    return taps


def _generate_expr_stack(stack, module_name, plan=None) -> Core:
    """Emit np.stack of pointwise expression lanes: C datapaths, one word.

    The matrix-mix case (a colour matrix: unpacked channels in, three dot
    products out). Every channel's DAG is emitted into ONE wire namespace,
    so subexpressions shared between channels fold, and the results
    concatenate into the output word -- channel 0 in the low bits, each
    field as wide as the input's samples. A timing plan cuts the shared
    namespace once for all lanes: shallow lanes ride delay lines to the
    final stage so the word always assembles from one input pixel.
    """
    image = stack.channels[0].image
    roots = [t.expr for t in stack.channels]
    in_bits = image.bits
    field = in_bits // max(1, stack.in_channels)
    for i, root in enumerate(roots):
        if root.lo < 0:
            raise NotImplementedError(
                f"np.stack channel {i} can be negative "
                f"([{root.lo}, {root.hi}]); clip channels to their unsigned "
                "range before stacking into a word")
        if root.hi.bit_length() > field:
            raise NotImplementedError(
                f"np.stack channel {i} needs {root.hi.bit_length()} bits but "
                f"the word's fields are {field} (the input sample width); "
                "clip it first")

    params = []
    arrays = []
    for kind, declared in _walk_expr_params(roots):
        if kind == "gather":
            arrays.append(declared)
            for index in range(declared.shape[0]):
                leaf = declared[index]
                params.append((leaf.name, leaf.bits, leaf.signed,
                               leaf.default, leaf.description))
        else:
            params.append((declared.name, declared.bits, declared.signed,
                           declared.default, declared.description))

    L = []
    a = L.append
    a(f"// generated by np2hw -- {len(roots)}-channel pointwise expression "
      f"datapath, one shared wire namespace")
    a("// Channels concatenate into the output word, channel 0 in the low")
    a("// bits; the field width is the input's sample width.")
    if plan:
        a(f"// timing: cut into {plan[1]} pipeline stages by the traced "
          f"depth model")
    a(f"module {module_name} #(")
    a(f"    parameter WIDTH  = {image.width},")
    a(f"    parameter HEIGHT = {image.height}")
    a(") (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    input  wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, doc in params:
        L.extend(_param_port(name, bits, psg, doc))
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")
    a("    output reg  out_eol,")
    a("    output reg  out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a(f"    reg [{POS-1}:0] col;")
    a(f"    reg [{POS-1}:0] row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    a("")
    for parent in arrays:
        size = parent.shape[0]
        a(f"    // {parent.name}: {size}-entry register array, read by index")
        a(f"    wire [{parent.bits-1}:0] {parent.name}_lut [0:{size-1}];")
        for index in range(size):
            a(f"    assign {parent.name}_lut[{index}] = "
              f"param_{parent.name}_{index};")
        a("")

    lanes, widths, nodes = _emit_expr_dag(a, roots, image, plan)
    out_bits = field * len(roots)
    parts = []
    for name, have in zip(reversed(lanes), reversed(widths)):
        if have >= field:
            parts.append(f"{name}[{field-1}:0]")
        else:
            parts.append(f"{{{field - have}'b0, {name}}}")
    a("")
    a(f"    wire [{out_bits-1}:0] word = {{{', '.join(parts)}}};")
    gate = "(erow < HEIGHT) && (ecol < WIDTH)"
    _emit_expr_framing(a, gate, plan, "out_data <= word;")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", ""))
    interface = _interface(in_bits, out_bits, False, params)
    interface["output"]["channels"] = len(roots)
    interface["output"]["field_bits"] = field
    return Core({
        "verilog": verilog,
        "interface": interface,
        "weighted": {(0, 0): 1},
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": False,
        "post": [],
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": 0, "N": 0,
        "channels": len(roots),
        "out_rows": image.height, "out_cols": image.width,
        "module": module_name,
        "image": image,
        "expr_nodes": nodes,
        "pipeline_stages": plan[1] if plan else 1,
        "flush_cycles": (plan[1] if plan else 1) + 1,
    })


def _generate_phase(canvas, module_name, clk_ns=None, label=None,
                    dev=None) -> dict:
    """Emit a phase-sliced model: one datapath, coefficients muxed by position.

    The model wrote four interleaved planes and they partition the image, so at
    any pixel exactly one plane applies. Rather than four quarter-rate datapaths
    and a recombiner, this is ONE full-rate datapath whose coefficients are
    selected by the pixel's position -- which is what the four planes actually
    describe, and a great deal less hardware than taking the slicing literally.

    The phase registers make the mapping from position to plane programmable, so
    one bitstream serves every CFA order.
    """
    rows, cols = canvas.validate()
    rows, cols = _phase_order(rows), _phase_order(cols)
    image = canvas.image
    planes = [[canvas.plane(r, c) for c in cols] for r in rows]
    flat = [p for row in planes for p in row]

    for plane in flat:
        if plane.taps != {(0, 0): 1}:
            raise NotImplementedError(
                "a phase-sliced plane must be pointwise in this version; a stencil "
                "across a strided plane needs its own line-buffer geometry")

    shapes = {_post_shape(p.post) for p in flat}
    if len(shapes) != 1:
        raise NotImplementedError(
            "the phase planes have different structures; this version muxes the "
            "coefficients of ONE datapath, so every plane must apply the same "
            "sequence of operations and differ only in its registers")

    template = flat[0]
    in_bits = image.bits
    in_lo, in_hi = ((-(1 << (in_bits - 1)), (1 << (in_bits - 1)) - 1)
                    if image.signed else (0, (1 << in_bits) - 1))

    # Phase registers first, then one muxed coefficient per register position.
    phase_params = []
    for desc in (rows[0], cols[0]):
        if isinstance(desc, PhaseRef):
            phase_params.append((desc.param.name, 1, False, desc.param.default,
                                 desc.param.description))

    muxed = []                                   # (wire_name, bits, signed, [names])
    for index, op in enumerate(template.post):
        if op[0] not in ("addp", "mulp"):
            continue
        names = [p.post[index][1] for p in flat]
        muxed.append((_common_name(names), op[2], op[3], names,
                      [p.post[index][4] for p in flat],
                      [p.post[index][5] for p in flat]))

    params = list(phase_params)
    for _, bits, signed, names, defaults, docs in muxed:
        params += [(name, bits, signed, default, doc)
                   for name, default, doc in zip(names, defaults, docs)]

    acc_lo, acc_hi = _acc_range(template.taps, in_lo, in_hi)
    range_signed = _range_bits(acc_lo, acc_hi)[1]
    signed = (template.spatial_signed or range_signed
              or any(entry[2] for entry in params))
    acc_bits = _range_bits_as(acc_lo, acc_hi, signed)
    sgn = "signed " if signed else ""
    out_rows, out_cols = image.height, image.width

    L = []
    a = L.append
    a(f"// generated by np2hw -- phase-sliced pointwise op, {len(flat)} planes")
    a("// One full-rate datapath; the coefficient is selected by the pixel's")
    a("// position, and the phase registers make that mapping programmable.")
    a(f"module {module_name} #(")
    a(f"    parameter WIDTH  = {image.width},")
    a(f"    parameter HEIGHT = {image.height}")
    a(") (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    input  wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, _doc in params:
        L.extend(_param_port(name, bits, psg, _doc))
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")
    a("    output reg  out_eol,")
    a("    output reg  out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a(f"    reg [{POS-1}:0] col;")
    a(f"    reg [{POS-1}:0] row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    a("")
    a("    // Which plane this pixel belongs to. A plane taken at `p::2` holds the")
    a("    // positions where (position & 1) == p, so XOR against the phase")
    a("    // register turns position into plane select.")
    row_x, col_x = _phase_expr(rows[0]), _phase_expr(cols[0])
    a(f"    wire sel_row = erow[0]{f' ^ {row_x}' if row_x else ''};")
    a(f"    wire sel_col = ecol[0]{f' ^ {col_x}' if col_x else ''};")
    a("")

    for wire, bits, psg, names, _, _ in muxed:
        kind = "signed " if psg else ""
        a(f"    reg {kind}[{bits-1}:0] {wire};")
        a("    always @(*) begin")
        a("        case ({sel_row, sel_col})")
        for code, name in zip(("2'b00", "2'b01", "2'b10"), names[:3]):
            a(f"            {code}:   {wire} = param_{name};")
        a(f"            default: {wire} = param_{names[3]};")
        a("        endcase")
        a("    end")
        a("")

    a(f"    wire [{in_bits-1}:0] row0 = in_data;")
    terms = [_tap_term("row0", w, signed, image.signed)
             for _, w in sorted(template.taps.items())]
    a(f"    wire {sgn}[{acc_bits-1}:0] acc = {' + '.join(terms)};")

    # Re-point the template's register operands at the muxed wires.
    effective, cursor = [], 0
    for op in template.post:
        if op[0] in ("addp", "mulp"):
            effective.append((op[0], muxed[cursor][0], op[2], op[3], op[4]))
            cursor += 1
        else:
            effective.append(op)
    wires = {name: name for name, _, _, _, _, _ in muxed}
    # The chain's own cutter handles the budget: the depth in front of
    # it is the coefficient mux and the tap adds, and a cut may not
    # land before a position-muxed operation (its select is ingest-
    # timed). Each cut registers the running value; the framing below
    # delays the flags to match.
    base = None
    no_cut = ()
    staging = []
    if clk_ns is not None:
        from .timing import device
        base = device(dev).adder_chain(len(template.taps), acc_bits,
                                       muxed=muxed)
        no_cut = tuple(i for i, op in enumerate(effective)
                       if op[0] in ("addp", "mulp"))
    result, out_bits, out_signed = _emit_post(
        a, effective, acc_bits, signed,
        signal=lambda n: wires.get(n, f"param_{n}"),
        clk_ns=clk_ns, base=base, no_cut_before=no_cut, staging=staging,
        label=label or module_name, dev=dev)
    plan = (None, len(staging) + 1) if staging else None

    gate = "(erow < HEIGHT) && (ecol < WIDTH)"
    _emit_expr_framing(a, gate, plan, f"out_data <= {result};")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if out_signed else ""))
    return Core({
        "verilog": verilog,
        "interface": _interface(in_bits, out_bits, out_signed, params),
        "weighted": template.taps,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": template.post,
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": 0, "N": 0,
        "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name,
        "image": image,
        "phases": len(flat),
        "pipeline_stages": len(staging) + 1,
        "flush_cycles": len(staging) + 2,
    })


def _timing_roots(out_line):
    """PExpr roots this result kind exposes -- the pointwise stages the
    depth model covers in v1. Spatial stencils get shapes of their own
    later; their stages are shallow MACs today."""
    from .ir import ExprLine
    if isinstance(out_line, ExprLine):
        yield out_line.root, "expr"
    elif type(out_line).__name__ == "ChannelStack" \
            and getattr(out_line, "kind", None) == "expr":
        # expr channels are traced lanes; their DAG root is `.expr`,
        # exactly what _generate_expr_stack emits from
        for i, channel in enumerate(out_line.channels):
            yield channel.expr, f"channel{i}"


def _reads_memory(root) -> bool:
    """Does this DAG read a constant table anywhere?"""
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, PExpr) or id(node) in seen:
            continue
        seen.add(id(node))
        if node.op == "rom":
            return True
        stack.extend(node.args)
    return False


def generate(out_line, module_name="np2hw_top", framing="height",
             max_width=None, clk_ns=None, label=None, dev=None) -> dict:
    """framing='height' (default): the core self-frames by counting to HEIGHT.
    framing='eof': height-agnostic -- an `in_eof` input (the sensor's frame-end /
    VSYNC, pulsed on the last input pixel) triggers the bottom flush, and output
    EOF (out_last) comes from the flush draining the line buffers. The frame can
    be ANY number of lines.

    max_width=N: runtime-variable WIDTH. Line buffers are sized to a MAX_WIDTH
    parameter (=N); an `active_width` register input marks where each row ends
    (wrap / EOL / right-edge). So one synthesized core processes any line length
    <= MAX_WIDTH, set live. Combine with framing='eof' for full dynamic
    resolution (a reprogrammable sensor: active_width register + VSYNC height)."""
    plan = None
    roots = [root for root, _ in _timing_roots(out_line)]
    # A memory read is registered whatever the clock is, so a model that
    # reads a constant table gets staged even with no budget stated.
    budget = clk_ns if clk_ns is not None else (
        float("inf") if any(_reads_memory(r) for r in roots) else None)
    if budget is not None and roots:
        # Timing is a TRACED property: a stage the clock cannot hold is
        # cut into pipeline stages at generation time -- pure retiming,
        # so the twin stays bit-exact. The only refusal left is the
        # floor: one operation deeper than the clock, named.
        from .timing import assign_stages
        stage_of, n_stages = assign_stages(
            roots, budget, dev, label=label or module_name)
        if n_stages > 1:
            plan = (stage_of, n_stages)
    if type(out_line).__name__ == "Mux":                 # np.where(enable, A, B)
        return _generate_mux(out_line, module_name)
    if isinstance(out_line, ExprLine):                   # pointwise DAG / LUTs
        return _generate_expr(out_line, module_name, plan=plan)
    if type(out_line).__name__ == "ChannelStack":        # np.stack([...], axis=-1)
        if out_line.kind == "expr":
            return _generate_expr_stack(out_line, module_name, plan=plan)
        return _generate_phase_stencil(out_line, module_name,
                                       clk_ns=clk_ns, label=label, dev=dev)
    if type(out_line).__name__ == "PhaseCanvas":         # out[py::2, px::2] = ...
        if any(value.taps != {(0, 0): 1} or value.mode != "none"
               for _, _, value in out_line.branches):
            return _generate_phase_stencil(out_line, module_name,
                                           clk_ns=clk_ns, label=label,
                                           dev=dev)
        return _generate_phase(out_line, module_name,
                               clk_ns=clk_ns, label=label, dev=dev)
    image = find_image(out_line)
    if image is None:
        raise ValueError("no image source found in pipeline")
    if getattr(out_line, "edge", None) is not None:
        return _generate_edge(out_line, image, module_name, framing=framing,
                              max_width=max_width)
    if framing == "eof" or max_width is not None:
        raise NotImplementedError("framing='eof' / max_width need a vertical edge "
                                  "(bottom flush); use an edge-padded model")
    spatial_line, post = peel_pointwise(out_line)
    weighted = flatten_weighted(spatial_line)

    in_bits = image.bits
    if image.signed:
        in_lo, in_hi = -(1 << (in_bits - 1)), (1 << (in_bits - 1)) - 1
    else:
        in_lo, in_hi = 0, (1 << in_bits) - 1
    M = max(r for r, _ in weighted)
    N = max(c for _, c in weighted)
    acc_lo, acc_hi = _acc_range(weighted, in_lo, in_hi)
    range_signed = _range_bits(acc_lo, acc_hi)[1]
    params = [(op[1], op[2], op[3], op[4], op[5])
              for op in post if op[0] in ("mulp", "addp")]
    params += _tap_params(weighted, params)              # programmable kernel coeffs
    signed = _datapath_signed(spatial_line, range_signed, params)
    acc_bits = _range_bits_as(acc_lo, acc_hi, signed)
    sgn = "signed " if signed else ""
    rows_used = sorted({r for r, _ in weighted})

    L = []
    a = L.append
    a(f"// generated by np2hw -- {len(weighted)} taps, {M} line buffer(s), "
      f"shift depth {N}, {'signed' if signed else 'unsigned'} {acc_bits}b acc")
    a(f"module {module_name} #(")
    a(f"    parameter WIDTH  = {image.width},")
    a(f"    parameter HEIGHT = {image.height}")
    a(") (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    input  wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")                            # frame start (AXI TUSER); tie 0 to free-run
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, _doc in params:
        L.extend(_param_port(name, bits, psg, _doc))
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")                         # start of frame (first pixel)
    a("    output reg  out_eol,")                         # end of line (last pixel of row)
    a("    output reg  out_last,")                        # end of frame (final pixel)
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")  # OUT_SGN/OUT_BITS fixed below
    a(");")
    if out_line.shape is not None:
        out_rows, out_cols = out_line.shape
    else:
        out_rows, out_cols = image.height - M, image.width - N
    a(f"    reg [{POS-1}:0] col;")
    a(f"    reg [{POS-1}:0] row;")
    a("    wire stall = out_valid && !out_ready;")       # holding an unaccepted output
    a("    assign in_ready = !stall;")
    # SOF re-anchors the current pixel to frame (0,0); tie in_sof=0 to free-run.
    # Output framing is DERIVED from the (effective) input position, so it tracks
    # SOF too -- no separate output counter to drift on re-anchor.
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")

    # line buffers: chain0 = in_data; chainK = input delayed K rows
    a(f"    wire [{in_bits-1}:0] chain0 = in_data;")
    for k in range(1, M + 1):
        a(f"    reg  [{in_bits-1}:0] mem{k} [0:WIDTH-1];")
        a(f"    wire [{in_bits-1}:0] chain{k} = mem{k}[ecol];")
    for r in rows_used:
        a(f"    wire [{in_bits-1}:0] row{r} = chain{M - r};")

    # per-row column shift registers
    for r in rows_used:
        for d in range(1, N + 1):
            a(f"    reg  [{in_bits-1}:0] row{r}_d{d};")

    def pixel(r, dc):
        delay = N - dc
        return f"row{r}" if delay == 0 else f"row{r}_d{delay}"

    # weighted tap sum (signed datapath when any coeff/the input is signed);
    # Param coeffs become real multipliers (a programmable kernel), Consts fold.
    terms = [_tap_term(pixel(r, c), w, signed, image.signed)
             for (r, c), w in sorted(weighted.items())]
    a(f"    wire {sgn}[{acc_bits-1}:0] acc = {' + '.join(terms)};")

    # trailing pointwise stages (track signedness per stage)
    result, out_bits, out_signed = _emit_post(a, post, acc_bits, signed)

    # size-dependent bounds use the WIDTH/HEIGHT parameters (resolution-agnostic);
    # only M/N (the kernel) are baked. valid region = [M,HEIGHT) x [N,WIDTH).
    gate = f"(erow >= {M}) && (ecol >= {N}) && (erow < HEIGHT) && (ecol < WIDTH)"
    sof = f"({gate}) && (erow == {M}) && (ecol == {N})"
    eol = f"({gate}) && (ecol == WIDTH-1)"
    last = f"({gate}) && (erow == HEIGHT-1) && (ecol == WIDTH-1)"
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            col <= 0; row <= 0; out_valid <= 1'b0;")
    a("            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;")
    a("        end else begin")
    a("            if (!stall) begin")                    # freeze while output is held
    a("            if (in_valid) begin")
    a(f"                out_valid <= {gate};")
    a(f"                out_sof <= {sof}; out_eol <= {eol}; out_last <= {last};")
    a(f"                out_data <= {result};")
    for k in range(1, M + 1):
        a(f"                mem{k}[ecol] <= chain{k-1};")
    for r in rows_used:
        for d in range(N, 0, -1):
            src = f"row{r}" if d == 1 else f"row{r}_d{d-1}"
            a(f"                row{r}_d{d} <= {src};")
    a("                if (in_sof) begin col <= 1; row <= 0; end")  # anchor to frame start
    a("                else if (col == WIDTH-1) begin")    # row wraps -> continuous frames
    a("                    col <= 0; row <= (row == HEIGHT-1) ? 0 : row + 1;")
    a("                end else col <= col + 1;")
    a("            end else begin out_valid <= 1'b0; out_sof <= 1'b0;")  # bubble
    a("                out_eol <= 1'b0; out_last <= 1'b0; end")
    a("            end")                                  # close if (!stall)
    a("        end")
    a("    end")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if out_signed else ""))
    return Core({
        "verilog": verilog,
        "interface": _interface(in_bits, out_bits, out_signed, params),
        "weighted": weighted,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": post,
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": M,
        "N": N,
        "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name,
        "image": image,
    })


# --------------------------------------------------------------------------- #
# Mux emission — np.where(enable, A, B): two branches over a shared window,
# selected per pixel by a 1-bit register. Sharing the window auto-aligns the
# branch latencies (a tap is delay-matched within the window). Valid-interior.
# --------------------------------------------------------------------------- #

def _shift_taps(taps, dr, dc):
    return {(r - dr, c - dc): w for (r, c), w in taps.items()}


def _extend(name, frm, to, signed):
    if frm >= to:
        return name
    pad = to - frm
    if signed:
        return f"{{{{{pad}{{{name}[{frm-1}]}}}}, {name}}}"
    return f"{{{pad}'b0, {name}}}"


def _generate_mux(mux, module_name) -> dict:
    image = mux.image
    A, B = mux.a, mux.b
    in_bits = image.bits
    if image.signed:
        in_lo, in_hi = -(1 << (in_bits - 1)), (1 << (in_bits - 1)) - 1
    else:
        in_lo, in_hi = 0, (1 << in_bits) - 1

    # shared (union) window: shift both tap maps so the combined min offset is 0
    keys = list(A.taps) + list(B.taps)
    minr = min(r for r, _ in keys)
    minc = min(c for _, c in keys)
    tapsA = _shift_taps(A.taps, minr, minc)
    tapsB = _shift_taps(B.taps, minr, minc)
    M = max(r for r, _ in tapsA | tapsB)
    N = max(c for _, c in tapsA | tapsB)
    rows_used = sorted({r for r, _ in tapsA} | {r for r, _ in tapsB})

    # signed datapath if either branch's value range or any coeff is signed
    sgA = _range_bits(*_acc_range(tapsA, in_lo, in_hi))[1]
    sgB = _range_bits(*_acc_range(tapsB, in_lo, in_hi))[1]
    sel = mux.select
    params = [(sel.name, sel.bits, sel.signed, sel.default, sel.description)]
    for br in (A, B):
        params += [(op[1], op[2], op[3], op[4], op[5])
                   for op in br.post if op[0] in ("mulp", "addp")]
        params += _tap_params({k: v for k, v in br.taps.items()}, params)
    # dedup by name, keep order
    seen, uniq = set(), []
    for p in params:
        if p[0] not in seen:
            uniq.append(p); seen.add(p[0])
    params = uniq
    # Declared dtype of either branch first (both share one datapath), then the
    # value ranges, then any signed coefficient register.
    signed = (A.spatial_signed or B.spatial_signed or sgA or sgB
              or any(ps for _, _, ps, _, _ in params[1:]))
    sgn = "signed " if signed else ""
    out_rows, out_cols = mux.shape
    total_out = out_rows * out_cols

    L = []
    a = L.append
    a(f"// generated by np2hw -- mux (np.where) over a shared {M+1}x{N+1} window")
    a(f"module {module_name} #(parameter WIDTH = {image.width}, "
      f"parameter HEIGHT = {image.height}) (")
    a("    input  wire clk, input wire rst,")
    a("    input  wire in_valid, output wire in_ready, input wire in_sof,")
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, _doc in params:
        a(f"    input  wire {'signed ' if psg else ''}[{bits-1}:0] param_{name},")
    a("    output reg  out_valid, input wire out_ready,")
    a("    output reg  out_sof, output reg out_eol, output reg out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a(f"    reg [{POS-1}:0] col; reg [{POS-1}:0] row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")          # SOF anchors to (0,0)
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    # shared window
    a(f"    wire [{in_bits-1}:0] chain0 = in_data;")
    for k in range(1, M + 1):
        a(f"    reg  [{in_bits-1}:0] mem{k} [0:WIDTH-1];")
        a(f"    wire [{in_bits-1}:0] chain{k} = mem{k}[ecol];")
    for r in rows_used:
        a(f"    wire [{in_bits-1}:0] row{r} = chain{M - r};")
    for r in rows_used:
        for d in range(1, N + 1):
            a(f"    reg  [{in_bits-1}:0] row{r}_d{d};")

    def pixel(r, dc):
        delay = N - dc
        return f"row{r}" if delay == 0 else f"row{r}_d{delay}"

    def emit_branch(taps, sbits, post, prefix):
        terms = [_tap_term(pixel(r, c), w, signed, image.signed)
                 for (r, c), w in sorted(taps.items())]
        acc_bits = _range_bits_as(*_acc_range(taps, in_lo, in_hi), signed)
        a(f"    wire {sgn}[{acc_bits-1}:0] {prefix}acc = {' + '.join(terms)};")
        eff_post = ([("trunc", sbits)] if sbits < acc_bits else []) + list(post)
        return _emit_post(a, eff_post, acc_bits, signed, prefix)

    rA, bA, _ = emit_branch(tapsA, A.spatial_bits, A.post, "a_")
    rB, bB, _ = emit_branch(tapsB, B.spatial_bits, B.post, "b_")
    out_bits = max(bA, bB)
    eA = _extend(rA, bA, out_bits, signed)
    eB = _extend(rB, bB, out_bits, signed)
    a(f"    wire {sgn}[{out_bits-1}:0] muxed = param_{sel.name} ? {eA} : {eB};")

    gate = f"(erow >= {M}) && (ecol >= {N})"
    sof = f"({gate}) && (erow == {M}) && (ecol == {N})"
    eol = f"({gate}) && (ecol == WIDTH-1)"
    last = f"({gate}) && (erow == HEIGHT-1) && (ecol == WIDTH-1)"
    a("    always @(posedge clk) begin")
    a("        if (rst) begin col<=0; row<=0; out_valid<=1'b0;")
    a("            out_sof<=1'b0; out_eol<=1'b0; out_last<=1'b0;")
    a("        end else begin")
    a("            if (!stall) begin")
    a("                if (in_valid) begin")
    a(f"                    out_valid <= {gate};")
    a(f"                    out_sof <= {sof}; out_eol <= {eol}; out_last <= {last};")
    a("                    out_data <= muxed;")
    for k in range(1, M + 1):
        a(f"                    mem{k}[ecol] <= chain{k-1};")
    for r in rows_used:
        for d in range(N, 0, -1):
            src = f"row{r}" if d == 1 else f"row{r}_d{d-1}"
            a(f"                    row{r}_d{d} <= {src};")
    a("                    if (in_sof) begin col <= 1; row <= 0; end")
    a("                    else if (col == WIDTH-1) begin")
    a("                        col <= 0; row <= (row == HEIGHT-1) ? 0 : row + 1;")
    a("                    end else col <= col + 1;")
    a("                end else begin out_valid<=1'b0; out_sof<=1'b0;")
    a("                    out_eol<=1'b0; out_last<=1'b0; end")
    a("            end")
    a("        end")
    a("    end")
    a("endmodule")
    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if signed else ""))
    return {
        "verilog": verilog, "in_bits": in_bits, "out_bits": out_bits,
        "signed": signed, "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": M, "N": N, "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name, "image": image,
    }


# --------------------------------------------------------------------------- #
# Phase-stencil emission — planes that are STENCILS, selected by position.
#
# The demosaic shape: each CFA site applies a DIFFERENT tap combination of one
# shared window. One edge-handled window feeds every plane's combination at
# full rate, and a positional mux (position parity XOR the phase registers)
# picks the one this pixel's site names. np.stack channels ride the same
# window and concatenate into one output word -- C muxes, one select, one
# set of line buffers.
# --------------------------------------------------------------------------- #

def _post_range(lo, hi, post):
    """Exact value range after a chain of CONSTANT post ops.

    The stencil-phase emitter packs channels by VALUE range, not carrier
    signedness: an int32-typed centre tap is still [0, top] and packs fine.
    Register operands are refused upstream, so every op here is exact."""
    for op in post:
        if op[0] == "shr":
            lo, hi = lo >> op[1], hi >> op[1]
        elif op[0] == "mulc":
            corners = (lo * op[1], hi * op[1])
            lo, hi = min(corners), max(corners)
        elif op[0] == "clip":
            lo, hi = max(lo, op[1]), min(hi, op[2])
            if lo > hi:
                lo = hi = op[1]
        elif op[0] == "trunc" and op[1] < max(hi.bit_length(), 1):
            lo, hi = 0, (1 << op[1]) - 1     # a narrowing trunc re-ranges
    return lo, hi


def _generate_phase_stencil(stack, module_name, clk_ns=None,
                            label=None, dev=None) -> Core:
    channels = (stack.channels if type(stack).__name__ == "ChannelStack"
                else [stack])
    image = channels[0].image

    # Every channel-canvas must partition the image with the SAME phase
    # structure: one position select serves all of them. A plain Traced
    # channel is a BROADCAST: the same value at every site.
    per_channel, phase_key, rows0, cols0 = [], None, None, None
    for canvas in channels:
        if type(canvas).__name__ != "PhaseCanvas":
            per_channel.append([canvas])
            continue
        rows, cols = canvas.validate()
        rows, cols = _phase_order(rows), _phase_order(cols)
        key = tuple(canvas._key(d) for d in (*rows, *cols))
        if phase_key is None:
            phase_key, rows0, cols0 = key, rows, cols
        elif key != phase_key:
            raise ValueError(
                "np.stack channels use different phase descriptors; one "
                "position select must serve every channel of a word")
        per_channel.append([canvas.plane(r, c) for r in rows for c in cols])
    if rows0 is None:
        raise ValueError("no phase canvas among the channels; a pointwise "
                         "word is the expression-stack case")

    flat = [plane for planes in per_channel for plane in planes]
    chains = [p for p in flat if p.expr is None]
    exprs = [p for p in flat if p.expr is not None]
    if any(isinstance(w, Param) for p in chains for w in p.taps.values()):
        raise NotImplementedError(
            "programmable tap coefficients under a phase select are out of "
            "scope; use constant kernels per plane")
    for plane in chains:
        for op in plane.post:
            if op[0] in ("mulp", "addp"):
                raise NotImplementedError(
                    "register operands inside phase-sliced stencil planes are "
                    "out of scope in this version; constant shifts and clips "
                    "only (put per-plane registers in a pointwise block)")
    pads = {p.pad for p in flat}
    modes = {p.mode for p in flat}
    if len(pads) != 1 or len(modes) != 1:
        raise ValueError(
            "the phase planes disagree on padding; every plane must window "
            "one identically padded image")
    (pt, pb, pl, pr), mode = pads.pop(), modes.pop()
    keys = set().union(*(set(p.taps) for p in chains)) if chains else set()
    for plane in exprs:
        keys |= _wexpr_taps(plane.expr)
    if not keys:
        raise ValueError("empty window: no taps anywhere")
    if min(r for r, _ in keys) < 0 or min(c for _, c in keys) < 0:
        raise ValueError("stencil offsets must be >= 0 (slice from 0)")
    M = max(r for r, _ in keys)
    N = max(c for _, c in keys)
    if mode == "none" and (M or N):
        raise NotImplementedError(
            "a phase partition needs same-size planes; pad the image (edge "
            "or constant) so the stencil keeps the input's shape")
    rep = mode == "edge"
    v_edge, h_edge = bool(pt or pb), bool(pl or pr)
    if v_edge and pt + pb != M:
        raise ValueError(f"row padding {(pt, pb)} must sum to the vertical "
                         f"span {M} for same-size output")
    if h_edge and pl + pr != N:
        raise ValueError(f"column padding {(pl, pr)} must sum to the "
                         f"horizontal span {N} for same-size output")

    in_bits = image.bits
    if image.signed:
        in_lo, in_hi = -(1 << (in_bits - 1)), (1 << (in_bits - 1)) - 1
    else:
        in_lo, in_hi = 0, (1 << in_bits) - 1
    signed = image.signed or any(
        p.spatial_signed or _range_bits(*_acc_range(p.taps, in_lo, in_hi))[1]
        for p in chains) or any(p.expr.lo < 0 for p in exprs)
    sgn = "signed " if signed else ""
    rows_used = sorted({r for r, _ in keys})
    realH, realW = image.height, image.width

    params = []
    for desc in (rows0[0], cols0[0]):
        if isinstance(desc, PhaseRef) and desc.param.name not in {
                n for n, *_ in params}:
            params.append((desc.param.name, 1, False, desc.param.default,
                           desc.param.description))
    for kind, declared in _walk_expr_params([p.expr for p in exprs]):
        if kind != "param":
            raise NotImplementedError(
                "register-array gathers inside a windowed expression are "
                "out of scope in this version")
        if declared.name not in {n for n, *_ in params}:
            params.append((declared.name, declared.bits, declared.signed,
                           declared.default, declared.description))

    vrow_lo = pb if v_edge else M
    hcol_lo = pr if h_edge else N

    # Timing as a traced property, stencil flavour. The module has two
    # natural cones: the WINDOW (a WIDTH-deep line-buffer column read
    # plus the edge muxes in front of it) and the ARITHMETIC (the
    # deepest plane's adder tree, its post ops, the phase select, the
    # pack). When their sum misses the budget, a snapshot register
    # between them is the cut -- the window machinery, counters and
    # write-backs are untouched, so the split is retiming by
    # construction. One cut is this shape's whole repertoire: a half
    # that still misses refuses, named.
    stages = 1
    aplan = None                 # (stage_of, n) for the arithmetic DAG
    stage_label = label or module_name
    if clk_ns is not None:
        from .timing import (Depth, assign_stages, device, expr_depth,
                             post_op_cost)
        fam = device(dev)
        window = fam.window(bool(M))
        worst = Depth()
        chain_worst = Depth()
        for plane in chains:
            acc_bits = _range_bits_as(
                *_acc_range(plane.taps, in_lo, in_hi), signed)
            d = fam.adder_tree(len(plane.taps), acc_bits)
            for op in plane.post:
                if op[0] in ("clip", "mulc"):
                    d = d + post_op_cost(op, acc_bits, fam)
            chain_worst = fam.worst(chain_worst, d)
        worst = chain_worst
        for plane in exprs:
            worst = fam.worst(worst, expr_depth(plane.expr, fam))
        arith = worst + fam.select()                     # the phase select
        if fam.ns(window + arith) > clk_ns:
            stages = 2
            if fam.ns(window) > clk_ns:
                raise ValueError(
                    f"{stage_label}: the stencil's window read alone "
                    f"estimates {fam.ns(window):.1f} ns against the "
                    f"{clk_ns:.1f} ns budget on {fam.name}. That read is one "
                    "memory and the edge selects -- there is nothing inside "
                    "it to cut. Slow the clock.")
            if fam.ns(arith) > clk_ns:
                # Behind the snapshot the planes read REGISTERS, so the
                # arithmetic is an ordinary pointwise DAG and cuts like
                # one. A tap chain is emitted as a single summed wire and
                # has no interior to cut, so a chain deeper than the clock
                # is still the floor -- and says which it was.
                if fam.ns(chain_worst) > clk_ns:
                    raise ValueError(
                        f"{stage_label}: a tap chain alone estimates "
                        f"{fam.ns(chain_worst):.1f} ns against the "
                        f"{clk_ns:.1f} ns budget on {fam.name}. A weighted "
                        "sum is emitted as one expression; only window "
                        "EXPRESSIONS carry pipeline stages today.")
                # one level is reserved for the phase select, which rides
                # the last arithmetic stage
                stage_of, n = assign_stages(
                    [pl.expr for pl in exprs], clk_ns - fam.level_ns,
                    fam, label=stage_label)
                if n > 1:
                    aplan = (stage_of, n)

    L = []
    a = L.append
    a(f"// generated by np2hw -- phase-selected stencils over one shared "
      f"{M+1}x{N+1} window,")
    a(f"// {len(flat)} plane datapath(s), {len(channels)} channel(s), "
      f"{mode} edges")
    if stages == 2:
        a("// timing: window snapshot registered -- the line-buffer read")
        a("// cone and the arithmetic cone each fit the clock; their sum")
        a("// did not (traced depth model, stencil flavour)")
    if aplan:
        a(f"// timing: the arithmetic is cut into {aplan[1]} further stages,")
        a("// the plane expressions being deeper than one clock alone")
    a(f"module {module_name} #(parameter WIDTH = {realW}, "
      f"parameter HEIGHT = {realH}) (")
    a("    input  wire clk, input wire rst, input wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, _doc in params:
        L.extend(_param_port(name, bits, psg, _doc))
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")
    a("    output reg  out_eol,")
    a("    output reg  out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a(f"    reg [{POS-1}:0] col; reg [{POS-1}:0] row; "
      f"reg [{POS-1}:0] fcol; reg [{POS-1}:0] frow;")
    a("    reg hf; reg vf; reg done;")
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    wire in_active = !done && !hf && !vf;")
    a("    assign in_ready = !stall && (in_active || in_sof);")
    a("    wire en = !stall && !done && ((in_active && in_valid) || hf || vf"
      " || (in_sof && in_valid));")
    vbc = "in_sof || ((!vf) && (row == 0))" if (v_edge and rep) else "1'b0"
    hbc = "in_sof || ((!hf) && (col == 0))" if (h_edge and rep) else "1'b0"
    a(f"    wire vbc = {vbc};")
    a(f"    wire hbc = {hbc};")
    # Shared window: line buffers, vertical taps, per-row column delays.
    #
    # The line buffers are read THROUGH A REGISTER, which is the only way
    # a memory becomes block RAM -- block RAM has no asynchronous read
    # port. An asynchronous read forces distributed RAM plus a select
    # tree whose depth grows with the LINE, which is why a stencil that
    # closes at 1280 pixels misses at 1920 on the same clock.
    #
    # The register costs no schedule, because the address is a COUNTER:
    # present the column the block will want NEXT, and the data lands
    # exactly when the existing taps expect it. `chain{k}_q` at any cycle
    # holds precisely what the asynchronous read held -- same values,
    # same order, one cycle earlier in flight.
    if M:
        if h_edge:
            end_of_line = "col"          # the block enters hflush; col holds
        else:
            end_of_line = "0"
        a("    // the column this block will read NEXT, from the counter's")
        a("    // own next-state: the address a cycle early, data on time")
        a(f"    wire [{POS-1}:0] rd_col = in_sof ? {POS}\'d1")
        a(f"                       : (!hf ? ((col == WIDTH-1) ? {end_of_line}"
          f" : (col + 1))")
        a(f"                              : ((fcol == {max(pr-1, 0)}) ? 0 : col));")
    for k in range(1, M + 1):
        a(f"    reg  [{in_bits-1}:0] mem{k} [0:WIDTH-1];")
        a(f"    reg  [{in_bits-1}:0] chain{k}_q;")
    if v_edge and M >= 1:
        # the vertical flush replays row 1, which is the same address the
        # window is already reading -- one memory port, not two
        flush_src = "chain1_q" if rep else "0"
        a(f"    wire [{in_bits-1}:0] chain0 = (vf && !in_sof) ? {flush_src} : in_data;")
    else:
        a(f"    wire [{in_bits-1}:0] chain0 = in_data;")
    for r in rows_used:
        delay = M - r
        base = f"chain{delay}_q" if delay else "chain0"
        if rep or delay == 0 or not v_edge:
            a(f"    wire [{in_bits-1}:0] row{r} = {base};")
        else:
            a(f"    wire [{in_bits-1}:0] row{r} = (!vf && erow < {delay}) ? 0 : {base};")
    zero_h = h_edge and not rep
    for r in rows_used:
        for d in range(1, N + 1):
            a(f"    reg  [{in_bits-1}:0] row{r}_d{d};")
    if h_edge and rep:
        for r in rows_used:
            a(f"    reg  [{in_bits-1}:0] vlast{r};")
    for r in rows_used:
        if h_edge:
            src = f"vlast{r}" if rep else "0"
            a(f"    wire [{in_bits-1}:0] cur{r} = hf ? {src} : row{r};")
        else:
            a(f"    wire [{in_bits-1}:0] cur{r} = row{r};")

    def pixel(r, dc):
        delay = N - dc
        if delay == 0:
            return f"cur{r}"
        base = f"row{r}_d{delay}"
        if zero_h:
            return f"((!hf && ecol < {delay}) ? 0 : {base})"
        return base

    def px_read(r, dc):
        # what the arithmetic reads: the live window in one-stage form,
        # the snapshot registers behind the cut in two-stage form
        if stages == 2:
            return f"q_px_{r}_{N - dc}"
        return pixel(r, dc)

    def term(r, c, w):
        px = px_read(r, c)
        if signed:
            px = f"$signed({px})" if image.signed else f"$signed({{1'b0, {px}}})"
        return px if w == 1 else f"{w}*{px}"

    # The output pixel's position drives the plane select: during an edge
    # flush the input counters sit on replicated columns, but the pixel
    # being EMITTED is the one whose CFA site must choose the taps.
    ORv = "HEIGHT" if v_edge else f"(HEIGHT - {M})"
    OCv = "WIDTH" if h_edge else f"(WIDTH - {N})"
    orow = (f"(vf ? (({ORv} - {pb}) + frow) : (erow - {vrow_lo}))" if v_edge
            else f"(erow - {vrow_lo})")
    ocolp = (f"(hf ? (({OCv} - {pr}) + fcol) : (ecol - {hcol_lo}))" if h_edge
             else f"(ecol - {hcol_lo})")
    a(f"    wire [{POS-1}:0] o_row = {orow};")
    a(f"    wire [{POS-1}:0] o_col = {ocolp};")
    row_x, col_x = _phase_expr(rows0[0]), _phase_expr(cols0[0])
    a(f"    wire sel_row = o_row[0]{f' ^ {row_x}' if row_x else ''};")
    a(f"    wire sel_col = o_col[0]{f' ^ {col_x}' if col_x else ''};")
    a("")

    rowok = f"(vf || (erow >= {vrow_lo}))"
    colok = f"(hf || (ecol >= {hcol_lo}))"
    okv = f"({rowok} && {colok})"
    eol = f"{okv} && (o_col == ({OCv} - 1))"
    sof = f"{okv} && (o_row == 0) && (o_col == 0)"
    last = f"{okv} && (o_row == ({ORv} - 1)) && (o_col == ({OCv} - 1))"

    if stages == 2:
        qpairs = sorted({(r, N - c) for (r, c) in keys})
        a("    // window snapshot: the stage boundary. The planes below")
        a("    // read these registers; the window machinery, counters and")
        a("    // write-backs above are untouched -- pure retiming.")
        for r, d in qpairs:
            a(f"    reg [{in_bits-1}:0] q_px_{r}_{d};")
        a("    reg q_selr, q_selc;")
        a("    reg q_v, q_sof, q_eol, q_last;")
        a("    // The line-buffer WRITE is registered too: its data is a")
        a("    // memory read away, and read-mux-write in one beat is the")
        a("    // other half of the window cone. One beat late is safe by")
        a("    // schedule: a column's next read is a full line away, and")
        a("    // a frame boundary's pending write lands on a cell row 0")
        a("    // rewrites before its first read.")
        a("    reg [31:0] wr_col;")
        a("    reg wr_en;")
        for k in range(1, M + 1):
            a(f"    reg [{in_bits-1}:0] wr_d{k};")
        a("    always @(posedge clk) begin")
        a("        if (rst) begin")
        a("            q_v <= 1'b0; q_sof <= 1'b0; "
          "q_eol <= 1'b0; q_last <= 1'b0;")
        a("            wr_en <= 1'b0;")
        a("        end else if (!stall) begin")
        a(f"            q_v <= en && {okv};")
        a(f"            q_sof <= en && {sof};")
        a(f"            q_eol <= en && {eol};")
        a(f"            q_last <= en && {last};")
        a("            wr_en <= en && (!hf || in_sof);")
        a("            if (en) begin")
        for r, d in qpairs:
            a(f"                q_px_{r}_{d} <= {pixel(r, N - d)};")
        a("                q_selr <= sel_row; q_selc <= sel_col;")
        a("                wr_col <= ecol;")
        for k in range(1, M + 1):
            src = "chain0" if k == 1 else f"chain{k-1}_q"
            a(f"                wr_d{k} <= vbc ? chain0 : {src};")
        a("            end")
        a("        end")
        a("    end")
        a("")

    # One wire namespace for every windowed expression in the module, so a
    # gradient shared between planes -- or between CHANNELS -- is computed
    # once, whatever plane the id-identical node appears in.
    wemitted = {}
    wcounter = [0]

    # Arithmetic pipeline. Behind the window snapshot the planes read
    # registers, so a plane deeper than one clock is cut exactly as a
    # pointwise DAG is: a value consumed in a later stage than the one
    # that computed it rides a delay line to get there.
    astage = aplan[0] if aplan else {}
    afinal = (aplan[1] - 1) if aplan else 0
    amark = len(L)               # pipeline regs are SPLICED in here: a
    # continuous assignment may not read a reg declared further down
    apipe = []                   # (reg, width, source, signed), in order
    aseen = set()

    def adelay(name, width, frm, to, sg=True):
        prev = name
        for k in range(1, to - frm + 1):
            rn = f"{name}_s{k}"
            if rn not in aseen:
                aseen.add(rn)
                apipe.append((rn, width, prev, sg))
            prev = rn
        return prev

    def warg(node, want):
        """An argument as the stage that consumes it sees it."""
        nm = wemit(node)
        if node.op in ("const", "param"):
            return nm            # a literal is wiring; a register port is
        s = astage.get(id(node), 0)   # stable across the pixel's flight
        return adelay(nm, wwidth(node), s, want) if want > s else nm

    def wwidth(node):
        negative = (-node.lo - 1).bit_length() if node.lo < 0 else 0
        return max(node.hi.bit_length(), negative) + 1

    def wemit(node):
        key = id(node)
        if key in wemitted:
            return wemitted[key]
        w = wwidth(node)
        ns = astage.get(key, 0)          # the stage this node computes in
        if node.op == "const":
            wemitted[key] = str(node.args[0])
            return wemitted[key]
        if node.op == "tap":
            r, c, shift, field = node.args
            base = px_read(r, c)
            if field is not None:
                base = f"{base}[{shift + field - 1}:{shift}]"
            expr = (f"$signed({base})" if image.signed and field is None
                    else f"$signed({{1'b0, {base}}})")
        elif node.op == "param":
            leaf = node.args[0]
            expr = (f"$signed(param_{leaf.name})" if leaf.signed
                    else f"$signed({{1'b0, param_{leaf.name}}})")
        elif node.op in ("add", "sub", "mul"):
            op = {"add": "+", "sub": "-", "mul": "*"}[node.op]
            expr = f"{warg(node.args[0], ns)} {op} {warg(node.args[1], ns)}"
        elif node.op == "shr":
            expr = f"{warg(node.args[0], ns)} >>> {node.args[1]}"
        elif node.op == "mask":
            expr = (f"$signed({{1'b0, {warg(node.args[0], ns)}"
                    f"[{node.args[1]-1}:0]}})")
        elif node.op == "clip":
            source, lo, hi = node.args
            inner = warg(source, ns)
            expr = (f"({inner} > {hi}) ? {hi} : "
                    f"(({inner} < {lo}) ? {lo} : {inner})")
        elif node.op == "abs":
            inner = warg(node.args[0], ns)
            expr = f"({inner} < 0) ? -{inner} : {inner}"
        elif node.op == "lt":
            expr = (f"({warg(node.args[0], ns)} < {warg(node.args[1], ns)}) "
                    "? 2'sd1 : 2'sd0")
        elif node.op == "sel":
            cond = warg(node.args[0], ns)
            yes, no = warg(node.args[1], ns), warg(node.args[2], ns)
            expr = f"({cond} != 0) ? {yes} : {no}"
        elif node.op == "acc":
            raise NotImplementedError(
                "a pointwise value (acc leaf) inside a windowed expression: "
                "derive everything from the padded image so it has a window "
                "position")
        else:
            raise NotImplementedError(
                f"expression op {node.op!r} in a windowed plane")
        name = f"x{wcounter[0]}"
        wcounter[0] += 1
        a(f"    wire signed [{w-1}:0] {name} = {expr};")
        wemitted[key] = name
        return name

    # per-plane tap combinations, then one positional mux per channel
    chan_wires = []
    single = len(channels) == 1
    for ci, planes in enumerate(per_channel):
        results = []
        value_lo, value_hi = 0, 0
        for bi, plane in enumerate(planes):
            if plane.expr is not None or plane.lane is not None:
                # Lane-tagged chains lift too: field extraction happens per
                # tap, never on the word's weighted sum.
                root = plane.expr if plane.expr is not None \
                    else plane._as_wexpr()
                res = warg(root, afinal)
                results.append((res, wwidth(root), root.lo < 0))
                value_lo = min(value_lo, root.lo)
                value_hi = max(value_hi, root.hi)
                continue
            prefix = f"c{ci}p{bi}_"
            acc_lo, acc_hi = _acc_range(plane.taps, in_lo, in_hi)
            acc_bits = _range_bits_as(acc_lo, acc_hi, signed)
            terms = [term(r, c, w) for (r, c), w in sorted(plane.taps.items())]
            a(f"    wire {sgn}[{acc_bits-1}:0] {prefix}acc = {' + '.join(terms)};")
            _cres, _cbits, _csg = _emit_post(a, list(plane.post), acc_bits,
                                             signed, prefix)
            # a shallow chain beside a cut expression must still arrive
            # with it: the mux reads one stage, not several
            results.append((adelay(_cres, _cbits, 0, afinal, _csg)
                            if afinal else _cres, _cbits, _csg))
            plane_lo, plane_hi = _post_range(acc_lo, acc_hi, plane.post)
            value_lo = min(value_lo, plane_lo)
            value_hi = max(value_hi, plane_hi)
        cb = max(bits for _, bits, _ in results)
        # Packability is a fact about the VALUE range, not the carrier:
        # a signed accumulator whose every plane lands in [0, hi] packs.
        csg = any(sg for _, _, sg in results) and value_lo < 0
        if csg and not single:
            raise NotImplementedError(
                "np.stack cannot pack a channel whose value range reaches "
                f"{value_lo}; clip channels to their unsigned range first")
        ext = [_extend(res, bits, cb, sg) for res, bits, sg in results]
        kind = "signed " if csg else ""
        if len(results) == 1:
            # A broadcast channel: the same value at every site, no mux.
            a(f"    wire {kind}[{cb-1}:0] chan{ci} = {ext[0]};")
        else:
            if afinal:
                sel = f"{{q_selr_s{afinal}, q_selc_s{afinal}}}"
            elif stages == 2:
                sel = "{q_selr, q_selc}"
            else:
                sel = "{sel_row, sel_col}"
            a(f"    reg {kind}[{cb-1}:0] chan{ci};")
            a("    always @(*) begin")
            a(f"        case ({sel})")
            for code, expr in zip(("2'b00", "2'b01", "2'b10"), ext[:3]):
                a(f"            {code}:   chan{ci} = {expr};")
            a(f"            default: chan{ci} = {ext[3]};")
            a("        endcase")
            a("    end")
        chan_wires.append((f"chan{ci}", cb, csg, value_hi))

    if single:
        result, out_bits, out_signed = chan_wires[0][0], chan_wires[0][1], chan_wires[0][2]
    else:
        # channel 0 in the low bits, each field as wide as the input's
        # SAMPLES (the word width over its channel count) -- the packing is
        # a property of the word, stated here once. Field capacity is
        # judged by VALUE range: a value in [0, top] packs from a wider
        # signed carrier by taking its low field bits.
        field = in_bits // max(1, getattr(stack, "in_channels", 1) or 1)
        for name, cb, _, vhi in chan_wires:
            if vhi.bit_length() > field:
                raise NotImplementedError(
                    f"channel {name} can reach {vhi}, which does not fit a "
                    f"{field}-bit field (the input sample width); clip it "
                    "first")
        out_bits = field * len(chan_wires)
        out_signed = False
        parts = [(f"{name}[{field-1}:0]" if cb >= field
                  else f"{{{field - cb}'b0, {name}}}")
                 for name, cb, _, _ in reversed(chan_wires)]
        a(f"    wire [{out_bits-1}:0] word = {{{', '.join(parts)}}};")
        result = "word"

    if afinal:
        decl = ["",
                f"    // arithmetic pipeline: {afinal + 1} stages behind the",
                "    // window snapshot. The planes were deeper than one",
                "    // clock, so the DAG is cut exactly as a pointwise",
                "    // datapath is -- a value consumed later than it is",
                "    // computed rides a delay line, and the flags and the",
                "    // phase select ride beside it, so the pixel that",
                "    // reaches the output is the one being described."]
        for rn, w, _src, sg in apipe:
            kind = "signed " if sg else ""
            decl.append(f"    reg {kind}[{w-1}:0] {rn};")
        for k in range(1, afinal + 1):
            decl.append(f"    reg q_v_s{k}, q_sof_s{k}, "
                        f"q_eol_s{k}, q_last_s{k};")
            decl.append(f"    reg q_selr_s{k}, q_selc_s{k};")
        L[amark:amark] = decl
        a("    always @(posedge clk) begin")
        a("        if (rst) begin")
        for k in range(1, afinal + 1):
            a(f"            q_v_s{k} <= 1'b0; q_sof_s{k} <= 1'b0;")
            a(f"            q_eol_s{k} <= 1'b0; q_last_s{k} <= 1'b0;")
        a("        end else if (!stall) begin")
        for rn, _w, src, _sg in apipe:
            a(f"            {rn} <= {src};")
        for k in range(1, afinal + 1):
            pv = "q_v" if k == 1 else f"q_v_s{k-1}"
            ps = "q_sof" if k == 1 else f"q_sof_s{k-1}"
            pe = "q_eol" if k == 1 else f"q_eol_s{k-1}"
            pl = "q_last" if k == 1 else f"q_last_s{k-1}"
            pr_ = "q_selr" if k == 1 else f"q_selr_s{k-1}"
            pc_ = "q_selc" if k == 1 else f"q_selc_s{k-1}"
            a(f"            q_v_s{k} <= {pv}; q_sof_s{k} <= {ps};")
            a(f"            q_eol_s{k} <= {pe}; q_last_s{k} <= {pl};")
            a(f"            q_selr_s{k} <= {pr_}; q_selc_s{k} <= {pc_};")
        a("        end")
        a("    end")
        a("")

    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            col<=0; row<=0; fcol<=0; frow<=0;")
    a("            hf<=1'b0; vf<=1'b0; done<=1'b0; out_valid<=1'b0;")
    a("            out_sof<=1'b0; out_eol<=1'b0; out_last<=1'b0;")
    a("        end else begin")
    a("            if (!stall) begin")
    if stages == 2:
        # the output registers read the snapshot's flags every advancing
        # beat: bubbles carry q_v = 0, so no en-gate here -- and the
        # registered line-buffer write fires on its own captured enable
        sfx = f"_s{afinal}" if afinal else ""
        a(f"            out_valid <= q_v{sfx}; out_sof <= q_sof{sfx};")
        a(f"            out_eol <= q_eol{sfx}; out_last <= q_last{sfx};")
        a(f"            out_data <= {result};")
        a("            if (wr_en) begin")
        for k in range(1, M + 1):
            a(f"                mem{k}[wr_col] <= wr_d{k};")
        a("            end")
    a("            if (en) begin")
    if stages == 1:
        a(f"                out_valid <= {okv};")
        a(f"                out_sof <= {sof}; out_eol <= {eol}; "
          f"out_last <= {last};")
        a(f"                out_data <= {result};")
    for k in range(1, M + 1):
        a(f"                chain{k}_q <= mem{k}[rd_col];")
    if stages == 1:
        for k in range(1, M + 1):
            src = "chain0" if k == 1 else f"chain{k-1}_q"
            a(f"                if (!hf || in_sof) mem{k}[ecol] <= "
              f"vbc ? chain0 : {src};")
    for r in rows_used:
        for d in range(N, 0, -1):
            src = f"cur{r}" if d == 1 else f"row{r}_d{d-1}"
            a(f"                row{r}_d{d} <= hbc ? cur{r} : {src};")
    if h_edge and rep:
        for r in rows_used:
            a(f"                if (!hf && col == WIDTH-1) vlast{r} <= row{r};")
    a("                if (in_sof) begin")
    a("                    col <= 1; row <= 0; hf <= 1'b0; vf <= 1'b0; fcol <= 0; frow <= 0;")
    a("                end else if (!hf) begin")
    a("                    if (col == WIDTH-1) begin")
    if h_edge:
        a("                        hf <= 1'b1; fcol <= 0;")
    else:
        a("                        col <= 0;")
        _emit_row_advance(a, pb, v_edge, 24)
    a("                    end else col <= col + 1;")
    a("                end else begin")
    a(f"                    if (fcol == {max(pr-1, 0)}) begin")
    a("                        hf <= 1'b0; col <= 0;")
    _emit_row_advance(a, pb, v_edge, 24)
    a("                    end else fcol <= fcol + 1;")
    a("                end")
    if stages == 1:
        a("            end else begin out_valid <= 1'b0; out_sof <= 1'b0;")
        a("                out_eol <= 1'b0; out_last <= 1'b0; end")
    else:
        a("            end")
    a("            end")
    a("        end")
    a("    end")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if out_signed else ""))
    interface = _interface(in_bits, out_bits, out_signed, params)
    if not single:
        interface["output"]["channels"] = len(channels)
        interface["output"]["field_bits"] = in_bits // max(
            1, getattr(stack, "in_channels", 1) or 1)
    return Core({
        "verilog": verilog,
        "interface": interface,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": M, "N": N,
        "phases": len(flat) // max(1, len(channels)),
        "channels": len(channels),
        "out_rows": realH if v_edge else realH - M,
        "out_cols": realW if h_edge else realW - N,
        "module": module_name,
        "image": image,
        "edge": mode != "none",
        "eof": False,
        "dynamic": False,
        "max_width": None,
        "aw_bits": 0,
        "hblank": pr + 2,
        "vdrain": (pb + 1) * (realW + pr) + 8,
        # the snapshot boundary plus however many the arithmetic needed:
        # the module's real depth, not the shape of the cut
        "pipeline_stages": stages + afinal,
    })


# --------------------------------------------------------------------------- #
# Edge emission — full-size output via row/column replicate (or zero)
# --------------------------------------------------------------------------- #

def _generate_edge(out_line, image, module_name, framing="height",
                   max_width=None) -> dict:
    """Same-size output by extending borders (idea.txt: copy first/last line).

    Vertical: top = broadcast first row into all line buffers (replicate); bottom
    = a flush phase recirculating the last row for `pb` extra row-times. These run
    during VBLANK. Horizontal: left = broadcast col 0 into the column shift
    registers each row; right = latch the last column's vertical vector (vlast)
    and replicate it for `pr` cycles during HBLANK. Edge flushes consume blanking
    idle cycles, so no backpressure is needed (min HBLANK>=pr, VBLANK>=pb rows).
    Zero mode is supported on the vertical axis only."""
    spatial_line, post = peel_pointwise(out_line)
    weighted = flatten_weighted(spatial_line)
    if any(isinstance(c, Param) for c in weighted.values()):
        raise NotImplementedError("programmable kernel (Param tap coeffs) + edge "
                                  "handling not yet supported; use valid-interior")
    pt, pb, pl, pr, mode, realH, realW = out_line.edge
    rep = mode == "edge"
    v_edge = bool(pt or pb)
    h_edge = bool(pl or pr)
    eof = framing == "eof"                  # height-agnostic: in_eof drives flush
    if eof and not (v_edge and pb > 0):
        raise NotImplementedError("framing='eof' needs a bottom edge (vertical "
                                  "flush to drain the line buffers)")
    dyn = max_width is not None             # runtime width: MAX_WIDTH buffers, active_width rows
    wparam = "MAX_WIDTH" if dyn else "WIDTH"   # line-buffer depth (fixed at synth)
    aw = "active_width" if dyn else "WIDTH"    # where the row ends (runtime if dyn)

    in_bits = image.bits
    if image.signed:
        in_lo, in_hi = -(1 << (in_bits - 1)), (1 << (in_bits - 1)) - 1
    else:
        in_lo, in_hi = 0, (1 << in_bits) - 1
    M = max(r for r, _ in weighted)
    N = max(c for _, c in weighted)
    acc_lo, acc_hi = _acc_range(weighted, in_lo, in_hi)
    range_signed = _range_bits(acc_lo, acc_hi)[1]
    rows_used = sorted({r for r, _ in weighted})
    params = [(op[1], op[2], op[3], op[4], op[5])
              for op in post if op[0] in ("mulp", "addp")]
    signed = _datapath_signed(spatial_line, range_signed, params)
    acc_bits = _range_bits_as(acc_lo, acc_hi, signed)
    sgn = "signed " if signed else ""

    vrow_lo = pb if v_edge else M          # first output row's input-row latency
    hcol_lo = pr if h_edge else N          # first output col's latency

    L = []
    a = L.append
    a(f"// generated by np2hw -- {mode} edges v={v_edge} h={h_edge}, "
      f"{M} line buffer(s), out {realH if v_edge else realH-M}x{realW if h_edge else realW-N}")
    a(f"module {module_name} #(parameter {wparam} = "
      f"{max_width if dyn else realW}, parameter HEIGHT = {realH}) (")
    a("    input  wire clk, input wire rst, input wire in_valid,")
    a("    output wire in_ready,")
    a("    input  wire in_sof,")                          # frame start (AXI TUSER); tie 0 to free-run
    if eof:
        a("    input  wire in_eof,")                      # frame end (VSYNC): last input pixel -> drives flush
    if dyn:
        a(f"    input  wire [31:0] active_width,")        # runtime line length (<= MAX_WIDTH)
    a(f"    input  wire [{in_bits-1}:0] in_data,")
    for name, bits, psg, _, _doc in params:
        a(f"    input  wire {'signed ' if psg else ''}[{bits-1}:0] param_{name},")
    a("    output reg  out_valid,")
    a("    input  wire out_ready,")
    a("    output reg  out_sof,")                         # start of frame (first pixel)
    a("    output reg  out_eol,")                         # end of line (last pixel of row)
    a("    output reg  out_last,")                        # end of frame (final pixel)
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    out_rows = realH if v_edge else realH - M
    out_cols = realW if h_edge else realW - N
    a(f"    reg [{POS-1}:0] col; reg [{POS-1}:0] row; "
      f"reg [{POS-1}:0] fcol; reg [{POS-1}:0] frow;")
    a("    reg hf; reg vf; reg done;")
    if eof:
        a("    reg eof_l;")                               # latched: this frame's end was seen
    a(f"    wire [{POS-1}:0] ecol = in_sof ? 0 : col;")          # SOF anchors to frame (0,0)
    a(f"    wire [{POS-1}:0] erow = in_sof ? 0 : row;")
    a("    wire stall = out_valid && !out_ready;")       # holding an unaccepted output
    a("    wire in_active = !done && !hf && !vf;")
    a("    assign in_ready = !stall && (in_active || in_sof);")  # SOF accepted even mid-flush
    # advance: consume a pixel, or run a flush cycle -- never while output is held
    a("    wire en = !stall && !done && ((in_active && in_valid) || hf || vf || (in_sof && in_valid));")
    vbc = "in_sof || ((!vf) && (row == 0))" if (v_edge and rep) else "1'b0"
    hbc = "in_sof || ((!hf) && (col == 0))" if (h_edge and rep) else "1'b0"
    a(f"    wire vbc = {vbc};")               # top broadcast (replicate / SOF)
    a(f"    wire hbc = {hbc};")               # left broadcast (replicate / SOF)
    # line buffers, read THROUGH A REGISTER so they land in block RAM
    # rather than distributed RAM behind a select tree that deepens
    # with the line. The address is a counter, so presenting the NEXT
    # column costs nothing and the data still arrives on the beat it
    # was due: rd_col mirrors the control below, one cycle early.
    if M:
        eol = "col" if h_edge else "0"   # h_edge holds col and enters hflush
        a(f"    wire [{POS-1}:0] rd_col = in_sof ? {POS}\'d1")
        a(f"                       : (!hf ? ((col == {aw}-1) ? {eol} : (col + 1))")
        a(f"                              : ((fcol == {max(pr-1,0)}) ? 0 : col));")
    for k in range(1, M + 1):
        a(f"    reg  [{in_bits-1}:0] mem{k} [0:{wparam}-1];")
        a(f"    reg  [{in_bits-1}:0] chain{k}_q;")       # mem{k} at the current column
    if v_edge and M >= 1:                                # bottom flush recirculates
        flush_src = "chain1_q" if rep else "0"           # the same registered read
        a(f"    wire [{in_bits-1}:0] chain0 = (vf && !in_sof) ? {flush_src} : in_data;")
    else:
        a(f"    wire [{in_bits-1}:0] chain0 = in_data;")
    # vertical taps (top: replicate via broadcast, or zero via mux)
    for r in rows_used:
        delay = M - r
        base = f"chain{delay}_q" if delay else "chain0"
        if rep or delay == 0 or not v_edge:
            a(f"    wire [{in_bits-1}:0] row{r} = {base};")
        else:                                            # zero-mode top
            a(f"    wire [{in_bits-1}:0] row{r} = (!vf && erow < {delay}) ? 0 : {base};")
    # horizontal shift registers + right-edge latch (vlast, replicate only)
    zero_h = h_edge and not rep
    for r in rows_used:
        for d in range(1, N + 1):
            a(f"    reg  [{in_bits-1}:0] row{r}_d{d};")
    if h_edge and rep:
        for r in rows_used:
            a(f"    reg  [{in_bits-1}:0] vlast{r};")
    # effective newest column tap: hflush replicates last column (vlast) or 0 (zero)
    for r in rows_used:
        if h_edge:
            src = f"vlast{r}" if rep else "0"
            a(f"    wire [{in_bits-1}:0] cur{r} = hf ? {src} : row{r};")
        else:
            a(f"    wire [{in_bits-1}:0] cur{r} = row{r};")

    def pixel(r, dc):
        delay = N - dc
        if delay == 0:
            return f"cur{r}"
        base = f"row{r}_d{delay}"
        if zero_h:                                       # left zero: cols < 0 -> 0
            return f"((!hf && ecol < {delay}) ? 0 : {base})"
        return base

    def term(r, c, w):
        px = pixel(r, c)
        if signed:
            px = f"$signed({px})" if image.signed else f"$signed({{1'b0, {px}}})"
        return px if w == 1 else f"{w}*{px}"
    terms = [term(r, c, w) for (r, c), w in sorted(weighted.items())]
    a(f"    wire {sgn}[{acc_bits-1}:0] acc = {' + '.join(terms)};")
    result, out_bits, out_signed = _emit_post(a, post, acc_bits, signed)

    rowok = f"(vf || (erow >= {vrow_lo}))"
    colok = f"(hf || (ecol >= {hcol_lo}))"
    okv = f"({rowok} && {colok})"
    # output dims as WIDTH/HEIGHT expressions (resolution-agnostic); pb/pr/M/N
    # are kernel constants. out_rows = HEIGHT[-M], out_cols = WIDTH[-N].
    ORv = "HEIGHT" if v_edge else f"(HEIGHT - {M})"
    OCv = aw if h_edge else f"({aw} - {N})"
    # output frame position, derived from input position + flush phase (so it
    # tracks SOF re-anchoring without a separate output counter)
    orow = (f"(vf ? (({ORv} - {pb}) + frow) : (erow - {vrow_lo}))" if v_edge
            else f"(erow - {vrow_lo})")
    ocolp = (f"(hf ? (({OCv} - {pr}) + fcol) : (ecol - {hcol_lo}))" if h_edge
             else f"(ecol - {hcol_lo})")
    eol = f"{okv} && ({ocolp} == ({OCv} - 1))"
    if eof:
        # height-free framing: SOF = first streaming output; EOF = last flush row's
        # last pixel (the flush has drained the line buffers). No HEIGHT anywhere.
        sof = f"{okv} && !vf && !hf && (erow == {vrow_lo}) && (ecol == {hcol_lo})"
        last = f"{okv} && vf && (frow == {pb - 1}) && ({ocolp} == ({OCv} - 1))"
    else:
        sof = f"{okv} && ({orow} == 0) && ({ocolp} == 0)"
        last = f"{okv} && ({orow} == ({ORv} - 1)) && ({ocolp} == ({OCv} - 1))"
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            col<=0; row<=0; fcol<=0; frow<=0;")
    a("            hf<=1'b0; vf<=1'b0; done<=1'b0; out_valid<=1'b0;")
    a("            out_sof<=1'b0; out_eol<=1'b0; out_last<=1'b0;")
    if eof:
        a("            eof_l<=1'b0;")
    a("        end else begin")
    a("            if (!stall) begin")                   # freeze while output is held
    a("            if (en) begin")
    a(f"                out_valid <= {okv};")
    a(f"                out_sof <= {sof}; out_eol <= {eol}; out_last <= {last};")
    a(f"                out_data <= {result};")
    if eof:                                              # latch the frame-end mark
        a("                if (in_active && in_valid && in_eof) eof_l <= 1'b1;")
    # line-buffer write at real columns (or a SOF pixel), never during hflush
    for k in range(1, M + 1):
        src = "chain0" if k == 1 else f"chain{k-1}_q"
        a(f"                if (!hf || in_sof) mem{k}[ecol] <= vbc ? chain0 : {src};")
        a(f"                chain{k}_q <= mem{k}[rd_col];")
    # horizontal shift: broadcast col0, else shift cur into the register chain
    for r in rows_used:
        for d in range(N, 0, -1):
            src = f"cur{r}" if d == 1 else f"row{r}_d{d-1}"
            a(f"                row{r}_d{d} <= hbc ? cur{r} : {src};")
    if h_edge and rep:                                   # latch last real column
        for r in rows_used:
            a(f"                if (!hf && col == {aw}-1) vlast{r} <= row{r};")
    # control: SOF re-anchors to frame start; else columns / hflush / row advance
    end_cond = "eof_l" if eof else "row == HEIGHT-1"
    a("                if (in_sof) begin")
    a("                    col <= 1; row <= 0; hf <= 1'b0; vf <= 1'b0; fcol <= 0; frow <= 0;"
      + (" eof_l <= 1'b0;" if eof else ""))
    a("                end else if (!hf) begin")
    a(f"                    if (col == {aw}-1) begin")
    if h_edge:
        a("                        hf <= 1'b1; fcol <= 0;")
    else:
        a("                        col <= 0;")
        _emit_row_advance(a, pb, v_edge, 24, end_cond, eof)
    a("                    end else col <= col + 1;")
    a("                end else begin")               # in hflush
    a(f"                    if (fcol == {max(pr-1,0)}) begin")
    a("                        hf <= 1'b0; col <= 0;")
    _emit_row_advance(a, pb, v_edge, 24, end_cond, eof)
    a("                    end else fcol <= fcol + 1;")
    a("                end")
    a("            end else begin out_valid <= 1'b0; out_sof <= 1'b0;")  # idle -> bubble
    a("                out_eol <= 1'b0; out_last <= 1'b0; end")
    a("            end")                                  # close if (!stall)
    a("        end")
    a("    end")
    a("endmodule")

    verilog = ("\n".join(L)
               .replace("OUT_BITS-1", f"{out_bits-1}")
               .replace("OUT_SGN", "signed " if out_signed else ""))
    return Core({
        "verilog": verilog,
        "interface": _interface(in_bits, out_bits, out_signed, params),
        "weighted": weighted,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": post,
        "params": [(n, b) for n, b, _, _, _ in params],
        "param_defaults": {n: d for n, _, _, d, _ in params},
        "M": M, "N": N,
        "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name,
        "image": image,
        "edge": True,
        "eof": eof,
        "dynamic": dyn,
        "max_width": max_width,
        "aw_bits": max(1, int(max_width).bit_length()) if dyn else 0,
        "hblank": pr + 2,
        "vdrain": (pb + 1) * (realW + pr) + 8,
    })


def _emit_row_advance(a, pb, v_edge, indent, end_cond="row == HEIGHT-1",
                      clear_eof=False):
    """Emit the row / vertical-flush advance logic (shared by the two col-end
    branches). At a frame boundary the core RE-ARMS (row<-0) rather than latching
    done, so it free-runs across frames (continuous video); the next frame's
    top-broadcast re-inits the line buffers and the handshake supplies VBLANK.
    `end_cond` is the end-of-frame test: 'row == HEIGHT-1' (count) or 'eof_l'
    (the latched input frame-end, height-agnostic). `a` appends lines."""
    sp = " " * indent
    clr = " eof_l <= 1'b0;" if clear_eof else ""
    a(f"{sp}if (!vf) begin")
    if v_edge and pb > 0:
        a(f"{sp}    if ({end_cond}) begin vf <= 1'b1; frow <= 0; end")
        a(f"{sp}    else row <= row + 1;")
    else:
        a(f"{sp}    row <= ({end_cond}) ? 0 : row + 1;")   # re-arm next frame
    a(f"{sp}end else begin")
    if pb > 0:
        a(f"{sp}    if (frow == {pb-1}) begin vf <= 1'b0; row <= 0;{clr} end")  # re-arm
        a(f"{sp}    else frow <= frow + 1;")
    else:
        a(f"{sp}    row <= 0;")
    a(f"{sp}end")


# --------------------------------------------------------------------------- #
# Testbench emission
# --------------------------------------------------------------------------- #

def testbench(meta, W, H, param_values=None, tb_name="tb") -> str:
    param_values = param_values or {}
    in_bits, out_bits = meta["in_bits"], meta["out_bits"]
    mod = meta["module"]
    L = []
    a = L.append
    a("`timescale 1ns/1ps")
    a(f"module {tb_name};")
    a("    reg clk = 0, rst = 1, in_valid = 0;")
    a(f"    reg  [{in_bits-1}:0] in_data;")
    for name, bits in meta["params"]:
        a(f"    reg  [{bits-1}:0] param_{name};")
    a("    wire out_valid; wire in_ready; wire out_last; reg out_ready = 1'b1;")
    a(f"    wire {'signed ' if meta.get('signed') else ''}[{out_bits-1}:0] out_data;")
    a("    integer i, f;")
    a(f"    reg [{in_bits-1}:0] img [0:{W*H-1}];")
    a("    always #5 clk = ~clk;")
    conns = ["        .clk(clk), .rst(rst),",
             "        .in_valid(in_valid), .in_ready(in_ready), .in_sof(1'b0), .in_data(in_data),"]
    for name, _ in meta["params"]:
        conns.append(f"        .param_{name}(param_{name}),")
    conns.append("        .out_valid(out_valid), .out_ready(out_ready), "
                 ".out_last(out_last), .out_data(out_data));")
    a(f"    {mod} #(.WIDTH({W}), .HEIGHT({H})) dut (")
    L.extend(conns)
    a("    always @(posedge clk) if (out_valid) $fdisplay(f, \"%0d\", out_data);")
    a("    initial begin")
    a("        $readmemh(\"in.hex\", img);")
    a("        f = $fopen(\"out.txt\", \"w\");")
    for name, _ in meta["params"]:
        a(f"        param_{name} = {int(param_values.get(name, 0))};")
    a("        @(negedge clk); rst = 0;")
    if meta.get("edge"):
        # stream row by row with HBLANK between rows; VBLANK drain at the end so
        # the module's horizontal/vertical edge flushes run in the idle cycles.
        a("        begin : feed")
        a("            integer r, c;")
        a(f"            for (r = 0; r < {H}; r = r + 1) begin")
        a(f"                for (c = 0; c < {W}; c = c + 1) begin")
        a(f"                    in_data = img[r*{W}+c]; in_valid = 1; @(negedge clk);")
        a("                end")
        a(f"                in_valid = 0; for (c = 0; c < {meta['hblank']}; c = c + 1) @(negedge clk);")
        a("            end")
        a(f"            for (c = 0; c < {meta['vdrain']}; c = c + 1) @(negedge clk);")
        a("        end")
    else:
        a(f"        for (i = 0; i < {W*H}; i = i + 1) begin")
        a("            in_data = img[i]; in_valid = 1; @(negedge clk);")
        a("        end")
        drain = meta.get("flush_cycles", 2)
        a(f"        in_valid = 0; for (i = 0; i < {drain}; i = i + 1) @(negedge clk);")
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
module {top} #(parameter WIDTH = {W}, parameter HEIGHT = {H}) (
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
    {core} #(.WIDTH(WIDTH), .HEIGHT(HEIGHT)) u_core (
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
    a(f"module {top} #(parameter WIDTH = {W}, parameter HEIGHT = {H}) (")
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
    a(f"    {core} #(.WIDTH(WIDTH), .HEIGHT(HEIGHT)) u_core (")
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
    a(f"module {top} #(parameter WIDTH = {W}, parameter HEIGHT = {H}) (")
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


def axil_regfile(params, module_name="np2hw_axil", addr_bits=8,
                 frame_sync=False, defaults=None) -> dict:
    """AXI4-Lite slave exposing one register per entry of `params`.

    `params` is a list of `(name, bits)` -- packed at word offset i*4 -- or of
    :class:`Reg` values, which carry their own offset, signedness, reset, access
    and description. Output `param_<name>` wires drive the core.

    frame_sync=True adds shadow registers + an `update` input: writes land in a
    shadow, and the live value (driving the core) copies from the shadow on an
    `update` pulse -- wire `update` to the core's out_last/SOF for glitch-free,
    frame-aligned coefficient updates (no mid-frame tearing)."""
    A = addr_bits
    regs = _normalise_regs(params, defaults, addr_bits=A)
    rw = [r for r in regs if r.writable]
    wreg = "shadow_" if frame_sync else "reg_"            # software-written reg
    dreg = "live_" if frame_sync else "reg_"              # drives param_*
    L = []
    a = L.append
    a("// generated by np2hw -- AXI4-Lite control register file"
      + (" (frame-synced shadow)" if frame_sync else ""))
    a(f"module {module_name} (")
    a("    input  wire aclk, input wire aresetn,")
    if frame_sync:
        a("    input  wire update,   // pulse (e.g. frame boundary): shadow -> live")
    a(f"    input  wire [{A-1}:0] s_axil_awaddr, input wire s_axil_awvalid,"
      " output reg s_axil_awready,")
    a("    input  wire [31:0] s_axil_wdata, input wire [3:0] s_axil_wstrb,"
      " input wire s_axil_wvalid, output reg s_axil_wready,")
    a("    output reg [1:0] s_axil_bresp, output reg s_axil_bvalid,"
      " input wire s_axil_bready,")
    a(f"    input  wire [{A-1}:0] s_axil_araddr, input wire s_axil_arvalid,"
      " output reg s_axil_arready,")
    a("    output reg [31:0] s_axil_rdata, output reg [1:0] s_axil_rresp,"
      " output reg s_axil_rvalid, input wire s_axil_rready")
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
    a(f"    reg [{A-1}:0] awaddr_q; reg [{A-1}:0] araddr_q; reg aw_en;")
    # write channel
    a("    always @(posedge aclk) begin")
    a("        if (!aresetn) begin")
    a("            s_axil_awready<=0; s_axil_wready<=0; s_axil_bvalid<=0;")
    a("            s_axil_bresp<=0; aw_en<=1;")
    for reg in rw:
        a(f"            {wreg}{reg.name}<={int(reg.reset)};")
    a("        end else begin")
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
        a(f"                    {reg.word}: begin {wreg}{reg.name} <= "
          f"s_axil_wdata[{reg.bits-1}:0]; s_axil_bresp<=2'b00; end")
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
        value = (f"32'h{reg.value & 0xFFFFFFFF:08x}" if not reg.writable
                 else _read_expr(reg, f"{wreg}{reg.name}"))
        a(f"                    {reg.word}: begin s_axil_rdata <= {value}; "
          "s_axil_rresp<=2'b00; end")
    a("                    default: s_axil_rdata <= 0;")
    a("                endcase")
    a("            end else if (s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;")
    a("        end")
    a("    end")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": module_name,
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
    """Top module = control register file + the core, with geometry register(s)
    folded in. For a dynamic-resolution core (generate(..., max_width=N)), the
    register file gains an `active_width` register and this top wires it to the
    core's active_width input -- so software sets the line length over AXI-Lite /
    UMI like any other config register (frame-synced when frame_sync=True). The
    user Params get registers too. Returns {verilog (regfile + top), module, reg}."""
    cm = core["module"]
    top = module_name or cm + "_ctrl"
    ib, ob = core["in_bits"], core["out_bits"]
    dyn = core.get("dynamic")
    awb = core.get("aw_bits", 0)
    eof = core.get("eof")
    # register list = user params (+ active_width geometry register if dynamic)
    regs = list(core["params"])
    defaults = dict(core.get("param_defaults", {}))
    if dyn:
        regs = regs + [("active_width", awb)]
        defaults["active_width"] = core["max_width"]      # boot at full width
    mk = axil_regfile if ctrl == "axil" else umi_regfile
    reg = mk(regs, module_name=f"{top}_rf", addr_bits=addr_bits,
             frame_sync=frame_sync, defaults=defaults)
    A = addr_bits

    L = []
    a = L.append
    a(reg["verilog"]); a("")
    a(f"// generated by np2hw -- control top: {ctrl} regfile + {cm} "
      + ("(active_width register wired)" if dyn else ""))
    wpar = "MAX_WIDTH" if dyn else "WIDTH"
    a(f"module {top} #(parameter {wpar} = {core.get('max_width') or core['image'].width}, "
      f"parameter HEIGHT = {core['image'].height}) (")
    a("    input  wire clk, input wire rst,")
    if ctrl == "axil":
        a(f"    input  wire [{A-1}:0] s_axil_awaddr, input wire s_axil_awvalid, output wire s_axil_awready,")
        a("    input  wire [31:0] s_axil_wdata, input wire [3:0] s_axil_wstrb, input wire s_axil_wvalid, output wire s_axil_wready,")
        a("    output wire [1:0] s_axil_bresp, output wire s_axil_bvalid, input wire s_axil_bready,")
        a(f"    input  wire [{A-1}:0] s_axil_araddr, input wire s_axil_arvalid, output wire s_axil_arready,")
        a("    output wire [31:0] s_axil_rdata, output wire [1:0] s_axil_rresp, output wire s_axil_rvalid, input wire s_axil_rready,")
    else:
        a("    input  wire udev_req_valid, output wire udev_req_ready, input wire [4:0] udev_req_cmd,")
        a(f"    input  wire [{A-1}:0] udev_req_addr, input wire [31:0] udev_req_data,")
        a("    output wire udev_resp_valid, input wire udev_resp_ready, output wire [31:0] udev_resp_data,")
    a("    input  wire in_valid, output wire in_ready, input wire in_sof,")
    if eof:
        a("    input  wire in_eof,")
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
    a(f"    {cm} #(.{wpar}({wpar}), .HEIGHT(HEIGHT)) u_core (")
    a("        .clk(clk), .rst(rst),")
    a("        .in_valid(in_valid), .in_ready(in_ready), .in_sof(in_sof),")
    if eof:
        a("        .in_eof(in_eof),")
    if dyn:                                              # geometry register -> core
        a(f"        .active_width({{{{{32 - awb}{{1'b0}}}}, param_active_width}}),")
    a("        .in_data(in_data),")
    for name, _ in core["params"]:                       # user params only
        a(f"        .param_{name}(param_{name}),")
    a("        .out_valid(out_valid), .out_ready(out_ready),")
    a("        .out_sof(out_sof), .out_eol(out_eol), .out_last(core_last), .out_data(out_data)")
    a("    );")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": top, "reg": reg,
            "regs": regs, "ctrl": ctrl, "frame_sync": frame_sync,
            "offsets": reg["offsets"], "addr_bits": A,
            "in_bits": ib, "out_bits": ob, "signed": core.get("signed"),
            "dynamic": dyn, "eof": eof}


def control_wrap(core, registers, bind, module_name=None, addr_bits=16,
                 frame_sync=True, commit=None, header=(), notes=(),
                 passthrough=()) -> dict:
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
                       frame_sync=frame_sync)
    rw = [r for r in reg["regs"] if r.writable]

    streams = interface.get("streams")
    if streams is None:
        raise KeyError(
            f"module {inner!r} does not describe its streams; control_wrap needs "
            "interface['streams'] to pass them through")

    declared = [name for name, _, _ in interface["params"]]
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
    a(f"module {top} (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    // AXI4-Lite control interface. Every configuration register, and every")
    a("    // read-only identity word, is behind this port.")
    a(f"    input  wire [{A-1}:0] s_axil_awaddr, input wire s_axil_awvalid,"
      " output wire s_axil_awready,")
    a("    input  wire [31:0] s_axil_wdata, input wire [3:0] s_axil_wstrb,"
      " input wire s_axil_wvalid, output wire s_axil_wready,")
    a("    output wire [1:0] s_axil_bresp, output wire s_axil_bvalid,"
      " input wire s_axil_bready,")
    a(f"    input  wire [{A-1}:0] s_axil_araddr, input wire s_axil_arvalid,"
      " output wire s_axil_arready,")
    a("    output wire [31:0] s_axil_rdata, output wire [1:0] s_axil_rresp,"
      " output wire s_axil_rvalid, input wire s_axil_rready")
    for name, bits in passthrough:
        a(f"    // context, not configuration: this arrives with the data")
        a(f"    , input  wire [{bits-1}:0] {name}")
    for stream in streams:
        p, bits = stream["prefix"], stream["data_bits"]
        sign = "signed " if stream.get("signed") else ""
        if stream["direction"] == "in":
            a(f"    , input  wire        {p}_valid")
            a(f"    , output wire        {p}_ready")
            a(f"    , input  wire {sign}[{bits-1}:0] {p}_data")
            for flag in stream["flags"]:
                a(f"    , input  wire        {p}_{flag}")
        else:
            a(f"    , output wire        {p}_valid")
            a(f"    , input  wire        {p}_ready")
            a(f"    , output wire {sign}[{bits-1}:0] {p}_data")
            for flag in stream["flags"]:
                a(f"    , output wire        {p}_{flag}")
    a(");")
    a("")
    for r in rw:
        a(f"    wire {'signed ' if r.signed else ''}[{r.bits-1}:0] param_{r.name};"
          f"   // @ 0x{r.offset:04x}")
    a("")
    if frame_sync:
        a("    // Shadow -> live at the frame boundary: software may write a register")
        a("    // at any time, and the frame that follows sees ALL of the writes or")
        a("    // none of them. A coefficient changing mid-frame would tear the image.")
        a(f"    wire csr_commit = {commit};")
        a("")
    a(f"    {reg['module']} u_csr (")
    a("        .aclk(clk), .aresetn(!rst),")
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
    a(f"    {inner} u_datapath (")
    a(f"        .{interface['clock']}(clk),")
    a(f"        .{interface['reset']}(rst),")
    prefix = interface.get("param_prefix", "param_")
    for name, _, _ in interface["params"]:
        a(f"        .{prefix}{name}({bind[name]}),")
    lines = []
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
            "frame_sync": frame_sync, "commit": commit}


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
    a(f"    {core['module']} #(.WIDTH({W}), .HEIGHT({H})) dut (")
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
        val = int(param_values.get(name, 0))
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
    a(f"    {mod} #(.WIDTH({W}), .HEIGHT({H})) dut (")
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
        a(f"        param_{name} = {int(param_values.get(name, 0))};")
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
module {top} #(parameter WIDTH = {W}, parameter HEIGHT = {H}) (
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
    {core} #(.WIDTH(WIDTH), .HEIGHT(HEIGHT)) u_core (
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
    a(f"    {mod} #(.WIDTH({W}), .HEIGHT({H})) dut (")
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
        a(f"        param_{name} = {int(param_values.get(name, 0))};")
    a("        in_offer = 0; out_ready = 0;")
    a("        @(negedge clk); @(negedge clk); rst = 0;")
    a("    end")
    a("endmodule")
    return "\n".join(L)
