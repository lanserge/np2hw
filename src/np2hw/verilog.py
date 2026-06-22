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

import numpy as np

from .ir import SourceLine, HProcLine, VProcLine, Image2D, Const, Param


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
    """(name, bits, signed, default) for each distinct Param tap coefficient not
    already listed (e.g. as a trailing mulp/addp)."""
    seen = {p[0] for p in already}
    out = []
    for coeff in weighted.values():
        if isinstance(coeff, Param) and coeff.name not in seen:
            out.append((coeff.name, coeff.bits, coeff.signed, coeff.default))
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


def _emit_post(emit, post, acc_bits, acc_signed, prefix=""):
    """Emit the trailing pointwise stages; return (result_wire, bits, signed).
    `emit(line)` appends one Verilog line. `prefix` namespaces the wires so
    multiple datapaths (mux branches) don't collide."""
    prev, cur_bits, cur_signed = f"{prefix}acc", acc_bits, acc_signed
    for i, op in enumerate(post):
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
            pexpr = (f"$signed(param_{name})" if psg else
                     (f"$signed({{1'b0, param_{name}}})" if res_signed else f"param_{name}"))
            prevx = (f"$signed({{1'b0, {prev}}})" if res_signed and not cur_signed else prev)
            cur_bits, cur_signed = max(cur_bits, pbits) + 1, res_signed
            rhs = f"{prevx} + {pexpr}"
        else:                                            # mulp
            name, pbits, psg = op[1], op[2], op[3]
            res_signed = cur_signed or psg
            pexpr = (f"$signed(param_{name})" if psg else
                     (f"$signed({{1'b0, param_{name}}})" if res_signed else f"param_{name}"))
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
            post.append(("addp", c.name, c.bits, c.signed, c.default))
        else:
            c = line.coeff
            post.append(("mulc", c.value) if isinstance(c, Const)
                        else ("mulp", c.name, c.bits, c.signed, c.default))
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

def generate(out_line, module_name="np2hw_top", framing="height",
             max_width=None) -> dict:
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
    if type(out_line).__name__ == "Mux":                 # np.where(enable, A, B)
        return _generate_mux(out_line, module_name)
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
    acc_bits, range_signed = _range_bits(*_acc_range(weighted, in_lo, in_hi))
    params = [(op[1], op[2], op[3], op[4]) for op in post if op[0] in ("mulp", "addp")]
    params += _tap_params(weighted, params)              # programmable kernel coeffs
    signed = range_signed or any(ps for _, _, ps, _ in params)
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
    for name, bits, psg, _ in params:
        a(f"    input  wire {'signed ' if psg else ''}[{bits-1}:0] param_{name},   // register")
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
    a("    integer col;")
    a("    integer row;")
    a("    wire stall = out_valid && !out_ready;")       # holding an unaccepted output
    a("    assign in_ready = !stall;")
    # SOF re-anchors the current pixel to frame (0,0); tie in_sof=0 to free-run.
    # Output framing is DERIVED from the (effective) input position, so it tracks
    # SOF too -- no separate output counter to drift on re-anchor.
    a("    wire [31:0] ecol = in_sof ? 0 : col;")
    a("    wire [31:0] erow = in_sof ? 0 : row;")

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
    return {
        "verilog": verilog,
        "weighted": weighted,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": post,
        "params": [(n, b) for n, b, _, _ in params],
        "param_defaults": {n: d for n, _, _, d in params},
        "M": M,
        "N": N,
        "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name,
        "image": image,
    }


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
    sA, sgA = _range_bits(*_acc_range(tapsA, in_lo, in_hi))
    sB, sgB = _range_bits(*_acc_range(tapsB, in_lo, in_hi))
    sel = mux.select
    params = [(sel.name, sel.bits, sel.signed, sel.default)]
    for br in (A, B):
        params += [(op[1], op[2], op[3], op[4]) for op in br.post if op[0] in ("mulp", "addp")]
        params += _tap_params({k: v for k, v in br.taps.items()}, params)
    # dedup by name, keep order
    seen, uniq = set(), []
    for p in params:
        if p[0] not in seen:
            uniq.append(p); seen.add(p[0])
    params = uniq
    signed = sgA or sgB or any(ps for _, _, ps, _ in params[1:])
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
    for name, bits, psg, _ in params:
        a(f"    input  wire {'signed ' if psg else ''}[{bits-1}:0] param_{name},")
    a("    output reg  out_valid, input wire out_ready,")
    a("    output reg  out_sof, output reg out_eol, output reg out_last,")
    a("    output reg  OUT_SGN[OUT_BITS-1:0] out_data")
    a(");")
    a("    integer col; integer row;")
    a("    wire stall = out_valid && !out_ready;")
    a("    assign in_ready = !stall;")
    a("    wire [31:0] ecol = in_sof ? 0 : col;")          # SOF anchors to (0,0)
    a("    wire [31:0] erow = in_sof ? 0 : row;")
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
        acc_bits = _range_bits(*_acc_range(taps, in_lo, in_hi))[0]
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
        "signed": signed, "params": [(n, b) for n, b, _, _ in params],
        "param_defaults": {n: d for n, _, _, d in params},
        "M": M, "N": N, "out_rows": out_rows, "out_cols": out_cols,
        "module": module_name, "image": image,
    }


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
    acc_bits, signed = _range_bits(*_acc_range(weighted, in_lo, in_hi))
    rows_used = sorted({r for r, _ in weighted})
    params = [(op[1], op[2], op[3], op[4]) for op in post if op[0] in ("mulp", "addp")]
    signed = signed or any(ps for _, _, ps, _ in params)
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
    for name, bits, psg, _ in params:
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
    a("    integer col; integer row; integer fcol; integer frow;")
    a("    reg hf; reg vf; reg done;")
    if eof:
        a("    reg eof_l;")                               # latched: this frame's end was seen
    a("    wire [31:0] ecol = in_sof ? 0 : col;")          # SOF anchors to frame (0,0)
    a("    wire [31:0] erow = in_sof ? 0 : row;")
    a("    wire stall = out_valid && !out_ready;")       # holding an unaccepted output
    a("    wire in_active = !done && !hf && !vf;")
    a("    assign in_ready = !stall && (in_active || in_sof);")  # SOF accepted even mid-flush
    # advance: consume a pixel, or run a flush cycle -- never while output is held
    a("    wire en = !stall && !done && ((in_active && in_valid) || hf || vf || (in_sof && in_valid));")
    vbc = "in_sof || ((!vf) && (row == 0))" if (v_edge and rep) else "1'b0"
    hbc = "in_sof || ((!hf) && (col == 0))" if (h_edge and rep) else "1'b0"
    a(f"    wire vbc = {vbc};")               # top broadcast (replicate / SOF)
    a(f"    wire hbc = {hbc};")               # left broadcast (replicate / SOF)
    # line buffers
    for k in range(1, M + 1):
        a(f"    reg  [{in_bits-1}:0] mem{k} [0:{wparam}-1];")
        a(f"    wire [{in_bits-1}:0] chain{k} = mem{k}[ecol];")
    if v_edge and M >= 1:                                # bottom flush recirculates
        flush_src = "mem1[ecol]" if rep else "0"
        a(f"    wire [{in_bits-1}:0] chain0 = (vf && !in_sof) ? {flush_src} : in_data;")
    else:
        a(f"    wire [{in_bits-1}:0] chain0 = in_data;")
    # vertical taps (top: replicate via broadcast, or zero via mux)
    for r in rows_used:
        delay = M - r
        base = f"chain{delay}"
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
        a(f"                if (!hf || in_sof) mem{k}[ecol] <= vbc ? chain0 : chain{k-1};")
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
    return {
        "verilog": verilog,
        "weighted": weighted,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "signed": out_signed,
        "post": post,
        "params": [(n, b) for n, b, _, _ in params],
        "param_defaults": {n: d for n, _, _, d in params},
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
    }


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
# Each Param maps to a word address; software writes it over a bus.
# --------------------------------------------------------------------------- #

def axil_regfile(params, module_name="np2hw_axil", addr_bits=8,
                 frame_sync=False, defaults=None) -> dict:
    """AXI4-Lite slave exposing one register per Param at word offset i*4.
    Output `param_<name>` wires drive the core. Software writes/reads registers
    the standard way. (params: list of (name, bits).)

    frame_sync=True adds shadow registers + an `update` input: writes land in a
    shadow, and the live value (driving the core) copies from the shadow on an
    `update` pulse -- wire `update` to the core's out_last/SOF for glitch-free,
    frame-aligned coefficient updates (no mid-frame tearing)."""
    A = addr_bits
    dv = defaults or {}                                   # reset/power-on values
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
    for name, bits in params:
        a(f"    , output wire [{bits-1}:0] param_{name}")
    a(");")
    for i, (name, bits) in enumerate(params):
        a(f"    reg [{bits-1}:0] {wreg}{name};   // @ 0x{i*4:02x}")
        if frame_sync:
            a(f"    reg [{bits-1}:0] {dreg}{name};")
        a(f"    assign param_{name} = {dreg}{name};")
    a(f"    reg [{A-1}:0] awaddr_q; reg [{A-1}:0] araddr_q; reg aw_en;")
    # write channel
    a("    always @(posedge aclk) begin")
    a("        if (!aresetn) begin")
    a("            s_axil_awready<=0; s_axil_wready<=0; s_axil_bvalid<=0;")
    a("            s_axil_bresp<=0; aw_en<=1;")
    for name, _ in params:
        a(f"            {wreg}{name}<={int(dv.get(name,0))};")
    a("        end else begin")
    a("            if (!s_axil_awready && s_axil_awvalid && s_axil_wvalid && aw_en) begin")
    a("                s_axil_awready<=1; awaddr_q<=s_axil_awaddr; aw_en<=0;")
    a("            end else if (s_axil_bready && s_axil_bvalid) begin")
    a("                aw_en<=1; s_axil_awready<=0;")
    a("            end else s_axil_awready<=0;")
    a("            if (!s_axil_wready && s_axil_wvalid && s_axil_awvalid && aw_en)")
    a("                s_axil_wready<=1; else s_axil_wready<=0;")
    a("            if (s_axil_awready && s_axil_awvalid && s_axil_wready && s_axil_wvalid) begin")
    a(f"                case (awaddr_q[{A-1}:2])")
    for i, (name, bits) in enumerate(params):
        a(f"                    {i}: {wreg}{name} <= s_axil_wdata[{bits-1}:0];")
    a("                    default: ;")
    a("                endcase")
    a("                s_axil_bvalid<=1; s_axil_bresp<=0;")
    a("            end else if (s_axil_bready && s_axil_bvalid) s_axil_bvalid<=0;")
    a("        end")
    a("    end")
    if frame_sync:                                        # shadow -> live at update
        a("    always @(posedge aclk) begin")
        a("        if (!aresetn) begin")
        for name, _ in params:
            a(f"            {dreg}{name}<={int(dv.get(name,0))};")
        a("        end else if (update) begin")
        for name, _ in params:
            a(f"            {dreg}{name} <= {wreg}{name};")
        a("        end")
        a("    end")
    # read channel (reads the written/shadow value)
    a("    always @(posedge aclk) begin")
    a("        if (!aresetn) begin s_axil_arready<=0; s_axil_rvalid<=0; s_axil_rresp<=0; araddr_q<=0; s_axil_rdata<=0; end")
    a("        else begin")
    a("            if (!s_axil_arready && s_axil_arvalid) begin s_axil_arready<=1; araddr_q<=s_axil_araddr; end")
    a("            else s_axil_arready<=0;")
    a("            if (s_axil_arready && s_axil_arvalid && !s_axil_rvalid) begin")
    a("                s_axil_rvalid<=1; s_axil_rresp<=0;")
    a(f"                case (araddr_q[{A-1}:2])")
    for i, (name, bits) in enumerate(params):
        a(f"                    {i}: s_axil_rdata <= {{{32-bits}'b0, {wreg}{name}}};"
          if bits < 32 else f"                    {i}: s_axil_rdata <= {wreg}{name};")
    a("                    default: s_axil_rdata <= 0;")
    a("                endcase")
    a("            end else if (s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;")
    a("        end")
    a("    end")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": module_name,
            "params": list(params), "addr_bits": A, "frame_sync": frame_sync,
            "offsets": {name: i * 4 for i, (name, _) in enumerate(params)}}


def umi_regfile(params, module_name="np2hw_umi", addr_bits=8,
                write_op=1, read_op=2, frame_sync=False, defaults=None) -> dict:
    """SIMPLIFIED UMI register slave for the Switchboard stack: a request channel
    (valid/ready + opcode + addr + 32-bit data) writes/reads one register per
    Param at word offset i*4; reads produce a response. opcode = udev_req_cmd[4:0].
    frame_sync=True adds shadow registers + an `update` input (see axil_regfile).

    NOTE: this models UMI write/read register semantics but is NOT the full UMI
    command encoding. For production, use switchboard's umi_endpoint/umi_regif and
    connect the param_* outputs to it. (params: list of (name, bits).)"""
    A = addr_bits
    dv = defaults or {}                                   # reset/power-on values
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
    for name, bits in params:
        a(f"    , output wire [{bits-1}:0] param_{name}")
    a(");")
    for i, (name, bits) in enumerate(params):
        a(f"    reg [{bits-1}:0] {wreg}{name};   // @ 0x{i*4:02x}")
        if frame_sync:
            a(f"    reg [{bits-1}:0] {dreg}{name};")
        a(f"    assign param_{name} = {dreg}{name};")
    a("    assign udev_req_ready = !udev_resp_valid;")
    a(f"    wire is_write = udev_req_cmd[4:0] == 5'd{write_op};")
    a(f"    wire is_read  = udev_req_cmd[4:0] == 5'd{read_op};")
    a("    always @(posedge clk) begin")
    a("        if (!nreset) begin")
    a("            udev_resp_valid<=0; udev_resp_data<=0;")
    for name, _ in params:
        a(f"            {wreg}{name}<={int(dv.get(name,0))};")
    a("        end else begin")
    a("            if (udev_req_valid && udev_req_ready) begin")
    a("                if (is_write) begin")
    a(f"                    case (udev_req_addr[{A-1}:2])")
    for i, (name, bits) in enumerate(params):
        a(f"                        {i}: {wreg}{name} <= udev_req_data[{bits-1}:0];")
    a("                        default: ;")
    a("                    endcase")
    a("                end else if (is_read) begin")
    a("                    udev_resp_valid <= 1'b1;")
    a(f"                    case (udev_req_addr[{A-1}:2])")
    for i, (name, bits) in enumerate(params):
        rd = f"{{{32-bits}'b0, {wreg}{name}}}" if bits < 32 else f"{wreg}{name}"
        a(f"                        {i}: udev_resp_data <= {rd};")
    a("                        default: udev_resp_data <= 0;")
    a("                    endcase")
    a("                end")
    a("            end else if (udev_resp_valid && udev_resp_ready) udev_resp_valid<=0;")
    a("        end")
    a("    end")
    if frame_sync:
        a("    always @(posedge clk) begin")
        a("        if (!nreset) begin")
        for name, _ in params:
            a(f"            {dreg}{name}<={int(dv.get(name,0))};")
        a("        end else if (update) begin")
        for name, _ in params:
            a(f"            {dreg}{name} <= {wreg}{name};")
        a("        end")
        a("    end")
    a("endmodule")
    return {"verilog": "\n".join(L), "module": module_name,
            "params": list(params), "addr_bits": A, "frame_sync": frame_sync,
            "write_op": write_op, "read_op": read_op,
            "offsets": {name: i * 4 for i, (name, _) in enumerate(params)}}


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
