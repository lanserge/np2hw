# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Lowering: the canonical model graph to a hardware module, as passes.

    rewrite      model kinds -> hardware kinds: the datapath
    control      the source's driving variables -> counters and flags
    schedule     every node a stage; the stage map verified
    materialise  every value crossing a stage -> a register
    (print)      np2hw.core.emit

Each pass takes a unit and returns one; the runner verifies after each.
Families come one at a time: a model a pass does not take raises
`Unsupported`, and the reference compiler serves it.

The first family is pointwise: every read is the source's own pixel (or a
field of it, on the unrolled channel axis), so there is no window and no
storage but the output stage.
"""
from __future__ import annotations

from ..ir import Param
from .canonical import Canonical
from .graph import Graph, GraphError, Value, walk
from .passes import Pass, Unit, run
from .trace import Unsupported

from . import hw as _hw  # noqa: F401  (registers the hardware kinds)

#: Width of a driving variable's counter.
POS = 16

_BINARY = {"model.add": "hw.add", "model.sub": "hw.sub", "model.mul": "hw.mul",
           "model.and": "hw.and", "model.or": "hw.or", "model.lt": "hw.lt",
           "model.gt": "hw.gt", "model.le": "hw.le", "model.ge": "hw.ge",
           "model.eq": "hw.eq", "model.ne": "hw.ne"}


def _pow2(k: int) -> int | None:
    return k.bit_length() - 1 if k > 0 and k & (k - 1) == 0 else None


# --------------------------------------------------------------------------- #
# rewrite: the datapath
# --------------------------------------------------------------------------- #

def rewrite(c: Canonical, dev=None):
    """A pass from the canonical model graph into a fresh hardware graph;
    `dev` the device it is built for, which answers how a constant table is
    built (np2hw.timing.Device.rom_in_logic)."""
    from ..timing import device
    fam = device(dev)

    def run_(unit: Unit) -> Unit:
        mg, src = c.g, c.source
        find = mg.vars.find
        src_axes = [find(a) for a in src.type.axes]
        img = c.image
        # a video's frames drive outermost, the frame inside them
        tsrc = src_axes.pop(0) if getattr(img, "frames", 0) else None
        driving, chan = src_axes[:2], src_axes[2] if len(src_axes) > 2 else None
        # A window over the source: its word streams through whole and the
        # window is kept -- the control pass gates it. Only the source, as
        # yet: a value computed inside a window belongs to the next block.
        win = c.output.node if c.output.node.kind == "model.window" else None
        out_axes = [find(a) if a else None for a in c.output.type.axes]

        def var_start(v):
            """An index `var + start`: (its variable, its start) -- or None."""
            n = v.node
            while n.kind == "model.expand":     # a broadcast keeps the value
                n = n.operands[0].node
            if n.kind == "model.var":
                return find(n.attr("name")), 0
            if n.kind == "model.add":
                a_, b_ = n.operands
                for x, k in ((a_, b_), (b_, a_)):
                    inner = var_start(x) if k.node.kind == "model.const" \
                        else None
                    if inner is not None:       # starts add up
                        return inner[0], inner[1] + k.node.attr("value")
            return None

        # The window rule: reads of a padded buffer, or of the source at an
        # offset, by the output's own variables at constant starts. Each such
        # buffer is a WINDOW SOURCE: the spread of its starts is its storage,
        # line buffers for the rows and shift registers for the columns, and
        # the latest starts set the lag the output trails the input by.
        stencil: dict = {}
        out_frame = None
        for n in walk([c.output]):
            if n.kind != "model.access":
                continue
            base = n.operands[0].node
            if base.kind == "model.pad":
                pad = base
            elif base is src.node and tsrc is None:
                rc = [var_start(i) for i in n.operands[1:3]]
                if all(r is not None and r[1] == 0 and r[0] == a
                       for r, a in zip(rc, driving)):
                    continue                    # a pointwise read
                pad = None
            else:
                continue
            rc = [var_start(i) for i in n.operands[1:3]]
            if None in rc:
                raise Unsupported("a window read that is not the output's "
                                  "frame at a constant offset")
            if out_frame is None:
                out_frame = (rc[0][0], rc[1][0])
            elif out_frame != (rc[0][0], rc[1][0]):
                raise Unsupported("window reads on two lattices")
            key = id(base)
            if key not in stencil:
                if pad is not None:
                    (pt, _), (pl, _) = pad.attr("widths")[:2]
                    zero = pad.attr("mode") == "zero"
                else:
                    pt = pl = 0
                    zero = False
                stencil[key] = {"node": base, "pad": (pt, pl), "zero": zero,
                                "taps": set()}
            stencil[key]["taps"].add((rc[0][1], rc[1][1]))
        if stencil:
            if tsrc is not None or win is not None:
                raise Unsupported("a window over a video, or within a window")
            # one lag for the core: the most any window reaches past the
            # output's own pixel. Each source's taps are counted back from
            # the newest row and column that lag brings -- a source reaching
            # less keeps more lines, read further back.
            lag = tuple(max(max(t[i] for t in st["taps"]) - st["pad"][i]
                            for st in stencil.values()) for i in (0, 1))
            if min(lag) < 0:
                raise Unsupported("a window reading ahead of the output")
            for st in stencil.values():
                st["max"] = tuple(lag[i] + st["pad"][i] for i in (0, 1))
                st["K"] = tuple(st["max"][i] - min(t[i] for t in st["taps"]) + 1
                                for i in (0, 1))
            frame_vars = list(out_frame)
        else:
            frame_vars = None
        # a video's output frames: the output's own variable on time, which
        # every read across time indexes
        tout = out_axes.pop(0) if tsrc is not None and out_axes else None
        if tsrc is not None and tout is None:
            raise Unsupported("a video's output with no frame axis")
        if tsrc is not None and win is not None:
            raise Unsupported("a window over a video")
        # a stack of the frame's values along a last axis: channels of one
        # word, as np2hw packs every multi-channel stream
        frame = frame_vars or driving
        lanes = (c.output.node.operands
                 if c.output.node.kind == "model.stack"
                 and out_axes == frame + [find(c.output.node.attr("var"))]
                 else None)
        if win is not None:
            if win.operands[0].node is not src.node:
                raise Unsupported("a window over a computed value: window "
                                  "the source, and compute in the next block")
        elif lanes is None and out_axes != frame:
            raise Unsupported("an output that is not the source's frame")
        g = Graph(intern=True, locations=False)
        g.vars = mg.vars
        field = img.bits // (mg.vars.count(chan) if chan else 1)

        def start_on_t(v) -> int:
            """An index on the frame axis, `tout + start`: its start. The
            window rule on time -- every read through the output's own
            frame variable, at a constant offset."""
            n = v.node
            if n.kind == "model.var" and find(n.attr("name")) == tout:
                return 0
            if n.kind == "model.add":
                a, b = n.operands
                for x, k in ((a, b), (b, a)):
                    if (x.node.kind == "model.var"
                            and find(x.node.attr("name")) == tout
                            and k.node.kind == "model.const"):
                        return k.node.attr("value")
            raise Unsupported("a read across time that is not the output's "
                              "frame at a constant offset")

        # The reads across time: their starts on the frame axis. The latest
        # is the live frame, and each other start that many frames before
        # it is a frame the store keeps: the spread is the store's depth,
        # derived from the reads (docs/one-graph.md, decision 6).
        starts = set()
        whole = False
        if tsrc is not None:
            whole = c.output.node is src.node
            for n in walk([c.output]):
                for i, o in enumerate(n.operands):
                    if o.node is not src.node:
                        continue
                    if n.kind == "model.access" and i == 0:
                        starts.add(start_on_t(n.operands[1]))
                    else:
                        whole = True            # the video as a value
            if whole:
                # its frames are the output's own: the live one
                if tout != tsrc:
                    raise Unsupported("a video read whole, beside its own "
                                      "frames")
                starts.add(0)
        live = max(starts) if starts else 0
        stores = live - min(starts) if starts else 0
        word = img.bits                         # one sample's word, kept
        history = (g.op("hw.memread", name="hist", lo=0,
                        hi=(1 << (stores * word)) - 1) if stores else None)

        def past(depth: int, lo: int, width: int) -> Value:
            """A field of the sample `depth` frames before the live one:
            the live word itself, or the store's word for this position."""
            if depth == 0:
                return g.op("hw.bits", port_in(), lo=lo, width=width)
            return g.op("hw.bits", history, lo=(depth - 1) * word + lo,
                        width=width)
        params: dict = {}
        values: dict = {}

        def port_in():
            return g.op("hw.port", name="in_data", bits=img.bits,
                        lo=0, hi=(1 << img.bits) - 1)

        def one(n) -> Value:
            k = n.kind
            ops = [values[id(o.node)] if id(o.node) in values else None
                   for o in n.operands]
            if k == "model.table":
                return None                         # only its reads are lowered
            if n is src.node:
                if chan:
                    return None                     # only fields are read
                if img.signed:
                    raise Unsupported("a signed source")
                if tsrc is not None:
                    # the video as a value is its live frame; otherwise it
                    # is only what its reads read, each lowered on its own
                    return past(live, 0, img.bits) if whole else None
                return port_in()
            if k == "model.access":
                if n.operands[0].node.kind == "model.table":
                    return lookup(n, ops[1])
                if id(n.operands[0].node) in stencil:
                    return tap(n)
                return field_of(n)
            if k == "model.pad":
                return None                         # only its reads are lowered
            if k == "model.var":
                v = find(n.attr("name"))
                # a position is data only where the control pass finds it
                # read: the output's, or the source's beside no window
                if v not in (frame_vars or []) + driving and v != tout:
                    raise Unsupported(f"variable {v} used as data")
                return g.op("hw.position", var=v, count=mg.vars.count(v))
            if k == "model.expand":
                return ops[0]
            if k == "model.const":
                return g.op("hw.const", value=n.attr("value"))
            if k == "model.param":
                p = mg.symbols[n.attr("name")]["object"]
                params.setdefault(p.name, ((p.name, 0), (
                    p.name, p.bits, bool(p.signed), p.default,
                    p.description)))
                return g.op("hw.param", name=p.name, bits=p.bits,
                            signed=bool(p.signed))
            if k == "model.mul":
                # by a constant +-2**k: a left shift, negated for the
                # second, and no multiplier (timing.as_shift's question,
                # answered once here for this compiler)
                for i in (0, 1):
                    c = n.operands[i].node
                    if c.kind != "model.const":
                        continue
                    value = c.attr("value")
                    e = _pow2(abs(value))
                    if e is None:
                        continue
                    out = g.op("hw.shl", ops[1 - i], amount=e) if e \
                        else ops[1 - i]
                    return g.op("hw.neg", out) if value < 0 else out
            if k in _BINARY:
                return g.op(_BINARY[k], ops[0], ops[1])
            if k in ("model.neg", "model.abs"):
                return g.op("hw." + k[6:], ops[0])
            if k == "model.clip":
                return g.op("hw.clip", ops[0], lo=n.attr("lo"),
                            hi=n.attr("hi"))
            if k == "model.shr":
                return g.op("hw.shr", ops[0], amount=n.attr("amount"))
            if k == "model.floordiv":
                e = _pow2(n.attr("divisor"))
                if e is None:
                    raise Unsupported(f"floor division by {n.attr('divisor')}")
                return g.op("hw.shr", ops[0], amount=e)
            if k == "model.mod":
                e = _pow2(n.attr("divisor"))
                if e is None:
                    raise Unsupported(f"a remainder by {n.attr('divisor')}")
                return g.op("hw.bits", ops[0], lo=0, width=e)
            if k == "model.wrap":
                kind, bits = n.attr("container")
                return g.op("hw.wrap", ops[0], bits=bits, signed=kind == "i")
            if k == "model.bits":
                return g.op("hw.bits", ops[0], lo=n.attr("lo"),
                            width=n.attr("width"))
            if k == "model.where":
                return g.op("hw.sel", ops[0], ops[1], ops[2])
            if k == "model.select":
                return select(n, ops)
            raise Unsupported(f"{k} in the pointwise family")

        def lookup(n, index) -> Value:
            """A table read at a data index: a register array's entries, or a
            constant table's where the device builds it as logic, chosen by
            the index's bits -- a balanced tree of selects, its depth the
            index's width. Entries the index's range cannot reach are not
            built. A constant table the device builds as a memory is a read
            through a register, a stage of its own, which the pointwise
            family does not schedule."""
            name = n.operands[0].node.attr("name")
            obj = mg.symbols[name]["object"]
            size = obj.shape[0] if isinstance(obj, Param) else len(obj.data)
            if index.type.lo < 0 or index.type.hi >= size:
                # the model's error, not the compiler's: NumPy would raise
                # on such a frame, and hardware has nothing to raise
                raise ValueError(
                    f"{name}[{size}]: the index can reach [{index.type.lo}, "
                    f"{index.type.hi}], which falls outside the table. Mask "
                    "or clip the index first; hardware has no IndexError.")
            if isinstance(obj, Param):              # a register array

                def entry(i):
                    leaf = obj[i]
                    params.setdefault(leaf.name, ((name, i), (
                        leaf.name, leaf.bits, bool(leaf.signed),
                        leaf.default, leaf.description)))
                    return g.op("hw.param", name=leaf.name, bits=leaf.bits,
                                signed=bool(leaf.signed))
            else:                                   # a constant table
                data = [int(v) for v in obj.data]
                if not fam.rom_in_logic(size):
                    # a memory: its contents inline, read through a register,
                    # a stage of its own (the schedule's)
                    return g.op("hw.rom", index, name=name, data=tuple(data))

                def entry(i):
                    return g.op("hw.const", value=data[i])
            top = index.type.hi
            # the entries the index can reach, chosen by it
            out = (entry(0) if top == 0 else
                   g.op("hw.mux", index, *(entry(i) for i in range(top + 1))))
            # past one lookup table the device reads it in a cycle of its
            # own: the entries spread out across the die, and the route to
            # them is the cost (Device.gather_is_read)
            if isinstance(obj, Param) and fam.gather_is_read(size):
                return g.op("hw.read", out)
            return out

        def tap(n) -> Value:
            """A read of a window source at (start row, start column): the
            value `a` rows and `b` columns back from the newest in its
            window, which the control pass builds from line buffers, shift
            registers and the edge rule. Reading a channel of the source
            directly, the field of the word it keeps."""
            base = n.operands[0].node
            st = stencil[id(base)]
            if "value" not in st:
                # what the window keeps: the padded value, as this pixel has
                # it -- or the source's whole word, read at an offset
                kept = (values[id(base.operands[0].node)] if base.kind ==
                        "model.pad" else port_in())
                if kept is None:
                    raise Unsupported("a window over a value the pointwise "
                                      "family does not compute")
                st["value"] = kept
            kept = st["value"]
            (_, dr), (_, dc) = (var_start(i) for i in n.operands[1:3])
            t = g.op("hw.tap", key=id(base), a=st["max"][0] - dr,
                     b=st["max"][1] - dc, lo=kept.type.lo, hi=kept.type.hi)
            if base.kind == "model.pad":
                return t
            rest = n.operands[3:]
            if not chan:
                return t
            if not rest or rest[0].node.kind != "model.const":
                raise Unsupported("a read the pointwise family does not take")
            ch = rest[0].node.attr("value")
            return g.op("hw.bits", t, lo=ch * field, width=field)

        def field_of(n) -> Value:
            """An access of the source by its own driving variables, and by a
            constant on its unrolled channel axis: a field of the word."""
            if n.operands[0].node is not src.node:
                raise Unsupported("a read of a computed buffer")
            idx = n.operands[1:]
            depth = 0
            if tsrc is not None:
                depth = live - start_on_t(idx[0])
                idx = idx[1:]
            for i, a in zip(idx[:2], driving):
                if not (i.node.kind == "model.var"
                        and find(i.node.attr("name")) == a):
                    raise Unsupported("a read at an offset: a window")
            if not chan:
                if tsrc is None or len(idx) != 2:
                    raise Unsupported("a read the pointwise family does not "
                                      "take")
                return past(depth, 0, img.bits)
            if idx[2].node.kind != "model.const":
                raise Unsupported("a read the pointwise family does not take")
            ch = idx[2].node.attr("value")
            return past(depth, ch * field, field)

        def select(n, ops) -> Value:
            sizes = n.attr("sizes")
            keys, vals = ops[:len(sizes)], ops[len(sizes):]
            # keys that are bit fields of their sizes, outermost first, make
            # one index: one mux of every choice, the shape a lookup table
            # selects in
            bits = [(s - 1).bit_length() for s in sizes]
            if all(s == 1 << b and k.type.lo >= 0 and k.type.hi < s
                   for s, b, k in zip(sizes, bits, keys)):
                index = (keys[0] if len(keys) == 1 else
                         g.op("hw.cat", *reversed(keys), field=max(bits))
                         if len(set(bits)) == 1 else None)
                if index is not None:
                    return g.op("hw.mux", index, *vals)

            def pick(level, choices):
                if level == len(sizes):
                    return choices[0]
                step = len(choices) // sizes[level]
                parts = [pick(level + 1, choices[i * step:(i + 1) * step])
                         for i in range(sizes[level])]
                key = keys[level]
                out = parts[-1]
                for i in range(len(parts) - 2, -1, -1):
                    hit = g.op("hw.eq", key, g.op("hw.const", value=i)) \
                        if sizes[level] > 2 else g.op("hw.not", key)
                    out = g.op("hw.sel", hit, parts[i], out)
                return out

            return pick(0, vals)

        if win is not None:
            if img.signed:
                raise Unsupported("a signed source")
            bounds = tuple(one(o.node) for o in win.operands[1:])
            unit = Unit(g, {"data": port_in()})
            unit.facts["window"] = bounds           # y0, x0, h, w
            unit.facts["channels"] = mg.vars.count(chan) if chan else 1
        elif lanes is not None:
            for n in walk(list(lanes)):
                values[id(n)] = one(n)
            words = [values[id(o.node)] for o in lanes]
            for i, w in enumerate(words):
                if w.type.lo < 0:
                    raise Unsupported(
                        f"np.stack channel {i} can be negative ([{w.type.lo}, "
                        f"{w.type.hi}]); clip channels to their unsigned "
                        "range before stacking into a word")
            # one width for every channel, the widest's: the consumer's
            # channels view splits the word into equal fields
            field = max(1, max(w.type.hi.bit_length() for w in words))
            unit = Unit(g, {"data": g.op("hw.cat", *words, field=field)})
            unit.facts.update(out_channels=len(words), out_field=field)
        else:
            for n in walk([c.output]):
                values[id(n)] = one(n)
            unit = Unit(g, {"data": values[id(c.output.node)]})
        # register ports in the order the reference lists them
        # (canonical.registers_in_order); a register array's entries at its
        # place, by index
        if stencil:
            unit.facts["stencil"] = {
                "lag": lag, "out": (mg.vars.count(frame_vars[0]),
                                    mg.vars.count(frame_vars[1])),
                "vars": frame_vars,
                "sources": [{"key": k, "value": st["value"], "K": st["K"],
                             "zero": st["zero"]}
                            for k, st in stencil.items() if "value" in st]}
        if tsrc is not None:
            # frames the output waits for (the live read's start), and the
            # store: how many, and the word it keeps per position
            unit.facts["frames"] = {"skip": live, "stores": stores,
                                    "word": word, "history": history}
        order = list(c.registers)
        ordered = [p for key, p in sorted(
            params.values(), key=lambda kp: (order.index(kp[0][0]),
                                             kp[0][1]))]
        unit.facts.update(image=img, driving=driving, params=ordered)
        return unit

    return Pass("rewrite", run_, levels={"hw"})


# --------------------------------------------------------------------------- #
# control: counters and flags from the driving variables
# --------------------------------------------------------------------------- #

def _substitute(g: Graph, roots: dict, fn) -> dict:
    """Rebuild `roots` with every node `fn` maps replaced."""
    memo: dict = {}

    def go(v: Value) -> Value:
        n = v.node
        if id(n) not in memo:
            new = fn(n)
            memo[id(n)] = new.node if new is not None else g.node(
                n.kind, [go(o) for o in n.operands], dict(n.attrs))
        return memo[id(n)].result(v.index)

    return {k: go(v) for k, v in roots.items()}


def _stencil_control(unit, g, c, AND, NOT, EQ, SEL, reg_symbol, in_valid,
                     in_sof, out_ready) -> Unit:
    """The control of a core with windows (docs/one-graph.md, the window
    rule and decision 7). Its counters run over a STEP raster -- the output's
    frame grown by the lag, (H_out + L_r) x (W_out + L_c), or the input's
    where the input reaches further (a crop's rows and columns past its
    window, which are taken and never read) -- one step a beat: a step takes an input pixel while it lies inside the input's
    frame, and gives an output pixel once it is past the lag. The steps
    past the input's frame are FLUSH steps, which take nothing and hold the
    next frame back through `in_ready`; the frame's start re-anchors the
    steps only at a step that wants a pixel, so a frame waiting while the
    last is flushed waits.

    Each window source keeps its rows in line buffers, read a column ahead
    so a column's word lands as its step arrives, and its columns in a
    shift register a row. The edge rule is the pad's. Replicate: a row or
    column before the frame's first reads the first -- a select on the
    step's position -- and one past its last reads the last, because a
    flush step feeds the last row or column back in as the newest. Zero:
    the same selects, and the same flush, with zero in place of the edge."""
    img = unit.facts["image"]
    st = unit.facts["stencil"]
    H, W = img.height, img.width
    (lr, lc), (ho, wo) = st["lag"], st["out"]
    # a crop's TAIL: input rows and columns past the window's last
    tr, tc = H - ho - lr, W - wo - lc
    hs, ws = max(ho + lr, H), max(wo + lc, W)
    LT = lambda x, y: g.op("hw.lt", x, y)
    GE = lambda x, y: g.op("hw.ge", x, y)
    clip = lambda x, hi: g.op("hw.clip", x, lo=0, hi=hi)
    one = lambda x: g.op("hw.add", x, c(1))
    # The frame as it arrives: the build's size, or at a run-time size the
    # frame's own (the printer reads WIDTH and HEIGHT from the frame
    # registers then), never more than the build's. The steps' last row and
    # column follow it: the output's frame -- the input's, less any border
    # an unpadded window drops -- grown by the lag, or with a tail the
    # input's own. An axis with a tail has the slice's own extent, so its
    # output ends where the slice does.
    extent = dict(lo=1, hi=(1 << POS) - 1, bits=POS)
    fw = g.op("hw.parameter", name="WIDTH", **extent)
    fh = g.op("hw.parameter", name="HEIGHT", **extent)
    last_c = g.op("hw.add", fw, c(ws - W - 1))
    last_r = g.op("hw.add", fh, c(hs - H - 1))
    out_c = c(lc + wo - 1) if tc > 0 else last_c
    out_r = c(lr + ho - 1) if tr > 0 else last_r

    cy = reg_symbol("step_r", 0, hs - 1, 0)
    cx = reg_symbol("step_c", 0, ws - 1, 0)
    wants = AND(LT(cy, fh), LT(cx, fw))
    anchor = AND(AND(in_valid, in_sof), wants)
    ey, ex = SEL(anchor, c(0), cy), SEL(anchor, c(0), cx)
    valid = reg_symbol("out_valid", 0, 1, 0)
    ready = NOT(AND(valid, NOT(out_ready)))
    take = SEL(wants, in_valid, c(1))            # a pixel, or a flush step
    go = AND(ready, take)
    line_end = EQ(ex, last_c)
    # past each counter's last step the select takes 0, so the clip changes
    # no value
    nx = SEL(line_end, c(0), clip(one(ex), ws - 1))
    ny = SEL(line_end, SEL(EQ(ey, last_r), c(0), clip(one(ey), hs - 1)),
             ey)
    unit.regs.append(g.node("hw.reg", [nx, go], {"name": "step_c"}))
    unit.regs.append(g.node("hw.reg", [ny, go], {"name": "step_r"}))

    emit = AND(GE(ey, c(lr)), GE(ex, c(lc)))
    if tr > 0:
        emit = AND(emit, LT(ey, c(lr + ho)))
    if tc > 0:
        emit = AND(emit, LT(ex, c(lc + wo)))
    out_eol = EQ(ex, out_c)
    flags = {"valid": emit,
             "sof": AND(emit, AND(EQ(ey, c(lr)), EQ(ex, c(lc)))),
             "eol": AND(emit, out_eol),
             "last": AND(emit, AND(EQ(ey, out_r), out_eol))}

    taps, mems = {}, []
    aw = max(1, (W - 1).bit_length())
    # the column the next step reads: a column ahead, the next line's first
    # after the last
    ahead = SEL(LT(nx, fw), clip(nx, W - 1), c(0))
    for j, src in enumerate(st["sources"]):
        val = src["value"]
        kr, kc = src["K"]
        zero = src["zero"]
        if val.type.lo < 0:
            raise Unsupported("a window over a value that can be negative")
        bits = max(1, val.type.hi.bit_length())
        word = dict(lo=0, hi=(1 << bits) - 1)
        reads = [g.op("hw.memread", name=f"lb{j}_{a}", **word)
                 for a in range(1, kr)]
        # the newest row: this pixel's value, or past the frame's last row
        # the last row again
        cur = val if kr == 1 else SEL(LT(ey, fh), val,
                                      c(0) if zero else reads[0])
        rows_v = [cur]
        for a in range(1, kr):
            # a row before the frame's first reads the first
            rows_v.append(SEL(GE(ey, c(a)), reads[a - 1],
                              c(0) if zero else rows_v[a - 1]))
        for a in range(1, kr):
            name = f"lb{j}_{a}"
            unit.roots.update({
                f"mem_{name}_we": AND(go, LT(ex, fw)),
                f"mem_{name}_wa": g.op("hw.bits", ex, lo=0, width=aw),
                f"mem_{name}_wd": cur if a == 1 else reads[a - 2],
                f"mem_{name}_re": go, f"mem_{name}_ra": ahead})
            mems.append({"instance": f"u_{name}", "name": name, "bits": bits,
                         "depth": W, "aw": aw, "read": "sync",
                         "seam": "linebuf"})
        for a in range(kr):
            names = [f"sr{j}_{a}_{b}" for b in range(1, kc)]
            regs = [reg_symbol(n, 0, (1 << bits) - 1, None) for n in names]
            # the newest column: the row's word here, or past the line's
            # last column the last again
            col = rows_v[a] if kc == 1 else SEL(LT(ex, fw), rows_v[a],
                                                 c(0) if zero else regs[0])
            for b, n in enumerate(names, start=1):
                unit.regs.append(g.node(
                    "hw.reg", [col if b == 1 else regs[b - 2], go],
                    {"name": n}))
            cols_v = [col]
            for b in range(1, kc):
                # a column before the line's first reads the first
                cols_v.append(SEL(GE(ex, c(b)), regs[b - 1],
                                  c(0) if zero else cols_v[b - 1]))
            for b in range(kc):
                taps[(src["key"], a, b)] = cols_v[b]

    out = dict(zip(st["vars"], ((ey, lr), (ex, lc))))

    def substitute(n):
        if n.kind == "hw.tap":
            return taps[(n.attr("key"), n.attr("a"), n.attr("b"))]
        if n.kind == "hw.position":
            v = n.attr("var")
            if v not in out:
                raise Unsupported("the source's position used as data "
                                  "beside a window")
            at, lag = out[v]
            w = max(1, (n.attr("count") - 1).bit_length())
            return g.op("hw.bits", g.op("hw.sub", at, c(lag)) if lag else at,
                        lo=0, width=w)
        return None

    roots = _substitute(g, unit.roots, substitute)
    unit.roots = {**roots, **{f"flag_{k}": v for k, v in flags.items()}}
    unit.facts.update(go=go, ready=ready, in_valid=in_valid, take=take,
                      in_ready=AND(ready, wants), memories=mems,
                      out_size=(ho, wo), border=(H - ho, W - wo),
                      # cycles past a frame's last pixel taken until its
                      # output is out: the flush steps left, and a margin
                      drain=(lr + 1) * ws + 8)
    return unit


def control() -> Pass:
    """Each driving variable of the source gets a counter -- the innermost
    advancing on every accepted beat, the outer when the inner wraps -- and
    a position read by the datapath is its counter at this pixel. The flags
    are the counters' boundaries."""

    def run_(unit: Unit) -> Unit:
        g = unit.g
        rows, cols = unit.facts["driving"]
        one_bit = dict(lo=0, hi=1, bits=1)
        pos = dict(lo=0, hi=(1 << POS) - 1, bits=POS)
        port = lambda name: g.op("hw.port", name=name, **one_bit)
        in_valid, in_sof, out_ready = (port("in_valid"), port("in_sof"),
                                       port("out_ready"))
        # a raster has at least one row and one column: so `limit - 1` is
        # never negative, and comparing a counter with it is unsigned, in
        # the counter's own width
        extent = dict(pos, lo=1)
        limit = {cols: g.op("hw.parameter", name="WIDTH", **extent),
                 rows: g.op("hw.parameter", name="HEIGHT", **extent)}
        c = lambda v: g.op("hw.const", value=v)
        AND = lambda x, y: g.op("hw.and", x, y)
        NOT = lambda x: g.op("hw.not", x)
        EQ = lambda x, y: g.op("hw.eq", x, y)
        SEL = lambda s, x, y: g.op("hw.sel", s, x, y)

        def reg_symbol(name, lo, hi, init):
            g.symbols[name] = {"lo": lo, "hi": hi, "init": init}
            return g.op("hw.regread", name=name)

        if "stencil" in unit.facts:
            return _stencil_control(unit, g, c, AND, NOT, EQ, SEL,
                                    reg_symbol, in_valid, in_sof, out_ready)
        cnt = {v: reg_symbol(f"cnt_{v}", 0, (1 << POS) - 1, 0)
               for v in (rows, cols)}
        valid = reg_symbol("out_valid", 0, 1, 0)
        stall = AND(valid, NOT(out_ready))
        ready = NOT(stall)
        go = AND(ready, in_valid)                        # a beat accepted
        epos = {v: SEL(in_sof, c(0), cnt[v]) for v in (rows, cols)}
        last = {v: g.op("hw.sub", limit[v], c(1)) for v in (rows, cols)}
        inc = lambda v: g.op("hw.wrap", g.op("hw.add", cnt[v], c(1)),
                             bits=POS, signed=False)
        wrap_col = EQ(cnt[cols], last[cols])
        nxt = {cols: SEL(in_sof, c(1), SEL(wrap_col, c(0), inc(cols))),
               rows: SEL(in_sof, c(0), SEL(wrap_col,
                                           SEL(EQ(cnt[rows], last[rows]),
                                               c(0), inc(rows)),
                                           cnt[rows]))}
        for v in (rows, cols):
            unit.regs.append(g.node("hw.reg", [nxt[v], go],
                                    {"name": f"cnt_{v}"}))
        inside = AND(g.op("hw.lt", epos[rows], limit[rows]),
                     g.op("hw.lt", epos[cols], limit[cols]))
        first = lambda v: EQ(epos[v], c(0))
        final = lambda v: EQ(epos[v], last[v])
        if "window" in unit.facts:
            # kept: inside the frame and the window; the frame starts at
            # the window's corner, and a line (the frame) ends at the
            # window's edge or the frame's own, whichever comes first
            y0, x0, h, w = unit.facts["window"]
            # The window's last row and column: frame-constant, since the
            # registers placing it change only at a commit, between
            # frames -- so worked out once into a register and compared
            # against, not added up again beneath every pixel's verdict,
            # where the sum and the compare were one carry chain too long
            # for the clock.
            end = {}
            for v, (at, n) in ((rows, (y0, h)), (cols, (x0, w))):
                bound = g.op("hw.sub", g.op("hw.add", at, n), c(1))
                end[v] = reg_symbol(f"win_end_{v}", bound.type.lo,
                                    bound.type.hi, 0)
                unit.regs.append(g.node("hw.reg", [bound, c(1)],
                                        {"name": f"win_end_{v}"}))
            start = {rows: y0, cols: x0}
            keep = lambda v: AND(g.op("hw.ge", epos[v], start[v]),
                                 g.op("hw.le", epos[v], end[v]))
            inside = AND(inside, AND(keep(rows), keep(cols)))
            edge = lambda v: g.op("hw.or", EQ(epos[v], end[v]), final(v))
            flags = {"valid": inside,
                     "sof": AND(inside, AND(EQ(epos[rows], y0),
                                            EQ(epos[cols], x0))),
                     "eol": AND(inside, edge(cols)),
                     "last": AND(inside, AND(edge(rows), edge(cols)))}
        else:
            flags = {"valid": inside,
                     "sof": AND(inside, AND(first(rows), first(cols))),
                     "eol": AND(inside, final(cols)),
                     "last": AND(inside, AND(final(rows), final(cols)))}
        fr = unit.facts.get("frames")
        if fr is not None:
            # how many frames have been taken, up to the number the output
            # waits for: the first frames make no output, as the model's
            # slices start the history once it is full
            skip, stores, word = fr["skip"], fr["stores"], fr["word"]
            end = AND(go, AND(final(rows), final(cols)))
            if skip:
                taken = reg_symbol("frames_taken", 0, skip, 0)
                # enabled only below `skip`, so the clip changes no value
                unit.regs.append(g.node("hw.reg", [
                    g.op("hw.clip", g.op("hw.add", taken, c(1)), lo=0,
                         hi=skip),
                    AND(end, g.op("hw.ne", taken, c(skip)))],
                    {"name": "frames_taken"}))
                full = EQ(taken, c(skip))
                flags = {k: AND(v, full) for k, v in flags.items()}
            if stores:
                # the store: one word per position, holding that position's
                # past samples, newest lowest. Each pixel taken writes its
                # word back SHIFTED -- its own sample in, the oldest out --
                # and the read runs a position ahead, from the address's
                # next state, so the next pixel's word lands as it arrives.
                img = unit.facts["image"]
                size = img.width * img.height
                aw = max(1, (size - 1).bit_length())
                addr = reg_symbol("hist_addr", 0, size - 1, 0)
                here = SEL(in_sof, c(0), addr)
                # past the last position the select takes 0, so the clip
                # changes no value
                ahead = SEL(EQ(here, c(size - 1)), c(0),
                            g.op("hw.clip", g.op("hw.add", here, c(1)), lo=0,
                                 hi=size - 1))
                unit.regs.append(g.node("hw.reg", [ahead, go],
                                        {"name": "hist_addr"}))
                hist = fr["history"]
                live = g.op("hw.port", name="in_data", bits=word, lo=0,
                            hi=(1 << word) - 1)
                kept = [live] + [g.op("hw.bits", hist, lo=k * word,
                                      width=word)
                                 for k in range(stores - 1)]
                unit.roots.update({
                    "mem_hist_we": go, "mem_hist_wa": here,
                    "mem_hist_wd": g.op("hw.cat", *kept, field=word),
                    "mem_hist_re": go, "mem_hist_ra": ahead})
                unit.facts["memories"] = [{
                    "instance": "u_hist", "name": "hist", "bits":
                    stores * word, "depth": size, "aw": aw, "read": "sync",
                    "seam": "mem"}]

        def position(n):
            """The datapath's position along a variable: the counter at this
            pixel, in the bits its count needs -- the range NumPy gave it."""
            if n.kind != "hw.position":
                return None
            if n.attr("var") not in epos:
                raise Unsupported("the frame's position used as data")
            w = max(1, (n.attr("count") - 1).bit_length())
            return g.op("hw.bits", epos[n.attr("var")], lo=0, width=w)

        roots = _substitute(g, unit.roots, position)
        unit.roots = {**roots, **{f"flag_{k}": v for k, v in flags.items()}}
        unit.facts.update(go=go, ready=ready, in_valid=in_valid)
        return unit

    return Pass("control", run_, levels={"hw"})


# --------------------------------------------------------------------------- #
# schedule and materialise
# --------------------------------------------------------------------------- #

# Leaves that FLOAT: a constant is wiring and a register port, or a build
# parameter, is stable for the whole frame, so each feeds any stage directly
# and never rides with a pixel. Everything else that is this pixel's -- its
# word, its position, its stored past -- is pinned to the stage it arrives
# in, and a later stage sees it through the registers that carry it.
_FLOATING = ("hw.const", "hw.param", "hw.parameter")


def schedule(clk_ns=None, dev=None) -> Pass:
    """Every node of the datapath a stage, so each stage fits `clk_ns` on
    the device `dev`, by the device's own cost model (np2hw.timing). A node
    lands in the latest stage any of its inputs lives in; when its path
    there would exceed the budget it starts the next stage, its inputs
    arriving through that stage's registers. With no budget, or one the
    whole datapath fits in, it is one stage: the datapath and the flags in
    the beat, registered at the output. An operation that alone exceeds the
    budget is refused: no register can land inside one."""

    def run_(unit: Unit) -> Unit:
        unit.facts["stages"] = 1
        # a constant table's memory read takes a stage whatever the budget:
        # with none, the only cuts are the reads'
        # the datapath and the flags beside it: what the output stage reads
        outs = [unit.roots["data"]] + [v for k, v in unit.roots.items()
                                       if k.startswith("flag_")]
        roms = any(n.kind == "hw.rom" for n in walk(outs))
        if clk_ns is None and not roms:
            return unit
        budget = float("inf") if clk_ns is None else clk_ns
        from ..timing import Depth, device
        from .pipeline import _cost
        fam = device(dev)
        stage: dict = {}
        depth: dict = {}
        for n in walk(outs):
            if n.kind in _FLOATING:
                continue
            cost = _cost(n, fam)
            if fam.ns(cost) > budget:
                raise ValueError(
                    f"operation '{n.kind.split('.', 1)[1]}' alone estimates "
                    f"{fam.ns(cost):.1f} ns against "
                    f"the {budget:.1f} ns budget on {fam.name} -- a pipeline "
                    "register cannot land inside one operation. Slow the "
                    "clock or restructure the model.")
            staged = [o.node for o in n.operands if o.node.kind not in _FLOATING]
            s = max((stage[id(o)] for o in staged), default=0)
            if n.kind == "hw.read":
                # a read owns its stage: what it read lands in the next,
                # through the register the stage boundary is, at the
                # clock-to-out the device costs a read at
                if clk_ns is not None:
                    stage[id(n)], depth[id(n)] = s + 1, fam.mem()
                    continue
            if n.kind == "hw.rom":
                # a memory read: its address registered at the end of the
                # stage that computes it, its word in the next -- always
                stage[id(n)], depth[id(n)] = s + 1, cost
                continue
            below = Depth()
            for o in staged:
                if stage[id(o)] == s:
                    below = fam.worst(below, depth[id(o)])
            d = below + cost
            if fam.ns(d) > budget:
                s, d = s + 1, cost
            stage[id(n)], depth[id(n)] = s, d
        unit.facts["stages"] = max(stage.get(id(v.node), 0) for v in outs) + 1
        unit.facts["stage_of"] = stage
        return unit

    return Pass("schedule", run_, levels={"hw"})


def materialise() -> Pass:
    """Every value crossing a stage becomes a register. In one stage: the
    data at the output -- no reset, it means something only while out_valid
    says so -- and the flags, reset. In several: each value a later stage
    reads is carried to it through registers of its own, one a stage,
    shared by every reader that needs it there; the flags ride the same
    stages beside the data; and one ready holds every stage while the
    output stalls."""

    def run_(unit: Unit) -> Unit:
        g, f = unit.g, unit.facts
        data = unit.roots["data"]
        stages = f.get("stages", 1)
        outputs = {"in_ready": f.get("in_ready", f["ready"])}
        # what makes a step's flags real: an input pixel taken, or a step
        # that needs none (a window's flush)
        taken_by = f.get("take", f["in_valid"])
        if stages == 1:
            g.symbols["out_data"] = {"lo": data.type.lo, "hi": data.type.hi,
                                     "init": None}
            unit.regs.append(g.node("hw.reg", [data, f["go"]],
                                    {"name": "out_data"}))
            for flag in ("valid", "sof", "eol", "last"):
                name = f"out_{flag}"
                g.symbols.setdefault(name, {"lo": 0, "hi": 1, "init": 0})
                unit.regs.append(g.node(
                    "hw.reg", [g.op("hw.and", taken_by,
                                    unit.roots[f"flag_{flag}"]), f["ready"]],
                    {"name": name}))
                outputs[name] = g.op("hw.regread", name=name)
        else:
            stage_of = f["stage_of"]
            ready = f["ready"]
            built: dict = {}
            carried: dict = {}

            def home_of(n) -> int:
                """The stage `n` is computed in: the schedule's, or for a
                node the schedule never saw -- one materialise makes, as
                the taken flags -- the latest stage among its inputs."""
                if id(n) not in stage_of:
                    stage_of[id(n)] = max(
                        (home_of(o.node) for o in n.operands
                         if o.node.kind not in _FLOATING), default=0)
                return stage_of[id(n)]

            def at(v: Value, k: int, reset=None) -> Value:
                """`v` as stage `k` sees it: computed there, or carried
                from the stage it was computed in through a register a
                stage."""
                n = v.node
                if n.kind in _FLOATING:
                    return v
                home = home_of(n)
                if k == home:
                    return here(v)
                key = (id(n), v.index, k)
                if key not in carried:
                    name = f"p{len(carried)}_s{k}"
                    g.symbols[name] = {"lo": v.type.lo, "hi": v.type.hi,
                                       "init": reset}
                    unit.regs.append(g.node(
                        "hw.reg", [at(v, k - 1, reset), ready],
                        {"name": name}))
                    carried[key] = g.op("hw.regread", name=name)
                return carried[key]

            def here(v: Value) -> Value:
                """`v` built in its own stage, its inputs as that stage
                sees them -- every node below it built first, in order, so
                no chain of operands is followed by recursion."""
                for n in walk([v]):
                    if id(n) in built:
                        continue
                    home = home_of(n)
                    built[id(n)] = g.node(
                        n.kind, [at(o, home) for o in n.operands],
                        dict(n.attrs))
                return built[id(v.node)].result(v.index)

            last = stages - 1
            word = at(data, last)
            g.symbols["out_data"] = {"lo": data.type.lo, "hi": data.type.hi,
                                     "init": None}
            unit.regs.append(g.node("hw.reg", [word, ready],
                                    {"name": "out_data"}))
            for flag in ("valid", "sof", "eol", "last"):
                name = f"out_{flag}"
                g.symbols.setdefault(name, {"lo": 0, "hi": 1, "init": 0})
                taken = g.op("hw.and", taken_by,
                             unit.roots[f"flag_{flag}"])
                unit.regs.append(g.node("hw.reg", [at(taken, last, 0), ready],
                                        {"name": name}))
                outputs[name] = g.op("hw.regread", name=name)
        outputs["out_data"] = g.op("hw.regread", name="out_data")
        # a memory's ports, wired to its instance by the printer
        outputs.update({k: v for k, v in unit.roots.items()
                        if k.startswith("mem_")})
        unit.roots = outputs
        f.update(out_bits=data.type.bits, out_signed=data.type.lo < 0)
        return unit

    return Pass("materialise", run_, levels={"hw"})


def lower(c: Canonical, dev=None, clk_ns=None) -> Unit:
    """The canonical graph, lowered to a hardware module for the device
    `dev` (the default device when None), its stages fitting `clk_ns` when
    one is given -- or Unsupported."""
    unit = Unit(c.g, {"out": c.output})
    try:
        return run(unit, [rewrite(c, dev), control(), schedule(clk_ns, dev),
                          materialise()])
    except GraphError as e:
        raise Unsupported(f"lowering: {e}") from e
