# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Lowering: the canonical model graph to a hardware module, as passes.

    rewrite      model kinds -> hardware kinds: the datapath
    control      the source's driving variables -> counters and flags
    schedule     every node a stage; the stage map verified
    materialise  every value crossing a stage -> a register
    (print)      np2hw.core.emit

Each pass takes a unit and returns one; the runner verifies after each. A
model a pass does not take raises `Unsupported`, naming what it does.

The families: pointwise, where every read is the source's own pixel (or a
field of it, on the unrolled channel axis); windows, reads of the source
at offsets, kept in line buffers and shift registers; and a video's frames,
reads across time, kept in a frame store.
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
           "model.eq": "hw.eq", "model.ne": "hw.ne", "model.max": "hw.max",
           "model.xor": "hw.xor",
           "model.min": "hw.min"}


def _pow2(k: int) -> int | None:
    return k.bit_length() - 1 if k > 0 and k & (k - 1) == 0 else None


def _folds(out: Value):
    """[(scatter node, the container its result wraps into, or None)] for
    a result that is a fold -- or a stack of folds, side by side along its
    last axis -- else None."""
    def one(v):
        n, wrap = v.node, None
        if n.kind == "model.wrap":
            wrap = n.attr("container")
            n = n.operands[0].node
        return (n, wrap) if n.kind == "model.scatter" else None
    n = out.node
    if n.kind == "model.stack":
        got = [one(o) for o in n.operands]
        if not all(got):
            return None
        if n.attr("axis") % len(out.type.axes) != len(out.type.axes) - 1:
            raise Unsupported("a stack of folds along an axis other than its "
                              "last: their entries side by side, an entry's "
                              "values together")
        return got
    got = one(out)
    return [got] if got else None


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
        # The source read at the output's own pixel -- whole, as a value, or
        # through its own variables -- by the datapath rather than by what a
        # window stores: beside a window, the pixel its lag has reached,
        # which only the source's own window has.
        direct = False
        if stencil and tsrc is None:
            seen, todo = set(), [c.output.node]
            while todo and not direct:
                n = todo.pop()
                if id(n) in seen or n.kind == "model.pad":
                    continue                    # what a window stores: live
                seen.add(id(n))
                for i, o in enumerate(n.operands):
                    if o.node is not src.node:
                        todo.append(o.node)
                    elif not (n.kind == "model.access" and i == 0):
                        direct = True           # the source as a value
                    elif all(r is not None and r[1] == 0 and r[0] == a
                             for r, a in zip((var_start(x) for x in
                                              n.operands[1:3]), driving)):
                        direct = True           # read at its own pixel
        if stencil and direct:
            st = stencil.setdefault(id(src.node), {
                "node": src.node, "pad": (0, 0), "zero": False,
                "taps": set()})
            st["taps"].add((0, 0))
            if out_frame is None:
                out_frame = tuple(driving)
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
        # A fold: every position of the frame into the entries of a result
        # that belongs to the whole frame -- the core a sink, its result
        # read back after the frame
        folds = _folds(c.output)
        if folds is not None and (tsrc is not None or win is not None):
            raise Unsupported("a fold over a video, or over a window")
        if win is not None:
            if win.operands[0].node is not src.node:
                raise Unsupported("a window over a computed value: window "
                                  "the source, and compute in the next block")
        elif folds is None and lanes is None and out_axes != frame:
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
            # A read across time is a read of the STREAM's frames: those are
            # what a store keeps. A value computed from the history and then
            # sliced on its frames is not a stream a store can keep, and
            # read as one it would be the live frame at every offset.
            for n in walk([c.output]):
                if n.kind != "model.access" or n.operands[0].node is src.node:
                    continue
                base = n.operands[0]
                if any(a and find(a) == tsrc for a in base.type.axes):
                    raise Unsupported(
                        "a read across time of a value computed from the "
                        "stream's history: slice the history first -- "
                        "video[1:], video[:-1] -- then compute on each frame")
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
        delivered = img.delivered
        if delivered and stores >= delivered:
            raise Unsupported(f"a read {stores} frames back, of a stream "
                              f"that brings {delivered} side by side")
        history = (g.op("hw.memread", name="hist", lo=0,
                        hi=(1 << (stores * word)) - 1)
                   if stores and not delivered else None)

        def past(depth: int, lo: int, width: int) -> Value:
            """A field of the sample `depth` frames before the live one:
            the live word itself, or the store's word for this position --
            or, where the stream brings its history, the word's own field."""
            if depth == 0 or delivered:
                return g.op("hw.bits", port_in(), lo=depth * word + lo,
                            width=width)
            return g.op("hw.bits", history, lo=(depth - 1) * word + lo,
                        width=width)
        params: dict = {}
        values: dict = {}
        stored: dict = {}       # what a window stores: the input's pixel

        def port_in():
            return g.op("hw.port", name="in_data", bits=img.word_bits,
                        lo=0, hi=(1 << img.word_bits) - 1)

        def one(n, now: bool = False) -> Value:
            """`n` as the datapath has it at the output's pixel -- or, `now`,
            at the input's, which is what a window stores."""
            k = n.kind
            memo = stored if now else values
            ops = [memo[id(o.node)] if id(o.node) in memo else None
                   for o in n.operands]
            lagged = id(src.node) in stencil and not now
            if lagged:
                # the source as a value beside a window: its own window, at
                # the output's pixel
                ops = [src_tap() if o.node is src.node and not (
                    k == "model.access" and i == 0) else v
                    for i, (o, v) in enumerate(zip(n.operands, ops))]
            if k == "model.table":
                return None                         # only its reads are lowered
            if n is src.node:
                if img.signed:
                    raise Unsupported("a signed source")
                if chan:
                    return None                     # only fields are read
                if tsrc is not None:
                    # the video as a value is its live frame; otherwise it
                    # is only what its reads read, each lowered on its own
                    return past(live, 0, img.bits) if whole else None
                return port_in()
            if k == "model.access":
                if n.operands[0].node.kind == "model.table":
                    return lookup(n, ops[1])
                if id(n.operands[0].node) in stencil and not (
                        now and n.operands[0].node is src.node):
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
                if getattr(p, "frame", False):
                    # the frame's own value: held and carried (_frame_values)
                    return g.op("hw.frameparam", name=p.name, bits=p.bits,
                                signed=bool(p.signed))
                return g.op("hw.param", name=p.name, bits=p.bits,
                            signed=bool(p.signed))
            if k == "model.mul":
                # by a constant +-2**k: a left shift, negated for the
                # second, and no multiplier
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

        def src_tap() -> Value:
            st = stencil[id(src.node)]
            st.setdefault("value", port_in())
            return g.op("hw.tap", key=id(src.node), a=st["max"][0],
                        b=st["max"][1], lo=st["value"].type.lo,
                        hi=st["value"].type.hi)

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
                if base.kind == "model.pad":
                    for m in walk([base.operands[0]]):
                        stored[id(m)] = one(m, now=True)
                    kept = stored[id(base.operands[0].node)]
                else:
                    kept = port_in()
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

        if folds is not None:
            unit = _fold_unit(g, mg, folds, values, one, frame, c.output)
        elif win is not None:
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
        # register ports in the order canonical.registers_in_order gives; a
        # register array's entries at its place, by index
        if stencil:
            unit.facts["stencil"] = {
                "lag": lag, "out": (mg.vars.count(frame_vars[0]),
                                    mg.vars.count(frame_vars[1])),
                "vars": frame_vars,
                # how each output axis ends, as the model's slices said
                "ends": tuple(mg.vars.ends(v) for v in frame_vars),
                "sources": [{"key": k, "value": st["value"], "K": st["K"],
                             "zero": st["zero"]}
                            for k, st in stencil.items() if "value" in st]}
        if tsrc is not None:
            # frames the output waits for (the live read's start), and the
            # store: how many, and the word it keeps per position
            # (a history the stream brings is whole from its first frame:
            # nothing waits, and nothing is kept)
            unit.facts["frames"] = (
                {"skip": 0, "stores": 0, "word": word, "history": None}
                if delivered else
                {"skip": live, "stores": stores, "word": word,
                 "history": history})
        order = list(c.registers)
        ordered = [p for key, p in sorted(
            params.values(), key=lambda kp: (order.index(kp[0][0]),
                                             kp[0][1]))]
        unit.facts.update(image=img, driving=driving, params=ordered)
        return unit

    return Pass("rewrite", run_, levels={"hw"})


def _fold_unit(g, mg, folds, values, one, frame, output) -> Unit:
    """A fold's pixel datapath: per fold, the entry each pixel folds into
    -- its keys made one index, row-major -- and its value, less the least
    it can be, side by side in one word. The word streams through the
    stages like any core's output, and the accumulate pass folds it."""
    find = mg.vars.find
    for n in walk([o for f, _ in folds for o in f.operands]):
        values[id(n)] = one(n)
    at = set(find(v) for v in frame)
    parts, widths, lanes = [], [], []
    bins = None
    for f, wrap in folds:
        v, *keys = f.operands
        reach = {find(a) for o in f.operands for a in o.type.axes if a}
        if reach != at:
            raise Unsupported(
                "a fold over other positions than the frame's: each pixel "
                "folds once, so the value and its keys range over the frame")
        counts = [mg.vars.count(a) for a in f.attr("axes")]
        n = 1
        for k in counts:
            n *= k
        if bins not in (None, n):
            raise GraphError("folds side by side of unequal sizes")
        bins = n
        flat = None
        for key, count in zip(keys, counts):
            k = values[id(key.node)]
            if flat is None:
                flat = k
                continue
            e = _pow2(count)
            flat = g.op("hw.add", g.op("hw.shl", flat, amount=e) if e is not None
                        else g.op("hw.mul", flat, g.op("hw.const", value=count)),
                        k)
        hv = values[id(v.node)]
        vlo = hv.type.lo
        field = g.op("hw.sub", hv, g.op("hw.const", value=vlo)) if vlo else hv
        t = f.value.type
        lane = {"op": f.attr("op"), "init": f.attr("init"), "lo": t.lo,
                "hi": t.hi, "vlo": vlo,
                "vbits": max(1, field.type.hi.bit_length()),
                "wrap": wrap}
        if n > 1:
            lane["kbits"] = max(1, (n - 1).bit_length())
            parts.append(flat)
            widths.append(lane["kbits"])
        parts.append(field)
        widths.append(lane["vbits"])
        lanes.append(lane)
    unit = Unit(g, {"data": g.op("hw.cat", *parts, fields=tuple(widths))})
    unit.facts["folds"] = {
        "lanes": lanes, "bins": bins,
        "stacked": output.node.kind == "model.stack",
        "shape": tuple(mg.vars.count(a) if a else 1
                       for a in output.type.axes)}
    return unit


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


def _ctx_port(g, img) -> Value:
    """The stream's context, as each frame's first beat brings it."""
    bits = img.context_bits
    return g.op("hw.port", name="in_ctx", bits=bits, lo=0, hi=(1 << bits) - 1)


def _ctx_field(g, img, want) -> Value | None:
    """The field of the context `want(name, follows)` picks, or None."""
    lo = 0
    for name, bits, follows in img.context:
        if want(name, follows):
            return g.op("hw.bits", _ctx_port(g, img), lo=lo, width=bits)
        lo += bits
    return None


def _out_ctx(g, unit, reg_symbol, handed: dict, load) -> Value:
    """The context a core hands on with its frame: the frame's size where a
    field follows it -- the size this core puts out -- and every other
    field as the frame brought it, each in a register loaded with the
    frame's first beat, like the size it sits beside."""
    img = unit.facts["image"]
    parts = []
    for name, bits, follows in img.context:
        if follows == "width":
            v = handed["out_frame_width"]
        elif follows == "height":
            v = handed["out_frame_height"]
        else:
            field = _ctx_field(g, img, lambda n, f: n == name)
            v = reg_symbol(f"ctx_{name}", 0, (1 << bits) - 1, 0)
            unit.regs.append(g.node("hw.reg", [field, load],
                                    {"name": f"ctx_{name}"}))
        parts.append(v)
    # the fields side by side, each leaving its own flop: wiring, no logic
    return g.op("hw.cat", *parts,
                fields=tuple(bits for _, bits, _ in img.context))


def _frame_values(g, unit, reg_symbol, SEL, load):
    """A substitution for each value a frame brings (``Param(frame=True)``):
    its port as the frame's first pixel is taken, and the copy held from
    then on, loaded with it -- so a core further down the chain takes the
    value its frame entered with, though the source has moved on to the
    next. Read at the first stage, it is pinned there, and the stages carry
    it with the frame's own beats: the frame before, still inside, keeps
    its own."""
    made: dict = {}

    def substitute(n):
        if n.kind != "hw.frameparam":
            return None
        name = n.attr("name")
        if name not in made:
            img = unit.facts["image"]
            port = _ctx_field(g, img, lambda f, _: f == name)
            if port is None:
                raise Unsupported(
                    f"{name} is read from each frame, and the stream's "
                    "context carries no such field")
            held = reg_symbol(f"frame_{name}", port.type.lo, port.type.hi, 0)
            unit.regs.append(g.node("hw.reg", [port, load],
                                    {"name": f"frame_{name}"}))
            made[name] = SEL(load, port, held)
        return made[name]
    return substitute


def _either(*fns):
    """The first of several substitutions that maps a node."""
    def substitute(n):
        for fn in fns:
            new = fn(n)
            if new is not None:
                return new
        return None
    return substitute


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
    # The frame as it arrives: its size is the frame registers', loaded as
    # each frame enters, never wider than the traced width a line buffer
    # holds. The steps' last row and column follow it: the output's frame
    # grown by the lag, or the input's own where a crop stops short.
    held, arriving, latches = _frame_size(g, reg_symbol, img)
    fw, fh = held["cols"], held["rows"]
    # Frame constants a step is compared against, worked out once per
    # frame into registers -- loaded with the frame's size, from the size
    # arriving -- and compared against, not added up again beneath every
    # step's verdict: the steps' last column and row, and the frame's last
    # column.
    constants = []

    def per_frame(name, axis, of, traced):
        node = of(arriving[axis])
        constants.append((name, node))
        return reg_symbol(name, node.type.lo, node.type.hi, traced)
    last_c = per_frame("last_c", "cols",
                       lambda w: g.op("hw.add", w, c(ws - W - 1)), ws - 1)
    last_r = per_frame("last_r", "rows",
                       lambda h: g.op("hw.add", h, c(hs - H - 1)), hs - 1)
    final_c = per_frame("final_c", "cols", lambda w: g.op("hw.sub", w, c(1)),
                        W - 1)
    final_r = per_frame("final_r", "rows", lambda h: g.op("hw.sub", h, c(1)),
                        H - 1)
    # The output's extent, one fact per axis, which both its flags and the
    # size it hands on are made from: a crop that stops short of the frame
    # has the slice's own count; otherwise the frame less the border an
    # unpadded window drops. Its last step is the lag plus that extent, one
    # less -- for a relative extent, the steps' own last.
    # How each output axis ends is the model's statement (its slices'
    # stops): after a fixed count, or the frame less the border its reads
    # drop -- `x[2:10]` against `x[1:-1]`, one graph at the traced size and
    # two answers at any other.
    ends = dict(zip(("rows", "cols"), st["ends"]))
    extent = {"rows": ("fixed", ho) if ends["rows"] == "fixed"
              else ("less", H - ho),
              "cols": ("fixed", wo) if ends["cols"] == "fixed"
              else ("less", W - wo)}
    # the size handed on is the arriving frame's, loaded as it is taken; the
    # output's last row and column, where the steps' own last is not it --
    # a fixed extent, or reads that stop short of the frame's end
    tail = {"rows": tr > 0, "cols": tc > 0}
    handed, out_end = {}, {}
    for axis, lag, last in (("rows", lr, last_r), ("cols", lc, last_c)):
        kind, k = extent[axis]
        if kind == "fixed":
            handed[axis], out_end[axis] = c(k), c(lag + k - 1)
            continue
        # passed on as it arrives, or worked out -- from the latched size,
        # a cycle on (_settled_load)
        handed[axis] = (arriving[axis] if k == 0 else g.op(
            "hw.clip", g.op("hw.sub", held[axis], c(k)), lo=0,
            hi=(1 << POS) - 1))
        out_end[axis] = (per_frame(f"out_end_{axis[0]}", axis,
                                   lambda n, k=k, lag=lag: g.op(
                                       "hw.add", n, c(lag - k - 1)),
                                   lag + (H if axis == "rows" else W) - k - 1)
                         if tail[axis] else last)
    unit.facts["extent"] = extent

    # rows step to the geometry's width: a window keeps lines, not frames,
    # so the frame's height bounds nothing it builds
    cy = reg_symbol("step_r", 0, (1 << POS) - 1, 0)
    cx = reg_symbol("step_c", 0, ws - 1, 0)
    # Whether the step lies inside the input frame, per axis: registers,
    # worked out from the steps' next values as they advance, so every
    # control path starts at a flop rather than behind a compare of the
    # step with the frame's size. Reset to the first step's: inside.
    in_r = reg_symbol("inside_r", 0, 1, 1)
    in_c = reg_symbol("inside_c", 0, 1, 1)
    wants = AND(in_r, in_c)
    anchor = AND(AND(in_valid, in_sof), wants)
    ey, ex = SEL(anchor, c(0), cy), SEL(anchor, c(0), cx)
    valid = reg_symbol("out_valid", 0, 1, 0)
    ready = NOT(AND(valid, NOT(out_ready)))
    take = SEL(wants, in_valid, c(1))            # a pixel, or a flush step
    go = AND(ready, take)
    # At the anchor the step is (0, 0), and everything worked out from the
    # step there is a constant: a frame is at least 2x2, so the step ends no
    # line, the next is column 1, and that column lies inside the frame. So
    # the steps' next values are worked out from the counters alone and the
    # anchor picks the constant at the end -- in parallel, not in series,
    # with what it took to decide the anchor.
    end_here = EQ(cx, last_c)
    line_end = AND(NOT(anchor), end_here)
    # past each counter's last step the select takes 0, so the clip changes
    # no value
    nx_on = SEL(end_here, c(0), clip(one(cx), ws - 1))
    ny_on = SEL(end_here, SEL(EQ(cy, last_r), c(0),
                              clip(one(cy), (1 << POS) - 1)), cy)
    nx = SEL(anchor, c(1), nx_on)
    ny = SEL(anchor, c(0), ny_on)
    # Inside, at the next step -- LT(ny, fh) and LT(nx, fw), each read off
    # what the step does, so every compare is of a counter with a frame
    # constant, side by side: after the anchor the next step is (0, 1),
    # inside any frame of at least 2x2; a step that ends no line keeps its
    # row; a line's end wraps the column to 0, inside, and the row to 0 or
    # to the next, inside while the row is before the frame's last; and
    # otherwise the next column is inside while this one is before the
    # frame's last.
    next_r = SEL(end_here, SEL(EQ(cy, last_r), c(1), LT(cy, final_r)), in_r)
    next_c = SEL(end_here, c(1), LT(cx, final_c))
    unit.regs.append(g.node("hw.reg", [SEL(anchor, c(1), next_r), go],
                            {"name": "inside_r"}))
    unit.regs.append(g.node("hw.reg", [SEL(anchor, c(1), next_c), go],
                            {"name": "inside_c"}))
    unit.regs.append(g.node("hw.reg", [nx, go], {"name": "step_c"}))
    unit.regs.append(g.node("hw.reg", [ny, go], {"name": "step_r"}))

    emit = AND(GE(ey, c(lr)), GE(ex, c(lc)))
    for axis, at in (("rows", ey), ("cols", ex)):
        if extent[axis][0] == "fixed" or tail[axis]:
            emit = AND(emit, g.op("hw.le", at, out_end[axis]))
    out_eol = EQ(ex, out_end["cols"])
    flags = {"valid": emit,
             "sof": AND(emit, AND(EQ(ey, c(lr)), EQ(ex, c(lc)))),
             "eol": AND(emit, out_eol),
             "last": AND(emit, AND(EQ(ey, out_end["rows"]), out_eol))}

    taps, mems = {}, []
    aw = max(1, (W - 1).bit_length())
    # the column the next step reads: a column ahead, the next line's first
    # after the last -- column 1 after the anchor; otherwise the next step's
    # column while it lies inside the frame (the step before the frame's
    # last column), the line's first once past it
    ahead = SEL(anchor, c(1),
                SEL(end_here, c(0), SEL(LT(cx, final_c),
                                        clip(one(cx), W - 1), c(0))))
    # the step inside the frame's rows and columns -- LT(ey, fh) and
    # LT(ex, fw), which at the anchor, (0, 0), hold
    inside_r = g.op("hw.or", anchor, in_r)
    inside_c = g.op("hw.or", anchor, in_c)
    # past the frame's first k rows / columns: GE(ey, k) and GE(ex, k),
    # which at the anchor, (0, 0), fail for every k > 0
    past = lambda at, k: AND(NOT(anchor), GE(at, c(k)))
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
        cur = val if kr == 1 else SEL(inside_r, val,
                                      c(0) if zero else reads[0])
        rows_v = [cur]
        for a in range(1, kr):
            # a row before the frame's first reads the first
            rows_v.append(SEL(past(cy, a), reads[a - 1],
                              c(0) if zero else rows_v[a - 1]))
        for a in range(1, kr):
            name = f"lb{j}_{a}"
            unit.roots.update({
                f"mem_{name}_we": AND(go, inside_c),
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
            col = rows_v[a] if kc == 1 else SEL(inside_c, rows_v[a],
                                                 c(0) if zero else regs[0])
            for b, n in enumerate(names, start=1):
                unit.regs.append(g.node(
                    "hw.reg", [col if b == 1 else regs[b - 2], go],
                    {"name": n}))
            cols_v = [col]
            for b in range(1, kc):
                # a column before the line's first reads the first
                cols_v.append(SEL(past(cx, b), regs[b - 1],
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

    load = _size_load(g, unit, reg_symbol, AND(go, anchor))
    roots = _substitute(g, unit.roots, _either(
        substitute, _frame_values(g, unit, reg_symbol, SEL, load)))
    worked = any(kind == "less" and k for kind, k in extent.values())
    sizes = _handed(g, unit, reg_symbol, handed,
                    _settled_load(g, unit, reg_symbol, load) if worked
                    else load, {"rows": H, "cols": W})
    unit.roots = {**roots, **{f"flag_{k}": v for k, v in flags.items()},
                  **sizes}
    unit.roots["out_ctx"] = _out_ctx(g, unit, reg_symbol, sizes, load)
    unit.regs.extend(latches(load))
    unit.regs.extend(g.node("hw.reg", [node, load], {"name": n})
                     for n, node in constants)
    unit.facts.update(go=go, ready=ready, in_valid=in_valid, take=take,
                      in_ready=AND(ready, wants), memories=mems,
                      out_size=(ho, wo),
                      # cycles past a frame's last pixel taken until its
                      # output is out: the flush steps left, and a margin
                      drain=(lr + 1) * ws + 8)
    return unit


def _frame_size(g, reg_symbol, img):
    """The frame's width and height: registers, latched from the context's
    size fields as a frame's first pixel is taken, and reset to the traced
    size.

    The latch is a frame's first beat late for that beat -- and need not be
    early: a frame is at least two pixels wide and two lines high, and at a
    frame's first pixel nothing a core decides depends on the size then.
    It is inside the frame, it ends no line, a window wants it and reads a
    column ahead that exists. So the datapath compares against flops, with
    nothing in front of them, and a size taken from the context reaches
    only the latch. The fields are clipped to the contract, which also
    keeps `size - 1` unsigned.
    Returns ({"cols", "rows"}: the latched size, {...}: the arriving size,
    latches(taken)), `latches` the two registers, enabled by the first
    pixel being taken."""
    held, arriving = {}, {}
    for axis, port, traced in (("cols", "frame_width", img.width),
                               ("rows", "frame_height", img.height)):
        # the context's field that follows this axis, at least two as the
        # contract has it -- or, where none does, the traced size
        field = _ctx_field(g, img, lambda _, f, a=port[6:]: f == a)
        arriving[axis] = (
            g.op("hw.clip", field, lo=2, hi=(1 << POS) - 1)
            if field is not None else g.op("hw.const", value=traced))
        held[axis] = reg_symbol(f"{port}_held", 2, (1 << POS) - 1,
                                max(2, traced))

    def latches(taken):
        return [g.node("hw.reg", [arriving[axis], taken],
                       {"name": f"{port}_held"})
                for axis, port in (("cols", "frame_width"),
                                   ("rows", "frame_height"))]
    return held, arriving, latches


def _size_load(g, unit, reg_symbol, taken):
    """When the registers made from the frame's size load: as a frame's
    first pixel is taken -- and, until the core has taken one, every
    cycle, so a core out of reset holds the size its context says is coming
    rather than the most it can hold, and hands that on."""
    started = reg_symbol("started", 0, 1, 0)
    unit.regs.append(g.node("hw.reg", [g.op("hw.const", value=1), taken],
                            {"name": "started"}))
    return g.op("hw.or", taken, g.op("hw.not", started))


def _settled_load(g, unit, reg_symbol, load):
    """The size registers' load a cycle late, for a size handed on that is
    worked out rather than passed: worked out from the latched size the
    cycle after the frame's first beat -- its own stage, not a chain behind
    the context's arrival in the same cycle as the latch -- and loaded
    then. The core then keeps at least two stages, so its first beat out,
    when the next core reads the context, leaves on the same edge the size
    lands on, never before."""
    name = "size_load_late"
    reg_symbol(name, 0, 1, 0)
    unit.regs.append(g.node("hw.reg", [load, g.op("hw.const", value=1)],
                            {"name": name}))
    unit.facts["min_stages"] = 2
    return g.op("hw.regread", name=name)


def _handed(g, unit, reg_symbol, handed: dict, starts, traced: dict) -> dict:
    """The size a core hands on, as two registers loaded as a frame's first
    beat is taken (`starts`, _size_load) -- from the size arriving with it,
    so they are current from the next cycle, when the next core can first
    take that frame -- and so they leave a flop: a size taken straight from
    the context would run one combinational path through every core of a
    pipeline. Their reset value is the size handed on at the traced frame
    (0 for a window placed by registers), which the context replaces a
    cycle out of reset. They are the size fields of `out_ctx`."""
    out = {}
    for axis, port in (("cols", "out_frame_width"), ("rows", "out_frame_height")):
        kind, *rest = unit.facts["extent"][axis]
        init = (traced[axis] - rest[0] if kind == "less" else
                rest[0] if kind == "fixed" else 0)
        out[port] = reg_symbol(port, 0, (1 << POS) - 1, init)
        unit.regs.append(g.node("hw.reg", [handed[axis], starts],
                                {"name": port}))
    return out


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
        c = lambda v: g.op("hw.const", value=v)
        in_valid, in_sof = port("in_valid"), port("in_sof")
        folds = unit.facts.get("folds")
        if folds is None:
            out_ready = port("out_ready")
        else:
            # its fold takes every beat, except while the frame before is
            # streamed out -- and, for a memory, cleared after reset
            # (accumulate)
            g.symbols["busy"] = {"lo": 0, "hi": 1,
                                 "init": 1 if folds["bins"] > 1 else 0}
            out_ready = g.op("hw.not", g.op("hw.regread", name="busy"))
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
        # the frame's size, latched (_frame_size): `limit - 1` is never
        # negative, so comparing a counter with it is unsigned, in the
        # counter's own width
        held, arriving, latches = _frame_size(g, reg_symbol,
                                              unit.facts["image"])
        limit = {cols: held["cols"], rows: held["rows"]}
        incoming = {cols: arriving["cols"], rows: arriving["rows"]}
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
        # The output's extent, per axis, and the size handed on made from
        # it: the frame itself, or the window's part of it -- a window
        # running past the frame's edge ends there, one starting past it
        # holds nothing.
        if "window" in unit.facts:
            y0, x0, h, w = unit.facts["window"]
            unit.facts["extent"] = {"rows": ("window", y0, h),
                                    "cols": ("window", x0, w)}

            def kept(frame_n, at, n):
                room = g.op("hw.sub", frame_n, at)
                held = SEL(g.op("hw.lt", room, n), room, n)
                return g.op("hw.clip", SEL(g.op("hw.ge", at, frame_n),
                                           c(0), held),
                            lo=0, hi=(1 << POS) - 1)
            # worked out from the latched size, a cycle on (_settled_load)
            handed = {"rows": kept(limit[rows], y0, h),
                      "cols": kept(limit[cols], x0, w)}
        else:
            unit.facts["extent"] = {"rows": ("less", 0), "cols": ("less", 0)}
            handed = {"rows": incoming[rows], "cols": incoming[cols]}
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
                # Its depth is the traced frame's, the most it holds; a
                # frame's positions run from its first pixel to its last,
                # so the address wraps where the frame ends.
                img = unit.facts["image"]
                size = img.width * img.height
                aw = max(1, (size - 1).bit_length())
                addr = reg_symbol("hist_addr", 0, size - 1, 0)
                here = SEL(in_sof, c(0), addr)
                # past the last position the select takes 0, so the clip
                # changes no value
                ahead = SEL(AND(final(rows), final(cols)), c(0),
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

        load = _size_load(g, unit, reg_symbol, AND(go, in_sof))
        roots = _substitute(g, unit.roots, _either(
            position, _frame_values(g, unit, reg_symbol, SEL, load)))
        img = unit.facts["image"]
        sizes = _handed(g, unit, reg_symbol, handed,
                        _settled_load(g, unit, reg_symbol, load)
                        if "window" in unit.facts else load,
                        {"rows": img.height, "cols": img.width})
        unit.roots = {**roots, **{f"flag_{k}": v for k, v in flags.items()},
                      **sizes}
        unit.roots["out_ctx"] = _out_ctx(g, unit, reg_symbol, sizes, load)
        unit.regs.extend(latches(load))
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
_FLOATING = ("hw.const", "hw.param")


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
        least = unit.facts.get("min_stages", 1)
        unit.facts["stages"] = least
        unit.facts.setdefault("stage_of", {})
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
        unit.facts["stages"] = max(
            least, max(stage.get(id(v.node), 0) for v in outs) + 1)
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
        # a memory's ports, wired to its instance by the printer; and the
        # context the module hands on, the size its fields -- a fact of the
        # frame made from the frame registers, not of a pixel, so no stage
        # carries it
        outputs.update({k: v for k, v in unit.roots.items()
                        if k.startswith("mem_") or k == "out_ctx"})
        unit.roots = outputs
        f.update(out_bits=data.type.bits, out_signed=data.type.lo < 0)
        return unit

    return Pass("materialise", run_, levels={"hw"})


def accumulate() -> Pass:
    """A fold: its result streamed out, a beat an entry, once its frame is
    folded.

    Each fold folds into an ACCUMULATOR memory: a read, the value folded
    in, a write -- two stages, the write of the beat before forwarded to a
    read it lands too late for. At a frame's last beat a SWEEP reads every
    entry in order and writes the fold's start back to it in the same
    cycle (a read beside a write of its own address reads the old word):
    each read is a beat of the result, held in the memory's read register
    until it is taken, and the accumulator is clear when the last is. The
    sweep after reset only clears. A one-entry fold accumulates in a
    register and streams one beat a frame.

    The result stream's frame is the result's: its rows and columns (a
    histogram one row), every fold's value of an entry side by side in a
    beat; its context the folded frame's, with the result's size. A frame
    arriving while the last is streamed out -- or while the memory is
    cleared after reset -- waits, as any stream does for a core that is
    not ready."""

    def run_(unit: Unit) -> Unit:
        f = unit.facts
        folds = f.get("folds")
        if folds is None:
            return unit
        g = unit.g
        c = lambda v: g.op("hw.const", value=v)
        AND = lambda x, y: g.op("hw.and", x, y)
        OR = lambda x, y: g.op("hw.or", x, y)
        NOT = lambda x: g.op("hw.not", x)
        EQ = lambda x, y: g.op("hw.eq", x, y)
        SEL = lambda s_, x, y: g.op("hw.sel", s_, x, y)
        bits = lambda x, lo, w: g.op("hw.bits", x, lo=lo, width=w)
        port = lambda name: g.op("hw.port", name=name, bits=1, lo=0, hi=1)

        def declare(name, lo, hi, init):
            g.symbols[name] = {"lo": lo, "hi": hi, "init": init}
            return g.op("hw.regread", name=name)

        def reg(name, nxt, en, lo, hi, init):
            out = declare(name, lo, hi, init)
            unit.regs.append(g.node("hw.reg", [nxt, en], {"name": name}))
            return out

        def load(name, nxt, en):
            unit.regs.append(g.node("hw.reg", [nxt, en], {"name": name}))

        def read(name):
            return g.op("hw.regread", name=name)

        lanes, bins = folds["lanes"], folds["bins"]
        shape = [n for n in folds["shape"]]
        if len(lanes) > 1 or (folds["stacked"]):
            shape = shape[:-1]                  # the values side by side
        cols = shape[-1] if shape else 1
        rows = bins // cols
        word = read("out_data")
        beat = read("out_valid")
        last = read("out_last")
        first = read("out_sof")
        busy = read("busy")
        memory = bins > 1
        fold = AND(beat, NOT(busy))             # taken, as control gave it
        out_ready = port("out_ready")
        # each lane's fields of the word, as _fold_unit laid them
        at, fields = 0, []
        for lane in lanes:
            key = None
            if memory:
                key = bits(word, at, lane["kbits"])
                at += lane["kbits"]
            v = bits(word, at, lane["vbits"])
            at += lane["vbits"]
            fields.append((key, g.op("hw.add", v, c(lane["vlo"]))
                           if lane["vlo"] else v))

        def folded(lane, base, v):
            if lane["op"] == "add":
                return g.op("hw.add", base, v)
            cmp = "hw.gt" if lane["op"] == "max" else "hw.lt"
            return SEL(g.op(cmp, v, base), v, base)

        def stored(lane):
            return max(1, (lane["hi"] - lane["lo"]).bit_length())

        def kept(lane, exact):
            """`exact` in the fold's word: less the least it can be. Where a
            step can leave the accumulator's container, NumPy's `at` wraps
            it back in -- and the entry's range is then the container's
            own, so this, its low bits above the container's least, IS
            that wrap."""
            return g.op("hw.wrap", g.op("hw.sub", exact, c(lane["lo"])),
                        bits=stored(lane), signed=False)

        roots = {k: v for k, v in unit.roots.items()
                 if k == "in_ready" or k.startswith("mem_")}
        mems = f.setdefault("memories", [])
        aw = max(1, (bins - 1).bit_length())
        values = []                     # each lane's value in the out beat
        taken = None                    # the result's beat taken
        if not memory:
            # the frame's value, and whether it waits to be taken
            done = AND(fold, last)
            taken = AND(busy, out_ready)
            load("busy", done, OR(done, taken))
            for j, (lane, (_, v)) in enumerate(zip(lanes, fields)):
                sb = stored(lane)
                acc = declare(f"acc{j}", 0, (1 << sb) - 1, 0)
                base = SEL(first, c(lane["init"]),
                           g.op("hw.add", acc, c(lane["lo"])))
                new = kept(lane, folded(lane, base, v))
                load(f"acc{j}", new, fold)
                held = reg(f"res{j}", new, done, 0, (1 << sb) - 1, None)
                values.append(g.op("hw.add", held, c(lane["lo"])))
            out_valid = busy
            flags = {"sof": c(1), "eol": c(1), "last": c(1)}
        else:
            b_valid = reg("b_valid", fold, c(1), 0, 1, 0)
            b_last = reg("b_last", last, c(1), 0, 1, 0)
            w_valid = reg("w_valid", b_valid, c(1), 0, 1, 0)
            landing = AND(b_valid, b_last)
            armed = declare("armed", 0, 1, 0)   # the reset's sweep is done
            # The sweep, once the frame's last write has landed: each
            # entry read -- a beat of the result, held in the memory's read
            # register until taken -- and its start written back at once.
            sw = declare("sweep", 0, bins, 0)
            sc = declare("sweep_c", 0, cols - 1, 0)
            held = declare("held", 0, 1, 0)     # a beat in the read register
            room = OR(NOT(armed), OR(NOT(held), out_ready))
            reading = AND(AND(busy, NOT(landing)),
                          AND(g.op("hw.lt", sw, c(bins)), room))
            load("sweep", SEL(busy, g.op("hw.clip", g.op(
                "hw.add", sw, c(1)), lo=0, hi=bins), c(0)),
                OR(reading, NOT(busy)))
            load("sweep_c", SEL(OR(NOT(busy), EQ(sc, c(cols - 1))), c(0),
                                g.op("hw.add", sc, c(1))),
                 OR(reading, NOT(busy)))
            taken = AND(held, out_ready)
            load("held", AND(reading, armed), OR(reading, taken))
            oe = reg("out_entry", g.op("hw.wrap", sw, bits=aw, signed=False),
                     reading, 0, (1 << aw) - 1, None)
            oc = reg("out_col", sc, reading, 0, cols - 1, None)
            # done: the last beat taken -- or, after reset, the last entry
            # cleared
            done = OR(AND(taken, EQ(oe, c(bins - 1))),
                      AND(AND(reading, NOT(armed)), EQ(sw, c(bins - 1))))
            load("armed", c(1), done)
            load("busy", SEL(AND(fold, last), c(1), c(0)),
                 OR(AND(fold, last), done))
            for j, (lane, (key, v)) in enumerate(zip(lanes, fields)):
                sb = stored(lane)
                name = f"acc{j}"
                rd = g.op("hw.memread", name=name, lo=0, hi=(1 << sb) - 1)
                top = (1 << lane["kbits"]) - 1
                b_key = reg(f"b_key{j}", key, c(1), 0, top, None)
                b_v = reg(f"b_v{j}", v, c(1), v.type.lo, v.type.hi, None)
                w_key = declare(f"w_key{j}", 0, top, None)
                w_s = declare(f"w_s{j}", 0, (1 << sb) - 1, None)
                fresh = AND(w_valid, EQ(w_key, b_key))
                base_s = SEL(fresh, w_s, rd)
                new = kept(lane, folded(
                    lane, g.op("hw.add", base_s, c(lane["lo"])), b_v))
                load(f"w_key{j}", b_key, c(1))
                load(f"w_s{j}", new, c(1))
                at_sw = g.op("hw.wrap", sw, bits=aw, signed=False)
                roots.update({
                    f"mem_{name}_we": OR(b_valid, reading),
                    f"mem_{name}_wa": SEL(reading, at_sw, b_key),
                    f"mem_{name}_wd": SEL(reading,
                                          c(lane["init"] - lane["lo"]), new),
                    f"mem_{name}_re": OR(fold, reading),
                    f"mem_{name}_ra": SEL(reading, at_sw, key)})
                mems.append({"instance": f"u_{name}", "name": name,
                             "bits": sb, "depth": bins, "aw": aw,
                             "read": "sync", "seam": "acc"})
                values.append(g.op("hw.add", rd, c(lane["lo"])))
            out_valid = held
            eol = EQ(oc, c(cols - 1))
            flags = {"sof": EQ(oe, c(0)), "eol": eol,
                     "last": EQ(oe, c(bins - 1))}
        # every fold's value of the entry side by side, as a stack's
        # channels are: one width, unsigned where there are several
        if len(values) > 1 and any(v.type.lo < 0 for v in values):
            raise Unsupported("folds side by side, one of them negative: a "
                              "stream's channels are unsigned")
        field = max(v.type.bits for v in values)
        data = values[0] if len(values) == 1 else g.op(
            "hw.cat", *values, field=field)
        # the context the result carries: the folded frame's, as it was at
        # its first beat, and the result's size
        img = f["image"]
        ctx_now = unit.roots["out_ctx"]
        cbits = img.context_bits
        kept_ctx = reg("res_ctx", ctx_now, AND(fold, first), 0,
                       (1 << cbits) - 1, 0)
        parts, lo_ = [], 0
        for name, b, follows in img.context:
            parts.append(c(cols) if follows == "width" else
                         c(rows) if follows == "height" else
                         bits(kept_ctx, lo_, b))
            lo_ += b
        roots.update({"out_valid": out_valid, "out_data": data,
                      "out_ctx": g.op("hw.cat", *parts, fields=tuple(
                          b for _, b, _ in img.context)),
                      **{f"out_{k}": AND(out_valid, v)
                         for k, v in flags.items()}})
        unit.roots = roots
        f.update(out_bits=data.type.bits, out_signed=data.type.lo < 0,
                 out_size=(rows, cols),
                 result={"shape": folds["shape"], "rows": rows,
                         "cols": cols})
        if len(values) > 1:
            f.update(out_channels=len(values), out_field=field)
        return unit

    return Pass("accumulate", run_, levels={"hw"})


def lower(c: Canonical, dev=None, clk_ns=None) -> Unit:
    """The canonical graph, lowered to a hardware module for the device
    `dev` (the default device when None), its stages fitting `clk_ns` when
    one is given -- or Unsupported."""
    unit = Unit(c.g, {"out": c.output})
    try:
        return run(unit, [rewrite(c, dev), control(), schedule(clk_ns, dev),
                          materialise(), accumulate()])
    except GraphError as e:
        raise Unsupported(f"lowering: {e}") from e
