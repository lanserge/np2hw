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
        # a video's output frames: the output's own variable on time, which
        # every read across time indexes
        tout = out_axes.pop(0) if tsrc is not None and out_axes else None
        if tsrc is not None and tout is None:
            raise Unsupported("a video's output with no frame axis")
        if tsrc is not None and win is not None:
            raise Unsupported("a window over a video")
        # a stack of the frame's values along a last axis: channels of one
        # word, as np2hw packs every multi-channel stream
        lanes = (c.output.node.operands
                 if c.output.node.kind == "model.stack"
                 and out_axes == driving + [find(c.output.node.attr("var"))]
                 else None)
        if win is not None:
            if win.operands[0].node is not src.node:
                raise Unsupported("a window over a computed value: window "
                                  "the source, and compute in the next block")
        elif lanes is None and out_axes != driving:
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
                return field_of(n)
            if k == "model.var":
                v = find(n.attr("name"))
                if v not in driving and v != tout:
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
            if index.type.lo < 0:
                raise Unsupported(f"{name}: an index that can be negative "
                                  f"([{index.type.lo}, {index.type.hi}])")
            if isinstance(obj, Param):              # a register array
                size = obj.shape[0]

                def entry(i):
                    leaf = obj[i]
                    params.setdefault(leaf.name, ((name, i), (
                        leaf.name, leaf.bits, bool(leaf.signed),
                        leaf.default, leaf.description)))
                    return g.op("hw.param", name=leaf.name, bits=leaf.bits,
                                signed=bool(leaf.signed))
            else:                                   # a constant table
                data = [int(v) for v in obj.data]
                size = len(data)
                if not fam.rom_in_logic(size):
                    raise Unsupported(
                        f"{name}: a {size}-entry constant table is a memory on "
                        f"{fam.name} -- a read through a register, a stage "
                        "the pointwise family does not schedule")

                def entry(i):
                    return g.op("hw.const", value=data[i])
            top = min(index.type.hi, size - 1)
            if index.type.hi > size - 1:
                raise Unsupported(f"{name}: an index reaching "
                                  f"{index.type.hi}, past its {size} entries")
            width = max(1, top.bit_length())

            def tree(base, bit):
                """Entries base .. base + 2**(bit+1) - 1 by the index's
                bits bit .. 0; past the reachable top, nothing is built."""
                if bit < 0:
                    return entry(base)
                upper = base + (1 << bit)
                low = tree(base, bit - 1)
                if upper > top:
                    return low
                return g.op("hw.sel", g.op("hw.bits", index, lo=bit, width=1),
                            tree(upper, bit - 1), low)

            return tree(0, width - 1)

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
                    stores * word, "depth": size, "aw": aw, "read": "sync"}]

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

def schedule() -> Pass:
    """Every node a stage. One stage for the pointwise family: the datapath
    and the flags computed in the beat, registered at the output. The
    budget check against a clock is np2hw.core.emit's, at generation."""

    def run_(unit: Unit) -> Unit:
        unit.facts["stages"] = 1
        return unit

    return Pass("schedule", run_, levels={"hw"})


def materialise() -> Pass:
    """Every value crossing the output stage becomes a register: the data --
    no reset, it means something only while out_valid says so -- and the
    flags, reset."""

    def run_(unit: Unit) -> Unit:
        g, f = unit.g, unit.facts
        data = unit.roots["data"]
        g.symbols["out_data"] = {"lo": data.type.lo, "hi": data.type.hi,
                                 "init": None}
        unit.regs.append(g.node("hw.reg", [data, f["go"]],
                                {"name": "out_data"}))
        outputs = {"in_ready": f["ready"]}
        for flag in ("valid", "sof", "eol", "last"):
            name = f"out_{flag}"
            g.symbols.setdefault(name, {"lo": 0, "hi": 1, "init": 0})
            unit.regs.append(g.node(
                "hw.reg", [g.op("hw.and", f["in_valid"],
                                unit.roots[f"flag_{flag}"]), f["ready"]],
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


def lower(c: Canonical, dev=None) -> Unit:
    """The canonical graph, lowered to a hardware module for the device
    `dev` (the default device when None) -- or Unsupported."""
    unit = Unit(c.g, {"out": c.output})
    try:
        return run(unit, [rewrite(c, dev), control(), schedule(),
                          materialise()])
    except GraphError as e:
        raise Unsupported(f"lowering: {e}") from e
