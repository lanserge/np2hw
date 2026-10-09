# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A loaded design as Verilog: one module per composite.

Every composite is written as it is described -- its children instanced
by name, its connections as wires -- from the hierarchy the loader kept
(``Design.hierarchy``) and the ports the netlist resolved for each leaf.
Nothing here is decided: a leaf is instanced by the module its owner
names, a vendor leaf as a black box by its cell's name and its
implementation's pin names (the toolchain makes that module), and a
composite's own ports are what they meet inside -- a signal of the width
it joins, or a bundle of the kind and side it joins, its signals named
``<port>_<signal>``. External pins are the FPGA's, so they rise through
every composite to the top.

    from np2hw.design.verilog import modules
    for name, text in modules(design): ...
"""
from __future__ import annotations

from np2hw.design.composite import CompositionError


class VerilogError(CompositionError):
    """A design that cannot be written as Verilog as it stands."""


def _leaf(design, path) -> dict:
    """What a leaf is instanced as: ``{"module", "parameters", "signals",
    "bundles", "pin", "tie"}`` -- ``pin`` maps a signal of the block's
    ports to the module's own pin name (a vendor implementation's)."""
    from np2hw.toolchain.vivado.vendor_check import pin_name
    inst = design.instances[path]
    net = design.netlist["instances"][path]
    leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
    if net["ports"] is None:
        raise VerilogError(f"{path} ({inst.ref}) states no ports; a design "
                           "is written as Verilog from every leaf's own "
                           "statement of them")
    cell = path.rpartition(".")[2]
    tool = leaf.get("vivado", {})
    if "ip" in tool or "tcl" in tool:
        # a black box: the toolchain makes a module by the cell's name,
        # with the implementation's pins, from the implementation's settings
        module, parameters = cell, {}
        pins = {signal: pin_name(leaf, signal) for signal in net["ports"]}
        tie = dict(leaf.get("tie", {}))
    elif "module" in leaf.get("vivado", {}):
        module = leaf["vivado"]["module"]
        parameters = dict(net["parameters"])
        pins = {signal: signal for signal in net["ports"]}
        tie = {}
    else:
        raise VerilogError(f"{path} ({inst.ref}) names no module and no "
                           "vendor implementation to instance")
    return {"module": module, "parameters": parameters,
            "signals": {s: tuple(v) for s, v in net["ports"].items()},
            "bundles": dict(net["bundles"] or {}), "pin": pins, "tie": tie}


def _role(bundle: dict, member: str) -> str:
    """A member's part in its bus, as the netlist pairs it: two bundles
    meet member to member by this."""
    return bundle["roles"][member]


class _Writer:
    def __init__(self, design, groups=None):
        import copy
        self.design = design
        self.h = copy.deepcopy(design.hierarchy)
        self.groups = {name: list(paths) for name, paths
                       in (groups or {}).items()}
        self.member = {p: name for name, paths in self.groups.items()
                       for p in paths}
        for name, paths in self.groups.items():
            parents = {p.rpartition(".")[0] for p in paths}
            if len(parents) != 1:
                raise VerilogError(f"group {name}: {sorted(paths)} are not "
                                   "in one composite; a group is one cell "
                                   "of the composite that holds them")
        self.leaves = {p: _leaf(design, p) for p in design.instances
                       if p not in self.member}
        self.group_ports: dict[str, dict] = {}   # group -> {port: record}
        self.ports: dict[str, dict] = {}     # composite -> its port specs
        self.texts: dict[str, list] = {}     # composite -> body lines
        self.names: dict[str, str] = {}      # composite -> module name

    # -- a group of leaves, one cell -------------------------------------- #

    def member_spec(self, path, port):
        """A group member's own statement of a port, or None."""
        net = self.design.netlist["instances"][path]
        if net["ports"] is None:
            return None
        bundles = net["bundles"] or {}
        if port in bundles:
            return ("bundle", bundles[port]["side"],
                    {_role(bundles[port], m): tuple(net["ports"][m])
                     for m in bundles[port]["members"]})
        if port in net["ports"]:
            return ("signal",) + tuple(net["ports"][port])
        return None

    @staticmethod
    def mirror(spec):
        flip = {"input": "output", "output": "input"}
        if spec[0] == "signal":
            return ("signal", flip.get(spec[1], spec[1]), spec[2])
        side = {"inputs": "outputs", "outputs": "inputs"}.get(spec[1],
                                                               spec[1])
        return ("bundle", side, {sfx: (flip.get(d, d), w)
                                 for sfx, (d, w) in spec[2].items()})

    def group(self, path):
        """Fold every group held by the composite at `path` into one child:
        its members' connections to the rest become the group's ports,
        named <cell>_<port>, as the member states the port or, where it
        states none, as the far end's mirror."""
        node = self.h[path]
        for name, paths in self.groups.items():
            if paths[0].rpartition(".")[0] != path:
                continue
            members = set(paths)
            ports, kept = {}, []
            for a, b in node["connections"]:
                ina, inb = a[0] in members, b[0] in members
                if ina and inb:
                    continue            # inside the group: the tool's
                if not (ina or inb):
                    kept.append((a, b))
                    continue
                mine, other = (a, b) if ina else (b, a)
                port = f"{mine[0].rpartition('.')[2]}_{mine[1]}"
                if port not in ports:
                    spec = self.member_spec(*mine)
                    if spec is None:
                        if other[0] == path:
                            raise VerilogError(
                                f"{mine[0]}.{mine[1]} meets {other[1]} and "
                                f"states nothing of the port")
                        spec = self.mirror(self.spec(*other))
                    ports[port] = {"spec": spec, "member": mine}
                key = f"@{name}"
                renamed = (key, port)
                kept.append((renamed, b) if ina else (a, renamed))
            self.group_ports[name] = ports
            signals, bundles = {}, {}
            for port, rec in ports.items():
                spec = rec["spec"]
                if spec[0] == "signal":
                    signals[port] = (spec[1], spec[2])
                else:
                    roles = {}
                    for sfx, (d, w) in spec[2].items():
                        signals[f"{port}_{sfx}"] = (d, w)
                        roles[f"{port}_{sfx}"] = sfx
                    bundles[port] = {"side": spec[1], "members": list(roles),
                                     "roles": roles}
            self.leaves[f"@{name}"] = {
                "module": name, "parameters": {}, "signals": signals,
                "bundles": bundles, "pin": {s: s for s in signals},
                "tie": {}}
            node["connections"] = kept
            node["children"] = {local: child for local, child
                                in node["children"].items()
                                if child not in members}
            node["children"][name] = f"@{name}"

    # -- what a pin is ---------------------------------------------------- #

    def spec(self, owner: str, port: str):
        """A child's port as ``("signal", direction, width)`` or
        ``("bundle", side, {suffix: (direction, width)})``."""
        if owner in self.leaves:
            leaf = self.leaves[owner]
            if port in leaf["bundles"]:
                b = leaf["bundles"][port]
                return ("bundle", b["side"],
                        {_role(b, m): leaf["signals"][m]
                         for m in b["members"]})
            if port not in leaf["signals"]:
                raise VerilogError(f"{owner} has no port {port!r}")
            return ("signal",) + leaf["signals"][port]
        return self.ports[owner][port]

    def signals(self, owner, port):
        """``{suffix: (name at the owner, direction, width)}`` -- one entry,
        suffix "", for a signal."""
        s = self.spec(owner, port)
        if s[0] == "signal":
            return {"": (port, s[1], s[2])}
        if owner in self.leaves:
            b = self.leaves[owner]["bundles"][port]
            return {_role(b, m): (m,) + self.leaves[owner]["signals"][m]
                    for m in b["members"]}
        return {sfx: (f"{port}_{sfx}",) + dw for sfx, dw in s[2].items()}

    # -- a composite, children first -------------------------------------- #

    def composite(self, path: str):
        node = self.h[path]
        for child in node["children"].values():
            if child in self.h and child not in self.ports:
                self.composite(child)
        self.group(path)
        own_in, own_out = set(node["inputs"]), set(node["outputs"])
        pins = self.pins_of(path)

        # Its own ports: what each meets inside.
        ports = {}
        for a, b in node["connections"]:
            for mine, other, role in ((a, b, "source"), (b, a, "sink")):
                # its declared inputs and outputs; a pin it owns is the
                # FPGA's, taken below
                if mine[0] != path or mine[1] in ports \
                        or mine[1] not in own_in | own_out:
                    continue
                if other[0] == path:
                    raise VerilogError(f"{path}: {mine[1]} is wired straight "
                                       f"to {other[1]}")
                spec = self.spec(*other)
                if spec[0] == "signal":
                    direction = "input" if role == "source" else "output"
                    ports[mine[1]] = ("signal", direction, spec[2])
                else:
                    ports[mine[1]] = spec
        # External pins in it: the FPGA's, named as declared, globally.
        for name in pins:
            for a, b in self.all_connections(path):
                for mine, other in ((a, b), (b, a)):
                    if mine == pins[name] and name not in ports:
                        # driven inside: an output of every module it
                        # passes through; read inside: an input
                        ports[name] = self.spec(*other)
        self.ports[path] = ports
        self.texts[path] = self.body(path, ports)

    def pins_of(self, path):
        """External pins owned anywhere in the composite at `path`, by
        name -> the pin as its owner's connections name it."""
        out = {}
        for p, node in self.h.items():
            if p != path and not (not path or p.startswith(path + ".")):
                continue
            for a, b in node["connections"]:
                for pin in (a, b):
                    # a pin is the composite's that declares it, never a
                    # port of the same name
                    if pin[0] == p and pin[1] in node["pins"]:
                        out[pin[1]] = pin
        return out

    def all_connections(self, path):
        for p, node in self.h.items():
            if p == path or (not path) or p.startswith(path + "."):
                yield from node["connections"]

    def body(self, path, ports) -> list:
        node = self.h[path]

        # Terminals: (owner, signal name at the owner); owner "" is this
        # module itself (its ports and the pins rising through it).
        group: dict[tuple, int] = {}
        nets: list[list] = []

        def join(x, y):
            gx, gy = group.get(x), group.get(y)
            if gx is None and gy is None:
                group[x] = group[y] = len(nets)
                nets.append([x, y])
            elif gy is None:
                group[y] = gx
                nets[gx].append(y)
            elif gx is None:
                group[x] = gy
                nets[gy].append(x)
            elif gx != gy:
                for t in nets[gy]:
                    group[t] = gx
                nets[gx] += nets[gy]
                nets[gy] = []

        def side(pin):
            """(owner key, {suffix: (name, dir, width)}) for a pin here."""
            owner, port = pin
            if owner == path:            # its own port, or a pin it owns
                spec = ports[port]
                if spec[0] == "signal":
                    return "", {"": (port, spec[1], spec[2])}
                return "", {sfx: (f"{port}_{sfx}",) + dw
                            for sfx, dw in spec[2].items()}
            return owner, self.signals(owner, port)

        for a, b in node["connections"]:
            (oa, sa), (ob, sb) = side(a), side(b)
            for sfx in sb:
                if sfx in sa:
                    join((oa, sa[sfx][0]), (ob, sb[sfx][0]))
        # A child composite's external pins rise through this module.
        for child in node["children"].values():
            if child in self.h:
                for name in self.pins_of(child):
                    for sfx, (sig, _, _) in side((path, name))[1].items():
                        join(("", sig), (child, sig))

        width_of = {}
        for owner, key in [t for n in nets for t in n]:
            if owner == "":
                continue
            width_of[(owner, key)] = self.width(owner, key)
        wire, widths = {}, {}
        lines, assigns = [], []
        for members in nets:
            if not members:
                continue
            own = [s for o, s in members if o == ""]
            if own:
                # one of the module's own ports names the net; any other
                # own port on it is driven from that one (an output tapped
                # twice, or an input passed straight out)
                drives = [s for s in own if self.own_direction(ports, s)
                          == "input"]
                name = drives[0] if drives else own[0]
                widths[name] = self.own_width(ports, name)
                for other in own:
                    if other != name:
                        assigns.append(f"    assign {other} = {name};")
            else:
                driver = next(((o, s) for o, s in members
                               if self.direction(o, s) == "output"),
                              members[0])
                cell = driver[0].rpartition(".")[2].lstrip("@")
                name = f"{cell}_{driver[1]}"
                w = width_of.get(driver, 1)
                widths[name] = w
                lines.append(f"    wire {self.range(w)}{name};")
            for t in members:
                wire[t] = name

        if assigns:
            lines += [""] + assigns
        # an instance shares a module's one namespace with its ports and
        # wires, which a block design kept apart: a clash is renamed
        taken = set(wire.values()) | {s for spec_port, spec in ports.items()
                                      for s in ([spec_port] if spec[0] ==
                                                "signal" else
                                                [f"{spec_port}_{x}" for x
                                                 in spec[2]])}
        for child_name, child in node["children"].items():
            lines.append("")
            label = child_name if child_name not in taken \
                else f"{child_name}_i"
            lines += self.instance(label, child, wire, widths)
        return lines

    @staticmethod
    def own_direction(ports, name):
        for port, spec in ports.items():
            if spec[0] == "signal" and port == name:
                return spec[1]
            if spec[0] == "bundle":
                for sfx, (d, _) in spec[2].items():
                    if f"{port}_{sfx}" == name:
                        return d
        return None

    @staticmethod
    def own_width(ports, name):
        for port, spec in ports.items():
            if spec[0] == "signal" and port == name:
                return spec[2]
            if spec[0] == "bundle":
                for sfx, (_, w) in spec[2].items():
                    if f"{port}_{sfx}" == name:
                        return w
        return 1

    def direction(self, owner, signal):
        if owner in self.leaves:
            return self.leaves[owner]["signals"][signal][0]
        for port, spec in self.ports[owner].items():
            if spec[0] == "signal" and port == signal:
                return spec[1]
            if spec[0] == "bundle":
                for sfx, (d, _) in spec[2].items():
                    if f"{port}_{sfx}" == signal:
                        return d
        return None

    def width(self, owner, signal):
        if owner in self.leaves:
            return self.leaves[owner]["signals"][signal][1]
        for port, spec in self.ports[owner].items():
            if spec[0] == "signal" and port == signal:
                return spec[2]
            if spec[0] == "bundle":
                for sfx, (_, w) in spec[2].items():
                    if f"{port}_{sfx}" == signal:
                        return w
        return 1

    @staticmethod
    def range(width):
        return "" if width == 1 else f"[{width - 1}:0] "

    def instance(self, name, child, wire, widths) -> list:
        if child in self.leaves:
            leaf = self.leaves[child]
            params = leaf["parameters"]
            head = f"    {leaf['module']}"
            if params:
                head += " #(" + ", ".join(
                    f".{k}({_literal(v)})" for k, v in params.items()) + ")"
            # every pin, an empty one where nothing is wired: what is left
            # open is written down, not implied
            conns = []
            for signal, pin in leaf["pin"].items():
                if (child, signal) in wire:
                    net = wire[(child, signal)]
                    direction, width = leaf["signals"][signal]
                    # a narrower input takes the low bits -- an AXI slave
                    # decodes its window's; any other mismatch the wiring
                    # check has refused already
                    if direction == "input" and widths.get(net, width) > width:
                        net = f"{net}[{width - 1}:0]"
                    conns.append(f".{pin}({net})")
                elif pin in leaf["tie"]:
                    conns.append(f".{pin}({leaf['tie'][pin]})")
                else:
                    conns.append(f".{pin}()")
            for pin, value in leaf["tie"].items():
                if pin not in leaf["pin"].values():
                    conns.append(f".{pin}({value})")
        else:
            # named once the module is settled (modules(), children first)
            head = f"    @@{child}@@"
            conns = []
            for port, spec in self.ports[child].items():
                names = [(port, spec[1], spec[2])] if spec[0] == "signal" \
                    else [(f"{port}_{sfx}", d, w)
                          for sfx, (d, w) in spec[2].items()]
                for sig, direction, width in names:
                    net = wire.get((child, sig), "")
                    if net and direction == "input" \
                            and widths.get(net, width) > width:
                        net = f"{net}[{width - 1}:0]"
                    conns.append(f".{sig}({net})")
        out = [f"{head} {name} ("]
        out += [f"        {c}," for c in conns[:-1]]
        if conns:
            out.append(f"        {conns[-1]}")
        out.append("    );")
        return out


def _literal(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        # past a 32-bit integer, sized: no tool guesses its width
        if value >= 1 << 31 or value < -(1 << 31):
            return f"{max(32, value.bit_length())}'d{value}" if value >= 0 \
                else f"-{max(32, (-value).bit_length())}'d{-value}"
        return str(value)
    return f'"{value}"'


def _header(name, ports) -> list:
    decls = []
    for port, spec in ports.items():
        if spec[0] == "signal":
            decls.append((spec[1], spec[2], port))
        else:
            for sfx, (direction, width) in spec[2].items():
                decls.append((direction, width, f"{port}_{sfx}"))
    lines = [f"module {name} ("]
    for i, (direction, width, port) in enumerate(decls):
        rng = "" if width == 1 else f"[{width - 1}:0] "
        end = "," if i < len(decls) - 1 else ""
        lines.append(f"    {direction:<6} wire {rng}{port}{end}")
    lines.append(");")
    return lines


def modules(design, groups=None) -> list[tuple[str, str]]:
    """Every composite of a loaded design as a Verilog module, in an order
    a tool can read -- each after what it instances -- the top last, named
    for the design. A composite instanced twice is one module where its
    two instances write alike, and one module each where they do not.

    ``groups`` -- ``{name: [leaf paths]}`` -- folds leaves a toolchain makes
    together (in one block design, say) into one black box by that name;
    :func:`group_ports` is what its ports are."""
    return _write(design, groups)[0]


def group_ports(design, groups) -> dict:
    """``{group: {port: {"spec", "member"}}}`` -- each group's ports as the
    modules instance them: a member's port that meets the rest of the
    design, named ``<cell>_<port>``, its spec (``("signal", direction,
    width)`` or ``("bundle", side, {suffix: (direction, width)})``) and the
    member port it is."""
    return _write(design, groups)[1]


def _write(design, groups):
    if design.netlist is None:
        raise VerilogError("a design loaded without generating has no "
                           "netlist to write from")
    w = _Writer(design, groups)
    w.composite("")
    order = sorted(w.texts, key=lambda p: -p.count(".") if p else 1)
    order = [p for p in order if p] + [""]
    by_text: dict[str, str] = {}
    used: set[str] = set()
    out = []
    for path in order:
        base = design.name if not path else \
            design.hierarchy[path]["ref"].rpartition(":")[2] \
            .rpartition(".")[2]
        lines = []
        for line in w.texts[path]:
            if "@@" in line:
                child = line.split("@@")[1]
                line = line.replace(f"@@{child}@@", w.names[child])
            lines.append(line)
        # its body names its children's modules, so it is final here
        key = "\n".join(_header("X", w.ports[path]) + lines)
        if key in by_text:
            w.names[path] = by_text[key]
            continue
        name, n = base, 1
        while name in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name)
        w.names[path] = by_text[key] = name
        text = "\n".join(_header(name, w.ports[path]) + lines
                         + ["endmodule", ""])
        out.append((name, text))
    return out, w.group_ports
