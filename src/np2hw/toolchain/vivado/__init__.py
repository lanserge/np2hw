# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A design's block design, for Vivado: from the loaded JSON to Tcl.

Nothing here knows a board or a design. The hierarchy is loaded by
np2hw.design.composite and flattened; this renders what it holds, in
phases, because order is not structure:

  project     the part, the board, the IP catalogues, every source and
              constraints file the design's blocks name
  cells       every leaf, in declaration order: vendor IP by its VLNV with
              its properties, our modules by name with their parameters,
              and a procedure where a leaf is built by one (a PS whose
              board preset must be applied before its configuration)
  pins        the FPGA's ports
  nets        every connection: a stream between two of our modules is an
              interface net (its six signals where one is tapped
              elsewhere), a declared interface an interface net, anything
              else a net -- each net made in one command, however many
              composites it crossed
  late        procedures that work on the connected design -- the bus
  finish      validate, save, the wrapper

A leaf names its Vivado implementation (``"vivado": {...}``); another
toolchain is another implementation of the same leaves, rendered by its own
emitter from the same design.
"""
from __future__ import annotations

from pathlib import Path

from np2hw.design.composite import source_file
from np2hw.design.interfaces import wiring


def tcl(value) -> str:
    """A JSON value as a Tcl word."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[list " + " ".join(tcl(v) for v in value) + "]"
    if isinstance(value, dict):
        return "[dict create " + " ".join(
            f"{tcl(k)} {tcl(v)}" for k, v in value.items()) + "]"
    return "{" + str(value) + "}"


class EmitError(ValueError):
    pass


def hex_word(value: int) -> str:
    """A 32-bit value as Vivado's IP writes one (``0x0000002A``), for a
    design's expression to set an IP property that takes that form:
    ``"@np2hw.toolchain.vivado:hex_word(default)"``."""
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


hex_word.expression = True


def mhz(khz: int) -> str:
    """A frequency in kHz as Vivado's IP properties spell one, in MHz to
    three places (``148.500``): for a design's expression to set a
    requested clock from the rate it derives --
    ``"@np2hw.toolchain.vivado:mhz(out_khz)"``."""
    return f"{int(khz) / 1000:.3f}"


mhz.expression = True


def _range(size: int) -> str:
    """A window's size as Vivado writes a range: 4K, 64K, 1M, 1G."""
    for unit, shift in (("G", 30), ("M", 20), ("K", 10)):
        if size >= 1 << shift and size % (1 << shift) == 0:
            return f"{size >> shift}{unit}"
    raise EmitError(f"a window of {size:#x} bytes is no Vivado range")


def cell_names(design) -> dict:
    """Each leaf's cell in the block design: its node's own name, which a
    block design holds once."""
    cell = {}
    for path in design.instances:
        name = path.rpartition(".")[2]
        if name in cell.values():
            other = next(p for p, n in cell.items() if n == name)
            raise EmitError(f"{path} and {other} would both be the Vivado "
                            f"cell {name!r}; a block design has one cell by "
                            "a name")
        cell[path] = name
    return cell


def emit(design, *, out: Path, part: str, board_part: str,
         board_files: Path, ip_repos: list[Path], name: str | None = None
         ) -> str:
    """The Tcl that builds ``design`` as a Vivado block design: the project
    and the block design both called ``name`` (the design's own name when
    none is given), its wrapper the top."""
    out = Path(out)
    bd = name or design.name
    leaves = design.instances
    cell = cell_names(design)

    def vivado(inst) -> dict:
        obj = inst.leaf if isinstance(inst.leaf, dict) else {}
        v = obj.get("vivado")
        if v is None:
            raise EmitError(f"{inst.path} ({inst.ref}) has no Vivado "
                            "implementation")
        return v

    # Sources: every module leaf's, beside it or generated into the build.
    sources, sources_seen = [], set()
    for inst in leaves.values():
        obj = inst.leaf if isinstance(inst.leaf, dict) else {}
        files = [source_file(inst.folder, f) for f in obj.get("sources", [])]
        files += [out / "hdl" / f for f in obj.get("generated", [])]
        for f in files:
            f = f.resolve()
            if f not in sources_seen:
                sources_seen.add(f)
                sources.append(f)

    # How each leaf is wired, from the ports it states -- its streams by
    # bundle, its other interfaces by name. Whether those wires agree is
    # not this file's question: the design was checked when it was loaded
    # (np2hw.design.wiring), for every toolchain alike.
    streams, named = {}, {}
    for path, inst in leaves.items():
        streams[path], named[path] = wiring(inst.leaf)
        renamed = (inst.leaf.get("port_map", {})
                   if isinstance(inst.leaf, dict) else {})
        named[path] = {renamed.get(n, n) for n in named[path]}

    L = []
    w = L.append
    w("# GENERATED by np2hw.toolchain.vivado from the design's JSON -- do not "
      "edit.")
    w(f"# {design.ref}")
    w("set out [file normalize [lindex $argv 0]]")
    w(f"set_param board.repoPaths {tcl(str(board_files))}")
    w(f"create_project {bd} [file join $out build {bd}] -part {part} "
      "-force")
    w(f"set_property board_part {board_part} [current_project]")
    w("set_property ip_repo_paths [list "
      + " ".join(tcl(str(p)) for p in ip_repos) + "] [current_project]")
    w("update_ip_catalog")
    w("add_files " + " ".join(tcl(str(f)) for f in sources))
    for c in design.constraints:
        w(f"add_files -fileset constrs_1 {tcl(str(c['file']))}")
        if c["scoped_to"]:
            # one file for every instance of a module, by the module's own
            # names, read once the clocks it asks about exist
            w(f"set_property SCOPED_TO_REF {c['scoped_to']} "
              f"[get_files {tcl(str(c['file']))}]")
            w(f"set_property PROCESSING_ORDER LATE "
              f"[get_files {tcl(str(c['file']))}]")
    w(f"create_bd_design {bd}")
    w("")

    sourced = set()

    def procedure(inst, v):
        src = (inst.folder / v["source"]).resolve()
        if src not in sourced:
            sourced.add(src)
            w(f"source {tcl(str(src))}")
        settings = {**inst.build, **inst.config}
        w(f"{v['tcl']} {cell[inst.path]} {tcl(settings)}")

    late = []
    for path, inst in leaves.items():
        v = vivado(inst)
        name = cell[path]
        if "tcl" in v:
            if v.get("phase") == "late":
                late.append((inst, v))
            else:
                procedure(inst, v)
            continue
        if "ip" in v:
            w(f"create_bd_cell -type ip -vlnv {v['ip']} {name}")
        elif "module" in v:
            w(f"create_bd_cell -type module -reference {v['module']} {name}")
        else:
            raise EmitError(f"{path}: vivado implementation names no ip, "
                            "module or tcl")
        # A generated leaf states the values its module's parameters take
        # in this use -- np2hw's fixed modules are files, sized here.
        own = inst.leaf.get("module_parameters", {}) \
            if "module" in v and isinstance(inst.leaf, dict) else {}
        props = {**own, **inst.build, **inst.config}
        if props:
            w("set_property -dict [list " + " ".join(
                f"CONFIG.{k} {tcl(val)}" for k, val in props.items())
              + f"] [get_bd_cells {name}]")

    # Pins and connections.
    def split(endpoint):
        if endpoint.startswith("pin:"):
            return None, endpoint[len("pin:"):]
        path, _, port = endpoint.rpartition(".")
        # a block filled by an implementation is wired by the block's port
        # names, and its cell has the implementation's
        leaf = leaves[path].leaf if path in leaves else {}
        renamed = leaf.get("port_map", {}) if isinstance(leaf, dict) else {}
        return path, renamed.get(port, port)

    def interfaces(path):
        return named[path]

    nets: list[list[str]] = []
    member: dict[str, int] = {}

    def join(a, b):
        ia, ib = member.get(a), member.get(b)
        if ia is None and ib is None:
            member[a] = member[b] = len(nets)
            nets.append([a, b])
        elif ib is None:
            nets[ia].append(b)
            member[b] = ia
        elif ia is None:
            nets[ib].insert(0, a)
            member[a] = ib
        elif ia != ib:
            nets[ia].extend(x for x in nets[ib] if x not in nets[ia])
            for x in nets[ib]:
                member[x] = ia
            nets[ib] = []

    # A pin of a stream that is also wired elsewhere -- a status tap on
    # its valid -- rules out an interface net for that stream: Vivado
    # lets a net on one pin of an interface OVERRIDE the interface's
    # connection of it, so the tap would take the pin away from the sink.
    # Such a stream is joined signal by signal, the tap on the same net.
    def is_stream_edge(sp, sport, dp, dport):
        return (sp is not None and dp is not None
                and sport in streams.get(sp, {}) and dport in streams.get(dp, {}))

    tapped = set()
    for source, sink in design.edges:
        (sp, sport), (dp, dport) = split(source), split(sink)
        if not is_stream_edge(sp, sport, dp, dport):
            tapped.update((path, port) for path, port in ((sp, sport), (dp, dport))
                          if path is not None)

    def untapped(path, prefix):
        return not any((path, f"{prefix}_{s}") in tapped
                       for s in streams[path][prefix]["signals"])

    intf_nets, external = [], {}
    for source, sink in design.edges:
        (sp, sport), (dp, dport) = split(source), split(sink)
        if sp is None or dp is None:
            pin_name = sport if sp is None else dport
            other = (dp, dport) if sp is None else (sp, sport)
            decl = design.pins[pin_name]
            if decl.get("interface"):
                external[pin_name] = f"{cell[other[0]]}/{other[1]}"
            else:
                join(f"port:{pin_name}", f"{cell[other[0]]}/{other[1]}")
            continue
        a, b = f"{cell[sp]}/{sport}", f"{cell[dp]}/{dport}"
        if sport in streams.get(sp, {}) and dport in streams.get(dp, {}):
            ga, gb = streams[sp][sport], streams[dp][dport]
            if untapped(sp, sport) and untapped(dp, dport):
                # both ends declare the interface: one connection, by the
                # name the block design gives it
                intf_nets.append((f"{cell[sp]}/{ga['bundle']}",
                                  f"{cell[dp]}/{gb['bundle']}"))
            else:
                # joined wire by wire: the signals the sink takes, which
                # the source carries (a core regenerates what it lacks)
                for s in gb["signals"]:
                    if s in ga["signals"]:
                        join(f"{a}_{s}", f"{b}_{s}")
        elif sport in interfaces(sp) or dport in interfaces(dp):
            intf_nets.append((a, b))
        else:
            join(a, b)

    w("")
    for name, decl in design.pins.items():
        if decl.get("interface"):
            if name not in external:
                raise EmitError(f"interface pin {name} is connected to "
                                "nothing")
            w(f"make_bd_intf_pins_external [get_bd_intf_pins "
              f"{external[name]}]")
            leaf_pin = external[name].rpartition("/")[2]
            w(f"set_property name {name} [get_bd_intf_ports {leaf_pin}_0]")
        else:
            w(f"create_bd_port -dir {decl['dir']} {name}")

    def obj(pin):
        return (f"[get_bd_ports {pin[5:]}]" if pin.startswith("port:")
                else f"[get_bd_pins {pin}]")

    for net in nets:
        if net:
            w("connect_bd_net " + " ".join(obj(p) for p in net))
    for a, b in intf_nets:
        w(f"connect_bd_intf_net [get_bd_intf_pins {a}] [get_bd_intf_pins {b}]")

    w("")
    for inst, v in late:
        procedure(inst, v)

    # Addresses. A leaf that decides where the slaves on its ports sit
    # publishes it -- the fact `windows`, {its port: [base, size]} -- and
    # each is set where Vivado records it: the slave's segment, in the
    # address space of the port it hangs off. Vivado's own assignment then
    # covers only the processor's map (its general-purpose port onto the
    # bus, its memory port onto memory), which it knows; anything it would
    # otherwise place on our ports at an offset of its own choosing is
    # already placed, at the address the hardware decodes.
    # Through a block that only forwards (np2hw.bus.hops), each hop is
    # set in the space of the port before it, at the same window.
    from np2hw.bus import hops

    def pin(path, port):
        return "/".join((cell[path], split(f"{path}.{port}")[1]))

    placed, done = False, set()
    for base, size, chain, _ in hops(design):
        for space, slave in chain:
            # a hop two masters' walks share is set once, by the first:
            # Vivado records a segment once in an address space
            if (space, slave) in done:
                continue
            done.add((space, slave))
            w(f"assign_bd_address -target_address_space /{pin(*space)} "
              f"-offset {base:#010x} -range {_range(size)} "
              f"[get_bd_addr_segs -of_objects [get_bd_intf_pins "
              f"{pin(*slave)}]]")
            placed = True
    w("assign_bd_address")
    if placed:
        w("")

    w("validate_bd_design")
    # The described vendor blocks' pins as the tool built them -- after
    # validation, which propagates widths from what is connected -- for the
    # check that holds each description to its implementation
    # (np2hw.toolchain.vivado.vendor_check). Read by that test, never by the build.
    from np2hw.toolchain.vivado.vendor_check import vendor
    described = [cell[p] for p, inst in leaves.items() if vendor(inst.leaf)]
    w("")
    w(f"file mkdir {tcl(str(out / 'generated'))}")
    w(f"set dump [open {tcl(str(out / 'generated' / 'vendor_ports.txt'))} w]")
    w("foreach c [list " + " ".join(described) + "] {")
    w('    puts $dump "CELL $c"')
    w("    foreach p [get_bd_pins -of [get_bd_cells $c]] {")
    w('        puts $dump "PIN [get_property NAME $p] [get_property DIR $p] '
      '[get_property LEFT $p] [get_property RIGHT $p] [get_property TYPE $p]"')
    w("    }")
    w("}")
    w("close $dump")
    w("")
    w("save_bd_design")
    w(f"make_wrapper -files [get_files {bd}.bd] -top -import")
    w(f"set_property top {bd}_wrapper [current_fileset]")
    w('puts "BD_DONE"')
    return "\n".join(L) + "\n"
