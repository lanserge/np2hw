# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""What a block-design script builds, read without Vivado.

Runs a block-design Tcl script under plain tclsh, with every Vivado command
it uses replaced by a stub that records what was asked, and returns the
design's STRUCTURE: its cells with their configuration, its nets as sets of
pins, its interface nets, its external ports, and the bus automation it
asked for. Order is not structure -- two scripts that create the same cells
and join the same pins describe the same design, whatever order they do it
in -- so two structures compare equal exactly when the designs do.

What the automation then builds is Vivado's; the automation CALLS are
recorded, in order, because their order is part of what they ask for.

    python3 -m np2hw.toolchain.vivado.bdtrace <bd.tcl> --out <build dir> > structure.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# The stubs. Objects are their names; a net is named by any pin on it, and
# the Python side joins pins into nets from the recorded connections.
STUBS = r'''
proc _log {args} {
    set clean {}
    foreach a $args { lappend clean [string map [list "\n" " "] $a] }
    puts $::_trace [join $clean "\x1f"]
}
proc _strip {x} { return [string trimleft $x /] }
proc _flags {argv} {
    # drop Vivado's -quiet / -verbose, keep everything else in order
    set out {}
    foreach a $argv { if {$a ni {-quiet -verbose}} { lappend out $a } }
    return $out
}
proc set_param {args} { _log set_param {*}$args }
proc create_project {args} { _log create_project }
proc current_project {args} { return project }
proc current_fileset {args} { return sources_1 }
proc update_ip_catalog {args} {}
proc add_files {args} { _log add_files {*}$args }
proc create_bd_design {args} { _log create_bd_design {*}$args }
proc validate_bd_design {args} {}
proc save_bd_design {args} {}
proc make_wrapper {args} {}
proc get_files {args} { return [lindex $args end] }
proc create_bd_cell {args} {
    set a [_flags $args]
    set name [lindex $a end]
    if {[set i [lsearch $a -vlnv]] >= 0} {
        _log cell $name ip [lindex $a [expr {$i + 1}]]
    } elseif {[set i [lsearch $a -reference]] >= 0} {
        _log cell $name module [lindex $a [expr {$i + 1}]]
    } else { error "create_bd_cell: unknown form $args" }
    set ::_cells($name) 1
    return $name
}
proc get_bd_cells {args} {
    set a [_flags $args]
    set name [_strip [lindex $a end]]
    if {[info exists ::_cells($name)]} { return $name }
    return ""
}
proc get_bd_pins {args} {
    set a [_flags $args]
    if {[set i [lsearch $a -of_objects]] >= 0} { return {} }
    set out {}
    foreach p $a { lappend out [_strip $p] }
    return [join $out " "]
}
proc get_bd_intf_pins {args} {
    set a [_flags $args]
    if {[set i [lsearch $a -of_objects]] >= 0} {
        # An interface net the automation made: the pin it was asked
        # about, and a stand-in for the master the automation chose.
        set net [lindex $a [expr {$i + 1}]]
        if {[string match intfnet:* $net]} {
            return [list [string range $net 8 end] auto:master]
        }
        return {}
    }
    return [_strip [lindex $a end]]
}
proc get_bd_ports {args} { return "port:[_strip [lindex [_flags $args] end]]" }
proc get_bd_intf_ports {args} { return "port:[_strip [lindex [_flags $args] end]]" }
proc get_bd_nets {args} {
    set a [_flags $args]
    if {[set i [lsearch $a -of_objects]] >= 0} {
        set pin [lindex $a [expr {$i + 1}]]
        if {$pin eq ""} { return {} }
        return "net:$pin"
    }
    return {}
}
proc get_bd_intf_nets {args} {
    set a [_flags $args]
    if {[set i [lsearch $a -of_objects]] >= 0} {
        return "intfnet:[lindex $a [expr {$i + 1}]]"
    }
    return {}
}
proc get_bd_addr_segs {args} { return [_strip [lindex [_flags $args] end]] }
proc get_property {prop obj} {
    if {$prop eq "NAME"} { return $obj }
    return ""
}
proc set_property {args} {
    set a [_flags $args]
    if {[lindex $a 0] eq "-dict"} {
        foreach {k v} [lindex $a 1] { _log prop [lindex $a 2] $k $v }
    } else {
        _log prop [lindex $a 2] [lindex $a 0] [lindex $a 1]
    }
}
proc connect_bd_net {args} {
    set a [_flags $args]
    if {[lindex $a 0] eq "-net"} {
        _log net [lindex $a 1] {*}[lrange $a 2 end]
    } else {
        _log net {} {*}$a
    }
}
proc disconnect_bd_net {args} { _log disconnect {*}[_flags $args] }
proc connect_bd_intf_net {args} { _log intf {*}[_flags $args] }
proc delete_bd_objs {args} { _log delete {*}$args }
proc create_bd_port {args} {
    set a [_flags $args]
    _log port [lindex $a end] [lindex $a [expr {[lsearch $a -dir] + 1}]]
}
proc make_bd_intf_pins_external {args} {
    set p [lindex [_flags $args] end]
    _log intfport [lindex [split $p /] end]_0 $p
}
proc apply_bd_automation {args} { _log automation {*}[_flags $args] }
proc assign_bd_address {args} { _log address {*}[_flags $args] }
'''


def trace(script: Path, argv: list[str]) -> list[list[str]]:
    """Run ``script`` under the stubs; return every recorded command."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "trace"
        runner = Path(tmp) / "run.tcl"
        runner.write_text(
            STUBS
            + f"set ::_trace [open {{{log}}} w]\n"
            + "set argv [list " + " ".join("{%s}" % a for a in argv) + "]\n"
            + f"set argc {len(argv)}\n"
            + f"set ::argv0 {{{script}}}\n"
            + "rename info _info\n"
            + "proc info {args} { if {$args eq \"script\"} "
            + f"{{ return {{{script}}} }}; return [_info {{*}}$args] }}\n"
            + f"source {{{script}}}\n"
            + "close $::_trace\n")
        result = subprocess.run(["tclsh", str(runner)], capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed under the stubs:\n"
                               f"{result.stderr}{result.stdout}")
        lines = log.read_text().splitlines()
    return [line.split("\x1f") for line in lines]


def structure(commands: list[list[str]]) -> dict:
    """The design the commands describe, with order taken out of it."""
    cells, props, ports, automation, edits, addresses, files = \
        {}, {}, {}, [], [], [], []
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    intf: dict[str, str] = {}
    ifind_parent = intf

    def ifind(x):
        ifind_parent.setdefault(x, x)
        while ifind_parent[x] != x:
            x = ifind_parent[x]
        return x

    for cmd in commands:
        kind, args = cmd[0], cmd[1:]
        if kind == "cell":
            name, typ, what = args
            cells[name] = {"type": typ, "of": what}
        elif kind == "prop":
            obj, key, value = args
            if obj.startswith("port:") and key == "name":
                # `set_property name TMDS [get_bd_intf_ports TMDS_0]`, at
                # the moment it happens: the next external port may be
                # called TMDS_0 too.
                old = obj[len("port:"):]
                ports[value] = ports.pop(old)
                continue
            props.setdefault(obj, {})[key] = value
        elif kind == "net":
            named, pins = args[0], [p for p in args[1:] if p]
            pins = [p[len("port:"):] if p.startswith("port:") else p
                    for p in pins]
            anchor = named[len("net:"):] if named.startswith("net:") else None
            for p in pins:
                find(p)
            if anchor is not None:
                for p in pins:
                    union(p, anchor)
            else:
                for p in pins[1:]:
                    union(p, pins[0])
        elif kind == "intf":
            a, b = args
            ifind_parent[ifind(a)] = ifind(b)
        elif kind == "port":
            ports[args[0]] = args[1]
        elif kind == "intfport":
            ports[args[0]] = f"external {args[1]}"
        elif kind == "automation":
            automation.append(args)
        elif kind in ("disconnect", "delete"):
            edits.append(cmd)
        elif kind == "address":
            addresses.append(args)
        elif kind == "add_files":
            files.append(args)

    nets: dict[str, set] = {}
    for p in list(parent):
        nets.setdefault(find(p), set()).add(p)
    inets: dict[str, set] = {}
    for p in list(ifind_parent):
        inets.setdefault(ifind(p), set()).add(p)
    for name, cell in cells.items():
        cell["config"] = dict(sorted(props.get(name, {}).items()))
    return {
        "cells": dict(sorted(cells.items())),
        "nets": sorted(sorted(n) for n in nets.values() if len(n) > 1),
        "intf_nets": sorted(sorted(n) for n in inets.values() if len(n) > 1),
        "ports": dict(sorted(ports.items())),
        "automation": automation,
        "edits": edits,
        "addresses": addresses,
        "files": files,
    }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python3 -m np2hw.toolchain.vivado.bdtrace")
    parser.add_argument("script", type=Path)
    parser.add_argument("--out", required=True,
                        help="the build directory the script is given")
    args = parser.parse_args(argv)
    json.dump(structure(trace(args.script.resolve(), [args.out])), sys.stdout,
              indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
