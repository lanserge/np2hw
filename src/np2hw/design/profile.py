# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Profiles: register values for one block, keyed relative to it.

A profile lives in a library file beside blocks and is referenced the same
way. It names the block it is FOR, and every key is a path relative to that
block -- which is what makes it reusable: a sensor's profile is for
``bayer_isp``, and it tunes every design that contains one.

    "profiles": {"ov5647": {
        "for": "acme.pipelines.isp:isp",
        "values": {"ccm.m": [[547, -167, -124], ...], ...}}}

A block's parameters include everything inside it, under the path it sits
at, so a profile for a bigger block reaches into the smaller ones -- and may
INCLUDE a profile for one of them by setting its node to a reference:

    "profiles": {"display": {
        "for": "acme.boards.board.isp:board_isp",
        "values": {"isp": "acme.sensors.ov5647.profile:ov5647",
                   "rgb_gamma.knots": [0, 53, 72, ...]}}}

A value set in a profile wins over one an included profile sets.

A resolved profile still has RELATIVE keys. flatten() anchors it in a built
design: the register map records where each composite instance sits, so the
keys become the design's absolute ones (``isp.ccm.m_0_0``), each checked
against the map -- its name, its shape, its range. The map is the truth of
the build; a profile that disagrees with it is refused, not masked.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from np2hw.design.composite import (LIBRARY_HELP, CompositionError,
                                      add_libraries, canonical, find, tree)


@dataclass
class Profile:
    ref: str
    for_ref: str
    description: str
    values: dict = field(default_factory=dict)    # "isp.ccm.m" -> literal
    sources: dict = field(default_factory=dict)   # "isp.ccm.m" -> profile ref


def resolve(ref: str, _stack: tuple = ()) -> Profile:
    """A profile with its includes expanded, every key checked against the
    structure of the block it is for."""
    ref = canonical(ref)
    if ref in _stack:
        raise CompositionError(
            f"{ref} includes itself ({' -> '.join([*_stack, ref])})")
    found = find(ref)
    if found.kind != "profile":
        raise CompositionError(f"{ref} is a {found.kind}, not a profile")
    for_ref = canonical(found.obj["for"])
    structure = tree(for_ref)
    if structure[""].kind == "profile":
        raise CompositionError(f"{ref}: for {for_ref}, which is a profile")
    profile = Profile(found.ref, for_ref, found.obj.get("description", ""))

    items = found.obj["values"].items()
    # Includes first, so what this profile sets itself wins over them.
    for key, value in items:
        if not isinstance(value, str):
            continue
        where = f"{ref}: {key}"
        if key not in structure or key == "":
            raise CompositionError(
                f"{where}: an included profile goes at a node, and {for_ref} "
                f"has no node {key!r}")
        inner = resolve(value, (*_stack, ref))
        if inner.for_ref != structure[key].ref:
            raise CompositionError(
                f"{where}: {value} is for {inner.for_ref}, but {key} is "
                f"{structure[key].ref}")
        for rel, literal in inner.values.items():
            profile.values[f"{key}.{rel}"] = literal
            profile.sources[f"{key}.{rel}"] = inner.sources[rel]
    for key, value in items:
        if isinstance(value, str):
            continue
        _check_register(structure, for_ref, key, f"{ref}: {key}")
        profile.values[key] = value
        profile.sources[key] = ref
    return profile


def _check_register(structure: dict, for_ref: str, key: str, where: str):
    """``key`` names a register of a block inside ``for_ref``."""
    instance, _, register = key.rpartition(".")
    leaf = structure.get(instance)
    if leaf is None:
        raise CompositionError(f"{where}: {for_ref} has no node "
                               f"{instance!r}")
    if leaf.kind != "block":
        raise CompositionError(f"{where}: {instance or for_ref} is a "
                               f"{leaf.kind}; only a block has registers")
    declared = {p.name for p in leaf.obj.params.params}
    if register not in declared:
        raise CompositionError(
            f"{where}: block {leaf.obj.name} has no register {register!r} "
            f"(registers {sorted(declared)})")


def elements(key: str, value) -> list[tuple[str, int]]:
    """A register's value as the map names it: a shaped register is one
    register per element, ``name_i_j``, row-major."""
    if not isinstance(value, list):
        return [(key, int(value))]
    out = []
    for i, item in enumerate(value):
        out.extend(elements(f"{key}_{i}", item))
    return out


def relative(profile: Profile) -> dict[str, int]:
    """Every element the profile sets, keyed relative to its block."""
    return dict(pair for key, value in profile.values.items()
                for pair in elements(key, value))


def flatten(profile: Profile, regmap: dict) -> dict[str, int]:
    """The profile anchored in a built design: the absolute keys it sets,
    at every instance of the block it is for, checked against the map."""
    anchors = [c["path"] for c in regmap.get("composites", [])
               if c["block"] == profile.for_ref]
    if not anchors:
        raise CompositionError(
            f"{profile.ref}: this design has no {profile.for_ref} (its "
            f"composites are "
            f"{[c['block'] for c in regmap.get('composites', [])]})")
    registers = {f"{b['path']}.{r['name']}": r
                 for b in regmap["blocks"] for r in b["registers"]}
    out = {}
    for anchor in anchors:
        for key, value in profile.values.items():
            base = f"{anchor}.{key}" if anchor else key
            where = f"{profile.sources[key]}: {key}"
            given = elements(base, value)
            prefix = base + "_"
            declared = {k for k in registers
                        if k == base or (k.startswith(prefix)
                                         and k[len(prefix):].replace("_", "")
                                         .isdigit())}
            if {k for k, _ in given} != declared:
                raise CompositionError(
                    f"{where}: the value's shape does not match {base} in "
                    f"this build ({len(given)} element(s) given, the map has "
                    f"{len(declared)})")
            for element, v in given:
                r = registers[element]
                if not r["min"] <= v <= r["max"]:
                    raise CompositionError(
                        f"{where}: {v} does not fit {element} "
                        f"({r['q_format']}, {r['min']}..{r['max']})")
                out[element] = v
    return out


def main(argv=None) -> int:
    """Print a profile anchored in a design, as absolute keys: what a tool
    with no np2hw to hand -- a script on the board -- writes."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(prog="python3 -m np2hw.design.profile")
    parser.add_argument("profile", help="package.module:name")
    parser.add_argument("--map", required=True,
                        help="the design's register map (<name>_regmap.json)")
    parser.add_argument("--library", action="append", metavar="DIR",
                        help=LIBRARY_HELP)
    args = parser.parse_args(argv)
    add_libraries(args.library)
    regmap = json.loads(Path(args.map).read_text())
    json.dump(flatten(resolve(args.profile), regmap), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
