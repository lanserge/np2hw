#!/usr/bin/env python3
# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Every source file carries this repository's licence, and the right one.

Checked mechanically because a licence header applied inconsistently is
worse than useless: a reviewer cannot tell whether a file without one was
an oversight or was deliberately contributed under different terms, and
that ambiguity is exactly what a licence audit stops on. A file carrying
the WRONG identifier is worse still -- it is a claim, not a gap.

The identifier is not written here. It is read from the packaging
metadata, which is where this project already keeps the version, so the
licence has one owner too and this check cannot drift from what is
actually published.

A script rather than a test, because this project's examples ARE its
suite: each one runs a NumPy function through the compiler and compares
the simulation bit-for-bit against it. A licence check demonstrates
nothing about the compiler and does not belong among them, and a second
suite directory holding one file would be a second place to look.

    python3 scripts/checklicence.py
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Walked rather than taken from `git ls-files`: a file that has been
# written but not yet added is exactly the one that slips through, and
# neither of these directories holds generated output.
SOURCE_ROOTS = ("src", "examples", "scripts")
SUFFIXES = (".py", ".c", ".h", ".v", ".sv", ".sh", ".tcl", ".xdc", ".core",
            ".mk")
# Files at the top level that are source too. A licence check scoped to
# source DIRECTORIES misses whatever sits beside them -- the FuseSoC core
# file went uncovered for exactly that reason.
ROOT_FILES = ("np2hw.core", "Makefile")
# JSON carries no comment syntax, so a description in .json cannot hold a
# header and is covered by the repository's LICENSE rather than by a line
# of its own. Named here so the gap is a decision, not an oversight.
NO_COMMENT_SYNTAX = (".json",)


def declared_licence() -> str:
    """The licence this project publishes under, from pyproject."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    licence = data.get("project", {}).get("license")
    if isinstance(licence, dict):          # the older table form
        licence = licence.get("text")
    if not licence:
        raise SystemExit("pyproject.toml declares no licence to check against")
    return licence


def sources() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        directory = ROOT / root
        if not directory.is_dir():
            continue
        files.extend(sorted(p for p in directory.rglob("*")
                            if (p.suffix in SUFFIXES or p.name in ROOT_FILES)
                            and "__pycache__" not in p.parts))
    files.extend(ROOT / name for name in ROOT_FILES if (ROOT / name).is_file())
    return files


def main() -> int:
    ident = declared_licence()
    problems: list[str] = []
    files = sources()

    for path in files:
        rel = path.relative_to(ROOT)
        head = "".join(path.read_text(encoding="utf-8",
                                      errors="replace").splitlines(True)[:6])
        found = re.search(r"SPDX-License-Identifier:\s*(.+?)\s*"
                          r"(?:-->|\*/)?\s*$", head, re.M)
        if not found:
            problems.append(
                f"{rel}: no SPDX-License-Identifier in its first lines; "
                f"every source file states the licence it is under "
                f"(expected {ident})")
        elif found.group(1) != ident:
            problems.append(
                f"{rel}: claims {found.group(1)!r} but this project "
                f"publishes as {ident!r} (pyproject.toml). A file under "
                "different terms is a decision, not a typo -- make it "
                "deliberately or fix the header.")

    # A check that finds nothing passes forever.
    if len(files) < 10:
        problems.append(f"only {len(files)} source files found; this check "
                        "is not covering the tree it is meant to cover")

    if problems:
        print(f"{len(problems)} licence problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"  licence: OK -- {len(files)} source files carry {ident}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
