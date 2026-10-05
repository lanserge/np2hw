# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""What each core costs the core next to it.

A timing claim is made about a module's own registers, so it is true in
isolation and a design is never isolation. If a module's output is
combinational, that depth is spent by its NEIGHBOUR, before the
neighbour reaches its own first register -- and the neighbour's model
cannot see it, because the depth is not in the neighbour.

This is not hypothetical. One bench lost two builds to it from a single
boundary, -1.424ns into one consumer's first stage and -1.792ns into
another's, from a receiver and a pipeline that each closed timing alone.
Over-constraining the pipeline's generator to 200MHz -- asking for more
stages than the real period needs -- moved the slack by exactly nothing,
because the depth was never in the pipeline to cut.

So the fact is published. `combinational_outputs` is what a module hands
on; empty is the good case, and `boundary_report()` collects it across a
design.

  A SLICE IS CLEAN          a module that ends in one hands its
                            consumers a flop, whatever they are.
  A TRACED CORE IS NOT      `out_*` leave the last pipeline register,
                            but `in_ready` is `!stall` and `stall` reads
                            `out_ready`.
  AND IT COMPOUNDS          chained, those cores pass `ready` straight
                            through: sink to source, combinationally,
                            one gate deeper per block. This is the claim
                            worth having, because it is invisible from
                            inside any single core and gets worse as the
                            pipeline gets better.
  A REPORT, NOT A REFUSAL   every traced core does this today, so a rule
                            that refused would refuse everything. What
                            it can do is stop the fact being invisible.

Run:  python examples/boundaries.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from np2hw import (Image2D, Param, boundary_report,
                   combinational_outputs, framebuffer_read, generate,
                   saturate, scanout, skid, to_ir)  # noqa: E402
from np2hw.hdl import closure  # noqa: E402


def traced_core():
    """A real generated core, not a stand-in: the fact under test is
    what `generate()` publishes, so a hand-written dict would be
    checking this file against itself."""
    def offset(img, gain):
        return saturate(img.astype(np.int32) + gain, 12)

    _, line = to_ir(offset, Image2D("img", 16, 8, bits=12),
                    Param("gain", bits=12, default=64))
    return generate(line, module_name="bl_probe")

def _combinational_in_rtl(verilog, module):
    """Read the boundary fact back out of the Verilog.

    An output port leaves a flop when it is declared `output reg` (and
    so assigned only under a clock) or is a plain alias for one. Anything
    driven by an expression, or by a concatenation on an instance port,
    is combinational until shown otherwise -- the conservative direction,
    since the cost of guessing "registered" wrongly is a build.
    """
    body = re.sub(r"//[^\n]*", "", verilog.split(f"module {module}")[1])
    header, rest = body.split(");", 1)
    body = rest.split("\nendmodule")[0]
    comb = []
    for m in re.finditer(
            r"output\s+(reg|wire)?\s*(?:\[[^\]]*\])?\s*(\w+)", header):
        kind, sig = m.group(1) or "wire", m.group(2)
        if kind == "reg":
            continue
        asg = re.search(rf"assign\s+{sig}\s*=\s*([^;]+);", body)
        if asg is None:
            # Driven by an instance -- registered only if that instance
            # says so. The slice is the one that does, and it is the
            # only thing generated here that drives a port that way.
            if re.search(rf"_slice\s*#?[^;]*?\.m_data\([^)]*\b{sig}\b",
                         body, re.S):
                continue
            if re.search(rf"_slice\s*#?[^;]*?\.m_valid\({sig}\)", body, re.S):
                continue
            comb.append(sig)
            continue
        expr = " ".join(asg.group(1).split())
        # Declarations live in BOTH parts: `output reg` is in the port
        # header, plain `reg` in the body, and a concat of registers
        # routinely mixes the two.
        if _depth_free(expr, header + "\n" + body):
            continue
        comb.append(sig)
    return comb


def _depth_free(expr, body):
    """Is this driver depth the neighbour has to pay for?

    Two things look like expressions and cost nothing. A CONSTANT is
    wiring to a rail -- every AXI attribute port is one. A CONCATENATION
    of registers is also wiring: each bit still leaves its own flop, and
    the braces are notation, not logic.

    Anything else counts, including a single inverter over a register.
    One LUT is nearly free and `nearly free` is a claim about the module
    on the other side, made without looking at it -- which is the habit
    this whole file exists to break.
    """
    def is_reg(term):
        # Per LINE, because a declaration can carry initialisers
        # (`reg en0 = 1'b0, en1 = 1'b0;`) and an attribute in front of
        # it, and scanning to the next `;` walks straight past both.
        term = term.split("[")[0].strip()
        pat = re.compile(
            rf"^\s*(?:\(\*.*?\*\)\s*)?(?:output\s+)?reg\b.*\b"
            rf"{re.escape(term)}\b")
        return any(pat.match(line) for line in body.splitlines())

    def is_const(term):
        return bool(re.fullmatch(r"\d*'[bdho][0-9a-fA-FxzXZ_]+|\d+",
                                 term.strip()))

    if is_const(expr):
        return True
    if re.fullmatch(r"\w+", expr):
        return is_reg(expr)
    m = re.fullmatch(r"\{(.*)\}", expr)
    if m:
        terms = [t.strip() for t in m.group(1).split(",")]
        return all(is_const(t) or is_reg(t) for t in terms)
    return False


ok = []


def check(label, passed, detail=""):
    ok.append(bool(passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


def main():
    print("boundaries: what each core costs the one next to it\n")

    sk = skid(16)
    check("a slice hands nothing on, by construction",
          combinational_outputs(sk["interface"]) == ())

    fb = framebuffer_read(data_bits=32, clk_hz=148_500_000)
    check("the framebuffer engine still hands on its whole stream side",
          len(combinational_outputs(fb["interface"])) == 4,
          f"{', '.join(combinational_outputs(fb['interface']))} -- the "
          "prefetch FIFO's head, read combinationally, exactly the shape "
          "the receiver had before")

    sc = scanout()
    check("the raster hands back its ready",
          combinational_outputs(sc) == ("s_axis_tready",),
          "the video side leaves flops; the stream side does not, so "
          "this lands in the producer's path")

    # The composition, which is where the cost actually appears. A chain
    # of traced cores is the interesting case: each is individually
    # unremarkable and together they are a wire from the sink to the
    # source.
    core = traced_core()
    check("a traced core keeps its data but hands back its ready",
          combinational_outputs(core["interface"]) == ("in_ready",),
          "`out_*` leave the last pipeline register; `in_ready` is "
          "`!stall` and `stall` reads `out_ready`")

    # The fact and the RTL are two descriptions of one thing, so they
    # are two things that can disagree. Whichever is easier to edit will
    # drift, and that is the dict -- so it gets checked against the
    # Verilog rather than believed.
    for name, r in [("slice", sk),
                    ("traced core", core), ("framebuffer", fb)]:
        claimed = set(combinational_outputs(r.get("interface", r)))
        actual = set(_combinational_in_rtl(closure(r), r["module"]))
        check(f"the {name}'s published fact matches its RTL",
              claimed == actual,
              f"claims {sorted(claimed) or 'none'}; RTL says "
              f"{sorted(actual) or 'none'}")

    CHAIN = ["bl", "wb", "hg", "hr", "cc", "gm", "lg"]
    traced = core["interface"]
    design = ([("slice", sk["interface"])]
              + [(n, traced) for n in CHAIN]
              + [("scanout", sc)])
    report = boundary_report(design)
    names = [n for n, _ in report]

    check("the report names every core that hands depth on, and no other,"
          " in pipeline order",
          names == CHAIN + ["scanout"],
          f"{len(report)} of {len(design)} modules, and the order is the "
          f"path: {' -> '.join(names)}")
    check("and it clears the ones that do not",
          "slice" not in names,
          "the slice is absent because it earned it, not because it "
          "was skipped")

    # The number that matters is not how many cores are dirty but how
    # far the path runs without meeting a register.
    depth = sum(1 for n, p in report if "in_ready" in p)
    check("a seven-block chain passes ready through all seven",
          depth == len(CHAIN),
          f"{depth} cores deep, plus the raster's own tready at the "
          f"end -- and one deeper for every block the pipeline gains")

    print("\n" + ("BOUNDARIES PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
