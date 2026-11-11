# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Statistics in a design: an ISP of model blocks forks its stream into a
fold beside its datapath, and the fold's result goes into the readout leaf
software reads it through -- sized by what the ISP's trace publishes of
that output, placed on the bus by the window the leaf states. Nothing
restates a size."""
import shutil
import subprocess

import pytest

from np2hw import hdl
from np2hw.design.composite import load
from np2hw.design.verilog import modules


def _isp():
    """in -> gain -> out, the gain's stream into a fold beside it."""
    return {"vocabulary": "toylib.blocks:VOCABULARY",
            "parameters": {"w": {"default": 16}},
            "inputs": [{"name": "in", "bits": 8, "max_width": "w",
                        "max_height": 8}],
            "outputs": [{"name": "out"}, {"name": "stats"}],
            "nodes": {"g": {"block": "toylib.blocks:gain"},
                      "m": {"block": "toylib.blocks:counts"}},
            "connections": [["in", "g.in"], ["g.out", "out"],
                            ["g.out", "m.in"], ["m.out", "stats"]]}


SOURCE = {"inputs": [{"name": "clk"}],
          "outputs": [{"name": "out", "width": 8,
                       "stream": ["sof", "eol", "last"]}],
          "vivado": {"module": "src"}, "sources": ["src.v"]}
SINK = {"inputs": [{"name": "in", "width": 8,
                    "stream": ["sof", "eol", "last"]}],
        "vivado": {"module": "dst"}, "sources": ["dst.v"]}


def _design(library):
    def files(name):
        top = {"inputs": [], "outputs": [],
               "pins": [{"name": "clk", "dir": "I"},
                        {"name": "rst", "dir": "I"},
                        {"name": "aclk", "dir": "I"},
                        {"name": "aresetn", "dir": "I"}],
               "nodes": {
                   "src": {"block": f"{name}.lib:source"},
                   "isp": {"block": f"{name}.lib:isp"},
                   "dst": {"block": f"{name}.lib:sink"},
                   # software's reading of the statistics: sized by the
                   # ISP's own facts of the output it reads
                   "ro": {"block": "np2hw.readout:readout", "build": {
                       "rows": "isp.stats.max_height",
                       "cols": "isp.stats.max_width",
                       "channels": "isp.stats.channels",
                       "bits": "isp.stats.bits",
                       "signed": "isp.stats.signed",
                       "context_bits": "isp.stats.context_bits"}},
                   "dec": {"block": "np2hw.bus:axil_decoder", "build": {
                       "sizes": ["isp.s_axil.window", "ro.s_axi.window"],
                       "base": 0x4000_0000, "span": 0x4000_0000}}},
               "connections": [
                   ["clk", "src.clk"], ["src.out", "isp.in"],
                   ["isp.out", "dst.in"], ["isp.stats", "ro.in"],
                   ["clk", "ro.clk"], ["rst", "ro.rst"],
                   ["aclk", "ro.aclk"], ["aresetn", "ro.aresetn"],
                   ["aclk", "dec.aclk"], ["aresetn", "dec.aresetn"],
                   ["dec.m00_axi", "isp.s_axil"],
                   ["dec.m01_axi", "ro.s_axi"]]}
        return {"lib": {"source": SOURCE, "sink": SINK, "isp": _isp(),
                        "top": top}}
    return load(f"{library(files)}.lib:top")


def test_the_readout_is_sized_by_the_isps_facts(library):
    design = _design(library)
    leaf = design.instances["ro"]
    shape = leaf.leaf["facts"]["layout"]
    # the counts' result: one row of two entries, one value each
    assert (shape["rows"], shape["cols"], shape["channels"]) == (1, 2, 1)
    assert shape["entries"] == 2 and not shape["signed"]
    windows = design.instances["dec"].leaf["facts"]["windows"]
    assert windows["m01_axi"][1] == 1 << shape["addr_bits"]


@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="needs verilator")
def test_what_is_written_lints(library, tmp_path):
    design = _design(library)
    files = []
    for name, text in modules(design):
        (tmp_path / f"{name}.v").write_text(text)
        files.append(f"{name}.v")
    for inst in design.instances.values():
        leaf = inst.leaf or {}
        for fname, text in leaf.get("files", {}).items():
            if fname.endswith(".v"):
                (tmp_path / fname).write_text(text)
                files.append(fname)
        for src in leaf.get("sources", ()):
            if src.startswith("np2hw.hdl:"):
                fname = src.split(":", 1)[1]
                (tmp_path / fname).write_text(hdl.text(fname[:-2]))
                files.append(fname)
    for stub, ports in (("src", "input wire clk, output wire out_valid, "
                                "input wire out_ready, output wire [7:0] "
                                "out_data, output wire out_sof, out_eol, "
                                "out_last"),
                        ("dst", "input wire in_valid, output wire in_ready, "
                                "input wire [7:0] in_data, input wire in_sof, "
                                "in_eol, in_last")):
        (tmp_path / f"{stub}.v").write_text(
            f"module {stub}({ports}); endmodule\n")
        files.append(f"{stub}.v")
    done = subprocess.run(
        ["verilator", "--lint-only", "-Wno-fatal", "-Wno-DECLFILENAME",
         "-Wno-UNUSEDSIGNAL", "-Wno-MULTITOP", "--top-module", "top",
         *sorted(set(files))],
        cwd=tmp_path, capture_output=True, text=True)
    errors = [l for l in done.stderr.splitlines() if l.startswith("%Error")]
    assert not errors, "\n".join(errors[:10])
