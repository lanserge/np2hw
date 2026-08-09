"""The FuseSoC generator protocol, end to end, without FuseSoC installed.

A build system integration is a contract, and a contract untested is a
contract broken at the consumer. This example plays FuseSoC's half of the
conversation exactly: write the generator-input file, invoke the generator
in a clean work root, and then hold what a consumer would hold --

  BUILDS        the generated core: a model file traced to Verilog in the
                work root, compiled by iverilog.
  MANIFEST      the emitted .core names the VLNV FuseSoC asked for, lists
                exactly the files that exist, and declares the toplevel.
  DETERMINISM   a second invocation in a second work root produces byte-
                identical Verilog -- reproducible builds are table stakes
                for a package manager.
  REFUSAL       missing geometry parameters are refused with their names;
                hardware sized by a defaulted guess is not hardware.

Run:  python examples/fusesoc_gen.py   (needs iverilog)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from np2hw.fusesoc import main as generator

MODEL = '''\
import numpy as np
from np2hw.ir import Param

PARAMS = [Param("offset", bits=8, signed=True,
                description="signed offset, saturated")]

def model(img, offset):
    return (img.astype(np.int16) + offset).clip(0, 255).astype(np.uint8)
'''


def run_generator(root: Path, parameters: dict, vlnv="lanserge:demo:offset:0",
                  work_name="work"):
    """Play FuseSoC: write the gapi input, run the generator in a work root."""
    work = root / work_name
    work.mkdir()
    # JSON, which is valid YAML -- so this example runs with or without
    # PyYAML while real FuseSoC input (always YAML) needs it.
    gapi = root / f"{work_name}-input.yml"
    gapi.write_text(json.dumps({
        "gapi": "1.0", "vlnv": vlnv,
        "files_root": str(root), "parameters": parameters,
    }))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        generator([str(gapi)])
    finally:
        os.chdir(cwd)
    return work


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "offset.py").write_text(MODEL)
        params = {"model": "offset.py", "width": 8, "height": 4, "bits": 8}

        work = run_generator(root, params)
        verilog = work / "model.v"
        core = work / "offset.core"
        built = verilog.exists() and core.exists()
        print(f"  generated core + manifest exist: {built}")
        ok &= built

        compiled = subprocess.run(
            ["iverilog", "-g2012", "-o", os.devnull, str(verilog)]).returncode == 0
        print(f"  iverilog compiles the generated core: {compiled}")
        ok &= compiled

        text = core.read_text()
        manifest = ("name: lanserge:demo:offset:0" in text
                    and "- model.v" in text and "toplevel: model" in text)
        print(f"  manifest names the requested VLNV, the file, the top: "
              f"{manifest}")
        ok &= manifest

        work2 = run_generator(root, params, work_name="again")
        same = (work2 / "model.v").read_text() == verilog.read_text()
        print(f"  second invocation is byte-identical: {same}")
        ok &= same

        try:
            run_generator(root, {"model": "offset.py"}, work_name="refused")
            refused = False
        except SystemExit as error:
            refused = "width" in str(error) and "bits" in str(error)
        print(f"  missing geometry refused by name: {refused}")
        ok &= refused

    print("\n" + ("FUSESOC GENERATOR PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
