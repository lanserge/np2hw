"""FuseSoC integration: the generator protocol, and the .core writer.

FuseSoC's *generators* mechanism is built for exactly what np2hw is: a
consuming core file names a generator and parameters, and FuseSoC calls the
tool at build time, collecting the files it writes plus a ``.core`` file
describing them. This module owns both halves of that conversation once:

  * :func:`read_generator_input` -- parse the file FuseSoC hands a generator
    (the "gapi 1.0" YAML: ``files_root``, ``vlnv``, ``parameters``).
  * :func:`write_core` -- emit a CAPI=2 ``.core`` file for a set of written
    files. Plain text, deterministic, no YAML library needed to WRITE.
  * :func:`main` -- np2hw's own generator: trace a user model file (the same
    ``file.py[:func]`` convention as the CLI -- one convention, stated once)
    into Verilog in the work root.

Downstream projects that compose np2hw output (revela's pipelines, for
instance) register their own thin generator and call :func:`write_core`
here, so the packaging protocol has one implementation.

The input file is YAML. PyYAML ships with FuseSoC itself, so a real
invocation always has it; ``pip install np2hw[fusesoc]`` declares it
explicitly. Without PyYAML the reader still accepts JSON input (YAML is a
superset of JSON), which is what the example suite uses.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GAPI = "1.0"


def read_generator_input(path) -> dict:
    """The dict FuseSoC passed to this generator invocation.

    Args:
        path: the single command-line argument FuseSoC gives a generator --
            a YAML file with ``gapi``, ``vlnv``, ``files_root`` and
            ``parameters``.
    """
    text = Path(path).read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise SystemExit(
                "this generator input is YAML and PyYAML is not installed; "
                "run inside a FuseSoC environment (which ships it) or "
                "install the extra: pip install np2hw[fusesoc]") from None
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping, got {type(data).__name__}")
    gapi = str(data.get("gapi", GAPI))
    if gapi.split(".")[0] != GAPI.split(".")[0]:
        raise SystemExit(f"{path}: generator API {gapi!r} is not {GAPI!r}")
    data.setdefault("parameters", {})
    data.setdefault("files_root", ".")
    return data


def write_core(directory, name: str, filesets: dict, toplevel: str | None = None,
               description: str = "") -> Path:
    """Write a CAPI=2 ``.core`` file describing files this tool produced.

    Deterministic plain-text emission -- a package manifest is a product
    surface like generated Verilog, so it is written to be read.

    Args:
        directory: where the core file goes (the generator work root).
        name: the core's VLNV, e.g. ``lanserge:revela:mono:0`` -- for a
            generated core, verbatim what the input's ``vlnv`` said.
        filesets: ``{fileset_name: {"files": [...], "file_type": "..."}}``.
            Order is preserved; every named file should already exist next
            to the core file.
        toplevel: module for the default target, if the core has one.
        description: one line for the manifest.
    """
    directory = Path(directory)
    lines = ["CAPI=2:", f"name: {name}"]
    if description:
        lines.append(f"description: {description}")
    lines += ["", "filesets:"]
    for fileset, spec in filesets.items():
        lines.append(f"  {fileset}:")
        lines.append("    files:")
        for file in spec["files"]:
            missing = not (directory / file).exists()
            if missing:
                raise FileNotFoundError(
                    f"core file would name {file!r}, which was not written -- "
                    "a manifest that lists files that do not exist fails at "
                    "the consumer, which is the worst place")
            lines.append(f"      - {file}")
        lines.append(f"    file_type: {spec['file_type']}")
    lines += ["", "targets:", "  default:",
              f"    filesets: [{', '.join(filesets)}]"]
    if toplevel:
        lines.append(f"    toplevel: {toplevel}")
    lines.append("")
    # The file takes its basename from the core name so several generated
    # cores can share one work root without clobbering each other.
    out = directory / (name.split(":")[2] + ".core" if name.count(":") == 3
                       else "generated.core")
    out.write_text("\n".join(lines))
    return out


def main(argv=None) -> int:
    """np2hw as a FuseSoC generator: NumPy model file in, Verilog core out.

    Parameters (in the consuming core's ``generate`` section):
        model: path to the model ``.py``, relative to the calling core
            (``file.py`` or ``file.py:function`` -- the CLI's convention).
        width, height, bits: input image geometry. Required: hardware is
            sized from its input, and a default geometry would be a guess
            wearing a build system.
        module_name: emitted Verilog module (default: the traced function's
            name).
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        raise SystemExit("usage: np2hw-fusesoc <generator-input.yml>")
    data = read_generator_input(argv[0])
    parameters = data["parameters"]

    missing = [key for key in ("model", "width", "height", "bits")
               if key not in parameters]
    if missing:
        raise SystemExit(f"generator parameters missing {missing}; "
                         "model, width, height and bits are required")

    from np2hw import Image2D, generate, to_ir
    from np2hw.cli import _load_model

    model_path = os.path.join(data["files_root"], str(parameters["model"]))
    fn, params, label = _load_model(model_path)
    module = str(parameters.get("module_name", fn.__name__))

    image = Image2D("img", int(parameters["width"]), int(parameters["height"]),
                    bits=int(parameters["bits"]))
    _, line = to_ir(fn, image, *params)
    core = generate(line, module_name=module)

    verilog = Path(f"{module}.v")
    verilog.write_text(core.verilog)
    write_core(Path.cwd(), str(data.get("vlnv") or f"::{module}:0"),
               {"rtl": {"files": [verilog.name],
                        "file_type": "verilogSource"}},
               toplevel=module,
               description=f"generated by np2hw from {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
