# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""``np2hw pipeline``: run a pipeline, or generate its design pack.

    np2hw pipeline run acme.isp:core in.npy out.png
    np2hw pipeline run <design> --profile acme.sensors.s1:s1 in.npy out.npy
    np2hw pipeline run <design> --from gamma.in mid.npy out.png
    np2hw pipeline run <design> --rtl in.npy out.npy
    np2hw pipeline generate <design> --out build --clock-mhz 155

``run`` treats a design as a library of image functions: the whole
pipeline by default, or any consecutive run of blocks between ``--from``
and ``--to``. Splitting a run at a port and feeding the intermediate back
in is exact by construction -- both halves together are the same block
models in the same order -- and there is a test that says so.

``generate`` is the same design headed the other way: build it, prove it
bit-exact against its own models under Verilator, and emit the pack a
hardware flow consumes (Verilog, register map, SystemRDL, FuseSoC core).
It refuses to emit anything unverified.
"""
from __future__ import annotations

import sys

from np2hw.design.composite import LIBRARY_HELP


def add_commands(commands) -> None:
    """Add ``pipeline run`` and ``pipeline generate`` to a CLI's
    subcommands."""
    group = commands.add_parser(
        "pipeline", help="run a pipeline, or generate its design pack")
    commands = group.add_subparsers(dest="pipeline_command", required=True)

    run = commands.add_parser(
        "run",
        help="run a pipeline (or a sub-chain of it) over an image file",
        description="Run a design's NumPy models -- or, with --rtl, its "
                    "generated Verilog under Verilator -- over one frame.")
    run.add_argument("design", help="a composite, package.module:name")
    run.add_argument("input", help=".npy (raw integers) or .png (8-bit view)")
    run.add_argument("output", help=".npy or .png, by extension")
    run.add_argument("--profile", metavar="PACKAGE.MODULE:NAME",
                     help="a sensor's profile, applied at every instance of "
                          "the block it is for")
    run.add_argument("--from", dest="inject", metavar="INSTANCE.PORT",
                     help="inject the input at this port "
                          "(default: the pipeline input)")
    run.add_argument("--to", dest="extract", metavar="INSTANCE.PORT",
                     help="extract the output at this port "
                          "(default: the pipeline output)")
    run.add_argument("--set", dest="sets", action="append", default=[],
                     metavar="INSTANCE.REG=VALUE",
                     help="override one register (raw integer; repeatable)")
    run.add_argument("--context", dest="contexts", action="append",
                     default=[], metavar="SIGNAL=VALUE",
                     help="override one context signal at the injection "
                          "point (raw integer; repeatable)")
    run.add_argument("--rtl", action="store_true",
                     help="run the generated Verilog under Verilator and "
                          "verify it against the model word for word")
    run.add_argument("--explain", action="store_true",
                     help="print where every register value came from")
    run.add_argument("--library", action="append", metavar="DIR",
                     help=LIBRARY_HELP)
    run.set_defaults(handler=_run)

    generate = commands.add_parser(
        "generate",
        help="build a design, verify it, and emit its pack",
        description="Build a design, prove it bit-exact against its own "
                    "NumPy models under Verilator, and emit the pack: "
                    "Verilog, register map, SystemRDL, FuseSoC core "
                    "manifest. Refuses to emit anything unverified.")
    generate.add_argument("design", help="a composite, package.module:name")
    generate.add_argument("--library", action="append", metavar="DIR",
                          help=LIBRARY_HELP)
    generate.add_argument("--vlnv", default=None,
                          help="the core's VLNV for the manifest (default: "
                               "no vendor, the models' package, the "
                               "pipeline's name)")
    generate.add_argument("--out", default=".",
                          help="directory the pack lands in (default: here)")
    generate.add_argument("--clock-mhz", type=float, default=None,
                          help="the clock the core must make; every "
                               "pointwise stage is depth-checked against it "
                               "and a too-deep stage is cut into pipeline "
                               "stages, by arithmetic on the traced "
                               "expression graph (default: unchecked)")
    generate.add_argument("--no-control", action="store_true",
                          help="stop at the datapath: no AXI4-Lite register "
                               "file, coefficients as flat input ports -- "
                               "the testbench form")
    generate.add_argument("--no-verify", action="store_true",
                          help="skip the Verilator twin -- only for flows "
                               "that prove the same thing elsewhere")
    generate.add_argument("--sync-memory", action="store_true",
                          help="refuse any block whose line buffers read "
                               "combinationally: every memory in the pack "
                               "must sit behind np2hw's seam, where a "
                               "memory macro can be bound. The pack for "
                               "silicon, not for an FPGA")
    generate.set_defaults(handler=_generate)


def handle(arguments) -> int:
    """Run a ``pipeline`` subcommand; a refusal is a message and status 2."""
    try:
        return arguments.handler(arguments)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        message = error.args[0] if error.args else error
        print(f"np2hw pipeline {arguments.pipeline_command}: {message}",
              file=sys.stderr)
        return 2


def _run(arguments) -> int:
    import numpy as np

    from np2hw import pipeline as pipelines
    from np2hw.pipeline import compose
    from np2hw.pipeline import run as runner
    from np2hw.design.composite import add_libraries

    add_libraries(arguments.library)
    pipeline = pipelines.load(arguments.design)

    chain = runner.pixel_chain(pipeline, arguments.inject, arguments.extract)
    values, origin = runner.resolve_values(
        pipeline, arguments.profile, runner.parse_sets(arguments.sets))

    owner = compose.vocabulary(pipeline).block.name
    context = dict(values.get(owner, {}))
    context.update(runner.parse_context(arguments.contexts, pipeline))

    bit_depth = compose.shape(pipeline).spec.bit_depth
    frame = runner.read_frame(arguments.input, bit_depth)

    if arguments.explain:
        print(runner.explain(chain, values, origin))

    result = runner.run_model(chain, frame, values, context, bit_depth)

    if arguments.rtl:
        rtl = runner.run_rtl(chain, frame, values, context, bit_depth)
        total = result.size // (result.shape[-1] if result.ndim == 3 else 1)
        if not np.array_equal(rtl, result):
            bad = np.argwhere(~np.all(rtl == result, axis=-1)
                              if result.ndim == 3 else (rtl != result))[:5]
            where = ", ".join(f"({y},{x})" for y, x in bad)
            print(f"rtl: DIFFERS from the model, first at {where}",
                  file=sys.stderr)
            runner.write_frame(arguments.output, rtl, bit_depth)
            return 1
        print(f"rtl: bit-exact with the model ({total} words)")
        result = rtl

    runner.write_frame(arguments.output, result, bit_depth)
    return 0


def _generate(arguments) -> int:
    from np2hw.design.composite import add_libraries
    from np2hw.pipeline import pack

    add_libraries(arguments.library)

    written = pack.emit(arguments.design, arguments.out,
                        vlnv=arguments.vlnv,
                        control=not arguments.no_control,
                        clock_mhz=arguments.clock_mhz,
                        verify=not arguments.no_verify,
                        sync_memory=arguments.sync_memory)
    if arguments.clock_mhz is not None:
        print(f"timing: every pointwise stage fits "
              f"{1000.0 / arguments.clock_mhz:.1f} ns "
              f"({arguments.clock_mhz:g} MHz), by the traced depth model")
    for artifact in ("verilog", "regmap", "systemrdl", "docs", "core"):
        print(f"{artifact}: {written[artifact]}")
    print(f"toplevel: {written['toplevel']} "
          f"(latency {written['latency']} pixels)")
    return 0
