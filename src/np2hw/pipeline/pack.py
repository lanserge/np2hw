# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A pipeline's design pack: built, proven, and written for a hardware flow.

:func:`emit` builds a pipeline, proves its Verilog bit-exact against its
own NumPy models under Verilator, and writes what a hardware flow consumes:
the Verilog, the register map, the SystemRDL, the register documentation, a
build summary and a FuseSoC ``.core`` manifest. ``np2hw pipeline generate``
and the FuseSoC generator (``np2hw-fusesoc`` with a ``design`` parameter)
both call it, so there is one emitter.
"""
from __future__ import annotations

import json
from pathlib import Path


def emit(design, output_dir, vlnv: str | None = None, control: bool = True,
         clock_mhz: float | None = None, verify: bool = True,
         sync_memory: bool = False) -> dict:
    """Build a design and write its pack: Verilog, maps, manifest.

    Args:
        design: a composite's reference, package.module:name -- or a
            design already loaded, of model blocks: a composite built whole
            inside a larger design, as that design sets it
            (``Design.pipelines``).
        output_dir: where the pack lands. Created if absent.
        vlnv: the core's VLNV for the manifest; by default no vendor, the
            models' package as the library and the pipeline's own name.
        control: emit the AXI4-Lite control plane in front of the datapath
            (the shippable form). ``False`` stops at the datapath, which is
            what a bit-exact testbench wants.
        clock_mhz: the clock the core must make. Every pointwise stage is
            depth-checked against it at generation time by arithmetic on
            the traced expression graph, and a too-deep stage is cut into
            pipeline stages there. None generates unchecked -- for a
            simulation-only build, never for one headed at a bitstream.
        verify: run the design's Verilog under Verilator against its own
            NumPy models before writing anything, on a synthetic frame with
            synthetic non-degenerate register values, and REFUSE the pack on
            any mismatch. On by default because this is the last gate
            before a hardware tool: an unverified pipeline should not reach
            a bitstream through packaging. ``False`` is for consumers that
            prove the same thing elsewhere (a packaging test, a flow whose
            own testbench is the twin).

    Returns:
        ``{artifact: Path}`` for everything written -- Verilog, register
        map, SystemRDL, register documentation, ``.core`` manifest --
        plus two facts about the build for whoever instantiates it:
        ``"toplevel"`` (the generated top module's name) and
        ``"latency"`` (its latency in pixels).
    """
    from np2hw.fusesoc import write_core
    from np2hw import pipeline as pipelines
    from np2hw.pipeline import compose
    from np2hw.pipeline.compose import register_map_markdown

    if isinstance(design, str):
        pipeline = pipelines.load(design)
    else:
        pipeline = design
        compose.netlist(pipeline)
    form = compose.shape(pipeline)

    if verify:
        # Generation and verification are ONE step: the same composition
        # runs under Verilator against the same block models. The values
        # are synthetic on purpose -- deterministic, nudged off every
        # transparent reset so the arithmetic is exercised, and never
        # anyone's calibration.
        import numpy as np

        from np2hw.pipeline import run as runner

        values = runner.synthetic_values(pipeline)
        rng = np.random.default_rng(20260813)
        # EVERY input's chain, not "the" chain. Asking for the sole input
        # was right for a mono pipeline and refused outright on a stereo
        # one -- which is the design that most wants proving, since it is
        # the one where a leg can be wrong on its own. A pipeline with two
        # sensors has two chains, and both are verified before anything is
        # emitted.
        clk_ns = None if clock_mhz is None else 1000.0 / clock_mhz
        for name in pipeline.inputs:
            chain = runner.pixel_chain(pipeline, name, None)
            # a chain reading the stream's history is proved over a run of
            # frames, each with its own context, as the stream brings it --
            # and every chain at the clock it is cut for, the stages the
            # hardware has
            run = runner.FRAMES_PROVED if runner.reads_history(chain) else None
            frame = rng.integers(0, 1 << form.spec.bit_depth,
                                 size=((run,) if run else ())
                                 + (form.height, form.width), dtype=np.uint16)
            context = runner.synthetic_context(pipeline, run)
            model = runner.run_model(chain, frame, values, context,
                                     form.spec.bit_depth)
            rtl = runner.run_rtl(chain, frame, values, context,
                                 form.spec.bit_depth, clk_ns=clk_ns)
            if not np.array_equal(model, rtl):
                raise ValueError(
                    f"{pipeline.name}: the generated RTL DIFFERS from the "
                    f"model on the chain from {name!r}; refusing to emit "
                    "an unverified pack")
            where = "" if len(pipeline.inputs) == 1 else f" from {name}"
            print(f"{pipeline.name}: twin bit-exact with the model"
                  f"{where} ({model.size} words)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = compose.generate(
        pipeline,
        control=control,
        clk_ns=None if clock_mhz is None else 1000.0 / clock_mhz,
        sync_memory=sync_memory)
    written = {
        "verilog": output_dir / f"{pipeline.name}.v",
        "regmap": output_dir / f"{pipeline.name}_regmap.json",
    }
    written["verilog"].write_text(generated.verilog)
    compose.write_register_map(pipeline, written["regmap"])
    written["systemrdl"] = compose.write_systemrdl(
        pipeline, output_dir / f"{pipeline.name}.rdl")
    # The documentation is the map in prose, and it ships WITH the map:
    # whoever reads a register in review reads this, and a pack whose
    # docs were rendered by a separate run is a pack whose docs can
    # describe a different build.
    written["docs"] = output_dir / f"{pipeline.name}-registers.md"
    written["docs"].write_text(register_map_markdown(compose.register_map(pipeline)))

    written["core"] = write_core(
        output_dir,
        vlnv or f":{compose.module_prefix(pipeline)}:{pipeline.name}:0",
        {
            "rtl": {"files": [written["verilog"].name],
                    "file_type": "verilogSource"},
            # The maps ride along as data: the register map JSON is what
            # hosts and tests read, the SystemRDL is the same allocation for
            # the integrator's register tooling.
            "maps": {"files": [written["regmap"].name,
                               written["systemrdl"].name],
                     "file_type": "user"},
            "docs": {"files": [written["docs"].name],
                     "file_type": "user"},
        },
        toplevel=generated.top,
        description=f"design pack for {pipeline.name}, generated by np2hw",
    )
    written["toplevel"] = generated.top
    # Not a file: the pixel latency through the core, which whoever
    # instantiates it needs and would otherwise re-derive by building
    # the design a second time.
    written["latency"] = generated.latency
    # Also not a file, and the reason this is published at all: the width
    # of the word each output carries is TRACED, so an integrator that
    # writes it down instead of reading it has a second owner of a fact
    # that changes whenever the models do.
    written["boundary"] = generated.meta["boundary"]
    # ...and on disk too, because the integrator here is a shell script
    # and a block design, neither of which can import Python.
    written["build"] = output_dir / f"{pipeline.name}_build.json"
    written["build"].write_text(json.dumps({
        "toplevel": generated.top,
        "latency": generated.latency,
        # What the cores were BUILT around, which is the maximum they can
        # process: the line buffers are this deep and the pointwise cores
        # reframe every line from it. An integrator handed a wider frame
        # must crop before this pack sees it -- the cores receive neither
        # the stream's end-of-line nor the header's width, so they cannot
        # notice, and the failure is a sheared picture with the CFA phase
        # inverted on alternate rows rather than an error.
        "geometry": {"width": form.width, "height": form.height},
        "boundary": generated.meta["boundary"],
        # Every memory the pack infers, per stage: instance, the seam
        # module it sits behind (or null), width, depth, and whether its
        # read is synchronous. An integrator binding macros starts here.
        "memories": generated.meta.get("memories", {}),
    }, indent=2) + "\n")
    return written
