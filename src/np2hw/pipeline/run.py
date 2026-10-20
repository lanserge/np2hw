# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Run a pipeline -- or any sub-chain of it -- as an image function.

A design already carries everything this needs: the netlist says which block
feeds which, and every block carries its own NumPy model. Composing those
gives each design a second life as a library of image functions: the whole
pipeline is one function, and any run of consecutive blocks is another --
inject a frame at one port, extract at another.

Injection is governed by the same contract as NumPy itself: the model's
arithmetic states its own requirements, and the person injecting mid-chain
is trusted to know what the port eats. An array the math cannot digest
fails inside the model; an array it CAN digest is accepted, meaningful or
not. That is a deliberate trade -- just math, no meaning tags -- and it is
the same one every NumPy user already lives with.

Register values layer exactly as they do on hardware (the design's reset,
then a sensor's profile -- see :mod:`np2hw.design.profile`), with one CLI
layer on top:
``--set instance.register=value``, checked against the declared range like
every other layer. ``explain`` reports where each value came from, because
"where did this number come from" is the first question a strange picture
raises.

The RTL twin runs the same sub-chain through generated Verilog under
Verilator, fed and compared word-for-word against the model. Verilator and
not Icarus: the composed first-light pipeline simulated a full 1296x972
frame in under a second compiled, where the event-driven run had not
reached time zero in an hour. Bit-exactness over full frames is this
project's definition of correct, so the twin is built to be cheap enough
to run on every tuning iteration.
"""
from __future__ import annotations

import hashlib
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
from np2hw.netlist import Endpoint

from np2hw.pipeline import compose

# Where compiled Verilator twins live, keyed by content hash: same design,
# same geometry, same values -> the build is already there. A tuning loop
# pays the C++ compile once per STRUCTURE, not once per parameter change
# (values travel in a side file the harness reads at run time).
CACHE_ROOT = Path.home() / ".cache" / "np2hw" / "verilator"


# --------------------------------------------------------------------------- #
# Chain selection
# --------------------------------------------------------------------------- #

def pixel_chain(pipeline, start: str | None = None, stop: str | None = None):
    """The consecutive pixel-carrying stages between two ports.

    Args:
        pipeline: a loaded design (:func:`revela.designs.load`).
        start: ``instance.port`` naming an INPUT port to inject at, or None
            for the pipeline's own input.
        stop: ``instance.port`` naming an OUTPUT port to extract at, or None
            for the pipeline's own output.

    Statistics taps are not part of any chain: they observe the stream and
    never transform it, which is exactly what the netlist's tap notion
    already records -- so the walk asks the netlist rather than keeping its
    own list of which blocks merely watch.
    """
    if start is None:
        if len(pipeline.inputs) != 1:
            raise ValueError(
                f"pipeline {pipeline.name!r} has inputs {list(pipeline.inputs)}; "
                "name the injection point with --from")
        first = _sole_consumer(pipeline, Endpoint(None, pipeline.inputs[0]))
    elif start in pipeline.inputs:
        # A PIPELINE INPUT by name, which is what the refusal above tells a
        # caller to supply and therefore what they will type. Accepting only
        # `instance.port` here meant the advice named a form the code would
        # not take.
        first = _sole_consumer(pipeline, Endpoint(None, start))
    else:
        instance, port = _split_port(start)
        stage = compose.stage(pipeline, instance)
        if port not in stage.block.ports.inputs:
            raise ValueError(
                f"--from {start!r}: {instance!r} has no input port {port!r}; "
                f"its inputs are {list(stage.block.ports.inputs)}")
        first = stage

    stop_instance = stop_port = None
    if stop is not None:
        stop_instance, stop_port = _split_port(stop)
        stage = compose.stage(pipeline, stop_instance)
        if stop_port not in stage.block.ports.outputs:
            raise ValueError(
                f"--to {stop!r}: {stop_instance!r} has no output port "
                f"{stop_port!r}; its outputs are "
                f"{list(stage.block.ports.outputs)}")

    chain = Chain([first], compose.vocabulary(pipeline))
    while True:
        stage = chain[-1]
        if stage.path == stop_instance:
            return chain
        outputs = stage.block.ports.outputs
        if not outputs:
            raise ValueError(
                f"the chain dead-ends at {stage.path!r}, which is a sink"
                + (f", before reaching --to {stop!r}" if stop else
                   " -- it never reaches the pipeline output"))
        if len(outputs) > 1:
            raise ValueError(
                f"{stage.path!r} has outputs {list(outputs)}; a forking chain "
                "has no single downstream -- name the leg with --to")
        consumers = compose.netlist(pipeline).consumers(Endpoint(stage.path, outputs[0]))
        if stop is None and any(c.node is None for c in consumers):
            return chain
        onward = [c for c in consumers
                  if c.node is not None and not compose.netlist(pipeline).is_tap(c)]
        if not onward:
            raise ValueError(
                f"nothing downstream of {stage.path}.{outputs[0]} carries "
                f"pixels" + (f"; --to {stop!r} is not on this chain"
                             if stop else ""))
        if len(onward) > 1:
            raise ValueError(
                f"{stage.path}.{outputs[0]} forks to "
                f"{[str(c) for c in onward]}; name the leg with --to")
        chain.append(compose.stage(pipeline, onward[0].node))


def _sole_consumer(pipeline, source: Endpoint):
    consumers = [c for c in compose.netlist(pipeline).consumers(source)
                 if c.node is not None and not compose.netlist(pipeline).is_tap(c)]
    if len(consumers) != 1:
        raise ValueError(
            f"{source} feeds {[str(c) for c in consumers]}; "
            "name the injection point with --from")
    return compose.stage(pipeline, consumers[0].node)


def _split_port(text: str) -> tuple[str, str]:
    instance, dot, port = text.rpartition(".")
    if not dot:
        raise ValueError(
            f"{text!r} does not name a port; ports are written "
            "instance.port, e.g. ha_green.out")
    return instance, port


# --------------------------------------------------------------------------- #
# Values: reset -> sensor -> profile -> --set, with provenance
# --------------------------------------------------------------------------- #

FROM_DEFAULT = "default"
FROM_PROFILE = "profile"
FROM_CLI = "cli"


def resolve_values(pipeline, profile: str | None = None,
                   sets: dict[str, dict[str, int]] | None = None):
    """Every register value for the run, and where each one came from.

    The design's reset values first -- a design's own values ARE its reset
    -- then a profile, by reference, anchored at every instance of the block
    it is for and checked against the map the way the board's tools check
    it; then the CLI's ``--set``, one more layer with the same range checks,
    never a bypass.
    """
    from np2hw.design import profile as profiles

    values = {stage.path: dict(stage.paramset.defaults())
              for stage in compose.stages(pipeline)}
    origin = {path: {name: FROM_DEFAULT for name in registers}
              for path, registers in values.items()}
    if profile is not None:
        anchored = profiles.flatten(profiles.resolve(profile),
                                    compose.register_map(pipeline))
        for key, value in anchored.items():
            path, _, name = key.rpartition(".")
            _assign(compose.stage(pipeline, path), values, origin, name, value,
                    FROM_PROFILE)
    for path, registers in (sets or {}).items():
        stage = compose.stage(pipeline, path)
        for name, value in registers.items():
            _assign(stage, values, origin, name, value, FROM_CLI)
    return values, origin


def _assign(stage, values, origin, name, value, source) -> None:
    if name not in values[stage.path]:
        raise KeyError(
            f"block {stage.path!r} has no register {name!r}; "
            f"it declares {sorted(values[stage.path])}")
    param = stage.paramset.param(name)
    low, high = param.limits
    if not low <= int(value) <= high:
        raise ValueError(
            f"{stage.path}.{name} = {value} is outside [{low}, {high}] for a "
            f"{param.bits}-bit {'signed' if param.signed else 'unsigned'} "
            f"register ({param.q_format}). {param.description}")
    values[stage.path][name] = int(value)
    origin[stage.path][name] = source


def parse_sets(entries) -> dict[str, dict[str, int]]:
    """``instance.register=value`` strings into ``{instance: {register: int}}``."""
    out: dict[str, dict[str, int]] = {}
    for entry in entries or ():
        name, eq, raw = entry.partition("=")
        if not eq:
            raise ValueError(f"--set {entry!r}: expected instance.register=value")
        instance, register = _split_port(name)
        try:
            value = int(raw, 0)
        except ValueError:
            raise ValueError(
                f"--set {entry!r}: {raw!r} is not an integer; registers hold "
                "raw fixed-point values -- quantise on the host first") from None
        out.setdefault(instance, {})[register] = value
    return out



def parse_context(entries, pipeline) -> dict[str, int]:
    """``signal=value`` strings into ``{signal: int}``, each signal one the
    pipeline's vocabulary declares and each value within its bits."""
    signals = {c.name: c for c in compose.vocabulary(pipeline).signals}
    out: dict[str, int] = {}
    for entry in entries or ():
        name, eq, raw = entry.partition("=")
        if not eq:
            raise ValueError(f"--context {entry!r}: expected signal=value")
        if name not in signals:
            raise ValueError(f"--context {entry!r}: the pipeline's context "
                             f"has no {name!r}; it has {sorted(signals)}")
        try:
            value = int(raw, 0)
        except ValueError:
            raise ValueError(
                f"--context {entry!r}: {raw!r} is not an integer") from None
        bits = signals[name].bits
        low, high = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                     if signals[name].signed else (0, (1 << bits) - 1))
        if not low <= value <= high:
            raise ValueError(f"--context {entry!r}: {name} holds "
                             f"{low}..{high}")
        out[name] = value
    return out

def _vocabulary(chain):
    """The pipeline context a chain was cut from: its design's."""
    return chain.vocabulary


class Chain(list):
    """The stages of a run, in order, with the pipeline context of the
    design they were cut from -- what a twin of them is built around."""

    def __init__(self, stages, vocabulary):
        super().__init__(stages)
        self.vocabulary = vocabulary


def explain(chain, values, origin) -> str:
    """Provenance for every register the run reads, one line each.

    Context first: the context block is what every block sees, and it is
    where a sensor-derived surprise (the wrong CFA order, say) will show up.
    """
    owner = _vocabulary(chain).block.name
    paths = ([owner] if owner in values else []) + [s.path for s in chain]
    lines = []
    for path in paths:
        for name in sorted(values[path]):
            lines.append(f"{path}.{name} = {values[path][name]}"
                         f"  ({origin[path][name]})")
    return "\n".join(lines)


def synthetic_values(pipeline, seed: int = 20260821):
    """Register values for TWIN VERIFICATION: deterministic, non-degenerate,
    and never anyone's calibration.

    Every reset value is nudged a little way off itself, within the
    register's declared range -- because the resets are deliberately
    transparent (unity gains, an identity matrix, a ramp LUT), and a
    transparent value masks the arithmetic it configures. An identity
    matrix never exercises its cross-terms; a unity gain never carries
    into its integer bits. The nudge is small (about 1.5% of the range)
    so the values stay in each block's plausible operating region rather
    than pinning everything to a saturation corner.

    The context block keeps its resets: perturbing the crop window or the
    maximum geometry does not exercise more arithmetic, it just shrinks the
    frame the twins compare.
    """
    rng = np.random.default_rng(seed)
    values = {}
    owner = compose.vocabulary(pipeline).block.name
    for stage in compose.stages(pipeline):
        registers = dict(stage.paramset.defaults())
        if stage.path != owner:
            for name, default in registers.items():
                param = stage.paramset.param(name)
                low, high = param.limits
                step = max(1, (high - low) >> 6)
                value = int(default) + int(rng.integers(-step, step + 1))
                value = min(max(value, low), high)
                if value == default:
                    value = (default - step if default - step >= low
                             else default + step)
                registers[name] = value
        values[stage.path] = registers
    return values



def synthetic_context(pipeline) -> dict:
    """Context values for TWIN VERIFICATION, read from the vocabulary.

    A signal a transport or software may set (``origin="either"``) and
    that follows no fact of the frame is moved off its reset by flipping
    its top bit -- a reset of zero masks whatever the signal indexes, a
    colour-filter phase say. A register-only choice (a crop) keeps its
    reset, for the reason :func:`synthetic_values` gives, and so does a
    fact the stream delivers anew every frame, which one frame cannot
    stand for.
    """
    return {c.name: c.default ^ (1 << (c.bits - 1))
            for c in compose.vocabulary(pipeline).signals
            if c.origin == "either" and not c.follows}

# --------------------------------------------------------------------------- #
# The model run
# --------------------------------------------------------------------------- #

def run_model(chain, frame: np.ndarray, values, context: dict,
              bit_depth: int) -> np.ndarray:
    """Run the chain's NumPy models in order. Every hop is the block's own
    :meth:`np2hw.blocks.Block.run`, so all its declaration-driven checks
    apply here too.

    Each block is told the stream it is GIVEN -- its depth, and the frame
    it is built for -- read from its driver's trace exactly as the composer
    reads it (:func:`np2hw.pipeline.compose.handed_on`): a block that
    saturates back hands on what it received, a display curve hands on
    less, a block that widens hands on more. Nothing declares any of it; the
    trace is its one owner. A model whose frame comes out other than its
    trace says is refused: the model and its hardware disagree.
    """
    from np2hw.blocks.stream import StreamSpec

    height, width = frame.shape[:2]
    spec = StreamSpec(bit_depth=bit_depth,
                      channels=1 if frame.ndim == 2 else int(frame.shape[-1]))
    for stage in chain:
        consumed = {name: context[name]
                    for name in stage.block.params.consumes if name in context}
        frame = stage.block.run(frame, values.get(stage.path),
                                bit_depth=spec.bit_depth, **consumed)
        if not stage.block.traceable:
            continue
        built = stage.block.generate(
            spec, width, height, module_name=f"run_{stage.module_prefix}")
        handed = compose.handed_on(built)
        got = (frame.shape[:2], 1 if frame.ndim == 2 else frame.shape[-1])
        traced = ((handed.height, handed.width), handed.channels)
        # a window's frame is its registers', anything up to the trace's
        inside = (compose.keeps_window(built) and got[1] == traced[1]
                  and all(g <= t for g, t in zip(got[0], traced[0])))
        if got != traced and not inside:
            raise ValueError(
                f"{stage.path}: the model hands on a {got[0][1]}x{got[0][0]} "
                f"frame of {got[1]} channel(s); its trace says "
                f"{handed.width}x{handed.height} of {handed.channels}")
        spec = spec.with_stream(handed.channels, handed.bit_depth)
        # the next block is traced at the frame it is given: the trace's,
        # or the part of it a window kept
        height, width = got[0]
    return frame


# --------------------------------------------------------------------------- #
# Frame files: .npy for exact integers, .png for eyeballs
# --------------------------------------------------------------------------- #

def read_frame(path: str | Path, bit_depth: int) -> np.ndarray:
    """Load a frame. ``.npy`` carries raw register-domain integers verbatim;
    ``.png`` is an 8-bit view scaled up to the datapath's ``bit_depth``."""
    path = Path(path)
    if path.suffix == ".npy":
        frame = np.load(path)
        if not np.issubdtype(frame.dtype, np.integer):
            raise ValueError(
                f"{path}: dtype {frame.dtype} -- a datapath frame holds raw "
                "integers; quantise floats on the host first")
        return frame.astype(np.int64)
    if path.suffix == ".png":
        return _png_read(path).astype(np.int64) << (bit_depth - 8)
    raise ValueError(f"{path}: expected .npy or .png")


def write_frame(path: str | Path, frame: np.ndarray, bit_depth: int) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, frame)
        return
    if path.suffix == ".png":
        if frame.ndim == 3 and frame.shape[-1] == 2:
            raise ValueError(
                "a 2-channel frame has no PNG meaning; write .npy "
                "and feed it back with --from")
        _png_write(path, (frame >> (bit_depth - 8)).astype(np.uint8))
        return
    raise ValueError(f"{path}: expected .npy or .png")


def _png_write(path: Path, image: np.ndarray) -> None:
    if image.ndim == 2:
        image = image[..., None]
    height, width, channels = image.shape
    colour = {1: 0, 3: 2}[channels]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body)))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour,
                                     0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def _png_read(path: Path) -> np.ndarray:
    """8-bit greyscale or RGB, non-interlaced. Written here rather than
    imported: np2hw's only array dependency is numpy, and a viewer format
    does not justify a second one."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, header = 8, b"", None
    while pos < len(data):
        (length,), tag = struct.unpack(">I", data[pos:pos + 4]), data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, colour, _, _, interlace = header
    if depth != 8 or colour not in (0, 2) or interlace:
        raise ValueError(
            f"{path}: only 8-bit greyscale or RGB PNG is read here "
            f"(depth {depth}, colour type {colour})")
    channels = 3 if colour == 2 else 1
    stride = width * channels
    raw = zlib.decompress(idat)
    out = np.zeros((height, stride), np.uint8)
    previous = np.zeros(stride, np.int64)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        line = np.frombuffer(
            raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int64)
        if kind == 0:
            recon = line
        elif kind == 2:                                   # up
            recon = (line + previous) & 0xff
        elif kind in (1, 3, 4):                           # sub / average / paeth
            recon = np.zeros(stride, np.int64)
            for x in range(stride):
                left = recon[x - channels] if x >= channels else 0
                above = previous[x]
                corner = previous[x - channels] if x >= channels else 0
                if kind == 1:
                    predict = left
                elif kind == 3:
                    predict = (left + above) // 2
                else:
                    p = left + above - corner
                    predict = min((left, above, corner),
                                  key=lambda v: abs(p - v))
                recon[x] = (line[x] + predict) & 0xff
        else:
            raise ValueError(f"{path}: unknown PNG filter {kind}")
        out[y] = recon
        previous = recon
    image = out.reshape(height, width, channels)
    return image[..., 0] if channels == 1 else image


# --------------------------------------------------------------------------- #
# The RTL twin
# --------------------------------------------------------------------------- #

def run_rtl(chain, frame: np.ndarray, values, context: dict,
            bit_depth: int) -> np.ndarray:
    """The same sub-chain, through generated Verilog under Verilator.

    The chain is composed into a throwaway pipeline at the FRAME's geometry
    (line buffers are sized at synthesis; a twin built wide for a small test
    frame would not be the hardware the test claims to exercise), generated,
    compiled once per structure, and fed the same words the model consumed.
    """
    from np2hw.pipeline import compose
    from np2hw.design.composite import assemble

    height, width = frame.shape[:2]
    in_channels = 1 if frame.ndim == 2 else int(frame.shape[-1])
    # A context with no register exists only where the stream delivers it,
    # so the twin takes from its "stream" -- the values the caller passes --
    # every stream-only context its chain consumes.
    consumed = {name for stage in chain for name in stage.block.params.consumes}
    from_stream = [c.name for c in _vocabulary(chain).signals
                   if c.origin == "stream" and c.name in consumed]
    # The chain as a design of its own: the same block instances, wired in
    # a line between one input and one output, entered by the frame's
    # stream. It is built by the same steps as any design.
    edges, source = [], "chain_in"
    for stage in chain:
        edges.append((source, f"{stage.path}.{stage.block.ports.inputs[0]}"))
        source = f"{stage.path}.{stage.block.ports.outputs[0]}"
    edges.append((source, "chain_out"))
    twin = assemble(
        "chain", {stage.path: stage.block for stage in chain}, edges,
        inputs=("chain_in",), outputs=("chain_out",),
        stream={"bits": bit_depth, "max_width": width, "max_height": height,
                "channels": in_channels, "context": from_stream},
        vocabulary=chain.vocabulary)
    generated = compose.generate(twin, control=False)
    verilog = "\n".join(text for _, text in generated.modules)

    # The top's ports as np2hw published them -- the ctx_*/param_* inputs
    # the twin drives, and the output word -- never read back from the text.
    ports = {p["name"]: p for p in generated.meta["ports"]}
    scraped = [name for name, p in ports.items() if p["direction"] == "input"
               and (name.startswith("ctx_") or name.startswith("param_"))]
    port_values = _port_values(twin, scraped, values, context,
                               width, height, bit_depth)

    # The output word is whatever the trace packed, and the trace SAYS so:
    # channels and lane width are published on the boundary. Dividing the
    # word by the INPUT depth was the old way, and it silently mis-unpacked
    # any block that narrows -- a display curve mapping 12-bit linear onto
    # 8-bit RGB reads back as two 12-bit lanes instead of three 8-bit ones,
    # and the twin then reports a model/RTL mismatch that is really a
    # mismatch between two readings of the same word.
    out = generated.meta["boundary"]["outputs"]["chain_out"]
    out_bits, out_channels = out["data_bits"], out["channels"]
    out_lane = out.get("channel_bits", out_bits // max(1, out_channels))
    # The port is the other statement of that width; if the two disagree,
    # one of them is wrong and neither is worth guessing between.
    port_bits = ports["chain_out_data"]["width"]
    if port_bits != out_bits:
        raise ValueError(
            f"the chain's output port is {port_bits} bits but its traced "
            f"boundary says {out_bits}; the generator and its own report "
            "disagree about the same word")
    sized = "chain_out_frame_width" in ports
    binary = _twin_binary(verilog, generated.top, scraped,
                          in_bits=in_channels * bit_depth, out_bits=out_bits,
                          sized=sized)
    words = _pack(frame, in_channels, bit_depth)

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        (scratch / "in.hex").write_text(
            "\n".join(f"{w:x}" for w in words) + "\n")
        (scratch / "ports.txt").write_text(
            "\n".join(f"{n} {v:x}" for n, v in port_values.items()) + "\n")
        done = subprocess.run(
            [str(binary), str(scratch / "in.hex"), str(scratch / "out.txt"),
             str(scratch / "ports.txt"), str(len(words))],
            capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(
                f"the RTL twin did not complete the frame: {done.stdout.strip()}"
                f" {done.stderr.strip()}")
        got = np.array([int(line, 16) for line in
                        (scratch / "out.txt").read_text().split()], dtype=object)
    # the frame the hardware says it put out: a window's is its registers'
    if sized:
        _, w, h = done.stdout.split()
        width, height = int(w), int(h)
    if got.size != width * height:
        raise RuntimeError(
            f"the RTL twin put out {got.size} words for the {width}x{height} "
            "frame it published")
    return _unpack(got, height, width, out_channels, out_lane)


def _pack(frame: np.ndarray, channels: int, bit_depth: int) -> np.ndarray:
    """Stream words as Python integers: a word of eight 16-bit channels is 128
    bits, and a 64-bit numpy word would drop the top half without a sound."""
    frame = frame.astype(np.int64).astype(object)
    if channels == 1:
        return frame.ravel()
    planes = [frame[..., c] << (c * bit_depth) for c in range(channels)]
    return sum(planes).ravel()


def _unpack(words: np.ndarray, height: int, width: int, channels: int,
            bit_depth: int) -> np.ndarray:
    mask = (1 << bit_depth) - 1
    if channels == 1:
        return (words & mask).astype(np.int64).reshape(height, width)
    planes = [((words >> (c * bit_depth)) & mask).astype(np.int64)
              for c in range(channels)]
    return np.stack(planes, axis=-1).reshape(height, width, channels)


def _port_values(twin, scraped: list[str], values, context, width, height,
                 bit_depth) -> dict[str, int]:
    """A value for every ctx_*/param_* port of the generated top.

    Port names are derived the same way the generator derives them -- from
    the instance path and the declared leaf register names -- so this is a
    read of the declarations, not a parallel naming scheme. The refusal on
    an uncovered port is what keeps it honest.
    """
    ports = {}
    # Every context the vocabulary declares, read from its declarations
    # rather than listed here: a signal that follows the frame's width,
    # height or depth takes this run's, anything else the caller's value or
    # the declared reset. A list of names kept here would be a second copy
    # of the vocabulary, and the first signal added there would meet a port
    # this file had never heard of.
    vocab = compose.vocabulary(twin)
    run_facts = {"width": width, "height": height, "bit_depth": bit_depth}
    ctx = {c.name: run_facts[c.follows] if c.follows
           else context.get(c.name, c.default)
           for c in vocab.signals}
    for stage in compose.stages(twin):
        if stage.path == vocab.block.name:
            continue
        for name, value in values[stage.path].items():
            param = stage.paramset.param(name)
            ports[f"param_{stage.module_prefix}_{name}"] = (
                value & ((1 << param.bits) - 1))

    chosen = {}
    for port in scraped:
        if port.startswith("ctx_"):
            chosen[port] = int(ctx[port[4:]])
        else:
            if port not in ports:
                raise AssertionError(
                    f"generated port {port!r} has no value; the naming here "
                    "has drifted from the generator's")
            chosen[port] = ports[port]
    return chosen


def _port_code(port: str, bits: int, direction: str) -> str:
    """C++ moving one stream port to or from 32-bit chunks, for its width.

    Verilator gives a port of up to 64 bits as one integer and a wider one as
    an array of 32-bit words, least significant first; the chunks here use
    that same order, so the wide case is a copy."""
    if direction == "in":
        if bits <= 32:
            return f"    top->{port} = w[0];"
        if bits <= 64:
            return f"    top->{port} = ((uint64_t)w[1] << 32) | w[0];"
        return f"    for (int k = 0; k < IW; k++) top->{port}[k] = w[k];"
    if bits <= 32:
        return f"    w[0] = top->{port};"
    if bits <= 64:
        return (f"    uint64_t v = top->{port};\n"
                "    w[0] = (uint32_t)v; w[1] = (uint32_t)(v >> 32);")
    return f"    for (int k = 0; k < OW; k++) w[k] = top->{port}[k];"


def _twin_binary(verilog: str, top: str, scraped: list[str],
                 in_bits: int, out_bits: int, sized: bool) -> Path:
    """Compile the twin, or reuse the cached build for this exact structure.
    A `sized` twin reports the frame size its top publishes at the last
    word."""
    assign = "\n".join(f'    top->{port} = take("{port}");'
                       for port in scraped)
    size = ('std::printf("size %d %d\\n", (int)top->chain_out_frame_width, '
            '(int)top->chain_out_frame_height);' if sized else "")
    harness = (_HARNESS.replace("@TOP@", top).replace("@ASSIGN@", assign)
               .replace("@SIZE@", size)
               .replace("@IN_WORDS@", str(-(-in_bits // 32)))
               .replace("@OUT_WORDS@", str(-(-out_bits // 32)))
               .replace("@SET_IN@", _port_code("chain_in_data", in_bits, "in"))
               .replace("@GET_OUT@", _port_code("chain_out_data", out_bits,
                                                "out")))
    key = hashlib.sha256((verilog + harness).encode()).hexdigest()[:16]
    build = CACHE_ROOT / key
    binary = build / "obj_dir" / f"V{top}"
    if binary.exists():
        return binary
    build.mkdir(parents=True, exist_ok=True)
    (build / "twin.v").write_text(verilog)
    (build / "harness.cpp").write_text(harness)
    done = subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-j", "0", "-O3",
         "-Wno-fatal", "--top-module", top, "twin.v", "harness.cpp"],
        cwd=build, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(
            f"verilator refused the twin:\n{done.stderr.strip()[-2000:]}")
    return binary


# The driver mirrors the generated testbench exactly: sample ready/valid
# BEFORE the rising edge -- that is what the always @(posedge) blocks see --
# and only then advance. Ports arrive as a name/value file so one compiled
# twin serves every parameter change.
_HARNESS = r"""#include "V@TOP@.h"
#include "verilated.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

// Stream words of any width travel as 32-bit chunks, least significant
// first -- the layout Verilator itself gives a port wider than 64 bits.
static const int IW = @IN_WORDS@, OW = @OUT_WORDS@;

static void parse_words(const char* hex, uint32_t* out, int nwords) {
    long len = (long)std::strlen(hex);
    for (int w = 0; w < nwords; w++) {
        long end = len - 8L * w, start = end - 8;
        out[w] = 0;
        if (end <= 0) continue;
        if (start < 0) start = 0;
        char chunk[9];
        std::memcpy(chunk, hex + start, end - start);
        chunk[end - start] = 0;
        out[w] = (uint32_t)std::strtoul(chunk, nullptr, 16);
    }
}

static void set_in(V@TOP@* top, const uint32_t* w) {
@SET_IN@
}

static void get_out(V@TOP@* top, uint32_t* w) {
@GET_OUT@
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc != 5) { std::fprintf(stderr, "in out ports total\n"); return 1; }
    auto* top = new V@TOP@;
    long total = std::atol(argv[4]);

    std::vector<uint32_t> mem((size_t)total * IW);
    FILE* fi = std::fopen(argv[1], "r");
    char hex[1024];
    for (long i = 0; i < total; i++) {
        if (std::fscanf(fi, "%1023s", hex) != 1) return 1;
        parse_words(hex, &mem[(size_t)i * IW], IW);
    }
    std::fclose(fi);

    std::map<std::string, uint64_t> port;
    FILE* fp = std::fopen(argv[3], "r");
    char name[256]; unsigned long long value;
    while (std::fscanf(fp, "%255s %llx", name, &value) == 2) port[name] = value;
    std::fclose(fp);
    auto take = [&](const char* n) {
        auto it = port.find(n);
        if (it == port.end()) { std::fprintf(stderr, "no port %s\n", n); std::exit(1); }
        return it->second;
    };
@ASSIGN@

    FILE* fo = std::fopen(argv[2], "w");
    long idx = 0, got = 0;
    int sending = 0, ended = 0;
    top->rst = 1; top->chain_in_valid = 0;
    top->chain_in_eol = 0; top->chain_in_last = 0; top->chain_out_ready = 1;
    set_in(top, &mem[0]); top->chain_in_sof = 1;
    for (int i = 0; i < 4; i++) { top->clk = 0; top->eval(); top->clk = 1; top->eval(); }
    top->rst = 0; sending = 1; top->chain_in_valid = 1;

    for (long guard = 0; guard < 4 * total + 4096; guard++) {
        top->clk = 0; top->eval();
        int ready = top->chain_in_ready, valid = top->chain_out_valid;
        int last = top->chain_out_last;
        uint32_t data[OW];
        get_out(top, data);
        top->clk = 1; top->eval();
        if (sending && ready) {
            if (idx == total - 1) { sending = 0; top->chain_in_valid = 0; }
            else idx++;
        }
        if (valid) {
            std::fprintf(fo, "%x", data[OW - 1]);
            for (int w = OW - 2; w >= 0; w--) std::fprintf(fo, "%08x", data[w]);
            std::fprintf(fo, "\n");
            got++;
            // the frame ends where the hardware says it does, and is the
            // size it publishes beside it
            if (last) { ended = 1; @SIZE@ break; }
        }
        set_in(top, &mem[(size_t)idx * IW]);
        top->chain_in_sof = (idx == 0);
    }
    std::fclose(fo);
    if (!ended)
        std::fprintf(stderr, "frame incomplete: %ld words and no last, "
                     "of %ld in\n", got, total);
    delete top;
    return ended ? 0 : 2;
}
"""
