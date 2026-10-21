# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Prove a design of model blocks and modelled leaves: its models against
its Verilog, under the same stimulus.

A composite made only of model blocks is proved as a chain
(:mod:`np2hw.pipeline.run`). A design that also holds a fixed module -- a
frame store, a FIFO -- can be proved too where every such module brings its
NumPy model (``"model"`` in its library description), and this is how. The
stream is followed from the design's input to its output, stage by stage:
each composite of model blocks built whole runs its blocks' models, as its
chain does, and each modelled leaf runs its model. Both runs are driven by
one :class:`~np2hw.hdl.models.Stimulus` -- the frames, and when they come
and go -- and nothing in the model run is read back from the hardware.

What travels beside the frames is followed the same way. A frame's context
rides its stream, as the stream's CTX member (``<port>_ctx``), and every
port of a stage other than its stream is a net of the design. Each frame
carries the values its nets and its stream's context hold for it -- the
design's inputs as the frame enters, then each stage's outputs as the stage
hands the frame on -- and a stage reads its other inputs from those. A
stage after one that re-times the stream (a frame store chooses which
frames go out, and when) may read only what came through it: a value read
around it would be whichever frame happened to be arriving, which no model
of frames can say, and the run refuses it, naming the net.

The Verilog run is the design's own, emitted as it is built, under
Verilator, driven by np2hw's one harness: the stream's framing as its source
sends it, each frame's other inputs set as its first beat goes in, the
input's gaps and the consumer's waits and stalls from the stimulus, every
clock one clock, every register at the design's reset values.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from np2hw.hdl.models import Stimulus
from np2hw.pipeline import compose
from np2hw.pipeline import run as chains


@dataclass
class Stage:
    """One stage of the stream through a design."""
    path: str
    kind: str                       # "pipeline" (built whole) | "leaf"
    stream_in: str
    stream_out: str


@dataclass
class Run:
    """A run of a design: the frames out, and for each the values of the
    design's other outputs as it went out; a Verilog run's frames are the
    words its stream carried, `word_bits` wide (:func:`words` packs a
    model's frames the same way)."""
    frames: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    word_bits: int = 0


class ProofError(ValueError):
    """A design that cannot be proved from its models as it stands."""


# --------------------------------------------------------------------------- #
# the stream's path
# --------------------------------------------------------------------------- #

def _signals(design, path: str) -> dict:
    return design.netlist["instances"][path]["ports"] or {}


def _stream_ports(design, path: str) -> tuple[list, list]:
    """A leaf's stream ports: those with a valid, a ready and a data,
    split by the direction of their data."""
    sig = _signals(design, path)
    ins, outs = [], []
    for name, (direction, _) in sig.items():
        if not name.endswith("_data"):
            continue
        port = name[:-len("_data")]
        if f"{port}_valid" in sig and f"{port}_ready" in sig:
            (ins if direction == "input" else outs).append(port)
    return ins, outs


def stages(design, stream_in: str = "in",
           stream_out: str = "out") -> list[Stage]:
    """The stages the stream passes through, from the design's input port
    `stream_in` to its output port `stream_out`, in order."""
    sinks: dict = {}
    for source, sink in design.edges:
        sinks.setdefault(source, []).append(sink)
    found, at = [], stream_in
    while True:
        nxt = []
        for sink in sinks.get(at, []):
            if sink == stream_out:
                return found
            path, _, port = sink.rpartition(".")
            if path and port in _stream_ports(design, path)[0]:
                nxt.append((path, port))
        if len(nxt) != 1:
            raise ProofError(
                f"{design.ref}: the stream at {at} goes "
                + (f"to {len(nxt)} stages" if nxt else "nowhere")
                + "; a proof follows one stream through")
        path, port = nxt[0]
        outs = _stream_ports(design, path)[1]
        if len(outs) != 1:
            raise ProofError(f"{path}: {len(outs)} streams out")
        inst = design.instances[path]
        leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
        if path in design.pipelines:
            kind = "pipeline"
        elif leaf.get("model"):
            kind = "leaf"
        else:
            raise ProofError(
                f"{path} ({inst.ref}) brings no model: a design holding it "
                "can be built, but not proved from its models")
        found.append(Stage(path, kind, port, outs[0]))
        at = f"{path}.{outs[0]}"


# --------------------------------------------------------------------------- #
# the model run
# --------------------------------------------------------------------------- #

def _driver(design, sink: str) -> str | None:
    for source, s in design.edges:
        if s == sink:
            return source
    return None


def _clocking(name: str) -> bool:
    """A clock or a reset: what holds still while frames go by."""
    tail = name.rpartition("_")[2] if "_" in name else name
    return tail in ("clk", "aclk", "rst", "resetn", "aresetn", "rstn")


def _model(ref: str):
    from importlib import import_module
    module, _, name = ref.partition(":")
    return getattr(import_module(module), name)


def _read(design, stage: Stage, port: str, env: list,
          member: str = "") -> list:
    """Each frame's value at `stage`'s input `port` -- or, with `member`,
    at that member of the stream port: ``ctx``, its frame's context."""
    driver = _driver(design, f"{stage.path}.{port}")
    if driver is None:
        raise ProofError(f"{stage.path}.{port} is driven by nothing")
    if member:
        port, driver = f"{port}_{member}", f"{driver}_{member}"
        if not any(driver in held for held in env):
            raise ProofError(
                f"{stage.path}.{port} reads {driver}: that stream carries no "
                f"{'context' if member == 'ctx' else member} with its frames")
    try:
        return [held[driver] for held in env]
    except KeyError:
        raise ProofError(
            f"{stage.path}.{port} reads {driver}, which does not ride with "
            "the frames it takes: a stage before it chose which frames go "
            "on, and the value is not one of theirs") from None


def run_model(design, frames, ports, stimulus: Stimulus,
              stream: tuple = ("in", "out"), outputs: int | None = None
              ) -> Run:
    """The design's models, stage by stage, on `frames` -- each ``(h, w)``
    -- and `ports`, each frame's values of the design's other inputs by
    their names, its stream's context among them as ``<stream>_ctx``, as
    `stimulus` times them."""
    frames = [np.asarray(f) for f in frames]
    env = [dict(p) for p in ports]
    for stage in stages(design, *stream):
        inst = design.instances[stage.path]
        if stage.kind == "leaf":
            leaf = inst.leaf
            own = {}
            for spec in leaf.get("inputs", []):
                name = spec["name"]
                if name == stage.stream_in:
                    if "ctx" in spec.get("stream", ()):
                        own[f"{name}_ctx"] = _read(design, stage, name, env,
                                                   "ctx")
                    continue
                if _clocking(name):
                    continue
                own[name] = _read(design, stage, name, env)
            given = [{k: v[i] for k, v in own.items()}
                     for i in range(len(frames))]
            params = leaf.get("block_parameters") or dict(inst.build)
            run = _model(leaf["model"])(frames, given, stimulus, params,
                                        outputs=outputs)
            frames = list(run.frames)
            # a context goes on only with a stream that declares one
            spec = next(p for p in leaf.get("outputs", [])
                        if p["name"] == stage.stream_out)
            dropped = () if "ctx" in spec.get("stream", ()) else \
                (f"{stage.stream_out}_ctx",)
            env = [{f"{stage.path}.{k}": v for k, v in told.items()
                    if k not in dropped} for told in run.ports]
        else:
            frames, env = _pipeline(design, stage, frames, env)
    out_ports = [p for p in design.outputs if p != stream[1]]
    told = []
    for held in env:
        row = {}
        for port in out_ports:
            driver = _driver(design, port)
            if driver in held:
                row[port] = held[driver]
        # the stream's own context, as it goes out
        driver = _driver(design, stream[1])
        if f"{driver}_ctx" in held:
            row[f"{stream[1]}_ctx"] = held[f"{driver}_ctx"]
        told.append(row)
    return Run(frames, told)


def _pipeline(design, stage: Stage, frames: list, env: list):
    """A composite of model blocks built whole, run as its chain: each
    frame's context the fields its stream carries, and the context block's
    registers, as they are from reset, for those it does not; a stream that
    brings its history run a frame of frames at a time. The context handed
    on is the core's: every field, the size the frame out's own."""
    from np2hw.stream import pack_context, unpack_context
    sub = design.pipelines[stage.path]
    chain = chains.pixel_chain(sub)
    values, _ = chains.resolve_values(sub)
    form = compose.shape(sub)
    bits = form.spec.bit_depth
    owner = compose.vocabulary(sub).block.name
    registers = next(b for b in compose.register_map(sub)["blocks"]
                     if b["path"] == owner)["registers"]
    held = {r["name"]: r["default"] for r in registers}
    carried = compose.carried_layout(sub)
    words = (_read(design, stage, stage.stream_in, env, "ctx") if carried
             else [0] * len(frames))
    layout = compose.context_layout(sub)
    brought = {name for name, _, _ in carried}
    fields = [dict({name: held[name] for name, _, _ in layout
                    if name not in brought},
                   **unpack_context(carried, word)) for word in words]
    context = {name: [f[name] for f in fields] for name, _, _ in layout}
    out = []
    if form.spec.frames:
        n = form.spec.frames
        for i, f in enumerate(frames):
            run = np.moveaxis(np.asarray(f), -1, 0)[::-1]   # oldest first
            ctx = {k: [v[i]] * n for k, v in context.items()}
            out.append(chains.run_model(chain, run, values, ctx, bits)[-1])
    elif chains.reads_history(chain):
        made = chains.run_model(chain, np.stack(frames), values, context, bits)
        skipped = len(frames) - len(made)
        out, env = list(made), env[skipped:]
        fields = fields[skipped:]
    else:
        for i, f in enumerate(frames):
            ctx = {k: v[i] for k, v in context.items()}
            out.append(chains.run_model(chain, f, values, ctx, bits))
    handed = [dict(held, **{f"{stage.path}.{stage.stream_out}_ctx": pack_context(
        layout, dict(given, width=f.shape[1], height=f.shape[0]))})
        for held, given, f in zip(env, fields, out)]
    return out, handed


# --------------------------------------------------------------------------- #
# the Verilog run
# --------------------------------------------------------------------------- #

def sources(design, folder: Path) -> tuple[str, list[str]]:
    """The design's Verilog, every file of it, written where it is made:
    ``(top module, files)``."""
    from np2hw.design.composite import source_file, write_generated
    from np2hw.design.verilog import modules
    folder = Path(folder)
    files = [str(p) for p in write_generated(design, folder / "generated")
             if p.suffix in (".v", ".sv")]
    for inst in design.instances.values():
        leaf = inst.leaf if isinstance(inst.leaf, dict) else {}
        for f in leaf.get("sources", []):
            files.append(str(source_file(inst.folder, f)))
    written = modules(design)
    for name, text in written:
        (folder / f"{name}.v").write_text(text)
        files.append(str(folder / f"{name}.v"))
    top = design.ref.rpartition(":")[2]
    if top not in {name for name, _ in written}:
        raise ProofError(f"{design.ref}: no module {top} among those written")
    return top, list(dict.fromkeys(files))


def run_rtl(design, frames, ports, stimulus: Stimulus,
            stream: tuple = ("in", "out"), outputs: int | None = None,
            cycles: int = 2_000_000) -> Run:
    """The design's Verilog under Verilator, driven as :func:`run_model`'s
    models are: the same frames, ports and stimulus."""
    from np2hw.ports import elaborated_ports
    frames = [np.asarray(f) for f in frames]
    count = outputs if outputs is not None else len(stimulus.waits) + 1
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        top, files = sources(design, scratch)
        pins = elaborated_ports(files, top)
        sin, sout = stream
        side_in = sorted({k for p in ports for k in p})
        side_out = [p for p in design.outputs if p != sout and p in pins]
        if f"{sout}_ctx" in pins:
            side_out.append(f"{sout}_ctx")
        for name in side_in:
            if name not in pins:
                raise ProofError(f"{top} has no input {name}")
        binary = _binary(files, top, pins, sin, sout, side_in, side_out)
        _stimulus_files(scratch, frames, ports, stimulus, pins, sin, side_in)
        done = subprocess.run(
            [str(binary), str(scratch), str(count), str(cycles)],
            capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"the design's Verilog did not finish: "
                               f"{done.stdout.strip()} {done.stderr.strip()}")
        run = _parse(scratch / "out.txt", pins[f"{sout}_data"][1])
        run.word_bits = pins[f"{sout}_data"][1]
        return run


def _stimulus_files(scratch, frames, ports, stimulus, pins, sin, side_in):
    words = []
    for f in frames:
        words += [int(v) for v in np.asarray(f, dtype=np.int64).ravel()]
    (scratch / "beats.txt").write_text("".join(f"{w:x}\n" for w in words))
    (scratch / "frames.txt").write_text("".join(
        f"{f.shape[1]} {f.shape[0]} {stimulus.gaps[k]}\n"
        for k, f in enumerate(frames)))
    (scratch / "ports.txt").write_text("".join(
        f"{k} {name} {int(p.get(name, 0)):x}\n"
        for k, p in enumerate(ports) for name in side_in))
    stalls = []
    for i in range(len(stimulus.waits) + 1):
        stalls += [f"{i} {b} {c}\n" for b, c in stimulus.stall(i)]
    (scratch / "consumer.txt").write_text(
        f"{stimulus.first}\n" + " ".join(map(str, stimulus.waits)) + "\n")
    (scratch / "stalls.txt").write_text("".join(stalls))


def _parse(path: Path, out_bits: int) -> Run:
    run, beats, told = Run(), [], {}
    for line in path.read_text().splitlines():
        kind, *rest = line.split()
        if kind == "O":
            told = {k: int(v, 16) for k, v in (r.split("=") for r in rest)}
        elif kind == "B":
            beats.append(int(rest[0], 16))
        elif kind == "E":
            w, h = int(rest[0]), int(rest[1])
            run.frames.append(np.array(beats, dtype=object).reshape(h, w))
            run.ports.append(told)
            beats, told = [], {}
    return run


def words(frame: np.ndarray, out_bits: int) -> np.ndarray:
    """A model's frame out as the words its stream carries: its channels
    packed in equal fields, channel 0 lowest, as np2hw packs every word."""
    frame = np.asarray(frame).astype(np.int64)
    if frame.ndim == 2:
        return frame.astype(object)
    channels = frame.shape[-1]
    field = out_bits // channels
    packed = np.zeros(frame.shape[:2], dtype=object)
    for c in range(channels):
        packed = packed + (frame[..., c].astype(object) << (c * field))
    return packed


CACHE_ROOT = Path(tempfile.gettempdir()) / "np2hw-proof"


def _code_set(port: str, bits: int, src: str) -> str:
    if bits <= 64:
        return f"top->{port} = {src}[0] | ((uint64_t){src}[1] << 32);"
    return (f"for (int q = 0; q < {-(-bits // 32)}; q++) "
            f"top->{port}[q] = {src}[q];")


def _code_get(port: str, bits: int, dst: str) -> str:
    if bits <= 64:
        return (f"{{ uint64_t v = top->{port}; {dst}[0] = (uint32_t)v; "
                f"{dst}[1] = (uint32_t)(v >> 32); }}")
    return (f"for (int q = 0; q < {-(-bits // 32)}; q++) "
            f"{dst}[q] = top->{port}[q];")


def _binary(files, top, pins, sin, sout, side_in, side_out) -> Path:
    clocks = [p for p, (d, _) in pins.items() if d == "input"
              and p.rpartition("_")[2] in ("clk", "aclk") or p == "clk"]
    highs = [p for p, (d, _) in pins.items() if d == "input" and
             (p == "rst" or p.endswith("_rst"))]
    lows = [p for p, (d, _) in pins.items() if d == "input" and
            p.rpartition("_")[2] in ("resetn", "aresetn", "rstn")]
    held = {*clocks, *highs, *lows, *side_in,
            *(f"{sin}_{s}" for s in ("valid", "data", "sof", "eol", "last")),
            f"{sout}_ready"}
    quiet = [p for p, (d, _) in pins.items() if d == "input"
             and p not in held]
    in_bits, out_bits = pins[f"{sin}_data"][1], pins[f"{sout}_data"][1]
    sides = "\n".join(
        f'        if (name == "{n}") {{ uint32_t w[2] = {{(uint32_t)v, '
        f'(uint32_t)(v >> 32)}}; {_code_set(n, pins[n][1], "w")} }}'
        for n in side_in)
    told = "\n".join(
        f'            {{ uint32_t w[2] = {{0, 0}}; {_code_get(n, pins[n][1], "w")} '
        f'std::fprintf(fo, " {n}=%llx", (unsigned long long)w[0] | '
        f'((unsigned long long)w[1] << 32)); }}'
        for n in side_out if pins[n][1] <= 64)
    text = (_HARNESS.replace("@TOP@", top)
            .replace("@CLOCKS@", " ".join(f"top->{c} = c;" for c in clocks))
            .replace("@RESET@", " ".join(
                [f"top->{r} = on;" for r in highs] +
                [f"top->{r} = !on;" for r in lows]))
            .replace("@QUIET@", " ".join(f"top->{q} = 0;" for q in quiet
                                          if pins[q][1] <= 64))
            .replace("@IW@", str(-(-in_bits // 32) + 2))
            .replace("@OW@", str(-(-out_bits // 32) + 2))
            .replace("@SET_DATA@", _code_set(f"{sin}_data", in_bits, "word"))
            .replace("@GET_DATA@", _code_get(f"{sout}_data", out_bits, "out"))
            .replace("@SIDES@", sides).replace("@TOLD@", told)
            .replace("@SIN@", sin).replace("@SOUT@", sout))
    key = hashlib.sha256("".join(
        [text] + [Path(f).read_text() for f in files]).encode()).hexdigest()[:16]
    build = CACHE_ROOT / key
    binary = build / "obj_dir" / f"V{top}"
    if binary.exists():
        return binary
    build.mkdir(parents=True, exist_ok=True)
    (build / "harness.cpp").write_text(text)
    done = subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-j", "0", "-O2",
         "-Wno-fatal", "-Wno-lint", "-Wno-style", "--top-module", top,
         *files, "harness.cpp"], cwd=build, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(
            f"verilator refused the design:\n{done.stderr.strip()[-3000:]}")
    return binary


# One harness for every design: the input stream sent frame by frame, each
# `gap` cycles after the last, a beat a cycle while the design takes it,
# with the framing its source gives -- sof, eol, last -- and the frame's
# other inputs set as its first beat is offered; the output taken from
# cycle `first`, each frame `wait` cycles after the last, holding off
# `cycles` before beat `beat` where the stimulus stalls. Inputs and outputs
# are sampled before the rising edge, which is what the always @(posedge)
# blocks see. Every clock is one clock.
_HARNESS = r"""#include "V@TOP@.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static void parse_hex(const char* hex, uint32_t* out, int n) {
    long len = (long)std::strlen(hex);
    for (int w = 0; w < n; w++) {
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

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc != 4) { std::fprintf(stderr, "dir frames cycles\n"); return 1; }
    std::string dir = argv[1];
    long want = std::atol(argv[2]), limit = std::atol(argv[3]);
    auto* top = new V@TOP@;
    const int IW = @IW@, OW = @OW@;

    // the input: each frame's size and gap, and its beats
    std::vector<long> fw, fh, gap;
    FILE* f = std::fopen((dir + "/frames.txt").c_str(), "r");
    long a, b, c;
    while (std::fscanf(f, "%ld %ld %ld", &a, &b, &c) == 3) {
        fw.push_back(a); fh.push_back(b); gap.push_back(c);
    }
    std::fclose(f);
    std::vector<std::vector<uint32_t>> beats;
    f = std::fopen((dir + "/beats.txt").c_str(), "r");
    char hex[2048];
    while (std::fscanf(f, "%2047s", hex) == 1) {
        std::vector<uint32_t> w(IW);
        parse_hex(hex, w.data(), IW);
        beats.push_back(w);
    }
    std::fclose(f);
    // each frame's other inputs
    std::vector<std::map<std::string, uint64_t>> side(fw.size());
    f = std::fopen((dir + "/ports.txt").c_str(), "r");
    char name[256]; unsigned long long v; long k;
    while (std::fscanf(f, "%ld %255s %llx", &k, name, &v) == 3)
        side[(size_t)k][name] = v;
    std::fclose(f);
    // the consumer
    long first = 0;
    std::vector<long> waits;
    f = std::fopen((dir + "/consumer.txt").c_str(), "r");
    if (std::fscanf(f, "%ld", &first) != 1) return 1;
    while (std::fscanf(f, "%ld", &a) == 1) waits.push_back(a);
    std::fclose(f);
    std::vector<std::vector<std::pair<long, long>>> stalls(waits.size() + 1);
    f = std::fopen((dir + "/stalls.txt").c_str(), "r");
    while (std::fscanf(f, "%ld %ld %ld", &a, &b, &c) == 3)
        if ((size_t)a < stalls.size()) stalls[(size_t)a].push_back({b, c});
    std::fclose(f);

    auto set_side = [&](long frame) {
        for (auto& kv : side[(size_t)frame]) {
            const std::string& name = kv.first; uint64_t v = kv.second;
@SIDES@
        }
    };
    auto edge = [&](int c) { @CLOCKS@ top->eval(); };

    // reset, every clock running
    @QUIET@
    top->@SIN@_valid = 0; top->@SOUT@_ready = 0;
    { int on = 1; @RESET@ }
    for (int i = 0; i < 4; i++) { edge(0); edge(1); }
    { int on = 0; @RESET@ }

    FILE* fo = std::fopen((dir + "/out.txt").c_str(), "w");
    // the source
    long fk = 0, idx = 0, wait_in = fw.empty() ? 0 : gap[0], at = 0;
    bool sending = !fw.empty() && wait_in == 0;
    // the consumer
    long oj = 0, taken = 0, hold = 0, start = first, si = 0;
    long cycle = 0, cols = 0, beats_out = 0;
    uint32_t word[64];
    for (; cycle < limit && oj < want; cycle++) {
        // drive this cycle's inputs, before the edge
        if (!sending && fk < (long)fw.size() && wait_in == 0) {
            sending = true; idx = 0;
        }
        if (sending) {
            if (idx == 0) set_side(fk);
            std::memset(word, 0, sizeof word);
            std::memcpy(word, beats[(size_t)(at + idx)].data(), IW * 4);
            @SET_DATA@
            top->@SIN@_valid = 1;
            top->@SIN@_sof = idx == 0;
            top->@SIN@_eol = idx % fw[(size_t)fk] == fw[(size_t)fk] - 1;
            top->@SIN@_last = idx == fw[(size_t)fk] * fh[(size_t)fk] - 1;
        } else {
            top->@SIN@_valid = 0;
        }
        bool ready_out = false;
        if (cycle >= start) {
            if (hold > 0) { hold--; }
            else if (si < (long)stalls[(size_t)oj].size()
                     && stalls[(size_t)oj][(size_t)si].first == taken) {
                hold = stalls[(size_t)oj][(size_t)si].second - 1; si++;
            } else ready_out = true;
        }
        top->@SOUT@_ready = ready_out;
        edge(0);
        bool in_take = sending && top->@SIN@_ready;
        bool out_take = ready_out && top->@SOUT@_valid;
        uint32_t out[64];
        std::memset(out, 0, sizeof out);
        if (out_take) {
            @GET_DATA@
            if (top->@SOUT@_sof) {
                std::fprintf(fo, "O");
@TOLD@
                std::fprintf(fo, "\n");
            }
            std::fprintf(fo, "B %x", out[OW - 3]);
            for (int q = OW - 4; q >= 0; q--) std::fprintf(fo, "%08x", out[q]);
            std::fprintf(fo, "\n");
        }
        bool out_last = out_take && top->@SOUT@_last;
        // the frame's width is where its first line ends
        if (out_take && cols == 0 && top->@SOUT@_eol) cols = beats_out + 1;
        edge(1);
        if (in_take) {
            if (idx == fw[(size_t)fk] * fh[(size_t)fk] - 1) {
                at += fw[(size_t)fk] * fh[(size_t)fk];
                fk++; sending = false;
                wait_in = fk < (long)fw.size() ? gap[(size_t)fk] : 0;
            } else idx++;
        } else if (!sending && wait_in > 0) {
            wait_in--;
        }
        if (out_take) {
            taken++; beats_out++;
            if (out_last) {
                long w = cols ? cols : beats_out;
                std::fprintf(fo, "E %ld %ld\n", w, beats_out / w);
                oj++; taken = 0; si = 0; hold = 0; cols = 0; beats_out = 0;
                start = cycle + (oj - 1 < (long)waits.size()
                                 ? waits[(size_t)(oj - 1)] : 1);
            }
        }
    }
    std::fclose(fo);
    delete top;
    if (oj < want) {
        std::printf("%ld of %ld frames out in %ld cycles\n", oj, want, cycle);
        return 2;
    }
    return 0;
}
"""
