"""np2hw command-line runner: apply an ISP model to an image or video, file to
file.

The model is always YOUR NumPy function -- the tool compiles arbitrary NumPy and
has no built-in model names. Point it at a .py file:

  np2hw run my_isp.py in.png out.png            # uses fn `model` (or `isp`)
  np2hw run my_isp.py:sharpen in.gif out.gif    # pick a function by name
  np2hw run my_isp.py in.png out.png --param gain=24

A model file is plain NumPy plus an optional param list:

  # my_isp.py
  import numpy as np
  from np2hw import Param
  PARAMS = [Param("gain", np.uint8)]            # optional; omit if no registers
  def model(img, gain):
      return ((gain * img.astype(np.uint16)) // 16).clip(0, 255).astype(np.uint8)

Ready-made example models live in examples/isp/ (gaussian.py, blur.py, ...).

Backends:
  numpy (default) -- the reference model; fast.
  rtl             -- generate Verilog and simulate it (iverilog) per frame, so
                     the output is what the actual hardware produces.

Media: still images via Pillow (PNG/JPG/BMP/...); "video" via animated GIF
(multi-frame, no ffmpeg needed). Frames are processed as 8-bit grayscale.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

from .ir import Image2D, Params, np_dtype
from .frontend import to_ir
from .verilog import generate, testbench


def _leaves(decl):
    """Scalar Params (registers) of a model declaration -- a plain list, or the
    flattened leaves of a Params namespace."""
    return decl.leaves() if isinstance(decl, Params) else list(decl)


def _trace_args(decl):
    """Args after `img` when tracing: the Params object itself, or *the list."""
    return (decl,) if isinstance(decl, Params) else tuple(decl)


def _value_args(decl, pvals):
    """Args after `img` for a NumPy run: a bound Params view, or *the values.
    Each value is a NumPy-typed scalar (Param dtype) so the oracle promotes like
    the hardware register -- a plain Python int would trip NEP-50 (e.g. a signed
    register added to a uint image)."""
    if isinstance(decl, Params):
        return (decl.bind({p.name: int(pvals[p.name]) for p in decl.leaves()}),)
    return tuple(np_dtype(p.bits, p.signed).type(int(pvals[p.name])) for p in decl)


# --------------------------------------------------------------------------- #
# Model resolution: always a user .py file (the tool compiles arbitrary NumPy)
# --------------------------------------------------------------------------- #

def _load_model(spec):
    """Return (fn, params, label). `spec` is 'file.py' or 'file.py:func'."""
    path, _, func = spec.partition(":")
    if not os.path.exists(path):
        raise SystemExit(f"model file not found: {path!r} "
                         "(pass a .py file, e.g. examples/isp/gaussian.py)")
    mod_spec = importlib.util.spec_from_file_location("np2hw_user_model", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    if func:
        fn = getattr(mod, func, None)
        if fn is None:
            raise SystemExit(f"{path} has no function {func!r}")
    else:
        fn = getattr(mod, "model", None) or getattr(mod, "isp", None)
        if fn is None:
            raise SystemExit(f"{path}: define `model(img, ...)` or use file.py:func")
    decl = getattr(mod, "PARAMS", [])
    params = decl if isinstance(decl, Params) else list(decl)  # keep namespace intact
    return fn, params, os.path.basename(path) + (f":{func}" if func else "")


# --------------------------------------------------------------------------- #
# Media IO (Pillow): still image or multi-frame GIF -> list of HxW uint8 arrays
# --------------------------------------------------------------------------- #

def _load(path):
    from PIL import Image, ImageSequence
    im = Image.open(path)
    frames = [np.asarray(f.convert("L"), dtype=np.uint8)
              for f in ImageSequence.Iterator(im)]
    is_video = len(frames) > 1
    meta = {"duration": im.info.get("duration", 80),
            "loop": im.info.get("loop", 0)}
    return frames, is_video, meta


def _save(path, frames, meta):
    from PIL import Image
    imgs = [Image.fromarray(f.astype(np.uint8), mode="L") for f in frames]
    if len(imgs) > 1:
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=meta.get("duration", 80), loop=meta.get("loop", 0))
    else:
        imgs[0].save(path)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def _run_numpy(fn, params, pvals, frames):
    tail = _value_args(params, pvals)
    return [np.asarray(fn(f, *tail)).astype(np.uint8) for f in frames]


def _compile(fn, params, W, H, module="model"):
    s, out = to_ir(fn, Image2D("img", W, H, bits=8), *_trace_args(params))
    meta = generate(out, module_name=module)
    meta.setdefault("out_rows", H)
    meta.setdefault("out_cols", W)
    return meta


def _run_iverilog(fn, params, pvals, frames, module="model"):
    """Compile to Verilog once, then stream each frame through iverilog (the same
    validated path the tests use). Robust; slower (a vvp run per frame)."""
    H, W = frames[0].shape
    work = tempfile.mkdtemp(prefix="np2hw_iv_")
    meta = _compile(fn, params, W, H, module)
    pv = {p.name: int(pvals[p.name]) for p in _leaves(params)}
    open(os.path.join(work, f"{module}.v"), "w").write(meta["verilog"] + "\n")
    open(os.path.join(work, "tb.v"), "w").write(testbench(meta, W, H, pv) + "\n")
    subprocess.run(["iverilog", "-o", "sim.vvp", f"{module}.v", "tb.v"],
                   cwd=work, check=True, capture_output=True)
    orows, ocols = meta["out_rows"], meta["out_cols"]
    results = []
    for fr in frames:
        open(os.path.join(work, "in.hex"), "w").write(
            "\n".join(f"{p:02x}" for p in fr.flatten()) + "\n")
        subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True)
        px = np.loadtxt(os.path.join(work, "out.txt"), dtype=np.int64)
        results.append(px.reshape(orows, ocols).astype(np.uint8))
    return results


def _mangle(name):
    return "p_" + name.replace("_", "__")


def _run_cxxrtl(fn, params, pvals, frames, module="model"):
    """Compile to a cxxrtl C++ model and stream ALL frames in one process. Fast
    (compiled, no per-frame startup) -- the right engine for video."""
    H, W = frames[0].shape
    work = tempfile.mkdtemp(prefix="np2hw_cxx_")
    meta = _compile(fn, params, W, H, module)
    orows, ocols = meta["out_rows"], meta["out_cols"]
    n_out = orows * ocols
    open(os.path.join(work, f"{module}.v"), "w").write(meta["verilog"] + "\n")
    subprocess.run(["yosys", "-q", "-p",
                    f"read_verilog {module}.v; write_cxxrtl {module}_cxx.cc"],
                   cwd=work, check=True, capture_output=True)
    pv = {p.name: int(pvals[p.name]) for p in _leaves(params)}
    open(os.path.join(work, "drv.cc"), "w").write(
        _cxxrtl_driver(module, W, H, len(frames), n_out, meta["params"], pv))
    datdir = subprocess.run(["yosys-config", "--datdir"], capture_output=True,
                            text=True, check=True).stdout.strip()
    inc = os.path.join(datdir, "include", "backends", "cxxrtl", "runtime")
    subprocess.run(["c++", "-std=c++14", "-O2", "-I", inc, "drv.cc", "-o", "sim"],
                   cwd=work, check=True, capture_output=True)
    # raw input: all frames, one byte per pixel
    np.concatenate([f.flatten() for f in frames]).astype(np.uint8).tofile(
        os.path.join(work, "in.bin"))
    subprocess.run(["./sim", "in.bin", "out.bin"], cwd=work, check=True,
                   capture_output=True)
    dt = np.int32 if meta.get("signed") else np.uint32
    flat = np.fromfile(os.path.join(work, "out.bin"), dtype=dt)
    return [flat[i*n_out:(i+1)*n_out].reshape(orows, ocols).astype(np.uint8)
            for i in range(len(frames))]


def _sb_design(Design, name, vpath, work):
    """Build a siliconcompiler Design carrying the core+wrapper Verilog with the
    fileset layout Switchboard expects: an 'rtl' fileset with the sources, plus
    'icarus'/'verilator' filesets that depend on it. SbDut then selects the
    fileset matching its `tool` (so we DON'T pass fileset= ourselves), which also
    pulls in the C++ testbench + DPI automatically -- no manual add_file needed.
    (Per zeroasiccorp/switchboard#309.)"""
    d = Design(name); d.set_dataroot("root", work)
    with d.active_fileset("rtl"):
        d.set_topmodule(name); d.add_file(vpath)
    for fs in ("icarus", "verilator"):
        with d.active_fileset(fs):
            d.set_topmodule(name); d.add_depfileset(d, "rtl")
    return d


def _sb_build(fn, params, pvals, W, H, tool="verilator", module="model"):
    """Build a persistent Switchboard sim for a WxH core (PACKED gearbox) and
    return (dut, tx, rx, meta, wrap). Params are baked at build time via autowrap
    tieoffs. Reused by the batch run and the live viewer."""
    try:                                                 # lazy: heavy optional dep
        import switchboard as sb
        from siliconcompiler import Design
    except ImportError as e:
        raise SystemExit("--backend switchboard needs the [switchboard] extra: "
                         "uv pip install -e \".[switchboard]\"  "
                         f"({e})")
    from .verilog import switchboard_wrap
    work = tempfile.mkdtemp(prefix="np2hw_sb_")
    meta = _compile(fn, params, W, H, module)
    wrap = switchboard_wrap(meta, W, H, module_name=module + "_sb",
                            native=True, pack=True)        # gearbox: many px/packet
    vpath = os.path.join(work, "g.v")
    open(vpath, "w").write(meta["verilog"] + "\n" + wrap["verilog"] + "\n")
    d = _sb_design(Design, module + "_sb", vpath, work)
    # tie each config register to its value -- width MUST be given (autowrap
    # defaults tieoff width to 1, which would truncate e.g. gain=24 to 0)
    tieoffs = {f"param_{p.name}": {"value": int(pvals[p.name]), "width": p.bits}
               for p in _leaves(params)}
    dut = sb.SbDut(
        design=d, autowrap=True, trace=False, tool=tool,
        parameters={"WIDTH": W, "HEIGHT": H},
        interfaces={"sb_in": {"type": "sb", "direction": "input", "dw": 416},
                    "sb_out": {"type": "sb", "direction": "output", "dw": 416}},
        clocks=["clk"], resets=["rst"], tieoffs=tieoffs,
        buildroot=os.path.join(work, "bld"))
    if tool == "verilator":                              # waive sim-module lint
        from switchboard.sbdut import get_task
        from siliconcompiler.tools.verilator import VerilatorTask
        t = get_task(dut, filter=VerilatorTask)
        for w in ("PINMISSING", "UNSIGNED", "WIDTHEXPAND", "WIDTHTRUNC", "TIMESCALEMOD"):
            t.add_warningoff(w)
    dut.build(); dut.simulate()
    return dut, dut.intfs["sb_in"], dut.intfs["sb_out"], meta, wrap


def _sb_stream_frame(tx, rx, meta, wrap, frame):
    """Stream ONE frame through the SB queues and return the output. Packs
    p_in = 416//in_bits pixels per packet, unpacks p_out per output packet."""
    import switchboard as sb
    ib, ob = meta["in_bits"], meta["out_bits"]
    p_in, p_out = wrap["p_in"], wrap["p_out"]
    imask, omask = (1 << ib) - 1, (1 << ob) - 1
    orows, ocols = meta["out_rows"], meta["out_cols"]
    nfo = orows * ocols
    flat = np.asarray(frame).reshape(-1)
    pkts = []                                            # final packet padded; core drops tail
    for s in range(0, flat.size, p_in):
        payload = 0
        for j, px in enumerate(flat[s:s + p_in]):
            payload |= (int(px) & imask) << (j * ib)
        pkts.append(sb.PySbPacket(0, 0,
                    np.frombuffer(payload.to_bytes(52, "little"), np.uint8).copy()))
    # bounded queues -> non-blocking send/recv interleave (no deadlock, no threads)
    cur, sent, idle = [], 0, 0
    while len(cur) < nfo:
        progressed = False
        if sent < len(pkts) and tx.send(pkts[sent], blocking=False):
            sent += 1; progressed = True
        p = rx.recv(blocking=False)
        if p is not None:
            payload = int.from_bytes(bytes(p.data), "little")
            for j in range(min(p_out, nfo - len(cur))):
                cur.append((payload >> (j * ob)) & omask)
            progressed = True
        idle = 0 if progressed else idle + 1
        if idle > 2_000_000:
            raise RuntimeError("switchboard stalled mid-frame")
    return np.array(cur, np.int64).reshape(orows, ocols).astype(np.uint8)


def _run_switchboard(fn, params, pvals, frames, module="model", tool="verilator"):
    """Batch: build the SB sim once, stream each frame, validate vs the NumPy."""
    H, W = frames[0].shape
    dut, tx, rx, meta, wrap = _sb_build(fn, params, pvals, W, H, tool, module)
    try:
        return [_sb_stream_frame(tx, rx, meta, wrap, fr) for fr in frames]
    finally:
        dut.terminate()


def _sb_build_ctrl(fn, params, pvals, W, H, tool="verilator", module="model"):
    """Build a persistent SB sim WITH an AXI-Lite control interface, so config
    registers can be written at RUNTIME (AxiLiteTxRx) while pixels stream over the
    SB queues. Returns (dut, tx, rx, ctrl, meta, wrap)."""
    try:
        import switchboard as sb
        from siliconcompiler import Design
    except ImportError as e:
        raise SystemExit("--backend switchboard needs the [switchboard] extra: "
                         "uv pip install -e \".[switchboard]\"  " f"({e})")
    from .verilog import switchboard_control_wrap
    work = tempfile.mkdtemp(prefix="np2hw_sbc_")
    meta = _compile(fn, params, W, H, module)
    wrap = switchboard_control_wrap(meta, W, H, module_name=module + "_sbc")
    vpath = os.path.join(work, "g.v")
    open(vpath, "w").write(meta["verilog"] + "\n" + wrap["verilog"] + "\n")
    d = _sb_design(Design, module + "_sbc", vpath, work)
    dut = sb.SbDut(
        design=d, autowrap=True, trace=False, tool=tool,
        parameters={"WIDTH": W, "HEIGHT": H},
        interfaces={"sb_in": {"type": "sb", "direction": "input", "dw": 416},
                    "sb_out": {"type": "sb", "direction": "output", "dw": 416},
                    "ctrl": {"type": "axil", "direction": "subordinate",
                             "dw": 32, "aw": wrap["addr_bits"]}},
        clocks=["clk"], resets=["rst"], buildroot=os.path.join(work, "bld"))
    if tool == "verilator":
        from switchboard.sbdut import get_task
        from siliconcompiler.tools.verilator import VerilatorTask
        t = get_task(dut, filter=VerilatorTask)
        for w in ("PINMISSING", "UNSIGNED", "WIDTHEXPAND", "WIDTHTRUNC", "TIMESCALEMOD"):
            t.add_warningoff(w)
    dut.build(); dut.simulate()
    return (dut, dut.intfs["sb_in"], dut.intfs["sb_out"], dut.intfs["ctrl"],
            meta, wrap)


def _cxxrtl_driver(module, W, H, nframes, n_out, params, pv):
    # params come from argv (argv[3], argv[4], ...) so the viewer can change them
    # live; the baked pv is the default when an arg isn't supplied (batch runs).
    sets = "\n  ".join(
        f"top.{_mangle('param_'+n)}.set<uint32_t>("
        f"argc > {3+j} ? (uint32_t)strtol(argv[{3+j}],0,10) : {int(pv.get(n,0))});"
        for j, (n, _) in enumerate(params))
    return f"""// generated by np2hw -- cxxrtl driver
#include "{module}_cxx.cc"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
using namespace cxxrtl_design;
int main(int argc, char** argv) {{
  p_{module} top;
  {sets}
  const int W={W}, H={H}, NF={nframes}, NOUT={n_out}, HW=W*H;
  FILE* fi=fopen(argv[1],"rb"); FILE* fo=fopen(argv[2],"wb");
  std::vector<uint8_t> buf((size_t)NF*HW);
  if (fread(buf.data(),1,buf.size(),fi)!=buf.size()) return 1;
  top.p_out__ready.set<bool>(true);
  // reset once: the core free-runs across frames (continuous video)
  top.p_rst.set<bool>(true); top.p_in__valid.set<bool>(false);
  for (int i=0;i<2;i++){{ top.p_clk.set<bool>(false); top.step(); top.p_clk.set<bool>(true); top.step(); }}
  top.p_rst.set<bool>(false);
  for (int f=0; f<NF; f++) {{
    long base=(long)f*HW; int idx=0, got=0; long cyc=0, cap=(long)(HW+NOUT)*8+100000;
    while (got<NOUT && cyc<cap) {{
      bool iv = idx<HW;
      top.p_in__valid.set<bool>(iv);
      top.p_in__data.set<uint32_t>(iv ? buf[base+idx] : 0);
      top.p_out__ready.set<bool>(true);
      top.p_clk.set<bool>(false); top.step();
      bool inr = top.p_in__ready.get<uint32_t>()!=0;        // combinational output
      bool ov  = top.p_out__valid.curr.get<uint32_t>()!=0;  // registered output
      uint32_t od = top.p_out__data.curr.get<uint32_t>();
      top.p_clk.set<bool>(true); top.step();
      if (iv && inr) idx++;
      if (ov) {{ fwrite(&od,4,1,fo); got++; }}
      cyc++;
    }}
  }}
  fclose(fo); fclose(fi); return 0;
}}
"""


# --------------------------------------------------------------------------- #
# Per-frame runner for the live viewer (numpy real-time, or cxxrtl "live HW")
# --------------------------------------------------------------------------- #

def _cxxrtl_runner(fn, params, pvals, W, H, module="vmodel"):
    """Compile the model to a cxxrtl binary once; return run(frame, pv)->frame
    that streams one frame through it, with live param values pv per call."""
    work = tempfile.mkdtemp(prefix="np2hw_view_")
    meta = _compile(fn, params, W, H, module)
    orows, ocols = meta["out_rows"], meta["out_cols"]
    pspec = meta["params"]                                # [(name, bits)] in port order
    open(os.path.join(work, f"{module}.v"), "w").write(meta["verilog"] + "\n")
    subprocess.run(["yosys", "-q", "-p",
                    f"read_verilog {module}.v; write_cxxrtl {module}_cxx.cc"],
                   cwd=work, check=True, capture_output=True)
    base_pv = {p.name: int(pvals[p.name]) for p in _leaves(params)}
    open(os.path.join(work, "drv.cc"), "w").write(
        _cxxrtl_driver(module, W, H, 1, orows * ocols, pspec, base_pv))
    datdir = subprocess.run(["yosys-config", "--datdir"], capture_output=True,
                            text=True, check=True).stdout.strip()
    inc = os.path.join(datdir, "include", "backends", "cxxrtl", "runtime")
    subprocess.run(["c++", "-std=c++14", "-O2", "-I", inc, "drv.cc", "-o", "sim"],
                   cwd=work, check=True, capture_output=True)
    dt = np.int32 if meta.get("signed") else np.uint32

    def run(frame, pv=None):
        pv = pv or base_pv
        frame.astype(np.uint8).tofile(os.path.join(work, "in.bin"))
        argv = ["./sim", "in.bin", "out.bin"] + [str(int(pv[n])) for n, _ in pspec]
        subprocess.run(argv, cwd=work, check=True, capture_output=True)
        flat = np.fromfile(os.path.join(work, "out.bin"), dtype=dt)
        return flat.reshape(orows, ocols).astype(np.uint8)
    return run


def _make_runner(fn, params, pvals, backend, sim, W, H, sb_tool="verilator"):
    """Return run(frame, pv)->frame, where pv overrides param values per frame."""
    base = {p.name: int(pvals[p.name]) for p in _leaves(params)}
    if backend == "numpy":
        def run(f, pv=None):
            return np.asarray(fn(f, *_value_args(params, pv or base))).astype(np.uint8)
        return run
    if backend in ("switchboard", "sb"):
        # persistent SB sim built ONCE; cleaned up on process exit.
        import atexit
        leaves = _leaves(params)
        if leaves:                                       # control plane -> LIVE registers
            dut, tx, rx, ctrl, meta, wrap = _sb_build_ctrl(fn, params, pvals, W, H, sb_tool)
            atexit.register(dut.terminate)
            off, applied = wrap["offsets"], {}

            def run(f, pv=None):
                pv = pv or base
                for n, v in pv.items():                  # write changed regs over AXI-Lite
                    if applied.get(n) != int(v):
                        ctrl.write(off[n], np.uint32(int(v) & 0xFFFFFFFF))
                        applied[n] = int(v)
                return _sb_stream_frame(tx, rx, meta, wrap, f)
            return run
        dut, tx, rx, meta, wrap = _sb_build(fn, params, pvals, W, H, sb_tool)
        atexit.register(dut.terminate)
        return lambda f, pv=None: _sb_stream_frame(tx, rx, meta, wrap, f)
    if sim == "cxxrtl":
        return _cxxrtl_runner(fn, params, pvals, W, H)
    raise SystemExit("live sim needs --sim cxxrtl (iverilog per-frame is too slow)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_params(items, params):
    pvals = {}
    for it in items or []:
        k, _, v = it.partition("=")
        pvals[k.strip()] = int(v)
    for p in _leaves(params):
        if p.name not in pvals:
            raise SystemExit(f"model needs --param {p.name}=<value>")
    return pvals


def cmd_run(args):
    fn, params, label = _load_model(args.model)
    pvals = _parse_params(args.param, params)
    frames, is_video, meta = _load(args.input)
    H, W = frames[0].shape
    kind = f"video ({len(frames)} frames)" if is_video else "image"
    how = {"numpy": "numpy", "switchboard": "switchboard", "sb": "switchboard"
           }.get(args.backend, f"sim:{args.sim}")
    print(f"{label}: {kind} {W}x{H}, run via {how}")
    if args.backend == "numpy":
        outs = _run_numpy(fn, params, pvals, frames)
    elif args.backend in ("switchboard", "sb"):
        outs = _run_switchboard(fn, params, pvals, frames, tool=args.sb_tool)
    elif args.sim == "cxxrtl":
        outs = _run_cxxrtl(fn, params, pvals, frames)
    else:
        outs = _run_iverilog(fn, params, pvals, frames)
    _save(args.output, outs, meta)
    oh, ow = outs[0].shape
    print(f"wrote {args.output}  ({ow}x{oh})")


def _wh(s):
    w, _, h = s.lower().partition("x")
    return (int(w), int(h))


def cmd_view(args):
    from . import view as V
    fn, params, label = _load_model(args.model)
    # lenient: sliders set the registers live, so --param is only the *initial*
    # value (defaults to mid-range when omitted).
    given = {}
    for it in args.param or []:
        k, _, v = it.partition("=")
        given[k.strip()] = int(v)
    leaves = _leaves(params)
    pvals = {}
    for p in leaves:
        if getattr(p, "default", 0):
            pvals[p.name] = given.get(p.name, p.default)
        else:
            lo, hi = ((-(1 << (p.bits - 1)), (1 << (p.bits - 1)) - 1) if p.signed
                      else (0, (1 << p.bits) - 1))
            pvals[p.name] = given.get(p.name, (lo + hi) // 2)
    W, H = _wh(args.size)
    sb = args.backend in ("switchboard", "sb")
    how = "numpy" if args.backend == "numpy" else ("switchboard" if sb else f"sim:{args.sim}")
    if sb:
        # SB sim is built once; with config registers it gets an AXI-Lite control
        # interface so sliders write registers live (over the SB control plane).
        print(f"{label}: source={args.source} {W}x{H}, run via switchboard "
              f"({args.sb_tool}) -- building sim once (~10s)...")
    run = _make_runner(fn, params, pvals, args.backend, args.sim, W, H,
                       sb_tool=args.sb_tool)
    frames = args.frames if args.snapshot else None
    src = V.open_source(args.source, (W, H), camera=args.camera, frames=frames)
    if not sb:
        print(f"{label}: source={args.source} {W}x{H}, run via {how}"
              + (f", live controls: {', '.join(p.name for p in leaves)}" if leaves else ""))
    if args.snapshot:
        shp = V.snapshot(src, run, args.snapshot, n=args.frames, pv=pvals)
        print(f"wrote {args.snapshot}  (montage {shp[1]}x{shp[0]}, input|output)")
    else:
        print("live window: drag the sliders to change registers"
              + (" (written over the SB AXI-Lite control plane)" if sb else "")
              + "; close to stop.")
        V.live(src, run, params=leaves, init_pv=pvals,
               title=f"np2hw: {label} [{how}] {W}x{H}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="np2hw", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="apply a model (.py file) from file to file")
    r.add_argument("model", help="path to a model .py (or file.py:func)")
    r.add_argument("input")
    r.add_argument("output")
    r.add_argument("--backend", choices=("numpy", "sim", "rtl", "switchboard", "sb"),
                   default="numpy",
                   help="numpy = reference model (fast); sim = run the generated "
                        "hardware (rtl is an alias for sim); switchboard = run "
                        "through ZeroAsic's Switchboard stack (PySbTx/PySbRx queues)")
    r.add_argument("--sim", choices=("iverilog", "cxxrtl"), default="iverilog",
                   help="simulator engine for --backend sim "
                        "(cxxrtl = compiled, fast, best for video)")
    r.add_argument("--sb-tool", choices=("verilator", "icarus"), default="verilator",
                   help="simulator engine for --backend switchboard "
                        "(verilator = fast; icarus = simpler build)")
    r.add_argument("--param", action="append", help="register value, e.g. gain=24")
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("view", help="live: capture -> model -> display (input|output)")
    v.add_argument("model", help="path to a model .py (or file.py:func)")
    v.add_argument("--source", choices=("camera", "screen", "test"), default="screen")
    v.add_argument("--backend", choices=("numpy", "sim", "switchboard", "sb"),
                   default="numpy", help="numpy/sim (live, sliders) or switchboard "
                        "(through the SB stack; params fixed at build, slower)")
    v.add_argument("--sim", choices=("cxxrtl",), default="cxxrtl")
    v.add_argument("--sb-tool", choices=("verilator", "icarus"), default="verilator",
                   help="engine for --backend switchboard")
    v.add_argument("--size", default="160x120", help="processing size WxH")
    v.add_argument("--camera", type=int, default=0, help="camera index")
    v.add_argument("--param", action="append", help="register value, e.g. gain=24")
    v.add_argument("--snapshot", help="headless: write a montage file instead of a window")
    v.add_argument("--frames", type=int, default=8, help="frames for --snapshot")
    v.set_defaults(func=cmd_view)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
