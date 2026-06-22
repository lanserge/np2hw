"""Shared test harness: compile a NumPy fn via the line IR, simulate, compare
to the SAME fn run on real uint8 data."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.ir import Image2D
from np2hw.frontend import to_ir
from np2hw.verilog import (generate, testbench, testbench_handshake,
                           switchboard_wrap, axis_video_wrap, testbench_sb_packed,
                           axil_regfile, umi_regfile, testbench_ctrl)

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
os.makedirs(BUILD, exist_ok=True)


def _param_arg(p, pv):
    """Reconstruct fn's argument for Param `p` from scalar-register values `pv`
    (keyed by scalar name). Scalar Param -> int; matrix Param -> ndarray."""
    if getattr(p, "shape", ()):
        out = np.zeros(p.shape, dtype=np.int64)
        for idx in np.ndindex(*p.shape):
            out[idx] = pv[f"{p.name}_" + "_".join(str(i) for i in idx)]
        return out
    return int(pv[p.name])


def _oracle(fn, A, params, pv):
    args = [A] + [_param_arg(p, pv) for p in params]
    return np.asarray(fn(*args)).astype(np.int64).flatten()


def _write_hex(A, bits):
    digits = (bits + 3) // 4                              # hex digits per pixel
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(p) & ((1 << bits) - 1):0{digits}x}"
                           for p in A.flatten()) + "\n")


def check_bp(name, fn, A, params=None, param_values=None, iface="core", bits=8):
    """Validate under RANDOMIZED ready/valid backpressure on one of:
      'core' — generic streaming core (out_last frames the frame)
      'sb'   — Switchboard adapter (last rides sb_out_flags[0])
      'axis' — AXI4-Stream Video adapter (tuser=SOF, tlast=EOL per line)
    Verifies output pixels AND the framing signals against the oracle."""
    params = params or []
    param_values = param_values or {}
    H, W = A.shape
    s, out = to_ir(fn, Image2D("img", W, H, bits=bits), *params)
    core = generate(out, module_name=name)
    expected = _oracle(fn, A, params, param_values)
    out_cols = core["out_cols"]

    files = [f"{name}.v"]
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    dut = core
    if iface == "sb":
        dut = switchboard_wrap(core, W, H, module_name=f"{name}_sb")
    elif iface == "axis":
        dut = axis_video_wrap(core, W, H, module_name=f"{name}_axis")
    if iface != "core":
        with open(os.path.join(BUILD, f"{dut['module']}.v"), "w") as fh:
            fh.write(dut["verilog"] + "\n")
        files.append(f"{dut['module']}.v")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(testbench_handshake(dut, W, H, expected.size, param_values,
                                     iface=iface) + "\n")
    _write_hex(A, bits)
    subprocess.run(["iverilog", "-o", "sim.vvp", *files, "tb.v"],
                   check=True, cwd=BUILD, capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD, capture_output=True)
    data = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64).reshape(-1, 3)
    got, f0, f1 = data[:, 0], data[:, 1], data[:, 2]
    n = expected.size
    idx = np.arange(n)
    if iface == "axis":
        exp0 = (idx == 0).astype(np.int64)               # SOF: first pixel
        exp1 = ((idx + 1) % out_cols == 0).astype(np.int64)  # EOL: each row end
        fnames = "(SOF,EOL)"
    else:
        exp0 = (idx == n - 1).astype(np.int64)           # last: final pixel
        exp1 = np.zeros(n, dtype=np.int64)
        fnames = "(last)"
    pix_ok = got.size == n and np.array_equal(got, expected)
    frm_ok = np.array_equal(f0, exp0) and np.array_equal(f1, exp1)
    ok = pix_ok and frm_ok
    label = f"{name} ({iface})"
    flag = "PASS" if ok else ("FAIL-pix" if not pix_ok else "FAIL-framing")
    print(f"  {label:<22} {n} px, random ready/valid, framing {fnames} -> {flag}")
    if not ok:
        print("    pix   :", got.tolist()[:12], "exp", expected.tolist()[:12])
        print("    frame0:", f0.tolist()[:12], "exp", exp0.tolist()[:12])
        print("    frame1:", f1.tolist()[:12], "exp", exp1.tolist()[:12])
    return ok


def check_sb_packed(name, fn, A, params=None, param_values=None, bits=8):
    """Validate the PACKED Switchboard adapter (gearbox): assemble input packets,
    drive SB ports under random backpressure, unpack output packets, compare to
    the oracle and check frame-aligned 'last'."""
    params = params or []
    param_values = param_values or {}
    H, W = A.shape
    s, out = to_ir(fn, Image2D("img", W, H, bits=bits), *params)
    core = generate(out, module_name=name)
    wrap = switchboard_wrap(core, W, H, module_name=f"{name}_sbp", pack=True)
    expected = _oracle(fn, A, params, param_values)

    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    with open(os.path.join(BUILD, f"{wrap['module']}.v"), "w") as fh:
        fh.write(wrap["verilog"] + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(testbench_sb_packed(wrap, W, H, expected.size, param_values) + "\n")
    _write_hex(A, bits)
    r = subprocess.run(["iverilog", "-o", "sim.vvp", f"{name}.v",
                        f"{wrap['module']}.v", "tb.v"],
                       check=True, cwd=BUILD, capture_output=True)
    out_run = subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD,
                             capture_output=True, text=True)
    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)
    pix_ok = got.size == expected.size and np.array_equal(got, expected)
    frm_ok = "FRAMING FAIL" not in out_run.stdout
    ok = pix_ok and frm_ok
    flag = "PASS" if ok else ("FAIL-pix" if not pix_ok else "FAIL-framing")
    print(f"  {name+' (sb-packed)':<22} {expected.size} px, "
          f"{wrap['p_in']}/{wrap['p_out']} px per packet -> {flag}")
    if not pix_ok:
        print("    DUT  :", got.tolist()[:12])
        print("    NumPy:", expected.tolist()[:12])
    return ok


def check(name, fn, A, params=None, param_values=None, show=0, bits=8):
    params = params or []
    param_values = param_values or {}
    H, W = A.shape
    s, out = to_ir(fn, Image2D("img", W, H, bits=bits), *params)
    meta = generate(out, module_name=name)

    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(meta["verilog"] + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(testbench(meta, W, H, param_values) + "\n")
    _write_hex(A, bits)
    subprocess.run(["iverilog", "-o", "sim.vvp", f"{name}.v", "tb.v"],
                   check=True, cwd=BUILD, capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD, capture_output=True)
    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)

    np_args = [A] + [int(param_values[p.name]) for p in params]
    expected = np.asarray(fn(*np_args)).astype(np.int64).flatten()
    ok = got.size == expected.size and np.array_equal(got, expected)
    tag = f"line_buffers={meta['M']} shift={meta['N']} out_bits={meta['out_bits']}"
    print(f"  {name:<12} {tag:<46} -> " + ("PASS" if ok else "FAIL"))
    if not ok or show:
        print("    DUT  :", got.tolist()[:show or 8])
        print("    NumPy:", expected.tolist()[:show or 8])
    return ok


def check_ctrl(name, fn, A, params, param_values, ctrl="axil", bits=8,
               frame_sync=False):
    """Program params over the control bus (AXI4-Lite or UMI) then stream a frame
    and validate the output reflects the written params. fn must be a plain
    (non-edge) pipeline (e.g. pointwise gain).

    frame_sync=True: shadow registers + update=out_last. Streams TWO frames; the
    written values take effect only at the frame boundary, so frame 1 uses the
    reset (zero) params and frame 2 uses the written ones -- proving glitch-free,
    frame-aligned updates."""
    H, W = A.shape
    s, out = to_ir(fn, Image2D("img", W, H, bits=bits), *params)
    core = generate(out, module_name=name)
    pspec = core["params"]                                # [(scalar_name, bits)]
    mk = axil_regfile if ctrl == "axil" else umi_regfile
    reg = mk(pspec, module_name=f"{name}_{ctrl}", frame_sync=frame_sync,
             defaults=core.get("param_defaults"))
    # register values keyed by scalar name (handles matrix Params transparently)
    pv = {n: int(param_values[n]) for n, _ in pspec}

    if frame_sync:
        zero_pv = {n: 0 for n, _ in pspec}                # reset/live value, frame 1
        exp1 = _oracle(fn, A, params, zero_pv)
        exp2 = _oracle(fn, A, params, param_values)       # frame 2, after update
        expected = np.concatenate([exp1, exp2])
        n_frames = 2
    else:
        expected = _oracle(fn, A, params, param_values)
        n_frames = 1

    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    with open(os.path.join(BUILD, f"{reg['module']}.v"), "w") as fh:
        fh.write(reg["verilog"] + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(testbench_ctrl(core, reg, W, H, expected.size, pv, ctrl=ctrl,
                                frame_sync=frame_sync, n_frames=n_frames) + "\n")
    _write_hex(A, bits)
    subprocess.run(["iverilog", "-o", "sim.vvp", f"{name}.v",
                    f"{reg['module']}.v", "tb.v"], check=True, cwd=BUILD,
                   capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD, capture_output=True)
    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)
    ok = got.size == expected.size and np.array_equal(got, expected)
    amap = ", ".join(f"{n}@0x{reg['offsets'][n]:02x}={pv[n]}" for n, _ in pspec)
    tag = ("shadow: frame1=reset, frame2=written" if frame_sync
           else f"streamed {expected.size} px")
    print(f"  {name+' ('+ctrl+('+sync' if frame_sync else '')+')':<22} "
          f"wrote [{amap}], {tag} -> " + ("PASS" if ok else "FAIL"))
    if not ok:
        print("    DUT  :", got.tolist()[:16])
        print("    NumPy:", expected.tolist()[:16])
    return ok
