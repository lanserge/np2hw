# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Snapshot what np2hw produces, and prove a change made nothing worse.

A refactor of the compiler's internals must leave its OUTPUT alone and must
not make it slower. Both are measured, not argued:

    python scripts/snapshot.py take BASE        # before the change
    python scripts/snapshot.py take NEW         # after it
    python scripts/snapshot.py compare BASE NEW

`take` runs every example (the examples are the test suite) and records

  * every Verilog file the example writes -- modules, wrappers, testbenches,
    in the order written -- by intercepting file writes, and every module
    `generate()` returns, written or not, so nothing an example emits can
    escape the comparison;
  * the time spent in tracing (`to_ir`) and generation (`generate`), the
    outermost call only, the best of `--repeat` runs.

`compare` is the gate, and its rule is BIT-EXACTNESS: every example must
pass, each checking the generated Verilog against NumPy in simulation.
Differing Verilog is listed; `--synth` also synthesises each differing
module both ways (Yosys, Xilinx 7-series, the best of several orderings)
and reports its figures -- a report, never a gate: at this grain the
tools' own noise exceeds the differences. `--strict` fails on any
difference, the rule for a refactor that promises byte-identical output.

Compile time is REPORTED, not gated: the compiler may spend more time for
simplicity.

The examples share `build/`, so nothing else may run them while a snapshot
is being taken.
"""
from __future__ import annotations

import argparse
import builtins
import difflib
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
SKIP = {"_harness.py", "switchboard.py"}     # as CI: switchboard needs its stack



def examples() -> list[str]:
    return sorted(f for f in os.listdir(EXAMPLES)
                  if f.endswith(".py") and f not in SKIP)


# --------------------------------------------------------------------------- #
# one example, in its own process
# --------------------------------------------------------------------------- #

def _run_one(example: str, out_json: str) -> int:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, EXAMPLES)
    import np2hw
    import np2hw.core.pipeline as pipeline
    import np2hw.verilog as verilog

    written: list[list[str]] = []            # [basename, text], in order
    real_open = builtins.open

    class _Capture:
        def __init__(self, fh, name):
            self._fh, self._name, self._parts = fh, name, []

        def write(self, text):
            self._parts.append(text)
            return self._fh.write(text)

        def close(self):
            if self._parts is not None:
                written.append([self._name, "".join(self._parts)])
                self._parts = None
            return self._fh.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

        def __getattr__(self, attr):
            return getattr(self._fh, attr)

    def capturing_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        if (isinstance(file, (str, os.PathLike)) and "w" in mode
                and str(file).endswith((".v", ".sv"))):
            return _Capture(fh, os.path.basename(str(file)))
        return fh

    builtins.open = capturing_open

    times = {"to_ir": 0.0, "generate": 0.0, "calls": 0}
    depth = [0]

    def timed(fn, key):
        def wrapper(*args, **kwargs):
            if depth[0]:
                return fn(*args, **kwargs)
            depth[0] += 1
            t = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                if key == "generate" and isinstance(result, dict) \
                        and "verilog" in result:
                    # what generate() RETURNS, written to disk or not
                    written.append([f"generate:{result.get('module')}",
                                    result["verilog"]])
                return result
            finally:
                times[key] += time.perf_counter() - t
                times["calls"] += 1
                depth[0] -= 1
        return wrapper

    t_ir = timed(pipeline.to_ir, "to_ir")
    t_gen = timed(verilog.generate, "generate")
    pipeline.to_ir = np2hw.to_ir = t_ir
    verilog.generate = np2hw.generate = t_gen

    old_cwd = os.getcwd()
    status = 0
    try:
        os.chdir(EXAMPLES)
        sys.argv = [example]
        runpy.run_path(os.path.join(EXAMPLES, example), run_name="__main__")
    except SystemExit as e:
        status = int(e.code or 0) if not isinstance(e.code, str) else 1
    except BaseException:                     # an example that raised FAILS,
        import traceback                      # and is still recorded
        traceback.print_exc()
        status = 1
    finally:
        os.chdir(old_cwd)
        import gc
        gc.collect()                          # flush unclosed captures
        builtins.open = real_open
    with real_open(out_json, "w") as fh:
        json.dump({"status": status, "times": times, "written": written}, fh)
    return 0


# --------------------------------------------------------------------------- #
# take / compare
# --------------------------------------------------------------------------- #

def take(out_dir: str, repeat: int, only: list[str] | None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    python = sys.executable
    failed = []
    for ex in examples():
        if only and ex not in only:
            continue
        best = None
        for _ in range(repeat):
            tmp = os.path.join(out_dir, f".{ex}.run.json")
            subprocess.run([python, __file__, "_one", ex, tmp],
                           cwd=ROOT, capture_output=True, text=True)
            with open(tmp) as fh:
                run = json.load(fh)
            os.remove(tmp)
            if best is None:
                best = run
            else:
                if run["written"] != best["written"]:
                    print(f"  {ex}: NONDETERMINISTIC output between runs")
                    failed.append(ex)
                for k in ("to_ir", "generate"):
                    best["times"][k] = min(best["times"][k], run["times"][k])
        record = {
            "status": best["status"],
            "times": best["times"],
            "files": [[name, hashlib.sha256(text.encode()).hexdigest()]
                      for name, text in best["written"]],
        }
        with open(os.path.join(out_dir, ex + ".json"), "w") as fh:
            json.dump(record, fh, indent=1)
        vdir = os.path.join(out_dir, ex[:-3])
        os.makedirs(vdir, exist_ok=True)
        for i, (name, text) in enumerate(best["written"]):
            with open(os.path.join(vdir, f"{i:03d}_{name}"), "w") as fh:
                fh.write(text)
        t = best["times"]["to_ir"] + best["times"]["generate"]
        flag = "" if best["status"] == 0 else "  (example FAILED)"
        print(f"  {ex:<24} {len(record['files']):>3} files  "
              f"{t * 1000:8.1f} ms{flag}")
        if best["status"]:
            failed.append(ex)
    return 1 if failed else 0


QUALITY = ("luts", "lutram", "ffs", "dsps", "brams", "carry", "delay_ps")


#: Orderings each module is synthesised in. The mapper's result depends on
#: the order logic is presented in: the same module with only its
#: continuous assignments reordered was measured at 67 to 72 LUTs. The best
#: of several orderings measures the design, not one ordering's luck.
ORDERINGS = 5


def quality(verilog_path: str, top: str) -> dict:
    """The design's figures: the best, figure by figure, over ORDERINGS
    logically identical orderings of its continuous assignments."""
    import random
    import tempfile
    with open(verilog_path) as fh:
        src = fh.read().split("\n")
    wires = [i for i, l in enumerate(src)
             if l.lstrip().startswith(("wire ", "assign ")) and l.rstrip().endswith(";")]
    best = None
    for seed in range(ORDERINGS):
        lines = src[:]
        if seed:
            order = [src[i] for i in wires]
            random.Random(seed).shuffle(order)
            for i, l in zip(wires, order):
                lines[i] = l
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as fh:
            fh.write("\n".join(lines))
            path = fh.name
        try:
            fig = _quality_once(path, top)
        finally:
            os.remove(path)
        best = fig if best is None else {k: min(best[k], fig[k]) for k in fig}
    return best


def _quality_once(verilog_path: str, top: str) -> dict:
    """Synthesise one module and measure it. Xilinx 7-series: the family np2hw
    designs run on, and a mapping whose figures mean something.

    `delay_ps` is the latest arrival time from static timing on the mapped
    cells, with the delays the Xilinx cell library states: the logic on the
    critical path, without routing. Enough to catch a rewrite that lengthens
    a path; a clock is still signed off by the vendor tool."""
    script = (f"read_verilog -sv {verilog_path}; "
              f"synth_xilinx -family xc7 -top {top} -abc9; flatten; "
              "stat; sta")
    r = subprocess.run(["yosys", "-q", "-p", script, "-l", "/dev/stdout"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"yosys failed on {top}: {r.stderr.strip()[-300:]}")
    fig = dict.fromkeys(QUALITY, 0)
    in_block = False
    for line in r.stdout.splitlines():
        if line.strip() == f"=== {top} ===":
            in_block = True               # the LAST statistics block: synth
            for k in QUALITY:             # prints its own before ours
                if k != "delay_ps":
                    fig[k] = 0
            continue
        if in_block and line.startswith("==="):
            in_block = False
        if line.startswith("Latest arrival time in"):
            fig["delay_ps"] = int(line.rsplit(" ", 1)[1].rstrip(":"))
        if not in_block:
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        n, cell = int(parts[0]), parts[1]
        if cell.startswith("LUT"):
            fig["luts"] += n
        elif cell.startswith(("RAM32", "RAM64", "RAM128", "RAM256", "SRL")):
            fig["lutram"] += n
        elif cell.startswith("FD"):
            fig["ffs"] += n
        elif cell.startswith("DSP48"):
            fig["dsps"] += n
        elif cell.startswith("RAMB"):
            fig["brams"] += n
        elif cell.startswith("CARRY"):
            fig["carry"] += n
    return fig


def compare(base: str, new: str, strict: bool = False,
            synth: bool = False) -> int:
    """The gate: every example must pass -- its results bit-exact against
    NumPy. Differing Verilog is listed; with `synth` each differing module
    is also synthesised both ways and its figures reported, never gated.
    With `strict`, any difference fails: the rule for a refactor that
    promises byte-identical output."""
    problems, base_total, new_total, differing = [], 0.0, 0.0, 0
    taken = sorted(f[:-5] for f in os.listdir(base) if f.endswith(".py.json"))
    for ex in taken:
        bp, np_ = (os.path.join(d, ex + ".json") for d in (base, new))
        if not os.path.exists(np_):
            problems.append(f"{ex}: in the base snapshot, missing from the new")
            continue
        with open(bp) as fh:
            b = json.load(fh)
        with open(np_) as fh:
            n = json.load(fh)
        if n["status"] != 0:
            problems.append(f"{ex}: the example fails -- not bit-exact")
        if b["files"] != n["files"]:
            differing += _report(base, new, ex, b["files"], n["files"], synth)
            if strict:
                problems.append(f"{ex}: Verilog differs (strict)")
        bt = b["times"]["to_ir"] + b["times"]["generate"]
        nt = n["times"]["to_ir"] + n["times"]["generate"]
        base_total += bt
        new_total += nt
    ratio = new_total / base_total if base_total else 1.0
    print(f"{differing} differing module(s)")
    print(f"compile time (reported, not gated): {base_total * 1000:.0f} ms "
          f"-> {new_total * 1000:.0f} ms ({ratio:.2f}x)")
    for p in problems:
        print("  FAIL", p)
    print("BIT-EXACT" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _report(base, new, ex, bf, nf, synth) -> int:
    """List what differs in one example; with `synth`, each differing
    module's figures, both ways."""
    if [f[0] for f in bf] != [f[0] for f in nf]:
        print(f"  {ex}: a different set of files is written")
        return 1
    count = 0
    for i, (b, n) in enumerate(zip(bf, nf)):
        if b == n or not b[0].startswith("generate:"):
            continue
        count += 1
        top = b[0].split(":", 1)[1]
        if not synth:
            print(f"  {ex} {top}: Verilog differs")
            continue
        name = f"{i:03d}_{b[0]}"
        qb = quality(os.path.join(base, ex[:-3], name), top)
        qn = quality(os.path.join(new, ex[:-3], name), top)
        line = "  ".join(f"{k} {qb[k]}->{qn[k]}" for k in QUALITY
                         if qb[k] != qn[k]) or "same figures"
        print(f"  {ex} {top}: {line}")
    return count


def _show_first_diff(base, new, ex, bf, nf):
    for i, (b, n) in enumerate(zip(bf, nf)):
        if b != n:
            name = f"{i:03d}_{b[0]}"
            with open(os.path.join(base, ex[:-3], name)) as fh:
                a = fh.read().splitlines()
            with open(os.path.join(new, ex[:-3], f"{i:03d}_{n[0]}")) as fh:
                c = fh.read().splitlines()
            for line in list(difflib.unified_diff(a, c, lineterm="", n=1))[:20]:
                print("      " + line)
            return
    print(f"      file count {len(bf)} -> {len(nf)}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_one":
        return _run_one(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("take")
    t.add_argument("out")
    t.add_argument("--repeat", type=int, default=3)
    t.add_argument("--only", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--strict", action="store_true",
                   help="any Verilog difference fails (refactors)")
    c.add_argument("--synth", action="store_true",
                   help="report synthesis figures of differing modules")
    m = sub.add_parser("quality", help="synthesise one module and measure it")
    m.add_argument("verilog")
    m.add_argument("top")
    a = ap.parse_args()
    if a.cmd == "take":
        return take(a.out, a.repeat, a.only)
    if a.cmd == "quality":
        print(json.dumps(quality(a.verilog, a.top)))
        return 0
    return compare(a.base, a.new, a.strict, a.synth)


if __name__ == "__main__":
    sys.exit(main())
