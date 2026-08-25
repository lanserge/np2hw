# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The tee: one stream, two DISPOSABLE consumers, and it never stalls.

A fork that advances its branches in lockstep is correct only for
consumers that always come back -- and a consumer owned by software
does not: a capture engine halted mid-frame leaves such a fork
waiting for a frame boundary that can never arrive, and a script
takes down the picture. This tee is a pure parallel tap. Upstream is always
accepted; a branch that cannot keep up loses its own beats, never its
sibling's; honesty replaces losslessness.

  NEVER STALLS            in_ready is constant: across every scenario
                          below, including a consumer murdered
                          mid-frame, the input is never once refused.
  KEEPING UP IS LOSSLESS  both consumers ready: each receives every
                          beat, in order, nothing torn.
  A DEATH IS PRIVATE      a consumer that dies mid-frame tears ITS
                          branch (sticky says so), drops out of the
                          frame, and the sibling receives every beat
                          as if nothing happened.
  TORN, NEVER HEADLESS    a slow consumer receives whole frame heads
                          or nothing: a tear truncates a tail; no
                          branch ever receives a frame it did not
                          start.
  BOTH OFF IS STANDBY     the input drains freely and nothing comes
                          out -- the pipeline lives while nothing is
                          displayed or grabbed.
  JOINS AT SOF, WHOLE     an enable flipped mid-frame changes nothing
                          until the next frame start; leaving by
                          DISABLE is not a tear and raises nothing.

Run:  python examples/tee.py   (needs iverilog)
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from np2hw.stream import tee  # noqa: E402

W = 12
FRAME = 16          # beats per frame in the testbench

TB = r"""
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1;
  reg        in_valid = 0;
  wire       in_ready;
  reg [%(W)d-1:0] in_data = 0;
  reg        in_sof = 0;
  reg en_a = %(ENA)d, en_b = %(ENB)d;
  wire a_valid, b_valid;
  reg  a_ready = 0, b_ready = 0;
  wire [%(W)d-1:0] a_data, b_data;
  wire torn_a, torn_b;

  tee dut (.clk(clk), .rst(rst),
    .in_valid(in_valid), .in_ready(in_ready),
    .in_data(in_data), .in_sof(in_sof),
    .en_a(en_a), .en_b(en_b),
    .a_valid(a_valid), .a_ready(a_ready), .a_data(a_data),
    .b_valid(b_valid), .b_ready(b_ready), .b_data(b_data),
    .torn_a(torn_a), .torn_b(torn_b));

  reg [15:0] ls = 16'hACE1, la = 16'h4242, lb = 16'h9E37;
  integer nxt = 0, sent = 0;
  integer FRAME = %(FRAME)d, N = %(N)d;

  always @(posedge clk) begin
    ls <= {ls[14:0], ls[15]^ls[13]^ls[12]^ls[10]};
    la <= {la[14:0], la[15]^la[13]^la[12]^la[10]};
    lb <= {lb[14:0], lb[15]^lb[13]^lb[12]^lb[10]};
    // A consumer models one of: KEEPING UP (ready high), flaky
    // (random ready), DEAD from the start, or MURDERED mid-run --
    // the case a lockstep fork cannot survive.
    a_ready <= ((la[2:0] != 0) || %(AFULL)d) && !%(ADEAD)d
               && !(%(ADIE)d && nxt > N/2);
    b_ready <= ((lb[2:0] != 0) || %(BFULL)d) && !%(BDEAD)d
               && !(%(BDIE)d && nxt > N/2);
    if (rst) begin in_valid <= 0; nxt <= 0; sent <= 0; end
    else begin
      if (in_valid && in_ready) begin sent <= sent + 1; in_valid <= 0; end
      if ((!in_valid || in_ready) && nxt < N && (ls[1:0] != 0)) begin
        in_data  <= nxt[%(W)d-1:0];
        in_sof   <= (nxt %% FRAME) == 0;
        in_valid <= 1;
        nxt      <= nxt + 1;
        // mid-frame enable flips: must not take effect until next SOF
        if ((nxt %% FRAME) == (FRAME/2)) begin
          en_a <= %(FLIPA)d ? !en_a : en_a;
          en_b <= %(FLIPB)d ? !en_b : en_b;
        end
      end
    end
  end

  // THE LAW: the input is never refused, whatever the branches do
  integer nrl = 0;
  always @(posedge clk) if (!rst && !in_ready) nrl = nrl + 1;

  integer got_a = 0, got_b = 0, err_a = 0;
  integer partial = 0;   // a branch received a beat of a frame it
                         // did not start -- must never happen
  reg started_a = 0, started_b = 0;
  always @(posedge clk) if (!rst) begin
    if (a_valid && a_ready) begin
      if (a_data[3:0] == 0) started_a <= 1;
      else if (!started_a) partial = partial + 1;
      if (a_data !== got_a_expect()) err_a = err_a + 1;
      got_a = got_a + 1;
    end
    if (b_valid && b_ready) begin
      if (b_data[3:0] == 0) started_b <= 1;
      else if (!started_b) partial = partial + 1;
      got_b = got_b + 1;
    end
  end
  // branch A's expectation: whatever arrives must be CONSECUTIVE
  // within a frame -- across a leave, a tear or a rejoin there are
  // gaps of whole frames or truncated tails, never scrambled beats.
  reg [%(W)d-1:0] last_a = {%(W)d{1'b1}};
  function [%(W)d-1:0] got_a_expect;
    got_a_expect = (last_a == {%(W)d{1'b1}}) ? a_data
                 : (a_data[3:0] == 0) ? a_data : last_a + 1;
  endfunction
  always @(posedge clk) if (a_valid && a_ready) last_a <= a_data;

  integer cycles = 0;
  always @(posedge clk) cycles = cycles + 1;
  initial begin
    repeat (4) @(posedge clk); rst = 0;
    wait (sent == N || cycles > N * 40);
    repeat (8) @(posedge clk);
    $display("RESULT sent=%%0d a=%%0d b=%%0d erra=%%0d partial=%%0d ta=%%0d tb=%%0d nrl=%%0d",
             sent, got_a, got_b, err_a, partial, torn_a, torn_b, nrl);
    $finish;
  end
  initial begin #%(TMAX)d; $display("RESULT TIMEOUT"); $finish; end
endmodule
"""

ok = []


def check(label, passed, detail=""):
    ok.append(bool(passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


def run(tmp, tag, verilog, n=320, ena=1, enb=1, afull=0, bfull=0,
        flipa=0, flipb=0, adead=0, bdead=0, adie=0, bdie=0):
    v = os.path.join(tmp, tag + ".v")
    with open(v, "w") as f:
        f.write(verilog + "\n" + TB % dict(W=W, FRAME=FRAME, N=n,
                                           ENA=ena, ENB=enb,
                                           AFULL=afull, BFULL=bfull,
                                           ADEAD=adead, BDEAD=bdead,
                                           ADIE=adie, BDIE=bdie,
                                           FLIPA=flipa, FLIPB=flipb,
                                           TMAX=n * 900))
    exe = os.path.join(tmp, tag)
    r = subprocess.run(["iverilog", "-g2012", "-o", exe, v],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return None
    out = subprocess.run([exe], capture_output=True, text=True).stdout
    line = next((l for l in out.splitlines() if l.startswith("RESULT")), "")
    if not line or "TIMEOUT" in line:
        return "TIMEOUT"
    return dict(kv.split("=") for kv in line.split()[1:])


def main():
    if subprocess.run(["which", "iverilog"], capture_output=True).returncode:
        print("iverilog not found -- skipping")
        return 0

    print("tee: one stream, two disposable consumers, never a stall\n")
    core = tee(W)
    v = core["verilog"]
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, "both", v, afull=1, bfull=1)
        check("both keeping up: lossless, in order, nothing torn",
              r not in (None, "TIMEOUT") and r["a"] == r["sent"] == r["b"]
              and r["erra"] == "0" and r["partial"] == "0"
              and r["ta"] == "0" and r["tb"] == "0" and r["nrl"] == "0",
              f"{r if isinstance(r, str) else (r['sent'], r['a'], r['b'])}")
        r = run(tmp, "murder", v, afull=1, bfull=1, bdie=1)
        check("B murdered mid-frame: its tear is PRIVATE -- A receives "
              "every beat, the input never waits",
              r not in (None, "TIMEOUT") and r["a"] == r["sent"]
              and r["erra"] == "0" and r["tb"] == "1" and r["ta"] == "0"
              and int(r["b"]) < int(r["sent"]) and r["nrl"] == "0",
              f"B got {r['b'] if isinstance(r, dict) else r} of "
              f"{r['sent'] if isinstance(r, dict) else '?'}, torn and "
              "honest; A got all")
        r = run(tmp, "flaky", v, afull=1, bfull=0)
        check("B flaky: whole heads or honest tears, never a headless "
              "frame; A untouched",
              r not in (None, "TIMEOUT") and r["a"] == r["sent"]
              and r["erra"] == "0" and r["partial"] == "0"
              and r["ta"] == "0" and r["nrl"] == "0")
        r = run(tmp, "standby", v, ena=0, enb=0, adead=1, bdead=1)
        check("both off, both sinks DEAD: the input still drains -- "
              "standby, not stall",
              r not in (None, "TIMEOUT") and r["sent"] == "320"
              and r["a"] == "0" and r["b"] == "0" and r["nrl"] == "0")
        r = run(tmp, "flip", v, ena=1, enb=0, flipb=1, afull=1, bfull=1)
        check("an enable flipped MID-FRAME joins only at the next SOF",
              r not in (None, "TIMEOUT") and r["partial"] == "0"
              and int(r["b"]) > 0 and int(r["b"]) % FRAME == 0
              and r["tb"] == "0",
              f"B received {r['b'] if isinstance(r, dict) else r} beats -- "
              f"a multiple of the {FRAME}-beat frame, so only whole frames")
        r = run(tmp, "leave", v, ena=1, enb=1, flipa=1, afull=1, bfull=1)
        check("...and leaving by DISABLE ends at a frame boundary and "
              "raises nothing",
              r not in (None, "TIMEOUT") and r["partial"] == "0"
              and int(r["a"]) % FRAME == 0 and r["ta"] == "0"
              and r["nrl"] == "0",
              f"A received {r['a'] if isinstance(r, dict) else r}")

    check("the interface admits the combinational fan-out",
          tuple(core["interface"]["combinational_outputs"])
          == ("in_ready", "a_valid", "b_valid"),
          "valid fans out combinationally from in_valid; in_ready is "
          "constant true, which the never-stalls scenarios prove")

    print("\n" + ("TEE PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
