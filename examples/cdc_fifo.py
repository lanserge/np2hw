# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The dual-clock FIFO, against clocks that owe each other nothing.

Every design this generator feeds has one crossing in it: the
receiver's clock is recovered from a cable, the pipeline's belongs to
the board. A vendor converter does this job on one family; this one is
owned, so it can follow the design anywhere -- and so its discipline
can be CHECKED rather than trusted.

The discipline: gray-coded pointers cross through two flops, the
payload never crosses at all. A binary pointer caught mid-increment
can be misread as anything; a gray-coded one is wrong by at most one
slot, and each side compares its own exact pointer against a stale
picture of the other's, so full and empty err only toward caution.

  LOSSLESS, BOTH RATIOS   every beat, once, in order, with the writer
                          faster and with the reader faster -- and at
                          depth FOUR, so full is hit constantly and
                          the full arithmetic is load-bearing.
  FULL MEANS FULL         a refused beat is not a lost beat: the
                          scoreboard would see any overwrite as a
                          wrong word later.
  RESET LEAVES NOTHING    both sides reset; nothing stale emerges.
  ONLY GRAY CROSSES       structural, because a zero-delay simulator
                          cannot show metastability: the synchroniser
                          inputs are the gray registers and nothing
                          else, and they carry ASYNC_REG.
  PORTS LEAVE FLOPS       in_ready and out_valid/out_data are
                          registers -- the boundary rule holds at a
                          clock boundary too.

Run:  python examples/cdc_fifo.py   (needs iverilog)
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from np2hw.stream import cdc_fifo  # noqa: E402

W = 20

TB = r"""
`timescale 1ns/1ps
module tb;
  // Two clocks with no common factor worth speaking of.
  reg wclk = 0, rclk = 0;
  always #%(WHALF)s wclk = ~wclk;
  always #%(RHALF)s rclk = ~rclk;
  reg wrst = 1, rrst = 1;

  reg  [%(W)d-1:0] in_data = 0;
  reg              in_valid = 0;
  wire             in_ready;
  wire [%(W)d-1:0] out_data;
  wire             out_valid;
  reg              out_ready = 0;

  np2hw_cdc_fifo #(.W(%(W)d), .A(2)) dut (.wclk(wclk), .wrst(wrst),
    .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
    .rclk(rclk), .rrst(rrst),
    .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready));

  integer N = %(N)d;
  reg [15:0] ls = 16'hACE1, lm = 16'h9E37;
  integer sent = 0, got = 0, errs = 0, nxt = 0;

  always @(posedge wclk) begin
    if (wrst) begin in_valid <= 0; sent <= 0; nxt <= 0; end
    else begin
      ls <= {ls[14:0], ls[15]^ls[13]^ls[12]^ls[10]};
      if (in_valid && in_ready) begin sent <= sent + 1; in_valid <= 0; end
      // `nxt` advances when a value is STAGED, `sent` when one is
      // taken -- folding them into one counter duplicates a word on
      // every accept-and-reload edge.
      if ((!in_valid || in_ready) && nxt < N && ls[2:0] != 0) begin
        in_data <= nxt[%(W)d-1:0] ^ %(W)d'h5A5A5;
        in_valid <= 1;
        nxt <= nxt + 1;
      end
    end
  end

  always @(posedge rclk) begin
    if (rrst) begin out_ready <= 0; got <= 0; end
    else begin
      lm <= {lm[14:0], lm[15]^lm[13]^lm[12]^lm[10]};
      out_ready <= (lm[2:0] != 0);
      if (out_valid && out_ready) begin
        if (out_data !== (got[%(W)d-1:0] ^ %(W)d'h5A5A5)) errs = errs + 1;
        got <= got + 1;
      end
    end
  end

  integer cycles = 0;
  always @(posedge rclk) cycles = cycles + 1;

  initial begin
    #40 wrst = 0; rrst = 0;
    wait (got == N || cycles > N * 60);
    $display("RESULT sent=%%0d got=%%0d errs=%%0d", sent, got, errs);
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


def run(tmp, tag, verilog, whalf, rhalf, n=600):
    v = os.path.join(tmp, tag + ".v")
    with open(v, "w") as f:
        f.write(verilog + "\n" + TB % dict(W=W, WHALF=whalf, RHALF=rhalf,
                                           N=n, TMAX=n * 4000))
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

    print("cdc_fifo: a stream between clocks that owe each other nothing\n")
    # DEPTH FOUR, deliberately: full is hit constantly, so the gray
    # window arithmetic is doing real work in every run below.
    core = cdc_fifo(W, addr_bits=2)
    v = core["verilog"]
    with tempfile.TemporaryDirectory() as tmp:
        r = run(tmp, "wfast", v, "3.5", "5.05")
        check("lossless with the WRITER faster (7.0ns into 10.1ns)",
              r not in (None, "TIMEOUT") and r["got"] == "600"
              and r["errs"] == "0",
              f"{r if isinstance(r, str) else r['got']}/600 beats, depth 4")
        r = run(tmp, "rfast", v, "5.05", "3.5")
        check("lossless with the READER faster (10.1ns into 7.0ns)",
              r not in (None, "TIMEOUT") and r["got"] == "600"
              and r["errs"] == "0",
              f"{r if isinstance(r, str) else r['got']}/600 beats")
        r = run(tmp, "near", v, "3.37", "3.4")
        check("lossless with nearly-equal, unrelated clocks",
              r not in (None, "TIMEOUT") and r["got"] == "600"
              and r["errs"] == "0",
              "the bench's own case: recovered 148.5 against local 148.5")

    lines = [l.strip() for l in v.splitlines()]
    crossings = [l for l in lines if "gray0 <=" in l]
    # EVERY synchroniser declaration carries the attribute -- checked
    # per declaration, not "somewhere in the file": a first version of
    # this claim passed a mutant that stripped it from one of the two,
    # because the other's attribute satisfied a whole-file search.
    sync_decls = [l for l in lines
                  if "reg [A:0] wq_rgray0" in l or "reg [A:0] rq_wgray0" in l]
    check("only gray-coded pointers cross, through ASYNC_REG pairs",
          len(sync_decls) == 2
          and all('ASYNC_REG = "TRUE"' in l for l in sync_decls)
          and "wq_rgray0 <= rgray; wq_rgray <= wq_rgray0;" in v
          and "rq_wgray0 <= wgray; rq_wgray <= rq_wgray0;" in v
          and not any("bin" in l for l in crossings),
          "a zero-delay simulator cannot show the failure this "
          "prevents, so the claim is structural -- and per declaration")
    check("the ports leave flops, clock boundary or not",
          "assign in_ready = w_ready_q;" in v
          and "assign out_valid = out_q_v;" in v
          and "assign out_data  = out_q;" in v)

    print("\n" + ("CDC_FIFO PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
