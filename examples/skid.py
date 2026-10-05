# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The elastic register slice, and why the obvious version is wrong.

A core's timing claim is made about its own registers, and it holds only
while nothing else is on the path. A combinational output hands its depth
to every consumer before the consumer does any work of its own, so two
cores that each meet timing alone can miss it together -- and no
per-module model can see that coming, least of all when the composing is
done by a tool the generator never sees. One bench hit that twice from
the same boundary: -1.424ns and -1.792ns, from a receiver and a pipeline
that both passed in isolation.

A register at the boundary makes the claim true instead of modelled. But
a valid/ready interface cannot simply be registered: put one flop in the
forward path and `ready` still runs back combinationally, which is the
path you were trying to break; register `ready` as well and the source
cannot know in time, so it stalls every other cycle. The skid buffer
keeps ONE beat in reserve, which is exactly what lets both directions be
registered at no cost in throughput.

  NOTHING COMBINATIONAL  no input can move any output without a clock
                         edge -- forward AND backward. This is the whole
                         reason the module exists, so it is the claim
                         that has to be checked rather than assumed.
  LOSSLESS               every beat, once, in order, under random stall
                         from both ends. A register slice that drops or
                         repeats a beat under backpressure is the
                         classic way to get this wrong.
  FULL RATE              N beats in N cycles when the sink never stalls.
                         This is the claim the reserve slot buys, and
                         the one a single-flop slice fails -- so it is
                         also the check that would catch someone
                         "simplifying" the skid register away.
  RESERVE IS ENOUGH      one beat held back is sufficient: the source is
                         never refused while the slot is free, and the
                         slot never needs to hold two.
  RESET IS CLEAN         no beat survives a reset to appear afterwards.

Run:  python examples/skid.py   (needs iverilog)
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from np2hw import skid  # noqa: E402

W = 24

# The stream testbench. The source holds data stable while it is refused,
# as any well-behaved master must; the sink accepts on its own schedule.
TB_STREAM = r"""
`timescale 1ns/1ps
module tb;
  localparam W = %(W)d;
  localparam N = %(N)d;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg  [W-1:0] s_data = 0;
  reg          s_valid = 0;
  wire         s_ready;
  wire [W-1:0] m_data;
  wire         m_valid;
  reg          m_ready = 0;

  np2hw_skid #(.W(W)) dut (.clk(clk), .rst(rst),
    .s_data(s_data), .s_valid(s_valid), .s_ready(s_ready),
    .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready));

  // Deterministic gaps: a fixed LFSR, so a failure is reproducible.
  reg [15:0] ls = 16'h%(SEED_S)04x, lm = 16'h%(SEED_M)04x;
  always @(posedge clk) begin
    ls <= {ls[14:0], ls[15]^ls[13]^ls[12]^ls[10]};
    lm <= {lm[14:0], lm[15]^lm[13]^lm[12]^lm[10]};
  end

  integer sent = 0, got = 0, errs = 0, cycles = 0;
  integer refused_free = 0;   // offered, slot free, still refused
  reg [W-1:0] nxt = 0;

  always @(posedge clk) begin
    if (rst) begin
      s_valid <= 1'b0; s_data <= 0; nxt <= 0; sent <= 0;
    end else begin
      if (s_valid && s_ready) begin
        sent <= sent + 1;
        s_valid <= 1'b0;          // may be overridden just below
      end
      // Offer only when the previous beat has gone: holding data stable
      // under refusal is the master's side of the handshake.
      if (!s_valid || s_ready) begin
        if (nxt < N && (%(SRC_FULL)d || ls[3:0] < 4'd12)) begin
          s_data  <= nxt;
          s_valid <= 1'b1;
          nxt     <= nxt + 1;
        end
      end
    end
  end

  always @(posedge clk)
    if (rst) m_ready <= 1'b0;
    else     m_ready <= (%(SNK_FULL)d != 0) || (lm[3:0] < 4'd12);

  always @(posedge clk) begin
    if (!rst) begin
      cycles <= cycles + 1;
      if (m_valid && m_ready) begin
        // In order, exactly once: the counter is the whole scoreboard.
        if (m_data !== got[W-1:0]) errs <= errs + 1;
        got <= got + 1;
      end
    end
  end

  // Enough is enough: with the reserve slot empty the source must never
  // be refused. If this ever trips, one beat of reserve was not enough.
  always @(posedge clk)
    if (!rst && s_valid && !s_ready && dut.slot_free)
      refused_free <= refused_free + 1;

  integer t0 = 0;
  initial begin
    repeat (8) @(posedge clk);
    rst = 0;
    t0 = 0;
    while (got < N && cycles < N * 40) @(posedge clk);
    $display("RESULT sent=%%0d got=%%0d errs=%%0d cycles=%%0d refused=%%0d",
             sent, got, errs, cycles, refused_free);
    $finish;
  end
endmodule
"""

# The claim the module exists to make: no path from any input to any
# output without a clock edge. Changed mid-cycle, the outputs must not
# move until the next edge -- checked in both directions.
TB_REG = r"""
`timescale 1ns/1ps
module tb;
  localparam W = %(W)d;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg  [W-1:0] s_data = 0;
  reg          s_valid = 0;
  wire         s_ready;
  wire [W-1:0] m_data;
  wire         m_valid;
  reg          m_ready = 0;

  np2hw_skid #(.W(W)) dut (.clk(clk), .rst(rst),
    .s_data(s_data), .s_valid(s_valid), .s_ready(s_ready),
    .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready));

  integer fwd = 0, bwd = 0, stale = 0;
  reg [W-1:0] snap_d; reg snap_v, snap_r;

  task settle_and_snap;
    begin
      #1;
      snap_d = m_data; snap_v = m_valid; snap_r = s_ready;
    end
  endtask

  initial begin
    repeat (4) @(posedge clk);
    rst = 0;
    @(posedge clk);

    // --- forward: slam the input mid-cycle -------------------------
    settle_and_snap;
    s_valid = 1'b1; s_data = {W{1'b1}};
    #1;
    if (m_valid !== snap_v || m_data !== snap_d) fwd = fwd + 1;
    s_data = 24'h5A5A5A;
    #1;
    if (m_valid !== snap_v || m_data !== snap_d) fwd = fwd + 1;

    // --- backward: slam m_ready mid-cycle --------------------------
    @(posedge clk); settle_and_snap;
    m_ready = 1'b1; #1;
    if (s_ready !== snap_r) bwd = bwd + 1;
    m_ready = 1'b0; #1;
    if (s_ready !== snap_r) bwd = bwd + 1;

    // Fill both slots so the interesting state is occupied, then
    // repeat: an output that is registered only while empty is not
    // registered.
    m_ready = 1'b0; s_valid = 1'b1;
    repeat (6) @(posedge clk);
    settle_and_snap;
    s_data = 24'h123456; s_valid = 1'b0; #1;
    if (m_valid !== snap_v || m_data !== snap_d) fwd = fwd + 1;
    m_ready = 1'b1; #1;
    if (s_ready !== snap_r) bwd = bwd + 1;

    // --- reset must not leave a beat behind ------------------------
    s_valid = 1'b1; s_data = 24'hDEAD01; m_ready = 1'b0;
    repeat (4) @(posedge clk);
    rst = 1; @(posedge clk); #1;
    if (m_valid !== 1'b0) stale = stale + 1;
    rst = 0; m_ready = 1'b1; s_valid = 1'b0;
    @(posedge clk); #1;
    if (m_valid !== 1'b0) stale = stale + 1;

    $display("RESULT fwd=%%0d bwd=%%0d stale=%%0d", fwd, bwd, stale);
    $finish;
  end
endmodule
"""

ok = []


def check(name, passed, detail=""):
    ok.append(bool(passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def run(tmp, tag, tb_src, subs):
    """Build the DUT plus a testbench and report the RESULT line."""
    v = os.path.join(tmp, f"{tag}.v")
    with open(v, "w") as f:
        f.write(skid(W)["verilog"] + "\n")
        f.write(tb_src % subs)
    exe = os.path.join(tmp, tag)
    r = subprocess.run(["iverilog", "-g2012", "-o", exe, v],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return None
    r = subprocess.run([exe], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return dict(kv.split("=") for kv in line.split()[1:])
    print(r.stdout)
    return None


def main():
    if subprocess.run(["which", "iverilog"], capture_output=True).returncode:
        print("iverilog not found -- skipping")
        return 0

    print("skid: an elastic register slice\n")
    with tempfile.TemporaryDirectory() as tmp:
        base = {"W": W, "SEED_S": 0xACE1, "SEED_M": 0x1234}

        r = run(tmp, "reg", TB_REG, base)
        check("no input moves any output without a clock edge",
              r and r["fwd"] == "0" and r["bwd"] == "0",
              f"forward violations {r['fwd']}, backward {r['bwd']} -- "
              "this is the entire point of the module"
              if r else "BUILD FAILED")
        check("reset leaves no beat behind",
              r and r["stale"] == "0",
              f"{r['stale']} stale beats after reset" if r else "")

        N = 400
        r = run(tmp, "rand", TB_STREAM,
                dict(base, N=N, SRC_FULL=0, SNK_FULL=0))
        check("every beat, once, in order, under stall from both ends",
              r and r["got"] == str(N) and r["errs"] == "0",
              f"{r['got']}/{N} beats, {r['errs']} out of order or lost"
              if r else "BUILD FAILED")

        # The claim the reserve slot buys. A single-flop register slice
        # scores about 0.5 here, so this is the check that fails if
        # anyone decides the skid register looks redundant.
        r = run(tmp, "rate", TB_STREAM,
                dict(base, N=N, SRC_FULL=1, SNK_FULL=1))
        rate = int(r["got"]) / int(r["cycles"]) if r else 0.0
        check("full rate: a beat every cycle, not every other one",
              r and r["errs"] == "0" and rate > 0.97,
              f"{rate:.3f} beats/clock over {r['cycles']} cycles -- a "
              "single-flop slice manages about 0.5" if r else "")

        check("one beat of reserve is enough: never refused with a free slot",
              r and r["refused"] == "0",
              f"{r['refused']} refusals while the reserve was empty"
              if r else "")

        # Backpressure is where a slice loses beats, so lean on it: a
        # source that always offers against a sink that mostly does not.
        r = run(tmp, "squeeze", TB_STREAM,
                dict(base, N=N, SRC_FULL=1, SNK_FULL=0))
        check("lossless when the source always offers and the sink stalls",
              r and r["got"] == str(N) and r["errs"] == "0",
              f"{r['got']}/{N} beats, {r['errs']} bad" if r else "")

    print("\n" + ("SKID PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
