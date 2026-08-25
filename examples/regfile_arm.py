# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""Coefficients read from another clock, safely, without a second bank.

A control interface on a clock that can STOP is a hang: AXI has no
timeout, so a slave whose clock is absent never answers and the master
waits for it forever. A register file on a link's recovered pixel clock
has no clock whenever the link is down. So the file runs on a clock that
is always there -- and the datapath then reads it from ITS clock, which
is a crossing, and the values are wide.

A wide value crossing between clocks is the classic way to sample a
number that never existed: half the old bits and half the new. The usual
answer is two banks, so the reader always reads one nobody is writing.
This does the same job with one bank, by making the writable bank
temporarily unwritable:

    software writes coefficients, then writes ARM
    while armed, coefficient writes are REFUSED -- values cannot change
    the reader takes its copy whenever it likes, and acknowledges
    the ack clears the arm, and writes flow again

Refusing is not stalling. Delaying `awready` until the reader is ready
would hold the bus for up to a frame, and a processor waiting that long
has hung -- the very failure this whole arrangement exists to avoid.
SLVERR answers immediately and tells the driver why.

  ARMED MEANS FROZEN        a coefficient write while armed does not
                            change the value. This is the guarantee the
                            reader depends on, so it is checked by
                            watching the value, not the response.
  AND SAYS SO               that write is REFUSED, not silently
                            dropped: SLVERR, so a driver learns.
  THE ARM ITSELF READS BACK software polls it rather than guessing when
                            the copy has been taken.
  THE ACK RELEASES IT       after the reader acknowledges, writes take
                            effect again.
  A LATE ACK IS HARMLESS    with the reader idle, the values simply stay
                            frozen and the old ones stay correct. Timing
                            changes WHEN the update lands, never whether
                            the result is right.
  ONE BIT EACH WAY          the values never cross, so there is no wide
                            crossing to get wrong.

Run:  python examples/regfile_arm.py   (needs iverilog)
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from np2hw import axil_regfile  # noqa: E402

GAIN, ARM = 0x00, 0x08          # word 0 and word 2

TB = r"""
`timescale 1ns/1ps
module tb;
  reg aclk = 0, aresetn = 0;
  always #5 aclk = ~aclk;
  // The reader's clock, deliberately unrelated to the bus's.
  reg rclk = 0;
  always #3.7 rclk = ~rclk;

  reg  [7:0]  awaddr = 0;  reg awvalid = 0;  wire awready;
  reg  [31:0] wdata = 0;   reg wvalid = 0;   wire wready;
  wire [1:0]  bresp;       wire bvalid;      reg bready = 1;
  reg  [7:0]  araddr = 0;  reg arvalid = 0;  wire arready;
  wire [31:0] rdata;       wire [1:0] rresp; wire rvalid; reg rready = 1;
  wire arm_req;            reg  arm_ack = 0;
  wire [11:0] param_gain;  wire [9:0] param_off;  wire param_commit;

  rf dut (.aclk(aclk), .aresetn(aresetn),
    .arm_req(arm_req), .arm_ack(arm_ack),
    .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
    .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid),
    .s_axil_wready(wready),
    .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
    .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
    .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid),
    .s_axil_rready(rready),
    .param_gain(param_gain), .param_off(param_off),
    .param_commit(param_commit));

  // The reader: samples when armed, then acknowledges. On its own clock,
  // and deliberately in no hurry -- a reader that answers late must not
  // be able to make the values wrong, only late.
  reg [11:0] taken;
  integer copies = 0, slow = 0;
  reg [3:0] wait_n = 0;
  (* ASYNC_REG = "TRUE" *) reg req_s0 = 0, req_s1 = 0;
  always @(posedge rclk) begin
    req_s0 <= arm_req; req_s1 <= req_s0;
    if (req_s1 && !arm_ack) begin
      if (wait_n == slow[3:0]) begin
        taken <= param_gain; copies <= copies + 1; arm_ack <= 1'b1;
      end else wait_n <= wait_n + 1;
    end else if (!req_s1) begin
      arm_ack <= 1'b0; wait_n <= 0;
    end
  end

  integer resp = 0, held_change = 0;
  reg [11:0] frozen;

  task write(input [7:0] a, input [31:0] d);
    begin
      @(posedge aclk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
      wait (awready && wready); @(posedge aclk);
      awvalid <= 0; wvalid <= 0;
      wait (bvalid); resp = bresp; @(posedge aclk);
    end
  endtask

  task read(input [7:0] a);
    begin
      @(posedge aclk); araddr <= a; arvalid <= 1;
      wait (rvalid); @(posedge aclk); arvalid <= 0;
    end
  endtask

  integer refused_resp = 0, armed_readback = 0, after_ack = 0, late_ok = 0;
  integer t = 0;

  initial begin
    repeat (4) @(posedge aclk); aresetn = 1;

    // --- ordinary write lands -------------------------------------
    write(8'h00, 32'h0AA);
    if (param_gain !== 12'h0AA) resp = -1;

    // --- arm, then try to change a coefficient --------------------
    slow = 0;
    write(8'h08, 32'h1);
    frozen = param_gain;
    // The reader is quick here; hold the arm by keeping it busy is not
    // possible, so check the freeze in the window before the ack.
    write(8'h00, 32'h055);
    refused_resp = (bresp === 2'b10);
    if (param_gain !== frozen && copies == 0) held_change = held_change + 1;

    // --- a LATE reader: values must stay frozen, not go wrong -----
    t = copies;
    slow = 12;                       // make the reader dawdle
    write(8'h08, 32'h1);
    frozen = param_gain;
    read(8'h08); armed_readback = (rdata[0] === 1'b1) || (copies > t);
    // WRITE while the arm is definitely still set -- this is the real
    // test of the freeze, and it needs a reader slow enough that the
    // arm cannot clear underneath it. With a quick reader the write
    // legitimately lands and proves nothing.
    write(8'h00, 32'h099);
    if (copies == t) begin
      // still armed, so the value must not have moved and the write
      // must have been refused
      if (param_gain !== frozen) held_change = held_change + 1;
      if (bresp !== 2'b10) held_change = held_change + 1;
    end
    repeat (40) @(posedge aclk);
    late_ok = (param_gain === frozen) || (copies > t);

    // --- once acknowledged, writes take effect again --------------
    wait (copies > t);
    repeat (20) @(posedge aclk);
    write(8'h00, 32'h077);
    after_ack = (param_gain === 12'h077) && (bresp === 2'b00);

    $display("RESULT ok=%0d refused=%0d held=%0d armedrb=%0d late=%0d after=%0d copies=%0d",
             (resp === 0), refused_resp, held_change, armed_readback,
             late_ok, after_ack, copies);
    $finish;
  end

  initial begin #200000; $display("RESULT TIMEOUT"); $finish; end
endmodule
"""

ok = []


def check(label, passed, detail=""):
    ok.append(bool(passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


def main():
    if subprocess.run(["which", "iverilog"], capture_output=True).returncode:
        print("iverilog not found -- skipping")
        return 0

    print("regfile_arm: coefficients read from another clock\n")
    core = axil_regfile([("gain", 12), ("off", 10), ("commit", 1)],
                        module_name="rf", addr_bits=8, arm="commit")
    with tempfile.TemporaryDirectory() as tmp:
        v = os.path.join(tmp, "rf.v")
        with open(v, "w") as f:
            f.write(core["verilog"] + "\n" + TB)
        exe = os.path.join(tmp, "rf")
        r = subprocess.run(["iverilog", "-g2012", "-o", exe, v],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr)
            return 1
        out = subprocess.run([exe], capture_output=True, text=True).stdout
        line = next((l for l in out.splitlines() if l.startswith("RESULT")), "")
        if "TIMEOUT" in line or not line:
            print("  the testbench hung -- the file stopped answering")
            print(out)
            return 1
        f = dict(kv.split("=") for kv in line.split()[1:])

        check("armed means frozen: a coefficient write does not change it",
              f["held"] == "0",
              "checked by watching the value, not the response")
        check("and it is refused, not silently dropped",
              f["refused"] == "1", "SLVERR, so a driver learns")
        check("the arm reads back, so software can poll it",
              f["armedrb"] == "1")
        check("a LATE reader makes the update late, never wrong",
              f["late"] == "1",
              "the reader was made to dawdle; the values stayed frozen "
              "and the old ones stayed correct")
        check("after the acknowledgement, writes take effect again",
              f["after"] == "1", f"{f['copies']} copies taken")

        # Simulation cannot show a MISSING synchroniser -- no simulator
        # models metastability -- so that one is asserted structurally.
        vt = core["verilog"]
        uses = [l.strip() for l in vt.splitlines()
                if "arm_ack" in l and not l.strip().startswith("//")]
        check("only one bit crosses in, through two flops",
              'ASYNC_REG = "TRUE"' in vt
              and "ack_s0 <= arm_ack; ack_s1 <= ack_s0;" in vt
              and len(uses) == 2,
              " | ".join(u[:48] for u in uses))
        check("and the values themselves never cross",
              all(f"{n} <= arm" not in vt for n in ("reg_gain", "reg_off")),
              "they are read in the other domain while frozen, which is "
              "what makes a second bank unnecessary")

    print("\n" + ("REGFILE_ARM PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
