# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The DDC slave: a register file reachable over the display cable.

The HDMI link already carries an I2C bus (DDC, the EDID wires). A
receiver that answers on it turns the one cable into the control
plane: address 0x50 says who the sink is (EDID -- the handshake that
makes a source believe a display exists at all), address 0x37 reads
and writes the ISP's registers through an AXI-lite master, with the
register file's refusal contract translated onto the bus: SLVERR
becomes NACK.

  WRITES LAND            two words in one transaction, auto-increment,
                         big-endian bytes, both acked.
  READS COME BACK        the same words, over a repeated start, with
                         the prefetch crossing the word boundary.
  REFUSAL TRANSLATES     SLVERR from the fabric is a NACK on the wire,
                         and NOTHING lands -- the armed-file contract,
                         seen from the cable.
  EDID IS THE HANDSHAKE  address 0x50 serves the ROM, offset write
                         then sequential reads, exactly the transaction
                         every source performs on hotplug.
  STRANGERS ARE IGNORED  a third address gets no ack and no state.
  OPEN-DRAIN, STRUCTURAL the only SDA output is sda_pull; no code path
                         can drive the line high, checked in the text
                         because a simulator with two push-pull drivers
                         happily shows a working bus that would be a
                         short on a PCB.

Run:  python examples/ddc_slave.py   (needs iverilog)
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from np2hw.i2c import ddc_slave  # noqa: E402


def minimal_edid():
    """A checksum-valid 128-byte EDID with a recognisable byte 8."""
    e = bytearray(128)
    e[0:8] = bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])
    e[8:10] = bytes([0x09, 0xE5])          # a manufacturer id
    e[18] = 1                               # EDID 1.x
    e[19] = 4
    e[127] = (-sum(e[:127])) % 256
    return bytes(e)


TB = r"""
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;      // 100 MHz board clock
  reg rst = 1;

  // The bus, open-drain: the master's release is a 1, and the line is
  // low iff SOMEBODY pulls. The slave never touches SCL.
  reg m_scl = 1, m_sda = 1;
  wire dut_pull;
  wire scl = m_scl;
  wire sda = m_sda & ~dut_pull;

  // AXI-lite behavioural register file, with a refusal switch: this
  // is the armed file's SLVERR, reduced to one bit of testbench.
  reg refuse = 0;
  reg [31:0] regs [0:63];
  wire [15:0] awaddr, araddr;
  wire awvalid, wvalid, arvalid;
  wire [31:0] wdata;  wire [3:0] wstrb;
  reg awr = 0, wr = 0, arr = 0;
  reg bvalid = 0; reg [1:0] bresp = 0;
  reg rvalid = 0; reg [1:0] rresp = 0; reg [31:0] rdata = 0;
  wire bready, rready;
  reg aw_seen = 0, w_seen = 0; reg [15:0] aw_q; reg [31:0] w_q;
  always @(posedge clk) begin
    awr <= awvalid && !aw_seen; wr <= wvalid && !w_seen; arr <= arvalid;
    if (awvalid && awr) begin aw_q <= awaddr; aw_seen <= 1; awr <= 0; end
    if (wvalid && wr)   begin w_q <= wdata;  w_seen <= 1; wr <= 0; end
    if (aw_seen && w_seen && !bvalid) begin
      if (!refuse) regs[aw_q[7:2]] <= w_q;
      bresp <= refuse ? 2'b10 : 2'b00;
      bvalid <= 1; aw_seen <= 0; w_seen <= 0;
    end
    if (bvalid && bready) bvalid <= 0;
    if (arvalid && arr) begin
      rdata <= regs[araddr[7:2]]; rresp <= 2'b00; rvalid <= 1; arr <= 0;
    end
    if (rvalid && rready) rvalid <= 0;
  end

  ddc_slave dut (.clk(clk), .rst(rst),
    .scl_i(scl), .sda_i(sda), .sda_pull(dut_pull),
    .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awr),
    .m_wdata(wdata), .m_wvalid(wvalid), .m_wready(wr), .m_wstrb(wstrb),
    .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready),
    .m_araddr(araddr), .m_arvalid(arvalid), .m_arready(arr),
    .m_rdata(rdata), .m_rresp(rresp), .m_rvalid(rvalid),
    .m_rready(rready));

  // ---- a bit-banged master, 400 kHz. Every phase gets real time:
  // a task chain with zero-width phases makes edges the slave's
  // glitch filter rightly swallows -- on a real bus those phases
  // are microseconds, and a testbench that skips them tests a bus
  // that cannot exist.
  integer HALF = 1250, Q = 312;
  task IST; begin
    #Q; m_sda = 1; #Q; m_scl = 1; #625; m_sda = 0; #625; m_scl = 0;
  end endtask
  task ISP; begin
    #Q; m_sda = 0; #Q; m_scl = 1; #625; m_sda = 1; #625;
  end endtask
  task IBIT(input b); begin
    #Q; m_sda = b; #Q; m_scl = 1; #HALF; m_scl = 0;
  end endtask
  task ISEND(input [7:0] b, output ack); integer i; begin
    for (i = 7; i >= 0; i = i - 1) IBIT(b[i]);
    #Q; m_sda = 1; #Q; m_scl = 1; #625; ack = ~sda; #625; m_scl = 0;
  end endtask
  task IRECV(input nack, output [7:0] b); integer i; begin
    #Q; m_sda = 1; #Q;
    for (i = 7; i >= 0; i = i - 1) begin
      m_scl = 1; #625; b[i] = sda; #625; m_scl = 0; #HALF;
    end
    m_sda = nack ? 1'b1 : 1'b0; #Q; m_scl = 1; #HALF; m_scl = 0;
    #Q; m_sda = 1;
  end endtask

  reg a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA;
  reg [7:0] r0, r1, r2, r3, r4, r5, r6, r7;
  integer errs = 0;
  task chk(input cond, input [127:0] tag);
    if (!cond) begin errs = errs + 1; $display("FAIL %0s", tag); end
  endtask

  initial begin
    regs[4] = 32'h0; regs[5] = 32'h0; regs[9] = 32'h55AA1234;
    #100 rst = 0; #100;

    // WRITE two words at 0x0010, one transaction
    IST; ISEND(8'h6E, a0);                    // 0x37 << 1 | W
    ISEND(8'h00, a1); ISEND(8'h10, a2);
    ISEND(8'hDE, a3); ISEND(8'hAD, a4); ISEND(8'hBE, a5); ISEND(8'hEF, a6);
    ISEND(8'hCA, a7); ISEND(8'hFE, a8); ISEND(8'h12, a9); ISEND(8'h34, aA);
    ISP;
    chk(a0 & a1 & a2 & a3 & a4 & a5 & a6, "write acks word0");
    chk(a7 & a8 & a9 & aA, "write acks word1");
    chk(regs[4] === 32'hDEADBEEF, "word0 landed");
    chk(regs[5] === 32'hCAFE1234, "word1 landed (auto-inc)");

    // READ them back through a repeated start, prefetch across words
    IST; ISEND(8'h6E, a0); ISEND(8'h00, a1); ISEND(8'h10, a2);
    IST; ISEND(8'h6F, a3);                    // 0x37 << 1 | R
    IRECV(0, r0); IRECV(0, r1); IRECV(0, r2); IRECV(0, r3);
    IRECV(0, r4); IRECV(0, r5); IRECV(0, r6); IRECV(1, r7);
    ISP;
    chk(a0 & a1 & a2 & a3, "read setup acks");
    chk({r0,r1,r2,r3} === 32'hDEADBEEF, "readback word0");
    chk({r4,r5,r6,r7} === 32'hCAFE1234, "readback word1");

    // REFUSAL: SLVERR from the fabric is a NACK on the wire
    refuse = 1;
    IST; ISEND(8'h6E, a0); ISEND(8'h00, a1); ISEND(8'h24, a2);
    ISEND(8'h11, a3); ISEND(8'h22, a4); ISEND(8'h33, a5); ISEND(8'h44, a6);
    ISP;
    refuse = 0;
    chk(a3 & a4 & a5, "bytes before the bus answers are acked");
    chk(!a6, "the word's last byte is NACKed on SLVERR");
    chk(regs[9] === 32'h55AA1234, "nothing landed");

    // EDID: offset write, sequential read -- the hotplug transaction
    IST; ISEND(8'hA0, a0); ISEND(8'h08, a1);
    IST; ISEND(8'hA1, a2);
    IRECV(0, r0); IRECV(1, r1);
    ISP;
    chk(a0 & a1 & a2, "edid acks");
    chk(r0 === 8'h09 && r1 === 8'hE5, "edid bytes at offset 8");
    IST; ISEND(8'hA0, a0); ISEND(8'h00, a1);
    IST; ISEND(8'hA1, a2);
    IRECV(0, r0); IRECV(0, r1); IRECV(1, r2);
    ISP;
    chk(r0 === 8'h00 && r1 === 8'hFF && r2 === 8'hFF, "edid header");

    // A STRANGER: no ack, and the engine holds no grudge
    IST; ISEND(8'h44, a0); ISP;
    chk(!a0, "unknown address is not acked");
    IST; ISEND(8'h6E, a0); ISEND(8'h00, a1); ISEND(8'h10, a2);
    IST; ISEND(8'h6F, a3); IRECV(1, r0); ISP;
    chk(a0 && r0 === 8'hDE, "and the next transaction is normal");

    $display("RESULT errs=%0d", errs);
    $finish;
  end
  initial begin #80_000_000; $display("RESULT TIMEOUT"); $finish; end
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

    print("ddc_slave: the register file, reachable over the display "
          "cable\n")
    core = ddc_slave(0x37, edid=minimal_edid(), edid_addr7=0x50)
    v = core["verilog"]

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "t.v")
        with open(src, "w") as f:
            f.write(v + "\n" + TB)
        exe = os.path.join(tmp, "t")
        r = subprocess.run(["iverilog", "-g2012", "-o", exe, src],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr)
            check("compiles", False)
        else:
            out = subprocess.run([exe], capture_output=True,
                                 text=True).stdout
            for line in out.splitlines():
                if line.startswith("FAIL"):
                    print("         " + line)
            res = next((l for l in out.splitlines()
                        if l.startswith("RESULT")), "RESULT none")
            check("writes, reads, refusal, EDID and strangers on one bus",
                  res == "RESULT errs=0", res)

    lines = [l.strip() for l in v.splitlines()]
    check("open-drain by construction: sda_pull is the only SDA output",
          [p["name"] for p in core["interface"]["ports"]
           if p["direction"] != "input" and "sda" in p["name"]]
          == ["sda_pull"]
          and not any("sda_o" in l or "assign sda " in l for l in lines),
          "a push-pull driver simulates fine and shorts on a PCB")
    check("the pins cross two flops and a majority before any logic",
          "scl_h <= {scl_h[1:0], scl_i};" in v
          and "sda_h <= {sda_h[1:0], sda_i};" in v
          and "(scl_h[2] & scl_h[1]) | (scl_h[1] & scl_h[0])" in v)
    check("SLVERR is consulted, not assumed",
          "m_bresp == 2'b00" in v and "wr_wait" in v)

    # The EDID validator refuses what a bench cannot debug.
    try:
        bad = bytearray(minimal_edid())
        bad[40] ^= 0x5A
        ddc_slave(0x37, edid=bytes(bad))
        check("a bad EDID checksum is refused at generation time", False)
    except ValueError:
        check("a bad EDID checksum is refused at generation time", True)

    print("\n" + ("DDC_SLAVE PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
