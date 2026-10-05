# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""A stream with its framing flags, across clocks that owe each other
nothing.

`stream_cdc` is `cdc_fifo` with a stream's ports on each face. The FIFO's
discipline is proven in examples/cdc_fifo.py; what is left to show is the
packing: every sample arrives once, in order, with its own sof, eol and
last -- not a neighbour's, not shifted -- under random back-pressure, with
the writer faster and with the reader faster.

Run:  python examples/stream_cdc.py   (needs iverilog)
"""
import os
import random
import subprocess
import sys
import tempfile

from np2hw.hdl import closure
from np2hw.stream import stream_cdc

BITS, BEATS = 10, 600

TB = """
`timescale 1ns/1ps
module tb;
  reg wclk = 0, rclk = 0, wrst = 1, rrst = 1;
  always #%(WHALF)s wclk = ~wclk;
  always #%(RHALF)s rclk = ~rclk;
  reg [%(W)d-1:0] mem [0:%(N)d-1];
  initial $readmemh("in.hex", mem);
  reg [15:0] wi = 0;
  wire in_ready, out_valid, out_sof, out_eol, out_last;
  wire [%(B)d-1:0] out_data;
  reg out_ready = 0;
  wire [%(W)d-1:0] word = mem[wi];
  // valid is a register, drawn afresh each write cycle and held until the
  // beat is taken: a continuous assignment of $random is evaluated only
  // when its operands change, and would hold one draw forever
  reg want = 0;
  wire in_valid = (wi < %(N)d) && want;
  always @(posedge wclk) if (!want || in_ready) want <= ($random %% 3 != 0);
  np2hw_stream_cdc #(.DATA_BITS(%(B)d), .A(3)) dut (
    .wclk(wclk), .wrst(wrst), .in_valid(in_valid), .in_ready(in_ready),
    .in_data(word[%(B)d-1:0]), .in_sof(word[%(B)d]),
    .in_eol(word[%(B)d+1]), .in_last(word[%(B)d+2]),
    .rclk(rclk), .rrst(rrst), .out_valid(out_valid),
    .out_ready(out_ready), .out_data(out_data), .out_sof(out_sof),
    .out_eol(out_eol), .out_last(out_last));
  always @(posedge wclk) if (!wrst && in_valid && in_ready) wi <= wi + 1;
  integer fd, got = 0;
  always @(posedge rclk) begin
    out_ready <= ($random %% 4 != 0);
    if (!rrst && out_valid && out_ready) begin
      $fwrite(fd, "%%0d %%0d %%0d %%0d\\n", out_data, out_sof, out_eol, out_last);
      got = got + 1;
    end
  end
  initial begin
    fd = $fopen("out.txt", "w");
    #100 wrst = 0; rrst = 0;
    wait (got == %(N)d);
    #50 $fclose(fd); $finish;
  end
  initial begin #2000000 $display("TIMEOUT"); $finish; end
endmodule
"""


def run(whalf, rhalf, beats):
    width = BITS + 3
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "in.hex"), "w") as fh:
            fh.write("\n".join(f"{(d | s << BITS | e << BITS + 1 | l << BITS + 2):x}"
                               for d, s, e, l in beats) + "\n")
        v = os.path.join(tmp, "all.v")
        with open(v, "w") as fh:
            fh.write(closure(stream_cdc(BITS)) + TB % {
                "WHALF": whalf, "RHALF": rhalf, "W": width, "B": BITS,
                "N": len(beats)})
        subprocess.run(["iverilog", "-g2012", "-o", "sim", "all.v"],
                       cwd=tmp, check=True, capture_output=True)
        out = subprocess.run(["vvp", "sim"], cwd=tmp, capture_output=True,
                             text=True)
        if "TIMEOUT" in out.stdout:
            return None
        with open(os.path.join(tmp, "out.txt")) as fh:
            return [tuple(int(x) for x in line.split()) for line in fh]


def main():
    if subprocess.run(["which", "iverilog"], capture_output=True).returncode:
        print("iverilog not found -- skipping")
        return 0
    rng = random.Random(7)
    beats = [(rng.randrange(1 << BITS), int(rng.random() < .1),
              int(rng.random() < .2), int(rng.random() < .05))
             for _ in range(BEATS)]
    ok = True
    for label, whalf, rhalf in (("writer faster", "3.1", "5.3"),
                                ("reader faster", "6.7", "2.9")):
        got = run(whalf, rhalf, beats)
        passed = got == beats
        ok &= passed
        print(f"  {label:<14} {BEATS} beats, flags with their samples  "
              f"{'PASS' if passed else 'FAIL'}")
    print("\n" + ("STREAM CDC PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
