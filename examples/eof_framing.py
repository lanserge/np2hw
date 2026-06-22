"""Height-agnostic framing: generate(..., framing="eof").

A reprogrammable sensor changes resolution at runtime. The EOF-framed core does
NOT count to HEIGHT -- instead an `in_eof` input (the sensor's frame-end / VSYNC,
pulsed on the last input pixel) triggers the bottom flush, and output EOF
(out_last) comes from the flush draining the line buffers. So ONE generated/
synthesized core processes ANY number of lines; only WIDTH is fixed (it sizes the
line buffers). We prove it by running two different frame heights through the
SAME core (generated with HEIGHT=4) and checking each against NumPy.

Run:  python examples/eof_framing.py   (needs iverilog)
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from np2hw.ir import Image2D
from np2hw.frontend import to_ir
from np2hw.verilog import generate

W = 8
BUILD = os.path.join(os.path.dirname(__file__), "..", "build", "eof_ex")
os.makedirs(BUILD, exist_ok=True)


def gauss(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def main():
    s, o = to_ir(gauss, Image2D("img", W, 4, 8))          # default HEIGHT=4
    core = generate(o, "gcore", framing="eof")
    open(os.path.join(BUILD, "g.v"), "w").write(core["verilog"] + "\n")
    ok_all = True
    for hact in (12, 6, 20):                               # any height, one core
        a = np.random.default_rng(hact).integers(0, 256, (hact, W), dtype=np.uint8)
        vb = 3 * (W + 2) + 8
        tb = f"""
`timescale 1ns/1ps
module tb; reg clk=0,rst=1,iv=0,isof=0,ieof=0; wire ir; reg[7:0] din;
 wire ov; reg orr=1; wire os,oe,ol; wire[7:0] od; reg[7:0] img[0:{W*hact-1}];
 integer r,c,k,fo; always #5 clk=~clk;
 gcore #(.WIDTH({W}),.HEIGHT(4)) dut(.clk(clk),.rst(rst),.in_valid(iv),.in_ready(ir),
   .in_sof(isof),.in_eof(ieof),.in_data(din),.out_valid(ov),.out_ready(orr),
   .out_sof(os),.out_eol(oe),.out_last(ol),.out_data(od));
 always @(posedge clk) if(ov&&orr) $fdisplay(fo,"%0d",od);
 initial begin $readmemh("in.hex",img); fo=$fopen("out.txt","w");
   @(negedge clk); rst=0;
   for(r=0;r<{hact};r=r+1) begin
     for(c=0;c<{W};c=c+1) begin
       din=img[r*{W}+c]; iv=1; isof=(r==0&&c==0); ieof=(r=={hact-1}&&c=={W-1});
       @(negedge clk); end
     iv=0; isof=0; ieof=0; for(k=0;k<3;k=k+1) @(negedge clk);
   end
   iv=0; for(k=0;k<{vb};k=k+1) @(negedge clk); $fclose(fo); $finish;
 end endmodule
"""
        open(os.path.join(BUILD, "tb.v"), "w").write(tb)
        open(os.path.join(BUILD, "in.hex"), "w").write(
            "\n".join(f"{p:02x}" for p in a.flatten()) + "\n")
        subprocess.run(["iverilog", "-o", "sim.vvp", "g.v", "tb.v"],
                       cwd=BUILD, check=True, capture_output=True)
        subprocess.run(["vvp", "sim.vvp"], cwd=BUILD, check=True, capture_output=True)
        got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)
        exp = gauss(a).astype(np.int64).flatten()
        ok = got.size == exp.size and np.array_equal(got, exp)
        ok_all &= ok
        print(f"  height={hact:>2} (core built for 4): {got.size} px -> "
              + ("PASS" if ok else "FAIL"))
    print("EOF framing, one core / many heights -> " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
