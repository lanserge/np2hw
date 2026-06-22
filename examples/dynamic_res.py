"""Full runtime-variable resolution: generate(..., framing="eof", max_width=N).

For a reprogrammable sensor that changes resolution live, ONE synthesized core
handles any frame size up to a maximum, with no rebuild:

  * width  -> `active_width` register (where each row ends: wrap / EOL / edge).
              Line buffers are sized to the MAX_WIDTH parameter.
  * height -> `in_eof` (sensor VSYNC, on the last input pixel) drives the bottom
              flush; output EOF falls out of the flush. No HEIGHT counter.

We drive several different (width, height) frames through the SAME core
(MAX_WIDTH=64) and check each against NumPy.

Run:  python examples/dynamic_res.py   (needs iverilog)
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from np2hw.ir import Image2D
from np2hw.frontend import to_ir
from np2hw.verilog import generate

MAXW = 64
BUILD = os.path.join(os.path.dirname(__file__), "..", "build", "dynres_ex")
os.makedirs(BUILD, exist_ok=True)


def gauss(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def main():
    s, o = to_ir(gauss, Image2D("img", 8, 4, 8))          # traced size irrelevant now
    core = generate(o, "gcore", framing="eof", max_width=MAXW)
    open(os.path.join(BUILD, "g.v"), "w").write(core["verilog"] + "\n")
    ok_all = True
    for (w, h) in [(10, 12), (20, 6), (7, 9), (64, 3)]:   # any size <= MAX_WIDTH
        a = np.random.default_rng(w * 100 + h).integers(0, 256, (h, w), dtype=np.uint8)
        vb = 3 * (w + 2) + 8
        tb = f"""
`timescale 1ns/1ps
module tb; reg clk=0,rst=1,iv=0,isof=0,ieof=0; reg[31:0] aw; wire ir; reg[7:0] din;
 wire ov; reg orr=1; wire os,oe,ol; wire[7:0] od; reg[7:0] img[0:{w*h-1}];
 integer r,c,k,fo; always #5 clk=~clk;
 gcore #(.MAX_WIDTH({MAXW}),.HEIGHT(4)) dut(.clk(clk),.rst(rst),.in_valid(iv),.in_ready(ir),
   .in_sof(isof),.in_eof(ieof),.active_width(aw),.in_data(din),
   .out_valid(ov),.out_ready(orr),.out_sof(os),.out_eol(oe),.out_last(ol),.out_data(od));
 always @(posedge clk) if(ov&&orr) $fdisplay(fo,"%0d",od);
 initial begin $readmemh("in.hex",img); fo=$fopen("out.txt","w"); aw={w};
   @(negedge clk); rst=0;
   for(r=0;r<{h};r=r+1) begin
     for(c=0;c<{w};c=c+1) begin
       din=img[r*{w}+c]; iv=1; isof=(r==0&&c==0); ieof=(r=={h-1}&&c=={w-1});
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
        print(f"  {w:>2}x{h:<2} (core: MAX_WIDTH={MAXW}): {got.size} px -> "
              + ("PASS" if ok else "FAIL"))
    print("dynamic resolution, one core / many sizes -> " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
