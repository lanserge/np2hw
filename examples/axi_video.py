"""AXI4-Stream Video conformance: the generated IP is driven by a realistic
video stream -- TUSER (Start Of Frame) on each frame's first pixel, HBLANK gaps
between rows, VBLANK between frames, TREADY honored. The slave consumes TUSER to
anchor frame boundaries (it does NOT just count pixels). Multi-frame output is
validated against the same NumPy model.

Run:  python examples/axi_video.py   (needs iverilog)
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from np2hw.ir import Image2D
from np2hw.frontend import to_ir
from np2hw.verilog import generate, axis_video_wrap


def gaussian(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)

W, H = 8, 6
BUILD = os.path.join(os.path.dirname(__file__), "..", "build", "axi")
os.makedirs(BUILD, exist_ok=True)
FRAMES = [np.random.default_rng(k).integers(0, 256, (H, W), dtype=np.uint8)
          for k in (1, 2, 3)]


def main():
    s, out = to_ir(gaussian, Image2D("img", W, H, 8))
    core = generate(out, "gcore")
    wrap = axis_video_wrap(core, W, H, module_name="gaxis")
    open(os.path.join(BUILD, "gcore.v"), "w").write(core["verilog"] + "\n")
    open(os.path.join(BUILD, "gaxis.v"), "w").write(wrap["verilog"] + "\n")
    nf, vb = len(FRAMES), (H + 4) * (W + 2)
    tb = f"""
`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1; reg tv=0,tu=0; reg[7:0] td; wire tr;
 wire mv; reg mr=1; wire[15:0] md; wire mu,ml;
 reg[7:0] img[0:{nf*H*W-1}]; integer fr,r,c,k,fo;
 always #5 clk=~clk;
 gaxis #(.WIDTH({W}),.HEIGHT({H})) dut(.aclk(clk),.aresetn(!rst),
   .s_axis_tvalid(tv),.s_axis_tready(tr),.s_axis_tdata(td),.s_axis_tuser(tu),.s_axis_tlast(1'b0),
   .m_axis_tvalid(mv),.m_axis_tready(mr),.m_axis_tdata(md),.m_axis_tuser(mu),.m_axis_tlast(ml));
 always @(posedge clk) if(mv&&mr) $fdisplay(fo,"%0d",md);
 task feed(input integer base); begin
   for(r=0;r<{H};r=r+1) begin
     for(c=0;c<{W};c=c+1) begin
       td=img[base+r*{W}+c]; tv=1; tu=(r==0&&c==0);
       @(negedge clk); while(!tr) @(negedge clk); tu=0;
     end
     tv=0; for(k=0;k<6;k=k+1) @(negedge clk);          // HBLANK
   end
   tv=0; for(k=0;k<{vb};k=k+1) @(negedge clk);         // VBLANK (covers flush)
 end endtask
 initial begin
   $readmemh("in.hex",img); fo=$fopen("out.txt","w");
   @(negedge clk); rst=0; @(negedge clk);
   for(fr=0;fr<{nf};fr=fr+1) feed(fr*{H*W});
   $fclose(fo); $finish;
 end
endmodule
"""
    open(os.path.join(BUILD, "tb.v"), "w").write(tb)
    allpix = np.concatenate([f.flatten() for f in FRAMES])
    open(os.path.join(BUILD, "in.hex"), "w").write(
        "\n".join(f"{p:02x}" for p in allpix) + "\n")
    subprocess.run(["iverilog", "-o", "sim.vvp", "gcore.v", "gaxis.v", "tb.v"],
                   cwd=BUILD, check=True, capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], cwd=BUILD, check=True, capture_output=True)
    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64)
    exp = np.concatenate([gaussian(f).astype(np.int64).flatten() for f in FRAMES])
    ok = got.size == exp.size and np.array_equal(got, exp)
    print(f"AXI4-Stream Video, {len(FRAMES)} frames with TUSER/SOF + blanking -> "
          + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
