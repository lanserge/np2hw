"""active_width driven from the control register file (control_top).

A dynamic-resolution core (framing="eof", max_width=N) exposes an `active_width`
input. control_top() folds it into the AXI-Lite / UMI register file as a normal
register and wires param_active_width -> core.active_width. So software sets the
line length over the control bus, exactly like a gain register:

    write  active_width = 1280   (AXI-Lite)   ->   core processes 1280-wide lines

frame_sync=False: the write takes effect immediately (initial bring-up).
frame_sync=True : the write lands in a shadow and goes live at the next frame
                  boundary (out_last) -- glitch-free runtime resolution change,
                  the same shadow->live path the coefficient registers use.

Here we write active_width over AXI-Lite, stream a frame of that width (height
framed by in_sof/in_eof), and check against NumPy.

Run:  python examples/active_width.py   (needs iverilog)
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from np2hw.ir import Image2D
from np2hw.frontend import to_ir
from np2hw.verilog import generate, control_top

MAXW = 64
BUILD = os.path.join(os.path.dirname(__file__), "..", "build", "awreg_ex")
os.makedirs(BUILD, exist_ok=True)


def gauss(img):
    x = np.pad(img.astype(np.uint16), 1, mode="edge")
    return ((    x[:-2, :-2] + 2*x[:-2, 1:-1] +   x[:-2, 2:]
             + 2*x[1:-1, :-2] + 4*x[1:-1, 1:-1] + 2*x[1:-1, 2:]
             +   x[2:, :-2] + 2*x[2:, 1:-1] +   x[2:, 2:]) // 16).astype(np.uint8)


def main():
    s, o = to_ir(gauss, Image2D("img", 8, 4, 8))
    core = generate(o, "gcore", framing="eof", max_width=MAXW)
    top = control_top(core, ctrl="axil", frame_sync=False)   # immediate config
    off = top["offsets"]["active_width"]
    open(os.path.join(BUILD, "g.v"), "w").write(
        core["verilog"] + "\n" + top["verilog"] + "\n")
    ok_all = True
    for (w, h) in [(10, 12), (20, 6), (33, 5)]:
        a = np.random.default_rng(w + h).integers(0, 256, (h, w), dtype=np.uint8)
        vb = 3 * (w + 2) + 8
        tb = f"""
`timescale 1ns/1ps
module tb; reg clk=0,rst=1,iv=0,isof=0,ieof=0; wire ir; reg[7:0] din;
 wire ov; reg orr=1; wire os,oe,ol; wire[7:0] od; reg[7:0] img[0:{w*h-1}];
 reg[7:0] awa; reg awv; wire awr; reg[31:0] wd; reg[3:0] wstr; reg wv; wire wrdy;
 wire[1:0] br; wire bv; reg brdy; reg[7:0] ara; reg arv; wire arr2;
 wire[31:0] rd; wire[1:0] rr; wire rv; reg rrdy;
 integer r,c,k,fo; always #5 clk=~clk;
 {top['module']} #(.MAX_WIDTH({MAXW}),.HEIGHT(4)) dut(.clk(clk),.rst(rst),
   .s_axil_awaddr(awa),.s_axil_awvalid(awv),.s_axil_awready(awr),
   .s_axil_wdata(wd),.s_axil_wstrb(wstr),.s_axil_wvalid(wv),.s_axil_wready(wrdy),
   .s_axil_bresp(br),.s_axil_bvalid(bv),.s_axil_bready(brdy),
   .s_axil_araddr(ara),.s_axil_arvalid(arv),.s_axil_arready(arr2),
   .s_axil_rdata(rd),.s_axil_rresp(rr),.s_axil_rvalid(rv),.s_axil_rready(rrdy),
   .in_valid(iv),.in_ready(ir),.in_sof(isof),.in_eof(ieof),.in_data(din),
   .out_valid(ov),.out_ready(orr),.out_sof(os),.out_eol(oe),.out_last(ol),.out_data(od));
 always @(posedge clk) if(ov&&orr) $fdisplay(fo,"%0d",od);
 task axil_w(input [7:0] ad, input [31:0] d); begin
   @(negedge clk); awa=ad; awv=1; wd=d; wstr=4'hf; wv=1; brdy=1;
   @(posedge clk); while(!(awr&&wrdy)) @(posedge clk);
   @(negedge clk); awv=0; wv=0; while(!bv) @(posedge clk); @(negedge clk); brdy=0;
 end endtask
 initial begin $readmemh("in.hex",img); fo=$fopen("out.txt","w");
   awv=0; wv=0; brdy=0; arv=0; rrdy=0;
   @(negedge clk); @(negedge clk); rst=0; @(negedge clk);
   axil_w({off}, {w});                     // set active_width over AXI-Lite
   @(negedge clk);
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
        print(f"  AXI-Lite active_width={w:>2} -> {w}x{h}: {got.size} px -> "
              + ("PASS" if ok else "FAIL"))
    print("active_width from regfile -> " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
