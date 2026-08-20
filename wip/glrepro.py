"""Prove scanout's v7 genlock: the clock is steered, the raster is still.

Two REAL clocks: the raster's (74.25-class, 13.468 ns) and the
producer's, offset by a runtime-variable ppm. The DUT's fine
phase-shift steps are modeled at 24 ps each: a retard step stretches
one raster half-cycle, an advance step stretches one producer
half-cycle (relative phase is what matters and it is exact).

Scenarios: worst-case mid-frame start (one jam allowed), +100 ppm,
a swing to -100 ppm, back to +50 ppm, and a 20-frame source outage
(one more jam allowed on return). The raster's frame length must
NEVER change -- that is the entire point.
"""
import subprocess
import sys

sys.path.insert(0, "/Users/serge/projects/np2hw/src")
from np2hw.video_out import scanout  # noqa: E402

core = scanout("sim_tiny", genlock=True, genlock_ps_per_pixel=560)

TB = r"""
`timescale 1ns/1fs  // 1ps rounds 100 ppm of 6.734 ns up to 148 ppm --
                    // past the stepper's authority; the instrument lied
module tb;
  // ---- the raster clock: 13.468 ns, retard steps stretch it ------
  real HP = 6.734;
  reg clk = 0;
  integer retp = 0;
  always begin
    if (retp > 0) begin retp = retp - 1; #(HP + 0.024); end
    else #(HP);
    clk = ~clk;
  end
  // ---- the producer clock: ppm off, advance steps stretch it -----
  real ppm = 100.0;
  reg pclk = 0;
  integer advp = 0;
  always begin
    if (advp > 0) begin advp = advp - 1; #(6.734 * (1.0 + ppm * 1e-6) + 0.024); end
    else #(6.734 * (1.0 + ppm * 1e-6));
    pclk = ~pclk;
  end
  reg rst = 1;

  // ---- PS model: count steps, pulse PSDONE 12 clocks later -------
  wire ps_en, ps_incdec;
  reg [11:0] done_pipe = 0;
  wire ps_done = done_pipe[11];
  integer ps_steps = 0, ret_steps = 0, adv_steps = 0;
  always @(posedge clk) begin
    done_pipe <= {done_pipe[10:0], ps_en};
    if (ps_en) begin
      ps_steps = ps_steps + 1;
      if (ps_incdec) begin retp = retp + 1; ret_steps = ret_steps + 1; end
      else           begin advp = advp + 1; adv_steps = adv_steps + 1; end
    end
  end
  // the estimator's real diet: sample AT the arrival, not at vsync
  integer arrs = 0;
  always @(posedge clk) if (!rst && dut.sof_arr) begin
    arrs = arrs + 1;
    if (arrs % 20 == 0 || arrs < 8)
      $display("ARR #%0d y=%0d x=%0d err_r=%0d late=%b owed_pre=%0d",
               arrs, dut.y, dut.x, dut.err_r, dut.late_r, dut.owed);
  end

  // ---- producer: its own raster on its own clock ------------------
  integer px = 0, py = 30;
  reg feeding = 1;                      // 0: source-gone window
  wire p_active = (px < 160) && (py < 64) && feeding;
  reg [23:0] pdat = 0;
  // SINGLE-WRITER pointers, occupancy DERIVED: a shared counter
  // updated blocking from two asynchronous processes loses one
  // update whenever both edges land in a timestep -- one lost beat
  // per frame, and the DUT starved on a fiction (the conservation
  // ledger convicted it: wr-win-drain-fn grew exactly +1/frame).
  reg [25:0] fifo [0:8191];
  reg [31:0] fw = 0, fr = 0;
  wire [31:0] fn = fw - fr;
  wire tvalid = (fw != fr);
  wire [25:0] head = fifo[fr[12:0]];
  wire tready;
  always @(posedge pclk) if (!rst) begin
    px <= (px == 199) ? 0 : px + 1;
    if (px == 199) py <= (py >= 79) ? 0 : py + 1;
    if (p_active && fn < 8192) begin
      fifo[fw[12:0]] <= {(py == 0 && px == 0), (px == 159), pdat};
      fw <= fw + 1;
      pdat <= pdat + 1;
    end
  end
  always @(posedge clk) if (!rst && tvalid && tready)
    fr <= fr + 1;
  reg povf = 0;
  always @(posedge pclk) if (p_active && fn >= 8192) povf <= 1;
  integer swr = 0, scon = 0, bwr = 0, bwin = 0, bdrain = 0;
  always @(posedge pclk) if (!rst && p_active && fn < 8192) bwr = bwr + 1;
  always @(posedge clk) if (!rst && tvalid && tready) begin
    if (dut.in_win) bwin = bwin + 1;
    else            bdrain = bdrain + 1;
  end
  always @(posedge pclk) if (!rst && p_active && fn < 8192 && py == 0 && px == 0) swr = swr + 1;
  always @(posedge clk)  if (!rst && tvalid && tready && head[25]) scon = scon + 1;

  wire vav; wire [23:0] vdat; wire vhs, vvs; wire [15:0] stat;
  wire [15:0] ax0, ay0;
  scanout dut (
    .clk(clk), .rst(rst), .locked(1'b1),
    .s_axis_tdata(head[23:0]), .s_axis_tvalid(tvalid),
    .s_axis_tready(tready), .s_axis_tuser(head[25]),
    .s_axis_tlast(head[24]),
    .param_win_x0(16'd0), .param_win_y0(16'd0),
    .param_win_w(16'd160), .param_win_h(16'd64),
    .param_center(1'b1),
    .param_follow(1'b1),
    .ps_en(ps_en), .ps_incdec(ps_incdec), .ps_done(ps_done),
    .act_x0(ax0), .act_y0(ay0),
    .vid_active_video(vav), .vid_data(vdat),
    .vid_hsync(vhs), .vid_vsync(vvs), .status(stat));

  // ---- the assertions ---------------------------------------------
  // 1. the raster NEVER changes length: measure every vsync period
  //    in clk cycles; after the first jam settles it must be exactly
  //    16000 except the (at most two) jam frames.
  integer frames = 0, cyc = 0;
  integer last_vs = 0, jams = 0, badper = 0;
  reg vvs_d = 0;
  reg locked_once = 0;
  always @(posedge clk) begin
    cyc = cyc + 1;
    vvs_d <= vvs;
    if (vvs && !vvs_d) begin
      frames = frames + 1;
      if (last_vs != 0 && (cyc - last_vs) != 16000) begin
        jams = jams + 1;
        if (locked_once && !(frames > 380 && frames < 415))
          badper = badper + 1;
        if (jams <= 5)
          $display("F%0d frame period %0d cycles (jam or drift)",
                   frames, cyc - last_vs);
      end
      last_vs = cyc;
      if (frames % 40 == 0)
        $display("F%0d beats wr=%0d win=%0d drain=%0d (wr-win-drain-fn=%0d) swr=%0d scon=%0d",
                 frames, bwr, bwin, bdrain, bwr - bwin - bdrain - fn,
                 swr, scon);
      // the ppm schedule and the outage
      if (frames == 150) ppm = -100.0;
      if (frames == 300) ppm = 50.0;
      if (frames == 380) feeding = 0;
      if (frames == 400) feeding = 1;
    end
    if (dut.glocked) locked_once <= 1;
  end
  // 2. no underflow after first lock, outside the outage window
  integer late_und = 0, und_first = -1, und_last = -1;
  always @(posedge clk)
    if (locked_once && dut.armed && dut.in_win && !tvalid
        && !(frames >= 379 && frames <= 416)) begin
      late_und = late_und + 1;
      if (und_first == -1) und_first = frames;
      und_last = frames;
      if (late_und <= 8 || (late_und % 100) == 0)
        $display("UND #%0d F%0d y=%0d x=%0d fn=%0d", late_und,
                 frames, dut.y, dut.x, fn);
    end

  initial begin
    repeat (30) @(posedge clk); rst = 0;
    while (frames < 470 && cyc < 12_000_000) @(posedge clk);
    $display("DONE frames=%0d glocked=%b jams=%0d badper=%0d late_und=%0d (F%0d..F%0d) steps=%0d povf=%b",
             frames, dut.glocked, jams, badper, late_und,
             und_first, und_last, ps_steps, povf);
    if (dut.glocked && !povf && badper == 0 && late_und == 0
        && jams <= 3)
      $display("VERDICT: PASS");
    else
      $display("VERDICT: FAIL");
    $finish;
  end
endmodule
"""

open("gl.v", "w").write(core["verilog"] + "\n" + TB)
r = subprocess.run(["iverilog", "-g2012", "-o", "gl", "gl.v"],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr)
    sys.exit(1)
out = subprocess.run(["./gl"], capture_output=True, text=True,
                     timeout=3600).stdout
print(out.strip())
