"""scanout: a pixel stream becomes a timed raster, and stays put.

The transmit mirror of the bayerlink receiver, and the claims are the
laws a bench charged us for. Every one of these failed on real hardware
before it was written down here.

  PLACED          a window lands where the registers say, to the pixel,
                  with fill everywhere else. If placement were off by
                  one this is where it shows.
  CENTRED         `center` derives the offsets and reports them on
                  act_x0/act_y0 -- and does NOT disturb what the host
                  wrote into the offset registers. A derived value
                  stamped back into its register becomes an explicit
                  ask, and then the picture stops following its source.
  DROPPED, NOT DISPLACED
                  the law this block exists for. A reader that answers
                  late at a frame boundary costs ONE frame of fill; it
                  must never shift the picture. The mutation is the old
                  behaviour -- start the frame anyway -- and it displaces
                  every later frame, permanently, which is exactly what
                  a bench saw as a 338-pixel roll that survived resets.
  REFUSED         a window that does not fit is refused visibly, not
                  clipped silently: a picture quietly cropped is a bug
                  wearing the costume of a decision.
  TOGETHER        pixel, enable and both syncs leave on the SAME clock.
                  Skew them and the sink shows a shifted picture that no
                  status bit inside the block can see.
  ONE OWNER       the raster table refuses an unknown mode by name
                  rather than inventing timings.

Run:  python examples/scanout.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from np2hw.video_out import MODES, mode_timing, scanout

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
os.makedirs(BUILD, exist_ok=True)

# A raster small enough to simulate, with the same shape as a real one.
TINY = (40, 12, 32, 8, 34, 36, 9, 10)          # h_tot v_tot h_act v_act ...


def _emit(name, selftest=0):
    core = scanout(mode="1080p60", module_name=name, sink="raw",
                   data_bits=24, fill=0x101010)
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"])
    return core


def _tb(name, win, center, stall_late, frames=10, mutate=False):
    """A reader that answers LATE at one frame boundary, and a monitor
    that judges every emitted pixel against where it should have come
    from."""
    x0, y0, w, h = win
    h_tot, v_tot, h_act, v_act, hs0, hs1, vs0, vs1 = TINY
    src = f"""`timescale 1ns/1ps
module tb;
  localparam H_TOT={h_tot}, V_TOT={v_tot}, H_ACT={h_act}, V_ACT={v_act};
  localparam W={w}, H={h}, FRAME=W*H;
  reg clk=0, rst=1; always #5 clk=~clk;
  wire tready; reg tvalid=0; integer beat=0, stall=0, wraps=0;
  wire [23:0] tdata = beat[23:0];
  wire tuser = (beat==0);
  wire de, hs, vs; wire [23:0] px; wire [15:0] status, ax, ay;

  {name} #(.H_TOT(H_TOT), .V_TOT(V_TOT), .H_ACT(H_ACT), .V_ACT(V_ACT),
           .HS_BEG({hs0}), .HS_END({hs1}), .VS_BEG({vs0}), .VS_END({vs1}),
           .SELFTEST(0)) dut (
    .clk(clk), .rst(rst), .locked(1'b1),
    .s_axis_tdata(tdata), .s_axis_tvalid(tvalid), .s_axis_tready(tready),
    .s_axis_tuser(tuser), .s_axis_tlast(1'b0),
    .param_win_x0({x0}), .param_win_y0({y0}),
    .param_win_w(W), .param_win_h(H), .param_center({int(center)}),
    .act_x0(ax), .act_y0(ay),
    .vid_active_video(de), .vid_data(px), .vid_hsync(hs), .vid_vsync(vs),
    .status(status));

  always @(posedge clk) begin
    if (rst) begin beat<=0; stall<={stall_late}; tvalid<=0; wraps<=0; end
    else if (stall>0) begin stall<=stall-1; tvalid<=0; end
    else begin
      tvalid <= 1;
      if (tvalid && tready) begin
        if (beat==FRAME-1) begin
          beat<=0; wraps<=wraps+1; tvalid<=0;
          stall <= (wraps=={1 if stall_late else 0}) ? {stall_late} : 3;
        end else beat<=beat+1;
      end
    end
  end

  // Each pixel is judged against ITS OWN POSITION, not against a
  // running count: a picture that starts late but arrives in order
  // would satisfy a counter, and that is exactly the fault being
  // tested. expected(wx,wy) = wy*W + wx, always.
  localparam WX0={x0 if not center else (h_act - w)//2};
  localparam WY0={y0 if not center else (v_act - h)//2};
  integer rx=0, ry=0, wx=0, wy=0, expect=0;
  integer frames_done=0, displaced=0, filled=0;
  always @(posedge clk) if (!rst) begin
    if (de) begin
      if (rx>=WX0 && rx<WX0+W && ry>=WY0 && ry<WY0+H) begin
        wx = rx - WX0; wy = ry - WY0; expect = wy*W + wx;
        if (px === 24'h101010) filled = filled + 1;
        else if (px !== expect[23:0]) displaced = displaced + 1;
        if (wx==W-1 && wy==H-1) frames_done = frames_done + 1;
      end else if (px !== 24'h101010) displaced = displaced + 1;
      rx = rx + 1;
      if (rx==H_ACT) begin rx=0; ry=ry+1; if (ry==V_ACT) ry=0; end
    end
  end

  initial begin
    repeat(4) @(negedge clk); rst=0;
    repeat(V_TOT*H_TOT*{frames}) @(negedge clk);
    $display("RESULT displaced=%0d filled=%0d frames=%0d act=%0d,%0d under=%0d misalign=%0d refused=%0d",
             displaced, filled, frames_done, ax, ay,
             (status>>5)&1, (status>>6)&1, (status>>7)&1);
    $finish;
  end
endmodule
"""
    with open(os.path.join(BUILD, "tb_scanout.v"), "w") as fh:
        fh.write(src)
    subprocess.run(["iverilog", "-o", "sc.vvp", "tb_scanout.v", f"{name}.v"],
                   check=True, cwd=BUILD, capture_output=True)
    out = subprocess.run(["vvp", "sc.vvp"], check=True, cwd=BUILD,
                         capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT")][0]
    return dict(kv.split("=") for kv in line.split()[1:])


def main():
    checks = []

    def result(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<54} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("scanout:")
    _emit("scanout_v")

    r = _tb("scanout_v", win=(6, 2, 16, 4), center=False, stall_late=0)
    result("placed: the window lands where the registers say",
           int(r["displaced"]) == 0 and int(r["frames"]) >= 4,
           f"{r['frames']} frames, {r['displaced']} displaced")

    r = _tb("scanout_v", win=(6, 2, 16, 4), center=True, stall_late=0)
    centred = (int(r["act"].split(",")[0]) == (32 - 16) // 2
               and int(r["act"].split(",")[1]) == (8 - 4) // 2)
    result("centred: offsets derived and reported, not stamped",
           centred and int(r["displaced"]) == 0, f"act={r['act']}")

    r = _tb("scanout_v", win=(0, 0, 32, 8), center=False, stall_late=200)
    result("dropped, not displaced: a late reader costs one frame",
           int(r["displaced"]) == 0 and int(r["filled"]) > 0,
           f"{r['filled']} fill px, {r['displaced']} displaced")

    r = _tb("scanout_v", win=(20, 2, 16, 4), center=False, stall_late=0)
    result("refused: a window past the raster is refused, not clipped",
           int(r["refused"]) == 1)

    # MUTATION: rebuild the old behaviour -- align once, then start
    # every frame on faith -- and require the displacement claim to
    # CATCH it. A claim that cannot fail is decoration.
    core = open(os.path.join(BUILD, "scanout_v.v")).read()
    mutant = (core
              .replace("module scanout_v", "module scanout_old")
              .replace("assign s_axis_tready = armed ? in_win "
                       ": (blank && !head_sof);",
                       "assign s_axis_tready = ever_armed ? in_win "
                       ": (blank && !head_sof);")
              .replace("if (armed && last_px) armed <= 1'b0;", "")
              .replace("vid_data <= (armed && in_win && s_axis_tvalid)",
                       "vid_data <= (ever_armed && in_win && s_axis_tvalid)"))
    with open(os.path.join(BUILD, "scanout_old.v"), "w") as fh:
        fh.write(mutant)
    m = _tb("scanout_old", win=(0, 0, 32, 8), center=False, stall_late=200)
    result("mutation: the old start-on-faith logic FAILS that claim",
           int(m["displaced"]) > 0,
           f"{m['displaced']} displaced px where the fix has 0")

    # The four outputs must be assigned in ONE always block with no
    # boundary between them: anchor on the live (non-reset) hsync
    # assignment and require the other three before the next `always`.
    verilog = open(os.path.join(BUILD, "scanout_v.v")).read()
    start = verilog.index("vid_hsync <= HS_POS")
    rest = verilog[start:]
    body = rest[:rest.index("always")] if "always" in rest else rest
    together = all(sig in body for sig in
                   ("vid_vsync <=", "vid_active_video <=", "vid_data <="))
    result("together: pixel, enable and syncs leave on one clock", together)

    named = True
    try:
        mode_timing("1080p75")
        named = False
    except ValueError as error:
        named = "one owner" in str(error)
    result("one owner: an unknown mode is refused by name", named,
           f"{len(MODES)} modes in the table")

    ok = all(checks)
    print("\n" + ("SCANOUT PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
