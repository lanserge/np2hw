"""The bayerlink receiver, proven against the protocol's own reference.

bayerlink (github.com/bayerlink/bayerlink) carries packed raw Bayer over a
display link. `np2hw.video_in.bayerlink_in()` emits the receiving end:
parallel video from an HDMI receiver in, np2hw's elastic 12-bit stream out.

The encoder here is not hand-rolled: containers come from the published
reference codec (`pip install bayerlink`), so this example is an INDEPENDENT
implementation meeting the reference across a simulated link -- the exact
situation the protocol's conformance story exists for. Four claims, each a
way the receiver could be wrong on a bench with a scope instead of here:

  BIT-EXACT      every sample and every framing flag out of the simulated
                 receiver equals the pattern that went into the encoder,
                 across two consecutive frames (re-anchoring included),
                 under downstream backpressure.
  LANES          a permuted byte-lane wiring, matched by `lane_map`, still
                 decodes exactly -- the permutation is configuration, not a
                 debugging season.
  RATE RULE      at the reference geometry's line budget (2028 samples in
                 2200 slots), the half-line FIFO never overflows.
  OVERFLOW       when downstream stalls beyond the FIFO's tolerance, the
                 sticky flag SETS. An error flag that has never been seen
                 to fire is decoration.

Run:  python examples/bayerlink_in.py   (needs iverilog and `pip install bayerlink`)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import BUILD

try:
    from bayerlink import encode_frame, pattern
except ImportError:
    print("bayerlink is not installed; this example decodes the reference "
          "codec's containers, so:  pip install bayerlink")
    sys.exit(0)

from np2hw.testing import Beat, check_framing
from np2hw.video_in import bayerlink_in

TB = """
`timescale 1ns/1ps
module tb;
  localparam DISPW = %(DISPW)d, LINES = %(LINES)d, BLANK = %(BLANK)d;
  reg clk = 0, rst = 1;
  reg vid_de = 0, vid_vsync = 0;
  reg [23:0] vid_data = 0;
  wire ov, out_valid, out_sof, out_eol, out_last;
  wire [11:0] out_data;
  reg out_ready = 0;
  integer stall = 0;

  %(MODULE)s dut (.clk(clk), .rst(rst),
    .vid_de(vid_de), .vid_vsync(vid_vsync), .vid_data(vid_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
    .overflow(ov));

  always #5 clk = ~clk;

  // One 24-bit word per display pixel, lane packing done by the host side.
  reg [23:0] pix [0:DISPW*LINES-1];
  integer x, y, f, cyc, fd;

  // Downstream backpressure: ready 4 cycles in 5, or a hard stall when the
  // overflow test asks for one.
  always @(posedge clk) begin
    cyc <= cyc + 1;
    out_ready <= (stall != 0 && cyc > 200 && cyc < 200 + %(STALL)d) ? 1'b0
                 : (cyc %% 5 != 3);
    if (out_valid && out_ready)
      $fwrite(fd, "%%0d %%0d %%0d %%0d\\n", out_data, out_sof, out_eol, out_last);
  end

  initial begin
    if ($value$plusargs("stall=%%d", stall)) ;
    $readmemh("in.hex", pix);
    fd = $fopen("out.txt", "w");
    cyc = 0;
    repeat (4) @(posedge clk);
    rst = 0;
    for (f = 0; f < %(FRAMES)d; f = f + 1) begin
      // vsync during blanking, then the frame's lines
      repeat (2) @(negedge clk);
      vid_vsync = 1; repeat (3) @(negedge clk); vid_vsync = 0;
      repeat (2) @(negedge clk);
      for (y = 0; y < LINES; y = y + 1) begin
        for (x = 0; x < DISPW; x = x + 1) begin
          @(negedge clk); vid_de = 1; vid_data = pix[y*DISPW + x];
        end
        @(negedge clk); vid_de = 0; vid_data = 0;
        repeat (BLANK) @(negedge clk);
      end
    end
    // drain
    repeat (DISPW*LINES*4 + 400) @(posedge clk);
    $fwrite(fd, "OVF %%0d\\n", ov);
    $fclose(fd);
    $finish;
  end
endmodule
"""


def pack_lanes(container: np.ndarray, lane_map) -> np.ndarray:
    """Container bytes -> 24-bit video words, byte k driven on lane lane_map[k].

    Lane 0 is vid_data[23:16], lane 1 [15:8], lane 2 [7:0]. This is the host
    side of the permutation the receiver's `lane_map` undoes.
    """
    height, width, _ = container.shape
    words = np.zeros((height, width), np.uint32)
    shift = {0: 16, 1: 8, 2: 0}
    for k in range(3):
        words |= container[:, :, k].astype(np.uint32) << shift[lane_map[k]]
    return words


def run(name, cam_w, cam_h, disp_w, blank, lane_map=(0, 1, 2), frames=2,
        stall=0, fifo_depth=None):
    """Build the receiver, drive encoded containers at it, return the log."""
    core = bayerlink_in(cam_w, cam_h, module_name=name, lane_map=lane_map,
                        fifo_depth=fifo_depth)
    raw = pattern.generate("corners", cam_w, cam_h)
    container = encode_frame(raw, "RGGB", frame_seq=1,
                             display=(disp_w, cam_h + 2))
    words = pack_lanes(container, lane_map)

    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(w):06x}" for w in words.ravel()) + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(TB % {"DISPW": disp_w, "LINES": cam_h + 2, "BLANK": blank,
                       "MODULE": name, "FRAMES": frames,
                       "STALL": stall or 1})

    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", f"{name}.v", "tb.v"],
                   check=True, cwd=BUILD, capture_output=True)
    plusargs = ["+stall=1"] if stall else []
    subprocess.run(["vvp", "sim.vvp"] + plusargs, check=True, cwd=BUILD,
                   capture_output=True)

    beats, overflow = [], None
    with open(os.path.join(BUILD, "out.txt")) as fh:
        for line in fh:
            if line.startswith("OVF"):
                overflow = int(line.split()[1])
            else:
                d, s, e, l = (int(v) for v in line.split())
                beats.append(Beat(d, bool(s), bool(e), bool(l)))
    return raw, beats, overflow


def main():
    checks = []

    def check(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<52} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("bayerlink_in:")

    # 1. Bit-exact, two frames, default lanes, backpressure.
    raw, beats, ov = run("blin_a", cam_w=64, cam_h=8, disp_w=64, blank=24)
    expected = raw.ravel()
    per_frame = expected.size
    ok = len(beats) == 2 * per_frame
    for f in range(2):
        got = beats[f * per_frame:(f + 1) * per_frame]
        ok = ok and all(b.data == int(v) for b, v in zip(got, expected))
        try:
            check_framing(got, 64, 8)
        except AssertionError:
            ok = False
    check("two frames bit-exact, framing exact, backpressured", ok,
          f"{len(beats)} beats")
    check("no overflow within the rate rule", ov == 0)

    # 2. Permuted lanes, matched by lane_map: still exact.
    raw, beats, ov = run("blin_b", cam_w=64, cam_h=8, disp_w=64, blank=24,
                         lane_map=(2, 0, 1), frames=1)
    ok = (len(beats) == raw.size
          and all(b.data == int(v) for b, v in zip(beats, raw.ravel()))
          and ov == 0)
    check("permuted byte lanes, matched by lane_map", ok)

    # 3. The reference line budget: 2028 samples in 2200 slots.
    raw, beats, ov = run("blin_c", cam_w=2028, cam_h=3, disp_w=1920,
                         blank=280 - 1, frames=1)   # 1920 active + 279 + de-gap
    ok = (len(beats) == raw.size
          and all(b.data == int(v) for b, v in zip(beats, raw.ravel()))
          and ov == 0)
    check("reference geometry (2028 in 2200 slots), no overflow", ok,
          f"{len(beats)} samples")

    # 4. The flag can actually fire: stall past the FIFO's tolerance.
    _, _, ov = run("blin_d", cam_w=64, cam_h=8, disp_w=64, blank=2,
                   frames=1, stall=600, fifo_depth=64)
    check("sticky overflow fires under a hard stall", ov == 1)

    ok = all(checks)
    print("\n" + ("BAYERLINK_IN PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
