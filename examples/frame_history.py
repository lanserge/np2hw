# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The frame history: each pixel delivered with the same pixel from the frames
before it.

`frame_history()` is the storage behind a model's time axis: channel k of each
output word is frame n - k, channel 0 the live frame. This proves the claims a
model downstream relies on without being able to check them itself.

  THE RIGHT PAST        every past channel of every pixel is exactly the
                        sample that position held k frames earlier -- for
                        one past frame, three, and four at 16 bits.
  THE LIVE FRAME        channel 0 is the input, untouched.
  THE RIGHT TAG         each frame's tag channel carries the tag of the loop
                        position that frame was captured at, from a table in
                        no particular order -- for a loop as long as the
                        window, longer (8 positions through 4 frames) and
                        shorter (2 through 4); live_tag reports the live
                        frame's.
  FRAMED                out_sof on each frame's first pixel, out_eol on each
                        line's last, out_last on each frame's last, and on
                        no other beat.
  UNDER BACKPRESSURE    gaps on the input and stalls on the output change
                        nothing -- and the stalls are counted, so a run in
                        which none happened cannot pass for one that
                        survived them.
  FULL RATE             with no gaps and no stalls, one pixel per clock.

Frames before the first `frames` are not checked on their past channels: the
memory holds whatever reset left there, and saying so is the core's contract.

Run:  python examples/frame_history.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _harness import BUILD
from np2hw import frame_history

TB = """
`timescale 1ns/1ps
module tb;
  localparam N = %(N)d, B = %(B)d, W = %(W)d, H = %(H)d, F = %(F)d;
  localparam P = W * H, M = %(LOOP)d, PB = %(PB)d, OFF = %(OFF)d;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg in_valid = 0, in_sof = 0, out_ready = 0;
  reg [B-1:0] in_data = 0;
  wire in_ready, out_valid, out_sof, out_eol, out_last;
  wire [2*(N+1)*B-1:0] out_data;
  wire [B-1:0] live_tag;
  reg  [PB-1:0] pos = 0;                  // the live frame's loop position
  reg  [B-1:0] tags [0:M-1];              // the tag table, in no order

  frame_history dut (.clk(clk), .rst(rst),
    .in_valid(in_valid), .in_ready(in_ready), .in_sof(in_sof),
    .in_data(in_data), .out_valid(out_valid), .out_ready(out_ready),
    .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
    .out_data(out_data), .live_tag(live_tag)%(PORTS)s);

  function integer at(input integer f);   // loop position of frame f
    at = (f + OFF) %% M;
  endfunction

  // Distinct per frame and per position, so a sample from the wrong frame
  // or the wrong pixel cannot pass for the right one.
  function [B-1:0] val(input integer f, input integer p);
    val = f * 53 + p * 7 + 11;
  endfunction

  integer seed = %(SEED)d;
  integer sf = 0, sp = 0;                       // next input beat
  integer rf = 0, rp = 0;                       // next expected output beat
  integer errs = 0, sofs = 0, eols = 0, lasts = 0, stalls = 0, checked = 0;
  integer tagchecks = 0, tagerrs = 0;
  integer cycles = 0, first_out = -1, last_out = -1, k;

  // Sample on the rising edge: every handshake is the value before the edge.
  always @(posedge clk) if (!rst) begin
    cycles = cycles + 1;
    if (out_valid && !out_ready) stalls = stalls + 1;
    if (out_valid && out_ready) begin
      if (first_out < 0) first_out = cycles;
      last_out = cycles;
      if (out_data[B-1:0] !== val(rf, rp)) errs = errs + 1;
      for (k = 1; k <= N; k = k + 1)
        if (rf >= k) begin
          checked = checked + 1;
          if (out_data[k*B +: B] !== val(rf - k, rp)) errs = errs + 1;
        end
      for (k = 0; k <= N; k = k + 1) begin
        tagchecks = tagchecks + 1;
        if (out_data[(N+1+k)*B +: B] !== tags[(at(rf) - k + M * N) %% M])
          tagerrs = tagerrs + 1;
      end
      if (out_sof  !== (rp == 0))         errs = errs + 1;
      if (out_eol  !== ((rp %% W) == W-1)) errs = errs + 1;
      if (out_last !== (rp == P-1))       errs = errs + 1;
      sofs = sofs + out_sof; eols = eols + out_eol; lasts = lasts + out_last;
      rp = rp + 1; if (rp == P) begin rp = 0; rf = rf + 1; end
    end
    if (in_valid && in_ready) begin
      if (live_tag !== tags[at(sf)]) tagerrs = tagerrs + 1;
      sp = sp + 1; if (sp == P) begin sp = 0; sf = sf + 1; end
    end
  end

  // Drive on the falling edge, half a cycle clear of the sampling.
  always @(negedge clk) if (!rst) begin
    // $random is SIGNED, so its remainder can be negative: take it
    // unsigned, or a gap of 0 still drops half the cycles.
    if (sf < F && ($unsigned($random(seed)) %% 100) >= %(GAP)d) begin
      in_valid <= 1; in_data <= val(sf, sp); in_sof <= (sp == 0);
      pos <= at(sf);
    end else begin
      in_valid <= 0; in_sof <= 0;
    end
    out_ready <= ($unsigned($random(seed)) %% 100) >= %(STALL)d;
  end

  integer t;
  initial begin
    for (t = 0; t < M; t = t + 1) tags[t] = (t * 97 + 29) ^ (t << 3);
    repeat (4) @(posedge clk);
    @(negedge clk) rst = 0;
    while (rf < F && cycles < %(TIMEOUT)d) @(posedge clk);
    $fwrite($fopen("%(OUT)s", "w"),
            "%%s %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d\\n",
            (rf < F) ? "TIMEOUT" : "DONE", errs, sofs, eols, lasts, stalls,
            checked, first_out, last_out, tagchecks, tagerrs);
    $finish;
  end
endmodule
"""


def run(name, frames, bits, width, height, n_frames, gap=0, stall=0,
        seed=1, timeout=200000, loop=None, offset=3):
    core = frame_history(frames, bits, width, height, loop=loop)
    loop = core["loop"]
    pb = max(1, (loop - 1).bit_length())
    ports = "".join(f", .param_pos_{i}(pos[{i}])" for i in range(pb))
    ports += "".join(f", .param_tag_{j}(tags[{j}])" for j in range(loop))
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    out = os.path.join(BUILD, f"{name}.txt")
    with open(os.path.join(BUILD, f"tb_{name}.v"), "w") as fh:
        fh.write(TB % {"N": frames, "B": bits, "W": width, "H": height,
                       "F": n_frames, "GAP": gap, "STALL": stall,
                       "LOOP": loop, "PB": pb, "OFF": offset % loop,
                       "PORTS": ports,
                       "SEED": seed, "TIMEOUT": timeout,
                       "OUT": out.replace("\\", "/")})
    subprocess.run(["iverilog", "-g2012", "-o", f"{name}.vvp",
                    f"{name}.v", f"tb_{name}.v"],
                   check=True, cwd=BUILD, capture_output=True)
    subprocess.run([os.path.join(BUILD, f"{name}.vvp")], check=True,
                   cwd=BUILD, capture_output=True)
    with open(out) as fh:
        line = fh.readline().split()
    if line[0] == "TIMEOUT":
        return None
    return dict(zip(("errs", "sofs", "eols", "lasts", "stalls", "checked",
                     "first_out", "last_out", "tagchecks", "tagerrs"),
                    (int(v) for v in line[1:])))


def main():
    ok = []

    def check(label, cond, detail=""):
        print(f"  {label:<56} {'PASS' if cond else 'FAIL'}  {detail}")
        ok.append(bool(cond))

    print("frame_history:")

    # Odd geometry, so an off-by-one in the position or the wrap shows.
    for name, n, bits in (("fh_1", 1, 8), ("fh_3", 3, 12), ("fh_4", 4, 16)):
        w, h, f = 7, 5, n + 5
        r = run(name, n, bits, w, h, f, gap=25, stall=30, seed=n)
        past = sum(max(0, f - k) for k in range(1, n + 1)) * w * h
        check(f"{n} past frame{'s' if n > 1 else ''}, {bits}-bit: every channel "
              "of every pixel exact",
              r and r["errs"] == 0 and r["checked"] == past,
              f"errs={r and r['errs']} past checked={r and r['checked']}/{past}")
        check(f"{n} past: framed -- sof/frame, eol/line, last/frame",
              r and (r["sofs"], r["eols"], r["lasts"]) == (f, f * h, f),
              f"sof={r and r['sofs']} eol={r and r['eols']} last={r and r['lasts']}")
        check(f"{n} past: backpressure happened, and changed nothing",
              r and r["stalls"] > 0 and r["errs"] == 0,
              f"stalled cycles={r and r['stalls']}")

        check(f"{n} past: every tag channel and live_tag exact",
              r and r["tagerrs"] == 0 and r["tagchecks"] == f * w * h * (n + 1),
              f"tag checks={r and r['tagchecks']} errs={r and r['tagerrs']}")

    # The loop need not be the window: 8 positions through a 4-frame
    # window (a multi-frequency sequence), and 2 through 4 (alternating
    # exposures seen twice each).
    for name, lp in (("fh_loop8", 8), ("fh_loop2", 2)):
        r = run(name, 3, 12, 7, 5, 11, gap=20, stall=20, seed=lp, loop=lp)
        check(f"a loop of {lp} through a 4-frame window: pixels and tags exact",
              r and r["errs"] == 0 and r["tagerrs"] == 0 and r["tagchecks"] > 0,
              f"errs={r and r['errs']} tag errs={r and r['tagerrs']}")

    # One pixel per clock when nothing holds it back.
    w, h, f = 16, 6, 6
    r = run("fh_rate", 3, 16, w, h, f)
    beats = f * w * h
    span = r and r["last_out"] - r["first_out"] + 1
    check("no gaps, no stalls: one pixel per clock",
          r and r["errs"] == 0 and span == beats,
          f"{beats} beats in {span} cycles")

    # Refusals.
    for label, args in (("no past frames", (0, 8, 4, 4)),
                        ("a one-pixel frame", (2, 8, 1, 1)),
                        ("an empty loop", (2, 8, 4, 4, 0))):
        try:
            frame_history(*args)
            refused = False
        except ValueError:
            refused = True
        check(f"refuses {label}", refused)

    print("all checks passed" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
