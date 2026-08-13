"""The bayerlink v2 receiver, proven against the protocol's own reference.

bayerlink (github.com/bayerlink/bayerlink) carries packed raw Bayer over a
display link. `np2hw.video_in.bayerlink_in()` emits the receiving end:
parallel video from an HDMI receiver in, np2hw's elastic 16-bit stream out,
with each frame's header owning depth and geometry AT RUN TIME.

The encoder here is not hand-rolled: containers come from the published
reference codec (`pip install bayerlink`), so this example is an INDEPENDENT
implementation meeting the reference across a simulated link. Seven claims,
each a way the receiver could be wrong on a bench with a scope instead:

  EVERY DEPTH     8, 10, 12, 14 and 16-bit frames -- one of each, in ONE
                  run of ONE build, the header switching the unpack engine
                  frame by frame -- all bit-exact with exact framing.
  GEOMETRY        frames of different width and height in one run: the
                  header owns geometry, the build owns only capacity.
  LANES           a permuted byte-lane wiring, matched by `lane_map`,
                  still decodes exactly.
  RATE RULE       the reference line budget (2028 samples in 2200 slots)
                  at 12 bits: no overflow.
  REFUSAL         an unknown fourcc and a line beyond capacity each refuse
                  THEIR frame -- zero beats -- latch the sticky code, and
                  the next good frame decodes exactly.
  OVERFLOW        when downstream stalls beyond the FIFOs' tolerance, the
                  sticky flag SETS. A flag never seen to fire is decoration.
  OBSERVABLE      hdr_* report the last accepted header's facts.

Run:  python examples/bayerlink_in.py   (needs iverilog and `pip install bayerlink`)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import BUILD

try:
    from bayerlink import encode_frame
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
  wire ov, refd, out_valid, out_sof, out_eol, out_last, hdrv;
  wire [2:0] rcode;
  wire [15:0] out_data, hdrw, hdrh;
  wire [4:0] hdrb;
  wire [1:0] hdrp;
  reg out_ready = 0;
  integer stall = 0;

  %(MODULE)s dut (.clk(clk), .rst(rst),
    .vid_de(vid_de), .vid_vsync(vid_vsync), .vid_data(vid_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
    .overflow(ov), .refused(refd), .refuse_code(rcode),
    .hdr_bits(hdrb), .hdr_width(hdrw), .hdr_height(hdrh),
    .hdr_phase(hdrp), .hdr_valid(hdrv));

  always #5 clk = ~clk;

  // One 24-bit word per display pixel, lane packing done by the host side.
  reg [23:0] pix [0:DISPW*LINES*%(FRAMES)d-1];
  integer x, y, f, cyc, fd;

  // Downstream backpressure: ready 4 cycles in 5, a hard stall when the
  // overflow test asks for one, or BURSTY reads (+burst=<off cycles>):
  // full-rate reading alternating with long dead gaps, the shape a DMA
  // packet reader on an ARM actually has.
  integer burst = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    out_ready <= (burst != 0) ? ((cyc %% (400 + burst)) < 400)
                 : (stall != 0 && cyc > 200 && cyc < 200 + %(STALL)d) ? 1'b0
                 : (cyc %% 5 != 3);
    if (out_valid && out_ready)
      $fwrite(fd, "%%0d %%0d %%0d %%0d\\n", out_data, out_sof, out_eol, out_last);
  end

  initial begin
    if ($value$plusargs("stall=%%d", stall)) ;
    if ($value$plusargs("burst=%%d", burst)) ;
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
          @(negedge clk); vid_de = 1; vid_data = pix[(f*LINES + y)*DISPW + x];
        end
        @(negedge clk); vid_de = 0; vid_data = 0;
        repeat (BLANK) @(negedge clk);
      end
      // let a frame's tail drain before the next header arrives
      repeat (%(GAP)d) @(negedge clk);
    end
    // drain
    repeat (DISPW*LINES*4 + 800) @(posedge clk);
    $fwrite(fd, "OVF %%0d\\n", ov);
    $fwrite(fd, "REF %%0d %%0d\\n", refd, rcode);
    $fwrite(fd, "HDR %%0d %%0d %%0d %%0d %%0d\\n", hdrb, hdrw, hdrh, hdrp, hdrv);
    $fclose(fd);
    $finish;
  end
endmodule
"""

ORDER_OF = {0: "RGGB", 1: "GRBG", 2: "GBRG", 3: "BGGR"}


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


def rand_frame(rng, bits, w, h):
    return rng.integers(0, 1 << bits, size=(h, w), dtype=np.uint16)


def run(name, frames, disp_w, disp_h, blank, lane_map=(0, 1, 2),
        max_line_bytes=4096, fifo_depth=1024, stall=0, gap=200, burst=0):
    """Build once, drive a LIST of frames (spec dicts or raw containers).

    Each frame spec: {"bits", "raw", "order"} -> encoded via the reference
    codec; or {"container": ndarray} pre-built (the refusal tests corrupt
    good containers, so even the refused bytes come from the reference).
    """
    core = bayerlink_in(max_line_bytes=max_line_bytes, module_name=name,
                        fifo_depth=fifo_depth, lane_map=lane_map)
    stream = []
    for i, spec in enumerate(frames):
        if "container" in spec:
            container = spec["container"]
        else:
            container = encode_frame(
                spec["raw"], spec.get("order", "RGGB"), frame_seq=i + 1,
                display=(disp_w, disp_h), bits=spec["bits"])
        stream.append(pack_lanes(container, lane_map))
    words = np.concatenate([s.ravel() for s in stream])

    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(w):06x}" for w in words) + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(TB % {"DISPW": disp_w, "LINES": disp_h, "BLANK": blank,
                       "MODULE": name, "FRAMES": len(frames),
                       "STALL": stall or 1, "GAP": gap})

    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", f"{name}.v", "tb.v"],
                   check=True, cwd=BUILD, capture_output=True)
    plusargs = ["+stall=1"] if stall else []
    if burst:
        plusargs = [f"+burst={burst}"]
    subprocess.run(["vvp", "sim.vvp"] + plusargs, check=True, cwd=BUILD,
                   capture_output=True)

    beats, overflow, refused, hdr = [], None, None, None
    with open(os.path.join(BUILD, "out.txt")) as fh:
        for line in fh:
            if line.startswith("OVF"):
                overflow = int(line.split()[1])
            elif line.startswith("REF"):
                refused = tuple(int(v) for v in line.split()[1:])
            elif line.startswith("HDR"):
                hdr = tuple(int(v) for v in line.split()[1:])
            else:
                d, s, e, l = (int(v) for v in line.split())
                beats.append(Beat(d, bool(s), bool(e), bool(l)))
    return beats, overflow, refused, hdr


def exact(beats, raws):
    """Beats == the concatenation of raw frames, each with exact framing."""
    at = 0
    for raw in raws:
        h, w = raw.shape
        got = beats[at:at + raw.size]
        if len(got) != raw.size:
            return False
        if not all(b.data == int(v) for b, v in zip(got, raw.ravel())):
            return False
        try:
            check_framing(got, w, h)
        except AssertionError:
            return False
        at += raw.size
    return at == len(beats)


def main():
    rng = np.random.default_rng(20260813)
    checks = []

    def check(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<56} {'PASS' if ok else 'FAIL'}"
              + (f"  {detail}" if detail else ""))

    print("bayerlink_in v2:")

    # 1. EVERY DEPTH, one build, one run: the header switches the engine.
    depths = (8, 10, 12, 14, 16)
    raws = [rand_frame(rng, b, 56, 6) for b in depths]
    frames = [{"bits": b, "raw": r} for b, r in zip(depths, raws)]
    beats, ov, refd, hdr = run("blv2_a", frames, disp_w=64, disp_h=10,
                               blank=24)
    check("five depths in one run, header-switched, bit-exact",
          exact(beats, raws), f"{len(beats)} beats")
    check("no overflow across the depth sweep", ov == 0)
    check("nothing refused across the depth sweep", refd == (0, 0))
    check("hdr_* report the last header (16-bit, 56x6)",
          hdr == (16, 56, 6, 0, 1), f"hdr={hdr}")

    # 2. GEOMETRY is the header's, per frame.
    raws = [rand_frame(rng, 12, 56, 6), rand_frame(rng, 12, 30, 8),
            rand_frame(rng, 12, 8, 3)]
    frames = [{"bits": 12, "raw": r} for r in raws]
    beats, ov, refd, _ = run("blv2_b", frames, disp_w=64, disp_h=10,
                             blank=24)
    check("three geometries in one run, header-owned",
          exact(beats, raws) and ov == 0 and refd == (0, 0))

    # 3. LANES: permuted wiring, matched by lane_map.
    raws = [rand_frame(rng, 10, 56, 6)]
    beats, ov, refd, _ = run("blv2_c", [{"bits": 10, "raw": raws[0]}],
                             disp_w=64, disp_h=10, blank=24,
                             lane_map=(2, 0, 1))
    check("permuted byte lanes, matched by lane_map",
          exact(beats, raws) and ov == 0)

    # 4. RATE RULE: the reference budget, 2028 samples in 2200 slots.
    raws = [rand_frame(rng, 12, 2028, 3)]
    beats, ov, refd, _ = run("blv2_d", [{"bits": 12, "raw": raws[0]}],
                             disp_w=1920, disp_h=5, blank=280 - 1)
    check("reference geometry (2028 in 2200 slots), no overflow",
          exact(beats, raws) and ov == 0, f"{len(beats)} samples")

    # 5. REFUSAL: unknown fourcc, then capacity; good frames still decode.
    good = rand_frame(rng, 12, 56, 6)
    bad = encode_frame(rand_frame(rng, 12, 56, 6), "RGGB", frame_seq=9,
                       display=(64, 10), bits=12).copy()
    # fourcc lives at header bytes 8..11: byte k is [0, k // 3, k % 3]
    for k, ch in zip(range(8, 12), b"XXXX"):
        bad[0, k // 3, k % 3] = ch
    beats, ov, refd, hdr = run(
        "blv2_e", [{"container": bad}, {"bits": 12, "raw": good}],
        disp_w=64, disp_h=10, blank=24)
    check("unknown fourcc: zero beats, sticky code 2, next frame OK",
          exact(beats, [good]) and refd == (1, 2), f"refd={refd}")

    wide = rand_frame(rng, 12, 200, 2)          # 300 bytes > 256 capacity
    beats, ov, refd, _ = run(
        "blv2_f", [{"bits": 12, "raw": wide}, {"bits": 12, "raw": good}],
        disp_w=128, disp_h=10, blank=24, max_line_bytes=256)
    check("line beyond capacity: refused code 4, next frame OK",
          exact(beats, [good]) and refd == (1, 4), f"refd={refd}")

    # 6. WRAP: a small sample FIFO wrapped many times by the one engine
    # whose batch stride (3) does not divide the depth -- the slot
    # addressing must wrap WITH the pointer, not run off the banks.
    raws = [rand_frame(rng, 8, 56, 40)]
    beats, ov, refd, _ = run("blv2_w", [{"bits": 8, "raw": raws[0]}],
                             disp_w=64, disp_h=44, blank=8, fifo_depth=256)
    check("8-bit through ~9 FIFO wraps, stride 3 vs depth 256",
          exact(beats, raws) and ov == 0)

    # 7. LOSS RECOVERY: a hard stall mangles ONE frame's data, but the
    # framing comes from ingest tags, so the damage cannot leak -- the
    # NEXT frame decodes exact, sof/eol/last included. (The bench found
    # this: drain-counted framing turned one lossy gap into every
    # subsequent DMA packet misframed.)
    from np2hw.testing import check_framing as _cf
    f1 = rand_frame(rng, 12, 56, 24)
    f2 = rand_frame(rng, 12, 56, 24)
    beats, ov, refd, _ = run(
        "blv2_r", [{"bits": 12, "raw": f1}, {"bits": 12, "raw": f2}],
        disp_w=64, disp_h=26, blank=2, stall=1500,
        max_line_bytes=128, fifo_depth=16)
    tail = beats[-f2.size:] if len(beats) >= f2.size else []
    ok = ov == 1 and len(tail) == f2.size and all(
        b.data == int(v) for b, v in zip(tail, f2.ravel()))
    if ok:
        try:
            _cf(tail, 56, 24)
        except AssertionError:
            ok = False
    check("hard-stall loss contained; NEXT frame exact, framed", ok,
          f"ov={ov} beats={len(beats)}")

    # 8. BURSTY READER: full-rate reads alternating with long dead gaps
    # -- the shape an ARM-side DMA packet loop actually has. Repeated
    # loss episodes must never wedge the stream: every delivered line
    # keeps its eol, and frame `last`s keep arriving to the end. (The
    # bench found the wedge: an orphan stub under the frame barrier.)
    raws = [rand_frame(rng, 10, 28, 6) for _ in range(4)]
    beats, ov, refd, _ = run("blv2_y", [{"bits": 10, "raw": r} for r in raws],
                             disp_w=32, disp_h=8, blank=4,
                             max_line_bytes=64, fifo_depth=16, burst=900)
    eols = sum(1 for b in beats if b.eol)
    lasts = sum(1 for b in beats if b.last)
    check("bursty reader: no wedge, eol per delivered line, lasts flow",
          ov == 1 and eols >= 12 and lasts >= 2 and eols == len(beats) // 28,
          f"{len(beats)} beats, {eols} eols, {lasts} lasts")

    # 9. OVERFLOW: stall downstream past the elastic; the flag fires.
    tight = rand_frame(rng, 12, 56, 24)
    _, ov, _, _ = run("blv2_g", [{"bits": 12, "raw": tight}],
                      disp_w=64, disp_h=26, blank=2, stall=3000,
                      max_line_bytes=128, fifo_depth=16)
    check("sticky overflow fires under a hard stall", ov == 1)

    ok = all(checks)
    print("\n" + ("BAYERLINK_IN V2 PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
