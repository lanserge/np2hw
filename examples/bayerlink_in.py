"""The bayerlink v2 receiver, proven against the protocol's own reference.

bayerlink (github.com/bayerlink/bayerlink) carries packed raw Bayer over a
display link. `np2hw.video_in.bayerlink_in()` emits the receiving end:
parallel video from an HDMI receiver in, np2hw's elastic stream out at the build's depth,
with each frame's header owning depth and geometry AT RUN TIME.

The encoder here is not hand-rolled: containers come from the published
reference codec (`pip install bayerlink`), so this example is an INDEPENDENT
implementation meeting the reference across a simulated link. Seven claims,
each a way the receiver could be wrong on a bench with a scope instead:

  EVERY DEPTH     8, 10, 12, 14 and 16-bit frames -- one of each, in ONE
                  run of ONE build, the header switching the unpack engine
                  frame by frame -- all bit-exact with exact framing.
  ALIGNMENT       the header's depth is run-time, the build's is not, so
                  samples leave aligned to the build's: full scale in is
                  full scale out at every source depth, a deeper source
                  shifts down rather than wrapping, and a source at the
                  build's own depth passes through untouched.
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

from np2hw import bayerlink_in
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
  wire [%(OUTBITS)d-1:0] out_data;
  wire [15:0] hdrw, hdrh;
  wire [4:0] hdrb;
  wire [7:0] hdrsrc;
  wire [31:0] hdrseq;
  wire [7:0] rescnt;
  wire [1:0] hdrp;
  reg out_ready = 0;
  reg diagclr = 0;
  integer stall = 0;

  %(MODULE)s dut (.clk(clk), .rst(rst), .diag_clear(diagclr),
    .vid_de(vid_de), .vid_vsync(vid_vsync), .vid_data(vid_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
    .overflow(ov), .refused(refd), .refuse_code(rcode),
    .hdr_bits(hdrb), .hdr_width(hdrw), .hdr_height(hdrh),
    .hdr_source_id(hdrsrc), .hdr_frame_seq(hdrseq),
    .resync_count(rescnt),
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
    $fwrite(fd, "HDR %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d\\n", hdrb, hdrw, hdrh, hdrp, hdrv, hdrsrc, hdrseq, rescnt);
    // Before and after a DELIBERATE clear. These tallies survive rst
    // on purpose -- rst now arrives on every cable drop, and a counter
    // reset by the event it counts tells you nothing -- so the one
    // thing that does clear them has to be shown to.
    $fwrite(fd, "TALLY %%0d %%0d %%0d\\n", ov, refd, rescnt);
    // A LINK RESET must not touch them. This is the regression: rst
    // arrives on every cable drop now, and the recovery was erasing
    // the evidence of whatever went wrong just before it.
    @(posedge clk); rst = 1;
    repeat (4) @(posedge clk); rst = 0;
    repeat (4) @(posedge clk);
    $fwrite(fd, "SURVIVED %%0d %%0d %%0d\\n", ov, refd, rescnt);
    @(posedge clk); diagclr = 1;
    repeat (4) @(posedge clk); diagclr = 0;
    repeat (4) @(posedge clk);
    $fwrite(fd, "CLEARED %%0d %%0d %%0d\\n", ov, refd, rescnt);
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
        max_line_bytes=4096, fifo_depth=1024, stall=0, gap=200, burst=0,
        out_bits=12):
    """Build once, drive a LIST of frames (spec dicts or raw containers).

    Each frame spec: {"bits", "raw", "order"} -> encoded via the reference
    codec; or {"container": ndarray} pre-built (the refusal tests corrupt
    good containers, so even the refused bytes come from the reference).
    """
    core = bayerlink_in(max_line_bytes=max_line_bytes, module_name=name,
                        fifo_depth=fifo_depth, lane_map=lane_map,
                        bits=out_bits)
    assert core["out_bits"] == out_bits
    stream = []
    for i, spec in enumerate(frames):
        if "container" in spec:
            container = spec["container"]
        else:
            container = encode_frame(
                spec["raw"], spec.get("order", "RGGB"), frame_seq=i + 1,
                display=(disp_w, disp_h), bits=spec["bits"],
                source_id=spec.get("source_id", 0))
            # a spec may name its own frame_seq to stage a gap
            if "frame_seq" in spec:
                container = encode_frame(
                    spec["raw"], spec.get("order", "RGGB"),
                    frame_seq=spec["frame_seq"],
                    display=(disp_w, disp_h), bits=spec["bits"],
                    source_id=spec.get("source_id", 0))
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
                       "STALL": stall or 1, "GAP": gap,
                       "OUTBITS": out_bits})

    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", f"{name}.v", "tb.v"],
                   check=True, cwd=BUILD, capture_output=True)
    plusargs = ["+stall=1"] if stall else []
    if burst:
        plusargs = [f"+burst={burst}"]
    subprocess.run(["vvp", "sim.vvp"] + plusargs, check=True, cwd=BUILD,
                   capture_output=True)

    beats, overflow, refused, hdr = [], None, None, None
    tally = cleared = survived = None
    with open(os.path.join(BUILD, "out.txt")) as fh:
        for line in fh:
            if line.startswith("OVF"):
                overflow = int(line.split()[1])
            elif line.startswith("REF"):
                refused = tuple(int(v) for v in line.split()[1:])
            elif line.startswith("HDR"):
                hdr = tuple(int(v) for v in line.split()[1:])
            elif line.startswith("TALLY"):
                tally = tuple(int(v) for v in line.split()[1:])
            elif line.startswith("SURVIVED"):
                survived = tuple(int(v) for v in line.split()[1:])
            elif line.startswith("CLEARED"):
                cleared = tuple(int(v) for v in line.split()[1:])
            else:
                d, s, e, l = (int(v) for v in line.split())
                beats.append(Beat(d, bool(s), bool(e), bool(l)))
    return beats, overflow, refused, hdr, tally, (survived, cleared)


def align(raw, in_bits, out_bits):
    """What the receiver must emit: `raw` at `in_bits`, full-scale-aligned
    to `out_bits`. Shallower shifts UP with the top bits replicated into
    what it vacated, so full scale lands on full scale; deeper shifts DOWN.

    Written from the SPEC, not from the Verilog -- an oracle copied off
    the implementation agrees with it by construction and proves nothing.
    """
    v = raw.astype(np.uint32)
    if in_bits > out_bits:
        return v >> (in_bits - out_bits)
    shl = out_bits - in_bits
    top = (v >> (in_bits - shl)) if shl < in_bits else 0
    return ((v << shl) | top) & ((1 << out_bits) - 1)


def exact(beats, raws, depths=None, out_bits=12):
    """Beats == the aligned raw frames, each with exact framing."""
    at = 0
    depths = list(depths or [out_bits] * len(raws))
    for raw, in_bits in zip(raws, depths):
        h, w = raw.shape
        got = beats[at:at + raw.size]
        if len(got) != raw.size:
            return False
        want = align(raw, in_bits, out_bits)
        if not all(b.data == int(v) for b, v in zip(got, want.ravel())):
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
    beats, ov, refd, hdr, _t, _c = run("blv2_a", frames, disp_w=64, disp_h=10,
                               blank=24)
    check("five depths in one run, header-switched, bit-exact",
          exact(beats, raws, depths), f"{len(beats)} beats")
    check("no overflow across the depth sweep", ov == 0)
    check("nothing refused across the depth sweep", refd == (0, 0))
    check("hdr_* report the last header (16-bit, 56x6, source 0)",
          hdr[:6] == (16, 56, 6, 0, 1, 0), f"hdr={hdr}")

    # 1b. WHICH CAMERA. Calibration belongs to a sensor and cannot be
    # derived from the stream, so the one thing a hot swap needs that
    # the pixels cannot tell it is identity -- and the protocol has
    # carried it since v2. Report the ACCEPTED header's, per frame.
    seen = []
    for sid in (0, 7, 255):
        raw = rand_frame(rng, 12, 56, 6)
        _b, _o, _r, h, _t, _c = run(f"blv2_id{sid}",
                            [{"bits": 12, "raw": raw, "source_id": sid}],
                            disp_w=64, disp_h=10, blank=24)
        seen.append(h[5] == sid)
    check("hdr_source_id reports the header's camera", all(seen),
          f"{sum(seen)}/3")

    # Two sources in one run: the field must FOLLOW the frames, not
    # latch once. This is the hot-swap case in miniature.
    raws2 = [rand_frame(rng, 12, 56, 6), rand_frame(rng, 12, 56, 6)]
    _b, _o, _r, h, _t, _c = run("blv2_id2",
                        [{"bits": 12, "raw": raws2[0], "source_id": 3},
                         {"bits": 12, "raw": raws2[1], "source_id": 9}],
                        disp_w=64, disp_h=10, blank=24)
    check("two cameras in one run: the last accepted one is reported",
          h[5] == 9, f"hdr={h}")

    # frame_seq: which frame these numbers belong to. A statistic read
    # without it is a statistic about an unknown frame, and an exposure
    # loop built on that oscillates.
    raws3 = [rand_frame(rng, 12, 56, 6) for _ in range(3)]
    _b, _o, _r, h, _t, _c = run("blv2_seq",
                        [{"bits": 12, "raw": r} for r in raws3],
                        disp_w=64, disp_h=10, blank=24)
    # run() numbers frames from 1, so the third is 3
    check("hdr_frame_seq reports the accepted frame's sequence",
          h[6] == 3, f"hdr={h}")

    # 1d. SELF-RECOVERY, which is the whole point of carrying identity.
    # A hot swap is: one camera stops mid-stream, another starts. The
    # buffers still hold a half-line belonging to a stream that ended,
    # and draining it into the new frame corrupts the new frame. The
    # receiver notices IN BAND -- the camera changed, or the sequence
    # did not follow -- and empties itself. No reset, no host.
    good_a = rand_frame(rng, 12, 56, 6)
    good_b = rand_frame(rng, 12, 56, 6)
    # camera 1 sends a frame; camera 2 then sends one, its own sequence
    # starting over, as a freshly booted source does
    beats, ov, refd, h, _t, _c = run(
        "blv2_swap",
        [{"bits": 12, "raw": good_a, "source_id": 1, "frame_seq": 41},
         {"bits": 12, "raw": good_b, "source_id": 2, "frame_seq": 1}],
        disp_w=64, disp_h=10, blank=24)
    check("hot swap: both frames exact, no reset in between",
          exact(beats, [good_a, good_b], [12, 12]) and refd == (0, 0),
          f"{len(beats)} beats")
    # 2, not 1: the FIRST frame after reset is a restart too, having
    # nothing before it to follow. The swap is the second.
    check("hot swap: the restart is counted", h[7] == 2, f"hdr={h}")

    # A GAP from ONE camera -- a frame the link lost -- is a
    # discontinuity too, and the frame after it must still be exact.
    g1, g2 = rand_frame(rng, 12, 56, 6), rand_frame(rng, 12, 56, 6)
    beats, ov, refd, h, _t, _c = run(
        "blv2_gap",
        [{"bits": 12, "raw": g1, "source_id": 4, "frame_seq": 10},
         {"bits": 12, "raw": g2, "source_id": 4, "frame_seq": 40}],
        disp_w=64, disp_h=10, blank=24)
    check("a forward gap is a dropped frame, not a restart",
          exact(beats, [g1, g2], [12, 12]) and h[7] == 1, f"hdr={h}")

    # And the case that must NOT fire: consecutive frames from one
    # camera. A tail draining under its own header is this design
    # working; flushing there would cut a frame in half.
    c1, c2, c3 = (rand_frame(rng, 12, 56, 6) for _ in range(3))
    beats, ov, refd, h, _t, _c = run(
        "blv2_cont",
        [{"bits": 12, "raw": c1, "source_id": 6, "frame_seq": 100},
         {"bits": 12, "raw": c2, "source_id": 6, "frame_seq": 101},
         {"bits": 12, "raw": c3, "source_id": 6, "frame_seq": 102}],
        disp_w=64, disp_h=10, blank=24)
    check("a consecutive run counts ONE restart, not three",
          exact(beats, [c1, c2, c3], [12] * 3) and h[7] == 1,
          f"restarts={h[7]}")

    # 1e. WHAT A REAL LINK LOOKS LIKE. The display repeats a container
    # when the camera is slower than the raster, and drops one when it
    # is not. Both are ordinary. Keyed on "the very next sequence" this
    # counted a restart once per frame forever -- found on hardware,
    # not here, because every case above counts cleanly.
    r1, r2, r3, r4 = (rand_frame(rng, 12, 56, 6) for _ in range(4))
    beats, ov, refd, h, _t, _c = run(
        "blv2_real",
        [{"bits": 12, "raw": r1, "source_id": 5, "frame_seq": 100},
         {"bits": 12, "raw": r2, "source_id": 5, "frame_seq": 100},
         {"bits": 12, "raw": r3, "source_id": 5, "frame_seq": 103},
         {"bits": 12, "raw": r4, "source_id": 5, "frame_seq": 104}],
        disp_w=64, disp_h=10, blank=24)
    check("repeats and forward gaps are NOT restarts",
          h[7] == 1 and exact(beats, [r1, r2, r3, r4], [12] * 4),
          f"restarts={h[7]} (1 = the first frame only)")

    # And what IS one: the sequence going backwards, as a freshly
    # started source does.
    b1, b2 = rand_frame(rng, 12, 56, 6), rand_frame(rng, 12, 56, 6)
    beats, ov, refd, h, _t, _c = run(
        "blv2_back",
        [{"bits": 12, "raw": b1, "source_id": 5, "frame_seq": 900},
         {"bits": 12, "raw": b2, "source_id": 5, "frame_seq": 2}],
        disp_w=64, disp_h=10, blank=24)
    check("a sequence going BACKWARDS is a restart", h[7] == 2, f"hdr={h}")

    # 2. GEOMETRY is the header's, per frame.
    raws = [rand_frame(rng, 12, 56, 6), rand_frame(rng, 12, 30, 8),
            rand_frame(rng, 12, 8, 3)]
    frames = [{"bits": 12, "raw": r} for r in raws]
    beats, ov, refd, _, _t, _c = run("blv2_b", frames, disp_w=64, disp_h=10,
                             blank=24)
    check("three geometries in one run, header-owned",
          exact(beats, raws, [12] * 3) and ov == 0 and refd == (0, 0))

    # 3. LANES: permuted wiring, matched by lane_map.
    raws = [rand_frame(rng, 10, 56, 6)]
    beats, ov, refd, _, _t, _c = run("blv2_c", [{"bits": 10, "raw": raws[0]}],
                             disp_w=64, disp_h=10, blank=24,
                             lane_map=(2, 0, 1))
    check("permuted byte lanes, matched by lane_map",
          exact(beats, raws, [10]) and ov == 0)

    # 4. RATE RULE: the reference budget, 2028 samples in 2200 slots.
    raws = [rand_frame(rng, 12, 2028, 3)]
    beats, ov, refd, _, _t, _c = run("blv2_d", [{"bits": 12, "raw": raws[0]}],
                             disp_w=1920, disp_h=5, blank=280 - 1)
    check("reference geometry (2028 in 2200 slots), no overflow",
          exact(beats, raws, [12]) and ov == 0, f"{len(beats)} samples")

    # 5. REFUSAL: unknown fourcc, then capacity; good frames still decode.
    good = rand_frame(rng, 12, 56, 6)
    bad = encode_frame(rand_frame(rng, 12, 56, 6), "RGGB", frame_seq=9,
                       display=(64, 10), bits=12).copy()
    # fourcc lives at header bytes 8..11: byte k is [0, k // 3, k % 3]
    for k, ch in zip(range(8, 12), b"XXXX"):
        bad[0, k // 3, k % 3] = ch
    beats, ov, refd, hdr, _t, _c = run(
        "blv2_e", [{"container": bad}, {"bits": 12, "raw": good}],
        disp_w=64, disp_h=10, blank=24)
    check("unknown fourcc: zero beats, sticky code 2, next frame OK",
          exact(beats, [good], [12]) and refd == (1, 2), f"refd={refd}")

    wide = rand_frame(rng, 12, 200, 2)          # 300 bytes > 256 capacity
    beats, ov, refd, _, _t, _c = run(
        "blv2_f", [{"bits": 12, "raw": wide}, {"bits": 12, "raw": good}],
        disp_w=128, disp_h=10, blank=24, max_line_bytes=256)
    check("line beyond capacity: refused code 4, next frame OK",
          exact(beats, [good], [12]) and refd == (1, 4), f"refd={refd}")

    # 6. WRAP: a small sample FIFO wrapped many times by the one engine
    # whose batch stride (3) does not divide the depth -- the slot
    # addressing must wrap WITH the pointer, not run off the banks.
    raws = [rand_frame(rng, 8, 56, 40)]
    beats, ov, refd, _, _t, _c = run("blv2_w", [{"bits": 8, "raw": raws[0]}],
                             disp_w=64, disp_h=44, blank=8, fifo_depth=256)
    check("8-bit through ~9 FIFO wraps, stride 3 vs depth 256",
          exact(beats, raws, [8]) and ov == 0)

    # 6b. FULL SCALE lands on FULL SCALE. The subtle half of alignment: a
    # plain left shift maps an 8-bit 255 to 4080 in 12 bits and leaves
    # every white in the picture 0.4% dark, forever, in a way no chart
    # measurement would separate from a gain error.
    tops, bottoms = [], []
    for b in (8, 10, 12, 14, 16):
        frame = np.full((2, 8), (1 << b) - 1, dtype=np.uint16)
        frame[1, :] = 0
        beats, _ov, _rd, _h, _t, _c = run(f"blv2_fs{b}", [{"bits": b, "raw": frame}],
                                  disp_w=64, disp_h=10, blank=24)
        tops.append(all(x.data == 4095 for x in beats[:8]))
        bottoms.append(all(x.data == 0 for x in beats[8:16]))
    check("full scale in is full scale out, every depth into 12",
          all(tops), f"{sum(tops)}/5 depths")
    check("zero stays zero, every depth into 12", all(bottoms))

    # 6c. DEEPER than the datapath: 16-bit into a 12-bit build keeps a
    # picture and loses the low bits, rather than truncating the top and
    # losing the picture.
    deep = rand_frame(rng, 16, 56, 6)
    beats, _ov, _rd, _h, _t, _c = run("blv2_deep", [{"bits": 16, "raw": deep}],
                              disp_w=64, disp_h=10, blank=24)
    check("16-bit source into a 12-bit build: shifted down, not wrapped",
          exact(beats, [deep], [16]))

    # 6d. The same link into a build of a DIFFERENT depth: 10-bit source
    # into a 10-bit build passes through untouched.
    ten = rand_frame(rng, 10, 56, 6)
    beats, _ov, _rd, _h, _t, _c = run("blv2_ten", [{"bits": 10, "raw": ten}],
                              disp_w=64, disp_h=10, blank=24, out_bits=10)
    check("10-bit source into a 10-bit build: unshifted",
          exact(beats, [ten], [10], out_bits=10)
          and all(x.data == int(v) for x, v in zip(beats, ten.ravel())))

    # 7. LOSS RECOVERY: a hard stall mangles ONE frame's data, but the
    # framing comes from ingest tags, so the damage cannot leak -- the
    # NEXT frame decodes exact, sof/eol/last included. (The bench found
    # this: drain-counted framing turned one lossy gap into every
    # subsequent DMA packet misframed.)
    from np2hw.testing import check_framing as _cf
    f1 = rand_frame(rng, 12, 56, 24)
    f2 = rand_frame(rng, 12, 56, 24)
    beats, ov, refd, _, _t, _c = run(
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
    beats, ov, refd, _, _t, _c = run("blv2_y", [{"bits": 10, "raw": r} for r in raws],
                             disp_w=32, disp_h=8, blank=4,
                             max_line_bytes=64, fifo_depth=16, burst=900)
    eols = sum(1 for b in beats if b.eol)
    lasts = sum(1 for b in beats if b.last)
    check("bursty reader: no wedge, eol per delivered line, lasts flow",
          ov == 1 and eols >= 12 and lasts >= 2 and eols == len(beats) // 28,
          f"{len(beats)} beats, {eols} eols, {lasts} lasts")

    # 9. OVERFLOW: stall downstream past the elastic; the flag fires.
    tight = rand_frame(rng, 12, 56, 24)
    _, ov, _, _, _t, _c = run("blv2_g", [{"bits": 12, "raw": tight}],
                      disp_w=64, disp_h=26, blank=2, stall=3000,
                      max_line_bytes=128, fifo_depth=16)
    check("sticky overflow fires under a hard stall", ov == 1)

    # A tally cleared by the event it counts is not a tally, which is
    # why these survive rst -- rst now arrives on every cable drop. So
    # the ONLY thing that clears them must be shown to.
    check("the tallies survive to be read", _t is not None and _t[0] == 1,
          f"before clear: ovf/refused/resyncs = {_t}")
    check("a LINK RESET does not touch the tallies",
          _c[0] == _t, f"after rst: {_c[0]} (was {_t})")
    check("diag_clear clears them, and only a deliberate ask does",
          _c[1] == (0, 0, 0), f"after clear: {_c[1]}")

    # The boundary. Everything above is about what this module produces;
    # this is about what it costs the module NEXT to it. The head of the
    # sample FIFO is a distributed-RAM read and a 4:1 select, and handing
    # that over raw puts it in front of the consumer's first stage --
    # which cost this design -1.424ns into one consumer and -1.792ns into
    # another, from a receiver and a pipeline that each closed alone.
    #
    # The slice's own behaviour is proven in examples/skid.py, so what is
    # left to show here is that the output actually goes through it: that
    # no port is driven combinationally from the FIFO any more. Behaviour
    # above already covers the rest -- the rate rule (2028 samples inside
    # 2200 slots) is what would fail if the slice ever cost throughput.
    r = bayerlink_in(max_line_bytes=4096, bits=12)
    v = r["verilog"]
    check("the output leaves a flop, not the FIFO's read mux",
          ".m_data({out_last, out_eol, out_sof, out_data})" in v
          and "assign {out_last, out_eol, out_sof, out_data}" not in v
          and "assign out_valid" not in v,
          "output ports driven by the register slice")
    check("and the interface says so, for a composer that cannot look",
          r["interface"].get("output_registered") is True)

    ok = all(checks)
    print("\n" + ("BAYERLINK_IN V2 PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
