"""Structural composition: instantiate generated cores, and nest the result.

`generate()` turns one NumPy function into one module. `compose()` turns several
of those into a design -- it declares the nets, instantiates the cores and wires
the streaming handshake. A composed module describes itself exactly as a
generated core does, so it can be an Instance inside another compose():

    gain_core ──▶ front_end ──▶ stereo_top
                      └────────────┘  (instantiated twice)

That is what makes the hierarchy real rather than two flat levels. A reusable
subsystem is built ONCE and instantiated per sensor, instead of having its graph
copied into every top level.

Two things compose() enforces before emitting a line, both of which used to be
possible to get wrong silently:

  TYPES     a source and sink must agree on width AND on an opaque domain tag.
            A 12-bit Bayer stream and a 12-bit luma stream are identical to a
            compiler and nonsense to connect.
  FAN-OUT   an output may feed one consumer that applies backpressure, plus any
            number of SINKS (modules that produce no stream and so never stall).
            Two blocking consumers need a buffering fork, which does not exist,
            so it is refused rather than emitted as something that deadlocks the
            first time a branch stalls.

Run:  python examples/hierarchy.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import BUILD
from np2hw import (Connection, Image2D, Instance, Param, Port, StreamType,
                   compose, generate, to_ir)

W, H, BITS = 8, 4, 12
FULL = (1 << BITS) - 1
N = W * H


def gain(img, g):
    """Fixed-point gain then saturate -- one stage of the subsystem."""
    return ((g * img.astype(np.uint32)) // 16).clip(0, FULL).astype(np.uint16)


def build():
    """core -> subsystem (two stages) -> top (subsystem twice)."""
    image = Image2D("img", W, H, bits=BITS)
    g = Param("gain", bits=8, default=16,
              description="Q4.4 gain; 16 is unity")
    _, out = to_ir(gain, image, g)
    core = generate(out, module_name="gain_core")

    bayer = StreamType(BITS, ("sof", "eol", "last"), "bayer")

    # A subsystem: two gain stages back to back, sharing one register.
    front = compose(
        module_name="front_end",
        instances=[
            Instance("a", core, bind={"gain": "param_gain"},
                     domains={"in": "bayer", "out": "bayer"},
                     comment="first gain stage"),
            Instance("b", core, bind={"gain": "param_gain"},
                     domains={"in": "bayer", "out": "bayer"},
                     comment="second gain stage"),
        ],
        connections=[Connection("src", "a.in"), Connection("a.out", "b.in"),
                     Connection("b.out", "dst")],
        ports=[Port("param_gain", "in", width=8,
                    comment="Shared by both stages of this subsystem"),
               Port("src", "in", stream=bayer), Port("dst", "out", stream=bayer)],
        header=["// front_end: two gain stages"],
    )

    # ...instantiated twice, with its OWN interface making that possible.
    top = compose(
        module_name="stereo_top",
        instances=[Instance("left", front, bind={"param_gain": "param_left_gain"}),
                   Instance("right", front, bind={"param_gain": "param_right_gain"})],
        connections=[Connection("l_in", "left.src"), Connection("left.dst", "l_out"),
                     Connection("r_in", "right.src"), Connection("right.dst", "r_out")],
        ports=[Port("param_left_gain", "in", width=8),
               Port("param_right_gain", "in", width=8),
               Port("l_in", "in", stream=bayer), Port("r_in", "in", stream=bayer),
               Port("l_out", "out", stream=bayer), Port("r_out", "out", stream=bayer)],
        header=["// stereo_top: ONE subsystem, instantiated twice"],
    )
    return core, front, top


TB = """
module tb;
  localparam W=%(W)d, H=%(H)d, N=W*H;
  reg clk=0, rst=1, iv=0, sof=0, ordy=1;
  reg [%(MSB)d:0] idata=0;
  wire lrdy, rrdy, lv, rv, lsof, leol, llast, rsof, reol, rlast;
  wire [%(MSB)d:0] ldata, rdata;
  stereo_top dut(.clk(clk), .rst(rst),
    .param_left_gain(8'd%(LG)d), .param_right_gain(8'd%(RG)d),
    .l_in_valid(iv), .l_in_ready(lrdy), .l_in_data(idata),
    .l_in_sof(sof), .l_in_eol(1'b0), .l_in_last(1'b0),
    .r_in_valid(iv), .r_in_ready(rrdy), .r_in_data(idata),
    .r_in_sof(sof), .r_in_eol(1'b0), .r_in_last(1'b0),
    .l_out_valid(lv), .l_out_ready(ordy), .l_out_data(ldata),
    .l_out_sof(lsof), .l_out_eol(leol), .l_out_last(llast),
    .r_out_valid(rv), .r_out_ready(ordy), .r_out_data(rdata),
    .r_out_sof(rsof), .r_out_eol(reol), .r_out_last(rlast));
  always #5 clk = ~clk;
  reg [%(MSB)d:0] pix [0:N-1];
  integer i, fd;
  initial begin
    $readmemh("in.hex", pix);
    fd = $fopen("out.txt", "w");
    @(posedge clk); rst=0;
    for (i=0; i<N+8; i=i+1) begin
      @(negedge clk); iv=(i<N); sof=(i==0); idata=(i<N)?pix[i]:0;
      @(posedge clk);
      #1 if (lv && rv) $fwrite(fd, "%%0d %%0d\\n", ldata, rdata);
    end
    $fclose(fd); $finish;
  end
endmodule
"""


def main():
    core, front, top = build()
    os.makedirs(BUILD, exist_ok=True)
    for meta in (core, front, top):
        with open(os.path.join(BUILD, f"{meta['module']}.v"), "w") as fh:
            fh.write(meta["verilog"] + "\n")

    print("hierarchy:")
    print(f"  gain_core   {len(core.verilog.splitlines()):3d} lines")
    print(f"  front_end   nestable={front['interface']['nestable']}, "
          f"stream {front['interface']['input']['prefix']} -> "
          f"{front['interface']['output']['prefix']}, "
          f"domain {front['interface']['output']['domain']!r}")
    instantiations = [ln.strip() for ln in top["verilog"].splitlines()
                      if "front_end u_" in ln]
    print(f"  stereo_top  instantiates the SAME subsystem twice: {instantiations}")

    left_gain, right_gain = 24, 8
    rng = np.random.default_rng(9)
    A = rng.integers(0, FULL + 1, (H, W)).astype(np.uint16)
    A[0, 0], A[0, 1] = 0, FULL
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(v):03x}" for v in A.ravel()) + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(TB % {"W": W, "H": H, "MSB": BITS - 1,
                       "LG": left_gain, "RG": right_gain})

    subprocess.run(["iverilog", "-o", "sim.vvp", "gain_core.v", "front_end.v",
                    "stereo_top.v", "tb.v"], check=True, cwd=BUILD,
                   capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD, capture_output=True)
    got = np.loadtxt(os.path.join(BUILD, "out.txt"), dtype=np.int64).reshape(-1, 2)

    # Oracle: the SAME NumPy function, applied twice (the subsystem is two
    # stages), per eye with that eye's gain.
    expect_l = gain(gain(A, left_gain), left_gain).ravel().astype(np.int64)
    expect_r = gain(gain(A, right_gain), right_gain).ravel().astype(np.int64)
    n = min(len(got), N)
    ok = (np.array_equal(got[:n, 0], expect_l[:n])
          and np.array_equal(got[:n, 1], expect_r[:n]) and n == N)

    print(f"\n  {n} pixels per eye, left gain {left_gain / 16:.2f}x twice, "
          f"right {right_gain / 16:.2f}x twice -> "
          + ("bit-exact vs NumPy" if ok else "MISMATCH"))
    if not ok:
        print("    left  DUT:", got[:8, 0].tolist(), "exp", expect_l[:8].tolist())
        print("    right DUT:", got[:8, 1].tolist(), "exp", expect_r[:8].tolist())

    nested = len(instantiations) == 2 and front["interface"]["nestable"]
    print("\n" + ("HIERARCHY PASS" if ok and nested else "FAIL"))
    return 0 if ok and nested else 1


if __name__ == "__main__":
    sys.exit(main())
