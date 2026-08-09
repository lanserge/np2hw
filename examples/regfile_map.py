"""A register file at addresses the CALLER allocated, not at i*4.

`axil_regfile(params, ...)` used to pack every register at word offset i*4. That
is fine for one core and wrong for a design: an application with its own address
map -- one aligned region per block, an identity word at each region's base --
would hold one layout while the emitted Verilog held another, and the two would
agree only on the day they were written.

So a register carries its own address. `Reg` says where it lives, how wide it
is, whether it is signed, what it resets to, whether it is writable at all, and
what it means:

    Reg("id_version", 32, offset=0x000, access="ro", value=0x8BAD0100)
    Reg("gain",        8, offset=0x100, reset=16)
    Reg("offset",     12, offset=0x104, signed=True)

Whoever allocates the address owns it. np2hw owns the decode. Four things fall
out of that, each checked below against a simulation rather than asserted:

  ADDRESSES   a register answers at the offset it was given, and nowhere else.
  READ-ONLY   an identity word is a constant wired into the read decode. It has
              no storage and no write path.
  ERRORS      a write to a read-only word, or to an unmapped address, is
              answered SLVERR. A silently dropped write presents as a
              configuration that had no effect, hours later and somewhere else.
  SIGN        a signed register reads back sign-extended, so a host that wrote
              -100 does not read 4044 out of a 12-bit register.

`control_wrap()` then puts that register file in front of ANY self-describing
module -- a generated core, or a whole composed design -- reading the clock, the
reset, the streams and the parameter ports out of its interface. The shadow
registers commit at the frame boundary, so a frame is processed with one
coherent set of coefficients.

Run:  python examples/regfile_map.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _harness import BUILD
from np2hw import Image2D, Param, Reg, control_wrap, generate, to_ir

W, H, BITS = 8, 4, 12
FULL = (1 << BITS) - 1
N = W * H

# The address map. Deliberately NOT contiguous: a gap the decode must refuse,
# and a block base far from word 0, which is precisely what i*4 cannot express.
ID_ADDRESS = 0x000
GAIN_ADDRESS = 0x100
OFFSET_ADDRESS = 0x104
UNMAPPED_ADDRESS = 0x0F0

IDENTITY = 0x8BAD0100
RESET_GAIN, RESET_OFFSET = 16, 0
NEW_GAIN, NEW_OFFSET = 24, -100


def tone(img, g, off):
    """Fixed-point gain, then a signed offset, then saturate."""
    return (((g * img.astype(np.int32)) // 16) + off).clip(0, FULL).astype(np.uint16)


def build():
    image = Image2D("img", W, H, bits=BITS)
    gain = Param("gain", bits=8, default=RESET_GAIN,
                 description="Q4.4 gain; 16 is unity")
    offset = Param("offset", bits=12, signed=True, default=RESET_OFFSET,
                   description="Signed offset added after the gain")
    _, out = to_ir(tone, image, gain, offset)
    core = generate(out, module_name="tone_core")

    registers = [
        Reg("id_version", 32, offset=ID_ADDRESS, access="ro", value=IDENTITY,
            description="Block ID in the high half, major.minor in the low half. "
                        "Read it before anything else."),
        Reg("gain", 8, offset=GAIN_ADDRESS, reset=RESET_GAIN,
            description="Q4.4 gain; 16 is unity"),
        Reg("offset", 12, offset=OFFSET_ADDRESS, signed=True, reset=RESET_OFFSET,
            description="Signed offset added after the gain"),
    ]
    # The core frames itself from sof and regenerates out_last, so the frame
    # boundary is on the OUTPUT side. Stated rather than guessed, which is what
    # the commit argument is for.
    wrap = control_wrap(
        core, registers,
        bind={"gain": "param_gain", "offset": "param_offset"},
        module_name="tone_ctrl", addr_bits=12,
        commit="out_valid && out_ready && out_last",
        header=["// tone_ctrl: a register file at caller-allocated addresses"],
    )
    return core, wrap


TB = """
`timescale 1ns/1ps
module tb;
  localparam W=%(W)d, H=%(H)d, N=W*H;
  reg clk=0, rst=1;
  reg [11:0] awaddr=0, araddr=0;
  reg [31:0] wdata=0;
  reg awvalid=0, wvalid=0, bready=0, arvalid=0, rready=0;
  wire awready, wready, bvalid, arready, rvalid;
  wire [1:0] bresp, rresp;
  wire [31:0] rdata;
  reg [1:0] last_bresp, last_rresp;
  reg [31:0] last_rdata;

  reg iv=0, sof=0, ordy=1;
  reg [%(MSB)d:0] idata=0;
  wire irdy, ov, osof, oeol, olast;
  wire [%(MSB)d:0] odata;

  tone_ctrl dut(.clk(clk), .rst(rst),
    .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
    .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid),
    .s_axil_wready(wready),
    .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
    .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
    .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid),
    .s_axil_rready(rready),
    .in_valid(iv), .in_ready(irdy), .in_data(idata), .in_sof(sof),
    .out_valid(ov), .out_ready(ordy), .out_data(odata),
    .out_sof(osof), .out_eol(oeol), .out_last(olast));

  always #5 clk = ~clk;

  // awvalid/wvalid are held until the response arrives: the slave registers
  // awready for one cycle and samples the write on the cycle AFTER, so
  // dropping them early loses the transfer. bready is held one cycle past
  // bvalid so the slave sees the acknowledgement and re-arms.
  task axi_write(input [11:0] a, input [31:0] d);
    begin
      @(negedge clk); awaddr=a; wdata=d; awvalid=1; wvalid=1; bready=1;
      wait (bvalid); last_bresp = bresp;
      @(negedge clk); awvalid=0; wvalid=0;
      @(posedge clk);
      @(negedge clk); bready=0;
      @(negedge clk);
    end
  endtask

  task axi_read(input [11:0] a);
    begin
      @(negedge clk); araddr=a; arvalid=1; rready=1;
      wait (rvalid); last_rdata = rdata; last_rresp = rresp;
      @(negedge clk); arvalid=0;
      @(posedge clk);
      @(negedge clk); rready=0;
      @(negedge clk);
    end
  endtask

  reg [%(MSB)d:0] pix [0:N-1];
  integer i, fd;
  initial begin
    $readmemh("in.hex", pix);
    fd = $fopen("out.txt", "w");
    repeat (4) @(posedge clk);
    @(negedge clk); rst=0;

    axi_read(12'h%(ID)03x);
    $fwrite(fd, "id %%0d %%0d\\n", last_rdata, last_rresp);

    axi_write(12'h%(ID)03x, 32'hDEADBEEF);      // read-only: must be refused
    $fwrite(fd, "wro %%0d\\n", last_bresp);
    axi_read(12'h%(ID)03x);
    $fwrite(fd, "id2 %%0d\\n", last_rdata);

    axi_write(12'h%(GAP)03x, 32'h1);            // unmapped: must be refused
    $fwrite(fd, "wgap %%0d\\n", last_bresp);
    axi_read(12'h%(GAP)03x);
    $fwrite(fd, "rgap %%0d\\n", last_rresp);

    axi_write(12'h%(GA)03x, 32'd%(NG)d);
    $fwrite(fd, "wgain %%0d\\n", last_bresp);
    axi_write(12'h%(OA)03x, 32'd%(NO)d);
    axi_read(12'h%(GA)03x);
    $fwrite(fd, "rgain %%0d\\n", last_rdata);
    axi_read(12'h%(OA)03x);
    $fwrite(fd, "roffset %%0d\\n", $signed(last_rdata));

    // Two frames. The first runs on the RESET configuration -- the writes are
    // in the shadow -- and the second on what was written.
    for (i=0; i<2*(N+8); i=i+1) begin
      @(negedge clk);
      iv = ((i %% (N+8)) < N);
      sof = ((i %% (N+8)) == 0);
      idata = iv ? pix[i %% (N+8)] : 0;
      @(posedge clk);
      #1 if (ov) $fwrite(fd, "pix %%0d\\n", odata);
    end
    $fclose(fd); $finish;
  end
endmodule
"""


def main():
    core, wrap = build()
    os.makedirs(BUILD, exist_ok=True)
    for name, text in (("tone_core", core["verilog"]), ("tone_ctrl", wrap["verilog"])):
        with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
            fh.write(text + "\n")

    print("register map (allocated by the caller, decoded by np2hw):")
    for name, offset in wrap["offsets"].items():
        print(f"  0x{offset:03x}  {name}")
    print(f"  commit pulse: {wrap['commit']}")

    rng = np.random.default_rng(11)
    A = rng.integers(0, FULL + 1, (H, W)).astype(np.uint16)
    A[0, 0], A[0, 1] = 0, FULL
    with open(os.path.join(BUILD, "in.hex"), "w") as fh:
        fh.write("\n".join(f"{int(v):03x}" for v in A.ravel()) + "\n")
    with open(os.path.join(BUILD, "tb.v"), "w") as fh:
        fh.write(TB % {"W": W, "H": H, "MSB": BITS - 1, "ID": ID_ADDRESS,
                       "GAP": UNMAPPED_ADDRESS, "GA": GAIN_ADDRESS,
                       "OA": OFFSET_ADDRESS, "NG": NEW_GAIN,
                       "NO": NEW_OFFSET & 0xFFFFFFFF})

    subprocess.run(["iverilog", "-o", "sim.vvp", "tone_core.v", "tone_ctrl.v",
                    "tb.v"], check=True, cwd=BUILD, capture_output=True)
    subprocess.run(["vvp", "sim.vvp"], check=True, cwd=BUILD, capture_output=True)

    log = {}
    pixels = []
    with open(os.path.join(BUILD, "out.txt")) as fh:
        for line in fh:
            key, _, rest = line.strip().partition(" ")
            if key == "pix":
                pixels.append(int(rest))
            else:
                log[key] = [int(v) for v in rest.split()]

    checks = []

    def check(label, ok, detail=""):
        checks.append(ok)
        print(f"  {label:<44} {'PASS' if ok else 'FAIL'}{'  ' + detail if detail else ''}")

    print("\nchecks:")
    identity, response = log["id"]
    check(f"id_version at 0x{ID_ADDRESS:03x} reads 0x{IDENTITY:08X}",
          identity == IDENTITY and response == 0, f"got 0x{identity:08X}")
    check("write to the read-only word is SLVERR", log["wro"][0] == 0b10)
    check("the read-only word did not change", log["id2"][0] == IDENTITY)
    check(f"write to unmapped 0x{UNMAPPED_ADDRESS:03x} is SLVERR",
          log["wgap"][0] == 0b10)
    check(f"read of unmapped 0x{UNMAPPED_ADDRESS:03x} is SLVERR",
          log["rgap"][0] == 0b10)
    check(f"write to 0x{GAIN_ADDRESS:03x} is OKAY", log["wgain"][0] == 0)
    check("gain reads back what was written", log["rgain"][0] == NEW_GAIN,
          f"got {log['rgain'][0]}")
    check("signed offset reads back sign-extended",
          log["roffset"][0] == NEW_OFFSET, f"got {log['roffset'][0]}")

    before = tone(A, RESET_GAIN, RESET_OFFSET).ravel().astype(np.int64)
    after = tone(A, NEW_GAIN, NEW_OFFSET).ravel().astype(np.int64)
    got = np.array(pixels[:2 * N], dtype=np.int64)
    check("the configuration under test changes the image",
          not np.array_equal(before, after))
    check("frame 1 runs on the RESET configuration (shadow not committed)",
          len(got) >= N and np.array_equal(got[:N], before))
    check("frame 2 runs on what was written (committed at the boundary)",
          len(got) >= 2 * N and np.array_equal(got[N:2 * N], after))

    ok = all(checks)
    print("\n" + ("REGFILE PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
