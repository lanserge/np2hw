"""The framebuffer read engine, against a memory that answers awkwardly.

A vendor DMA fetches a frame back for the raster and, in doing so, makes
one thing impossible: knowing when its first beat arrives relative to the
raster's first pixel. That offset lands wherever it lands each time the
link locks -- 0, 16, 48 and 80 pixels were all measured on one bench from
reloads of an unchanged bitstream -- and it was corrected by hand, with a
number passed on a command line. A picture that comes back shifted by a
random amount whenever a cable is replugged does not recover by itself.

`framebuffer_read()` cannot have that offset, because it owns the
addressing: its first beat IS the frame's first pixel and it says so on
`tuser`. This proves that, and the things that would quietly undo it.

  IN ORDER, EXACTLY     every beat of a frame, once, in raster order,
                        with the picture read as a WINDOW of a wider
                        frame (line pitch != line length).
  FRAMED                tuser on the first beat of each frame and no
                        other; tlast on each line's last beat and no
                        other. This is what the raster arms on.
  FRAME AFTER FRAME     the engine restarts and the second frame is as
                        exact as the first, since a display asks
                        forever.
  4K IS NEVER CROSSED   AXI forbids it and an interconnect may mangle
                        rather than report a burst that does. Checked
                        per burst, against a base deliberately placed
                        so that unclipped bursts WOULD cross.
  A SLOW MEMORY         latency and gaps in the read data, and a raster
                        that stalls: the picture is unchanged, because
                        prefetch is the point.
  NEVER OVERRUN         the engine never has more beats in flight than
                        the FIFO can take, or a slave is left holding
                        data it cannot deliver.

Run:  python examples/framebuffer_read.py   (needs iverilog)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _harness import BUILD
from np2hw.video_mem import framebuffer_read

DW, BURST, FIFO = 32, 16, 128

TB = """
`timescale 1ns/1ps
module tb;
  localparam DW = %(DW)d;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg  [31:0] base   = 32'h%(BASE)08x;
  reg  [15:0] lbeats = 16'd%(LBEATS)d;
  reg  [15:0] lns    = 16'd%(LINES)d;
  reg  [31:0] stride = 32'd%(STRIDE)d;
  reg         en     = 0;

  wire [3:0] arid; wire [31:0] araddr; wire [7:0] arlen;
  wire [2:0] arsize, arprot; wire [1:0] arburst; wire [3:0] arcache;
  wire arvalid; wire arready;
  reg [DW-1:0] rdata = 0; reg [1:0] rresp = 0;
  reg rlast = 0, rvalid = 0; wire rready;
  wire [DW-1:0] tdata; wire tvalid, tuser, tlast; reg tready = 0;
  wire underflow, slverr;

  framebuffer_read dut (
    .clk(clk), .rst_n(!rst), .base_addr(base), .line_beats(lbeats),
    .lines(lns), .stride_bytes(stride), .enable(en),
    .m_axi_arid(arid), .m_axi_araddr(araddr), .m_axi_arlen(arlen),
    .m_axi_arsize(arsize), .m_axi_arburst(arburst),
    .m_axi_arcache(arcache), .m_axi_arprot(arprot),
    .m_axi_arvalid(arvalid), .m_axi_arready(arready),
    .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid), .m_axi_rready(rready),
    .m_axis_tdata(tdata), .m_axis_tvalid(tvalid), .m_axis_tready(tready),
    .m_axis_tuser(tuser), .m_axis_tlast(tlast),
    .underflow(underflow), .slverr(slverr));

  // ---- a memory with ONE outstanding transaction ----------------------
  // arready drops while a burst is being delivered, which is a legal and
  // common slave. Holding it high while serving would let a second
  // address overwrite the first mid-delivery -- a fault in the model
  // that reads exactly like a fault in the engine.
  //
  // Data at an address IS that address, so the checker knows what every
  // beat should have been without holding a copy of the picture.
  integer lat = 0, gap = 0;
  reg [8:0] k = 0;
  reg [31:0] q_addr = 0; reg [7:0] q_len = 0;
  integer bursts = 0, cross4k = 0, maxflight = 0, flight = 0;

  // A clocked state machine, not an initial/forever with blocking
  // assignments. The first version of this model raced the DUT's
  // sampling: it believed it had accepted a burst the engine never saw
  // accepted, and the engine sat correctly waiting while the model
  // insisted it had answered. A testbench that can lie about a
  // handshake will cost more than the design it is checking.
  // DEPTH>1 models a memory that overlaps the latency of later bursts
  // with the delivery of earlier ones -- which is the only way a raster
  // is ever fed, and therefore the only way to find out whether this
  // engine keeps enough addresses in the air to be fed by it.
  localparam QD = %(QDEPTH)d;
  reg [31:0] q_a [0:7];
  reg [7:0]  q_l [0:7];
  integer    q_t [0:7];
  integer    qh = 0, qt = 0, qn = 0, qi;

  localparam S_IDLE = 0, S_LAT = 1, S_DATA = 2, S_GAP = 3;
  reg [1:0] st = S_IDLE;
  integer   wait_n = 0;

  // COMBINATIONAL. Registered from `qn < QD` it could still be high in
  // the cycle the queue filled, and AXI says a transfer HAS happened
  // when ARVALID and ARREADY are both high -- so the engine counted a
  // burst the model then dropped, and waited forever for data that was
  // never going to come. The engine was right; the model was lying.
  assign arready = !rst && (qn < QD);

  always @(posedge clk) begin
    for (qi = 0; qi < 8; qi = qi + 1)
      if (q_t[qi] > 0) q_t[qi] <= q_t[qi] - 1;
    if (arvalid && arready) begin
      q_a[qt] = araddr; q_l[qt] = arlen; q_t[qt] = lat;
      qt = (qt + 1) %% 8; qn = qn + 1;
      if ((araddr[11:0] + ((arlen + 1) * (DW/8))) > 4096)
        cross4k = cross4k + 1;
      bursts = bursts + 1;
      flight = flight + arlen + 1;
      if (flight > maxflight) maxflight = flight;
    end
    if (rst) begin
      st <= S_IDLE; rvalid <= 0; rlast <= 0; k <= 0;
      qh = 0; qt = 0; qn = 0;
    end else case (st)
      S_IDLE: if (qn != 0 && q_t[qh] == 0) begin
                q_addr <= q_a[qh]; q_len <= q_l[qh];
                k <= 0; st <= S_LAT;
              end
      S_LAT: begin
               rdata <= q_addr; rlast <= (q_len == 0); rvalid <= 1;
               st <= S_DATA;
             end
      S_DATA: if (rvalid && rready) begin
                flight = flight - 1;
                if (k == q_len) begin
                  rvalid <= 0; rlast <= 0; st <= S_IDLE;
                  qh = (qh + 1) %% 8; qn = qn - 1;
                end else begin
                  k <= k + 1;
                  if (gap) begin
                    rvalid <= 0; rlast <= 0; wait_n <= gap; st <= S_GAP;
                  end else begin
                    rdata <= q_addr + ((k + 1) * (DW/8));
                    rlast <= (k + 1 == q_len);
                  end
                end
              end
      S_GAP: if (wait_n == 0) begin
               rdata <= q_addr + (k * (DW/8));
               rlast <= (k == q_len); rvalid <= 1; st <= S_DATA;
             end else wait_n <= wait_n - 1;
    endcase
  end

  // ---- the checker ----------------------------------------------------
  integer beat = 0, line = 0, frame = 0, errs = 0;
  integer users = 0, lasts = 0, want, delivered = 0, starve = 0;
  always @(posedge clk) begin
    // Steady state only: frame 0 includes filling the pipeline.
    if (frame >= 1 && tready && !tvalid) starve = starve + 1;
    if (tvalid && tready) begin
      want = base + (line * stride) + (beat * (DW/8));
      if (tdata !== want[DW-1:0]) begin
        if (errs < 5) $display("  beat %%0d,%%0d: got %%h want %%h",
                               line, beat, tdata, want[DW-1:0]);
        errs = errs + 1;
      end
      if (tuser) users = users + 1;
      if (tuser !== ((beat == 0) && (line == 0)))
        begin $display("  tuser wrong at %%0d,%%0d", line, beat); errs = errs + 1; end
      if (tlast !== (beat == lbeats - 1))
        begin $display("  tlast wrong at %%0d,%%0d", line, beat); errs = errs + 1; end
      if (tlast) lasts = lasts + 1;
      delivered = delivered + 1;
      beat = beat + 1;
      if (beat == lbeats) begin
        beat = 0; line = line + 1;
        if (line == lns) begin line = 0; frame = frame + 1; end
      end
    end
  end

  // The raster's readiness, driven on the clock like everything else.
  // Blocking-assigned from an initial block it raced the checker, and
  // the two disagreed about whether a beat had moved -- which reads as
  // a duplicated pixel and is not one.
  integer cyc = 0;
  reg go = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    // BLANK models the real consumer: a 1080p60 raster asks for 1920
    // of every 2200 columns and is idle for the rest, and that idle is
    // exactly when a prefetch FIFO catches up. A testbench that
    // consumes every cycle is asking for something no display asks for.
    tready <= go && !(%(STALL)d && ((cyc %% 97) < 7))
                 && !(%(BLANK)d && ((cyc %% 2200) >= 1920));
  end

  integer fd;
  initial begin
    lat = %(LAT)d; gap = %(GAP)d;
    fd = $fopen("%(OUT)s", "w");
    repeat (8) @(posedge clk); rst = 0; en = 1;
    repeat (20) @(posedge clk);
    go = 1;
    while (frame < 2) @(posedge clk);
    $fwrite(fd, "RESULT %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d %%0d\\n",
            errs, users, lasts, bursts, cross4k, maxflight, starve,
            delivered, cyc);
    $fclose(fd);
    $finish;
  end

  initial begin
    #%(TIMEOUT)d;
    $fwrite(fd, "TIMEOUT %%0d %%0d %%0d %%0d %%0d\\n", frame, line, beat, delivered, cyc);
    $fclose(fd);
    $finish;
  end
endmodule
"""


def run(name, line_beats, lines, stride, base=0x1000, lat=6, gap=0,
        stall=0, timeout=4000000, qdepth=1, burst=BURST, fifo=FIFO,
        blank=0):
    core = framebuffer_read(data_bits=DW, burst_len=burst, fifo_depth=fifo,
                            module_name="framebuffer_read")
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, f"{name}.v"), "w") as fh:
        fh.write(core["verilog"] + "\n")
    out = os.path.join(BUILD, f"{name}.txt")
    with open(os.path.join(BUILD, f"tb_{name}.v"), "w") as fh:
        fh.write(TB % {"DW": DW, "BASE": base, "LBEATS": line_beats,
                       "LINES": lines, "STRIDE": stride, "LAT": lat,
                       "GAP": gap, "STALL": stall,
                       "OUT": out.replace("\\", "/"), "TIMEOUT": timeout,
                       "QDEPTH": qdepth, "BLANK": blank})
    subprocess.run(["iverilog", "-g2012", "-o", f"{name}.vvp",
                    f"{name}.v", f"tb_{name}.v"],
                   check=True, cwd=BUILD, capture_output=True)
    subprocess.run([os.path.join(BUILD, f"{name}.vvp")], check=True,
                   cwd=BUILD, capture_output=True)
    with open(out) as fh:
        line = fh.readline().split()
    if line[0] == "TIMEOUT":
        return None
    return dict(zip(("errs", "users", "lasts", "bursts", "cross4k",
                     "maxflight", "starve", "delivered", "cycles"),
                    (int(v) for v in line[1:])))


def main():
    ok = []

    def check(label, cond, detail=""):
        print(f"  {label:<54} {'PASS' if cond else 'FAIL'}  {detail}")
        ok.append(bool(cond))

    print("framebuffer_read:")

    # A window of a wider frame: 40 beats of picture on a 64-beat pitch,
    # which is how a smaller sensor sits centred in a display-sized frame.
    r = run("fbr_a", line_beats=40, lines=8, stride=64 * 4)
    check("two frames, every beat exact, read as a window",
          r and r["errs"] == 0, f"{r}")
    check("tuser once per frame, tlast once per line",
          r and r["users"] == 2 and r["lasts"] == 16,
          f"users={r['users']} lasts={r['lasts']}")

    # 0x0FD0 + 16 beats x 4 bytes = 0x1010: an unclipped burst steps
    # OVER the line. 0x0FC0 would land exactly ON it and prove
    # nothing -- the first version of this test did that.
    r = run("fbr_b", line_beats=40, lines=6, stride=40 * 4, base=0x0FD0)
    check("no burst crosses a 4K boundary", r and r["cross4k"] == 0
          and r["errs"] == 0, f"bursts={r['bursts']} crossing={r['cross4k']}")

    # Never more in flight than the FIFO can take.
    check("beats in flight never exceed the FIFO",
          r and r["maxflight"] <= FIFO, f"peak={r['maxflight']} of {FIFO}")

    # A bus with other users on it, and a raster that stalls.
    r = run("fbr_c", line_beats=40, lines=6, stride=64 * 4, lat=25, gap=3,
            stall=1)
    check("slow memory and a stalling raster change nothing",
          r and r["errs"] == 0 and r["users"] == 2, f"{r}")

    # A line that is an exact multiple of the burst, and one that is not.
    r = run("fbr_d", line_beats=64, lines=4, stride=64 * 4)
    check("a line that divides the burst exactly", r and r["errs"] == 0)
    r = run("fbr_e", line_beats=37, lines=4, stride=48 * 4)
    check("a line that does NOT divide the burst", r and r["errs"] == 0)

    # THROUGHPUT, which is the whole question of whether this can
    # replace a vendor DMA. The bar is not 1.0 beats/clock: a raster
    # only asks during active video, so 1080p60 averages
    # 1920*1080 / (2200*1125) = 0.84 of the pixel clock. Anything above
    # that with margin feeds the screen; below it tears, whatever the
    # peak rate looks like.
    DEMAND_1080P60 = (1920 * 1080) / (2200 * 1125)

    r = run("fbr_t1", line_beats=256, lines=4, stride=256 * 4, lat=30,
            qdepth=1)
    rate1 = r["delivered"] / r["cycles"]
    check("one burst in flight cannot feed a 1080p60 raster",
          rate1 < DEMAND_1080P60 and r["errs"] == 0,
          f"{rate1:.2f} < {DEMAND_1080P60:.2f} demanded -- this is why a "
          "DMA needs the bus pipelined")

    r = run("fbr_t4", line_beats=256, lines=4, stride=256 * 4, lat=30,
            qdepth=4)
    rate4 = r["delivered"] / r["cycles"]
    check("pipelined, 16-beat bursts clear the demand",
          rate4 > DEMAND_1080P60 and r["errs"] == 0,
          f"{rate4:.2f} vs {DEMAND_1080P60:.2f} -- margin "
          f"{(rate4 / DEMAND_1080P60 - 1) * 100:.0f}%")

    # The shortfall from 1.0 is the address channel: arvalid cannot rise
    # again until the cycle after it is accepted, so every burst costs
    # about two cycles it does not spend moving data. Longer bursts
    # amortise that, and a framebuffer is exactly where long bursts are
    # natural -- the addresses are consecutive by definition.
    r = run("fbr_t64", line_beats=256, lines=4, stride=256 * 4, lat=30,
            qdepth=4, burst=64, fifo=512)
    rate64 = r["delivered"] / r["cycles"]
    check("64-beat bursts amortise the address gap",
          rate64 > rate4 and r["errs"] == 0,
          f"{rate64:.2f} beats/clock, up from {rate4:.2f}")

    # The residual bubbles are the address gap, two cycles per burst,
    # and a real raster's blanking absorbs them: this is the case the
    # hardware will actually run.
    r = run("fbr_tb", line_beats=1920, lines=4, stride=1920 * 4, lat=30,
            qdepth=4, burst=64, fifo=512, blank=1, timeout=40000000)
    check("against a 1080p60 raster's real duty cycle, it never starves",
          r and r["errs"] == 0 and r["starve"] == 0,
          f"{r['starve']} starved cycles across a whole frame"
          if r else "TIMEOUT")

    print("\n" + ("FRAMEBUFFER_READ PASS" if all(ok) else "SOME FAILED"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
