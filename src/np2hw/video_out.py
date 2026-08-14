"""scanout: a pixel stream becomes a timed raster.

The mirror of :mod:`video_in`. That module decodes a link and follows
whatever the sender states; this one DEFINES a link -- it drives a
display, so its timing is not negotiable and not per-frame. That
asymmetry decides everything below.

  THE MODE IS A BUILD FACT, because it is a CLOCK.  A raster's geometry
      and its pixel clock are one thing: 1080p60 is 148.5 MHz and
      1280x720p60 is 74.25. You cannot change the active width per frame
      without changing the clock, and changing the clock means the
      display re-locks. So the mode is chosen at generation, from ONE
      table (below) -- the same table the transmitting side should read,
      because "what a 1080p60 raster is" must have a single owner.

  THE PICTURE'S PLACEMENT IS A RUN-TIME FACT.  Where a picture sits in
      that raster, and how big it is, are registers: any size source on
      any size screen. `center` derives the offsets instead of stating
      them -- and the derived values are NEVER written back into the
      offset registers, because a derived value stamped into a register
      becomes an explicit ask, and then the picture stops following its
      source. The offsets read back what the host wrote; `act_x0/act_y0`
      report where it actually landed.

  A FRAME MAY BE DROPPED, NEVER DISPLACED.  This is the law this block
      exists to keep, and it was paid for on a bench: if the reader has
      not delivered the frame's first beat when the picture's first
      pixel is due, starting anyway paints the WHOLE frame displaced by
      the gap -- and nothing downstream can detect it, because a beat
      that has not arrived carries no evidence. So the raster ARMS every
      frame: it refuses to start until that frame's first beat is at the
      head, and an engine that is late costs one frame of fill, not a
      picture that slides for the rest of the session.

  A WINDOW THAT DOES NOT FIT IS REFUSED, VISIBLY.  Not clipped silently.
      A sticky status bit, the same posture the receiver takes toward a
      header it cannot honour.

  DATA AND DATA-ENABLE LEAVE TOGETHER.  One register stage for the
      pixel, the enable and both syncs. A block that skews them by even
      one clock hands the sink a picture shifted by one, and no status
      bit can see it from inside.

What this block does NOT do is fetch. It consumes an elastic stream and
someone else reads memory -- today a vendor DMA, whose data lags its own
start-of-frame marker by a fixed number of beats that the caller has to
measure and compensate. Emitting our own reader is the next step and
retires that compensation; the interface here does not change when it
lands.
"""
from __future__ import annotations

# ONE owner for "what a raster is". Totals, active area, sync window and
# pixel clock, from CEA-861. Anything transmitting to us should read this
# table rather than keep a second copy: two owners of a raster is how a
# container and a display disagree about a line.
MODES = {
    #            h_tot v_tot h_act v_act hs_beg hs_end vs_beg vs_end  MHz  pol
    "640x480p60":  (800,  525,  640,  480,  656,  752,  490,  492,  25.175, 0, 0),
    "720x480p60":  (858,  525,  720,  480,  736,  798,  489,  495,  27.000, 0, 0),
    "1280x720p60": (1650, 750, 1280,  720, 1390, 1430,  725,  730,  74.250, 1, 1),
    "1280x720p50": (1980, 750, 1280,  720, 1720, 1760,  725,  730,  74.250, 1, 1),
    "1920x1080p30": (2200, 1125, 1920, 1080, 2008, 2052, 1084, 1089, 74.250, 1, 1),
    "1920x1080p50": (2640, 1125, 1920, 1080, 2448, 2492, 1084, 1089, 148.500, 1, 1),
    "1920x1080p60": (2200, 1125, 1920, 1080, 2008, 2052, 1084, 1089, 148.500, 1, 1),
}

ALIASES = {"720p60": "1280x720p60", "720p50": "1280x720p50",
           "1080p30": "1920x1080p30", "1080p50": "1920x1080p50",
           "1080p60": "1920x1080p60", "480p60": "720x480p60",
           "vga": "640x480p60"}

SINKS = {
    # what the emitted video interface is called and how it is annotated,
    # so the block drops onto an encoder without a shim
    "dvi": "vid_io",          # Digilent rgb2dvi and its kin
    "raw": None,              # plain ports, for a testbench or a panel
}


def mode_timing(mode: str) -> dict:
    """The named raster, as numbers. Raises on an unknown name, listing
    what exists -- a typo must not become a silently different mode."""
    key = ALIASES.get(mode, mode)
    if key not in MODES:
        raise ValueError(
            f"unknown display mode {mode!r}. This table is the one owner "
            f"of raster geometry; add the mode here rather than passing "
            f"timings around. Known: {', '.join(sorted(MODES))}")
    h_tot, v_tot, h_act, v_act, hs0, hs1, vs0, vs1, mhz, hp, vp = MODES[key]
    return {"name": key, "h_total": h_tot, "v_total": v_tot,
            "h_active": h_act, "v_active": v_act,
            "hs_beg": hs0, "hs_end": hs1, "vs_beg": vs0, "vs_end": vs1,
            "pixel_mhz": mhz, "hsync_positive": bool(hp),
            "vsync_positive": bool(vp)}


def _bits(value: int) -> int:
    return max(1, int(value).bit_length())


def scanout(mode: str = "1080p60",
            module_name: str = "scanout",
            sink: str = "dvi",
            data_bits: int = 24,
            fill: int = 0x101010) -> dict:
    """Emit a raster that paints a placed window from an elastic stream.

    Args:
        mode: a name from :data:`MODES` (or :data:`ALIASES`). Fixes the
            timing AND the pixel clock the caller must provide.
        module_name: the emitted module's name.
        sink: ``"dvi"`` annotates the video outputs as a Xilinx vid_io
            bundle so an encoder connects without a shim; ``"raw"``
            leaves them as plain ports.
        data_bits: width of one pixel on the stream and on the sink.
        fill: what shows where the window does not cover, and where a
            frame had to be dropped. Deliberately not black: a fill that
            is visibly not video makes a dropped frame legible.

    Returns:
        ``{"verilog", "module", "mode", "interface", ...}``. Run-time
        placement arrives on ``param_win_x0/y0/w/h`` and ``param_center``;
        ``act_x0/act_y0`` report where the picture actually landed, and
        the status word carries armed/underflow/misalign/refused plus a
        count of start-of-frame beats per frame.
    """
    if sink not in SINKS:
        raise ValueError(f"unknown sink {sink!r}; known: "
                         f"{', '.join(sorted(SINKS))}")
    t = mode_timing(mode)
    h_act, v_act = t["h_active"], t["v_active"]
    xb, yb = _bits(t["h_total"]), _bits(t["v_total"])
    iface = SINKS[sink]

    def port(logical, physical):
        if iface is None:
            return ""
        return (f'    (* X_INTERFACE_INFO = '
                f'"xilinx.com:interface:{iface}:1.0 {iface} {logical}" *)\n')

    L = []
    a = L.append
    a(f"// generated by np2hw -- scanout for {t['name']} "
      f"({t['pixel_mhz']:g} MHz pixel clock)")
    a("// The mode is a BUILD fact because it is a clock; the picture's")
    a("// size and place are RUN-TIME facts, on registers. A frame may be")
    a("// dropped, never displaced: the raster arms every frame and will")
    a("// not start one until that frame's first beat is at the head.")
    a(f"module {module_name} #(")
    a(f"    parameter H_TOT  = {t['h_total']}, parameter V_TOT  = {t['v_total']},")
    a(f"    parameter H_ACT  = {h_act}, parameter V_ACT  = {v_act},")
    a(f"    parameter HS_BEG = {t['hs_beg']}, parameter HS_END = {t['hs_end']},")
    a(f"    parameter VS_BEG = {t['vs_beg']}, parameter VS_END = {t['vs_end']},")
    a(f"    parameter HS_POS = {int(t['hsync_positive'])}, "
      f"parameter VS_POS = {int(t['vsync_positive'])},")
    a(f"    parameter [{data_bits-1}:0] FILL = {data_bits}'h{fill:06x},")
    a("    // SELFTEST paints from the raster's OWN counters and ignores")
    a("    // the stream. It answers one question and no other: is the")
    a("    // data this block emits where its own enable says it is? A")
    a("    // displacement that survives this is downstream of here.")
    a("    parameter SELFTEST = 0")
    a(") (")
    a("    input  wire clk,")
    a("    input  wire rst,")
    a("    input  wire locked,          // the pixel clock's own testimony")
    a("    // the picture, as an elastic stream")
    a('    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)')
    a(f"    input  wire [{data_bits-1}:0] s_axis_tdata,")
    a('    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)')
    a("    input  wire        s_axis_tvalid,")
    a('    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)')
    a("    output wire        s_axis_tready,")
    a('    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TUSER" *)')
    a("    input  wire        s_axis_tuser,   // start of frame")
    a('    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST" *)')
    a("    input  wire        s_axis_tlast,   // end of line (the raster rules)")
    a("    // WHERE the picture goes: run-time, because any size source may")
    a("    // land on any size screen. Reset places a full-raster window at")
    a("    // the origin, so an unconfigured build shows the whole picture.")
    a("    input  wire [15:0] param_win_x0,")
    a("    input  wire [15:0] param_win_y0,")
    a("    input  wire [15:0] param_win_w,")
    a("    input  wire [15:0] param_win_h,")
    a("    input  wire        param_center,  // derive the offsets instead")
    a("    // where it ACTUALLY landed -- read-only, because a derived")
    a("    // value written back into its register becomes an explicit ask")
    a("    output wire [15:0] act_x0,")
    a("    output wire [15:0] act_y0,")
    a(port("ACTIVE_VIDEO", "vid_active_video") + "    output reg         vid_active_video,")
    a(port("DATA", "vid_data") + f"    output reg  [{data_bits-1}:0] vid_data,")
    a(port("HSYNC", "vid_hsync") + "    output reg         vid_hsync,")
    a(port("VSYNC", "vid_vsync") + "    output reg         vid_vsync,")
    a("    output wire [15:0] status")
    a(");")
    a(f"    reg [{xb-1}:0] x;")
    a(f"    reg [{yb-1}:0] y;")
    a("")
    a("    // The window, as the hardware sees it. Centring is a shift, not")
    a("    // a divider, and it is computed LIVE -- never stored back.")
    a("    wire [16:0] c_x0 = (H_ACT > param_win_w)")
    a("                     ? ((H_ACT - param_win_w) >> 1) : 17'd0;")
    a("    wire [16:0] c_y0 = (V_ACT > param_win_h)")
    a("                     ? ((V_ACT - param_win_h) >> 1) : 17'd0;")
    a("    wire [16:0] w_x0 = param_center ? c_x0 : {1'b0, param_win_x0};")
    a("    wire [16:0] w_y0 = param_center ? c_y0 : {1'b0, param_win_y0};")
    a("    assign act_x0 = w_x0[15:0];")
    a("    assign act_y0 = w_y0[15:0];")
    a("")
    a("    // A window that does not fit is REFUSED, not silently clipped:")
    a("    // a picture quietly cropped is a bug that looks like a choice.")
    a("    wire fits = (param_win_w != 0) && (param_win_h != 0)")
    a("             && ((w_x0 + param_win_w) <= H_ACT)")
    a("             && ((w_y0 + param_win_h) <= V_ACT);")
    a("")
    a("    wire active = (x < H_ACT) && (y < V_ACT);")
    a("    wire blank  = (y >= V_ACT);")
    a("    wire in_win = fits && active")
    a("               && (x >= w_x0) && (x < (w_x0 + param_win_w))")
    a("               && (y >= w_y0) && (y < (w_y0 + param_win_h));")
    a("    wire first_px = in_win && (x == w_x0[" + f"{xb-1}:0]) && (y == w_y0[{yb-1}:0]);")
    a("    wire last_px  = in_win && (x == (w_x0 + param_win_w - 1))")
    a("                           && (y == (w_y0 + param_win_h - 1));")
    a("")
    a("    reg armed;                   // this frame's first beat is HERE")
    a("    reg ever_armed;")
    a("    wire head_sof = s_axis_tvalid && s_axis_tuser;")
    a("    // Consume inside the window when armed; otherwise drain during")
    a("    // vertical blanking until the next frame start is at the head.")
    a("    assign s_axis_tready = armed ? in_win : (blank && !head_sof);")
    a("")
    a("    reg [24:0] alive;")
    a("    reg [3:0]  vsyncs;")
    a("    reg vs_d, underflow, misalign, refused;")
    a("    reg [2:0] sof_run, sof_cnt;")
    a("    always @(posedge clk) begin")
    a("        alive <= alive + 1;")
    a("        if (rst || !locked) begin")
    a("            x <= 0; y <= 0; armed <= 0; ever_armed <= 0;")
    a("            vid_active_video <= 0; vid_hsync <= 0; vid_vsync <= 0;")
    a("            vid_data <= FILL; vsyncs <= 0; vs_d <= 0;")
    a("            underflow <= 0; misalign <= 0; refused <= 0;")
    a("            sof_run <= 0; sof_cnt <= 0;")
    a("        end else begin")
    a(f"            x <= (x == H_TOT-1) ? {xb}'d0 : x + {xb}'d1;")
    a(f"            if (x == H_TOT-1) y <= (y == V_TOT-1) ? {yb}'d0 : y + {yb}'d1;")
    a("")
    a("            // ONE register stage for pixel, enable and both syncs:")
    a("            // skew them and the sink shows a picture shifted by the")
    a("            // skew, invisibly to every status bit in this module.")
    a("            vid_hsync <= HS_POS ? ((x >= HS_BEG) && (x < HS_END))")
    a("                                : !((x >= HS_BEG) && (x < HS_END));")
    a("            vid_vsync <= VS_POS ? ((y >= VS_BEG) && (y < VS_END))")
    a("                                : !((y >= VS_BEG) && (y < VS_END));")
    a("            vid_active_video <= active;")
    a("            if (SELFTEST)")
    a("                vid_data <= (x < 4 || x >= H_ACT-4)")
    a(f"                            ? {{{data_bits}{{1'b1}}}} : {{3{{x[7:0]}}}};")
    a("            else")
    a("                vid_data <= (armed && in_win && s_axis_tvalid)")
    a("                            ? s_axis_tdata : FILL;")
    a("")
    a("            vs_d <= vid_vsync;")
    a("            if (vid_vsync && !vs_d) vsyncs <= vsyncs + 4'd1;")
    a("            if (!fits) refused <= 1;")
    a("")
    a("            // Arm on the frame start, in the blanking before it is")
    a("            // due; disarm when the window's last pixel is spent, so")
    a("            // the next frame must arm again. Alignment is per FRAME.")
    a("            if (blank && head_sof) begin armed <= 1; ever_armed <= 1; end")
    a("            if (armed && last_px) armed <= 1'b0;")
    a("            if (armed && in_win && !s_axis_tvalid) underflow <= 1;")
    a("            if (armed && first_px && s_axis_tvalid && !s_axis_tuser)")
    a("                misalign <= 1;")
    a("            if (head_sof && s_axis_tready && sof_run != 3'd7)")
    a("                sof_run <= sof_run + 3'd1;")
    a("            if (first_px) begin")
    a("                sof_cnt <= sof_run;")
    a("                sof_run <= (head_sof && s_axis_tready) ? 3'd1 : 3'd0;")
    a("            end")
    a("        end")
    a("    end")
    a("    assign status = {vsyncs, alive[24:22], refused, misalign,")
    a("                     underflow, ever_armed, locked, sof_cnt};")
    a("endmodule")

    return {
        "verilog": "\n".join(L) + "\n",
        "module": module_name,
        "mode": t,
        "sink": sink,
        "data_bits": data_bits,
        "params": [("win_x0", 16), ("win_y0", 16), ("win_w", 16),
                   ("win_h", 16), ("center", 1)],
        "status_bits": {"sof_per_frame": (0, 3), "locked": 3, "armed": 4,
                        "underflow": 5, "misalign": 6, "refused": 7,
                        "alive": (8, 3), "vsyncs": (11, 4)},
    }
