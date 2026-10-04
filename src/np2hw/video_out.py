# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""scanout: a pixel stream becomes a timed raster.

The mirror of :mod:`video_in`. That module decodes a link and follows
whatever the sender states; this one DEFINES a link -- it drives a
display, so its timing is never negotiated with the sink. The one
concession points the other way: `genlock` lets vertical blanking
breathe so the raster can follow its SOURCE -- the display still sees
a raster that never stutters. That asymmetry decides everything below.

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

What this block does NOT do is fetch. It consumes an elastic stream;
who produces it -- a memory reader like :func:`np2hw.video_mem.
framebuffer_read`, or the live pipeline itself -- is the caller's
composition, and this interface is the same for both.
"""
from __future__ import annotations

from .ports import Pin, header, published

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
    # Simulation-scale raster: the control laws (genlock, windowing,
    # arming) are geometry-independent, and a 200x80 frame simulates
    # three orders of magnitude faster than 2200x1125. Not a display
    # mode; a testbench's.
    "sim_tiny": (200, 80, 160, 64, 170, 180, 70, 74, 10.0, 1, 1),
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
            fill: int = 0x101010,
            genlock: bool = False,
            genlock_lead_lines: int = 2,
            genlock_ps_per_pixel: int = 560) -> dict:
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
        genlock: slave this raster to the incoming stream by
            steering the PIXEL CLOCK, not the line count. The
            raster's geometry NEVER changes -- a television drops
            sync on any vertical-total motion, however slow
            (bench-paid: one line per frame held fabric lock
            perfectly and lost the TV entirely; a fixed raster on
            the same clock was solid). Instead the module emits
            fine phase-shift steps for the MMCM that makes its
            clock (ps_en/ps_incdec/ps_done -- PSCLK is this
            module's own clk): ~24 ps per step, cumulative and
            unbounded, which is a fractional frequency trim with
            hundreds of ppm of authority. Phase is measured at
            each frame start's arrival at the queue head, in
            PIXELS, against a target G_LEAD lines before the
            window; small errors drain through the phase shifter,
            invisible to any sink. An error beyond two lines is
            not steerable at 24 ps -- the counters JAM to the
            target once (one visibly short frame, the same price
            as a mode set, cooled down so a wild source cannot
            jam repeatedly). Crystal ppm is the genlock's job;
            an fps step is a mode switch and stays software's.
        genlock_lead_lines: how early the SOF should arrive, in
            lines. THIS IS STORAGE, not preference: with no frame
            store, every line of lead is a line the upstream
            elastic must hold. Two lines wants a ~4096-deep
            crossing FIFO.
        genlock_ps_per_pixel: fine phase-shift steps per pixel
            tick -- a fact about the MMCM feeding this raster
            (pixel_period / (VCO_period/56); 74.25 MHz from a
            742.5 VCO is 560). The board states it because the
            board owns the clock.

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

    L = []
    a = L.append
    a(f"// generated by np2hw -- scanout for {t['name']} "
      f"({t['pixel_mhz']:g} MHz pixel clock)")
    a("// The mode is a BUILD fact because it is a clock; the picture's")
    a("// size and place are RUN-TIME facts, on registers. A frame may be")
    a("// dropped, never displaced: the raster arms every frame and will")
    a("// not start one until that frame's first beat is at the head.")
    params = [
        f"    parameter H_TOT  = {t['h_total']}, parameter V_TOT  = {t['v_total']},",
        f"    parameter H_ACT  = {h_act}, parameter V_ACT  = {v_act},",
        f"    parameter HS_BEG = {t['hs_beg']}, parameter HS_END = {t['hs_end']},",
        f"    parameter VS_BEG = {t['vs_beg']}, parameter VS_END = {t['vs_end']},",
        f"    parameter HS_POS = {int(t['hsync_positive'])}, "
        f"parameter VS_POS = {int(t['vsync_positive'])},",
        f"    parameter [{data_bits-1}:0] FILL = {data_bits}'h{fill:06x},",
        "    // SELFTEST paints from the raster's OWN counters and ignores",
        "    // the stream. It answers one question and no other: is the",
        "    // data this block emits where its own enable says it is? A",
        "    // displacement that survives this is downstream of here.",
        "    parameter SELFTEST = 0",
    ]

    def axis(signal, name, width=1, direction="input", **kw):
        return Pin(name, direction, width,
                   bus=("xilinx.com:interface:axis:1.0", "s_axis", signal),
                   **kw)

    def vid(signal, name, width=1):
        return Pin(name, "output", width, reg=True, bus=(
            None if iface is None else
            (f"xilinx.com:interface:{iface}:1.0", iface, signal)))

    pins = [
        Pin("clk", "input"),
        Pin("rst", "input"),
        Pin("locked", "input", note="the pixel clock's own testimony"),
        axis("TDATA", "s_axis_tdata", data_bits,
             before=("the picture, as an elastic stream",)),
        axis("TVALID", "s_axis_tvalid"),
        axis("TREADY", "s_axis_tready", direction="output"),
        axis("TUSER", "s_axis_tuser", note="start of frame"),
        axis("TLAST", "s_axis_tlast", note="end of line (the raster rules)"),
        Pin("param_win_x0", "input", 16, before=(
            "WHERE the picture goes: run-time, because any size source may",
            "land on any size screen. Reset places a full-raster window at",
            "the origin, so an unconfigured build shows the whole picture.")),
        Pin("param_win_y0", "input", 16),
        Pin("param_win_w", "input", 16),
        Pin("param_win_h", "input", 16),
        Pin("param_center", "input", note="derive the offsets instead"),
    ]
    if genlock:
        pins += [
            Pin("param_follow", "input", note="0: free-run the mode", before=(
                "GENLOCK by clock steering: the raster's geometry NEVER",
                "moves (a TV drops sync on any vertical-total motion --",
                "bench-paid). The stream is followed through the MMCM's",
                "fine phase shifter, ~24 ps at a time; PSCLK is this",
                "module's own clk. Everything is derived from the stream",
                "itself -- no occupancy tap, no side channel.")),
            Pin("ps_en", "output", reg=True, note="one PSEN pulse per step"),
            Pin("ps_incdec", "output", reg=True,
                note="1: delay (retard raster)"),
            Pin("ps_done", "input", note="MMCM's PSDONE"),
        ]
    pins += [
        Pin("act_x0", "output", 16, before=(
            "where it ACTUALLY landed -- read-only, because a derived",
            "value written back into its register becomes an explicit ask")),
        Pin("act_y0", "output", 16),
        vid("ACTIVE_VIDEO", "vid_active_video"),
        vid("DATA", "vid_data", data_bits),
        vid("HSYNC", "vid_hsync"),
        vid("VSYNC", "vid_vsync"),
        Pin("status", "output", 16),
    ]
    L += header(module_name, pins, parameter_lines=params)
    a(f"    reg [{xb-1}:0] x;")
    a(f"    reg [{yb-1}:0] y;")
    a("")
    a("    // The window, as the hardware sees it. Centring is a shift, not")
    a("    // a divider, and it is DERIVED -- never stored back into the")
    a("    // register the host wrote.")
    a("    wire [16:0] c_x0 = (H_ACT > param_win_w)")
    a("                     ? ((H_ACT - param_win_w) >> 1) : 17'd0;")
    a("    wire [16:0] c_y0 = (V_ACT > param_win_h)")
    a("                     ? ((V_ACT - param_win_h) >> 1) : 17'd0;")
    a("    wire [16:0] w_x0_n = param_center ? c_x0 : {1'b0, param_win_x0};")
    a("    wire [16:0] w_y0_n = param_center ? c_y0 : {1'b0, param_win_y0};")
    a("    wire [17:0] x_end_n = w_x0_n + param_win_w;")
    a("    wire [17:0] y_end_n = w_y0_n + param_win_h;")
    a("")
    a("    // A window that does not fit is REFUSED, not silently clipped:")
    a("    // a picture quietly cropped is a bug that looks like a choice.")
    a("    wire fits_n = (param_win_w != 0) && (param_win_h != 0)")
    a("               && (x_end_n <= H_ACT) && (y_end_n <= V_ACT);")
    a("")
    a("    // Every one of those is FRAME-CONSTANT, so it is worked out")
    a("    // once and compared against -- not worked out again beneath")
    a("    // every pixel. While placement was a build-time constant this")
    a("    // cost nothing, because a constant folds away; the first build")
    a("    // that put it on real registers missed setup by 124 ps, with a")
    a("    // subtract, a shift, a mux and two adds -- nineteen levels of")
    a("    // logic and seven carry chains -- sitting under the pixel path")
    a("    // at 74.25 MHz. The arithmetic now gets a clock of its own and")
    a("    // the pixel path sees registers. (Run-time placement is the")
    a("    // whole point of these being ports, so the cheap case must be")
    a("    // the one where they actually change.)")
    a("    reg [16:0] w_x0 = 17'd0, w_y0 = 17'd0;")
    a("    reg [17:0] x_end = 18'd0, y_end = 18'd0;")
    a("    reg fits = 1'b0;")
    a("    always @(posedge clk) begin")
    a("        w_x0  <= w_x0_n;  w_y0  <= w_y0_n;")
    a("        x_end <= x_end_n; y_end <= y_end_n;")
    a("        fits  <= fits_n;")
    a("    end")
    a("    // ...and the readback follows the copy the raster is USING, so")
    a("    // what act_x0/act_y0 report is where the picture actually is,")
    a("    // not where it is about to be.")
    a("    assign act_x0 = w_x0[15:0];")
    a("    assign act_y0 = w_y0[15:0];")
    a("")
    a("    wire active = (x < H_ACT) && (y < V_ACT);")
    a("    wire blank  = (y >= V_ACT);")
    a("    wire in_win = fits && active")
    a("               && (x >= w_x0) && (x < x_end)")
    a("               && (y >= w_y0) && (y < y_end);")
    a("    wire first_px = in_win && (x == w_x0[" + f"{xb-1}:0]) && (y == w_y0[{yb-1}:0]);")
    a("    wire last_px  = in_win && (x == (x_end - 1))")
    a("                           && (y == (y_end - 1));")
    a("")
    a("    // The frame length this raster is RUNNING -- and it NEVER")
    a("    // changes: a television drops sync on any vertical-total")
    a("    // motion, however slow (bench-paid: one line per frame held")
    a("    // fabric lock perfectly and lost the TV entirely). Genlocked,")
    a("    // the following happens in the CLOCK, 24 ps at a time.")
    a(f"    reg [{yb}:0] v_tot_eff;")
    if genlock:
        frame_px = t["h_total"] * t["v_total"]
        pb = _bits(frame_px)
        owed_cap = 2 * t["h_total"] * genlock_ps_per_pixel
        ob = _bits(owed_cap)
        a("    reg [15:0] in_period;  // line ticks since last ARRIVAL")
        a("    reg head_sof_d;        // one arrival edge per input frame")
        a("    // the steering debt: phase-shift steps still to issue,")
        a("    // OVERWRITTEN at each arrival (a proportional re-target;")
        a("    // the slew-limited drain is the smoothing)")
        a(f"    reg [{ob-1}:0] owed;")
        a("    reg owed_ret;          // 1: retard the raster (delay)")
        a("    reg ps_busy;           // one step in flight")
        a("    reg [4:0] ps_tmo;      // PSDONE comes ~12 PSCLKs later;")
        a("                           // a timeout so a miswired done")
        a("                           // pin degrades to slow, not stuck")
        a("    reg [5:0] jam_cool;    // frames since the last counter")
        a("                           // jam: a wild source must not")
        a("                           // jam every frame")

        a("    reg glocked;             // STATUS, not a control state")
        a("    reg [2:0] lock_run;")

        a(f"    localparam [{yb}:0] G_LEAD = {genlock_lead_lines};")
        a(f"    localparam [{ob-1}:0] PS_PER_PX = {genlock_ps_per_pixel};")
    a("    reg armed;                   // this frame's first beat is HERE")
    a("    reg ever_armed;")
    a("    wire head_sof = s_axis_tvalid && s_axis_tuser;")
    if genlock:
        a("    // ONE consumption rule: window positions consume, and")
        a("    // nothing else does. Pacing falls out (an eager drain")
        a("    // would let the SOF's timing measure queue occupancy")
        a("    // instead of phase -- simulated, not imagined); an")
        a("    // EARLY frame start waits naturally, because blanking")
        a("    // simply has no window positions to consume it; a LATE")
        a("    // one passes mid-window, where its position IS the")
        a("    // error. No lock state gates the data path. Blanking")
        a("    // additionally DRAINS non-SOF beats: without a discard")
        a("    // path, any transient where the raster runs long grows")
        a("    // the backlog forever (window consumption is exactly")
        a("    // one frame per raster frame -- it can never catch")
        a("    // up; simulated as an unbounded pile-up). Converged,")
        a("    // the drain is a no-op, so the wait measurement is")
        a("    // clean precisely in the regime that uses it.")
        a("    assign s_axis_tready = in_win || (blank && !head_sof);")
    else:
        a("    // Consume inside the window when armed; otherwise drain")
        a("    // during vertical blanking until the next frame start is")
        a("    // at the head.")
        a("    assign s_axis_tready = armed ? in_win"
          " : (blank && !head_sof);")
    a("")
    if genlock:
        a("    // measurement helpers (combinational into registers)")
        a(f"    wire [{yb-1}:0] pass_dy = y - w_y0[{yb-1}:0];")
        a("    // the SOF reaching the queue head is the arrival: it")
        a("    // fires exactly once per input frame regardless of drain")
        a("    // or window state. Target: arrive G_LEAD lines before")
        a("    // the window origin and wait out the lead in blanking.")
        a("    wire sof_arr = head_sof && !head_sof_d;")
        a(f"    wire [{yb}:0] arr_tgt = (w_y0[{yb}:0] >= G_LEAD)")
        a(f"        ? (w_y0[{yb}:0] - G_LEAD)")
        a(f"        : (v_tot_eff + w_y0[{yb}:0] - G_LEAD);")
        a("    // signed-shortest-way phase error, in whole lines")
        a(f"    wire [{yb}:0] arr_dy = ({{1'b0, y}} >= arr_tgt)")
        a(f"        ? ({{1'b0, y}} - arr_tgt)")
        a(f"        : ({{1'b0, y}} + v_tot_eff - arr_tgt);")
        a("    wire arr_late = (arr_dy != 0)")
        a("                 && (arr_dy < (v_tot_eff >> 1));")
        a(f"    wire [{yb}:0] arr_err = arr_late ? arr_dy")
        a(f"        : (arr_dy == 0) ? {{{yb + 1}{{1'b0}}}}")
        a("        : (v_tot_eff - arr_dy);")
        a("    // the same error, in PIXELS: what the phase shifter can")
        a("    // actually steer -- computed in a REGISTERED pipeline,")
        a("    // one multiply per cycle: the single-cycle cone (y*H_TOT,")
        a("    // wrap subtract, x560, clamp) missed timing by 2.3 ns.")
        a("    // The measurement is once per frame; three cycles of")
        a("    // latency are free. The memory reader's burst decision")
        a("    // paid for this same lesson first.")
        a(f"    reg [{xb-1}:0] arr_x;   // the arrival, captured")
        a(f"    reg [{yb-1}:0] arr_y;")
        a("    reg [2:0] calc;        // one-hot measurement pipeline")
        a(f"    reg [{pb}:0] pos_r, tgt_r, err_r;")
        a("    reg late_r;")
    a("    reg [24:0] alive;")
    a("    reg [3:0]  vsyncs;")
    a("    reg vs_d, underflow, misalign, refused;")
    a("    reg [2:0] sof_run, sof_cnt;")
    a("    always @(posedge clk) begin")
    a("        alive <= alive + 1;")
    a("        if (rst || !locked) begin")
    a("            x <= 0; y <= 0; armed <= 0; ever_armed <= 0;")
    a("            v_tot_eff <= V_TOT;")
    if genlock:
        a("            in_period <= 0; head_sof_d <= 0;")
        a("            owed <= 0; owed_ret <= 0; calc <= 0;")
        a("            ps_en <= 0; ps_incdec <= 0;")
        a("            ps_busy <= 0; ps_tmo <= 0; jam_cool <= 0;")
        a("            glocked <= 0; lock_run <= 0;")
    a("            vid_active_video <= 0; vid_hsync <= 0; vid_vsync <= 0;")
    a("            vid_data <= FILL; vsyncs <= 0; vs_d <= 0;")
    a("            underflow <= 0; misalign <= 0; refused <= 0;")
    a("            sof_run <= 0; sof_cnt <= 0;")
    a("        end else begin")
    a(f"            x <= (x == H_TOT-1) ? {xb}'d0 : x + {xb}'d1;")
    a(f"            if (x == H_TOT-1) begin")
    a(f"                y <= (y >= v_tot_eff-1) ? {yb}'d0 : y + {yb}'d1;")
    a(f"                if (y >= v_tot_eff-1) begin")
    a("                    // frame boundary: the ONE place the length")
    a("                    // may change")
    a("                    v_tot_eff <= V_TOT;")
    a("                end")
    a("            end")
    if genlock:
        a("")
        a("            // ---- the genlock: the clock is steered, the")
        a("            // counters are still. Each arrival re-targets the")
        a("            // steering debt (proportional overwrite); the")
        a("            // stepper drains it one MMCM fine-phase step at a")
        a("            // time. An error past two lines is not steerable")
        a("            // at 24 ps -- the counters JAM to target once,")
        a("            // under a cooldown.")
        a("            if (x == H_TOT-1) begin")
        a("                in_period <= (in_period == 16'hFFFF)")
        a("                             ? in_period : in_period + 1;")
        a(f"                if (y >= v_tot_eff-1 && jam_cool != 0)")
        a("                    jam_cool <= jam_cool - 1;")
        a("            end")
        a("            if (s_axis_tvalid && s_axis_tready")
        a("                && s_axis_tuser) begin")
        a("                // lock is a REPORT: frame starts passing at")
        a("                // the window origin, four in a row.")
        a(f"                if (pass_dy == {yb}'d0) begin")
        a("                    if (lock_run != 3'd7)")
        a("                        lock_run <= lock_run + 3'd1;")
        a("                    glocked <= (lock_run >= 3'd3);")
        a("                end else begin")
        a("                    lock_run <= 0; glocked <= 0;")
        a("                end")
        a("            end")
        a("            // the stepper: one PSEN pulse, then wait for")
        a("            // PSDONE (a timeout degrades a miswired done pin")
        a("            // to slow, never to stuck)")
        a("            ps_en <= 1'b0;")
        a("            if (ps_busy) begin")
        a("                if (ps_done || ps_tmo == 0) ps_busy <= 0;")
        a("                else ps_tmo <= ps_tmo - 5'd1;")
        a("            end else if (param_follow && owed != 0) begin")
        a("                ps_en <= 1'b1;")
        a("                ps_incdec <= owed_ret;")
        a("                ps_busy <= 1'b1; ps_tmo <= 5'd31;")
        a(f"                owed <= owed - {ob}'d1;")
        a("            end")
        a("            head_sof_d <= head_sof;")
        a("            if (sof_arr) begin")
        a("                in_period <= 0;")
        a("                if (param_follow) begin")
        a(f"                    if (arr_err > {yb+1}'d2) begin")
        a("                        // beyond steering: jam ONCE, cooled")
        a("                        if (jam_cool == 0) begin")
        a(f"                            x <= {xb}'d0;")
        a(f"                            y <= arr_tgt[{yb-1}:0];")
        a("                            jam_cool <= 6'd63;")
        a("                        end")
        a("                        owed <= 0; calc <= 3'b000;")
        a("                    end else begin")
        a("                        // within steering: capture the")
        a("                        // arrival, measure over the next")
        a("                        // three cycles")
        a("                        arr_x <= x; arr_y <= y;")
        a("                        calc <= 3'b001;")
        a("                    end")
        a("                end")
        a("            end")
        a("            if (calc[0]) begin")
        a(f"                pos_r <= arr_y * {t['h_total']}")
        a(f"                         + {{{pb+1-xb}'d0, arr_x}};")
        a(f"                tgt_r <= arr_tgt * {t['h_total']};")
        a("                calc <= 3'b010;")
        a("            end")
        a("            if (calc[1]) begin")
        a("                if (pos_r >= tgt_r) begin")
        a(f"                    late_r <= (pos_r - tgt_r) < {frame_px // 2};")
        a(f"                    err_r <= ((pos_r - tgt_r) < {frame_px // 2})")
        a("                        ? (pos_r - tgt_r)")
        a(f"                        : ({frame_px} - (pos_r - tgt_r));")
        a("                end else begin")
        a(f"                    late_r <= (pos_r + {frame_px} - tgt_r)")
        a(f"                              < {frame_px // 2};")
        a(f"                    err_r <= ((pos_r + {frame_px} - tgt_r)")
        a(f"                              < {frame_px // 2})")
        a(f"                        ? (pos_r + {frame_px} - tgt_r)")
        a(f"                        : (tgt_r - pos_r);")
        a("                end")
        a("                calc <= 3'b100;")
        a("            end")
        a("            if (calc[2]) begin")
        a(f"                if (err_r > {pb+1}'d1) begin")
        a(f"                    owed <= (err_r > {2 * t['h_total']})")
        a(f"                        ? {ob}'d{owed_cap}")
        a("                        : err_r * PS_PER_PX;")
        a("                    owed_ret <= late_r;")
        a("                end else begin")
        a("                    owed <= 0;")
        a("                end")
        a("                calc <= 3'b000;")
        a("            end")
        a("            // source gone: stop steering and say so. The")
        a("            // raster itself needs no restoring -- it never")
        a("            // moved.")
        a(f"            if (in_period > 16'd{t['v_total'] * 4}) begin")
        a("                glocked <= 0; lock_run <= 0; owed <= 0;")
        a("                calc <= 0;")
        a("            end")
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
    if genlock:
        a("            // a late frame start consumed mid-window arms")
        a("            // too: show the joined frame from where it joins")
        a("            if (s_axis_tvalid && s_axis_tready && s_axis_tuser)")
        a("                begin armed <= 1; ever_armed <= 1; end")
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
    if genlock:
        a("    assign status = {glocked, vsyncs[2:0], alive[24:22],")
        a("                     refused, misalign,")
        a("                     underflow, ever_armed, locked, sof_cnt};")
    else:
        a("    assign status = {vsyncs, alive[24:22], refused, misalign,")
        a("                     underflow, ever_armed, locked, sof_cnt};")
    a("endmodule")

    return {
        "verilog": "\n".join(L) + "\n",
        "module": module_name,
        "mode": t,
        "sink": sink,
        "data_bits": data_bits,
        "interface": {"clock": "clk", "reset": "rst",
                      "param_prefix": "param_", "ports": published(pins)},
        "params": [("win_x0", 16), ("win_y0", 16), ("win_w", 16),
                   ("win_h", 16), ("center", 1)],
        # The video side leaves flops -- a raster that jittered would be
        # visible. The stream side does not: `s_axis_tready` is decided
        # combinationally from the window state, so it lands in the
        # producer's path. On a chain of traced cores, whose `in_ready`
        # is also combinational, that path runs all the way back to the
        # source without meeting a register.
        "combinational_outputs": ("s_axis_tready",),
        # The genlock variant narrows the vsync counter by one bit to
        # make room for glocked; everything below is common. Bit 15 is
        # the zero-extension pad in both.
        "status_bits": ({"sof_per_frame": (0, 3), "locked": 3, "armed": 4,
                         "underflow": 5, "misalign": 6, "refused": 7,
                         "alive": (8, 3), "vsyncs": (11, 3), "glocked": 14}
                        if genlock else
                        {"sof_per_frame": (0, 3), "locked": 3, "armed": 4,
                         "underflow": 5, "misalign": 6, "refused": 7,
                         "alive": (8, 3), "vsyncs": (11, 4)}),
    }
