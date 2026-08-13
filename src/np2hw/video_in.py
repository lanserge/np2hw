"""bayerlink receiver: parallel video in, one sample per clock out.

The bayerlink protocol (github.com/bayerlink/bayerlink) carries packed raw
Bayer over a display link: line 0 of the active area is a header, every
following line holds one camera line of CSI-2 packed bytes, zero-padded to
the display width. An HDMI receiver (dvi2rgb and its kin) recovers the
parallel video -- pixel clock, data enable, vsync, 24 data bits -- and this
adapter turns that into np2hw's elastic stream: `valid/ready/data` with
`sof/eol/last` framing, one unshifted sample per beat in a 16-bit lane.

This is the protocol's version-2 receiver: the header is PARSED, in
hardware, and it is the single owner of every per-stream fact:

  DEPTH IS RUN-TIME   the fourcc names the packing (8/10/12/14/16-bit, the
      one CSI-2 rule at five depths); the unpack engine carries the whole
      table, so one build serves any sensor. There is no depth register --
      a register would be a second owner of a fact the header already
      states, and one fact with two owners is the bug.

  GEOMETRY IS RUN-TIME   width and height come from the header, per frame.
      The build fixes only CAPACITY (``max_line_bytes`` sizes the buffers);
      a header the silicon cannot honour is refused, visibly.

  REFUSAL IS THE API   a bad magic, an unknown fourcc, a width that
      misaligns its packing groups, a line that exceeds capacity: each
      refuses the FRAME (not the stream), latches a sticky code for
      software, and waits for the next header. Nothing is guessed.

Three facts of the link shape the datapath, as in v1:

  BYTES BURST FASTER THAN SAMPLES   a display pixel carries 3 payload
      bytes; at 12 bits that is 2 samples, at 16 bits 1.5. A byte FIFO
      deep enough for a full line absorbs the burst; the protocol's rate
      rule (camera width <= the display mode's total line slots) is the
      condition under which the line always drains before the next one.

  NO BACKPRESSURE UPSTREAM   video cannot be stalled, so a FIFO overflow
      is unavoidable data loss, and unobservable data loss is how a demo
      becomes a debugging season. `overflow` is STICKY until reset.

  LANES ARE A PERMUTATION   which memory byte of the container arrives on
      which 8-bit video lane depends on the scanout and the receiver;
      `lane_map` fixes the permutation at generation time -- resolved once
      per platform pair with a test pattern, not discovered per frame.

The header's CRC is NOT checked in fabric: the TMDS link either decodes a
byte or does not, and the magic plus fourcc table already reject the
plausible corruptions; hosts holding a raw tap can always run the
reference codec's full verification. The fourcc and phase tables are
imported from the reference codec (`pip install bayerlink`) at GENERATION
time, so the protocol tables have exactly one owner.
"""
from __future__ import annotations


# depth code (3 bits, in hdr_bits order) -> (bits, bytes/emit, samples/emit)
# One emit = one packing group, except 8-bit (three 1-byte groups per cycle
# to outpace the 3-byte/pixel ingest; 16-bit tolerates 1/cycle because the
# byte FIFO holds a whole line).
_ENGINES = {0: (8, 3, 3), 1: (10, 5, 4), 2: (12, 3, 2),
            3: (14, 7, 4), 4: (16, 2, 1)}
_WINDOW = 12                      # bytes; >= 7 (widest group) + 3 (refill)


def bayerlink_in(max_line_bytes: int = 4096,
                 module_name: str = "bayerlink_in",
                 fifo_depth: int = 1024,
                 lane_map: tuple[int, int, int] = (0, 1, 2),
                 vsync_active_high: bool = True) -> dict:
    """Emit the version-2 receiver for one CAPACITY, any geometry or depth.

    Args:
        max_line_bytes: the largest packed camera line the buffers hold.
            A header whose ``width * bits / 8`` exceeds this is refused
            with a sticky code. Sizes the byte FIFO (one full line plus
            slack, in 3-byte words).
        fifo_depth: sample-side elastic, in samples, power of two, at
            least 8. Four banks so a whole emit batch lands in one cycle.
        lane_map: which video lane carries container byte 0, 1, 2. Lane 0
            is ``vid_data[23:16]``, lane 1 ``[15:8]``, lane 2 ``[7:0]``.
        vsync_active_high: polarity of the frame pulse.

    Returns:
        ``{"verilog", "module", "interface", ...}`` -- a self-describing
        source module: no stream input, one 16-bit output stream carrying
        unshifted samples, so ``compose()`` can instantiate it at the head
        of a pipeline. Sticky status: ``overflow``, ``refused`` +
        ``refuse_code``; observability: the last accepted header's
        ``hdr_bits/hdr_width/hdr_height/hdr_phase/hdr_valid``.
    """
    try:
        from bayerlink.protocol import _FOURCC, BAYER_PHASE, MAGIC
    except ImportError as error:                # pragma: no cover
        raise ImportError(
            "generating the bayerlink receiver needs the reference codec's "
            "protocol tables (the fourcc and phase maps have ONE owner): "
            "pip install bayerlink") from error

    if max_line_bytes < 16:
        raise ValueError(f"max_line_bytes {max_line_bytes} is below any "
                         "real camera line")
    if sorted(lane_map) != [0, 1, 2]:
        raise ValueError(
            f"lane_map {lane_map} must be a permutation of (0, 1, 2)")
    if fifo_depth & (fifo_depth - 1) or fifo_depth < 8:
        raise ValueError(f"fifo_depth {fifo_depth} must be a power of two "
                         ">= 8 (four banks, batches of up to 4)")

    def clog2(n: int) -> int:
        return max(1, (n - 1).bit_length())

    # Byte FIFO: 3-byte words, a full line plus slack.
    bwords = 1
    while bwords < (max_line_bytes + 2) // 3 + 8:
        bwords *= 2
    bptrw = clog2(bwords)
    bcntw = clog2(bwords + 1)
    half = fifo_depth // 4                     # per-bank sample capacity
    sptrw = clog2(fifo_depth)
    scntw = clog2(fifo_depth + 1)
    lbw = clog2(max_line_bytes + 1)            # line-byte counters
    vs = "" if vsync_active_high else "~"
    lane_hi = {0: 23, 1: 15, 2: 7}

    bits_of = {code: eng[0] for code, eng in _ENGINES.items()}
    code_of = {b: c for c, b in bits_of.items()}
    group_samples = {8: 1, 10: 4, 12: 2, 14: 4, 16: 1}

    L = []
    a = L.append
    a("// generated by np2hw -- bayerlink v2 receiver: parallel video in,")
    a("// one unshifted sample per clock out (16-bit lane). Depth and")
    a("// geometry are RUN-TIME, owned by each frame's header; this build")
    a(f"// fixes capacity only: max_line_bytes {max_line_bytes}, byte FIFO")
    a(f"// {bwords} words of 3, sample FIFO {fifo_depth}. Container byte k")
    a(f"// on video lane {tuple(lane_map)} (resolved per platform with a")
    a("// test pattern). Header CRC is host business; fabric refuses on")
    a("// magic, fourcc, group alignment and capacity -- sticky, coded.")
    a(f"module {module_name} (")
    a("    input  wire        clk,     // the recovered pixel clock")
    a("    input  wire        rst,")
    a("    // Parallel video from the HDMI/DVI receiver. No ready exists on")
    a("    // this side: video cannot be stalled, only missed.")
    a("    input  wire        vid_de,")
    a("    input  wire        vid_vsync,")
    a("    input  wire [23:0] vid_data,")
    a("    // The elastic stream a generated core consumes. Samples are")
    a("    // unshifted: a 10-bit sample is 0..1023 in the 16-bit lane.")
    a("    output wire        out_valid,")
    a("    input  wire        out_ready,")
    a("    output wire [15:0] out_data,")
    a("    output wire        out_sof,")
    a("    output wire        out_eol,")
    a("    output wire        out_last,")
    a("    // STICKY until reset: unobservable loss is a debugging season.")
    a("    output reg         overflow,")
    a("    output reg         refused,")
    a("    output reg  [2:0]  refuse_code,  // 1 magic, 2 fourcc, 3 group")
    a("                                     // alignment, 4 capacity, 5 zero")
    a("    // The last ACCEPTED header, for hosts to observe.")
    a("    output reg  [4:0]  hdr_bits,")
    a("    output reg  [15:0] hdr_width,")
    a("    output reg  [15:0] hdr_height,")
    a("    output reg  [1:0]  hdr_phase,")
    a("    output reg         hdr_valid")
    a(");")
    a(f"    localparam MAX_LINE_BYTES = {max_line_bytes};")
    a("")
    a("    // ---- position in the display frame ----")
    a("    reg de_q, vs_q;")
    a(f"    wire frame_start = {vs}vid_vsync & ~vs_q;")
    a("    wire line_end    = de_q & ~vid_de;   // DE falling edge")
    a("    reg [15:0] line;                     // display lines since vsync")
    a(f"    reg [{lbw-1}:0] bytecnt;                // payload bytes into the line")
    a("")
    a("    // ---- container bytes, lane permutation undone ----")
    a(f"    wire [7:0] byte0 = vid_data[{lane_hi[lane_map[0]]}:{lane_hi[lane_map[0]]-7}];")
    a(f"    wire [7:0] byte1 = vid_data[{lane_hi[lane_map[1]]}:{lane_hi[lane_map[1]]-7}];")
    a(f"    wire [7:0] byte2 = vid_data[{lane_hi[lane_map[2]]}:{lane_hi[lane_map[2]]-7}];")
    a("")
    a("    // ---- header capture: line 0, first 7 pixels = bytes 0..20 ----")
    a("    reg [7:0] hb [0:22];")
    a("    reg [4:0] hbi;                       // header bytes stored, caps at 21")
    a("    wire hdr_line = (line == 16'd0);")
    a("    // The decode is PIPELINED over three cycles -- the header sits")
    a("    // a full display line ahead of its payload, so the fourcc table")
    a("    // and the capacity compare never share a cycle. Stage 1 latches")
    a("    // the fields, stage 2 the table lookups, stage 3 the verdict.")
    a("    reg [31:0] h_magic, h_fourcc, h_width, h_height;")
    a("")
    a("    // fourcc -> depth code + CFA phase, the reference codec's table")
    a("    // verbatim (one owner, imported at generation time).")
    a("    reg        f_known;")
    a("    reg [2:0]  f_code;")
    a("    reg [1:0]  f_phase;")
    a("    always @* begin")
    a("        f_known = 1'b1; f_code = 3'd0; f_phase = 2'd0;")
    a("        case (h_fourcc)")
    for text, (bits, order) in sorted(_FOURCC.items()):
        code = int.from_bytes(text.encode("ascii"), "little")
        phase = BAYER_PHASE.get(order, 0)
        a(f"            32'h{code:08X}: begin f_code = 3'd{code_of[bits]}; "
          f"f_phase = 2'd{phase}; end // {text!r}: {bits}-bit "
          f"{order or 'mono'}")
    a("            default: f_known = 1'b0;")
    a("        endcase")
    a("    end")
    a("")
    a("    reg [17:0] f_linebytes;   // width * bits / 8")
    a("    always @* case (f_code)")
    a("        3'd0: f_linebytes = {2'b0, h_width[15:0]};                       // w")
    a("        3'd1: f_linebytes = {2'b0, h_width[15:0]} + {4'b0, h_width[15:2]}; // w + w/4")
    a("        3'd2: f_linebytes = {2'b0, h_width[15:0]} + {3'b0, h_width[15:1]}; // w + w/2")
    a("        3'd3: f_linebytes = {1'b0, h_width[15:0], 1'b0} - {4'b0, h_width[15:2]}; // 2w - w/4")
    a("        default: f_linebytes = {1'b0, h_width[15:0], 1'b0};              // 2w")
    a("    endcase")
    a("")
    a("    // group alignment: width % group_samples == 0")
    a("    reg f_aligned;")
    a("    always @* case (f_code)")
    a("        3'd1, 3'd3: f_aligned = (h_width[1:0] == 2'b00);   // groups of 4")
    a("        3'd2:       f_aligned = (h_width[0] == 1'b0);      // pairs")
    a("        default:    f_aligned = 1'b1;                      // bytes/LE16")
    a("    endcase")
    a("")
    a("    // ---- stage-2 registers, then the verdict on registered facts ----")
    a("    reg        d_known, d_aligned;")
    a("    reg [2:0]  d_code;")
    a("    reg [1:0]  d_phase;")
    a("    reg [17:0] d_linebytes;")
    a("    wire [2:0] verdict =")
    a("        (h_magic != 32'h%08X) ? 3'd1 :" % MAGIC)
    a("        (!d_known)            ? 3'd2 :")
    a("        (!d_aligned)          ? 3'd3 :")
    a("        (d_linebytes > MAX_LINE_BYTES || h_width[31:16] != 16'h0")
    a("            || h_height[31:16] != 16'h0) ? 3'd4 :")
    a("        (h_width == 32'd0 || h_height == 32'd0) ? 3'd5 : 3'd0;")
    a("")
    a("    // Pending = this frame's accepted facts; active copies arrive on")
    a("    // the drain side with the frame's first byte, so a stalled tail")
    a("    // of the previous frame still unpacks under ITS OWN header.")
    a("    reg        pend_ok;")
    a("    reg [2:0]  pend_code;")
    a("    reg [1:0]  pend_phase;")
    a("    reg [15:0] pend_width, pend_height;")
    a(f"    reg [{lbw-1}:0] pend_linebytes;")
    a("    reg [1:0] hstage;   // 0 idle/latched, 1 fields held, 2 decoded, 3 done")
    a("")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            de_q <= 0; vs_q <= 0; line <= 16'hFFFF; bytecnt <= 0;")
    a("            hbi <= 0; hstage <= 0; pend_ok <= 0;")
    a("            h_magic <= 0; h_fourcc <= 0; h_width <= 0; h_height <= 0;")
    a("            d_known <= 0; d_aligned <= 0; d_code <= 0; d_phase <= 0;")
    a("            d_linebytes <= 0;")
    a("            refused <= 0; refuse_code <= 0;")
    a("            hdr_valid <= 0; hdr_bits <= 0; hdr_width <= 0;")
    a("            hdr_height <= 0; hdr_phase <= 0;")
    a("        end else begin")
    a("            de_q <= vid_de;")
    a(f"            vs_q <= {vs}vid_vsync;")
    a("            if (vid_de && hdr_line && hbi != 5'd21) begin")
    a("                hb[hbi] <= byte0;")
    a("                hb[hbi + 5'd1] <= byte1;")
    a("                hb[hbi + 5'd2] <= byte2;")
    a("                hbi <= hbi + 5'd3;")
    a("            end")
    a("            if (hdr_line && hbi == 5'd21 && hstage == 2'd0) begin")
    a("                hstage <= 2'd1;   // stage 1: hold the fields")
    a("                h_magic  <= {hb[3],  hb[2],  hb[1],  hb[0]};")
    a("                h_fourcc <= {hb[11], hb[10], hb[9],  hb[8]};")
    a("                h_width  <= {hb[15], hb[14], hb[13], hb[12]};")
    a("                h_height <= {hb[19], hb[18], hb[17], hb[16]};")
    a("            end")
    a("            if (hstage == 2'd1) begin")
    a("                hstage <= 2'd2;   // stage 2: table lookups, held")
    a("                d_known <= f_known; d_code <= f_code;")
    a("                d_phase <= f_phase; d_aligned <= f_aligned;")
    a("                d_linebytes <= f_linebytes;")
    a("            end")
    a("            if (hstage == 2'd2) begin")
    a("                hstage <= 2'd3;   // stage 3: the verdict")
    a("                if (verdict == 3'd0) begin")
    a("                    pend_ok <= 1'b1;")
    a("                    pend_code <= d_code; pend_phase <= d_phase;")
    a("                    pend_width <= h_width[15:0];")
    a("                    pend_height <= h_height[15:0];")
    a(f"                    pend_linebytes <= d_linebytes[{lbw-1}:0];")
    a("                    hdr_valid <= 1'b1;")
    a("                    case (d_code)   // observable, in plain bits")
    a("                        3'd0: hdr_bits <= 5'd8;  3'd1: hdr_bits <= 5'd10;")
    a("                        3'd2: hdr_bits <= 5'd12; 3'd3: hdr_bits <= 5'd14;")
    a("                        default: hdr_bits <= 5'd16;")
    a("                    endcase")
    a("                    hdr_width <= h_width[15:0];")
    a("                    hdr_height <= h_height[15:0];")
    a("                    hdr_phase <= d_phase;")
    a("                end else begin")
    a("                    pend_ok <= 1'b0;")
    a("                    refused <= 1'b1;      // sticky; frame is dropped")
    a("                    refuse_code <= verdict;")
    a("                end")
    a("            end")
    a("            if (line_end) begin")
    a("                bytecnt <= 0;")
    a("                if (line != 16'hFFFF) line <= line + 16'd1;")
    a("            end else if (vid_de && !hdr_line)")
    a(f"                bytecnt <= (bytecnt > MAX_LINE_BYTES - 3) ? bytecnt")
    a("                           : bytecnt + 3'd3;")
    a("            if (frame_start) begin")
    a("                line <= 16'd0; bytecnt <= 0; hbi <= 0; hstage <= 0;")
    a("            end")
    a("        end")
    a("    end")
    a("")
    a("    // ---- payload ingest: 3-byte words with a valid count + frame tag ----")
    a("    wire in_frame = pend_ok && !hdr_line")
    a("                    && line != 16'hFFFF && line <= {1'b0, pend_height};")
    a("    wire payload  = vid_de && in_frame && bytecnt < pend_linebytes;")
    a(f"    wire [{lbw-1}:0] left = pend_linebytes - bytecnt;")
    a("    wire [1:0] vb = (left >= 3) ? 2'd3 : left[1:0];")
    a("    wire ff = (line == 16'd1) && (bytecnt == 0);   // frame's first word")
    a("    // FRAMING DECIDED AT INGEST, the v1 law: position is known")
    a("    // here and only here. le marks each line's final payload word,")
    a("    // fe the frame's; the drain fires eol/last on the TAG, so a")
    a("    // lossy stretch corrupts its own frame and nothing after it.")
    a("    wire le = (left <= 3);")
    a("    wire fe = le && (line == {1'b0, pend_height});")
    a("")
    a("    // The ingest DECISION is registered: the line-bytes compare")
    a("    // never shares a cycle with the FIFO bookkeeping.")
    a("    reg        p_valid, p_ff, p_le, p_fe;")
    a("    reg [1:0]  p_vb;")
    a("    reg [23:0] p_b;")
    a(f"    reg [28:0] bfifo [0:{bwords-1}];   // {{fe, le, ff, vb, b0, b1, b2}}")
    a(f"    reg [{bptrw-1}:0] bwr, brd;")
    a(f"    reg [{bcntw-1}:0] bcount;")
    a(f"    wire bspace = (bcount != {bcntw}'d{bwords});")
    a("    wire bwr_en = p_valid && bspace;")
    a("")
    a("    // ---- drain: window of bytes, one packing group per cycle ----")
    a(f"    reg [7:0] win [0:{_WINDOW-1}];      // win[0] is oldest")
    a("    reg [3:0] wcnt;")
    a("    // Active facts travel with the frame tag, not the wire clock:")
    a("    reg [2:0]  act_code;")
    a("    reg [15:0] act_width, act_height;")
    a("    reg        act_valid;")
    a("    // Position: `rem` (samples left in the line) exists ONLY for")
    a("    // the 8-bit clamp and resyncs at every tagged line end; the")
    a("    // FLAGS come from the ingest tags, never from delivered-sample")
    a("    // counts -- loss corrupts its own frame and nothing after it.")
    a("    reg [15:0] rem;")
    a("    reg        sof_p;")
    a("    reg        eol_pend, fe_pend;   // a tagged line end is in the window")
    a("    reg [3:0]  eol_mark;            // bytes from window head to it")
    a("    reg [3:0]  need_r;              // registered shift: min(mark, be)")
    a("")
    a("    // The byte FIFO reads REGISTERED (block RAM, not a 2048-deep")
    a("    // LUT mux): bdout always holds the word at brd, one cycle after")
    a("    // brd settles; bprimed covers the latency and the same-address")
    a("    // write-through corner by simply waiting a cycle.")
    a("    reg [28:0] bdout;")
    a("    reg        bprimed;")
    a("    wire [28:0] bhead = bdout;")
    a("    // A frame tag waits for the window to drain, then swaps geometry.")
    a("    // One-word SKID between the BRAM and the window: the BRAM's")
    a("    // slow output lands only in registers, and every control")
    a("    // decision reads registers, never the RAM.")
    a("    reg        sk_valid, sk_ff, sk_le, sk_fe;")
    a("    reg [1:0]  sk_vb;")
    a("    reg [23:0] sk_b;")
    a("    wire sk_take = sk_valid")
    a(f"               && (wcnt <= 4'd{_WINDOW-3})")
    a("               && (!sk_ff || wcnt == 4'd0)")
    a("               && (!sk_le || !eol_pend);")
    a("    wire pop = bprimed && (bcount != 0) && (!sk_valid || sk_take);")
    a("    // A lossy frame can strand a stub: leftover window bytes too")
    a("    // few to emit, with the next frame's barrier waiting for an")
    a("    // empty window that can never come. The stub belongs to a dead")
    a("    // frame -- flush it, and the barrier opens.")
    a("    wire stub_flush = sk_valid && sk_ff && (wcnt != 4'd0)")
    a("                      && ({1'b0, wcnt} < {1'b0, need_r});")
    a(f"    wire [{bptrw-1}:0] brd_n = pop ? brd + {bptrw}'d1 : brd;")
    a("")
    a("    function [2:0] eng_be; input [2:0] c;")
    a("        case (c)")
    a("            3'd0: eng_be = 3'd3;")
    a("            3'd1: eng_be = 3'd5;")
    a("            3'd2: eng_be = 3'd3;")
    a("            3'd3: eng_be = 3'd7;")
    a("            3'd4: eng_be = 3'd2;")
    a("            default: eng_be = 3'd3;")
    a("        endcase")
    a("    endfunction")
    a("    function [2:0] eng_se; input [2:0] c;")
    a("        case (c)")
    a("            3'd0: eng_se = 3'd3;")
    a("            3'd1: eng_se = 3'd4;")
    a("            3'd2: eng_se = 3'd2;")
    a("            3'd3: eng_se = 3'd4;")
    a("            3'd4: eng_se = 3'd1;")
    a("            default: eng_se = 3'd2;")
    a("        endcase")
    a("    endfunction")
    a("    reg [2:0] be; reg [2:0] se;         // bytes, samples per emit")
    a("    always @* case (act_code)")
    for code, (bits, be_, se_) in sorted(_ENGINES.items()):
        a(f"        3'd{code}: begin be = 3'd{be_}; se = 3'd{se_}; end // {bits}-bit")
    a("        default: begin be = 3'd3; se = 3'd2; end")
    a("    endcase")
    a("    // 8-bit is the one engine whose emit batch is not a packing")
    a("    // group (three 1-byte groups per cycle, to outpace ingest), so")
    a("    // it alone may meet a line boundary mid-batch: clamp it there,")
    a("    // and the whole-groups-per-line invariant holds for every depth.")
    a("    // The clamp is REGISTERED: both writers of `rem` know its next")
    a("    // value, so the compare rides their paths, not the emit loop's.")
    a("    reg [2:0] be_eff, se_eff;")
    a("    function [2:0] clamp8;")
    a("        input [2:0] code; input [15:0] r; input [2:0] full;")
    a("        clamp8 = (code == 3'd0 && r[15:2] == 14'd0")
    a("                  && r[1:0] != 2'd3 && r != 16'd0)")
    a("                 ? {1'b0, r[1:0]} : full;")
    a("    endfunction")
    a("")
    a(f"    reg [{sptrw-1}:0] swr, srd;")
    a("    // Two counters, one truth each: RESERVED (bumped when a batch is")
    a("    // extracted, gates emission) and AVAILABLE (bumped a cycle later")
    a("    // when the banks are physically written, gates the reader) --")
    a("    // the bank write is its own pipeline stage, so the rotation mux")
    a("    // never shares a cycle with the sample extraction.")
    a(f"    reg [{scntw-1}:0] scount_res, scount_avail;")
    a(f"    wire sroom = (scount_res <= {scntw}'d{fifo_depth - 8});")
    a("    wire emit = act_valid && ({1'b0, wcnt} >= {1'b0, need_r}) && sroom;")
    a("")
    a("    // Samples of the head group, every depth in parallel; act_code picks.")
    a("    wire [15:0] smp0, smp1, smp2, smp3;")
    a("    reg  [15:0] s0m, s1m, s2m, s3m;")
    a("    always @* case (act_code)")
    a("        3'd0: begin                                        // 8-bit")
    a("            s0m = {8'b0, win[0]}; s1m = {8'b0, win[1]};")
    a("            s2m = {8'b0, win[2]}; s3m = 16'd0;")
    a("        end")
    a("        3'd1: begin                                        // 10P")
    a("            s0m = {6'b0, win[0], win[4][1:0]};")
    a("            s1m = {6'b0, win[1], win[4][3:2]};")
    a("            s2m = {6'b0, win[2], win[4][5:4]};")
    a("            s3m = {6'b0, win[3], win[4][7:6]};")
    a("        end")
    a("        3'd2: begin                                        // 12P")
    a("            s0m = {4'b0, win[0], win[2][3:0]};")
    a("            s1m = {4'b0, win[1], win[2][7:4]};")
    a("            s2m = 16'd0; s3m = 16'd0;")
    a("        end")
    a("        3'd3: begin                                        // 14P")
    a("            s0m = {2'b0, win[0], win[4][5:0]};")
    a("            s1m = {2'b0, win[1], win[5][3:0], win[4][7:6]};")
    a("            s2m = {2'b0, win[2], win[6][1:0], win[5][7:4]};")
    a("            s3m = {2'b0, win[3], win[6][7:2]};")
    a("        end")
    a("        default: begin                                     // 16 LE")
    a("            s0m = {win[1], win[0]};")
    a("            s1m = 16'd0; s2m = 16'd0; s3m = 16'd0;")
    a("        end")
    a("    endcase")
    a("    assign smp0 = s0m; assign smp1 = s1m;")
    a("    assign smp2 = s2m; assign smp3 = s3m;")
    a("")
    a("    // Flags: an emit batch is whole packing groups, and a line is")
    a("    // whole groups too, so a batch never straddles a line boundary --")
    a("    // eol can only be the batch's LAST sample, sof only its first.")
    a("    wire batch_eol  = eol_pend && (eol_mark <= {1'b0, be_eff});")
    a("    wire batch_sof  = sof_p;")
    a("    wire batch_last = batch_eol && fe_pend;")
    a("    wire [15:0] rem_next = batch_eol ? act_width")
    a("                                     : rem - {13'b0, se_eff};")

    a("")
    a(f"    reg [18:0] sbank0 [0:{half-1}];   // {{last, eol, sof, data}}")
    a(f"    reg [18:0] sbank1 [0:{half-1}];")
    a(f"    reg [18:0] sbank2 [0:{half-1}];")
    a(f"    reg [18:0] sbank3 [0:{half-1}];")
    a("    // The extracted batch, registered: stage B writes the banks.")
    a("    reg        q_valid;")
    a("    reg [2:0]  q_se;")
    a(f"    reg [{sptrw-1}:0] q_swr;")
    a("    reg        q_sof, q_eol, q_last;")
    a("    reg [15:0] q_smp0, q_smp1, q_smp2, q_smp3;")
    a("")
    a("    // window shift by be, then refill from the popped word")
    a("    integer k;")
    a("    reg [7:0] wnext [0:%d];" % (_WINDOW - 1))
    a("    reg [3:0] cnt_s;")

    a("    always @* begin")
    a("        for (k = 0; k < %d; k = k + 1)" % _WINDOW)
    a("            wnext[k] = emit ? ((k + need_r < %d) ? win[k + need_r] : 8'd0)"
      % _WINDOW)
    a("                            : win[k];")
    a("        cnt_s = emit ? (wcnt - need_r) : wcnt;")
    a("    end")
    a("")
    a("    // next-values feeding need_r's register (mirrors the updates)")
    a("    wire nr_ins  = sk_take && sk_le;")
    a("    wire nr_pend = nr_ins ? 1'b1 : (emit && batch_eol ? 1'b0 : eol_pend);")
    a("    wire [3:0] nr_mark = nr_ins ? (cnt_s + {2'b0, sk_vb})")
    a("                       : ((emit && eol_pend) ? eol_mark - need_r")
    a("                                             : eol_mark);")
    a("    wire [2:0] nr_be = (sk_take && sk_ff)")
    a("                       ? clamp8(pend_code, pend_width, eng_be(pend_code))")
    a("                       : (emit ? clamp8(act_code, rem_next, be) : be_eff);")

    a("    // Per-slot addresses at POINTER width: a batch that straddles")
    a("    // the wrap must wrap with it, not run off the end of the banks.")
    for j in range(4):
        a(f"    wire [{sptrw-1}:0] q_adr{j} = q_swr + {sptrw}'d{j};")
    a("    always @(posedge clk) begin")
    a("        if (rst) begin")
    a("            bwr <= 0; brd <= 0; bcount <= 0; overflow <= 0;")
    a("            bprimed <= 0; bdout <= 0;")
    a("            p_valid <= 0; p_ff <= 0; p_le <= 0; p_fe <= 0;")
    a("            p_vb <= 0; p_b <= 0;")
    a("            sk_valid <= 0; sk_ff <= 0; sk_le <= 0; sk_fe <= 0;")
    a("            sk_vb <= 0; sk_b <= 0;")
    a("            wcnt <= 0; swr <= 0; srd <= 0;")
    a("            scount_res <= 0; scount_avail <= 0; q_valid <= 0;")
    a("            q_se <= 0; q_swr <= 0; q_sof <= 0; q_eol <= 0;")
    a("            q_last <= 0; q_smp0 <= 0; q_smp1 <= 0; q_smp2 <= 0;")
    a("            q_smp3 <= 0;")
    a("            act_valid <= 0; act_code <= 0; act_width <= 0;")
    a("            act_height <= 0; rem <= 0; sof_p <= 0;")
    a("            be_eff <= 3'd3; se_eff <= 3'd3; need_r <= 4'd3;")
    a("            eol_pend <= 0; fe_pend <= 0; eol_mark <= 0;")
    a("            for (k = 0; k < %d; k = k + 1) win[k] <= 8'd0;" % _WINDOW)
    a("        end else begin")
    a("            // ingest side, one registered stage after the compare")
    a("            p_valid <= payload;")
    a("            p_ff <= ff; p_le <= le; p_fe <= fe; p_vb <= vb;")
    a("            p_b <= {byte0, byte1, byte2};")
    a("            if (bwr_en) begin")
    a("                bfifo[bwr] <= {p_fe, p_le, p_ff, p_vb, p_b};")
    a(f"                bwr <= bwr + {bptrw}'d1;")
    a("            end")
    a("            if (p_valid && !bspace) overflow <= 1'b1;  // sticky loss")
    a("            // registered FIFO head: next cycle bdout matches brd")
    a("            bdout <= bfifo[brd_n];")
    a(f"            bprimed <= ((bcount + (bwr_en ? {bcntw}'d1 : {bcntw}'d0)")
    a(f"                         - (pop ? {bcntw}'d1 : {bcntw}'d0)) != 0)")
    a("                       && !(bwr_en && (bwr == brd_n));")
    a("")
    a("            // drain side: shift, then insert the popped word's bytes")
    a("            for (k = 0; k < %d; k = k + 1) win[k] <= wnext[k];" % _WINDOW)
    a("            wcnt <= cnt_s;")
    a("            if (pop) begin")
    a("                sk_valid <= 1'b1;")
    a("                sk_fe <= bhead[28];")
    a("                sk_le <= bhead[27];")
    a("                sk_ff <= bhead[26];")
    a("                sk_vb <= bhead[25:24];")
    a("                sk_b  <= bhead[23:0];")
    a("                brd <= brd_n;")
    a("            end else if (sk_take)")
    a("                sk_valid <= 1'b0;")
    a("            if (sk_take) begin")
    a("                if (sk_ff) begin     // the frame's facts arrive HERE")
    a("                    act_valid <= 1'b1;")
    a("                    act_code <= pend_code;")
    a("                    act_width <= pend_width;")
    a("                    act_height <= pend_height;")
    a("                    rem <= pend_width;")
    a("                    sof_p <= 1'b1;")
    a("                    be_eff <= clamp8(pend_code, pend_width,")
    a("                                     eng_be(pend_code));")
    a("                    se_eff <= clamp8(pend_code, pend_width,")
    a("                                     eng_se(pend_code));")
    a("                end")
    a("                // bytes land at the post-shift count")
    a("                if (sk_vb >= 2'd1) win[cnt_s + 4'd0] <= sk_b[23:16];")
    a("                if (sk_vb >= 2'd2) win[cnt_s + 4'd1] <= sk_b[15:8];")
    a("                if (sk_vb == 2'd3) win[cnt_s + 4'd2] <= sk_b[7:0];")
    a("                wcnt <= cnt_s + {2'b0, sk_vb};")
    a("            end")
    a(f"            bcount <= bcount + (bwr_en ? {bcntw}'d1 : {bcntw}'d0)")
    a(f"                             - (pop ? {bcntw}'d1 : {bcntw}'d0);")
    a("")
    a("            // stage A: extract the batch and reserve its slots")
    a("            q_valid <= emit;")
    a("            if (emit) begin")
    a("                q_se <= se_eff; q_swr <= swr;")
    a("                q_sof <= batch_sof; q_eol <= batch_eol;")
    a("                q_last <= batch_last;")
    a("                q_smp0 <= smp0; q_smp1 <= smp1;")
    a("                q_smp2 <= smp2; q_smp3 <= smp3;")
    a("                swr <= swr + {%d'b0, se_eff};" % (sptrw - 3))
    a("                sof_p <= 1'b0;")
    a("                rem <= rem_next;")
    a("                be_eff <= clamp8(act_code, rem_next, be);")
    a("                se_eff <= clamp8(act_code, rem_next, se);")
    a("                if (batch_eol) begin")
    a("                    eol_pend <= 1'b0; fe_pend <= 1'b0;")
    a("                end else if (eol_pend)")
    a("                    eol_mark <= eol_mark - need_r;")
    a("            end")
    a("            if (sk_take && sk_le) begin")
    a("                eol_pend <= 1'b1;")
    a("                fe_pend <= sk_fe;")
    a("                eol_mark <= cnt_s + {2'b0, sk_vb};")
    a("            end")
    a("            if (stub_flush) begin")
    a("                wcnt <= 4'd0;")
    a("                eol_pend <= 1'b0; fe_pend <= 1'b0;")
    a("            end")
    a("            // registered shift: min(next mark, next be), from the")
    a("            // same next-values its sources register this cycle")
    a("            // the SELECT compares registers only (be_eff, not its")
    a("            // next): at ff-loads pend is provably clear, and the")
    a("            // 8-bit clamp coincides with the tag on intact streams,")
    a("            // so the one-cycle skew never picks a wrong minimum.")
    a("            need_r <= nr_pend && ({1'b0, nr_mark} < {2'b0, be_eff})")
    a("                      ? nr_mark : {1'b0, nr_be};")
    a("            // stage B: the physical bank writes, from registers only")
    a("            if (q_valid) begin")
    a("                case (q_swr[1:0])")
    for rot in range(4):
        banks = [(rot + j) % 4 for j in range(4)]
        a(f"                    2'd{rot}: begin")
        for j in range(4):
            b = banks[j]
            a(f"                        if (q_se > 3'd{j}) sbank{b}"
              f"[q_adr{j} >> 2] <= {{")
            a(f"                            (q_se == 3'd{j+1}) ? q_last : 1'b0,")
            a(f"                            (q_se == 3'd{j+1}) ? q_eol  : 1'b0,")
            first = "q_sof" if j == 0 else "1'b0"
            a(f"                            {first}, q_smp{j}}};")
        a("                    end")
    a("                endcase")
    a("            end")
    a("            if (out_valid && out_ready) srd <= srd + %d'd1;" % sptrw)
    a("            scount_res <= scount_res + (emit ? {%d'b0, se_eff} : %d'd0)"
      % (scntw - 3, scntw))
    a("                             - ((out_valid && out_ready) ? %d'd1 : %d'd0);"
      % (scntw, scntw))
    a("            scount_avail <= scount_avail + (q_valid ? {%d'b0, q_se} : %d'd0)"
      % (scntw - 3, scntw))
    a("                             - ((out_valid && out_ready) ? %d'd1 : %d'd0);"
      % (scntw, scntw))
    a("        end")
    a("    end")
    a("")
    a("    reg [18:0] shead;")
    a("    always @* case (srd[1:0])")
    a("        2'd0: shead = sbank0[srd >> 2];")
    a("        2'd1: shead = sbank1[srd >> 2];")
    a("        2'd2: shead = sbank2[srd >> 2];")
    a("        default: shead = sbank3[srd >> 2];")
    a("    endcase")
    a("    assign out_valid = (scount_avail != 0);")
    a("    assign {out_last, out_eol, out_sof, out_data} = shead;")
    a("endmodule")

    interface = {
        "clock": "clk",
        "reset": "rst",
        "param_prefix": "param_",
        # A SOURCE: no stream input (the video side is not an np2hw stream,
        # and cannot carry ready), one stream output.
        "input": None,
        "output": {"prefix": "out", "data_bits": 16,
                   "flags": ("sof", "eol", "last")},
        "streams": [{"prefix": "out", "direction": "out", "data_bits": 16,
                     "flags": ("sof", "eol", "last"), "domain": ""}],
        "params": [],
    }
    return {"verilog": "\n".join(L), "module": module_name,
            "interface": interface, "max_line_bytes": max_line_bytes,
            "fifo_depth": fifo_depth, "lane_map": tuple(lane_map),
            "out_bits": 16, "depths": tuple(sorted(group_samples))}
