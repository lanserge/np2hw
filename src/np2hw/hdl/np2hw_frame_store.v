// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_store -- a stream's recent frames, delivered together at the
// pace their consumer takes them.
//
// A model that reads across time -- a time-of-flight sensor's four
// reference phases, a temporal filter's frames -- needs the same pixel from
// several frames at once. Where frames arrive as their source sends them
// and leave as a display scans them, the two cadences owe each other
// nothing: a sensor may send four frames in a burst and then nothing, and a
// display reads at its own rate. This store stands between them. Every
// frame that comes in is kept, in a ring of SLOTS, and the consumer is
// handed the newest FRAMES whole ones, side by side, as often as it asks.
//
// THE INPUT NEVER WAITS FOR THE OUTPUT. A frame is written into the next
// slot of the ring as it arrives; the only back-pressure on the input is
// the memory's own (wr_ready). What the consumer reads is chosen when its
// frame starts -- the newest FRAMES frames that are whole by then, in age
// order -- and held for the frame, so every frame out is the same FRAMES
// frames from its first pixel to its last. A slow consumer repeats a frame
// of frames, a fast source skips some; neither tears one.
//
// SLOTS - FRAMES IS THE SLACK. While the consumer reads one frame of
// frames, the source may begin SLOTS - FRAMES frames past the newest whole
// one -- the frames being written, and those written but not yet whole
// (external memory's writes in flight), both hold a slot -- before it
// reaches a slot being read. A source that gets there anyway is never
// stopped -- a sensor cannot be paused -- and `overrun` says so, sticky
// until reset, whether the writer reached a frame being read or a read
// began on a frame already being overwritten: that frame out may hold rows
// of two frames.
//
// WHAT EACH FRAME IS rides with it. A frame's CTX_BITS of context -- the
// stream's CTX member, whatever its facts, carried whole and never read --
// is latched as its first beat goes in, and its size is the one its own
// beats draw, its width up to the first eol and its rows up to its last:
// both are published with the frame of frames that is read out, the
// newest's, for it is what the frame out is computed AT. `out_source` counts the frame
// out came from, the input's frames counted from reset, so a reader can
// tell a repeat from a new one.
//
// HISTORY IS NOT TRUSTWORTHY UNTIL IT EXISTS. Nothing goes out until
// FRAMES frames are whole, and two things start the history over: a frame
// cut short -- a start arriving before the last frame's last beat -- for
// the ring would otherwise hand out frames that were never consecutive,
// and a frame of another size than the one before, for frames of two sizes
// are not one history.
//
// A RESET IS THE STREAM'S. `rst` starts the store over -- no history, the
// count from zero, the read in progress abandoned -- whenever its stream
// does, a link lost and found among them, while the memory behind the seam
// may be in the middle of answering: the binding finishes what it has
// promised its memory, and a frame it answers whole afterwards is of the
// history before, so it never counts, and the ring goes on from the slot it
// had reached, never back over one still being answered.
//
// THE MEMORY IS BEHIND A SEAM. The frames live wherever the binding on the
// seam ports keeps them -- block RAM (np2hw_frame_mem_bram), or external
// memory -- under a sequential contract, since frames are only ever written
// and read in raster order:
//
//   write  wr_valid/wr_ready beats, wr_sof on a frame's first, wr_last on
//          its last, wr_slot the slot it goes to. The binding answers
//          wr_done with wr_done_slot once the frame's every beat is where a
//          read will find it -- for external memory, once the writes are
//          acknowledged, not merely issued.
//   read   rd_start, with rd_slots (FRAMES slot indices, newest in the low
//          bits) and rd_pixels: then rd_pixels beats on rd_valid/rd_ready,
//          each FRAMES samples, newest in the low bits.
module np2hw_frame_store #(
    parameter BITS = 12,                // one sample
    parameter FRAMES = 4,               // frames delivered together
    parameter SLOTS = 8,                // frames kept, FRAMES + slack
    parameter CTX_BITS = 1,             // context carried with each frame
    parameter SB = $clog2(SLOTS)        // bits of a slot index
) (
    input  wire                     clk,
    input  wire                     rst,
    // the frames, as their source sends them
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [BITS-1:0]          in_data,
    input  wire                     in_sof,
    input  wire                     in_eol,
    input  wire                     in_last,
    input  wire [CTX_BITS-1:0]      in_ctx,
    // the newest FRAMES frames, as the consumer takes them: channel k is
    // the frame k before the newest
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [FRAMES*BITS-1:0]   out_data,
    output wire                     out_sof,
    output wire                     out_eol,
    output wire                     out_last,
    output wire [CTX_BITS-1:0]      out_ctx,
    // status
    output wire [15:0]              out_source,
    output wire                     cold,
    output reg                      overrun = 1'b0,
    // the seam: write
    output wire                     wr_valid,
    input  wire                     wr_ready,
    output wire [BITS-1:0]          wr_data,
    output wire                     wr_sof,
    output wire                     wr_last,
    output wire [SB-1:0]            wr_slot,
    input  wire                     wr_done,
    input  wire [SB-1:0]            wr_done_slot,
    // the seam: read
    output reg                      rd_start = 1'b0,
    output wire [FRAMES*SB-1:0]     rd_slots,
    output wire [31:0]              rd_pixels,
    input  wire                     rd_valid,
    output wire                     rd_ready,
    input  wire [FRAMES*BITS-1:0]   rd_data
);
    // Every beat in through a skid, so all that follows starts at a flop
    // whatever drives the store -- an entry whose framing is decided
    // combinationally from its counters, say -- and the input's ready is
    // a register too.
    wire                b_valid, b_ready, b_sof, b_eol, b_last;
    wire [BITS-1:0]     b_data;
    wire [CTX_BITS-1:0] b_ctx;
    np2hw_skid #(.W(CTX_BITS + BITS + 3)) u_in (
        .clk(clk), .rst(rst),
        .s_data({in_ctx, in_last, in_eol, in_sof, in_data}),
        .s_valid(in_valid), .s_ready(in_ready),
        .m_data({b_ctx, b_last, b_eol, b_sof, b_data}),
        .m_valid(b_valid), .m_ready(b_ready));

    // ---- what each slot holds: its frame's size, context and number
    reg [31:0]          size_of [0:SLOTS-1];
    reg [CTX_BITS-1:0]  ctx_of  [0:SLOTS-1];
    reg [15:0]          num_of  [0:SLOTS-1];
    reg [31:0]          beats_of [0:SLOTS-1];   // its samples, counted

    // ---- write: every frame into the next slot of the ring
    reg [SB-1:0] next = {SB{1'b0}};     // the slot the next frame takes
    reg [SB-1:0] wslot = {SB{1'b0}};    // the slot being written
    reg          writing = 1'b0;        // a frame begun, its last not yet in
    reg [15:0]   count = 16'd0;         // frames begun since reset
    reg [31:0]   last_size = 32'd0;     // the frame before's
    // one past the beat's place in its row, and one past the rows ended
    // in the frame, so the size a beat draws is a choice between
    // registers; the width once a row ends
    reg [15:0]   in_x1 = 16'd1, in_y1 = 16'd1;
    reg [31:0]   in_n1 = 32'd1;         // one past the beats of the frame
    reg [15:0]   in_w = 16'd0;
    reg          rowed = 1'b0;          // a row of the frame has ended
    wire take = b_valid && b_ready;
    // the frame's size as its last beat draws it: {rows, width}
    wire [15:0] beat_x1 = b_sof ? 16'd1 : in_x1;
    wire [15:0] beat_y1 = b_sof ? 16'd1 : in_y1;
    wire        beat_rowed = !b_sof && rowed;
    wire [31:0] drawn = {beat_y1, beat_rowed ? in_w : beat_x1};
    assign b_ready = wr_ready;
    assign wr_valid = b_valid;
    assign wr_data  = b_data;
    assign wr_sof   = b_sof;
    assign wr_last  = b_last;
    assign wr_slot  = b_sof ? next : wslot;

    // ---- the frames whole: the newest, and how many in a row are
    reg [SB-1:0] newest = {SB{1'b0}};
    reg [7:0]    whole = 8'd0;          // up to FRAMES
    assign cold = whole < FRAMES;

    // ---- read: the frame of frames the consumer is taking
    reg          reading = 1'b0;
    reg [SB-1:0] r_new = {SB{1'b0}};
    reg [15:0]   r_num = 16'd0;
    reg [31:0]   r_beats = 32'd0;
    reg [15:0]   r_w1 = 16'd0, r_h1 = 16'd0;   // one short of each
    reg [CTX_BITS-1:0] r_ctx = {CTX_BITS{1'b0}};
    reg [15:0]   x = 16'd0, y = 16'd0;

    // the slots a frame k back from `n` sits in
    function [SB-1:0] back(input [SB-1:0] n, input integer k);
        back = (n >= k) ? n - k : n + SLOTS - k;
    endfunction
    genvar g;
    generate for (g = 0; g < FRAMES; g = g + 1) begin : slots
        assign rd_slots[g*SB +: SB] = back(r_new, g);
    end endgenerate
    // whether a slot is one a read in progress is taking
    function read_holds(input [SB-1:0] s);
        integer k;
        begin
            read_holds = 1'b0;
            for (k = 0; k < FRAMES; k = k + 1)
                if (reading && back(r_new, k) == s) read_holds = 1'b1;
        end
    endfunction

    // A read starts as soon as there is history and nothing is being
    // read; the frames begun past the newest whole one must not reach the
    // oldest it takes
    wire starts = !rst && !reading && !cold;
    wire behind = starts &&
        ((wslot >= newest) ? wslot - newest : wslot + SLOTS - newest)
            > SLOTS - FRAMES;

    // A history is one run of consecutive frames of one size: its EPOCH
    // tags every slot written in it, so a frame the binding finishes after
    // a later one has started the history over is never counted in it. A
    // frame cut short starts it over as the next one begins; a frame of
    // another size, as its own last beat shows it -- before the binding
    // can say the frame is whole, which is a cycle after its last beat at
    // the soonest -- and it is the new history's first.
    reg [7:0] epoch = 8'd0;
    reg [7:0] epoch1 = 8'd1, epoch2 = 8'd2;     // the two after it
    reg [7:0] epoch_of [0:SLOTS-1];
    wire      begins = take && b_sof;
    wire      ends = take && b_last;
    wire      cut = begins && writing;
    wire      resized = ends && drawn != last_size;
    wire      restart = cut || resized;
    wire [7:0] epoch_next = (cut && resized) ? epoch2
                          : (cut || resized) ? epoch1 : epoch;
    wire      counts = wr_done && epoch_of[wr_done_slot] == epoch;
    always @(posedge clk) begin
        if (rst) begin
            // the ring keeps its place: a frame written before the reset
            // may yet be answered whole, and its slot is behind the next
            writing <= 1'b0;
            count <= 16'd0; whole <= 8'd0; overrun <= 1'b0;
            last_size <= 32'd0;
            epoch <= epoch1; epoch1 <= epoch2; epoch2 <= epoch2 + 8'd1;
            in_x1 <= 16'd1; in_y1 <= 16'd1; in_n1 <= 32'd1;
            in_w <= 16'd0; rowed <= 1'b0;
        end else begin
            // the source reaching a frame being read, or a read starting
            // on frames it has reached: one driver for the flag
            if ((begins && read_holds(next)) || behind) overrun <= 1'b1;
            if (begins) begin
                ctx_of[next] <= b_ctx;
                num_of[next] <= count;
                count <= count + 16'd1;
                wslot <= next;
                next <= (next == SLOTS - 1) ? {SB{1'b0}} : next + 1'b1;
            end
            // the slot the beat goes to is tagged with the epoch the beat
            // leaves: the one it begins, or the one its frame's size starts
            if (begins || resized)
                epoch_of[wr_slot] <= epoch_next;
            epoch <= epoch_next;
            if (cut && resized) begin
                epoch1 <= epoch2 + 8'd1; epoch2 <= epoch2 + 8'd2;
            end else if (cut || resized) begin
                epoch1 <= epoch2; epoch2 <= epoch2 + 8'd1;
            end
            if (ends) begin
                last_size <= drawn;
                size_of[wr_slot] <= drawn;
                beats_of[wr_slot] <= b_sof ? 32'd1 : in_n1;
            end
            if (take) begin
                writing <= !b_last;
                in_n1 <= (b_sof ? 32'd1 : in_n1) + 32'd1;
                if (b_eol) begin
                    in_x1 <= 16'd1; in_y1 <= beat_y1 + 16'd1;
                    if (!beat_rowed) in_w <= beat_x1;
                    rowed <= 1'b1;
                end else begin
                    in_x1 <= beat_x1 + 16'd1; in_y1 <= beat_y1;
                    rowed <= beat_rowed;
                end
            end
            // a frame is whole once the binding says every beat of it is
            // where a read will find it, and counts if it is of this history
            if (restart) begin
                whole <= 8'd0;
            end else if (counts) begin
                newest <= wr_done_slot;
                if (whole < FRAMES) whole <= whole + 8'd1;
            end
        end
    end

    // A frame of frames starts as soon as there is history and nothing is
    // being read: the consumer's taking paces it, so a consumer that waits
    // between frames gets the newest frames as they are when it is ready.
    wire beat = out_valid && out_ready;
    always @(posedge clk) begin
        rd_start <= 1'b0;
        if (rst) begin
            reading <= 1'b0;
        end else if (starts) begin
            reading <= 1'b1;
            rd_start <= 1'b1;
            r_new <= newest;
            r_ctx <= ctx_of[newest];
            r_num <= num_of[newest];
            r_beats <= beats_of[newest];
            r_w1 <= size_of[newest][15:0] - 16'd1;
            r_h1 <= size_of[newest][31:16] - 16'd1;
            x <= 16'd0; y <= 16'd0;
        end else if (beat) begin
            if (out_last) begin
                reading <= 1'b0;
            end else if (x == r_w1) begin
                x <= 16'd0; y <= y + 16'd1;
            end else begin
                x <= x + 16'd1;
            end
        end
    end
    assign rd_pixels = r_beats;
    // the beats are offered only once the read has been asked for
    wire live = reading && !rd_start;
    assign out_valid = live && rd_valid;
    assign rd_ready  = live && out_ready;
    assign out_data  = rd_data;
    assign out_sof   = x == 16'd0 && y == 16'd0;
    assign out_eol   = x == r_w1;
    assign out_last  = out_eol && y == r_h1;
    assign out_ctx    = r_ctx;
    assign out_source = r_num;
endmodule
