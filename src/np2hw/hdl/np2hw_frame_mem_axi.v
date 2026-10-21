// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_mem_axi -- np2hw_frame_store's frames, kept in external
// memory over AXI4.
//
// The external binding of the store's seam. Each slot is a region of
// memory from BASE, SLOT_BYTES apart -- a frame's samples, each in the
// smallest of one, two or four bytes that holds it, rounded up to 4K so a
// slot starts on a page -- written and read as 64-bit words, the frame's
// samples packed into them in order, the first in the low bits.
//
// WRITES. Samples are packed into words, and words queue until a burst can
// go: sixteen of them, or the rest of a frame. A burst never crosses a
// frame's end, so it never crosses a slot, and since a frame starts on a
// page and its bursts start sixteen words apart, never a 4K boundary
// either -- which AXI forbids and an interconnect is entitled to mangle
// rather than report. A burst's address goes only once its words are all
// queued, so its data never waits. A frame is WHOLE once the response to
// its last burst is back -- the writes landed, not merely issued -- and
// wr_done says so then, with its slot: a read of it from then on finds
// every sample.
//
// READS. A read takes FRAMES frames at one position together. Each frame
// read has a queue of its own; bursts are asked for round the frames in
// turn, each only while its queue has room for the whole burst on top of
// what it holds and what is already asked for, so rready never drops -- an
// AXI read cannot be paused politely. A beat goes out once every queue
// holds the word at its position, each frame's sample of it side by side.
//
// One ID, so responses come in the order they were asked for, which is how
// each read beat finds its queue. Bursts are sixteen beats at most, which a
// Zynq's AXI3 ports take as they are.
module np2hw_frame_mem_axi #(
    parameter BITS = 12,
    parameter FRAMES = 4,
    parameter SLOTS = 8,
    parameter PIXELS = 43200,
    parameter BASE = 32'h0,
    parameter SB = $clog2(SLOTS)
) (
    input  wire                     clk,
    input  wire                     rst,
    // the seam
    input  wire                     wr_valid,
    output wire                     wr_ready,
    input  wire [BITS-1:0]          wr_data,
    input  wire                     wr_sof,
    input  wire                     wr_last,
    input  wire [SB-1:0]            wr_slot,
    output reg                      wr_done = 1'b0,
    output reg  [SB-1:0]            wr_done_slot = {SB{1'b0}},
    input  wire                     rd_start,
    input  wire [FRAMES*SB-1:0]     rd_slots,
    input  wire [31:0]              rd_pixels,
    output wire                     rd_valid,
    input  wire                     rd_ready,
    output wire [FRAMES*BITS-1:0]   rd_data,
    // the memory
    output wire [31:0]              mem_awaddr,
    output wire [7:0]               mem_awlen,
    output wire [2:0]               mem_awsize,
    output wire [1:0]               mem_awburst,
    output wire [3:0]               mem_awcache,
    output wire [2:0]               mem_awprot,
    output reg                      mem_awvalid = 1'b0,
    input  wire                     mem_awready,
    output wire [63:0]              mem_wdata,
    output wire [7:0]               mem_wstrb,
    output wire                     mem_wlast,
    output wire                     mem_wvalid,
    input  wire                     mem_wready,
    input  wire [1:0]               mem_bresp,
    input  wire                     mem_bvalid,
    output wire                     mem_bready,
    output wire [31:0]              mem_araddr,
    output wire [7:0]               mem_arlen,
    output wire [2:0]               mem_arsize,
    output wire [1:0]               mem_arburst,
    output wire [3:0]               mem_arcache,
    output wire [2:0]               mem_arprot,
    output reg                      mem_arvalid = 1'b0,
    input  wire                     mem_arready,
    input  wire [63:0]              mem_rdata,
    input  wire [1:0]               mem_rresp,
    input  wire                     mem_rlast,
    input  wire                     mem_rvalid,
    output wire                     mem_rready
);
    localparam SBYTES = BITS <= 8 ? 1 : BITS <= 16 ? 2 : 4;
    localparam LANE = SBYTES * 8;               // a sample's bits in a word
    localparam SPW = 8 / SBYTES;                // samples a word
    localparam SLOT_BYTES = (PIXELS * SBYTES + 4095) / 4096 * 4096;
    localparam BURST = 16;
    localparam WQ = 64;                         // words queued to write
    localparam RQ = 64;                         // words a frame read holds
    localparam OFF = 24;                        // bits of a word offset

    assign mem_awsize = 3'd3;   assign mem_arsize = 3'd3;   // 8 bytes
    assign mem_awburst = 2'b01; assign mem_arburst = 2'b01; // INCR
    assign mem_awcache = 4'b0011; assign mem_arcache = 4'b0011;
    assign mem_awprot = 3'b000; assign mem_arprot = 3'b000;
    assign mem_wstrb = 8'hff;

    function [31:0] addr(input [SB-1:0] slot, input [OFF-1:0] word);
        addr = BASE + slot * SLOT_BYTES + {word, 3'b000};
    endfunction

    // ================================================================ write
    // ---- packing samples into words
    reg  [63:0]     pack = 64'd0;
    reg  [3:0]      lane = 4'd0;
    reg  [OFF-1:0]  woff = {OFF{1'b0}};
    wire            take = wr_valid && wr_ready;
    wire [3:0]      at = wr_sof ? 4'd0 : lane;
    wire [OFF-1:0]  off = wr_sof ? {OFF{1'b0}} : woff;
    wire [63:0]     shifted = {{(64-BITS){1'b0}}, wr_data} << (at * LANE);
    wire [63:0]     word = (wr_sof ? 64'd0 : pack) | shifted;
    wire            push = take && (at == SPW - 1 || wr_last);

    // ---- the queue of words: data, where it goes, whether a frame ends
    reg  [63:0]     wq_data [0:WQ-1];
    reg  [SB-1:0]   wq_slot [0:WQ-1];
    reg  [OFF-1:0]  wq_off  [0:WQ-1];
    reg             wq_end  [0:WQ-1];
    reg  [6:0]      wq_w = 7'd0, wq_r = 7'd0;   // pushes and pops, mod 128
    wire [6:0]      wq_count = wq_w - wq_r;
    assign wr_ready = wq_count < WQ - 2;
    // where each queued frame ends: the push count of its last word
    reg  [6:0]      ends [0:15];
    reg  [4:0]      ends_w = 5'd0, ends_r = 5'd0;
    wire            ending = ends_w != ends_r;
    wire [6:0]      to_end = ends[ends_r[3:0]] - wq_r + 7'd1;

    always @(posedge clk) begin
        if (rst) begin
            lane <= 4'd0; woff <= {OFF{1'b0}}; wq_w <= 7'd0; ends_w <= 5'd0;
        end else if (take) begin
            if (push) begin
                wq_data[wq_w[5:0]] <= word;
                wq_slot[wq_w[5:0]] <= wr_slot;
                wq_off[wq_w[5:0]]  <= off;
                wq_end[wq_w[5:0]]  <= wr_last;
                wq_w <= wq_w + 7'd1;
                if (wr_last) begin
                    ends[ends_w[3:0]] <= wq_w;
                    ends_w <= ends_w + 5'd1;
                end
                lane <= 4'd0;
                pack <= 64'd0;
                woff <= off + 1'b1;
            end else begin
                lane <= at + 4'd1;
                pack <= word;
                woff <= off;
            end
        end
    end

    // ---- bursts: the address once the words are queued, then the words
    reg  [7:0]      len = 8'd0;                 // beats of the burst
    reg  [7:0]      sent = 8'd0;
    reg             wsending = 1'b0;
    reg             last_of_frame = 1'b0;
    reg  [SB-1:0]   burst_slot = {SB{1'b0}};
    reg  [31:0]     awaddr = 32'd0;
    // the responses owed: whether each burst ends a frame, and its slot
    reg             bq_end  [0:7];
    reg  [SB-1:0]   bq_slot [0:7];
    reg  [3:0]      bq_w = 4'd0, bq_r = 4'd0;
    wire            bq_full = bq_w - bq_r == 4'd8;
    wire [7:0]      ready_len = (ending && to_end <= BURST) ? {1'b0, to_end}
                                                            : BURST;
    wire            can_burst = !mem_awvalid && !wsending && !bq_full &&
                                (wq_count >= BURST || ending);
    assign mem_awaddr = awaddr;
    assign mem_awlen = len - 8'd1;
    always @(posedge clk) begin
        if (rst) begin
            mem_awvalid <= 1'b0; wsending <= 1'b0; wq_r <= 7'd0;
            ends_r <= 5'd0; bq_w <= 4'd0;
        end else begin
            if (can_burst) begin
                len <= ready_len;
                awaddr <= addr(wq_slot[wq_r[5:0]], wq_off[wq_r[5:0]]);
                burst_slot <= wq_slot[wq_r[5:0]];
                last_of_frame <= ending && to_end <= BURST;
                mem_awvalid <= 1'b1;
            end
            if (mem_awvalid && mem_awready) begin
                mem_awvalid <= 1'b0;
                wsending <= 1'b1;
                sent <= 8'd0;
                bq_end[bq_w[2:0]] <= last_of_frame;
                bq_slot[bq_w[2:0]] <= burst_slot;
                bq_w <= bq_w + 4'd1;
                if (last_of_frame) ends_r <= ends_r + 5'd1;
            end
            if (mem_wvalid && mem_wready) begin
                wq_r <= wq_r + 7'd1;
                sent <= sent + 8'd1;
                if (mem_wlast) wsending <= 1'b0;
            end
        end
    end
    assign mem_wvalid = wsending;
    assign mem_wdata = wq_data[wq_r[5:0]];
    assign mem_wlast = sent == len - 8'd1;

    // ---- a frame is whole when its last burst's response is back
    assign mem_bready = 1'b1;
    always @(posedge clk) begin
        wr_done <= 1'b0;
        if (rst) begin
            bq_r <= 4'd0;
        end else if (mem_bvalid) begin
            if (bq_end[bq_r[2:0]]) begin
                wr_done <= 1'b1;
                wr_done_slot <= bq_slot[bq_r[2:0]];
            end
            bq_r <= bq_r + 4'd1;
        end
    end

    // ================================================================= read
    reg  [31:0]          left = 32'd0;          // samples still to go out
    reg  [OFF-1:0]       words = {OFF{1'b0}};   // words a frame read takes
    reg  [FRAMES*SB-1:0] which = {FRAMES*SB{1'b0}};
    reg  [OFF-1:0]       asked [0:FRAMES-1];    // words asked for, each
    reg  [7:0]           owed  [0:FRAMES-1];    // asked for, not yet in
    reg  [3:0]           rlane = 4'd0;
    // the bursts asked for, in order: whose, how long
    reg  [7:0]           aq_who [0:15];
    reg  [3:0]           aq_w = 4'd0, aq_r = 4'd0;
    reg  [7:0]           turn = 8'd0;           // whose turn to ask
    reg  [7:0]           ask_who = 8'd0;
    reg  [7:0]           ask_len = 8'd0;
    reg  [31:0]          araddr = 32'd0;
    assign mem_araddr = araddr;
    assign mem_arlen = ask_len - 8'd1;
    assign mem_rready = 1'b1;

    // each frame read's queue
    wire [FRAMES-1:0]    has;
    wire [FRAMES*7-1:0]  held;
    wire                 out = rd_valid && rd_ready;
    wire                 word_done = out && (rlane == SPW - 1 || left == 32'd1);
    genvar g;
    generate for (g = 0; g < FRAMES; g = g + 1) begin : q
        reg [63:0] mem [0:RQ-1];
        reg [6:0]  w = 7'd0, r = 7'd0;
        wire in = mem_rvalid && aq_who[aq_r] == g;
        always @(posedge clk) begin
            if (rst || rd_start) begin
                w <= 7'd0; r <= 7'd0;
            end else begin
                if (in) begin
                    mem[w[5:0]] <= mem_rdata;
                    w <= w + 7'd1;
                end
                if (word_done) r <= r + 7'd1;
            end
        end
        assign has[g] = w != r;
        assign held[g*7 +: 7] = w - r;
        assign rd_data[g*BITS +: BITS] = mem[r[5:0]] >> (rlane * LANE);
    end endgenerate
    assign rd_valid = left != 32'd0 && &has;

    // whose turn it is, and whether its queue takes a whole burst more
    integer k;
    reg            fits;
    reg  [7:0]     pick, pick_len;
    reg  [OFF-1:0] rest;
    always @* begin
        fits = 1'b0; pick = 8'd0; pick_len = 8'd0;
        for (k = FRAMES - 1; k >= 0; k = k - 1) begin : look
            integer j;
            j = (turn + k) % FRAMES;
            rest = words - asked[j];
            if (rest != 0 && held[j*7 +: 7] + owed[j] + BURST <= RQ) begin
                fits = 1'b1; pick = j[7:0];
                pick_len = rest < BURST ? rest[7:0] : BURST;
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            left <= 32'd0; mem_arvalid <= 1'b0; aq_w <= 4'd0; aq_r <= 4'd0;
            for (k = 0; k < FRAMES; k = k + 1) begin
                asked[k] <= {OFF{1'b0}}; owed[k] <= 8'd0;
            end
        end else if (rd_start) begin
            left <= rd_pixels;
            words <= (rd_pixels + SPW - 1) / SPW;
            which <= rd_slots;
            rlane <= 4'd0;
            mem_arvalid <= 1'b0; aq_w <= 4'd0; aq_r <= 4'd0;
            for (k = 0; k < FRAMES; k = k + 1) begin
                asked[k] <= {OFF{1'b0}}; owed[k] <= 8'd0;
            end
        end else begin
            if (!mem_arvalid && fits && aq_w - aq_r != 4'd15) begin
                ask_who <= pick;
                ask_len <= pick_len;
                araddr <= addr(which[pick*SB +: SB], asked[pick]);
                asked[pick] <= asked[pick] + pick_len;
                owed[pick] <= owed[pick] + pick_len;
                mem_arvalid <= 1'b1;
                turn <= (pick + 8'd1) % FRAMES;
            end
            if (mem_arvalid && mem_arready) begin
                mem_arvalid <= 1'b0;
                aq_who[aq_w] <= ask_who;
                aq_w <= aq_w + 4'd1;
            end
            if (mem_rvalid) begin
                owed[aq_who[aq_r]] <= owed[aq_who[aq_r]] - 8'd1
                    + ((!mem_arvalid && fits && aq_w - aq_r != 4'd15
                        && pick == aq_who[aq_r]) ? pick_len : 8'd0);
                if (mem_rlast) aq_r <= aq_r + 4'd1;
            end
            if (out) begin
                left <= left - 32'd1;
                rlane <= word_done ? 4'd0 : rlane + 4'd1;
            end
        end
    end
endmodule
