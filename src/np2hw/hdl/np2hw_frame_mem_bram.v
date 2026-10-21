// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_mem_bram -- np2hw_frame_store's frames, kept in block RAM.
//
// The resident binding of the store's seam: one memory per slot, PIXELS
// samples deep, each written at the position its frame's beats count to
// and all of them read at one position together, so a beat out carries the
// same pixel of every frame read. Which memory answers for which frame is
// the read's own choice, latched at its start (rd_slots).
//
// Every memory keeps the seam's four rules (docs/memories.md): written on
// the edge `we` is high, read through a register that holds while it is
// not enabled, read-before-write, no reset. The read runs ahead of the
// consumer by at most two beats, which a two-entry buffer holds, so
// rd_ready may drop at any beat, and a beat is offered every cycle while it
// is high. A frame is whole the cycle after its last
// beat is written, when a read of it finds every beat.
module np2hw_frame_mem_bram #(
    parameter BITS = 12,
    parameter FRAMES = 4,
    parameter SLOTS = 8,
    parameter PIXELS = 43200,           // the largest frame the store keeps
    parameter SB = $clog2(SLOTS),
    parameter AW = $clog2(PIXELS)
) (
    input  wire                     clk,
    input  wire                     rst,
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
    output wire [FRAMES*BITS-1:0]   rd_data
);
    // ---- write: the position counts from each frame's first beat
    reg  [AW-1:0] wa = {AW{1'b0}};
    wire          we = wr_valid && wr_ready;
    wire [AW-1:0] waddr = wr_sof ? {AW{1'b0}} : wa;
    assign wr_ready = 1'b1;
    always @(posedge clk) begin
        wr_done <= 1'b0;
        if (we) begin
            wa <= waddr + 1'b1;
            if (wr_last) begin
                wr_done <= 1'b1;
                wr_done_slot <= wr_slot;
            end
        end
    end

    // ---- read: one position at a time, every memory at once
    reg  [31:0]          left = 32'd0;
    reg  [AW-1:0]        ra = {AW{1'b0}};
    reg  [FRAMES*SB-1:0] which = {FRAMES*SB{1'b0}};
    reg                  flight = 1'b0;   // a read issued last cycle
    reg  [1:0]           held = 2'd0;     // beats in the buffer
    wire                 pop = rd_valid && rd_ready;
    // room for one more after this cycle, counting the beat leaving now:
    // a read issues every cycle a beat is taken, so the consumer is never
    // kept waiting once the first two are in
    wire                 issue = left != 32'd0 &&
                                 {1'b0, held} + flight <= 3'd1 + pop;
    always @(posedge clk) begin
        if (rst) begin
            left <= 32'd0; flight <= 1'b0;
        end else if (rd_start) begin
            left <= rd_pixels; ra <= {AW{1'b0}}; which <= rd_slots;
            flight <= 1'b0;
        end else begin
            flight <= issue;
            if (issue) begin
                left <= left - 32'd1;
                ra <= ra + 1'b1;
            end
        end
    end

    wire [SLOTS*BITS-1:0] word;         // every memory's word at `ra`
    genvar s;
    generate for (s = 0; s < SLOTS; s = s + 1) begin : slot
        (* ram_decomp = "power" *) reg [BITS-1:0] mem [0:PIXELS-1];
        reg [BITS-1:0] rd = {BITS{1'b0}};
        always @(posedge clk) begin
            if (we && wr_slot == s) mem[waddr] <= wr_data;
            if (issue) rd <= mem[ra];
        end
        assign word[s*BITS +: BITS] = rd;
    end endgenerate

    // each frame read: the word of the memory its slot names
    reg [FRAMES*BITS-1:0] picked;
    integer k;
    always @* for (k = 0; k < FRAMES; k = k + 1)
        picked[k*BITS +: BITS] = word[which[k*SB +: SB]*BITS +: BITS];

    // ---- the two-entry buffer the consumer takes from
    reg [FRAMES*BITS-1:0] b0, b1;
    always @(posedge clk) begin
        if (rst || rd_start) begin
            held <= 2'd0;
        end else begin
            case ({flight, pop})
                2'b10: begin
                    if (held == 2'd0) b0 <= picked; else b1 <= picked;
                    held <= held + 2'd1;
                end
                2'b01: begin b0 <= b1; held <= held - 2'd1; end
                2'b11: begin
                    if (held == 2'd1) b0 <= picked;
                    else begin b0 <= b1; b1 <= picked; end
                end
                default: ;
            endcase
        end
    end
    assign rd_valid = held != 2'd0;
    assign rd_data = b0;
endmodule
