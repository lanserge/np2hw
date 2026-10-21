// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_store_bram -- the frame store with its frames in block RAM:
// np2hw_frame_store on the block-RAM binding of its seam,
// np2hw_frame_mem_bram. The ports are the frame store's own, as a design
// is wired to them; what this implementation adds is only where the
// frames live, PIXELS samples a slot.
module np2hw_frame_store_bram #(
    parameter BITS = 12,
    parameter FRAMES = 4,
    parameter SLOTS = 8,
    parameter CTX_BITS = 1,
    parameter PIXELS = 43200            // the largest frame the store keeps
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [BITS-1:0]          in_data,
    input  wire                     in_sof,
    input  wire                     in_eol,
    input  wire                     in_last,
    input  wire [CTX_BITS-1:0]      in_ctx,
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [FRAMES*BITS-1:0]   out_data,
    output wire                     out_sof,
    output wire                     out_eol,
    output wire                     out_last,
    output wire [CTX_BITS-1:0]      out_ctx,
    output wire [15:0]              out_source,
    output wire                     cold,
    output wire                     overrun
);
    localparam SB = $clog2(SLOTS);
    wire                    wr_valid, wr_ready, wr_sof, wr_last, wr_done;
    wire [BITS-1:0]         wr_data;
    wire [SB-1:0]           wr_slot, wr_done_slot;
    wire                    rd_start, rd_valid, rd_ready;
    wire [FRAMES*SB-1:0]    rd_slots;
    wire [31:0]             rd_pixels;
    wire [FRAMES*BITS-1:0]  rd_data;
    np2hw_frame_store #(.BITS(BITS), .FRAMES(FRAMES), .SLOTS(SLOTS),
                        .CTX_BITS(CTX_BITS)) u_store (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .in_ctx(in_ctx),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ctx(out_ctx),
        .out_source(out_source), .cold(cold), .overrun(overrun),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_sof(wr_sof), .wr_last(wr_last), .wr_slot(wr_slot),
        .wr_done(wr_done), .wr_done_slot(wr_done_slot),
        .rd_start(rd_start), .rd_slots(rd_slots), .rd_pixels(rd_pixels),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data));
    np2hw_frame_mem_bram #(.BITS(BITS), .FRAMES(FRAMES), .SLOTS(SLOTS),
                           .PIXELS(PIXELS)) u_mem (
        .clk(clk), .rst(rst),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_sof(wr_sof), .wr_last(wr_last), .wr_slot(wr_slot),
        .wr_done(wr_done), .wr_done_slot(wr_done_slot),
        .rd_start(rd_start), .rd_slots(rd_slots), .rd_pixels(rd_pixels),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data));
endmodule
