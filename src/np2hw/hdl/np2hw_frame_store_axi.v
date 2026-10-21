// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_store_axi -- the frame store with its frames in external
// memory: np2hw_frame_store on the AXI4 binding of its seam,
// np2hw_frame_mem_axi. The ports are the frame store's own, as a design is
// wired to them, and the memory's; what this implementation adds is only
// where the frames live, slot by slot from BASE.
module np2hw_frame_store_axi #(
    parameter BITS = 12,
    parameter FRAMES = 4,
    parameter SLOTS = 8,
    parameter CTX_BITS = 1,
    parameter PIXELS = 43200,           // the largest frame the store keeps
    parameter BASE = 32'h0              // where the first slot starts
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
    output wire                     overrun,
    output wire [31:0]              mem_awaddr,
    output wire [7:0]               mem_awlen,
    output wire [2:0]               mem_awsize,
    output wire [1:0]               mem_awburst,
    output wire [3:0]               mem_awcache,
    output wire [2:0]               mem_awprot,
    output wire                     mem_awvalid,
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
    output wire                     mem_arvalid,
    input  wire                     mem_arready,
    input  wire [63:0]              mem_rdata,
    input  wire [1:0]               mem_rresp,
    input  wire                     mem_rlast,
    input  wire                     mem_rvalid,
    output wire                     mem_rready
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
    np2hw_frame_mem_axi #(.BITS(BITS), .FRAMES(FRAMES), .SLOTS(SLOTS),
                          .PIXELS(PIXELS), .BASE(BASE)) u_mem (
        .clk(clk), .rst(rst),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_data(wr_data),
        .wr_sof(wr_sof), .wr_last(wr_last), .wr_slot(wr_slot),
        .wr_done(wr_done), .wr_done_slot(wr_done_slot),
        .rd_start(rd_start), .rd_slots(rd_slots), .rd_pixels(rd_pixels),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_data(rd_data),
        .mem_awaddr(mem_awaddr), .mem_awlen(mem_awlen),
        .mem_awsize(mem_awsize), .mem_awburst(mem_awburst),
        .mem_awcache(mem_awcache), .mem_awprot(mem_awprot),
        .mem_awvalid(mem_awvalid), .mem_awready(mem_awready),
        .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb), .mem_wlast(mem_wlast),
        .mem_wvalid(mem_wvalid), .mem_wready(mem_wready),
        .mem_bresp(mem_bresp), .mem_bvalid(mem_bvalid),
        .mem_bready(mem_bready),
        .mem_araddr(mem_araddr), .mem_arlen(mem_arlen),
        .mem_arsize(mem_arsize), .mem_arburst(mem_arburst),
        .mem_arcache(mem_arcache), .mem_arprot(mem_arprot),
        .mem_arvalid(mem_arvalid), .mem_arready(mem_arready),
        .mem_rdata(mem_rdata), .mem_rresp(mem_rresp),
        .mem_rlast(mem_rlast), .mem_rvalid(mem_rvalid),
        .mem_rready(mem_rready));
endmodule
