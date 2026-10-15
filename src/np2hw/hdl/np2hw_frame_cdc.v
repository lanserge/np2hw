// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_frame_cdc -- np2hw_stream_cdc for a stream that carries its frame's
// size: the samples and their flags through one np2hw_cdc_fifo, and each
// frame's width and height through a second, one entry per frame, written
// as the frame's first beat goes in.
//
// On the read side a frame's size is published from the moment its first
// beat is PRESENTED, before it is taken, so a consumer that places the
// frame -- a raster centring it, say -- has it in hand before it takes a
// pixel; and it stays published until the next frame's first beat is
// presented, so the tail of a frame draining behind the next one's size
// keeps its own. The first beat waits for its size: the two crossings'
// pointers are synchronised apart, and either may arrive first.
//
// The size is sampled as the first beat is taken, which is when a core
// latches the size it is given, so a size that changes mid-stream lands
// at a frame's start on both sides. Both crossings' discipline, reset
// contract and timing intent are np2hw_cdc_fifo's; its constraint file is
// scoped to that module and so covers both.
module np2hw_frame_cdc #(
    parameter DATA_BITS = 12,
    parameter A = 4,
    // frames the size crossing holds: a frame needs at least one beat in
    // the sample crossing, so it is full only when 2**SA frames are
    parameter SA = 2
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 wclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in_stream, ASSOCIATED_RESET wrst" *)
    input  wire                 wclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 wrst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire                 wrst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream VALID" *)
    input  wire                 in_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream READY" *)
    output wire                 in_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream DATA" *)
    input  wire [DATA_BITS-1:0] in_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream SOF" *)
    input  wire                 in_sof,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream EOL" *)
    input  wire                 in_eol,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream LAST" *)
    input  wire                 in_last,
    input  wire [15:0]          in_frame_width,
    input  wire [15:0]          in_frame_height,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 rclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF out_stream, ASSOCIATED_RESET rrst" *)
    input  wire                 rclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rrst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire                 rrst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream VALID" *)
    output wire                 out_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream READY" *)
    input  wire                 out_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream DATA" *)
    output wire [DATA_BITS-1:0] out_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream SOF" *)
    output wire                 out_sof,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream EOL" *)
    output wire                 out_eol,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream LAST" *)
    output wire                 out_last,
    output wire [15:0]          out_frame_width,
    output wire [15:0]          out_frame_height
);
    // ---- write side: a beat goes in when BOTH crossings have room, so
    // ready is two registers ANDed, never a function of the beat
    wire m_ready, s_ready;
    wire [DATA_BITS-1:0] m_data;
    wire m_sof, m_eol, m_last, m_valid, m_take;
    wire [31:0] s_size;
    wire s_valid, s_take;
    wire head = m_valid && m_sof;
    assign in_ready = m_ready && s_ready;
    np2hw_cdc_fifo #(.W(DATA_BITS + 3), .A(A)) u_samples (
        .wclk(wclk), .wrst(wrst),
        .in_data({in_last, in_eol, in_sof, in_data}),
        .in_valid(in_valid && s_ready), .in_ready(m_ready),
        .rclk(rclk), .rrst(rrst),
        .out_data({m_last, m_eol, m_sof, m_data}),
        .out_valid(m_valid), .out_ready(m_take));
    np2hw_cdc_fifo #(.W(32), .A(SA)) u_sizes (
        .wclk(wclk), .wrst(wrst),
        .in_data({in_frame_height, in_frame_width}),
        .in_valid(in_valid && in_sof && m_ready), .in_ready(s_ready),
        .rclk(rclk), .rrst(rrst),
        .out_data(s_size), .out_valid(s_valid), .out_ready(s_take));

    // ---- read side: a first beat is offered only with its size, and the
    // size is taken with it
    assign out_valid = m_valid && (!m_sof || s_valid);
    assign m_take = out_ready && (!m_sof || s_valid);
    assign s_take = head && s_valid && out_ready;
    assign out_data = m_data;
    assign out_sof  = m_sof;
    assign out_eol  = m_eol;
    assign out_last = m_last;

    reg [31:0] size = 32'd0;            // the frame being drained
    always @(posedge rclk)
        if (rrst) size <= 32'd0;
        else if (s_take) size <= s_size;
    wire [31:0] shown = (head && s_valid) ? s_size : size;
    assign out_frame_width  = shown[15:0];
    assign out_frame_height = shown[31:16];
endmodule
