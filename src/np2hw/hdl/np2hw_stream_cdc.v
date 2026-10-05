// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_stream_cdc -- an np2hw stream crossing between two unrelated
// clocks: its samples and their framing flags (sof, eol, last) through
// np2hw_cdc_fifo, 2**A deep, packed {last, eol, sof, data}. The crossing's
// discipline, its reset contract (a link reset asserts BOTH sides) and its
// timing constraint (np2hw_cdc_fifo.xdc, scoped to that module) are the
// FIFO's; this only gives it a stream's ports on each face.
module np2hw_stream_cdc #(
    parameter DATA_BITS = 12,
    parameter A = 4
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
    output wire                 out_last
);
    np2hw_cdc_fifo #(.W(DATA_BITS + 3), .A(A)) u_fifo (
        .wclk(wclk), .wrst(wrst),
        .in_data({in_last, in_eol, in_sof, in_data}),
        .in_valid(in_valid), .in_ready(in_ready),
        .rclk(rclk), .rrst(rrst),
        .out_data({out_last, out_eol, out_sof, out_data}),
        .out_valid(out_valid), .out_ready(out_ready));
endmodule
