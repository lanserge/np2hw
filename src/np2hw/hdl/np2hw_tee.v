// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_tee -- a frame-atomic, NEVER-STALLING two-way tee. W is the
// payload width, flags packed in it; in_sof marks a frame's first beat.
//
// Upstream is always accepted; each branch is a disposable tap.
// A branch that misses a beat mid-frame drops out of the frame
// and a sticky torn output says so; the sibling and the
// pipeline never feel it. Enables are sampled at SOF and held
// for the frame, so no branch ever receives a partial head.
module np2hw_tee #(
    parameter W = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in_stream:a:b, ASSOCIATED_RESET rst" *)
    input  wire         clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire         rst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream VALID" *)
    input  wire         in_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream READY" *)
    output wire         in_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream DATA" *)
    input  wire [W-1:0] in_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream SOF" *)
    input  wire         in_sof,
    input  wire         en_a,
    input  wire         en_b,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 a VALID" *)
    output wire         a_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 a READY" *)
    input  wire         a_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 a DATA" *)
    output wire [W-1:0] a_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 b VALID" *)
    output wire         b_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 b READY" *)
    input  wire         b_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 b DATA" *)
    output wire [W-1:0] b_data,
    // sticky until reset: that branch lost mid-frame beats
    output reg          torn_a,
    output reg          torn_b
);
    // What each branch is doing THIS frame. Latched across the
    // frame; re-decided only at a SOF beat, and the SOF beat
    // itself already obeys the new decision, so a joining branch
    // gets the frame from its first pixel.
    reg act_a = 1'b0, act_b = 1'b0;
    // a branch takes the SOF beat only if it is ready FOR that
    // beat; joining late would hand it a headless frame
    wire join_a = en_a && a_ready;
    wire join_b = en_b && b_ready;
    wire sel_a = in_sof ? join_a : act_a;
    wire sel_b = in_sof ? join_b : act_b;

    assign a_data  = in_data;
    assign b_data  = in_data;
    assign a_valid = in_valid && sel_a;
    assign b_valid = in_valid && sel_b;
    // the whole law in one line: nothing downstream of a tap
    // may slow the stream that feeds it
    assign in_ready = 1'b1;

    always @(posedge clk) begin
        if (rst) begin
            act_a <= 1'b0; act_b <= 1'b0;
            torn_a <= 1'b0; torn_b <= 1'b0;
        end else if (in_valid) begin
            if (in_sof) begin
                act_a <= join_a; act_b <= join_b;
            end else begin
                // a missed beat mid-frame: that branch is out
                // of THIS frame -- a partial tail is worse than
                // an honest absence -- and the sticky testifies
                if (act_a && !a_ready) begin
                    act_a <= 1'b0; torn_a <= 1'b1;
                end
                if (act_b && !b_ready) begin
                    act_b <= 1'b0; torn_b <= 1'b1;
                end
            end
        end
    end
endmodule
