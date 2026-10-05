// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_cdc_fifo -- a stream crossing between two clocks that owe each
// other nothing: a dual-clock FIFO, 2**A deep, W wide (flags packed in
// the payload by the caller). Both sides carry their own reset, and a
// link reset must assert BOTH: pointers that reset on one side only
// disagree forever after. docs/clock-crossings.md says what a timing
// flow has to be told about the two pointer paths.
//
// Gray-coded pointers cross; the payload never does. Each side
// compares its own exact pointer with a two-flop-stale picture
// of the other's, so full and empty are PESSIMISTIC by up to
// the staleness -- a slot of headroom, never a corrupt word.
module np2hw_cdc_fifo #(
    parameter W = 8,
    parameter A = 4
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 wclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in_stream, ASSOCIATED_RESET wrst" *)
    input  wire         wclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 wrst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire         wrst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream DATA" *)
    input  wire [W-1:0] in_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream VALID" *)
    input  wire         in_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 in_stream READY" *)
    output wire         in_ready,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 rclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF out_stream, ASSOCIATED_RESET rrst" *)
    input  wire         rclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rrst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire         rrst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream DATA" *)
    output wire [W-1:0] out_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream VALID" *)
    output wire         out_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 out_stream READY" *)
    input  wire         out_ready
);
    // Block RAM at every depth, so the payload never crosses on a path a
    // timer sees: a block RAM's read port is clocked by the reader. A
    // LUT RAM's read is combinational from what the write clock stored,
    // which puts every word on a path from the write domain into out_q --
    // safe by construction, but a path. Ignored by tools without block RAM.
    (* ram_style = "block" *) reg [W-1:0] mem [0:(1<<A)-1];
    // BOTH sides' pointers up front: each side's always block
    // samples the other's gray pointer, so neither can be
    // declared after its first use.
    // The gray pointers are KEPT: their top bit equals the binary's, and a
    // synthesis tool merging the two would leave the crossing launched from
    // a register the timing constraint does not name (np2hw_cdc_fifo.xdc).
    reg  [A:0] wbin = 0, rbin = 0;
    (* KEEP = "TRUE" *) reg [A:0] wgray = 0;
    (* KEEP = "TRUE" *) reg [A:0] rgray = 0;

    // ---- write side, in wclk ------------------------------
    (* ASYNC_REG = "TRUE" *) reg [A:0] wq_rgray0 = 0, wq_rgray = 0;
    // Ready is a REGISTER, judged from the pointer the pending
    // write will leave behind -- so it is never optimistic, and
    // the port leaves a flop (the boundary rule, again).
    reg w_ready_q = 1'b1;
    assign in_ready = w_ready_q;
    wire        w_take  = in_valid && in_ready;
    wire [A:0]  wbin_n  = wbin + {{A{1'b0}}, w_take};
    wire [A:0]  wgray_n = (wbin_n >> 1) ^ wbin_n;
    // Full: the next write pointer, gray-coded, equals the read
    // pointer with its top two bits inverted -- one lap ahead.
    wire w_full_n = (wgray_n == {~wq_rgray[A:A-1], wq_rgray[A-2:0]});
    always @(posedge wclk) begin
        if (wrst) begin
            wbin <= 0; wgray <= 0; w_ready_q <= 1'b1;
            wq_rgray0 <= 0; wq_rgray <= 0;
        end else begin
            wq_rgray0 <= rgray; wq_rgray <= wq_rgray0;
            if (w_take) mem[wbin[A-1:0]] <= in_data;
            wbin <= wbin_n; wgray <= wgray_n;
            w_ready_q <= !w_full_n;
        end
    end

    // ---- read side, in rclk -------------------------------
    (* ASYNC_REG = "TRUE" *) reg [A:0] rq_wgray0 = 0, rq_wgray = 0;
    reg          out_q_v = 1'b0;
    reg  [W-1:0] out_q;
    assign out_valid = out_q_v;
    assign out_data  = out_q;
    wire r_empty = (rgray == rq_wgray);
    // First-word-fall-through, through a register: a word is
    // popped into the output flop whenever that flop is free or
    // being drained, so the port never exposes the RAM read.
    wire r_pop   = !r_empty && (!out_q_v || out_ready);
    wire [A:0] rbin_n = rbin + {{A{1'b0}}, r_pop};
    always @(posedge rclk) begin
        if (rrst) begin
            rbin <= 0; rgray <= 0; out_q_v <= 1'b0;
            rq_wgray0 <= 0; rq_wgray <= 0;
        end else begin
            rq_wgray0 <= wgray; rq_wgray <= rq_wgray0;
            if (r_pop) begin
                out_q   <= mem[rbin[A-1:0]];
                out_q_v <= 1'b1;
            end else if (out_ready) out_q_v <= 1'b0;
            rbin <= rbin_n; rgray <= (rbin_n >> 1) ^ rbin_n;
        end
    end
endmodule
