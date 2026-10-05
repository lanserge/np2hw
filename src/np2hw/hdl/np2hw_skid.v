// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_skid -- an elastic register slice (skid buffer): every port
// leaves a flop. W is the payload width; framing flags ride in it,
// packed by whoever assembles the stream.
//
// WHY, since a wire would be shorter: a core's timing claim is
// about its own registers, and stays true only while nothing else
// shares the path. A combinational output hands its depth to every
// consumer, so two modules that each meet timing alone can miss it
// together -- and no per-module model can see that coming.
//
// The skid register is what lets `ready` be registered too. Register
// a valid/ready interface with one flop and you either combinationally
// pass `ready` back (the path you were trying to break) or stall every
// other cycle. Holding one beat in reserve costs a register and buys
// full throughput with nothing combinational in either direction.
module np2hw_skid #(
    parameter W = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s:m, ASSOCIATED_RESET rst" *)
    input  wire         clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    input  wire         rst,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 s DATA" *)
    input  wire [W-1:0] s_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 s VALID" *)
    input  wire         s_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 s READY" *)
    output wire         s_ready,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 m DATA" *)
    output wire [W-1:0] m_data,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 m VALID" *)
    output wire         m_valid,
    (* X_INTERFACE_INFO = "lanserge:interface:np2hw_stream_rtl:1.0 m READY" *)
    input  wire         m_ready
);
    reg [W-1:0] out_data;
    reg         out_valid;
    reg [W-1:0] skid_data;   // the beat accepted while the sink
    reg         slot_free;   // was busy; never more than one

    // The reserve slot's state is kept in the polarity the PORT
    // wants, so `s_ready` is a flop and not an inverter after one.
    // An inverter is nearly free and that is exactly the reasoning
    // this module exists to stop: `nearly free` is a claim about
    // the consumer, made without seeing it.
    assign s_ready = slot_free;
    assign m_data  = out_data;
    assign m_valid = out_valid;

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            slot_free <= 1'b1;
        end else if (slot_free) begin
            if (out_valid && !m_ready) begin
                // The sink is busy and the source is offering: take
                // it into the reserve rather than refusing it.
                if (s_valid) begin
                    skid_data <= s_data;
                    slot_free <= 1'b0;
                end
            end else begin
                out_data  <= s_data;
                out_valid <= s_valid;
            end
        end else if (m_ready) begin
            // Draining the reserve. s_ready is low this cycle, so
            // nothing arrives to overwrite it.
            out_data  <= skid_data;
            out_valid <= 1'b1;
            slot_free <= 1'b1;
        end
    end
endmodule
