// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_reset_sync -- a reset for one clock domain: asserted the moment
// any source asks, released only on that domain's clock, STAGES flops
// after the last source lets go.
//
// WHY both halves: a reset that asserts on the clock cannot reach logic
// whose clock has stopped -- a link that drops takes its clock with it,
// and the logic must still let go of the bus -- so assertion is
// asynchronous. A reset that RELEASES asynchronously lets flops leave
// reset on different cycles, and a state machine half out of reset
// runs from a state no one wrote; so release is synchronous.
//
// Sources are active low, as AXI's and the processor's are: rst_n_in
// (a reset input, say the processor's fabric reset) and locked (a clock
// generator's lock -- tie it high where there is none). Both outputs are
// the same reset in the two polarities blocks expect.
module np2hw_reset_sync #(
    parameter STAGES = 3
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET rst_n:rst" *)
    input  wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n_in RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire rst_n_in,
    input  wire locked,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    output wire rst_n,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_HIGH" *)
    output wire rst
);
    wire asserted = !rst_n_in || !locked;

    (* ASYNC_REG = "TRUE" *) reg [STAGES-1:0] chain;

    always @(posedge clk or posedge asserted)
        if (asserted) chain <= {STAGES{1'b0}};
        else          chain <= {chain[STAGES-2:0], 1'b1};

    // Two output flops, one per polarity, so neither output is an
    // inverter after the other: a reset fans out to everything in its
    // domain, and its timing is its own.
    reg out_n, out_p;
    always @(posedge clk or posedge asserted)
        if (asserted) begin
            out_n <= 1'b0;
            out_p <= 1'b1;
        end else begin
            out_n <= chain[STAGES-1];
            out_p <= !chain[STAGES-1];
        end

    assign rst_n = out_n;
    assign rst   = out_p;
endmodule
