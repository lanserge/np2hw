// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_slice -- bits HI down to LO of an IN-bit word.
module np2hw_slice #(
    parameter IN = 32,
    parameter HI = 0,
    parameter LO = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [IN-1:0]  d,      // the other bits are the point
    /* verilator lint_on UNUSEDSIGNAL */
    output wire [HI-LO:0] q
);
    assign q = d[HI:LO];
endmodule
