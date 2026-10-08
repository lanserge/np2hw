// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_not -- every bit of a WIDTH-bit word inverted.
module np2hw_not #(
    parameter WIDTH = 1
) (
    input  wire [WIDTH-1:0] a,
    output wire [WIDTH-1:0] y
);
    assign y = ~a;
endmodule
