// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_const -- a constant, WIDTH bits of VALUE.
module np2hw_const #(
    parameter WIDTH = 1,
    parameter VALUE = 0
) (
    output wire [WIDTH-1:0] q
);
    assign q = VALUE;
endmodule
