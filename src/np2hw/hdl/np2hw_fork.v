// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_fork -- one stream into N consumers, each taking every beat.
//
// The source's beat is held until every branch has taken it: a branch that
// takes it early is marked done and offered nothing more until the beat
// moves on, so the branches may each take it in a different cycle. Nothing
// is dropped and nothing is duplicated; a branch that is slow holds the
// source, and so every branch, as one consumer would.
//
// Only the handshake passes through here. The data, the flags and the
// context are the source's own, wired straight to every branch: they hold
// while the source's beat does.
//
// No branch's valid depends on any ready -- each is the source's valid less
// the branches done -- so a fork closes no loop through the readies, and
// the source's ready is every branch either done or taking the beat now.
module np2hw_fork #(
    parameter N = 2
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         s_valid,
    output wire         s_ready,
    output wire [N-1:0] m_valid,
    input  wire [N-1:0] m_ready
);
    reg [N-1:0] done;

    assign m_valid = {N{s_valid}} & ~done;
    assign s_ready = &(done | m_ready);

    always @(posedge clk)
        if (rst || (s_valid && s_ready)) done <= {N{1'b0}};
        else done <= done | (m_valid & m_ready);
endmodule
