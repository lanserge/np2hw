// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_readout -- a stream's frames, read by software: each frame written
// into a memory on the stream's clock, and read over AXI4-Lite on the
// bus's.
//
// The memory holds two frames. A frame is written into one half while the
// other holds the frame slot_less, which is what a read reads; the halves
// turn as a frame's last beat is written. Taking a beat never waits, so
// the stream is always ready. The memory has a port on each clock -- block
// RAM, its write port on clk and its read port on aclk -- and is itself the
// crossing: nothing written is read until its frame is published, and the
// half being written is never the one read.
//
// The frames published are counted on clk and the count crosses to aclk
// Gray coded, through two flops; which half holds the last frame published
// is the count's low bit, read on aclk, so the two always agree. Software
// reads the count, the words it wants, and the count again: a count that
// changed across the read means the half read may have been rewritten, so
// it reads again.
//
// The window, as 32-bit words:
//   word 0                  the frames published, in the low 16 bits
//   slot e + 1              entry e: SLOT_WORDS words, VALUE_WORDS a value,
//                           DATA_BITS / FIELD values, each two's complement
//                           where SIGNED, the slot's words beyond them zero
// An entry is a beat of the stream, in the order the frame brings them.
// A write is answered SLVERR: nothing here is software's to set.
module np2hw_readout #(
    parameter DATA_BITS   = 8,     // the stream's word
    parameter CTX_BITS    = 32,    // its context, carried and unread here
    parameter FIELD       = 8,     // bits a value, side by side in a beat, the first lowest
    parameter SIGNED      = 0,     // a value two's complement (one channel)
    parameter ENTRIES     = 1,     // beats a frame
    parameter VALUE_WORDS = 1,     // words a value: a power of two
    parameter SLOT_WORDS  = 1,     // words an entry: a power of two
    parameter AW          = 12     // the window's address bits
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire [DATA_BITS-1:0]  in_data,
    input  wire                  in_sof,
    input  wire                  in_eol,
    input  wire                  in_last,
    input  wire [CTX_BITS-1:0]   in_ctx,

    input  wire                  aclk,
    input  wire                  aresetn,
    input  wire [AW-1:0]         s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output reg                   s_axi_awready,
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output reg                   s_axi_wready,
    output wire [1:0]            s_axi_bresp,
    output reg                   s_axi_bvalid,
    input  wire                  s_axi_bready,
    input  wire [AW-1:0]         s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,
    output reg  [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,
    output reg                   s_axi_rvalid,
    input  wire                  s_axi_rready
);
    localparam EB = ENTRIES > 1 ? $clog2(ENTRIES) : 1;     // an entry's bits
    localparam SB = $clog2(SLOT_WORDS);                    // a slot's word bits
    localparam VB = $clog2(VALUE_WORDS);                   // a value's word bits

    // -- the stream's side: each beat written, each frame published -------
    (* ram_style = "block" *) reg [DATA_BITS-1:0] mem [0:(2 << EB) - 1];
    reg [EB-1:0] entry;                 // the next beat's, in_slot its frame
    reg [15:0]   frames;                // frames published
    reg [15:0]   frames_gray;
    wire [EB-1:0] at = in_sof ? {EB{1'b0}} : entry;

    assign in_ready = 1'b1;
    always @(posedge clk) begin
        if (in_valid) mem[{frames[0], at}] <= in_data;
        if (rst) begin
            entry <= {EB{1'b0}};
            frames <= 16'd0;
        end else if (in_valid) begin
            entry <= at + 1'b1;
            if (in_last) frames <= frames + 16'd1;
        end
        frames_gray <= frames ^ (frames >> 1);
    end

    // -- the count, to the bus's clock -------------------------------------
    (* ASYNC_REG = "TRUE" *) reg [15:0] frames_sync1, frames_sync2;
    always @(posedge aclk) begin
        frames_sync1 <= frames_gray;
        frames_sync2 <= frames_sync1;
    end
    wire [15:0] published;
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : binary
        assign published[i] = ^frames_sync2[15:i];
    end endgenerate

    // -- the bus's side: reads ---------------------------------------------
    // idle, then the memory read, then the word, then waiting to be taken
    reg [1:0]    state;
    reg [AW-1:0] addr;
    reg [15:0]   count;                 // the count the read is of
    reg [DATA_BITS-1:0] word;
    localparam VBITS = VALUE_WORDS * 32;                   // a value's words
    wire [AW-3:0] w = addr[AW-1:2];
    wire [AW-3-SB:0] slot = w[AW-3:SB];
    wire [AW-3-SB:0] slot_less = slot - 1'b1;
    wire [EB-1:0] read_entry = slot_less[EB-1:0];
    // the last frame published is in the half its count's low bit is not
    wire [EB:0] read_at = {~count[0], read_entry};
    // the value the word is of: its channel, then its word of the value
    wire [SB:0] in_slot;                // the word's place in its slot
    generate if (SB == 0) begin : one_word
        assign in_slot = 1'b0;
    end else begin : words
        assign in_slot = {1'b0, w[SB-1:0]};
    end endgenerate
    wire [SB:0] channel = in_slot >> VB;
    wire [SB:0] part = in_slot & (VALUE_WORDS - 1);
    wire [DATA_BITS-1:0] shifted = word >> (channel * FIELD);
    wire [FIELD-1:0] field = shifted[FIELD-1:0];
    wire [VBITS-1:0] extended, zeroed;  // the value, sign- and zero-extended
    generate if (VBITS == FIELD) begin : whole
        assign extended = field;
        assign zeroed = field;
    end else begin : widened
        assign extended = {{(VBITS - FIELD){field[FIELD-1]}}, field};
        assign zeroed = {{(VBITS - FIELD){1'b0}}, field};
    end endgenerate
    // a slot's channels past the beat's are shifted past its word: zero
    wire [VBITS-1:0] value = SIGNED ? extended : zeroed;
    wire [VBITS-1:0] down = value >> (part * 32);
    wire [31:0] picked = slot == 0 ? (w == 0 ? {16'd0, count} : 32'd0) :
                         slot > ENTRIES ? 32'd0 : down[31:0];

    assign s_axi_arready = aresetn && state == 2'd0;
    assign s_axi_rresp = 2'b00;
    always @(posedge aclk) begin
        if (!aresetn) begin
            state <= 2'd0;
            s_axi_rvalid <= 1'b0;
        end else case (state)
            2'd0: if (s_axi_arvalid) begin
                addr <= s_axi_araddr;
                count <= published;
                state <= 2'd1;
            end
            2'd1: begin
                word <= mem[read_at];
                state <= 2'd2;
            end
            2'd2: begin
                s_axi_rdata <= picked;
                s_axi_rvalid <= 1'b1;
                state <= 2'd3;
            end
            2'd3: if (s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                state <= 2'd0;
            end
        endcase
    end

    // -- writes: none are software's -----------------------------------------
    assign s_axi_bresp = 2'b10;
    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
        end else begin
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            if (s_axi_bvalid) begin
                if (s_axi_bready) s_axi_bvalid <= 1'b0;
            end else if (s_axi_awvalid && s_axi_wvalid && !s_axi_awready) begin
                s_axi_awready <= 1'b1;
                s_axi_wready <= 1'b1;
                s_axi_bvalid <= 1'b1;
            end
        end
    end
endmodule
