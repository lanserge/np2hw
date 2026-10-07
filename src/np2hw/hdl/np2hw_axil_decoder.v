// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_axil_decoder -- one AXI4-Lite master onto N slaves, by address.
//
// Slave i owns every address a with (a & MASK[i]) == BASE[i]: a window
// whose size is a power of two, aligned to it, so the decode is a
// compare of the top bits. The windows are the design's allocator's;
// nothing here chooses an address. An address in no window is answered
// DECERR by this module, so a typo in a host's address map comes back as
// an error instead of hanging the processor that asked.
//
// One transaction at a time in each direction, writes and reads
// independent. That is the whole of the bus a register file needs, and
// it keeps the decoder free of ordering rules: a response always belongs
// to the one request in flight. The slaves see the full address and
// decode their own low bits, as every AXI slave does.
//
// The ports are packed, slave i at [i*W +: W]; a design's leaf gives each
// slave its own named interface around this core.
module np2hw_axil_decoder #(
    parameter            N    = 2,
    parameter            AW   = 32,
    parameter [N*AW-1:0] BASE = 0,
    parameter [N*AW-1:0] MASK = 0
) (
    input  wire            aclk,
    input  wire            aresetn,

    input  wire [AW-1:0]   s_awaddr,
    input  wire            s_awvalid,
    output wire            s_awready,
    input  wire [31:0]     s_wdata,
    input  wire [3:0]      s_wstrb,
    input  wire            s_wvalid,
    output wire            s_wready,
    output wire [1:0]      s_bresp,
    output wire            s_bvalid,
    input  wire            s_bready,
    input  wire [AW-1:0]   s_araddr,
    input  wire            s_arvalid,
    output wire            s_arready,
    output wire [31:0]     s_rdata,
    output wire [1:0]      s_rresp,
    output wire            s_rvalid,
    input  wire            s_rready,

    output wire [N*AW-1:0] m_awaddr,
    output wire [N-1:0]    m_awvalid,
    input  wire [N-1:0]    m_awready,
    output wire [N*32-1:0] m_wdata,
    output wire [N*4-1:0]  m_wstrb,
    output wire [N-1:0]    m_wvalid,
    input  wire [N-1:0]    m_wready,
    input  wire [N*2-1:0]  m_bresp,
    input  wire [N-1:0]    m_bvalid,
    output wire [N-1:0]    m_bready,
    output wire [N*AW-1:0] m_araddr,
    output wire [N-1:0]    m_arvalid,
    input  wire [N-1:0]    m_arready,
    input  wire [N*32-1:0] m_rdata,
    input  wire [N*2-1:0]  m_rresp,
    input  wire [N-1:0]    m_rvalid,
    output wire [N-1:0]    m_rready
);
    localparam [1:0] DECERR = 2'b11;

    // The window an address falls in, one-hot; zero for none.
    function [N-1:0] hit(input [AW-1:0] a);
        integer i;
        begin
            hit = {N{1'b0}};
            for (i = 0; i < N; i = i + 1)
                if ((a & MASK[i*AW +: AW]) == BASE[i*AW +: AW])
                    hit[i] = 1'b1;
        end
    endfunction

    // -- writes --------------------------------------------------------- //
    localparam [1:0] W_IDLE = 2'd0, W_ADDR = 2'd1, W_RESP = 2'd2;
    reg [1:0]    w_state;
    reg [AW-1:0] w_addr;
    reg [N-1:0]  w_sel;
    reg          aw_done, w_done;

    wire w_miss  = (w_sel == {N{1'b0}});
    wire aw_hs   = |(m_awvalid & m_awready);
    assign s_awready = (w_state == W_IDLE);
    assign s_wready  = (w_state == W_ADDR) && !w_done &&
                       (w_miss || |(m_wready & w_sel));
    wire w_hs    = s_wvalid && s_wready;

    reg       b_valid_sel;
    reg [1:0] b_resp_sel;
    integer   bi;
    always @* begin
        b_valid_sel = 1'b0;
        b_resp_sel  = 2'b00;
        for (bi = 0; bi < N; bi = bi + 1)
            if (w_sel[bi]) begin
                b_valid_sel = m_bvalid[bi];
                b_resp_sel  = m_bresp[bi*2 +: 2];
            end
    end
    assign s_bvalid = (w_state == W_RESP) && (w_miss || b_valid_sel);
    assign s_bresp  = w_miss ? DECERR : b_resp_sel;

    assign m_awaddr  = {N{w_addr}};
    assign m_awvalid = (w_state == W_ADDR && !aw_done) ? w_sel : {N{1'b0}};
    assign m_wdata   = {N{s_wdata}};
    assign m_wstrb   = {N{s_wstrb}};
    assign m_wvalid  = (w_state == W_ADDR && !w_done && s_wvalid)
                       ? w_sel : {N{1'b0}};
    assign m_bready  = (w_state == W_RESP && s_bready) ? w_sel : {N{1'b0}};

    always @(posedge aclk)
        if (!aresetn) begin
            w_state <= W_IDLE;
            w_sel   <= {N{1'b0}};
            aw_done <= 1'b0;
            w_done  <= 1'b0;
        end else case (w_state)
            W_IDLE: if (s_awvalid) begin
                w_addr  <= s_awaddr;
                w_sel   <= hit(s_awaddr);
                // a miss has no slave to take the address
                aw_done <= (hit(s_awaddr) == {N{1'b0}});
                w_done  <= 1'b0;
                w_state <= W_ADDR;
            end
            W_ADDR: begin
                if (aw_hs) aw_done <= 1'b1;
                if (w_hs)  w_done  <= 1'b1;
                if ((aw_done || aw_hs) && (w_done || w_hs))
                    w_state <= W_RESP;
            end
            W_RESP: if (s_bvalid && s_bready) w_state <= W_IDLE;
            default: w_state <= W_IDLE;
        endcase

    // -- reads ---------------------------------------------------------- //
    localparam [1:0] R_IDLE = 2'd0, R_ADDR = 2'd1, R_DATA = 2'd2;
    reg [1:0]    r_state;
    reg [AW-1:0] r_addr;
    reg [N-1:0]  r_sel;

    wire r_miss = (r_sel == {N{1'b0}});
    assign s_arready = (r_state == R_IDLE);

    reg        r_valid_sel;
    reg [31:0] r_data_sel;
    reg [1:0]  r_resp_sel;
    integer    ri;
    always @* begin
        r_valid_sel = 1'b0;
        r_data_sel  = 32'd0;
        r_resp_sel  = 2'b00;
        for (ri = 0; ri < N; ri = ri + 1)
            if (r_sel[ri]) begin
                r_valid_sel = m_rvalid[ri];
                r_data_sel  = m_rdata[ri*32 +: 32];
                r_resp_sel  = m_rresp[ri*2 +: 2];
            end
    end
    assign s_rvalid = (r_state == R_DATA) && (r_miss || r_valid_sel);
    assign s_rdata  = r_miss ? 32'd0 : r_data_sel;
    assign s_rresp  = r_miss ? DECERR : r_resp_sel;

    assign m_araddr  = {N{r_addr}};
    assign m_arvalid = (r_state == R_ADDR) ? r_sel : {N{1'b0}};
    assign m_rready  = (r_state == R_DATA && s_rready) ? r_sel : {N{1'b0}};

    always @(posedge aclk)
        if (!aresetn) begin
            r_state <= R_IDLE;
            r_sel   <= {N{1'b0}};
        end else case (r_state)
            R_IDLE: if (s_arvalid) begin
                r_addr  <= s_araddr;
                r_sel   <= hit(s_araddr);
                r_state <= (hit(s_araddr) == {N{1'b0}}) ? R_DATA : R_ADDR;
            end
            R_ADDR: if (|(m_arvalid & m_arready)) r_state <= R_DATA;
            R_DATA: if (s_rvalid && s_rready) r_state <= R_IDLE;
            default: r_state <= R_IDLE;
        endcase
endmodule
