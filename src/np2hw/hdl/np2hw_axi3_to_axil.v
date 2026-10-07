// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_axi3_to_axil -- an AXI3 master (a processor's general-purpose
// port) onto an AXI4-Lite bus.
//
// AXI3 transactions carry an ID and may be bursts; AXI4-Lite has neither.
// The bridge takes one transaction at a time in each direction, echoes
// its ID on the response, and turns a burst into one AXI4-Lite access per
// beat: INCR advances the address by the beat size, FIXED repeats it
// (WRAP, which no processor issues to device memory, is taken as INCR).
// A write burst answers with the worst response any of its beats got, so
// an error part way through is not lost behind the beats that worked.
//
// Unused AXI3 fields -- lock, cache, prot, qos, the write ID -- are inputs
// so the processor's port connects as the interface it is.
module np2hw_axi3_to_axil #(
    parameter AW   = 32,
    parameter ID_W = 12
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:m_axi, ASSOCIATED_RESET aresetn" *)
    input  wire            aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire            aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI3" *)
    input  wire [AW-1:0]   s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWBURST" *)
    input  wire [1:0]      s_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWCACHE" *)
    input  wire [3:0]      s_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWID" *)
    input  wire [ID_W-1:0] s_axi_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWLEN" *)
    input  wire [3:0]      s_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWLOCK" *)
    input  wire [1:0]      s_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWPROT" *)
    input  wire [2:0]      s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWQOS" *)
    input  wire [3:0]      s_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output wire            s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWSIZE" *)
    input  wire [2:0]      s_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire            s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [31:0]     s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WID" *)
    input  wire [ID_W-1:0] s_axi_wid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WLAST" *)
    input  wire            s_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output wire            s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [3:0]      s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire            s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BID" *)
    output wire [ID_W-1:0] s_axi_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire            s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]      s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output wire            s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [AW-1:0]   s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARBURST" *)
    input  wire [1:0]      s_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARCACHE" *)
    input  wire [3:0]      s_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARID" *)
    input  wire [ID_W-1:0] s_axi_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARLEN" *)
    input  wire [3:0]      s_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARLOCK" *)
    input  wire [1:0]      s_axi_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARPROT" *)
    input  wire [2:0]      s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARQOS" *)
    input  wire [3:0]      s_axi_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output wire            s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARSIZE" *)
    input  wire [2:0]      s_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire            s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output wire [31:0]     s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RID" *)
    output wire [ID_W-1:0] s_axi_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RLAST" *)
    output wire            s_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire            s_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]      s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output wire            s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE" *)
    output wire [AW-1:0]   m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWVALID" *)
    output wire            m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWREADY" *)
    input  wire            m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WDATA" *)
    output wire [31:0]     m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WSTRB" *)
    output wire [3:0]      m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WVALID" *)
    output wire            m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WREADY" *)
    input  wire            m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BRESP" *)
    input  wire [1:0]      m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BVALID" *)
    input  wire            m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BREADY" *)
    output wire            m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARADDR" *)
    output wire [AW-1:0]   m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARVALID" *)
    output wire            m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARREADY" *)
    input  wire            m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RDATA" *)
    input  wire [31:0]     m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RRESP" *)
    input  wire [1:0]      m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RVALID" *)
    input  wire            m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RREADY" *)
    output wire            m_axi_rready
);
    // -- writes --------------------------------------------------------- //
    localparam [1:0] W_IDLE = 2'd0, W_BEAT = 2'd1, W_B = 2'd2, W_RESP = 2'd3;
    reg [1:0]      w_state;
    reg [ID_W-1:0] w_id;
    reg [AW-1:0]   w_addr;
    reg [3:0]      w_left;      // beats after this one
    reg            w_fixed;
    reg [7:0]      w_step;
    reg [1:0]      w_resp;      // the worst of the beats so far
    reg            aw_done, w_done;

    assign s_axi_awready = (w_state == W_IDLE);
    assign m_axi_awaddr  = w_addr;
    assign m_axi_awvalid = (w_state == W_BEAT) && !aw_done;
    assign m_axi_wdata   = s_axi_wdata;
    assign m_axi_wstrb   = s_axi_wstrb;
    assign m_axi_wvalid  = (w_state == W_BEAT) && !w_done && s_axi_wvalid;
    assign s_axi_wready  = (w_state == W_BEAT) && !w_done && m_axi_wready;
    assign m_axi_bready  = (w_state == W_B);
    assign s_axi_bvalid  = (w_state == W_RESP);
    assign s_axi_bid     = w_id;
    assign s_axi_bresp   = w_resp;

    wire aw_hs = m_axi_awvalid && m_axi_awready;
    wire w_hs  = m_axi_wvalid  && m_axi_wready;

    always @(posedge aclk)
        if (!aresetn) begin
            w_state <= W_IDLE;
            aw_done <= 1'b0;
            w_done  <= 1'b0;
        end else case (w_state)
            W_IDLE: if (s_axi_awvalid) begin
                w_id    <= s_axi_awid;
                w_addr  <= s_axi_awaddr;
                w_left  <= s_axi_awlen;
                w_fixed <= (s_axi_awburst == 2'b00);
                w_step  <= 8'd1 << s_axi_awsize;
                w_resp  <= 2'b00;
                aw_done <= 1'b0;
                w_done  <= 1'b0;
                w_state <= W_BEAT;
            end
            W_BEAT: begin
                if (aw_hs) aw_done <= 1'b1;
                if (w_hs)  w_done  <= 1'b1;
                if ((aw_done || aw_hs) && (w_done || w_hs))
                    w_state <= W_B;
            end
            W_B: if (m_axi_bvalid) begin
                if (m_axi_bresp > w_resp) w_resp <= m_axi_bresp;
                if (w_left == 4'd0)
                    w_state <= W_RESP;
                else begin
                    w_left  <= w_left - 4'd1;
                    if (!w_fixed) w_addr <= w_addr + {{(AW-8){1'b0}}, w_step};
                    aw_done <= 1'b0;
                    w_done  <= 1'b0;
                    w_state <= W_BEAT;
                end
            end
            W_RESP: if (s_axi_bready) w_state <= W_IDLE;
        endcase

    // -- reads ---------------------------------------------------------- //
    localparam [1:0] R_IDLE = 2'd0, R_ADDR = 2'd1, R_DATA = 2'd2;
    reg [1:0]      r_state;
    reg [ID_W-1:0] r_id;
    reg [AW-1:0]   r_addr;
    reg [3:0]      r_left;
    reg            r_fixed;
    reg [7:0]      r_step;

    assign s_axi_arready = (r_state == R_IDLE);
    assign m_axi_araddr  = r_addr;
    assign m_axi_arvalid = (r_state == R_ADDR);
    assign s_axi_rvalid  = (r_state == R_DATA) && m_axi_rvalid;
    assign m_axi_rready  = (r_state == R_DATA) && s_axi_rready;
    assign s_axi_rdata   = m_axi_rdata;
    assign s_axi_rresp   = m_axi_rresp;
    assign s_axi_rid     = r_id;
    assign s_axi_rlast   = (r_left == 4'd0);

    always @(posedge aclk)
        if (!aresetn)
            r_state <= R_IDLE;
        else case (r_state)
            R_IDLE: if (s_axi_arvalid) begin
                r_id    <= s_axi_arid;
                r_addr  <= s_axi_araddr;
                r_left  <= s_axi_arlen;
                r_fixed <= (s_axi_arburst == 2'b00);
                r_step  <= 8'd1 << s_axi_arsize;
                r_state <= R_ADDR;
            end
            R_ADDR: if (m_axi_arready) r_state <= R_DATA;
            R_DATA: if (m_axi_rvalid && s_axi_rready) begin
                if (r_left == 4'd0)
                    r_state <= R_IDLE;
                else begin
                    r_left  <= r_left - 4'd1;
                    if (!r_fixed) r_addr <= r_addr + {{(AW-8){1'b0}}, r_step};
                    r_state <= R_ADDR;
                end
            end
            default: r_state <= R_IDLE;
        endcase
endmodule
