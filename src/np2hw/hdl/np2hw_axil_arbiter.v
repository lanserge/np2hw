// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_axil_arbiter -- two AXI4-Lite masters onto one slave.
//
// A register file with two ways in -- the processor's bus and a second
// master, say a bridge from a display cable's DDC lines -- takes one
// transaction at a time from either. Writes and reads are arbitrated
// apart: each direction grants one master its whole transaction, address
// to response, and alternates when both ask, so neither can starve the
// other. The granted master is connected straight through; nothing is
// buffered, so the slave's timing is the master's.
module np2hw_axil_arbiter #(
    parameter AW = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s0_axi:s1_axi:m_axi, ASSOCIATED_RESET aresetn" *)
    input  wire          aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire          aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE" *)
    input  wire [AW-1:0] s0_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi AWVALID" *)
    input  wire          s0_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi AWREADY" *)
    output wire          s0_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi WDATA" *)
    input  wire [31:0]   s0_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi WSTRB" *)
    input  wire [3:0]    s0_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi WVALID" *)
    input  wire          s0_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi WREADY" *)
    output wire          s0_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi BRESP" *)
    output wire [1:0]    s0_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi BVALID" *)
    output wire          s0_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi BREADY" *)
    input  wire          s0_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi ARADDR" *)
    input  wire [AW-1:0] s0_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi ARVALID" *)
    input  wire          s0_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi ARREADY" *)
    output wire          s0_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi RDATA" *)
    output wire [31:0]   s0_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi RRESP" *)
    output wire [1:0]    s0_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi RVALID" *)
    output wire          s0_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s0_axi RREADY" *)
    input  wire          s0_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE" *)
    input  wire [AW-1:0] s1_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi AWVALID" *)
    input  wire          s1_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi AWREADY" *)
    output wire          s1_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi WDATA" *)
    input  wire [31:0]   s1_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi WSTRB" *)
    input  wire [3:0]    s1_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi WVALID" *)
    input  wire          s1_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi WREADY" *)
    output wire          s1_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi BRESP" *)
    output wire [1:0]    s1_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi BVALID" *)
    output wire          s1_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi BREADY" *)
    input  wire          s1_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi ARADDR" *)
    input  wire [AW-1:0] s1_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi ARVALID" *)
    input  wire          s1_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi ARREADY" *)
    output wire          s1_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi RDATA" *)
    output wire [31:0]   s1_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi RRESP" *)
    output wire [1:0]    s1_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi RVALID" *)
    output wire          s1_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s1_axi RREADY" *)
    input  wire          s1_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE" *)
    output wire [AW-1:0] m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWVALID" *)
    output wire          m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi AWREADY" *)
    input  wire          m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WDATA" *)
    output wire [31:0]   m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WSTRB" *)
    output wire [3:0]    m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WVALID" *)
    output wire          m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi WREADY" *)
    input  wire          m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BRESP" *)
    input  wire [1:0]    m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BVALID" *)
    input  wire          m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi BREADY" *)
    output wire          m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARADDR" *)
    output wire [AW-1:0] m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARVALID" *)
    output wire          m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi ARREADY" *)
    input  wire          m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RDATA" *)
    input  wire [31:0]   m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RRESP" *)
    input  wire [1:0]    m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RVALID" *)
    input  wire          m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi RREADY" *)
    output wire          m_axi_rready
);
    // -- writes --------------------------------------------------------- //
    localparam [1:0] W_IDLE = 2'd0, W_ADDR = 2'd1, W_RESP = 2'd2;
    reg [1:0] w_state;
    reg       w_grant;        // the master holding the write path
    reg       w_last;         // the master granted last, for fairness
    reg       aw_done, w_done;

    wire w0 = s0_axi_awvalid, w1 = s1_axi_awvalid;
    // both asking: the one that did not go last; else whoever asks
    wire w_pick = (w0 && w1) ? !w_last : w1;

    wire g_awvalid = w_grant ? s1_axi_awvalid : s0_axi_awvalid;
    wire g_wvalid  = w_grant ? s1_axi_wvalid  : s0_axi_wvalid;
    wire g_bready  = w_grant ? s1_axi_bready  : s0_axi_bready;

    assign m_axi_awaddr  = w_grant ? s1_axi_awaddr : s0_axi_awaddr;
    assign m_axi_wdata   = w_grant ? s1_axi_wdata  : s0_axi_wdata;
    assign m_axi_wstrb   = w_grant ? s1_axi_wstrb  : s0_axi_wstrb;
    assign m_axi_awvalid = (w_state == W_ADDR) && !aw_done && g_awvalid;
    assign m_axi_wvalid  = (w_state == W_ADDR) && !w_done  && g_wvalid;
    assign m_axi_bready  = (w_state == W_RESP) && g_bready;

    wire aw_ok = (w_state == W_ADDR) && !aw_done && m_axi_awready;
    wire w_ok  = (w_state == W_ADDR) && !w_done  && m_axi_wready;
    wire b_ok  = (w_state == W_RESP) && m_axi_bvalid;
    assign s0_axi_awready = !w_grant && aw_ok;
    assign s1_axi_awready =  w_grant && aw_ok;
    assign s0_axi_wready  = !w_grant && w_ok;
    assign s1_axi_wready  =  w_grant && w_ok;
    assign s0_axi_bvalid  = !w_grant && b_ok;
    assign s1_axi_bvalid  =  w_grant && b_ok;
    assign s0_axi_bresp   = m_axi_bresp;
    assign s1_axi_bresp   = m_axi_bresp;

    wire aw_hs = m_axi_awvalid && m_axi_awready;
    wire w_hs  = m_axi_wvalid  && m_axi_wready;

    always @(posedge aclk)
        if (!aresetn) begin
            w_state <= W_IDLE;
            w_grant <= 1'b0;
            w_last  <= 1'b1;
            aw_done <= 1'b0;
            w_done  <= 1'b0;
        end else case (w_state)
            W_IDLE: if (w0 || w1) begin
                w_grant <= w_pick;
                w_last  <= w_pick;
                aw_done <= 1'b0;
                w_done  <= 1'b0;
                w_state <= W_ADDR;
            end
            W_ADDR: begin
                if (aw_hs) aw_done <= 1'b1;
                if (w_hs)  w_done  <= 1'b1;
                if ((aw_done || aw_hs) && (w_done || w_hs))
                    w_state <= W_RESP;
            end
            W_RESP: if (m_axi_bvalid && m_axi_bready) w_state <= W_IDLE;
            default: w_state <= W_IDLE;
        endcase

    // -- reads ---------------------------------------------------------- //
    localparam [1:0] R_IDLE = 2'd0, R_ADDR = 2'd1, R_DATA = 2'd2;
    reg [1:0] r_state;
    reg       r_grant;
    reg       r_last;

    wire r0 = s0_axi_arvalid, r1 = s1_axi_arvalid;
    wire r_pick = (r0 && r1) ? !r_last : r1;

    wire g_arvalid = r_grant ? s1_axi_arvalid : s0_axi_arvalid;
    wire g_rready  = r_grant ? s1_axi_rready  : s0_axi_rready;

    assign m_axi_araddr  = r_grant ? s1_axi_araddr : s0_axi_araddr;
    assign m_axi_arvalid = (r_state == R_ADDR) && g_arvalid;
    assign m_axi_rready  = (r_state == R_DATA) && g_rready;

    wire ar_ok = (r_state == R_ADDR) && m_axi_arready;
    wire r_ok  = (r_state == R_DATA) && m_axi_rvalid;
    assign s0_axi_arready = !r_grant && ar_ok;
    assign s1_axi_arready =  r_grant && ar_ok;
    assign s0_axi_rvalid  = !r_grant && r_ok;
    assign s1_axi_rvalid  =  r_grant && r_ok;
    assign s0_axi_rdata   = m_axi_rdata;
    assign s1_axi_rdata   = m_axi_rdata;
    assign s0_axi_rresp   = m_axi_rresp;
    assign s1_axi_rresp   = m_axi_rresp;

    always @(posedge aclk)
        if (!aresetn) begin
            r_state <= R_IDLE;
            r_grant <= 1'b0;
            r_last  <= 1'b1;
        end else case (r_state)
            R_IDLE: if (r0 || r1) begin
                r_grant <= r_pick;
                r_last  <= r_pick;
                r_state <= R_ADDR;
            end
            R_ADDR: if (m_axi_arvalid && m_axi_arready) r_state <= R_DATA;
            R_DATA: if (m_axi_rvalid && m_axi_rready) r_state <= R_IDLE;
            default: r_state <= R_IDLE;
        endcase
endmodule
