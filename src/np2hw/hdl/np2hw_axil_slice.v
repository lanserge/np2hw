// Copyright 2026 Serge Rabyking
// SPDX-License-Identifier: MIT
//
// np2hw_axil_slice -- an AXI4-Lite register slice: every channel through
// an np2hw_skid, so every port of both buses leaves or enters a flop.
//
// A register bus is combinational from end to end -- a processor's port,
// a bridge, a decoder, an arbiter, a register file's write decode -- and
// each piece meets timing alone. Put together they are one path, starting
// at whatever drives the bus (a processor's port, slow out of its hard
// block). A slice ends that path where it is placed. It costs a cycle each
// way and no throughput: each skid holds one beat in reserve, so both
// valid and ready are registered and nothing waits a cycle to be taken.
module np2hw_axil_slice #(
    parameter AW = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:m_axi, ASSOCIATED_RESET aresetn" *)
    input  wire          aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire          aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE" *)
    input  wire [AW-1:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire          s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output wire          s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [31:0]   s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [3:0]    s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire          s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output wire          s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]    s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output wire          s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire          s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [AW-1:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire          s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output wire          s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output wire [31:0]   s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]    s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output wire          s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire          s_axi_rready,
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
    wire rst = !aresetn;

    // requests travel s -> m
    np2hw_skid #(.W(AW)) aw (
        .clk(aclk), .rst(rst),
        .s_data(s_axi_awaddr), .s_valid(s_axi_awvalid),
        .s_ready(s_axi_awready),
        .m_data(m_axi_awaddr), .m_valid(m_axi_awvalid),
        .m_ready(m_axi_awready));
    np2hw_skid #(.W(36)) w (
        .clk(aclk), .rst(rst),
        .s_data({s_axi_wstrb, s_axi_wdata}), .s_valid(s_axi_wvalid),
        .s_ready(s_axi_wready),
        .m_data({m_axi_wstrb, m_axi_wdata}), .m_valid(m_axi_wvalid),
        .m_ready(m_axi_wready));
    np2hw_skid #(.W(AW)) ar (
        .clk(aclk), .rst(rst),
        .s_data(s_axi_araddr), .s_valid(s_axi_arvalid),
        .s_ready(s_axi_arready),
        .m_data(m_axi_araddr), .m_valid(m_axi_arvalid),
        .m_ready(m_axi_arready));
    // responses travel m -> s
    np2hw_skid #(.W(2)) b (
        .clk(aclk), .rst(rst),
        .s_data(m_axi_bresp), .s_valid(m_axi_bvalid),
        .s_ready(m_axi_bready),
        .m_data(s_axi_bresp), .m_valid(s_axi_bvalid),
        .m_ready(s_axi_bready));
    np2hw_skid #(.W(34)) r (
        .clk(aclk), .rst(rst),
        .s_data({m_axi_rresp, m_axi_rdata}), .s_valid(m_axi_rvalid),
        .s_ready(m_axi_rready),
        .m_data({s_axi_rresp, s_axi_rdata}), .m_valid(s_axi_rvalid),
        .m_ready(s_axi_rready));
endmodule
