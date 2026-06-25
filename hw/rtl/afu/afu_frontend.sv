// Copyright (c) 2026
// AFU Frontend - CSRs and OBI Target Interface

module afu_frontend #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic                    clk_i,
    input  logic                    rst_ni,

    // OBI target interface
    input  logic                    obi_s_req_i,
    output logic                    obi_s_gnt_o,
    input  logic [ADDR_WIDTH-1:0]   obi_s_addr_i,
    input  logic                    obi_s_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] obi_s_be_i,
    input  logic [DATA_WIDTH-1:0]   obi_s_wdata_i,
    output logic                    obi_s_rvalid_o,
    output logic [DATA_WIDTH-1:0]   obi_s_rdata_o,

    // CSR Outputs
    output logic [31:0]             cfg_src_ptr_o,
    output logic [31:0]             cfg_dst_ptr_o,
    output logic [31:0]             cfg_length_o,
    output logic [1:0]              cfg_mode_o,
    output logic                    cfg_start_o,
    input  logic                    afu_done_i,
    input  logic                    afu_busy_i,
    input  logic                    afu_error_i,
    
    // LUT write interface
    output logic                    lut_we_o,
    output logic [7:0]              lut_addr_o,
    output logic [31:0]             lut_wdata_o,
    output logic [3:0]              lut_be_o
);

    logic [31:0] cfg_src_ptr_q,  cfg_src_ptr_n;
    logic [31:0] cfg_dst_ptr_q,  cfg_dst_ptr_n;
    logic [31:0] cfg_length_q,   cfg_length_n;
    logic [1:0]  cfg_mode_q,     cfg_mode_n;
    logic        cfg_start_q,    cfg_start_n;

    logic        obi_s_rvalid_q;
    logic [31:0] obi_s_rdata_q;
    logic        lut_sel;
    logic        csr_sel;

    assign obi_s_gnt_o    = 1'b1;
    assign obi_s_rvalid_o = obi_s_rvalid_q;
    assign obi_s_rdata_o  = obi_s_rdata_q;

    assign cfg_src_ptr_o = cfg_src_ptr_q;
    assign cfg_dst_ptr_o = cfg_dst_ptr_q;
    assign cfg_length_o  = cfg_length_q;
    assign cfg_mode_o    = cfg_mode_q;
    assign cfg_start_o   = cfg_start_q;

    assign lut_sel = (obi_s_addr_i[15:12] == 4'h0) && (obi_s_addr_i[11:10] == 2'b00);
    assign csr_sel = (obi_s_addr_i[11:10] == 2'b01) || (obi_s_addr_i[15:12] == 4'h1);

    function automatic logic [31:0] apply_cfg_be(
        input logic [31:0] current,
        input logic [31:0] written,
        input logic [3:0]  be
    );
        logic [31:0] result;
        result[7:0]   = be[0] ? written[7:0]   : current[7:0];
        result[15:8]  = be[1] ? written[15:8]  : current[15:8];
        result[23:16] = be[2] ? written[23:16] : current[23:16];
        result[31:24] = be[3] ? written[31:24] : current[31:24];
        return result;
    endfunction

    always_comb begin
        cfg_src_ptr_n = cfg_src_ptr_q;
        cfg_dst_ptr_n = cfg_dst_ptr_q;
        cfg_length_n  = cfg_length_q;
        cfg_mode_n    = cfg_mode_q;
        cfg_start_n   = 1'b0; // start is a 1-cycle pulse
        
        lut_we_o      = 1'b0;
        lut_addr_o    = obi_s_addr_i[9:2];
        lut_wdata_o   = obi_s_wdata_i;
        lut_be_o      = obi_s_be_i;

        if (obi_s_req_i) begin
            if (obi_s_we_i) begin
                if (lut_sel) begin
                    lut_we_o = 1'b1;
                end else if (csr_sel) begin
                    unique case (obi_s_addr_i[5:0])
                        6'h00: cfg_start_n   = obi_s_wdata_i[0];
                        6'h04: cfg_src_ptr_n = apply_cfg_be(cfg_src_ptr_q, obi_s_wdata_i, obi_s_be_i);
                        6'h08: cfg_dst_ptr_n = apply_cfg_be(cfg_dst_ptr_q, obi_s_wdata_i, obi_s_be_i);
                        6'h0c: cfg_length_n  = apply_cfg_be(cfg_length_q, obi_s_wdata_i, obi_s_be_i);
                        6'h10: cfg_mode_n    = apply_cfg_be({30'd0, cfg_mode_q}, obi_s_wdata_i, obi_s_be_i)[1:0];
                        default: ;
                    endcase
                end
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            obi_s_rvalid_q <= 1'b0;
            obi_s_rdata_q  <= '0;
            cfg_src_ptr_q  <= '0;
            cfg_dst_ptr_q  <= '0;
            cfg_length_q   <= '0;
            cfg_mode_q     <= '0;
            cfg_start_q    <= 1'b0;
        end else begin
            obi_s_rvalid_q <= obi_s_req_i;
            obi_s_rdata_q  <= '0;
            
            if (obi_s_req_i && !obi_s_we_i) begin
                if (csr_sel) begin
                    unique case (obi_s_addr_i[5:0])
                        6'h00: obi_s_rdata_q <= {29'd0, afu_error_i, afu_busy_i, afu_done_i};
                        6'h04: obi_s_rdata_q <= cfg_src_ptr_q;
                        6'h08: obi_s_rdata_q <= cfg_dst_ptr_q;
                        6'h0c: obi_s_rdata_q <= cfg_length_q;
                        6'h10: obi_s_rdata_q <= {30'd0, cfg_mode_q};
                        default: obi_s_rdata_q <= '0;
                    endcase
                end
            end

            cfg_src_ptr_q <= cfg_src_ptr_n;
            cfg_dst_ptr_q <= cfg_dst_ptr_n;
            cfg_length_q  <= cfg_length_n;
            cfg_mode_q    <= cfg_mode_n;
            cfg_start_q   <= cfg_start_n;
        end
    end

endmodule
