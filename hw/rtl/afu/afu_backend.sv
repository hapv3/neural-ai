// Copyright (c) 2026
// AFU Backend - DMA Engine for OBI Master

module afu_backend #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned BE_WIDTH   = 32
)(
    input  logic clk_i,
    input  logic rst_ni,
    
    // CSRs
    input  logic [31:0] cfg_src_ptr_i,
    input  logic [31:0] cfg_dst_ptr_i,
    input  logic [31:0] cfg_length_i,
    input  logic [1:0]  cfg_mode_i,
    input  logic        cfg_start_i,
    
    // OBI Master Interface
    output logic                    obi_m_req_o,
    input  logic                    obi_m_gnt_i,
    output logic [ADDR_WIDTH-1:0]   obi_m_addr_o,
    output logic                    obi_m_we_o,
    output logic [BE_WIDTH-1:0]     obi_m_be_o,
    output logic [DATA_WIDTH-1:0]   obi_m_wdata_o,
    input  logic                    obi_m_rvalid_i,
    input  logic [DATA_WIDTH-1:0]   obi_m_rdata_i,
    
    // Read FIFO Interface
    input  logic                    rfifo_almost_full_i,
    output logic                    rfifo_push_o,
    output logic [DATA_WIDTH-1:0]   rfifo_data_o,
    
    // Write FIFO Interface
    input  logic                    wfifo_empty_i,
    output logic                    wfifo_pop_o,
    input  logic [DATA_WIDTH+31:0]  wfifo_data_i // 256 data + 32 BE = 288 bits
);

    // Arbiter
    logic write_req;
    logic read_req;
    
    logic        we_req;
    logic [31:0] we_addr;
    logic [31:0] we_be;
    logic [255:0] we_data;
    logic        we_gnt;
    
    logic        re_req;
    logic [31:0] re_addr;
    logic        re_gnt;
    
    // OBI requires holding request until gnt
    logic hold_req_q;
    logic hold_we_q;
    
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            hold_req_q <= 1'b0;
            hold_we_q  <= 1'b0;
        end else begin
            if (obi_m_req_o && !obi_m_gnt_i) begin
                hold_req_q <= 1'b1;
                hold_we_q  <= obi_m_we_o;
            end else if (obi_m_gnt_i) begin
                hold_req_q <= 1'b0;
            end
        end
    end

    assign write_req = hold_req_q ? hold_we_q  : we_req;
    assign read_req  = hold_req_q ? !hold_we_q : (re_req && !we_req);
    
    assign obi_m_req_o   = write_req | read_req;
    assign obi_m_we_o    = write_req;
    assign obi_m_addr_o  = write_req ? we_addr : re_addr;
    assign obi_m_be_o    = write_req ? we_be   : 32'hFFFF_FFFF;
    assign obi_m_wdata_o = we_data;
    
    assign we_gnt = write_req & obi_m_gnt_i;
    assign re_gnt = read_req  & obi_m_gnt_i;

    // Read Engine
    logic [31:0] re_addr_q, re_addr_n;
    logic [31:0] re_end_addr_q, re_end_addr_n;
    logic        re_active_q, re_active_n;
    
    assign re_addr = re_addr_q;
    
    always_comb begin
        logic [31:0] byte_len;
        byte_len = '0;
        
        re_req        = 1'b0;
        re_addr_n     = re_addr_q;
        re_end_addr_n = re_end_addr_q;
        re_active_n   = re_active_q;
        
        if (cfg_start_i && cfg_length_i > 0) begin
            if (cfg_mode_i == 2'd0) byte_len = cfg_length_i;
            else if (cfg_mode_i == 2'd1) byte_len = {cfg_length_i[30:0], 1'b0};
            else byte_len = {cfg_length_i[29:0], 2'b00};
            
            re_active_n   = 1'b1;
            re_addr_n     = cfg_src_ptr_i & ~32'h1F;
            re_end_addr_n = (cfg_src_ptr_i + byte_len - 1) & ~32'h1F;
        end else begin
            if (re_active_q) begin
                if (!rfifo_almost_full_i) begin
                    re_req = 1'b1;
                end
                
                if (re_gnt) begin
                    re_addr_n = re_addr_q + 32;
                    if (re_addr_n > re_end_addr_q) begin
                        re_active_n = 1'b0;
                    end
                end
            end
        end
    end
    
    assign rfifo_push_o = obi_m_rvalid_i;
    assign rfifo_data_o = obi_m_rdata_i;

    // Write Engine
    logic [31:0] we_addr_q, we_addr_n;
    
    always_comb begin
        we_req      = 1'b0;
        we_addr_n   = we_addr_q;
        wfifo_pop_o = 1'b0;
        
        if (cfg_start_i) begin
            we_addr_n = cfg_dst_ptr_i & ~32'h1F;
        end else begin
            if (!wfifo_empty_i) begin
                we_req = 1'b1;
            end
            if (we_gnt) begin
                wfifo_pop_o = 1'b1;
                we_addr_n   = we_addr_q + 32;
            end
        end
        
        we_addr = we_addr_q;
        we_be   = wfifo_data_i[287:256];
        we_data = wfifo_data_i[255:0];
    end
    
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            re_addr_q     <= '0;
            re_end_addr_q <= '0;
            re_active_q   <= 1'b0;
            we_addr_q     <= '0;
        end else begin
            re_addr_q     <= re_addr_n;
            re_end_addr_q <= re_end_addr_n;
            re_active_q   <= re_active_n;
            we_addr_q     <= we_addr_n;
        end
    end

endmodule
