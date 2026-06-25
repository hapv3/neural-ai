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
    input  logic [DATA_WIDTH+31:0]  wfifo_data_i, // 256 data + 32 BE = 288 bits

    output logic                    idle_o
);

    logic        we_req;
    logic [31:0] we_addr;
    logic [31:0] we_be;
    logic [255:0] we_data;
    logic        we_gnt;

    logic        re_req;
    logic [31:0] re_addr;
    logic        re_gnt;

    logic        pending_valid_q, pending_valid_n;
    logic        pending_we_q,    pending_we_n;
    logic [31:0] pending_addr_q,  pending_addr_n;
    logic [31:0] pending_be_q,    pending_be_n;
    logic [255:0] pending_data_q, pending_data_n;
    logic        start_txn;
    logic        start_txn_we;
    logic [31:0] start_txn_addr;
    logic [31:0] start_txn_be;
    logic [255:0] start_txn_data;
    logic        txn_gnt;

    always_comb begin
        start_txn      = 1'b0;
        start_txn_we   = 1'b0;
        start_txn_addr = '0;
        start_txn_be   = 32'hFFFF_FFFF;
        start_txn_data = '0;

        if (we_req) begin
            start_txn      = 1'b1;
            start_txn_we   = 1'b1;
            start_txn_addr = we_addr;
            start_txn_be   = we_be;
            start_txn_data = we_data;
        end else if (re_req) begin
            start_txn      = 1'b1;
            start_txn_we   = 1'b0;
            start_txn_addr = re_addr;
            start_txn_be   = 32'hFFFF_FFFF;
            start_txn_data = '0;
        end

        pending_valid_n = pending_valid_q;
        pending_we_n    = pending_we_q;
        pending_addr_n  = pending_addr_q;
        pending_be_n    = pending_be_q;
        pending_data_n  = pending_data_q;

        if (txn_gnt) begin
            pending_valid_n = 1'b0;
        end

        if (!pending_valid_q && start_txn && !obi_m_gnt_i) begin
            pending_valid_n = 1'b1;
            pending_we_n    = start_txn_we;
            pending_addr_n  = start_txn_addr;
            pending_be_n    = start_txn_be;
            pending_data_n  = start_txn_data;
        end
    end

    assign obi_m_req_o   = pending_valid_q | start_txn;
    assign obi_m_we_o    = pending_valid_q ? pending_we_q    : start_txn_we;
    assign obi_m_addr_o  = pending_valid_q ? pending_addr_q  : start_txn_addr;
    assign obi_m_be_o    = pending_valid_q ? pending_be_q    : start_txn_be;
    assign obi_m_wdata_o = pending_valid_q ? pending_data_q  : start_txn_data;

    assign txn_gnt = obi_m_req_o & obi_m_gnt_i;
    assign we_gnt  = txn_gnt & obi_m_we_o;
    assign re_gnt  = txn_gnt & !obi_m_we_o;

    // Read Engine
    logic [31:0] re_addr_q, re_addr_n;
    logic [31:0] re_end_addr_q, re_end_addr_n;
    logic        re_active_q, re_active_n;
    logic        read_outstanding_q, read_outstanding_n;
    
    assign re_addr = re_addr_q;
    
    always_comb begin
        re_req        = 1'b0;
        re_addr_n     = re_addr_q;
        re_end_addr_n = re_end_addr_q;
        re_active_n   = re_active_q;
        read_outstanding_n = read_outstanding_q;

        if (obi_m_rvalid_i) begin
            read_outstanding_n = 1'b0;
        end
        
        if (cfg_start_i && cfg_length_i > 0) begin
            re_active_n   = 1'b1;
            re_addr_n     = cfg_src_ptr_i & ~32'h1F;
            re_end_addr_n = (cfg_src_ptr_i + cfg_length_i - 1) & ~32'h1F;
            read_outstanding_n = 1'b0;
        end else begin
            if (re_active_q) begin
                if (!rfifo_almost_full_i && !read_outstanding_q && !pending_valid_q) begin
                    re_req = 1'b1;
                end
                
                if (re_gnt) begin
                    read_outstanding_n = 1'b1;
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
            if (!wfifo_empty_i && !pending_valid_q) begin
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
            read_outstanding_q <= 1'b0;
            we_addr_q     <= '0;
            pending_valid_q <= 1'b0;
            pending_we_q    <= 1'b0;
            pending_addr_q  <= '0;
            pending_be_q    <= '0;
            pending_data_q  <= '0;
        end else begin
            re_addr_q     <= re_addr_n;
            re_end_addr_q <= re_end_addr_n;
            re_active_q   <= re_active_n;
            read_outstanding_q <= read_outstanding_n;
            we_addr_q     <= we_addr_n;
            pending_valid_q <= pending_valid_n;
            pending_we_q    <= pending_we_n;
            pending_addr_q  <= pending_addr_n;
            pending_be_q    <= pending_be_n;
            pending_data_q  <= pending_data_n;
        end
    end

    assign idle_o = !re_active_q &&
                    !read_outstanding_q &&
                    !pending_valid_q &&
                    !obi_m_req_o &&
                    wfifo_empty_i;

endmodule
