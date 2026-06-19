`default_nettype none

module dma_engine #(
    parameter int unsigned AXI_ADDR_WIDTH = 32,
    parameter int unsigned AXI_DATA_WIDTH = 256,
    parameter int unsigned AXI_ID_WIDTH   = 4,
    parameter int unsigned OBI_ADDR_WIDTH = 32,
    parameter int unsigned OBI_DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Control Interface
    input  logic [31:0] cfg_src_addr_i,
    input  logic [31:0] cfg_dst_addr_i,
    input  logic [31:0] cfg_length_i,
    input  logic        cfg_start_i,
    output logic        cfg_done_o,

    // AXI4 Master Interface
    output logic [AXI_ADDR_WIDTH-1:0] axi_aw_addr_o,
    output logic [7:0]                axi_aw_len_o,
    output logic [2:0]                axi_aw_size_o,
    output logic [1:0]                axi_aw_burst_o,
    output logic                      axi_aw_valid_o,
    input  logic                      axi_aw_ready_i,

    output logic [AXI_DATA_WIDTH-1:0] axi_w_data_o,
    output logic [(AXI_DATA_WIDTH/8)-1:0] axi_w_strb_o,
    output logic                      axi_w_last_o,
    output logic                      axi_w_valid_o,
    input  logic                      axi_w_ready_i,

    input  logic [1:0]                axi_b_resp_i,
    input  logic                      axi_b_valid_i,
    output logic                      axi_b_ready_o,

    output logic [AXI_ADDR_WIDTH-1:0] axi_ar_addr_o,
    output logic [7:0]                axi_ar_len_o,
    output logic [2:0]                axi_ar_size_o,
    output logic [1:0]                axi_ar_burst_o,
    output logic                      axi_ar_valid_o,
    input  logic                      axi_ar_ready_i,

    input  logic [AXI_DATA_WIDTH-1:0] axi_r_data_i,
    input  logic [1:0]                axi_r_resp_i,
    input  logic                      axi_r_last_i,
    input  logic                      axi_r_valid_i,
    output logic                      axi_r_ready_o,

    // OBI Master Interface
    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [OBI_ADDR_WIDTH-1:0] obi_addr_o,
    output logic                      obi_we_o,
    output logic [(OBI_DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [OBI_DATA_WIDTH-1:0] obi_wdata_o,
    input  logic                      obi_rvalid_i,
    input  logic [OBI_DATA_WIDTH-1:0] obi_rdata_i
);

    typedef enum logic [3:0] {
        IDLE,
        AXI_READ_REQ,
        AXI_READ_WAIT,
        OBI_WRITE,
        OBI_READ_REQ,
        OBI_READ_WAIT,
        AXI_WRITE_ADDR,
        AXI_WRITE_DATA,
        AXI_WRITE_RESP,
        DONE
    } state_e;

    state_e state_d, state_q;

    logic [31:0] src_addr_q, dst_addr_q, bytes_left_q;
    logic [255:0] data_buffer_q;
    logic is_l1_src_q;
    logic is_l1_dst_q;

    always_comb begin
        state_d = state_q;
        cfg_done_o = 1'b0;

        // Default AXI Read signals
        axi_ar_valid_o = 1'b0;
        axi_ar_addr_o  = src_addr_q;
        axi_ar_len_o   = 8'd0;
        axi_ar_size_o  = 3'b101; // 32 bytes
        axi_ar_burst_o = 2'b01; // INCR
        axi_r_ready_o  = 1'b0;

        // Default AXI Write signals
        axi_aw_valid_o = 1'b0;
        axi_aw_addr_o  = dst_addr_q;
        axi_aw_len_o   = 8'd0;
        axi_aw_size_o  = 3'b101;
        axi_aw_burst_o = 2'b01;
        
        axi_w_valid_o  = 1'b0;
        axi_w_data_o   = data_buffer_q;
        axi_w_strb_o   = { (AXI_DATA_WIDTH/8) {1'b1} };
        axi_w_last_o   = 1'b1;
        
        axi_b_ready_o  = 1'b1;

        // Default OBI signals
        obi_req_o   = 1'b0;
        obi_addr_o  = dst_addr_q;
        obi_we_o    = 1'b0;
        obi_be_o    = { (OBI_DATA_WIDTH/8) {1'b1} };
        obi_wdata_o = data_buffer_q;

        case (state_q)
            IDLE: begin
                if (cfg_start_i) begin
                    if (cfg_length_i > 0) begin
                        if (cfg_src_addr_i[31:24] == 8'h10) begin
                            state_d = OBI_READ_REQ; // L1 -> L2
                        end else begin
                            state_d = AXI_READ_REQ; // L2 -> L1
                        end
                    end
                end
            end

            // ==========================================
            // PATH 1: L2 -> L1 (AXI Read -> OBI Write)
            // ==========================================
            AXI_READ_REQ: begin
                axi_ar_valid_o = 1'b1;
                if (axi_ar_ready_i) begin
                    state_d = AXI_READ_WAIT;
                end
            end

            AXI_READ_WAIT: begin
                axi_r_ready_o = 1'b1;
                if (axi_r_valid_i) begin
                    if (is_l1_dst_q) state_d = OBI_WRITE;
                    else state_d = AXI_WRITE_ADDR;
                end
            end

            OBI_WRITE: begin
                obi_req_o  = 1'b1;
                obi_we_o   = 1'b1;
                obi_addr_o = dst_addr_q;
                if (obi_gnt_i) begin
                    if (bytes_left_q <= 32) state_d = DONE;
                    else if (is_l1_src_q) state_d = OBI_READ_REQ;
                    else state_d = AXI_READ_REQ;
                end
            end

            // ==========================================
            // PATH 2: L1 -> L2 (OBI Read -> AXI Write)
            // ==========================================
            OBI_READ_REQ: begin
                obi_req_o  = 1'b1;
                obi_we_o   = 1'b0; // Read
                obi_addr_o = src_addr_q;
                if (obi_gnt_i) begin
                    state_d = OBI_READ_WAIT;
                end
            end

            OBI_READ_WAIT: begin
                if (obi_rvalid_i) begin
                    if (is_l1_dst_q) state_d = OBI_WRITE;
                    else state_d = AXI_WRITE_ADDR;
                end
            end

            AXI_WRITE_ADDR: begin
                axi_aw_valid_o = 1'b1;
                if (axi_aw_ready_i) begin
                    state_d = AXI_WRITE_DATA;
                end
            end

            AXI_WRITE_DATA: begin
                axi_w_valid_o = 1'b1;
                if (axi_w_ready_i) begin
                    state_d = AXI_WRITE_RESP;
                end
            end

            AXI_WRITE_RESP: begin
                // axi_b_ready_o is default 1
                if (axi_b_valid_i) begin
                    if (bytes_left_q <= 32) state_d = DONE;
                    else if (is_l1_src_q) state_d = OBI_READ_REQ;
                    else state_d = AXI_READ_REQ;
                end
            end

            DONE: begin
                cfg_done_o = 1'b1;
                state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q       <= IDLE;
            src_addr_q    <= '0;
            dst_addr_q    <= '0;
            bytes_left_q  <= '0;
            data_buffer_q <= '0;
            is_l1_src_q   <= 1'b0;
            is_l1_dst_q   <= 1'b0;
        end else begin
            state_q <= state_d;

            if (state_q == IDLE && cfg_start_i) begin
                src_addr_q   <= cfg_src_addr_i;
                dst_addr_q   <= cfg_dst_addr_i;
                bytes_left_q <= cfg_length_i;
                is_l1_src_q  <= (cfg_src_addr_i[31:24] == 8'h10);
                is_l1_dst_q  <= (cfg_dst_addr_i[31:24] == 8'h10);
            end else if (state_q == AXI_READ_WAIT && axi_r_valid_i && axi_r_ready_o) begin
                data_buffer_q <= axi_r_data_i;
            end else if (state_q == OBI_WRITE && obi_gnt_i) begin
                src_addr_q   <= src_addr_q + 32;
                dst_addr_q   <= dst_addr_q + 32;
                bytes_left_q <= bytes_left_q - 32;
            end else if (state_q == OBI_READ_WAIT && obi_rvalid_i) begin
                data_buffer_q <= obi_rdata_i;
            end else if (state_q == AXI_WRITE_RESP && axi_b_valid_i) begin
                src_addr_q   <= src_addr_q + 32;
                dst_addr_q   <= dst_addr_q + 32;
                bytes_left_q <= bytes_left_q - 32;
            end
        end
    end

endmodule
