`default_nettype none

//-------------------------------------------------------------------------
// NPU DMA Engine
// Vai trò: Trực tiếp Copy dữ liệu giữa AXI4 (L2/DRAM ngoài) và OBI (TCDM L1 nội bộ).
// Hỗ trợ AXI Burst Transfer để đạt thông lượng đọc/ghi liên tục.
//-------------------------------------------------------------------------
module dma_engine #(
    parameter int unsigned AXI_ADDR_WIDTH = 32,
    parameter int unsigned AXI_DATA_WIDTH = 256,
    parameter int unsigned AXI_ID_WIDTH   = 4,
    parameter int unsigned OBI_ADDR_WIDTH = 32,
    parameter int unsigned OBI_DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    //---------------------------------------------------
    // Control Interface (Từ Snitch Core cấu hình)
    //---------------------------------------------------
    input  logic [31:0] cfg_src_addr_i,
    input  logic [31:0] cfg_dst_addr_i,
    input  logic [31:0] cfg_length_i,   // Số lượng bytes cần truyền
    input  logic        cfg_start_i,
    output logic        cfg_done_o,

    //---------------------------------------------------
    // AXI4 Master Interface (Nối ra L2/DRAM Interconnect)
    //---------------------------------------------------
    // Address Write Channel
    output logic [AXI_ADDR_WIDTH-1:0] axi_aw_addr_o,
    output logic [7:0]                axi_aw_len_o,
    output logic [2:0]                axi_aw_size_o,
    output logic [1:0]                axi_aw_burst_o,
    output logic                      axi_aw_valid_o,
    input  logic                      axi_aw_ready_i,

    // Write Channel
    output logic [AXI_DATA_WIDTH-1:0] axi_w_data_o,
    output logic [(AXI_DATA_WIDTH/8)-1:0] axi_w_strb_o,
    output logic                      axi_w_last_o,
    output logic                      axi_w_valid_o,
    input  logic                      axi_w_ready_i,

    // Write Response Channel
    input  logic [1:0]                axi_b_resp_i,
    input  logic                      axi_b_valid_i,
    output logic                      axi_b_ready_o,

    // Address Read Channel
    output logic [AXI_ADDR_WIDTH-1:0] axi_ar_addr_o,
    output logic [7:0]                axi_ar_len_o,
    output logic [2:0]                axi_ar_size_o,
    output logic [1:0]                axi_ar_burst_o,
    output logic                      axi_ar_valid_o,
    input  logic                      axi_ar_ready_i,

    // Read Channel
    input  logic [AXI_DATA_WIDTH-1:0] axi_r_data_i,
    input  logic [1:0]                axi_r_resp_i,
    input  logic                      axi_r_last_i,
    input  logic                      axi_r_valid_i,
    output logic                      axi_r_ready_o,

    //---------------------------------------------------
    // OBI Master Interface (Nối vào D-TCDM Interconnect)
    //---------------------------------------------------
    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [OBI_ADDR_WIDTH-1:0] obi_addr_o,
    output logic                      obi_we_o,
    output logic [(OBI_DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [OBI_DATA_WIDTH-1:0] obi_wdata_o,
    input  logic                      obi_rvalid_i,
    input  logic [OBI_DATA_WIDTH-1:0] obi_rdata_i
);

    // Basic FSM for DMA Read from AXI (L2) to OBI (L1)
    // Trong thực tế, DMA Engine sẽ có 2 luồng độc lập: L2->L1 và L1->L2.
    // Đoạn code này chỉ minh hoạ State Machine đơn giản cho luồng L2->L1.

    typedef enum logic [2:0] {
        IDLE,
        AXI_READ_REQ,
        AXI_READ_WAIT,
        OBI_WRITE,
        DONE
    } state_e;

    state_e state_d, state_q;

    logic [31:0] src_addr_q, dst_addr_q, bytes_left_q;
    logic [255:0] data_buffer_q;

    always_comb begin
        state_d = state_q;
        cfg_done_o = 1'b0;

        // Default AXI signals
        axi_ar_valid_o = 1'b0;
        axi_ar_addr_o  = src_addr_q;
        axi_ar_len_o   = 8'd0; // 1 beat per request (Simplified)
        axi_ar_size_o  = 3'b101; // 32 bytes = 256 bits per beat
        axi_ar_burst_o = 2'b01; // INCR
        axi_r_ready_o  = 1'b0;

        // Default OBI signals
        obi_req_o   = 1'b0;
        obi_addr_o  = dst_addr_q;
        obi_we_o    = 1'b0;
        obi_be_o    = { (OBI_DATA_WIDTH/8) {1'b1} };
        obi_wdata_o = data_buffer_q;

        // Unused Write channel tie-offs
        axi_aw_valid_o = 1'b0;
        axi_aw_addr_o  = '0;
        axi_aw_len_o   = '0;
        axi_aw_size_o  = '0;
        axi_aw_burst_o = '0;
        axi_w_valid_o  = 1'b0;
        axi_w_data_o   = '0;
        axi_w_strb_o   = '0;
        axi_w_last_o   = 1'b0;
        axi_b_ready_o  = 1'b1;

        case (state_q)
            IDLE: begin
                if (cfg_start_i && cfg_length_i > 0) begin
                    state_d = AXI_READ_REQ;
                end
            end

            AXI_READ_REQ: begin
                axi_ar_valid_o = 1'b1;
                if (axi_ar_ready_i) begin
                    state_d = AXI_READ_WAIT;
                end
            end

            AXI_READ_WAIT: begin
                axi_r_ready_o = 1'b1;
                if (axi_r_valid_i) begin
                    state_d = OBI_WRITE;
                end
            end

            OBI_WRITE: begin
                obi_req_o = 1'b1;
                obi_we_o  = 1'b1; // Memory Write into L1
                if (obi_gnt_i) begin
                    if (bytes_left_q <= 32) begin
                        state_d = DONE;
                    end else begin
                        state_d = AXI_READ_REQ;
                    end
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
        end else begin
            state_q <= state_d;

            if (state_q == IDLE && cfg_start_i) begin
                src_addr_q   <= cfg_src_addr_i;
                dst_addr_q   <= cfg_dst_addr_i;
                bytes_left_q <= cfg_length_i;
            end else if (state_q == AXI_READ_WAIT && axi_r_valid_i && axi_r_ready_o) begin
                data_buffer_q <= axi_r_data_i;
            end else if (state_q == OBI_WRITE && obi_gnt_i) begin
                src_addr_q   <= src_addr_q + 32;
                dst_addr_q   <= dst_addr_q + 32;
                bytes_left_q <= bytes_left_q - 32;
            end
        end
    end

endmodule
