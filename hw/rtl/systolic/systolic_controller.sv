`default_nettype none

module systolic_controller #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Configuration from MMIO
    input  logic                      cfg_sys_start_i,
    input  logic [31:0]               cfg_sys_weight_ptr_i,
    input  logic [31:0]               cfg_sys_ifm_ptr_i,
    input  logic [31:0]               cfg_sys_ofm_ptr_i,
    input  logic [31:0]               cfg_sys_dim_m_i, // Number of IFM rows (skewed)
    output logic                      cfg_sys_done_o,

    // 1x OBI Master for I-TCDM (Read Weights & IFM)
    output logic                      obi_i_req_o,
    input  logic                      obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_i_addr_o,
    output logic                      obi_i_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_i_be_o,
    output logic [DATA_WIDTH-1:0]     obi_i_wdata_o,
    input  logic                      obi_i_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_i_rdata_i,

    // 4x OBI Masters for O-TCDM (Write OFM)
    output logic [3:0]                obi_o_req_o,
    input  logic [3:0]                obi_o_gnt_i,
    output logic [3:0][ADDR_WIDTH-1:0]obi_o_addr_o,
    output logic [3:0]                obi_o_we_o,
    output logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o,
    output logic [3:0][DATA_WIDTH-1:0]obi_o_wdata_o,
    input  logic [3:0]                obi_o_rvalid_i,
    input  logic [3:0][DATA_WIDTH-1:0]obi_o_rdata_i,

    // Systolic Array Interface
    output logic                      weight_load_en_o,
    output logic                      clear_acc_o,
    output logic                      compute_en_o,
    output logic [31:0][7:0]          weight_data_o,
    output logic [31:0][7:0]          ifm_data_o,
    output logic [31:0][31:0]         psum_data_o,
    input  logic [31:0][31:0]         ofm_data_i,
    input  logic                      ofm_valid_i
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        WAIT_DRAIN,
        DONE
    } state_e;

    state_e state_q, state_d;

    logic [31:0] w_ptr_q, w_ptr_d;
    logic [31:0] i_ptr_q, i_ptr_d;
    logic [31:0] o_ptr_q, o_ptr_d;
    logic [31:0] req_cnt_q, req_cnt_d; // Counter for requests
    logic [31:0] rsp_cnt_q, rsp_cnt_d; // Counter for responses
    logic [31:0] drain_cnt_q, drain_cnt_d; // Counter for valid outputs

    logic [DATA_WIDTH-1:0] rdata_buf_q;

    // OFM output buffer for backpressure handling (I3)
    logic [31:0][31:0]     ofm_buf_q;
    logic                  ofm_buf_valid_q;

    // The Systolic Array has a 32-cycle latency.
    localparam int unsigned SYS_LATENCY = 32;
    localparam int unsigned ARRAY_DIM = 32;

    // Tie off unused
    assign psum_data_o = '0;
    assign obi_i_we_o = 1'b0;
    assign obi_i_be_o = '1;
    assign obi_i_wdata_o = '0;

    // 4 OBI write ports are always write-only
    for (genvar i = 0; i < 4; i++) begin : gen_obi_o
        assign obi_o_we_o[i] = 1'b1;
        assign obi_o_be_o[i] = '1;
    end

    // FSM
    always_comb begin
        state_d = state_q;
        w_ptr_d = w_ptr_q;
        i_ptr_d = i_ptr_q;
        o_ptr_d = o_ptr_q;
        req_cnt_d   = req_cnt_q;
        rsp_cnt_d   = rsp_cnt_q;
        drain_cnt_d = drain_cnt_q;

        cfg_sys_done_o = 1'b0;

        obi_i_req_o = 1'b0;
        obi_i_addr_o = '0;

        weight_load_en_o = 1'b0;
        compute_en_o     = 1'b0;
        clear_acc_o      = 1'b0;
        weight_data_o    = '0;  // Only driven during LOAD_WEIGHTS
        ifm_data_o       = '0;  // Only driven during COMPUTE

        // OBI Output ports (Write)
        obi_o_req_o = '0;
        obi_o_addr_o[0] = o_ptr_q + 0;
        obi_o_addr_o[1] = o_ptr_q + 32;
        obi_o_addr_o[2] = o_ptr_q + 64;
        obi_o_addr_o[3] = o_ptr_q + 96;
        
        obi_o_wdata_o[0] = ofm_buf_q[ 7: 0]; // Elements 0-7 (256-bit)
        obi_o_wdata_o[1] = ofm_buf_q[15: 8]; // Elements 8-15
        obi_o_wdata_o[2] = ofm_buf_q[23:16]; // Elements 16-23
        obi_o_wdata_o[3] = ofm_buf_q[31:24]; // Elements 24-31

        // Handle OFM Writes with backpressure (I3)
        // When ofm_buf_valid_q is set (either from new ofm_valid_i or a retry),
        // we issue 4 writes. If not all granted, we hold and retry next cycle.
        if (ofm_buf_valid_q) begin
            obi_o_req_o = 4'b1111;
            // Only advance when all 4 ports are granted
            if (obi_o_gnt_i == 4'b1111) begin
                o_ptr_d = o_ptr_q + 128; // 128 bytes = 4 x 256-bit
                drain_cnt_d = drain_cnt_q - 1;
                // Buffer consumed — will be re-filled if ofm_valid_i arrives
            end
        end

        case (state_q)
            IDLE: begin
                if (cfg_sys_start_i) begin
                    w_ptr_d = cfg_sys_weight_ptr_i + ((ARRAY_DIM - 1) * 32);
                    i_ptr_d = cfg_sys_ifm_ptr_i;
                    o_ptr_d = cfg_sys_ofm_ptr_i;
                    req_cnt_d   = ARRAY_DIM; // 32 rows of weights
                    rsp_cnt_d   = ARRAY_DIM;
                    drain_cnt_d = cfg_sys_dim_m_i;
                    state_d = LOAD_WEIGHTS;
                end
            end

            LOAD_WEIGHTS: begin
                if (req_cnt_q > 0) begin
                    obi_i_req_o = 1'b1;
                    obi_i_addr_o = w_ptr_q;
                    if (obi_i_gnt_i) begin
                        w_ptr_d = w_ptr_q - 32;
                        req_cnt_d   = req_cnt_q - 1;
                    end
                end
                if (obi_i_rvalid_i) begin
                    weight_load_en_o = 1'b1;
                    weight_data_o    = obi_i_rdata_i;  // I1: only drive during weight load
                    rsp_cnt_d = rsp_cnt_q - 1;
                end
                if (rsp_cnt_q == 0 || (rsp_cnt_q == 1 && obi_i_rvalid_i)) begin // Wait for all reads to complete
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    state_d = COMPUTE;
                end
            end

            COMPUTE: begin
                if (req_cnt_q > 0) begin
                    obi_i_req_o = 1'b1;
                    obi_i_addr_o = i_ptr_q;
                    if (obi_i_gnt_i) begin
                        i_ptr_d = i_ptr_q + 32;
                        req_cnt_d   = req_cnt_q - 1;
                    end
                end
                if (obi_i_rvalid_i) begin
                    compute_en_o = 1'b1;
                    clear_acc_o  = 1'b0;
                    ifm_data_o   = obi_i_rdata_i;  // I1: only drive during compute
                    rsp_cnt_d = rsp_cnt_q - 1;
                end
                if (rsp_cnt_q == 0 || (rsp_cnt_q == 1 && obi_i_rvalid_i)) begin
                    state_d = WAIT_DRAIN;
                end
            end

            WAIT_DRAIN: begin
                // Wait until all outputs are drained to memory
                if (drain_cnt_q == 0 && !ofm_buf_valid_q) begin
                    state_d = DONE;
                end
            end

            DONE: begin
                cfg_sys_done_o = 1'b1;
                state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q         <= IDLE;
            w_ptr_q         <= '0;
            i_ptr_q         <= '0;
            o_ptr_q         <= '0;
            req_cnt_q       <= '0;
            rsp_cnt_q       <= '0;
            drain_cnt_q     <= '0;
            ofm_buf_q       <= '0;
            ofm_buf_valid_q <= 1'b0;
        end else begin
            state_q     <= state_d;
            w_ptr_q     <= w_ptr_d;
            i_ptr_q     <= i_ptr_d;
            o_ptr_q     <= o_ptr_d;
            req_cnt_q   <= req_cnt_d;
            rsp_cnt_q   <= rsp_cnt_d;
            drain_cnt_q <= drain_cnt_d;

            // OFM buffer management (I3: backpressure)
            if (ofm_buf_valid_q && obi_o_gnt_i == 4'b1111) begin
                // Buffer was consumed
                if (ofm_valid_i) begin
                    // New data arrives same cycle — refill buffer
                    ofm_buf_q       <= ofm_data_i;
                    ofm_buf_valid_q <= 1'b1;
                end else begin
                    ofm_buf_valid_q <= 1'b0;
                end
            end else if (!ofm_buf_valid_q && ofm_valid_i) begin
                // Fresh data from systolic array
                ofm_buf_q       <= ofm_data_i;
                ofm_buf_valid_q <= 1'b1;
            end
            // If buf is valid but not all granted, hold — retry next cycle
        end
    end

endmodule
