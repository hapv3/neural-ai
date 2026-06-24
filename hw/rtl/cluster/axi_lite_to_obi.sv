`default_nettype none

module axi_lite_to_obi #(
    parameter int unsigned AXI_ADDR_WIDTH = 32,
    parameter int unsigned AXI_DATA_WIDTH = 32,
    parameter int unsigned OBI_ADDR_WIDTH = 32,
    parameter int unsigned OBI_DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // AXI Lite Slave
    input  logic [AXI_ADDR_WIDTH-1:0]       s_axi_aw_addr_i,
    input  logic                            s_axi_aw_valid_i,
    output logic                            s_axi_aw_ready_o,

    input  logic [AXI_DATA_WIDTH-1:0]       s_axi_w_data_i,
    input  logic [(AXI_DATA_WIDTH/8)-1:0]   s_axi_w_strb_i,
    input  logic                            s_axi_w_valid_i,
    output logic                            s_axi_w_ready_o,

    output logic [1:0]                      s_axi_b_resp_o,
    output logic                            s_axi_b_valid_o,
    input  logic                            s_axi_b_ready_i,

    input  logic [AXI_ADDR_WIDTH-1:0]       s_axi_ar_addr_i,
    input  logic                            s_axi_ar_valid_i,
    output logic                            s_axi_ar_ready_o,

    output logic [AXI_DATA_WIDTH-1:0]       s_axi_r_data_o,
    output logic [1:0]                      s_axi_r_resp_o,
    output logic                            s_axi_r_valid_o,
    input  logic                            s_axi_r_ready_i,

    // OBI Master
    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [OBI_ADDR_WIDTH-1:0] obi_addr_o,
    output logic                      obi_we_o,
    output logic [(OBI_DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [OBI_DATA_WIDTH-1:0] obi_wdata_o,
    input  logic                      obi_rvalid_i,
    input  logic [OBI_DATA_WIDTH-1:0] obi_rdata_i
);

    typedef enum logic [2:0] {IDLE, WAIT_W, WAIT_AW, WAIT_GNT, WAIT_RVALID, WAIT_BREADY, WAIT_RREADY} state_e;
    state_e state_q, state_d;

    logic [AXI_ADDR_WIDTH-1:0] aw_addr_q;
    logic [AXI_DATA_WIDTH-1:0] w_data_q;
    logic [(AXI_DATA_WIDTH/8)-1:0] w_strb_q;
    logic [AXI_ADDR_WIDTH-1:0] ar_addr_q;
    logic [AXI_DATA_WIDTH-1:0] r_data_q, r_data_d;
    logic is_write_q, is_write_d;

    assign s_axi_b_resp_o = 2'b00; // OKAY
    assign s_axi_r_resp_o = 2'b00; // OKAY

    logic aw_hs, w_hs, ar_hs;
    assign aw_hs = s_axi_aw_valid_i & s_axi_aw_ready_o;
    assign w_hs  = s_axi_w_valid_i & s_axi_w_ready_o;
    assign ar_hs = s_axi_ar_valid_i & s_axi_ar_ready_o;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q    <= IDLE;
            is_write_q <= 1'b0;
            aw_addr_q  <= '0;
            w_data_q   <= '0;
            w_strb_q   <= '0;
            ar_addr_q  <= '0;
            r_data_q   <= '0;
        end else begin
            state_q    <= state_d;
            is_write_q <= is_write_d;
            r_data_q   <= r_data_d;
            if (aw_hs) aw_addr_q <= s_axi_aw_addr_i;
            if (w_hs) begin
                w_data_q <= s_axi_w_data_i;
                w_strb_q <= s_axi_w_strb_i;
            end
            if (ar_hs) ar_addr_q <= s_axi_ar_addr_i;
        end
    end

    always_comb begin
        state_d          = state_q;
        is_write_d       = is_write_q;
        r_data_d         = r_data_q;
        
        s_axi_aw_ready_o = 1'b0;
        s_axi_w_ready_o  = 1'b0;
        s_axi_ar_ready_o = 1'b0;
        s_axi_b_valid_o  = 1'b0;
        s_axi_r_valid_o  = 1'b0;
        s_axi_r_data_o   = r_data_q;
        
        obi_req_o   = 1'b0;
        obi_addr_o  = '0;
        obi_we_o    = 1'b0;
        obi_be_o    = '0;
        obi_wdata_o = '0;

        case (state_q)
            IDLE: begin
                if (s_axi_aw_valid_i && s_axi_w_valid_i) begin
                    s_axi_aw_ready_o = 1'b1;
                    s_axi_w_ready_o  = 1'b1;
                    is_write_d       = 1'b1;
                    state_d          = WAIT_GNT;
                end else if (s_axi_aw_valid_i) begin
                    s_axi_aw_ready_o = 1'b1;
                    is_write_d       = 1'b1;
                    state_d          = WAIT_W;
                end else if (s_axi_w_valid_i) begin
                    s_axi_w_ready_o  = 1'b1;
                    is_write_d       = 1'b1;
                    state_d          = WAIT_AW;
                end else if (s_axi_ar_valid_i) begin
                    s_axi_ar_ready_o = 1'b1;
                    is_write_d       = 1'b0;
                    state_d          = WAIT_GNT;
                end
            end

            WAIT_W: begin
                if (s_axi_w_valid_i) begin
                    s_axi_w_ready_o = 1'b1;
                    state_d         = WAIT_GNT;
                end
            end

            WAIT_AW: begin
                if (s_axi_aw_valid_i) begin
                    s_axi_aw_ready_o = 1'b1;
                    state_d          = WAIT_GNT;
                end
            end

            WAIT_GNT: begin
                obi_req_o = 1'b1;
                if (is_write_q) begin
                    obi_we_o    = 1'b1;
                    obi_addr_o  = aw_addr_q;
                    if (OBI_DATA_WIDTH == AXI_DATA_WIDTH) begin
                        obi_be_o    = w_strb_q;
                        obi_wdata_o = w_data_q;
                    end else begin
                        // AXI w_strb is already aligned to AXI_DATA_WIDTH.
                        // We only shift by the offset of the AXI_DATA_WIDTH word within the OBI_DATA_WIDTH line.
                        obi_be_o    = {{(OBI_DATA_WIDTH/8 - AXI_DATA_WIDTH/8){1'b0}}, w_strb_q} << ((aw_addr_q[4:0] / (AXI_DATA_WIDTH/8)) * (AXI_DATA_WIDTH/8));

                        // Replicate the AXI data to fill the OBI data width
                        obi_wdata_o = {(OBI_DATA_WIDTH/AXI_DATA_WIDTH){w_data_q}};
                    end
                end else begin
                    obi_we_o    = 1'b0;
                    obi_addr_o  = ar_addr_q;
                    obi_be_o    = '1;
                end

                if (obi_gnt_i) begin
                    state_d = WAIT_RVALID;
                end
            end

            WAIT_RVALID: begin
                if (obi_rvalid_i) begin
                    if (is_write_q) begin
                        s_axi_b_valid_o = 1'b1;
                        if (s_axi_b_ready_i) begin
                            state_d = IDLE;
                        end else begin
                            state_d = WAIT_BREADY;
                        end
                    end else begin
                        s_axi_r_valid_o = 1'b1;
                        if (OBI_DATA_WIDTH == AXI_DATA_WIDTH) begin
                            r_data_d = obi_rdata_i[AXI_DATA_WIDTH-1:0];
                        end else begin
                            r_data_d = obi_rdata_i >> ((ar_addr_q[4:0] / (AXI_DATA_WIDTH/8)) * AXI_DATA_WIDTH);
                        end
                        s_axi_r_data_o  = r_data_d;
                        if (s_axi_r_ready_i) begin
                            state_d = IDLE;
                        end else begin
                            state_d = WAIT_RREADY;
                        end
                    end
                end
            end
            
            WAIT_BREADY: begin
                s_axi_b_valid_o = 1'b1;
                if (s_axi_b_ready_i) state_d = IDLE;
            end
            
            WAIT_RREADY: begin
                s_axi_r_valid_o = 1'b1;
                s_axi_r_data_o  = r_data_q;
                if (s_axi_r_ready_i) state_d = IDLE;
            end
        endcase
    end

endmodule
