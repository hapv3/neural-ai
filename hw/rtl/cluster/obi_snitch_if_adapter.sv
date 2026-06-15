module obi_snitch_if_adapter #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256
) (
    input  logic                    clk_i,
    input  logic                    rst_ni,

    // Snitch Instruction Fetch Interface
    input  logic                    snitch_req_valid_i,
    output logic                    snitch_req_ready_o,
    input  logic [ADDR_WIDTH-1:0]   snitch_req_addr_i,

    output logic [DATA_WIDTH-1:0]   snitch_rsp_data_o,

    // OBI Master Interface
    output logic                    obi_req_o,
    input  logic                    obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]   obi_addr_o,
    input  logic                    obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]   obi_rdata_i
);

    typedef enum logic { IDLE, WAIT_RVALID } state_e;
    state_e state_d, state_q;

    always_comb begin
        state_d            = state_q;
        
        obi_req_o          = 1'b0;
        obi_addr_o         = snitch_req_addr_i;
        
        snitch_req_ready_o = 1'b0;
        snitch_rsp_data_o  = '0;

        case (state_q)
            IDLE: begin
                if (snitch_req_valid_i) begin
                    obi_req_o  = 1'b1;
                    if (obi_gnt_i) begin
                        state_d = WAIT_RVALID;
                    end
                end
            end

            WAIT_RVALID: begin
                if (obi_rvalid_i) begin
                    snitch_req_ready_o = 1'b1;
                    snitch_rsp_data_o  = obi_rdata_i;
                    state_d = IDLE;
                end
            end
            
            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
        end else begin
            state_q <= state_d;
        end
    end

endmodule
