`default_nettype none

module cluster_sram_bank #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned SIZE_BYTES = 32768 // 32KB
)(
    input  logic                      clk_i,
    input  logic                      rst_ni,
    
    // OBI slave interface
    input  logic                      req_i,
    input  logic                      we_i,
    input  logic [31:0]               addr_i,
    input  logic [DATA_WIDTH-1:0]     wdata_i,
    input  logic [(DATA_WIDTH/8)-1:0] be_i,
    output logic                      gnt_o,
    output logic                      rvalid_o,
    output logic [DATA_WIDTH-1:0]     rdata_o
);

    localparam int unsigned NUM_WORDS = SIZE_BYTES / (DATA_WIDTH / 8);
    localparam int unsigned ADDR_BITS = $clog2(NUM_WORDS);

    // BRAM memory array
    logic [DATA_WIDTH-1:0] mem [NUM_WORDS];

    logic rvalid_q;
    logic [DATA_WIDTH-1:0] rdata_q;

    // Grant is immediate for simple SRAM
    assign gnt_o = req_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            rvalid_q <= 1'b0;
            rdata_q  <= '0;
            // Khởi tạo bộ nhớ với giá trị 0 (để dễ debug)
            for (int i = 0; i < NUM_WORDS; i++) begin
                mem[i] <= '0;
            end
        end else begin
            rvalid_q <= req_i & ~we_i; // rvalid takes 1 cycle
            
            if (req_i) begin
                if (we_i) begin
                    // Write with Byte Enable
                    for (int i = 0; i < DATA_WIDTH/8; i++) begin
                        if (be_i[i]) begin
                            mem[addr_i[ADDR_BITS-1:0]][i*8 +: 8] <= wdata_i[i*8 +: 8];
                        end
                    end
                end else begin
                    // Read
                    rdata_q <= mem[addr_i[ADDR_BITS-1:0]];
                end
            end
        end
    end

    assign rvalid_o = rvalid_q;
    assign rdata_o  = rdata_q;

endmodule
