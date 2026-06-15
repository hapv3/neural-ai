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
    logic [DATA_WIDTH-1:0] temp_data;

    // Grant is immediate for simple SRAM
    assign gnt_o = req_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            rvalid_q <= 1'b0;
            rdata_q  <= '0;
            // Do NOT clear memory on reset, otherwise we wipe the loaded firmware
            // Real SRAMs don't clear on reset anyway.
        end else begin
            rvalid_q <= req_i; // OBI requires rvalid for both reads and writes
            
            if (req_i) begin
                if (we_i) begin
                    $display("[SRAM %m] WRITE to Addr=%h, Data=%h, BE=%h", addr_i, wdata_i, be_i);
                    // Write with Byte Enable
                    // Workaround for Verilator partial array assignment bug
                    temp_data = mem[addr_i[ADDR_BITS-1:0]];
                    for (int i = 0; i < DATA_WIDTH/8; i++) begin
                        if (be_i[i]) begin
                            temp_data[i*8 +: 8] = wdata_i[i*8 +: 8];
                        end
                    end
                    mem[addr_i[ADDR_BITS-1:0]] <= temp_data;
                end else begin
                    // Read
                    rdata_q <= mem[addr_i[ADDR_BITS-1:0]];
                    $display("[SRAM %m] READ from Addr=%h, DataWord=%h", addr_i, mem[addr_i[ADDR_BITS-1:0]]);
                end
            end
        end
    end

    assign rvalid_o = rvalid_q;
    assign rdata_o  = rdata_q;

endmodule
