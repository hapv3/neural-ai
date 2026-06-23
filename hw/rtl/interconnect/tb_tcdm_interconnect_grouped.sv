`default_nettype none

module tb_tcdm_interconnect_grouped;
    localparam int unsigned NUM_MASTERS = 10;
    localparam int unsigned NUM_BANKS   = 16;
    localparam int unsigned ADDR_WIDTH  = 32;
    localparam int unsigned DATA_WIDTH  = 256;

    logic clk_i;
    logic rst_ni;

    logic [NUM_MASTERS-1:0]                     master_req;
    logic [NUM_MASTERS-1:0]                     master_gnt;
    logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]     master_addr;
    logic [NUM_MASTERS-1:0]                     master_we;
    logic [NUM_MASTERS-1:0][DATA_WIDTH/8-1:0]   master_be;
    logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]     master_wdata;
    logic [NUM_MASTERS-1:0]                     master_rvalid;
    logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]     master_rdata;

    logic [NUM_BANKS-1:0]                       bank_req;
    logic [NUM_BANKS-1:0][ADDR_WIDTH-1:0]       bank_addr;
    logic [NUM_BANKS-1:0]                       bank_we;
    logic [NUM_BANKS-1:0][DATA_WIDTH/8-1:0]     bank_be;
    logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]       bank_wdata;
    logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]       bank_rdata;

    tcdm_interconnect #(
        .NUM_MASTERS      (NUM_MASTERS),
        .NUM_BANKS        (NUM_BANKS),
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .HWPE_MASTER_MASK (10'h1FA),
        .DMA_MASTER_MASK  (10'h204),
        .CORE_MASTER_MASK (10'h001)
    ) dut (
        .clk_i,
        .rst_ni,
        .master_req_i     (master_req),
        .master_gnt_o     (master_gnt),
        .master_addr_i    (master_addr),
        .master_we_i      (master_we),
        .master_be_i      (master_be),
        .master_wdata_i   (master_wdata),
        .master_rvalid_o  (master_rvalid),
        .master_rdata_o   (master_rdata),
        .bank_req_o       (bank_req),
        .bank_addr_o      (bank_addr),
        .bank_we_o        (bank_we),
        .bank_be_o        (bank_be),
        .bank_wdata_o     (bank_wdata),
        .bank_rdata_i     (bank_rdata)
    );

    always #5 clk_i = ~clk_i;

    function automatic logic [ADDR_WIDTH-1:0] addr_for_bank(input int unsigned bank);
        return 32'h1010_0000 + (bank << 5);
    endfunction

    task automatic clear_masters();
        master_req   = '0;
        master_addr  = '0;
        master_we    = '0;
        master_be    = '1;
        master_wdata = '0;
    endtask

    task automatic request_master(input int unsigned master, input int unsigned bank);
        master_req[master]  = 1'b1;
        master_addr[master] = addr_for_bank(bank);
        master_we[master]   = 1'b1;
    endtask

    task automatic expect_onehot(input logic [NUM_MASTERS-1:0] expected, input string label);
        #1;
        if (master_gnt !== expected) begin
            $error("%s: got gnt=%b expected=%b", label, master_gnt, expected);
            $fatal(1);
        end
    endtask

    task automatic expect_eventual_onehot(
        input logic [NUM_MASTERS-1:0] expected,
        input int unsigned max_cycles,
        input string label
    );
        for (int unsigned cycle = 0; cycle <= max_cycles; cycle++) begin
            #1;
            if (master_gnt === expected) begin
                return;
            end
            @(posedge clk_i);
        end
        #1;
        $error("%s: got gnt=%b expected eventual=%b", label, master_gnt, expected);
        $fatal(1);
    endtask

    task automatic tick();
        @(posedge clk_i);
        #1;
    endtask

    initial begin
        clk_i = 1'b0;
        rst_ni = 1'b0;
        clear_masters();
        bank_rdata = '0;
        repeat (4) @(posedge clk_i);
        rst_ni = 1'b1;
        tick();

        clear_masters();
        request_master(0, 0);
        request_master(2, 0);
        request_master(3, 0);
        expect_onehot(10'b0000001000, "strict priority HWPE over DMA/CORE");
        tick();

        clear_masters();
        request_master(0, 0);
        request_master(2, 0);
        expect_onehot(10'b0000000100, "strict priority DMA over CORE");
        tick();

        clear_masters();
        request_master(0, 0);
        expect_onehot(10'b0000000001, "CORE granted when alone");
        tick();

        clear_masters();
        request_master(1, 1);
        request_master(3, 1);
        expect_onehot(10'b0000000010, "HWPE RR first grant");
        tick();
        expect_eventual_onehot(10'b0000001000, NUM_MASTERS, "HWPE RR eventually grants second requester");
        expect_eventual_onehot(10'b0000000010, NUM_MASTERS, "HWPE RR eventually wraps");

        clear_masters();
        request_master(2, 2);
        request_master(9, 2);
        expect_onehot(10'b0000000100, "DMA RR first grant");
        tick();
        expect_eventual_onehot(10'b1000000000, NUM_MASTERS, "DMA RR eventually grants second requester");

        clear_masters();
        request_master(0, 3);
        request_master(3, 4);
        expect_onehot(10'b0000001001, "independent banks grant concurrently");
        tick();

        $display("TCDM grouped interconnect unit test passed");
        $finish;
    end
endmodule

`default_nettype wire
