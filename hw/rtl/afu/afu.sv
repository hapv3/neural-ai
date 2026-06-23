`default_nettype none

module afu #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic clk_i,
    input  logic rst_ni,

    // OBI Slave Port (Configuration & LUT Loading)
    input  logic                      obi_s_req_i,
    output logic                      obi_s_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     obi_s_addr_i,
    input  logic                      obi_s_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] obi_s_be_i,
    input  logic [DATA_WIDTH-1:0]     obi_s_wdata_i,
    output logic                      obi_s_rvalid_o,
    output logic [DATA_WIDTH-1:0]     obi_s_rdata_o,

    // OBI Master Port (Memory Read/Write)
    output logic                      obi_m_req_o,
    input  logic                      obi_m_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_m_addr_o,
    output logic                      obi_m_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_m_be_o,
    output logic [DATA_WIDTH-1:0]     obi_m_wdata_o,
    input  logic                      obi_m_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_m_rdata_i,

    // Interrupt/Done signal
    output logic                      done_o
);

    //----------------------------------------------------------------------
    // Configuration Registers
    //----------------------------------------------------------------------
    // Address Map:
    // 0x0000 - 0x03FC: LUT SRAM (256 x 32-bit = 1024 bytes)
    // 0x1000: START (W), DONE (R)
    // 0x1004: SRC_PTR (R/W)
    // 0x1008: DST_PTR (R/W)
    // 0x100C: LENGTH (R/W) - Number of elements (bytes) to process
    // 0x1010: MODE (R/W) - 0: 8-bit, 1: 16-bit, 2: 32-bit output precision
    
    logic        cfg_start;
    logic        cfg_done;
    logic [31:0] cfg_src_ptr, cfg_src_ptr_n;
    logic [31:0] cfg_dst_ptr, cfg_dst_ptr_n;
    logic [31:0] cfg_length,  cfg_length_n;
    logic [1:0]  cfg_mode,    cfg_mode_n;

    logic lut_we;
    logic [7:0]  lut_addr;
    logic [31:0] lut_wdata;
    logic [31:0] lut_rdata_cfg;

    // OBI Slave Response
    logic obi_s_rvalid_q;
    logic [DATA_WIDTH-1:0] obi_s_rdata_q;
    
    assign obi_s_gnt_o    = 1'b1; // Always accept requests immediately
    assign obi_s_rvalid_o = obi_s_rvalid_q;
    assign obi_s_rdata_o  = obi_s_rdata_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cfg_src_ptr    <= '0;
            cfg_dst_ptr    <= '0;
            cfg_length     <= '0;
            cfg_mode       <= '0;
            cfg_start      <= '0;
            obi_s_rvalid_q <= '0;
            obi_s_rdata_q  <= '0;
        end else begin
            cfg_start      <= 1'b0; // Auto-clear
            obi_s_rvalid_q <= obi_s_req_i; // Respond in next cycle
            
            // Register writes
            if (obi_s_req_i && obi_s_we_i) begin
                if (obi_s_addr_i[15:12] == 4'h1) begin
                    case (obi_s_addr_i[7:0])
                        8'h00: cfg_start   <= obi_s_wdata_i[0];
                        8'h04: cfg_src_ptr <= obi_s_wdata_i;
                        8'h08: cfg_dst_ptr <= obi_s_wdata_i;
                        8'h0C: cfg_length  <= obi_s_wdata_i;
                        8'h10: cfg_mode    <= obi_s_wdata_i[1:0];
                    endcase
                end
            end
            
            // Register reads
            if (obi_s_req_i && !obi_s_we_i) begin
                if (obi_s_addr_i[15:12] == 4'h1) begin
                    case (obi_s_addr_i[7:0])
                        8'h00: obi_s_rdata_q <= {31'd0, cfg_done};
                        8'h04: obi_s_rdata_q <= cfg_src_ptr;
                        8'h08: obi_s_rdata_q <= cfg_dst_ptr;
                        8'h0C: obi_s_rdata_q <= cfg_length;
                        8'h10: obi_s_rdata_q <= {30'd0, cfg_mode};
                        default: obi_s_rdata_q <= '0;
                    endcase
                end else if (obi_s_addr_i[15:12] == 4'h0) begin
                    obi_s_rdata_q <= lut_rdata_cfg;
                end else begin
                    obi_s_rdata_q <= '0;
                end
            end
        end
    end

    // LUT write logic from OBI Slave
    assign lut_we    = (obi_s_req_i && obi_s_we_i && (obi_s_addr_i[15:12] == 4'h0));
    assign lut_addr  = obi_s_addr_i[9:2]; // Word address
    assign lut_wdata = obi_s_wdata_i;

    //----------------------------------------------------------------------
    // LUT SRAM (256 words x 32-bit)
    //----------------------------------------------------------------------
    logic [1:0]          sram_req;
    logic [1:0]          sram_we;
    logic [1:0][7:0]     sram_addr;
    logic [1:0][31:0]    sram_wdata;
    logic [1:0][3:0]     sram_be;
    logic [1:0][31:0]    sram_rdata;

    // Port 0: Configuration
    assign sram_req[0]   = (obi_s_req_i && (obi_s_addr_i[15:12] == 4'h0));
    assign sram_we[0]    = lut_we;
    assign sram_addr[0]  = lut_addr;
    assign sram_wdata[0] = lut_wdata;
    assign sram_be[0]    = obi_s_be_i;
    assign lut_rdata_cfg = sram_rdata[0];

    // Port 1: FSM Lookup
    logic       fsm_lut_req;
    logic [7:0] fsm_lut_addr;
    
    assign sram_req[1]   = fsm_lut_req;
    assign sram_we[1]    = 1'b0; // FSM only reads
    assign sram_addr[1]  = fsm_lut_addr;
    assign sram_wdata[1] = '0;
    assign sram_be[1]    = 4'hF;

    tc_sram #(
        .NumWords   (256),
        .DataWidth  (32),
        .ByteWidth  (8),
        .NumPorts   (2),
        .Latency    (1),
        .SimInit    ("zeros")
    ) i_lut_sram (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .req_i      (sram_req),
        .we_i       (sram_we),
        .addr_i     (sram_addr),
        .wdata_i    (sram_wdata),
        .be_i       (sram_be),
        .rdata_o    (sram_rdata)
    );

    //----------------------------------------------------------------------
    // FSM Core
    //----------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_READ_MEM,
        ST_WAIT_RDATA,
        ST_LOOKUP_SRAM,
        ST_WAIT_SRAM,
        ST_WRITE_MEM,
        ST_DONE
    } state_e;

    state_e state_q, state_n;
    
    logic [31:0] elem_cnt_q, elem_cnt_n;
    logic [31:0] src_addr_q, src_addr_n;
    logic [31:0] dst_addr_q, dst_addr_n;
    logic [1:0]  byte_idx_q, byte_idx_n;
    logic [31:0] read_buf_q, read_buf_n;
    
    assign done_o   = cfg_done;
    
    always_comb begin
        state_n      = state_q;
        elem_cnt_n   = elem_cnt_q;
        src_addr_n   = src_addr_q;
        dst_addr_n   = dst_addr_q;
        byte_idx_n   = byte_idx_q;
        read_buf_n   = read_buf_q;
        
        obi_m_req_o  = 1'b0;
        obi_m_we_o   = 1'b0;
        obi_m_addr_o = '0;
        obi_m_wdata_o= '0;
        obi_m_be_o   = '0;
        
        fsm_lut_req  = 1'b0;
        fsm_lut_addr = '0;
        
        case (state_q)
            ST_IDLE: begin
                if (cfg_start) begin
                    state_n    = ST_READ_MEM;
                    elem_cnt_n = '0;
                    src_addr_n = cfg_src_ptr;
                    dst_addr_n = cfg_dst_ptr;
                    byte_idx_n = cfg_src_ptr[1:0]; // Handle unaligned start
                end
            end
            
            ST_READ_MEM: begin
                obi_m_req_o  = 1'b1;
                obi_m_addr_o = {src_addr_q[31:2], 2'b00}; // Word aligned read
                obi_m_we_o   = 1'b0;
                if (obi_m_gnt_i) begin
                    state_n = ST_WAIT_RDATA;
                end
            end
            
            ST_WAIT_RDATA: begin
                if (obi_m_rvalid_i) begin
                    read_buf_n = obi_m_rdata_i;
                    state_n = ST_LOOKUP_SRAM;
                end
            end
            
            ST_LOOKUP_SRAM: begin
                fsm_lut_req = 1'b1;
                // Select byte based on index
                case (byte_idx_q)
                    2'd0: fsm_lut_addr = read_buf_q[7:0];
                    2'd1: fsm_lut_addr = read_buf_q[15:8];
                    2'd2: fsm_lut_addr = read_buf_q[23:16];
                    2'd3: fsm_lut_addr = read_buf_q[31:24];
                endcase
                state_n = ST_WAIT_SRAM;
            end
            
            ST_WAIT_SRAM: begin
                // SRAM latency is 1 cycle, data available next cycle
                state_n = ST_WRITE_MEM;
            end
            
            ST_WRITE_MEM: begin
                obi_m_req_o = 1'b1;
                obi_m_we_o  = 1'b1;
                
                // Format output based on mode and destination address
                if (cfg_mode == 2'd0) begin
                    // 8-bit output
                    obi_m_addr_o = {dst_addr_q[31:2], 2'b00};
                    case (dst_addr_q[1:0])
                        2'd0: begin obi_m_wdata_o[7:0]   = sram_rdata[1][7:0]; obi_m_be_o = 4'b0001; end
                        2'd1: begin obi_m_wdata_o[15:8]  = sram_rdata[1][7:0]; obi_m_be_o = 4'b0010; end
                        2'd2: begin obi_m_wdata_o[23:16] = sram_rdata[1][7:0]; obi_m_be_o = 4'b0100; end
                        2'd3: begin obi_m_wdata_o[31:24] = sram_rdata[1][7:0]; obi_m_be_o = 4'b1000; end
                    endcase
                end else if (cfg_mode == 2'd1) begin
                    // 16-bit output
                    obi_m_addr_o = {dst_addr_q[31:2], 2'b00};
                    if (dst_addr_q[1] == 1'b0) begin
                        obi_m_wdata_o[15:0]  = sram_rdata[1][15:0]; 
                        obi_m_be_o = 4'b0011;
                    end else begin
                        obi_m_wdata_o[31:16] = sram_rdata[1][15:0]; 
                        obi_m_be_o = 4'b1100;
                    end
                end else begin
                    // 32-bit output
                    obi_m_addr_o = {dst_addr_q[31:2], 2'b00};
                    obi_m_wdata_o = sram_rdata[1];
                    obi_m_be_o = 4'b1111;
                end
                
                if (obi_m_gnt_i) begin
                    elem_cnt_n = elem_cnt_q + 1;
                    byte_idx_n = byte_idx_q + 1;
                    src_addr_n = src_addr_q + 1; // Input is always 8-bit
                    
                    // Increment destination based on output mode
                    if (cfg_mode == 2'd0)      dst_addr_n = dst_addr_q + 1;
                    else if (cfg_mode == 2'd1) dst_addr_n = dst_addr_q + 2;
                    else                       dst_addr_n = dst_addr_q + 4;
                    
                    if (elem_cnt_n == cfg_length) begin
                        state_n = ST_DONE;
                    end else begin
                        // If we wrapped around byte index, need to read next word
                        if (byte_idx_n == 2'd0) begin
                            state_n = ST_READ_MEM;
                        end else begin
                            state_n = ST_LOOKUP_SRAM;
                        end
                    end
                end
            end
            
            ST_DONE: begin
                state_n = ST_IDLE;
            end
            
            default: state_n = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q    <= ST_IDLE;
            elem_cnt_q <= '0;
            src_addr_q <= '0;
            dst_addr_q <= '0;
            byte_idx_q <= '0;
            read_buf_q <= '0;
            cfg_done   <= 1'b0;
        end else begin
            state_q    <= state_n;
            elem_cnt_q <= elem_cnt_n;
            src_addr_q <= src_addr_n;
            dst_addr_q <= dst_addr_n;
            byte_idx_q <= byte_idx_n;
            read_buf_q <= read_buf_n;
            
            if (state_q == ST_DONE) begin
                cfg_done <= 1'b1;
            end else if (cfg_start) begin
                cfg_done <= 1'b0;
            end
        end
    end

endmodule
