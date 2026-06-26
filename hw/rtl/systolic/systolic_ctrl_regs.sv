`default_nettype none

module systolic_ctrl_regs #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      req_i,
    output logic                      gnt_o,
    input  logic [ADDR_WIDTH-1:0]     addr_i,
    input  logic                      we_i,
    input  logic [(DATA_WIDTH/8)-1:0] be_i,
    input  logic [DATA_WIDTH-1:0]     wdata_i,
    output logic                      rvalid_o,
    output logic [DATA_WIDTH-1:0]     rdata_o,

    output logic                      cfg_sys_start_o,
    output logic [31:0]               cfg_sys_weight_ptr_o,
    output logic [31:0]               cfg_sys_ifm_ptr_o,
    output logic [31:0]               cfg_sys_ofm_ptr_o,
    output logic [31:0]               cfg_sys_psum_ptr_o,
    output logic [31:0]               cfg_sys_dim_m_o,
    output logic                      cfg_sys_accum_en_o,
    output logic                      cfg_sys_ifm_stream_en_o,
    output logic                      cfg_requant_en_o,
    output logic [31:0][31:0]         cfg_requant_bias_o,
    output logic [31:0][31:0]         cfg_requant_multiplier_o,
    output logic [31:0][7:0]          cfg_requant_shift_o,
    output logic [31:0][31:0]         cfg_requant_zero_point_o,
    output logic [31:0]               cfg_requant_clamp_min_o,
    output logic [31:0]               cfg_requant_clamp_max_o,
    input  logic                      cfg_sys_done_i,

    output logic                      cfg_conv_start_o,
    output logic [31:0]               cfg_conv_input_ptr_o,
    output logic [31:0]               cfg_conv_output_ptr_o,
    output logic [31:0]               cfg_conv_rows_o,
    output logic [31:0]               cfg_conv_output_w_o,
    output logic [31:0]               cfg_conv_input_h_o,
    output logic [31:0]               cfg_conv_input_w_o,
    output logic [31:0]               cfg_conv_input_c_o,
    output logic [15:0]               cfg_conv_kernel_h_o,
    output logic [15:0]               cfg_conv_kernel_w_o,
    output logic [15:0]               cfg_conv_stride_h_o,
    output logic [15:0]               cfg_conv_stride_w_o,
    output logic [15:0]               cfg_conv_pad_h_o,
    output logic [15:0]               cfg_conv_pad_w_o,
    output logic [15:0]               cfg_conv_dilation_h_o,
    output logic [15:0]               cfg_conv_dilation_w_o,
    output logic [31:0]               cfg_conv_k_base_o,
    output logic                      cfg_conv_stream_en_o,
    input  logic                      cfg_conv_done_i,
    input  logic                      cfg_conv_busy_i
);

    localparam int unsigned DATA_BYTES = DATA_WIDTH / 8;

    localparam logic [ADDR_WIDTH-1:0] REG_SYS_W_PTR = 32'h0100;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_I_PTR = 32'h0104;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_O_PTR = 32'h0108;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DIM_M = 32'h010C;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_START = 32'h0110;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DONE  = 32'h0114;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_PSUM_PTR = 32'h0118;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_ACCUM_CTRL = 32'h011C;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CTRL   = 32'h0120;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CMIN   = 32'h0124;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CMAX   = 32'h0128;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_BIAS_BASE = 32'h0200;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_MULT_BASE = 32'h0280;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_SHIFT_BASE = 32'h0300;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_ZP_BASE = 32'h0380;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_INPUT_PTR = 32'h0400;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_OUTPUT_PTR = 32'h0404;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_ROWS = 32'h0408;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_OUTPUT_W = 32'h040C;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_INPUT_H = 32'h0410;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_INPUT_W = 32'h0414;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_INPUT_C = 32'h0418;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_KERNEL = 32'h041C;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_STRIDE = 32'h0420;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_PAD = 32'h0424;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_DILATION = 32'h0428;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_K_BASE = 32'h042C;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_START = 32'h0430;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_DONE = 32'h0434;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_STATUS = 32'h0438;
    localparam logic [ADDR_WIDTH-1:0] REG_CONV_MODE = 32'h043C;

    logic [31:0] r_sys_w_ptr;
    logic [31:0] r_sys_i_ptr;
    logic [31:0] r_sys_o_ptr;
    logic [31:0] r_sys_psum_ptr;
    logic [31:0] r_sys_dim_m;
    logic        r_sys_start;
    logic        r_sys_done;
    logic        r_sys_accum_en;
    logic        r_sys_ifm_stream_en;
    logic        r_requant_en;
    logic [31:0][31:0] r_requant_bias;
    logic [31:0][31:0] r_requant_multiplier;
    logic [31:0][7:0]  r_requant_shift;
    logic [31:0][31:0] r_requant_zero_point;
    logic [31:0]       r_requant_clamp_min;
    logic [31:0]       r_requant_clamp_max;
    logic              r_conv_start;
    logic              r_conv_done;
    logic [31:0]       r_conv_input_ptr;
    logic [31:0]       r_conv_output_ptr;
    logic [31:0]       r_conv_rows;
    logic [31:0]       r_conv_output_w;
    logic [31:0]       r_conv_input_h;
    logic [31:0]       r_conv_input_w;
    logic [31:0]       r_conv_input_c;
    logic [15:0]       r_conv_kernel_h;
    logic [15:0]       r_conv_kernel_w;
    logic [15:0]       r_conv_stride_h;
    logic [15:0]       r_conv_stride_w;
    logic [15:0]       r_conv_pad_h;
    logic [15:0]       r_conv_pad_w;
    logic [15:0]       r_conv_dilation_h;
    logic [15:0]       r_conv_dilation_w;
    logic [31:0]       r_conv_k_base;
    logic              r_conv_stream_en;
    logic [ADDR_WIDTH-1:0] r_addr_q;

    assign gnt_o = 1'b1;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            r_sys_w_ptr <= '0;
            r_sys_i_ptr <= '0;
            r_sys_o_ptr <= '0;
            r_sys_psum_ptr <= '0;
            r_sys_dim_m <= '0;
            r_sys_start <= 1'b0;
            r_sys_done  <= 1'b0;
            r_sys_accum_en <= 1'b0;
            r_sys_ifm_stream_en <= 1'b0;
            r_requant_en <= 1'b0;
            r_requant_clamp_min <= 32'hFFFF_FF80;
            r_requant_clamp_max <= 32'h0000_007F;
            r_conv_start <= 1'b0;
            r_conv_done <= 1'b0;
            r_conv_input_ptr <= '0;
            r_conv_output_ptr <= '0;
            r_conv_rows <= '0;
            r_conv_output_w <= '0;
            r_conv_input_h <= '0;
            r_conv_input_w <= '0;
            r_conv_input_c <= '0;
            r_conv_kernel_h <= 16'd1;
            r_conv_kernel_w <= 16'd1;
            r_conv_stride_h <= 16'd1;
            r_conv_stride_w <= 16'd1;
            r_conv_pad_h <= '0;
            r_conv_pad_w <= '0;
            r_conv_dilation_h <= 16'd1;
            r_conv_dilation_w <= 16'd1;
            r_conv_k_base <= '0;
            r_conv_stream_en <= 1'b0;
            for (int unsigned ch = 0; ch < 32; ch++) begin
                r_requant_bias[ch] <= '0;
                r_requant_multiplier[ch] <= 32'd1;
                r_requant_shift[ch] <= '0;
                r_requant_zero_point[ch] <= '0;
            end
            r_addr_q <= '0;
            rvalid_o <= 1'b0;
        end else begin
            r_sys_start <= 1'b0;
            r_conv_start <= 1'b0;

            if (cfg_sys_done_i) begin
                r_sys_done <= 1'b1;
            end
            if (cfg_conv_done_i) begin
                r_conv_done <= 1'b1;
            end

            if (req_i && gnt_o) begin
                if (we_i) begin
                    for (int i = 0; i < DATA_WIDTH/32; i++) begin
                        if (be_i[i*4 +: 4] != 4'b0000) begin
                            logic [31:0] exact_addr;
                            logic [31:0] local_addr;
                            logic [31:0] wdata_word;

                            exact_addr = (addr_i & ~(32'(DATA_BYTES - 1))) + (i * 4);
                            local_addr = exact_addr & 32'hFFFF;
                            wdata_word = wdata_i[i*32 +: 32];

                            unique case (local_addr)
                                REG_SYS_W_PTR: r_sys_w_ptr <= wdata_word;
                                REG_SYS_I_PTR: r_sys_i_ptr <= wdata_word;
                                REG_SYS_O_PTR: r_sys_o_ptr <= wdata_word;
                                REG_SYS_PSUM_PTR: r_sys_psum_ptr <= wdata_word;
                                REG_SYS_DIM_M: r_sys_dim_m <= wdata_word;
                                REG_SYS_START: r_sys_start <= wdata_word[0];
                                REG_SYS_DONE:  r_sys_done  <= 1'b0;
                                REG_SYS_ACCUM_CTRL: begin
                                    r_sys_accum_en <= wdata_word[0];
                                    r_sys_ifm_stream_en <= wdata_word[1];
                                end
                                REG_RQ_CTRL:   r_requant_en <= wdata_word[0];
                                REG_RQ_CMIN:   r_requant_clamp_min <= wdata_word;
                                REG_RQ_CMAX:   r_requant_clamp_max <= wdata_word;
                                REG_CONV_INPUT_PTR: r_conv_input_ptr <= wdata_word;
                                REG_CONV_OUTPUT_PTR: r_conv_output_ptr <= wdata_word;
                                REG_CONV_ROWS: r_conv_rows <= wdata_word;
                                REG_CONV_OUTPUT_W: r_conv_output_w <= wdata_word;
                                REG_CONV_INPUT_H: r_conv_input_h <= wdata_word;
                                REG_CONV_INPUT_W: r_conv_input_w <= wdata_word;
                                REG_CONV_INPUT_C: r_conv_input_c <= wdata_word;
                                REG_CONV_KERNEL: begin
                                    r_conv_kernel_h <= wdata_word[15:0];
                                    r_conv_kernel_w <= wdata_word[31:16];
                                end
                                REG_CONV_STRIDE: begin
                                    r_conv_stride_h <= wdata_word[15:0];
                                    r_conv_stride_w <= wdata_word[31:16];
                                end
                                REG_CONV_PAD: begin
                                    r_conv_pad_h <= wdata_word[15:0];
                                    r_conv_pad_w <= wdata_word[31:16];
                                end
                                REG_CONV_DILATION: begin
                                    r_conv_dilation_h <= wdata_word[15:0];
                                    r_conv_dilation_w <= wdata_word[31:16];
                                end
                                REG_CONV_K_BASE: r_conv_k_base <= wdata_word;
                                REG_CONV_MODE: r_conv_stream_en <= wdata_word[0];
                                REG_CONV_START: r_conv_start <= wdata_word[0];
                                REG_CONV_DONE: r_conv_done <= 1'b0;
                                default: begin
                                    if ((local_addr & 32'hFF80) == REG_RQ_BIAS_BASE) begin
                                        r_requant_bias[(local_addr - REG_RQ_BIAS_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_MULT_BASE) begin
                                        r_requant_multiplier[(local_addr - REG_RQ_MULT_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_SHIFT_BASE) begin
                                        r_requant_shift[(local_addr - REG_RQ_SHIFT_BASE) >> 2] <= wdata_word[7:0];
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_ZP_BASE) begin
                                        r_requant_zero_point[(local_addr - REG_RQ_ZP_BASE) >> 2] <= wdata_word;
                                    end
                                end
                            endcase
                        end
                    end
                end else begin
                    r_addr_q <= addr_i & ~(32'(DATA_BYTES - 1));
                end
                rvalid_o <= 1'b1;
            end else begin
                rvalid_o <= 1'b0;
            end
        end
    end

    always_comb begin
        rdata_o = '0;
        if (rvalid_o) begin
            for (int i = 0; i < DATA_WIDTH/32; i++) begin
                logic [31:0] rdata_word;
                logic [31:0] exact_addr;

                exact_addr = (r_addr_q & 32'hFFFF) + (i * 4);
                rdata_word = '0;
                unique case (exact_addr)
                    REG_SYS_W_PTR: rdata_word = r_sys_w_ptr;
                    REG_SYS_I_PTR: rdata_word = r_sys_i_ptr;
                    REG_SYS_O_PTR: rdata_word = r_sys_o_ptr;
                    REG_SYS_PSUM_PTR: rdata_word = r_sys_psum_ptr;
                    REG_SYS_DIM_M: rdata_word = r_sys_dim_m;
                    REG_SYS_START: rdata_word = {31'd0, r_sys_start};
                    REG_SYS_DONE:  rdata_word = {31'd0, r_sys_done};
                    REG_SYS_ACCUM_CTRL: rdata_word = {30'd0, r_sys_ifm_stream_en, r_sys_accum_en};
                    REG_RQ_CTRL:   rdata_word = {31'd0, r_requant_en};
                    REG_RQ_CMIN:   rdata_word = r_requant_clamp_min;
                    REG_RQ_CMAX:   rdata_word = r_requant_clamp_max;
                    REG_CONV_INPUT_PTR: rdata_word = r_conv_input_ptr;
                    REG_CONV_OUTPUT_PTR: rdata_word = r_conv_output_ptr;
                    REG_CONV_ROWS: rdata_word = r_conv_rows;
                    REG_CONV_OUTPUT_W: rdata_word = r_conv_output_w;
                    REG_CONV_INPUT_H: rdata_word = r_conv_input_h;
                    REG_CONV_INPUT_W: rdata_word = r_conv_input_w;
                    REG_CONV_INPUT_C: rdata_word = r_conv_input_c;
                    REG_CONV_KERNEL: rdata_word = {r_conv_kernel_w, r_conv_kernel_h};
                    REG_CONV_STRIDE: rdata_word = {r_conv_stride_w, r_conv_stride_h};
                    REG_CONV_PAD: rdata_word = {r_conv_pad_w, r_conv_pad_h};
                    REG_CONV_DILATION: rdata_word = {r_conv_dilation_w, r_conv_dilation_h};
                    REG_CONV_K_BASE: rdata_word = r_conv_k_base;
                    REG_CONV_START: rdata_word = {31'd0, r_conv_start};
                    REG_CONV_DONE: rdata_word = {31'd0, r_conv_done};
                    REG_CONV_STATUS: rdata_word = {30'd0, r_conv_stream_en, cfg_conv_busy_i};
                    REG_CONV_MODE: rdata_word = {31'd0, r_conv_stream_en};
                    default: begin
                        if ((exact_addr & 32'hFF80) == REG_RQ_BIAS_BASE) begin
                            rdata_word = r_requant_bias[(exact_addr - REG_RQ_BIAS_BASE) >> 2];
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_MULT_BASE) begin
                            rdata_word = r_requant_multiplier[(exact_addr - REG_RQ_MULT_BASE) >> 2];
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_SHIFT_BASE) begin
                            rdata_word = {24'd0, r_requant_shift[(exact_addr - REG_RQ_SHIFT_BASE) >> 2]};
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_ZP_BASE) begin
                            rdata_word = r_requant_zero_point[(exact_addr - REG_RQ_ZP_BASE) >> 2];
                        end
                    end
                endcase
                rdata_o[i*32 +: 32] = rdata_word;
            end
        end
    end

    assign cfg_sys_start_o      = r_sys_start;
    assign cfg_sys_weight_ptr_o = r_sys_w_ptr;
    assign cfg_sys_ifm_ptr_o    = r_sys_i_ptr;
    assign cfg_sys_ofm_ptr_o    = r_sys_o_ptr;
    assign cfg_sys_psum_ptr_o   = r_sys_psum_ptr;
    assign cfg_sys_dim_m_o      = r_sys_dim_m;
    assign cfg_sys_accum_en_o   = r_sys_accum_en;
    assign cfg_sys_ifm_stream_en_o = r_sys_ifm_stream_en;
    assign cfg_requant_en_o     = r_requant_en;
    assign cfg_requant_bias_o   = r_requant_bias;
    assign cfg_requant_multiplier_o = r_requant_multiplier;
    assign cfg_requant_shift_o  = r_requant_shift;
    assign cfg_requant_zero_point_o = r_requant_zero_point;
    assign cfg_requant_clamp_min_o = r_requant_clamp_min;
    assign cfg_requant_clamp_max_o = r_requant_clamp_max;
    assign cfg_conv_start_o = r_conv_start;
    assign cfg_conv_input_ptr_o = r_conv_input_ptr;
    assign cfg_conv_output_ptr_o = r_conv_output_ptr;
    assign cfg_conv_rows_o = r_conv_rows;
    assign cfg_conv_output_w_o = r_conv_output_w;
    assign cfg_conv_input_h_o = r_conv_input_h;
    assign cfg_conv_input_w_o = r_conv_input_w;
    assign cfg_conv_input_c_o = r_conv_input_c;
    assign cfg_conv_kernel_h_o = r_conv_kernel_h;
    assign cfg_conv_kernel_w_o = r_conv_kernel_w;
    assign cfg_conv_stride_h_o = r_conv_stride_h;
    assign cfg_conv_stride_w_o = r_conv_stride_w;
    assign cfg_conv_pad_h_o = r_conv_pad_h;
    assign cfg_conv_pad_w_o = r_conv_pad_w;
    assign cfg_conv_dilation_h_o = r_conv_dilation_h;
    assign cfg_conv_dilation_w_o = r_conv_dilation_w;
    assign cfg_conv_k_base_o = r_conv_k_base;
    assign cfg_conv_stream_en_o = r_conv_stream_en;

endmodule
