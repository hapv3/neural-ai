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
    output logic [31:0]               cfg_sys_ofm_row_stride_bytes_o,
    output logic [31:0]               cfg_sys_ofm_tile_cols_o,
    output logic [31:0]               cfg_sys_psum_row_stride_bytes_o,
    output logic                      cfg_requant_en_o,
    output logic [31:0][31:0]         cfg_requant_bias_o,
    output logic [31:0][31:0]         cfg_requant_multiplier_o,
    output logic [31:0][7:0]          cfg_requant_shift_o,
    output logic [31:0][31:0]         cfg_requant_zero_point_o,
    output logic [31:0]               cfg_requant_clamp_min_o,
    output logic [31:0]               cfg_requant_clamp_max_o,
    output logic                      cfg_linebuf_en_o,
    output logic                      cfg_linebuf_coalesce_o,
    output logic                      cfg_linebuf_pool_o,
    output logic                      cfg_linebuf_c32_fast_o,
    output logic                      cfg_linebuf_depthwise_o,
    output logic [31:0]               cfg_linebuf_input_base_o,
    output logic [15:0]               cfg_linebuf_input_h_o,
    output logic [15:0]               cfg_linebuf_input_w_o,
    output logic [15:0]               cfg_linebuf_input_c_o,
    output logic [15:0]               cfg_linebuf_output_w_o,
    output logic [15:0]               cfg_linebuf_stride_h_o,
    output logic [15:0]               cfg_linebuf_stride_w_o,
    output logic [15:0]               cfg_linebuf_pad_h_o,
    output logic [15:0]               cfg_linebuf_pad_w_o,
    output logic [31:0]               cfg_linebuf_row_stride_bytes_o,
    output logic [31:0]               cfg_linebuf_pixel_stride_bytes_o,
    output logic [31:0]               cfg_linebuf_ow_step_bytes_o,
    output logic [31:0]               cfg_linebuf_oh_step_bytes_o,
    output logic [15:0]               cfg_linebuf_kernel_h_o,
    output logic [15:0]               cfg_linebuf_kernel_w_o,
    output logic [15:0]               cfg_linebuf_c_base_o,
    output logic [5:0]                cfg_linebuf_lane_base_o,
    output logic                      cfg_linebuf_kgen_o,
    output logic [31:0]               cfg_linebuf_k_tiles_o,
    output logic [15:0]               cfg_linebuf_k_seed_ic_o,
    output logic [7:0]                cfg_linebuf_k_seed_kw_o,
    output logic [7:0]                cfg_linebuf_k_seed_kh_o,
    output logic [31:0]               cfg_linebuf_spatial_m_o,
    output logic [5:0]                cfg_linebuf_block_valid_bytes_o,
    output logic [31:0]               cfg_linebuf_channel_addr_offset_o,
    output logic [31:0]               cfg_linebuf_coalesce_k_bytes_o,
    input  logic                      cfg_sys_done_i
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
    localparam logic [ADDR_WIDTH-1:0] REG_LB_CTRL = 32'h0400;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_INPUT_BASE = 32'h0404;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_INPUT_H = 32'h0408;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_INPUT_W = 32'h040C;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_INPUT_C = 32'h0410;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_OUTPUT_W = 32'h0414;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_STRIDE = 32'h0418;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_PAD = 32'h041C;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_ROW_STRIDE = 32'h0428;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_PIXEL_STRIDE = 32'h042C;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_OW_STEP = 32'h0430;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_OH_STEP = 32'h0434;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_KERNEL = 32'h0438;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_C_BASE = 32'h043C;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_SPATIAL_M = 32'h0440;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_LANE_BASE = 32'h0444;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_K_TILES = 32'h0448;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_K_SEED = 32'h044C;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_OFM_ROW_STRIDE = 32'h0450;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_OFM_TILE_COLS = 32'h0454;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_PSUM_ROW_STRIDE = 32'h0458;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_PRECOMP0 = 32'h045C;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_CHANNEL_OFFSET = 32'h0460;
    localparam logic [ADDR_WIDTH-1:0] REG_LB_COALESCE_K_BYTES = 32'h0464;

    logic [31:0] r_sys_w_ptr;
    logic [31:0] r_sys_i_ptr;
    logic [31:0] r_sys_o_ptr;
    logic [31:0] r_sys_psum_ptr;
    logic [31:0] r_sys_dim_m;
    logic [31:0] r_sys_ofm_row_stride_bytes;
    logic [31:0] r_sys_ofm_tile_cols;
    logic [31:0] r_sys_psum_row_stride_bytes;
    logic        r_sys_start;
    logic        r_sys_done;
    logic        r_sys_accum_en;
    logic        r_requant_en;
    logic [31:0][31:0] r_requant_bias;
    logic [31:0][31:0] r_requant_multiplier;
    logic [31:0][7:0]  r_requant_shift;
    logic [31:0][31:0] r_requant_zero_point;
    logic [31:0]       r_requant_clamp_min;
    logic [31:0]       r_requant_clamp_max;
    logic              r_linebuf_en;
    logic              r_linebuf_coalesce;
    logic              r_linebuf_pool;
    logic              r_linebuf_kgen;
    logic              r_linebuf_c32_fast;
    logic              r_linebuf_depthwise;
    logic [31:0]       r_linebuf_input_base;
    logic [15:0]       r_linebuf_input_h;
    logic [15:0]       r_linebuf_input_w;
    logic [15:0]       r_linebuf_input_c;
    logic [15:0]       r_linebuf_output_w;
    logic [15:0]       r_linebuf_stride_h;
    logic [15:0]       r_linebuf_stride_w;
    logic [15:0]       r_linebuf_pad_h;
    logic [15:0]       r_linebuf_pad_w;
    logic [31:0]       r_linebuf_row_stride_bytes;
    logic [31:0]       r_linebuf_pixel_stride_bytes;
    logic [31:0]       r_linebuf_ow_step_bytes;
    logic [31:0]       r_linebuf_oh_step_bytes;
    logic [15:0]       r_linebuf_kernel_h;
    logic [15:0]       r_linebuf_kernel_w;
    logic [15:0]       r_linebuf_c_base;
    logic [5:0]        r_linebuf_lane_base;
    logic [31:0]       r_linebuf_k_tiles;
    logic [15:0]       r_linebuf_k_seed_ic;
    logic [7:0]        r_linebuf_k_seed_kw;
    logic [7:0]        r_linebuf_k_seed_kh;
    logic [31:0]       r_linebuf_spatial_m;
    logic [5:0]        r_linebuf_block_valid_bytes;
    logic [31:0]       r_linebuf_channel_addr_offset;
    logic [31:0]       r_linebuf_coalesce_k_bytes;

    logic [31:0] s_sys_w_ptr;
    logic [31:0] s_sys_i_ptr;
    logic [31:0] s_sys_o_ptr;
    logic [31:0] s_sys_psum_ptr;
    logic [31:0] s_sys_dim_m;
    logic [31:0] s_sys_ofm_row_stride_bytes;
    logic [31:0] s_sys_ofm_tile_cols;
    logic [31:0] s_sys_psum_row_stride_bytes;
    logic        s_sys_accum_en;
    logic        s_requant_en;
    logic [31:0][31:0] s_requant_bias;
    logic [31:0][31:0] s_requant_multiplier;
    logic [31:0][7:0]  s_requant_shift;
    logic [31:0][31:0] s_requant_zero_point;
    logic [31:0]       s_requant_clamp_min;
    logic [31:0]       s_requant_clamp_max;
    logic              s_linebuf_en;
    logic              s_linebuf_coalesce;
    logic              s_linebuf_pool;
    logic              s_linebuf_kgen;
    logic              s_linebuf_c32_fast;
    logic              s_linebuf_depthwise;
    logic [31:0]       s_linebuf_input_base;
    logic [15:0]       s_linebuf_input_h;
    logic [15:0]       s_linebuf_input_w;
    logic [15:0]       s_linebuf_input_c;
    logic [15:0]       s_linebuf_output_w;
    logic [15:0]       s_linebuf_stride_h;
    logic [15:0]       s_linebuf_stride_w;
    logic [15:0]       s_linebuf_pad_h;
    logic [15:0]       s_linebuf_pad_w;
    logic [31:0]       s_linebuf_row_stride_bytes;
    logic [31:0]       s_linebuf_pixel_stride_bytes;
    logic [31:0]       s_linebuf_ow_step_bytes;
    logic [31:0]       s_linebuf_oh_step_bytes;
    logic [15:0]       s_linebuf_kernel_h;
    logic [15:0]       s_linebuf_kernel_w;
    logic [15:0]       s_linebuf_c_base;
    logic [5:0]        s_linebuf_lane_base;
    logic [31:0]       s_linebuf_k_tiles;
    logic [15:0]       s_linebuf_k_seed_ic;
    logic [7:0]        s_linebuf_k_seed_kw;
    logic [7:0]        s_linebuf_k_seed_kh;
    logic [31:0]       s_linebuf_spatial_m;
    logic [5:0]        s_linebuf_block_valid_bytes;
    logic [31:0]       s_linebuf_channel_addr_offset;
    logic [31:0]       s_linebuf_coalesce_k_bytes;
    logic [ADDR_WIDTH-1:0] r_addr_q;

    assign gnt_o = 1'b1;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            r_sys_w_ptr <= '0;
            r_sys_i_ptr <= '0;
            r_sys_o_ptr <= '0;
            r_sys_psum_ptr <= '0;
            r_sys_dim_m <= '0;
            r_sys_ofm_row_stride_bytes <= '0;
            r_sys_ofm_tile_cols <= '0;
            r_sys_psum_row_stride_bytes <= '0;
            r_sys_start <= 1'b0;
            r_sys_done  <= 1'b0;
            r_sys_accum_en <= 1'b0;
            r_requant_en <= 1'b0;
            r_requant_clamp_min <= 32'hFFFF_FF80;
            r_requant_clamp_max <= 32'h0000_007F;
            r_linebuf_en <= 1'b0;
            r_linebuf_coalesce <= 1'b0;
            r_linebuf_pool <= 1'b0;
            r_linebuf_kgen <= 1'b0;
            r_linebuf_c32_fast <= 1'b0;
            r_linebuf_depthwise <= 1'b0;
            r_linebuf_input_base <= '0;
            r_linebuf_input_h <= '0;
            r_linebuf_input_w <= '0;
            r_linebuf_input_c <= '0;
            r_linebuf_output_w <= '0;
            r_linebuf_stride_h <= 16'd1;
            r_linebuf_stride_w <= 16'd1;
            r_linebuf_pad_h <= '0;
            r_linebuf_pad_w <= '0;
            r_linebuf_row_stride_bytes <= '0;
            r_linebuf_pixel_stride_bytes <= '0;
            r_linebuf_ow_step_bytes <= '0;
            r_linebuf_oh_step_bytes <= '0;
            r_linebuf_kernel_h <= 16'd3;
            r_linebuf_kernel_w <= 16'd3;
            r_linebuf_c_base <= '0;
            r_linebuf_lane_base <= '0;
            r_linebuf_k_tiles <= '0;
            r_linebuf_k_seed_ic <= '0;
            r_linebuf_k_seed_kw <= '0;
            r_linebuf_k_seed_kh <= '0;
            r_linebuf_spatial_m <= '0;
            r_linebuf_block_valid_bytes <= '0;
            r_linebuf_channel_addr_offset <= '0;
            r_linebuf_coalesce_k_bytes <= '0;
            s_sys_w_ptr <= '0;
            s_sys_i_ptr <= '0;
            s_sys_o_ptr <= '0;
            s_sys_psum_ptr <= '0;
            s_sys_dim_m <= '0;
            s_sys_ofm_row_stride_bytes <= '0;
            s_sys_ofm_tile_cols <= '0;
            s_sys_psum_row_stride_bytes <= '0;
            s_sys_accum_en <= 1'b0;
            s_requant_en <= 1'b0;
            s_requant_clamp_min <= 32'hFFFF_FF80;
            s_requant_clamp_max <= 32'h0000_007F;
            s_linebuf_en <= 1'b0;
            s_linebuf_coalesce <= 1'b0;
            s_linebuf_pool <= 1'b0;
            s_linebuf_kgen <= 1'b0;
            s_linebuf_c32_fast <= 1'b0;
            s_linebuf_depthwise <= 1'b0;
            s_linebuf_input_base <= '0;
            s_linebuf_input_h <= '0;
            s_linebuf_input_w <= '0;
            s_linebuf_input_c <= '0;
            s_linebuf_output_w <= '0;
            s_linebuf_stride_h <= 16'd1;
            s_linebuf_stride_w <= 16'd1;
            s_linebuf_pad_h <= '0;
            s_linebuf_pad_w <= '0;
            s_linebuf_row_stride_bytes <= '0;
            s_linebuf_pixel_stride_bytes <= '0;
            s_linebuf_ow_step_bytes <= '0;
            s_linebuf_oh_step_bytes <= '0;
            s_linebuf_kernel_h <= 16'd3;
            s_linebuf_kernel_w <= 16'd3;
            s_linebuf_c_base <= '0;
            s_linebuf_lane_base <= '0;
            s_linebuf_k_tiles <= '0;
            s_linebuf_k_seed_ic <= '0;
            s_linebuf_k_seed_kw <= '0;
            s_linebuf_k_seed_kh <= '0;
            s_linebuf_spatial_m <= '0;
            s_linebuf_block_valid_bytes <= '0;
            s_linebuf_channel_addr_offset <= '0;
            s_linebuf_coalesce_k_bytes <= '0;
            for (int unsigned ch = 0; ch < 32; ch++) begin
                r_requant_bias[ch] <= '0;
                r_requant_multiplier[ch] <= 32'd1;
                r_requant_shift[ch] <= '0;
                r_requant_zero_point[ch] <= '0;
                s_requant_bias[ch] <= '0;
                s_requant_multiplier[ch] <= 32'd1;
                s_requant_shift[ch] <= '0;
                s_requant_zero_point[ch] <= '0;
            end
            r_addr_q <= '0;
            rvalid_o <= 1'b0;
        end else begin
            r_sys_start <= 1'b0;

            if (cfg_sys_done_i) begin
                r_sys_done <= 1'b1;
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
                                REG_SYS_W_PTR: s_sys_w_ptr <= wdata_word;
                                REG_SYS_I_PTR: s_sys_i_ptr <= wdata_word;
                                REG_SYS_O_PTR: s_sys_o_ptr <= wdata_word;
                                REG_SYS_PSUM_PTR: s_sys_psum_ptr <= wdata_word;
                                REG_SYS_DIM_M: s_sys_dim_m <= wdata_word;
                                REG_SYS_OFM_ROW_STRIDE: s_sys_ofm_row_stride_bytes <= wdata_word;
                                REG_SYS_OFM_TILE_COLS: s_sys_ofm_tile_cols <= wdata_word;
                                REG_SYS_PSUM_ROW_STRIDE: s_sys_psum_row_stride_bytes <= wdata_word;
                                REG_SYS_START: begin
                                    if (wdata_word[0]) begin
                                        r_sys_w_ptr <= s_sys_w_ptr;
                                        r_sys_i_ptr <= s_sys_i_ptr;
                                        r_sys_o_ptr <= s_sys_o_ptr;
                                        r_sys_psum_ptr <= s_sys_psum_ptr;
                                        r_sys_dim_m <= s_sys_dim_m;
                                        r_sys_ofm_row_stride_bytes <= s_sys_ofm_row_stride_bytes;
                                        r_sys_ofm_tile_cols <= s_sys_ofm_tile_cols;
                                        r_sys_psum_row_stride_bytes <= s_sys_psum_row_stride_bytes;
                                        r_sys_accum_en <= s_sys_accum_en;
                                        r_requant_en <= s_requant_en;
                                        r_requant_clamp_min <= s_requant_clamp_min;
                                        r_requant_clamp_max <= s_requant_clamp_max;
                                        r_linebuf_en <= s_linebuf_en;
                                        r_linebuf_coalesce <= s_linebuf_coalesce;
                                        r_linebuf_kgen <= s_linebuf_kgen;
                                        r_linebuf_pool <= s_linebuf_pool;
                                        r_linebuf_c32_fast <= s_linebuf_c32_fast;
                                        r_linebuf_depthwise <= s_linebuf_depthwise;
                                        r_linebuf_input_base <= s_linebuf_input_base;
                                        r_linebuf_input_h <= s_linebuf_input_h;
                                        r_linebuf_input_w <= s_linebuf_input_w;
                                        r_linebuf_input_c <= s_linebuf_input_c;
                                        r_linebuf_output_w <= s_linebuf_output_w;
                                        r_linebuf_stride_h <= s_linebuf_stride_h;
                                        r_linebuf_stride_w <= s_linebuf_stride_w;
                                        r_linebuf_pad_h <= s_linebuf_pad_h;
                                        r_linebuf_pad_w <= s_linebuf_pad_w;
                                        r_linebuf_row_stride_bytes <= s_linebuf_row_stride_bytes;
                                        r_linebuf_pixel_stride_bytes <= s_linebuf_pixel_stride_bytes;
                                        r_linebuf_ow_step_bytes <= s_linebuf_ow_step_bytes;
                                        r_linebuf_oh_step_bytes <= s_linebuf_oh_step_bytes;
                                        r_linebuf_kernel_h <= s_linebuf_kernel_h;
                                        r_linebuf_kernel_w <= s_linebuf_kernel_w;
                                        r_linebuf_c_base <= s_linebuf_c_base;
                                        r_linebuf_lane_base <= s_linebuf_lane_base;
                                        r_linebuf_k_tiles <= s_linebuf_k_tiles;
                                        r_linebuf_k_seed_ic <= s_linebuf_k_seed_ic;
                                        r_linebuf_k_seed_kw <= s_linebuf_k_seed_kw;
                                        r_linebuf_k_seed_kh <= s_linebuf_k_seed_kh;
                                        r_linebuf_spatial_m <= s_linebuf_spatial_m;
                                        r_linebuf_block_valid_bytes <= s_linebuf_block_valid_bytes;
                                        r_linebuf_channel_addr_offset <= s_linebuf_channel_addr_offset;
                                        r_linebuf_coalesce_k_bytes <= s_linebuf_coalesce_k_bytes;
                                        for (int unsigned ch = 0; ch < 32; ch++) begin
                                            r_requant_bias[ch] <= s_requant_bias[ch];
                                            r_requant_multiplier[ch] <= s_requant_multiplier[ch];
                                            r_requant_shift[ch] <= s_requant_shift[ch];
                                            r_requant_zero_point[ch] <= s_requant_zero_point[ch];
                                        end
                                        r_sys_done <= 1'b0;
                                        r_sys_start <= 1'b1;
                                    end
                                end
                                REG_SYS_DONE:  r_sys_done  <= 1'b0;
                                REG_SYS_ACCUM_CTRL: s_sys_accum_en <= wdata_word[0];
                                REG_RQ_CTRL:   s_requant_en <= wdata_word[0];
                                REG_RQ_CMIN:   s_requant_clamp_min <= wdata_word;
                                REG_RQ_CMAX:   s_requant_clamp_max <= wdata_word;
                                REG_LB_CTRL: begin
                                    s_linebuf_en <= wdata_word[0];
                                    s_linebuf_coalesce <= wdata_word[1];
                                    s_linebuf_kgen <= wdata_word[2];
                                    s_linebuf_pool <= wdata_word[3];
                                    s_linebuf_c32_fast <= wdata_word[4];
                                    s_linebuf_depthwise <= wdata_word[5];
                                end
                                REG_LB_INPUT_BASE: s_linebuf_input_base <= wdata_word;
                                REG_LB_INPUT_H: s_linebuf_input_h <= wdata_word[15:0];
                                REG_LB_INPUT_W: s_linebuf_input_w <= wdata_word[15:0];
                                REG_LB_INPUT_C: s_linebuf_input_c <= wdata_word[15:0];
                                REG_LB_OUTPUT_W: s_linebuf_output_w <= wdata_word[15:0];
                                REG_LB_STRIDE: begin
                                    s_linebuf_stride_h <= wdata_word[15:0];
                                    s_linebuf_stride_w <= wdata_word[31:16];
                                end
                                REG_LB_PAD: begin
                                    s_linebuf_pad_h <= wdata_word[15:0];
                                    s_linebuf_pad_w <= wdata_word[31:16];
                                end
                                REG_LB_ROW_STRIDE: s_linebuf_row_stride_bytes <= wdata_word;
                                REG_LB_PIXEL_STRIDE: s_linebuf_pixel_stride_bytes <= wdata_word;
                                REG_LB_OW_STEP: s_linebuf_ow_step_bytes <= wdata_word;
                                REG_LB_OH_STEP: s_linebuf_oh_step_bytes <= wdata_word;
                                REG_LB_KERNEL: begin
                                    s_linebuf_kernel_h <= wdata_word[15:0];
                                    s_linebuf_kernel_w <= wdata_word[31:16];
                                end
                                REG_LB_C_BASE: s_linebuf_c_base <= wdata_word[15:0];
                                REG_LB_SPATIAL_M: s_linebuf_spatial_m <= wdata_word;
                                REG_LB_LANE_BASE: s_linebuf_lane_base <= wdata_word[5:0];
                                REG_LB_K_TILES: s_linebuf_k_tiles <= wdata_word;
                                REG_LB_K_SEED: begin
                                    s_linebuf_k_seed_ic <= wdata_word[15:0];
                                    s_linebuf_k_seed_kw <= wdata_word[23:16];
                                    s_linebuf_k_seed_kh <= wdata_word[31:24];
                                end
                                REG_LB_PRECOMP0: s_linebuf_block_valid_bytes <= wdata_word[5:0];
                                REG_LB_CHANNEL_OFFSET: s_linebuf_channel_addr_offset <= wdata_word;
                                REG_LB_COALESCE_K_BYTES: s_linebuf_coalesce_k_bytes <= wdata_word;
                                default: begin
                                    if ((local_addr & 32'hFF80) == REG_RQ_BIAS_BASE) begin
                                        s_requant_bias[(local_addr - REG_RQ_BIAS_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_MULT_BASE) begin
                                        s_requant_multiplier[(local_addr - REG_RQ_MULT_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_SHIFT_BASE) begin
                                        s_requant_shift[(local_addr - REG_RQ_SHIFT_BASE) >> 2] <= wdata_word[7:0];
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_ZP_BASE) begin
                                        s_requant_zero_point[(local_addr - REG_RQ_ZP_BASE) >> 2] <= wdata_word;
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
                    REG_SYS_OFM_ROW_STRIDE: rdata_word = r_sys_ofm_row_stride_bytes;
                    REG_SYS_OFM_TILE_COLS: rdata_word = r_sys_ofm_tile_cols;
                    REG_SYS_PSUM_ROW_STRIDE: rdata_word = r_sys_psum_row_stride_bytes;
                    REG_SYS_START: rdata_word = {31'd0, r_sys_start};
                    REG_SYS_DONE:  rdata_word = {31'd0, r_sys_done};
                    REG_SYS_ACCUM_CTRL: rdata_word = {31'd0, r_sys_accum_en};
                    REG_RQ_CTRL:   rdata_word = {31'd0, r_requant_en};
                    REG_RQ_CMIN:   rdata_word = r_requant_clamp_min;
                    REG_RQ_CMAX:   rdata_word = r_requant_clamp_max;
                    REG_LB_CTRL: rdata_word = {26'd0, r_linebuf_depthwise, r_linebuf_c32_fast, r_linebuf_pool, r_linebuf_kgen,
                                               r_linebuf_coalesce, r_linebuf_en};
                    REG_LB_INPUT_BASE: rdata_word = r_linebuf_input_base;
                    REG_LB_INPUT_H: rdata_word = {16'd0, r_linebuf_input_h};
                    REG_LB_INPUT_W: rdata_word = {16'd0, r_linebuf_input_w};
                    REG_LB_INPUT_C: rdata_word = {16'd0, r_linebuf_input_c};
                    REG_LB_OUTPUT_W: rdata_word = {16'd0, r_linebuf_output_w};
                    REG_LB_STRIDE: rdata_word = {r_linebuf_stride_w, r_linebuf_stride_h};
                    REG_LB_PAD: rdata_word = {r_linebuf_pad_w, r_linebuf_pad_h};
                    REG_LB_ROW_STRIDE: rdata_word = r_linebuf_row_stride_bytes;
                    REG_LB_PIXEL_STRIDE: rdata_word = r_linebuf_pixel_stride_bytes;
                    REG_LB_OW_STEP: rdata_word = r_linebuf_ow_step_bytes;
                    REG_LB_OH_STEP: rdata_word = r_linebuf_oh_step_bytes;
                    REG_LB_KERNEL: rdata_word = {r_linebuf_kernel_w, r_linebuf_kernel_h};
                    REG_LB_C_BASE: rdata_word = {16'd0, r_linebuf_c_base};
                    REG_LB_SPATIAL_M: rdata_word = r_linebuf_spatial_m;
                    REG_LB_LANE_BASE: rdata_word = {26'd0, r_linebuf_lane_base};
                    REG_LB_K_TILES: rdata_word = r_linebuf_k_tiles;
                    REG_LB_K_SEED: rdata_word = {r_linebuf_k_seed_kh, r_linebuf_k_seed_kw, r_linebuf_k_seed_ic};
                    REG_LB_PRECOMP0: rdata_word = {26'd0, r_linebuf_block_valid_bytes};
                    REG_LB_CHANNEL_OFFSET: rdata_word = r_linebuf_channel_addr_offset;
                    REG_LB_COALESCE_K_BYTES: rdata_word = r_linebuf_coalesce_k_bytes;
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
    assign cfg_sys_ofm_row_stride_bytes_o = r_sys_ofm_row_stride_bytes;
    assign cfg_sys_ofm_tile_cols_o = r_sys_ofm_tile_cols;
    assign cfg_sys_psum_row_stride_bytes_o = r_sys_psum_row_stride_bytes;
    assign cfg_requant_en_o     = r_requant_en;
    assign cfg_requant_bias_o   = r_requant_bias;
    assign cfg_requant_multiplier_o = r_requant_multiplier;
    assign cfg_requant_shift_o  = r_requant_shift;
    assign cfg_requant_zero_point_o = r_requant_zero_point;
    assign cfg_requant_clamp_min_o = r_requant_clamp_min;
    assign cfg_requant_clamp_max_o = r_requant_clamp_max;
    assign cfg_linebuf_en_o = r_linebuf_en;
    assign cfg_linebuf_coalesce_o = r_linebuf_coalesce;
    assign cfg_linebuf_pool_o = r_linebuf_pool;
    assign cfg_linebuf_kgen_o = r_linebuf_kgen;
    assign cfg_linebuf_c32_fast_o = r_linebuf_c32_fast;
    assign cfg_linebuf_depthwise_o = r_linebuf_depthwise;
    assign cfg_linebuf_input_base_o = r_linebuf_input_base;
    assign cfg_linebuf_input_h_o = r_linebuf_input_h;
    assign cfg_linebuf_input_w_o = r_linebuf_input_w;
    assign cfg_linebuf_input_c_o = r_linebuf_input_c;
    assign cfg_linebuf_output_w_o = r_linebuf_output_w;
    assign cfg_linebuf_stride_h_o = r_linebuf_stride_h;
    assign cfg_linebuf_stride_w_o = r_linebuf_stride_w;
    assign cfg_linebuf_pad_h_o = r_linebuf_pad_h;
    assign cfg_linebuf_pad_w_o = r_linebuf_pad_w;
    assign cfg_linebuf_row_stride_bytes_o = r_linebuf_row_stride_bytes;
    assign cfg_linebuf_pixel_stride_bytes_o = r_linebuf_pixel_stride_bytes;
    assign cfg_linebuf_ow_step_bytes_o = r_linebuf_ow_step_bytes;
    assign cfg_linebuf_oh_step_bytes_o = r_linebuf_oh_step_bytes;
    assign cfg_linebuf_kernel_h_o = r_linebuf_kernel_h;
    assign cfg_linebuf_kernel_w_o = r_linebuf_kernel_w;
    assign cfg_linebuf_c_base_o = r_linebuf_c_base;
    assign cfg_linebuf_lane_base_o = r_linebuf_lane_base;
    assign cfg_linebuf_k_tiles_o = r_linebuf_k_tiles;
    assign cfg_linebuf_k_seed_ic_o = r_linebuf_k_seed_ic;
    assign cfg_linebuf_k_seed_kw_o = r_linebuf_k_seed_kw;
    assign cfg_linebuf_k_seed_kh_o = r_linebuf_k_seed_kh;
    assign cfg_linebuf_spatial_m_o = r_linebuf_spatial_m;
    assign cfg_linebuf_block_valid_bytes_o = r_linebuf_block_valid_bytes;
    assign cfg_linebuf_channel_addr_offset_o = r_linebuf_channel_addr_offset;
    assign cfg_linebuf_coalesce_k_bytes_o = r_linebuf_coalesce_k_bytes;

endmodule
