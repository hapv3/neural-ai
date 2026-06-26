#include "conv2d_feeder_sw.h"
#include "hal_systolic.h"

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static int32_t signed_coord(uint32_t base, uint32_t offset, uint32_t pad) {
    return (int32_t)base + (int32_t)offset - (int32_t)pad;
}

static int8_t load_nhwc_i8(uint32_t input_addr,
                           uint32_t input_h,
                           uint32_t input_w,
                           uint32_t input_c,
                           int32_t ih,
                           int32_t iw,
                           uint32_t ic) {
    if ((ih < 0) || (iw < 0) || ((uint32_t)ih >= input_h) || ((uint32_t)iw >= input_w) || (ic >= input_c)) {
        return 0;
    }

    uint32_t offset = ((((uint32_t)ih * input_w) + (uint32_t)iw) * input_c) + ic;
    volatile const uint32_t *input_words = (volatile const uint32_t *)(input_addr & ~3u);
    uint32_t word = input_words[offset >> 2];
    uint32_t shift = (offset & 3u) * 8u;
    return (int8_t)((word >> shift) & 0xFFu);
}

void conv2d_feeder_sw_materialize_k_tile(const conv2d_feeder_sw_cfg_t *cfg, uint32_t k_block) {
    volatile uint32_t *im2col_words = (volatile uint32_t *)cfg->im2col_addr;
    uint32_t rows = cfg->output_h * cfg->output_w;
    uint32_t kernel_spatial = cfg->kernel_h * cfg->kernel_w;
    uint32_t k_total = kernel_spatial * cfg->input_c;
    uint32_t k_base = k_block * CONV2D_FEEDER_K_TILE;

    for (uint32_t row = 0; row < rows; row++) {
        uint32_t oh = row / cfg->output_w;
        uint32_t ow = row - (oh * cfg->output_w);

        for (uint32_t lane_word = 0; lane_word < (CONV2D_FEEDER_K_TILE / 4u); lane_word++) {
            uint32_t packed = 0;

            for (uint32_t byte = 0; byte < 4u; byte++) {
                uint32_t lane = (lane_word * 4u) + byte;
                uint32_t k_index = k_base + lane;
                int8_t value = 0;

                if (k_index < k_total) {
                    uint32_t spatial_index = k_index / cfg->input_c;
                    uint32_t ic = k_index - (spatial_index * cfg->input_c);
                    uint32_t kh = spatial_index / cfg->kernel_w;
                    uint32_t kw = spatial_index - (kh * cfg->kernel_w);
                    int32_t ih = signed_coord(oh * cfg->stride_h, kh * cfg->dilation_h, cfg->pad_h);
                    int32_t iw = signed_coord(ow * cfg->stride_w, kw * cfg->dilation_w, cfg->pad_w);

                    value = load_nhwc_i8(cfg->input_addr,
                                         cfg->input_h,
                                         cfg->input_w,
                                         cfg->input_c,
                                         ih,
                                         iw,
                                         ic);
                }

                packed |= ((uint32_t)((uint8_t)value)) << (byte * 8u);
            }

            im2col_words[(row * (CONV2D_FEEDER_K_TILE / 4u)) + lane_word] = packed;
        }
    }
}

void conv2d_feeder_sw_run_oc32(const conv2d_feeder_sw_cfg_t *cfg) {
    uint32_t rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_blocks = ceil_div_u32(k_total, CONV2D_FEEDER_K_TILE);

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        uint32_t weight_addr = cfg->weight_addr + (k_block * CONV2D_FEEDER_K_TILE * CONV2D_FEEDER_OC_TILE);

        conv2d_feeder_sw_materialize_k_tile(cfg, k_block);

        if (k_block == 0u) {
            systolic_gemm32(weight_addr, cfg->im2col_addr, cfg->output_addr, rows);
        } else {
            systolic_gemm32_accumulate(weight_addr, cfg->im2col_addr, cfg->output_addr, cfg->output_addr, rows);
        }
    }
}
