#include "hal_conv2d_feeder.h"
#include "npu_memory_map.h"

static uint32_t pack_u16_pair(uint32_t lo, uint32_t hi) {
    return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16);
}

void conv2d_feeder_hw_start_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base, uint32_t stream_en) {
    REG_WRITE(REG_CONV_INPUT_PTR, cfg->input_addr);
    REG_WRITE(REG_CONV_OUTPUT_PTR, cfg->output_addr);
    REG_WRITE(REG_CONV_ROWS, cfg->rows);
    REG_WRITE(REG_CONV_OUTPUT_W, cfg->output_w);
    REG_WRITE(REG_CONV_INPUT_H, cfg->input_h);
    REG_WRITE(REG_CONV_INPUT_W, cfg->input_w);
    REG_WRITE(REG_CONV_INPUT_C, cfg->input_c);
    REG_WRITE(REG_CONV_KERNEL, pack_u16_pair(cfg->kernel_h, cfg->kernel_w));
    REG_WRITE(REG_CONV_STRIDE, pack_u16_pair(cfg->stride_h, cfg->stride_w));
    REG_WRITE(REG_CONV_PAD, pack_u16_pair(cfg->pad_h, cfg->pad_w));
    REG_WRITE(REG_CONV_DILATION, pack_u16_pair(cfg->dilation_h, cfg->dilation_w));
    REG_WRITE(REG_CONV_K_BASE, k_base);
    REG_WRITE(REG_CONV_MODE, stream_en ? REG_CONV_MODE_STREAM_EN : 0u);
    REG_WRITE(REG_CONV_DONE, 0u);
    REG_WRITE(REG_CONV_START, REG_CONV_START_EN);
}

void conv2d_feeder_hw_wait(void) {
    while (REG_READ(REG_CONV_DONE) == 0u) {
    }

    REG_WRITE(REG_CONV_DONE, 0u);
    REG_WRITE(REG_CONV_MODE, 0u);
}

void conv2d_feeder_hw_materialize_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base) {
    conv2d_feeder_hw_start_k_tile(cfg, k_base, 0u);
    conv2d_feeder_hw_wait();
}

void conv2d_feeder_hw_start_stream_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base) {
    conv2d_feeder_hw_start_k_tile(cfg, k_base, 1u);
}
