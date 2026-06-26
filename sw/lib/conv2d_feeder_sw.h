#ifndef CONV2D_FEEDER_SW_H
#define CONV2D_FEEDER_SW_H

#include "npu_types.h"

#define CONV2D_FEEDER_OC_TILE 32u
#define CONV2D_FEEDER_K_TILE  32u

typedef struct {
    uint32_t input_addr;
    uint32_t weight_addr;
    uint32_t im2col_addr;
    uint32_t output_addr;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
} conv2d_feeder_sw_cfg_t;

void conv2d_feeder_sw_materialize_k_tile(const conv2d_feeder_sw_cfg_t *cfg, uint32_t k_block);
void conv2d_feeder_sw_run_oc32(const conv2d_feeder_sw_cfg_t *cfg);

#endif
