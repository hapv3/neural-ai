#ifndef HAL_CONV2D_FEEDER_H
#define HAL_CONV2D_FEEDER_H

#include "npu_types.h"

typedef struct {
    uint32_t input_addr;
    uint32_t output_addr;
    uint32_t rows;
    uint32_t output_w;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
} conv2d_feeder_hw_cfg_t;

void conv2d_feeder_hw_start_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base, uint32_t stream_en);
void conv2d_feeder_hw_wait(void);
void conv2d_feeder_hw_materialize_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base);
void conv2d_feeder_hw_start_stream_k_tile(const conv2d_feeder_hw_cfg_t *cfg, uint32_t k_base);

#endif
