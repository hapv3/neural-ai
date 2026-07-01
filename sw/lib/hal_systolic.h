#ifndef HAL_SYSTOLIC_H
#define HAL_SYSTOLIC_H

#include "npu_types.h"

#define SYSTOLIC_GEMM32_K 32u
#define SYSTOLIC_GEMM32_N 32u
#define SYSTOLIC_GEMM32_TILE_M 1024u
#define SYSTOLIC_GEMM32_ACCUM_TILE_M 16u

typedef struct {
    uint32_t input_base;
    uint16_t input_h;
    uint16_t input_w;
    uint16_t input_c;
    uint16_t output_w;
    uint16_t stride_h;
    uint16_t stride_w;
    uint16_t pad_h;
    uint16_t pad_w;
    uint16_t tile_oh_base;
    uint16_t tile_ow_base;
    uint32_t lane_valid;
    uint8_t lane_kh[32];
    uint8_t lane_kw[32];
    uint16_t lane_ic[32];
} systolic_linebuf_cfg_t;

void systolic_linebuf_disable(void);
void systolic_linebuf_config(const systolic_linebuf_cfg_t *cfg);
void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_gemm32_linebuf(uint32_t weight_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_gemm32_accumulate(uint32_t weight_addr,
                                uint32_t ifm_addr,
                                uint32_t psum_addr,
                                uint32_t ofm_addr,
                                uint32_t dim_m);
void systolic_gemm32_accumulate_requant(uint32_t weight_addr,
                                        uint32_t ifm_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m);
void systolic_requant_disable(void);
void systolic_requant_config_per_channel(const int32_t *bias,
                                         const int32_t *multiplier,
                                         const uint8_t *shift,
                                         const int32_t *zero_point,
                                         int32_t clamp_min,
                                         int32_t clamp_max);
void systolic_gemm32_requant(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);

#endif
