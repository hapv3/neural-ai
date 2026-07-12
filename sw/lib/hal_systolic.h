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
    uint32_t row_stride_bytes;
    uint32_t pixel_stride_bytes;
    uint32_t ow_step_bytes;
    uint32_t oh_step_bytes;
    uint16_t kernel_h;
    uint16_t kernel_w;
    uint16_t c_base;
    uint16_t lane_base;
    uint16_t coalesce;
    uint16_t kgen;
    uint16_t k_seed_kh;
    uint16_t k_seed_kw;
    uint16_t k_seed_ic;
    uint32_t k_tiles;
    uint32_t spatial_m;
} systolic_linebuf_cfg_t;

void systolic_linebuf_disable(void);
void systolic_linebuf_config(const systolic_linebuf_cfg_t *cfg);
void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_gemm32_linebuf(uint32_t weight_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_gemm32_linebuf_requant(uint32_t weight_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_gemm32_linebuf_ktiles(uint32_t weight_addr,
                                    uint32_t psum_addr,
                                    uint32_t ofm_addr,
                                    uint32_t dim_m);
void systolic_gemm32_linebuf_ktiles_strided(uint32_t weight_addr,
                                            uint32_t psum_addr,
                                            uint32_t ofm_addr,
                                            uint32_t dim_m,
                                            uint32_t ofm_row_stride_bytes,
                                            uint32_t ofm_tile_cols);
void systolic_gemm32_linebuf_ktiles_requant(uint32_t weight_addr,
                                            uint32_t psum_addr,
                                            uint32_t ofm_addr,
                                            uint32_t dim_m);
void systolic_gemm32_linebuf_ktiles_requant_strided(uint32_t weight_addr,
                                                    uint32_t psum_addr,
                                                    uint32_t ofm_addr,
                                                    uint32_t dim_m,
                                                    uint32_t ofm_row_stride_bytes,
                                                    uint32_t ofm_tile_cols);
void systolic_gemm32_linebuf_ktiles_accumulate(uint32_t weight_addr,
                                               uint32_t psum_addr,
                                               uint32_t ofm_addr,
                                               uint32_t dim_m);
void systolic_gemm32_linebuf_ktiles_accumulate_strided(uint32_t weight_addr,
                                                       uint32_t psum_addr,
                                                       uint32_t ofm_addr,
                                                       uint32_t dim_m,
                                                       uint32_t ofm_row_stride_bytes,
                                                       uint32_t ofm_tile_cols,
                                                       uint32_t psum_row_stride_bytes);
void systolic_gemm32_linebuf_accumulate_requant(uint32_t weight_addr,
                                                uint32_t psum_addr,
                                                uint32_t ofm_addr,
                                                uint32_t dim_m);
void systolic_gemm32_linebuf_accumulate(uint32_t weight_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m);
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
