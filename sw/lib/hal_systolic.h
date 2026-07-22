#ifndef HAL_SYSTOLIC_H
#define HAL_SYSTOLIC_H

#include "npu_types.h"

#define SYSTOLIC_GEMM32_K 32u
#define SYSTOLIC_GEMM32_N 32u
#define SYSTOLIC_GEMM32_TILE_M 1024u
#define SYSTOLIC_GEMM32_ACCUM_TILE_M 256u

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
    uint16_t pool;
    uint16_t c32_fast;
    uint16_t depthwise;
    uint16_t c32_group_stationary;
    uint16_t block_valid_bytes;
    uint16_t k_seed_kh;
    uint16_t k_seed_kw;
    uint16_t k_seed_ic;
    uint32_t k_tiles;
    uint32_t spatial_m;
    /*
     * Host-precomputed channel addressing.
     * - Normal/generic descriptors: byte offset from input_base to c_base.
     * - C32 group-stationary descriptors: byte span between consecutive C32
     *   channel groups. The controller keeps the current group offset in a
     *   register and advances it with adds; the linebuffer hot path does not
     *   multiply k_seed_ic by this span.
     */
    uint32_t channel_addr_offset;
    uint32_t coalesce_k_bytes;
} systolic_linebuf_cfg_t;

typedef struct {
    uint32_t weight_addr;
    uint32_t ifm_addr;
    uint32_t psum_addr;
    uint32_t ofm_addr;
    uint32_t dim_m;
    uint32_t accum_en;
    uint32_t ofm_row_stride_bytes;
    uint32_t ofm_tile_cols;
    uint32_t psum_row_stride_bytes;
} systolic_gemm32_req_t;

void systolic_gemm32_preload(const systolic_gemm32_req_t *req);
void systolic_gemm32_start_preloaded(void);
uint32_t systolic_gemm32_done(void);
void systolic_gemm32_wait_done(void);
void systolic_linebuf_disable(void);
void systolic_linebuf_config(const systolic_linebuf_cfg_t *cfg);
void systolic_maxpool5x5s1p2_c32_linebuf(uint32_t input_addr,
                                         uint32_t output_addr,
                                         uint32_t height,
                                         uint32_t width);
void systolic_depthwise3x3s1p1_c32_requant(uint32_t input_addr,
                                           uint32_t weight_addr,
                                           uint32_t output_addr,
                                           uint32_t height,
                                           uint32_t width);
void systolic_depthwise3x3s1p1_c32_requant_channels(uint32_t input_addr,
                                                    uint32_t weight_addr,
                                                    uint32_t output_addr,
                                                    uint32_t height,
                                                    uint32_t width,
                                                    uint32_t channels);
void systolic_depthwise3x3_c32_requant_channels(uint32_t input_addr,
                                                uint32_t weight_addr,
                                                uint32_t output_addr,
                                                uint32_t input_h,
                                                uint32_t input_w,
                                                uint32_t output_h,
                                                uint32_t output_w,
                                                uint32_t channels,
                                                uint32_t stride_h,
                                                uint32_t stride_w,
                                                uint32_t pad_h,
                                                uint32_t pad_w);
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
void systolic_gemm32_linebuf_ktiles_accumulate_requant_strided(uint32_t weight_addr,
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
void systolic_gemm32_strided(uint32_t weight_addr,
                             uint32_t ifm_addr,
                             uint32_t ofm_addr,
                             uint32_t dim_m,
                             uint32_t ofm_row_stride_bytes);
void systolic_gemm32_requant_strided(uint32_t weight_addr,
                                     uint32_t ifm_addr,
                                     uint32_t ofm_addr,
                                     uint32_t dim_m,
                                     uint32_t ofm_row_stride_bytes);
void systolic_gemm32_accumulate_strided(uint32_t weight_addr,
                                        uint32_t ifm_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m,
                                        uint32_t ofm_row_stride_bytes,
                                        uint32_t psum_row_stride_bytes);
void systolic_gemm32_accumulate_requant_strided(uint32_t weight_addr,
                                                uint32_t ifm_addr,
                                                uint32_t psum_addr,
                                                uint32_t ofm_addr,
                                                uint32_t dim_m,
                                                uint32_t ofm_row_stride_bytes,
                                                uint32_t psum_row_stride_bytes);
void systolic_requant_disable(void);
void systolic_requant_config_per_channel(const int32_t *bias,
                                         const int32_t *multiplier,
                                         const uint8_t *shift,
                                         const int32_t *zero_point,
                                         int32_t clamp_min,
                                         int32_t clamp_max);
void systolic_gemm32_requant(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);
void systolic_pointwise1x1_c32_multi_requant(uint32_t input_addr,
                                             uint32_t weight_addr,
                                             uint32_t psum_addr,
                                             uint32_t output_addr,
                                             uint32_t rows,
                                             uint32_t input_c32_groups,
                                             uint32_t output_c32_groups);

#endif
