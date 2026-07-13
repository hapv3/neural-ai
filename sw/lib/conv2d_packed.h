#ifndef CONV2D_PACKED_H
#define CONV2D_PACKED_H

#include "hal_systolic.h"
#include "npu_types.h"

#define NPU_CONV2D_PACKED_OC_TILE 32u
#define NPU_CONV2D_PACKED_K_TILE  32u

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
    uint32_t input_c_stride;
    uint32_t input_row_stride_bytes;
    uint32_t input_c_base;
    uint32_t accumulate;
} npu_conv2d_packed_cfg_t;

typedef struct {
    uint32_t rows;
    uint32_t k_tiles;
    uint32_t prepare_cycles;
    uint32_t gemm_cycles;
    uint32_t total_cycles;
    uint32_t last_prepare_cycles;
    uint32_t last_gemm_cycles;
    uint32_t status;
    uint32_t prepare_idma_tiles;
    uint32_t prepare_idma_transfers;
    uint32_t prepare_spatz_tiles;
    uint32_t prepare_scalar_tiles;
} npu_conv2d_packed_stats_t;

typedef struct {
    uint32_t oh_base;
    uint32_t ow_base;
    uint32_t tile_oh;
    uint32_t tile_ow;
} npu_conv2d_spatial_tile_t;

typedef struct npu_conv2d_linebuf_job_desc {
    systolic_linebuf_cfg_t linebuf;
    systolic_gemm32_req_t gemm;
    uint32_t rows;
    uint32_t k_tiles;
} npu_conv2d_linebuf_job_desc_t;

typedef struct npu_conv2d_l2_copy_job_desc {
    npu_conv2d_linebuf_job_desc_t job;
    uint32_t l2_addr;
    uint32_t tile_output_addr;
    uint32_t tile_oh;
    uint32_t tile_ow;
} npu_conv2d_l2_copy_job_desc_t;

enum {
    NPU_CONV2D_PACKED_OK = 0,
    NPU_CONV2D_PACKED_ERR_DILATION = 0xBAD20001u,
    NPU_CONV2D_PACKED_ERR_BAD_SHAPE = 0xBAD20002u,
    NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES = 0xBAD20003u
};

uint32_t npu_conv2d_packed_run_oc32(const npu_conv2d_packed_cfg_t *cfg,
                                    npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_requant(const npu_conv2d_packed_cfg_t *cfg,
                                            npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_linebuf(const npu_conv2d_packed_cfg_t *cfg,
                                            npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_linebuf_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                    npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_linebuf_default_tile_oh(const npu_conv2d_packed_cfg_t *cfg);
uint32_t npu_conv2d_packed_run_oc32_linebuf_tiles(const npu_conv2d_packed_cfg_t *cfg,
                                                  const npu_conv2d_spatial_tile_t *tiles,
                                                  uint32_t tile_count,
                                                  uint32_t output_elem_bytes,
                                                  npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_linebuf_tiles_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                          const npu_conv2d_spatial_tile_t *tiles,
                                                          uint32_t tile_count,
                                                          uint32_t psum_addr,
                                                          npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                                    const npu_conv2d_spatial_tile_t *tile,
                                                                    uint32_t psum_addr,
                                                                    uint32_t output_addr,
                                                                    npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_linebuf_job_descs(const npu_conv2d_linebuf_job_desc_t *jobs,
                                                 uint32_t job_count,
                                                 npu_conv2d_packed_stats_t *stats);

#endif
