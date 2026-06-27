#ifndef CONV2D_PACKED_H
#define CONV2D_PACKED_H

#include "conv2d_feeder_sw.h"
#include "npu_types.h"

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
    uint32_t prepare_spatz_tiles;
    uint32_t prepare_scalar_tiles;
} npu_conv2d_packed_stats_t;

enum {
    NPU_CONV2D_PACKED_OK = 0,
    NPU_CONV2D_PACKED_ERR_DILATION = 0xBAD20001u,
    NPU_CONV2D_PACKED_ERR_BAD_SHAPE = 0xBAD20002u
};

uint32_t npu_conv2d_packed_run_oc32(const conv2d_feeder_sw_cfg_t *cfg,
                                    npu_conv2d_packed_stats_t *stats);
uint32_t npu_conv2d_packed_run_oc32_requant(const conv2d_feeder_sw_cfg_t *cfg,
                                            npu_conv2d_packed_stats_t *stats);

#endif
