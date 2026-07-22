#ifndef NPU_CMD_DESC_V2_H
#define NPU_CMD_DESC_V2_H

#include "npu_model_abi.h"

#define NAI_CMD_FLAG_OPTIONAL  (1u << 0)
#define NAI_CMD_FLAG_SKIPPABLE (1u << 1)

typedef enum {
    NAI_CMD_END = 0,
    NAI_CMD_BARRIER = 1,
    NAI_CMD_DMA_1D = 2,
    NAI_CMD_DMA_2D = 3,
    NAI_CMD_DMA_3D = 4,
    NAI_CMD_RQ_LOAD = 5,
    NAI_CMD_GEMM32 = 6,
    NAI_CMD_GEMM32_ACCUM = 7,
    NAI_CMD_GEMM32_REQUANT = 8,
    NAI_CMD_LINEBUF_JOB = 9,
    NAI_CMD_POINTWISE_C32 = 10,
    NAI_CMD_DEPTHWISE_C32 = 11,
    NAI_CMD_AFU_LUT = 12,
    NAI_CMD_AFU_BINARY = 13,
    NAI_CMD_AFU_GLOBAL_AVGPOOL = 14,
    NAI_CMD_SPATZ_REQUANT = 15,
    NAI_CMD_SPATZ_ADD = 16,
    NAI_CMD_SPATZ_MUL = 17,
    NAI_CMD_COPY_LAYOUT = 18,
    NAI_CMD_MAXPOOL = 19,
    NAI_CMD_UPSAMPLE_NEAREST = 20,
    NAI_CMD_ROLLING_RESET = 21,
    NAI_CMD_ROLLING_PRODUCE = 22,
    NAI_CMD_ROLLING_CONSUME_RELEASE = 23,
    NAI_CMD_DMA_SUBMIT_1D = 24,
    NAI_CMD_DMA_SUBMIT_2D = 25,
    NAI_CMD_DMA_SUBMIT_3D = 26,
    NAI_CMD_DMA_WAIT = 27
} nai_cmd_type_v2_t;

typedef struct {
    uint16_t type;
    uint16_t size_bytes;
    uint32_t flags;
    uint32_t layer_id;
    uint32_t tile_id;
} nai_cmd_header_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t source;
    nai_ref_v1_t destination;
    uint32_t length;
    uint32_t direction;
    uint32_t reserved[6];
} nai_cmd_dma_1d_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t source;
    nai_ref_v1_t destination;
    uint32_t length;
    uint32_t source_stride_2;
    uint32_t destination_stride_2;
    uint32_t repetitions_2;
    uint32_t direction;
    uint32_t reserved[3];
} nai_cmd_dma_2d_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t source;
    nai_ref_v1_t destination;
    uint32_t length;
    uint32_t source_stride_2;
    uint32_t destination_stride_2;
    uint32_t repetitions_2;
    uint32_t source_stride_3;
    uint32_t destination_stride_3;
    uint32_t repetitions_3;
    uint32_t direction;
} nai_cmd_dma_3d_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t weights;
    nai_ref_v1_t ifm;
    nai_ref_v1_t partial_sums;
    nai_ref_v1_t ofm;
    uint32_t dim_m;
    uint32_t ofm_row_stride;
    uint32_t partial_sum_row_stride;
    uint32_t qparam_block;
    uint32_t reserved[8];
} nai_cmd_gemm32_v2_t;

typedef enum {
    NAI_COPY_NHWC_TO_ROW32 = 1,
    NAI_COPY_ROW32_TO_NHWC = 2,
    NAI_COPY_NHWC_TO_C32 = 3,
    NAI_COPY_C32_TO_NHWC = 4
} nai_copy_layout_mode_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t source;
    nai_ref_v1_t destination;
    uint16_t mode;
    uint16_t source_layout;
    uint16_t destination_layout;
    uint16_t data_type;
    uint32_t dimensions[4];
    uint32_t valid_channels;
    uint32_t source_row_stride;
    uint32_t destination_row_stride;
    uint32_t reserved[7];
} nai_cmd_copy_layout_v2_t;

_Static_assert(sizeof(nai_cmd_header_v2_t) == 16, "nai_cmd_header_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_dma_1d_v2_t) == 64, "nai_cmd_dma_1d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_dma_2d_v2_t) == 64, "nai_cmd_dma_2d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_dma_3d_v2_t) == 64, "nai_cmd_dma_3d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_gemm32_v2_t) == 96, "nai_cmd_gemm32_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_copy_layout_v2_t) == 96, "nai_cmd_copy_layout_v2_t ABI size");
_Static_assert(offsetof(nai_cmd_gemm32_v2_t, weights) == 16, "GEMM reference offset");
_Static_assert(offsetof(nai_cmd_gemm32_v2_t, dim_m) == 48, "GEMM dimension offset");

#endif
