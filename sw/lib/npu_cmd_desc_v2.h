#ifndef NPU_CMD_DESC_V2_H
#define NPU_CMD_DESC_V2_H

#include "npu_model_loader.h"
#include "hal_systolic.h"

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
    uint32_t reserved[4];
} nai_cmd_control_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    uint32_t qparam_index;
    uint32_t qparam_count;
    uint32_t qparam_block;
    uint32_t reserved;
} nai_cmd_rq_load_v2_t;

typedef enum {
    NAI_DMA_EXTERNAL_TO_LOCAL = 0,
    NAI_DMA_LOCAL_TO_EXTERNAL = 1,
    NAI_DMA_LOCAL_TO_LOCAL = 2
} nai_dma_direction_v2_t;

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

typedef struct {
    systolic_linebuf_cfg_t linebuf;
    systolic_gemm32_req_t gemm;
    uint32_t rows;
    uint32_t k_tiles;
} nai_linebuf_job_wire_v1_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_linebuf_job_wire_v1_t job;
    uint8_t reserved[20];
} nai_cmd_linebuf_job_v2_t;

/* Pointwise 1x1 C32 command. Activations are group-major C32 blocked;
   weights are OCG-major, then ICG-major, with one 32x32 tile per pair. */
typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t weights;
    nai_ref_v1_t ifm;
    nai_ref_v1_t partial_sums;
    nai_ref_v1_t ofm;
    uint32_t rows;
    uint32_t input_c32_groups;
    uint32_t output_c32_groups;
    uint32_t qparam_block;
    uint32_t input_group_stride_bytes;
    uint32_t output_group_stride_bytes;
    uint32_t reserved[6];
} nai_cmd_pointwise_c32_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t weights;
    nai_ref_v1_t ifm;
    nai_ref_v1_t ofm;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t channels;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t qparam_block;
    uint32_t reserved[4];
} nai_cmd_depthwise_c32_v2_t;

typedef enum {
    NAI_AFU_BINARY_ADD_I8 = 1
} nai_afu_binary_mode_v2_t;

typedef struct {
    nai_cmd_header_v2_t header;
    nai_ref_v1_t lhs;
    nai_ref_v1_t rhs;
    nai_ref_v1_t ofm;
    uint32_t length;
    uint32_t mode;
    uint32_t reserved[4];
} nai_cmd_afu_binary_v2_t;

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
_Static_assert(sizeof(nai_cmd_control_v2_t) == 32, "nai_cmd_control_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_rq_load_v2_t) == 32, "nai_cmd_rq_load_v2_t ABI size");
_Static_assert(offsetof(nai_cmd_rq_load_v2_t, qparam_index) == 16, "RQ load index offset");
_Static_assert(offsetof(nai_cmd_rq_load_v2_t, qparam_block) == 24, "RQ load block offset");
_Static_assert(sizeof(nai_cmd_dma_1d_v2_t) == 64, "nai_cmd_dma_1d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_dma_2d_v2_t) == 64, "nai_cmd_dma_2d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_dma_3d_v2_t) == 64, "nai_cmd_dma_3d_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_gemm32_v2_t) == 96, "nai_cmd_gemm32_v2_t ABI size");
_Static_assert(sizeof(nai_linebuf_job_wire_v1_t) == 124, "nai_linebuf_job_wire_v1_t ABI size");
_Static_assert(sizeof(nai_cmd_linebuf_job_v2_t) == 160, "nai_cmd_linebuf_job_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_pointwise_c32_v2_t) == 96, "nai_cmd_pointwise_c32_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_depthwise_c32_v2_t) == 96, "nai_cmd_depthwise_c32_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_afu_binary_v2_t) == 64, "nai_cmd_afu_binary_v2_t ABI size");
_Static_assert(sizeof(nai_cmd_copy_layout_v2_t) == 96, "nai_cmd_copy_layout_v2_t ABI size");
_Static_assert(offsetof(nai_cmd_gemm32_v2_t, weights) == 16, "GEMM reference offset");
_Static_assert(offsetof(nai_cmd_gemm32_v2_t, dim_m) == 48, "GEMM dimension offset");

typedef enum {
    NAI_DISPATCH_OK = 0,
    NAI_DISPATCH_BAD_STREAM = 1,
    NAI_DISPATCH_BAD_COMMAND = 2,
    NAI_DISPATCH_UNSUPPORTED = 3,
    NAI_DISPATCH_BAD_REFERENCE = 4,
    NAI_DISPATCH_OPERATION_FAILED = 5
} nai_dispatch_status_v2_t;

typedef struct {
    void *context;
    uint32_t (*dma_1d)(void *context, uint32_t source, uint32_t destination,
                       uint32_t length, uint32_t direction);
    uint32_t (*dma_2d)(void *context, uint32_t source, uint32_t destination,
                       uint32_t length, uint32_t source_stride, uint32_t destination_stride,
                       uint32_t repetitions, uint32_t direction);
    uint32_t (*dma_3d)(void *context, uint32_t source, uint32_t destination,
                       uint32_t length, uint32_t source_stride_2, uint32_t destination_stride_2,
                       uint32_t repetitions_2, uint32_t source_stride_3,
                       uint32_t destination_stride_3, uint32_t repetitions_3,
                       uint32_t direction);
    uint32_t (*gemm32)(void *context, const nai_cmd_gemm32_v2_t *command,
                       uint32_t weights, uint32_t ifm, uint32_t partial_sums, uint32_t ofm);
    uint32_t (*pointwise_c32)(void *context, const nai_cmd_pointwise_c32_v2_t *command,
                              uint32_t weights, uint32_t ifm, uint32_t partial_sums, uint32_t ofm);
    uint32_t (*depthwise_c32)(void *context, const nai_cmd_depthwise_c32_v2_t *command,
                              uint32_t weights, uint32_t ifm, uint32_t ofm);
    uint32_t (*afu_binary)(void *context, const nai_cmd_afu_binary_v2_t *command,
                           uint32_t lhs, uint32_t rhs, uint32_t ofm);
    uint32_t (*linebuf_job)(void *context, const nai_cmd_linebuf_job_v2_t *command);
    uint32_t (*copy_layout)(void *context, const nai_cmd_copy_layout_v2_t *command,
                           uint32_t source, uint32_t destination);
    uint32_t (*barrier)(void *context);
    uint32_t (*rq_load)(void *context, uint32_t qparam_address,
                        uint32_t qparam_count, uint32_t qparam_block);
} nai_runtime_ops_v2_t;

nai_dispatch_status_v2_t nai_cmd_dispatch_v2(const nai_model_view_v1_t *view,
                                             const nai_resolver_v1_t *resolver,
                                             const nai_runtime_ops_v2_t *ops,
                                             uint32_t *completed_commands,
                                             uint32_t *failure_command_offset);
nai_dispatch_status_v2_t nai_cmd_dispatch_stream_v2(const nai_model_view_v1_t *view,
                                                    const nai_resolver_v1_t *resolver,
                                                    const nai_runtime_ops_v2_t *ops,
                                                    const nai_model_reader_v1_t *reader,
                                                    void *command_buffer,
                                                    uint32_t command_buffer_bytes,
                                                    uint32_t *completed_commands,
                                                    uint32_t *failure_command_offset);

#endif
