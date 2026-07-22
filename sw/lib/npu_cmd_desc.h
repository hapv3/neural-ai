#ifndef NPU_CMD_DESC_H
#define NPU_CMD_DESC_H

#include "npu_types.h"

#define NPU_CMD_TABLE_MAGIC 0x4E505543u
#define NPU_CMD_VERSION     1u
#define NPU_CMD_HEADER_SIZE 16u
#define NPU_CMD_ALIGN_BYTES 32u

#define NPU_CMD_FLAG_WAIT        0x00000001u
#define NPU_CMD_FLAG_ACCUM       0x00000002u
#define NPU_CMD_FLAG_REQUANT     0x00000004u

#define NPU_CMD_FAIL_NONE          0x00000000u
#define NPU_CMD_FAIL_BAD_SIZE      0xBADCD001u
#define NPU_CMD_FAIL_BAD_ALIGN     0xBADCD002u
#define NPU_CMD_FAIL_COPY          0xBADCD003u
#define NPU_CMD_FAIL_BAD_TABLE     0xBADCD004u
#define NPU_CMD_FAIL_BAD_COMMAND   0xBADCD005u
#define NPU_CMD_FAIL_UNSUPPORTED   0xBADCD006u
#define NPU_CMD_FAIL_DMA_TIMEOUT   0xBADCD007u
#define NPU_CMD_FAIL_DMA_START     0xBADCD008u
#define NPU_CMD_FAIL_ROLLING       0xBADCD009u
#define NPU_CMD_FAIL_BAD_INVOCATION 0xBADCD00Au
#define NPU_CMD_FAIL_BAD_MODEL      0xBADCD00Bu
#define NPU_CMD_FAIL_BAD_BINDING    0xBADCD00Cu
#define NPU_CMD_FAIL_V2_DISPATCH    0xBADCD00Du

typedef enum {
    NPU_CMD_TYPE_END = 0,
    NPU_CMD_TYPE_IDMA_1D = 1,
    NPU_CMD_TYPE_IDMA_2D = 2,
    NPU_CMD_TYPE_IDMA_3D = 3,
    NPU_CMD_TYPE_SYSTOLIC_GEMM32 = 4,
    NPU_CMD_TYPE_BARRIER = 5,
    NPU_CMD_TYPE_ROLLING_BUFFER = 6
} npu_cmd_type_t;

typedef enum {
    NPU_CMD_ROLLING_RESET = 0,
    NPU_CMD_ROLLING_PRODUCE = 1,
    NPU_CMD_ROLLING_CONSUME_RELEASE = 2
} npu_cmd_rolling_op_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;
    uint32_t entry_offset;
    uint32_t cmd_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} npu_cmd_table_header_t;

typedef struct {
    uint16_t type;
    uint16_t size_bytes;
    uint32_t flags;
    uint32_t layer_id;
    uint32_t tile_id;
} npu_cmd_header_t;

typedef struct {
    npu_cmd_header_t header;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;
    uint32_t direction;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
} npu_cmd_idma_1d_t;

typedef struct {
    npu_cmd_header_t header;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;
    uint32_t src_stride_2;
    uint32_t dst_stride_2;
    uint32_t reps_2;
    uint32_t direction;
    uint32_t reserved0;
    uint32_t reserved1;
} npu_cmd_idma_2d_t;

typedef struct {
    npu_cmd_header_t header;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;
    uint32_t src_stride_2;
    uint32_t dst_stride_2;
    uint32_t reps_2;
    uint32_t src_stride_3;
    uint32_t dst_stride_3;
    uint32_t reps_3;
    uint32_t direction;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} npu_cmd_idma_3d_t;

typedef struct {
    npu_cmd_header_t header;
    uint32_t weight_addr;
    uint32_t ifm_addr;
    uint32_t psum_addr;
    uint32_t ofm_addr;
    uint32_t dim_m;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} npu_cmd_systolic_gemm32_t;

typedef struct {
    npu_cmd_header_t header;
    uint32_t op;
    uint32_t buffer_id;
    uint32_t base_addr;
    uint32_t slot_bytes;
    uint32_t slot_count;
    uint32_t expected_slot;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
} npu_cmd_rolling_buffer_t;

uint32_t npu_cmd_dispatch_from_ctrl(void);
uint32_t npu_cmd_dispatch(uint32_t cmd_tcdm_base, uint32_t cmd_total_bytes);

#endif
