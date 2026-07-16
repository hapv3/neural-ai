#ifndef NPU_GRAPH_H
#define NPU_GRAPH_H

#include "npu_tensor.h"
#include "npu_types.h"

typedef struct npu_conv2d_linebuf_job_desc npu_conv2d_linebuf_job_desc_t;
typedef struct npu_conv2d_l2_copy_job_desc npu_conv2d_l2_copy_job_desc_t;

typedef enum {
    DMA_IN = 1,
    DMA_OUT = 2,
    LEGACY_IM2COL_C3_PAD32 = 3,
    SYSTOLIC_GEMM32 = 4,
    SPATZ_REQUANT = 5,
    SYSTOLIC_GEMM32_REQUANT = 6,
    LEGACY_IM2COL_C3_DOWNSAMPLE_PAD32 = 7,
    CONV2D_RGB_LINEBUF_REQUANT = 8,
    LOGISTIC_LUT_I8 = 9,
    MUL_I8 = 10,
    CONV2D_C32_LINEBUF = 11,
    CONV2D_C32_LINEBUF_REQUANT = 12,
    ADD_I8 = 13,
    CONV2D_C32_DOWNSAMPLE_LINEBUF_REQUANT = 14,
    MAXPOOL2D_I8 = 15,
    UPSAMPLE_NEAREST_I8 = 16,
    CONV2D_DUAL_SOURCE_C32_LINEBUF_REQUANT_L2 = 17,
    DFL_SOFTMAX_I8_Q8 = 18,
    CLASS_SIGMOID_ROW32_HIGH16_I8 = 19,
    CONV2D_POINTWISE_C32_REQUANT = 20,
    DEPTHWISE_CONV2D_C32_REQUANT = 21,
    DEPTHWISE_CONV2D_C32_DOWNSAMPLE_REQUANT = 22,
    GLOBAL_AVGPOOL_C32_REDUCE = 23,
    CLAMP_I8 = 24,
    CONV2D_C32_MULTI_LINEBUF_REQUANT = 25
} npu_op_type_t;

typedef struct {
    npu_op_type_t op;
    uint32_t src;
    uint32_t src2;
    uint32_t dst;
    uint32_t aux;
    uint32_t aux2;
    uint32_t l2_addr;
    uint32_t bytes;
    uint32_t dim_m;
    uint32_t dim_n;
    int32_t multiplier;
    uint32_t shift;
    int32_t min_val;
    int32_t max_val;
    const npu_conv2d_linebuf_job_desc_t *linebuf_jobs;
    uint32_t linebuf_job_count;
    const npu_conv2d_l2_copy_job_desc_t *linebuf_l2_jobs;
    uint32_t linebuf_l2_job_count;
} npu_layer_t;

typedef struct {
    const npu_tensor_t *tensors;
    uint32_t num_tensors;
    const npu_layer_t *layers;
    uint32_t num_layers;
} npu_graph_t;

typedef struct {
    uint32_t base;
    uint32_t size;
    uint32_t offset;
} npu_graph_scratch_t;

enum {
    NPU_GRAPH_OK = 0,
    NPU_GRAPH_ERR_BAD_OP = 0xBAD10001,
    NPU_GRAPH_ERR_BAD_TENSOR = 0xBAD10002,
    NPU_GRAPH_ERR_DMA = 0xBAD10003,
    NPU_GRAPH_ERR_SCRATCH = 0xBAD10004,
    NPU_GRAPH_ERR_ACCEL = 0xBAD10005
};

void npu_graph_scratch_init(npu_graph_scratch_t *scratch, uint32_t base, uint32_t size);
uint32_t npu_graph_scratch_alloc(npu_graph_scratch_t *scratch, uint32_t bytes, uint32_t align);
uint32_t npu_graph_run(const npu_graph_t *graph);
void npu_im2col3x3s1p1_c3_pad32(const int8_t *input_hwc, int8_t *output_row32);
void npu_im2col3x3s2p1_c3_pad32(const int8_t *input_hwc, uint32_t input_h, uint32_t input_w,
                                int8_t *output_row32);
void npu_graph_trace(uint32_t layer_index, npu_op_type_t op, uint32_t event);

#endif
