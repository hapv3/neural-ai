#ifndef NPU_GRAPH_H
#define NPU_GRAPH_H

#include "npu_types.h"

typedef enum {
    NPU_DTYPE_I8 = 1,
    NPU_DTYPE_I32 = 4
} npu_dtype_t;

typedef enum {
    NPU_LAYOUT_HWC = 1,
    NPU_LAYOUT_ROW32 = 2
} npu_layout_t;

typedef enum {
    NPU_OP_DMA_IN = 1,
    NPU_OP_DMA_OUT = 2,
    NPU_OP_IM2COL3X3S1P1_C3_PAD32 = 3,
    NPU_OP_SYSTOLIC_GEMM32 = 4,
    NPU_OP_SPATZ_REQUANT = 5
} npu_op_type_t;

typedef struct {
    uint32_t addr;
    uint16_t h;
    uint16_t w;
    uint16_t c;
    uint16_t reserved;
    uint32_t bytes;
    npu_dtype_t dtype;
    npu_layout_t layout;
} npu_tensor_t;

typedef struct {
    npu_op_type_t op;
    uint32_t src;
    uint32_t dst;
    uint32_t aux;
    uint32_t l2_addr;
    uint32_t bytes;
    uint32_t dim_m;
    int32_t multiplier;
    uint32_t shift;
    int32_t min_val;
    int32_t max_val;
} npu_layer_t;

typedef struct {
    const npu_tensor_t *tensors;
    uint32_t num_tensors;
    const npu_layer_t *layers;
    uint32_t num_layers;
} npu_graph_t;

enum {
    NPU_GRAPH_OK = 0,
    NPU_GRAPH_ERR_BAD_OP = 0xBAD10001,
    NPU_GRAPH_ERR_BAD_TENSOR = 0xBAD10002
};

uint32_t npu_graph_run(const npu_graph_t *graph);
void npu_im2col3x3s1p1_c3_pad32(const int8_t *input_hwc, int8_t *output_row32);
void npu_graph_trace(uint32_t layer_index, npu_op_type_t op, uint32_t event);

#endif
