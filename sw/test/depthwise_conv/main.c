#include "npu_graph.h"
#include "spatz_rt.h"

/*
 * Scenario: graph-level depthwise Conv3x3 S1 P1 C32 fast path.
 * Target: prove a C32-blocked 1x32x48x48 tensor uses the RTL linebuffer-fed
 * lane-wise depthwise MAC path, with one C32 tap vector consumed per cycle.
 */

#define L2_INPUT   0x80000000u
#define L2_WEIGHT  0x80020000u
#define L2_OUTPUT  0x80030000u

#define T_INPUT    0x10100000u
#define T_WEIGHT   0x10113000u
#define T_OUTPUT   0x10114000u

#ifndef DEPTHWISE_H
#define DEPTHWISE_H 48u
#endif

#ifndef DEPTHWISE_W
#define DEPTHWISE_W 48u
#endif

#ifndef DEPTHWISE_C
#define DEPTHWISE_C 32u
#endif

#define H DEPTHWISE_H
#define W DEPTHWISE_W
#define C DEPTHWISE_C
#define ROWS (H * W)
#define C32_GROUPS ((C + 31u) / 32u)
#define ACTIVATION_BYTES (ROWS * C32_GROUPS * 32u)
#define WEIGHT_BYTES (3u * 3u * C32_GROUPS * 32u)

enum {
    TENSOR_INPUT = 0,
    TENSOR_WEIGHT,
    TENSOR_OUTPUT,
    TENSOR_COUNT
};

enum {
    L_DMA_IN_INPUT = 0,
    L_DMA_IN_WEIGHT,
    L_DEPTHWISE,
    L_DMA_OUT,
    LAYER_COUNT
};

static npu_tensor_t tensors[TENSOR_COUNT];
static npu_layer_t layers[LAYER_COUNT];

void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *ptr = (uint8_t *)dst;
    for (uint32_t i = 0; i < bytes; i++) {
        ptr[i] = (uint8_t)value;
    }
    return dst;
}

static void init_tensor(npu_tensor_t *tensor,
                        uint32_t addr,
                        uint16_t h,
                        uint16_t w,
                        uint16_t c,
                        uint32_t bytes,
                        npu_dtype_t dtype,
                        npu_layout_t layout) {
    tensor->addr = addr;
    tensor->h = h;
    tensor->w = w;
    tensor->c = c;
    tensor->reserved = 0u;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    tensor->layout = layout;
    tensor->scale_q31 = 0;
    tensor->zero_point = 0;
}

static void init_graph(void) {
    init_tensor(&tensors[TENSOR_INPUT], T_INPUT, H, W, C, ACTIVATION_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[TENSOR_WEIGHT], T_WEIGHT, 3u, 3u, C, WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[TENSOR_OUTPUT], T_OUTPUT, H, W, C, ACTIVATION_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);

    layers[L_DMA_IN_INPUT].op = NPU_OP_DMA_IN;
    layers[L_DMA_IN_INPUT].dst = TENSOR_INPUT;
    layers[L_DMA_IN_INPUT].l2_addr = L2_INPUT;
    layers[L_DMA_IN_INPUT].bytes = ACTIVATION_BYTES;

    layers[L_DMA_IN_WEIGHT].op = NPU_OP_DMA_IN;
    layers[L_DMA_IN_WEIGHT].dst = TENSOR_WEIGHT;
    layers[L_DMA_IN_WEIGHT].l2_addr = L2_WEIGHT;
    layers[L_DMA_IN_WEIGHT].bytes = WEIGHT_BYTES;

    layers[L_DEPTHWISE].op = NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT;
    layers[L_DEPTHWISE].src = TENSOR_INPUT;
    layers[L_DEPTHWISE].dst = TENSOR_OUTPUT;
    layers[L_DEPTHWISE].aux = TENSOR_WEIGHT;
    layers[L_DEPTHWISE].multiplier = 1;
    layers[L_DEPTHWISE].shift = 0u;
    layers[L_DEPTHWISE].min_val = -128;
    layers[L_DEPTHWISE].max_val = 127;

    layers[L_DMA_OUT].op = NPU_OP_DMA_OUT;
    layers[L_DMA_OUT].src = TENSOR_OUTPUT;
    layers[L_DMA_OUT].l2_addr = L2_OUTPUT;
    layers[L_DMA_OUT].bytes = ACTIVATION_BYTES;
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1u, 0u);
    init_graph();

    npu_graph_t graph;
    graph.tensors = tensors;
    graph.num_tensors = TENSOR_COUNT;
    graph.layers = layers;
    graph.num_layers = LAYER_COUNT;

    uint32_t status = npu_graph_run(&graph);
    if (status != NPU_GRAPH_OK) {
        spatz_rt_fail_at(1u, 0u, (int32_t)status, (int32_t)NPU_GRAPH_OK);
    }

    spatz_rt_pass();
    return 0;
}
