#include "npu_graph.h"
#include "npu_memory_map.h"

/*
 * Phase 3a minimal graph:
 *   32x32x3 HWC -> im2col 3x3s1p1 pad32 -> GEMM32 -> clamp/requant
 *                -> GEMM32 1x1 -> clamp/requant -> L2 output.
 */
#define PASS_SIGNATURE 0xDEADBEEFu
#define FAIL_SIGNATURE 0xBAD30000u

#define SIG_STATUS     (*(volatile uint32_t *)0x10008000u)
#define SIG_FAIL_CODE  (*(volatile uint32_t *)0x10008004u)
#define SIG_LAYER      (*(volatile uint32_t *)0x10008008u)
#define SIG_OP         (*(volatile uint32_t *)0x1000800Cu)
#define SIG_EVENT      (*(volatile uint32_t *)0x10008010u)

#define L2_INPUT   0x80000000u
#define L2_WEIGHT0 0x80002000u
#define L2_WEIGHT1 0x80002400u
#define L2_OUTPUT  0x80010000u

#define SCRATCH_BASE 0x10100000u
#define SCRATCH_SIZE (512u * 1024u)

#define INPUT_H 32u
#define INPUT_W 32u
#define INPUT_C 3u
#define ROWS    (INPUT_H * INPUT_W)
#define WEIGHT_BYTES (32u * 32u)
#define ACT_BYTES    (ROWS * 32u)
#define PSUM_BYTES   (ROWS * 32u * 4u)

enum {
    T_INPUT = 0,
    T_WEIGHT0,
    T_WEIGHT1,
    T_IM2COL,
    T_PSUM0,
    T_ACT0,
    T_PSUM1,
    T_OUT,
    TENSOR_COUNT
};

static npu_tensor_t tensors[TENSOR_COUNT];
static npu_layer_t layers[9];
static npu_graph_t graph;

void npu_graph_trace(uint32_t layer_index, npu_op_type_t op, uint32_t event) {
    SIG_LAYER = layer_index;
    SIG_OP = (uint32_t)op;
    SIG_EVENT = event;
}

static void fail(uint32_t code) {
    SIG_FAIL_CODE = code;
    SIG_STATUS = FAIL_SIGNATURE | (code & 0xffffu);
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, SIG_STATUS);
    while (1) {
    }
}

static void init_tensor(npu_tensor_t *tensor, uint32_t addr,
                        uint16_t h, uint16_t w, uint16_t c,
                        uint32_t bytes, npu_dtype_t dtype, npu_layout_t layout,
                        int32_t scale_q31, int32_t zero_point) {
    tensor->addr = addr;
    tensor->h = h;
    tensor->w = w;
    tensor->c = c;
    tensor->reserved = 0;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    tensor->layout = layout;
    tensor->scale_q31 = scale_q31;
    tensor->zero_point = zero_point;
}

static void clear_layer(npu_layer_t *layer) {
    layer->op = 0;
    layer->src = 0;
    layer->dst = 0;
    layer->aux = 0;
    layer->l2_addr = 0;
    layer->bytes = 0;
    layer->dim_m = 0;
    layer->multiplier = 0;
    layer->shift = 0;
    layer->min_val = 0;
    layer->max_val = 0;
}

static void init_layers(void) {
    for (uint32_t i = 0; i < (sizeof(layers) / sizeof(layers[0])); i++) {
        clear_layer(&layers[i]);
    }

    layers[0].op = NPU_OP_DMA_IN;
    layers[0].dst = T_INPUT;
    layers[0].l2_addr = L2_INPUT;
    layers[0].bytes = INPUT_H * INPUT_W * INPUT_C;

    layers[1].op = NPU_OP_DMA_IN;
    layers[1].dst = T_WEIGHT0;
    layers[1].l2_addr = L2_WEIGHT0;
    layers[1].bytes = WEIGHT_BYTES;

    layers[2].op = NPU_OP_DMA_IN;
    layers[2].dst = T_WEIGHT1;
    layers[2].l2_addr = L2_WEIGHT1;
    layers[2].bytes = WEIGHT_BYTES;

    layers[3].op = NPU_OP_IM2COL3X3S1P1_C3_PAD32;
    layers[3].src = T_INPUT;
    layers[3].dst = T_IM2COL;

    layers[4].op = NPU_OP_SYSTOLIC_GEMM32;
    layers[4].src = T_IM2COL;
    layers[4].dst = T_PSUM0;
    layers[4].aux = T_WEIGHT0;
    layers[4].dim_m = ROWS;

    layers[5].op = NPU_OP_SPATZ_REQUANT;
    layers[5].src = T_PSUM0;
    layers[5].dst = T_ACT0;
    layers[5].multiplier = 1;
    layers[5].shift = 0;
    layers[5].min_val = 0;
    layers[5].max_val = 127;

    layers[6].op = NPU_OP_SYSTOLIC_GEMM32;
    layers[6].src = T_ACT0;
    layers[6].dst = T_PSUM1;
    layers[6].aux = T_WEIGHT1;
    layers[6].dim_m = ROWS;

    layers[7].op = NPU_OP_SPATZ_REQUANT;
    layers[7].src = T_PSUM1;
    layers[7].dst = T_OUT;
    layers[7].multiplier = 1;
    layers[7].shift = 0;
    layers[7].min_val = -128;
    layers[7].max_val = 127;

    layers[8].op = NPU_OP_DMA_OUT;
    layers[8].src = T_OUT;
    layers[8].l2_addr = L2_OUTPUT;
    layers[8].bytes = ACT_BYTES;

    graph.tensors = tensors;
    graph.num_tensors = TENSOR_COUNT;
    graph.layers = layers;
    graph.num_layers = sizeof(layers) / sizeof(layers[0]);
}

static uint32_t alloc_or_fail(npu_graph_scratch_t *scratch, uint32_t bytes) {
    uint32_t addr = npu_graph_scratch_alloc(scratch, bytes, 32u);
    if (addr == 0u) {
        fail(NPU_GRAPH_ERR_SCRATCH);
    }
    return addr;
}

int main(void) {
    SIG_STATUS = 0x30000001u;
    SIG_FAIL_CODE = 0;
    SIG_LAYER = 0;
    SIG_OP = 0;
    SIG_EVENT = 0;

    npu_graph_scratch_t scratch;
    npu_graph_scratch_init(&scratch, SCRATCH_BASE, SCRATCH_SIZE);
    SIG_STATUS = 0x30000002u;

    init_tensor(&tensors[T_INPUT], alloc_or_fail(&scratch, INPUT_H * INPUT_W * INPUT_C),
                INPUT_H, INPUT_W, INPUT_C, INPUT_H * INPUT_W * INPUT_C,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
    init_tensor(&tensors[T_WEIGHT0], alloc_or_fail(&scratch, WEIGHT_BYTES),
                32, 32, 32, WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_WEIGHT1], alloc_or_fail(&scratch, WEIGHT_BYTES),
                32, 32, 32, WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_IM2COL], alloc_or_fail(&scratch, ACT_BYTES),
                INPUT_H, INPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_PSUM0], alloc_or_fail(&scratch, PSUM_BYTES),
                INPUT_H, INPUT_W, 32, PSUM_BYTES,
                NPU_DTYPE_I32, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_ACT0], alloc_or_fail(&scratch, ACT_BYTES),
                INPUT_H, INPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_PSUM1], alloc_or_fail(&scratch, PSUM_BYTES),
                INPUT_H, INPUT_W, 32, PSUM_BYTES,
                NPU_DTYPE_I32, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_OUT], alloc_or_fail(&scratch, ACT_BYTES),
                INPUT_H, INPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    SIG_STATUS = 0x30000003u;

    init_layers();
    SIG_STATUS = 0x30000004u;

    uint32_t status = npu_graph_run(&graph);
    if (status != NPU_GRAPH_OK) {
        fail(status);
    }

    SIG_STATUS = PASS_SIGNATURE;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, PASS_SIGNATURE);
    while (1) {
    }
}
