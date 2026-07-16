#include "npu_graph.h"
#include "spatz_rt.h"

#define L2_INPUT           0x80000000u
#define L2_W_STEM          0x80010000u
#define L2_W_DW0           0x80011000u
#define L2_W_PW0           0x80012000u
#define L2_W_DW1           0x80013000u
#define L2_W_PW1           0x80014000u
#define L2_W_PW2           0x80015000u
#define L2_W_DW2           0x80018000u
#define L2_W_PW3           0x80019000u
#define L2_W_VALIDATE      0x8001C000u
#define L2_W_CLASSIFIER    0x80026000u
#define L2_OUTPUT          0x80030000u

#define SCRATCH_BASE       0x10100000u

#define TRACE_LAYER        (*(volatile uint32_t *)0x10008020u)
#define TRACE_OP           (*(volatile uint32_t *)0x10008024u)
#define TRACE_EVENT        (*(volatile uint32_t *)0x10008028u)

#define INPUT_H 96u
#define INPUT_W 96u
#define STEM_H  48u
#define STEM_W  48u
#define MID_H   24u
#define MID_W   24u

#define ACT48_C32_BYTES    (STEM_H * STEM_W * 32u)
#define ACT24_C64_BYTES    (MID_H * MID_W * 64u)
#define ACT24_C128_BYTES   (MID_H * MID_W * 128u)
#define PSUM24_BYTES       (MID_H * MID_W * 32u * 4u)
#define INPUT_BYTES        (INPUT_H * INPUT_W * 3u)
#define WEIGHT_MAX_BYTES   (2u * 2u * 3u * 3u * 32u * 32u)

#define T_INPUT_ADDR       (SCRATCH_BASE)
#define T_WEIGHT_ADDR      (T_INPUT_ADDR + INPUT_BYTES)
#define T_ACT_A_ADDR       (T_WEIGHT_ADDR + WEIGHT_MAX_BYTES)
#define T_ACT_B_ADDR       (T_ACT_A_ADDR + ACT48_C32_BYTES)
#define T_ACT_C_ADDR       (T_ACT_B_ADDR + ACT48_C32_BYTES)
#define T_ACT_D_ADDR       (T_ACT_C_ADDR + ACT48_C32_BYTES)
#define T_PSUM_ADDR        (T_ACT_D_ADDR + ACT24_C128_BYTES)

#define STEM_WEIGHT_BYTES       (32u * 32u)
#define DW32_WEIGHT_BYTES       (3u * 3u * 32u)
#define DW128_WEIGHT_BYTES      (4u * 3u * 3u * 32u)
#define PW32_32_WEIGHT_BYTES    (1u * 1u * 32u * 32u)
#define PW32_64_WEIGHT_BYTES    (2u * 1u * 32u * 32u)
#define PW64_128_WEIGHT_BYTES   (4u * 2u * 32u * 32u)
#define PW128_64_WEIGHT_BYTES   (2u * 4u * 32u * 32u)
#define PW64_32_WEIGHT_BYTES    (1u * 2u * 32u * 32u)
#define CONV64_64_WEIGHT_BYTES  (2u * 2u * 3u * 3u * 32u * 32u)

enum {
    T_INPUT = 0,
    T_WEIGHT,
    T_A_48_C32_ROW32,
    T_B_48_C32_ROW32,
    T_C_48_C32_ROW32,
    T_A_48_C32,
    T_B_48_C32,
    T_C_48_C32,
    T_A_24_C32,
    T_B_24_C32,
    T_A_24_C64,
    T_B_24_C64,
    T_C_24_C64,
    T_C_24_C128,
    T_D_24_C128,
    T_A_1_C64,
    T_B_1_C32,
    T_PSUM,
    TENSOR_COUNT
};

enum {
    L_DMA_IN_INPUT = 0,
    L_DMA_W_STEM,
    L_STEM_CONV,
    L_STEM_CLAMP,
    L_DMA_W_DW0,
    L_DW0,
    L_DW0_CLAMP,
    L_DMA_W_PW0,
    L_PW0,
    L_RESIDUAL0,
    L_DMA_W_DW1,
    L_DW1,
    L_DMA_W_PW1,
    L_PW1,
    L_PW1_CLAMP,
    L_DMA_W_PW2,
    L_PW2,
    L_DMA_W_DW2,
    L_DW2,
    L_DMA_W_PW3,
    L_PW3,
    L_RESIDUAL1,
    L_DMA_W_VALIDATE,
    L_VALIDATE_CONV,
    L_GLOBAL_AVG,
    L_DMA_W_CLASSIFIER,
    L_CLASSIFIER,
    L_DMA_OUT,
    LAYER_COUNT
};

static npu_tensor_t tensors[TENSOR_COUNT];
static npu_layer_t layers[LAYER_COUNT];

void npu_graph_trace(uint32_t layer_index, npu_op_type_t op, uint32_t event) {
    TRACE_LAYER = layer_index;
    TRACE_OP = (uint32_t)op;
    TRACE_EVENT = event;
}

void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *ptr = (uint8_t *)dst;
    for (uint32_t i = 0u; i < bytes; i++) {
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

static void clear_layer(npu_layer_t *layer) {
    layer->op = 0;
    layer->src = 0;
    layer->src2 = 0;
    layer->dst = 0;
    layer->aux = 0;
    layer->aux2 = 0;
    layer->l2_addr = 0;
    layer->bytes = 0;
    layer->dim_m = 0;
    layer->dim_n = 0;
    layer->multiplier = 0;
    layer->shift = 0;
    layer->min_val = 0;
    layer->max_val = 0;
    layer->linebuf_jobs = 0;
    layer->linebuf_job_count = 0;
    layer->linebuf_l2_jobs = 0;
    layer->linebuf_l2_job_count = 0;
}

static void set_dma_in(uint32_t layer, uint32_t tensor, uint32_t l2_addr, uint32_t bytes) {
    layers[layer].op = NPU_OP_DMA_IN;
    layers[layer].dst = tensor;
    layers[layer].l2_addr = l2_addr;
    layers[layer].bytes = bytes;
}

static void set_requant(npu_layer_t *layer, int32_t min_val, int32_t max_val) {
    layer->multiplier = 1;
    layer->shift = 0u;
    layer->min_val = min_val;
    layer->max_val = max_val;
}

static void init_graph(npu_graph_t *graph) {
    for (uint32_t i = 0u; i < LAYER_COUNT; i++) {
        clear_layer(&layers[i]);
    }

    init_tensor(&tensors[T_INPUT], T_INPUT_ADDR, INPUT_H, INPUT_W, 3u, INPUT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC);
    init_tensor(&tensors[T_WEIGHT], T_WEIGHT_ADDR, 1u, 1u, 32u, WEIGHT_MAX_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[T_A_48_C32_ROW32], T_ACT_A_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[T_B_48_C32_ROW32], T_ACT_B_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[T_C_48_C32_ROW32], T_ACT_C_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[T_A_48_C32], T_ACT_A_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_B_48_C32], T_ACT_B_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_C_48_C32], T_ACT_C_ADDR, STEM_H, STEM_W, 32u, ACT48_C32_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_A_24_C32], T_ACT_A_ADDR, MID_H, MID_W, 32u, MID_H * MID_W * 32u,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_B_24_C32], T_ACT_B_ADDR, MID_H, MID_W, 32u, MID_H * MID_W * 32u,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_A_24_C64], T_ACT_A_ADDR, MID_H, MID_W, 64u, ACT24_C64_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_B_24_C64], T_ACT_B_ADDR, MID_H, MID_W, 64u, ACT24_C64_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_C_24_C64], T_ACT_C_ADDR, MID_H, MID_W, 64u, ACT24_C64_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_C_24_C128], T_ACT_C_ADDR, MID_H, MID_W, 128u, ACT24_C128_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_D_24_C128], T_ACT_D_ADDR, MID_H, MID_W, 128u, ACT24_C128_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_A_1_C64], T_ACT_A_ADDR, 1u, 1u, 64u, 64u,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_B_1_C32], T_ACT_B_ADDR, 1u, 1u, 32u, 32u,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[T_PSUM], T_PSUM_ADDR, MID_H, MID_W, 32u, PSUM24_BYTES,
                NPU_DTYPE_I32, NPU_LAYOUT_ROW32);

    set_dma_in(L_DMA_IN_INPUT, T_INPUT, L2_INPUT, INPUT_BYTES);

    set_dma_in(L_DMA_W_STEM, T_WEIGHT, L2_W_STEM, STEM_WEIGHT_BYTES);
    layers[L_STEM_CONV].op = NPU_OP_CONV2D3X3S2P1_C3_LINEBUF_REQUANT;
    layers[L_STEM_CONV].src = T_INPUT;
    layers[L_STEM_CONV].dst = T_A_48_C32_ROW32;
    layers[L_STEM_CONV].aux = T_WEIGHT;
    set_requant(&layers[L_STEM_CONV], -128, 127);

    layers[L_STEM_CLAMP].op = NPU_OP_CLAMP_I8;
    layers[L_STEM_CLAMP].src = T_A_48_C32_ROW32;
    layers[L_STEM_CLAMP].dst = T_B_48_C32_ROW32;
    layers[L_STEM_CLAMP].bytes = ACT48_C32_BYTES;
    layers[L_STEM_CLAMP].min_val = 0;
    layers[L_STEM_CLAMP].max_val = 6;

    set_dma_in(L_DMA_W_DW0, T_WEIGHT, L2_W_DW0, DW32_WEIGHT_BYTES);
    layers[L_DW0].op = NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT;
    layers[L_DW0].src = T_B_48_C32_ROW32;
    layers[L_DW0].dst = T_A_48_C32_ROW32;
    layers[L_DW0].aux = T_WEIGHT;
    set_requant(&layers[L_DW0], -128, 127);

    layers[L_DW0_CLAMP].op = NPU_OP_CLAMP_I8;
    layers[L_DW0_CLAMP].src = T_A_48_C32_ROW32;
    layers[L_DW0_CLAMP].dst = T_C_48_C32_ROW32;
    layers[L_DW0_CLAMP].bytes = ACT48_C32_BYTES;
    layers[L_DW0_CLAMP].min_val = 0;
    layers[L_DW0_CLAMP].max_val = 6;

    set_dma_in(L_DMA_W_PW0, T_WEIGHT, L2_W_PW0, PW32_32_WEIGHT_BYTES);
    layers[L_PW0].op = NPU_OP_CONV2D1X1_C32_REQUANT;
    layers[L_PW0].src = T_C_48_C32;
    layers[L_PW0].dst = T_A_48_C32;
    layers[L_PW0].aux = T_WEIGHT;
    layers[L_PW0].aux2 = T_PSUM;
    set_requant(&layers[L_PW0], -128, 127);

    layers[L_RESIDUAL0].op = NPU_OP_ADD_I8;
    layers[L_RESIDUAL0].src = T_A_48_C32;
    layers[L_RESIDUAL0].aux = T_B_48_C32;
    layers[L_RESIDUAL0].dst = T_A_48_C32;
    layers[L_RESIDUAL0].bytes = ACT48_C32_BYTES;
    layers[L_RESIDUAL0].min_val = -128;
    layers[L_RESIDUAL0].max_val = 127;

    set_dma_in(L_DMA_W_DW1, T_WEIGHT, L2_W_DW1, DW32_WEIGHT_BYTES);
    layers[L_DW1].op = NPU_OP_DEPTHWISE3X3S2P1_C32_REQUANT;
    layers[L_DW1].src = T_A_48_C32;
    layers[L_DW1].dst = T_B_24_C32;
    layers[L_DW1].aux = T_WEIGHT;
    set_requant(&layers[L_DW1], -128, 127);

    set_dma_in(L_DMA_W_PW1, T_WEIGHT, L2_W_PW1, PW32_64_WEIGHT_BYTES);
    layers[L_PW1].op = NPU_OP_CONV2D1X1_C32_REQUANT;
    layers[L_PW1].src = T_B_24_C32;
    layers[L_PW1].dst = T_C_24_C64;
    layers[L_PW1].aux = T_WEIGHT;
    layers[L_PW1].aux2 = T_PSUM;
    set_requant(&layers[L_PW1], -128, 127);

    layers[L_PW1_CLAMP].op = NPU_OP_CLAMP_I8;
    layers[L_PW1_CLAMP].src = T_C_24_C64;
    layers[L_PW1_CLAMP].dst = T_A_24_C64;
    layers[L_PW1_CLAMP].bytes = ACT24_C64_BYTES;
    layers[L_PW1_CLAMP].min_val = 0;
    layers[L_PW1_CLAMP].max_val = 6;

    set_dma_in(L_DMA_W_PW2, T_WEIGHT, L2_W_PW2, PW64_128_WEIGHT_BYTES);
    layers[L_PW2].op = NPU_OP_CONV2D1X1_C32_REQUANT;
    layers[L_PW2].src = T_A_24_C64;
    layers[L_PW2].dst = T_D_24_C128;
    layers[L_PW2].aux = T_WEIGHT;
    layers[L_PW2].aux2 = T_PSUM;
    set_requant(&layers[L_PW2], -128, 127);

    set_dma_in(L_DMA_W_DW2, T_WEIGHT, L2_W_DW2, DW128_WEIGHT_BYTES);
    layers[L_DW2].op = NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT;
    layers[L_DW2].src = T_D_24_C128;
    layers[L_DW2].dst = T_C_24_C128;
    layers[L_DW2].aux = T_WEIGHT;
    set_requant(&layers[L_DW2], -128, 127);

    set_dma_in(L_DMA_W_PW3, T_WEIGHT, L2_W_PW3, PW128_64_WEIGHT_BYTES);
    layers[L_PW3].op = NPU_OP_CONV2D1X1_C32_REQUANT;
    layers[L_PW3].src = T_C_24_C128;
    layers[L_PW3].dst = T_B_24_C64;
    layers[L_PW3].aux = T_WEIGHT;
    layers[L_PW3].aux2 = T_PSUM;
    set_requant(&layers[L_PW3], -128, 127);

    layers[L_RESIDUAL1].op = NPU_OP_ADD_I8;
    layers[L_RESIDUAL1].src = T_B_24_C64;
    layers[L_RESIDUAL1].aux = T_A_24_C64;
    layers[L_RESIDUAL1].dst = T_B_24_C64;
    layers[L_RESIDUAL1].bytes = ACT24_C64_BYTES;
    layers[L_RESIDUAL1].min_val = -128;
    layers[L_RESIDUAL1].max_val = 127;

    set_dma_in(L_DMA_W_VALIDATE, T_WEIGHT, L2_W_VALIDATE, CONV64_64_WEIGHT_BYTES);
    layers[L_VALIDATE_CONV].op = NPU_OP_CONV2D3X3S1P1_C32_MULTI_LINEBUF_REQUANT;
    layers[L_VALIDATE_CONV].src = T_B_24_C64;
    layers[L_VALIDATE_CONV].dst = T_C_24_C64;
    layers[L_VALIDATE_CONV].aux = T_WEIGHT;
    layers[L_VALIDATE_CONV].aux2 = T_PSUM;
    layers[L_VALIDATE_CONV].dim_m = 3u;
    layers[L_VALIDATE_CONV].dim_n = MID_W;
    set_requant(&layers[L_VALIDATE_CONV], -128, 127);

    layers[L_GLOBAL_AVG].op = NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE;
    layers[L_GLOBAL_AVG].src = T_C_24_C64;
    layers[L_GLOBAL_AVG].dst = T_A_1_C64;

    set_dma_in(L_DMA_W_CLASSIFIER, T_WEIGHT, L2_W_CLASSIFIER, PW64_32_WEIGHT_BYTES);
    layers[L_CLASSIFIER].op = NPU_OP_CONV2D1X1_C32_REQUANT;
    layers[L_CLASSIFIER].src = T_A_1_C64;
    layers[L_CLASSIFIER].dst = T_B_1_C32;
    layers[L_CLASSIFIER].aux = T_WEIGHT;
    layers[L_CLASSIFIER].aux2 = T_PSUM;
    set_requant(&layers[L_CLASSIFIER], -128, 127);

    layers[L_DMA_OUT].op = NPU_OP_DMA_OUT;
    layers[L_DMA_OUT].src = T_B_1_C32;
    layers[L_DMA_OUT].l2_addr = L2_OUTPUT;
    layers[L_DMA_OUT].bytes = 32u;

    graph->tensors = tensors;
    graph->num_tensors = TENSOR_COUNT;
    graph->layers = layers;
    graph->num_layers = LAYER_COUNT;
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1u, 0u);

    npu_graph_t graph;
    init_graph(&graph);

    uint32_t status = npu_graph_run(&graph);
    if (status != NPU_GRAPH_OK) {
        spatz_rt_fail_at(1u, 0u, (int32_t)status, (int32_t)NPU_GRAPH_OK);
    }

    spatz_rt_pass();
    return 0;
}
