#include "npu_graph.h"
#include "npu_memory_map.h"

/*
 * Phase 3h Conv_Stem + SiLU + C2f_Conv + residual Add + Conv_Down + SPPF MaxPool + Upsample checkpoint:
 *   96x96x3 HWC -> linebuffer 3x3s2p1 -> GEMM32 -> clamp/requant
 *                -> Logistic LUT -> Mul -> fused linebuffer Conv3x3s1p1 C32
 *                -> Add residual(SiLU) -> fused linebuffer Conv3x3s2p1 C32
 *                -> MaxPool2D 5x5s1p2
 *                -> Upsample nearest 2x
 *                -> 48x48x32 INT8 L2 output.
 */
#define PASS_SIGNATURE 0xDEADBEEFu
#define FAIL_SIGNATURE 0xBAD30000u

#define SIG_STATUS     (*(volatile uint32_t *)0x10008000u)
#define SIG_FAIL_CODE  (*(volatile uint32_t *)0x10008004u)
#define SIG_LAYER      (*(volatile uint32_t *)0x10008008u)
#define SIG_OP         (*(volatile uint32_t *)0x1000800Cu)
#define SIG_EVENT      (*(volatile uint32_t *)0x10008010u)

#define L2_INPUT   0x80000000u
#define L2_WEIGHT0 0x80008000u
#define L2_SIG_LUT 0x80009000u
#define L2_WEIGHT1 0x8000A000u
#define L2_WEIGHT2 0x8000D000u
#define L2_OUTPUT  0x80020000u

#define SCRATCH_BASE 0x10100000u
#define SCRATCH_SIZE (512u * 1024u)

#define INPUT_H 96u
#define INPUT_W 96u
#define INPUT_C 3u
#define OUTPUT_H ((INPUT_H + 1u) / 2u)
#define OUTPUT_W ((INPUT_W + 1u) / 2u)
#define ROWS    (OUTPUT_H * OUTPUT_W)
#define DOWN_H  ((OUTPUT_H + 1u) / 2u)
#define DOWN_W  ((OUTPUT_W + 1u) / 2u)
#define DOWN_ROWS (DOWN_H * DOWN_W)
#define WEIGHT_BYTES (32u * 32u)
#define C2F_WEIGHT_BYTES (3u * 3u * 32u * 32u)
#define DOWN_WEIGHT_BYTES C2F_WEIGHT_BYTES
#define LUT_BYTES    256u
#define ACT_BYTES    (ROWS * 32u)
#define DOWN_ACT_BYTES (DOWN_ROWS * 32u)
#define C2F_PSUM_BYTES (ROWS * 32u * 4u)
#define DOWN_PSUM_BYTES (DOWN_ROWS * 32u * 4u)

#ifndef MICRO_YOLO_C2F_TILE_OH
#define MICRO_YOLO_C2F_TILE_OH 16u
#endif

#ifndef MICRO_YOLO_C2F_TILE_OW
#define MICRO_YOLO_C2F_TILE_OW 16u
#endif

#ifndef MICRO_YOLO_DOWN_TILE_OH
#define MICRO_YOLO_DOWN_TILE_OH 16u
#endif

#ifndef MICRO_YOLO_DOWN_TILE_OW
#define MICRO_YOLO_DOWN_TILE_OW 16u
#endif

void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *ptr = (uint8_t *)dst;
    for (uint32_t i = 0; i < bytes; i++) {
        ptr[i] = (uint8_t)value;
    }
    return dst;
}

enum {
    T_INPUT = 0,
    T_WEIGHT0,
    T_WEIGHT1,
    T_WEIGHT2,
    T_SIG_LUT,
    T_STEM,
    T_SIG,
    T_SILU,
    T_C2F_PSUM,
    T_DOWN_PSUM,
    T_OUT,
    T_DOWN,
    T_POOL,
    T_UPSAMPLE,
    TENSOR_COUNT
};

static npu_tensor_t tensors[TENSOR_COUNT];
static npu_layer_t layers[14];
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
    layer->aux2 = 0;
    layer->l2_addr = 0;
    layer->bytes = 0;
    layer->dim_m = 0;
    layer->dim_n = 0;
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
    layers[2].bytes = C2F_WEIGHT_BYTES;

    layers[3].op = NPU_OP_DMA_IN;
    layers[3].dst = T_WEIGHT2;
    layers[3].l2_addr = L2_WEIGHT2;
    layers[3].bytes = DOWN_WEIGHT_BYTES;

    layers[4].op = NPU_OP_DMA_IN;
    layers[4].dst = T_SIG_LUT;
    layers[4].l2_addr = L2_SIG_LUT;
    layers[4].bytes = LUT_BYTES;

    layers[5].op = NPU_OP_CONV2D3X3S2P1_C3_LINEBUF_REQUANT;
    layers[5].src = T_INPUT;
    layers[5].dst = T_STEM;
    layers[5].aux = T_WEIGHT0;
    layers[5].dim_m = ROWS;
    layers[5].multiplier = 1;
    layers[5].shift = 0;
    layers[5].min_val = -128;
    layers[5].max_val = 127;

    layers[6].op = NPU_OP_LOGISTIC_LUT_I8;
    layers[6].src = T_STEM;
    layers[6].dst = T_SIG;
    layers[6].aux = T_SIG_LUT;
    layers[6].bytes = ACT_BYTES;

    layers[7].op = NPU_OP_MUL_I8;
    layers[7].src = T_STEM;
    layers[7].dst = T_SILU;
    layers[7].aux = T_SIG;
    layers[7].bytes = ACT_BYTES;
    layers[7].multiplier = 1;
    layers[7].shift = 7;
    layers[7].min_val = -128;
    layers[7].max_val = 127;

    layers[8].op = NPU_OP_CONV2D3X3S1P1_C32_LINEBUF_REQUANT;
    layers[8].src = T_SILU;
    layers[8].dst = T_OUT;
    layers[8].aux = T_WEIGHT1;
    layers[8].aux2 = T_C2F_PSUM;
    layers[8].bytes = C2F_PSUM_BYTES;
    layers[8].dim_m = MICRO_YOLO_C2F_TILE_OH;
    layers[8].dim_n = MICRO_YOLO_C2F_TILE_OW;
    layers[8].multiplier = 1;
    layers[8].shift = 0;
    layers[8].min_val = -128;
    layers[8].max_val = 127;

    layers[9].op = NPU_OP_ADD_I8;
    layers[9].src = T_OUT;
    layers[9].dst = T_OUT;
    layers[9].aux = T_SILU;
    layers[9].bytes = ACT_BYTES;
    layers[9].min_val = -128;
    layers[9].max_val = 127;

    layers[10].op = NPU_OP_CONV2D3X3S2P1_C32_LINEBUF_REQUANT;
    layers[10].src = T_OUT;
    layers[10].dst = T_DOWN;
    layers[10].aux = T_WEIGHT2;
    layers[10].aux2 = T_DOWN_PSUM;
    layers[10].bytes = DOWN_PSUM_BYTES;
    layers[10].dim_m = MICRO_YOLO_DOWN_TILE_OH;
    layers[10].dim_n = MICRO_YOLO_DOWN_TILE_OW;
    layers[10].multiplier = 1;
    layers[10].shift = 0;
    layers[10].min_val = -128;
    layers[10].max_val = 127;

    layers[11].op = NPU_OP_MAXPOOL2D5X5S1P2_I8;
    layers[11].src = T_DOWN;
    layers[11].dst = T_POOL;
    layers[11].bytes = DOWN_ACT_BYTES;

    layers[12].op = NPU_OP_UPSAMPLE_NEAREST2X_I8;
    layers[12].src = T_POOL;
    layers[12].dst = T_UPSAMPLE;
    layers[12].bytes = ACT_BYTES;

    layers[13].op = NPU_OP_DMA_OUT;
    layers[13].src = T_UPSAMPLE;
    layers[13].l2_addr = L2_OUTPUT;
    layers[13].bytes = ACT_BYTES;

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

    uint32_t input_addr = alloc_or_fail(&scratch, INPUT_H * INPUT_W * INPUT_C);
    uint32_t weight0_addr = alloc_or_fail(&scratch, WEIGHT_BYTES);
    uint32_t weight1_addr = alloc_or_fail(&scratch, C2F_WEIGHT_BYTES);
    uint32_t weight2_addr = alloc_or_fail(&scratch, DOWN_WEIGHT_BYTES);
    uint32_t lut_addr = alloc_or_fail(&scratch, LUT_BYTES);
    uint32_t act_a_addr = alloc_or_fail(&scratch, ACT_BYTES);
    uint32_t psum_or_sig_addr = alloc_or_fail(&scratch, C2F_PSUM_BYTES);
    uint32_t act_c_addr = alloc_or_fail(&scratch, ACT_BYTES);

    init_tensor(&tensors[T_INPUT], input_addr,
                INPUT_H, INPUT_W, INPUT_C, INPUT_H * INPUT_W * INPUT_C,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
    init_tensor(&tensors[T_WEIGHT0], weight0_addr,
                32, 32, 32, WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_WEIGHT1], weight1_addr,
                288, 32, 32, C2F_WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_WEIGHT2], weight2_addr,
                288, 32, 32, DOWN_WEIGHT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_SIG_LUT], lut_addr,
                1, 1, 256, LUT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
    init_tensor(&tensors[T_STEM], act_a_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_SIG], psum_or_sig_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_SILU], act_c_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_C2F_PSUM], psum_or_sig_addr,
                OUTPUT_H, OUTPUT_W, 32, C2F_PSUM_BYTES,
                NPU_DTYPE_I32, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_DOWN_PSUM], psum_or_sig_addr,
                DOWN_H, DOWN_W, 32, DOWN_PSUM_BYTES,
                NPU_DTYPE_I32, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_OUT], act_a_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_DOWN], act_c_addr,
                DOWN_H, DOWN_W, 32, DOWN_ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_POOL], act_a_addr,
                DOWN_H, DOWN_W, 32, DOWN_ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_UPSAMPLE], act_c_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
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
