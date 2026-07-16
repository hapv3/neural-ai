#include "npu_graph.h"
#include "spatz_rt.h"

/*
 * Scenario: graph-level depthwise Conv3x3 S1 P1 C32 fast path.
 * Target: prove a C32-blocked 1x32x48x48 tensor uses the RTL linebuffer-fed
 * lane-wise depthwise MAC path, with one C32 tap vector consumed per cycle.
 */

#define L2_INPUT   0x80000000u
#define L2_WEIGHT  0x80040000u
#define L2_OUTPUT  0x80050000u
#define L2_CONFIG  0x8007F000u

#define T_INPUT    0x10100000u
#define T_OUTPUT   0x10138000u
#define T_WEIGHT   0x10170000u
#define T_CONFIG   0x1017F000u

#define DEPTHWISE_CONFIG_MAGIC 0x44574346u

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

typedef struct {
    uint16_t h;
    uint16_t w;
    uint16_t c;
    uint16_t stride;
} depthwise_case_t;

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

static uint32_t c32_groups(uint32_t channels) {
    return (channels + 31u) >> 5;
}

static uint32_t output_dim(uint32_t size, uint32_t stride) {
    return ((size - 1u) / stride) + 1u;
}

static uint32_t input_bytes(const depthwise_case_t *cfg) {
    return (uint32_t)cfg->h * (uint32_t)cfg->w * c32_groups(cfg->c) * 32u;
}

static uint32_t output_bytes(const depthwise_case_t *cfg) {
    return output_dim(cfg->h, cfg->stride) *
           output_dim(cfg->w, cfg->stride) *
           c32_groups(cfg->c) * 32u;
}

static uint32_t weight_bytes(const depthwise_case_t *cfg) {
    return 3u * 3u * c32_groups(cfg->c) * 32u;
}

static depthwise_case_t select_case(uint32_t case_id) {
    depthwise_case_t cfg;
    cfg.h = 48u;
    cfg.w = 48u;
    cfg.c = 32u;
    cfg.stride = 1u;

    switch (case_id) {
    case 1u:
        cfg.h = 24u;
        cfg.w = 24u;
        cfg.c = 64u;
        break;
    case 2u:
        cfg.c = 33u;
        break;
    case 3u:
        cfg.c = 64u;
        break;
    case 4u:
        cfg.c = 65u;
        cfg.stride = 2u;
        break;
    case 5u:
        cfg.c = 96u;
        break;
    case 6u:
        cfg.stride = 2u;
        break;
    default:
        break;
    }
    return cfg;
}

static uint32_t load_case_id(void) {
    volatile uint32_t *cfg = (volatile uint32_t *)T_CONFIG;
    spatz_rt_dma_1d(T_CONFIG, L2_CONFIG, 2u * sizeof(uint32_t));
    spatz_rt_dma_wait_all();
    if (cfg[0] == DEPTHWISE_CONFIG_MAGIC) {
        return cfg[1];
    }
    return 0u;
}

static void init_graph(const depthwise_case_t *cfg) {
    uint32_t out_h = output_dim(cfg->h, cfg->stride);
    uint32_t out_w = output_dim(cfg->w, cfg->stride);
    uint32_t in_bytes = input_bytes(cfg);
    uint32_t out_bytes = output_bytes(cfg);
    uint32_t wgt_bytes = weight_bytes(cfg);

    init_tensor(&tensors[TENSOR_INPUT], T_INPUT, cfg->h, cfg->w, cfg->c, in_bytes,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);
    init_tensor(&tensors[TENSOR_WEIGHT], T_WEIGHT, 3u, 3u, cfg->c, wgt_bytes,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32);
    init_tensor(&tensors[TENSOR_OUTPUT], T_OUTPUT, out_h, out_w, cfg->c, out_bytes,
                NPU_DTYPE_I8, NPU_LAYOUT_C32_BLOCKED);

    layers[L_DMA_IN_INPUT].op = DMA_IN;
    layers[L_DMA_IN_INPUT].dst = TENSOR_INPUT;
    layers[L_DMA_IN_INPUT].l2_addr = L2_INPUT;
    layers[L_DMA_IN_INPUT].bytes = in_bytes;

    layers[L_DMA_IN_WEIGHT].op = DMA_IN;
    layers[L_DMA_IN_WEIGHT].dst = TENSOR_WEIGHT;
    layers[L_DMA_IN_WEIGHT].l2_addr = L2_WEIGHT;
    layers[L_DMA_IN_WEIGHT].bytes = wgt_bytes;

    layers[L_DEPTHWISE].op = (cfg->stride == 2u) ?
                              DEPTHWISE_CONV2D_C32_DOWNSAMPLE_REQUANT :
                              DEPTHWISE_CONV2D_C32_REQUANT;
    layers[L_DEPTHWISE].src = TENSOR_INPUT;
    layers[L_DEPTHWISE].dst = TENSOR_OUTPUT;
    layers[L_DEPTHWISE].aux = TENSOR_WEIGHT;
    layers[L_DEPTHWISE].multiplier = 1;
    layers[L_DEPTHWISE].shift = 0u;
    layers[L_DEPTHWISE].min_val = -128;
    layers[L_DEPTHWISE].max_val = 127;

    layers[L_DMA_OUT].op = DMA_OUT;
    layers[L_DMA_OUT].src = TENSOR_OUTPUT;
    layers[L_DMA_OUT].l2_addr = L2_OUTPUT;
    layers[L_DMA_OUT].bytes = out_bytes;
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1u, 0u);
    depthwise_case_t cfg = select_case(load_case_id());
    init_graph(&cfg);

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
