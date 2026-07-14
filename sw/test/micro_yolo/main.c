#include "conv2d_packed.h"
#include "idma_mm_utils.h"
#include "npu_graph.h"
#include "npu_memory_map.h"

/*
 * Phase 3j Conv_Stem + SiLU + C2f_Conv + residual Add + Conv_Down + SPPF MaxPool + Upsample + Head_Conv checkpoint:
 *   96x96x3 HWC -> linebuffer 3x3s2p1 -> GEMM32 -> clamp/requant
 *                -> Logistic LUT -> Mul -> fused linebuffer Conv3x3s1p1 C32
 *                -> Add residual(SiLU) -> fused linebuffer Conv3x3s2p1 C32
 *                -> MaxPool2D 5x5s1p2
 *                -> Upsample nearest 2x
 *                -> bypass Concat(L9, L3) with two C32 Head_Conv chunks
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
#define L2_WEIGHT3 0x80010000u
#define L2_OUTPUT  0x80020000u
#define L2_SKIP    0x80040000u
#define L2_LINEBUF_DESC_BASE 0x80052000u
#define L2_DFL_EXP_LUT 0x80054000u
#define L2_DFL_RECIP_LUT 0x80054400u
#define L2_DFL_OUTPUT 0x80060000u
#define LINEBUF_DESC_MANIFEST_MAGIC 0x4D594C42u
#define LINEBUF_DESC_MANIFEST_VERSION 1u
#define LINEBUF_DESC_MANIFEST_ENTRIES 5u
#define LINEBUF_DESC_KIND_LINEBUF 1u
#define LINEBUF_DESC_KIND_L2_COPY 2u

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
#define HEAD_WEIGHT_BYTES (2u * C2F_WEIGHT_BYTES)
#define LUT_BYTES    256u
#define ACT_BYTES    (ROWS * 32u)
#define DOWN_ACT_BYTES (DOWN_ROWS * 32u)
#define C2F_PSUM_BYTES (ROWS * 32u * 4u)
#define DOWN_PSUM_BYTES (DOWN_ROWS * 32u * 4u)
#define DFL_BOX_CHANNELS 16u
#define DFL_SIDES 4u
#define DFL_LUT_BYTES (256u * 4u)
#define DFL_OUTPUT_BYTES (ROWS * DFL_SIDES * 2u)

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

#ifndef MICRO_YOLO_HEAD_TILE_OH
#define MICRO_YOLO_HEAD_TILE_OH 16u
#endif

#ifndef MICRO_YOLO_HEAD_TILE_OW
#define MICRO_YOLO_HEAD_TILE_OW 16u
#endif

#define HEAD_TILE_BYTES (MICRO_YOLO_HEAD_TILE_OH * MICRO_YOLO_HEAD_TILE_OW * 32u)

_Static_assert(sizeof(systolic_linebuf_cfg_t) == 76u, "linebuffer config ABI changed");
_Static_assert(sizeof(npu_conv2d_linebuf_job_desc_t) == 120u, "linebuffer job ABI changed");
_Static_assert(sizeof(npu_conv2d_l2_copy_job_desc_t) == 136u, "linebuffer L2 job ABI changed");

typedef struct {
    uint32_t l2_addr;
    uint32_t count;
    uint32_t bytes;
    uint32_t kind;
} micro_yolo_linebuf_manifest_entry_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint32_t reserved;
    micro_yolo_linebuf_manifest_entry_t entries[LINEBUF_DESC_MANIFEST_ENTRIES];
} micro_yolo_linebuf_manifest_t;

enum {
    LB_DESC_STEM = 0,
    LB_DESC_C2F = 1,
    LB_DESC_DOWN = 2,
    LB_DESC_HEAD0 = 3,
    LB_DESC_HEAD1_L2 = 4,
};

_Static_assert(sizeof(micro_yolo_linebuf_manifest_t) == 96u,
               "linebuffer manifest ABI changed");

/*
 * Static 3j graph contract:
 *
 * - All C32 activations use ROW32 layout: one contiguous 32-byte vector per
 *   spatial pixel.
 * - The logical concat before Head_Conv is intentionally not materialized.
 *   Layer 16 consumes two tensors: T_UPSAMPLE as chunk 0 and T_SKIP_RELOAD as
 *   chunk 1. The graph runtime accumulates both chunks into INT32 psum and
 *   requants only after chunk 1.
 * - T_C2F_PSUM and T_DOWN_PSUM intentionally alias the same large INT32 scratch
 *   allocation. Their lifetimes do not overlap.
 * - T_SIG also aliases that scratch allocation before any Conv psum use.
 * - T_HEAD_TILE must not alias T_SKIP_RELOAD or T_UPSAMPLE because layer 16
 *   reads both input branches while writing the output tile.
 */

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
    T_WEIGHT3,
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
    T_SKIP_RELOAD,
    T_HEAD_TILE,
    T_RAW_HEAD,
    T_DFL_OUT,
    T_DFL_EXP_LUT,
    T_DFL_RECIP_LUT,
    TENSOR_COUNT
};

static npu_tensor_t tensors[TENSOR_COUNT];
static npu_layer_t layers[20];
static npu_graph_t graph;
static npu_conv2d_linebuf_job_desc_t *micro_yolo_lb_stem_jobs;
static npu_conv2d_linebuf_job_desc_t *micro_yolo_lb_c2f_jobs;
static npu_conv2d_linebuf_job_desc_t *micro_yolo_lb_down_jobs;
static npu_conv2d_linebuf_job_desc_t *micro_yolo_lb_head0_jobs;
static npu_conv2d_l2_copy_job_desc_t *micro_yolo_lb_head1_l2_jobs;
static uint32_t micro_yolo_lb_stem_count;
static uint32_t micro_yolo_lb_c2f_count;
static uint32_t micro_yolo_lb_down_count;
static uint32_t micro_yolo_lb_head0_count;
static uint32_t micro_yolo_lb_head1_l2_count;

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

static uint32_t alloc_or_fail(npu_graph_scratch_t *scratch, uint32_t bytes);

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

static void set_linebuf_jobs(npu_layer_t *layer,
                             const npu_conv2d_linebuf_job_desc_t *jobs,
                             uint32_t job_count) {
    layer->linebuf_jobs = jobs;
    layer->linebuf_job_count = job_count;
}

static void set_linebuf_l2_jobs(npu_layer_t *layer,
                                const npu_conv2d_l2_copy_job_desc_t *jobs,
                                uint32_t job_count) {
    layer->linebuf_l2_jobs = jobs;
    layer->linebuf_l2_job_count = job_count;
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
    layers[4].dst = T_WEIGHT3;
    layers[4].l2_addr = L2_WEIGHT3;
    layers[4].bytes = HEAD_WEIGHT_BYTES;

    layers[5].op = NPU_OP_DMA_IN;
    layers[5].dst = T_SIG_LUT;
    layers[5].l2_addr = L2_SIG_LUT;
    layers[5].bytes = LUT_BYTES;

    layers[6].op = NPU_OP_CONV2D3X3S2P1_C3_LINEBUF_REQUANT;
    layers[6].src = T_INPUT;
    layers[6].dst = T_STEM;
    layers[6].aux = T_WEIGHT0;
    layers[6].dim_m = ROWS;
    layers[6].multiplier = 1;
    layers[6].shift = 0;
    layers[6].min_val = -128;
    layers[6].max_val = 127;
    set_linebuf_jobs(&layers[6], micro_yolo_lb_stem_jobs, micro_yolo_lb_stem_count);

    layers[7].op = NPU_OP_LOGISTIC_LUT_I8;
    layers[7].src = T_STEM;
    layers[7].dst = T_SIG;
    layers[7].aux = T_SIG_LUT;
    layers[7].bytes = ACT_BYTES;

    layers[8].op = NPU_OP_MUL_I8;
    layers[8].src = T_STEM;
    layers[8].dst = T_SILU;
    layers[8].aux = T_SIG;
    layers[8].bytes = ACT_BYTES;
    layers[8].multiplier = 1;
    layers[8].shift = 7;
    layers[8].min_val = -128;
    layers[8].max_val = 127;

    /*
     * Preserve the L3 skip branch before T_SILU/T_DOWN reuse overwrites this
     * TCDM storage. This replaces materialized concat: the skip branch is later
     * reloaded and consumed as Head_Conv chunk 1.
     */
    layers[9].op = NPU_OP_DMA_OUT;
    layers[9].src = T_SILU;
    layers[9].l2_addr = L2_SKIP;
    layers[9].bytes = ACT_BYTES;

    layers[10].op = NPU_OP_CONV2D3X3S1P1_C32_LINEBUF_REQUANT;
    layers[10].src = T_SILU;
    layers[10].dst = T_OUT;
    layers[10].aux = T_WEIGHT1;
    layers[10].aux2 = T_C2F_PSUM;
    layers[10].bytes = C2F_PSUM_BYTES;
    layers[10].dim_m = MICRO_YOLO_C2F_TILE_OH;
    layers[10].dim_n = MICRO_YOLO_C2F_TILE_OW;
    layers[10].multiplier = 1;
    layers[10].shift = 0;
    layers[10].min_val = -128;
    layers[10].max_val = 127;
    set_linebuf_jobs(&layers[10], micro_yolo_lb_c2f_jobs, micro_yolo_lb_c2f_count);

    layers[11].op = NPU_OP_ADD_I8;
    layers[11].src = T_OUT;
    layers[11].dst = T_OUT;
    layers[11].aux = T_SILU;
    layers[11].bytes = ACT_BYTES;
    layers[11].min_val = -128;
    layers[11].max_val = 127;

    layers[12].op = NPU_OP_CONV2D3X3S2P1_C32_LINEBUF_REQUANT;
    layers[12].src = T_OUT;
    layers[12].dst = T_DOWN;
    layers[12].aux = T_WEIGHT2;
    layers[12].aux2 = T_DOWN_PSUM;
    layers[12].bytes = DOWN_PSUM_BYTES;
    layers[12].dim_m = MICRO_YOLO_DOWN_TILE_OH;
    layers[12].dim_n = MICRO_YOLO_DOWN_TILE_OW;
    layers[12].multiplier = 1;
    layers[12].shift = 0;
    layers[12].min_val = -128;
    layers[12].max_val = 127;
    set_linebuf_jobs(&layers[12], micro_yolo_lb_down_jobs, micro_yolo_lb_down_count);

    layers[13].op = NPU_OP_MAXPOOL2D5X5S1P2_I8;
    layers[13].src = T_DOWN;
    layers[13].dst = T_POOL;
    layers[13].bytes = DOWN_ACT_BYTES;

    layers[14].op = NPU_OP_UPSAMPLE_NEAREST2X_I8;
    layers[14].src = T_POOL;
    layers[14].dst = T_UPSAMPLE;
    layers[14].bytes = ACT_BYTES;

    /* Reload the preserved skip branch for logical concat bypass. */
    layers[15].op = NPU_OP_DMA_IN;
    layers[15].dst = T_SKIP_RELOAD;
    layers[15].l2_addr = L2_SKIP;
    layers[15].bytes = ACT_BYTES;

    /*
     * Fused logical Concat + Head_Conv:
     *   src  = upsample branch, weight3 chunk 0 -> INT32 psum
     *   src2 = skip branch,     weight3 chunk 1 -> accumulate + requant
     * The op writes the final INT8 head output tile-by-tile to L2_OUTPUT.
     */
    layers[16].op = NPU_OP_CONV2D3X3S1P1_C32X2_LINEBUF_REQUANT_L2;
    layers[16].src = T_UPSAMPLE;
    layers[16].src2 = T_SKIP_RELOAD;
    layers[16].dst = T_HEAD_TILE;
    layers[16].aux = T_WEIGHT3;
    layers[16].aux2 = T_C2F_PSUM;
    layers[16].l2_addr = L2_OUTPUT;
    layers[16].bytes = ACT_BYTES;
    layers[16].dim_m = MICRO_YOLO_HEAD_TILE_OH;
    layers[16].dim_n = MICRO_YOLO_HEAD_TILE_OW;
    layers[16].multiplier = 1;
    layers[16].shift = 0;
    layers[16].min_val = -128;
    layers[16].max_val = 127;
    set_linebuf_jobs(&layers[16], micro_yolo_lb_head0_jobs, micro_yolo_lb_head0_count);
    set_linebuf_l2_jobs(&layers[16], micro_yolo_lb_head1_l2_jobs, micro_yolo_lb_head1_l2_count);

    /*
     * Post-head DFL graph stage:
     *   - Raw head is produced tile-by-tile directly to L2_OUTPUT.
     *   - Reload the complete ROW32 head once into T_RAW_HEAD after layer 16.
     *   - Run AFU-assisted exp LUT over the first 16 channels of each pixel
     *     and reduce 4 bins per side into Q8.8 distances.
     */
    layers[17].op = NPU_OP_DMA_IN;
    layers[17].dst = T_RAW_HEAD;
    layers[17].l2_addr = L2_OUTPUT;
    layers[17].bytes = ACT_BYTES;

    layers[18].op = NPU_OP_DFL_SOFTMAX4_I8_Q8;
    layers[18].src = T_RAW_HEAD;
    layers[18].dst = T_DFL_OUT;
    layers[18].aux = T_DFL_EXP_LUT;
    layers[18].aux2 = T_DFL_RECIP_LUT;
    layers[18].bytes = DFL_OUTPUT_BYTES;

    layers[19].op = NPU_OP_DMA_OUT;
    layers[19].src = T_DFL_OUT;
    layers[19].l2_addr = L2_DFL_OUTPUT;
    layers[19].bytes = DFL_OUTPUT_BYTES;

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

static uint32_t load_desc_or_fail(npu_graph_scratch_t *scratch,
                                  const micro_yolo_linebuf_manifest_entry_t *entry,
                                  uint32_t expected_kind,
                                  uint32_t desc_size) {
    SIG_EVENT = 0xD1000000u | expected_kind;
    if (entry->kind != expected_kind ||
        entry->count == 0u ||
        entry->bytes != (entry->count * desc_size)) {
        fail(NPU_GRAPH_ERR_BAD_TENSOR);
    }

    uint32_t bytes = entry->bytes;
    uint32_t addr = alloc_or_fail(scratch, bytes);
    if (!idma_memcpy_blocking(entry->l2_addr, addr, bytes)) {
        fail(NPU_GRAPH_ERR_DMA);
    }
    return addr;
}

static void load_runtime_linebuf_jobs(npu_graph_scratch_t *scratch) {
    uint32_t manifest_addr = alloc_or_fail(scratch, sizeof(micro_yolo_linebuf_manifest_t));
    micro_yolo_linebuf_manifest_t *manifest = (micro_yolo_linebuf_manifest_t *)manifest_addr;

    SIG_EVENT = 0xD0000001u;
    if (!idma_memcpy_blocking(L2_LINEBUF_DESC_BASE,
                              manifest_addr,
                              sizeof(micro_yolo_linebuf_manifest_t))) {
        fail(NPU_GRAPH_ERR_DMA);
    }
    SIG_EVENT = 0xD0000002u;
    if (manifest->magic != LINEBUF_DESC_MANIFEST_MAGIC ||
        manifest->version != LINEBUF_DESC_MANIFEST_VERSION ||
        manifest->entry_count != LINEBUF_DESC_MANIFEST_ENTRIES) {
        fail(NPU_GRAPH_ERR_BAD_TENSOR);
    }

    micro_yolo_lb_stem_jobs = (npu_conv2d_linebuf_job_desc_t *)
        load_desc_or_fail(scratch, &manifest->entries[LB_DESC_STEM],
                          LINEBUF_DESC_KIND_LINEBUF,
                          sizeof(npu_conv2d_linebuf_job_desc_t));
    micro_yolo_lb_c2f_jobs = (npu_conv2d_linebuf_job_desc_t *)
        load_desc_or_fail(scratch, &manifest->entries[LB_DESC_C2F],
                          LINEBUF_DESC_KIND_LINEBUF,
                          sizeof(npu_conv2d_linebuf_job_desc_t));
    micro_yolo_lb_down_jobs = (npu_conv2d_linebuf_job_desc_t *)
        load_desc_or_fail(scratch, &manifest->entries[LB_DESC_DOWN],
                          LINEBUF_DESC_KIND_LINEBUF,
                          sizeof(npu_conv2d_linebuf_job_desc_t));
    micro_yolo_lb_head0_jobs = (npu_conv2d_linebuf_job_desc_t *)
        load_desc_or_fail(scratch, &manifest->entries[LB_DESC_HEAD0],
                          LINEBUF_DESC_KIND_LINEBUF,
                          sizeof(npu_conv2d_linebuf_job_desc_t));
    micro_yolo_lb_head1_l2_jobs = (npu_conv2d_l2_copy_job_desc_t *)
        load_desc_or_fail(scratch, &manifest->entries[LB_DESC_HEAD1_L2],
                          LINEBUF_DESC_KIND_L2_COPY,
                          sizeof(npu_conv2d_l2_copy_job_desc_t));

    micro_yolo_lb_stem_count = manifest->entries[LB_DESC_STEM].count;
    micro_yolo_lb_c2f_count = manifest->entries[LB_DESC_C2F].count;
    micro_yolo_lb_down_count = manifest->entries[LB_DESC_DOWN].count;
    micro_yolo_lb_head0_count = manifest->entries[LB_DESC_HEAD0].count;
    micro_yolo_lb_head1_l2_count = manifest->entries[LB_DESC_HEAD1_L2].count;
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
    uint32_t weight3_addr = alloc_or_fail(&scratch, HEAD_WEIGHT_BYTES);
    uint32_t lut_addr = alloc_or_fail(&scratch, LUT_BYTES);
    uint32_t act_a_addr = alloc_or_fail(&scratch, ACT_BYTES);
    uint32_t psum_or_sig_addr = alloc_or_fail(&scratch, C2F_PSUM_BYTES);
    uint32_t act_c_addr = alloc_or_fail(&scratch, ACT_BYTES);
    uint32_t head_tile_addr = alloc_or_fail(&scratch, HEAD_TILE_BYTES);
    uint32_t dfl_exp_lut_addr = alloc_or_fail(&scratch, DFL_LUT_BYTES);
    uint32_t dfl_recip_lut_addr = alloc_or_fail(&scratch, DFL_LUT_BYTES);
    load_runtime_linebuf_jobs(&scratch);
    if (!idma_memcpy_blocking(L2_DFL_EXP_LUT, dfl_exp_lut_addr, DFL_LUT_BYTES) ||
        !idma_memcpy_blocking(L2_DFL_RECIP_LUT, dfl_recip_lut_addr, DFL_LUT_BYTES)) {
        fail(NPU_GRAPH_ERR_DMA);
    }

    /*
     * Buffer lifetime map:
     *
     * act_a:
     *   T_STEM -> T_OUT -> T_POOL -> T_SKIP_RELOAD
     * act_c:
     *   T_SILU -> T_DOWN -> T_UPSAMPLE
     * psum_or_sig:
     *   T_SIG -> T_C2F_PSUM/T_DOWN_PSUM/head full psum -> T_DFL_OUT
     *
     * After Head_Conv completes, T_SKIP_RELOAD is dead and act_a is reused as
     * T_RAW_HEAD for the L2 reload feeding DFL. DFL exp/reciprocal LUTs are
     * host-precomputed blobs loaded from L2 and then written to AFU fixed LUT
     * windows by the graph DFL operator.
     *
     * The static aliases above are part of the test contract. If a new layer is
     * inserted, update this map first; otherwise the graph can pass build but
     * silently overwrite an input branch.
     */
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
    init_tensor(&tensors[T_WEIGHT3], weight3_addr,
                576, 32, 32, HEAD_WEIGHT_BYTES,
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
    init_tensor(&tensors[T_SKIP_RELOAD], act_a_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_HEAD_TILE], head_tile_addr,
                MICRO_YOLO_HEAD_TILE_OH, MICRO_YOLO_HEAD_TILE_OW, 32, HEAD_TILE_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_RAW_HEAD], act_a_addr,
                OUTPUT_H, OUTPUT_W, 32, ACT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_ROW32, 1, 0);
    init_tensor(&tensors[T_DFL_OUT], psum_or_sig_addr,
                OUTPUT_H, OUTPUT_W, DFL_SIDES, DFL_OUTPUT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
    init_tensor(&tensors[T_DFL_EXP_LUT], dfl_exp_lut_addr,
                1, 1, 256, DFL_LUT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
    init_tensor(&tensors[T_DFL_RECIP_LUT], dfl_recip_lut_addr,
                1, 1, 256, DFL_LUT_BYTES,
                NPU_DTYPE_I8, NPU_LAYOUT_HWC, 1, 0);
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
