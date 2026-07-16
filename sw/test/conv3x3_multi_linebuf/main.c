#include "conv2d_packed.h"
#include "hal_systolic.h"
#include "spatz_rt.h"

#define L2_INPUT          0x80000000u
#define L2_WEIGHT_SPLIT   0x80010000u
#define L2_WEIGHT_FUSED   0x80020000u
#define L2_OUT_SPLIT      0x80030000u
#define L2_OUT_FUSED      0x80040000u
#define L2_STATS_SPLIT    0x80050000u
#define L2_STATS_FUSED    0x80050100u
#define L2_MODE           0x80050200u

#define T_INPUT           0x10100000u
#define T_WEIGHT_SPLIT    0x10110000u
#define T_WEIGHT_FUSED    0x10120000u
#define T_OUT_SPLIT       0x10130000u
#define T_OUT_FUSED       0x10140000u
#define T_PSUM            0x10150000u
#define T_MODE            0x10170000u

#define H                 24u
#define W                 24u
#define IC                64u
#define OC                64u
#define C32               32u
#define KERNEL            3u
#define TILE_OH           3u
#define ROWS              (H * W)
#define GROUPS            2u
#define INPUT_BYTES       (ROWS * IC)
#define OUTPUT_BYTES      (ROWS * OC)
#define CHUNK_WEIGHT_BYTES (KERNEL * KERNEL * C32 * C32)
#define WEIGHT_BYTES      (GROUPS * GROUPS * CHUNK_WEIGHT_BYTES)
#define PSUM_BYTES        (ROWS * C32 * 4u)

static int32_t rq_bias[32];
static int32_t rq_multiplier[32];
static uint8_t rq_shift[32];
static int32_t rq_zero_point[32];

void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *ptr = (uint8_t *)dst;
    for (uint32_t i = 0u; i < bytes; i++) {
        ptr[i] = (uint8_t)value;
    }
    return dst;
}

static void clear_stats(npu_conv2d_packed_stats_t *stats) {
    uint32_t *words = (uint32_t *)stats;
    for (uint32_t i = 0u; i < (sizeof(*stats) / sizeof(uint32_t)); i++) {
        words[i] = 0u;
    }
}

static void add_stats(npu_conv2d_packed_stats_t *dst,
                      const npu_conv2d_packed_stats_t *src) {
    dst->rows += src->rows;
    dst->k_tiles += src->k_tiles;
    dst->prepare_cycles += src->prepare_cycles;
    dst->gemm_cycles += src->gemm_cycles;
    dst->total_cycles += src->total_cycles;
    dst->last_prepare_cycles = src->last_prepare_cycles;
    dst->last_gemm_cycles = src->last_gemm_cycles;
    dst->status = src->status;
    dst->prepare_idma_tiles += src->prepare_idma_tiles;
    dst->prepare_idma_transfers += src->prepare_idma_transfers;
    dst->prepare_spatz_tiles += src->prepare_spatz_tiles;
    dst->prepare_scalar_tiles += src->prepare_scalar_tiles;
}

static void init_requant(void) {
    for (uint32_t ch = 0u; ch < 32u; ch++) {
        rq_bias[ch] = 0;
        rq_multiplier[ch] = 1;
        rq_shift[ch] = 0u;
        rq_zero_point[ch] = 0;
    }
    systolic_requant_config_per_channel(rq_bias, rq_multiplier, rq_shift, rq_zero_point,
                                        -128, 127);
}

static void build_tiles(npu_conv2d_spatial_tile_t *tiles, uint32_t *tile_count) {
    uint32_t idx = 0u;
    for (uint32_t oh = 0u; oh < H; oh += TILE_OH) {
        tiles[idx].oh_base = oh;
        tiles[idx].ow_base = 0u;
        tiles[idx].tile_oh = TILE_OH;
        tiles[idx].tile_ow = W;
        idx++;
    }
    *tile_count = idx;
}

static void init_cfg(npu_conv2d_packed_cfg_t *cfg,
                     uint32_t input_addr,
                     uint32_t weight_addr,
                     uint32_t output_addr,
                     uint32_t input_c) {
    cfg->input_addr = input_addr;
    cfg->weight_addr = weight_addr;
    cfg->im2col_addr = 0u;
    cfg->output_addr = output_addr;
    cfg->input_h = H;
    cfg->input_w = W;
    cfg->input_c = input_c;
    cfg->output_h = H;
    cfg->output_w = W;
    cfg->kernel_h = KERNEL;
    cfg->kernel_w = KERNEL;
    cfg->stride_h = 1u;
    cfg->stride_w = 1u;
    cfg->pad_h = 1u;
    cfg->pad_w = 1u;
    cfg->dilation_h = 1u;
    cfg->dilation_w = 1u;
    cfg->input_c_stride = C32;
    cfg->input_row_stride_bytes = W * C32;
    cfg->input_c_base = 0u;
    cfg->accumulate = 0u;
}

static void publish_stats(uint32_t l2_addr, const npu_conv2d_packed_stats_t *stats) {
    spatz_rt_dma_1d(l2_addr, (uint32_t)stats, sizeof(*stats));
    spatz_rt_dma_wait_all();
}

static void run_split_path(npu_conv2d_spatial_tile_t *tiles,
                           uint32_t tile_count,
                           npu_conv2d_packed_stats_t *total) {
    clear_stats(total);

    for (uint32_t ocg = 0u; ocg < GROUPS; ocg++) {
        uint32_t out_group_addr = T_OUT_SPLIT + (ocg * ROWS * C32);
        uint32_t oc_weight_base = T_WEIGHT_SPLIT + (ocg * GROUPS * CHUNK_WEIGHT_BYTES);

        for (uint32_t icg = 0u; icg < GROUPS; icg++) {
            npu_conv2d_packed_cfg_t cfg = {0};
            npu_conv2d_packed_stats_t stats;
            uint32_t input_group_addr = T_INPUT + (icg * ROWS * C32);
            uint32_t weight_addr = oc_weight_base + (icg * CHUNK_WEIGHT_BYTES);
            uint32_t is_last = (icg + 1u) == GROUPS;

            init_cfg(&cfg, input_group_addr, weight_addr,
                     is_last ? out_group_addr : T_PSUM, C32);
            cfg.accumulate = icg != 0u;

            if (!is_last) {
                uint32_t status = npu_conv2d_packed_run_oc32_linebuf_tiles(&cfg,
                                                                           tiles,
                                                                           tile_count,
                                                                           4u,
                                                                           &stats);
                if (status != NPU_CONV2D_PACKED_OK) {
                    spatz_rt_fail_at(0x3100u, ocg * GROUPS + icg,
                                     (int32_t)status, NPU_CONV2D_PACKED_OK);
                }
                add_stats(total, &stats);
            } else {
                init_requant();
                for (uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++) {
                    const npu_conv2d_spatial_tile_t *tile = &tiles[tile_idx];
                    uint32_t tile_output_addr =
                        out_group_addr + ((tile->oh_base * W + tile->ow_base) * C32);
                    uint32_t status =
                        npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant(&cfg,
                                                                                   tile,
                                                                                   T_PSUM,
                                                                                   tile_output_addr,
                                                                                   &stats);
                    if (status != NPU_CONV2D_PACKED_OK) {
                        spatz_rt_fail_at(0x3200u, tile_idx,
                                         (int32_t)status, NPU_CONV2D_PACKED_OK);
                    }
                    add_stats(total, &stats);
                }
                systolic_requant_disable();
            }
        }
    }
}

static void run_fused_path(npu_conv2d_spatial_tile_t *tiles,
                           uint32_t tile_count,
                           npu_conv2d_packed_stats_t *total) {
    clear_stats(total);

    for (uint32_t ocg = 0u; ocg < GROUPS; ocg++) {
        npu_conv2d_packed_cfg_t cfg = {0};
        npu_conv2d_packed_stats_t stats;
        uint32_t out_group_addr = T_OUT_FUSED + (ocg * ROWS * C32);
        uint32_t oc_weight_base = T_WEIGHT_FUSED + (ocg * GROUPS * CHUNK_WEIGHT_BYTES);

        init_cfg(&cfg, T_INPUT, oc_weight_base, out_group_addr, IC);
        init_requant();
        uint32_t status = npu_conv2d_packed_run_oc32_linebuf_tiles_requant(&cfg,
                                                                           tiles,
                                                                           tile_count,
                                                                           T_PSUM,
                                                                           &stats);
        systolic_requant_disable();
        if (status != NPU_CONV2D_PACKED_OK) {
            spatz_rt_fail_at(0x3300u, ocg, (int32_t)status, NPU_CONV2D_PACKED_OK);
        }
        add_stats(total, &stats);
    }
}

int main(void) {
    npu_conv2d_spatial_tile_t tiles[8];
    uint32_t tile_count;
    npu_conv2d_packed_stats_t split_stats;
    npu_conv2d_packed_stats_t fused_stats;
    volatile uint32_t *mode_ptr = (volatile uint32_t *)T_MODE;
    uint32_t mode;

    spatz_rt_init();
    spatz_rt_set_phase(1u, 0u);

    spatz_rt_dma_1d(T_INPUT, L2_INPUT, INPUT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_WEIGHT_SPLIT, L2_WEIGHT_SPLIT, WEIGHT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_WEIGHT_FUSED, L2_WEIGHT_FUSED, WEIGHT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_MODE, L2_MODE, sizeof(uint32_t));
    spatz_rt_dma_wait_all();
    mode = *mode_ptr;

    build_tiles(tiles, &tile_count);

    if (mode == 0u) {
        run_split_path(tiles, tile_count, &split_stats);
        spatz_rt_dma_1d(L2_OUT_SPLIT, T_OUT_SPLIT, OUTPUT_BYTES);
        spatz_rt_dma_wait_all();
        publish_stats(L2_STATS_SPLIT, &split_stats);
    } else {
        run_fused_path(tiles, tile_count, &fused_stats);
        spatz_rt_dma_1d(L2_OUT_FUSED, T_OUT_FUSED, OUTPUT_BYTES);
        spatz_rt_dma_wait_all();
        publish_stats(L2_STATS_FUSED, &fused_stats);
    }

    spatz_rt_pass();
    return 0;
}
