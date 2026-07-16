#include "conv2d_packed.h"
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "spatz_rt.h"

/*
 * Scenario: P0/P1 packed Conv2D software scheduler benchmark.
 * Target: keep Conv2D performance work in software/iDMA/Spatz-prepared
 * Mx32 buffers and measure cycle cost before adding new Conv2D hardware.
 */
#define L2_CONV1_INPUT   0x80000000u
#define L2_CONV1_WEIGHT  0x80002000u
#define L2_CONV1_OUT     0x80010000u
#define L2_CONV1_STATS   0x80018000u

#define L2_CONV3_INPUT   0x80020000u
#define L2_CONV3_WEIGHT  0x80022000u
#define L2_CONV3_OUT     0x80030000u
#define L2_CONV3_STATS   0x80038000u

#define L2_CONV1_C32_INPUT  0x80040000u
#define L2_CONV1_C32_WEIGHT 0x80044000u
#define L2_CONV1_C32_OUT    0x80050000u
#define L2_CONV1_C32_STATS  0x80058000u

#define L2_CONV1_C64_INPUT  0x80060000u
#define L2_CONV1_C64_WEIGHT 0x80068000u
#define L2_CONV1_C64_OUT    0x80070000u
#define L2_CONV1_C64_STATS  0x80078000u
#define L2_CONV_PERF_CONFIG 0x8007F000u

#define L2_P3_BASE          0x80080000u
#define P3_CASE_STRIDE      0x00040000u
#define P3_C120_INPUT_ADDR  0x81000000u
#define P3_C120_WEIGHT_ADDR 0x81110000u
#define P3_C120_OUT_ADDR    0x81120000u
#define P3_C120_STATS_ADDR  0x81240000u
#define P3_C120_C32B_INPUT_ADDR  0x81300000u
#define P3_C120_C32B_WEIGHT_ADDR 0x81410000u
#define P3_C120_C32B_OUT_ADDR    0x81420000u
#define P3_C120_C32B_STATS_ADDR  0x81540000u
#define P3_C32T_INPUT_ADDR       0x81600000u
#define P3_C32T_WEIGHT_ADDR      0x81610000u
#define P3_C32T_OUT_ADDR         0x81620000u
#define P3_C32T_STATS_ADDR       0x81630000u
#define P3_INPUT_ADDR(id)   (((id) == 20u) ? P3_C120_INPUT_ADDR : (((id) == 22u) ? P3_C120_C32B_INPUT_ADDR : (((id) == 23u) ? P3_C32T_INPUT_ADDR : (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x0000u))))
#define P3_WEIGHT_ADDR(id)  (((id) == 20u) ? P3_C120_WEIGHT_ADDR : (((id) == 22u) ? P3_C120_C32B_WEIGHT_ADDR : (((id) == 23u) ? P3_C32T_WEIGHT_ADDR : (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x3000u))))
#define P3_OUT_ADDR(id)     (((id) == 20u) ? P3_C120_OUT_ADDR : (((id) == 22u) ? P3_C120_C32B_OUT_ADDR : (((id) == 23u) ? P3_C32T_OUT_ADDR : (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x10000u))))
#define P3_STATS_ADDR(id)   (((id) == 20u) ? P3_C120_STATS_ADDR : (((id) == 22u) ? P3_C120_C32B_STATS_ADDR : (((id) == 23u) ? P3_C32T_STATS_ADDR : (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x3E000u))))

#define T_INPUT          0x10100000u
#define T_WEIGHT         0x10102000u
#define T_IM2COL         0x10108000u
#define T_OUTPUT         0x10140000u
#define T_OUTPUT_OC1     0x10150000u
#define T_STATS          0x10178000u
#define T_CONFIG         0x1017F000u
#define T_C120_INPUT     0x10100000u
#define T_C120_WEIGHT    0x10110000u
#define T_C120_OUTPUT    0x10140000u
#define T_C120_PSUM      0x10150000u

#define CONV1_H          4u
#define CONV1_W          5u
#define CONV1_C          33u
#define CONV1_ROWS       (CONV1_H * CONV1_W)
#define CONV1_INPUT_BYTES  (CONV1_ROWS * CONV1_C)
#define CONV1_WEIGHT_BYTES (2u * 32u * 32u)
#define CONV1_OUT_BYTES    (CONV1_ROWS * 32u * 4u)

#define CONV3_H          5u
#define CONV3_W          5u
#define CONV3_C          3u
#define CONV3_ROWS       (CONV3_H * CONV3_W)
#define CONV3_INPUT_BYTES  (CONV3_ROWS * CONV3_C)
#define CONV3_WEIGHT_BYTES (32u * 32u)
#define CONV3_OUT_BYTES    (CONV3_ROWS * 32u * 4u)

#define P3_H             4u
#define P3_W             4u
#define P3_ROWS          (P3_H * P3_W)
#define P3_C120_H        16u
#define P3_C120_W        16u
#define P3_C120_TILE_OH  16u
#define P3_C32T_H        32u
#define P3_C32T_W        32u
#define P3_C32T_TILE     16u
#define P3_C32           32u
#define P3_C64           64u
#define P3_C32_INPUT_BYTES  (P3_ROWS * P3_C32)
#define P3_C64_INPUT_BYTES  (P3_ROWS * P3_C64)
#define P3_C32_WEIGHT_BYTES (1u * 32u * 32u)
#define P3_C64_WEIGHT_BYTES (2u * 32u * 32u)
#define P3_OUT_BYTES        (P3_ROWS * 32u * 4u)

#define P3_CASE_IC1       0u
#define P3_CASE_IC3       1u
#define P3_CASE_IC31      2u
#define P3_CASE_OC64      3u
#define P3_CASE_3X3_P0_C32 4u
#define P3_CASE_3X3_P1_C32 5u
#define P3_CASE_5X5_P2_C3  6u
#define P3_CASE_7X7_P3_C1  7u
#define P3_CASE_1X3_C3     8u
#define P3_CASE_3X1_C3     9u
#define P3_CASE_1X5_C3     10u
#define P3_CASE_5X1_C3     11u
#define P3_CASE_3X3_S2_C3  12u
#define P3_CASE_3X3_C1     13u
#define P3_CASE_3X3_C5     14u
#define P3_CASE_REQUANT    15u
#define P3_CASE_YOLO_RGB_TCDM_SPATZ 16u
#define P3_CASE_LINEBUF_3X3_C3 17u
#define P3_CASE_LINEBUF_KGEN_3X3_C32 18u
#define P3_CASE_LINEBUF_KGEN_3X3_C96 19u
#define P3_CASE_LINEBUF_3X3_C120 20u
#define P3_CASE_LINEBUF_KGEN_3X3_C65 21u
#define P3_CASE_LINEBUF_3X3_C120_C32B 22u
#define P3_CASE_LINEBUF_3X3_C32_TILED_REQUANT 23u

#define YOLO_RGB_H          64u
#define YOLO_RGB_W          64u
#define YOLO_RGB_C          3u
#define YOLO_RGB_OH         32u
#define YOLO_RGB_OW         32u
#define YOLO_RGB_TILE_OH    8u
#define YOLO_RGB_INPUT_BYTES (YOLO_RGB_H * YOLO_RGB_W * YOLO_RGB_C)
#define YOLO_RGB_WEIGHT_BYTES (32u * 32u)
#define YOLO_RGB_TILE_OUT_BYTES (YOLO_RGB_TILE_OH * YOLO_RGB_OW * 32u * 4u)

#define CONV_PERF_GROUP_ALL       0
#define CONV_PERF_GROUP_POINTWISE 1
#define CONV_PERF_GROUP_KERNELS   2
#define CONV_PERF_GROUP_REQUANT   3
#define CONV_PERF_GROUP_YOLO      4
#define CONV_PERF_CONFIG_MAGIC    0x43504647u

static uint32_t conv_perf_group_q;
static uint32_t conv_perf_case_q;
static uint32_t conv_perf_case_valid_q;

static void load_conv_perf_config(void) {
    volatile uint32_t *cfg = (volatile uint32_t *)T_CONFIG;

    spatz_rt_dma_1d(T_CONFIG, L2_CONV_PERF_CONFIG, 3u * sizeof(uint32_t));
    spatz_rt_dma_wait_all();

    if (cfg[0] == CONV_PERF_CONFIG_MAGIC) {
        conv_perf_group_q = cfg[1];
        conv_perf_case_q = cfg[2];
        conv_perf_case_valid_q = (cfg[2] != 0xFFFFFFFFu);
    }
}

static uint32_t should_run_legacy(void) {
    if (conv_perf_case_valid_q) {
        return 0u;
    }
    return (conv_perf_group_q == CONV_PERF_GROUP_ALL) || (conv_perf_group_q == CONV_PERF_GROUP_POINTWISE);
}

static uint32_t should_run_case(uint32_t case_id) {
    if (conv_perf_case_valid_q) {
        return case_id == conv_perf_case_q;
    }
    if (conv_perf_group_q == CONV_PERF_GROUP_ALL) {
        return 1u;
    }
    if (conv_perf_group_q == CONV_PERF_GROUP_POINTWISE) {
        return case_id <= P3_CASE_OC64;
    }
    if (conv_perf_group_q == CONV_PERF_GROUP_KERNELS) {
        return ((case_id >= P3_CASE_3X3_P0_C32) && (case_id <= P3_CASE_3X3_C5)) ||
               (case_id == P3_CASE_LINEBUF_KGEN_3X3_C96) ||
               (case_id == P3_CASE_LINEBUF_3X3_C120) ||
               (case_id == P3_CASE_LINEBUF_KGEN_3X3_C65) ||
               (case_id == P3_CASE_LINEBUF_3X3_C32_TILED_REQUANT);
    }
    if (conv_perf_group_q == CONV_PERF_GROUP_REQUANT) {
        return case_id == P3_CASE_REQUANT;
    }
    if (conv_perf_group_q == CONV_PERF_GROUP_YOLO) {
        return case_id == P3_CASE_YOLO_RGB_TCDM_SPATZ;
    }
    return 0u;
}

static void publish_stats(uint32_t l2_addr, const npu_conv2d_packed_stats_t *stats) {
    spatz_rt_memcpy((void *)T_STATS, stats, sizeof(*stats));
    spatz_rt_dma_1d(l2_addr, T_STATS, sizeof(*stats));
    spatz_rt_dma_wait_all();
}

static void publish_stats_pair(uint32_t l2_addr,
                               const npu_conv2d_packed_stats_t *stats0,
                               const npu_conv2d_packed_stats_t *stats1) {
    spatz_rt_memcpy((void *)T_STATS, stats0, sizeof(*stats0));
    spatz_rt_memcpy((void *)(T_STATS + sizeof(*stats0)), stats1, sizeof(*stats1));
    spatz_rt_dma_1d(l2_addr, T_STATS, sizeof(*stats0) + sizeof(*stats1));
    spatz_rt_dma_wait_all();
}

static void accumulate_stats(npu_conv2d_packed_stats_t *accum,
                             const npu_conv2d_packed_stats_t *tile) {
    accum->rows += tile->rows;
    accum->k_tiles += tile->k_tiles;
    accum->prepare_cycles += tile->prepare_cycles;
    accum->gemm_cycles += tile->gemm_cycles;
    accum->total_cycles += tile->total_cycles;
    accum->last_prepare_cycles = tile->last_prepare_cycles;
    accum->last_gemm_cycles = tile->last_gemm_cycles;
    accum->status = tile->status;
    accum->prepare_idma_tiles += tile->prepare_idma_tiles;
    accum->prepare_idma_transfers += tile->prepare_idma_transfers;
    accum->prepare_spatz_tiles += tile->prepare_spatz_tiles;
    accum->prepare_scalar_tiles += tile->prepare_scalar_tiles;
}

static void reset_stats(npu_conv2d_packed_stats_t *stats) {
    stats->rows = 0u;
    stats->k_tiles = 0u;
    stats->prepare_cycles = 0u;
    stats->gemm_cycles = 0u;
    stats->total_cycles = 0u;
    stats->last_prepare_cycles = 0u;
    stats->last_gemm_cycles = 0u;
    stats->status = 0u;
    stats->prepare_idma_tiles = 0u;
    stats->prepare_idma_transfers = 0u;
    stats->prepare_spatz_tiles = 0u;
    stats->prepare_scalar_tiles = 0u;
}

static void init_cfg(npu_conv2d_packed_cfg_t *cfg,
                     uint32_t input_addr,
                     uint32_t weight_addr,
                     uint32_t output_addr,
                     uint32_t input_h,
                     uint32_t input_w,
                     uint32_t input_c,
                     uint32_t output_h,
                     uint32_t output_w,
                     uint32_t kernel_h,
                     uint32_t kernel_w,
                     uint32_t stride_h,
                     uint32_t stride_w,
                     uint32_t pad_h,
                     uint32_t pad_w) {
    cfg->input_addr = input_addr;
    cfg->weight_addr = weight_addr;
    cfg->im2col_addr = T_IM2COL;
    cfg->output_addr = output_addr;
    cfg->input_h = input_h;
    cfg->input_w = input_w;
    cfg->input_c = input_c;
    cfg->output_h = output_h;
    cfg->output_w = output_w;
    cfg->kernel_h = kernel_h;
    cfg->kernel_w = kernel_w;
    cfg->stride_h = stride_h;
    cfg->stride_w = stride_w;
    cfg->pad_h = pad_h;
    cfg->pad_w = pad_w;
    cfg->dilation_h = 1u;
    cfg->dilation_w = 1u;
    cfg->input_c_stride = input_c;
    cfg->input_row_stride_bytes = 0u;
    cfg->input_c_base = 0u;
    cfg->accumulate = 0u;
}

static uint32_t k_tiles_for(uint32_t input_c, uint32_t kernel_h, uint32_t kernel_w) {
    uint32_t k_total = input_c * kernel_h * kernel_w;
    return (k_total + 31u) / 32u;
}

static void run_oc32_case(uint32_t case_id,
                          uint32_t input_in_l2,
                          uint32_t input_h,
                          uint32_t input_w,
                          uint32_t input_c,
                          uint32_t output_h,
                          uint32_t output_w,
                          uint32_t kernel_h,
                          uint32_t kernel_w,
                          uint32_t stride_h,
                          uint32_t stride_w,
                          uint32_t pad_h,
                          uint32_t pad_w,
                          uint32_t fail_code) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;
    uint32_t rows = output_h * output_w;
    uint32_t input_bytes = input_h * input_w * input_c;
    uint32_t weight_bytes = k_tiles_for(input_c, kernel_h, kernel_w) * 32u * 32u;
    uint32_t out_bytes = rows * 32u * 4u;
    uint32_t input_addr = P3_INPUT_ADDR(case_id);

    if (!input_in_l2) {
        spatz_rt_dma_1d(T_INPUT, P3_INPUT_ADDR(case_id), input_bytes);
        spatz_rt_dma_wait_all();
        input_addr = T_INPUT;
    }

    spatz_rt_dma_1d(T_WEIGHT, P3_WEIGHT_ADDR(case_id), weight_bytes);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             input_addr,
             T_WEIGHT,
             T_OUTPUT,
             input_h,
             input_w,
             input_c,
             output_h,
             output_w,
             kernel_h,
             kernel_w,
             stride_h,
             stride_w,
             pad_h,
             pad_w);

    uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(fail_code, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), T_OUTPUT, out_bytes);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &stats);
}

static void merge_split_stats(npu_conv2d_packed_stats_t *total,
                              const npu_conv2d_packed_stats_t *tail) {
    total->k_tiles += tail->k_tiles;
    total->prepare_cycles += tail->prepare_cycles;
    total->gemm_cycles += tail->gemm_cycles;
    total->total_cycles += tail->total_cycles;
    total->last_prepare_cycles = tail->last_prepare_cycles;
    total->last_gemm_cycles = tail->last_gemm_cycles;
    total->status = tail->status;
    total->prepare_idma_tiles += tail->prepare_idma_tiles;
    total->prepare_idma_transfers += tail->prepare_idma_transfers;
    total->prepare_spatz_tiles += tail->prepare_spatz_tiles;
    total->prepare_scalar_tiles += tail->prepare_scalar_tiles;
}

static void copy_stats(npu_conv2d_packed_stats_t *dst,
                       const npu_conv2d_packed_stats_t *src) {
    dst->rows = src->rows;
    dst->k_tiles = src->k_tiles;
    dst->prepare_cycles = src->prepare_cycles;
    dst->gemm_cycles = src->gemm_cycles;
    dst->total_cycles = src->total_cycles;
    dst->last_prepare_cycles = src->last_prepare_cycles;
    dst->last_gemm_cycles = src->last_gemm_cycles;
    dst->status = src->status;
    dst->prepare_idma_tiles = src->prepare_idma_tiles;
    dst->prepare_idma_transfers = src->prepare_idma_transfers;
    dst->prepare_spatz_tiles = src->prepare_spatz_tiles;
    dst->prepare_scalar_tiles = src->prepare_scalar_tiles;
}

static void run_oc32_case_channel_slices(uint32_t case_id,
                                         uint32_t input_in_l2,
                                         uint32_t input_h,
                                         uint32_t input_w,
                                         uint32_t input_c_total,
                                         uint32_t output_h,
                                         uint32_t output_w,
                                         uint32_t kernel_h,
                                         uint32_t kernel_w,
                                         uint32_t stride_h,
                                         uint32_t stride_w,
                                         uint32_t pad_h,
                                         uint32_t pad_w,
                                         uint32_t main_c,
                                         uint32_t tail_c,
                                         uint32_t fail_code) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t total_stats;
    uint32_t rows = output_h * output_w;
    uint32_t input_bytes = input_h * input_w * input_c_total;
    uint32_t output_bytes = rows * 32u * 4u;
    uint32_t input_addr = P3_INPUT_ADDR(case_id);
    uint32_t t_input = T_INPUT;
    uint32_t t_weight = T_WEIGHT;
    uint32_t t_output = T_OUTPUT;
    uint32_t total_weight_bytes = 0u;
    uint32_t weight_offset = 0u;

    if (case_id == 20u) {
        t_input = T_C120_INPUT;
        t_weight = T_C120_WEIGHT;
        t_output = T_C120_OUTPUT;
    }

    reset_stats(&total_stats);

    total_weight_bytes = k_tiles_for(main_c, kernel_h, kernel_w) * 32u * 32u;
    if (tail_c != 0u) {
        total_weight_bytes += k_tiles_for(tail_c, kernel_h, kernel_w) * 32u * 32u;
    }

    if (!input_in_l2) {
        spatz_rt_dma_1d(t_input, P3_INPUT_ADDR(case_id), input_bytes);
        input_addr = t_input;
    }

    spatz_rt_dma_1d(t_weight, P3_WEIGHT_ADDR(case_id), total_weight_bytes);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             input_addr,
             t_weight,
             t_output,
             input_h,
             input_w,
             input_c_total,
             output_h,
             output_w,
             kernel_h,
             kernel_w,
             stride_h,
             stride_w,
             pad_h,
             pad_w);
    cfg.input_c_stride = input_c_total;

    for (uint32_t idx = 0; idx < 2u; idx++) {
        npu_conv2d_packed_stats_t slice_stats;
        uint32_t slice_c_base = (idx == 0u) ? 0u : main_c;
        uint32_t slice_c_count = (idx == 0u) ? main_c : tail_c;
        uint32_t status;

        if (slice_c_count == 0u) {
            break;
        }

        cfg.weight_addr = t_weight + weight_offset;
        cfg.input_c = slice_c_count;
        cfg.input_c_base = slice_c_base;
        cfg.accumulate = (idx == 0u) ? 0u : 1u;

        status = npu_conv2d_packed_run_oc32(&cfg, &slice_stats);
        if (status != NPU_CONV2D_PACKED_OK) {
            spatz_rt_fail_at(fail_code, idx, (int32_t)status, NPU_CONV2D_PACKED_OK);
        }

        if (idx == 0u) {
            copy_stats(&total_stats, &slice_stats);
        } else {
            merge_split_stats(&total_stats, &slice_stats);
        }
        weight_offset += k_tiles_for(slice_c_count, kernel_h, kernel_w) * 32u * 32u;
    }

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), t_output, output_bytes);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &total_stats);
}

static void run_oc32_case_c32_blocked_slices(uint32_t case_id,
                                             uint32_t input_h,
                                             uint32_t input_w,
                                             uint32_t input_c_total,
                                             uint32_t output_h,
                                             uint32_t output_w,
                                             uint32_t kernel_h,
                                             uint32_t kernel_w,
                                             uint32_t stride_h,
                                             uint32_t stride_w,
                                             uint32_t pad_h,
                                             uint32_t pad_w,
                                             uint32_t fail_code) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t total_stats;
    uint32_t rows = output_h * output_w;
    uint32_t c_blocks = (input_c_total + 31u) / 32u;
    uint32_t c_block_bytes = input_h * input_w * 32u;
    uint32_t input_bytes = c_blocks * c_block_bytes;
    uint32_t output_bytes = rows * 32u * 4u;
    uint32_t weight_offset = 0u;

    reset_stats(&total_stats);

    spatz_rt_dma_1d(T_C120_INPUT, P3_INPUT_ADDR(case_id), input_bytes);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_C120_WEIGHT,
                    P3_WEIGHT_ADDR(case_id),
                    k_tiles_for(input_c_total, kernel_h, kernel_w) * 32u * 32u);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             T_C120_INPUT,
             T_C120_WEIGHT,
             T_C120_OUTPUT,
             input_h,
             input_w,
             32u,
             output_h,
             output_w,
             kernel_h,
             kernel_w,
             stride_h,
             stride_w,
             pad_h,
             pad_w);
    cfg.input_c_stride = 32u;
    cfg.input_c_base = 0u;

    for (uint32_t c_base = 0u; c_base < input_c_total; c_base += 32u) {
        npu_conv2d_packed_stats_t slice_stats;
        uint32_t block_idx = c_base / 32u;
        uint32_t c_count = input_c_total - c_base;
        uint32_t status;

        if (c_count > 32u) {
            c_count = 32u;
        }

        cfg.input_addr = T_C120_INPUT + (block_idx * c_block_bytes);
        cfg.weight_addr = T_C120_WEIGHT + weight_offset;
        cfg.input_c = c_count;
        cfg.accumulate = (c_base == 0u) ? 0u : 1u;

        status = npu_conv2d_packed_run_oc32(&cfg, &slice_stats);
        if (status != NPU_CONV2D_PACKED_OK) {
            spatz_rt_fail_at(fail_code, c_base, (int32_t)status, NPU_CONV2D_PACKED_OK);
        }

        if (c_base == 0u) {
            copy_stats(&total_stats, &slice_stats);
        } else {
            merge_split_stats(&total_stats, &slice_stats);
        }
        weight_offset += k_tiles_for(c_count, kernel_h, kernel_w) * 32u * 32u;
    }

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), T_C120_OUTPUT, output_bytes);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &total_stats);
}

static void copy_oc32_to_oc64_l2(uint32_t l2_addr, uint32_t src_addr, uint32_t rows, uint32_t oc_base) {
    int tx = idma_L1ToL2_2d(src_addr, l2_addr + (oc_base * 4u), 32u * 4u, 32u * 4u, 64u * 4u, rows);
    if (!idma_mm_wait_for_completion(IDMA_DIR_L1_TO_L2, (uint32_t)tx)) {
        spatz_rt_fail_at(0xC0D0u, oc_base, tx, 1);
    }
}

static void run_oc64_case(uint32_t case_id) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats0;
    npu_conv2d_packed_stats_t stats1;
    uint32_t rows = P3_ROWS;
    uint32_t k_tiles = k_tiles_for(33u, 1u, 1u);
    uint32_t weight_tile_bytes = k_tiles * 32u * 32u;

    spatz_rt_dma_1d(T_WEIGHT, P3_WEIGHT_ADDR(case_id), weight_tile_bytes * 2u);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             P3_INPUT_ADDR(case_id),
             T_WEIGHT,
             T_OUTPUT,
             P3_H,
             P3_W,
             33u,
             P3_H,
             P3_W,
             1u,
             1u,
             1u,
             1u,
             0u,
             0u);

    uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &stats0);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC640u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }
    copy_oc32_to_oc64_l2(P3_OUT_ADDR(case_id), T_OUTPUT, rows, 0u);

    cfg.weight_addr = T_WEIGHT + weight_tile_bytes;
    cfg.output_addr = T_OUTPUT_OC1;
    status = npu_conv2d_packed_run_oc32(&cfg, &stats1);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC641u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }
    copy_oc32_to_oc64_l2(P3_OUT_ADDR(case_id), T_OUTPUT_OC1, rows, 32u);
    publish_stats_pair(P3_STATS_ADDR(case_id), &stats0, &stats1);
}

static void init_qparams(void) {
    static int32_t bias[32];
    static int32_t multiplier[32];
    static uint8_t shift[32];
    static int32_t zero_point[32];

    for (uint32_t ch = 0; ch < 32u; ch++) {
        bias[ch] = ((int32_t)ch - 16) * 3;
        multiplier[ch] = (int32_t)((ch % 5u) + 1u);
        shift[ch] = (uint8_t)(ch % 4u);
        zero_point[ch] = (int32_t)(ch % 7u) - 3;
    }

    systolic_requant_config_per_channel(bias, multiplier, shift, zero_point, -50, 60);
}

static void run_requant_case(uint32_t case_id) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;
    uint32_t rows = P3_ROWS;
    uint32_t weight_bytes = k_tiles_for(64u, 1u, 1u) * 32u * 32u;

    spatz_rt_dma_1d(T_WEIGHT, P3_WEIGHT_ADDR(case_id), weight_bytes);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             P3_INPUT_ADDR(case_id),
             T_WEIGHT,
             T_OUTPUT,
             P3_H,
             P3_W,
             64u,
             P3_H,
             P3_W,
             1u,
             1u,
             1u,
             1u,
             0u,
             0u);

    init_qparams();
    uint32_t status = npu_conv2d_packed_run_oc32_requant(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC0F0u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }
    systolic_requant_disable();

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), T_OUTPUT, rows * 32u);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &stats);
}

static void run_linebuf_3x3_c32_tiled_requant_case(uint32_t case_id) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;
    npu_conv2d_spatial_tile_t tiles[4];
    uint32_t input_bytes = P3_C32T_H * P3_C32T_W * P3_C32;
    uint32_t weight_bytes = 9u * 32u * 32u;
    uint32_t out_bytes = P3_C32T_H * P3_C32T_W * 32u;
    uint32_t tile_idx = 0u;

    spatz_rt_dma_1d(T_C120_INPUT, P3_INPUT_ADDR(case_id), input_bytes);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_C120_WEIGHT, P3_WEIGHT_ADDR(case_id), weight_bytes);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             T_C120_INPUT,
             T_C120_WEIGHT,
             T_C120_OUTPUT,
             P3_C32T_H,
             P3_C32T_W,
             P3_C32,
             P3_C32T_H,
             P3_C32T_W,
             3u,
             3u,
             1u,
             1u,
             1u,
             1u);

    for (uint32_t oh = 0u; oh < P3_C32T_H; oh += P3_C32T_TILE) {
        for (uint32_t ow = 0u; ow < P3_C32T_W; ow += P3_C32T_TILE) {
            tiles[tile_idx].oh_base = oh;
            tiles[tile_idx].ow_base = ow;
            tiles[tile_idx].tile_oh = P3_C32T_TILE;
            tiles[tile_idx].tile_ow = P3_C32T_TILE;
            tile_idx++;
        }
    }

    init_qparams();
    uint32_t status = npu_conv2d_packed_run_oc32_linebuf_tiles_requant(&cfg,
                                                                       tiles,
                                                                       tile_idx,
                                                                       T_C120_PSUM,
                                                                       &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC9E6u, tile_idx, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }
    systolic_requant_disable();

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), T_C120_OUTPUT, out_bytes);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &stats);
}

static void run_yolo_rgb_tcdm_spatz_case(uint32_t case_id) {
    npu_conv2d_packed_stats_t total_stats;
    uint32_t weight_addr = P3_WEIGHT_ADDR(case_id);
    uint32_t input_addr = P3_INPUT_ADDR(case_id);
    uint32_t out_addr = P3_OUT_ADDR(case_id);

    spatz_rt_memset(&total_stats, 0, sizeof(total_stats));
    spatz_rt_dma_1d(T_WEIGHT, weight_addr, YOLO_RGB_WEIGHT_BYTES);
    spatz_rt_dma_wait_all();

    for (uint32_t oh_start = 0; oh_start < YOLO_RGB_OH; oh_start += YOLO_RGB_TILE_OH) {
        uint32_t tile_oh = YOLO_RGB_TILE_OH;
        if ((oh_start + tile_oh) > YOLO_RGB_OH) {
            tile_oh = YOLO_RGB_OH - oh_start;
        }

        uint32_t first_ih = 0u;
        if ((oh_start * 2u) > 0u) {
            first_ih = (oh_start * 2u) - 1u;
        }

        uint32_t last_oh = oh_start + tile_oh - 1u;
        uint32_t last_ih = (last_oh * 2u) + 1u;
        if (last_ih >= YOLO_RGB_H) {
            last_ih = YOLO_RGB_H - 1u;
        }

        uint32_t local_input_h = last_ih - first_ih + 1u;
        uint32_t local_pad_h = 1u + first_ih - (oh_start * 2u);
        uint32_t input_offset = first_ih * YOLO_RGB_W * YOLO_RGB_C;
        uint32_t input_bytes = local_input_h * YOLO_RGB_W * YOLO_RGB_C;
        npu_conv2d_packed_cfg_t cfg = {0};
        npu_conv2d_packed_stats_t tile_stats;

        spatz_rt_dma_1d(T_INPUT, input_addr + input_offset, input_bytes);
        spatz_rt_dma_wait_all();

        init_cfg(&cfg,
                 T_INPUT,
                 T_WEIGHT,
                 T_OUTPUT,
                 local_input_h,
                 YOLO_RGB_W,
                 YOLO_RGB_C,
                 tile_oh,
                 YOLO_RGB_OW,
                 3u,
                 3u,
                 2u,
                 2u,
                 local_pad_h,
                 1u);

        uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &tile_stats);
        if (status != NPU_CONV2D_PACKED_OK) {
            spatz_rt_fail_at(0xC800u, oh_start, (int32_t)status, NPU_CONV2D_PACKED_OK);
        }

        spatz_rt_dma_1d(out_addr + (oh_start * YOLO_RGB_OW * 32u * 4u),
                        T_OUTPUT,
                        tile_oh * YOLO_RGB_OW * 32u * 4u);
        spatz_rt_dma_wait_all();
        accumulate_stats(&total_stats, &tile_stats);
    }

    publish_stats(P3_STATS_ADDR(case_id), &total_stats);
}

static void run_linebuf_3x3_c3_case(uint32_t case_id) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;
    uint32_t input_bytes = P3_H * P3_W * 3u;
    uint32_t weight_bytes = 32u * 32u;

    spatz_rt_dma_1d(T_INPUT, P3_INPUT_ADDR(case_id), input_bytes);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_WEIGHT, P3_WEIGHT_ADDR(case_id), weight_bytes);
    spatz_rt_dma_wait_all();

    init_cfg(&cfg,
             T_INPUT,
             T_WEIGHT,
             T_OUTPUT,
             P3_H,
             P3_W,
             3u,
             P3_H,
             P3_W,
             3u,
             3u,
             1u,
             1u,
             1u,
             1u);

    uint32_t status = npu_conv2d_packed_run_oc32_linebuf(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC1B0u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }

    spatz_rt_dma_1d(P3_OUT_ADDR(case_id), T_OUTPUT, P3_OUT_BYTES);
    spatz_rt_dma_wait_all();
    publish_stats(P3_STATS_ADDR(case_id), &stats);
}

static void run_conv1x1_k33(void) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;

    cfg.input_addr = L2_CONV1_INPUT;
    cfg.weight_addr = T_WEIGHT;
    cfg.im2col_addr = T_IM2COL;
    cfg.output_addr = T_OUTPUT;
    cfg.input_h = CONV1_H;
    cfg.input_w = CONV1_W;
    cfg.input_c = CONV1_C;
    cfg.output_h = CONV1_H;
    cfg.output_w = CONV1_W;
    cfg.kernel_h = 1u;
    cfg.kernel_w = 1u;
    cfg.stride_h = 1u;
    cfg.stride_w = 1u;
    cfg.pad_h = 0u;
    cfg.pad_w = 0u;
    cfg.dilation_h = 1u;
    cfg.dilation_w = 1u;
    cfg.input_c_stride = CONV1_C;
    cfg.input_c_base = 0u;
    cfg.accumulate = 0u;

    spatz_rt_dma_1d(T_WEIGHT, L2_CONV1_WEIGHT, CONV1_WEIGHT_BYTES);
    spatz_rt_dma_wait_all();

    uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC001u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }

    spatz_rt_dma_1d(L2_CONV1_OUT, T_OUTPUT, CONV1_OUT_BYTES);
    spatz_rt_dma_wait_all();
    publish_stats(L2_CONV1_STATS, &stats);
}

static void run_conv3x3_pad1_c3(void) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;

    cfg.input_addr = L2_CONV3_INPUT;
    cfg.weight_addr = T_WEIGHT;
    cfg.im2col_addr = T_IM2COL;
    cfg.output_addr = T_OUTPUT;
    cfg.input_h = CONV3_H;
    cfg.input_w = CONV3_W;
    cfg.input_c = CONV3_C;
    cfg.output_h = CONV3_H;
    cfg.output_w = CONV3_W;
    cfg.kernel_h = 3u;
    cfg.kernel_w = 3u;
    cfg.stride_h = 1u;
    cfg.stride_w = 1u;
    cfg.pad_h = 1u;
    cfg.pad_w = 1u;
    cfg.dilation_h = 1u;
    cfg.dilation_w = 1u;
    cfg.input_c_stride = CONV3_C;
    cfg.input_c_base = 0u;
    cfg.accumulate = 0u;

    spatz_rt_dma_1d(T_WEIGHT, L2_CONV3_WEIGHT, CONV3_WEIGHT_BYTES);
    spatz_rt_dma_wait_all();

    uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(0xC003u, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }

    spatz_rt_dma_1d(L2_CONV3_OUT, T_OUTPUT, CONV3_OUT_BYTES);
    spatz_rt_dma_wait_all();
    publish_stats(L2_CONV3_STATS, &stats);
}

static void run_conv1x1_p3(uint32_t input_addr,
                           uint32_t weight_addr,
                           uint32_t output_addr,
                           uint32_t stats_addr,
                           uint32_t input_c,
                           uint32_t weight_bytes,
                           uint32_t fail_code) {
    npu_conv2d_packed_cfg_t cfg = {0};
    npu_conv2d_packed_stats_t stats;

    cfg.input_addr = input_addr;
    cfg.weight_addr = T_WEIGHT;
    cfg.im2col_addr = T_IM2COL;
    cfg.output_addr = T_OUTPUT;
    cfg.input_h = P3_H;
    cfg.input_w = P3_W;
    cfg.input_c = input_c;
    cfg.output_h = P3_H;
    cfg.output_w = P3_W;
    cfg.kernel_h = 1u;
    cfg.kernel_w = 1u;
    cfg.stride_h = 1u;
    cfg.stride_w = 1u;
    cfg.pad_h = 0u;
    cfg.pad_w = 0u;
    cfg.dilation_h = 1u;
    cfg.dilation_w = 1u;
    cfg.input_c_stride = input_c;
    cfg.input_c_base = 0u;
    cfg.accumulate = 0u;

    spatz_rt_dma_1d(T_WEIGHT, weight_addr, weight_bytes);
    spatz_rt_dma_wait_all();

    uint32_t status = npu_conv2d_packed_run_oc32(&cfg, &stats);
    if (status != NPU_CONV2D_PACKED_OK) {
        spatz_rt_fail_at(fail_code, 0u, (int32_t)status, NPU_CONV2D_PACKED_OK);
    }

    spatz_rt_dma_1d(output_addr, T_OUTPUT, P3_OUT_BYTES);
    spatz_rt_dma_wait_all();
    publish_stats(stats_addr, &stats);
}

int main(void) {
    spatz_rt_init();
    load_conv_perf_config();

    if (should_run_legacy()) {
        spatz_rt_set_phase(1, 1);
        run_conv1x1_k33();
        spatz_rt_pass_step();

        spatz_rt_set_phase(2, 3);
        run_conv3x3_pad1_c3();
        spatz_rt_pass_step();

        spatz_rt_set_phase(3, 32);
        run_conv1x1_p3(L2_CONV1_C32_INPUT,
                       L2_CONV1_C32_WEIGHT,
                       L2_CONV1_C32_OUT,
                       L2_CONV1_C32_STATS,
                       P3_C32,
                       P3_C32_WEIGHT_BYTES,
                       0xC032u);
        spatz_rt_pass_step();

        spatz_rt_set_phase(4, 64);
        run_conv1x1_p3(L2_CONV1_C64_INPUT,
                       L2_CONV1_C64_WEIGHT,
                       L2_CONV1_C64_OUT,
                       L2_CONV1_C64_STATS,
                       P3_C64,
                       P3_C64_WEIGHT_BYTES,
                       0xC064u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_IC1)) {
        spatz_rt_set_phase(5, P3_CASE_IC1);
        run_oc32_case(P3_CASE_IC1, 1u, P3_H, P3_W, 1u, P3_H, P3_W, 1u, 1u, 1u, 1u, 0u, 0u, 0xC101u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_IC3)) {
        spatz_rt_set_phase(6, P3_CASE_IC3);
        run_oc32_case(P3_CASE_IC3, 1u, P3_H, P3_W, 3u, P3_H, P3_W, 1u, 1u, 1u, 1u, 0u, 0u, 0xC103u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_IC31)) {
        spatz_rt_set_phase(7, P3_CASE_IC31);
        run_oc32_case(P3_CASE_IC31, 1u, P3_H, P3_W, 31u, P3_H, P3_W, 1u, 1u, 1u, 1u, 0u, 0u, 0xC131u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_OC64)) {
        spatz_rt_set_phase(8, P3_CASE_OC64);
        run_oc64_case(P3_CASE_OC64);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X3_P0_C32)) {
        spatz_rt_set_phase(9, P3_CASE_3X3_P0_C32);
        run_oc32_case(P3_CASE_3X3_P0_C32, 1u, P3_H, P3_W, 32u, 2u, 2u, 3u, 3u, 1u, 1u, 0u, 0u, 0xC330u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X3_P1_C32)) {
        spatz_rt_set_phase(10, P3_CASE_3X3_P1_C32);
        run_oc32_case(P3_CASE_3X3_P1_C32, 1u, P3_H, P3_W, 32u, P3_H, P3_W, 3u, 3u, 1u, 1u, 1u, 1u, 0xC331u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_5X5_P2_C3)) {
        spatz_rt_set_phase(11, P3_CASE_5X5_P2_C3);
        run_oc32_case(P3_CASE_5X5_P2_C3, 0u, P3_H, P3_W, 3u, P3_H, P3_W, 5u, 5u, 1u, 1u, 2u, 2u, 0xC552u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_7X7_P3_C1)) {
        spatz_rt_set_phase(12, P3_CASE_7X7_P3_C1);
        run_oc32_case(P3_CASE_7X7_P3_C1, 0u, P3_H, P3_W, 1u, P3_H, P3_W, 7u, 7u, 1u, 1u, 3u, 3u, 0xC773u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_1X3_C3)) {
        spatz_rt_set_phase(13, P3_CASE_1X3_C3);
        run_oc32_case(P3_CASE_1X3_C3, 0u, P3_H, P3_W, 3u, P3_H, P3_W, 1u, 3u, 1u, 1u, 0u, 1u, 0xC013u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X1_C3)) {
        spatz_rt_set_phase(14, P3_CASE_3X1_C3);
        run_oc32_case(P3_CASE_3X1_C3, 0u, P3_H, P3_W, 3u, P3_H, P3_W, 3u, 1u, 1u, 1u, 1u, 0u, 0xC031u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_1X5_C3)) {
        spatz_rt_set_phase(15, P3_CASE_1X5_C3);
        run_oc32_case(P3_CASE_1X5_C3, 0u, P3_H, P3_W, 3u, P3_H, P3_W, 1u, 5u, 1u, 1u, 0u, 2u, 0xC015u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_5X1_C3)) {
        spatz_rt_set_phase(16, P3_CASE_5X1_C3);
        run_oc32_case(P3_CASE_5X1_C3, 0u, P3_H, P3_W, 3u, P3_H, P3_W, 5u, 1u, 1u, 1u, 2u, 0u, 0xC051u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X3_S2_C3)) {
        spatz_rt_set_phase(17, P3_CASE_3X3_S2_C3);
        run_oc32_case(P3_CASE_3X3_S2_C3, 0u, P3_H, P3_W, 3u, 2u, 2u, 3u, 3u, 2u, 2u, 1u, 1u, 0xC332u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X3_C1)) {
        spatz_rt_set_phase(18, P3_CASE_3X3_C1);
        run_oc32_case(P3_CASE_3X3_C1, 0u, P3_H, P3_W, 1u, P3_H, P3_W, 3u, 3u, 1u, 1u, 1u, 1u, 0xC301u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_3X3_C5)) {
        spatz_rt_set_phase(19, P3_CASE_3X3_C5);
        run_oc32_case(P3_CASE_3X3_C5, 0u, P3_H, P3_W, 5u, P3_H, P3_W, 3u, 3u, 1u, 1u, 1u, 1u, 0xC305u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_REQUANT)) {
        spatz_rt_set_phase(20, P3_CASE_REQUANT);
        run_requant_case(P3_CASE_REQUANT);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_YOLO_RGB_TCDM_SPATZ)) {
        spatz_rt_set_phase(21, P3_CASE_YOLO_RGB_TCDM_SPATZ);
        run_yolo_rgb_tcdm_spatz_case(P3_CASE_YOLO_RGB_TCDM_SPATZ);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_3X3_C3)) {
        spatz_rt_set_phase(22, P3_CASE_LINEBUF_3X3_C3);
        run_linebuf_3x3_c3_case(P3_CASE_LINEBUF_3X3_C3);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_KGEN_3X3_C32)) {
        spatz_rt_set_phase(23, P3_CASE_LINEBUF_KGEN_3X3_C32);
        run_oc32_case(P3_CASE_LINEBUF_KGEN_3X3_C32,
                  0u,
                  P3_H,
                  P3_W,
                  32u,
                  P3_H,
                  P3_W,
                      3u,
                      3u,
                      1u,
                      1u,
                      1u,
                      1u,
                      0xC9E0u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_KGEN_3X3_C96)) {
        spatz_rt_set_phase(24, P3_CASE_LINEBUF_KGEN_3X3_C96);
        run_oc32_case_channel_slices(P3_CASE_LINEBUF_KGEN_3X3_C96,
                                      0u,
                                      P3_H,
                                      P3_W,
                                      96u,
                                      P3_H,
                                      P3_W,
                                      3u,
                                      3u,
                                      1u,
                                      1u,
                                      1u,
                                      1u,
                                      96u,
                                      0u,
                                      0xC9E1u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_3X3_C120)) {
        spatz_rt_set_phase(25, P3_CASE_LINEBUF_3X3_C120);
        run_oc32_case_channel_slices(P3_CASE_LINEBUF_3X3_C120,
                                      0u,
                                      P3_C120_H,
                                      P3_C120_W,
                                      120u,
                                      P3_C120_H,
                                      P3_C120_W,
                                      3u,
                                      3u,
                                      1u,
                                      1u,
                                      1u,
                                      1u,
                                      96u,
                                      24u,
                                      0xC9E2u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_KGEN_3X3_C65)) {
        spatz_rt_set_phase(26, P3_CASE_LINEBUF_KGEN_3X3_C65);
        run_oc32_case_channel_slices(P3_CASE_LINEBUF_KGEN_3X3_C65,
                                      0u,
                                      P3_H,
                                      P3_W,
                                      65u,
                                      P3_H,
                                      P3_W,
                                      3u,
                                      3u,
                                      1u,
                                      1u,
                                      1u,
                                      1u,
                                      64u,
                                      1u,
                                      0xC9E4u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_3X3_C120_C32B)) {
        spatz_rt_set_phase(27, P3_CASE_LINEBUF_3X3_C120_C32B);
        run_oc32_case_c32_blocked_slices(P3_CASE_LINEBUF_3X3_C120_C32B,
                                          P3_C120_H,
                                          P3_C120_W,
                                          120u,
                                          P3_C120_H,
                                          P3_C120_W,
                                          3u,
                                          3u,
                                          1u,
                                          1u,
                                          1u,
                                          1u,
                                          0xC9E5u);
        spatz_rt_pass_step();
    }

    if (should_run_case(P3_CASE_LINEBUF_3X3_C32_TILED_REQUANT)) {
        spatz_rt_set_phase(28, P3_CASE_LINEBUF_3X3_C32_TILED_REQUANT);
        run_linebuf_3x3_c32_tiled_requant_case(P3_CASE_LINEBUF_3X3_C32_TILED_REQUANT);
        spatz_rt_pass_step();
    }

    spatz_rt_pass();
    return 0;
}
