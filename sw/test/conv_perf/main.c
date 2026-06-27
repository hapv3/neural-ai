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

#define L2_P3_BASE          0x80080000u
#define P3_CASE_STRIDE      0x00010000u
#define P3_INPUT_ADDR(id)   (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x0000u)
#define P3_WEIGHT_ADDR(id)  (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x3000u)
#define P3_OUT_ADDR(id)     (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0x6000u)
#define P3_STATS_ADDR(id)   (L2_P3_BASE + ((id) * P3_CASE_STRIDE) + 0xE000u)

#define T_INPUT          0x10100000u
#define T_WEIGHT         0x10102000u
#define T_IM2COL         0x10108000u
#define T_OUTPUT         0x10140000u
#define T_OUTPUT_OC1     0x10150000u
#define T_STATS          0x10178000u

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

#ifndef CONV_PERF_GROUP
#define CONV_PERF_GROUP 0
#endif

#define CONV_PERF_GROUP_ALL       0
#define CONV_PERF_GROUP_POINTWISE 1
#define CONV_PERF_GROUP_KERNELS   2
#define CONV_PERF_GROUP_REQUANT   3

static uint32_t should_run_legacy(void) {
    return (CONV_PERF_GROUP == CONV_PERF_GROUP_ALL) || (CONV_PERF_GROUP == CONV_PERF_GROUP_POINTWISE);
}

static uint32_t should_run_case(uint32_t case_id) {
    if (CONV_PERF_GROUP == CONV_PERF_GROUP_ALL) {
        return 1u;
    }
    if (CONV_PERF_GROUP == CONV_PERF_GROUP_POINTWISE) {
        return case_id <= P3_CASE_OC64;
    }
    if (CONV_PERF_GROUP == CONV_PERF_GROUP_KERNELS) {
        return (case_id >= P3_CASE_3X3_P0_C32) && (case_id <= P3_CASE_3X3_C5);
    }
    if (CONV_PERF_GROUP == CONV_PERF_GROUP_REQUANT) {
        return case_id == P3_CASE_REQUANT;
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
    npu_conv2d_packed_cfg_t cfg;
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

static void copy_oc32_to_oc64_l2(uint32_t l2_addr, uint32_t src_addr, uint32_t rows, uint32_t oc_base) {
    int tx = idma_L1ToL2_2d(src_addr, l2_addr + (oc_base * 4u), 32u * 4u, 32u * 4u, 64u * 4u, rows);
    if (!idma_mm_wait_for_completion(IDMA_DIR_L1_TO_L2, (uint32_t)tx)) {
        spatz_rt_fail_at(0xC0D0u, oc_base, tx, 1);
    }
}

static void run_oc64_case(uint32_t case_id) {
    npu_conv2d_packed_cfg_t cfg;
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
    npu_conv2d_packed_cfg_t cfg;
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

static void run_conv1x1_k33(void) {
    npu_conv2d_packed_cfg_t cfg;
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
    npu_conv2d_packed_cfg_t cfg;
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
    npu_conv2d_packed_cfg_t cfg;
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

    spatz_rt_pass();
    return 0;
}
