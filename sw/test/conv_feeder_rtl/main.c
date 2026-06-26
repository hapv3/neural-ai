#include "hal_conv2d_feeder.h"
#include "hal_systolic.h"
#include "spatz_rt.h"

/*
 * Scenario: P2 RTL Conv2D feeder for systolic Conv lowering.
 * Target: stream tile-local IFM rows directly into systolic while keeping
 * materialized im2col dumps as a debug/verification mode.
 */
#define L2_CONV1_INPUT   0x80000000u
#define L2_CONV1_WEIGHT  0x80002000u
#define L2_CONV1_IM2COL0 0x80008000u
#define L2_CONV1_IM2COL1 0x80009000u
#define L2_CONV1_OUT     0x80010000u

#define L2_CONV3_INPUT   0x80020000u
#define L2_CONV3_WEIGHT  0x80022000u
#define L2_CONV3_OUT     0x80030000u

#define T_INPUT          0x10100000u
#define T_WEIGHT         0x10102000u
#define T_IM2COL         0x10108000u
#define T_OUTPUT         0x10140000u

#define CONV1_H          4u
#define CONV1_W          5u
#define CONV1_C          33u
#define CONV1_ROWS       (CONV1_H * CONV1_W)
#define CONV1_INPUT_BYTES  (CONV1_ROWS * CONV1_C)
#define CONV1_WEIGHT_BYTES (2u * 32u * 32u)
#define CONV1_IM2COL_BYTES (CONV1_ROWS * 32u)
#define CONV1_OUT_BYTES    (CONV1_ROWS * 32u * 4u)

#define CONV3_H          5u
#define CONV3_W          5u
#define CONV3_C          3u
#define CONV3_ROWS       (CONV3_H * CONV3_W)
#define CONV3_INPUT_BYTES  (CONV3_ROWS * CONV3_C)
#define CONV3_WEIGHT_BYTES (32u * 32u)
#define CONV3_OUT_BYTES    (CONV3_ROWS * 32u * 4u)

static void init_conv1_cfg(conv2d_feeder_hw_cfg_t *cfg) {
    cfg->input_addr = T_INPUT;
    cfg->output_addr = T_IM2COL;
    cfg->rows = CONV1_ROWS;
    cfg->output_w = CONV1_W;
    cfg->input_h = CONV1_H;
    cfg->input_w = CONV1_W;
    cfg->input_c = CONV1_C;
    cfg->kernel_h = 1u;
    cfg->kernel_w = 1u;
    cfg->stride_h = 1u;
    cfg->stride_w = 1u;
    cfg->pad_h = 0u;
    cfg->pad_w = 0u;
    cfg->dilation_h = 1u;
    cfg->dilation_w = 1u;
}

static void init_conv3_cfg(conv2d_feeder_hw_cfg_t *cfg) {
    cfg->input_addr = T_INPUT;
    cfg->output_addr = T_IM2COL;
    cfg->rows = CONV3_ROWS;
    cfg->output_w = CONV3_W;
    cfg->input_h = CONV3_H;
    cfg->input_w = CONV3_W;
    cfg->input_c = CONV3_C;
    cfg->kernel_h = 3u;
    cfg->kernel_w = 3u;
    cfg->stride_h = 1u;
    cfg->stride_w = 1u;
    cfg->pad_h = 1u;
    cfg->pad_w = 1u;
    cfg->dilation_h = 1u;
    cfg->dilation_w = 1u;
}

static void run_conv1x1_k33(void) {
    conv2d_feeder_hw_cfg_t cfg;
    init_conv1_cfg(&cfg);

    spatz_rt_dma_1d(T_INPUT, L2_CONV1_INPUT, CONV1_INPUT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_WEIGHT, L2_CONV1_WEIGHT, CONV1_WEIGHT_BYTES);
    spatz_rt_dma_wait_all();

    conv2d_feeder_hw_materialize_k_tile(&cfg, 0u);
    spatz_rt_dma_1d(L2_CONV1_IM2COL0, T_IM2COL, CONV1_IM2COL_BYTES);
    spatz_rt_dma_wait_all();

    conv2d_feeder_hw_start_stream_k_tile(&cfg, 0u);
    systolic_gemm32_stream(T_WEIGHT, T_OUTPUT, CONV1_ROWS);
    conv2d_feeder_hw_wait();

    conv2d_feeder_hw_materialize_k_tile(&cfg, 32u);
    spatz_rt_dma_1d(L2_CONV1_IM2COL1, T_IM2COL, CONV1_IM2COL_BYTES);
    spatz_rt_dma_wait_all();

    conv2d_feeder_hw_start_stream_k_tile(&cfg, 32u);
    systolic_gemm32_accumulate_stream(T_WEIGHT + (32u * 32u), T_OUTPUT, T_OUTPUT, CONV1_ROWS);
    conv2d_feeder_hw_wait();

    spatz_rt_dma_1d(L2_CONV1_OUT, T_OUTPUT, CONV1_OUT_BYTES);
    spatz_rt_dma_wait_all();
}

static void run_conv3x3_pad1_c3(void) {
    conv2d_feeder_hw_cfg_t cfg;
    init_conv3_cfg(&cfg);

    spatz_rt_dma_1d(T_INPUT, L2_CONV3_INPUT, CONV3_INPUT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_WEIGHT, L2_CONV3_WEIGHT, CONV3_WEIGHT_BYTES);
    spatz_rt_dma_wait_all();

    conv2d_feeder_hw_materialize_k_tile(&cfg, 0u);
    conv2d_feeder_hw_start_stream_k_tile(&cfg, 0u);
    systolic_gemm32_stream(T_WEIGHT, T_OUTPUT, CONV3_ROWS);
    conv2d_feeder_hw_wait();

    spatz_rt_dma_1d(L2_CONV3_OUT, T_OUTPUT, CONV3_OUT_BYTES);
    spatz_rt_dma_wait_all();
}

int main(void) {
    spatz_rt_init();

    spatz_rt_set_phase(1, 1);
    run_conv1x1_k33();
    spatz_rt_pass_step();

    spatz_rt_set_phase(2, 3);
    run_conv3x3_pad1_c3();
    spatz_rt_pass_step();

    spatz_rt_pass();
    return 0;
}
