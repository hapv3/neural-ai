#include "hal_systolic.h"
#include "spatz_rt.h"

/*
 * Scenario: systolic GEMM32 with RTL requant output packing.
 * Target: prove INT32 accumulators are converted to packed INT8 bytes in the
 * controller drain path, with per-channel qparams and exact output checking in cocotb.
 */
#define L2_WEIGHT   0x80000000u
#define L2_IFM      0x80001000u
#define L2_OUT      0x80018000u

#define T_WEIGHT    0x10100000u
#define T_IFM       0x10101000u
#define T_OFM_I8    0x10150000u

#define WEIGHT_BYTES (32u * 32u)
#define MAX_M        64u
#define IFM_BYTES    (MAX_M * 32u)

#define NUM_DIMS 6u

static int32_t rq_bias[32];
static int32_t rq_multiplier[32];
static uint8_t rq_shift[32];
static int32_t rq_zero_point[32];

static uint32_t dim_at(uint32_t index) {
    if (index == 0u) return 1u;
    if (index == 1u) return 2u;
    if (index == 2u) return 31u;
    if (index == 3u) return 32u;
    if (index == 4u) return 33u;
    return 64u;
}

static void init_qparams(void) {
    for (uint32_t ch = 0; ch < 32u; ch++) {
        rq_bias[ch] = ((int32_t)ch - 16) * 3;
        rq_multiplier[ch] = (int32_t)((ch % 5u) + 1u);
        rq_shift[ch] = (uint8_t)(ch % 4u);
        rq_zero_point[ch] = (int32_t)(ch % 7u) - 3;
    }
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1, 0);

    spatz_rt_dma_1d(T_WEIGHT, L2_WEIGHT, WEIGHT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_IFM, L2_IFM, IFM_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_pass_step();

    init_qparams();
    systolic_requant_config_per_channel(rq_bias, rq_multiplier, rq_shift, rq_zero_point, -50, 60);

    uint32_t out_offset = 0;
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
        uint32_t dim_m = dim_at(i);
        uint32_t out_bytes = dim_m * 32u;

        spatz_rt_set_phase(2, dim_m);
        systolic_gemm32_requant(T_WEIGHT, T_IFM, T_OFM_I8, dim_m);

        spatz_rt_set_phase(3, dim_m);
        spatz_rt_dma_1d(L2_OUT + out_offset, T_OFM_I8, out_bytes);
        spatz_rt_dma_wait_all();
        out_offset += out_bytes;
        spatz_rt_pass_step();
    }

    systolic_requant_disable();
    spatz_rt_pass();
    return 0;
}
