#include "hal_systolic.h"
#include "spatz_rt.h"

/*
 * Scenario: independent systolic GEMM32 regression.
 * Target: run deterministic INT8xINT8->INT32 GEMM for boundary M sizes and
 * copy every Mx32 INT32 result back to L2 for full Python golden comparison.
 */
#define SIG_START   (*(volatile uint32_t *)0x10008020u)

#define L2_WEIGHT   0x80000000u
#define L2_IFM      0x80001000u
#define L2_OUT      0x80010000u

#define T_WEIGHT    0x10100000u
#define T_IFM       0x10101000u
#define T_OFM       0x10140000u

#define WEIGHT_BYTES (32u * 32u)
#define MAX_M        1024u
#define IFM_BYTES    (MAX_M * 32u)

static const uint32_t dims[] = {1u, 2u, 31u, 32u, 33u, 64u, 128u, 1024u};

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1, 0);

    while (SIG_START == 0) {
    }
    SIG_START = 0;

    // Stage fixed 32x32 weights and maximum IFM rows once; each M reuses prefix rows.
    spatz_rt_set_phase(2, 1);
    spatz_rt_dma_1d(T_WEIGHT, L2_WEIGHT, WEIGHT_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_dma_1d(T_IFM, L2_IFM, IFM_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_pass_step();

    uint32_t out_offset = 0;
    for (uint32_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        uint32_t dim_m = dims[i];
        uint32_t out_bytes = dim_m * 32u * 4u;

        // Each M stresses a different controller boundary: tiny, tile edge, and long burst.
        spatz_rt_set_phase(3, dim_m);
        systolic_gemm32(T_WEIGHT, T_IFM, T_OFM, dim_m);

        // Preserve all outputs contiguously in L2 so cocotb can compare full tensors.
        spatz_rt_set_phase(4, dim_m);
        spatz_rt_dma_1d(L2_OUT + out_offset, T_OFM, out_bytes);
        spatz_rt_dma_wait_all();
        out_offset += out_bytes;
        spatz_rt_pass_step();
    }

    spatz_rt_pass();
    return 0;
}
