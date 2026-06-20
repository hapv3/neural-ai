#include "hal_systolic.h"
#include "spatz_rt.h"

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

        spatz_rt_set_phase(3, dim_m);
        spatz_rt_memset((void *)T_OFM, 0x5A, out_bytes);
        systolic_gemm32(T_WEIGHT, T_IFM, T_OFM, dim_m);

        spatz_rt_set_phase(4, dim_m);
        spatz_rt_dma_1d(L2_OUT + out_offset, T_OFM, out_bytes);
        spatz_rt_dma_wait_all();
        out_offset += out_bytes;
        spatz_rt_pass_step();
    }

    spatz_rt_pass();
    return 0;
}
