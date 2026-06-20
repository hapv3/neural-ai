#include "spatz_rt.h"

#define SIG_START      (*(volatile uint32_t *)0x10008020u)

#define L2_SRC         0x80000000u
#define L2_DST         0x80001000u
#define TCDM_SRC       0x10100000u
#define TCDM_DST       0x10100400u
#define TCDM_BANK_LOW  0x10102000u
#define TCDM_BANK_HIGH 0x1017E000u
#define DMA_BYTES      512u

static uint8_t expected_l2(uint32_t index) {
    return (uint8_t)((index * 13u + 7u) & 0xFFu);
}

static void verify_bytes(const volatile uint8_t *ptr, uint32_t count, uint32_t test_base) {
    for (uint32_t i = 0; i < count; i++) {
        uint8_t expected = expected_l2(i);
        uint8_t got = ptr[i];
        if (got != expected) {
            spatz_rt_fail_at(test_base, i, got, expected);
        }
    }
}

static void fill_tcdm_dst(void) {
    volatile uint8_t *dst = (volatile uint8_t *)TCDM_DST;
    for (uint32_t i = 0; i < DMA_BYTES; i++) {
        dst[i] = (uint8_t)(0xA0u ^ ((i * 5u) & 0xFFu));
    }
}

static void verify_bank_addresses(void) {
    volatile uint32_t *low;
    volatile uint32_t *high;

    for (uint32_t bank = 0; bank < 16; bank++) {
        low = (volatile uint32_t *)(TCDM_BANK_LOW + bank * 32u);
        high = (volatile uint32_t *)(TCDM_BANK_HIGH + bank * 32u);
        *low = 0x11000000u | bank;
        *high = 0x22000000u | bank;
    }

    for (uint32_t bank = 0; bank < 16; bank++) {
        low = (volatile uint32_t *)(TCDM_BANK_LOW + bank * 32u);
        high = (volatile uint32_t *)(TCDM_BANK_HIGH + bank * 32u);
        uint32_t low_exp = 0x11000000u | bank;
        uint32_t high_exp = 0x22000000u | bank;
        if (*low != low_exp) {
            spatz_rt_fail_at(4, bank, (int32_t)*low, (int32_t)low_exp);
        }
        if (*high != high_exp) {
            spatz_rt_fail_at(5, bank, (int32_t)*high, (int32_t)high_exp);
        }
    }
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1, 0);

    while (SIG_START == 0) {
    }
    SIG_START = 0;

    spatz_rt_set_phase(2, 1);
    spatz_rt_memset((void *)TCDM_SRC, 0, DMA_BYTES);
    spatz_rt_dma_1d(TCDM_SRC, L2_SRC, DMA_BYTES);
    spatz_rt_dma_wait_all();
    verify_bytes((const volatile uint8_t *)TCDM_SRC, DMA_BYTES, 1);
    spatz_rt_pass_step();

    spatz_rt_set_phase(3, 2);
    fill_tcdm_dst();
    spatz_rt_dma_1d(L2_DST, TCDM_DST, DMA_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_pass_step();

    spatz_rt_set_phase(4, 3);
    verify_bank_addresses();
    spatz_rt_pass_step();

    spatz_rt_pass();
    return 0;
}
