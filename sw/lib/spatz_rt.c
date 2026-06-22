#include "spatz_rt.h"
#include "idma_mm_utils.h"

static volatile uint32_t *const sig = (volatile uint32_t *)SPATZ_RT_STATUS_ADDR;

void spatz_rt_init(void) {
    for (uint32_t i = 0; i < 8; i++) {
        sig[i] = 0;
    }
}

void spatz_rt_set_phase(uint32_t phase, uint32_t op) {
    sig[SPATZ_RT_SIG_PHASE] = phase;
    sig[SPATZ_RT_SIG_OP] = op;
}

void spatz_rt_pass_step(void) {
    sig[SPATZ_RT_SIG_PASS_COUNT] = sig[SPATZ_RT_SIG_PASS_COUNT] + 1;
}

void spatz_rt_pass(void) {
    sig[SPATZ_RT_SIG_STATUS] = SPATZ_RT_PASS_SIGNATURE;
    while (1) {
    }
}

void spatz_rt_fail_at(uint32_t test_id, uint32_t index, int32_t got, int32_t expected) {
    sig[SPATZ_RT_SIG_FAIL_TEST] = test_id;
    sig[SPATZ_RT_SIG_FAIL_INDEX] = index;
    sig[SPATZ_RT_SIG_GOT] = (uint32_t)got;
    sig[SPATZ_RT_SIG_EXPECTED] = (uint32_t)expected;
    sig[SPATZ_RT_SIG_STATUS] = SPATZ_RT_FAIL_PREFIX | (test_id & 0xFFFFu);
    while (1) {
    }
}

void *spatz_rt_memset(void *ptr, int value, uint32_t num) {
    uint8_t *dst = (uint8_t *)ptr;
    for (uint32_t i = 0; i < num; i++) {
        dst[i] = (uint8_t)value;
    }
    return ptr;
}

void *spatz_rt_memcpy(void *dst, const void *src, uint32_t num) {
    uint8_t *dst_bytes = (uint8_t *)dst;
    const uint8_t *src_bytes = (const uint8_t *)src;
    for (uint32_t i = 0; i < num; i++) {
        dst_bytes[i] = src_bytes[i];
    }
    return dst;
}

void spatz_rt_dma_1d(uint32_t dst_addr, uint32_t src_addr, uint32_t size) {
    if (!idma_memcpy_blocking(src_addr, dst_addr, size)) {
        spatz_rt_fail_at(0x0DADu, 0, (int32_t)size, 1);
    }
}

void spatz_rt_dma_wait_all(void) {
}
