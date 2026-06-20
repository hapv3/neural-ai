#ifndef SPATZ_RT_H
#define SPATZ_RT_H

#include "npu_types.h"

#define SPATZ_RT_STATUS_ADDR     0x10008000u
#define SPATZ_RT_PASS_SIGNATURE  0xDEADBEEFu
#define SPATZ_RT_FAIL_PREFIX     0xBAD00000u

enum {
    SPATZ_RT_SIG_STATUS = 0,
    SPATZ_RT_SIG_PASS_COUNT = 1,
    SPATZ_RT_SIG_FAIL_TEST = 2,
    SPATZ_RT_SIG_FAIL_INDEX = 3,
    SPATZ_RT_SIG_GOT = 4,
    SPATZ_RT_SIG_EXPECTED = 5,
    SPATZ_RT_SIG_PHASE = 6,
    SPATZ_RT_SIG_OP = 7,
};

void spatz_rt_init(void);
void spatz_rt_set_phase(uint32_t phase, uint32_t op);
void spatz_rt_pass_step(void);
void spatz_rt_pass(void);
void spatz_rt_fail_at(uint32_t test_id, uint32_t index, int32_t got, int32_t expected);
void *spatz_rt_memset(void *ptr, int value, uint32_t num);
void *spatz_rt_memcpy(void *dst, const void *src, uint32_t num);
void spatz_rt_dma_1d(uint32_t dst_addr, uint32_t src_addr, uint32_t size);
void spatz_rt_dma_wait_all(void);

#endif
