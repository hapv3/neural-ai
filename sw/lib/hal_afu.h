#ifndef HAL_AFU_H
#define HAL_AFU_H

#include "npu_memory_map.h"
#include "npu_types.h"

static inline void afu_load_lut_entry(uint32_t index, uint32_t value) {
    REG_WRITE(NPU_AFU_LUT_BASE + (index * 4u), value);
}

static inline void afu_load_lut(const uint32_t *lut, uint32_t entries) {
    for (uint32_t i = 0; i < entries; i++) {
        afu_load_lut_entry(i, lut[i]);
    }
}

static inline uint32_t afu_status(void) {
    return REG_READ(NPU_AFU_STATUS);
}

static inline uint32_t afu_done(void) {
    return (afu_status() & NPU_AFU_STATUS_DONE) != 0u;
}

static inline uint32_t afu_busy(void) {
    return (afu_status() & NPU_AFU_STATUS_BUSY) != 0u;
}

static inline uint32_t afu_error(void) {
    return (afu_status() & NPU_AFU_STATUS_ERROR) != 0u;
}

static inline void afu_start(uint32_t src, uint32_t dst, uint32_t length, uint32_t mode) {
    REG_WRITE(NPU_AFU_SRC_PTR, src);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, length);
    REG_WRITE(NPU_AFU_MODE, mode);
    REG_WRITE(NPU_AFU_STATUS, 1u);
}

static inline uint32_t afu_wait_done(uint32_t timeout_cycles) {
    while (timeout_cycles-- > 0u) {
        uint32_t status = afu_status();
        if ((status & NPU_AFU_STATUS_ERROR) != 0u) {
            return 0u;
        }
        if ((status & NPU_AFU_STATUS_DONE) != 0u) {
            return 1u;
        }
        __asm__ volatile("nop");
    }
    return 0u;
}

#endif
