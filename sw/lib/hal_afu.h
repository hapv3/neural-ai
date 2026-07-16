#ifndef HAL_AFU_H
#define HAL_AFU_H

#include "npu_memory_map.h"
#include "npu_types.h"

static inline void afu_load_lut_entry(uint32_t index, uint32_t value) {
    REG_WRITE(NPU_AFU_LUT_BASE + (index * 4u), value);
}

static inline void afu_load_dfl_exp_lut_entry(uint32_t index, uint32_t value) {
    REG_WRITE(NPU_AFU_DFL_EXP_LUT_BASE + (index * 4u), value);
}

static inline void afu_load_dfl_recip_lut_entry(uint32_t index, uint32_t value) {
    REG_WRITE(NPU_AFU_DFL_RECIP_LUT_BASE + (index * 4u), value);
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

static inline void afu_preload(uint32_t src, uint32_t dst, uint32_t length, uint32_t mode) {
    REG_WRITE(NPU_AFU_SRC_PTR, src);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, length);
    REG_WRITE(NPU_AFU_MODE, mode);
}

static inline void afu_start_preloaded(void) {
    REG_WRITE(NPU_AFU_STATUS, 1u);
}

static inline void afu_preload_binary(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                      uint32_t length, uint32_t mode) {
    REG_WRITE(NPU_AFU_SRC_PTR, lhs);
    REG_WRITE(NPU_AFU_SRC2_PTR, rhs);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, length);
    REG_WRITE(NPU_AFU_MODE, mode);
}

static inline void afu_start(uint32_t src, uint32_t dst, uint32_t length, uint32_t mode) {
    afu_preload(src, dst, length, mode);
    afu_start_preloaded();
}

static inline void afu_start_binary(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                    uint32_t length, uint32_t mode) {
    afu_preload_binary(lhs, rhs, dst, length, mode);
    afu_start_preloaded();
}

static inline void afu_start_mul_q7(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                    uint32_t length) {
    afu_start_binary(lhs, rhs, dst, length, NPU_AFU_MODE_MUL_Q7);
}

static inline void afu_start_add_i8(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                    uint32_t length) {
    afu_start_binary(lhs, rhs, dst, length, NPU_AFU_MODE_ADD_I8);
}

static inline void afu_preload_class_sigmoid_row32_high16(uint32_t src, uint32_t dst,
                                                          uint32_t input_bytes) {
    afu_preload(src, dst, input_bytes, NPU_AFU_MODE_CLASS_SIGMOID_ROW32_HIGH16);
}

static inline void afu_start_class_sigmoid_row32_high16(uint32_t src, uint32_t dst,
                                                        uint32_t input_bytes) {
    afu_preload_class_sigmoid_row32_high16(src, dst, input_bytes);
    afu_start_preloaded();
}

static inline void afu_preload_global_avgpool_c32(uint32_t src, uint32_t dst,
                                                  uint32_t input_bytes,
                                                  uint32_t spatial_count) {
    REG_WRITE(NPU_AFU_SRC_PTR, src);
    REG_WRITE(NPU_AFU_SRC2_PTR, spatial_count);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, input_bytes);
    REG_WRITE(NPU_AFU_MODE, NPU_AFU_MODE_GLOBAL_AVGPOOL_C32);
}

static inline void afu_start_global_avgpool_c32(uint32_t src, uint32_t dst,
                                                uint32_t input_bytes,
                                                uint32_t spatial_count) {
    afu_preload_global_avgpool_c32(src, dst, input_bytes, spatial_count);
    afu_start_preloaded();
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
