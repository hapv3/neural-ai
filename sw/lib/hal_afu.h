#ifndef HAL_AFU_H
#define HAL_AFU_H

#include "npu_memory_map.h"
#include "npu_types.h"

typedef struct {
    int32_t lhs_multiplier;
    uint32_t lhs_shift;
    int32_t rhs_multiplier;
    uint32_t rhs_shift;
    int32_t output_multiplier;
    uint32_t output_shift;
    int32_t lhs_zero_point;
    int32_t rhs_zero_point;
    int32_t output_zero_point;
    int32_t clamp_min;
    int32_t clamp_max;
    uint32_t double_round_shift;
    uint32_t operation;
} afu_binary_quant_params_t;

enum {
    AFU_BINARY_QUANT_ADD = 0u,
    AFU_BINARY_QUANT_SUBTRACT = 1u,
    AFU_BINARY_QUANT_MULTIPLY = 2u
};

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
    REG_WRITE(NPU_AFU_ADD_BIAS, 0u);
}

static inline void afu_preload_binary_bias(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                           uint32_t length, uint32_t mode,
                                           int32_t bias) {
    REG_WRITE(NPU_AFU_SRC_PTR, lhs);
    REG_WRITE(NPU_AFU_SRC2_PTR, rhs);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, length);
    REG_WRITE(NPU_AFU_MODE, mode);
    REG_WRITE(NPU_AFU_ADD_BIAS, (uint32_t)bias);
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

static inline void afu_start_add_i8_bias(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                         uint32_t length, int32_t bias) {
    afu_preload_binary_bias(lhs, rhs, dst, length, NPU_AFU_MODE_ADD_I8, bias);
    afu_start_preloaded();
}

static inline void afu_preload_binary_quant(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                             uint32_t length, uint32_t lut_chain,
                                             const afu_binary_quant_params_t *params) {
    REG_WRITE(NPU_AFU_SRC_PTR, lhs);
    REG_WRITE(NPU_AFU_SRC2_PTR, rhs);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, length);
    REG_WRITE(NPU_AFU_BINARY_OP, params->operation);
    REG_WRITE(NPU_AFU_BINARY_LHS_MULTIPLIER, (uint32_t)params->lhs_multiplier);
    REG_WRITE(NPU_AFU_BINARY_LHS_SHIFT, params->lhs_shift);
    REG_WRITE(NPU_AFU_BINARY_RHS_MULTIPLIER, (uint32_t)params->rhs_multiplier);
    REG_WRITE(NPU_AFU_BINARY_RHS_SHIFT, params->rhs_shift);
    REG_WRITE(NPU_AFU_BINARY_OUT_MULTIPLIER, (uint32_t)params->output_multiplier);
    REG_WRITE(NPU_AFU_BINARY_OUT_SHIFT, params->output_shift);
    REG_WRITE(NPU_AFU_BINARY_ZERO_POINTS,
              ((uint32_t)params->lhs_zero_point & 0xffu) |
              (((uint32_t)params->rhs_zero_point & 0xffu) << 8) |
              (((uint32_t)params->output_zero_point & 0xffu) << 16));
    REG_WRITE(NPU_AFU_BINARY_CLAMP,
              ((uint32_t)params->clamp_min & 0xffu) |
              (((uint32_t)params->clamp_max & 0xffu) << 8));
    REG_WRITE(NPU_AFU_BINARY_DOUBLE_ROUND, params->double_round_shift);
    REG_WRITE(NPU_AFU_MODE, lut_chain != 0u ?
              NPU_AFU_MODE_LUT_BINARY_QUANT : NPU_AFU_MODE_BINARY_QUANT);
}

static inline void afu_start_binary_quant(uint32_t lhs, uint32_t rhs, uint32_t dst,
                                           uint32_t length, uint32_t lut_chain,
                                           const afu_binary_quant_params_t *params) {
    afu_preload_binary_quant(lhs, rhs, dst, length, lut_chain, params);
    afu_start_preloaded();
}

static inline void afu_preload_dfl_row32(uint32_t src, uint32_t dst,
                                         uint32_t input_bytes,
                                         uint32_t bins) {
    REG_WRITE(NPU_AFU_SRC_PTR, src);
    REG_WRITE(NPU_AFU_SRC2_PTR, bins);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, input_bytes);
    REG_WRITE(NPU_AFU_MODE, NPU_AFU_MODE_DFL4_ROW32_Q8);
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

static inline void afu_preload_global_avgpool_requant_c32(
    uint32_t src, uint32_t dst, uint32_t input_bytes, uint32_t spatial_count,
    int32_t output_multiplier, uint32_t output_shift, int32_t input_offset,
    int32_t output_zero_point, uint32_t double_round_shift) {
    REG_WRITE(NPU_AFU_SRC_PTR, src);
    REG_WRITE(NPU_AFU_SRC2_PTR, spatial_count);
    REG_WRITE(NPU_AFU_DST_PTR, dst);
    REG_WRITE(NPU_AFU_LENGTH, input_bytes);
    REG_WRITE(NPU_AFU_ADD_BIAS, (uint32_t)input_offset);
    REG_WRITE(NPU_AFU_BINARY_OUT_MULTIPLIER, (uint32_t)output_multiplier);
    REG_WRITE(NPU_AFU_BINARY_OUT_SHIFT, output_shift);
    REG_WRITE(NPU_AFU_BINARY_ZERO_POINTS,
              ((uint32_t)output_zero_point & 0xffu) << 16);
    REG_WRITE(NPU_AFU_BINARY_CLAMP, 0x00007f80u);
    REG_WRITE(NPU_AFU_BINARY_DOUBLE_ROUND, double_round_shift);
    REG_WRITE(NPU_AFU_MODE, NPU_AFU_MODE_GLOBAL_AVGPOOL_REQUANT_C32);
}

static inline void afu_start_global_avgpool_requant_c32(
    uint32_t src, uint32_t dst, uint32_t input_bytes, uint32_t spatial_count,
    int32_t output_multiplier, uint32_t output_shift, int32_t input_offset,
    int32_t output_zero_point, uint32_t double_round_shift) {
    afu_preload_global_avgpool_requant_c32(src, dst, input_bytes, spatial_count,
        output_multiplier, output_shift, input_offset, output_zero_point,
        double_round_shift);
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
