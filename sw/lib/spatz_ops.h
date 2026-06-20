#ifndef SPATZ_OPS_H
#define SPATZ_OPS_H

#include "npu_types.h"

#define SPATZ_OP_SCRATCH_I32_ADDR 0x10140000u

void spatz_vec_copy_i8(const int8_t *src, int8_t *dst, uint32_t count);
void spatz_vec_relu_i8(int8_t *data, uint32_t count);
void spatz_requant_i32_to_i32(const int32_t *src, int32_t *dst, uint32_t count,
                              int32_t multiplier, uint32_t shift,
                              int32_t min_val, int32_t max_val);
void spatz_requant_i32_to_i8(const int32_t *src, int8_t *dst, uint32_t count,
                             int32_t multiplier, uint32_t shift,
                             int32_t min_val, int32_t max_val);

#endif
