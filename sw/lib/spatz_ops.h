#ifndef SPATZ_OPS_H
#define SPATZ_OPS_H

#include "npu_types.h"

#define SPATZ_OP_SCRATCH_I32_ADDR 0x10140000u

void spatz_vec_copy_i8(const int8_t *src, int8_t *dst, uint32_t count);
void spatz_vec_zero_i8(int8_t *dst, uint32_t count);
void spatz_vec_zero_i32(uint32_t *dst, uint32_t count);
void spatz_vec_strided_copy_i8(const int8_t *src, int8_t *dst, uint32_t count,
                               int32_t src_stride, int32_t dst_stride);
void spatz_im2col_rect_i8(const int8_t *src, int8_t *dst,
                          uint32_t rows, uint32_t cols,
                          int32_t src_col_stride, int32_t dst_col_stride,
                          uint32_t src_row_stride, uint32_t dst_row_stride);
void spatz_im2col_rect_channels_i8(const int8_t *src, int8_t *dst,
                                   uint32_t channels, uint32_t rows, uint32_t cols,
                                   int32_t src_col_stride, int32_t dst_col_stride,
                                   uint32_t src_row_stride, uint32_t dst_row_stride);
void spatz_vec_relu_i8(int8_t *data, uint32_t count);
void spatz_requant_i32_to_i32(const int32_t *src, int32_t *dst, uint32_t count,
                              int32_t multiplier, uint32_t shift,
                              int32_t min_val, int32_t max_val);
void spatz_requant_i32_to_i8(const int32_t *src, int8_t *dst, uint32_t count,
                             int32_t multiplier, uint32_t shift,
                             int32_t min_val, int32_t max_val);

#endif
