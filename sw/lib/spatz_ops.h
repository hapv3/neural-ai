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
void spatz_add_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                  uint32_t count, int32_t min_val, int32_t max_val);
void spatz_mul_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                  uint32_t count, int32_t multiplier, uint32_t shift,
                  int32_t min_val, int32_t max_val);
uint32_t npu_logistic_i8(const int8_t *src, int8_t *dst,
                         uint32_t count, const uint8_t *lut);
void spatz_maxpool2d_i8(const int8_t *src, int8_t *dst,
                        uint32_t input_h, uint32_t input_w, uint32_t channels,
                        uint32_t kernel_h, uint32_t kernel_w,
                        uint32_t stride_h, uint32_t stride_w,
                        uint32_t pad_h, uint32_t pad_w);
void spatz_upsample_nearest_i8(const int8_t *src, int8_t *dst,
                               uint32_t input_h, uint32_t input_w, uint32_t channels,
                               uint32_t scale_h, uint32_t scale_w);
void spatz_concat_c32_i8(const int8_t *src0, uint32_t c0,
                         const int8_t *src1, uint32_t c1,
                         int8_t *dst, uint32_t height, uint32_t width);

#endif
