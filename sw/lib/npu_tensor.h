#ifndef NPU_TENSOR_H
#define NPU_TENSOR_H

#include "npu_types.h"

typedef enum {
    NPU_DTYPE_I8 = 1,
    NPU_DTYPE_I32 = 4
} npu_dtype_t;

typedef enum {
    NPU_LAYOUT_HWC = 1,
    NPU_LAYOUT_ROW32 = 2,
    NPU_LAYOUT_C32_BLOCKED = 3
} npu_layout_t;

typedef struct {
    uint32_t addr;
    uint16_t h;
    uint16_t w;
    uint16_t c;
    uint16_t reserved;
    uint32_t bytes;
    npu_dtype_t dtype;
    npu_layout_t layout;
    int32_t scale_q31;
    int32_t zero_point;
} npu_tensor_t;

static inline uint32_t npu_align_up(uint32_t value, uint32_t align) {
    return (value + align - 1u) & ~(align - 1u);
}

static inline uint32_t npu_c32_blocks(uint32_t channels) {
    return (channels + 31u) >> 5;
}

static inline uint32_t npu_tensor_hwc_bytes(uint32_t height, uint32_t width, uint32_t channels) {
    return height * width * channels;
}

static inline uint32_t npu_tensor_c32_bytes(uint32_t height, uint32_t width, uint32_t channels) {
    return height * width * npu_c32_blocks(channels) * 32u;
}

static inline uint32_t npu_tensor_row32_bytes(uint32_t rows) {
    return rows * 32u;
}

static inline uint32_t npu_tensor_i32_row32_bytes(uint32_t rows) {
    return rows * 32u * 4u;
}

static inline uint32_t npu_tensor_hwc_offset(uint32_t width, uint32_t channels,
                                             uint32_t y, uint32_t x, uint32_t c) {
    return ((y * width + x) * channels) + c;
}

static inline uint32_t npu_tensor_c32_offset(uint32_t height, uint32_t width, uint32_t channels,
                                             uint32_t y, uint32_t x, uint32_t c) {
    (void)height;
    (void)channels;
    uint32_t pixel = y * width + x;
    return (((c >> 5) * (height * width) + pixel) * 32u) + (c & 31u);
}

static inline uint32_t npu_tensor_offset(const npu_tensor_t *tensor,
                                         uint32_t y, uint32_t x, uint32_t c) {
    if (tensor->layout == NPU_LAYOUT_C32_BLOCKED) {
        return npu_tensor_c32_offset(tensor->h, tensor->w, tensor->c, y, x, c);
    }
    return npu_tensor_hwc_offset(tensor->w, tensor->c, y, x, c);
}

#endif
