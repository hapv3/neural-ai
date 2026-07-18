#include "spatz_ops.h"
#include "hal_afu.h"

static int8_t pack_i8(int32_t value) {
    return (int8_t)value;
}

static int32_t clamp_i32_local(int32_t value, int32_t min_val, int32_t max_val) {
    if (value < min_val) {
        return min_val;
    }
    if (value > max_val) {
        return max_val;
    }
    return value;
}

void spatz_requant_i32_to_i8(const int32_t *src, int8_t *dst, uint32_t count,
                             int32_t multiplier, uint32_t shift,
                             int32_t min_val, int32_t max_val) {
    int32_t *scratch = (int32_t *)SPATZ_OP_SCRATCH_I32_ADDR;

    spatz_requant_i32_to_i32(src, scratch, count, multiplier, shift, min_val, max_val);

    for (uint32_t i = 0; i < count; i++) {
        dst[i] = pack_i8(scratch[i]);
    }
}

void spatz_add_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                  uint32_t count, int32_t min_val, int32_t max_val) {
    if (min_val == -128 && max_val == 127) {
        if (npu_add_i8(lhs, rhs, dst, count)) {
            return;
        }
    } else {
        int8_t *scratch = (int8_t *)SPATZ_OP_SCRATCH_I8_ADDR;
        if (npu_add_i8(lhs, rhs, scratch, count) &&
            npu_clamp_i8(scratch, dst, count, min_val, max_val)) {
            return;
        }
    }

    for (uint32_t i = 0; i < count; i++) {
        int32_t value = (int32_t)lhs[i] + (int32_t)rhs[i];
        dst[i] = pack_i8(clamp_i32_local(value, min_val, max_val));
    }
}

void spatz_mul_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                  uint32_t count, int32_t multiplier, uint32_t shift,
                  int32_t min_val, int32_t max_val) {
    for (uint32_t i = 0; i < count; i++) {
        int32_t value = (int32_t)lhs[i] * (int32_t)rhs[i];
        value = (value * multiplier) >> shift;
        dst[i] = pack_i8(clamp_i32_local(value, min_val, max_val));
    }
}

uint32_t npu_logistic_i8(const int8_t *src, int8_t *dst,
                         uint32_t count, const uint8_t *lut) {
    afu_preload((uint32_t)src, (uint32_t)dst, count, NPU_AFU_MODE_E8);
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_lut_entry(i, (uint32_t)lut[i]);
    }

    afu_start_preloaded();
    return afu_wait_done(1000000u);
}

uint32_t npu_clamp_i8(const int8_t *src, int8_t *dst,
                      uint32_t count, int32_t min_val, int32_t max_val) {
    if (!src || !dst || count == 0u || min_val > max_val ||
        min_val < -128 || max_val > 127) {
        return 0u;
    }

    afu_preload((uint32_t)src, (uint32_t)dst, count, NPU_AFU_MODE_E8);
    for (uint32_t i = 0; i < 256u; i++) {
        int32_t value = (i < 128u) ? (int32_t)i : ((int32_t)i - 256);
        int32_t clamped = clamp_i32_local(value, min_val, max_val);
        afu_load_lut_entry(i, (uint32_t)(uint8_t)(int8_t)clamped);
    }

    afu_start_preloaded();
    return afu_wait_done(100000u + (count * 16u));
}

uint32_t npu_mul_q7_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                       uint32_t count) {
    afu_preload_binary((uint32_t)lhs, (uint32_t)rhs, (uint32_t)dst,
                       count, NPU_AFU_MODE_MUL_Q7);
    afu_start_preloaded();
    return afu_wait_done(1000000u);
}

uint32_t npu_add_i8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                    uint32_t count) {
    afu_preload_binary((uint32_t)lhs, (uint32_t)rhs, (uint32_t)dst,
                       count, NPU_AFU_MODE_ADD_I8);
    afu_start_preloaded();
    return afu_wait_done(1000000u);
}

static uint8_t dfl_delta_index(int8_t value, int8_t max_value) {
    return (uint8_t)((int32_t)value - (int32_t)max_value);
}

static uint16_t dfl_distance_q8_from_exp4(const uint16_t *exp_values) {
    uint32_t sum = (uint32_t)exp_values[0] +
                   (uint32_t)exp_values[1] +
                   (uint32_t)exp_values[2] +
                   (uint32_t)exp_values[3];
    uint32_t weighted = (uint32_t)exp_values[1] +
                        (2u * (uint32_t)exp_values[2]) +
                        (3u * (uint32_t)exp_values[3]);

    if (sum == 0u) {
        return 0u;
    }

    return (uint16_t)(((weighted << 8) + (sum >> 1)) / sum);
}

static uint32_t dfl_afu_timeout(uint32_t elem_count) {
    return 100000u + (elem_count * 256u);
}

uint32_t spatz_dfl_softmax4_i8_q8(const int8_t *src, uint16_t *dst,
                                  uint32_t locations,
                                  const uint16_t *exp_lut) {
    if (!src || !dst || !exp_lut) {
        return 0u;
    }

    for (uint32_t loc = 0; loc < locations; loc++) {
        const int8_t *in_pixel = &src[loc * 16u];
        uint16_t *out_pixel = &dst[loc * 4u];

        for (uint32_t side = 0; side < 4u; side++) {
            const int8_t *logits = &in_pixel[side * 4u];
            int8_t max_value = logits[0];
            uint16_t exp_values[4];

            for (uint32_t bin = 1u; bin < 4u; bin++) {
                if (logits[bin] > max_value) {
                    max_value = logits[bin];
                }
            }

            for (uint32_t bin = 0u; bin < 4u; bin++) {
                exp_values[bin] = exp_lut[dfl_delta_index(logits[bin], max_value)];
            }
            out_pixel[side] = dfl_distance_q8_from_exp4(exp_values);
        }
    }

    return 1u;
}

uint32_t npu_dfl_softmax4_i8_q8(const int8_t *src, uint16_t *dst,
                                uint32_t locations,
                                const uint16_t *exp_lut,
                                uint8_t *delta_scratch,
                                uint16_t *exp_scratch) {
    uint32_t elem_count = locations * 16u;

    if (!src || !dst || !exp_lut || !delta_scratch || !exp_scratch) {
        return 0u;
    }

    for (uint32_t loc = 0; loc < locations; loc++) {
        const int8_t *in_pixel = &src[loc * 16u];
        uint8_t *delta_pixel = &delta_scratch[loc * 16u];

        for (uint32_t side = 0; side < 4u; side++) {
            const int8_t *logits = &in_pixel[side * 4u];
            uint8_t *delta_side = &delta_pixel[side * 4u];
            int8_t max_value = logits[0];

            for (uint32_t bin = 1u; bin < 4u; bin++) {
                if (logits[bin] > max_value) {
                    max_value = logits[bin];
                }
            }

            for (uint32_t bin = 0u; bin < 4u; bin++) {
                delta_side[bin] = dfl_delta_index(logits[bin], max_value);
            }
        }
    }

    afu_preload((uint32_t)delta_scratch, (uint32_t)exp_scratch,
                elem_count, NPU_AFU_MODE_E16);
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_lut_entry(i, (uint32_t)exp_lut[i]);
    }
    afu_start_preloaded();
    if (!afu_wait_done(dfl_afu_timeout(elem_count))) {
        return 0u;
    }

    for (uint32_t loc = 0; loc < locations; loc++) {
        const uint16_t *exp_pixel = &exp_scratch[loc * 16u];
        uint16_t *out_pixel = &dst[loc * 4u];

        for (uint32_t side = 0; side < 4u; side++) {
            out_pixel[side] = dfl_distance_q8_from_exp4(&exp_pixel[side * 4u]);
        }
    }

    return 1u;
}

uint32_t npu_dfl_softmax4_row32_i8_q8(const int8_t *src_row32, uint16_t *dst,
                                      uint32_t locations,
                                      const uint32_t *exp_lut,
                                      const uint32_t *recip_lut) {
    uint32_t input_bytes = locations * 32u;

    if (!src_row32 || !dst || !exp_lut || !recip_lut) {
        return 0u;
    }

    afu_preload((uint32_t)src_row32, (uint32_t)dst,
                input_bytes, NPU_AFU_MODE_DFL4_ROW32_Q8);
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_dfl_exp_lut_entry(i, exp_lut[i]);
        afu_load_dfl_recip_lut_entry(i, recip_lut[i]);
    }

    afu_start_preloaded();
    return afu_wait_done(100000u + (locations * 128u));
}

uint32_t npu_class_sigmoid_row32_high16_i8(const int8_t *src_row32, int8_t *dst,
                                           uint32_t locations,
                                           const uint8_t *lut) {
    uint32_t input_bytes = locations * 32u;

    if (!src_row32 || !dst || !lut) {
        return 0u;
    }

    afu_preload_class_sigmoid_row32_high16((uint32_t)src_row32,
                                           (uint32_t)dst,
                                           input_bytes);
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_lut_entry(i, (uint32_t)lut[i]);
    }

    afu_start_preloaded();
    return afu_wait_done(100000u + (locations * 64u));
}

uint32_t npu_global_avgpool_c32_i8(const int8_t *src_c32, int8_t *dst_c32,
                                   uint32_t input_h, uint32_t input_w,
                                   uint32_t channels) {
    uint32_t spatial_count = input_h * input_w;
    uint32_t groups = (channels + 31u) >> 5;
    uint32_t input_bytes = spatial_count * groups * 32u;
    uint32_t recip_q31;

    if (!src_c32 || !dst_c32 || spatial_count == 0u || channels == 0u) {
        return 0u;
    }

    recip_q31 = (uint32_t)((1ull << 31) / (uint64_t)spatial_count);
    afu_load_lut_entry(0u, recip_q31);
    afu_preload_global_avgpool_c32((uint32_t)src_c32,
                                   (uint32_t)dst_c32,
                                   input_bytes,
                                   spatial_count);
    afu_start_preloaded();
    return afu_wait_done(100000u + (spatial_count * groups * 32u));
}

void spatz_maxpool2d_i8(const int8_t *src, int8_t *dst,
                        uint32_t input_h, uint32_t input_w, uint32_t channels,
                        uint32_t kernel_h, uint32_t kernel_w,
                        uint32_t stride_h, uint32_t stride_w,
                        uint32_t pad_h, uint32_t pad_w) {
    if (channels == 32u && kernel_h == 5u && kernel_w == 5u &&
        stride_h == 1u && stride_w == 1u && pad_h == 2u && pad_w == 2u &&
        src != dst) {
        spatz_maxpool2d_5x5s1p2_c32_i8(src, dst, input_h, input_w);
        return;
    }

    uint32_t output_h = ((input_h + (2u * pad_h) - kernel_h) / stride_h) + 1u;
    uint32_t output_w = ((input_w + (2u * pad_w) - kernel_w) / stride_w) + 1u;

    for (uint32_t oh = 0; oh < output_h; oh++) {
        int32_t ih_base = (int32_t)(oh * stride_h) - (int32_t)pad_h;
        uint32_t ih_start = (ih_base < 0) ? 0u : (uint32_t)ih_base;
        uint32_t ih_end = (ih_base + (int32_t)kernel_h > (int32_t)input_h)
            ? input_h
            : (uint32_t)(ih_base + (int32_t)kernel_h);

        for (uint32_t ow = 0; ow < output_w; ow++) {
            int32_t iw_base = (int32_t)(ow * stride_w) - (int32_t)pad_w;
            uint32_t iw_start = (iw_base < 0) ? 0u : (uint32_t)iw_base;
            uint32_t iw_end = (iw_base + (int32_t)kernel_w > (int32_t)input_w)
                ? input_w
                : (uint32_t)(iw_base + (int32_t)kernel_w);
            int8_t *out_pixel = &dst[((oh * output_w + ow) * channels)];

            for (uint32_t c = 0; c < channels; c++) {
                out_pixel[c] = (int8_t)-128;
            }

            for (uint32_t ih = ih_start; ih < ih_end; ih++) {
                for (uint32_t iw = iw_start; iw < iw_end; iw++) {
                    const int8_t *in_pixel = &src[((ih * input_w + iw) * channels)];
                    for (uint32_t c = 0; c < channels; c++) {
                        if (in_pixel[c] > out_pixel[c]) {
                            out_pixel[c] = in_pixel[c];
                        }
                    }
                }
            }
        }
    }
}

void spatz_upsample_nearest_i8(const int8_t *src, int8_t *dst,
                               uint32_t input_h, uint32_t input_w, uint32_t channels,
                               uint32_t scale_h, uint32_t scale_w) {
    if (channels == 32u && scale_h == 2u && scale_w == 2u && src != dst) {
        spatz_upsample_nearest2x_c32_i8(src, dst, input_h, input_w);
        return;
    }

    uint32_t output_w = input_w * scale_w;

    for (uint32_t ih = 0; ih < input_h; ih++) {
        for (uint32_t iw = 0; iw < input_w; iw++) {
            const int8_t *src_pixel = &src[((ih * input_w + iw) * channels)];
            for (uint32_t sh = 0; sh < scale_h; sh++) {
                uint32_t oh = ih * scale_h + sh;
                for (uint32_t sw = 0; sw < scale_w; sw++) {
                    uint32_t ow = iw * scale_w + sw;
                    int8_t *dst_pixel = &dst[((oh * output_w + ow) * channels)];
                    spatz_vec_copy_i8(src_pixel, dst_pixel, channels);
                }
            }
        }
    }
}

void spatz_concat_c32_i8(const int8_t *src0, uint32_t c0,
                         const int8_t *src1, uint32_t c1,
                         int8_t *dst, uint32_t height, uint32_t width) {
    uint32_t pixels = height * width;
    uint32_t dst_c = c0 + c1;
    uint32_t dst_blocks = (dst_c + 31u) / 32u;
    uint32_t src0_blocks = (c0 + 31u) / 32u;
    uint32_t src1_blocks = (c1 + 31u) / 32u;

    if ((c0 & 31u) == 0u && (c1 & 31u) == 0u) {
        uint32_t src0_bytes = src0_blocks * pixels * 32u;
        uint32_t src1_bytes = src1_blocks * pixels * 32u;
        spatz_vec_copy_i8(src0, dst, src0_bytes);
        spatz_vec_copy_i8(src1, dst + src0_bytes, src1_bytes);
        return;
    }

    spatz_vec_zero_i8(dst, dst_blocks * pixels * 32u);

    for (uint32_t p = 0; p < pixels; p++) {
        for (uint32_t c = 0; c < c0; c++) {
            uint32_t src_index = (((c / 32u) * pixels + p) * 32u) + (c % 32u);
            uint32_t dst_index = (((c / 32u) * pixels + p) * 32u) + (c % 32u);
            dst[dst_index] = src0[src_index];
        }
        for (uint32_t c = 0; c < c1; c++) {
            uint32_t out_c = c0 + c;
            uint32_t src_index = (((c / 32u) * pixels + p) * 32u) + (c % 32u);
            uint32_t dst_index = (((out_c / 32u) * pixels + p) * 32u) + (out_c % 32u);
            dst[dst_index] = src1[src_index];
        }
    }
}
