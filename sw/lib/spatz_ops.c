#include "spatz_ops.h"

static int8_t pack_i8(int32_t value) {
    return (int8_t)value;
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
