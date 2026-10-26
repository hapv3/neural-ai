#ifndef SPATZ_COPY_H
#define SPATZ_COPY_H

#include "spatz_ops.h"

/* The descriptor denotes runs, never their bounding rectangle. */
typedef struct {
    uint32_t length;
    uint32_t source_stride_2;
    uint32_t destination_stride_2;
    uint32_t repetitions_2;
    uint32_t source_stride_3;
    uint32_t destination_stride_3;
    uint32_t repetitions_3;
} spatz_copy_geometry_t;

/* Vectorize across pixels in 32-bit words, not single strided bytes. */
void spatz_vec_copy_blocks_i32(const int8_t *src, int8_t *dst,
                              uint32_t words, uint32_t runs,
                              uint32_t source_stride, uint32_t destination_stride);

static inline void spatz_copy_affine_i8(const int8_t *src, int8_t *dst,
                                       spatz_copy_geometry_t copy)
{
    if (!copy.length || !copy.repetitions_2 || !copy.repetitions_3) return;
    uint32_t alignment = (uint32_t)(unsigned long)src |
        (uint32_t)(unsigned long)dst | copy.length;
    if (copy.repetitions_2 > 1u)
        alignment |= copy.source_stride_2 | copy.destination_stride_2;
    if (copy.repetitions_3 > 1u)
        alignment |= copy.source_stride_3 | copy.destination_stride_3;
    const uint64_t source_span = copy.length +
        (uint64_t)(copy.repetitions_2 - 1u) * copy.source_stride_2 +
        (uint64_t)(copy.repetitions_3 - 1u) * copy.source_stride_3;
    const uint64_t destination_span = copy.length +
        (uint64_t)(copy.repetitions_2 - 1u) * copy.destination_stride_2 +
        (uint64_t)(copy.repetitions_3 - 1u) * copy.destination_stride_3;
    const uint64_t source_begin = (unsigned long)src;
    const uint64_t destination_begin = (unsigned long)dst;
    const uint32_t disjoint = source_begin + source_span <= destination_begin ||
        destination_begin + destination_span <= source_begin;
    if (disjoint) {
        if (copy.repetitions_3 > 1u &&
            (uint64_t)copy.source_stride_2 * copy.repetitions_2 == copy.source_stride_3 &&
            (uint64_t)copy.destination_stride_2 * copy.repetitions_2 == copy.destination_stride_3 &&
            (uint64_t)copy.repetitions_2 * copy.repetitions_3 <= 0xffffffffu) {
            copy.repetitions_2 *= copy.repetitions_3;
            copy.repetitions_3 = 1u;
        }
        /* The compiler does this before scheduling too. Keep old descriptors
         * efficient, but do not reorder overlapping legacy transfers. */
        for (uint32_t dimension = 0; dimension < 2u; dimension++) {
            if (copy.repetitions_2 == 1u ||
                (copy.source_stride_2 == copy.length &&
                 copy.destination_stride_2 == copy.length &&
                 (uint64_t)copy.length * copy.repetitions_2 <= 0xffffffffu)) {
                copy.length *= copy.repetitions_2;
                copy.source_stride_2 = copy.source_stride_3;
                copy.destination_stride_2 = copy.destination_stride_3;
                copy.repetitions_2 = copy.repetitions_3;
                copy.repetitions_3 = 1u;
                copy.source_stride_3 = copy.destination_stride_3 = 0u;
            }
        }
    }
    const uint32_t batch = !(alignment & 3u) && copy.length <= 32u &&
        copy.repetitions_2 > 1u && copy.destination_stride_2 >= copy.length && disjoint;
    for (uint32_t plane = 0; plane < copy.repetitions_3; plane++) {
        const int8_t *source_row = src + plane * copy.source_stride_3;
        int8_t *destination_row = dst + plane * copy.destination_stride_3;
        if (batch) {
            spatz_vec_copy_blocks_i32(source_row, destination_row, copy.length / 4u,
                copy.repetitions_2, copy.source_stride_2, copy.destination_stride_2);
        } else {
            /* Preserve the legacy run order for byte tails or overlapping
             * views. In particular this is not a new memmove contract. */
            for (uint32_t run = 0; run < copy.repetitions_2; run++)
                spatz_vec_copy_i8(source_row + run * copy.source_stride_2,
                    destination_row + run * copy.destination_stride_2, copy.length);
        }
    }
}

#endif
