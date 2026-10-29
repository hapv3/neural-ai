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

/* Unit-stride within each long row; one call for all rows in a plane. */
void spatz_vec_copy_rows_i8(const int8_t *src, int8_t *dst,
                           uint32_t length, uint32_t runs,
                           uint32_t source_stride, uint32_t destination_stride);

#define NAI_LONG_COPY_1D_MIN_BYTES 256u

/* Unit-stride e8 supports byte alignment and VL tails. Only change grouping
 * when the ranges are disjoint: overlapping copies retain their old ordering,
 * rather than accidentally promising memmove semantics. Pointer differences
 * avoid overflowing an RV32 end address. */
static inline void spatz_copy_1d_i8(const int8_t *src, int8_t *dst, uint32_t length)
{
    const unsigned long source = (unsigned long)src;
    const unsigned long destination = (unsigned long)dst;
    const unsigned long distance = source < destination ?
        destination - source : source - destination;
    if (length >= NAI_LONG_COPY_1D_MIN_BYTES && length <= distance) {
        spatz_vec_copy_rows_i8(src, dst, length, 1u, 0u, 0u);
    } else {
        spatz_vec_copy_i8(src, dst, length);
    }
}

/* Regor/loader already prove the complete TCDM spans in the trusted runtime.
 * Keep wide arithmetic for the standalone/legacy, untrusted helper. */
#if defined(NAI_TRUSTED_FIRMWARE) && defined(__riscv) && __riscv_xlen == 32
typedef uint32_t spatz_copy_span_t;
#else
typedef uint64_t spatz_copy_span_t;
#endif

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
    const spatz_copy_span_t source_span = copy.length +
        (spatz_copy_span_t)(copy.repetitions_2 - 1u) * copy.source_stride_2 +
        (spatz_copy_span_t)(copy.repetitions_3 - 1u) * copy.source_stride_3;
    const spatz_copy_span_t destination_span = copy.length +
        (spatz_copy_span_t)(copy.repetitions_2 - 1u) * copy.destination_stride_2 +
        (spatz_copy_span_t)(copy.repetitions_3 - 1u) * copy.destination_stride_3;
    const spatz_copy_span_t source_begin = (unsigned long)src;
    const spatz_copy_span_t destination_begin = (unsigned long)dst;
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
    const uint32_t row_batch = !(alignment & 3u) && copy.length > 32u &&
        (copy.repetitions_2 == 1u || copy.destination_stride_2 >= copy.length) && disjoint;
    for (uint32_t plane = 0; plane < copy.repetitions_3; plane++) {
        const int8_t *source_row = src + plane * copy.source_stride_3;
        int8_t *destination_row = dst + plane * copy.destination_stride_3;
        if (batch) {
            spatz_vec_copy_blocks_i32(source_row, destination_row, copy.length / 4u,
                copy.repetitions_2, copy.source_stride_2, copy.destination_stride_2);
        } else if (row_batch) {
            spatz_vec_copy_rows_i8(source_row, destination_row, copy.length,
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
