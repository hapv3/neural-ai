#include "spatz_copy.h"

#include <assert.h>
#include <string.h>

static int8_t s_memory[32768];
static int8_t s_expected[32768];
static uint32_t s_batches;
static uint32_t s_copies;

/* Host contract tests for dispatch/geometry. The real RVV kernel is tested
 * separately by the Spatz operator-library firmware gate. */
void spatz_vec_copy_i8(const int8_t *src, int8_t *dst, uint32_t count)
{
    s_copies++;
    for (uint32_t i = 0; i < count; i++) dst[i] = src[i];
}

void spatz_vec_copy_blocks_i32(const int8_t *src, int8_t *dst,
                              uint32_t words, uint32_t runs,
                              uint32_t source_stride, uint32_t destination_stride)
{
    s_batches++;
    for (uint32_t word = 0; word < words; word++)
        for (uint32_t run = 0; run < runs; run++)
            for (uint32_t byte = 0; byte < 4; byte++)
                dst[run * destination_stride + word * 4 + byte] =
                    src[run * source_stride + word * 4 + byte];
}

static void check(spatz_copy_geometry_t copy, uint32_t source, uint32_t destination)
{
    for (uint32_t i = 0; i < sizeof(s_memory); i++) s_memory[i] = (int8_t)(i * 17u + 93u);
    memcpy(s_expected, s_memory, sizeof(s_memory));
    for (uint32_t plane = 0; plane < copy.repetitions_3; plane++)
        for (uint32_t run = 0; run < copy.repetitions_2; run++)
            for (uint32_t byte = 0; byte < copy.length; byte++)
                s_expected[destination + plane * copy.destination_stride_3 +
                    run * copy.destination_stride_2 + byte] =
                    s_expected[source + plane * copy.source_stride_3 +
                    run * copy.source_stride_2 + byte];
    s_batches = s_copies = 0;
    spatz_copy_affine_i8(s_memory + source, s_memory + destination, copy);
    assert(memcmp(s_memory, s_expected, sizeof(s_memory)) == 0);
}

int main(void)
{
    check((spatz_copy_geometry_t){16, 32, 16, 127, 4096, 2048, 3}, 0, 16384);
    assert(s_batches == 3 && s_copies == 0);
    check((spatz_copy_geometry_t){16, 16, 32, 127, 2048, 4096, 3}, 0, 16384);
    assert(s_batches == 3 && s_copies == 0);
    check((spatz_copy_geometry_t){16, 0, 32, 255, 0, 0, 1}, 0, 16384);
    assert(s_batches == 1 && s_copies == 0);
    check((spatz_copy_geometry_t){32, 32, 32, 20, 640, 640, 9}, 0, 16384);
    assert(s_batches == 0 && s_copies == 1);
    check((spatz_copy_geometry_t){16, 32, 32, 16, 512, 544, 3}, 0, 16384);
    assert(s_batches == 3);
    check((spatz_copy_geometry_t){16, 32, 16, 19, 608, 304, 3}, 0, 16384);
    assert(s_batches == 1 && s_copies == 0);
    check((spatz_copy_geometry_t){15, 32, 32, 10, 0, 0, 1}, 1, 16385);
    assert(s_batches == 0 && s_copies == 10);
    check((spatz_copy_geometry_t){16, 32, 32, 10, 0, 0, 1}, 0, 32);
    assert(s_batches == 0 && s_copies == 10);
    for (uint32_t trial = 0; trial < 1000; trial++) {
        uint32_t length = 1u + trial % 65u;
        uint32_t runs = 1u + trial % 17u;
        uint32_t source_stride = length + trial % 9u;
        uint32_t destination_stride = length + trial % 13u;
        check((spatz_copy_geometry_t){length, trial % 11u ? source_stride : 0,
            destination_stride, runs, source_stride * runs + 8u,
            destination_stride * runs + 12u, 1u + trial % 5u}, trial % 4u, 16384u + trial % 4u);
    }
    return 0;
}
