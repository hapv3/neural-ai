#include "npu_layout_ops.h"

#include <assert.h>
#include <string.h>

static void configure(nai_cmd_copy_layout_v2_t *command, uint16_t mode, uint32_t channels)
{
    memset(command, 0, sizeof(*command));
    command->mode = mode;
    command->data_type = NAI_DTYPE_I8;
    command->dimensions[0] = 1;
    command->dimensions[1] = 1;
    command->dimensions[2] = 2;
    command->dimensions[3] = channels;
    command->valid_channels = channels;
}

static void check_round_trip(uint16_t mode_to_native, uint16_t mode_to_compact,
                             uint32_t channels, uint32_t pixels, uint16_t data_type,
                             uint32_t source_offset, uint32_t destination_offset)
{
    const uint32_t element_bytes = data_type == NAI_DTYPE_I8 ? 1u : 4u;
    const uint32_t compact_bytes = pixels * channels * element_bytes;
    const uint32_t padded_channels = (channels + 31u) & ~31u;
    const uint32_t native_bytes = pixels * padded_channels * element_bytes;
    uint8_t compact[704];
    uint8_t native_storage[704];
    uint8_t output[704];
    nai_cmd_copy_layout_v2_t command;

    assert(compact_bytes <= sizeof(compact));
    assert(native_bytes + destination_offset <= sizeof(native_storage));
    assert(compact_bytes + source_offset <= sizeof(compact));
    for (uint32_t index = 0; index < compact_bytes; index++)
        compact[source_offset + index] = (uint8_t)(index * 13u + 7u);

    configure(&command, mode_to_native, channels);
    command.data_type = data_type;
    command.dimensions[2] = pixels;
    command.dimensions[3] = channels;
    command.valid_channels = channels;
    memset(native_storage, 0xa5, sizeof(native_storage));
    assert(nai_copy_layout_v2(&command, compact + source_offset,
                              native_storage + destination_offset) == 0u);

    command.mode = mode_to_compact;
    memset(output, 0, sizeof(output));
    assert(nai_copy_layout_v2(&command, native_storage + destination_offset, output) == 0u);
    assert(memcmp(output, compact + source_offset, compact_bytes) == 0);
}

int main(void)
{
    uint8_t compact[66];
    uint8_t native[128];
    uint8_t output[66];
    nai_cmd_copy_layout_v2_t command;

    for (uint32_t index = 0; index < sizeof(compact); index++) compact[index] = (uint8_t)(index + 1u);

    configure(&command, NAI_COPY_NHWC_TO_ROW32, 33);
    memset(native, 0xa5, sizeof(native));
    assert(nai_copy_layout_v2(&command, compact, native) == 0u);
    assert(memcmp(native, compact, 33) == 0);
    assert(memcmp(native + 64, compact + 33, 33) == 0);
    for (uint32_t index = 33; index < 64; index++) assert(native[index] == 0u);
    for (uint32_t index = 97; index < 128; index++) assert(native[index] == 0u);
    command.mode = NAI_COPY_ROW32_TO_NHWC;
    memset(output, 0, sizeof(output));
    assert(nai_copy_layout_v2(&command, native, output) == 0u);
    assert(memcmp(output, compact, sizeof(compact)) == 0);

    configure(&command, NAI_COPY_NHWC_TO_C32, 33);
    memset(native, 0xa5, sizeof(native));
    assert(nai_copy_layout_v2(&command, compact, native) == 0u);
    assert(memcmp(native, compact, 32) == 0);
    assert(memcmp(native + 32, compact + 33, 32) == 0);
    assert(native[64] == compact[32]);
    assert(native[96] == compact[65]);
    command.mode = NAI_COPY_C32_TO_NHWC;
    memset(output, 0, sizeof(output));
    assert(nai_copy_layout_v2(&command, native, output) == 0u);
    assert(memcmp(output, compact, sizeof(compact)) == 0);

    /* Compact rows are intentionally allowed to be unaligned and shorter than
       one 32-byte AXI beat.  Exercise both row32 and C32 mappings at offsets
       that would be rejected by an over-constrained DMA endpoint contract. */
    check_round_trip(NAI_COPY_NHWC_TO_ROW32, NAI_COPY_ROW32_TO_NHWC,
                     3u, 6u, NAI_DTYPE_I8, 1u, 3u);
    check_round_trip(NAI_COPY_NHWC_TO_ROW32, NAI_COPY_ROW32_TO_NHWC,
                     31u, 5u, NAI_DTYPE_I8, 7u, 5u);
    check_round_trip(NAI_COPY_NHWC_TO_C32, NAI_COPY_C32_TO_NHWC,
                     3u, 6u, NAI_DTYPE_I8, 11u, 9u);
    check_round_trip(NAI_COPY_NHWC_TO_C32, NAI_COPY_C32_TO_NHWC,
                     31u, 5u, NAI_DTYPE_I32, 13u, 1u);

    configure(&command, NAI_COPY_NHWC_TO_ROW32, 3u);
    command.dimensions[2] = 1u;
    command.dimensions[3] = 4u;
    command.valid_channels = 3u;
    assert(nai_copy_layout_v2(&command, compact, native) != 0u);
    command.valid_channels = 0xffffffffu;
    command.dimensions[3] = 0xffffffffu;
    assert(nai_copy_layout_v2(&command, compact, native) != 0u);
    return 0;
}
