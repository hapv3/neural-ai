#include "npu_layout_ops.h"

#if defined(NAI_TRUSTED_FIRMWARE)
#define NAI_TRUSTED_INVALID(condition) 0u
#else
#define NAI_TRUSTED_INVALID(condition) (condition)
#endif

static uint32_t checked_multiply(uint32_t lhs, uint32_t rhs, uint32_t *result)
{
    if (lhs != 0u && rhs > 0xffffffffu / lhs) return 0u;
    *result = lhs * rhs;
    return 1u;
}

static uint32_t c32_offset(uint32_t pixels, uint32_t pixel, uint32_t channel,
                           uint32_t element_bytes)
{
    return (((channel >> 5) * pixels + pixel) * 32u + (channel & 31u)) * element_bytes;
}

static void copy_element(uint8_t *destination, const uint8_t *source, uint32_t element_bytes)
{
    for (uint32_t byte = 0; byte < element_bytes; byte++) destination[byte] = source[byte];
}

uint32_t nai_copy_layout_v2(const nai_cmd_copy_layout_v2_t *command,
                            const void *source_pointer, void *destination_pointer)
{
    const uint8_t *source = (const uint8_t *)source_pointer;
    uint8_t *destination = (uint8_t *)destination_pointer;
    uint32_t element_bytes;
    uint32_t pixels;
    uint32_t channels;
    uint32_t padded_channels;
    uint32_t native_bytes;

    if (command == 0 || source == 0 || destination == 0) return 1u;
    if (command->data_type == NAI_DTYPE_I8) element_bytes = 1u;
    else if (command->data_type == NAI_DTYPE_I32) element_bytes = 4u;
    else return 1u;
    if (NAI_TRUSTED_INVALID(command->valid_channels == 0u) ||
        !checked_multiply(command->dimensions[0], command->dimensions[1], &pixels) ||
        !checked_multiply(pixels, command->dimensions[2], &pixels)) return 1u;
    channels = command->valid_channels;
    if (channels > 0xffffffffu - 31u) return 1u;
    padded_channels = (channels + 31u) & ~31u;
    if (!checked_multiply(pixels, padded_channels, &native_bytes) ||
        !checked_multiply(native_bytes, element_bytes, &native_bytes)) return 1u;

    if (NAI_TRUSTED_INVALID(command->valid_channels != command->dimensions[3])) return 1u;
    if (command->mode == NAI_COPY_C32_TO_CHW) {
        if (NAI_TRUSTED_INVALID(command->data_type != NAI_DTYPE_I8 || command->dimensions[0] != 1u ||
            channels != 144u || command->dimensions[1] != command->dimensions[2] ||
            (command->dimensions[1] != 10u && command->dimensions[1] != 20u &&
             command->dimensions[1] != 40u) ||
            command->source_row_stride != pixels * 32u ||
            command->destination_row_stride != pixels)) return 1u;
        for (uint32_t channel = 0u; channel < channels; channel++) {
            for (uint32_t pixel = 0u; pixel < pixels; pixel++) {
                destination[channel * pixels + pixel] =
                    source[c32_offset(pixels, pixel, channel, 1u)];
            }
        }
        return 0u;
    }
    if (command->mode != NAI_COPY_NHWC_TO_ROW32 &&
        command->mode != NAI_COPY_ROW32_TO_NHWC &&
        command->mode != NAI_COPY_NHWC_TO_C32 &&
        command->mode != NAI_COPY_C32_TO_NHWC) return 1u;

    if (command->mode == NAI_COPY_NHWC_TO_ROW32 || command->mode == NAI_COPY_NHWC_TO_C32) {
        for (uint32_t byte = 0; byte < native_bytes; byte++) destination[byte] = 0u;
    }
    for (uint32_t pixel = 0; pixel < pixels; pixel++) {
        for (uint32_t channel = 0; channel < channels; channel++) {
            uint32_t compact = (pixel * channels + channel) * element_bytes;
            uint32_t native = command->mode == NAI_COPY_NHWC_TO_C32 || command->mode == NAI_COPY_C32_TO_NHWC ?
                c32_offset(pixels, pixel, channel, element_bytes) :
                (pixel * padded_channels + channel) * element_bytes;
            if (command->mode == NAI_COPY_NHWC_TO_ROW32 || command->mode == NAI_COPY_NHWC_TO_C32)
                copy_element(destination + native, source + compact, element_bytes);
            else if (command->mode == NAI_COPY_ROW32_TO_NHWC || command->mode == NAI_COPY_C32_TO_NHWC)
                copy_element(destination + compact, source + native, element_bytes);
            else return 1u;
        }
    }
    return 0u;
}
