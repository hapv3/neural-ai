#include "npu_types.h"

void *memset(void *destination, int value, unsigned long bytes)
{
    uint8_t *output = (uint8_t *)destination;
    for (unsigned long index = 0; index < bytes; index++) output[index] = (uint8_t)value;
    return destination;
}

void *memcpy(void *destination, const void *source, unsigned long bytes)
{
    uint8_t *output = (uint8_t *)destination;
    const uint8_t *input = (const uint8_t *)source;
    for (unsigned long index = 0; index < bytes; index++) output[index] = input[index];
    return destination;
}
