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
    return 0;
}
