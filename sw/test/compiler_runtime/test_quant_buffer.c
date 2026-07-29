#include "npu_quant_buffer.h"

#include <assert.h>

void systolic_requant_disable(void)
{
}

void systolic_requant_config_per_channel(const int32_t *bias,
                                         const int32_t *multiplier,
                                         const uint8_t *shift,
                                         const int32_t *zero_point,
                                         int32_t clamp_min,
                                         int32_t clamp_max)
{
    (void)bias;
    (void)multiplier;
    (void)shift;
    (void)zero_point;
    (void)clamp_min;
    (void)clamp_max;
}

static void make_qparams(nai_qparam_v1_t qparams[NAI_QUANT_CHANNELS_V1])
{
    for (uint32_t channel = 0; channel < NAI_QUANT_CHANNELS_V1; channel++) {
        qparams[channel] = (nai_qparam_v1_t){
            (int32_t)channel - 16,
            (int32_t)(channel + 1u),
            channel & 7u,
            (int32_t)(channel % 5u) - 2,
            -100,
            100,
            {0, 0}
        };
    }
}

int main(void)
{
    nai_qparam_v1_t qparams[NAI_QUANT_CHANNELS_V1];
    nai_quant_buffer_v1_t buffer;

    make_qparams(qparams);
    assert(nai_quant_buffer_decode_v1(qparams, NAI_QUANT_CHANNELS_V1, 7,
        &buffer) == NAI_QUANT_BUFFER_OK);
    assert(buffer.block == 7u);
    assert(buffer.bias[0] == -16 && buffer.bias[31] == 15);
    assert(buffer.multiplier[31] == 32);
    assert(buffer.shift[31] == 7u);
    assert(buffer.zero_point[31] == -1);
    assert(buffer.clamp_min == -100 && buffer.clamp_max == 100);

    assert(nai_quant_buffer_decode_v1(qparams, 31, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_COUNT);
    qparams[3].reserved[0] = 1;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    qparams[3].reserved[0] = 0;
    qparams[9].shift = 32;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    qparams[9].shift = 1;
    qparams[20].clamp_max = 99;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    make_qparams(qparams);
    qparams[0].clamp_min = -129;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    qparams[0].clamp_min = -100;
    qparams[0].clamp_max = 128;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    qparams[0].clamp_min = 101;
    qparams[0].clamp_max = 100;
    assert(nai_quant_buffer_decode_v1(qparams, 32, 0,
        &buffer) == NAI_QUANT_BUFFER_BAD_PARAMETER);
    return 0;
}
