#include "npu_quant_buffer.h"

#ifndef NAI_QUANT_BUFFER_HOST_TEST
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "npu_memory_map.h"

static nai_quant_buffer_v1_t g_quant_buffer;
static uint32_t g_quant_buffer_loaded;
#endif

static uint32_t reserved_zero(const nai_qparam_v1_t *qparam)
{
    return qparam->reserved[0] == 0u && qparam->reserved[1] == 0u;
}

nai_quant_buffer_status_t nai_quant_buffer_decode_v1(const nai_qparam_v1_t *qparams,
                                                      uint32_t count,
                                                      uint32_t block,
                                                      nai_quant_buffer_v1_t *buffer)
{
    if (qparams == 0 || buffer == 0) return NAI_QUANT_BUFFER_BAD_ARGUMENT;
    if (count != NAI_QUANT_CHANNELS_V1) return NAI_QUANT_BUFFER_BAD_COUNT;
    if (qparams[0].clamp_min < -128 || qparams[0].clamp_max > 127 ||
        qparams[0].clamp_min > qparams[0].clamp_max) return NAI_QUANT_BUFFER_BAD_PARAMETER;

    for (uint32_t channel = 0; channel < NAI_QUANT_CHANNELS_V1; channel++) {
        const nai_qparam_v1_t *qparam = &qparams[channel];
        if (!reserved_zero(qparam) || qparam->shift > 31u ||
            qparam->clamp_min != qparams[0].clamp_min ||
            qparam->clamp_max != qparams[0].clamp_max) return NAI_QUANT_BUFFER_BAD_PARAMETER;
        buffer->bias[channel] = qparam->bias;
        buffer->multiplier[channel] = qparam->multiplier;
        buffer->shift[channel] = (uint8_t)qparam->shift;
        buffer->zero_point[channel] = qparam->zero_point;
    }
    buffer->clamp_min = qparams[0].clamp_min;
    buffer->clamp_max = qparams[0].clamp_max;
    buffer->block = block;
    return NAI_QUANT_BUFFER_OK;
}

#ifndef NAI_QUANT_BUFFER_HOST_TEST
nai_quant_buffer_status_t nai_quant_buffer_load_l2_v1(uint32_t qparam_address,
                                                       uint32_t count,
                                                       uint32_t block)
{
    nai_quant_buffer_status_t status;
    uint32_t bytes;

    g_quant_buffer_loaded = 0u;
    systolic_requant_disable();
    if (count != NAI_QUANT_CHANNELS_V1) return NAI_QUANT_BUFFER_BAD_COUNT;
    bytes = count * (uint32_t)sizeof(nai_qparam_v1_t);
    if ((qparam_address & (NAI_ALIGNMENT_BYTES - 1u)) != 0u ||
        qparam_address > 0xffffffffu - bytes) return NAI_QUANT_BUFFER_BAD_ARGUMENT;
    if (!idma_memcpy_blocking(qparam_address, NPU_CMD_TCDM_BASE, bytes))
        return NAI_QUANT_BUFFER_DMA_FAILED;
    status = nai_quant_buffer_decode_v1(
        (const nai_qparam_v1_t *)(unsigned long)NPU_CMD_TCDM_BASE,
        count, block, &g_quant_buffer);
    if (status != NAI_QUANT_BUFFER_OK) return status;
    systolic_requant_config_per_channel(g_quant_buffer.bias, g_quant_buffer.multiplier,
        g_quant_buffer.shift, g_quant_buffer.zero_point,
        g_quant_buffer.clamp_min, g_quant_buffer.clamp_max);
    g_quant_buffer_loaded = 1u;
    return NAI_QUANT_BUFFER_OK;
}

uint32_t nai_quant_buffer_is_loaded_v1(uint32_t block)
{
    return g_quant_buffer_loaded != 0u && g_quant_buffer.block == block;
}

void nai_quant_buffer_reset_v1(void)
{
    g_quant_buffer_loaded = 0u;
    systolic_requant_disable();
}
#endif
