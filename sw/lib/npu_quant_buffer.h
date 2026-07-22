#ifndef NPU_QUANT_BUFFER_H
#define NPU_QUANT_BUFFER_H

#include "npu_model_abi.h"

#define NAI_QUANT_CHANNELS_V1 32u

typedef enum {
    NAI_QUANT_BUFFER_OK = 0,
    NAI_QUANT_BUFFER_BAD_ARGUMENT = 1,
    NAI_QUANT_BUFFER_BAD_COUNT = 2,
    NAI_QUANT_BUFFER_BAD_PARAMETER = 3,
    NAI_QUANT_BUFFER_DMA_FAILED = 4
} nai_quant_buffer_status_t;

typedef struct {
    int32_t bias[NAI_QUANT_CHANNELS_V1];
    int32_t multiplier[NAI_QUANT_CHANNELS_V1];
    uint8_t shift[NAI_QUANT_CHANNELS_V1];
    int32_t zero_point[NAI_QUANT_CHANNELS_V1];
    int32_t clamp_min;
    int32_t clamp_max;
    uint32_t block;
} nai_quant_buffer_v1_t;

nai_quant_buffer_status_t nai_quant_buffer_decode_v1(const nai_qparam_v1_t *qparams,
                                                      uint32_t count,
                                                      uint32_t block,
                                                      nai_quant_buffer_v1_t *buffer);
nai_quant_buffer_status_t nai_quant_buffer_load_l2_v1(uint32_t qparam_address,
                                                       uint32_t count,
                                                       uint32_t block);
uint32_t nai_quant_buffer_is_loaded_v1(uint32_t block);
void nai_quant_buffer_reset_v1(void);

#endif
