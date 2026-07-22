#ifndef NPU_MODEL_RUNTIME_H
#define NPU_MODEL_RUNTIME_H

#include "npu_types.h"

uint32_t nai_runtime_dispatch_from_ctrl(uint32_t invocation_base,
                                        uint32_t invocation_bytes,
                                        uint32_t staging_base,
                                        uint32_t staging_bytes);

#endif
