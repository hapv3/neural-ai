#include "npu_cmd_desc.h"
#include "npu_memory_map.h"
#include "npu_model_runtime.h"

uint32_t npu_cmd_dispatch_from_ctrl(void)
{
    uint32_t invocation_base = REG_READ(NPU_CMD_L2_BASE);
    uint32_t invocation_bytes = REG_READ(NPU_CMD_TOTAL_BYTES);
    uint32_t staging_base = REG_READ(NPU_CMD_TCDM_BASE_REG);
    uint32_t staging_bytes = REG_READ(NPU_CMD_TCDM_BYTES);

    REG_WRITE(NPU_CMD_FAIL_CODE, NPU_CMD_FAIL_NONE);
    REG_WRITE(NPU_CMD_FAIL_PTR, 0u);
    REG_WRITE(NPU_CMD_DONE_COUNT, 0u);
    REG_WRITE(NPU_CMD_STATUS, NPU_CMD_STATUS_LOADING);

    return nai_runtime_dispatch_from_ctrl(invocation_base, invocation_bytes,
                                          staging_base, staging_bytes);
}
