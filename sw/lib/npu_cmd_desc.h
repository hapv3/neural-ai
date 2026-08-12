#ifndef NPU_CMD_DESC_H
#define NPU_CMD_DESC_H

#include "npu_types.h"

#define NPU_CMD_FAIL_NONE          0x00000000u
#define NPU_CMD_FAIL_BAD_INVOCATION 0xBADCD00Au
#define NPU_CMD_FAIL_BAD_MODEL      0xBADCD00Bu
#define NPU_CMD_FAIL_BAD_BINDING    0xBADCD00Cu
#define NPU_CMD_FAIL_V2_DISPATCH    0xBADCD00Du

uint32_t npu_cmd_dispatch_from_ctrl(void);

#endif
