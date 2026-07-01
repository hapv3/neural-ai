#include "npu_cmd_desc.h"
#include "npu_memory_map.h"

/*
 * Scenario: legacy systolic matmul application regression.
 * Target: run one M=64 systolic GEMM fixture from L2, return Mx32 INT32 output
 * to L2, and notify the host through the interrupt controller.
 */
#define PASS_SIGNATURE   0xDEADBEEFu
#define FAIL_SIGNATURE   0xBAD30000u

int main() {
    uint32_t status = npu_cmd_dispatch_from_ctrl();

    REG_WRITE(NPU_IRQ_HOST_NOTIFY, status == NPU_CMD_FAIL_NONE ? PASS_SIGNATURE : (FAIL_SIGNATURE | (status & 0xFFFFu)));
    while (1) {
    }

    return 0;
}
