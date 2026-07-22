#include "npu_cmd_desc.h"
#include "npu_memory_map.h"

#define NAI_RUNTIME_PASS_SIGNATURE 0xDEADBEEFu
#define NAI_RUNTIME_FAIL_SIGNATURE 0xBAD40000u

int main(void)
{
    uint32_t status = npu_cmd_dispatch_from_ctrl();
    uint32_t signature = status == NPU_CMD_FAIL_NONE ? NAI_RUNTIME_PASS_SIGNATURE :
        (NAI_RUNTIME_FAIL_SIGNATURE | (status & 0xffffu));
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, signature);
    while (1) {
    }
    return 0;
}
