#include "npu_types.h"
#include "npu_memory_map.h"
#include "idma_mm_utils.h"

// For testing purposes, we use Shared Data TCDM
#define TEST_SRC_ADDR 0x10100000 // TCDM Bank 0
#define TEST_DST_ADDR 0x10100100 // TCDM Bank 0 offset 0x100
#define TEST_LEN      64

volatile uint32_t *src_array = (volatile uint32_t*)TEST_SRC_ADDR;
volatile uint32_t *dst_array = (volatile uint32_t*)TEST_DST_ADDR;

int main(void) {
    // 1. Write some test data to the Source Address
    for (int i = 0; i < TEST_LEN/4; i++) {
        src_array[i] = 0xCAFEBABE + i;
    }

    // 2. Clear Destination Address
    for (int i = 0; i < TEST_LEN/4; i++) {
        dst_array[i] = 0;
    }

    // 3. Test writing to iDMA MMIO registers via OBI
    REG_WRITE(IDMA_LENGTH_LOW(IDMA_DIR_L2_TO_L1), 0x1234);
    uint32_t readback = REG_READ(IDMA_LENGTH_LOW(IDMA_DIR_L2_TO_L1));
    if (readback != 0x1234) {
        // MMIO write/read failed, halt here
        *((volatile uint32_t*)(NPU_DTCM_BASE)) = 0xBADBAD01;
        while(1);
    }

    // 4. Test local copy through the iDMA-compatible runtime API
    idma_L1ToL1(TEST_SRC_ADDR, TEST_DST_ADDR, TEST_LEN);

    // 5. Verify DMA Transfer Result
    int success = 1;
    for (int i = 0; i < TEST_LEN/4; i++) {
        if (dst_array[i] != src_array[i]) {
            success = 0;
            *((volatile uint32_t*)(NPU_DTCM_BASE + 0x10)) = dst_array[i];
            *((volatile uint32_t*)(NPU_DTCM_BASE + 0x14)) = src_array[i];
            *((volatile uint32_t*)(NPU_DTCM_BASE + 0x18)) = i;
            break;
        }
    }

    // 6. Signal completion to D-TCM (so testbench can backdoor read it)
    if (success) {
        *((volatile uint32_t*)(NPU_DTCM_BASE)) = 0xDEADBEEF; // Success signature
    } else {
        *((volatile uint32_t*)(NPU_DTCM_BASE)) = 0xBADBAD00; // Failure signature
    }

    // Sleep or idle
    while(1) {
        // Halt
    }

    return 0;
}
