#include "npu_types.h"
#include "npu_memory_map.h"
#include "idma_mm_utils.h"

/*
 * Scenario: minimal boot firmware smoke test.
 * Target: prove Snitch can execute code, access D-TCM status words, write/read
 * iDMA-compatible MMIO registers, and run one local TCDM-to-TCDM copy.
 */
#define TEST_SRC_ADDR 0x10100000 // TCDM Bank 0
#define TEST_DST_ADDR 0x10100100 // TCDM Bank 0 offset 0x100
#define TEST_LEN      64

volatile uint32_t *src_array = (volatile uint32_t*)TEST_SRC_ADDR;
volatile uint32_t *dst_array = (volatile uint32_t*)TEST_DST_ADDR;

int main(void) {
    // Seed a deterministic TCDM fixture that the copy result can compare byte-exactly.
    for (int i = 0; i < TEST_LEN/4; i++) {
        src_array[i] = 0xCAFEBABE + i;
    }

    // Clear destination first so a stale boot image cannot accidentally pass.
    for (int i = 0; i < TEST_LEN/4; i++) {
        dst_array[i] = 0;
    }

    // Check OBI/MMIO plumbing before using the DMA helper path.
    REG_WRITE(IDMA_LENGTH_LOW(IDMA_DIR_L2_TO_L1), 0x1234);
    uint32_t readback = REG_READ(IDMA_LENGTH_LOW(IDMA_DIR_L2_TO_L1));
    if (readback != 0x1234) {
        // MMIO write/read failed, halt here
        *((volatile uint32_t*)(NPU_DTCM_BASE)) = 0xBADBAD01;
        while(1);
    }

    // Exercise the iDMA-compatible local-copy API on current TCDM backend.
    idma_L1ToL1(TEST_SRC_ADDR, TEST_DST_ADDR, TEST_LEN);

    // Verify every copied word and leave debug words for cocotb on first mismatch.
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

    // Publish final signature through D-TCM so cocotb can poll without UART.
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
