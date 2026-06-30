#include "npu_memory_map.h"
#include "npu_types.h"

/*
 * Scenario: PMU host-access smoke workload.
 * Target: generate deterministic Snitch/TCDM traffic while the Python test
 * controls and reads PMU counters through the host AXI4-Lite slave port.
 */

#define STATUS_ADDR      (NPU_DTCM_BASE + 0x00u)
#define FAIL_TEST_ADDR   (NPU_DTCM_BASE + 0x08u)
#define GOT_ADDR         (NPU_DTCM_BASE + 0x10u)
#define EXPECTED_ADDR    (NPU_DTCM_BASE + 0x14u)
#define SCRATCH_WORDS    64u
#define SCRATCH_ARRAY    ((volatile uint32_t *)NPU_TCDM_BASE)

static void fail_at(uint32_t test_id, uint32_t got, uint32_t expected) {
    REG_WRITE(FAIL_TEST_ADDR, test_id);
    REG_WRITE(GOT_ADDR, got);
    REG_WRITE(EXPECTED_ADDR, expected);
    REG_WRITE(STATUS_ADDR, 0xBAD00000u | test_id);
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, 0xBAD00000u | test_id);
    while (1) {
    }
}

int main(void) {
    volatile uint32_t sum = 0;

    // Generate deterministic Snitch D-bus write traffic to Shared TCDM.
    for (uint32_t idx = 0; idx < SCRATCH_WORDS; idx++) {
        SCRATCH_ARRAY[idx] = 0x12340000u + idx;
    }

    // Generate deterministic Snitch D-bus read traffic from Shared TCDM.
    for (uint32_t idx = 0; idx < SCRATCH_WORDS; idx++) {
        sum += SCRATCH_ARRAY[idx];
    }

    if (sum == 0u) {
        fail_at(2u, 0u, 1u);
    }

    REG_WRITE(STATUS_ADDR, 0xDEADBEEFu);
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, 0xDEADBEEFu);

    while (1) {
    }

    return 0;
}
