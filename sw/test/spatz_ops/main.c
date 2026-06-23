#include "npu_types.h"
#include "spatz_ops.h"

/*
 * Scenario: C-callable Spatz operator library smoke/regression test.
 * Target: prove scheduler-facing wrappers produce exact data for copy, ReLU,
 * and INT32->INT8 requant before graph firmware is allowed to depend on them.
 */
#define PASS_SIGNATURE 0xDEADBEEFu
#define FAIL_SIGNATURE 0xBAD20000u

#define SIG_STATUS     (*(volatile uint32_t *)0x10008000u)
#define SIG_PASS_COUNT (*(volatile uint32_t *)0x10008004u)
#define SIG_FAIL_TEST  (*(volatile uint32_t *)0x10008008u)
#define SIG_FAIL_INDEX (*(volatile uint32_t *)0x1000800Cu)
#define SIG_FAIL_GOT   (*(volatile uint32_t *)0x10008010u)
#define SIG_FAIL_EXP   (*(volatile uint32_t *)0x10008014u)

#define SRC_I8       ((volatile int8_t *)0x10100000u)
#define DST_I8       ((volatile int8_t *)0x10100100u)
#define RELU_I8      ((volatile int8_t *)0x10100200u)
#define SRC_I32      ((volatile int32_t *)0x10100300u)
#define DST_REQUANT  ((volatile int8_t *)0x10100500u)

#define VL 32u

static void fail(uint32_t test_id, uint32_t index, int32_t got, int32_t expected) {
    // Standard status/debug page lets cocotb report failing op, element, got, expected.
    SIG_FAIL_TEST = test_id;
    SIG_FAIL_INDEX = index;
    SIG_FAIL_GOT = (uint32_t)got;
    SIG_FAIL_EXP = (uint32_t)expected;
    SIG_STATUS = FAIL_SIGNATURE | test_id;
    while (1) {
    }
}

static int32_t clamp_i32(int32_t value, int32_t min_val, int32_t max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

int main(void) {
    SIG_STATUS = 0;
    SIG_PASS_COUNT = 0;

    // Test 1: vector copy must preserve signed int8 payload exactly.
    for (uint32_t i = 0; i < VL; i++) {
        SRC_I8[i] = (int8_t)((int32_t)i - 16);
        DST_I8[i] = 0;
    }

    spatz_vec_copy_i8((const int8_t *)SRC_I8, (int8_t *)DST_I8, VL);
    // Test 2: ReLU is in-place and clamps only negative int8 lanes.
    for (uint32_t i = 0; i < VL; i++) {
        if (DST_I8[i] != SRC_I8[i]) {
            fail(1, i, DST_I8[i], SRC_I8[i]);
        }
    }
    SIG_PASS_COUNT = SIG_PASS_COUNT + 1;

    // Test 3: requant uses integer multiply, arithmetic shift, and int8 clamp.
    for (uint32_t i = 0; i < VL; i++) {
        RELU_I8[i] = (int8_t)((int32_t)i - 12);
    }

    spatz_vec_relu_i8((int8_t *)RELU_I8, VL);
    for (uint32_t i = 0; i < VL; i++) {
        int8_t expected = (i < 12) ? 0 : (int8_t)((int32_t)i - 12);
        if (RELU_I8[i] != expected) {
            fail(2, i, RELU_I8[i], expected);
        }
    }
    SIG_PASS_COUNT = SIG_PASS_COUNT + 1;

    for (uint32_t i = 0; i < VL; i++) {
        SRC_I32[i] = ((int32_t)i - 16) * 37;
        DST_REQUANT[i] = 0;
    }

    spatz_requant_i32_to_i8((const int32_t *)SRC_I32, (int8_t *)DST_REQUANT,
                            VL, 2, 3, -20, 31);
    for (uint32_t i = 0; i < VL; i++) {
        int32_t expected = clamp_i32((SRC_I32[i] * 2) >> 3, -20, 31);
        if (DST_REQUANT[i] != (int8_t)expected) {
            fail(3, i, DST_REQUANT[i], expected);
        }
    }
    SIG_PASS_COUNT = SIG_PASS_COUNT + 1;

    SIG_STATUS = PASS_SIGNATURE;
    while (1) {
    }
}
