#include "npu_types.h"
#include "npu_memory_map.h"
#include "spatz_ops.h"

/*
 * Scenario: C-callable Spatz/AFU operator library smoke/regression test.
 * Target: prove scheduler-facing wrappers produce exact data for the non-Conv
 * operators needed by the Micro-YOLOv8 raw-head path.
 */
#define PASS_SIGNATURE 0xDEADBEEFu
#define FAIL_SIGNATURE 0xBAD20000u

#define SPATZ_OP_TEST_ALL      0u
#define SPATZ_OP_TEST_COPY     1u
#define SPATZ_OP_TEST_RELU     2u
#define SPATZ_OP_TEST_REQUANT  3u
#define SPATZ_OP_TEST_ADD      4u
#define SPATZ_OP_TEST_MUL      5u
#define SPATZ_OP_TEST_LOGISTIC 6u
#define SPATZ_OP_TEST_MAXPOOL  7u
#define SPATZ_OP_TEST_UPSAMPLE 8u
#define SPATZ_OP_TEST_CONCAT   9u

#ifndef SPATZ_OP_TEST_ID
#define SPATZ_OP_TEST_ID SPATZ_OP_TEST_ALL
#endif

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

#define ADD_LHS      ((volatile int8_t *)0x10100600u)
#define ADD_RHS      ((volatile int8_t *)0x10100700u)
#define ADD_DST      ((volatile int8_t *)0x10100800u)
#define MUL_LHS      ((volatile int8_t *)0x10100900u)
#define MUL_RHS      ((volatile int8_t *)0x10100A00u)
#define MUL_DST      ((volatile int8_t *)0x10100B00u)
#define LOG_SRC      ((volatile int8_t *)0x10100C00u)
#define LOG_DST      ((volatile int8_t *)0x10100D00u)
#define LOG_LUT      ((volatile uint8_t *)0x10100E00u)
#define POOL_SRC     ((volatile int8_t *)0x10101000u)
#define POOL_DST     ((volatile int8_t *)0x10101200u)
#define UP_SRC       ((volatile int8_t *)0x10101400u)
#define UP_DST       ((volatile int8_t *)0x10101500u)
#define CONCAT_SRC0  ((volatile int8_t *)0x10101800u)
#define CONCAT_SRC1  ((volatile int8_t *)0x10101A00u)
#define CONCAT_DST   ((volatile int8_t *)0x10101C00u)

#define VL 32u
#define POOL_H 4u
#define POOL_W 5u
#define POOL_C 2u
#define POOL_K 5u
#define POOL_PAD 2u
#define UP_H 2u
#define UP_W 3u
#define UP_C 2u
#define UP_SCALE 2u
#define CONCAT_H 2u
#define CONCAT_W 3u
#define CONCAT_C0 32u
#define CONCAT_C1 32u

static void fail(uint32_t test_id, uint32_t index, int32_t got, int32_t expected) {
    // Standard status/debug page lets cocotb report failing op, element, got, expected.
    SIG_FAIL_TEST = test_id;
    SIG_FAIL_INDEX = index;
    SIG_FAIL_GOT = (uint32_t)got;
    SIG_FAIL_EXP = (uint32_t)expected;
    SIG_STATUS = FAIL_SIGNATURE | test_id;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, FAIL_SIGNATURE | test_id);
    while (1) {
    }
}

static int32_t clamp_i32(int32_t value, int32_t min_val, int32_t max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

static int8_t pool_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)((h * 11u + w * 7u + c * 5u) % 31u) - 15);
}

static int8_t expected_pool_value(uint32_t oh, uint32_t ow, uint32_t c) {
    int8_t best = (int8_t)-128;
    for (uint32_t kh = 0; kh < POOL_K; kh++) {
        int32_t ih = (int32_t)(oh + kh) - (int32_t)POOL_PAD;
        if (ih < 0 || ih >= (int32_t)POOL_H) {
            continue;
        }
        for (uint32_t kw = 0; kw < POOL_K; kw++) {
            int32_t iw = (int32_t)(ow + kw) - (int32_t)POOL_PAD;
            if (iw < 0 || iw >= (int32_t)POOL_W) {
                continue;
            }
            int8_t value = pool_input_value((uint32_t)ih, (uint32_t)iw, c);
            if (value > best) {
                best = value;
            }
        }
    }
    return best;
}

static int8_t up_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)(h * 17u + w * 9u + c * 3u) - 20);
}

static void mark_pass(void) {
    SIG_PASS_COUNT = SIG_PASS_COUNT + 1;
}

static void run_copy(void) {
    for (uint32_t i = 0; i < VL; i++) {
        SRC_I8[i] = (int8_t)((int32_t)i - 16);
        DST_I8[i] = 0;
    }

    spatz_vec_copy_i8((const int8_t *)SRC_I8, (int8_t *)DST_I8, VL);
    for (uint32_t i = 0; i < VL; i++) {
        if (DST_I8[i] != SRC_I8[i]) {
            fail(1, i, DST_I8[i], SRC_I8[i]);
        }
    }
    mark_pass();
}

static void run_relu(void) {
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
    mark_pass();
}

static void run_requant(void) {
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
    mark_pass();
}

static void run_add(void) {
    for (uint32_t i = 0; i < VL; i++) {
        ADD_LHS[i] = (int8_t)((int32_t)i - 16);
        ADD_RHS[i] = (int8_t)((int32_t)(i % 9u) - 4);
        ADD_DST[i] = 0;
    }
    spatz_add_i8((const int8_t *)ADD_LHS, (const int8_t *)ADD_RHS,
                 (int8_t *)ADD_DST, VL, -12, 18);
    for (uint32_t i = 0; i < VL; i++) {
        int32_t expected = clamp_i32((int32_t)ADD_LHS[i] + (int32_t)ADD_RHS[i], -12, 18);
        if (ADD_DST[i] != (int8_t)expected) {
            fail(4, i, ADD_DST[i], expected);
        }
    }
    mark_pass();
}

static void run_mul(void) {
    for (uint32_t i = 0; i < VL; i++) {
        MUL_LHS[i] = (int8_t)((int32_t)(i % 9u) - 4);
        MUL_RHS[i] = (int8_t)((int32_t)(i % 7u) - 3);
        MUL_DST[i] = 0;
    }
    spatz_mul_i8((const int8_t *)MUL_LHS, (const int8_t *)MUL_RHS,
                 (int8_t *)MUL_DST, VL, 3, 2, -30, 31);
    for (uint32_t i = 0; i < VL; i++) {
        int32_t product = (int32_t)MUL_LHS[i] * (int32_t)MUL_RHS[i];
        int32_t expected = clamp_i32((product * 3) >> 2, -30, 31);
        if (MUL_DST[i] != (int8_t)expected) {
            fail(5, i, MUL_DST[i], expected);
        }
    }
    mark_pass();
}

static void run_logistic(void) {
    for (uint32_t i = 0; i < 256u; i++) {
        LOG_LUT[i] = (uint8_t)((i * 3u + 7u) & 0xffu);
    }
    for (uint32_t i = 0; i < VL; i++) {
        LOG_SRC[i] = (int8_t)((i * 13u + 5u) & 0xffu);
        LOG_DST[i] = 0;
    }
    if (!npu_logistic_i8((const int8_t *)LOG_SRC, (int8_t *)LOG_DST,
                         VL, (const uint8_t *)LOG_LUT)) {
        fail(6, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }
    for (uint32_t i = 0; i < VL; i++) {
        uint8_t input = (uint8_t)LOG_SRC[i];
        int8_t expected = (int8_t)LOG_LUT[input];
        if (LOG_DST[i] != expected) {
            fail(6, i, LOG_DST[i], expected);
        }
    }
    mark_pass();
}

static void run_maxpool(void) {
    for (uint32_t h = 0; h < POOL_H; h++) {
        for (uint32_t w = 0; w < POOL_W; w++) {
            for (uint32_t c = 0; c < POOL_C; c++) {
                POOL_SRC[((h * POOL_W + w) * POOL_C) + c] = pool_input_value(h, w, c);
            }
        }
    }
    spatz_maxpool2d_i8((const int8_t *)POOL_SRC, (int8_t *)POOL_DST,
                       POOL_H, POOL_W, POOL_C,
                       POOL_K, POOL_K, 1, 1, POOL_PAD, POOL_PAD);
    for (uint32_t h = 0; h < POOL_H; h++) {
        for (uint32_t w = 0; w < POOL_W; w++) {
            for (uint32_t c = 0; c < POOL_C; c++) {
                uint32_t index = ((h * POOL_W + w) * POOL_C) + c;
                int8_t expected = expected_pool_value(h, w, c);
                if (POOL_DST[index] != expected) {
                    fail(7, index, POOL_DST[index], expected);
                }
            }
        }
    }
    mark_pass();
}

static void run_upsample(void) {
    for (uint32_t h = 0; h < UP_H; h++) {
        for (uint32_t w = 0; w < UP_W; w++) {
            for (uint32_t c = 0; c < UP_C; c++) {
                UP_SRC[((h * UP_W + w) * UP_C) + c] = up_input_value(h, w, c);
            }
        }
    }
    spatz_upsample_nearest_i8((const int8_t *)UP_SRC, (int8_t *)UP_DST,
                              UP_H, UP_W, UP_C, UP_SCALE, UP_SCALE);
    for (uint32_t h = 0; h < (UP_H * UP_SCALE); h++) {
        for (uint32_t w = 0; w < (UP_W * UP_SCALE); w++) {
            for (uint32_t c = 0; c < UP_C; c++) {
                uint32_t index = ((h * (UP_W * UP_SCALE) + w) * UP_C) + c;
                int8_t expected = up_input_value(h / UP_SCALE, w / UP_SCALE, c);
                if (UP_DST[index] != expected) {
                    fail(8, index, UP_DST[index], expected);
                }
            }
        }
    }
    mark_pass();
}

static void run_concat(void) {
    for (uint32_t p = 0; p < (CONCAT_H * CONCAT_W); p++) {
        for (uint32_t c = 0; c < 32u; c++) {
            CONCAT_SRC0[(p * 32u) + c] = (int8_t)((int32_t)p + (int32_t)c - 20);
            CONCAT_SRC1[(p * 32u) + c] = (int8_t)(50 + (int32_t)p - (int32_t)c);
        }
    }
    spatz_concat_c32_i8((const int8_t *)CONCAT_SRC0, CONCAT_C0,
                        (const int8_t *)CONCAT_SRC1, CONCAT_C1,
                        (int8_t *)CONCAT_DST, CONCAT_H, CONCAT_W);
    for (uint32_t p = 0; p < (CONCAT_H * CONCAT_W); p++) {
        for (uint32_t c = 0; c < 64u; c++) {
            uint32_t index = (((c / 32u) * (CONCAT_H * CONCAT_W) + p) * 32u) + (c % 32u);
            int8_t expected = (c < 32u)
                ? (int8_t)((int32_t)p + (int32_t)c - 20)
                : (int8_t)(50 + (int32_t)p - (int32_t)(c - 32u));
            if (CONCAT_DST[index] != expected) {
                fail(9, index, CONCAT_DST[index], expected);
            }
        }
    }
    mark_pass();
}

int main(void) {
    SIG_STATUS = 0;
    SIG_PASS_COUNT = 0;
    SIG_FAIL_TEST = 0;
    SIG_FAIL_INDEX = 0;
    SIG_FAIL_GOT = 0;
    SIG_FAIL_EXP = 0;

    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_COPY) {
        run_copy();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_RELU) {
        run_relu();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_REQUANT) {
        run_requant();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ADD) {
        run_add();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_MUL) {
        run_mul();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_LOGISTIC) {
        run_logistic();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_MAXPOOL) {
        run_maxpool();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_UPSAMPLE) {
        run_upsample();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ALL || SPATZ_OP_TEST_ID == SPATZ_OP_TEST_CONCAT) {
        run_concat();
    }

    SIG_STATUS = PASS_SIGNATURE;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, PASS_SIGNATURE);
    while (1) {
    }
}
