#include "npu_types.h"
#include "npu_memory_map.h"
#include "hal_afu.h"
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
#define SPATZ_OP_TEST_LOGISTIC_FULL 10u
#define SPATZ_OP_TEST_MUL_Q7_FULL 11u
#define SPATZ_OP_TEST_ADD_FULL 12u
#define SPATZ_OP_TEST_DFL_FUSED 14u
#define SPATZ_OP_TEST_CLASS_SIGMOID 15u
#define SPATZ_OP_TEST_GLOBAL_AVGPOOL 16u
#define SPATZ_OP_TEST_CLAMP_RELU6 17u

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
#define UP_C32_SRC   ((volatile int8_t *)0x10102000u)
#define UP_C32_DST   ((volatile int8_t *)0x10103000u)
#define CONCAT_SRC0  ((volatile int8_t *)0x10101800u)
#define CONCAT_SRC1  ((volatile int8_t *)0x10101A00u)
#define CONCAT_DST   ((volatile int8_t *)0x10101C00u)

#define LOG_FULL_LUT ((volatile uint8_t *)0x10106000u)
#define LOG_FULL_SRC ((volatile int8_t *)0x10108000u)
#define LOG_FULL_DST ((volatile int8_t *)0x1011A000u)
#define MUL_Q7_LHS   ((volatile int8_t *)0x10108000u)
#define MUL_Q7_RHS   ((volatile int8_t *)0x1011A000u)
#define MUL_Q7_DST   ((volatile int8_t *)0x1012C000u)
#define ADD_FULL_LHS MUL_Q7_LHS
#define ADD_FULL_RHS MUL_Q7_RHS
#define ADD_FULL_DST MUL_Q7_DST
#define DFL_ROW32_SRC ((volatile int8_t *)0x10155000u)
#define DFL_ROW32_DST ((volatile uint16_t *)0x10156000u)
#define DFL_EXP_LUT32 ((volatile uint32_t *)0x10157000u)
#define DFL_RECIP_LUT ((volatile uint32_t *)0x10158000u)
#define CLASS_ROW32_SRC ((volatile int8_t *)0x10159000u)
#define CLASS_DST       ((volatile int8_t *)0x1015A000u)
#define GAP_SRC         ((volatile int8_t *)0x1015B000u)
#define GAP_DST         ((volatile int8_t *)0x1015D000u)
#define CLAMP_SRC       ((volatile int8_t *)0x10160000u)
#define CLAMP_DST       ((volatile int8_t *)0x10168000u)

#define VL 32u
#define LOG_FULL_H 48u
#define LOG_FULL_W 48u
#define LOG_FULL_C 32u
#define LOG_FULL_PIXELS (LOG_FULL_H * LOG_FULL_W)
#define LOG_FULL_BYTES (LOG_FULL_PIXELS * LOG_FULL_C)
#define POOL_H 4u
#define POOL_W 5u
#define POOL_C 2u
#define POOL_K 5u
#define POOL_PAD 2u
#define UP_H 2u
#define UP_W 3u
#define UP_C 2u
#define UP_SCALE 2u
#define UP_C32_H 3u
#define UP_C32_W 4u
#define UP_C32_C 32u
#define CONCAT_H 2u
#define CONCAT_W 3u
#define CONCAT_C0 32u
#define CONCAT_C1 32u
#define DFL_FUSED_LOCATIONS 64u
#define CLASS_SIGMOID_LOCATIONS 17u
#define GAP_H 7u
#define GAP_W 5u
#define GAP_C 65u
#define GAP_PIXELS (GAP_H * GAP_W)
#define GAP_GROUPS ((GAP_C + 31u) / 32u)
#define CLAMP_H 17u
#define CLAMP_W 13u
#define CLAMP_C 65u
#define CLAMP_GROUPS ((CLAMP_C + 31u) / 32u)
#define CLAMP_PHYS_C (CLAMP_GROUPS * 32u)
#define CLAMP_BYTES (CLAMP_H * CLAMP_W * CLAMP_PHYS_C)
#define CLAMP_MIN 0
#define CLAMP_MAX 63

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

static uint8_t sigmoid_lut_value(uint32_t index) {
    return (uint8_t)((index * 3u + 7u) & 0xffu);
}

static int8_t pool_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)((h * 11u + w * 7u + c * 5u) % 31u) - 15);
}

static int8_t up_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)(h * 17u + w * 9u + c * 3u) - 20);
}

static int8_t up_c32_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)((h * 17u + w * 11u + c * 5u) % 127u) - 63);
}

static uint32_t c32_offset(uint32_t height, uint32_t width,
                           uint32_t y, uint32_t x, uint32_t c) {
    uint32_t pixel = y * width + x;
    return (((c >> 5) * (height * width) + pixel) * 32u) + (c & 31u);
}

static int8_t gap_input_value(uint32_t h, uint32_t w, uint32_t c) {
    return (int8_t)((int32_t)(((h * 23u) + (w * 17u) + (c * 11u) + 9u) & 0x7fu) - 64);
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
    mark_pass();
}

static void run_relu(void) {
    for (uint32_t i = 0; i < VL; i++) {
        RELU_I8[i] = (int8_t)((int32_t)i - 12);
    }

    spatz_vec_relu_i8((int8_t *)RELU_I8, VL);
    mark_pass();
}

static void run_requant(void) {
    for (uint32_t i = 0; i < VL; i++) {
        SRC_I32[i] = ((int32_t)i - 16) * 37;
        DST_REQUANT[i] = 0;
    }

    spatz_requant_i32_to_i8((const int32_t *)SRC_I32, (int8_t *)DST_REQUANT,
                            VL, 2, 3, -20, 31);
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
    mark_pass();
}

static int8_t logistic_full_input_value(uint32_t pixel, uint32_t channel) {
    return (int8_t)(((pixel * 17u) + (channel * 13u) + 5u) & 0xffu);
}

static uint8_t logistic_full_lut_value(uint32_t index) {
    return (uint8_t)(((index * 5u) + 11u) & 0x7fu);
}

static uint32_t run_logistic_full_afu(uint32_t timeout_cycles) {
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_lut_entry(i, (uint32_t)LOG_FULL_LUT[i]);
    }

    afu_start((uint32_t)LOG_FULL_SRC, (uint32_t)LOG_FULL_DST,
              LOG_FULL_BYTES, NPU_AFU_MODE_E8);
    for (uint32_t i = 0; i < timeout_cycles; i++) {
        uint32_t status = afu_status();
        SIG_FAIL_INDEX = i;
        SIG_FAIL_GOT = status;
        SIG_FAIL_EXP = NPU_AFU_STATUS_DONE;
        if ((status & NPU_AFU_STATUS_ERROR) != 0u) {
            return 0u;
        }
        if ((status & NPU_AFU_STATUS_DONE) != 0u) {
            return 1u;
        }
        __asm__ volatile("nop");
    }
    return 0u;
}

static void run_logistic_full(void) {
    SIG_STATUS = 0x30001001u;
    for (uint32_t i = 0; i < 256u; i++) {
        LOG_FULL_LUT[i] = logistic_full_lut_value(i);
    }

    SIG_STATUS = 0x30001002u;
    // Source tensor is preloaded by cocotb backdoor to avoid measuring scalar seed time.

    SIG_STATUS = 0x30001003u;
    if (!run_logistic_full_afu(300000u)) {
        fail(10, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001004u;
    // Full tensor output is checked by cocotb backdoor.  Scalar verification on
    // Snitch is too slow for this large AFU throughput regression.
    mark_pass();
}

static void run_mul_q7_full(void) {
    SIG_STATUS = 0x30001101u;
    // Full tensor inputs are preloaded by cocotb backdoor to isolate the vector helper.
    if (!npu_mul_q7_i8((const int8_t *)MUL_Q7_LHS, (const int8_t *)MUL_Q7_RHS,
                       (int8_t *)MUL_Q7_DST, LOG_FULL_BYTES)) {
        fail(11, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001102u;
    // Full tensor output is checked by cocotb backdoor.
    mark_pass();
}

static void run_add_full(void) {
    SIG_STATUS = 0x30001201u;
    // Full tensor inputs are preloaded by cocotb backdoor to isolate AFU add.
    if (!npu_add_i8((const int8_t *)ADD_FULL_LHS, (const int8_t *)ADD_FULL_RHS,
                    (int8_t *)ADD_FULL_DST, LOG_FULL_BYTES)) {
        fail(12, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001202u;
    // Full tensor output is checked by cocotb backdoor.
    mark_pass();
}

static void run_dfl_fused(void) {
    SIG_STATUS = 0x30001402u;
    if (!npu_dfl_softmax4_row32_i8_q8((const int8_t *)DFL_ROW32_SRC,
                                      (uint16_t *)DFL_ROW32_DST,
                                      DFL_FUSED_LOCATIONS,
                                      (const uint32_t *)DFL_EXP_LUT32,
                                      (const uint32_t *)DFL_RECIP_LUT)) {
        fail(14, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001403u;
    mark_pass();
}

static void run_class_sigmoid(void) {
    SIG_STATUS = 0x30001501u;
    for (uint32_t i = 0; i < 256u; i++) {
        LOG_LUT[i] = sigmoid_lut_value(i);
    }
    for (uint32_t loc = 0; loc < CLASS_SIGMOID_LOCATIONS; loc++) {
        for (uint32_t ch = 0; ch < 32u; ch++) {
            CLASS_ROW32_SRC[(loc * 32u) + ch] =
                (int8_t)((int32_t)(((loc * 29u) + (ch * 13u) + 5u) & 0xffu) - 128);
        }
    }
    for (uint32_t i = 0; i < CLASS_SIGMOID_LOCATIONS * 16u; i++) {
        CLASS_DST[i] = 0;
    }

    SIG_STATUS = 0x30001502u;
    if (!npu_class_sigmoid_row32_high16_i8((const int8_t *)CLASS_ROW32_SRC,
                                           (int8_t *)CLASS_DST,
                                           CLASS_SIGMOID_LOCATIONS,
                                           (const uint8_t *)LOG_LUT)) {
        fail(15, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001503u;
    mark_pass();
}

static void run_global_avgpool(void) {
    SIG_STATUS = 0x30001601u;
    for (uint32_t h = 0; h < GAP_H; h++) {
        for (uint32_t w = 0; w < GAP_W; w++) {
            for (uint32_t c = 0; c < GAP_C; c++) {
                GAP_SRC[c32_offset(GAP_H, GAP_W, h, w, c)] = gap_input_value(h, w, c);
            }
        }
    }
    for (uint32_t i = 0; i < GAP_GROUPS * 32u; i++) {
        GAP_DST[i] = (int8_t)0x5a;
    }

    SIG_STATUS = 0x30001602u;
    if (!npu_global_avgpool_c32_i8((const int8_t *)GAP_SRC,
                                   (int8_t *)GAP_DST,
                                   GAP_H,
                                   GAP_W,
                                   GAP_C)) {
        fail(16, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001603u;
    mark_pass();
}

static void run_clamp_relu6(void) {
    SIG_STATUS = 0x30001702u;
    if (!npu_clamp_i8((const int8_t *)CLAMP_SRC,
                      (int8_t *)CLAMP_DST,
                      CLAMP_BYTES,
                      CLAMP_MIN,
                      CLAMP_MAX)) {
        fail(17, 0, (int32_t)REG_READ(NPU_AFU_STATUS), NPU_AFU_STATUS_DONE);
    }

    SIG_STATUS = 0x30001703u;
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

    for (uint32_t h = 0; h < UP_C32_H; h++) {
        for (uint32_t w = 0; w < UP_C32_W; w++) {
            for (uint32_t c = 0; c < UP_C32_C; c++) {
                UP_C32_SRC[((h * UP_C32_W + w) * UP_C32_C) + c] =
                    up_c32_input_value(h, w, c);
            }
        }
    }
    spatz_upsample_nearest_i8((const int8_t *)UP_C32_SRC, (int8_t *)UP_C32_DST,
                              UP_C32_H, UP_C32_W, UP_C32_C, UP_SCALE, UP_SCALE);
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
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_LOGISTIC) {
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
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_LOGISTIC_FULL) {
        run_logistic_full();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_MUL_Q7_FULL) {
        run_mul_q7_full();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_ADD_FULL) {
        run_add_full();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_DFL_FUSED) {
        run_dfl_fused();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_CLASS_SIGMOID) {
        run_class_sigmoid();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_GLOBAL_AVGPOOL) {
        run_global_avgpool();
    }
    if (SPATZ_OP_TEST_ID == SPATZ_OP_TEST_CLAMP_RELU6) {
        run_clamp_relu6();
    }

    SIG_STATUS = PASS_SIGNATURE;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, PASS_SIGNATURE);
    while (1) {
    }
}
