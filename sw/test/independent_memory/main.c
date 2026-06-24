#include "spatz_rt.h"
#include "idma_mm_utils.h"

/*
 * Scenario: independent memory subsystem regression.
 * Target: verify boot-controlled L2 fixtures, 1D/2D/3D iDMA-compatible
 * transfers in both directions, and representative TCDM bank/boundary aliases.
 */
#define L2_SRC         0x80000000u
#define L2_DST         0x80001000u
#define L2_SRC_2D      0x80002000u
#define L2_DST_2D      0x80003000u
#define L2_SRC_3D      0x80004000u
#define L2_DST_3D      0x80005000u
#define TCDM_SRC       0x10100000u
#define TCDM_DST       0x10100400u
#define TCDM_DST_2D    0x10103000u
#define TCDM_SRC_2D    0x10104000u
#define TCDM_DST_3D    0x10105000u
#define TCDM_SRC_3D    0x10106000u
#define TCDM_BANK_LOW  0x10102000u
#define TCDM_BANK_HIGH 0x1017E000u
#define DMA_BYTES      512u

#define DMA_2D_LEN        8u
#define DMA_2D_REPS       5u
#define DMA_2D_L2_STRIDE  13u
#define DMA_2D_TCDM_STRIDE 16u

#define DMA_2D_OUT_LEN         6u
#define DMA_2D_OUT_REPS        6u
#define DMA_2D_OUT_TCDM_STRIDE 17u
#define DMA_2D_OUT_L2_STRIDE   11u

#define DMA_3D_LEN             4u
#define DMA_3D_REPS2           3u
#define DMA_3D_REPS3           2u
#define DMA_3D_L2_STRIDE2      7u
#define DMA_3D_TCDM_STRIDE2    8u
#define DMA_3D_L2_STRIDE3      40u
#define DMA_3D_TCDM_STRIDE3    32u

#define DMA_3D_OUT_LEN          5u
#define DMA_3D_OUT_REPS2        2u
#define DMA_3D_OUT_REPS3        3u
#define DMA_3D_OUT_TCDM_STRIDE2 9u
#define DMA_3D_OUT_L2_STRIDE2   8u
#define DMA_3D_OUT_TCDM_STRIDE3 64u
#define DMA_3D_OUT_L2_STRIDE3   48u

static uint8_t expected_l2(uint32_t index) {
    return (uint8_t)((index * 13u + 7u) & 0xFFu);
}

static uint8_t expected_l2_2d(uint32_t index) {
    return (uint8_t)((index * 17u + 3u) & 0xFFu);
}

static uint8_t expected_l2_3d(uint32_t index) {
    return (uint8_t)((index * 29u + 5u) & 0xFFu);
}

static uint8_t tcdm_src_2d_pattern(uint32_t index) {
    return (uint8_t)((index * 19u + 0x31u) & 0xFFu);
}

static uint8_t tcdm_src_3d_pattern(uint32_t index) {
    return (uint8_t)((index * 23u + 0x41u) & 0xFFu);
}

static void verify_bytes(const volatile uint8_t *ptr, uint32_t count, uint32_t test_base) {
    for (uint32_t i = 0; i < count; i++) {
        uint8_t expected = expected_l2(i);
        uint8_t got = ptr[i];
        if (got != expected) {
            spatz_rt_fail_at(test_base, i, got, expected);
        }
    }
}

static void fill_tcdm_dst(void) {
    volatile uint8_t *dst = (volatile uint8_t *)TCDM_DST;
    for (uint32_t i = 0; i < DMA_BYTES; i++) {
        dst[i] = (uint8_t)(0xA0u ^ ((i * 5u) & 0xFFu));
    }
}

static void verify_l2_to_tcdm_2d(void) {
    // Keep source/destination strides different to catch swapped stride wiring.
    volatile uint8_t *dst = (volatile uint8_t *)TCDM_DST_2D;
    uint32_t total = (DMA_2D_REPS - 1u) * DMA_2D_TCDM_STRIDE + DMA_2D_LEN;
    spatz_rt_memset((void *)TCDM_DST_2D, 0xEE, total);

    int tx = idma_L2ToL1_2d(L2_SRC_2D, TCDM_DST_2D, DMA_2D_LEN,
                            DMA_2D_L2_STRIDE, DMA_2D_TCDM_STRIDE, DMA_2D_REPS);
    if (REG_READ(IDMA_REPS_2(IDMA_DIR_L2_TO_L1)) != DMA_2D_REPS) {
        spatz_rt_fail_at(6, 1, REG_READ(IDMA_REPS_2(IDMA_DIR_L2_TO_L1)), DMA_2D_REPS);
    }
    if (((REG_READ(IDMA_CONF(IDMA_DIR_L2_TO_L1)) >> IDMA_CONF_ENABLE_ND_SHIFT) & 0x3u) != IDMA_2D) {
        spatz_rt_fail_at(6, 2, REG_READ(IDMA_CONF(IDMA_DIR_L2_TO_L1)), IDMA_2D);
    }
    if (!idma_mm_wait_for_completion(IDMA_DIR_L2_TO_L1, (uint32_t)tx)) {
        spatz_rt_fail_at(6, 0, tx, 1);
    }

    for (uint32_t rep = 0; rep < DMA_2D_REPS; rep++) {
        for (uint32_t col = 0; col < DMA_2D_LEN; col++) {
            uint32_t dst_index = rep * DMA_2D_TCDM_STRIDE + col;
            uint32_t src_index = rep * DMA_2D_L2_STRIDE + col;
            uint8_t got = dst[dst_index];
            uint8_t expected = expected_l2_2d(src_index);
            if (got != expected) {
                spatz_rt_fail_at(7, dst_index, got, expected);
            }
        }
    }
}

static void fill_tcdm_src_2d(void) {
    volatile uint8_t *src = (volatile uint8_t *)TCDM_SRC_2D;
    uint32_t total = (DMA_2D_OUT_REPS - 1u) * DMA_2D_OUT_TCDM_STRIDE + DMA_2D_OUT_LEN;
    for (uint32_t i = 0; i < total; i++) {
        src[i] = tcdm_src_2d_pattern(i);
    }
}

static void transfer_tcdm_to_l2_2d(void) {
    // Firmware launches the transfer; cocotb verifies sparse L2 rows afterward.
    fill_tcdm_src_2d();
    int tx = idma_L1ToL2_2d(TCDM_SRC_2D, L2_DST_2D, DMA_2D_OUT_LEN,
                            DMA_2D_OUT_TCDM_STRIDE, DMA_2D_OUT_L2_STRIDE, DMA_2D_OUT_REPS);
    if (!idma_mm_wait_for_completion(IDMA_DIR_L1_TO_L2, (uint32_t)tx)) {
        spatz_rt_fail_at(8, 0, tx, 1);
    }
}

static void verify_l2_to_tcdm_3d(void) {
    // 3D coverage uses distinct stride-2/stride-3 values on L2 and TCDM.
    volatile uint8_t *dst = (volatile uint8_t *)TCDM_DST_3D;
    uint32_t total = (DMA_3D_REPS3 - 1u) * DMA_3D_TCDM_STRIDE3 +
                     (DMA_3D_REPS2 - 1u) * DMA_3D_TCDM_STRIDE2 + DMA_3D_LEN;
    spatz_rt_memset((void *)TCDM_DST_3D, 0xDD, total);

    int tx = idma_L2ToL1_3d(L2_SRC_3D, TCDM_DST_3D, DMA_3D_LEN,
                            DMA_3D_L2_STRIDE2, DMA_3D_TCDM_STRIDE2, DMA_3D_REPS2,
                            DMA_3D_L2_STRIDE3, DMA_3D_TCDM_STRIDE3, DMA_3D_REPS3);
    if (REG_READ(IDMA_REPS_3(IDMA_DIR_L2_TO_L1)) != DMA_3D_REPS3) {
        spatz_rt_fail_at(9, 1, REG_READ(IDMA_REPS_3(IDMA_DIR_L2_TO_L1)), DMA_3D_REPS3);
    }
    if (((REG_READ(IDMA_CONF(IDMA_DIR_L2_TO_L1)) >> IDMA_CONF_ENABLE_ND_SHIFT) & 0x3u) != IDMA_3D) {
        spatz_rt_fail_at(9, 2, REG_READ(IDMA_CONF(IDMA_DIR_L2_TO_L1)), IDMA_3D);
    }
    if (!idma_mm_wait_for_completion(IDMA_DIR_L2_TO_L1, (uint32_t)tx)) {
        spatz_rt_fail_at(9, 0, tx, 1);
    }

    for (uint32_t rep3 = 0; rep3 < DMA_3D_REPS3; rep3++) {
        for (uint32_t rep2 = 0; rep2 < DMA_3D_REPS2; rep2++) {
            for (uint32_t col = 0; col < DMA_3D_LEN; col++) {
                uint32_t dst_index = rep3 * DMA_3D_TCDM_STRIDE3 + rep2 * DMA_3D_TCDM_STRIDE2 + col;
                uint32_t src_index = rep3 * DMA_3D_L2_STRIDE3 + rep2 * DMA_3D_L2_STRIDE2 + col;
                uint8_t got = dst[dst_index];
                uint8_t expected = expected_l2_3d(src_index);
                if (got != expected) {
                    spatz_rt_fail_at(10, dst_index, got, expected);
                }
            }
        }
    }
}

static void fill_tcdm_src_3d(void) {
    volatile uint8_t *src = (volatile uint8_t *)TCDM_SRC_3D;
    uint32_t total = (DMA_3D_OUT_REPS3 - 1u) * DMA_3D_OUT_TCDM_STRIDE3 +
                     (DMA_3D_OUT_REPS2 - 1u) * DMA_3D_OUT_TCDM_STRIDE2 + DMA_3D_OUT_LEN;
    for (uint32_t i = 0; i < total; i++) {
        src[i] = tcdm_src_3d_pattern(i);
    }
}

static void transfer_tcdm_to_l2_3d(void) {
    // Output-side 3D transfer is checked in the testbench to avoid self-aliasing.
    fill_tcdm_src_3d();
    int tx = idma_L1ToL2_3d(TCDM_SRC_3D, L2_DST_3D, DMA_3D_OUT_LEN,
                            DMA_3D_OUT_TCDM_STRIDE2, DMA_3D_OUT_L2_STRIDE2, DMA_3D_OUT_REPS2,
                            DMA_3D_OUT_TCDM_STRIDE3, DMA_3D_OUT_L2_STRIDE3, DMA_3D_OUT_REPS3);
    if (!idma_mm_wait_for_completion(IDMA_DIR_L1_TO_L2, (uint32_t)tx)) {
        spatz_rt_fail_at(11, 0, tx, 1);
    }
}

static void verify_bank_addresses(void) {
    // Probe low/high representatives for every TCDM bank decode lane.
    volatile uint32_t *low;
    volatile uint32_t *high;

    for (uint32_t bank = 0; bank < 16; bank++) {
        low = (volatile uint32_t *)(TCDM_BANK_LOW + bank * 32u);
        high = (volatile uint32_t *)(TCDM_BANK_HIGH + bank * 32u);
        *low = 0x11000000u | bank;
        *high = 0x22000000u | bank;
    }

    for (uint32_t bank = 0; bank < 16; bank++) {
        low = (volatile uint32_t *)(TCDM_BANK_LOW + bank * 32u);
        high = (volatile uint32_t *)(TCDM_BANK_HIGH + bank * 32u);
        uint32_t low_exp = 0x11000000u | bank;
        uint32_t high_exp = 0x22000000u | bank;
        if (*low != low_exp) {
            spatz_rt_fail_at(4, bank, (int32_t)*low, (int32_t)low_exp);
        }
        if (*high != high_exp) {
            spatz_rt_fail_at(5, bank, (int32_t)*high, (int32_t)high_exp);
        }
    }
}

int main(void) {
    spatz_rt_init();
    spatz_rt_set_phase(1, 0);

    // Phase 2: L2 fixture -> TCDM 1D copy, verified locally by firmware.
    spatz_rt_set_phase(2, 1);
    spatz_rt_memset((void *)TCDM_SRC, 0, DMA_BYTES);
    spatz_rt_dma_1d(TCDM_SRC, L2_SRC, DMA_BYTES);
    spatz_rt_dma_wait_all();
    verify_bytes((const volatile uint8_t *)TCDM_SRC, DMA_BYTES, 1);
    spatz_rt_pass_step();

    // Phase 3: TCDM -> L2 1D copy, verified by cocotb in external memory.
    spatz_rt_set_phase(3, 2);
    fill_tcdm_dst();
    spatz_rt_dma_1d(L2_DST, TCDM_DST, DMA_BYTES);
    spatz_rt_dma_wait_all();
    spatz_rt_pass_step();

    // Phase 4: direct TCDM bank low/high address decode probe.
    spatz_rt_set_phase(4, 3);
    verify_bank_addresses();
    spatz_rt_pass_step();

    // Phase 5: L2 -> TCDM 2D copy with non-equal source/destination strides.
    spatz_rt_set_phase(5, 4);
    verify_l2_to_tcdm_2d();
    spatz_rt_pass_step();

    // Phase 6: TCDM -> L2 2D copy; external L2 data is checked by cocotb.
    spatz_rt_set_phase(6, 5);
    transfer_tcdm_to_l2_2d();
    spatz_rt_pass_step();

    // Phase 7: L2 -> TCDM 3D copy with independent plane and row strides.
    spatz_rt_set_phase(7, 6);
    verify_l2_to_tcdm_3d();
    spatz_rt_pass_step();

    // Phase 8: TCDM -> L2 3D copy; cocotb checks exact output layout.
    spatz_rt_set_phase(8, 7);
    transfer_tcdm_to_l2_3d();
    spatz_rt_pass_step();

    spatz_rt_pass();
    return 0;
}
