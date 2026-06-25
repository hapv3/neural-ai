#include "hal_afu.h"
#include "npu_memory_map.h"
#include "npu_types.h"

#define AFU_SRC0_ADDR 0x10100120u
#define AFU_DST0_ADDR 0x10100480u
#define AFU_SRC1_ADDR 0x10101140u
#define AFU_DST1_ADDR 0x10101580u
#define AFU_SRC2_ADDR 0x10102260u
#define AFU_DST2_ADDR 0x10102680u

#define AFU_LEN0 64u
#define AFU_LEN1 33u
#define AFU_LEN2 17u

#define DBG_STATUS   ((volatile uint32_t *)(NPU_DTCM_BASE + 0x00))
#define DBG_CASE     ((volatile uint32_t *)(NPU_DTCM_BASE + 0x10))
#define DBG_INDEX    ((volatile uint32_t *)(NPU_DTCM_BASE + 0x14))
#define DBG_GOT      ((volatile uint32_t *)(NPU_DTCM_BASE + 0x18))
#define DBG_EXPECTED ((volatile uint32_t *)(NPU_DTCM_BASE + 0x1c))
#define DBG_PHASE    ((volatile uint32_t *)(NPU_DTCM_BASE + 0x20))

static uint32_t lut_value(uint32_t mode, uint32_t input, uint32_t salt) {
    if (mode == NPU_AFU_MODE_E8) {
        return ((input * 7u) + salt + 0x5au) & 0xffu;
    }
    if (mode == NPU_AFU_MODE_E16) {
        return ((input * 257u) + 0x1234u + salt) & 0xffffu;
    }
    return (input * 0x01010101u) ^ (0xdeadbeefu + salt);
}

static uint8_t input_value(uint32_t index, uint32_t salt) {
    return (uint8_t)(((index * 37u) + 11u + salt) & 0xffu);
}

static uint16_t read_u16(uint32_t addr) {
    volatile uint8_t *ptr = (volatile uint8_t *)addr;
    return (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
}

static uint32_t read_u32(uint32_t addr) {
    volatile uint8_t *ptr = (volatile uint8_t *)addr;
    return (uint32_t)ptr[0] |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

static void fail(uint32_t code, uint32_t case_id, uint32_t index, uint32_t got, uint32_t expected) {
    *DBG_STATUS = code;
    *DBG_CASE = case_id;
    *DBG_INDEX = index;
    *DBG_GOT = got;
    *DBG_EXPECTED = expected;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, code);
    while (1) {
    }
}

static void load_case_lut(uint32_t mode, uint32_t salt) {
    for (uint32_t i = 0; i < 256u; i++) {
        afu_load_lut_entry(i, lut_value(mode, i, salt));
    }
}

static void seed_case(uint32_t src, uint32_t dst, uint32_t length, uint32_t mode, uint32_t salt) {
    volatile uint8_t *src_ptr = (volatile uint8_t *)src;
    uint32_t output_bytes = (mode == NPU_AFU_MODE_E8) ? 1u : ((mode == NPU_AFU_MODE_E16) ? 2u : 4u);

    for (uint32_t i = 0; i < length; i++) {
        src_ptr[i] = input_value(i, salt);
    }

    for (uint32_t i = 0; i < (length * output_bytes + 64u); i++) {
        ((volatile uint8_t *)(dst - 16u))[i] = 0xa5u;
    }
}

static void check_case(uint32_t case_id, uint32_t src, uint32_t dst, uint32_t length, uint32_t mode, uint32_t salt) {
    uint32_t output_bytes = (mode == NPU_AFU_MODE_E8) ? 1u : ((mode == NPU_AFU_MODE_E16) ? 2u : 4u);

    *DBG_PHASE = 0x100u + case_id;
    load_case_lut(mode, salt);
    seed_case(src, dst, length, mode, salt);

    REG_WRITE(NPU_IRQ_INT_CLEAR, NPU_IRQ_SRC_AFU);
    REG_WRITE(NPU_IRQ_INT_ENABLE, NPU_IRQ_SRC_AFU);

    *DBG_PHASE = 0x200u + case_id;
    afu_start(src, dst, length, mode);
    if (!afu_wait_done(1000000u)) {
        fail(0xBADAF001u, case_id, 0u, afu_status(), NPU_AFU_STATUS_DONE);
    }

    if ((REG_READ(NPU_IRQ_INT_PENDING) & NPU_IRQ_SRC_AFU) == 0u) {
        fail(0xBADAF002u, case_id, 0u, REG_READ(NPU_IRQ_INT_PENDING), NPU_IRQ_SRC_AFU);
    }
    REG_WRITE(NPU_IRQ_INT_CLEAR, NPU_IRQ_SRC_AFU);

    *DBG_PHASE = 0x300u + case_id;
    for (uint32_t i = 0; i < length; i++) {
        uint32_t expected = lut_value(mode, input_value(i, salt), salt);
        uint32_t out_addr = dst + (i * output_bytes);
        uint32_t got;

        if (mode == NPU_AFU_MODE_E8) {
            got = *((volatile uint8_t *)out_addr);
        } else if (mode == NPU_AFU_MODE_E16) {
            got = read_u16(out_addr);
        } else {
            got = read_u32(out_addr);
        }

        if (got != expected) {
            fail(0xBADAF003u, case_id, i, got, expected);
        }
    }
}

int main(void) {
    *DBG_STATUS = 0u;
    *DBG_PHASE = 0u;

    check_case(0u, AFU_SRC0_ADDR, AFU_DST0_ADDR, AFU_LEN0, NPU_AFU_MODE_E8, 1u);
    check_case(1u, AFU_SRC1_ADDR, AFU_DST1_ADDR, AFU_LEN1, NPU_AFU_MODE_E16, 2u);
    check_case(2u, AFU_SRC2_ADDR, AFU_DST2_ADDR, AFU_LEN2, NPU_AFU_MODE_E32, 3u);

    *DBG_STATUS = 0xDEADBEEFu;
    REG_WRITE(NPU_IRQ_HOST_NOTIFY, 0xDEADBEEFu);
    while (1) {
    }

    return 0;
}
