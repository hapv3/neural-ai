#ifndef NPU_MEMORY_MAP_H
#define NPU_MEMORY_MAP_H

#include "npu_types.h"

// 1. I-TCM (32 KB) - Instruction Memory
#define NPU_ITCM_BASE   0x10000000
#define NPU_ITCM_SIZE   0x00008000

// 2. D-TCM (8 KB) - Snitch Local Data
#define NPU_DTCM_BASE   0x10008000
#define NPU_DTCM_SIZE   0x00002000

// 3. Shared Data TCDM (512 KB) - Compute Data
#define NPU_TCDM_BASE   0x10100000
#define NPU_TCDM_SIZE   0x00080000

// 4. MMIO Control Registers (64 KB)
#define NPU_CTRL_BASE   0x20000000
#define NPU_IDMA_BASE   (NPU_CTRL_BASE + 0x1000)
#define NPU_IRQ_BASE    (NPU_CTRL_BASE + 0x2000)
#define NPU_AFU_BASE    (NPU_CTRL_BASE + 0x3000)
#define NPU_PMU_BASE    (NPU_CTRL_BASE + 0x4000)
#define NPU_CMD_CTRL_BASE (NPU_CTRL_BASE + 0x5000)

// 5. External L2 / simulation memory window
#define NPU_L2_BASE     0x80000000

// Reserved Shared TCDM command staging buffer. Host writes command streams to L2;
// firmware refills this 4 KB window with iDMA while dispatching descriptors.
#define NPU_CMD_TCDM_BASE 0x1017F000u
#define NPU_CMD_TCDM_SIZE 0x00001000u

// Legacy DMA Configuration Registers
#define REG_DMA_START   (NPU_CTRL_BASE + 0x00)
#define REG_DMA_SRC     (NPU_CTRL_BASE + 0x20)
#define REG_DMA_DST     (NPU_CTRL_BASE + 0x40)
#define REG_DMA_LEN     (NPU_CTRL_BASE + 0x60)
#define REG_DMA_DONE    (NPU_CTRL_BASE + 0x80)

// PULP iDMA reg32_3d-compatible MMIO register window.
// Keep these offsets aligned with hw/idma/target/rtl/idma_reg32_3d_reg_pkg.sv.
#define IDMA_DIR_AXI2OBI            0u
#define IDMA_DIR_OBI2AXI            1u
#define IDMA_DIR_OFFSET             0x200u

#define IDMA_CONF_OFFSET            0x000u
#define IDMA_STATUS_OFFSET          0x004u
#define IDMA_NEXT_ID_OFFSET         0x044u
#define IDMA_DONE_ID_OFFSET         0x084u
#define IDMA_DST_ADDR_LOW_OFFSET    0x0D0u
#define IDMA_SRC_ADDR_LOW_OFFSET    0x0D8u
#define IDMA_LENGTH_LOW_OFFSET      0x0E0u
#define IDMA_DST_STRIDE_2_OFFSET    0x0E8u
#define IDMA_SRC_STRIDE_2_OFFSET    0x0F0u
#define IDMA_REPS_2_OFFSET          0x0F8u
#define IDMA_DST_STRIDE_3_OFFSET    0x100u
#define IDMA_SRC_STRIDE_3_OFFSET    0x108u
#define IDMA_REPS_3_OFFSET          0x110u

#define IDMA_DIR_BASE(dir)          (NPU_IDMA_BASE + ((dir) ? IDMA_DIR_OFFSET : 0u))
#define IDMA_CONF(dir)              (IDMA_DIR_BASE(dir) + IDMA_CONF_OFFSET)
#define IDMA_STATUS(dir)            (IDMA_DIR_BASE(dir) + IDMA_STATUS_OFFSET)
#define IDMA_NEXT_ID(dir)           (IDMA_DIR_BASE(dir) + IDMA_NEXT_ID_OFFSET)
#define IDMA_DONE_ID(dir)           (IDMA_DIR_BASE(dir) + IDMA_DONE_ID_OFFSET)
#define IDMA_DST_ADDR_LOW(dir)      (IDMA_DIR_BASE(dir) + IDMA_DST_ADDR_LOW_OFFSET)
#define IDMA_SRC_ADDR_LOW(dir)      (IDMA_DIR_BASE(dir) + IDMA_SRC_ADDR_LOW_OFFSET)
#define IDMA_LENGTH_LOW(dir)        (IDMA_DIR_BASE(dir) + IDMA_LENGTH_LOW_OFFSET)
#define IDMA_DST_STRIDE_2(dir)      (IDMA_DIR_BASE(dir) + IDMA_DST_STRIDE_2_OFFSET)
#define IDMA_SRC_STRIDE_2(dir)      (IDMA_DIR_BASE(dir) + IDMA_SRC_STRIDE_2_OFFSET)
#define IDMA_REPS_2(dir)            (IDMA_DIR_BASE(dir) + IDMA_REPS_2_OFFSET)
#define IDMA_DST_STRIDE_3(dir)      (IDMA_DIR_BASE(dir) + IDMA_DST_STRIDE_3_OFFSET)
#define IDMA_SRC_STRIDE_3(dir)      (IDMA_DIR_BASE(dir) + IDMA_SRC_STRIDE_3_OFFSET)
#define IDMA_REPS_3(dir)            (IDMA_DIR_BASE(dir) + IDMA_REPS_3_OFFSET)

// Systolic GEMM32 control registers
#define REG_SYS_W_PTR   (NPU_CTRL_BASE + 0x100)
#define REG_SYS_I_PTR   (NPU_CTRL_BASE + 0x104)
#define REG_SYS_O_PTR   (NPU_CTRL_BASE + 0x108)
#define REG_SYS_DIM_M   (NPU_CTRL_BASE + 0x10C)
#define REG_SYS_START   (NPU_CTRL_BASE + 0x110)
#define REG_SYS_DONE    (NPU_CTRL_BASE + 0x114)
#define REG_SYS_PSUM_PTR (NPU_CTRL_BASE + 0x118)
#define REG_SYS_ACCUM_CTRL (NPU_CTRL_BASE + 0x11C)
#define REG_RQ_CTRL     (NPU_CTRL_BASE + 0x120)
#define REG_RQ_CMIN     (NPU_CTRL_BASE + 0x124)
#define REG_RQ_CMAX     (NPU_CTRL_BASE + 0x128)
#define REG_RQ_BIAS(ch) (NPU_CTRL_BASE + 0x200 + ((ch) * 4u))
#define REG_RQ_MULT(ch) (NPU_CTRL_BASE + 0x280 + ((ch) * 4u))
#define REG_RQ_SHIFT(ch) (NPU_CTRL_BASE + 0x300 + ((ch) * 4u))
#define REG_RQ_ZP(ch)   (NPU_CTRL_BASE + 0x380 + ((ch) * 4u))

#define REG_LB_CTRL       (NPU_CTRL_BASE + 0x400)
#define REG_LB_INPUT_BASE (NPU_CTRL_BASE + 0x404)
#define REG_LB_INPUT_H    (NPU_CTRL_BASE + 0x408)
#define REG_LB_INPUT_W    (NPU_CTRL_BASE + 0x40C)
#define REG_LB_INPUT_C    (NPU_CTRL_BASE + 0x410)
#define REG_LB_OUTPUT_W   (NPU_CTRL_BASE + 0x414)
#define REG_LB_STRIDE     (NPU_CTRL_BASE + 0x418)
#define REG_LB_PAD        (NPU_CTRL_BASE + 0x41C)
#define REG_LB_ROW_STRIDE (NPU_CTRL_BASE + 0x428)
#define REG_LB_PIXEL_STRIDE (NPU_CTRL_BASE + 0x42C)
#define REG_LB_OW_STEP    (NPU_CTRL_BASE + 0x430)
#define REG_LB_OH_STEP    (NPU_CTRL_BASE + 0x434)
#define REG_LB_KERNEL     (NPU_CTRL_BASE + 0x438)
#define REG_LB_C_BASE     (NPU_CTRL_BASE + 0x43C)
#define REG_LB_SPATIAL_M  (NPU_CTRL_BASE + 0x440)
#define REG_LB_LANE_BASE  (NPU_CTRL_BASE + 0x444)
#define REG_LB_K_TILES    (NPU_CTRL_BASE + 0x448)
#define REG_LB_K_SEED     (NPU_CTRL_BASE + 0x44C)

#define REG_RQ_CTRL_EN  0x00000001u
#define REG_SYS_ACCUM_CTRL_EN 0x00000001u
#define REG_LB_CTRL_EN  0x00000001u
#define REG_LB_CTRL_COALESCE 0x00000002u
#define REG_LB_CTRL_KGEN 0x00000004u

// Interrupt controller registers
#define NPU_IRQ_INT_ENABLE    (NPU_IRQ_BASE + 0x00)
#define NPU_IRQ_INT_PENDING   (NPU_IRQ_BASE + 0x04)
#define NPU_IRQ_INT_CLEAR     (NPU_IRQ_BASE + 0x08)
#define NPU_IRQ_EXT_ENABLE    (NPU_IRQ_BASE + 0x0C)
#define NPU_IRQ_EXT_PENDING   (NPU_IRQ_BASE + 0x10)
#define NPU_IRQ_EXT_CLEAR     (NPU_IRQ_BASE + 0x14)
#define NPU_IRQ_HOST_NOTIFY   (NPU_IRQ_BASE + 0x18)
#define NPU_IRQ_HOST_STATUS   (NPU_IRQ_BASE + 0x1C)

#define NPU_IRQ_SRC_DMA       0x00000001u
#define NPU_IRQ_SRC_SYSTOLIC  0x00000002u
#define NPU_IRQ_SRC_AFU       0x00000004u
#define NPU_IRQ_SRC_SPATZ     0x00000008u
#define NPU_IRQ_HOST_DONE     0x00000001u

// AFU control window. LUT entries occupy 0x000..0x3ff; CSRs start at 0x400.
#define NPU_AFU_LUT_BASE      (NPU_AFU_BASE + 0x000)
#define NPU_AFU_STATUS        (NPU_AFU_BASE + 0x400)
#define NPU_AFU_SRC_PTR       (NPU_AFU_BASE + 0x404)
#define NPU_AFU_DST_PTR       (NPU_AFU_BASE + 0x408)
#define NPU_AFU_LENGTH        (NPU_AFU_BASE + 0x40C)
#define NPU_AFU_MODE          (NPU_AFU_BASE + 0x410)

#define NPU_AFU_STATUS_DONE   0x00000001u
#define NPU_AFU_STATUS_BUSY   0x00000002u
#define NPU_AFU_STATUS_ERROR  0x00000004u

#define NPU_AFU_MODE_E8       0u
#define NPU_AFU_MODE_E16      1u
#define NPU_AFU_MODE_E32      2u

// Host command-control bootstrap/status registers.
#define NPU_CMD_L2_BASE       (NPU_CMD_CTRL_BASE + 0x00)
#define NPU_CMD_TOTAL_BYTES   (NPU_CMD_CTRL_BASE + 0x04)
#define NPU_CMD_TCDM_BASE_REG (NPU_CMD_CTRL_BASE + 0x08)
#define NPU_CMD_TCDM_BYTES    (NPU_CMD_CTRL_BASE + 0x0C)
#define NPU_CMD_START         (NPU_CMD_CTRL_BASE + 0x10)
#define NPU_CMD_STATUS        (NPU_CMD_CTRL_BASE + 0x14)
#define NPU_CMD_FAIL_CODE     (NPU_CMD_CTRL_BASE + 0x18)
#define NPU_CMD_FAIL_PTR      (NPU_CMD_CTRL_BASE + 0x1C)
#define NPU_CMD_DONE_COUNT    (NPU_CMD_CTRL_BASE + 0x20)

#define NPU_CMD_STATUS_IDLE    0u
#define NPU_CMD_STATUS_LOADING 1u
#define NPU_CMD_STATUS_RUNNING 2u
#define NPU_CMD_STATUS_PASS    3u
#define NPU_CMD_STATUS_FAIL    4u

// Register Access Macros
#define REG_WRITE(addr, val) *((volatile uint32_t*)(addr)) = (val)
#define REG_READ(addr)       *((volatile uint32_t*)(addr))

#endif // NPU_MEMORY_MAP_H
