#ifndef IDMA_MM_UTILS_H
#define IDMA_MM_UTILS_H

#include "npu_memory_map.h"
#include "npu_types.h"

#define IDMA_MM_DIRECTION_OFFSET IDMA_DIR_OFFSET
#define IDMA_MM_BASE_AXI2OBI NPU_IDMA_BASE
#define IDMA_MM_BASE_OBI2AXI (NPU_IDMA_BASE + IDMA_MM_DIRECTION_OFFSET)

#define IDMA_CONF_ADDR(is_l1_to_l2) IDMA_CONF(is_l1_to_l2)
#define IDMA_STATUS_ADDR(is_l1_to_l2, id) (IDMA_DIR_BASE(is_l1_to_l2) + IDMA_STATUS_OFFSET + ((id) * 4u))
#define IDMA_NEXT_ID_ADDR(is_l1_to_l2, id) (IDMA_DIR_BASE(is_l1_to_l2) + IDMA_NEXT_ID_OFFSET + ((id) * 4u))
#define IDMA_DONE_ID_ADDR(is_l1_to_l2, id) (IDMA_DIR_BASE(is_l1_to_l2) + IDMA_DONE_ID_OFFSET + ((id) * 4u))
#define IDMA_DST_ADDR_LOW_ADDR(is_l1_to_l2) IDMA_DST_ADDR_LOW(is_l1_to_l2)
#define IDMA_SRC_ADDR_LOW_ADDR(is_l1_to_l2) IDMA_SRC_ADDR_LOW(is_l1_to_l2)
#define IDMA_LENGTH_LOW_ADDR(is_l1_to_l2) IDMA_LENGTH_LOW(is_l1_to_l2)
#define IDMA_DST_STRIDE_2_LOW_ADDR(is_l1_to_l2) IDMA_DST_STRIDE_2(is_l1_to_l2)
#define IDMA_SRC_STRIDE_2_LOW_ADDR(is_l1_to_l2) IDMA_SRC_STRIDE_2(is_l1_to_l2)
#define IDMA_REPS_2_LOW_ADDR(is_l1_to_l2) IDMA_REPS_2(is_l1_to_l2)
#define IDMA_DST_STRIDE_3_LOW_ADDR(is_l1_to_l2) IDMA_DST_STRIDE_3(is_l1_to_l2)
#define IDMA_SRC_STRIDE_3_LOW_ADDR(is_l1_to_l2) IDMA_SRC_STRIDE_3(is_l1_to_l2)
#define IDMA_REPS_3_LOW_ADDR(is_l1_to_l2) IDMA_REPS_3(is_l1_to_l2)

#define IDMA_CONF_DECOUPLE_AW_BIT    0u
#define IDMA_CONF_DECOUPLE_RW_BIT    1u
#define IDMA_CONF_SRC_REDUCE_LEN_BIT 2u
#define IDMA_CONF_DST_REDUCE_LEN_BIT 3u
#define IDMA_CONF_SRC_MAX_LLEN_SHIFT 4u
#define IDMA_CONF_DST_MAX_LLEN_SHIFT 7u
#define IDMA_CONF_ENABLE_ND_SHIFT    10u
#define IDMA_CONF_SRC_PROTOCOL       12u
#define IDMA_CONF_DST_PROTOCOL       15u

#define IDMA_STATUS_BUSY_MASK        0x3FFu

#define IDMA_DIR_L2_TO_L1 IDMA_DIR_AXI2OBI
#define IDMA_DIR_L1_TO_L2 IDMA_DIR_OBI2AXI
#define IDMA_EXT2LOC      IDMA_DIR_L2_TO_L1
#define IDMA_LOC2EXT      IDMA_DIR_L1_TO_L2

#define IDMA_1D 0u
#define IDMA_2D 1u
#define IDMA_3D 2u

#define IDMA_MM_MAX_1D_CHUNK 4096u

typedef enum {
    IDMA_PROT_AXI = 0,
    IDMA_PROT_OBI = 1
} idma_prot_t;

typedef uint32_t dma_ext_t;

#ifndef mmio32
#define mmio32(addr) (*(volatile uint32_t *)(addr))
#endif

static inline void wait_nop(uint32_t nops) {
    for (uint32_t i = 0; i < nops; i++) {
        __asm__ volatile("nop");
    }
}

static inline void idma_cpu_copy(uint32_t src, uint32_t dst, uint32_t size) {
    volatile const uint8_t *src_ptr = (volatile const uint8_t *)src;
    volatile uint8_t *dst_ptr = (volatile uint8_t *)dst;
    for (uint32_t i = 0; i < size; i++) {
        dst_ptr[i] = src_ptr[i];
    }
}

static inline uint32_t idma_mm_is_l1_addr(uint32_t addr) {
    return ((addr >> 24) == (NPU_ITCM_BASE >> 24));
}

static inline void idma_mm_conf_dir(uint32_t is_l1_to_l2, uint32_t decouple_aw, uint32_t decouple_rw,
                                    uint32_t src_reduce_len, uint32_t dst_reduce_len,
                                    uint32_t src_max_llen, uint32_t dst_max_llen,
                                    uint32_t enable_nd) {
    uint32_t conf_val = 0;

    if (decouple_aw) {
        conf_val |= (1u << IDMA_CONF_DECOUPLE_AW_BIT);
    }
    if (decouple_rw) {
        conf_val |= (1u << IDMA_CONF_DECOUPLE_RW_BIT);
    }
    if (src_reduce_len) {
        conf_val |= (1u << IDMA_CONF_SRC_REDUCE_LEN_BIT);
    }
    if (dst_reduce_len) {
        conf_val |= (1u << IDMA_CONF_DST_REDUCE_LEN_BIT);
    }

    conf_val |= ((src_max_llen & 0x7u) << IDMA_CONF_SRC_MAX_LLEN_SHIFT);
    conf_val |= ((dst_max_llen & 0x7u) << IDMA_CONF_DST_MAX_LLEN_SHIFT);
    conf_val |= ((enable_nd & 0x3u) << IDMA_CONF_ENABLE_ND_SHIFT);

    if (is_l1_to_l2) {
        conf_val |= (IDMA_PROT_OBI << IDMA_CONF_SRC_PROTOCOL);
        conf_val |= (IDMA_PROT_AXI << IDMA_CONF_DST_PROTOCOL);
    } else {
        conf_val |= (IDMA_PROT_AXI << IDMA_CONF_SRC_PROTOCOL);
        conf_val |= (IDMA_PROT_OBI << IDMA_CONF_DST_PROTOCOL);
    }

    mmio32(IDMA_CONF_ADDR(is_l1_to_l2)) = conf_val;
}

static inline void idma_mm_conf_default_dir(uint32_t is_l1_to_l2) {
    idma_mm_conf_dir(is_l1_to_l2, 0, 0, 0, 0, 4, 4, IDMA_1D);
}

static inline uint32_t idma_mm_is_busy_dir(uint32_t is_l1_to_l2, uint32_t stream_id) {
    if (stream_id >= 16u) {
        return 0;
    }
    return (mmio32(IDMA_STATUS_ADDR(is_l1_to_l2, stream_id)) & IDMA_STATUS_BUSY_MASK) ? 1u : 0u;
}

static inline uint32_t idma_mm_start_transfer_dir(uint32_t is_l1_to_l2, uint32_t stream_id) {
    if (stream_id >= 16u) {
        return 0;
    }
    return mmio32(IDMA_NEXT_ID_ADDR(is_l1_to_l2, stream_id));
}

static inline uint32_t idma_mm_get_done_id_dir(uint32_t is_l1_to_l2, uint32_t stream_id) {
    if (stream_id >= 16u) {
        return 0;
    }
    return mmio32(IDMA_DONE_ID_ADDR(is_l1_to_l2, stream_id));
}

static inline void idma_mm_set_addr_len_dir(uint32_t is_l1_to_l2, uint32_t dst_addr, uint32_t src_addr, uint32_t length) {
    mmio32(IDMA_DST_ADDR_LOW_ADDR(is_l1_to_l2)) = dst_addr;
    mmio32(IDMA_SRC_ADDR_LOW_ADDR(is_l1_to_l2)) = src_addr;
    mmio32(IDMA_LENGTH_LOW_ADDR(is_l1_to_l2)) = length;
}

static inline void idma_mm_set_2d_params_dir(uint32_t is_l1_to_l2, uint32_t dst_stride_2, uint32_t src_stride_2, uint32_t reps_2) {
    mmio32(IDMA_DST_STRIDE_2_LOW_ADDR(is_l1_to_l2)) = dst_stride_2;
    mmio32(IDMA_SRC_STRIDE_2_LOW_ADDR(is_l1_to_l2)) = src_stride_2;
    mmio32(IDMA_REPS_2_LOW_ADDR(is_l1_to_l2)) = reps_2;
}

static inline void idma_mm_set_3d_params_dir(uint32_t is_l1_to_l2, uint32_t dst_stride_3, uint32_t src_stride_3, uint32_t reps_3) {
    mmio32(IDMA_DST_STRIDE_3_LOW_ADDR(is_l1_to_l2)) = dst_stride_3;
    mmio32(IDMA_SRC_STRIDE_3_LOW_ADDR(is_l1_to_l2)) = src_stride_3;
    mmio32(IDMA_REPS_3_LOW_ADDR(is_l1_to_l2)) = reps_3;
}

static inline uint32_t idma_mm_wait_for_completion(uint32_t direction, uint32_t transfer_id) {
    if (transfer_id == 0) {
        return 0;
    }

    uint32_t is_l1_to_l2 = (direction == IDMA_DIR_L1_TO_L2) ? 1u : 0u;
    uint32_t timeout = 1000000u;

    while (timeout-- > 0u) {
        if (!idma_mm_is_busy_dir(is_l1_to_l2, 0) &&
            idma_mm_get_done_id_dir(is_l1_to_l2, 0) == transfer_id) {
            return 1;
        }
        wait_nop(10);
    }

    return 0;
}

static inline int idma_L1ToL2(uint32_t src, uint32_t dst, uint32_t size) {
    idma_mm_conf_default_dir(1);
    idma_mm_set_addr_len_dir(1, dst, src, size);
    idma_mm_set_2d_params_dir(1, 0, 0, 1);
    idma_mm_set_3d_params_dir(1, 0, 0, 1);
    return (int)idma_mm_start_transfer_dir(1, 0);
}

static inline int idma_L2ToL1(uint32_t src, uint32_t dst, uint32_t size) {
    idma_mm_conf_default_dir(0);
    idma_mm_set_addr_len_dir(0, dst, src, size);
    idma_mm_set_2d_params_dir(0, 0, 0, 1);
    idma_mm_set_3d_params_dir(0, 0, 0, 1);
    return (int)idma_mm_start_transfer_dir(0, 0);
}

static inline int idma_L1ToL1(uint32_t src, uint32_t dst, uint32_t size) {
    idma_cpu_copy(src, dst, size);
    return 1;
}

static inline int dma_memcpy(dma_ext_t ext, uint32_t loc, uint32_t size, int ext2loc) {
    return ext2loc ? idma_L2ToL1(ext, loc, size) : idma_L1ToL2(loc, ext, size);
}

static inline int dma_l1ToExt(dma_ext_t ext, uint32_t loc, uint32_t size) {
    return idma_L1ToL2(loc, ext, size);
}

static inline int dma_extToL1(uint32_t loc, dma_ext_t ext, uint32_t size) {
    return idma_L2ToL1(ext, loc, size);
}

static inline int idma_memcpy(uint32_t src, uint32_t dst, uint32_t size,
                              idma_prot_t src_prot, idma_prot_t dst_prot) {
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_AXI) {
        return idma_L1ToL2(src, dst, size);
    }
    if (src_prot == IDMA_PROT_AXI && dst_prot == IDMA_PROT_OBI) {
        return idma_L2ToL1(src, dst, size);
    }
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_OBI) {
        return idma_L1ToL1(src, dst, size);
    }
    return 0;
}

static inline uint32_t idma_memcpy_blocking(uint32_t src, uint32_t dst, uint32_t size) {
    uint32_t src_is_l1 = idma_mm_is_l1_addr(src);
    uint32_t dst_is_l1 = idma_mm_is_l1_addr(dst);
    uint32_t offset = 0;

    if (src_is_l1 && dst_is_l1) {
        idma_cpu_copy(src, dst, size);
        return 1;
    }

    if (!src_is_l1 && !dst_is_l1) {
        return 0;
    }

    while (offset < size) {
        uint32_t chunk = size - offset;
        uint32_t dir;
        int tx_id;

        if (chunk > IDMA_MM_MAX_1D_CHUNK) {
            chunk = IDMA_MM_MAX_1D_CHUNK;
        }

        if (src_is_l1) {
            dir = IDMA_DIR_L1_TO_L2;
            tx_id = idma_L1ToL2(src + offset, dst + offset, chunk);
        } else {
            dir = IDMA_DIR_L2_TO_L1;
            tx_id = idma_L2ToL1(src + offset, dst + offset, chunk);
        }

        if (!idma_mm_wait_for_completion(dir, (uint32_t)tx_id)) {
            return 0;
        }

        offset += chunk;
    }

    return 1;
}

static inline int idma_L1ToL1_pull(uint32_t remote_src, uint32_t local_dst, uint32_t size) {
    return idma_L2ToL1(remote_src, local_dst, size);
}

static inline int idma_L1ToL1_push(uint32_t local_src, uint32_t remote_dst, uint32_t size) {
    return idma_L1ToL2(local_src, remote_dst, size);
}

static inline int idma_L1ToL2_2d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride, uint32_t dst_stride, uint32_t num_reps) {
    idma_mm_conf_dir(1, 0, 0, 0, 0, 4, 4, IDMA_2D);
    idma_mm_set_addr_len_dir(1, dst, src, size);
    idma_mm_set_2d_params_dir(1, dst_stride, src_stride, num_reps);
    idma_mm_set_3d_params_dir(1, 0, 0, 1);
    return (int)idma_mm_start_transfer_dir(1, 0);
}

static inline int idma_L2ToL1_2d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride, uint32_t dst_stride, uint32_t num_reps) {
    idma_mm_conf_dir(0, 0, 0, 0, 0, 4, 4, IDMA_2D);
    idma_mm_set_addr_len_dir(0, dst, src, size);
    idma_mm_set_2d_params_dir(0, dst_stride, src_stride, num_reps);
    idma_mm_set_3d_params_dir(0, 0, 0, 1);
    return (int)idma_mm_start_transfer_dir(0, 0);
}

static inline int idma_L1ToL1_2d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride, uint32_t dst_stride, uint32_t num_reps) {
    for (uint32_t rep = 0; rep < num_reps; rep++) {
        idma_cpu_copy(src + rep * src_stride, dst + rep * dst_stride, size);
    }
    return 1;
}

static inline int idma_L1ToL1_pull_2d(uint32_t remote_src, uint32_t local_dst, uint32_t size,
                                      uint32_t src_stride, uint32_t dst_stride, uint32_t num_reps) {
    return idma_L2ToL1_2d(remote_src, local_dst, size, src_stride, dst_stride, num_reps);
}

static inline int idma_L1ToL1_push_2d(uint32_t local_src, uint32_t remote_dst, uint32_t size,
                                      uint32_t src_stride, uint32_t dst_stride, uint32_t num_reps) {
    return idma_L1ToL2_2d(local_src, remote_dst, size, src_stride, dst_stride, num_reps);
}

static inline int idma_memcpy_2d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride, uint32_t dst_stride,
                                 uint32_t num_reps, idma_prot_t src_prot, idma_prot_t dst_prot) {
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_AXI) {
        return idma_L1ToL2_2d(src, dst, size, src_stride, dst_stride, num_reps);
    }
    if (src_prot == IDMA_PROT_AXI && dst_prot == IDMA_PROT_OBI) {
        return idma_L2ToL1_2d(src, dst, size, src_stride, dst_stride, num_reps);
    }
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_OBI) {
        return idma_L1ToL1_2d(src, dst, size, src_stride, dst_stride, num_reps);
    }
    return 0;
}

static inline int idma_L1ToL2_3d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride_2, uint32_t dst_stride_2, uint32_t reps_2,
                                 uint32_t src_stride_3, uint32_t dst_stride_3, uint32_t reps_3) {
    uint32_t src_hw_stride_3 = src_stride_3 - ((reps_2 - 1u) * src_stride_2);
    uint32_t dst_hw_stride_3 = dst_stride_3 - ((reps_2 - 1u) * dst_stride_2);
    idma_mm_conf_dir(1, 0, 0, 0, 0, 4, 4, IDMA_3D);
    idma_mm_set_addr_len_dir(1, dst, src, size);
    idma_mm_set_2d_params_dir(1, dst_stride_2, src_stride_2, reps_2);
    idma_mm_set_3d_params_dir(1, dst_hw_stride_3, src_hw_stride_3, reps_3);
    return (int)idma_mm_start_transfer_dir(1, 0);
}

static inline int idma_L2ToL1_3d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride_2, uint32_t dst_stride_2, uint32_t reps_2,
                                 uint32_t src_stride_3, uint32_t dst_stride_3, uint32_t reps_3) {
    uint32_t src_hw_stride_3 = src_stride_3 - ((reps_2 - 1u) * src_stride_2);
    uint32_t dst_hw_stride_3 = dst_stride_3 - ((reps_2 - 1u) * dst_stride_2);
    idma_mm_conf_dir(0, 0, 0, 0, 0, 4, 4, IDMA_3D);
    idma_mm_set_addr_len_dir(0, dst, src, size);
    idma_mm_set_2d_params_dir(0, dst_stride_2, src_stride_2, reps_2);
    idma_mm_set_3d_params_dir(0, dst_hw_stride_3, src_hw_stride_3, reps_3);
    return (int)idma_mm_start_transfer_dir(0, 0);
}

static inline int idma_L1ToL1_3d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride_2, uint32_t dst_stride_2, uint32_t reps_2,
                                 uint32_t src_stride_3, uint32_t dst_stride_3, uint32_t reps_3) {
    for (uint32_t rep3 = 0; rep3 < reps_3; rep3++) {
        for (uint32_t rep2 = 0; rep2 < reps_2; rep2++) {
            idma_cpu_copy(src + rep3 * src_stride_3 + rep2 * src_stride_2,
                          dst + rep3 * dst_stride_3 + rep2 * dst_stride_2,
                          size);
        }
    }
    return 1;
}

static inline int idma_memcpy_3d(uint32_t src, uint32_t dst, uint32_t size,
                                 uint32_t src_stride_2, uint32_t dst_stride_2, uint32_t reps_2,
                                 uint32_t src_stride_3, uint32_t dst_stride_3, uint32_t reps_3,
                                 idma_prot_t src_prot, idma_prot_t dst_prot) {
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_AXI) {
        return idma_L1ToL2_3d(src, dst, size, src_stride_2, dst_stride_2, reps_2,
                              src_stride_3, dst_stride_3, reps_3);
    }
    if (src_prot == IDMA_PROT_AXI && dst_prot == IDMA_PROT_OBI) {
        return idma_L2ToL1_3d(src, dst, size, src_stride_2, dst_stride_2, reps_2,
                              src_stride_3, dst_stride_3, reps_3);
    }
    if (src_prot == IDMA_PROT_OBI && dst_prot == IDMA_PROT_OBI) {
        return idma_L1ToL1_3d(src, dst, size, src_stride_2, dst_stride_2, reps_2,
                              src_stride_3, dst_stride_3, reps_3);
    }
    return 0;
}

static inline uint32_t idma_tx_cplt(uint32_t dma_tx_id) {
    return (idma_mm_get_done_id_dir(0, 0) == dma_tx_id) ||
           (idma_mm_get_done_id_dir(1, 0) == dma_tx_id);
}

static inline uint32_t dma_status(void) {
    return idma_mm_is_busy_dir(0, 0) || idma_mm_is_busy_dir(1, 0);
}

static inline void dma_wait(uint32_t dma_tx_id) {
    while (!idma_tx_cplt(dma_tx_id)) {
        wait_nop(1);
    }
}

static inline void dma_barrier(void) {
    while (dma_status()) {
        wait_nop(1);
    }
}

#endif
