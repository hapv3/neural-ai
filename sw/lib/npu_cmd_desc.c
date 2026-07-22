#include "npu_cmd_desc.h"
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "npu_memory_map.h"
#include "npu_model_abi.h"
#include "npu_model_runtime.h"

extern uint32_t nai_runtime_dispatch_from_ctrl(uint32_t invocation_base,
                                               uint32_t invocation_bytes,
                                               uint32_t staging_base,
                                               uint32_t staging_bytes) __attribute__((weak));

static uint32_t g_cmd_done_count;

typedef struct {
    uint32_t configured;
    uint32_t base_addr;
    uint32_t slot_bytes;
    uint32_t slot_count;
    uint32_t produce_slot;
    uint32_t consume_slot;
    uint32_t occupancy;
} npu_cmd_rolling_state_t;

static npu_cmd_rolling_state_t g_rolling_state[4];

static void cmd_record_phase(uint32_t phase, uint32_t op) {
    volatile uint32_t *debug_words = (volatile uint32_t *)NPU_DTCM_BASE;
    debug_words[6] = phase;
    debug_words[7] = op;
}

static void cmd_set_status(uint32_t status) {
    REG_WRITE(NPU_CMD_STATUS, status);
}

static uint32_t cmd_fail(uint32_t code, uint32_t fail_ptr) {
    REG_WRITE(NPU_CMD_FAIL_CODE, code);
    REG_WRITE(NPU_CMD_FAIL_PTR, fail_ptr);
    REG_WRITE(NPU_CMD_DONE_COUNT, g_cmd_done_count);
    cmd_set_status(NPU_CMD_STATUS_FAIL);
    return code;
}

static uint32_t is_aligned32(uint32_t value) {
    return (value & (NPU_CMD_ALIGN_BYTES - 1u)) == 0u;
}

static void copy_from_cmd(void *dst, const volatile void *src, uint32_t bytes) {
    uint32_t *dst_words = (uint32_t *)dst;
    const volatile uint32_t *src_words = (const volatile uint32_t *)src;
    uint32_t words = bytes >> 2;
    for (uint32_t i = 0; i < words; i++) {
        dst_words[i] = src_words[i];
    }
}

typedef struct {
    uint32_t l2_base;
    uint32_t total_bytes;
    uint32_t buffer_base;
    uint32_t buffer_bytes;
    uint32_t window_offset;
    uint32_t window_bytes;
} npu_cmd_stream_t;

static uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

static void reset_rolling_state(void) {
    for (uint32_t idx = 0; idx < 4u; idx++) {
        g_rolling_state[idx].configured = 0u;
        g_rolling_state[idx].base_addr = 0u;
        g_rolling_state[idx].slot_bytes = 0u;
        g_rolling_state[idx].slot_count = 0u;
        g_rolling_state[idx].produce_slot = 0u;
        g_rolling_state[idx].consume_slot = 0u;
        g_rolling_state[idx].occupancy = 0u;
    }
}


static uint32_t cmd_refill_window(npu_cmd_stream_t *stream, uint32_t offset, uint32_t bytes, uint32_t fail_ptr) {
    if (bytes == 0u || offset > stream->total_bytes || bytes > (stream->total_bytes - offset)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_SIZE, fail_ptr);
    }
    if (bytes > stream->buffer_bytes) {
        return cmd_fail(NPU_CMD_FAIL_BAD_SIZE, fail_ptr);
    }
    if (stream->window_bytes != 0u &&
        offset >= stream->window_offset &&
        (offset + bytes) <= (stream->window_offset + stream->window_bytes)) {
        return NPU_CMD_FAIL_NONE;
    }

    uint32_t copy_offset = offset;
    uint32_t copy_bytes = min_u32(stream->buffer_bytes, stream->total_bytes - copy_offset);
    if (!is_aligned32(copy_offset) || !is_aligned32(copy_bytes)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_ALIGN, fail_ptr);
    }
    if (!idma_memcpy_blocking(stream->l2_base + copy_offset, stream->buffer_base, copy_bytes)) {
        return cmd_fail(NPU_CMD_FAIL_COPY, fail_ptr);
    }

    stream->window_offset = copy_offset;
    stream->window_bytes = copy_bytes;
    return NPU_CMD_FAIL_NONE;
}

static uint32_t stream_copy_from_cmd(npu_cmd_stream_t *stream, void *dst, uint32_t offset, uint32_t bytes, uint32_t fail_ptr) {
    uint32_t status = cmd_refill_window(stream, offset, bytes, fail_ptr);
    if (status != NPU_CMD_FAIL_NONE) {
        return status;
    }
    copy_from_cmd(dst, (const volatile void *)(stream->buffer_base + (offset - stream->window_offset)), bytes);
    return NPU_CMD_FAIL_NONE;
}

static uint32_t wait_tx_or_fail(uint32_t direction, int tx_id, uint32_t fail_ptr) {
    if (tx_id <= 0) {
        return cmd_fail(NPU_CMD_FAIL_DMA_START, fail_ptr);
    }

    uint32_t status_dir = (direction == IDMA_DIR_L1_TO_L2) ? 1u : 0u;
    if (idma_mm_wait_for_completion(direction, (uint32_t)tx_id)) {
        return NPU_CMD_FAIL_NONE;
    }

    REG_WRITE(NPU_CMD_FAIL_PTR, fail_ptr);
    REG_WRITE(NPU_CMD_DONE_COUNT, REG_READ(IDMA_STATUS(status_dir)));
    return cmd_fail(NPU_CMD_FAIL_DMA_TIMEOUT | ((uint32_t)tx_id & 0xFFu), fail_ptr);
}

static uint32_t resolve_dma_direction(uint32_t src_addr, uint32_t dst_addr, uint32_t requested_direction) {
    uint32_t src_is_l1 = idma_mm_is_l1_addr(src_addr);
    uint32_t dst_is_l1 = idma_mm_is_l1_addr(dst_addr);

    if (src_is_l1 && dst_is_l1) {
        return 2u;
    }
    if (!src_is_l1 && dst_is_l1) {
        return IDMA_DIR_L2_TO_L1;
    }
    if (src_is_l1 && !dst_is_l1) {
        return IDMA_DIR_L1_TO_L2;
    }
    return requested_direction;
}

static uint32_t run_idma_1d(const npu_cmd_idma_1d_t *cmd, uint32_t cmd_ptr) {
    int tx_id;
    uint32_t direction = resolve_dma_direction(cmd->src_addr, cmd->dst_addr, cmd->direction);

    if (direction == IDMA_DIR_L2_TO_L1) {
        tx_id = idma_L2ToL1(cmd->src_addr, cmd->dst_addr, cmd->length);
        return wait_tx_or_fail(IDMA_DIR_L2_TO_L1, tx_id, cmd_ptr);
    }
    if (direction == IDMA_DIR_L1_TO_L2) {
        tx_id = idma_L1ToL2(cmd->src_addr, cmd->dst_addr, cmd->length);
        return wait_tx_or_fail(IDMA_DIR_L1_TO_L2, tx_id, cmd_ptr);
    }
    if (direction == 2u) {
        idma_L1ToL1(cmd->src_addr, cmd->dst_addr, cmd->length);
        return NPU_CMD_FAIL_NONE;
    }
    return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_ptr);
}

static uint32_t run_idma_2d(const npu_cmd_idma_2d_t *cmd, uint32_t cmd_ptr) {
    int tx_id;
    uint32_t direction = resolve_dma_direction(cmd->src_addr, cmd->dst_addr, cmd->direction);

    if (direction == IDMA_DIR_L2_TO_L1) {
        tx_id = idma_L2ToL1_2d(cmd->src_addr, cmd->dst_addr, cmd->length,
                               cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2);
        return wait_tx_or_fail(IDMA_DIR_L2_TO_L1, tx_id, cmd_ptr);
    }
    if (direction == IDMA_DIR_L1_TO_L2) {
        tx_id = idma_L1ToL2_2d(cmd->src_addr, cmd->dst_addr, cmd->length,
                               cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2);
        return wait_tx_or_fail(IDMA_DIR_L1_TO_L2, tx_id, cmd_ptr);
    }
    if (direction == 2u) {
        idma_L1ToL1_2d(cmd->src_addr, cmd->dst_addr, cmd->length,
                       cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2);
        return NPU_CMD_FAIL_NONE;
    }
    return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_ptr);
}

static uint32_t run_idma_3d(const npu_cmd_idma_3d_t *cmd, uint32_t cmd_ptr) {
    int tx_id;
    uint32_t direction = resolve_dma_direction(cmd->src_addr, cmd->dst_addr, cmd->direction);

    if (direction == IDMA_DIR_L2_TO_L1) {
        tx_id = idma_L2ToL1_3d(cmd->src_addr, cmd->dst_addr, cmd->length,
                               cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2,
                               cmd->src_stride_3, cmd->dst_stride_3, cmd->reps_3);
        return wait_tx_or_fail(IDMA_DIR_L2_TO_L1, tx_id, cmd_ptr);
    }
    if (direction == IDMA_DIR_L1_TO_L2) {
        tx_id = idma_L1ToL2_3d(cmd->src_addr, cmd->dst_addr, cmd->length,
                               cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2,
                               cmd->src_stride_3, cmd->dst_stride_3, cmd->reps_3);
        return wait_tx_or_fail(IDMA_DIR_L1_TO_L2, tx_id, cmd_ptr);
    }
    if (direction == 2u) {
        idma_L1ToL1_3d(cmd->src_addr, cmd->dst_addr, cmd->length,
                       cmd->src_stride_2, cmd->dst_stride_2, cmd->reps_2,
                       cmd->src_stride_3, cmd->dst_stride_3, cmd->reps_3);
        return NPU_CMD_FAIL_NONE;
    }
    return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_ptr);
}

static uint32_t run_systolic(const npu_cmd_systolic_gemm32_t *cmd, uint32_t cmd_ptr) {
    uint32_t flags = cmd->header.flags;
    if ((flags & NPU_CMD_FLAG_ACCUM) && (flags & NPU_CMD_FLAG_REQUANT)) {
        systolic_gemm32_accumulate_requant(cmd->weight_addr, cmd->ifm_addr,
                                           cmd->psum_addr, cmd->ofm_addr, cmd->dim_m);
        return NPU_CMD_FAIL_NONE;
    }
    if (flags & NPU_CMD_FLAG_ACCUM) {
        systolic_gemm32_accumulate(cmd->weight_addr, cmd->ifm_addr,
                                   cmd->psum_addr, cmd->ofm_addr, cmd->dim_m);
        return NPU_CMD_FAIL_NONE;
    }
    if (flags & NPU_CMD_FLAG_REQUANT) {
        systolic_gemm32_requant(cmd->weight_addr, cmd->ifm_addr, cmd->ofm_addr, cmd->dim_m);
        return NPU_CMD_FAIL_NONE;
    }
    (void)cmd_ptr;
    systolic_gemm32(cmd->weight_addr, cmd->ifm_addr, cmd->ofm_addr, cmd->dim_m);
    return NPU_CMD_FAIL_NONE;
}

static uint32_t run_rolling_buffer(const npu_cmd_rolling_buffer_t *cmd, uint32_t cmd_ptr) {
    if (cmd->buffer_id >= 4u) {
        return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x01u, cmd_ptr);
    }

    npu_cmd_rolling_state_t *state = &g_rolling_state[cmd->buffer_id];

    if (cmd->op == NPU_CMD_ROLLING_RESET) {
        if (cmd->slot_count == 0u || cmd->slot_bytes == 0u) {
            return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x02u, cmd_ptr);
        }
        state->configured = 1u;
        state->base_addr = cmd->base_addr;
        state->slot_bytes = cmd->slot_bytes;
        state->slot_count = cmd->slot_count;
        state->produce_slot = 0u;
        state->consume_slot = 0u;
        state->occupancy = 0u;
        return NPU_CMD_FAIL_NONE;
    }

    if (!state->configured) {
        return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x03u, cmd_ptr);
    }
    if (cmd->base_addr != 0u && cmd->base_addr != state->base_addr) {
        return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x04u, cmd_ptr);
    }

    if (cmd->op == NPU_CMD_ROLLING_PRODUCE) {
        if (state->occupancy >= state->slot_count) {
            return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x05u, cmd_ptr);
        }
        if (cmd->expected_slot != state->produce_slot) {
            return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x06u, cmd_ptr);
        }
        state->produce_slot++;
        if (state->produce_slot == state->slot_count) {
            state->produce_slot = 0u;
        }
        state->occupancy++;
        return NPU_CMD_FAIL_NONE;
    }

    if (cmd->op == NPU_CMD_ROLLING_CONSUME_RELEASE) {
        if (state->occupancy == 0u) {
            return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x07u, cmd_ptr);
        }
        if (cmd->expected_slot != state->consume_slot) {
            return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x08u, cmd_ptr);
        }
        state->consume_slot++;
        if (state->consume_slot == state->slot_count) {
            state->consume_slot = 0u;
        }
        state->occupancy--;
        return NPU_CMD_FAIL_NONE;
    }

    return cmd_fail(NPU_CMD_FAIL_ROLLING | 0x09u, cmd_ptr);
}

uint32_t npu_cmd_dispatch(uint32_t cmd_tcdm_base, uint32_t cmd_total_bytes) {
    npu_cmd_table_header_t table;
    uint32_t cmd_ptr;
    uint32_t cmd_end;

    g_cmd_done_count = 0u;
    reset_rolling_state();

    if (!is_aligned32(cmd_tcdm_base) || !is_aligned32(cmd_total_bytes)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_ALIGN, cmd_tcdm_base);
    }
    if (cmd_total_bytes < sizeof(npu_cmd_table_header_t)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_SIZE, cmd_tcdm_base);
    }

    copy_from_cmd(&table, (const volatile void *)cmd_tcdm_base, sizeof(table));
    cmd_record_phase(0x100u, table.magic);
    if (table.magic != NPU_CMD_TABLE_MAGIC || table.version != NPU_CMD_VERSION ||
        table.total_bytes > cmd_total_bytes || table.entry_offset < sizeof(table) ||
        table.entry_offset >= table.total_bytes || !is_aligned32(table.entry_offset)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_TABLE, cmd_tcdm_base);
    }

    cmd_ptr = cmd_tcdm_base + table.entry_offset;
    cmd_end = cmd_tcdm_base + table.total_bytes;
    cmd_set_status(NPU_CMD_STATUS_RUNNING);
    cmd_record_phase(0x110u, cmd_ptr);

    while (cmd_ptr < cmd_end) {
        npu_cmd_header_t header;
        copy_from_cmd(&header, (const volatile void *)cmd_ptr, sizeof(header));
        cmd_record_phase(0x200u | (uint32_t)header.type, header.size_bytes);

        if (header.size_bytes < sizeof(header) || !is_aligned32(header.size_bytes) ||
            (cmd_ptr + header.size_bytes) > cmd_end) {
            return cmd_fail(NPU_CMD_FAIL_BAD_COMMAND, cmd_ptr);
        }

        if (header.type == NPU_CMD_TYPE_END) {
            REG_WRITE(NPU_CMD_DONE_COUNT, g_cmd_done_count);
            REG_WRITE(NPU_CMD_FAIL_CODE, NPU_CMD_FAIL_NONE);
            REG_WRITE(NPU_CMD_FAIL_PTR, 0u);
            cmd_set_status(NPU_CMD_STATUS_PASS);
            return NPU_CMD_FAIL_NONE;
        }

        uint32_t status = NPU_CMD_FAIL_NONE;
        switch (header.type) {
            case NPU_CMD_TYPE_IDMA_1D: {
                npu_cmd_idma_1d_t cmd;
                cmd_record_phase(0x301u, cmd_ptr);
                copy_from_cmd(&cmd, (const volatile void *)cmd_ptr, sizeof(cmd));
                cmd_record_phase(0x302u, cmd.length);
                status = run_idma_1d(&cmd, cmd_ptr);
                cmd_record_phase(0x303u, status);
                break;
            }
            case NPU_CMD_TYPE_IDMA_2D: {
                npu_cmd_idma_2d_t cmd;
                cmd_record_phase(0x321u, cmd_ptr);
                copy_from_cmd(&cmd, (const volatile void *)cmd_ptr, sizeof(cmd));
                cmd_record_phase(0x322u, cmd.length);
                status = run_idma_2d(&cmd, cmd_ptr);
                cmd_record_phase(0x323u, status);
                break;
            }
            case NPU_CMD_TYPE_IDMA_3D: {
                npu_cmd_idma_3d_t cmd;
                cmd_record_phase(0x341u, cmd_ptr);
                copy_from_cmd(&cmd, (const volatile void *)cmd_ptr, sizeof(cmd));
                cmd_record_phase(0x342u, cmd.length);
                status = run_idma_3d(&cmd, cmd_ptr);
                cmd_record_phase(0x343u, status);
                break;
            }
            case NPU_CMD_TYPE_SYSTOLIC_GEMM32: {
                npu_cmd_systolic_gemm32_t cmd;
                cmd_record_phase(0x361u, cmd_ptr);
                copy_from_cmd(&cmd, (const volatile void *)cmd_ptr, sizeof(cmd));
                cmd_record_phase(0x362u, cmd.dim_m);
                status = run_systolic(&cmd, cmd_ptr);
                cmd_record_phase(0x363u, status);
                break;
            }
            case NPU_CMD_TYPE_BARRIER:
                cmd_record_phase(0x380u, cmd_ptr);
                dma_barrier();
                break;
            case NPU_CMD_TYPE_ROLLING_BUFFER: {
                npu_cmd_rolling_buffer_t cmd;
                cmd_record_phase(0x3A1u, cmd_ptr);
                copy_from_cmd(&cmd, (const volatile void *)cmd_ptr, sizeof(cmd));
                status = run_rolling_buffer(&cmd, cmd_ptr);
                cmd_record_phase(0x3A2u, status);
                break;
            }
            default:
                return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_ptr);
        }

        if (status != NPU_CMD_FAIL_NONE) {
            return status;
        }

        g_cmd_done_count++;
        REG_WRITE(NPU_CMD_DONE_COUNT, g_cmd_done_count);
        cmd_ptr += header.size_bytes;
    }

    return cmd_fail(NPU_CMD_FAIL_BAD_TABLE, cmd_end);
}

static uint32_t npu_cmd_dispatch_stream(uint32_t cmd_l2_base, uint32_t cmd_total_bytes,
                                        uint32_t cmd_tcdm_base, uint32_t cmd_tcdm_bytes) {
    npu_cmd_stream_t stream;
    npu_cmd_table_header_t table;
    uint32_t cmd_offset;
    uint32_t cmd_end;
    uint32_t status;

    g_cmd_done_count = 0u;
    reset_rolling_state();

    if (!is_aligned32(cmd_l2_base) || !is_aligned32(cmd_total_bytes) ||
        !is_aligned32(cmd_tcdm_base) || !is_aligned32(cmd_tcdm_bytes)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_ALIGN, cmd_tcdm_base);
    }
    if (cmd_total_bytes < sizeof(npu_cmd_table_header_t) ||
        cmd_tcdm_bytes < NPU_CMD_TCDM_SIZE) {
        return cmd_fail(NPU_CMD_FAIL_BAD_SIZE, cmd_tcdm_base);
    }

    stream.l2_base = cmd_l2_base;
    stream.total_bytes = cmd_total_bytes;
    stream.buffer_base = cmd_tcdm_base;
    stream.buffer_bytes = NPU_CMD_TCDM_SIZE;
    stream.window_offset = 0u;
    stream.window_bytes = 0u;

    status = stream_copy_from_cmd(&stream, &table, 0u, sizeof(table), cmd_l2_base);
    if (status != NPU_CMD_FAIL_NONE) {
        return status;
    }

    cmd_record_phase(0x100u, table.magic);
    if (table.magic != NPU_CMD_TABLE_MAGIC || table.version != NPU_CMD_VERSION ||
        table.total_bytes > cmd_total_bytes || table.entry_offset < sizeof(table) ||
        table.entry_offset >= table.total_bytes || !is_aligned32(table.entry_offset)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_TABLE, cmd_l2_base);
    }

    cmd_offset = table.entry_offset;
    cmd_end = table.total_bytes;
    cmd_set_status(NPU_CMD_STATUS_RUNNING);
    cmd_record_phase(0x110u, cmd_l2_base + cmd_offset);

    while (cmd_offset < cmd_end) {
        npu_cmd_header_t header;
        uint32_t cmd_ptr = cmd_l2_base + cmd_offset;

        status = stream_copy_from_cmd(&stream, &header, cmd_offset, sizeof(header), cmd_ptr);
        if (status != NPU_CMD_FAIL_NONE) {
            return status;
        }
        cmd_record_phase(0x200u | (uint32_t)header.type, header.size_bytes);

        if (header.size_bytes < sizeof(header) || !is_aligned32(header.size_bytes) ||
            (cmd_offset + header.size_bytes) > cmd_end) {
            return cmd_fail(NPU_CMD_FAIL_BAD_COMMAND, cmd_ptr);
        }

        if (header.type == NPU_CMD_TYPE_END) {
            REG_WRITE(NPU_CMD_DONE_COUNT, g_cmd_done_count);
            REG_WRITE(NPU_CMD_FAIL_CODE, NPU_CMD_FAIL_NONE);
            REG_WRITE(NPU_CMD_FAIL_PTR, 0u);
            cmd_set_status(NPU_CMD_STATUS_PASS);
            return NPU_CMD_FAIL_NONE;
        }

        status = NPU_CMD_FAIL_NONE;
        switch (header.type) {
            case NPU_CMD_TYPE_IDMA_1D: {
                npu_cmd_idma_1d_t cmd;
                cmd_record_phase(0x301u, cmd_ptr);
                status = stream_copy_from_cmd(&stream, &cmd, cmd_offset, sizeof(cmd), cmd_ptr);
                if (status == NPU_CMD_FAIL_NONE) {
                    cmd_record_phase(0x302u, cmd.length);
                    status = run_idma_1d(&cmd, cmd_ptr);
                    cmd_record_phase(0x303u, status);
                }
                break;
            }
            case NPU_CMD_TYPE_IDMA_2D: {
                npu_cmd_idma_2d_t cmd;
                cmd_record_phase(0x321u, cmd_ptr);
                status = stream_copy_from_cmd(&stream, &cmd, cmd_offset, sizeof(cmd), cmd_ptr);
                if (status == NPU_CMD_FAIL_NONE) {
                    cmd_record_phase(0x322u, cmd.length);
                    status = run_idma_2d(&cmd, cmd_ptr);
                    cmd_record_phase(0x323u, status);
                }
                break;
            }
            case NPU_CMD_TYPE_IDMA_3D: {
                npu_cmd_idma_3d_t cmd;
                cmd_record_phase(0x341u, cmd_ptr);
                status = stream_copy_from_cmd(&stream, &cmd, cmd_offset, sizeof(cmd), cmd_ptr);
                if (status == NPU_CMD_FAIL_NONE) {
                    cmd_record_phase(0x342u, cmd.length);
                    status = run_idma_3d(&cmd, cmd_ptr);
                    cmd_record_phase(0x343u, status);
                }
                break;
            }
            case NPU_CMD_TYPE_SYSTOLIC_GEMM32: {
                npu_cmd_systolic_gemm32_t cmd;
                cmd_record_phase(0x361u, cmd_ptr);
                status = stream_copy_from_cmd(&stream, &cmd, cmd_offset, sizeof(cmd), cmd_ptr);
                if (status == NPU_CMD_FAIL_NONE) {
                    cmd_record_phase(0x362u, cmd.dim_m);
                    status = run_systolic(&cmd, cmd_ptr);
                    cmd_record_phase(0x363u, status);
                }
                break;
            }
            case NPU_CMD_TYPE_BARRIER:
                cmd_record_phase(0x380u, cmd_ptr);
                dma_barrier();
                break;
            case NPU_CMD_TYPE_ROLLING_BUFFER: {
                npu_cmd_rolling_buffer_t cmd;
                cmd_record_phase(0x3A1u, cmd_ptr);
                status = stream_copy_from_cmd(&stream, &cmd, cmd_offset, sizeof(cmd), cmd_ptr);
                if (status == NPU_CMD_FAIL_NONE) {
                    status = run_rolling_buffer(&cmd, cmd_ptr);
                    cmd_record_phase(0x3A2u, status);
                }
                break;
            }
            default:
                return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_ptr);
        }

        if (status != NPU_CMD_FAIL_NONE) {
            return status;
        }

        g_cmd_done_count++;
        REG_WRITE(NPU_CMD_DONE_COUNT, g_cmd_done_count);
        cmd_offset += header.size_bytes;
    }

    return cmd_fail(NPU_CMD_FAIL_BAD_TABLE, cmd_l2_base + cmd_end);
}

uint32_t npu_cmd_dispatch_from_ctrl(void) {
    uint32_t cmd_l2_base = REG_READ(NPU_CMD_L2_BASE);
    uint32_t cmd_total_bytes = REG_READ(NPU_CMD_TOTAL_BYTES);
    uint32_t cmd_tcdm_base = REG_READ(NPU_CMD_TCDM_BASE_REG);
    uint32_t cmd_tcdm_bytes = REG_READ(NPU_CMD_TCDM_BYTES);

    REG_WRITE(NPU_CMD_FAIL_CODE, NPU_CMD_FAIL_NONE);
    REG_WRITE(NPU_CMD_FAIL_PTR, 0u);
    REG_WRITE(NPU_CMD_DONE_COUNT, 0u);
    cmd_set_status(NPU_CMD_STATUS_LOADING);

    if (cmd_total_bytes == 0u || cmd_tcdm_bytes < NPU_CMD_TCDM_SIZE) {
        return cmd_fail(NPU_CMD_FAIL_BAD_SIZE, cmd_tcdm_base);
    }
    if (!is_aligned32(cmd_l2_base) || !is_aligned32(cmd_tcdm_base) || !is_aligned32(cmd_total_bytes)) {
        return cmd_fail(NPU_CMD_FAIL_BAD_ALIGN, cmd_tcdm_base);
    }

    if (!idma_memcpy_blocking(cmd_l2_base, cmd_tcdm_base, NPU_CMD_ALIGN_BYTES)) {
        return cmd_fail(NPU_CMD_FAIL_COPY, cmd_l2_base);
    }
    if (*(volatile uint32_t *)(unsigned long)cmd_tcdm_base == NAI_INVOCATION_MAGIC) {
        if (nai_runtime_dispatch_from_ctrl == 0) {
            return cmd_fail(NPU_CMD_FAIL_UNSUPPORTED, cmd_l2_base);
        }
        return nai_runtime_dispatch_from_ctrl(cmd_l2_base, cmd_total_bytes, cmd_tcdm_base, cmd_tcdm_bytes);
    }
    return npu_cmd_dispatch_stream(cmd_l2_base, cmd_total_bytes, cmd_tcdm_base, cmd_tcdm_bytes);
}
