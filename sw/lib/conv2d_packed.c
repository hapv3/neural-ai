#include "conv2d_packed.h"
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "spatz_ops.h"
#include "spatz_rt.h"

#ifndef NPU_CONV2D_ENABLE_SPATZ_RECT
#define NPU_CONV2D_ENABLE_SPATZ_RECT 1
#endif

#define NPU_CONV2D_MAX_PREPARE_CMDS 256u
#define NPU_CONV2D_LINEBUF_K_MAX 5u
#define NPU_CONV2D_LINEBUF_MAX_INPUT_W 640u
#define NPU_CONV2D_LINEBUF_KGEN_MAX_M 1024u
#define NPU_CONV2D_LINEBUF_PSUM_BUF_M 256u

typedef enum {
    NPU_CONV2D_PREPARE_SCALAR = 0,
    NPU_CONV2D_PREPARE_IDMA = 1,
    NPU_CONV2D_PREPARE_SPATZ = 2
} npu_conv2d_prepare_backend_t;

typedef struct {
    uint32_t k_block;
    uint32_t src_base;
    uint32_t dst_base;
    uint32_t channels;
    uint32_t valid_oh;
    uint32_t valid_ow;
    uint32_t src_row_stride;
    uint32_t dst_row_stride;
    int32_t src_col_stride;
    int32_t dst_col_stride;
} npu_conv2d_prepare_cmd_t;

typedef struct {
    npu_conv2d_prepare_cmd_t cmds[NPU_CONV2D_MAX_PREPARE_CMDS];
    uint32_t count;
    uint32_t overflow;
} npu_conv2d_prepare_plan_t;

typedef struct {
    systolic_linebuf_cfg_t linebuf;
    systolic_gemm32_req_t gemm;
    uint32_t rows;
    uint32_t k_tiles;
} npu_conv2d_linebuf_job_t;

static uint32_t input_c_stride(const npu_conv2d_packed_cfg_t *cfg);
static uint32_t input_row_stride_bytes(const npu_conv2d_packed_cfg_t *cfg);
static void make_linebuf_output_tile_cfg(const npu_conv2d_packed_cfg_t *cfg,
                                         uint32_t oh_base,
                                         uint32_t ow_base,
                                         uint32_t tile_oh,
                                         uint32_t tile_ow,
                                         uint32_t output_elem_bytes,
                                         npu_conv2d_packed_cfg_t *tile_cfg);

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static void clear_stats(npu_conv2d_packed_stats_t *stats) {
    if (stats) {
        stats->rows = 0;
        stats->k_tiles = 0;
        stats->prepare_cycles = 0;
        stats->gemm_cycles = 0;
        stats->total_cycles = 0;
        stats->last_prepare_cycles = 0;
        stats->last_gemm_cycles = 0;
        stats->status = NPU_CONV2D_PACKED_OK;
        stats->prepare_idma_tiles = 0;
        stats->prepare_idma_transfers = 0;
        stats->prepare_spatz_tiles = 0;
        stats->prepare_scalar_tiles = 0;
    }
}

static uint32_t validate_cfg(const npu_conv2d_packed_cfg_t *cfg) {
    if (!cfg || cfg->input_c == 0u || cfg->kernel_h == 0u || cfg->kernel_w == 0u ||
        cfg->output_h == 0u || cfg->output_w == 0u) {
        return NPU_CONV2D_PACKED_ERR_BAD_SHAPE;
    }

    if (cfg->dilation_h != 1u || cfg->dilation_w != 1u) {
        return NPU_CONV2D_PACKED_ERR_DILATION;
    }

    uint32_t stride_c = input_c_stride(cfg);
    if (cfg->input_c_base + cfg->input_c > stride_c) {
        uint32_t c32_blocked_group =
            (cfg->input_c_stride == NPU_CONV2D_PACKED_K_TILE) &&
            (cfg->input_c_base == 0u) &&
            (cfg->input_c > NPU_CONV2D_PACKED_K_TILE) &&
            ((cfg->input_c % NPU_CONV2D_PACKED_K_TILE) == 0u);
        if (!c32_blocked_group) {
            return NPU_CONV2D_PACKED_ERR_BAD_SHAPE;
        }
    }

    return NPU_CONV2D_PACKED_OK;
}

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

static uint32_t input_c_stride(const npu_conv2d_packed_cfg_t *cfg) {
    return cfg->input_c_stride ? cfg->input_c_stride : cfg->input_c;
}

static uint32_t input_row_stride_bytes(const npu_conv2d_packed_cfg_t *cfg) {
    return cfg->input_row_stride_bytes ? cfg->input_row_stride_bytes :
                                         (cfg->input_w * input_c_stride(cfg));
}

static uint32_t linebuf_valid_bytes(uint32_t input_c, uint32_t c_base, uint32_t lane_base) {
    uint32_t lane_room = (lane_base >= SYSTOLIC_GEMM32_K) ? 0u : (SYSTOLIC_GEMM32_K - lane_base);
    if (c_base >= input_c || lane_room == 0u) {
        return 0u;
    }
    return min_u32(input_c - c_base, lane_room);
}

static void linebuf_finalize_precompute(systolic_linebuf_cfg_t *cfg, uint32_t request_c32_fast) {
    cfg->block_valid_bytes = (uint16_t)linebuf_valid_bytes(cfg->input_c, cfg->c_base, cfg->lane_base);
    cfg->channel_addr_offset = cfg->c_base;
    cfg->coalesce_k_bytes = (uint32_t)cfg->kernel_h * (uint32_t)cfg->kernel_w *
                            (uint32_t)cfg->block_valid_bytes;
    cfg->c32_fast = (uint16_t)(request_c32_fast &&
                               cfg->block_valid_bytes == SYSTOLIC_GEMM32_K &&
                               cfg->pixel_stride_bytes == SYSTOLIC_GEMM32_K &&
                               cfg->lane_base == 0u &&
                               cfg->c_base == 0u &&
                               ((cfg->input_base & (SYSTOLIC_GEMM32_K - 1u)) == 0u) &&
                               ((cfg->channel_addr_offset & (SYSTOLIC_GEMM32_K - 1u)) == 0u));
}

static uint32_t input_pixel_addr(const npu_conv2d_packed_cfg_t *cfg,
                                 uint32_t ih,
                                 uint32_t iw,
                                 uint32_t local_ic) {
    uint32_t stride_c = input_c_stride(cfg);
    return cfg->input_addr + ((((ih * cfg->input_w) + iw) * stride_c) + cfg->input_c_base + local_ic);
}

static uint32_t is_contiguous_conv1x1(const npu_conv2d_packed_cfg_t *cfg) {
    return cfg->kernel_h == 1u &&
           cfg->kernel_w == 1u &&
           cfg->stride_h == 1u &&
           cfg->stride_w == 1u &&
           cfg->pad_h == 0u &&
           cfg->pad_w == 0u &&
           cfg->output_h == cfg->input_h &&
           cfg->output_w == cfg->input_w;
}

static uint32_t is_l1_addr(uint32_t addr) {
    return idma_mm_is_l1_addr(addr);
}

static uint32_t is_linebuf_supported(const npu_conv2d_packed_cfg_t *cfg) {
    return is_l1_addr(cfg->input_addr) &&
           is_l1_addr(cfg->weight_addr) &&
           is_l1_addr(cfg->output_addr) &&
           cfg->kernel_h <= NPU_CONV2D_LINEBUF_K_MAX &&
           cfg->kernel_w <= NPU_CONV2D_LINEBUF_K_MAX &&
           cfg->input_w <= NPU_CONV2D_LINEBUF_MAX_INPUT_W;
}

static uint32_t is_linebuf_coalesce_supported(const npu_conv2d_packed_cfg_t *cfg,
                                              uint32_t spatial_rows,
                                              uint32_t k_total) {
    return is_linebuf_supported(cfg) &&
           k_total <= NPU_CONV2D_PACKED_K_TILE &&
           spatial_rows <= SYSTOLIC_GEMM32_TILE_M;
}

static uint32_t is_linebuf_kgen_channel_block_safe(const npu_conv2d_packed_cfg_t *cfg) {
    return cfg->input_c <= NPU_CONV2D_PACKED_K_TILE ||
           (cfg->input_c % NPU_CONV2D_PACKED_K_TILE) == 0u;
}

static uint32_t is_linebuf_kgen_supported(const npu_conv2d_packed_cfg_t *cfg,
                                          uint32_t spatial_rows,
                                          uint32_t k_total) {
    return is_linebuf_supported(cfg) &&
           is_linebuf_kgen_channel_block_safe(cfg) &&
           k_total > NPU_CONV2D_PACKED_K_TILE &&
           spatial_rows <= NPU_CONV2D_LINEBUF_KGEN_MAX_M;
}

static uint32_t is_linebuf_kgen_shape_supported(const npu_conv2d_packed_cfg_t *cfg,
                                                uint32_t k_total) {
    return is_linebuf_supported(cfg) &&
           is_linebuf_kgen_channel_block_safe(cfg) &&
           k_total > NPU_CONV2D_PACKED_K_TILE;
}

static void accumulate_conv_stats(npu_conv2d_packed_stats_t *accum,
                                  const npu_conv2d_packed_stats_t *tile) {
    if (!accum || !tile) {
        return;
    }

    accum->rows += tile->rows;
    accum->k_tiles += tile->k_tiles;
    accum->prepare_cycles += tile->prepare_cycles;
    accum->gemm_cycles += tile->gemm_cycles;
    accum->total_cycles += tile->total_cycles;
    accum->last_prepare_cycles = tile->last_prepare_cycles;
    accum->last_gemm_cycles = tile->last_gemm_cycles;
    accum->status = tile->status;
    accum->prepare_idma_tiles += tile->prepare_idma_tiles;
    accum->prepare_idma_transfers += tile->prepare_idma_transfers;
    accum->prepare_spatz_tiles += tile->prepare_spatz_tiles;
    accum->prepare_scalar_tiles += tile->prepare_scalar_tiles;
}

static void copy_conv_stats(npu_conv2d_packed_stats_t *dst,
                            const npu_conv2d_packed_stats_t *src) {
    if (!dst || !src) {
        return;
    }

    dst->rows = src->rows;
    dst->k_tiles = src->k_tiles;
    dst->prepare_cycles = src->prepare_cycles;
    dst->gemm_cycles = src->gemm_cycles;
    dst->total_cycles = src->total_cycles;
    dst->last_prepare_cycles = src->last_prepare_cycles;
    dst->last_gemm_cycles = src->last_gemm_cycles;
    dst->status = src->status;
    dst->prepare_idma_tiles = src->prepare_idma_tiles;
    dst->prepare_idma_transfers = src->prepare_idma_transfers;
    dst->prepare_spatz_tiles = src->prepare_spatz_tiles;
    dst->prepare_scalar_tiles = src->prepare_scalar_tiles;
}

static void zero_im2col_tile(const npu_conv2d_packed_cfg_t *cfg, uint32_t rows) {
    uint32_t bytes = rows * NPU_CONV2D_PACKED_K_TILE;
    if (((cfg->im2col_addr | bytes) & 0x3u) == 0u) {
        spatz_vec_zero_i32((uint32_t *)cfg->im2col_addr, bytes >> 2);
    } else {
        spatz_vec_zero_i8((int8_t *)cfg->im2col_addr, bytes);
    }
}

static uint32_t wait_idma_or_fail(uint32_t direction, int tx_id) {
    if (tx_id <= 0) {
        return 0;
    }
    return idma_mm_wait_for_completion(direction, (uint32_t)tx_id);
}

static npu_conv2d_prepare_backend_t prepare_conv1x1_contiguous(const npu_conv2d_packed_cfg_t *cfg,
                                                               uint32_t k_block,
                                                               uint32_t rows,
                                                               uint32_t *idma_transfers) {
    uint32_t k_base = k_block * NPU_CONV2D_PACKED_K_TILE;
    uint32_t valid = (k_base < cfg->input_c) ? min_u32(NPU_CONV2D_PACKED_K_TILE, cfg->input_c - k_base) : 0u;
    uint32_t stride_c = input_c_stride(cfg);

    zero_im2col_tile(cfg, rows);
    if (valid == 0u) {
        return NPU_CONV2D_PREPARE_SPATZ;
    }

    uint32_t src = cfg->input_addr + cfg->input_c_base + k_base;
    uint32_t dst = cfg->im2col_addr;

    if (!is_l1_addr(src)) {
        int tx_id = idma_L2ToL1_2d(src, dst, valid, stride_c, NPU_CONV2D_PACKED_K_TILE, rows);
        if (wait_idma_or_fail(IDMA_DIR_L2_TO_L1, tx_id)) {
            *idma_transfers = 1u;
            return NPU_CONV2D_PREPARE_IDMA;
        }
    }

    for (uint32_t row = 0; row < rows; row++) {
        const int8_t *row_src = (const int8_t *)(src + (row * stride_c));
        int8_t *row_dst = (int8_t *)(dst + (row * NPU_CONV2D_PACKED_K_TILE));
        spatz_vec_copy_i8(row_src, row_dst, valid);
    }

    return NPU_CONV2D_PREPARE_SPATZ;
}

static int32_t signed_coord(uint32_t base, uint32_t offset, uint32_t pad) {
    return (int32_t)base + (int32_t)offset - (int32_t)pad;
}

static uint32_t output_valid_range(uint32_t output_len,
                                   uint32_t stride,
                                   uint32_t kernel_offset,
                                   uint32_t pad,
                                   uint32_t input_len,
                                   uint32_t *first,
                                   uint32_t *count) {
    uint32_t begin = output_len;
    uint32_t end = 0;

    for (uint32_t out = 0; out < output_len; out++) {
        int32_t in = signed_coord(out * stride, kernel_offset, pad);
        if (in >= 0 && (uint32_t)in < input_len) {
            if (begin == output_len) {
                begin = out;
            }
            end = out + 1u;
        }
    }

    if (begin == output_len) {
        *first = 0;
        *count = 0;
        return 0;
    }

    *first = begin;
    *count = end - begin;
    return 1;
}

static void build_prepare_plan(const npu_conv2d_packed_cfg_t *cfg,
                               uint32_t k_tiles,
                               npu_conv2d_prepare_plan_t *plan) {
    plan->count = 0;
    plan->overflow = 0;

    uint32_t kernel_spatial = cfg->kernel_h * cfg->kernel_w;
    uint32_t k_total = kernel_spatial * cfg->input_c;
    uint32_t stride_c = input_c_stride(cfg);

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t k_base = k_block * NPU_CONV2D_PACKED_K_TILE;
        uint32_t lane = 0;

        while (lane < NPU_CONV2D_PACKED_K_TILE) {
            uint32_t k_index = k_base + lane;
            if (k_index >= k_total) {
                break;
            }

            uint32_t spatial_index = k_index / cfg->input_c;
            uint32_t channel_base = k_index - (spatial_index * cfg->input_c);
            uint32_t kh = spatial_index / cfg->kernel_w;
            uint32_t kw = spatial_index - (kh * cfg->kernel_w);
            uint32_t valid = min_u32(NPU_CONV2D_PACKED_K_TILE - lane, cfg->input_c - channel_base);
            uint32_t first_oh;
            uint32_t valid_oh;
            uint32_t first_ow;
            uint32_t valid_ow;

            if (k_index + valid > k_total) {
                valid = k_total - k_index;
            }
            if (valid == 0u) {
                plan->overflow = 1u;
                return;
            }

            if (output_valid_range(cfg->output_h, cfg->stride_h, kh, cfg->pad_h, cfg->input_h, &first_oh, &valid_oh) &&
                output_valid_range(cfg->output_w, cfg->stride_w, kw, cfg->pad_w, cfg->input_w, &first_ow, &valid_ow)) {
                if (plan->count >= NPU_CONV2D_MAX_PREPARE_CMDS) {
                    plan->overflow = 1u;
                    return;
                }

                int32_t ih = signed_coord(first_oh * cfg->stride_h, kh * cfg->dilation_h, cfg->pad_h);
                int32_t iw = signed_coord(first_ow * cfg->stride_w, kw * cfg->dilation_w, cfg->pad_w);
                npu_conv2d_prepare_cmd_t *cmd = &plan->cmds[plan->count++];
                cmd->k_block = k_block;
                cmd->src_base = input_pixel_addr(cfg, (uint32_t)ih, (uint32_t)iw, channel_base);
                cmd->dst_base = cfg->im2col_addr + (((first_oh * cfg->output_w) + first_ow) * NPU_CONV2D_PACKED_K_TILE) + lane;
                cmd->channels = valid;
                cmd->valid_oh = valid_oh;
                cmd->valid_ow = valid_ow;
                cmd->src_row_stride = cfg->stride_h * cfg->input_w * stride_c;
                cmd->dst_row_stride = cfg->output_w * NPU_CONV2D_PACKED_K_TILE;
                cmd->src_col_stride = (int32_t)(cfg->stride_w * stride_c);
                cmd->dst_col_stride = (int32_t)NPU_CONV2D_PACKED_K_TILE;
            }

            lane += valid;
        }
    }
}

static uint32_t try_prepare_plan_idma(const npu_conv2d_packed_cfg_t *cfg,
                                      uint32_t k_block,
                                      const npu_conv2d_prepare_plan_t *plan,
                                      uint32_t *idma_transfers) {
    int last_tx_id = 0;
    uint32_t transfers = 0;

    if (is_l1_addr(cfg->input_addr) || plan->overflow) {
        return 0;
    }

    for (uint32_t idx = 0; idx < plan->count; idx++) {
        const npu_conv2d_prepare_cmd_t *cmd = &plan->cmds[idx];
        if (cmd->k_block != k_block) {
            continue;
        }

        int tx_id = idma_L2ToL1_3d(cmd->src_base, cmd->dst_base, cmd->channels,
                                   (uint32_t)cmd->src_col_stride,
                                   (uint32_t)cmd->dst_col_stride,
                                   cmd->valid_ow,
                                   cmd->src_row_stride,
                                   cmd->dst_row_stride,
                                   cmd->valid_oh);
        if (tx_id <= 0) {
            if (last_tx_id > 0) {
                (void)wait_idma_or_fail(IDMA_DIR_L2_TO_L1, last_tx_id);
            }
            return 0;
        }
        last_tx_id = tx_id;
        transfers++;
    }

    if (last_tx_id <= 0) {
        return 0;
    }
    if (!wait_idma_or_fail(IDMA_DIR_L2_TO_L1, last_tx_id)) {
        return 0;
    }

    *idma_transfers = transfers;
    return 1;
}

static void prepare_spatz_segmented(const npu_conv2d_packed_cfg_t *cfg,
                                    uint32_t k_block,
                                    uint32_t rows) {
    uint32_t kernel_spatial = cfg->kernel_h * cfg->kernel_w;
    uint32_t k_total = kernel_spatial * cfg->input_c;
    uint32_t k_base = k_block * NPU_CONV2D_PACKED_K_TILE;

    zero_im2col_tile(cfg, rows);

    for (uint32_t row = 0; row < rows; row++) {
        uint32_t oh = row / cfg->output_w;
        uint32_t ow = row - (oh * cfg->output_w);
        uint32_t lane = 0;

        while (lane < NPU_CONV2D_PACKED_K_TILE) {
            uint32_t k_index = k_base + lane;
            if (k_index >= k_total) {
                break;
            }

            uint32_t spatial_index = k_index / cfg->input_c;
            uint32_t ic = k_index - (spatial_index * cfg->input_c);
            uint32_t kh = spatial_index / cfg->kernel_w;
            uint32_t kw = spatial_index - (kh * cfg->kernel_w);
            int32_t ih = signed_coord(oh * cfg->stride_h, kh * cfg->dilation_h, cfg->pad_h);
            int32_t iw = signed_coord(ow * cfg->stride_w, kw * cfg->dilation_w, cfg->pad_w);
            uint32_t run = min_u32(NPU_CONV2D_PACKED_K_TILE - lane, cfg->input_c - ic);

            if (k_index + run > k_total) {
                run = k_total - k_index;
            }

            if (ih >= 0 && iw >= 0 && (uint32_t)ih < cfg->input_h && (uint32_t)iw < cfg->input_w) {
                uint32_t src = input_pixel_addr(cfg, (uint32_t)ih, (uint32_t)iw, ic);
                uint32_t dst = cfg->im2col_addr + (row * NPU_CONV2D_PACKED_K_TILE) + lane;
                spatz_vec_copy_i8((const int8_t *)src, (int8_t *)dst, run);
            }

            lane += run;
        }
    }
}

static uint32_t prepare_spatz_rect_strided(const npu_conv2d_packed_cfg_t *cfg,
                                           uint32_t k_block,
                                           uint32_t rows,
                                           const npu_conv2d_prepare_plan_t *plan) {
    uint32_t emitted = 0;

    if (!is_l1_addr(cfg->input_addr) || rows != (cfg->output_h * cfg->output_w) || plan->overflow) {
        return 0;
    }

    for (uint32_t idx = 0; idx < plan->count; idx++) {
        const npu_conv2d_prepare_cmd_t *cmd = &plan->cmds[idx];
        if (cmd->k_block != k_block) {
            continue;
        }

        spatz_im2col_rect_channels_i8((const int8_t *)cmd->src_base,
                                      (int8_t *)cmd->dst_base,
                                      cmd->channels,
                                      cmd->valid_oh,
                                      cmd->valid_ow,
                                      cmd->src_col_stride,
                                      cmd->dst_col_stride,
                                      cmd->src_row_stride,
                                      cmd->dst_row_stride);
        emitted = 1u;
    }

    return emitted;
}

static npu_conv2d_prepare_backend_t prepare_k_tile(const npu_conv2d_packed_cfg_t *cfg,
                                                   uint32_t k_block,
                                                   uint32_t rows,
                                                   const npu_conv2d_prepare_plan_t *plan,
                                                   uint32_t *idma_transfers) {
    *idma_transfers = 0;
    if (is_contiguous_conv1x1(cfg)) {
        return prepare_conv1x1_contiguous(cfg, k_block, rows, idma_transfers);
    }

    zero_im2col_tile(cfg, rows);
    if (try_prepare_plan_idma(cfg, k_block, plan, idma_transfers)) {
        return NPU_CONV2D_PREPARE_IDMA;
    }

#if NPU_CONV2D_ENABLE_SPATZ_RECT
    if (prepare_spatz_rect_strided(cfg, k_block, rows, plan)) {
        return NPU_CONV2D_PREPARE_SPATZ;
    }
#endif

    prepare_spatz_segmented(cfg, k_block, rows);
    return NPU_CONV2D_PREPARE_SPATZ;
}

uint32_t npu_conv2d_packed_run_oc32(const npu_conv2d_packed_cfg_t *cfg,
                                    npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    if (is_linebuf_supported(cfg)) {
        return npu_conv2d_packed_run_oc32_linebuf(cfg, stats);
    }

    uint32_t rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    npu_conv2d_prepare_plan_t prepare_plan;

    build_prepare_plan(cfg, k_tiles, &prepare_plan);
    uint32_t total_start = spatz_rt_read_cycle();

    if (stats) {
        stats->rows = rows;
        stats->k_tiles = k_tiles;
    }

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t weight_addr = cfg->weight_addr + (k_block * NPU_CONV2D_PACKED_K_TILE * NPU_CONV2D_PACKED_OC_TILE);
        uint32_t idma_transfers = 0;

        uint32_t prepare_start = spatz_rt_read_cycle();
        npu_conv2d_prepare_backend_t backend = prepare_k_tile(cfg, k_block, rows, &prepare_plan, &idma_transfers);
        uint32_t prepare_cycles = spatz_rt_read_cycle() - prepare_start;

        uint32_t gemm_start = spatz_rt_read_cycle();
        if (k_block == 0u && !cfg->accumulate) {
            systolic_gemm32(weight_addr, cfg->im2col_addr, cfg->output_addr, rows);
        } else {
            systolic_gemm32_accumulate(weight_addr, cfg->im2col_addr, cfg->output_addr, cfg->output_addr, rows);
        }
        uint32_t gemm_cycles = spatz_rt_read_cycle() - gemm_start;

        if (stats) {
            stats->prepare_cycles += prepare_cycles;
            stats->gemm_cycles += gemm_cycles;
            stats->last_prepare_cycles = prepare_cycles;
            stats->last_gemm_cycles = gemm_cycles;
            if (backend == NPU_CONV2D_PREPARE_IDMA) {
                stats->prepare_idma_tiles++;
                stats->prepare_idma_transfers += idma_transfers;
            } else if (backend == NPU_CONV2D_PREPARE_SPATZ) {
                stats->prepare_spatz_tiles++;
            } else {
                stats->prepare_scalar_tiles++;
            }
        }
    }

    if (stats) {
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_run_oc32_requant(const npu_conv2d_packed_cfg_t *cfg,
                                            npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    uint32_t psum_addr = cfg->output_addr + (rows * NPU_CONV2D_PACKED_OC_TILE);
    npu_conv2d_prepare_plan_t prepare_plan;

    build_prepare_plan(cfg, k_tiles, &prepare_plan);
    uint32_t total_start = spatz_rt_read_cycle();

    if (stats) {
        stats->rows = rows;
        stats->k_tiles = k_tiles;
    }

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t weight_addr = cfg->weight_addr + (k_block * NPU_CONV2D_PACKED_K_TILE * NPU_CONV2D_PACKED_OC_TILE);
        uint32_t is_last = (k_block + 1u) == k_tiles;
        uint32_t idma_transfers = 0;

        uint32_t prepare_start = spatz_rt_read_cycle();
        npu_conv2d_prepare_backend_t backend = prepare_k_tile(cfg, k_block, rows, &prepare_plan, &idma_transfers);
        uint32_t prepare_cycles = spatz_rt_read_cycle() - prepare_start;

        uint32_t gemm_start = spatz_rt_read_cycle();
        if (k_block == 0u && is_last) {
            systolic_gemm32_requant(weight_addr, cfg->im2col_addr, cfg->output_addr, rows);
        } else if (k_block == 0u) {
            systolic_gemm32(weight_addr, cfg->im2col_addr, psum_addr, rows);
        } else if (is_last) {
            systolic_gemm32_accumulate_requant(weight_addr, cfg->im2col_addr, psum_addr, cfg->output_addr, rows);
        } else {
            systolic_gemm32_accumulate(weight_addr, cfg->im2col_addr, psum_addr, psum_addr, rows);
        }
        uint32_t gemm_cycles = spatz_rt_read_cycle() - gemm_start;

        if (stats) {
            stats->prepare_cycles += prepare_cycles;
            stats->gemm_cycles += gemm_cycles;
            stats->last_prepare_cycles = prepare_cycles;
            stats->last_gemm_cycles = gemm_cycles;
            if (backend == NPU_CONV2D_PREPARE_IDMA) {
                stats->prepare_idma_tiles++;
                stats->prepare_idma_transfers += idma_transfers;
            } else if (backend == NPU_CONV2D_PREPARE_SPATZ) {
                stats->prepare_spatz_tiles++;
            } else {
                stats->prepare_scalar_tiles++;
            }
        }
    }

    if (stats) {
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

static void linebuf_config_from_conv(const npu_conv2d_packed_cfg_t *cfg,
                                     uint32_t spatial_rows,
                                     uint32_t coalesce,
                                     uint32_t kgen,
                                     uint32_t k_tiles,
                                     systolic_linebuf_cfg_t *linebuf_cfg) {
    uint32_t stride_c = input_c_stride(cfg);
    uint32_t row_stride_bytes = input_row_stride_bytes(cfg);

    linebuf_cfg->input_base = cfg->input_addr + cfg->input_c_base - (cfg->pad_h * row_stride_bytes);
    linebuf_cfg->input_h = (uint16_t)cfg->input_h;
    linebuf_cfg->input_w = (uint16_t)cfg->input_w;
    linebuf_cfg->input_c = (uint16_t)cfg->input_c;
    linebuf_cfg->output_w = (uint16_t)cfg->output_w;
    linebuf_cfg->stride_h = (uint16_t)cfg->stride_h;
    linebuf_cfg->stride_w = (uint16_t)cfg->stride_w;
    linebuf_cfg->pad_h = (uint16_t)cfg->pad_h;
    linebuf_cfg->pad_w = (uint16_t)cfg->pad_w;
    linebuf_cfg->row_stride_bytes = row_stride_bytes;
    linebuf_cfg->pixel_stride_bytes = stride_c;
    linebuf_cfg->ow_step_bytes = cfg->stride_w * stride_c;
    linebuf_cfg->oh_step_bytes = cfg->stride_h * row_stride_bytes;
    linebuf_cfg->kernel_h = (uint16_t)cfg->kernel_h;
    linebuf_cfg->kernel_w = (uint16_t)cfg->kernel_w;
    linebuf_cfg->c_base = 0u;
    linebuf_cfg->lane_base = 0u;
    linebuf_cfg->coalesce = (uint16_t)coalesce;
    linebuf_cfg->kgen = (uint16_t)kgen;
    linebuf_cfg->pool = 0u;
    linebuf_cfg->k_seed_kh = 0u;
    linebuf_cfg->k_seed_kw = 0u;
    linebuf_cfg->k_seed_ic = 0u;
    linebuf_cfg->k_tiles = k_tiles;
    linebuf_cfg->spatial_m = spatial_rows;
    linebuf_finalize_precompute(linebuf_cfg,
                                (stride_c == SYSTOLIC_GEMM32_K) &&
                                (cfg->input_c == SYSTOLIC_GEMM32_K) &&
                                (cfg->input_c_base == 0u));
}

static void linebuf_job_from_tile_cfg(const npu_conv2d_packed_cfg_t *cfg,
                                      uint32_t psum_addr,
                                      uint32_t accum_en,
                                      uint32_t ofm_row_stride_bytes,
                                      uint32_t ofm_tile_cols,
                                      uint32_t psum_row_stride_bytes,
                                      npu_conv2d_linebuf_job_t *job) {
    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);

    linebuf_config_from_conv(cfg, spatial_rows, 1u, 1u, k_tiles, &job->linebuf);

    job->gemm.weight_addr = cfg->weight_addr;
    job->gemm.ifm_addr = 0u;
    job->gemm.psum_addr = psum_addr;
    job->gemm.ofm_addr = cfg->output_addr;
    job->gemm.dim_m = spatial_rows;
    job->gemm.accum_en = accum_en;
    job->gemm.ofm_row_stride_bytes = ofm_row_stride_bytes;
    job->gemm.ofm_tile_cols = ofm_tile_cols;
    job->gemm.psum_row_stride_bytes = psum_row_stride_bytes;
    job->rows = spatial_rows;
    job->k_tiles = k_tiles;
}

static void linebuf_job_preload(const npu_conv2d_linebuf_job_t *job) {
    systolic_linebuf_config(&job->linebuf);
    systolic_gemm32_preload(&job->gemm);
}

static void linebuf_job_record(npu_conv2d_packed_stats_t *stats,
                               const npu_conv2d_linebuf_job_t *job) {
    if (stats) {
        stats->rows += job->rows;
        stats->k_tiles += job->k_tiles;
    }
}

static void run_linebuf_kgen_tile(const npu_conv2d_packed_cfg_t *cfg,
                                  uint32_t ofm_row_stride_bytes,
                                  uint32_t ofm_tile_cols,
                                  npu_conv2d_packed_stats_t *stats) {
    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    systolic_linebuf_cfg_t linebuf_cfg;
    uint32_t total_start;
    uint32_t gemm_start;
    uint32_t gemm_cycles;

    clear_stats(stats);
    total_start = spatz_rt_read_cycle();
    gemm_start = spatz_rt_read_cycle();

    linebuf_config_from_conv(cfg, spatial_rows, 1u, 1u, k_tiles, &linebuf_cfg);
    systolic_linebuf_config(&linebuf_cfg);
    if (cfg->accumulate) {
        if (ofm_row_stride_bytes != 0u && ofm_tile_cols != 0u) {
            systolic_gemm32_linebuf_ktiles_accumulate_strided(cfg->weight_addr,
                                                              cfg->output_addr,
                                                              cfg->output_addr,
                                                              spatial_rows,
                                                              ofm_row_stride_bytes,
                                                              ofm_tile_cols,
                                                              ofm_row_stride_bytes);
        } else {
            systolic_gemm32_linebuf_ktiles_accumulate(cfg->weight_addr, cfg->output_addr, cfg->output_addr, spatial_rows);
        }
    } else {
        if (ofm_row_stride_bytes != 0u && ofm_tile_cols != 0u) {
            systolic_gemm32_linebuf_ktiles_strided(cfg->weight_addr, cfg->output_addr,
                                                   cfg->output_addr, spatial_rows,
                                                   ofm_row_stride_bytes, ofm_tile_cols);
        } else {
            systolic_gemm32_linebuf_ktiles(cfg->weight_addr, cfg->output_addr, cfg->output_addr, spatial_rows);
        }
    }

    gemm_cycles = spatz_rt_read_cycle() - gemm_start;
    if (stats) {
        stats->rows = spatial_rows;
        stats->k_tiles = k_tiles;
        stats->prepare_cycles = 0u;
        stats->gemm_cycles = gemm_cycles;
        stats->last_prepare_cycles = 0u;
        stats->last_gemm_cycles = gemm_cycles;
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }
}

static uint32_t run_linebuf_kgen_tile_requant(const npu_conv2d_packed_cfg_t *cfg,
                                              uint32_t psum_addr,
                                              uint32_t ofm_row_stride_bytes,
                                              uint32_t ofm_tile_cols,
                                              npu_conv2d_packed_stats_t *stats) {
    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    systolic_linebuf_cfg_t linebuf_cfg;
    uint32_t total_start;
    uint32_t gemm_start;

    clear_stats(stats);
    if (cfg->accumulate || k_tiles == 0u || spatial_rows > NPU_CONV2D_LINEBUF_PSUM_BUF_M) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    total_start = spatz_rt_read_cycle();
    gemm_start = spatz_rt_read_cycle();

    linebuf_config_from_conv(cfg, spatial_rows, 1u, 1u, k_tiles, &linebuf_cfg);
    systolic_linebuf_config(&linebuf_cfg);
    if (ofm_row_stride_bytes != 0u && ofm_tile_cols != 0u) {
        systolic_gemm32_linebuf_ktiles_requant_strided(cfg->weight_addr, psum_addr,
                                                       cfg->output_addr, spatial_rows,
                                                       ofm_row_stride_bytes,
                                                       ofm_tile_cols);
    } else if (k_tiles == 1u) {
        systolic_gemm32_linebuf_requant(cfg->weight_addr, cfg->output_addr, spatial_rows);
    } else {
        systolic_gemm32_linebuf_ktiles_requant(cfg->weight_addr, psum_addr,
                                               cfg->output_addr, spatial_rows);
    }

    if (stats) {
        stats->rows = spatial_rows;
        stats->k_tiles = k_tiles;
        stats->prepare_cycles = 0u;
        stats->gemm_cycles = spatz_rt_read_cycle() - gemm_start;
        stats->last_prepare_cycles = 0u;
        stats->last_gemm_cycles = stats->gemm_cycles;
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

static uint32_t run_linebuf_kgen_tile_accumulate_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                         uint32_t psum_addr,
                                                         uint32_t ofm_row_stride_bytes,
                                                         uint32_t ofm_tile_cols,
                                                         uint32_t psum_row_stride_bytes,
                                                         npu_conv2d_packed_stats_t *stats) {
    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    systolic_linebuf_cfg_t linebuf_cfg;
    uint32_t total_start;
    uint32_t gemm_start;

    clear_stats(stats);
    if (!cfg->accumulate || k_tiles == 0u || spatial_rows > NPU_CONV2D_LINEBUF_PSUM_BUF_M) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    total_start = spatz_rt_read_cycle();
    gemm_start = spatz_rt_read_cycle();

    linebuf_config_from_conv(cfg, spatial_rows, 1u, 1u, k_tiles, &linebuf_cfg);
    systolic_linebuf_config(&linebuf_cfg);
    systolic_gemm32_linebuf_ktiles_accumulate_requant_strided(cfg->weight_addr,
                                                              psum_addr,
                                                              cfg->output_addr,
                                                              spatial_rows,
                                                              ofm_row_stride_bytes,
                                                              ofm_tile_cols,
                                                              psum_row_stride_bytes);

    if (stats) {
        stats->rows = spatial_rows;
        stats->k_tiles = k_tiles;
        stats->prepare_cycles = 0u;
        stats->gemm_cycles = spatz_rt_read_cycle() - gemm_start;
        stats->last_prepare_cycles = 0u;
        stats->last_gemm_cycles = stats->gemm_cycles;
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_linebuf_default_tile_oh(const npu_conv2d_packed_cfg_t *cfg) {
    uint32_t by_m = NPU_CONV2D_LINEBUF_KGEN_MAX_M / cfg->output_w;
    uint32_t by_cache = 1u;

    if (by_m == 0u) {
        by_m = 1u;
    }

    if (cfg->kernel_h == 1u && cfg->kernel_w == 1u && cfg->pad_h == 0u && cfg->pad_w == 0u) {
        return by_m;
    }

    if (cfg->kernel_h <= NPU_CONV2D_LINEBUF_K_MAX && cfg->stride_h != 0u) {
        by_cache = ((NPU_CONV2D_LINEBUF_K_MAX - cfg->kernel_h) / cfg->stride_h) + 1u;
        if (by_cache == 0u) {
            by_cache = 1u;
        }
    }

    return min_u32(by_m, by_cache);
}

uint32_t npu_conv2d_packed_run_oc32_linebuf_tiles(const npu_conv2d_packed_cfg_t *cfg,
                                                  const npu_conv2d_spatial_tile_t *tiles,
                                                  uint32_t tile_count,
                                                  uint32_t output_elem_bytes,
                                                  npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    if (!tiles || tile_count == 0u ||
        !is_linebuf_kgen_shape_supported(cfg, k_total) ||
        (output_elem_bytes != 1u && output_elem_bytes != 4u)) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    uint32_t total_start = spatz_rt_read_cycle();
    npu_conv2d_packed_stats_t total_stats;
    clear_stats(&total_stats);

    for (uint32_t idx = 0; idx < tile_count; idx++) {
        const npu_conv2d_spatial_tile_t *tile = &tiles[idx];
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        if (tile->tile_oh == 0u || tile_ow == 0u ||
            tile->oh_base >= cfg->output_h || tile->ow_base >= cfg->output_w ||
            tile->oh_base + tile->tile_oh > cfg->output_h ||
            tile->ow_base + tile_ow > cfg->output_w) {
            if (stats) {
                stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
            }
            return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
    }

    npu_conv2d_linebuf_job_t job_slots[2];
    npu_conv2d_linebuf_job_t *cur_job = &job_slots[0];
    npu_conv2d_linebuf_job_t *next_job = &job_slots[1];
    {
        const npu_conv2d_spatial_tile_t *tile = &tiles[0];
        npu_conv2d_packed_cfg_t tile_cfg;
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        uint32_t row_stride = (tile_ow != cfg->output_w) ?
            (cfg->output_w * NPU_CONV2D_PACKED_OC_TILE * output_elem_bytes) : 0u;
        uint32_t tile_cols = (tile_ow != cfg->output_w) ? tile_ow : 0u;

        make_linebuf_output_tile_cfg(cfg, tile->oh_base, tile->ow_base,
                                     tile->tile_oh, tile_ow,
                                     output_elem_bytes, &tile_cfg);
        linebuf_job_from_tile_cfg(&tile_cfg,
                                  tile_cfg.output_addr,
                                  cfg->accumulate ? 1u : 0u,
                                  row_stride,
                                  tile_cols,
                                  row_stride,
                                  cur_job);
        linebuf_job_preload(cur_job);
        systolic_gemm32_start_preloaded();
    }

    for (uint32_t idx = 1; idx < tile_count; idx++) {
        const npu_conv2d_spatial_tile_t *tile = &tiles[idx];
        npu_conv2d_packed_cfg_t tile_cfg;
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        uint32_t row_stride = (tile_ow != cfg->output_w) ?
            (cfg->output_w * NPU_CONV2D_PACKED_OC_TILE * output_elem_bytes) : 0u;
        uint32_t tile_cols = (tile_ow != cfg->output_w) ? tile_ow : 0u;

        make_linebuf_output_tile_cfg(cfg, tile->oh_base, tile->ow_base,
                                     tile->tile_oh, tile_ow,
                                     output_elem_bytes, &tile_cfg);
        linebuf_job_from_tile_cfg(&tile_cfg,
                                  tile_cfg.output_addr,
                                  cfg->accumulate ? 1u : 0u,
                                  row_stride,
                                  tile_cols,
                                  row_stride,
                                  next_job);
        linebuf_job_preload(next_job);
        systolic_gemm32_wait_done();
        linebuf_job_record(&total_stats, cur_job);
        systolic_gemm32_start_preloaded();
        {
            npu_conv2d_linebuf_job_t *tmp = cur_job;
            cur_job = next_job;
            next_job = tmp;
        }
    }

    systolic_gemm32_wait_done();
    linebuf_job_record(&total_stats, cur_job);
    systolic_linebuf_disable();

    if (stats) {
        copy_conv_stats(stats, &total_stats);
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->gemm_cycles = stats->total_cycles;
        stats->last_gemm_cycles = stats->total_cycles;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_run_oc32_linebuf_tiles_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                          const npu_conv2d_spatial_tile_t *tiles,
                                                          uint32_t tile_count,
                                                          uint32_t psum_addr,
                                                          npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    if (!tiles || tile_count == 0u || psum_addr == 0u || cfg->accumulate ||
        !is_linebuf_kgen_shape_supported(cfg, k_total)) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    uint32_t total_start = spatz_rt_read_cycle();
    npu_conv2d_packed_stats_t total_stats;
    clear_stats(&total_stats);

    for (uint32_t idx = 0; idx < tile_count; idx++) {
        const npu_conv2d_spatial_tile_t *tile = &tiles[idx];
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        if (tile->tile_oh == 0u || tile_ow == 0u ||
            tile->oh_base >= cfg->output_h || tile->ow_base >= cfg->output_w ||
            tile->oh_base + tile->tile_oh > cfg->output_h ||
            tile->ow_base + tile_ow > cfg->output_w) {
            if (stats) {
                stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
            }
            return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
    }

    npu_conv2d_linebuf_job_t job_slots[2];
    npu_conv2d_linebuf_job_t *cur_job = &job_slots[0];
    npu_conv2d_linebuf_job_t *next_job = &job_slots[1];
    {
        const npu_conv2d_spatial_tile_t *tile = &tiles[0];
        npu_conv2d_packed_cfg_t tile_cfg;
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        uint32_t tile_psum_addr = psum_addr +
                                  (((tile->oh_base * cfg->output_w) + tile->ow_base) *
                                   NPU_CONV2D_PACKED_OC_TILE * 4u);
        uint32_t row_stride = (tile_ow != cfg->output_w) ?
            (cfg->output_w * NPU_CONV2D_PACKED_OC_TILE) : 0u;
        uint32_t tile_cols = (tile_ow != cfg->output_w) ? tile_ow : 0u;

        make_linebuf_output_tile_cfg(cfg, tile->oh_base, tile->ow_base,
                                     tile->tile_oh, tile_ow,
                                     1u, &tile_cfg);
        linebuf_job_from_tile_cfg(&tile_cfg,
                                  tile_psum_addr,
                                  0u,
                                  row_stride,
                                  tile_cols,
                                  0u,
                                  cur_job);
        linebuf_job_preload(cur_job);
        systolic_gemm32_start_preloaded();
    }

    for (uint32_t idx = 1; idx < tile_count; idx++) {
        const npu_conv2d_spatial_tile_t *tile = &tiles[idx];
        npu_conv2d_packed_cfg_t tile_cfg;
        uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
        uint32_t tile_psum_addr = psum_addr +
                                  (((tile->oh_base * cfg->output_w) + tile->ow_base) *
                                   NPU_CONV2D_PACKED_OC_TILE * 4u);
        uint32_t row_stride = (tile_ow != cfg->output_w) ?
            (cfg->output_w * NPU_CONV2D_PACKED_OC_TILE) : 0u;
        uint32_t tile_cols = (tile_ow != cfg->output_w) ? tile_ow : 0u;

        make_linebuf_output_tile_cfg(cfg, tile->oh_base, tile->ow_base,
                                     tile->tile_oh, tile_ow,
                                     1u, &tile_cfg);
        linebuf_job_from_tile_cfg(&tile_cfg,
                                  tile_psum_addr,
                                  0u,
                                  row_stride,
                                  tile_cols,
                                  0u,
                                  next_job);
        linebuf_job_preload(next_job);
        systolic_gemm32_wait_done();
        linebuf_job_record(&total_stats, cur_job);
        systolic_gemm32_start_preloaded();
        {
            npu_conv2d_linebuf_job_t *tmp = cur_job;
            cur_job = next_job;
            next_job = tmp;
        }
    }

    systolic_gemm32_wait_done();
    linebuf_job_record(&total_stats, cur_job);
    systolic_linebuf_disable();

    if (stats) {
        copy_conv_stats(stats, &total_stats);
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->gemm_cycles = stats->total_cycles;
        stats->last_gemm_cycles = stats->total_cycles;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                                    const npu_conv2d_spatial_tile_t *tile,
                                                                    uint32_t psum_addr,
                                                                    uint32_t output_addr,
                                                                    npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    if (!tile || psum_addr == 0u || output_addr == 0u || !cfg->accumulate ||
        !is_linebuf_kgen_shape_supported(cfg, k_total)) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    uint32_t tile_ow = tile->tile_ow ? tile->tile_ow : cfg->output_w;
    if (tile->tile_oh == 0u || tile_ow == 0u ||
        tile->oh_base >= cfg->output_h || tile->ow_base >= cfg->output_w ||
        tile->oh_base + tile->tile_oh > cfg->output_h ||
        tile->ow_base + tile_ow > cfg->output_w) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    npu_conv2d_packed_cfg_t tile_cfg;
    make_linebuf_output_tile_cfg(cfg, tile->oh_base, tile->ow_base,
                                 tile->tile_oh, tile_ow, 1u, &tile_cfg);
    tile_cfg.output_addr = output_addr;
    uint32_t tile_psum_addr = psum_addr +
                              (((tile->oh_base * cfg->output_w) + tile->ow_base) *
                               NPU_CONV2D_PACKED_OC_TILE * 4u);
    uint32_t tile_output_row_stride = tile_ow * NPU_CONV2D_PACKED_OC_TILE;
    uint32_t psum_row_stride = cfg->output_w * NPU_CONV2D_PACKED_OC_TILE * 4u;

    return run_linebuf_kgen_tile_accumulate_requant(&tile_cfg,
                                                    tile_psum_addr,
                                                    tile_output_row_stride,
                                                    tile_ow,
                                                    psum_row_stride,
                                                    stats);
}

static void make_linebuf_output_tile_cfg(const npu_conv2d_packed_cfg_t *cfg,
                                         uint32_t oh_base,
                                         uint32_t ow_base,
                                         uint32_t tile_oh,
                                         uint32_t tile_ow,
                                         uint32_t output_elem_bytes,
                                         npu_conv2d_packed_cfg_t *tile_cfg) {
    uint32_t stride_c = input_c_stride(cfg);
    uint32_t row_stride_bytes = input_row_stride_bytes(cfg);
    uint32_t last_oh = oh_base + tile_oh - 1u;
    uint32_t last_ow = ow_base + tile_ow - 1u;
    uint32_t first_y_unpadded = oh_base * cfg->stride_h;
    uint32_t last_y_kernel = (last_oh * cfg->stride_h) + cfg->kernel_h - 1u;
    uint32_t first_x_unpadded = ow_base * cfg->stride_w;
    uint32_t last_x_kernel = (last_ow * cfg->stride_w) + cfg->kernel_w - 1u;
    uint32_t first_ih = (first_y_unpadded > cfg->pad_h) ? (first_y_unpadded - cfg->pad_h) : 0u;
    uint32_t last_ih = (last_y_kernel > cfg->pad_h) ? (last_y_kernel - cfg->pad_h) : 0u;
    uint32_t first_iw = (first_x_unpadded > cfg->pad_w) ? (first_x_unpadded - cfg->pad_w) : 0u;
    uint32_t last_iw = (last_x_kernel > cfg->pad_w) ? (last_x_kernel - cfg->pad_w) : 0u;

    if (last_ih >= cfg->input_h) {
        last_ih = cfg->input_h - 1u;
    }
    if (first_ih >= cfg->input_h) {
        first_ih = cfg->input_h - 1u;
    }
    if (last_iw >= cfg->input_w) {
        last_iw = cfg->input_w - 1u;
    }
    if (first_iw >= cfg->input_w) {
        first_iw = cfg->input_w - 1u;
    }

    tile_cfg->input_addr = cfg->input_addr;
    tile_cfg->weight_addr = cfg->weight_addr;
    tile_cfg->im2col_addr = cfg->im2col_addr;
    tile_cfg->output_addr = cfg->output_addr;
    tile_cfg->input_h = cfg->input_h;
    tile_cfg->input_w = cfg->input_w;
    tile_cfg->input_c = cfg->input_c;
    tile_cfg->output_h = cfg->output_h;
    tile_cfg->output_w = cfg->output_w;
    tile_cfg->kernel_h = cfg->kernel_h;
    tile_cfg->kernel_w = cfg->kernel_w;
    tile_cfg->stride_h = cfg->stride_h;
    tile_cfg->stride_w = cfg->stride_w;
    tile_cfg->pad_h = cfg->pad_h;
    tile_cfg->pad_w = cfg->pad_w;
    tile_cfg->dilation_h = cfg->dilation_h;
    tile_cfg->dilation_w = cfg->dilation_w;
    tile_cfg->input_c_stride = cfg->input_c_stride;
    tile_cfg->input_row_stride_bytes = row_stride_bytes;
    tile_cfg->input_c_base = cfg->input_c_base;
    tile_cfg->accumulate = cfg->accumulate;
    tile_cfg->input_addr = cfg->input_addr + (first_ih * row_stride_bytes) + (first_iw * stride_c);
    tile_cfg->input_h = last_ih - first_ih + 1u;
    tile_cfg->input_w = last_iw - first_iw + 1u;
    tile_cfg->output_h = tile_oh;
    tile_cfg->output_w = tile_ow;
    tile_cfg->output_addr = cfg->output_addr +
                            (((oh_base * cfg->output_w) + ow_base) *
                             NPU_CONV2D_PACKED_OC_TILE * output_elem_bytes);
    {
        uint32_t shifted_pad_h = cfg->pad_h + first_ih;
        tile_cfg->pad_h = (shifted_pad_h > first_y_unpadded) ? (shifted_pad_h - first_y_unpadded) : 0u;
    }
    {
        uint32_t shifted_pad_w = cfg->pad_w + first_iw;
        tile_cfg->pad_w = (shifted_pad_w > first_x_unpadded) ? (shifted_pad_w - first_x_unpadded) : 0u;
    }
}

static uint32_t linebuf_coalesce_requant_tile(const npu_conv2d_packed_cfg_t *cfg,
                                              npu_conv2d_packed_stats_t *stats) {
    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;

    if (!is_linebuf_coalesce_supported(cfg, spatial_rows, k_total)) {
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    systolic_linebuf_cfg_t linebuf_cfg;
    linebuf_config_from_conv(cfg, spatial_rows, 1u, 0u, 0u, &linebuf_cfg);
    systolic_linebuf_config(&linebuf_cfg);
    systolic_gemm32_linebuf_requant(cfg->weight_addr, cfg->output_addr, spatial_rows);

    if (stats) {
        stats->rows = spatial_rows;
        stats->k_tiles = 1u;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_run_oc32_linebuf(const npu_conv2d_packed_cfg_t *cfg,
                                            npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t spatial_rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);

    if (!is_linebuf_supported(cfg)) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    uint32_t total_start = spatz_rt_read_cycle();
    uint32_t gemm_start = spatz_rt_read_cycle();

    if (is_linebuf_coalesce_supported(cfg, spatial_rows, k_total)) {
        systolic_linebuf_cfg_t linebuf_cfg;
        linebuf_config_from_conv(cfg, spatial_rows, 1u, 0u, 0u, &linebuf_cfg);

        systolic_linebuf_config(&linebuf_cfg);
        if (cfg->accumulate) {
            systolic_gemm32_linebuf_accumulate(cfg->weight_addr, cfg->output_addr, cfg->output_addr, spatial_rows);
        } else {
            systolic_gemm32_linebuf(cfg->weight_addr, cfg->output_addr, spatial_rows);
        }

        uint32_t gemm_cycles = spatz_rt_read_cycle() - gemm_start;
        if (stats) {
            stats->rows = spatial_rows;
            stats->k_tiles = 1u;
            stats->prepare_cycles = 0u;
            stats->gemm_cycles = gemm_cycles;
            stats->last_prepare_cycles = 0u;
            stats->last_gemm_cycles = gemm_cycles;
            stats->total_cycles = spatz_rt_read_cycle() - total_start;
            stats->status = NPU_CONV2D_PACKED_OK;
        }

        return NPU_CONV2D_PACKED_OK;
    }

    if (is_linebuf_kgen_supported(cfg, spatial_rows, k_total)) {
        run_linebuf_kgen_tile(cfg, 0u, 0u, stats);
        return NPU_CONV2D_PACKED_OK;
    }

    if (is_linebuf_kgen_shape_supported(cfg, k_total)) {
        uint32_t tile_oh_step = npu_conv2d_packed_linebuf_default_tile_oh(cfg);
        npu_conv2d_packed_stats_t total_stats;

        clear_stats(&total_stats);
        if (tile_oh_step == 0u) {
            tile_oh_step = 1u;
        }

        for (uint32_t oh_base = 0; oh_base < cfg->output_h; oh_base += tile_oh_step) {
            npu_conv2d_packed_cfg_t tile_cfg;
            npu_conv2d_packed_stats_t tile_stats;
            uint32_t tile_oh = cfg->output_h - oh_base;

            if (tile_oh > tile_oh_step) {
                tile_oh = tile_oh_step;
            }

            make_linebuf_output_tile_cfg(cfg, oh_base, 0u, tile_oh, cfg->output_w,
                                         4u, &tile_cfg);
            run_linebuf_kgen_tile(&tile_cfg, 0u, 0u, &tile_stats);
            accumulate_conv_stats(&total_stats, &tile_stats);
        }

        if (stats) {
            copy_conv_stats(stats, &total_stats);
            stats->total_cycles = spatz_rt_read_cycle() - total_start;
            stats->status = NPU_CONV2D_PACKED_OK;
        }
        return NPU_CONV2D_PACKED_OK;
    }

    if (!cfg->accumulate) {
        spatz_vec_zero_i32((uint32_t *)cfg->output_addr, spatial_rows * NPU_CONV2D_PACKED_OC_TILE);
    }

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t lane = 0;
        uint32_t weight_addr = cfg->weight_addr + (k_block * NPU_CONV2D_PACKED_K_TILE * NPU_CONV2D_PACKED_OC_TILE);
        uint32_t stride_c = input_c_stride(cfg);

        while (lane < NPU_CONV2D_PACKED_K_TILE) {
            uint32_t k_index = (k_block * NPU_CONV2D_PACKED_K_TILE) + lane;
            if (k_index >= k_total) {
                break;
            }

            uint32_t spatial_index = k_index / cfg->input_c;
            uint32_t c_base = k_index - (spatial_index * cfg->input_c);
            uint32_t kh = spatial_index / cfg->kernel_w;
            uint32_t kw = spatial_index - (kh * cfg->kernel_w);
            uint32_t valid_lanes = min_u32(NPU_CONV2D_PACKED_K_TILE - lane, cfg->input_c - c_base);
            if (k_index + valid_lanes > k_total) {
                valid_lanes = k_total - k_index;
            }

            for (uint32_t oh = 0; oh < cfg->output_h; oh++) {
                int32_t ih = signed_coord(oh * cfg->stride_h, kh * cfg->dilation_h, cfg->pad_h);
                if (ih < 0 || (uint32_t)ih >= cfg->input_h) {
                    continue;
                }

                uint32_t first_ow;
                uint32_t valid_ow;
                if (!output_valid_range(cfg->output_w, cfg->stride_w, kw * cfg->dilation_w,
                                        cfg->pad_w, cfg->input_w, &first_ow, &valid_ow)) {
                    continue;
                }

                uint32_t ow_done = 0;
                while (ow_done < valid_ow) {
                    uint32_t chunk_ow = valid_ow - ow_done;
                    if (chunk_ow > SYSTOLIC_GEMM32_ACCUM_TILE_M) {
                        chunk_ow = SYSTOLIC_GEMM32_ACCUM_TILE_M;
                    }

                    uint32_t ow = first_ow + ow_done;
                    int32_t iw = signed_coord(ow * cfg->stride_w, kw * cfg->dilation_w, cfg->pad_w);
                    uint32_t input_base = input_pixel_addr(cfg, (uint32_t)ih, (uint32_t)iw, c_base);
                    uint32_t output_base = cfg->output_addr +
                                           (((oh * cfg->output_w) + ow) * NPU_CONV2D_PACKED_OC_TILE * 4u);
                    uint32_t local_input_w = ((chunk_ow - 1u) * cfg->stride_w) + 1u;
                    systolic_linebuf_cfg_t linebuf_cfg;

                    linebuf_cfg.input_base = input_base;
                    linebuf_cfg.input_h = 1u;
                    linebuf_cfg.input_w = (uint16_t)local_input_w;
                    linebuf_cfg.input_c = (uint16_t)valid_lanes;
                    linebuf_cfg.output_w = (uint16_t)chunk_ow;
                    linebuf_cfg.stride_h = 1u;
                    linebuf_cfg.stride_w = (uint16_t)cfg->stride_w;
                    linebuf_cfg.pad_h = 0u;
                    linebuf_cfg.pad_w = 0u;
                    linebuf_cfg.row_stride_bytes = input_row_stride_bytes(cfg);
                    linebuf_cfg.pixel_stride_bytes = stride_c;
                    linebuf_cfg.ow_step_bytes = cfg->stride_w * stride_c;
                    linebuf_cfg.oh_step_bytes = linebuf_cfg.row_stride_bytes;
                    linebuf_cfg.kernel_h = 1u;
                    linebuf_cfg.kernel_w = 1u;
                    linebuf_cfg.c_base = 0u;
                    linebuf_cfg.lane_base = (uint16_t)lane;
                    linebuf_cfg.coalesce = 0u;
                    linebuf_cfg.kgen = 0u;
                    linebuf_cfg.pool = 0u;
                    linebuf_cfg.k_seed_kh = 0u;
                    linebuf_cfg.k_seed_kw = 0u;
                    linebuf_cfg.k_seed_ic = 0u;
                    linebuf_cfg.k_tiles = 0u;
                    linebuf_cfg.spatial_m = chunk_ow;
                    linebuf_finalize_precompute(&linebuf_cfg,
                                                (stride_c == SYSTOLIC_GEMM32_K) &&
                                                (valid_lanes == SYSTOLIC_GEMM32_K) &&
                                                (lane == 0u));

                    systolic_linebuf_config(&linebuf_cfg);
                    systolic_gemm32_linebuf_accumulate(weight_addr, output_base, output_base, chunk_ow);

                    ow_done += chunk_ow;
                }
            }

            lane += valid_lanes;
        }
    }

    uint32_t gemm_cycles = spatz_rt_read_cycle() - gemm_start;

    if (stats) {
        stats->rows = spatial_rows;
        stats->k_tiles = k_tiles;
        stats->prepare_cycles = 0u;
        stats->gemm_cycles = gemm_cycles;
        stats->last_prepare_cycles = 0u;
        stats->last_gemm_cycles = gemm_cycles;
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}

uint32_t npu_conv2d_packed_run_oc32_linebuf_requant(const npu_conv2d_packed_cfg_t *cfg,
                                                    npu_conv2d_packed_stats_t *stats) {
    clear_stats(stats);

    uint32_t status = validate_cfg(cfg);
    if (status != NPU_CONV2D_PACKED_OK) {
        if (stats) {
            stats->status = status;
        }
        return status;
    }

    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    if (!is_linebuf_supported(cfg) || cfg->accumulate || k_total > NPU_CONV2D_PACKED_K_TILE) {
        if (stats) {
            stats->status = NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
        }
        return NPU_CONV2D_PACKED_ERR_LINEBUF_K_TILES;
    }

    uint32_t tile_oh_step = SYSTOLIC_GEMM32_TILE_M / cfg->output_w;
    if (tile_oh_step == 0u) {
        tile_oh_step = 1u;
    }

    uint32_t total_start = spatz_rt_read_cycle();
    npu_conv2d_packed_stats_t total_stats;
    clear_stats(&total_stats);

    for (uint32_t oh_base = 0; oh_base < cfg->output_h; oh_base += tile_oh_step) {
        npu_conv2d_packed_cfg_t tile_cfg;
        npu_conv2d_packed_stats_t tile_stats;
        uint32_t tile_oh = cfg->output_h - oh_base;
        if (tile_oh > tile_oh_step) {
            tile_oh = tile_oh_step;
        }

        make_linebuf_output_tile_cfg(cfg, oh_base, 0u, tile_oh, cfg->output_w,
                                     1u, &tile_cfg);

        uint32_t gemm_start = spatz_rt_read_cycle();
        status = linebuf_coalesce_requant_tile(&tile_cfg, &tile_stats);
        uint32_t gemm_cycles = spatz_rt_read_cycle() - gemm_start;
        if (status != NPU_CONV2D_PACKED_OK) {
            if (stats) {
                stats->status = status;
            }
            return status;
        }

        tile_stats.gemm_cycles = gemm_cycles;
        tile_stats.last_gemm_cycles = gemm_cycles;
        tile_stats.total_cycles = gemm_cycles;
        accumulate_conv_stats(&total_stats, &tile_stats);
    }

    if (stats) {
        copy_conv_stats(stats, &total_stats);
        stats->prepare_cycles = 0u;
        stats->last_prepare_cycles = 0u;
        stats->total_cycles = spatz_rt_read_cycle() - total_start;
        stats->status = NPU_CONV2D_PACKED_OK;
    }

    return NPU_CONV2D_PACKED_OK;
}
