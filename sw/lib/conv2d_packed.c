#include "conv2d_packed.h"
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "spatz_ops.h"
#include "spatz_rt.h"

typedef enum {
    NPU_CONV2D_PREPARE_SCALAR = 0,
    NPU_CONV2D_PREPARE_IDMA = 1,
    NPU_CONV2D_PREPARE_SPATZ = 2
} npu_conv2d_prepare_backend_t;

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

    return NPU_CONV2D_PACKED_OK;
}

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
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

static void zero_im2col_tile(const npu_conv2d_packed_cfg_t *cfg, uint32_t rows) {
    spatz_rt_memset((void *)cfg->im2col_addr, 0, rows * NPU_CONV2D_PACKED_K_TILE);
}

static uint32_t wait_idma_or_fail(uint32_t direction, int tx_id) {
    if (tx_id <= 0) {
        return 0;
    }
    return idma_mm_wait_for_completion(direction, (uint32_t)tx_id);
}

static npu_conv2d_prepare_backend_t prepare_conv1x1_contiguous(const npu_conv2d_packed_cfg_t *cfg,
                                                               uint32_t k_block,
                                                               uint32_t rows) {
    uint32_t k_base = k_block * NPU_CONV2D_PACKED_K_TILE;
    uint32_t valid = (k_base < cfg->input_c) ? min_u32(NPU_CONV2D_PACKED_K_TILE, cfg->input_c - k_base) : 0u;

    zero_im2col_tile(cfg, rows);
    if (valid == 0u) {
        return NPU_CONV2D_PREPARE_SPATZ;
    }

    uint32_t src = cfg->input_addr + k_base;
    uint32_t dst = cfg->im2col_addr;

    if (!is_l1_addr(src)) {
        int tx_id = idma_L2ToL1_2d(src, dst, valid, cfg->input_c, NPU_CONV2D_PACKED_K_TILE, rows);
        if (wait_idma_or_fail(IDMA_DIR_L2_TO_L1, tx_id)) {
            return NPU_CONV2D_PREPARE_IDMA;
        }
    }

    for (uint32_t row = 0; row < rows; row++) {
        const int8_t *row_src = (const int8_t *)(src + (row * cfg->input_c));
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

static uint32_t try_prepare_single_spatial_idma(const npu_conv2d_packed_cfg_t *cfg,
                                                uint32_t k_block,
                                                uint32_t rows) {
    uint32_t k_base = k_block * NPU_CONV2D_PACKED_K_TILE;
    uint32_t kernel_spatial = cfg->kernel_h * cfg->kernel_w;
    uint32_t k_total = kernel_spatial * cfg->input_c;

    if (k_base >= k_total || is_l1_addr(cfg->input_addr)) {
        return 0;
    }

    uint32_t spatial_index = k_base / cfg->input_c;
    uint32_t channel_base = k_base - (spatial_index * cfg->input_c);
    uint32_t tile_valid = min_u32(NPU_CONV2D_PACKED_K_TILE, k_total - k_base);
    uint32_t valid = min_u32(NPU_CONV2D_PACKED_K_TILE, cfg->input_c - channel_base);
    if (valid == 0u || valid != tile_valid || k_base + valid > k_total) {
        return 0;
    }

    uint32_t end_spatial_index = (k_base + valid - 1u) / cfg->input_c;
    if (end_spatial_index != spatial_index) {
        return 0;
    }

    uint32_t kh = spatial_index / cfg->kernel_w;
    uint32_t kw = spatial_index - (kh * cfg->kernel_w);
    uint32_t first_oh;
    uint32_t valid_oh;
    uint32_t first_ow;
    uint32_t valid_ow;

    if (!output_valid_range(cfg->output_h, cfg->stride_h, kh, cfg->pad_h, cfg->input_h, &first_oh, &valid_oh) ||
        !output_valid_range(cfg->output_w, cfg->stride_w, kw, cfg->pad_w, cfg->input_w, &first_ow, &valid_ow)) {
        return 1;
    }

    int32_t ih = signed_coord(first_oh * cfg->stride_h, kh, cfg->pad_h);
    int32_t iw = signed_coord(first_ow * cfg->stride_w, kw, cfg->pad_w);
    uint32_t src = cfg->input_addr + ((((uint32_t)ih * cfg->input_w + (uint32_t)iw) * cfg->input_c) + channel_base);
    uint32_t dst = cfg->im2col_addr + (((first_oh * cfg->output_w) + first_ow) * NPU_CONV2D_PACKED_K_TILE);
    uint32_t src_stride_2 = cfg->stride_w * cfg->input_c;
    uint32_t dst_stride_2 = NPU_CONV2D_PACKED_K_TILE;
    uint32_t src_stride_3 = cfg->stride_h * cfg->input_w * cfg->input_c;
    uint32_t dst_stride_3 = cfg->output_w * NPU_CONV2D_PACKED_K_TILE;
    int tx_id = idma_L2ToL1_3d(src, dst, valid,
                               src_stride_2, dst_stride_2, valid_ow,
                               src_stride_3, dst_stride_3, valid_oh);

    (void)rows;
    return wait_idma_or_fail(IDMA_DIR_L2_TO_L1, tx_id);
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
                uint32_t src = cfg->input_addr + ((((uint32_t)ih * cfg->input_w + (uint32_t)iw) * cfg->input_c) + ic);
                uint32_t dst = cfg->im2col_addr + (row * NPU_CONV2D_PACKED_K_TILE) + lane;
                spatz_vec_copy_i8((const int8_t *)src, (int8_t *)dst, run);
            }

            lane += run;
        }
    }
}

static npu_conv2d_prepare_backend_t prepare_k_tile(const npu_conv2d_packed_cfg_t *cfg,
                                                   uint32_t k_block,
                                                   uint32_t rows) {
    if (is_contiguous_conv1x1(cfg)) {
        return prepare_conv1x1_contiguous(cfg, k_block, rows);
    }

    zero_im2col_tile(cfg, rows);
    if (try_prepare_single_spatial_idma(cfg, k_block, rows)) {
        return NPU_CONV2D_PREPARE_IDMA;
    }

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

    uint32_t rows = cfg->output_h * cfg->output_w;
    uint32_t k_total = cfg->kernel_h * cfg->kernel_w * cfg->input_c;
    uint32_t k_tiles = ceil_div_u32(k_total, NPU_CONV2D_PACKED_K_TILE);
    uint32_t psum_addr = cfg->output_addr + (rows * NPU_CONV2D_PACKED_OC_TILE);
    uint32_t total_start = spatz_rt_read_cycle();

    if (stats) {
        stats->rows = rows;
        stats->k_tiles = k_tiles;
    }

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t weight_addr = cfg->weight_addr + (k_block * NPU_CONV2D_PACKED_K_TILE * NPU_CONV2D_PACKED_OC_TILE);

        uint32_t prepare_start = spatz_rt_read_cycle();
        npu_conv2d_prepare_backend_t backend = prepare_k_tile(cfg, k_block, rows);
        uint32_t prepare_cycles = spatz_rt_read_cycle() - prepare_start;

        uint32_t gemm_start = spatz_rt_read_cycle();
        if (k_block == 0u) {
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
    uint32_t total_start = spatz_rt_read_cycle();

    if (stats) {
        stats->rows = rows;
        stats->k_tiles = k_tiles;
    }

    for (uint32_t k_block = 0; k_block < k_tiles; k_block++) {
        uint32_t weight_addr = cfg->weight_addr + (k_block * NPU_CONV2D_PACKED_K_TILE * NPU_CONV2D_PACKED_OC_TILE);
        uint32_t is_last = (k_block + 1u) == k_tiles;

        uint32_t prepare_start = spatz_rt_read_cycle();
        npu_conv2d_prepare_backend_t backend = prepare_k_tile(cfg, k_block, rows);
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
