#include "hal_systolic.h"
#include "npu_memory_map.h"

void systolic_gemm32_preload(const systolic_gemm32_req_t *req) {
    REG_WRITE(REG_SYS_W_PTR, req->weight_addr);
    REG_WRITE(REG_SYS_I_PTR, req->ifm_addr);
    REG_WRITE(REG_SYS_O_PTR, req->ofm_addr);
    REG_WRITE(REG_SYS_PSUM_PTR, req->psum_addr);
    REG_WRITE(REG_SYS_DIM_M, req->dim_m);
    REG_WRITE(REG_SYS_ACCUM_CTRL, req->accum_en ? REG_SYS_ACCUM_CTRL_EN : 0u);
    REG_WRITE(REG_SYS_OFM_ROW_STRIDE, req->ofm_row_stride_bytes);
    REG_WRITE(REG_SYS_OFM_TILE_COLS, req->ofm_tile_cols);
    REG_WRITE(REG_SYS_PSUM_ROW_STRIDE, req->psum_row_stride_bytes);
}

void systolic_gemm32_start_preloaded(void) {
    REG_WRITE(REG_SYS_DONE, 0);
    REG_WRITE(REG_SYS_START, 1);
}

uint32_t systolic_gemm32_done(void) {
    return REG_READ(REG_SYS_DONE) != 0u;
}

void systolic_gemm32_wait_done(void) {
    while (!systolic_gemm32_done()) {
    }
    REG_WRITE(REG_SYS_DONE, 0);
}

static void systolic_gemm32_tile_ex(uint32_t weight_addr,
                                    uint32_t ifm_addr,
                                    uint32_t psum_addr,
                                    uint32_t ofm_addr,
                                    uint32_t dim_m,
                                    uint32_t accum_en,
                                    uint32_t ofm_row_stride_bytes,
                                    uint32_t ofm_tile_cols,
                                    uint32_t psum_row_stride_bytes) {
    systolic_gemm32_req_t req;

    req.weight_addr = weight_addr;
    req.ifm_addr = ifm_addr;
    req.psum_addr = psum_addr;
    req.ofm_addr = ofm_addr;
    req.dim_m = dim_m;
    req.accum_en = accum_en;
    req.ofm_row_stride_bytes = ofm_row_stride_bytes;
    req.ofm_tile_cols = ofm_tile_cols;
    req.psum_row_stride_bytes = psum_row_stride_bytes;

    systolic_gemm32_preload(&req);
    systolic_gemm32_start_preloaded();
    systolic_gemm32_wait_done();
    REG_WRITE(REG_SYS_ACCUM_CTRL, 0u);
}

static void systolic_gemm32_tile(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    systolic_gemm32_tile_ex(weight_addr, ifm_addr, 0u, ofm_addr, dim_m, 0u, 0u, 0u, 0u);
}

void systolic_linebuf_disable(void) {
    REG_WRITE(REG_LB_CTRL, 0u);
}

void systolic_linebuf_config(const systolic_linebuf_cfg_t *cfg) {
    REG_WRITE(REG_LB_CTRL, 0u);
    REG_WRITE(REG_LB_INPUT_BASE, cfg->input_base);
    REG_WRITE(REG_LB_INPUT_H, cfg->input_h);
    REG_WRITE(REG_LB_INPUT_W, cfg->input_w);
    REG_WRITE(REG_LB_INPUT_C, cfg->input_c);
    REG_WRITE(REG_LB_OUTPUT_W, cfg->output_w);
    REG_WRITE(REG_LB_STRIDE, ((uint32_t)cfg->stride_w << 16) | cfg->stride_h);
    REG_WRITE(REG_LB_PAD, ((uint32_t)cfg->pad_w << 16) | cfg->pad_h);
    REG_WRITE(REG_LB_ROW_STRIDE, cfg->row_stride_bytes);
    REG_WRITE(REG_LB_PIXEL_STRIDE, cfg->pixel_stride_bytes);
    REG_WRITE(REG_LB_OW_STEP, cfg->ow_step_bytes);
    REG_WRITE(REG_LB_OH_STEP, cfg->oh_step_bytes);
    REG_WRITE(REG_LB_KERNEL, ((uint32_t)cfg->kernel_w << 16) | cfg->kernel_h);
    REG_WRITE(REG_LB_C_BASE, cfg->c_base);
    REG_WRITE(REG_LB_SPATIAL_M, cfg->spatial_m);
    REG_WRITE(REG_LB_LANE_BASE, cfg->lane_base);
    REG_WRITE(REG_LB_K_TILES, cfg->k_tiles);
    REG_WRITE(REG_LB_K_SEED, ((uint32_t)(cfg->k_seed_kh & 0xFFu) << 24) |
                             ((uint32_t)(cfg->k_seed_kw & 0xFFu) << 16) |
                             (uint32_t)cfg->k_seed_ic);
    REG_WRITE(REG_LB_PRECOMP0, cfg->block_valid_bytes & 0x3Fu);
    REG_WRITE(REG_LB_CHANNEL_OFFSET, cfg->channel_addr_offset);
    REG_WRITE(REG_LB_COALESCE_K_BYTES, cfg->coalesce_k_bytes);

    REG_WRITE(REG_LB_CTRL, REG_LB_CTRL_EN |
                           (cfg->coalesce ? REG_LB_CTRL_COALESCE : 0u) |
                           (cfg->kgen ? REG_LB_CTRL_KGEN : 0u) |
                           (cfg->pool ? REG_LB_CTRL_POOL : 0u) |
                           (cfg->c32_fast ? REG_LB_CTRL_C32_FAST : 0u));
}

void systolic_maxpool5x5s1p2_c32_linebuf(uint32_t input_addr,
                                         uint32_t output_addr,
                                         uint32_t height,
                                         uint32_t width) {
    systolic_linebuf_cfg_t cfg;
    uint32_t row_stride = width * 32u;

    cfg.input_base = input_addr - (2u * row_stride);
    cfg.input_h = (uint16_t)height;
    cfg.input_w = (uint16_t)width;
    cfg.input_c = 32u;
    cfg.output_w = (uint16_t)width;
    cfg.stride_h = 1u;
    cfg.stride_w = 1u;
    cfg.pad_h = 2u;
    cfg.pad_w = 2u;
    cfg.row_stride_bytes = row_stride;
    cfg.pixel_stride_bytes = 32u;
    cfg.ow_step_bytes = 32u;
    cfg.oh_step_bytes = row_stride;
    cfg.kernel_h = 5u;
    cfg.kernel_w = 5u;
    cfg.c_base = 0u;
    cfg.lane_base = 0u;
    cfg.coalesce = 0u;
    cfg.kgen = 0u;
    cfg.pool = 1u;
    cfg.c32_fast = 1u;
    cfg.block_valid_bytes = 32u;
    cfg.k_seed_kh = 0u;
    cfg.k_seed_kw = 0u;
    cfg.k_seed_ic = 0u;
    cfg.k_tiles = 0u;
    cfg.spatial_m = height * width;
    cfg.channel_addr_offset = 0u;
    cfg.coalesce_k_bytes = cfg.kernel_h * cfg.kernel_w * cfg.block_valid_bytes;

    systolic_requant_disable();
    systolic_linebuf_config(&cfg);
    systolic_gemm32_tile_ex(0u, 0u, 0u, output_addr, cfg.spatial_m,
                            0u, row_stride, width, 0u);
    systolic_linebuf_disable();
}

void systolic_requant_disable(void) {
    REG_WRITE(REG_RQ_CTRL, 0u);
}

void systolic_requant_config_per_channel(const int32_t *bias,
                                         const int32_t *multiplier,
                                         const uint8_t *shift,
                                         const int32_t *zero_point,
                                         int32_t clamp_min,
                                         int32_t clamp_max) {
    REG_WRITE(REG_RQ_CTRL, 0u);
    REG_WRITE(REG_RQ_CMIN, (uint32_t)clamp_min);
    REG_WRITE(REG_RQ_CMAX, (uint32_t)clamp_max);

    for (uint32_t ch = 0; ch < SYSTOLIC_GEMM32_N; ch++) {
        REG_WRITE(REG_RQ_BIAS(ch), (uint32_t)bias[ch]);
        REG_WRITE(REG_RQ_MULT(ch), (uint32_t)multiplier[ch]);
        REG_WRITE(REG_RQ_SHIFT(ch), (uint32_t)shift[ch]);
        REG_WRITE(REG_RQ_ZP(ch), (uint32_t)zero_point[ch]);
    }

    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
}

void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;
    systolic_requant_disable();
    systolic_linebuf_disable();

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr,
                             ifm_addr + row * 32u,
                             ofm_addr + row * 32u * 4u,
                             tile_m);
        row += tile_m;
    }
}

void systolic_gemm32_linebuf(uint32_t weight_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;
    systolic_requant_disable();

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr, 0u, ofm_addr + row * 32u * 4u, tile_m);
        row += tile_m;
    }

    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_requant(uint32_t weight_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr, 0u, ofm_addr + row * 32u, tile_m);
        row += tile_m;
    }

    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles(uint32_t weight_addr,
                                    uint32_t psum_addr,
                                    uint32_t ofm_addr,
                                    uint32_t dim_m) {
    systolic_requant_disable();
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 0u, 0u, 0u, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_strided(uint32_t weight_addr,
                                            uint32_t psum_addr,
                                            uint32_t ofm_addr,
                                            uint32_t dim_m,
                                            uint32_t ofm_row_stride_bytes,
                                            uint32_t ofm_tile_cols) {
    systolic_requant_disable();
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 0u,
                            ofm_row_stride_bytes, ofm_tile_cols, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_requant(uint32_t weight_addr,
                                            uint32_t psum_addr,
                                            uint32_t ofm_addr,
                                            uint32_t dim_m) {
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 0u, 0u, 0u, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_requant_strided(uint32_t weight_addr,
                                                    uint32_t psum_addr,
                                                    uint32_t ofm_addr,
                                                    uint32_t dim_m,
                                                    uint32_t ofm_row_stride_bytes,
                                                    uint32_t ofm_tile_cols) {
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 0u,
                            ofm_row_stride_bytes, ofm_tile_cols, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_accumulate(uint32_t weight_addr,
                                               uint32_t psum_addr,
                                               uint32_t ofm_addr,
                                               uint32_t dim_m) {
    systolic_requant_disable();
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 1u, 0u, 0u, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_accumulate_strided(uint32_t weight_addr,
                                                       uint32_t psum_addr,
                                                       uint32_t ofm_addr,
                                                       uint32_t dim_m,
                                                       uint32_t ofm_row_stride_bytes,
                                                       uint32_t ofm_tile_cols,
                                                       uint32_t psum_row_stride_bytes) {
    systolic_requant_disable();
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 1u,
                            ofm_row_stride_bytes, ofm_tile_cols, psum_row_stride_bytes);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_ktiles_accumulate_requant_strided(uint32_t weight_addr,
                                                               uint32_t psum_addr,
                                                               uint32_t ofm_addr,
                                                               uint32_t dim_m,
                                                               uint32_t ofm_row_stride_bytes,
                                                               uint32_t ofm_tile_cols,
                                                               uint32_t psum_row_stride_bytes) {
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 1u,
                            ofm_row_stride_bytes, ofm_tile_cols, psum_row_stride_bytes);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_accumulate_requant(uint32_t weight_addr,
                                                uint32_t psum_addr,
                                                uint32_t ofm_addr,
                                                uint32_t dim_m) {
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 1u, 0u, 0u, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_linebuf_accumulate(uint32_t weight_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m) {
    systolic_requant_disable();
    systolic_gemm32_tile_ex(weight_addr, 0u, psum_addr, ofm_addr, dim_m, 1u, 0u, 0u, 0u);
    systolic_linebuf_disable();
}

void systolic_gemm32_accumulate(uint32_t weight_addr,
                                uint32_t ifm_addr,
                                uint32_t psum_addr,
                                uint32_t ofm_addr,
                                uint32_t dim_m) {
    uint32_t row = 0;
    systolic_requant_disable();
    systolic_linebuf_disable();

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_ACCUM_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_ACCUM_TILE_M;
        }

        systolic_gemm32_tile_ex(weight_addr,
                                ifm_addr + row * 32u,
                                psum_addr + row * 32u * 4u,
                                ofm_addr + row * 32u * 4u,
                                tile_m,
                                1u,
                                0u,
                                0u,
                                0u);
        row += tile_m;
    }
}

void systolic_gemm32_accumulate_requant(uint32_t weight_addr,
                                        uint32_t ifm_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m) {
    uint32_t row = 0;
    systolic_linebuf_disable();
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_ACCUM_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_ACCUM_TILE_M;
        }

        systolic_gemm32_tile_ex(weight_addr,
                                ifm_addr + row * 32u,
                                psum_addr + row * 32u * 4u,
                                ofm_addr + row * 32u,
                                tile_m,
                                1u,
                                0u,
                                0u,
                                0u);
        row += tile_m;
    }
}

void systolic_gemm32_requant(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;
    systolic_linebuf_disable();
    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr,
                             ifm_addr + row * 32u,
                             ofm_addr + row * 32u,
                             tile_m);
        row += tile_m;
    }
}
