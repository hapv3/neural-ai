from dataclasses import dataclass


YOLO_CHANNELS = [3, 16, 32, 64, 96, 128, 192, 256]
SQUARE_KERNELS = [(kernel, kernel) for kernel in [1, 2, 3, 4, 5]]
ALL_KERNELS = [(kernel_h, kernel_w) for kernel_h in range(1, 6) for kernel_w in range(1, 6)]


@dataclass(frozen=True)
class DirectCase:
    case_id: str
    dim_m: int
    mode: str = "direct"


@dataclass(frozen=True)
class LinebufCase:
    case_id: str
    mode: str
    kernel_h: int
    kernel_w: int
    input_h: int
    input_w: int
    input_c: int
    output_h: int
    output_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    c_base: int = 0
    logical_channels: int = 0
    coalesce: bool = False
    kgen: bool = False
    c32_fast: bool = False
    pool: bool = False
    block_valid_bytes: int = 0
    channel_offset: int = 0
    coalesce_k_bytes: int = 0

    @property
    def spatial_m(self):
        return self.output_h * self.output_w

    @property
    def k_tiles(self):
        if self.kgen:
            return self.kernel_h * self.kernel_w
        return 1

    @property
    def expected_channels(self):
        if self.logical_channels:
            return min(32, self.logical_channels)
        return max(0, min(32, self.input_c - self.c_base))


def conv_output_dim(input_size, kernel, stride, pad):
    return ((input_size + 2 * pad - kernel) // stride) + 1


def shape_variants(kernel_h, kernel_w=None):
    if kernel_w is None:
        kernel_w = kernel_h
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    return [
        ("tiny_square", 3, 3, 1, 1, pad_h, pad_w),
        ("rect_wide", 3, 7, 1, 1, pad_h, pad_w),
        ("rect_tall", 7, 3, 1, 1, pad_h, pad_w),
        ("mid_rect", 5, 6, 1, 1, pad_h, pad_w),
        ("stride2_square", 7, 7, 2, 2, pad_h, pad_w),
        ("stride2_rect", 6, 9, 2, 2, pad_h, pad_w),
        ("valid_rect", max(kernel_h + 1, 5), max(kernel_w + 2, 6), 1, 1, 0, 0),
    ]


def kernel_name(kernel_h, kernel_w):
    return f"{kernel_h}x{kernel_w}"


def coalesce_channels_for(kernel_h, kernel_w):
    area = kernel_h * kernel_w
    if area <= 1:
        return [1, 3, 16, 32]
    if area <= 4:
        return [1, 3, 8]
    if area <= 9:
        return [1, 3]
    if area <= 16:
        return [1, 2]
    return [1]


def build_direct_cases():
    return [DirectCase(case_id=f"direct_m_{m:03d}", dim_m=m) for m in range(1, 129)]


def build_linebuf_cases():
    cases = []

    # Bypass and channel-boundary cases for pointwise mode.
    pointwise_channels = [
        ("c3_rgb", 3, 0, 3),
        ("c16_tail", 16, 0, 16),
        ("c32_full", 32, 0, 32),
        ("c33_cross", 33, 1, 32),
        ("c33_tail", 33, 32, 1),
        ("c65_cross", 65, 33, 32),
        ("c65_tail", 65, 64, 1),
    ]
    for shape_name, ih, iw, sh, sw, ph, pw in shape_variants(1, 1):
        oh = conv_output_dim(ih, 1, sh, ph)
        ow = conv_output_dim(iw, 1, sw, pw)
        for chan_name, input_c, c_base, logical in pointwise_channels:
            cases.append(LinebufCase(
                case_id=f"lb_bypass_1x1_{shape_name}_{chan_name}",
                mode="bypass",
                kernel_h=1,
                kernel_w=1,
                input_h=ih,
                input_w=iw,
                input_c=input_c,
                output_h=oh,
                output_w=ow,
                stride_h=sh,
                stride_w=sw,
                pad_h=ph,
                pad_w=pw,
                c_base=c_base,
                logical_channels=logical,
            ))

    # Coalesced small-K cases where KH*KW*C fits in one 32-lane vector.
    for kernel_h, kernel_w in ALL_KERNELS:
        for shape_name, ih, iw, sh, sw, ph, pw in shape_variants(kernel_h, kernel_w):
            oh = conv_output_dim(ih, kernel_h, sh, ph)
            ow = conv_output_dim(iw, kernel_w, sw, pw)
            if oh <= 0 or ow <= 0:
                continue
            for input_c in coalesce_channels_for(kernel_h, kernel_w):
                cases.append(LinebufCase(
                    case_id=f"lb_coal_{kernel_name(kernel_h, kernel_w)}_{shape_name}_c{input_c}",
                    mode="coalesce",
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                    input_h=ih,
                    input_w=iw,
                    input_c=input_c,
                    output_h=oh,
                    output_w=ow,
                    stride_h=sh,
                    stride_w=sw,
                    pad_h=ph,
                    pad_w=pw,
                    logical_channels=input_c,
                    coalesce=True,
                ))

    # C32 KGEN/C32-fast cases model the standard YOLO internal C32 view.
    for kernel_h, kernel_w in ALL_KERNELS:
        for shape_name, ih, iw, sh, sw, ph, pw in shape_variants(kernel_h, kernel_w):
            oh = conv_output_dim(ih, kernel_h, sh, ph)
            ow = conv_output_dim(iw, kernel_w, sw, pw)
            if oh <= 0 or ow <= 0:
                continue
            for logical_channels in [32, 64, 96, 128, 192, 256]:
                cases.append(LinebufCase(
                    case_id=f"lb_kgen_c32fast_{kernel_name(kernel_h, kernel_w)}_{shape_name}_yoloC{logical_channels}",
                    mode="kgen_c32fast",
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                    input_h=ih,
                    input_w=iw,
                    input_c=32,
                    output_h=oh,
                    output_w=ow,
                    stride_h=sh,
                    stride_w=sw,
                    pad_h=ph,
                    pad_w=pw,
                    logical_channels=logical_channels,
                    coalesce=True,
                    kgen=True,
                    c32_fast=True,
                    block_valid_bytes=32,
                    channel_offset=0,
                    coalesce_k_bytes=kernel_h * kernel_w * 32,
                ))

    # Pool path is linebuffer-backed but bypasses systolic MACs.
    for kernel_h, kernel_w in ALL_KERNELS:
        for shape_name, ih, iw, sh, sw, ph, pw in shape_variants(kernel_h, kernel_w):
            oh = conv_output_dim(ih, kernel_h, sh, ph)
            ow = conv_output_dim(iw, kernel_w, sw, pw)
            if oh <= 0 or ow <= 0:
                continue
            for input_c in [16, 32]:
                cases.append(LinebufCase(
                    case_id=f"lb_pool_{kernel_name(kernel_h, kernel_w)}_{shape_name}_c{input_c}",
                    mode="pool",
                    kernel_h=kernel_h,
                    kernel_w=kernel_w,
                    input_h=ih,
                    input_w=iw,
                    input_c=input_c,
                    output_h=oh,
                    output_w=ow,
                    stride_h=sh,
                    stride_w=sw,
                    pad_h=ph,
                    pad_w=pw,
                    logical_channels=input_c,
                    pool=True,
                ))

    return cases


DIRECT_CASES = build_direct_cases()
LINEBUF_CASES = build_linebuf_cases()
ALL_CASES = {case.case_id: case for case in [*DIRECT_CASES, *LINEBUF_CASES]}


def get_case(case_id):
    return ALL_CASES[case_id]


def default_smoke_case_ids():
    return [
        "direct_m_001",
        "direct_m_017",
        "direct_m_064",
        "direct_m_128",
        "lb_bypass_1x1_rect_wide_c33_cross",
        "lb_coal_2x2_rect_wide_c8",
        "lb_coal_1x5_rect_wide_c3",
        "lb_coal_5x1_rect_tall_c3",
        "lb_coal_2x4_mid_rect_c3",
        "lb_coal_4x2_mid_rect_c3",
        "lb_coal_4x4_mid_rect_c2",
        "lb_kgen_c32fast_3x3_stride2_square_yoloC96",
        "lb_kgen_c32fast_1x5_rect_wide_yoloC96",
        "lb_kgen_c32fast_5x1_rect_tall_yoloC96",
        "lb_kgen_c32fast_2x4_mid_rect_yoloC96",
        "lb_kgen_c32fast_4x2_mid_rect_yoloC96",
        "lb_kgen_c32fast_5x5_rect_tall_yoloC256",
        "lb_pool_4x4_stride2_rect_c16",
        "lb_pool_1x5_rect_wide_c16",
        "lb_pool_5x1_rect_tall_c16",
        "lb_pool_2x4_mid_rect_c16",
        "lb_pool_4x2_mid_rect_c16",
    ]
