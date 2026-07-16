#!/usr/bin/env python3
"""Generate host-planned Conv2D linebuffer job descriptors for Micro-YOLO."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


BEAT_BYTES = 32
OC_TILE = 32
K_TILE = 32
SCRATCH_BASE = 0x10100000
L2_OUTPUT = 0x80020000
MICRO_YOLO_DESC_L2_BASE = 0x80052000
MICRO_YOLO_DESC_MANIFEST_MAGIC = 0x4D594C42
MICRO_YOLO_DESC_MANIFEST_VERSION = 1
MICRO_YOLO_DESC_MANIFEST_ENTRIES = 5
MICRO_YOLO_DESC_MANIFEST_SIZE = 16 + (MICRO_YOLO_DESC_MANIFEST_ENTRIES * 16)
DESC_KIND_LINEBUF = 1
DESC_KIND_L2_COPY = 2
LINEBUF_DESC_SIZE = 124
L2_COPY_DESC_SIZE = 140


@dataclass(frozen=True)
class ConvPlan:
    input_addr: int
    weight_addr: int
    output_addr: int
    input_h: int
    input_w: int
    input_c: int
    output_h: int
    output_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    input_c_stride: int
    input_row_stride_bytes: int = 0
    input_c_base: int = 0

    @property
    def row_stride_bytes(self) -> int:
        return self.input_row_stride_bytes or (self.input_w * self.input_c_stride)

    @property
    def k_total(self) -> int:
        return self.kernel_h * self.kernel_w * self.input_c

    @property
    def k_tiles(self) -> int:
        return (self.k_total + K_TILE - 1) // K_TILE


@dataclass(frozen=True)
class MemoryMap:
    input_addr: int
    weight0_addr: int
    weight1_addr: int
    weight2_addr: int
    weight3_addr: int
    lut_addr: int
    act_a_addr: int
    psum_or_sig_addr: int
    act_c_addr: int
    head_tile_addr: int


@dataclass(frozen=True)
class DescriptorBlob:
    name: str
    kind: str
    l2_addr: int
    count: int
    data: bytes


def alloc_sequence(head_tile_oh: int, head_tile_ow: int) -> MemoryMap:
    input_h = 96
    input_w = 96
    output_h = 48
    output_w = 48
    input_bytes = input_h * input_w * 3
    weight0_bytes = 32 * 32
    c2f_weight_bytes = 3 * 3 * 32 * 32
    down_weight_bytes = c2f_weight_bytes
    head_weight_bytes = 2 * c2f_weight_bytes
    lut_bytes = 256
    act_bytes = output_h * output_w * 32
    psum_bytes = output_h * output_w * 32 * 4
    head_tile_bytes = head_tile_oh * head_tile_ow * 32

    addr = SCRATCH_BASE

    def take(size: int) -> int:
        nonlocal addr
        out = addr
        addr += size
        return out

    return MemoryMap(
        input_addr=take(input_bytes),
        weight0_addr=take(weight0_bytes),
        weight1_addr=take(c2f_weight_bytes),
        weight2_addr=take(down_weight_bytes),
        weight3_addr=take(head_weight_bytes),
        lut_addr=take(lut_bytes),
        act_a_addr=take(act_bytes),
        psum_or_sig_addr=take(psum_bytes),
        act_c_addr=take(act_bytes),
        head_tile_addr=take(head_tile_bytes),
    )


def valid_bytes(input_c: int, c_base: int, lane_base: int) -> int:
    lane_room = 0 if lane_base >= BEAT_BYTES else BEAT_BYTES - lane_base
    if c_base >= input_c or lane_room == 0:
        return 0
    return min(input_c - c_base, lane_room)


def linebuf_precompute(plan: ConvPlan, input_base: int, lane_base: int = 0) -> dict[str, int]:
    block_valid_bytes = valid_bytes(plan.input_c, plan.input_c_base, lane_base)
    channel_addr_offset = plan.input_c_base
    request_c32_fast = (
        plan.input_c_stride == BEAT_BYTES
        and plan.input_c >= BEAT_BYTES
        and (plan.input_c % BEAT_BYTES) == 0
        and plan.input_c_base == 0
    )
    c32_fast = (
        request_c32_fast
        and block_valid_bytes == BEAT_BYTES
        and plan.input_c_stride == BEAT_BYTES
        and lane_base == 0
        and plan.input_c_base == 0
        and (input_base & (BEAT_BYTES - 1)) == 0
        and (channel_addr_offset & (BEAT_BYTES - 1)) == 0
    )
    c32_group_stationary = c32_fast and plan.input_c > BEAT_BYTES
    # For C32 group-stationary, channel_addr_offset is intentionally the
    # byte span between consecutive C32 groups. RTL keeps the current group
    # offset in a register and advances by this span, avoiding seed*span math
    # in the linebuffer hot path.
    return {
        "c32_fast": 1 if c32_fast else 0,
        "c32_group_stationary": 1 if c32_group_stationary else 0,
        "block_valid_bytes": block_valid_bytes,
        "channel_addr_offset": (
            plan.input_h * plan.row_stride_bytes
            if c32_group_stationary else channel_addr_offset
        ),
        "coalesce_k_bytes": plan.kernel_h * plan.kernel_w * block_valid_bytes,
    }


def make_tile_plan(plan: ConvPlan, oh_base: int, ow_base: int, tile_oh: int, tile_ow: int,
                   output_elem_bytes: int) -> ConvPlan:
    row_stride_bytes = plan.row_stride_bytes
    last_oh = oh_base + tile_oh - 1
    last_ow = ow_base + tile_ow - 1
    first_y_unpadded = oh_base * plan.stride_h
    last_y_kernel = (last_oh * plan.stride_h) + plan.kernel_h - 1
    first_x_unpadded = ow_base * plan.stride_w
    last_x_kernel = (last_ow * plan.stride_w) + plan.kernel_w - 1

    first_ih = first_y_unpadded - plan.pad_h if first_y_unpadded > plan.pad_h else 0
    last_ih = last_y_kernel - plan.pad_h if last_y_kernel > plan.pad_h else 0
    first_iw = first_x_unpadded - plan.pad_w if first_x_unpadded > plan.pad_w else 0
    last_iw = last_x_kernel - plan.pad_w if last_x_kernel > plan.pad_w else 0

    first_ih = min(first_ih, plan.input_h - 1)
    last_ih = min(last_ih, plan.input_h - 1)
    first_iw = min(first_iw, plan.input_w - 1)
    last_iw = min(last_iw, plan.input_w - 1)

    input_addr = plan.input_addr + (first_ih * row_stride_bytes) + (first_iw * plan.input_c_stride)
    output_addr = plan.output_addr + (((oh_base * plan.output_w) + ow_base) * OC_TILE * output_elem_bytes)
    shifted_pad_h = plan.pad_h + first_ih
    shifted_pad_w = plan.pad_w + first_iw
    pad_h = shifted_pad_h - first_y_unpadded if shifted_pad_h > first_y_unpadded else 0
    pad_w = shifted_pad_w - first_x_unpadded if shifted_pad_w > first_x_unpadded else 0

    return ConvPlan(
        input_addr=input_addr,
        weight_addr=plan.weight_addr,
        output_addr=output_addr,
        input_h=last_ih - first_ih + 1,
        input_w=last_iw - first_iw + 1,
        input_c=plan.input_c,
        output_h=tile_oh,
        output_w=tile_ow,
        kernel_h=plan.kernel_h,
        kernel_w=plan.kernel_w,
        stride_h=plan.stride_h,
        stride_w=plan.stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        input_c_stride=plan.input_c_stride,
        input_row_stride_bytes=row_stride_bytes,
        input_c_base=plan.input_c_base,
    )


def make_linebuf(plan: ConvPlan, coalesce: int, kgen: int, k_tiles: int) -> dict[str, int]:
    spatial_m = plan.output_h * plan.output_w
    input_base = plan.input_addr + plan.input_c_base - (plan.pad_h * plan.row_stride_bytes)
    pre = linebuf_precompute(plan, input_base)
    out = {
        "input_base": input_base,
        "input_h": plan.input_h,
        "input_w": plan.input_w,
        "input_c": plan.input_c,
        "output_w": plan.output_w,
        "stride_h": plan.stride_h,
        "stride_w": plan.stride_w,
        "pad_h": plan.pad_h,
        "pad_w": plan.pad_w,
        "row_stride_bytes": plan.row_stride_bytes,
        "pixel_stride_bytes": plan.input_c_stride,
        "ow_step_bytes": plan.stride_w * plan.input_c_stride,
        "oh_step_bytes": plan.stride_h * plan.row_stride_bytes,
        "kernel_h": plan.kernel_h,
        "kernel_w": plan.kernel_w,
        "c_base": 0,
        "lane_base": 0,
        "coalesce": coalesce,
        "kgen": kgen,
        "pool": 0,
        "c32_fast": pre["c32_fast"],
        "depthwise": 0,
        "c32_group_stationary": pre["c32_group_stationary"],
        "block_valid_bytes": pre["block_valid_bytes"],
        "k_seed_kh": 0,
        "k_seed_kw": 0,
        "k_seed_ic": 0,
        "k_tiles": k_tiles,
        "spatial_m": spatial_m,
        "channel_addr_offset": pre["channel_addr_offset"],
        "coalesce_k_bytes": pre["coalesce_k_bytes"],
    }
    return out


def make_job(plan: ConvPlan, psum_addr: int, accum_en: int, ofm_row_stride_bytes: int,
             ofm_tile_cols: int, psum_row_stride_bytes: int, coalesce: int, kgen: int,
             linebuf_k_tiles: int, stat_k_tiles: int | None = None) -> dict[str, object]:
    spatial_m = plan.output_h * plan.output_w
    return {
        "linebuf": make_linebuf(plan, coalesce, kgen, linebuf_k_tiles),
        "gemm": {
            "weight_addr": plan.weight_addr,
            "ifm_addr": 0,
            "psum_addr": psum_addr,
            "ofm_addr": plan.output_addr,
            "dim_m": spatial_m,
            "accum_en": accum_en,
            "ofm_row_stride_bytes": ofm_row_stride_bytes,
            "ofm_tile_cols": ofm_tile_cols,
            "psum_row_stride_bytes": psum_row_stride_bytes,
        },
        "rows": spatial_m,
        "k_tiles": stat_k_tiles if stat_k_tiles is not None else plan.k_tiles,
    }


def iter_tiles(output_h: int, output_w: int, tile_oh: int, tile_ow: int):
    for oh_base in range(0, output_h, tile_oh):
        this_tile_oh = min(tile_oh, output_h - oh_base)
        for ow_base in range(0, output_w, tile_ow):
            this_tile_ow = min(tile_ow, output_w - ow_base)
            yield oh_base, ow_base, this_tile_oh, this_tile_ow


def micro_yolo_jobs(c2f_tile_oh: int, c2f_tile_ow: int,
                    down_tile_oh: int, down_tile_ow: int,
                    head_tile_oh: int, head_tile_ow: int) -> dict[str, object]:
    mem = alloc_sequence(head_tile_oh, head_tile_ow)
    input_h = 96
    input_w = 96
    output_h = 48
    output_w = 48
    down_h = 24
    down_w = 24
    c2f_weight_bytes = 3 * 3 * 32 * 32

    stem = ConvPlan(mem.input_addr, mem.weight0_addr, mem.act_a_addr,
                    input_h, input_w, 3, output_h, output_w,
                    3, 3, 2, 2, 1, 1, 3)
    c2f = ConvPlan(mem.act_c_addr, mem.weight1_addr, mem.act_a_addr,
                   output_h, output_w, 32, output_h, output_w,
                   3, 3, 1, 1, 1, 1, 32)
    down = ConvPlan(mem.act_a_addr, mem.weight2_addr, mem.act_c_addr,
                    output_h, output_w, 32, down_h, down_w,
                    3, 3, 2, 2, 1, 1, 32)
    head0 = ConvPlan(mem.act_c_addr, mem.weight3_addr, mem.psum_or_sig_addr,
                     output_h, output_w, 32, output_h, output_w,
                     3, 3, 1, 1, 1, 1, 32)
    head1 = ConvPlan(mem.act_a_addr, mem.weight3_addr + c2f_weight_bytes, mem.head_tile_addr,
                     output_h, output_w, 32, output_h, output_w,
                     3, 3, 1, 1, 1, 1, 32)

    stem_jobs = []
    stem_tile_oh = max(1, 1024 // output_w)
    for oh_base, ow_base, tile_oh, tile_ow in iter_tiles(output_h, output_w, stem_tile_oh, output_w):
        tile_plan = make_tile_plan(stem, oh_base, ow_base, tile_oh, tile_ow, 1)
        stem_jobs.append(make_job(tile_plan, 0, 0, 0, 0, 0, 1, 0, 0, 1))

    c2f_jobs = []
    for oh_base, ow_base, tile_oh, tile_ow in iter_tiles(output_h, output_w, c2f_tile_oh, c2f_tile_ow):
        tile_plan = make_tile_plan(c2f, oh_base, ow_base, tile_oh, tile_ow, 1)
        row_stride = output_w * OC_TILE if tile_ow != output_w else 0
        tile_cols = tile_ow if tile_ow != output_w else 0
        tile_psum = mem.psum_or_sig_addr + (((oh_base * output_w) + ow_base) * OC_TILE * 4)
        c2f_jobs.append(make_job(tile_plan, tile_psum, 0, row_stride, tile_cols, 0, 1, 1, tile_plan.k_tiles))

    down_jobs = []
    for oh_base, ow_base, tile_oh, tile_ow in iter_tiles(down_h, down_w, down_tile_oh, down_tile_ow):
        tile_plan = make_tile_plan(down, oh_base, ow_base, tile_oh, tile_ow, 1)
        row_stride = down_w * OC_TILE if tile_ow != down_w else 0
        tile_cols = tile_ow if tile_ow != down_w else 0
        tile_psum = mem.psum_or_sig_addr + (((oh_base * down_w) + ow_base) * OC_TILE * 4)
        down_jobs.append(make_job(tile_plan, tile_psum, 0, row_stride, tile_cols, 0, 1, 1, tile_plan.k_tiles))

    head0_jobs = []
    for oh_base, ow_base, tile_oh, tile_ow in iter_tiles(output_h, output_w, head_tile_oh, head_tile_ow):
        tile_plan = make_tile_plan(head0, oh_base, ow_base, tile_oh, tile_ow, 4)
        row_stride = output_w * OC_TILE * 4 if tile_ow != output_w else 0
        tile_cols = tile_ow if tile_ow != output_w else 0
        head0_jobs.append(make_job(tile_plan, tile_plan.output_addr, 0, row_stride, tile_cols,
                                   row_stride, 1, 1, tile_plan.k_tiles))

    head1_l2_jobs = []
    for oh_base, ow_base, tile_oh, tile_ow in iter_tiles(output_h, output_w, head_tile_oh, head_tile_ow):
        tile_plan = make_tile_plan(head1, oh_base, ow_base, tile_oh, tile_ow, 1)
        tile_plan = ConvPlan(tile_plan.input_addr, tile_plan.weight_addr, mem.head_tile_addr,
                             tile_plan.input_h, tile_plan.input_w, tile_plan.input_c,
                             tile_plan.output_h, tile_plan.output_w, tile_plan.kernel_h,
                             tile_plan.kernel_w, tile_plan.stride_h, tile_plan.stride_w,
                             tile_plan.pad_h, tile_plan.pad_w, tile_plan.input_c_stride,
                             tile_plan.input_row_stride_bytes, tile_plan.input_c_base)
        tile_psum = mem.psum_or_sig_addr + (((oh_base * output_w) + ow_base) * OC_TILE * 4)
        compact_stride = tile_ow * OC_TILE
        full_psum_stride = output_w * OC_TILE * 4
        copy_l2 = L2_OUTPUT + (((oh_base * output_w) + ow_base) * OC_TILE)
        head1_l2_jobs.append({
            "job": make_job(tile_plan, tile_psum, 1, compact_stride, tile_ow,
                            full_psum_stride, 1, 1, tile_plan.k_tiles),
            "l2_addr": copy_l2,
            "tile_output_addr": mem.head_tile_addr,
            "tile_oh": tile_oh,
            "tile_ow": tile_ow,
        })

    return {
        "STEM": stem_jobs,
        "C2F": c2f_jobs,
        "DOWN": down_jobs,
        "HEAD0": head0_jobs,
        "HEAD1_L2": head1_l2_jobs,
    }


LINEBUF_FIELDS = [
    "input_base", "input_h", "input_w", "input_c", "output_w",
    "stride_h", "stride_w", "pad_h", "pad_w", "row_stride_bytes",
    "pixel_stride_bytes", "ow_step_bytes", "oh_step_bytes", "kernel_h",
    "kernel_w", "c_base", "lane_base", "coalesce", "kgen", "pool",
    "c32_fast", "depthwise", "c32_group_stationary", "block_valid_bytes",
    "k_seed_kh", "k_seed_kw", "k_seed_ic", "k_tiles", "spatial_m",
    "channel_addr_offset", "coalesce_k_bytes",
]
GEMM_FIELDS = [
    "weight_addr", "ifm_addr", "psum_addr", "ofm_addr", "dim_m",
    "accum_en", "ofm_row_stride_bytes", "ofm_tile_cols", "psum_row_stride_bytes",
]


def c_u32(value: int) -> str:
    return f"0x{value:08X}u" if value >= 0x10000 else f"{value}u"


def pack_linebuf_cfg(fields: dict[str, int]) -> bytes:
    return struct.pack(
        "<I8H4I14H4I",
        fields["input_base"],
        fields["input_h"],
        fields["input_w"],
        fields["input_c"],
        fields["output_w"],
        fields["stride_h"],
        fields["stride_w"],
        fields["pad_h"],
        fields["pad_w"],
        fields["row_stride_bytes"],
        fields["pixel_stride_bytes"],
        fields["ow_step_bytes"],
        fields["oh_step_bytes"],
        fields["kernel_h"],
        fields["kernel_w"],
        fields["c_base"],
        fields["lane_base"],
        fields["coalesce"],
        fields["kgen"],
        fields["pool"],
        fields["c32_fast"],
        fields["depthwise"],
        fields["c32_group_stationary"],
        fields["block_valid_bytes"],
        fields["k_seed_kh"],
        fields["k_seed_kw"],
        fields["k_seed_ic"],
        fields["k_tiles"],
        fields["spatial_m"],
        fields["channel_addr_offset"],
        fields["coalesce_k_bytes"],
    )


def pack_gemm_req(fields: dict[str, int]) -> bytes:
    return struct.pack("<9I", *(fields[field] for field in GEMM_FIELDS))


def pack_linebuf_job(job: dict[str, object]) -> bytes:
    data = (
        pack_linebuf_cfg(job["linebuf"])
        + pack_gemm_req(job["gemm"])
        + struct.pack("<2I", job["rows"], job["k_tiles"])
    )
    if len(data) != LINEBUF_DESC_SIZE:
        raise ValueError(f"linebuffer descriptor ABI size mismatch: {len(data)}")
    return data


def pack_l2_job(entry: dict[str, object]) -> bytes:
    data = (
        pack_linebuf_job(entry["job"])
        + struct.pack("<4I", entry["l2_addr"], entry["tile_output_addr"],
                      entry["tile_oh"], entry["tile_ow"])
    )
    if len(data) != L2_COPY_DESC_SIZE:
        raise ValueError(f"L2 copy descriptor ABI size mismatch: {len(data)}")
    return data


def align_up(value: int, align: int) -> int:
    return (value + align - 1) & ~(align - 1)


def micro_yolo_descriptor_blobs(c2f_tile_oh: int = 16, c2f_tile_ow: int = 16,
                                down_tile_oh: int = 16, down_tile_ow: int = 16,
                                head_tile_oh: int = 16, head_tile_ow: int = 16
                                ) -> list[DescriptorBlob]:
    jobs = micro_yolo_jobs(c2f_tile_oh, c2f_tile_ow,
                           down_tile_oh, down_tile_ow,
                           head_tile_oh, head_tile_ow)
    specs = [
        ("STEM", "linebuf", b"".join(pack_linebuf_job(job) for job in jobs["STEM"])),
        ("C2F", "linebuf", b"".join(pack_linebuf_job(job) for job in jobs["C2F"])),
        ("DOWN", "linebuf", b"".join(pack_linebuf_job(job) for job in jobs["DOWN"])),
        ("HEAD0", "linebuf", b"".join(pack_linebuf_job(job) for job in jobs["HEAD0"])),
        ("HEAD1_L2", "l2_copy", b"".join(pack_l2_job(job) for job in jobs["HEAD1_L2"])),
    ]

    blobs: list[DescriptorBlob] = []
    addr = align_up(MICRO_YOLO_DESC_L2_BASE + MICRO_YOLO_DESC_MANIFEST_SIZE, BEAT_BYTES)
    for name, kind, data in specs:
        desc_size = L2_COPY_DESC_SIZE if kind == "l2_copy" else LINEBUF_DESC_SIZE
        blobs.append(DescriptorBlob(name=name, kind=kind, l2_addr=addr,
                                    count=len(data) // desc_size, data=data))
        addr = align_up(addr + len(data), BEAT_BYTES)
    return blobs


def micro_yolo_descriptor_manifest(blobs: list[DescriptorBlob]) -> bytes:
    if len(blobs) != MICRO_YOLO_DESC_MANIFEST_ENTRIES:
        raise ValueError(f"expected {MICRO_YOLO_DESC_MANIFEST_ENTRIES} descriptor blobs")
    data = struct.pack(
        "<4I",
        MICRO_YOLO_DESC_MANIFEST_MAGIC,
        MICRO_YOLO_DESC_MANIFEST_VERSION,
        len(blobs),
        0,
    )
    for blob in blobs:
        kind = DESC_KIND_L2_COPY if blob.kind == "l2_copy" else DESC_KIND_LINEBUF
        data += struct.pack("<4I", blob.l2_addr, blob.count, len(blob.data), kind)
    if len(data) != MICRO_YOLO_DESC_MANIFEST_SIZE:
        raise ValueError(f"descriptor manifest ABI size mismatch: {len(data)}")
    return data


def emit_linebuf(job: dict[str, object], indent: str) -> list[str]:
    linebuf = job["linebuf"]
    gemm = job["gemm"]
    lines = [indent + "{"]
    lines.append(indent + "    .linebuf = {")
    for field in LINEBUF_FIELDS:
        lines.append(indent + f"        .{field} = {c_u32(linebuf[field])},")
    lines.append(indent + "    },")
    lines.append(indent + "    .gemm = {")
    for field in GEMM_FIELDS:
        lines.append(indent + f"        .{field} = {c_u32(gemm[field])},")
    lines.append(indent + "    },")
    lines.append(indent + f"    .rows = {c_u32(job['rows'])},")
    lines.append(indent + f"    .k_tiles = {c_u32(job['k_tiles'])},")
    lines.append(indent + "},")
    return lines


def emit_job_array(name: str, jobs: list[dict[str, object]]) -> list[str]:
    symbol = f"MICRO_YOLO_LB_{name}_JOBS"
    lines = [f"static const npu_conv2d_linebuf_job_desc_t {symbol}[] "
             "__attribute__((section(\".data.linebuf_desc\"), used)) = {"]
    for job in jobs:
        lines.extend(emit_linebuf(job, "    "))
    lines.append("};")
    lines.append(f"#define {symbol}_COUNT ((uint32_t)(sizeof({symbol}) / sizeof({symbol}[0])))")
    lines.append("")
    return lines


def emit_l2_job_array(name: str, jobs: list[dict[str, object]]) -> list[str]:
    symbol = f"MICRO_YOLO_LB_{name}_JOBS"
    lines = [f"static const npu_conv2d_l2_copy_job_desc_t {symbol}[] "
             "__attribute__((section(\".data.linebuf_desc\"), used)) = {"]
    for entry in jobs:
        lines.append("    {")
        job_lines = emit_linebuf(entry["job"], "        ")
        job_lines[0] = "        .job = {"
        job_lines[-1] = "        },"
        lines.extend(job_lines)
        lines.append(f"        .l2_addr = {c_u32(entry['l2_addr'])},")
        lines.append(f"        .tile_output_addr = {c_u32(entry['tile_output_addr'])},")
        lines.append(f"        .tile_oh = {c_u32(entry['tile_oh'])},")
        lines.append(f"        .tile_ow = {c_u32(entry['tile_ow'])},")
        lines.append("    },")
    lines.append("};")
    lines.append(f"#define {symbol}_COUNT ((uint32_t)(sizeof({symbol}) / sizeof({symbol}[0])))")
    lines.append("")
    return lines


def emit_micro_yolo_header(output: Path, args: argparse.Namespace) -> None:
    jobs = micro_yolo_jobs(args.c2f_tile_oh, args.c2f_tile_ow,
                           args.down_tile_oh, args.down_tile_ow,
                           args.head_tile_oh, args.head_tile_ow)
    lines = [
        "#ifndef MICRO_YOLO_LINEBUF_PRECOMPUTE_H",
        "#define MICRO_YOLO_LINEBUF_PRECOMPUTE_H",
        "",
        "#include \"conv2d_packed.h\"",
        "",
        "/* Generated by tools/npu_linebuf_precompute.py. */",
        "",
    ]
    lines.extend(emit_job_array("STEM", jobs["STEM"]))
    lines.extend(emit_job_array("C2F", jobs["C2F"]))
    lines.extend(emit_job_array("DOWN", jobs["DOWN"]))
    lines.extend(emit_job_array("HEAD0", jobs["HEAD0"]))
    lines.extend(emit_l2_job_array("HEAD1_L2", jobs["HEAD1_L2"]))
    lines.append("#endif")
    lines.append("")
    output.write_text("\n".join(lines), encoding="ascii")


def emit_micro_yolo_blob_dir(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    blobs = micro_yolo_descriptor_blobs(args.c2f_tile_oh, args.c2f_tile_ow,
                                        args.down_tile_oh, args.down_tile_ow,
                                        args.head_tile_oh, args.head_tile_ow)
    (output_dir / "micro_yolo_linebuf_manifest.bin").write_bytes(
        micro_yolo_descriptor_manifest(blobs)
    )
    manifest = [
        f"# binary_manifest_l2_addr 0x{MICRO_YOLO_DESC_L2_BASE:08X}",
        "# name kind l2_addr count bytes filename",
    ]
    for blob in blobs:
        filename = f"micro_yolo_lb_{blob.name.lower()}.bin"
        (output_dir / filename).write_bytes(blob.data)
        manifest.append(
            f"{blob.name} {blob.kind} 0x{blob.l2_addr:08X} "
            f"{blob.count} {len(blob.data)} {filename}"
        )
    (output_dir / "micro_yolo_linebuf_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="ascii"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("micro-yolo-header")
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--c2f-tile-oh", type=int, default=16)
    gen.add_argument("--c2f-tile-ow", type=int, default=16)
    gen.add_argument("--down-tile-oh", type=int, default=16)
    gen.add_argument("--down-tile-ow", type=int, default=16)
    gen.add_argument("--head-tile-oh", type=int, default=16)
    gen.add_argument("--head-tile-ow", type=int, default=16)
    blob = sub.add_parser("micro-yolo-blob")
    blob.add_argument("--output-dir", required=True, type=Path)
    blob.add_argument("--c2f-tile-oh", type=int, default=16)
    blob.add_argument("--c2f-tile-ow", type=int, default=16)
    blob.add_argument("--down-tile-oh", type=int, default=16)
    blob.add_argument("--down-tile-ow", type=int, default=16)
    blob.add_argument("--head-tile-oh", type=int, default=16)
    blob.add_argument("--head-tile-ow", type=int, default=16)
    args = parser.parse_args()

    if args.cmd == "micro-yolo-header":
        emit_micro_yolo_header(args.output, args)
        return 0
    if args.cmd == "micro-yolo-blob":
        emit_micro_yolo_blob_dir(args.output_dir, args)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
