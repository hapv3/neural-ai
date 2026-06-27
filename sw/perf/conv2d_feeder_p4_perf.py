#!/usr/bin/env python3
"""Software performance estimator for Conv2D feeder P4 options.

The model is intentionally architectural, not cycle-exact RTL simulation.  It
answers the P4 design question: does the current byte-serial feeder dominate
runtime enough to justify a packed-buffer or wide-read path?
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from math import ceil
from typing import Iterable, TextIO


K_TILE = 32
BEAT_BYTES = 32
DEFAULT_CACHE_ENTRIES = 128
DEFAULT_CACHE_BANKS = 4


@dataclass(frozen=True)
class ConvShape:
    name: str
    input_h: int
    input_w: int
    input_c: int
    kernel_h: int
    kernel_w: int
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    oc: int = 32

    @property
    def output_h(self) -> int:
        effective_kh = (self.kernel_h - 1) * self.dilation_h + 1
        return ((self.input_h + (2 * self.pad_h) - effective_kh) // self.stride_h) + 1

    @property
    def output_w(self) -> int:
        effective_kw = (self.kernel_w - 1) * self.dilation_w + 1
        return ((self.input_w + (2 * self.pad_w) - effective_kw) // self.stride_w) + 1

    @property
    def rows(self) -> int:
        return self.output_h * self.output_w

    @property
    def k_total(self) -> int:
        return self.kernel_h * self.kernel_w * self.input_c

    @property
    def k_tiles(self) -> int:
        return ceil(self.k_total / K_TILE)

    @property
    def oc_tiles(self) -> int:
        return ceil(self.oc / 32)

    @property
    def has_padding(self) -> bool:
        return self.pad_h != 0 or self.pad_w != 0

    @property
    def has_dilation(self) -> bool:
        return self.dilation_h != 1 or self.dilation_w != 1


@dataclass
class CurrentFeederStats:
    cycles: int = 0
    valid_lanes: int = 0
    zero_lanes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tcdm_reads: int = 0
    rows: int = 0


@dataclass
class WideReadStats:
    cycles: int = 0
    rows: int = 0
    one_segment_rows: int = 0
    two_segment_rows: int = 0
    slow_rows: int = 0
    zero_only_rows: int = 0
    unaligned_rows: int = 0
    tcdm_reads: int = 0


@dataclass(frozen=True)
class PerfModel:
    hit_cycles: int = 2
    miss_extra_cycles: int = 2
    zero_lane_cycles: int = 1
    write_row_cycles: int = 1
    precompute_base_cycles: int = 14
    cache_entries: int = DEFAULT_CACHE_ENTRIES
    cache_banks: int = DEFAULT_CACHE_BANKS
    weight_cycles: int = 32
    compute_cycles_per_row: int = 1
    drain_cycles_per_row: int = 1
    p4_segment_cycles: int = 1
    p4_unaligned_cycles: int = 2
    p4_two_segment_cycles: int = 3
    p4_slow_row_cycles: int = 32
    spatz_im2col_cycles_per_row: int = 4
    dma_prepare_cycles_per_row: int = 1

    @property
    def miss_cycles(self) -> int:
        return self.hit_cycles + self.miss_extra_cycles

    @property
    def flush_cycles(self) -> int:
        return ceil(self.cache_entries / self.cache_banks) + 2


def default_shapes() -> list[ConvShape]:
    return [
        ConvShape("rgb_3x3_c3_32x32", 32, 32, 3, 3, 3, pad_h=1, pad_w=1),
        ConvShape("conv1x1_c48_16x16", 16, 16, 48, 1, 1),
        ConvShape("conv3x3_c48_16x16", 16, 16, 48, 3, 3, pad_h=1, pad_w=1),
        ConvShape("conv1x1_c192_8x8", 8, 8, 192, 1, 1),
        ConvShape("conv3x3_c192_8x8", 8, 8, 192, 3, 3, pad_h=1, pad_w=1),
        ConvShape("conv1x1_c576_4x4", 4, 4, 576, 1, 1),
        ConvShape("conv3x3_c576_4x4", 4, 4, 576, 3, 3, pad_h=1, pad_w=1),
        ConvShape("conv3x3_c33_8x8", 8, 8, 33, 3, 3, pad_h=1, pad_w=1),
        ConvShape("conv5x5_c48_8x8", 8, 8, 48, 5, 5, pad_h=2, pad_w=2),
    ]


def parse_shape(spec: str) -> ConvShape:
    fields = spec.split(",")
    if len(fields) not in (6, 8, 10, 12):
        raise argparse.ArgumentTypeError(
            "shape must be name,H,W,IC,KH,KW[,SH,SW[,PH,PW[,DH,DW]]]"
        )

    name = fields[0]
    values = [int(value, 0) for value in fields[1:]]
    input_h, input_w, input_c, kernel_h, kernel_w = values[:5]
    stride_h = stride_w = 1
    pad_h = pad_w = 0
    dilation_h = dilation_w = 1

    if len(values) >= 7:
        stride_h, stride_w = values[5:7]
    if len(values) >= 9:
        pad_h, pad_w = values[7:9]
    if len(values) >= 11:
        dilation_h, dilation_w = values[9:11]

    shape = ConvShape(
        name,
        input_h,
        input_w,
        input_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    )
    if shape.has_dilation:
        raise argparse.ArgumentTypeError("dilation > 1 is unsupported by the current Conv2D P4 scope")
    return shape


def k_to_offsets(shape: ConvShape, k_index: int) -> tuple[int, int, int] | None:
    if k_index >= shape.k_total:
        return None

    spatial = k_index // shape.input_c
    ic = k_index - (spatial * shape.input_c)
    kh = spatial // shape.kernel_w
    kw = spatial - (kh * shape.kernel_w)
    return kh, kw, ic


def lane_addresses(shape: ConvShape, row: int, k_base: int) -> list[int | None]:
    oh = row // shape.output_w
    ow = row - (oh * shape.output_w)
    addresses: list[int | None] = []

    for lane in range(K_TILE):
        offsets = k_to_offsets(shape, k_base + lane)
        if offsets is None:
            addresses.append(None)
            continue

        kh, kw, ic = offsets
        ih = (oh * shape.stride_h) + (kh * shape.dilation_h) - shape.pad_h
        iw = (ow * shape.stride_w) + (kw * shape.dilation_w) - shape.pad_w
        if ih < 0 or iw < 0 or ih >= shape.input_h or iw >= shape.input_w:
            addresses.append(None)
            continue

        addresses.append(((ih * shape.input_w) + iw) * shape.input_c + ic)

    return addresses


def contiguous_segments(addresses: list[int | None]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None

    for address in addresses:
        if address is None:
            if start is not None and previous is not None:
                segments.append((start, previous))
            start = None
            previous = None
            continue

        if start is None:
            start = address
            previous = address
        elif previous is not None and address == previous + 1:
            previous = address
        else:
            segments.append((start, previous))
            start = address
            previous = address

    if start is not None and previous is not None:
        segments.append((start, previous))

    return segments


def cache_lookup(cache: dict[tuple[int, int], int], line: int, model: PerfModel) -> bool:
    bank = line % model.cache_banks
    index = (line // model.cache_banks) % (model.cache_entries // model.cache_banks)
    key = (bank, index)
    hit = cache.get(key) == line
    cache[key] = line
    return hit


def estimate_current_feeder(shape: ConvShape, model: PerfModel) -> CurrentFeederStats:
    stats = CurrentFeederStats(rows=shape.rows * shape.k_tiles)

    for k_tile in range(shape.k_tiles):
        k_base = k_tile * K_TILE
        cache: dict[tuple[int, int], int] = {}
        stats.cycles += model.flush_cycles + model.precompute_base_cycles + k_base

        for row in range(shape.rows):
            stats.cycles += 1
            addresses = lane_addresses(shape, row, k_base)
            for address in addresses:
                if address is None:
                    stats.zero_lanes += 1
                    stats.cycles += model.zero_lane_cycles
                    continue

                stats.valid_lanes += 1
                line = address // BEAT_BYTES
                if cache_lookup(cache, line, model):
                    stats.cache_hits += 1
                    stats.cycles += model.hit_cycles
                else:
                    stats.cache_misses += 1
                    stats.tcdm_reads += 1
                    stats.cycles += model.miss_cycles

            stats.cycles += model.write_row_cycles

    return stats


def estimate_p4_raw_wide(shape: ConvShape, model: PerfModel) -> WideReadStats:
    stats = WideReadStats(rows=shape.rows * shape.k_tiles)

    for k_tile in range(shape.k_tiles):
        k_base = k_tile * K_TILE
        for row in range(shape.rows):
            segments = contiguous_segments(lane_addresses(shape, row, k_base))
            stats.tcdm_reads += len(segments)

            if not segments:
                stats.zero_only_rows += 1
                stats.cycles += model.p4_segment_cycles
            elif len(segments) > 2:
                stats.slow_rows += 1
                stats.cycles += model.p4_slow_row_cycles
            elif len(segments) == 1:
                stats.one_segment_rows += 1
                if segments[0][0] % BEAT_BYTES == 0:
                    stats.cycles += model.p4_segment_cycles
                else:
                    stats.unaligned_rows += 1
                    stats.cycles += model.p4_unaligned_cycles
            else:
                stats.two_segment_rows += 1
                stats.cycles += model.p4_two_segment_cycles

    return stats


def estimate_packed_prepare(shape: ConvShape, model: PerfModel) -> int:
    if shape.input_c < K_TILE:
        return shape.rows * shape.k_tiles * model.spatz_im2col_cycles_per_row

    border_penalty = 1 if shape.has_padding else 0
    return shape.rows * shape.k_tiles * (model.dma_prepare_cycles_per_row + border_penalty)


def compute_cycles(shape: ConvShape, model: PerfModel, feed_cycles: int | None = None) -> int:
    row_cycles = shape.rows * model.compute_cycles_per_row
    if feed_cycles is not None:
        row_cycles = max(row_cycles, feed_cycles)
    return model.weight_cycles + row_cycles + (shape.rows * model.drain_cycles_per_row)


def overlapped_pipeline_cycles(prepare_per_tile: list[int], compute_per_tile: list[int]) -> int:
    if not prepare_per_tile:
        return 0

    total = prepare_per_tile[0]
    for index in range(len(prepare_per_tile) - 1):
        total += max(compute_per_tile[index], prepare_per_tile[index + 1])
    total += compute_per_tile[-1]
    return total


def broken_k_tiles(shape: ConvShape) -> int:
    broken = 0
    for k_tile in range(shape.k_tiles):
        k_base = k_tile * K_TILE
        k_end = min(k_base + K_TILE, shape.k_total)
        kh_values = {
            offsets[0]
            for k_index in range(k_base, k_end)
            if (offsets := k_to_offsets(shape, k_index)) is not None
        }
        if len(kh_values) > 1:
            broken += 1
    return broken


def summarize_shape(shape: ConvShape, model: PerfModel) -> dict[str, int | str | float]:
    current = estimate_current_feeder(shape, model)
    raw = estimate_p4_raw_wide(shape, model)
    prepare_total = estimate_packed_prepare(shape, model)

    raw_compute_per_tile: list[int] = []
    packed_compute_per_tile: list[int] = []
    packed_prepare_per_tile: list[int] = []

    for k_tile in range(shape.k_tiles):
        k_base = k_tile * K_TILE
        tile_feed = 0
        for row in range(shape.rows):
            segments = contiguous_segments(lane_addresses(shape, row, k_base))
            if not segments:
                tile_feed += model.p4_segment_cycles
            elif shape.has_dilation or len(segments) > 2:
                tile_feed += model.p4_slow_row_cycles
            elif len(segments) == 1:
                tile_feed += model.p4_segment_cycles if segments[0][0] % BEAT_BYTES == 0 else model.p4_unaligned_cycles
            else:
                tile_feed += model.p4_two_segment_cycles

        raw_compute_per_tile.append(compute_cycles(shape, model, tile_feed))

        if shape.input_c < K_TILE:
            packed_prepare = shape.rows * model.spatz_im2col_cycles_per_row
        else:
            packed_prepare = shape.rows * (model.dma_prepare_cycles_per_row + (1 if shape.has_padding else 0))
        packed_prepare_per_tile.append(packed_prepare)
        packed_compute_per_tile.append(compute_cycles(shape, model, shape.rows))

    p4_raw_total = sum(raw_compute_per_tile)
    p4_packed_no_overlap = prepare_total + sum(packed_compute_per_tile)
    p4_packed_overlap = overlapped_pipeline_cycles(packed_prepare_per_tile, packed_compute_per_tile)

    return {
        "name": shape.name,
        "OHxOW": f"{shape.output_h}x{shape.output_w}",
        "IC": shape.input_c,
        "K": f"{shape.kernel_h}x{shape.kernel_w}",
        "K_total": shape.k_total,
        "K_tiles": shape.k_tiles,
        "broken_k_tiles": broken_k_tiles(shape),
        "current_cycles": current.cycles,
        "current_hit_rate": round((current.cache_hits / current.valid_lanes) if current.valid_lanes else 0.0, 3),
        "current_tcdm_reads": current.tcdm_reads,
        "p4_raw_cycles": p4_raw_total,
        "p4_raw_slow_rows": raw.slow_rows,
        "p4_raw_two_seg_rows": raw.two_segment_rows,
        "p4_packed_no_overlap": p4_packed_no_overlap,
        "p4_packed_overlap": p4_packed_overlap,
        "speedup_raw": round(current.cycles / p4_raw_total, 2) if p4_raw_total else 0.0,
        "speedup_packed": round(current.cycles / p4_packed_overlap, 2) if p4_packed_overlap else 0.0,
    }


def print_markdown(rows: list[dict[str, int | str | float]], out: TextIO) -> None:
    columns = [
        "name",
        "OHxOW",
        "IC",
        "K",
        "K_total",
        "K_tiles",
        "broken_k_tiles",
        "current_cycles",
        "p4_raw_cycles",
        "p4_packed_overlap",
        "speedup_raw",
        "speedup_packed",
    ]
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns}

    print("| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |", file=out)
    print("| " + " | ".join("-" * widths[column] for column in columns) + " |", file=out)
    for row in rows:
        print("| " + " | ".join(str(row[column]).ljust(widths[column]) for column in columns) + " |", file=out)


def write_csv(rows: list[dict[str, int | str | float]], out: TextIO) -> None:
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="Add shape: name,H,W,IC,KH,KW[,SH,SW[,PH,PW[,DH,DW]]]. May be repeated.",
    )
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of Markdown.")
    parser.add_argument("--hit-cycles", type=int, default=2)
    parser.add_argument("--miss-extra-cycles", type=int, default=2)
    parser.add_argument("--spatz-im2col-cycles-per-row", type=int, default=4)
    parser.add_argument("--dma-prepare-cycles-per-row", type=int, default=1)
    parser.add_argument("--p4-slow-row-cycles", type=int, default=32)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model = PerfModel(
        hit_cycles=args.hit_cycles,
        miss_extra_cycles=args.miss_extra_cycles,
        spatz_im2col_cycles_per_row=args.spatz_im2col_cycles_per_row,
        dma_prepare_cycles_per_row=args.dma_prepare_cycles_per_row,
        p4_slow_row_cycles=args.p4_slow_row_cycles,
    )
    shapes = args.shape if args.shape else default_shapes()

    rows = [summarize_shape(shape, model) for shape in shapes]
    if args.csv:
        write_csv(rows, sys.stdout)
    else:
        print_markdown(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
