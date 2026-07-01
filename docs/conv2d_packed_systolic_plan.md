# Packed Systolic Conv2D Plan

## Goal

Add a packed Conv2D execution path around the existing `32x32` systolic array
without redesigning the MAC array. Conv2D is lowered to tiled GEMM:

- `M = OH * OW` rows, tiled by `Mtile`.
- `N = OC` columns, tiled by `32`.
- `K = IC * KH * KW`, tiled by `32`.

The first implementation target is correctness and reusable control flow. Full
performance optimization, line-buffer reuse, and direct-conv dataflow are later
work.

## Current Baseline

The current cluster has no dedicated IFM SRAM. Input activation, weights,
partial sums, and outputs live in Shared Data TCDM. The systolic controller
reads weight and IFM rows from TCDM, stages them through small FIFOs, feeds the
array, and drains OFM rows back to TCDM.

The systolic array already exposes a `psum_data_i` boundary, but the controller
currently drives it as zero. Therefore native GEMM32 only supports one
`K=32` block per invocation unless partial sums are accumulated outside the
array.

## Architecture

```text
L2
 │
 DMA
 │
Shared Data TCDM
 │
 ├─ input tensor
 ├─ weight tiles packed as K32 x OC32
 ├─ INT32 psum tiles
 └─ INT8/INT32 output tiles
        │
        ▼
conv2d packed scheduler
        │
        ├─ generate tile-local IFM rows: Mtile x 32
        ├─ load weight tile: 32 x 32
        ├─ load or reuse INT32 psum tile for K>32
        ▼
systolic_controller
        ▼
32x32 systolic array
        ▼
OFM INT32 rows
        ▼
psum writeback or requant + activation + store
```

## P5 Candidate: Shared Generic Max-9-Line Linebuffer

Status: architecture proposal only. P3/P4 continue to use the current
software+iDMA+Spatz packed prepare path. This section defines the direct Conv2D
feeder architecture to revisit if measured packed-prepare cost remains the
dominant bottleneck.

### Requirement Envelope

- Filter size: `1x1` through `9x9`, including asymmetric filters such as `1x9`
  and `9x1`.
- Stride modes: `1x1`, `2x2`, `2x1`, `1x2`.
- `1x1` with stride `2x2` is explicitly unsupported.
- Input tile width: `7..4096` for non-`1x1`; `1..4096` for `1x1`.
- Input tile height: `1..4096`.
- Output tile width: `5..4096` for non-`1x1`; `1..4096` for `1x1`.
- Output tile height: `1..4096`.
- Input/output maps: `1..4096`, processed as `IC` and `OC` tiles.
- Internal output micro-tile shapes: `4x4`, `8x4`, `16x4`, `8x8`, `16x8`,
  `16x16`. These are compute blocks; larger logical tiles are decomposed into
  these blocks. The `4x4` block is kept for tails/debug even when the external
  non-`1x1` tile-width requirement starts at `5`.

### Top-Level Dataflow

The linebuffer is shared per systolic controller instance. It is not replicated
per filter or output channel, because all output-channel filters consume the
same IFM spatial window.

```text
L2 / DRAM
 │
 │  optional iDMA stripe prefetch
 ▼
Shared Data TCDM
 │
 ├─ input activation stripe / tile
 ├─ weight K32 x OC32 tiles
 ├─ INT32 psum tiles
 └─ INT8/INT32 output tiles
        │
        ▼
conv_linebuf_scheduler
        │
        ├─ config/register block
        ├─ output micro-tile sequencer
        ├─ row loader into shared 9-line ring buffer
        ├─ window/tap extractor with pad-zero injection
        ├─ K32 IFM row packer
        └─ systolic feed adapter
                │
                ▼
        systolic_controller
                │
                ▼
        32x32 systolic array
                │
                ▼
        psum accumulation or final requant/store
```

The feeder reads input activation from Shared Data TCDM. If a source tensor
lives in L2, the host-side compiler/runtime precomputes the required input
stripe copies and emits them as command descriptors. Snitch firmware only
dispatches those descriptors through iDMA and accelerator MMIO. The feeder then
avoids materializing `M x 32` im2col rows in TCDM; it emits IFM rows directly to
the systolic input path.

### Host-Compiled Descriptor Model

To reduce Snitch workload, all expensive prepare/configuration calculations are
done by the host before launching the cluster. In current simulation this host
compiler is Python; a later deployment can move the same logic into a runtime
driver.

The host computes a descriptor table, writes the table to L2/DRAM, then programs
a small host-visible command control/status register block through the cluster
AXI-slave path. Snitch uses that bootstrap information to copy the descriptor
table into a fixed reserved Shared TCDM command region. After the copy, Snitch
treats the local TCDM table as immutable work items:

- graph/layer order;
- output micro-tile decomposition;
- L2-to-TCDM input stripe copy ranges;
- weight tile and psum/output addresses;
- `OC` and `K` tile loop expansion;
- pre-decoded K32 lane descriptors;
- padding/zero-injection masks;
- linebuffer register images;
- expected debug/golden metadata for test mode.

Hot-path firmware must not perform division/modulo-heavy Conv2D coordinate
math. Its job is reduced to: fetch descriptor, program iDMA/MMIO registers,
start accelerator, wait for completion/IRQ, record status, and advance.

```text
Python host compiler
  -> layer graph + tensor shapes + addresses
  -> tile descriptors + lane descriptors + DMA copy descriptors
  -> write command stream into L2/DRAM
  -> AXI-slave writes command control/status registers
  -> Snitch iDMA copy into fixed TCDM command region
  -> Snitch dispatch loop from TCDM
  -> iDMA / linebuffer / systolic / requant
```

Descriptor memory placement policy for P0/P1:

- The command stream lives in L2/DRAM before launch.
- A fixed high Shared TCDM region is reserved as the local command staging
  window, currently `0x1017_F000..0x1017_FFFF` for 4 KB. Tile allocators must
  not use this range for input, weight, psum, output, or temporary im2col data.
- Snitch refills the staging window from the L2 command stream with iDMA. The
  command stream itself may be larger than 4 KB; firmware copies the chunk
  containing the next descriptor before decoding it.
- Descriptors are 32-byte aligned to match the native TCDM beat and simplify
  future hardware descriptor prefetch.
- Snitch reads each descriptor into local scalar variables before starting the
  corresponding iDMA/compute operation. This avoids descriptor fetch traffic
  overlapping the hot compute/data movement phase.
- A single descriptor must fit in the 4 KB staging window. If the window is too
  small, firmware fails with `BAD_CMD_BAD_SIZE`.
- Command streams larger than 4 KB are legal and are consumed by L2
  streaming/chunking.
- Descriptor format must remain endian-stable and versioned so Python golden,
  firmware, and RTL decode the same fields.

Host command control/status register block:

| Register | Direction | Role |
|----------|-----------|------|
| `CMD_L2_BASE` | Host write, Snitch read | L2/DRAM base address of the command stream. |
| `CMD_TOTAL_BYTES` | Host write, Snitch read | Exact command stream length. |
| `CMD_TCDM_BASE` | Host write, Snitch read | Reserved local TCDM command staging window base. |
| `CMD_TCDM_BYTES` | Host write, Snitch read | Reserved local TCDM staging window size; current firmware expects at least 4 KB. |
| `CMD_START` | Host write, Snitch read/clear | Optional launch bit for multi-run control; P0 can still start after fetch release. |
| `CMD_STATUS` | Snitch write, Host read | Idle/running/pass/fail state. |
| `CMD_FAIL_CODE` | Snitch write, Host read | Failure reason such as bad magic, bad size, unsupported command, or hardware timeout. |
| `CMD_FAIL_PTR` | Snitch write, Host read | L2 address of the failing descriptor, or staging-buffer base for bootstrap failures. |

The register block must be accessible from both the host AXI-slave frontend and
the Snitch MMIO path. It should not expose D-TCM or the whole Shared TCDM to the
host.

### Layer 0: Firmware/Scheduler Contract

The host owns model-level tiling, L2-to-TCDM staging decisions, and OC/K loop
expansion. Firmware owns descriptor dispatch. Hardware owns only one
direct-feeder micro-tile at a time.

```c
cmd_l2_base = REG_READ(CMD_L2_BASE);
cmd_total_bytes = REG_READ(CMD_TOTAL_BYTES);
cmd_tcdm_base = REG_READ(CMD_TCDM_BASE);
cmd_tcdm_bytes = REG_READ(CMD_TCDM_BYTES);

if (cmd_tcdm_bytes < 4096) {
  fail(BAD_CMD_BAD_SIZE);
}
if ((cmd_tcdm_base & 31) != 0 || (cmd_total_bytes & 31) != 0) {
  fail(BAD_CMD_ALIGNMENT);
}

refill_window_from_l2(offset = 0);
table = prefetch_table_header(cmd_tcdm_base);
if (table.magic != NPU_CMD_TABLE_MAGIC || table.total_bytes > cmd_total_bytes) {
  fail(BAD_CMD_TABLE);
}

cmd_offset = table.entry_offset;
cmd_end = table.total_bytes;

while (cmd_offset != cmd_end) {
  refill_window_from_l2(offset = cmd_offset);
  // Read the full descriptor into D-TCM/register-backed scalar state before
  // starting the engine. Descriptor fetch must not overlap compute traffic.
  cmd = prefetch_descriptor(cmd_offset);

  switch (cmd.type) {
    case CMD_IDMA_COPY:
      program_idma_from_descriptor(cmd.idma);
      wait_idma_done(cmd.idma.tx_id);
      break;

    case CMD_LINEBUF_CONV_TILE:
      program_linebuf_regs_from_descriptor(cmd.linebuf);
      start_linebuf_tile();
      wait_linebuf_done_irq_or_poll_status();
      break;

    case CMD_SYSTOLIC_GEMM_TILE:
      program_systolic_regs_from_descriptor(cmd.gemm);
      start_systolic_tile();
      wait_systolic_done_irq_or_poll_status();
      break;

    case CMD_BARRIER:
      wait_all_outstanding_work();
      break;
  }

  cmd_ptr += cmd.size_bytes;
}
```

Scheduler rules:

- `1x1/s1`, contiguous `IC` blocks should bypass the linebuffer and use the
  cheapest direct C32 stream path.
- `1x1/s2x2` must trap as unsupported.
- `K > 32` is expanded by the host into multiple descriptors. The linebuffer
  emits one `M x 32` logical block per linebuffer descriptor.
- Intermediate K descriptors write/read INT32 psum. Only host-marked final K
  descriptors may enable fused requant/activation.
- Firmware may validate descriptor version, bounds, and unsupported mode bits,
  but must not recompute tile geometry in the performance path.

### Layer 1: Config/Register Block

The register block should live with the systolic controller/direct feeder, not
inside the cluster-wide control register file. It exposes shape, tile, stride,
padding, address, mode, and lane-descriptor registers.

```c
enum npu_cmd_type {
  CMD_IDMA_COPY = 1,
  CMD_LINEBUF_CONV_TILE = 2,
  CMD_SYSTOLIC_GEMM_TILE = 3,
  CMD_BARRIER = 4,
};

struct npu_cmd_header {
  uint16_t version;
  uint16_t size_bytes;
  uint16_t type;
  uint16_t flags;
};

struct lane_desc {
  bool valid;
  uint8_t kh;
  uint8_t kw;
  uint16_t ic;
};

struct linebuf_cfg {
  uint32_t input_base;
  uint32_t weight_base;
  uint32_t psum_base;
  uint32_t output_base;

  uint16_t input_h, input_w, input_c;
  uint16_t output_h, output_w, output_c;
  uint8_t kernel_h, kernel_w;
  uint8_t stride_h, stride_w;
  int8_t pad_top, pad_left;
  int8_t pad_bottom, pad_right;

  uint16_t tile_oh, tile_ow;
  uint16_t tile_oh_base, tile_ow_base;
  uint16_t oc_base;
  uint16_t k_base;

  bool first_k;
  bool last_k;
  bool requant_en;
  bool relu_clamp_en;
  bool debug_materialize_en;
};

struct linebuf_conv_cmd {
  struct npu_cmd_header header;
  struct linebuf_cfg cfg;
  struct lane_desc lane[32];
  uint32_t pad_zero_mask;
  uint32_t k_tail_zero_mask;
  uint32_t debug_im2col_base;
};
```

Validation pseudo-code:

```c
bool linebuf_cmd_valid(cmd) {
  if (cmd.header.version != NPU_CMD_VERSION) return false;
  if (cmd.header.size_bytes < sizeof(struct linebuf_conv_cmd)) return false;

  cfg = cmd.cfg;
  if (cfg.kernel_h < 1 || cfg.kernel_h > 9) return false;
  if (cfg.kernel_w < 1 || cfg.kernel_w > 9) return false;
  if (!stride_supported(cfg.stride_h, cfg.stride_w)) return false;
  if (cfg.kernel_h == 1 && cfg.kernel_w == 1 &&
      cfg.stride_h == 2 && cfg.stride_w == 2) return false;
  if (!micro_tile_supported(cfg.tile_oh, cfg.tile_ow)) return false;
  if (cfg.input_c < 1 || cfg.input_c > 4096) return false;
  if (cfg.output_c < 1 || cfg.output_c > 4096) return false;
  return true;
}
```

### Layer 2: Output Micro-Tile Sequencer

The host decomposes a logical output tile into supported micro-tiles. The RTL
sequencer consumes one already-decoded micro-tile descriptor and only advances
within that descriptor. It does not choose tile shape at runtime.

```c
// Host-side Python compiler.
for each micro_tile in split_preferred(tile_oh, tile_ow,
                                      {16x16, 16x8, 16x4, 8x8, 8x4, 4x4}) {
  input_y_min = micro_tile.oh0 * stride_h - pad_top;
  input_y_max = (micro_tile.oh1 - 1) * stride_h + kernel_h - 1 - pad_top;
  input_x_min = micro_tile.ow0 * stride_w - pad_left;
  input_x_max = (micro_tile.ow1 - 1) * stride_w + kernel_w - 1 - pad_left;

  desc.tile = micro_tile;
  desc.input_y_min = clamp(input_y_min, 0, input_h - 1);
  desc.input_y_max = clamp(input_y_max, 0, input_h - 1);
  desc.input_x_min = clamp(input_x_min, 0, input_w - 1);
  desc.input_x_max = clamp(input_x_max, 0, input_w - 1);
  desc.top_pad_mask = compute_top_pad_mask(micro_tile, shape);
  desc.left_right_pad_masks = compute_lr_pad_masks(micro_tile, shape);
  emit_descriptor(desc);
}

// RTL/firmware-visible sequencer.
for oh = desc.tile.oh0; oh < desc.tile.oh1; oh++ {
  for ow = desc.tile.ow0; ow < desc.tile.ow1; ow++ {
    request_row_loader_for_predecoded_range(desc.input_y_min,
                                            desc.input_y_max);
    emit_window_for_predecoded_output_point(oh, ow);
  }
}
```

For the worst supported `16x16`, `9x9`, `stride=2` micro-tile, the input span is
`39x39`. The design still needs only a max-9-line ring because rows are streamed
through the ring; it must not allocate a full `39`-row tile buffer.

### Layer 3: Shared 9-Line Ring Buffer

The ring buffer stores at most nine input rows for the current channel/K slice.
It should be banked around the native TCDM beat width so full-width reads are
used when loading rows. The physical storage is shared across all kernel modes.

```c
struct row_slot {
  bool valid;
  int32_t global_ih_tag;
  uint8_t data[MAX_SEGMENT_W][CHANNEL_SLICE_BYTES];
};

row_slot rows[9];

void ensure_row_resident(int32_t global_ih) {
  if (global_ih < 0 || global_ih >= input_h) return; // pad row
  slot = global_ih % 9;
  if (rows[slot].valid && rows[slot].global_ih_tag == global_ih) return;

  rows[slot].valid = false;
  for (x = input_x_min; x <= input_x_max; x += TCDM_BEAT_PIXELS) {
    beat = tcdm_read(input_base + input_offset(global_ih, x, channel_slice));
    rows[slot].data[x - input_x_min] = beat;
  }
  rows[slot].global_ih_tag = global_ih;
  rows[slot].valid = true;
}
```

Implementation notes:

- The buffer should be segment-based, not full-image-width based. A full
  `9 * 4096 * 32B` design is too large for the cluster SRAM budget.
- The loader operates on the current `K_TILE=32` lane descriptor. For common
  `IC >= 32` aligned chunks, the channel slice is a contiguous `C32` vector.
- For small-IC first layers such as RGB, the row data can store only valid
  channels and let the K32 packer zero-fill the remaining lanes.

### Layer 4: Lane Descriptor / K32 Mapping

Each systolic IFM row is 32 bytes. The host pre-decodes each K-lane into
`(kh, kw, ic)` and stores the result in the command descriptor.

```c
// Host-side Python/C model.
void build_lane_desc_on_host(desc[32], uint32_t k_base,
                             uint32_t IC, uint32_t KH, uint32_t KW) {
  for (lane = 0; lane < 32; lane++) {
    k = k_base + lane;
    if (k >= IC * KH * KW) {
      desc[lane].valid = false;
      continue;
    }
    desc[lane].kh = k / (KW * IC);
    rem = k % (KW * IC);
    desc[lane].kw = rem / IC;
    desc[lane].ic = rem % IC;
    desc[lane].valid = true;
  }
}
```

This mapping is the same contract currently used by the software packed prepare
path, so the existing Python golden model can be reused.

Hardware consumes the pre-decoded `lane_desc[32]` from the descriptor table. It
must not divide by `IC` or `KW * IC` in the hot path.

### Layer 5: Window Extractor and Pad-Zero Injection

The extractor consumes output coordinates and lane descriptors, then emits one
packed 32-byte IFM row. Pad and K-tail lanes are zero-injected in hardware.

```c
uint8_t make_ifm_lane(uint32_t oh, uint32_t ow, lane_desc d) {
  if (!d.valid) return 0;

  ih = oh * stride_h + d.kh - pad_top;
  iw = ow * stride_w + d.kw - pad_left;
  if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) return 0;
  if (!row_resident(ih)) return stall_until_row_loaded();

  return linebuf_read(ih % 9, iw - input_x_min, d.ic);
}

// Host may precompute descriptor-level masks. Hardware still keeps final
// boundary checks for safety and for micro-tiles touching image borders.
void emit_ifm_row(uint32_t oh, uint32_t ow) {
  uint8_t ifm[32];
  for (lane = 0; lane < 32; lane++) {
    ifm[lane] = make_ifm_lane(oh, ow, lane_desc[lane]);
  }
  push_ifm_to_systolic(ifm);
}
```

ASIC note: a fully generic `32`-lane arbitrary byte gather is expensive. The
first implementation should prioritize:

- contiguous C32 lane groups for `IC >= 32`;
- RGB/small-IC grouped taps for `3x3` first-layer style workloads;
- a slower generic fallback for rare asymmetric/tail cases.

The generic functional path must be correct; the common paths should be
separately optimized after PMU data confirms where cycles are spent.

### Layer 6: Systolic Feed Adapter

The feed adapter turns extracted IFM rows into the same ready/valid contract
used by the current systolic controller. Weight rows remain loaded from TCDM
using the existing weight path.

```c
while (micro_tile_active) {
  if (ifm_fifo_ready && weight_fifo_ready && linebuf_row_available) {
    ifm_row = next_extracted_ifm_row();
    weight_row = read_weight_row(weight_base, k_base, oc_base);
    push_to_systolic(ifm_row, weight_row);
    emitted_rows++;
  } else {
    stall_counter++;
  }
}
```

The feed adapter must respect true OFM ready/valid backpressure. It must not
drop rows if the output drain stalls.

### Layer 7: Accumulation, Requant, and Store

The output side should reuse the existing systolic controller functionality.

```c
for each output row from systolic {
  if (!first_k) {
    acc = ofm_row + read_psum(psum_base, m, oc_base);
  } else {
    acc = ofm_row;
  }

  if (last_k && requant_en) {
    int8_row = requant_relu_clamp(acc, qparam[oc_base : oc_base + 31]);
    write_output_int8(output_base, m, oc_base, int8_row);
  } else {
    write_psum_int32(psum_base, m, oc_base, acc);
  }
}
```

This keeps the direct feeder scoped to input-window generation. It does not
duplicate the requant pipeline or OFM backpressure logic.

### Layer 8: Debug Materialize and PMU Counters

The feeder needs a debug mode that materializes the generated `M x 32` rows into
TCDM before feeding the array. This mode is not a performance path; it exists to
compare hardware-generated rows against the current software packed prepare.

```c
if (debug_materialize_en) {
  for each emitted IFM row {
    tcdm_write(debug_im2col_base + row_index * 32, ifm_row);
  }
}
```

Required counters:

- input rows loaded;
- IFM rows emitted;
- pad-zero lanes;
- K-tail zero lanes;
- TCDM read stall cycles;
- IFM feed stall cycles;
- OFM backpressure stall cycles;
- debug materialized rows.

Acceptance rule: direct-feeder tests must check both final Conv output and, in
debug mode, the generated `M x 32` packed rows against the existing software
golden.

## K-Tile Accumulation

For `K <= 32`, one systolic invocation is sufficient. For `K > 32`, firmware or
a future graph controller issues multiple GEMM32 blocks:

```text
for oc_tile in 0..OC step 32:
  for m_tile in 0..OH*OW step Mtile:
    for k_tile in 0..IC*KH*KW step 32:
      prepare packed A[Mtile, 32]
      load W[32, 32]
      if k_tile == 0:
        psum = A * W
      else:
        psum = psum + A * W
      if k_tile is last:
        requant/activation/store final output
      else:
        store INT32 psum tile
```

Phase 0 implements psum accumulation in the systolic output drain path. This is
intentionally conservative: the array computes one `A*W` block, the controller
reads the previous psum row from TCDM, adds it to the OFM row, then either
writes the accumulated INT32 row back or feeds the accumulated row into the
requant pipeline for the final K-block.

## Packed Tile Contract

The packed prepare path maps each logical GEMM row and K lane back to Conv2D
coordinates:

```text
m      -> oh, ow
k_lane -> kh, kw, ic
ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w
```

If `(ih, iw)` is outside the input image, the prepare path injects zero by
leaving the pre-cleared destination lane untouched. Otherwise it copies the
corresponding activation byte into a packed `32` byte row matching the current
systolic IFM input format.

## Operator Coverage

The verification plan must cover the packed Conv2D operator family, not only
`3x3`:

- Pointwise Conv `1x1`.
- Standard Conv `3x3`, `5x5`, `7x7`.
- Asymmetric Conv `1x3`, `3x1`, `1x5`, `5x1`.
- Stride `1/2`, padding `0/1`, dilation when enabled.
- Channel boundary cases: `IC/OC = 1, 3, 31, 32, 33, 64`.
- `K` boundary cases: `<32`, `=32`, `>32`, non-multiple of `32`.
- Depthwise/grouped Conv as separate functional paths; they are not the
  performance target for dense systolic GEMM.

## Implementation Phases

### P0: GEMM K-Block Foundation

- Add systolic MMIO fields for psum pointer and accumulation enable.
- Add controller support for `OFM + previous_psum -> OFM`.
- Add HAL helper for tiled `K>32` accumulation.
- Add exact data-output tests for `K=64` INT32 accumulation and fused final
  requant.

### Removed Legacy Conv2D Lowering Paths

The previous software materializer test and RTL debug path have been dropped
from active RTL/software. Conv2D
lowering now uses the packed prepare path in `sw/lib/conv2d_packed.*`, with
iDMA for contiguous and regular multi-spatial L2 tiles, plus Spatz RVV copies
for fallback L1/TCDM or irregular tiles.

Active Conv2D verification lives in `sw/test/conv_perf` and
`test_conv_perf`. Do not add new roadmap features or acceptance criteria to the
removed path. If prepare performance is insufficient, add a new
wide-segment extractor architecture instead of reviving byte-serial lowering.

### P3: Packed Conv2D Functional Completeness

Goal: turn the software+iDMA+Spatz packed prepare path into a reusable Conv2D
operator backend for model layers. There is no active direct-Conv2D RTL dependency for this phase.

Features to complete:

- **Output-channel tiling (`OC > 32`)**:
  - Add firmware scheduler loops for `oc_tile = 0..OC step 32`.
  - Define weight layout for `K32 x OC32` tiles across multiple output channel
    groups.
  - Store each OC tile to the correct output tensor offset.
- **K tiling edge cases**:
  - Keep `K_TILE=32`, but formally support `K < 32`, `K = 32`,
    `K > 32`, and non-multiple K tails through zero-injected lanes.
  - Ensure only final K-block can enable requant/final store; intermediate
    blocks must store INT32 psum.
- **Final block postprocess**:
  - Use existing systolic requant path for final accumulated block.
  - Support final output as either INT32 debug output or INT8 requantized output.
  - Add optional ReLU via requant clamp configuration (`clamp_min = 0`) for
    integer-only activation fusion.
- **Shape coverage**:
  - Support standard Conv `1x1`, `3x3`, `5x5`, `7x7`.
  - Support asymmetric Conv `1x3`, `3x1`, `1x5`, `5x1`.
  - Support stride `1` and `2`.
  - Support symmetric and asymmetric padding.
  - Support dilation `1` only. Dilation greater than `1` is out of scope for
    the current YOLO/CNN/Vision Transformer target path.
- **Backend policy**:
  - Conv1x1/contiguous K tiles use iDMA 2D pack.
  - Regular multi-spatial Conv2D tiles from L2 use iDMA 3D segment pack.
  - RGB/padding/border cases use iDMA 3D when representable as regular L2
    segments; otherwise they fall back to Spatz RVV pack.
  - L1/TCDM source tensors use Spatz RVV pack until an L1-side DMA/backend is
    introduced.
  - Scalar prepare must remain disabled for performance tests.
- **Unsupported in P3**:
  - Depthwise/grouped conv are tracked as separate paths; dense systolic Conv2D
    should not claim them.
  - No performance overlap, no line buffer, no weight reuse cache.
  - Dilation greater than `1` remains out of scope.

P3 implementation status:

- **Pointwise coverage**:
  - Conv1x1 `IC=1`, `IC=3`, `IC=31`, `IC=32`, `IC=33`, `IC=64`, `OC=32`
    exact INT32 compare is covered by `test_conv_perf`.
  - Conv1x1 `IC=33`, `OC=64` is covered as two OC tiles and two K tiles.
- **Kernel coverage**:
  - Conv3x3 pad0 and pad1 with `IC=32`, `OC=32` are covered.
  - Conv5x5 pad2 with `IC=3`, `OC=32` is covered.
  - Conv7x7 pad3 with `IC=1`, `OC=32` is covered.
  - Conv1x3, Conv3x1, Conv1x5, Conv5x1 are covered with exact INT32 compare.
- **Stride/padding coverage**:
  - Conv3x3 stride2 pad1 is covered.
  - Asymmetric kernel coverage is implemented; true asymmetric top/bottom or
    left/right padding remains a future register/API extension because the
    current packed scheduler config exposes symmetric `pad_h/pad_w`.
- **Tail and zero-injection coverage**:
  - `K < 32`: Conv3x3 `IC=1` (`K=9`) and Conv1x1 `IC=3` are covered.
  - `K = 32`: Conv1x1 `IC=32` is covered.
  - `K > 32` non-multiple: Conv1x1 `IC=33` and Conv3x3 `IC=5` (`K=45`)
    are covered.
  - Padding/tail zero lanes are checked through exact final output compare.
- **Final store coverage**:
  - INT32 output mode exact compare is covered.
  - Final-block requant INT32-to-INT8 exact compare is covered for Conv1x1
    `IC=64`.
  - ReLU-through-clamp uses the same requant clamp mechanism; a dedicated
    clamp-min-zero fixture remains optional.
- **Regression gates**:
  - `test_conv_perf` covers packed prepare shape/backend coverage.
  - `CONV_PERF_GROUP=1` is the dedicated OC tiling/pointwise regression.
  - `CONV_PERF_GROUP=2` is the dedicated kernel/stride/tail-K regression.
  - `CONV_PERF_GROUP=3` is the dedicated final-block requant regression.
  - Existing `test_independent_systolic`, `test_systolic_requant`, and
    `test_dma_tcm` must continue passing.

P3 known limits:

- Dilation greater than `1` remains unsupported by policy.
- True asymmetric top/bottom or left/right padding needs an API/register
  extension; current coverage is symmetric padding plus asymmetric kernels.
- Depthwise/grouped convolution remains a separate path, not dense systolic
  Conv2D.

### P4: Packed Prepare Performance

Goal: improve throughput without changing the correctness contract established
in P3.

Features to complete:

- **Tile scheduling**:
  - Choose `Mtile` policy for small/large feature maps.
  - Keep output FIFO high-water protection until true OFM backpressure exists.
  - Decide when to split large `M` into multiple controller invocations.
- **DMA/compute overlap**:
  - Double-buffer input, weight, and output/psum tiles where TCDM capacity
    allows it.
  - Prefetch next K tile or next OC tile while current tile computes.
  - Keep a non-overlapped debug mode for deterministic bring-up.
- **Weight/data reuse**:
  - Reuse weight tile across multiple `Mtile` chunks when TCDM placement allows.
  - Avoid full L2 im2col tensors; packed `M x 32` TCDM tiles are the compute
    interface to systolic.
- **Performance counters**:
  - Revisit OFM ready/valid into systolic controller so FIFO depth can be
    reduced safely.
  - Add counters for packed prepare cycles, iDMA wait cycles, Spatz pack cycles,
    systolic active cycles, OFM FIFO stalls, TCDM grant stalls, and DMA wait cycles.
- **Unsupported in P4**:
  - Re-architecting the MAC array or adding a full line-buffer direct-conv
    engine is out of scope unless packed prepare measurements cannot meet target.

P4 required tests:

- **Performance invariant tests**:
  - Run the same P3 correctness suite and compare output bit-exactly while
    checking backend tile counters.
  - Randomized bounded Conv1x1 and Conv3x3 fixtures with `M` crossing tile
    boundaries.
- **Large-shape smoke tests**:
  - Conv1x1 with `M=1024`, `IC=32`, `OC=32`.
  - Conv3x3 pad1 with `M=1024`, `IC=3`, `OC=32`.
  - Conv3x3 with `IC=32`, `OC=64`, `M` split into multiple tiles.
- **Stress tests**:
  - TCDM bank-conflict fixture for input/weight/output addresses.
  - Back-to-back Conv layers sharing output/input buffers.
  - DMA overlap test with intentionally delayed TCDM grants if testbench
    support exists.
- **Performance reporting tests**:
  - Log cycles per output element and systolic utilization for Conv1x1,
    Conv3x3, and Conv3x3 with `IC=32`.
  - Record packed prepare breakdown: iDMA wait, Spatz pack, systolic active,
    OFM drain wait.
  - Compare against current `test_conv_perf` baseline before accepting P4 changes.

## Initial Limits

- Phase 0 supports INT32 accumulation and fused requant on the final accumulated
  K-block.
- Accumulation mode uses small `Mtile` until the systolic output path has true
  output backpressure.
- No full im2col tensor is created in L2.
