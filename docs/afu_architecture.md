# AFU Architecture

This document describes the current implemented AFU, not a proposed design.
The RTL lives under `hw/rtl/afu`, the cluster integration is in
`hw/rtl/cluster/npu_cluster.sv`, and the firmware wrappers are in
`sw/lib/hal_afu.h` and `sw/lib/spatz_ops.c`.

Last checked against RTL/firmware: 2026-07-21.

## Current Design Snapshot

| Area | Current implementation | Practical limit / contract |
|---|---|---|
| Control path | 32-bit OBI target at `NPU_AFU_BASE = 0x2000_3000` | Config writes update shadow CSRs. A `STATUS/START` bit0 write commits shadow to active only when AFU is not busy. |
| Data path | 256-bit TCDM beats, 32 bytes per beat | Production graph buffers should remain 32-byte aligned even though byte-enable tails are supported. |
| TCDM masters | Primary read/write master plus RHS read-only master | RHS port is used only by `MUL_Q7` and `ADD_I8`. Other modes use the primary port only. |
| Input FIFOs | RFIFO depth 2, RHS_RFIFO depth 2 | Only one primary read and one RHS read can be outstanding; this is a low-area streaming engine, not a many-request DMA. |
| Output FIFO | WFIFO depth 2, 256-bit data + 32 byte enables | `done_o` waits for WFIFO drain and backend idle, so completion is architecturally after writes reach TCDM. |
| LUT storage | `LUT_LANES=4`, two 256x32 banks per lane | Normal modes ping-pong active/stage banks. DFL uses fixed bank0/bank1 interpretation. |
| Generic LUT modes | E8/E16/E32, 4 input bytes per core step | Good for activation/clamp/table staging; not a general vector ISA. |
| Binary modes | ADD_I8 and MUL_Q7, 32 byte lanes per paired beat | `MUL_Q7` is fixed to `(lhs * rhs) >> 7` with i8 saturation; arbitrary multiplier/shift still uses software fallback. |
| YOLO fused modes | DFL `reg_max=4` ROW32 low16 and class sigmoid ROW32 high16 | Fixed raw-head layout: box logits in lanes `0..15`, class logits in lanes `16..31`. |
| GlobalAvgPool | C32-blocked spatial reduction | `SRC2_PTR` carries `spatial_count`; firmware loads reciprocal Q31 into normal LUT entry 0 before start. |

## 1. Role

The AFU is a small streaming accelerator for post-processing and element-wise
operators that are inefficient or unsupported on the current Snitch/Spatz path.
It is attached to the shared TCDM as an independent HWPE-style master and is
configured through the Snitch MMIO path.

The current AFU covers:

- INT8 LUT transforms: sigmoid/logistic, clamp/ReLU6, and other 256-entry
  activation tables.
- Wider LUT staging: INT8 input to E16/E32 output for exp-like intermediate
  tensors.
- Dual-source INT8 arithmetic: Q7 multiply and saturating add.
- YOLO raw-head postprocess: fused DFL `reg_max=4` and class sigmoid.
- C32 global average pooling.

The AFU is not a general vector engine. Spatz remains the owner for general
vector instructions and reductions that are already supported there. The AFU is
used when a narrow fixed datapath avoids scalar loops or unsupported vector
operations.

## 2. Cluster Integration

```text
Snitch D-side MMIO
        |
        v
  0x2000_3000 AFU target window
        |
        v
+--------------------------------------------------------------+
|                           AFU                                |
|                                                              |
|  afu_frontend                                                |
|    - MMIO target                                             |
|    - shadow config registers                                 |
|    - normal ping-pong LUT writes                             |
|    - fixed DFL exp/recip LUT writes                          |
|                                                              |
|  afu_backend                                                 |
|    - primary TCDM master: reads src and writes dst            |
|    - RHS read-only TCDM master: reads src2 for binary modes   |
|    - pending transaction register for OBI gnt stalls          |
|                                                              |
|  RFIFO      RHS_RFIFO      afu_core      WFIFO                |
|    256b        256b        compute        256b + 32b BE       |
|                                                              |
+--------------------------------------------------------------+
        |                         |
        v                         v
Shared TCDM master 10       Shared TCDM master 12
primary read/write          RHS read for binary modes
```

Current cluster parameters:

- AFU MMIO base: `0x2000_3000`.
- AFU MMIO aperture: `0x1000` bytes.
- Shared TCDM data width: `256` bits, or 32 bytes per beat.
- Normal LUT lanes: `4`.
- RFIFO/RHS_RFIFO depth: `2` beats.
- WFIFO depth: `2` entries, each entry stores `256` data bits plus `32` byte
  enables.
- TCDM masters:
  - master 10: AFU primary read/write port.
  - master 12: AFU RHS read-only port for binary modes.
- `done_o` is connected to interrupt source `NPU_IRQ_SRC_AFU`.

`done_o` is asserted only when all three conditions are true:

```text
core_done && wfifo_all_empty && backend_idle
```

This prevents the firmware from observing completion before the final write beat
has drained into TCDM.

The AFU does not own an iDMA channel internally. Current graph firmware may use
iDMA before or after AFU jobs to move tensors or LUT blobs between L2 and TCDM,
but AFU execution itself is driven by its OBI masters and MMIO CSRs.

## 3. RTL Blocks

### 3.1. `afu.sv`

Top-level wrapper. It instantiates:

- `afu_frontend`
- `afu_backend`
- `afu_core`
- `afu_fifo_ff` for RFIFO
- `afu_fifo_ff` for RHS_RFIFO
- `afu_fifo_ff` for WFIFO

It also combines core/backend status into the final `done_o` signal.

### 3.2. `afu_frontend.sv`

The frontend is the MMIO target.

Important behavior:

- MMIO grants are immediate (`obi_s_gnt_o = 1`).
- Config writes go to shadow registers.
- A write of bit 0 to the status/start register starts the AFU only when it is
  not busy.
- On start, shadow registers are atomically copied into active registers and a
  one-cycle `cfg_start_o` pulse is emitted.
- Reads from config CSRs return shadow values, not active values. `STATUS` is
  the exception and returns `{error,busy,done}`.
- Normal LUT writes are staged into the inactive ping-pong LUT bank.
- Starting a non-DFL mode with pending LUT writes swaps the active/stage banks.
- DFL fixed LUT writes target dedicated banks and are blocked while AFU is busy.

Writes to DFL fixed LUT windows are ignored while busy because
`lut_we_o = req && we && lut_sel && !(lut_dfl_sel && afu_busy_i)`. Normal
ping-pong LUT writes are not blocked by busy, but runtime software should only
modify the inactive staging bank and must respect the next-start bank-swap
semantics.

The shadow-register behavior is important for non-blocking firmware: software
can preload the next AFU job while a previous job is still running, then launch
it with one start write once the current job is complete.

### 3.3. `afu_backend.sv`

The backend owns TCDM access.

Primary port:

- Reads source data from `SRC_PTR`.
- Writes output data to `DST_PTR`.
- Uses one 256-bit OBI request at a time.
- Holds an unganted request in a pending register.
- Gives write traffic priority over new primary reads.
- Aligns source/destination beats down to 32-byte boundaries.
- Uses byte enables from WFIFO to support tails and packed sub-beat outputs.

RHS port:

- Active only in `MUL_Q7` and `ADD_I8`.
- Reads `SRC2_PTR` through a separate read-only TCDM master.
- Feeds RHS_RFIFO so binary modes can consume LHS/RHS beats together.

The backend uses at most one outstanding primary read and one outstanding RHS
read. It does not coalesce beyond naturally aligned 32-byte beats, and it does
not reorder writes around reads except for the local primary-port arbitration
rule: WFIFO write traffic has priority over issuing a new primary read. This is
intentional to keep completion precise and avoid a deep write backlog.

The backend stops issuing further reads when `core_done` is observed, then waits
until outstanding requests, pending requests, and WFIFO writes are drained before
reporting idle.

### 3.4. `afu_core.sv`

The core is a mode-dependent FSM.

Common LUT modes use a two-stage flow:

```text
RFIFO beat -> byte select / LUT index -> 1-cycle LUT read -> output pack -> WFIFO
```

The normal LUT path has 4 lanes, so E8/E16/E32 transform up to four input bytes
per core step. Output packing and WFIFO byte enables handle E8, E16, E32, and
tail writes.

Binary modes bypass the LUT SRAM:

```text
RFIFO beat + RHS_RFIFO beat -> 32 byte-wise ALU lanes -> WFIFO
```

`ADD_I8` packs directly from the stage-1 lane bundle. `MUL_Q7` has an explicit
three-stage registered path:

```text
S1: latch LHS/RHS beat, lane count, output address
S2: 32 signed 8x8 products
S3: arithmetic shift by 7, i8 saturation, byte-enable pack
```

This keeps the multiply path synthesizable at a high target frequency while
preserving one 32-lane beat per cycle once the short pipeline is full and the
TCDM/WFIFO interfaces do not stall.

Special fused modes have dedicated FSM states:

- `ST_DFL_EXP_REQ`, `ST_DFL_EXP_WAIT`, `ST_DFL_RECIP_WAIT`,
  `ST_DFL_MUL_WRITE`, `ST_DFL_WRITE`, `ST_DFL_PUSH`
- `ST_CLASS_LUT_REQ`, `ST_CLASS_LUT_WAIT`, `ST_CLASS_PUSH`
- `ST_GAP_ACCUM`, `ST_GAP_RECIP_REQ`, `ST_GAP_RECIP_WAIT`,
  `ST_GAP_MUL_WRITE`, `ST_GAP_WRITE`, `ST_GAP_PUSH`

These modes use the same backend and WFIFO infrastructure but have custom
packing and loop semantics.

### 3.5. LUT Banks

The core instantiates two LUT banks per lane.

Normal modes:

- Firmware writes through `NPU_AFU_LUT_BASE`.
- Writes go to the inactive staging bank.
- At `START`, if a LUT update is pending, active and staging banks swap.
- This preserves the old active LUT while firmware prepares a new LUT.

DFL fused mode:

- `NPU_AFU_DFL_EXP_LUT_BASE` maps to fixed bank0.
- `NPU_AFU_DFL_RECIP_LUT_BASE` maps to fixed bank1.
- Bank swapping is disabled for DFL mode.
- Writes to the fixed DFL LUT windows are blocked while AFU is busy.

GlobalAvgPool does not have a dedicated reciprocal SRAM. It reuses normal LUT
entry 0 from whichever normal bank is active at start. The current wrapper
therefore writes `recip_q31 = floor(2^31 / spatial_count)` into
`NPU_AFU_LUT_BASE + 0` before launching `NPU_AFU_MODE_GLOBAL_AVGPOOL_C32`.
If future firmware starts caching normal activation LUTs aggressively, it must
treat GAP reciprocal entry 0 as a separate LUT identity.

## 4. Programming Model

### 4.1. Register Map

All offsets are relative to `NPU_AFU_BASE = 0x2000_3000`.

| Offset | Name | Access | Meaning |
|---:|---|---|---|
| `0x000..0x3ff` | normal LUT | W | 256 32-bit entries for ping-pong LUT modes |
| `0x400` | `STATUS/START` | R/W | read `{error,busy,done}`; write bit0 to start |
| `0x404` | `SRC_PTR` | R/W shadow | source pointer or LHS pointer |
| `0x408` | `DST_PTR` | R/W shadow | destination pointer |
| `0x40c` | `LENGTH` | R/W shadow | mode-specific input length in bytes/elements |
| `0x410` | `MODE` | R/W shadow | 3-bit mode ID |
| `0x414` | `SRC2_PTR` | R/W shadow | RHS pointer or mode-specific metadata |
| `0x800..0xbff` | DFL exp LUT | W | 256 32-bit fixed exp entries |
| `0xc00..0xfff` | DFL reciprocal LUT | W | 256 32-bit fixed reciprocal entries |

CSR address decoding uses offset bits `[5:0]` for the five CSRs, so software
should use the symbolic addresses in `sw/lib/npu_memory_map.h` and avoid relying
on mirrored aliases inside the AFU aperture.

Status bits:

| Bit | Name |
|---:|---|
| 0 | `DONE` |
| 1 | `BUSY` |
| 2 | `ERROR` |

`ERROR` is currently tied low in `afu.sv`; timeout handling is done in firmware
wrappers and tests.

### 4.2. Mode IDs

| Mode | ID | Wrapper | Input | Output |
|---|---:|---|---|---|
| `NPU_AFU_MODE_E8` | 0 | `npu_logistic_i8`, `npu_clamp_i8` | compact INT8 | compact INT8 |
| `NPU_AFU_MODE_E16` | 1 | legacy assisted DFL helper | compact INT8 | compact uint16 |
| `NPU_AFU_MODE_E32` | 2 | generic LUT staging | compact INT8 | compact uint32 |
| `NPU_AFU_MODE_MUL_Q7` | 3 | `npu_mul_q7_i8` | two compact INT8 streams | compact INT8 |
| `NPU_AFU_MODE_ADD_I8` | 4 | `npu_add_i8` | two compact INT8 streams | compact INT8 |
| `NPU_AFU_MODE_DFL4_ROW32_Q8` | 5 | `npu_dfl_softmax4_row32_i8_q8` | ROW32 low 16 logits | four Q8.8 distances/location |
| `NPU_AFU_MODE_CLASS_SIGMOID_ROW32_HIGH16` | 6 | `npu_class_sigmoid_row32_high16_i8` | ROW32 high 16 logits | compact 16 INT8 scores/location |
| `NPU_AFU_MODE_GLOBAL_AVGPOOL_C32` | 7 | `npu_global_avgpool_c32_i8` | C32-blocked tensor | one C32 output group per channel group |

`LENGTH` meaning is mode-specific:

| Mode group | `LENGTH` unit | Notes |
|---|---|---|
| E8/E16/E32 | input elements, one byte per input element | Output bytes are `LENGTH`, `2*LENGTH`, or `4*LENGTH`. |
| MUL_Q7/ADD_I8 | input/output elements, one byte per lane | LHS and RHS must cover the same logical byte count. |
| DFL4_ROW32_Q8 | input bytes | Must be `locations * 32`. Output bytes are `locations * 8`. |
| CLASS_SIGMOID_ROW32_HIGH16 | input bytes | Must be `locations * 32`. Output bytes are `locations * 16`. |
| GLOBAL_AVGPOOL_C32 | input bytes | Must be `spatial_count * ceil(channels/32) * 32`; `SRC2_PTR` is `spatial_count`. |

### 4.3. Alignment Contract

The backend internally aligns TCDM read/write beats down to 32-byte boundaries
and the core selects bytes within the returned beat. The current firmware/test
contract is still conservative:

- Use 32-byte aligned source and destination pointers for production paths.
- Keep ROW32/C32 data physically padded to 32 channels.
- Tails are represented with byte enables on the final write beat.
- Arbitrary unaligned E16/E32 destination layouts are supported by the RTL test
  bench but are not yet a scheduler contract for production graph generation.

For binary modes, both `SRC_PTR` and `SRC2_PTR` are aligned down independently.
The core tracks source byte offsets, so unaligned block-level cases work, but
production graph generation should keep both streams aligned to preserve the
32-lane throughput model.

## 5. Mode Semantics

### 5.1. E8/E16/E32 LUT Modes

These modes read one INT8 input element and emit one LUT value:

```text
E8 : output byte  = lut[input_u8][7:0]
E16: output half  = lut[input_u8][15:0]
E32: output word  = lut[input_u8][31:0]
```

Current uses:

- E8 logistic/class-like activation.
- E8 clamp/ReLU6 through a generated clamp LUT.
- E16/E32 for exp-like intermediate staging when a fused mode is not available.

The wrapper currently loads 256 LUT entries before each standalone call. Runtime
graph code should cache LUT identity and avoid reloading unchanged tables.

### 5.2. `MUL_Q7`

`MUL_Q7` uses both TCDM read ports:

```text
lhs_i8 = SRC_PTR[i]
rhs_i8 = SRC2_PTR[i]
dst_i8 = clamp_i8((lhs_i8 * rhs_i8) >> 7)
```

The core computes up to 32 byte lanes from a pair of 256-bit input beats. This is
intended for SiLU-style activation multiply and other Q7 element-wise products.

### 5.3. `ADD_I8`

`ADD_I8` also uses both TCDM read ports:

```text
dst_i8 = clamp_i8(lhs_i8 + rhs_i8)
```

The wrapper `spatz_add_i8()` uses AFU only for the full-range clamp case
`[-128, 127]`; other min/max ranges still fall back to the scalar helper path.

### 5.4. Fused YOLO DFL

Mode `NPU_AFU_MODE_DFL4_ROW32_Q8` consumes one 32-byte raw-head row per spatial
location. Bytes `0..15` are interpreted as:

```text
side0 bin0..3, side1 bin0..3, side2 bin0..3, side3 bin0..3
```

For each side:

```text
max      = max(logits[0..3])
exp[i]   = exp_lut[(logits[i] - max) & 0xff]
sum      = exp[0] + exp[1] + exp[2] + exp[3]
weighted = exp[1] + 2*exp[2] + 3*exp[3]
recip    = recip_lut[normalized_sum_index(sum)]
distance = round((weighted << 8) / sum) using the reciprocal LUT
```

Output is four little-endian Q8.8 `uint16_t` values per location. The current
micro-YOLO path uses host/Python-generated LUT contents and checks the output
byte-exactly against the same fixed-point model.

Implementation details:

- Exp values are 16-bit values stored in 32-bit LUT words.
- The reciprocal index is generated from an 18-bit sum by normalizing the sum
  into a Q8 range; the RTL clamps the table index to `0..255`.
- The reciprocal LUT output is consumed through fixed bank1 lane 0.
- `weighted << 8` is multiplied by the reciprocal LUT entry and rounded in a
  registered stage before being packed into the 32-byte output beat.

Limitations:

- `reg_max` is fixed at 4.
- Exactly four box sides are computed.
- Input must be ROW32; low lanes `0..15` carry the logits. High lanes `16..31`
  are ignored by this mode.
- This is not a general softmax engine. Attention softmax or arbitrary class
  softmax needs a new row-reduction mode.

### 5.5. Fused Class Sigmoid

Mode `NPU_AFU_MODE_CLASS_SIGMOID_ROW32_HIGH16` consumes one 32-byte raw-head row
per location and applies the active E8 LUT to bytes `16..31`.

Output is compact:

```text
dst[location][0..15] = sigmoid_lut(src_row32[location][16..31])
```

This removes the need to materialize or scan the low DFL logits when only class
scores are being produced.

The class mode still uses the normal ping-pong LUT path. A pending normal LUT
update swaps active/stage banks on start, unlike DFL fixed LUT windows.

Limitations:

- It is fixed to 16 class logits per ROW32 location.
- It only reads lanes `16..31`.
- It writes compact class bytes, not ROW32-padded class rows.

### 5.6. C32 Global Average Pool

Mode `NPU_AFU_MODE_GLOBAL_AVGPOOL_C32` consumes C32-blocked input:

```text
input[spatial][c32_group][lane0..31]
```

`SRC2_PTR` is reinterpreted as `spatial_count`. The wrapper computes:

```text
spatial_count = input_h * input_w
groups        = ceil(channels / 32)
length        = spatial_count * groups * 32
```

For each C32 group, the core accumulates 32 signed lanes across
`spatial_count` rows, multiplies the absolute sum by the Q31 reciprocal stored in
normal LUT entry 0, applies one correction step, restores the sign, saturates to
INT8, and writes one 32-byte C32 output group.

Tail channels in the final C32 group are physically present because the format is
C32-padded. The software/golden model is responsible for ignoring inactive tail
lanes where the logical channel count is not a multiple of 32.

Limitations:

- Input order is spatial-major C32 groups:
  `input[spatial][c32_group][lane0..31]`.
- The mode assumes every C32 group has exactly `spatial_count` consecutive rows.
- `spatial_count == 0` is rejected by the firmware wrapper and maps to `ST_DONE`
  in RTL.
- The divide behavior is deterministic reciprocal-multiply with correction, not
  a hardware divider. Python golden tests should match this integer model.

## 6. Firmware and Host Flow

Current low-level HAL helpers:

- `afu_load_lut_entry()`
- `afu_load_dfl_exp_lut_entry()`
- `afu_load_dfl_recip_lut_entry()`
- `afu_preload()`
- `afu_preload_binary()`
- `afu_preload_class_sigmoid_row32_high16()`
- `afu_preload_global_avgpool_c32()`
- `afu_start_preloaded()`
- `afu_start()`
- `afu_start_binary()`
- `afu_start_global_avgpool_c32()`
- `afu_wait_done()`

The preferred runtime pattern is:

```text
1. Host/Python places tensors and, where possible, LUT blobs in L2.
2. Firmware/DMA moves data/LUTs to 32-byte aligned TCDM scratch windows where
   the current graph ABI requires TCDM execution.
3. Firmware preloads AFU shadow registers for the next job.
4. Firmware starts the AFU with one write to STATUS/START.
5. Firmware polls or waits for AFU done.
6. Python/cocotb validates output in tests; firmware does not do golden checks.
```

Current firmware wrappers follow this pattern. `npu_logistic_i8()`,
`npu_clamp_i8()`, `npu_dfl_softmax4_row32_i8_q8()`, and
`npu_class_sigmoid_row32_high16_i8()` write the AFU shadow config before the LUT
fill loop, then issue only the start write after the LUT contents are ready.
Binary Add/Mul and GlobalAvgPool also use explicit preload followed by
`afu_start_preloaded()`. This removes late config writes from the immediate
start path and makes the wrappers compatible with future graph-level
non-blocking dispatch.

Important caveat: LUT writes are still MMIO writes performed by firmware in the
current wrappers. Host/Python already owns LUT value generation in the tests and
can place LUT blobs in L2, but there is no AFU-side DMA ingestion of the LUT
windows yet. A future graph scheduler may use iDMA or a small firmware loop to
copy L2 LUT blobs into AFU MMIO windows before the start pulse.

Standalone AFU operator tests follow the same rule as the rest of the cleaned
test suite: firmware configures and dispatches only; Python/cocotb owns input
preload and golden comparison. The fused DFL cluster wrapper test now preloads
ROW32 input plus exp/reciprocal LUTs directly into TCDM and checks TCDM output,
so the measured path excludes firmware-side DFL setup loops and iDMA copy-back
noise.

## 7. Performance Model

The AFU has two different throughput regimes.

### 7.1. LUT Transform Modes

E8/E16/E32 are limited primarily by the 4 LUT lanes:

```text
compute_cycles ~= ceil(input_elements / 4)
read_beats     ~= ceil(input_bytes / 32)
write_beats    ~= ceil(output_bytes / 32)
```

For long tensors without TCDM contention, the observed active cycles should be
closer to the compute bound than to raw traffic because backend and core work
overlap. Small standalone tests are dominated by firmware boot, LUT loading,
MMIO writes, and polling.

### 7.2. Binary Modes

`MUL_Q7` and `ADD_I8` process up to 32 lanes per paired beat once both RFIFOs
have data. They consume more read bandwidth because they use two TCDM read
streams plus one write stream:

```text
traffic ~= lhs_read + rhs_read + dst_write
```

The second AFU TCDM master was added specifically to avoid interleaving RHS/LHS
reads through one port for these modes.

### 7.3. Fused DFL/Class Modes

Fused DFL and class sigmoid are intentionally ROW32-specialized:

- One 32-byte read per location.
- DFL writes 8 bytes per location, packed into 32-byte write beats.
- Class sigmoid writes 16 bytes per location, packed into 32-byte write beats.

DFL is not one output location per cycle. It performs four side computations per
location. Each side issues an exp LUT read, waits for the exp data, issues a
reciprocal LUT read, registers the reciprocal multiply, rounds/packs the result,
and then advances to the next side. It is still much faster and cleaner than CPU
delta/reduce loops plus separate AFU E16 staging.

Class sigmoid is lighter: it processes four high-half class bytes per LUT step,
so one ROW32 location requires four LUT groups and produces 16 compact output
bytes.

### 7.4. Global Average Pool

Global average pool is reduction-dominated. It reads all spatial C32 beats and
writes one C32 output beat per channel group:

```text
read_beats  = spatial_count * c32_groups
write_beats = c32_groups
```

The current average path already uses reciprocal-multiply rather than a
sequential divider:

```text
recip_q31    = floor(2^31 / spatial_count) loaded into normal LUT entry 0
quotient     = (abs(sum) * recip_q31) >> 31
correction   = ((quotient * spatial_count) + spatial_count) <= abs(sum)
avg_abs      = quotient + correction
avg_signed   = sign_restore(avg_abs)
dst_i8       = saturate_i8(avg_signed)
```

The datapath contains 32 parallel 32x32-style products for GAP finalization.
This is high throughput for C32 output groups, but it is not free for synthesis.
If GAP becomes timing-critical, preserve the mode semantics and add pipeline
registers around the reciprocal multiply/correction path rather than replacing
it with scalar division.

## 8. PMU Observability

Current cluster PMU events include AFU counters:

| Counter index | Meaning |
|---:|---|
| 16 | AFU done pulse count |
| 17 | AFU TCDM requests, primary + RHS |
| 18 | AFU TCDM stalls, primary + RHS |

The PMU does not yet split AFU read/write requests, primary/RHS requests, core
stall cycles, RFIFO empty stalls, or WFIFO full stalls. For deeper tuning, add:

- AFU active cycles.
- Core wait-for-input cycles.
- Core wait-for-WFIFO cycles.
- Primary read requests and stalls.
- RHS read requests and stalls.
- Write requests and stalls.
- Per-mode completion counters.

## 9. Tests and Coverage

Current tests:

- `hw/rtl/afu/tb/tb_afu.sv`
  - Block-level AFU testbench.
  - Covers E8/E16/E32 aligned and unaligned cases.
  - Covers fused DFL and class sigmoid block behavior.
  - Checks normal LUT ping-pong behavior after DFL fixed-bank use.
  - The current block-level harness ties the RHS OBI port inactive, so binary
    ADD/MUL coverage is currently stronger at cluster/operator-wrapper level.
- `sw/test/afu_ops`
  - Standalone C-callable AFU wrapper binaries:
    - `afu_ops_add_full.bin`
    - `afu_ops_mul_q7_full.bin`
    - `afu_ops_logistic.bin`
    - `afu_ops_logistic_full.bin`
    - `afu_ops_clamp_relu6.bin`
    - `afu_ops_dfl_fused.bin`
    - `afu_ops_class_sigmoid.bin`
    - `afu_ops_global_avgpool.bin`
- `hw/rtl/cluster/tb/tests/test_spatz_operator_library.py`
  - Cluster-level wrapper tests.
  - Python/cocotb owns preload and golden output comparison.
- `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py`
  - End-to-end micro-YOLO graph with fused DFL and class sigmoid.

Recent reference result for the standalone fused DFL cluster wrapper:

```text
test_afu_op_dfl_fused: PASS
cycles=5580
idma: busy=0 start=0 done=0
afu: done=43 tcdm_req=80 stall=0
```

That number is a cluster test PMU result, not a pure RTL core latency number.

## 10. Current Limitations and Next Work

- Normal LUT wrappers still reload all 256 entries per standalone call. Runtime
  firmware should cache active LUT identity and skip unchanged reloads.
- The AFU has shadow registers but no firmware job queue yet. Current wrappers
  preload one job before start; graph-level overlap of a future AFU job with a
  current non-AFU layer still needs scheduler support and careful LUT/weight
  lifetime handling.
- AFU LUT programming is still a firmware/MMIO loop. Host-precomputed LUT blobs
  are supported at the graph-data level, but there is not yet a native DMA-to-AFU
  LUT loader ABI.
- The production scheduler should keep using 32-byte aligned TCDM buffers until
  unaligned E16/E32 destinations are explicitly supported at the graph level.
- `ERROR` status is not yet driven by real hardware error conditions.
- `spatz_add_i8()` only uses AFU for full-range saturation. Narrow clamp add
  still uses scalar fallback.
- `spatz_mul_i8()` still uses scalar software for arbitrary
  multiply/requant/clamp. The native AFU path only covers Q7 multiply through
  `npu_mul_q7_i8()`.
- Legacy assisted DFL (`npu_dfl_softmax4_i8_q8`) still exists as a software/API
  fallback, but the optimized YOLO path should use fused ROW32 DFL.
- DFL fused mode is fixed to YOLO `reg_max=4` and cannot be reused as a general
  softmax for LLM attention.
- Class sigmoid fused mode is fixed to ROW32 high16 and compact 16-byte output.
- GlobalAvgPool assumes C32-blocked spatial-major input and uses normal LUT entry
  0 as reciprocal storage.
- More detailed AFU PMU events are needed before optimizing FIFO depth, OBI
  outstanding policy, or reduction datapaths further.
