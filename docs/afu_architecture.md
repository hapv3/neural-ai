# AFU Architecture

This document describes the current implemented AFU, not a proposed design.
The RTL lives under `hw/rtl/afu`, the cluster integration is in
`hw/rtl/cluster/npu_cluster.sv`, and the firmware wrappers are in
`sw/lib/hal_afu.h` and `sw/lib/spatz_ops.c`.

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
- Reads from config CSRs return shadow values, not active values.
- Normal LUT writes are staged into the inactive ping-pong LUT bank.
- Starting a non-DFL mode with pending LUT writes swaps the active/stage banks.
- DFL fixed LUT writes target dedicated banks and are blocked while AFU is busy.

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

Special fused modes have dedicated FSM states:

- `ST_DFL_EXP_REQ`, `ST_DFL_EXP_WAIT`, `ST_DFL_RECIP_WAIT`, `ST_DFL_PUSH`
- `ST_CLASS_LUT_REQ`, `ST_CLASS_LUT_WAIT`, `ST_CLASS_PUSH`
- `ST_GAP_ACCUM`, `ST_GAP_PUSH`

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

### 4.3. Alignment Contract

The backend internally aligns TCDM read/write beats down to 32-byte boundaries
and the core selects bytes within the returned beat. The current firmware/test
contract is still conservative:

- Use 32-byte aligned source and destination pointers for production paths.
- Keep ROW32/C32 data physically padded to 32 channels.
- Tails are represented with byte enables on the final write beat.
- Arbitrary unaligned E16/E32 destination layouts are supported by the RTL test
  bench but are not yet a scheduler contract for production graph generation.

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

### 5.5. Fused Class Sigmoid

Mode `NPU_AFU_MODE_CLASS_SIGMOID_ROW32_HIGH16` consumes one 32-byte raw-head row
per location and applies the active E8 LUT to bytes `16..31`.

Output is compact:

```text
dst[location][0..15] = sigmoid_lut(src_row32[location][16..31])
```

This removes the need to materialize or scan the low DFL logits when only class
scores are being produced.

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
`spatial_count` rows, divides each lane by `spatial_count` with truncation toward
zero, and writes one 32-byte C32 output group.

Tail channels in the final C32 group are physically present because the format is
C32-padded. The software/golden model is responsible for ignoring inactive tail
lanes where the logical channel count is not a multiple of 32.

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
2. Firmware/DMA moves data/LUTs to 32-byte aligned TCDM scratch windows.
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
location, including fixed exp and reciprocal LUT accesses. It is still much
faster and cleaner than CPU delta/reduce loops plus separate AFU E16 staging.

### 7.4. Global Average Pool

Global average pool is reduction-dominated. It reads all spatial C32 beats and
writes one C32 output beat per channel group:

```text
read_beats  = spatial_count * c32_groups
write_beats = c32_groups
```

The current divider/average logic is simple and deterministic. If global average
pool becomes a major full-model bottleneck, the next improvement should be a more
pipelined reduction tree or reciprocal-multiply average path.

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
- The production scheduler should keep using 32-byte aligned TCDM buffers until
  unaligned E16/E32 destinations are explicitly supported at the graph level.
- `ERROR` status is not yet driven by real hardware error conditions.
- `spatz_add_i8()` only uses AFU for full-range saturation. Narrow clamp add
  still uses scalar fallback.
- Legacy assisted DFL (`npu_dfl_softmax4_i8_q8`) still exists as a software/API
  fallback, but the optimized YOLO path should use fused ROW32 DFL.
- More detailed AFU PMU events are needed before optimizing FIFO depth, OBI
  outstanding policy, or reduction datapaths further.
