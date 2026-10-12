# YOLO Neural-AI reference replay

This diagnostic starts from reset or resumes a pinned first-cascade memory snapshot and
checks intermediate INT8 tensors against the Neural-AI hybrid TFLite runtime.
Bounded mode is not a full-model correctness test or a general snapshot-state restorer.
Full-graph mode additionally compares the complete output binding.
Conv/Depthwise/FC use Neural-AI reference arithmetic; other operators still use
TFLite reference implementations.

## Prepare expected data without simulation

Run from the neural-ai repository with the sibling neural-compiler reference
runtime already built (see its `tools/neural_ai_tflite_reference/README.md`):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest hw/rtl/cluster/tb/sim/test_full_models_cursor/yolo320/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_stem_byte_exact/command-23.snapshot \
  --output hw/rtl/cluster/tb/sim/test_yolo_neural_reference_24_75_firmware_restore
```

Choose a fresh output directory. The snapshot directory must contain its
`byte_exact.json` provenance and `input.bin`. Preparation verifies model/firmware
hashes, source identity, snapshot integrity, restart-safe boundaries, and all
saved initial checkpoints against freshly computed reference tensors. It then
derives checkpoint addresses and rows from commands, saves expected binaries,
and writes `case.json` with artifact hashes and reference build provenance.

Supported intermediate mapping is limited to the initial batch-one, C<=32
Conv/SiLU cascade, including rolling producer storage and async jobs with
explicit waits. Async consumer outputs are compared at WAIT completion, not at
submission. Partial outputs and multi-group mappings are rejected instead of
guessed. Disjoint later-layer LUT work can overlap an async consumer, but is not
silently counted as a verified cascade tensor.

## Checkpoint-first workflow

Verify numerical checkpoints before enabling full-model regression. A checkpoint
PASS means zero byte differences, not merely successful command completion.
Add `--fail-fast` to stop at the first mismatching intermediate checkpoint.
The report is saved as `FAIL` with `failed_command`, complete tensor/row/address
mapping, mismatch statistics and actual bytes before the simulator aborts;
`failure-tcdm.bin` preserves the diagnostic memory. No end snapshot is promised
when a gate stops early. A passing individual check does not mark the entire
case PASS: firmware success and all requested checks are still required.

For the pinned 2026-10-12 YOLO NAI, gates through commands 50 and 107 have
passed byte-exact. The next small gate adds checks at 118, 121, 127 and 157:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --consumer-tiles 5 --fail-fast --timeout-cycles 2000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-157
```

This prepares 13 comparisons totaling 261,120 active bytes, ending at command
157. It does not certify later layers. Resume from memory snapshots only when
both the stored data and the required hardware configuration restoration are
verified. In particular, snapshot 107 omits a LUT reused by subsequent commands;
use reset replay for this gate. Advance to subsequent bounded gates only after
this gate passes, then run full regression after all numerical gates pass.

### Run the complete supported checkpoint list in one pass

Use `--all-cascade-checkpoints` to derive every tile through completion of the
initial consumer tensor. Omit `--fail-fast` when collecting all comparisons and
checking the result after the run; any mismatch still makes the final verdict
FAIL. Firmware errors or timeout remain ERROR, never numerical PASS.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --all-cascade-checkpoints --timeout-cycles 4000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-all-cascade
```

For the pinned 2026-10-12 artifact this derives 33 checkpoints through command
399: all 25,600 stem SiLU rows (409,600 bytes), and all 6,400 consumer Conv rows
(204,800 bytes). Coverage validation rejects missing, duplicated or overlapping
logical rows. The last Conv submits at 363 and completes at WAIT 399;
`submit_command` identifies where the pre-compute TCDM dump was captured.
This batch supersedes separate 211/260/262/311/313/360 runs: earlier checkpoints
are collected while the same simulation advances, without repeating boot.

The scope is the complete initial cascade, **not every YOLO layer**. Later
layer/AFU/DFL tensors still require their own mappings before full numerical
regression can be called covered. Actual bytes, expected hashes and mismatch
statistics are retained for every listed checkpoint, with command PMU CSV.

### After recompiling the model

A different NAI hash invalidates previous snapshots. Pin the new `.nai` and
unchanged compatible `.elf` next to a manifest containing their SHA-256 values
under `.nai` and `.elf`, plus an `environment` object. Then start from reset:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_neural_reference_fresh_20261008/artifacts/dma_events.json \
  --from-reset --consumer-tiles 1 \
  --output hw/rtl/cluster/tb/sim/test_yolo_neural_reference_fresh_20261008/run
```

`--from-reset` and `--snapshot` are mutually exclusive. Reset mode derives
buffer sizes from the model bindings, zero-initializes TCDM/temporary/output
memory and saves deterministic input bytes `(index * 37 + 11) & 255`.
It accepts exactly one input, output and temporary binding, each index zero.
Input/output each have a 1 MiB L2 window; temporary has 14 MiB up to the model
base at `0x81000000`. Control data is placed above the encoded model, including
packages larger than 4 MiB.
No snapshot or AFU restoration preamble is used. Commands begin at 1 and
checkpoint numbers are derived from the new ABI, not copied from an older run.
`--consumer-tiles` defaults to 2; use 1 for a shorter first-consumer diagnostic.
Unsupported mappings fail explicitly.

For the pinned 2026-10-08 artifact this selects commands 1–56, with SiLU checks
at 15/25 (35840 active bytes each) and consumer Conv at 56 (17920 active bytes).
These preparation results are not numerical PASS; the simulation must compare
the actual outputs. Old-model command numbers below apply only to snapshot mode.

## Replay

Use the normal cluster cocotb/Verilator environment, adding
`sw/test/compiler_runtime`, `hw/rtl/cluster/tb`, and
`hw/rtl/cluster/tb/tests` to `PYTHONPATH`. Select:

```bash
export COCOTB_TEST_MODULES=test_yolo_reference_replay
export COCOTB_TESTCASE=test_yolo_reference_replay
export YOLO_REFERENCE_CASE=/absolute/path/to/output/case.json
export NAI_REPLAY_MANIFEST=/absolute/path/to/yolo320/dma_events.json
```

Keep `NAI_REPLAY_FIRMWARE_ELF`, `NAI_DMA_TRACE_CSV`, and
`NAI_PMU_COMMAND_FILTER` unset. Run the cluster simulator with log and results
paths inside the case directory. The test uses the pinned firmware and original
selected command payloads. Runtime timeout defaults to 1,000,000 cycles for
bounded mode (50,000,000 for full graph), configurable with `--timeout-cycles`.
The cluster clock is 1 ns.

Memory snapshots do not preserve AFU LUT SRAM or requant registers. For this
case the harness repeats the immediately preceding original LUT-load command
through the pinned firmware. Host AXI only exposes boot ITCM, PMU and command
control; it cannot program AFU MMIO directly. Offline checks require unflagged,
aligned, disjoint TCDM source/destination ranges outside firmware scratch, and
prove that the complete saved destination already equals LUT(source). The
runtime checks this destination again after the preamble, including C32 padding.
Command flags and payloads are unchanged. A requant-load command must precede
the first replay Conv. Other AFU residency modes and non-immediate restoration
commands are deliberately unsupported.

## Inspect results

For the pinned command-23 snapshot, command 23 is replayed once as restoration
setup, followed by original commands 24-75. Setup cycles are reported separately:

| Command | Reference tensor | Logical rows | Compared bytes |
| --- | --- | --- | --- |
| 25 | Stem SiLU, 159 | 3200-6399 | 51200 |
| 70 | Consumer Conv, 161 | 0-799 | 25600 |
| 75 | Consumer Conv, 161 | 800-1599 | 25600 |

- `byte_exact.json`: running/PASS/FAIL and mismatch counts, first difference,
  maximum absolute difference. PASS requires firmware success and zero byte
  differences at every listed checkpoint; it does not certify unlisted tensors.
- `command-N-expected.bin` and `command-N-actual.bin`: active INT8 channels in
  logical row order, with C32 padding removed.
- `command_pmu.csv`: completed commands using original logical command numbers.
- `restore_pmu.csv`, `restore-actual.bin`: command-23 setup cost and physical
  destination data; `restore_check` in the report must show zero differences.
- `before-command-70-tcdm.bin`, `before-command-75-tcdm.bin`: pre-compute memory
  for diagnosing input, weights, and in-place output overwrites.
- `command-75.snapshot`: complete memory snapshot saved after normal runtime
  completion, even when numerical comparisons fail. Inspect the report before
  using it as a known-good reference. Timeout does not promise an end snapshot.

A result file is never overwritten. For reruns, prepare a fresh case directory.
Do not treat restored-memory readback or firmware completion alone as byte-exact
success.

The earlier `test_yolo_neural_reference_24_75_lut` attempt stalled on its first
host write to AFU address `0x20003000`, before executing any model command. Its
empty checkpoint list is not a numerical failure. Preserve that log for
diagnostics and use a newly prepared firmware-restoration case.

## Offline regressions

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=sw/test/compiler_runtime:hw/rtl/cluster/tb:hw/rtl/cluster/tb/tests \
python3 -m unittest test_yolo_reference_helpers test_snapshot_boundary \
  test_yolo_byte_exact_helpers -v
```

These tests validate mapping, LUT restoration planning, artifact hashes, and
snapshot helpers. They do not replace the Verilator replay.
