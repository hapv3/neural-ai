# YOLO Neural-AI reference replay

This bounded diagnostic starts from reset or resumes a pinned first-cascade memory snapshot and
checks intermediate INT8 tensors against the Neural-AI hybrid TFLite runtime.
It is not a full-model correctness test or a general snapshot-state restorer.
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

Supported mapping is limited to the initial batch-one, C<=32 Conv/SiLU cascade
and the first two contiguous tiles of its Conv consumer. Async/partial or
multi-group mappings are rejected instead of guessed.

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
It accepts exactly one input, output and temporary binding, each index zero and
at most 1 MiB, matching the diagnostic's non-overlapping L2 address windows.
No snapshot or AFU restoration preamble is used. Commands begin at 1 and
checkpoint numbers are derived from the new ABI, not copied from an older run.
`--consumer-tiles` defaults to 2; use 1 for a shorter first-consumer diagnostic.
Mapping still rejects a request that crosses a producer restart.

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
selected commands, with a 1,000,000-cycle runtime timeout and a whole-test
1,200,000 ns watchdog (1 ns clock), including setup and firmware loading.

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
