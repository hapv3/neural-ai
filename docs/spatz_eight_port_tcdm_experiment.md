# Eight-port Spatz and native mixed-width TCDM experiment

## Scope and isolation

Parent repository branch: `experiment/spatz8-tcdm64x32-native`, based on
`c315ed7`. Worktree: `/home/dev01/neural-ai-br`. The original neural-ai
worktree, its pending firmware optimizations, and its running simulation are
not modified. Spatz-only LMUL counter work is isolated on the submodule branch
`experiment/neural-ai-eight-port`; commit its verified change before committing
the parent gitlink. No results are claimed before the regression passes.

| Property | Baseline | Candidate |
|---|---|---|
| Integer vector lanes / native VLSU ports | 2 / 2 × 32 bits | 8 / 8 × 32 bits |
| VLEN / ELEN / floating point | 512 / 32 / none | unchanged |
| Shared SRAM | 16 × 32 KiB × 256 bits | 64 × 8 KiB × 32 bits |
| Shared capacity | 512 KiB | unchanged |
| Theoretical aggregate bank throughput | 512 bytes/cycle | 256 bytes/cycle |
| Snitch and Spatz shared path | narrow-to-wide adapters | direct 32-bit requests |
| DMA, AFU and systolic ports | 256 bits | native 256-bit gang requests |
| Shared master count | 14 | 20 |

Increasing the number of Spatz ports also increases integer execution lanes:
this Spatz version requires `NrMemPorts == NFU` for the normal VLSU. It is not
an isolated port-count experiment. VRF capacity remains 32 × 512 bits, but its
datapath becomes 256 bits. Private ITCM/DTCM, AXI width, memory bindings,
command ABI and model/firmware artifacts are unchanged.

## Interconnect

`mixed_tcdm_interconnect.sv` exposes per-master widths and directly maps native
32-bit words to SRAM banks. Upper zeros in a maximum-width packed signal are
wiring only: there is no upsizer, transaction serialization or lane-shifting
adapter on scalar/vector ports. The generic router also accepts 64/128-bit
masters, exercised by the block gate; the current cluster uses 32 and 256 bits.

For byte address `a`, bank = `(a >> 2) & 63`, row = `(a >> 8) & 2047`.
Each master aligns to its own bus width, matching the old aligned-bus semantics.
Logical snapshots remain 512-KiB byte arrays, independent of physical banking.
The previous address aliases are retained, not silently redesigned.

Eight adjacent 32-bit banks form a 256-bit arbitration group. A group guide
arbitrates a wide transaction against narrow requesters, then independent
bank arbiters grant narrow requests on remaining banks. All needed banks of a
wide beat are granted atomically, with one fixed-cycle response; no partial
acceptance, rollback, response queue or stalled-SRAM protocol is introduced.
Sparse writes occupy only their strobed banks, zero-byte stores acknowledge
as no-ops, and reads ignore write strobes. The SRAM contract remains immediate
acceptance, synchronous response, one read **or** write per bank per cycle.

Priority remains HWPE > DMA > scalar, round-robin within each class. Ports
1, 8, 14–19 belong to Spatz; other engine master IDs remain stable. Strict
priority does not guarantee low-priority progress under endless HWPE traffic.
Changed grouping and removal of bridge bubbles are part of the experiment.
No synthesis, area or 1-GHz timing closure is asserted by lint/simulation.

The pinned Spatz revision already has eight-lane reduction iterations. The
VRF word index is widened to cover LMUL=8, matching upstream spatz_vpu fix
`235c830982391ced0b63baa42915eba711b703fb`. No wholesale upstream migration is
part of this change.

## Verification and full YOLO A/B

All results live under:

`hw/rtl/cluster/tb/sim/spatz8_tcdm64x32_20261027/`

The baseline `baseline_model/Vtop` was archived **before** editing RTL, from
`c315ed7`, with its build log, source stamp and git identity. Its preceding
boot gate passed an independent 64-byte comparison. Do not rebuild the
baseline binary from candidate sources. The experimental binary is built in
`candidate_model/`; sources/stamps and both binary hashes are pinned before run.

Compile-only examples, from the repository root:

```bash
make -j10 -C hw/rtl/cluster \
  "$PWD/hw/rtl/cluster/tb/sim/spatz8_tcdm64x32_20261027/candidate_model/Vtop" \
  CLUSTER_SIM_NAME=spatz8_tcdm64x32_20261027/candidate_model VERILATOR_JOBS=10

verilator --binary --timing --assert -sv -j 10 -Wno-fatal -Wno-WIDTH \
  --top-module tb_mixed_tcdm_interconnect \
  --Mdir hw/rtl/cluster/tb/sim/spatz8_tcdm64x32_20261027/unit \
  hw/common_cells/src/cf_math_pkg.sv hw/common_cells/src/lzc.sv \
  hw/common_cells/src/rr_arb_tree.sv \
  hw/rtl/interconnect/mixed_tcdm_interconnect.sv \
  hw/rtl/interconnect/tb_mixed_tcdm_interconnect.sv \
  +incdir+hw/common_cells/include

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=hw/rtl/cluster/tb/tests \
  python3 -m unittest test_tcdm_geometry_host -v
```

The unit binary is compiled with assertions enabled. Build assembly gates with
`make -j10 -C sw/test/spatz_vector SPATZ_INSTALL=<installed-toolchain>`.
Build `sw/test/conv3x3_multi_linebuf` and the `spatz_ops_{copy,relu,requant,dfl16_pack}.bin`
targets in `sw/test/spatz_ops` with the same toolchain; Clang `-ffreestanding`
avoids incompatible hosted stdint typedefs from the old GCC installation.

Run the following gates **individually**, checking their results before starting
the next one. The early workflow helpers are local historical tools, not the
current execution method.

1. Mixed-width block scoreboard: all 32/64/128/256-bit widths, eight simultaneous
   adjacent narrow ports, atomic wide grants, sparse/no-op writes, read strobes,
   unaligned bus words, bank/row boundaries, back-to-back responses, random
   contention, round-robin progress, traffic-class priority and reset.
2. Basic/strided/indexed/m8 RVV memory arithmetic with output checks.
3. Memory widths, unmasked arithmetic, reduction, and LMUL=8 full/tail transfers
   with independent host byte comparison. Keep the failing masked diagnostic
   separate; it is not a supported gate or counted as PASS.
4. Spatz copy/ReLU/requant/DFL packing operator gates.
5. C64/OC64 Conv3×3 split/fused linebuffer gate, both outputs checked.
6. Full YOLO candidate and baseline in separate runs, each reset-to-end and
   byte-exact. No automatic transition between model runs.
7. Only after both PASS: compare total/engine/state/phase and per-command cycles.

After preparing and hash-validating the pinned case, run the candidate directly
from its own output directory, without a workflow script:

```bash
repo=/home/dev01/neural-ai-br
exp="$repo/hw/rtl/cluster/tb/sim/spatz8_tcdm64x32_20261027"
cd "$exp/full-yolo/candidate"
env -u COCOTB_TESTCASE -u NAI_REPLAY_FIRMWARE_ELF \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$repo/sw/test/compiler_runtime:$repo/hw/rtl/cluster/tb/tests:$repo/hw/rtl/cluster/tb" \
  PYGPI_PYTHON_BIN=/usr/bin/python3 LIBPYTHON_LOC="$(cocotb-config --libpython)" \
  COCOTB_TEST_MODULES=test_model_reference_replay \
  COCOTB_TEST_FILTER='[.]test_model_reference_replay$' \
  COCOTB_TOPLEVEL=tb_npu_cluster TOPLEVEL_LANG=verilog \
  COCOTB_RESULTS_FILE="$PWD/results.xml" \
  MODEL_REFERENCE_CASE="$PWD/case.json" \
  NAI_REPLAY_MANIFEST=/home/dev01/neural-ai/hw/rtl/cluster/tb/sim/test_dog_models_20261022_restart1/yolo/artifacts/dma_events.json \
  "$exp/retry2/candidate_model/Vtop" > sim.log 2>&1
```

Both variants execute the **same 5,520 commands**, same archived ELF, same dog
input, same expected 176,400 output bytes, 1-ns clock and 50-million-cycle timeout.
No compiler changes or model regeneration confound the comparison. The case's
existing intermediate snapshot boundaries are retained, now using geometry-aware
logical memory dumps. An intermediate snapshot's integrity does not by itself
prove all its bytes correct; full output byte-exact is the final numerical gate.

Inspect `unit/sim.log`, the per-test directories under `gates/`, and
`full-yolo/candidate/` or the separately prepared baseline directory. Each full run emits
`command_pmu.csv`, `byte_exact.json`, `output-actual.bin`, and snapshots.
The final `comparison.json` and `command-comparison.csv` are produced only on
complete numerical PASS, during a separate analysis step. A missing final
comparison means incomplete/failed, not a speedup. No simulator or subsequent
test is started automatically after any test ends.

Use total measured cycles, instruction counts, engine active/stalled cycles,
and per-command deltas. Separate Spatz gains from systolic/AFU/DMA regressions.
Raw bank-operation counts have different widths and must not be compared as
bytes; legacy PMU read-byte counters also depend on read strobes. Timing in ms
at 1 GHz is a normalization, not a validated achievable clock. More banks but
half the raw aggregate SRAM width can improve narrow concurrency and still
regress wide-engine throughput; measure rather than assume a speedup.

## Initial gate evidence and verifier correction

The first run passed the mixed-width block scoreboard (2,682 cycles) and
the cluster basic/strided/indexed/m8 RVV gate. Memory-width verification also
passed. The arithmetic/mask gate then failed at index 16, beyond its VL=16.
Inspection found that all four scalar verifier functions called `value_a` /
`value_b` using `jal ra` without preserving their own return address. Their
final `ret` returned into the verifier loop, not to the caller. Save/restore
`ra` on the stack fixes this test-firmware defect; no RTL change is made for it.
The host now independently checks all memory-width and arithmetic/mask outputs,
requires the exact pass count, and clears stale signatures between firmware runs.

Original evidence is retained without overwriting. The continuation is in
`hw/rtl/cluster/tb/sim/spatz8_tcdm64x32_20261027/retry1/`. Its local
`run_remaining.py` reuses the two already-passed gates only after matching
simulator binary hashes, RTL source hashes, the relevant Python helpers and
basic/strided firmware hashes. `reused-gates.json` records this provenance.
The corrected eight-port gate must pass before operator/Conv gates and full
YOLO A/B can proceed. This is not a claim that the remaining gates have passed.

Retry1 was stopped before firmware execution: the new stale-signature clear
incorrectly used host AXI to address private DTCM, which is not in that host
window. The harness now uses the existing ELF-loader DTCM backdoor with fetch
disabled, and bounds firmware loading, fetch release and the complete gate with
timeouts. No additional RTL change is involved. Retry2 preserves both earlier
attempts and retains the same unchanged unit/basic gate evidence.

The standalone Retry2 gate passed memory widths, then encountered an illegal
`vmerge.vvm` at PC `0x10000134` (opcode `0x5c1101d7`). Spatz decodes VMERGE but
Snitch does not forward it in this configuration. Per the chosen compatibility
policy, do not extend RTL: the arithmetic test now checks compare plus masked
`vadd.vi`, preserving the max(a,b) expected output. The LMUL test also replaces
unsupported `vid.v` with scalar-seeded input plus `vle8.v`. Both changes retain
mask/tail and independent output checks; they do not claim VMERGE/VID support.

The standalone Retry3 had no illegal instruction, but the masked update failed:
lane 0 retained -24 instead of -7. The integer decoder accepts VMSLT/VMSEQ, but
the pinned SIMD lane has no corresponding result cases (its default result is
zero); decoder membership alone is insufficient evidence of execution support.
The tests now load packed even-lane masks directly from scalar-seeded SRAM,
and exercise masked arithmetic independently of those unimplemented compare
operations. Arithmetic-mask expected output is b on even lanes, a on odd lanes;
the 512-element LMUL expected output is unchanged. This does not claim integer
comparison support or alter RTL. Failed results remain retained.

The seeded-mask diagnostic also failed on both hardware configurations:
candidate lane 9 (1 instead of 3), baseline lane 15 (10 instead of 21). Baseline
output evidence confirms the four unmasked arithmetic vectors match exactly;
the failing last lane retains the earlier shift result. This establishes an
existing masked-destination problem, not that all new-topology behavior is
equivalent. Its detailed RTL cause remains unresolved and no fix is claimed.

Following the compatibility policy, `arith_mask` remains an explicitly failing
diagnostic, selectable with `SPATZ_GATE_CASE=arith_mask`, not a passed gate.
`arith_basic` is a distinct four-operation unmasked gate with scalar and host
golden checks. The default instruction suite uses it instead. The LMUL=8 gate
now verifies unmasked 512-byte load/add/store and a 509-byte tail overwrite;
its last three bytes must remain from the full store. Mask comparison, merge,
VID and reliable masked writeback support are not claimed by these gates.

The standalone `retry6/arith-basic` gate passed: firmware signature
`0xDEADBEEF`, four completed arithmetic checks, and all 64 result words
(256 bytes) independently exact in `arith-outputs.json`. Its simulation time
was 4,579 ns including reset/AXI loading, not an isolated arithmetic benchmark.
Subsequent gates are launched individually, without the workflow runner; the
next gate is `gates/reduction`, checking the first reduced word equals 136.

The isolated reduction gate passed with scalar signature/pass count and host
verification of sum=136. The next standalone gate is `gates/lmul8-tail`:
firmware checks all 512 full-store bytes before the 509-byte overwrite, then
checks the complete final buffer including its three preserved tail bytes.
Both firmware checks must pass; the host independently compares the final
512 bytes and saves binary outputs plus `output-check.json`.

`gates/lmul8-tail` passed both firmware checks and the host's full 512-byte
comparison with zero mismatches; preserved tail bytes are `[4, 5, 6]`.
`gates/op-copy` (`test_spatz_op_copy`) passed its firmware check and independent
32-byte host comparison. The small test's PMU count is 537 cycles, including
firmware work, not an isolated copy-kernel throughput measurement.
`gates/op-relu` (`test_spatz_op_relu`) also passed its firmware and independent
32-byte host checks (small-test PMU: 207 cycles). `gates/op-requant` passed its
firmware and 32-byte host checks (small-test PMU: 941 cycles).
`gates/op-dfl16-pack` passed its CHW/C32 full/tail firmware checks and independent
1,024-byte final C32 host comparison, including retained records and channel
guards (small-test PMU: 76,889 cycles, including fixture/verification work).
The next standalone gate is `gates/conv-linebuffer`, comparing each 3,072-byte
Conv3x3 C64/OC64 split/fused output against its host golden.

`gates/conv-linebuffer` passed both complete 3,072-byte output comparisons.
Both firmware paths currently use the safe C32-group schedule and each measured
28,583 PMU cycles; this is not evidence of a distinct fused-schedule speedup.
Firmware-exported detailed stats were zero, so they are not used for performance
claims. The RTL PMU report is available in the log. All required functional gates
for this unmasked experiment are now passed; full-model correctness/performance
remain pending until the separate full YOLO runs complete.

## Firmware integration gate — audited 2026-11-02

After importing the affine-copy, long-row and guarded-prefetch firmware from
main and merging `origin/main`, the eight-port `test_spatz_op_copy_batch` gate
passed all 20 firmware byte-oracle/canary cases. Evidence is under
`hw/rtl/cluster/tb/sim/test_yolo_latest_compiler_20261030/gates-copy-batch-main-merge/`.
The test completed on 2026-10-30; its XML and all 20 timing rows were audited on
2026-11-02. Other XML tests are explicitly skipped by the selected test filter.

Overall PMU cycles were 4,557,519, including scalar fixture initialization and
in-firmware validation. The sum of the separately bracketed call-side timings
was 84,488 legacy cycles versus 11,694 batched cycles (7.225x for this fixture,
not full-model speedup). Cases 8, 17 and 18 regressed: 2,114 -> 2,336,
280 -> 334 and 512 -> 588 cycles. No numerical tolerance was relaxed.

This run loaded the older reporter, so its aggregate PMU is retained in
`sim.log`, not a reconstructed 163-counter JSON. Subsequent operator runs use
the complete JSON/CSV exporter introduced in `1996df5`. Standalone firmware
does not open ABI command contexts: `spatz_active_cycles` can therefore remain
zero despite nonzero issue/response and TCDM traffic. Do not interpret that zero
as an inactive VPU or use it as a utilization denominator for these gates.

Next: verify long 1D copy, then stream prefetch on eight ports, before running
the prepared 5,428-command full-reset YOLO with the updated firmware. These
remaining gates are separate simulations, not automatically chained runs.

### Long 1D copy — verified 2026-11-02

`test_spatz_op_copy_1d` passed all 12 firmware checks and the host's full-memory
payload/canary comparison. Eleven timing records cover lengths 0 through
51,200 bytes, independent source/destination byte alignment and VL tails. The
additional check preserves legacy self/overlap behavior, not a memmove contract.
Evidence: `test_yolo_latest_compiler_20261030/gates-copy-1d/` below the simulation
root used above.

Aggregate PMU cycles were 44,099. All 163 counters and metadata match exactly
between `test_spatz_op_copy_1d_pmu.json` and the single-row CSV, with zero
overflow. The exporter reports PASS only after the host comparison succeeds.
The 51,200-byte case (source alignment 2, destination alignment 3) measured
19,996 legacy versus 13,562 batched call-side cycles. Cases of 0, 31 and 255
bytes each cost eight extra dispatch cycles. These are fixture timings, not
full-model latency. Next required gate: `test_spatz_op_stream_prefetch`.

### Stream prefetch — verified 2026-11-02

`test_spatz_op_stream_prefetch` passed all four firmware checks and the host's
full-memory payload/canary comparison on the eight-port candidate. Evidence is
in `test_yolo_latest_compiler_20261030/gates-stream-prefetch/`. All 163 PMU
counters and metadata match between CSV and JSON, with zero overflow and
769,557 aggregate cycles. This includes scalar reader/control work and is not
an isolated DMA-bandwidth measurement.

The 1/2/4 KiB slot cases submitted/joined 14/14, 9/9 and 7/7 private jobs;
early finish joined both outstanding jobs (2/2). Cases include 4 KiB boundary
crossings and interleaved data DMA. Together with copy-batch and copy-1D, this
closes the focused firmware gates before the updated full-model regression.
The full run retains the already compiled 5,428-command NAI, input and deployed
AFU oracle; it does not silently recompile using a subsequently advanced main.

## Updated-firmware full YOLO — verified 2026-11-04

The full-reset run completed on 2026-11-03 and its artifacts were independently
audited on 2026-11-04. Evidence, relative to the experiment simulation root:

`o3_20261102/paired-rerun1/os/`

This is the eight-port candidate using the archived GCC `-Os` simulator, not
the original two-port hardware baseline. It retains the NAI produced by compiler
commit `3d6a1e2418635c306c7f34460447989bc87ffe14`; no compilation from a later
compiler HEAD is implied. The integrated firmware includes the separately
verified copy-batch, long-copy and guarded command-prefetch changes.

| Pinned artifact | SHA-256 |
|---|---|
| Simulator | `64f195052e737f188bebb28b8b25aa877e91599c7e8066a24f33cad15c6bed4a` |
| NAI | `ecde7fa438d9e9f0a45cab510d0dca3748d836442d073a341731af656a2c1c68` |
| Firmware ELF | `de037cedaa8127ab08d4008d8ac7b7aa566bef714cf6555783c22508ce2bc989` |
| Actual and expected output | `17ddd14ec29fec3e1e1705f4e1241adc50fe05cb02c6137534336742465f5962` |

Verification results:

- Cocotb passed its single selected test; all **5,428 commands** completed.
- All **176,400 INT8 output bytes** match the pinned Neural-AI MAC/AFU DFL
  reference: zero mismatches, MAE=0, max_abs=0. This is one pinned input and the
  deployed arithmetic contract, not an unmodified-TFLite or dataset-accuracy claim.
- All 16 intermediate snapshots plus the final command-5428 snapshot passed
  format/identity/integrity checks. The final snapshot output also equals the
  actual output and oracle. Intermediate snapshots have no numerical-oracle
  observations; integrity alone is not an intermediate tensor correctness PASS.
- `command_pmu.csv` contains 5,428 dense, ordered command rows and 169 columns:
  six identity fields plus all 163 raw counters. Global overflow status is zero.
- `verification-audit-20261104.json` records output, CSV, XML and snapshot hashes
  and the independent checks. Logs, snapshots and raw reports remain local.

| Measured counter | Cycles |
|---|---:|
| Total firmware/command execution PMU window | 13,183,595 |
| Sum of per-command windows / command_active_cycles | 10,732,584 |
| Systolic active | 4,047,055 |
| Spatz active | 2,266,939 |
| AFU active | 1,309,885 |
| Command engine idle | 3,463,898 |

Engine/state counters can overlap and must not be added as exclusive runtime
shares. The per-command windows do not cover all global-window gaps; command
engine idle is not interchangeable with time outside command contexts.
Cocotb host wall time was 93,700.316 seconds (about 26 h 1 min 40 s), including
simulation and host-side verification, not physical inference latency.

The earlier candidate run in `full-yolo/candidate/` used the **same simulator**,
source model, input and final-output oracle, but different NAI/firmware. Its
5,520 commands took 18,365,775 cycles. The updated software/artifact combination
therefore reduces total cycles by **28.22%**, to 13,183,595. This is not an
isolated compiler-only, firmware-only, hardware-port-count or host-optimization
speedup. The earlier two-port/eight-port A/B remains a separate experiment.

### Host O3 experiment disposition

The paired GCC O3 run was explicitly stopped on 2026-11-03 after 2,991 commands,
without a final inference verdict. Its 2,991 PMU rows match the completed Os run
across all 169 columns, and all seven shared snapshots are byte-identical.
`o3_20261102/paired-rerun1/o3/simulation-stop.json` records the user-requested
SIGTERM; retain the interrupted results and do not count them as a full PASS.

At shared checkpoint 2824, O3 required 38.36% more elapsed host time. That is a
partial, concurrent host measurement, not extra simulated NPU cycles or a
full-model O3 result. The local `host-profile-20261103.json` and
`host-optimization-analysis-20261103.md` contain the non-stopping CPU profile,
code-size/disassembly evidence and phase/confounding limitations. No O2 build
or replacement simulation was started; the user-stopped O3 run stays stopped.

## Fresh latest-compiler full YOLO — verified 2026-11-05

The requested fresh compilation used compiler commit
`7a9f4f5b63a1c705a67ca66030ce5ce4c463107b`. The Release Python backend was
rebuilt with 10 jobs before compiling the pinned YOLO320 source; its SHA-256 is
`39bcb9768127a76c1acdb25cd88530c4ab8c8f8f401ae4fc482feb8454a5ed74`.
The newly generated NAI is byte-identical to the prior `ecde7fa4...` package:
5,428 logical commands. This records a new compilation, not relabeled old
compiler provenance. The independent source/input reference was also recomputed
and matched the existing pinned oracle before simulation.

The full reset run retains the verified GCC Os simulator and `de037ced...`
firmware above. It does **not** merge the newer main-repository CPU FETCH
firmware, change RTL or restart O3. Evidence under cluster `tb/sim`:

`test_yolo_latest_compiler_20261104/full_yolo/`

- Cocotb PASS and simulator exit 0; all 5,428 command descriptors match the
  dense, ordered 169-column PMU CSV.
- All 176,400 final output bytes are exact against the pinned Neural-AI
  MAC/deployed-AFU-DFL reference; zero mismatches, MAE=0, max_abs=0.
- Total PMU cycles remain **13,183,595**; command windows sum to **10,732,584**.
  All 163 global counters, PMU metadata and every command CSV row match the
  previous completed Os baseline exactly. Counter overflow remains zero.
- All 16 intermediate snapshots and the final command-5428 snapshot pass
  format, model identity, size and payload CRC checks, and are byte-identical
  to the corresponding baseline snapshots. Final snapshot output equals the
  actual output and oracle. Intermediate independent tensor-oracle comparisons
  remain zero; baseline equality is not all-layer TensorFlow verification.
- `verification-audit-20261105.json` records the independent audit and artifact
  hashes. Logs, CSV, snapshots, reference and compiled artifacts stay local.

Host elapsed time was 22:03:51. Different host load prevents treating its
difference from the earlier run as an isolated optimization gain. Simulated
NPU cycles are unchanged, as expected for unchanged NAI, firmware and RTL.

## Main merge: full YOLO and MobileNet — verified 2026-11-09

Merge commit `c775b1e6a69616d17078d433510d6f9683e0aada` integrates main's CPU
FETCH firmware into the eight-port branch without changing RTL. Both models
were freshly compiled with compiler `3bd1bbc683e49fb83eacfed5ca6d6356644629b6`.
The rebuilt firmware SHA-256 is
`90e85d25080bb81d51719868e2029921a57114334ddba1a77b9fc62aba7598a6`.
The simulator remains the verified GCC Os binary `64f19505...`; no O3 run was
restarted. Source, input and independent reference artifacts were pinned before
the two full-reset runs were launched with 50M-cycle timeouts.

Evidence is under cluster `tb/sim/test_main_merge_models_20261105/`, in separate
`yolo/` and `mobilenet/` directories. Both cocotb tests pass and simulator exit
codes are zero. Independent output comparisons and descriptor/PMU audits give:

| Full-reset result | YOLO320 | MobileNet224 |
|---|---:|---:|
| Commands completed and audited | 5,428 | 12,784 |
| Output bytes compared | 176,400 | 1,000 |
| Mismatches / max_abs | 0 / 0 | 0 / 0 |
| Total PMU cycles | 12,479,475 | 14,585,749 |
| Command execute windows | 10,748,490 | 12,024,562 |
| FETCH cycles | 1,689,244 | 2,519,446 |
| Command CSV rows / columns | 5,428 / 169 | 12,784 / 169 |
| Counter overflow | 0 | 0 |

YOLO retains NAI `ecde7fa4...` and the Neural-AI MAC/deployed-AFU-DFL reference;
MobileNet uses newly generated NAI
`6c4d58c6388a03f8db2a03eec320c3d96a392412c02db7705c062b77ddb4c94c` and the
Neural-AI MAC/AFU-Mean reference. This is final-output numerical coverage for
one pinned input per model, not unmodified-TFLite or dataset-wide accuracy.

All 16 intermediate snapshots plus the final snapshot of each model pass
format, identity, size, payload CRC and model-command restart-boundary checks.
Final snapshot outputs equal actual outputs and their respective oracles.
Intermediate independent tensor-oracle comparison counts are zero; snapshot
integrity alone does not establish per-layer numerical correctness. Private
command-reader jobs remain outside the model-command restart guarantee.

The prior eight-port YOLO run used identical NAI/source/input/oracle but older
firmware: 13,183,595 -> 12,479,475 cycles, **704,120 saved / 5.34% reduction**.
FETCH saves 720,026 cycles while execute windows increase 15,906; other phases
are unchanged, reconciling the invocation saving exactly. Systolic useful
cycles, AFU active cycles and global AXI bytes are unchanged.

For a separate hardware-configuration comparison, the verified main-repository
MobileNet run in `test_mobilenet_recompile_20261105/full_mobilenet/` uses the
same NAI, firmware, source, input and oracle: 16,364,403 -> 14,585,749 cycles,
**1,778,654 saved / 10.87% reduction**. FETCH is identical; the reduction is
entirely in execute windows. This compares the complete original-main and
eight-port/64-bank mixed-width configurations, not Spatz port count in isolation.

Each model's `verification-audit-20261109.json` records the independent checks,
artifact hashes, raw PMU and scoped baseline comparison. Command-type CSVs and
`verification-summary-20261109.json` are archived beside the runs and stay local.
Host wall times were 23:01:54 (YOLO) and 25:44:59 (MobileNet), with concurrent
simulations; these are not isolated host benchmarks or inference latency.
