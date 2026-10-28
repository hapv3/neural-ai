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
