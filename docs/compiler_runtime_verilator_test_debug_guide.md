# Compiler-runtime Verilator test and debug guide

This guide describes how to build, run, and debug Neural-AI tests that cross
the Regor compiler, the `.nai` ABI, trusted firmware, and RTL. It concentrates
on the compiler-runtime tests and the selected YOLO320 model. General firmware
test inventory remains in [test_flow.md](test_flow.md).

## 1. Repository layout and test layers

The examples assume these paths:

```text
/home/dev01/neural-compiler   Regor compiler and Vela frontend
/home/dev01/neural-ai         firmware, runtime, testbench, and RTL
```

Use the cheapest layer that can expose the suspected defect:

| Layer | What it verifies | Typical runtime |
|---|---|---|
| Regor unit test | graph constraints, tiling, schedule selection, serialized fields | seconds |
| Runtime host test | ABI sizes, parser/dispatcher validation, address and command contracts | seconds |
| Firmware cross-build | trusted firmware compiles and fits its memory image | seconds |
| RTL lint | synthesizable syntax, widths, connectivity, combinational warnings | seconds to minutes |
| Systolic block simulation | controller, linebuffer, array, requant, MMIO contract | seconds after build |
| Cluster simulation | real Snitch firmware, DMA, TCDM, Spatz, AFU, systolic, `.nai` | minutes or hours |

Do not use a full-model cluster run to discover a basic operator bug. First
reduce it to a Regor test, host ABI test, or focused block test.

## 2. Build prerequisites

### 2.1 Build the Regor Python extension

The YOLO tests compile the TFLite model at test time and require the in-tree
`ethosu/regor*.so` to match the source.

```bash
cd /home/dev01/neural-compiler
env CMAKE_BUILD_PARALLEL_LEVEL=10 CCACHE_DISABLE=1 \
  python3 setup.py build_ext --inplace
```

`CCACHE_DISABLE=1` avoids failures when the default ccache temporary directory
is read-only. If an editable pip install is preferred and all build
dependencies are already installed:

```bash
env CMAKE_BUILD_PARALLEL_LEVEL=10 CCACHE_DISABLE=1 \
  pip3 install -e . --no-deps --no-build-isolation
```

Avoid build isolation on an offline machine because pip will try to download
build dependencies.

Verify which extension Python imports:

```bash
cd /home/dev01/neural-compiler
python3 -c 'import ethosu.regor; print(ethosu.regor.__file__)'
```

### 2.2 Build trusted firmware

```bash
cd /home/dev01/neural-ai/sw/runtime/neural_ai
make -j10
```

The build produces `neural_ai.elf`, `neural_ai.bin`, and `neural_ai.hex`, then
prints section sizes. Track at least `.text`, `.data`, and `.bss`; an unexpected
`.text` jump usually means validation, formatting, or a fallback loop entered
the trusted firmware image.

Example:

```text
.text  23560
.data     72
.bss    6045
```

The cluster tests use `neural_ai.elf`, not a stale binary copied elsewhere.

## 3. Fast gates before cluster simulation

### 3.1 Regor unit tests

Build with multiple threads:

```bash
cd /home/dev01/neural-compiler
cmake --build build-unit-tests --target unit_tests -j10
```

Run all Neural-AI cases:

```bash
LSAN_OPTIONS=detect_leaks=0 \
  ./build-unit-tests/unit_tests 'Neural-AI*'
```

Run only linebuffer planning cases:

```bash
LSAN_OPTIONS=detect_leaks=0 \
  ./build-unit-tests/unit_tests 'Neural-AI linebuffer planner*'
```

`LSAN_OPTIONS=detect_leaks=0` is only a workaround for environments where
LeakSanitizer cannot attach through ptrace. It does not disable AddressSanitizer
checks for invalid accesses.

### 3.2 Runtime and cross-ABI host tests

```bash
cd /home/dev01/neural-ai/sw/test/compiler_runtime
make -j10 check
```

This gate checks, among other things:

- compiler/runtime ABI constants, enums, and structure sizes;
- model and invocation parsing;
- command validation and region resolution;
- copy-layout and quant-buffer behavior;
- trusted and non-trusted dispatcher contracts.

If the cross-repository manifest fails, do not debug RTL first. Fix the ABI
definition or serialization mismatch.

### 3.3 Source hygiene

Run in each modified repository:

```bash
git diff --check
git status --short
```

`git diff --check` detects whitespace errors. `git status --short` is important
when test diagnostics and production changes are being developed separately.

## 4. Systolic block simulation

The block testbench excludes Snitch firmware and most cluster interconnect. Use
it for linebuffer, KGEN, systolic drain, requant, and shadow-register bugs.

### 4.1 One focused test

```bash
cd /home/dev01/neural-ai/hw/rtl/systolic
make -j10 sim \
  MODULE=test_systolic_controller_linebuf_matrix \
  COCOTB_TEST_FILTER=systolic_controller_linebuf_generic_kgen_consecutive_spatial_tiles
```

### 4.2 Full relevant controller suite

```bash
cd /home/dev01/neural-ai/hw/rtl/systolic
make -j10 sim \
  MODULE=test_systolic_controller_linebuf_matrix,test_systolic_controller
```

Expected summary form:

```text
TESTS=16 PASS=16 FAIL=0 SKIP=0
```

Important block assertions include:

- expected compute and weight-load pulse counts;
- exact output bytes or words;
- all K tiles complete;
- a second spatial job starts without reset;
- `DONE` is not asserted while the linebuffer is still busy.

### 4.3 Interpreting build warnings

Warnings in shared SRAM models or imported libraries are not automatically a
failure. Classify them by source path:

- `tc_sram.sv` `WIDTH`/`UNSIGNED`: commonly pre-existing simulation-model
  warnings;
- imported Spatz/fpnew warnings: record separately unless the modified path
  changes their parameters;
- warnings in a newly edited Neural-AI module: inspect before accepting the
  test, even if `-Wno-fatal` allows the build.

An executable build followed by a Cocotb `PASS` proves behavior for the tested
configuration; it does not make a new timing or synthesis guarantee.

## 5. Cluster simulation commands

Run cluster commands from:

```bash
cd /home/dev01/neural-ai/hw/rtl/cluster
```

The compiler-runtime Python module is outside the cluster test directory, so
add it to `PYTHONPATH`:

```bash
export PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime
```

### 5.1 Normal compiler-runtime test

```bash
make -j10 sim \
  COCOTB_TEST_MODULES=test_compiled_model \
  COCOTB_TEST_FILTER=test_compiler_runtime_dma_package \
  CLUSTER_SIM_NAME=test_compiled_model
```

`CLUSTER_SIM_NAME` selects the reusable Verilator build directory:

```text
hw/rtl/cluster/tb/sim/test_compiled_model/
```

Keeping the same name reuses `Vtop` when RTL sources and generic parameters are
unchanged. Changing only Python test code or environment variables should not
require rebuilding the RTL model.

### 5.2 Selected YOLO320 prefix

```bash
env \
  PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  YOLO320_PREFIX_COMMANDS=22 \
  YOLO320_PREFIX_TIMEOUT_CYCLES=1000000 \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_first_stem_tile \
    CLUSTER_SIM_NAME=test_compiled_model
```

`YOLO320_PREFIX_COMMANDS` keeps commands `1..N`, appends an END command, and
updates command-section metadata. It is useful for binary-searching the first
failing command while preserving all preceding graph state.

Current reference checkpoints are examples, not permanent performance limits:

| Prefix | Measured PMU cycles | Result |
|---:|---:|---|
| 22 | 214,193 | PASS |
| 129 | 937,925 | PASS |
| 131 | 958,417 | PASS |

The 1,000,000-cycle timeout is a debug bound. A larger prefix can exceed it
because of valid work, not because of deadlock.

### 5.3 Segmented prefix continuation

For a model whose valid prefix already consumes most of the timeout, use
segmented continuation:

```bash
env \
  PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  YOLO320_SEGMENT_ENDS=131,141 \
  YOLO320_SEGMENT_TIMEOUT_CYCLES=1000000 \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_segmented_prefix \
    CLUSTER_SIM_NAME=test_compiled_model
```

`YOLO320_SEGMENT_ENDS=131,141` runs:

```text
segment 1: commands 1..131
reset logic while SRAM contents remain intact
segment 2: commands 132..141
```

Each segment gets its own timeout. The next segment is repackaged at the start
of the command section, but tensor references, model constants, bindings, TCDM,
and L2 temporary data keep their original addresses.

Rules for valid segmented testing:

- boundaries are cumulative, strictly increasing command counts;
- the first segment must start at command 1;
- every preceding segment must PASS before the next starts;
- do not run a suffix alone because its intermediate tensors do not exist;
- reset must not clear TCDM or AXI simulation memory;
- segmented completion proves execution continuity, not final numerical
  correctness; a final full-output comparison remains required.

Use a small smoke test after changing segmentation logic:

```bash
env \
  PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  YOLO320_SEGMENT_ENDS=5,6 \
  YOLO320_SEGMENT_TIMEOUT_CYCLES=100000 \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_segmented_prefix \
    CLUSTER_SIM_NAME=test_compiled_model
```

This splits RGB staging/configuration from the first linebuffer job and proves
that the second segment consumes preserved SRAM state. The current smoke test
passes both segments; command 6 completes in 52,081 PMU cycles after reset.

### 5.4 Full selected graph

```bash
env PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_full_graph \
    CLUSTER_SIM_NAME=test_compiled_model
```

The full test compiles all 3,910 commands and compares the public output with
TensorFlow Lite `BUILTIN_REF`. Use it only after focused and segmented gates
are green.

### 5.5 Parallel broad regression

```bash
cd /home/dev01/neural-ai
python3 hw/rtl/cluster/tb/run_cluster_tests.py \
  --build-fw --jobs 4 --tests all
```

Focused examples:

```bash
python3 hw/rtl/cluster/tb/run_cluster_tests.py \
  --jobs 4 --tests test_conv_perf --conv-perf-cases 0-23

python3 hw/rtl/cluster/tb/run_cluster_tests.py \
  --jobs 4 --tests test_depthwise_conv --depthwise-cases 0-6
```

The runner shares a Verilator binary for matching RTL generics and writes
separate XML results and logs under:

```text
hw/rtl/cluster/tb/sim/<shared-build>/
```

## 6. Waveform debug

Enable FST tracing for a focused reproduction:

```bash
cd /home/dev01/neural-ai/hw/rtl/cluster
env PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  make -j10 sim DEBUG=1 TRACE_FORMAT=fst \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=<focused_test> \
    CLUSTER_SIM_NAME=<trace_build_name>
```

Use a distinct `CLUSTER_SIM_NAME` because trace support is compiled into
`Vtop`. FST is normally much smaller than VCD. Do not enable waveforms for a
large full-model run until a short failing prefix is known.

Useful waveform groups:

- command control registers at `0x2000_5000`;
- Snitch PC and command-buffer DTCM words;
- iDMA start/done and source/destination requests;
- systolic `state_q`, `drain_state_q`, request/response counters;
- linebuffer channel/background state, FIFO count, emitted vectors, and busy;
- OFM FIFO valid/ready/empty;
- host IRQ and PMU enable/snapshot.

## 7. Progress log format

The cluster wait loop is event-driven and prints a snapshot every 100,000
cycles. A typical line is:

```text
waiting for host irq: 300,000/1,000,000 cycles;
command=18/22 (index=17), type=9, size=160, layer=2, tile=39;
sys=WAIT_DRAIN, drain=IDLE, linebuf=STREAM_DONE,
req=240, rsp=240, remaining=0, k_tile=8,
weight_empty=1, ofm_empty=1, lb_prefetch_busy=0,
bg=0, beat_fifo=0, emitted=240
```

### 7.1 Command fields

| Field | Meaning |
|---|---|
| `command=18/22` | one-based command position in the current prefix or segment |
| `index=17` | zero-based position |
| `type` | ABI v2 command type |
| `size` | serialized command bytes |
| `layer` | compiler scheduler operation index |
| `tile` | compiler-generated tile/stripe sequence number |

The command header shown in the log is read from the trusted firmware command
buffer in DTCM, then matched against the `.nai` command section. If identical
headers occur more than once, the log reports `command_candidates` instead of
claiming one unique index.

Important command type values:

| Type | Command |
|---:|---|
| 0 | END |
| 2/3/4 | DMA 1D/2D/3D |
| 5 | requant parameter load |
| 6/7/8 | GEMM32/base accumulate/requant variants |
| 9 | linebuffer+systolic job |
| 10 | pointwise C32 |
| 11 | depthwise C32 |
| 12 | AFU LUT |
| 13 | AFU binary |
| 14 | AFU global average pool |
| 15/16/17 | Spatz requant/add/mul |
| 18 | copy-layout |
| 19 | maxpool |
| 20 | nearest-neighbor upsample |
| 21/22/23 | rolling reset/produce/consume-release |
| 24/25/26/27 | asynchronous DMA submit/wait |
| 28 | fused AFU DFL16 |

### 7.2 Systolic and linebuffer fields

`sys` is the top systolic controller state:

- `IDLE`: no active job;
- `LOAD_WEIGHTS`: weight tile loading;
- `COMPUTE`: array input/compute phase;
- `WAIT_DRAIN`: compute input finished; output or formatter still draining;
- `DONE`: completion pulse/state.

`drain` describes the output/partial-sum drain sub-FSM:

- `IDLE`;
- `ACCUM_READ`;
- `ACCUM_WRITE`;
- `ACCUM_REQUANT`.

`linebuf` describes formatter activity. The most useful states are:

- `ENSURE`, `FILL_REQ0/1`, `FILL_DRAIN`: populate required rows;
- `WINDOW_REQ`, `WINDOW_WAIT`: form a convolution window;
- `STREAM_PRIME`, `STREAM_EMIT`: emit vectors to the array;
- `BYPASS_*`: direct/bypass path;
- `STREAM_DONE`: final formatted vectors have been emitted and the pipeline is
  releasing state.

Additional fields:

| Field | Interpretation |
|---|---|
| `req`, `rsp` | systolic input requests issued and responses accepted |
| `remaining` | output drain count still outstanding |
| `k_tile` | current internal K tile |
| `weight_empty` | weight FIFO empty flag |
| `ofm_empty` | output FIFO empty flag |
| `lb_prefetch_busy` | linebuffer prefetch engine active |
| `bg` | linebuffer background-fill state value |
| `beat_fifo` | buffered input beat count |
| `emitted` | formatter vectors emitted for the current job |

## 8. Completion and failure registers

After IRQ or timeout, the test reads:

| Register | Meaning |
|---|---|
| `NPU_CMD_STATUS` | 0 idle, 1 loading, 2 running, 3 pass, 4 fail |
| `NPU_CMD_FAIL_CODE` | model/runtime/dispatcher failure code |
| `NPU_CMD_FAIL_PTR` | failing invocation/model/command address |
| `NPU_CMD_DONE_COUNT` | commands successfully completed before return/failure |

Base failures:

| Code | Meaning |
|---|---|
| `0xBADCD00A` | malformed or inaccessible invocation |
| `0xBADCD00B` | malformed model/package |
| `0xBADCD00C` | invalid binding table/address/size |
| `0xBADCD00E` | bad command stream framing |
| `0xBADCD00F` | invalid command fields or unsupported schedule combination |
| `0xBADCD010` | unsupported command |
| `0xBADCD011` | invalid reference/region/span/alignment |
| `0xBADCD012` | operator execution failed |

Dispatcher codes are formed as `0xBADCD00D + dispatch_status`, where status 1
through 5 means bad stream, bad command, unsupported, bad reference, and
operation failed.

Use `FAIL_PTR` together with the command section base to locate the serialized
descriptor. `DONE_COUNT` identifies the last completed boundary even when the
current DTCM command header is ambiguous.

## 9. PMU report interpretation

A successful run prints:

```text
PMU performance report:
  cycles=958417
  snitch: instr=479076 load=68132 tcdm_req=15912 stall=0
  systolic: compute=28800 (3.00%) ifm_req=77992 ofm_req=13660 ofm_stall=220
  spatz: issue=26517 rsp=8917 tcdm_req=88000 stall=17600
  idma: busy=29242 (3.05%) start=239 done=239 tcdm_stall=0
  afu: done=759421 tcdm_req=25600 stall=0
  tcdm: req=240402 gnt=222382 stall=18020 read=150704 write=89698
```

Interpretation:

- `cycles`: fetch release to firmware completion; excludes Python backdoor
  model/input loading;
- `snitch instr/load`: firmware control cost and scalar load pressure;
- `snitch tcdm_req/stall`: scalar-core pressure on shared TCDM;
- `systolic compute`: cycles with array compute enabled; the percentage is
  utilization relative to total inference cycles, not MAC utilization;
- `ifm_req`, `ofm_req`, `ofm_stall`: systolic input/output traffic and output
  backpressure;
- `spatz issue/rsp`: vector instruction requests/responses;
- `spatz tcdm_req/stall`: vector memory pressure and arbitration loss;
- `idma busy/start/done`: DMA occupancy and transaction count; `start != done`
  at final completion indicates a synchronization defect;
- `afu done`: AFU progress event counter; compare deltas between equivalent
  runs rather than treating it as a universal byte/cycle value;
- `afu tcdm_req/stall`: AFU memory activity and contention;
- `tcdm req/gnt/stall`: aggregate arbitration; normally
  `stall` is close to `req - gnt`;
- `read/write`: aggregate TCDM traffic split;
- `overflow_status`: nonzero means one or more counters wrapped or overflowed,
  so performance conclusions are unsafe.

For regression comparisons, keep model, command prefix, firmware build, RTL
generics, and simulator version constant. PMU numbers are architecture
references, not a requirement to match an older implementation cycle-for-cycle.

## 10. Distinguishing valid long work from deadlock

### Likely valid long-running work

- the current command is AFU LUT/DFL, Spatz, DMA, or a large linebuffer tile;
- command index stays fixed but internal counters continue changing;
- `k_tile`, `emitted`, `req/rsp`, FIFO count, or DMA completion advances;
- increasing the prefix by a known byte length causes a proportional cycle
  increase;
- the previous prefix completes close to the timeout.

Example: prefix 129 used 937,925 cycles and prefix 131 used 958,417 cycles.
The extra two AFU LUT commands therefore added about 20,492 cycles. A later
timeout near one million cycles would first be treated as a bound issue, not a
deadlock.

### Likely deadlock or lost handshake

- the same command and all state/counter fields repeat over several 100,000
  cycle snapshots;
- `sys=WAIT_DRAIN` while `remaining=0`, FIFOs are empty, but linebuffer remains
  busy indefinitely;
- `linebuf=STREAM_DONE` never returns to IDLE;
- `idma start > done` and neither value changes;
- request count grows without responses, or both freeze with outstanding work;
- firmware PC is unchanged in an unexpected polling loop;
- `DONE` was seen while a producer remained busy, then the next one-cycle START
  was lost.

### Likely validation or ABI failure

- status changes to FAIL quickly;
- `FAIL_CODE` is `BADCD00A..BADCD012`;
- `DONE_COUNT` points just before one descriptor;
- host ABI tests fail for the same descriptor;
- Regor-serialized flags do not match the trusted firmware's expected schedule.

### Likely numerical/layout failure

- status PASS and command count is complete, but output bytes differ;
- DMA dimensions/strides or C32/C16 tail layout are wrong;
- qparam block, multiplier, shift, clamp, or zero point differs;
- a segmented boundary failed to preserve a required intermediate;
- output materialization wrote correct values in the wrong plane/order.

## 11. Practical debug workflow

1. Record the exact model hash, compiler commit, runtime/RTL commit, firmware
   section size, test command, simulator version, and RTL generic values.
2. Reproduce with the smallest passing prefix `N-1` and failing prefix `N`.
3. Decode command `N`: type, size, layer/tile, references, dimensions, strides,
   schedule bits, and expected work.
4. Add or run the matching Regor serialization/planner test.
5. Add or run runtime host validation for both valid and adjacent-invalid
   descriptors.
6. Reproduce the hardware behavior in a focused block test, including two
   consecutive jobs when START/DONE or retained state is involved.
7. Run RTL lint after any RTL edit.
8. Run the focused cluster prefix and inspect status, fail pointer, done count,
   progress FSM, and PMU report.
9. Extend by bounded prefixes or segmented continuation.
10. Run the full graph and compare the complete public output with an
    independent reference.

When a failure appears only at cluster level, separate the possible owners:

```text
Regor serialization
  -> model loader/reference resolver
  -> trusted dispatcher/HAL
  -> MMIO shadow/active register transfer
  -> DMA/TCDM arbitration
  -> operator RTL
  -> output layout/materialization
```

Do not change several layers speculatively. Use the first observable mismatch
to decide which layer needs a focused test.

## 12. Output files and preserving evidence

For a direct Makefile run, preserve:

- the complete console log;
- `results.xml` or the configured `COCOTB_RESULTS_FILE`;
- the exact environment variables and command line;
- the generated `Vtop` build name;
- FST/VCD only for the smallest focused failure;
- compiler report and `.nai` package when debugging serialization.

For the parallel runner, logs and result XML files live under the selected
shared simulation directory. A useful failure report contains:

```text
first failing command and preceding passing prefix
status / fail code / fail pointer / done count
last two progress snapshots
PMU report from the closest passing run
decoded command fields
focused host/block-test result
compiler and runtime/RTL commits
```

This is enough to distinguish compiler contract, trusted runtime, and RTL
execution defects without attaching a full multi-gigabyte waveform.
