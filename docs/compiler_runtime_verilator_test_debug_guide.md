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

### 5.4 Persistent snapshots between simulator runs

Segmented continuation avoids replay inside one simulator process. To avoid
replaying an established prefix after stopping and starting a new process,
write a persistent snapshot at a clean command boundary:

```bash
env \
  PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  YOLO320_SEGMENT_ENDS=131 \
  YOLO320_SEGMENT_TIMEOUT_CYCLES=1000000 \
  YOLO320_SNAPSHOT_OUT=/tmp/yolo320-command-131.snapshot \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_segmented_prefix \
    CLUSTER_SIM_NAME=test_compiled_model
```

Resume in a separate simulator run without executing commands 1–131:

```bash
env \
  PYTHONPATH=/home/dev01/neural-ai/sw/test/compiler_runtime \
  YOLO320_SNAPSHOT_IN=/tmp/yolo320-command-131.snapshot \
  YOLO320_SEGMENT_ENDS=141 \
  YOLO320_SEGMENT_TIMEOUT_CYCLES=1000000 \
  YOLO320_SNAPSHOT_OUT=/tmp/yolo320-command-141.snapshot \
  make -j10 sim \
    COCOTB_TEST_MODULES=test_compiled_model \
    COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_segmented_prefix \
    CLUSTER_SIM_NAME=test_compiled_model
```

The snapshot contains:

- the command boundary;
- CRC32 of the complete compiled model;
- all 16 x 1,024 x 32-byte physical TCDM storage in logical-address order;
- the 307,200-byte selected-model L2 temporary arena;
- the 176,400-byte public output buffer;
- sizes and CRC32 of the uncompressed state payload;
- a zlib-compressed payload.

The deterministic input binding and `.nai` model are loaded normally. Model
constants do not need to be duplicated in the snapshot. Restore is performed
while logic is held in reset; TCDM and L2 state are deposited before reset is
released and trusted firmware starts the suffix command stream.

When `YOLO320_SEGMENT_ENDS` contains multiple boundaries, the harness writes a
snapshot immediately after every segment that reports PASS. Use a `{command}`
placeholder for explicit filenames:

```bash
YOLO320_SEGMENT_ENDS=611,623,635,647,659,671,683,700 \
YOLO320_SNAPSHOT_OUT='/tmp/yolo320-command-{command}.snapshot'
```

This creates snapshots for commands 611, 623, 635, 647, 659, 671, 683, and
700. A path such as `/tmp/yolo320-command-700.snapshot` is also accepted; the
trailing command value is replaced for each boundary. For a generic path such
as `/tmp/yolo320-checkpoint.snapshot`, the generated names are
`yolo320-checkpoint-command-611.snapshot`, etc. A single boundary retains the
exact output path for backward compatibility. Because each snapshot is written
before the next logic reset, completed checkpoints remain available if a later
segment times out or fails.

Snapshot safety rules:

- create snapshots only after a segment reports PASS and IRQ;
- a snapshot must match the exact full-model CRC, so recompiling to a different
  package invalidates it even if tensor shapes look unchanged;
- the boundary stored in the snapshot must be lower than the first requested
  segment end;
- do not edit or concatenate snapshot files;
- keep snapshots in `/tmp` or another artifact directory, not in Git;
- a restored suffix must produce the same status, command count, PMU behavior,
  and output as continuation in one simulator process.

Minimal two-process validation uses command 5 as the saved boundary. First
generate `/tmp/yolo320-command-5.snapshot`, then restore it with
`YOLO320_SEGMENT_ENDS=6`. This checks both VPI TCDM restore and L2 restore before
using a large checkpoint. The verified restore run executes only command 6 and
reports 52,081 PMU cycles, exactly matching command 6 after in-process segmented
continuation; its systolic, DMA, and TCDM event counts also match.

### 5.5 Byte-exact snapshot validation against TensorFlow Lite

A segmented run that stops before the final command verifies command status,
command count, snapshot integrity, and memory restore. It does **not** compare
intermediate tensor data with TensorFlow Lite. `PASS` at such a boundary means
that the selected command range completed without a runtime/RTL protocol error;
it is not proof of numerical correctness.

Use the following procedure when a suffix passes but the final public output is
wrong, or when the first numerically wrong operator must be identified.

#### Step 1: freeze all inputs to the comparison

Record all of the following before inspecting bytes:

- compiler commit and generated `.nai` file;
- RTL/runtime commit and Verilator build name;
- source `.tflite` file and its hash;
- snapshot boundary and snapshot file;
- exact input binding bytes;
- command range and PMU CSV used to create the snapshot.

The selected YOLO320 test uses this deterministic input:

```python
input_bytes = 320 * 320 * 3
input_data = bytes(((index * 37 + 11) & 0xFF) for index in range(input_bytes))
```

Use those exact bytes for both the RTL run and TensorFlow Lite. Do not generate
random input independently in the two processes.

#### Step 2: validate and decode the snapshot container

The snapshot header is `<8s6I>` and contains magic, command boundary, model
CRC32, TCDM size, L2-temporary size, output size, and payload CRC32. The payload
is `TCDM || L2 temporary || public output`, compressed with zlib.

The following standalone check rejects a truncated snapshot, a snapshot from a
different compiled model, or corrupt state before any numerical comparison:

```python
from pathlib import Path
import struct
import zlib

model = Path("/tmp/yolo320-compiled/yolov8n_320_int8.nai").read_bytes()
snapshot = Path("/tmp/yolo320-command-4090.snapshot").read_bytes()
header = struct.Struct("<8s6I")
(
    magic,
    boundary,
    model_crc,
    tcdm_bytes,
    temporary_bytes,
    output_bytes,
    payload_crc,
) = header.unpack_from(snapshot)

assert magic == b"NAISNP01"
assert model_crc == (zlib.crc32(model) & 0xFFFFFFFF)
payload = zlib.decompress(snapshot[header.size :])
assert len(payload) == tcdm_bytes + temporary_bytes + output_bytes
assert payload_crc == (zlib.crc32(payload) & 0xFFFFFFFF)

tcdm = payload[:tcdm_bytes]
l2_temporary = payload[tcdm_bytes : tcdm_bytes + temporary_bytes]
public_output = payload[tcdm_bytes + temporary_bytes :]
print(boundary, len(tcdm), len(l2_temporary), len(public_output))
```

The test harness implements the same checks in
`_decode_yolo320_snapshot()` and `_verify_yolo320_snapshot()` in
`sw/test/compiler_runtime/test_compiled_model.py`.

#### Step 3: generate the independent TensorFlow Lite reference

Use the reference op resolver and preserve intermediate tensors. Optimized
TensorFlow Lite kernels can have different rounding behavior and are not the
byte-exact oracle used by the full selected-model test.

```python
from pathlib import Path
import numpy as np
import tensorflow as tf

model_path = Path("/home/dev01/neural-compiler/test/model/yolov8n_320_int8.tflite")
input_bytes = 320 * 320 * 3
input_data = bytes(((index * 37 + 11) & 0xFF) for index in range(input_bytes))

interpreter = tf.lite.Interpreter(
    model_path=str(model_path),
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    experimental_preserve_all_tensors=True,
)
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
input_tensor = np.frombuffer(input_data, dtype=np.uint8).view(np.int8)
input_tensor = input_tensor.reshape(input_detail["shape"])
interpreter.set_tensor(input_detail["index"], input_tensor)
interpreter.invoke()

for detail in interpreter.get_tensor_details():
    if "multiply_242" in detail["name"] or "transpose_25" in detail["name"]:
        print(detail["index"], detail["name"], detail["shape"],
              detail["quantization_parameters"])
```

Select an intermediate tensor by verified name, shape, and quantization, not by
index alone. Tensor indices can change when the source model is regenerated.
For the DFL investigation below, the merged scaled box tensor was tensor 397
with shape `[1, 4, 2100]`.

#### Step 4: map the reference tensor to resident accelerator memory

A TFLite tensor name does not directly identify a TCDM address. Establish the
mapping from the exact compiled package:

1. Use compiler verbose schedule/debug-map output to map the TFLite operation
   to its Neural-AI layer and ABI commands.
2. Decode source/destination references, dimensions, layouts, and DMA gathers
   from those commands.
3. Track the compiler memory-state/liveness information to identify whether the
   tensor is in TCDM, L2 temporary, or the public output at boundary `N`.
4. Choose a snapshot after the producer and any required gather have completed,
   but before that allocation is reused.
5. Confirm tensor byte size from data type and shape. Never compare an entire
   TCDM image with TensorFlow output: TCDM also contains live unrelated data,
   reusable scratch, padding, and dead allocations.

For compact INT8 tensors, the snapshot byte offset is the logical TCDM address
used by the ABI reference. For blocked layouts, first undo C32/C16 padding and
the emitted layout transformation before comparing with the NHWC/CHW reference.

#### Step 5: compare bytes and useful error statistics

This example compares the compact merged DFL result resident at TCDM offset
1600 in the command-4090 snapshot:

```python
import numpy as np
import zlib

tensor_index = 397
reference = interpreter.get_tensor(tensor_index).astype(np.int8).reshape(-1)
actual = np.frombuffer(tcdm[1600 : 1600 + 8400], dtype=np.int8)
assert actual.size == reference.size == 8400

delta = actual.astype(np.int16) - reference.astype(np.int16)
mismatch = np.flatnonzero(delta)
print(f"actual_crc=0x{zlib.crc32(actual.tobytes()) & 0xFFFFFFFF:08x}")
print(f"reference_crc=0x{zlib.crc32(reference.tobytes()) & 0xFFFFFFFF:08x}")
print(f"exact={np.mean(delta == 0) * 100:.4f}%")
print(f"mae={np.mean(np.abs(delta)):.6f}")
print(f"max_abs={np.max(np.abs(delta))}")
print(f"bias={np.mean(delta):.6f}")
print(f"actual_range=[{actual.min()}, {actual.max()}]")
print(f"reference_range=[{reference.min()}, {reference.max()}]")
if mismatch.size:
    index = int(mismatch[0])
    print("first_mismatch", index, int(actual[index]), int(reference[index]))
```

CRC equality is a convenient final check, but the distribution metrics are
what make a failure diagnosable:

- one constant extreme value usually indicates saturation or a bad clamp;
- a large one-sided bias suggests zero-point or signedness error;
- values with the right shape but periodic mismatches suggest layout/stride;
- small differences around rounding thresholds suggest multiplier/shift or
  rounding-mode mismatch;
- a correct producer tensor followed by a wrong gathered tensor points to DMA
  dimensions, strides, ordering, or destination overlap.

#### Step 6: split a merged tensor and locate the first bad producer

For a tensor assembled from multiple heads, compare both the final gather and
each source allocation. The YOLO DFL reference is side-major and can be split
along its 2100 locations:

```python
reference_2d = reference.reshape(4, 2100)
head_specs = [
    # (name, first_location, locations, TCDM byte offset)
    ("40x40", 0,    1600, 427104),
    ("20x20", 1600,  400,      0),
    ("10x10", 2000,  100,  41600),
]

for name, first, locations, address in head_specs:
    expected = reference_2d[:, first : first + locations].reshape(-1)
    observed = np.frombuffer(
        tcdm[address : address + 4 * locations], dtype=np.int8
    )
    difference = observed.astype(np.int16) - expected.astype(np.int16)
    print(name, "exact", np.mean(difference == 0),
          "mae", np.mean(np.abs(difference)),
          "range", (int(observed.min()), int(observed.max())))
```

Addresses are build-specific and must be re-derived after scheduling or memory
allocation changes. The values above document the investigated package only.

#### Step 7: compare adjacent snapshots without confusing writes and reuse

When snapshots exist at `N-1` and `N`, report changed byte ranges separately
for TCDM, L2 temporary, and public output. Intersect those ranges with the
decoded destination spans of command `N`. Unexpected changes outside legal
destinations indicate overwrite or an incomplete hazard; no change inside the
expected destination indicates a lost start, stale read, or missing store.

Do not interpret every difference as an error. Later commands legally reuse
TCDM allocations. Compare data to TensorFlow only while the compiler memory
model says that tensor is live.

#### Worked example: DFL16 saturation in commands 3951-4090

The replay restored command 3950 and executed commands 3951-4090. Runtime
status was PASS for all 140 commands, but the snapshot comparison found:

- merged DFL tensor: 8400/8400 bytes equal to `127`;
- exact match against TensorFlow tensor 397: 0%;
- MAE: about 216.45, with bias about +216.45;
- all three pre-gather head tensors were already wrong;
- the class branch at TCDM offset 42016 was 99.8006% exact, MAE 0.00386, and
  maximum absolute error 9.

This ruled out the final gather and four post-DFL box Add/Sub commands. Command
decoding then showed these DFL16 requantization parameters:

| Expanded command | Locations | Wrong multiplier/shift | Correct multiplier/shift |
|---:|---:|---:|---:|
| 3339 | 100 | 51003 / 8 | 58039 / 19 |
| 3617 | 400 | 51203 / 9 | 58267 / 20 |
| 4081 | 1600 | 51203 / 10 | 58267 / 21 |

The compiler's explicit-quantization pass had converted the `Mul` input scale
to an execution scale of 1. The later structural DFL fusion incorrectly treated
that value as the original physical scale. Consequently raw scale deltas
64/128/255 were used instead of approximately 0.0251/0.0502/0.1, and the DFL
output saturated. The fix reads the original scalar quantization from TFLite
tensor metadata for the fused DFL interface while preserving explicit
quantization for the following Concat/Add/Sub operations.

The regression gate checks the exact generated qparams, then recompiles the
real YOLO model and checks its three ABI commands. This is stronger than merely
checking that multiplier and shift are in their legal ranges.

#### Snapshot reuse after a compiler fix

Normally, never bypass the model-CRC check. Rerun from a snapshot created by the
same `.nai` package. A narrowly scoped exception is useful for diagnosis only
when a binary comparison proves that old and new packages have identical size,
section layout, command boundaries, addresses, and constants, and that all
changed bytes are understood command fields executed after the restored
boundary.

For this DFL issue, old and new packages had identical size and differed by
only nine bytes: the low bytes of three DFL multiplier/shift pairs. Therefore a
command-3950 snapshot could be CRC-rebound to test the third DFL at command
4081. It could **not** prove final-model correctness, because the first two DFL
commands at 3339 and 3617 had already produced stale wrong state. Full
byte-exact validation must restart before the earliest changed command, or from
command zero, and must eventually compare all 176400 public output bytes with
TensorFlow Lite.

### 5.6 Per-command PMU trace

Set `YOLO320_COMMAND_PMU_CSV` on a segmented run to persist one PMU row for
every completed ABI command:

```bash
YOLO320_SNAPSHOT_IN=/tmp/yolo320-command-1464.snapshot \
YOLO320_SEGMENT_ENDS=1571 \
YOLO320_COMMAND_PMU_CSV=/tmp/yolo320-command-pmu.csv \
make -j10 sim \
  COCOTB_TEST_MODULES=test_compiled_model \
  COCOTB_TEST_FILTER=test_compiler_generated_selected_yolo320_segmented_prefix \
  CLUSTER_SIM_NAME=test_compiled_model
```

The CSV records the global command index, ABI header metadata, and deltas for
all hardware PMU counters. Rows are flushed as soon as the following command
header is loaded, so completed rows survive a later timeout or interruption.
The boundary tracer watches the streamed D-TCM command buffer through cocotb
and adds no host AXI transactions. PMU profiling firmware separately emits two
command-tag MMIO writes per command; build with `NAI_PMU_PROFILE=0` when a
production-overhead comparison is required. Each delta spans one command
header becoming available through the following header becoming available; it
therefore includes dispatch transition and fetch cost at the boundary, as seen
by the real ABI runtime.

For exact engine attribution of one zero-based command ID, set
`NAI_PMU_COMMAND_FILTER=<id>`. The PMU then follows tags latched by DMA,
systolic, AFU and Spatz instead of attributing asynchronous work to whichever
descriptor happens to be in the firmware command buffer later.

### 5.7 Full selected graph

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

### 5.8 Parallel broad regression

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
  cycles=...
  systolic: active=... useful=... ifm=... ofm=... blocked=...
  spatz: active=... issue=... rsp=... tcdm=... blocked=...
  idma: load_busy=... store_busy=... overlap=... read=...B write=...B
  afu: active=... start=... done=... input_wait=... output_stall=...
  tcdm: accept=... blocked=... conflict_cycles=... read=...B write=...B
  overlap/control: compute_dma=... command_idle=... commands=.../...
```

Interpretation:

- `cycles`: fetch release to firmware completion; excludes Python backdoor
  model/input loading;
- `snitch instr/load`: firmware control cost and scalar load pressure;
- `snitch tcdm_req/stall`: scalar-core pressure on shared TCDM;
- `systolic useful`: cycles with array compute enabled; the percentage is
  utilization relative to total inference cycles, not MAC utilization;
- `ifm`, `ofm`, `blocked`: accepted systolic input/output traffic and primary
  backpressure;
- `spatz issue/rsp`: vector instruction requests/responses;
- `spatz tcdm_req/stall`: vector memory pressure and arbitration loss;
- load/store DMA busy and overlap are independent; start/done counters remain
  available in the CSV, and `start != done`
  at final completion indicates a synchronization defect;
- `afu done`: a rising-edge completion pulse. The legacy ID 16 remains a done
  level-cycle counter and must not be used as completion count;
- `tcdm accept/blocked`: exact handshakes and denied cycles. Conflict counters
  additionally distinguish physical bank collisions;
- `read/write`: accepted AXI or TCDM bytes, not asserted request cycles;
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
