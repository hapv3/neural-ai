# Performance Management Unit (PMU) Design Plan

Goal: build a hardware PMU that collects real-time hardware performance counters (HPCs) for each major NPU cluster component. Firmware or the host CPU can then profile the system, identify bottlenecks, and compute metrics such as TOPS, memory bandwidth, and hardware utilization.

## 1. PMU Operating Model

- **Architecture:** The PMU is a set of 32-bit or 64-bit counter registers.
- **Interface:** The counters are mapped into a dedicated MMIO range on the AXI4-Lite host slave port. The P0 implementation uses `0x2000_4000` because `0x2000_2000` belongs to the interrupt controller. Python/cocotb host code can write this block to clear counters, start/stop counting, snapshot counters, and read results after each test.
- **Routing:** Each component (DMA, Systolic, TCDM) emits event wires such as `is_active`, `is_stalled`, or `conflict_pulse`. These wires connect directly to PMU inputs and increment the corresponding counter on each clock cycle.

---

## 2. Metrics by Component

> [!TIP]
> **Profiling rule of thumb:** To identify where the NPU bottlenecks, measure three basic states for every module: **Active** (doing useful work), **Idle** (waiting for work), and **Stalled** (ready to work but blocked by I/O or arbitration).

### 2.1. Systolic Array (Matrix Engine)

Characteristic: a data-hungry compute engine. The PMU should measure MAC activity and data-starvation rate.

> [!NOTE]
> **Hardware performance impact:** Adding PMU counters for the Systolic Array does not reduce throughput or increase the array critical path. The PMU only snoops existing control wires such as `valid`, `ready`, and FSM states, then counts through independent accumulators outside the datapath.

- **`SYS_ACTIVE_CYCLES`**: Number of cycles in which the Systolic Array is actively performing MAC work.
- **`SYS_STALL_CYCLES`**: Number of cycles in which the Systolic Array wants to compute but stalls because the TCDM interconnect cannot feed Weight/IFM or drain OFM fast enough.
- **`SYS_IDLE_CYCLES`**: Number of cycles in which the array is idle and waiting for Snitch to configure the next layer.
- **`SYS_TOTAL_MACS`**: Optional total number of MAC operations performed, or derived from M/N/K configuration.

Analysis: utilization = `SYS_ACTIVE_CYCLES` / total cycles. A high `SYS_STALL_CYCLES` value indicates that TCDM I/O is the bottleneck.

### 2.2. iDMA (Data Movement Engine)

Characteristic: an asynchronous data mover.

> **P0 reality:** The current iDMA wrapper does not yet expose the full native event struct at cluster top. PMU P0 counts existing signals: `busy`, `start`, `done`, and TCDM master request/stall. Once the wrapper exposes the native event bus, byte counters and detailed AXI/L1 stall counters can be added.

Counters map directly from iDMA flags:

- **`DMA_ACTIVE_CYCLES`**: Connected to `dma_busy`; counts cycles with an in-flight iDMA transfer.
- **`DMA_L2_STALL_CYCLES`**: Connected to `ar_stall` and `aw_stall`; counts cycles blocked by slow L2/AXI interconnect readiness.
- **`DMA_L1_STALL_CYCLES`**: Connected to `w_stall` and `r_stall`; counts cycles blocked by L1/TCDM interconnect readiness.
- **`DMA_BYTES_TRANSFERRED`**: Accumulated from iDMA byte-count signals such as `num_bytes_written` and `r_bw`.

Analysis: actual bandwidth (GB/s) = `DMA_BYTES_TRANSFERRED` / (`DMA_ACTIVE_CYCLES` * 1/Freq).

### 2.3. Snitch Core (Control Core)

Characteristic: control-flow management. Most time should be spent in low-power sleep (`WFI`) while DMA and Systolic execute work.

> **Native integration:** Snitch already provides a strong performance monitor. When `SNITCH_ENABLE_PERF` is enabled, Snitch automatically counts `mcycle` and `minstret` through CSRs. The core also emits a `core_events_t` struct with pulses such as `retired_instr`, `retired_load`, and `retired_acc`.

Counter mapping:

- **`CORE_ACTIVE_CYCLES`**: Can be read directly from CSR `mcycle`.
- **`CORE_INSTR_RETIRED`**: Can be read directly from CSR `minstret`.
- **`CORE_WFI_CYCLES`**: Can be derived from the difference between `mcycle` and executed instructions, or counted from an internal sleep signal.

Analysis: a well-designed NPU should spend more than 90% of CPU time in `CORE_WFI_CYCLES`.

### 2.4. TCDM Interconnect (Memory Subsystem)

Characteristic: the main data-routing crossbar in the NPU and the most likely source of I/O bottlenecks from bank conflicts.

> **Native integration:** Similar to PULP/Spatz library patterns, each SRAM-bank port can use `popcount` logic to collect two metrics: requests touching the bank (`accessed`) and requests rejected due to conflict (`congested`).

Counter mapping:

- **`TCDM_BANK_CONFLICTS`**: Counts total master requests rejected or stalled because of priority/bank conflicts.
- **`TCDM_TOTAL_REQ`**: Counts total successful requests sent to SRAM banks.

Analysis: conflict rate = `TCDM_BANK_CONFLICTS` / `TCDM_TOTAL_REQ`. If this exceeds roughly 5-10%, firmware should optimize memory layout to distribute matrices across banks and avoid multiple ports concentrating on one bank.

---

## 3. Hardware PMU Architecture

```text
                                              +-----------------------------------+
[Systolic Array] ---- (active, stall) ------> |                                   |
[iDMA] -------------- (active, stall) ------> |         Performance               |
[TCDM Arbiter] ------ (conflict) -----------> |         Management                |
[Snitch Core] ------- (sleep, active) ------> |         Unit (PMU)                |
                                              |                                   |
                                              |  - Counter 0: SYS_ACTIVE (32b)    |
   MMIO Bus (0x2000_4000)                     |  - Counter 1: SYS_STALL  (32b)    |
   (Read results / reset counters) ---------> |  - Counter N: ...                 |
                                              +-----------------------------------+
```

## 4. PMU v2 Implementation

PMU v2.1 is instantiated in `npu_cluster` with 163 fixed 64-bit metrics and a
32-bit host AXI4-Lite MMIO interface. Counters 0-31 retain the original P0 ABI
for comparisons with saved runs. Counters 32-162 are the authoritative metrics
for optimization because they distinguish asserted requests from accepted
transactions and completion levels from completion pulses, and include complete
systolic-drain and linebuffer FSM occupancy.

- `0x2000_4000` `CTRL`: bit0 enable, bit1 clear, bit2 snapshot.
- `0x2000_4004` `STATUS`: overflow sticky bits.
- `0x2000_4008` `NUM_COUNTERS`: number of fixed counters.
- `0x2000_400c..0x2000_4014`: overflow words for counters 32-127.
- `0x2000_402c..0x2000_4030`: overflow words for counters 128-162.
- `0x2000_4018` `VERSION`: `0x0002_0001`.
- `0x2000_401c` `FILTER_CTRL`: bit 0 enables logical-command filtering.
- `0x2000_4020` `FILTER_CONTEXT`: selected zero-based command ID.
- `0x2000_4024` `CONTEXT`: current firmware command ID.
- `0x2000_4028` `PHASE`: bit 4 is command-active; bits 3:0 are the runtime phase.
- `0x2000_4100 + id*8`: counter low/high 32-bit.

Counter groups:

| ID | Counter |
| --- | --- |
| 0 | cycle |
| 1-4 | Snitch retired instruction/load/int/acc events |
| 5-6 | Snitch TCDM request/stall |
| 7-10 | Spatz issue/response/TCDM request/stall |
| 11-15 | iDMA busy/start/done/TCDM request/stall |
| 16-18 | AFU done/TCDM request/stall |
| 19-25 | Systolic compute/weight/ofm/IFM/OFM request/stall |
| 26-31 | Aggregate TCDM request/grant/stall/bank/read/write request |
| 32-43 | Logical command active/begin/end and invocation/model/binding/validation/fetch/execute/barrier/complete/fail phase cycles |
| 44-54 | Independent load/store DMA busy, overlap, start/done, queue occupancy sum and peak |
| 55-73 | AXI request/response handshakes, bytes, blocked cycles, outstanding sum/peak, and completion-latency sum/max |
| 74-95 | Systolic load/compute/drain/done states, useful work, OFM backpressure, accepted/blocked IFM/weight/RHS/OFM traffic, linebuffer/prefetch/binary activity, start/done |
| 96-111 | AFU active/start/done pulse, grouped core FSM states, backend drain, core input/output beats and primary stalls |
| 112-116 | Spatz active/issue/response and accepted/blocked VLSU TCDM traffic |
| 117-127 | Accepted/blocked TCDM transactions, active/conflicting banks, accepted read/write transactions and bytes, compute-DMA overlap, command time with every engine idle |
| 128-131 | Output-drain FSM: idle, accumulation read, accumulation write and accumulation requant |
| 132-146 | Linebuffer spatial scheduler: every state from idle/ensure through fill, window, stream, bypass and done |
| 147-150 | Linebuffer main-fetch FSM: request beat 0/1, drain and idle |
| 151-155 | Linebuffer background-prefetch FSM: idle, scan, request beat 0/1 and drain |
| 156-162 | Linebuffer bypass-engine FSM: idle, prepare, request/wait beat 0/1 and emit |

The exact ID-to-name mapping used by reports is `PMU_COUNTER_NAMES` in
`hw/rtl/cluster/tb/tests/npu_test_utils.py`.

### 4.1 Command and engine attribution

Profiling firmware writes a zero-based logical command ID to `NPU_CMD_PMU_BEGIN`
immediately before dispatch and to `NPU_CMD_PMU_END` immediately after dispatch.
These registers live in the existing command-control block, so no new Snitch
MMIO route is required. The command-control block also records coarse runtime
phase changes.

Each engine latches the current command ID at its real start/issue event. DMA
load and store directions each keep a 16-entry tag FIFO matching their hardware
job queues, while systolic, AFU and Spatz hold their active tag until completion.
Consequently a command filter continues to count an asynchronous job after
firmware has begun another command. DMA and AXI response attribution assumes
the existing in-order response contract; changing the DMA AXI IDs to permit
out-of-order completion requires propagating tags with those IDs as well.

Set `NAI_PMU_PROFILE=0` when building `sw/runtime/neural_ai` to remove the two
command-marker MMIO writes from production firmware. Profiling builds default
to `NAI_PMU_PROFILE=1`. This makes the measurement overhead explicit and
removable rather than silently charging all production command dispatches.

### 4.2 Counter semantics

- A name ending in `_accept`, `_fire`, `_beat`, `_start`, `_done`, or `_pulse`
  counts a handshake/event, not time asserted.
- A name ending in `_cycles` counts clocks for which the condition is true.
- Queue/outstanding `occupancy` counters integrate occupancy over time; divide
  by an applicable busy-cycle count for the mean. Their `peak` companions are
  max-mode counters, not sums.
- AXI latency sum is measured from accepted AR to accepted final R beat and from
  accepted AW to accepted B. Divide by the matching completed burst count.
- TCDM byte metrics use the accepted request byte enables. Conflict cycles mean
  at least one bank had multiple contenders; `tcdm_conflicting_banks` sums how
  many banks conflicted in each cycle.
- P0 IDs 5-31 intentionally preserve asserted-request semantics. Use IDs
  83-90, 115-125 for transaction and stall analysis.
- State counters 128-162 are gated by the active systolic job tag. Their IDLE
  counters therefore measure idle occupancy within a systolic job and do not
  accumulate between jobs.

### 4.3 Reporting policy

Python/cocotb only configures the PMU, reads a stable snapshot, derives ratios,
and writes inference/per-command CSV. It does not sample every clock. For a
specific asynchronous command, call `pmu_start(..., command_id=<id>)`; the RTL
filter applies the engine tags described above. Per-command CSV remains an
event-driven command-boundary facility and includes all 163 counters.

There is deliberately no cycle timeline trace tier. It would create very large
files and is unnecessary for the optimization loop. When finer diagnosis is
needed, use the command filter and rerun only a snapshot-bounded segment.

Access model:

- Host AXI4-Lite slave port decodes `0x1000_0000` I-TCM for firmware boot and `0x2000_4000` PMU for profiling.
- Snitch D-bus `0x2000_4000` remains intentionally disconnected. Firmware emits
  tags through the command-control window at `0x2000_5024..0x2000_502c`.
- Cocotb starts PMU before `fetch_enable_i`, snapshots/stops it after `irq_o`, then prints a performance report.

Validation gates:

- `make -C sw/test/pmu`
- `make -C sw/test/compiler_runtime check`
- `make -C hw/rtl/cluster pmu_unit`
- `make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot`
- `test_pmu_basic` firmware smoke generates Snitch/TCDM traffic; Python host verifies PMU MMIO, snapshot, and non-zero TCDM counters. If building a dedicated simulator for this module is too heavy, it can reuse any up-to-date `tb_npu_cluster` Verilator binary because the cocotb test module is selected at runtime.
