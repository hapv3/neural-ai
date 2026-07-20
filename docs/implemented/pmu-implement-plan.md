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

## 4. P0 Implementation Status

P0 is instantiated in `npu_cluster` with 32 fixed 64-bit counters and 32-bit host AXI4-Lite MMIO:

- `0x2000_4000` `CTRL`: bit0 enable, bit1 clear, bit2 snapshot.
- `0x2000_4004` `STATUS`: overflow sticky bits.
- `0x2000_4008` `NUM_COUNTERS`: number of fixed counters.
- `0x2000_4100 + id*8`: counter low/high 32-bit.

P0 counter map:

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

P0 access model:

- Host AXI4-Lite slave port decodes `0x1000_0000` I-TCM for firmware boot and `0x2000_4000` PMU for profiling.
- Snitch D-bus `0x2000_4000` is intentionally not connected to PMU; that window returns a sink response to avoid firmware hangs.
- Cocotb starts PMU before `fetch_enable_i`, snapshots/stops it after `irq_o`, then prints a performance report.

P0 validation:

- `make -C sw/test/pmu`
- `make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot`
- `test_pmu_basic` firmware smoke generates Snitch/TCDM traffic; Python host verifies PMU MMIO, snapshot, and non-zero TCDM counters. If building a dedicated simulator for this module is too heavy, it can reuse any up-to-date `tb_npu_cluster` Verilator binary because the cocotb test module is selected at runtime.
