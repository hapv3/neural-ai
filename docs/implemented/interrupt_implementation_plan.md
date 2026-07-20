# NPU Cluster Interrupt Controller Plan

Goal: replace the existing power-hungry and inefficient MMIO polling loop with an event-driven interrupt mechanism. This allows the Snitch core to enter `WFI` while Systolic, AFU, or DMA blocks are doing heavy work, improving power behavior and responsiveness.

## Open Questions

> [!WARNING]
> **What is the width of `irq_i` on the Snitch core?**
> RISC-V commonly standardizes interrupts as M-mode External/Timer/Software (MEIP, MTIP, MSIP), or as a Fast Local Interrupt vector such as `[15:0]`. This plan assumes Snitch uses a bit-vector for internal fast interrupts. If not, the design needs an MMIO `IRQ_CAUSE` register so Snitch can read which source raised the interrupt. Confirm the current Snitch interrupt architecture before finalizing the trap path.

## Proposed Changes

## Implementation Status

Initial interrupt support is implemented for verification and firmware completion:

- `hw/rtl/cluster/npu_interrupt_ctrl.sv` provides internal pending/enable/clear registers plus external host pending/enable/clear registers.
- Hardware done events from DMA, Systolic, AFU, and Spatz are latched into the internal interrupt domain.
- `snitch_irq_o.mcip` is driven when an enabled internal event is pending.
- Firmware reports test completion by writing `NPU_IRQ_HOST_NOTIFY`; this latches `HOST_STATUS` internally and asserts cluster `irq_o`.
- Cocotb tests now boot firmware through AXI I-TCM, release Snitch with `fetch_enable_i`, wait on `irq_o`, and validate correctness through L2/TCDM output data. The current host AXI path intentionally reaches I-TCM only, so it does not read IRQ MMIO.
- D-TCM is no longer used as a testbench preload/start/status path for the active boot, memory, systolic, Spatz, operator, and matmul gates.

Remaining work is the true low-power event-driven firmware path: install a Snitch trap vector, enable M-mode interrupts, replace selected `REG_DONE` polling loops with `wfi`, and clear `IRQ_INT_PENDING` in the handler.

### 1. Create `npu_interrupt_ctrl.sv` (Event Unit)

This block acts as an interrupt aggregator, or PLIC-lite, inside the cluster. It receives signals from hardware engines and routes them into two fully separate directions:

- **Inputs (interrupt sources):**
  - `dma_done_i`: interrupt from DMA after A2O or O2A completes.
  - `sys_done_i`: interrupt from Systolic Controller.
  - `afu_done_i`: interrupt from AFU.
  - `spatz_done_i`: interrupt from Vector Engine.

---

### Direction 1: Internal Interrupt into Snitch Core

**Purpose:** Synchronize work inside the cluster, letting Snitch escape polling loops and enter `WFI` for lower power.

- **Output signal:** `snitch_irq_o`, connected directly to the Snitch core `irq_i` port.
- **Behavior:**
  - Provide a dedicated MMIO register range such as `IRQ_ENABLE_INTERNAL`, `IRQ_PENDING_INTERNAL`, and `IRQ_CLEAR_INTERNAL`.
  - When DMA or Systolic finishes a small data tile, it reports the event to this block.
  - The block wakes Snitch.
  - Snitch runs the trap handler, updates configuration/state, assigns new work to Systolic/DMA for the next tile, and returns to `WFI`.

### Direction 2: External Interrupt to Host CPU

**Purpose:** Report progress for a larger work unit, such as finishing an entire ResNet or YOLO layer, to the external host CPU.

- **Output signal:** `host_irq_o`, wired directly to the NPU cluster `irq_o` pin.
- **Behavior:**
  - Provide a separate MMIO register range such as `IRQ_ENABLE_EXTERNAL`, `IRQ_PENDING_EXTERNAL`, and `IRQ_CLEAR_EXTERNAL`.
  - Hardware blocks such as DMA and Systolic do **not** automatically trigger external interrupts.
  - Only after Snitch firmware has counted enough inner-loop work, for example 100 tiles for one layer, does it write `HOST_NOTIFY` in the event unit.
  - The event unit asserts `host_irq_o`.
  - The host CPU services the interrupt, reads cluster status, and loads firmware/data for the next layer as needed.

---

### 2. Interface-Level RTL Modifications

#### [MODIFY] `hw/rtl/cluster/npu_cluster.sv`

- Instantiate the new `npu_interrupt_ctrl` block.
- Connect the currently dangling DMA, Systolic, and AFU `done` wires into the interrupt controller.
- Update the `snitch_core` instantiation from `.irq_i('0)` to `.irq_i(snitch_irq_o)`.
- Assign module output `irq_o` from the generated `host_irq_o` signal.

#### [MODIFY] `hw/rtl/cluster/snitch_core.sv`

- If needed, expose the `irq_i` port, if it is currently hardcoded inside wrappers, so it connects directly to `i_snitch`.

---

### 3. Firmware Impact

The `WFI` mechanism changes the firmware loop:

1. **Initialize:** In `main()`, enable `MIE` (Machine Interrupt Enable) in RISC-V `mstatus`. Enable the relevant interrupt bit in `npu_interrupt_ctrl`.
2. **Start hardware:** Write a `start` command to Systolic Controller.
3. **Sleep:** Execute `wfi;`. Snitch can stop active work and wait.
4. **Wakeup:** When Systolic finishes, it asserts `sys_done`. `npu_interrupt_ctrl` asserts Snitch IRQ.
5. **Trap handler:** PC jumps to `trap_vector`. The handler reads `IRQ_PENDING`, identifies Systolic completion, updates firmware state, writes `1` to `IRQ_CLEAR`, and returns to `main()`.

## Verification Plan

### Automated Tests

Write a cocotb testbench script `test_interrupt.py`:

1. Load a small firmware image with `trap_vector`. Firmware prints `A`, starts Systolic, then executes `wfi`.
2. The testbench checks that `wfi` is effective while Systolic runs, for example by observing that Snitch PC does not advance or that a sleep flag is set.
3. When Systolic asserts `done`, the testbench checks that Snitch exits `wfi` and prints `B` from the interrupt handler.
4. If Snitch prints `B` and continues execution, the interrupt mechanism passes.
