# NPU Cluster Boot Flow

**Scope**: Per-cluster Snitch boot through host AXI4-Lite  
**Baseline**: Phase 3B-A with interrupt-driven test completion

---

## 1. Actors

| Actor | Role |
|-------|------|
| Host / testbench | Owns reset/fetch-enable, writes firmware into I-TCM, observes completion IRQ |
| AXI4-Lite slave | Converts host writes into OBI requests |
| I-TCM arbiter | Arbitrates host boot writes vs. Snitch instruction fetch |
| I-TCM SRAM | Holds firmware at `0x1000_0000` |
| Snitch Core | Executes firmware and programs DMA/compute CSRs |
| MMIO / CSR block | Exposes DMA, Systolic Controller, AFU, and interrupt registers |
| Interrupt Controller | Converts firmware `HOST_NOTIFY` writes into cluster `irq_o` |

---

## 2. Reset and Firmware Load

```text
Host keeps fetch_enable_i low
  |
  v
Host writes firmware bytes to 0x1000_0000+
  |
  v
AXI4-Lite Slave
  |
  v
AXI-to-OBI bridge
  |
  v
I-TCM Arbiter
  |
  v
I-TCM SRAM
```

1. Testbench holds `fetch_enable_i=0` so Snitch cannot fetch before firmware load.
2. Host loads `sw/test/boot/boot.bin` or `sw/test/matmul/matmul.bin` through AXI4-Lite.
3. AXI4-Lite writes are converted into OBI writes.
4. I-TCM arbiter grants bootloader writes into local I-TCM.
5. Firmware image is now resident in L1 I-TCM.

---

## 3. Snitch Release

```text
Host asserts fetch_enable_i
  |
  v
Snitch reset PC = 0x1000_0000
  |
  v
I-fetch from I-TCM
  |
  v
Firmware begins execution
```

Snitch fetches through its instruction OBI path. Because I-TCM is local L1 memory, instruction fetch is independent from Shared Data TCDM traffic.

---

## 4. Firmware Responsibilities

For the boot smoke test:

1. Initialize stack/private data in D-TCM.
2. Optionally exercise DMA/AFU/compute paths.
3. Write success signature `0xDEADBEEF` to `NPU_IRQ_HOST_NOTIFY`.

For the Matrix Engine test:

1. Start immediately after fetch release.
2. Program DMA CSRs to load weights from L2 to I-TCDM.
3. Program DMA CSRs to load IFM from L2 to I-TCDM.
4. Program Systolic Controller CSRs.
5. Wait for `REG_SYS_DONE`.
6. Program DMA CSRs to copy OFM from O-TCDM back to L2.
7. Write `0xDEADBEEF` to `NPU_IRQ_HOST_NOTIFY`.

---

## 5. Interrupt Completion Path

```text
Firmware running on Snitch
  |
  | OBI write through Snitch D-Bus
  v
MMIO decode @ 0x2000_2000
  |
  v
npu_interrupt_ctrl.NPU_IRQ_HOST_NOTIFY
  |
  | latch HOST_STATUS, set EXT_PENDING
  v
cluster irq_o
  |
  v
Host / cocotb observes completion
```

The completion path is deliberately separate from D-TCM:

1. Firmware may keep private debug words in D-TCM for its own diagnosis.
2. Testbench must not preload start flags or poll pass/fail through D-TCM.
3. Active cluster tests now use `irq_o` as completion and compare output data in L2/TCDM.
4. Current host AXI-Lite path remains I-TCM-only; it does not access D-TCM or IRQ MMIO.

### Interrupt Registers Used During Boot Tests

| Register | Address | Usage |
|----------|---------|-------|
| `NPU_IRQ_HOST_NOTIFY` | `0x2000_2018` | Firmware writes `0xDEADBEEF` pass or `0xBADxxxxx` fail/progress code. |
| `NPU_IRQ_HOST_STATUS` | `0x2000_201c` | Internal status latch for future host-control path; not read over host AXI today. |
| `irq_o` | top-level pin | Completion signal observed by cocotb/host. |

---

## 6. Internal Engine Interrupt Path

The same controller also has an internal interrupt domain for future event-driven firmware:

```text
DMA/Systolic/AFU/Spatz done
  |
  v
INT_PENDING bit set in npu_interrupt_ctrl
  |
  | if enabled by INT_ENABLE
  v
snitch_irq_o.mcip
  |
  v
Snitch trap / WFI wakeup path
```

This path is wired in RTL, but current firmware still polls selected `REG_DONE` bits for simple deterministic sequencing. Replacing those loops with `wfi` and a trap handler is future work.

Current AFU firmware already enables `NPU_IRQ_SRC_AFU`, waits for AFU done, and
checks that the AFU event is latched in `INT_PENDING`. It still reports final
test completion to the host through `NPU_IRQ_HOST_NOTIFY`.

---

## 7. Boot Success Criteria

| Test | Success signal |
|------|----------------|
| `test_snitch_boot.py` | `irq_o` asserts |
| `test_matmul.py` | `irq_o` asserts, then OFM matches golden data |
| `test_afu_basic.py` | `irq_o` asserts, then AFU e8/e16/e32 TCDM outputs match golden data |

---

## 8. Known Log Noise

`fetch_enable_i` prevents Snitch from fetching before the AXI boot image is resident in I-TCM, so illegal-instruction noise during firmware load should not be part of normal tests.
