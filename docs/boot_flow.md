# NPU Cluster Boot Flow

**Scope**: Per-cluster Snitch boot through host AXI4-Lite  
**Baseline**: Phase 3B-A

---

## 1. Actors

| Actor | Role |
|-------|------|
| Host / testbench | Owns reset/fetch-enable, writes firmware into I-TCM, observes completion IRQ |
| AXI4-Lite slave | Converts host writes into OBI requests |
| I-TCM arbiter | Arbitrates host boot writes vs. Snitch instruction fetch |
| I-TCM SRAM | Holds firmware at `0x1000_0000` |
| Snitch Core | Executes firmware and programs DMA/compute CSRs |
| MMIO / CSR block | Exposes DMA and Systolic Controller registers |

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
2. Optionally exercise DMA path.
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

## 5. Boot Success Criteria

| Test | Success signal |
|------|----------------|
| `test_snitch_boot.py` | `irq_o` asserts |
| `test_matmul.py` | `irq_o` asserts, then OFM matches golden data |

---

## 6. Known Log Noise

`fetch_enable_i` prevents Snitch from fetching before the AXI boot image is resident in I-TCM, so illegal-instruction noise during firmware load should not be part of normal tests.
