# NPU Cluster Boot Flow

**Scope**: Per-cluster Snitch boot through host AXI4-Lite  
**Baseline**: Phase 3B-A

---

## 1. Actors

| Actor | Role |
|-------|------|
| Host / testbench | Owns reset, writes firmware into I-TCM, observes completion |
| AXI4-Lite slave | Converts host writes into OBI requests |
| I-TCM arbiter | Arbitrates host boot writes vs. Snitch instruction fetch |
| I-TCM SRAM | Holds firmware at `0x1000_0000` |
| Snitch Core | Executes firmware and programs DMA/compute CSRs |
| MMIO / CSR block | Exposes DMA and Systolic Controller registers |

---

## 2. Reset and Firmware Load

```text
Host keeps rst_ni low
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

1. Testbench asserts reset so Snitch cannot fetch unstable instructions.
2. Host loads `sw/test/boot/boot.bin` or `sw/test/matmul/matmul.bin` through AXI4-Lite.
3. AXI4-Lite writes are converted into OBI writes.
4. I-TCM arbiter grants bootloader writes into local I-TCM.
5. Firmware image is now resident in L1 I-TCM.

---

## 3. Snitch Release

```text
Host de-asserts rst_ni
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
3. Write success signature `0xDEADBEEF` to the test-observed location.

For the Matrix Engine test:

1. Poll D-TCM start flag from testbench.
2. Read `dim_m` from D-TCM.
3. Program DMA CSRs to load weights from L2 to I-TCDM.
4. Program DMA CSRs to load IFM from L2 to I-TCDM.
5. Program Systolic Controller CSRs.
6. Wait for `REG_SYS_DONE`.
7. Program DMA CSRs to copy OFM from O-TCDM back to L2.
8. Set D-TCM done flag for the testbench.

---

## 5. Boot Success Criteria

| Test | Success signal |
|------|----------------|
| `test_snitch_boot.py` | Signature value equals `0xDEADBEEF` |
| `test_matmul.py` | Done flag in D-TCM becomes `1`, then OFM matches golden data |

---

## 6. Known Log Noise

During reset or before firmware load, Snitch can print illegal-instruction messages because I-TCM contains zeros. These messages are expected in current simulation as long as the final pass criteria are met.
