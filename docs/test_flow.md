# NPU Cluster Test Flow

**Scope**: Current verification flows for Phase 3B-A  
**Baseline**: Matrix Engine integrated into `npu_cluster`

---

## 1. Test Pyramid

| Level | Test | Target |
|-------|------|--------|
| Unit | `hw/rtl/systolic/tb/test_systolic.py` | Standalone 32×32 Systolic Array |
| Cluster boot | `hw/rtl/cluster/tb/tests/test_snitch_boot.py` | I-TCM boot path, Snitch execution, signature write |
| Cluster compute | `hw/rtl/cluster/tb/tests/test_matmul.py` | Firmware-controlled DMA + Systolic + OFM writeback |

---

## 2. Standalone Systolic Flow

```text
Python test item
  |
  +--> Generate weights / IFM
  +--> Compute Python golden output
  |
Driver
  |
  +--> Reset DUT
  +--> Load 32 weight rows
  +--> Stream IFM rows
  |
Monitor
  |
  +--> Capture ofm_valid_o
  |
Scoreboard
  |
  +--> Compare 32 INT32 columns
```

Pass criteria:

- Scoreboard captures at least one OFM output.
- `last_error_count == 0`.

---

## 3. Snitch Boot Flow Test

```text
test_snitch_boot.py
  |
  +--> Assert reset
  +--> Load boot.bin into I-TCM over AXI4-Lite
  +--> Release reset
  +--> Poll signature
  |
  +--> PASS when signature == 0xDEADBEEF
```

This test validates:

- Host AXI4-Lite → AXI-to-OBI → I-TCM write path.
- Snitch instruction fetch from I-TCM.
- D-Bus/MMIO-visible completion path.

---

## 4. Cluster MatMul Flow

```text
Testbench
  |
  +--> Randomize dim_m in [1, 64]
  +--> Randomize W[32][32] and IFM[dim_m][32]
  +--> Compute OFM_golden = IFM × W
  +--> Write W and IFM to L2 AXI sim memory
  +--> Write dim_m and start flag to D-TCM
  |
Snitch firmware
  |
  +--> DMA W:   L2 0x8000_0000 -> L1 I-TCDM 0x1011_0000
  +--> DMA IFM: L2 0x8000_1000 -> L1 I-TCDM 0x1012_0000
  +--> Start Systolic Controller
  +--> Wait for REG_SYS_DONE
  +--> DMA OFM: L1 O-TCDM 0x1020_0000 -> L2 0x8000_2000
  +--> Set done flag in D-TCM
  |
Testbench
  |
  +--> Poll done flag
  +--> Read OFM from L2
  +--> Compare every INT32 output against NumPy golden
```

Pass criteria:

- All randomized iterations complete.
- Every OFM word matches golden model.
- No firmware timeout/hang.

---

## 5. Useful Commands

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/systolic
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul
```

If firmware binaries were cleaned, rebuild them first:

```bash
make -C sw/boot_app
make -C sw/matmul_app
```

---

## 6. Current Gaps to Add Later

- Concurrent DMA + Systolic + Vector TCDM arbitration test.
- L1→L1 DMA stress test with overlapping bank access patterns.
- Multi-cluster top-level test once Phase 4 starts.
- End-to-end YOLO layer test after tiling and manager orchestration are available.
