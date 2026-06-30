# PMU Smoke Test

## Scenario

This firmware generates deterministic Snitch/TCDM traffic for the host-side PMU
smoke test.

## Target

- Clear and enable PMU counters through the AXI4-Lite host slave port.
- Generate Snitch D-bus traffic with deterministic TCDM load/store loops.
- Snapshot and read PMU counters from Python through the `0x2000_4000` host window.
- Report pass/fail through D-TCM status words and host interrupt.

## Pass Criteria

- Host read of PMU `NUM_COUNTERS` reports at least 32 counters.
- Cycle, Snitch TCDM request, aggregate TCDM request, read request, and write request counters are non-zero.
- Firmware writes `0xDEADBEEF` to `0x10008000` and raises host IRQ.
