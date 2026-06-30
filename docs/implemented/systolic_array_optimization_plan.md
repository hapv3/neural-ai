# Systolic Array Optimization Plan

## OFM Backpressure And FIFO Sizing

### Current Status

- `systolic_controller` currently uses input FIFOs for weight/IFM and an OFM FIFO after the array.
- Current cluster configuration:
  - Weight FIFO: `4 × 32B = 128B`
  - IFM FIFO: `4 × 32B = 128B`
  - OFM FIFO: `8 × 128B = 1KB`
- OFM FIFO entry format is one full output row: `32 × int32 = 1024b = 128B`.
- The array output now has a true `ofm_ready` signal from the controller/FIFO back into the array pipeline.

### Current Backpressure Mechanism

- OFM rows are pushed into the OFM FIFO when `ofm_valid_i && ofm_ready_o`.
- `ofm_ready_o` is driven directly from FIFO space: `!ofm_fifo_full`.
- If `ofm_valid_i && !ofm_ready_o`, the systolic array output pipeline holds its current row stable instead of advancing/dropping data.
- The controller drains one OFM row to O-TCDM using `4 × 256-bit` OBI writes.
- A FIFO pop only occurs when all four OBI write ports grant in the same cycle: `obi_o_gnt_i == 4'b1111`.
- The old indirect `OFM_FIFO_STOP_LEVEL` reserve path is removed from the compute-feed decision.
- Compute feed is now gated by real array pipeline readiness, not by a latency-based FIFO reserve estimate.

### Measured Baseline

- Stress test: independent systolic GEMM32 with `M={1,2,31,32,33,64,128,1024}`.
- Direct `M=1024` stream passes with OFM FIFO depth `8`.
- No-stall high-water with default depth `8`:
  - `usage=7`
  - `write_idx=7`
  - `read_idx=7`
- Synthetic O-TCDM stall injection uses `SYSTOLIC_OTCDM_STALL_PERIOD=6`, `SYSTOLIC_OTCDM_STALL_HOLD=2`.
- Stall sweep verifies full independent systolic output byte-exact for `M={1,2,31,32,33,64,128,1024}`.

| OFM FIFO Depth | Capacity | Max Usage | Full Cycles | Array Backpressure Cycles | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `64` | `8KB` | `63` | `330` | `272` | Pass |
| `32` | `4KB` | `31` | `393` | `326` | Pass |
| `16` | `2KB` | `15` | `453` | `374` | Pass |
| `8` | `1KB` | `7` | `547` | `468` | Pass |
| `4` | `512B` | `3` | `613` | `530` | Pass |

Decision:

- Default OFM FIFO depth is reduced from `64` to `8`.
- Depth `8` saves `7KB` per array instance versus depth `64`, an `87.5%` OFM FIFO capacity reduction.
- Depth `4` passes the current stress suite, but remains an aggressive option until more random and system-level stall scenarios are covered.
- Keep tile `M<=64` as a scheduler fallback, but prefer direct large-`M` streaming when ready/valid is proven.

## Direct Large-M Versus Tiled GEMM

### Direct `M=1024` With OFM FIFO

Benefits:

- Loads weights once for the full `M=1024` GEMM.
- Avoids repeated controller start/wait overhead.
- Better matches CNN/YOLO layers where `M = H × W` can be large, e.g. `32 × 32 = 1024`.
- Exercises the real long-stream datapath instead of hiding issues behind small tiles.

### Tiled `M<=64`

Benefits:

- Useful fallback when output backpressure is not proven.
- Easier to debug because each transaction is short.
- Reduces pressure on OFM buffering.

Cost:

- For `M=1024`, `M=64` tiling requires 16 GEMM calls.
- If weights are reloaded per tile, weight load grows from `32` rows to `16 × 32 = 512` rows.
- More scheduler/control overhead.

Decision:

- Keep direct large-`M` as the target path.
- Keep `M<=64` tiling as a safe fallback option for scheduler/debug.
- OFM ready/valid is now implemented, so direct large-`M` streaming is the preferred path with default OFM FIFO depth `8`.
