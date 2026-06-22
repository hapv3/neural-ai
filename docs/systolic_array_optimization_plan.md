# Systolic Array Optimization Plan

## OFM Backpressure And FIFO Sizing

### Current Status

- `systolic_controller` currently uses input FIFOs for weight/IFM and an OFM FIFO after the array.
- Current cluster configuration:
  - Weight FIFO: `4 × 32B = 128B`
  - IFM FIFO: `4 × 32B = 128B`
  - OFM FIFO: `64 × 128B = 8KB`
- OFM FIFO entry format is one full output row: `32 × int32 = 1024b = 128B`.
- The array output currently has valid-only behavior; there is no true `ofm_ready` signal back into the array pipeline.

### Current Backpressure Mechanism

- OFM rows are pushed into the OFM FIFO when `ofm_valid_i && !ofm_fifo_full`.
- The controller drains one OFM row to O-TCDM using `4 × 256-bit` OBI writes.
- A FIFO pop only occurs when all four OBI write ports grant in the same cycle: `obi_o_gnt_i == 4'b1111`.
- Since the systolic array cannot be stalled directly at the output, the controller applies indirect backpressure by stopping new IFM feed when the OFM FIFO reaches an almost-full threshold.
- With `OFM_FIFO_DEPTH=64`, the controller reserves:
  - `SYS_LATENCY=32`
  - `INPUT_FIFO_DEPTH=4`
  - margin `8`
  - total reserve `44` entries
- Therefore `OFM_FIFO_STOP_LEVEL = 64 - 44 = 20`.
- Once `ofm_fifo_usage >= 20`, the controller stops issuing/popping IFM into the array and leaves 44 entries free for OFM rows already in flight.

### Measured Baseline

- Stress test: independent systolic GEMM32 with `M={1,2,31,32,33,64,128,1024}`.
- Direct `M=1024` stream passes with OFM FIFO enabled.
- Measured OFM FIFO high-water on the direct `M=1024` suite:
  - `usage=20`
  - `write_idx=63`
  - `read_idx=63`
- `write_idx/read_idx=63` only proves the circular FIFO pointers wrapped through the whole depth; it is not occupancy.
- `usage=20` is the relevant occupancy high-water and matches the current stop threshold.

### Why Not Reduce OFM FIFO Below 64 Yet

- Depth `32` is smaller than the current reserve requirement of `44` entries.
- Without true output `ready/valid`, output rows already in flight after IFM stop still need guaranteed storage.
- Reducing below `64` without changing the array output protocol risks overflow under O-TCDM stalls.

### Future Optimization: True OFM Ready/Valid

Goal: add a real `ofm_ready` handshake from the controller/FIFO back into the systolic array output stage.

Expected effect:

- Convert backpressure from indirect IFM stop to direct OFM output stall.
- Remove the need for `SYS_LATENCY + margin` worth of FIFO reserve.
- Reduce OFM FIFO from `64` entries toward:
  - conservative first target: `8` entries = `1KB`
  - aggressive target after stall-injection tests: `4` entries = `512B`
  - theoretical minimum: `1–2` entries if the array output stage can hold data stable under stall

Required correctness contract:

- If `ofm_valid && !ofm_ready`, `ofm_data` must remain stable.
- The array output pipeline must not advance/drop rows while `ofm_ready=0`.
- Backpressure must propagate far enough into the array output/accumulator pipeline, or an elastic output buffer must absorb the remaining in-flight rows.
- The controller must continue popping the OFM FIFO only when all four OBI write ports grant.

Required tests before reducing FIFO further:

- Add O-TCDM write grant stall injection in systolic tests.
- Sweep OFM FIFO depth: `64`, `32`, `16`, `8`, `4`.
- Verify direct `M=1024` output byte-exact against golden.
- Record FIFO high-water and stall cycles for each depth.
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
- Prioritize true OFM ready/valid before reducing OFM FIFO below `64` entries.
