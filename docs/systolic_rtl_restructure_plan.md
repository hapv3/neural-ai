# Systolic RTL Restructure Plan

## 1. Purpose and scope

This plan restructures the two remaining monolithic systolic control blocks:

- `hw/rtl/systolic/systolic_controller.sv`
- `hw/rtl/systolic/conv_linebuf_stream_packer.sv`

The work is structural. It must not change the command ABI, programmed register
semantics, arithmetic, scheduling policy, memory request order, externally
visible latency, or the `npu_systolic_array` implementation. Performance changes
are separate follow-up work and require explicit review.

The target is not a fixed line count. A module boundary is accepted only when
each sequential state element has one owner, data crosses the boundary through
valid/ready channels, lifecycle control crosses through explicit events, and
each OBI port has one owner or one named arbiter.

## 2. Baseline and invariants

The numeric baseline at RTL commit `6081ced` is:

| Suite | Invocations | Start-to-done cycles |
|---|---:|---:|
| Linebuffer block | 39 | 25,641 |
| Controller core | 9 | 5,903 |
| Controller execution modes | 4 | 428 |
| Controller linebuffer matrix | 9 | 13,481 |
| Controller pool matrix | 2 | 570 |
| Controller binary Add/Sub/Mul | 3 | 353 |
| Controller focused total | 27 | 20,735 |

Every structural increment must preserve functional output, OBI transaction
counts/order, PMU event counts, and the applicable cycle totals. Any intentional
difference stops the restructure and is reviewed as a performance or functional
change.

The following invariants apply throughout the migration:

1. The parent never writes state owned by a child.
2. A child receives `start`, `tile_start`, `prefetch_start`, or an equivalent
   one-cycle event; it returns `busy`, `done`, and stream status.
3. A producer holds `valid` and payload stable until `ready`.
4. OBI request payload stays stable until grant, and response ownership remains
   unambiguous even when foreground and background fetches overlap.
5. No task may update `_d` state belonging to more than one functional block.
6. Reset values, FIFO depths, SRAM organization, response timing, and debug
   state encoding remain unchanged during the restructure.

## 3. Final hierarchy

```text
systolic_controller
|-- systolic_ctrl_regs                    existing MMIO/shadow register owner
|-- systolic_job_sequencer                job and execution-mode lifecycle
|-- systolic_k_tile_scheduler             tile index, K seed and channel offset
|-- systolic_weight_engine                initial load, preload and weight FIFO
|-- systolic_input_engine                 direct IFM FIFO or linebuffer stream
|   `-- conv_linebuf_stream_packer
|       |-- conv_linebuf_config_decoder
|       |-- conv_linebuf_spatial_scheduler
|       |-- conv_linebuf_fetch_engine
|       |-- conv_linebuf_row_store
|       |-- conv_linebuf_window_engine
|       |-- conv_linebuf_formatter_pipeline
|       `-- conv_linebuf_bypass_engine
|-- npu_systolic_array                    unchanged
|-- depthwise_mac_engine                  existing
|-- systolic_maxpool_engine               existing
`-- systolic_output_drain
    |-- OFM FIFO
    |-- PSUM FIFO and ping-pong PSUM SRAM
    |-- requant and binary post-op routing
    |-- OFM, PSUM and binary address walkers
    `-- OBI-B/OBI-O transaction engines
```

The two named parent files become integration shells. They select streams,
forward lifecycle events, expose debug/performance status, and perform only the
OBI arbitration explicitly assigned to the shell.

## 4. Linebuffer ownership and interfaces

| Block | Solely owned state | Main requests/events | Main responses |
|---|---|---|---|
| Config decoder | None; combinational derived configuration only | Raw active linebuffer configuration | Mode flags, lane/tap map, effective C base, address constants, cache mode |
| Spatial scheduler | Main spatial FSM, output row/column, kernel walk, emitted-vector counters, tile lifecycle | `start`, `tile_start`, child availability/completion | Fetch/window/bypass commands, formatter metadata, `busy`, `done` |
| Fetch engine | Foreground/background fetch FSMs, OBI request state, response metadata FIFO, crossing-beat register, outstanding and pending counts, fetch counter | Row-fetch descriptors, prefetch event, OBI grant/response | OBI request, row-store write/commit events, row ready/completion |
| Row store | SRAM banks, row tags, valid bits, cached channel base and ring/full-cache allocation state | Allocate/invalidate, fetch writes, window reads | Tag hit/ready, bank read data |
| Window engine | Window request/wait state, kernel-column state, slide-window registers and stage-1 metadata | Spatial position, row-store readiness/data, formatter ready | Formatted-vector input stream and spatial completion |
| Formatter pipeline | Existing stage-2/output data and valid state | Window vector stream | Packed row stream |
| Bypass engine | Bypass address, first crossing beat, valid-byte metadata and bypass response state | Bypass descriptor and output ready | OBI request and packed row stream |
| Parent shell | No functional datapath state | Child streams and OBI responses | One priority-defined OBI request and response routing |

The fetch engine owns both foreground and background request generation so the
response metadata FIFO and row-pending bookkeeping cannot be split across
modules. The row store owns stored contents and tags, but it changes a tag only
on explicit allocate, commit, or invalidate events from the fetch/scheduler
side. The bypass engine remains independent because its response assembly never
writes row-store state.

### Linebuffer migration order

1. Extract the pure config decoder and lock its combinational outputs with a
   focused unit test.
2. Move SRAM banks, tags, row validity, and allocation into the row store.
3. Move foreground/background OBI request generation, metadata FIFO, response
   merge, and row pending/completion into one fetch engine.
4. Move window read/slide state and stage-1 generation into the window engine.
5. Move the complete 1x1 direct path into the bypass engine.
6. Move the remaining spatial/tile state into the scheduler and reduce the
   parent to child wiring plus the fetch-versus-bypass OBI arbiter.

This order follows data dependencies from storage and transport toward policy.
It avoids temporarily giving the parent and a child shared ownership of a state
group.

## 5. Controller ownership and interfaces

| Block | Solely owned state/resources | Main requests/events | Main responses |
|---|---|---|---|
| Job sequencer | Top-level job state and array flush lifecycle | Register `start`; engine, scheduler and drain completion | Per-engine start/tile/prefetch events, array phase control, job `done` |
| K-tile scheduler | Tile index, K seeds, channel offset and next-tile decision | Job start, advance request | Current/next tile metadata, `has_next`, advance acknowledgement |
| Weight engine | Weight pointer/counters, initial and preload request state, weight FIFO, depthwise weight bank | Load/preload start, tile metadata, OBI-W response | OBI-W master, array weight stream, preload/load completion, depthwise weights |
| Input engine | IFM pointer/counters, IFM FIFO, linebuffer instance and linebuffer prefetch lifecycle | Feed start, tile metadata, array/side-engine readiness | OBI-I master, normalized input stream, feed/prefetch completion and linebuffer debug |
| Output drain | Output/PSUM pointers and columns, drain counters/FSM, OFM and PSUM FIFOs, PSUM SRAM, requant/binary pipelines, binary operand stream | Normalized result stream and tile metadata | OBI-B/OBI-O masters, result ready, drain busy/done and drain debug |
| Parent controller | No low-level datapath state | Child status and streams | Engine interconnect, execution-mode stream selection, public ports |

Output address walkers are part of the output drain rather than standalone
helpers: their advancement is atomic with an accepted write, invalid requant
result, external PSUM read, or PSUM-buffer transition. Keeping those operations
together preserves request timing and eliminates cross-module update enables.

The output drain accepts one normalized result channel. The selected producer
is the systolic array, depthwise engine, or max-pool engine. Producer-specific
quantization metadata is carried with the start/tile descriptor; mode-specific
writeback code does not remain in the top-level sequencer.

### Controller migration order

1. Extract the complete output drain, including all FIFOs/SRAM and OBI-B/OBI-O
   traffic. This removes the largest independent state domain first.
2. Extract the complete weight engine, including initial load and overlapped
   next-tile preload, so OBI-W has one owner.
3. Extract the input engine around the direct FIFO and the complete linebuffer
   subsystem, so OBI-I has one owner.
4. Extract the K-tile scheduler after weight/input/drain tile interfaces are
   stable.
5. Replace the residual controller FSM with the job sequencer and reduce
   `systolic_controller` to integration/routing.

## 6. Increment and verification gates

Each ownership migration is one reviewable increment. The minimum gate is:

1. `git diff --check`.
2. Verilator lint for the new child and its parent.
3. Unit tests for the child, including reset, backpressure, OBI grant stalls,
   delayed responses, crossing beats, and final/tail conditions as applicable.
4. Parent block regression for every mode that uses the migrated state.
5. Comparison against the baseline cycle, PMU, and OBI transaction numbers.
6. An English commit message containing only that verified increment.

Cluster simulation is run only at two major endpoints: after the linebuffer
shell is complete and after the controller shell is complete. Full-model
compiler/Verilator work remains paused until this restructure is closed.

## 7. Completion criteria

The restructure is complete when:

- all state appears in exactly one ownership row above;
- both parent files contain only integration, routing, named arbitration, and
  public debug/performance mapping;
- no cross-functional side-effect tasks remain;
- `npu_systolic_array.sv` is unchanged from the approved baseline;
- all unit, block, controller-matrix, cycle-trace, lint, and endpoint cluster
  regressions pass;
- architecture documentation reflects the implemented hierarchy and interfaces.
