# Retained command-window verification

The streaming dispatcher keeps a cursor into the 2 KiB command buffer. It only
compacts an incomplete tail when refilling, rather than moving every remaining
byte after each command. Affine-loop bodies execute in the retained window;
patches never modify the original model. `g_nai_current_command` identifies the
actual descriptor for progress reporting, including affine children. PMU begin/end
IDs remain logical command IDs.

## Host checks

```sh
make -j10 -C sw/test/compiler_runtime check
make -j10 -C sw/runtime/neural_ai all
```

`test_cmd_stream_window` and its trusted variant cover nine window sizes,
split headers/records, affine patches, source preservation, buffer guards,
nonzero entry offsets, read failures, PMU IDs, and absence of per-command shifts.
The test also passes AddressSanitizer/UndefinedBehaviorSanitizer.

## Cluster evidence

Local logs: `hw/rtl/cluster/tb/sim/test_command_stream_cursor/`.

- `conv.log`: compiled 2x3x33 -> 34-channel pointwise Conv; all 204 output
  bytes match the reference; 6/6 commands, 55,415 PMU cycles.
- `range.log`: saved YOLO commands [423,629), plus 18 leading barriers;
  224/224 commands complete in 386,969 PMU cycles under a 1,000,000-cycle limit.
  This zero-initialized range checks liveness, **not full-model accuracy**.
- Same range with corrected AXI timing but the earlier shift-based dispatcher
  timed out at command 50. This is not a completed-run speedup measurement.
- Firmware `.text`: 26,248 bytes; the 32 KiB ITCM limit is respected.

Verified ELF SHA256:
`5003c93460c376b5b8c32c18cf2053168eaf1aa980bbbfc4f947d7f437976035`.
Range NAI SHA256:
`0a6f5d17bbf7a16afe10a7adc7ab4ab76b351514f6274e51fbc9226e01d89a04`.

## Controlled replay

`NAI_REPLAY_MANIFEST` validates the saved `.nai`, `.elf`, and `YOLO320_RANGE_*`
settings, then replays without recompilation or reinserting padding. By default
the saved ELF is loaded. An explicit `NAI_REPLAY_FIRMWARE_ELF` permits a new
firmware comparison while keeping the model fixed; both original artifact hashes
are still validated. `NAI_DMA_TRACE_CSV` records the actual ELF/model and hashes
next to its event CSV. `NAI_DMA_TRACE_MAX_ROWS` bounds event storage (default
20,000); `NAI_DMA_TRACE_START_CYCLE` controls the observation window.

Full YOLO320/MobileNet byte-exact output verification is a separate gate.
