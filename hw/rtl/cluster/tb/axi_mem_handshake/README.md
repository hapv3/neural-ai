# AXI simulation-memory handshake regression

This simulation-only test reproduces a stalled final read beat with the cluster's
1 ns clock and a 57-beat, 256-bit burst. Five backpressure phases each compare a
zero-delay negative control against 100 ps application / 400 ps acquisition
delays. The positive controls check every data byte, stable stalled payload,
accepted beat count, and exactly one accepted RLAST.

```sh
cd hw/rtl/cluster/tb/axi_mem_handshake
make -j10 lint
make -j10 run
```

Artifacts default to `../sim/axi_mem_handshake`; override `SIM_DIR` to retain a
separate run. PASS requires all positive controls to pass and all zero-delay
controls to expose the race. This is not an NPU/model accuracy test.

Keep a 1 ns time unit and 1 ps precision. Delay parameters must be `realtime`:
integer `time` parameters round these sub-nanosecond delays to zero at a 1 ns
time unit. Changing precision to 1 ns would also remove the intended phases.

## Cluster replay evidence

The saved 224-command isolated range (source commands 423..628, plus 18 leading
barriers) was replayed with identical model and firmware bytes before and after
the timing correction:

- NAI SHA256: `0a6f5d17bbf7a16afe10a7adc7ab4ab76b351514f6274e51fbc9226e01d89a04`
- ELF SHA256: `761bf36f025f3d309466e307bd42806c6d7de28224bd5ff165f82a58de6cf3de`
- Before: DMA job 9 accepted only 56/57 read beats; no RLAST handshake or
  completion. Timeout at command 32 with DMA still busy.
- After: 57 read beats, 57 OBI writes/responses, one RLAST handshake and one
  completion for job 9. The 1,000,000-cycle timeout reached command 50 with DMA
  idle and firmware copying its retained command buffer.

Local evidence lives under `../sim/test_command_refill_dma_serialization` and
`../sim/test_command_refill_dma_timing`. These results validate this handshake
fix, not completion or byte-exact output of the full model.
