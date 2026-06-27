# Conv2D Packed Prepare Performance Estimator

## Scenario

Host-side software model for evaluating the packed-prepare Conv2D performance
path without adding more Conv2D RTL.

## Target

- Estimate current P2.5 legacy RTL path cost with the `tc_sram` cache hit
  latency, cache flush, K-base decode, padding zeros, and direct-mapped cache
  misses.
- Estimate P4 raw-wide read cost with one-segment, two-segment, unaligned, and
  slow rows.
- Estimate P4 packed-buffer cost with prepare/compute double buffering.
- Report broken K-tiles so the `KH` boundary assumption can be checked across
  `IC`, kernel, and padding variants.

## Command

```sh
make -C sw/perf report
make -C sw/perf csv
```

Custom shape format:

```sh
python3 sw/perf/conv2d_feeder_p4_perf.py \
  --shape conv3x3_c48,16,16,48,3,3,1,1,1,1
```

The shape fields are:

```text
name,H,W,IC,KH,KW[,SH,SW[,PH,PW[,DH,DW]]]
```

`DH` and `DW` must be `1`. Dilation greater than `1` is outside the current P4
scope and is rejected by the estimator.

## Modeling Notes

- The estimator is architectural, not cycle-exact RTL simulation.
- Current feeder defaults assume cache hits cost 2 cycles per valid byte and
  misses add 2 cycles for read request/response.
- P4 packed mode assumes `M x 32` rows are already materialized before systolic
  compute; raw NHWC cannot generally use `base + row * 32` unless `IC == 32`
  and the layer/tile is contiguous.
- Padding is deliberately visible in the report because border zero insertion is
  one of the cases most likely to invalidate the fast path.
