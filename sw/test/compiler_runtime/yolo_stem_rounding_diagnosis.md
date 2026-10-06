# YOLO320 stem byte-exact mismatch

## Scope and frozen artifacts

The full graph completed 3,811 commands but failed the TensorFlow Lite
`BUILTIN_REF` byte comparison. Prefix 23 isolates the first eight Conv output
tiles (40 image rows) and the first fused SiLU LUT. No RTL or compiler fix is
included in this diagnosis.

- NAI SHA256: `23d24dc617e6c9fe33de08149c6c9b4af62b3e4d94b94d4a7f7b7d38e1dd5c6a`
- TFLite SHA256: `2330cae180c11c0d3203883efe63ef677935867754dc6a81d4c88c773436921f`
- ELF SHA256: `5003c93460c376b5b8c32c18cf2053168eaf1aa980bbbfc4f947d7f437976035`
- Local evidence: `hw/rtl/cluster/tb/sim/test_yolo_stem_byte_exact/`.

`test_yolo_stem_byte_exact.py` decodes output addresses/pitches from the saved
ABI, ignores padded channels, and captures each checkpoint at PMU command end.
It writes actual/expected INT8 bytes, `byte_exact.json`, PMU CSV and a final
command-23 snapshot. Its current numerical failure is expected and must not be
weakened to a tolerance-based PASS.

## Measured explanation

| Comparison | Mismatches |
| --- | ---: |
| RTL Conv vs TensorFlow, first 102,400 elements | 100 (all magnitude 1) |
| RTL Conv vs reconstructed accumulator + NAI single-round requant | 0 |
| Reconstructed TFLite double-round requant vs TensorFlow | 0 |
| Full-Q31 precision single-round vs TensorFlow | 100 |
| NAI single-round vs full-Q31 precision single-round | 0 |
| AFU output vs compiled LUT applied to observed Conv | 0 |

The compiled bias equals the TFLite bias with input zero-point correction
folded in. The software reconstruction uses the original TFLite weights and
input bytes, not the recorded RTL output. Thus the observed mismatch is fully
explained by requantization semantics on this prefix, not lost DMA data or a
separate LUT error. Qparam precision reduction contributes no additional
differences on this measured set.

Example: spatial position 27, channel 14 has biased accumulator 4,155.
NAI multiplier/shift `3359085 / 31` produces 11 after output zero-point 5;
TFLite Q31 multiplier `1719851499`, exponent -9, produces 12 using rounded
high-multiply followed by rounded division by 512. A negative example is
accumulator -29,088: NAI -40, TFLite -41.

The fused LUT's 31/51,200 mismatches against the independent activation tensor
are consistent with the already differing Conv input; maximum deviation is 2.
This explains the **first** mismatch, not necessarily every downstream mismatch
in the full graph. Full-model byte-exact verification remains required.

## Reproduce offline

```sh
PYTHONPATH=sw/test/compiler_runtime:hw/rtl/cluster/tb/tests \
  python3 -m unittest sw/test/compiler_runtime/test_yolo_byte_exact_helpers.py
python3 sw/test/compiler_runtime/analyze_yolo_stem_requant.py \
  --checkpoints hw/rtl/cluster/tb/sim/test_yolo_stem_byte_exact \
  --model hw/rtl/cluster/tb/sim/test_full_models_cursor/yolo320/dma_events.nai \
  --source /home/dev01/neural-compiler/test/model/yolov8n_320_int8.tflite
```

The analyzer validates the recorded model/source hashes and writes
`requant_analysis.json`. No Verilator run is needed to repeat this analysis.

## Proposed next change (requires RTL permission)

Add an explicit TensorFlow-compatible double-rounding mode to systolic requant,
with compiler-emitted Q31 multiplier/exponent and firmware configuration.
Preserve the existing single-round mode for its current ABI/users. A general
semantic fix is needed; do not tune biases/scales for the observed input values.
Verify positive/negative ties and saturation in unit/block tests, rerun this
prefix, then continue full-model byte-exact checkpoints. Do not assume that
fixing the stem alone proves the whole graph correct.
