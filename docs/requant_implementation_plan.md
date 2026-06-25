# Requantization Implementation Plan

**Status:** RTL first implementation integrated in systolic controller drain path
**Scope:** Define exact arithmetic, document the implemented hardware contract, and keep Spatz/C parity as follow-up verification rather than a blocker.

---

## 1. Why Requantization Is Required

Systolic GEMM/Conv computes `INT8 × INT8` and accumulates into `INT32`. Most follow-on neural-network layers expect compact INT8 activations, not raw INT32 accumulators. Requantization is therefore required to turn each `INT32` accumulator into a clipped `INT8` activation with the correct quantization parameters.

Requantization is high priority for:

1. **YOLO/CNN Conv chains:** Conv output must become INT8 before the next Conv/GEMM consumes it.
2. **Bandwidth and capacity:** INT32 OFM is 4× larger than INT8 OFM, increasing TCDM pressure and DMA writeback cost.
3. **Fused post-processing:** Bias, BN/folded scale, clamp, ReLU/ReLU6, and signed/unsigned output ranges can be handled in one pass.
4. **Graph scheduler correctness:** The scheduler needs a precise operator contract so hardware/Spatz/Snitch fallback all produce identical bytes.

The current implementation intentionally moves directly to RTL fusion because YOLO/CNN chaining benefits immediately from avoiding the intermediate `M × 32 × INT32` activation writeback. Spatz/C parity remains useful as a fallback and cross-check, but it is no longer the required first phase.

---

## 2. Exact Arithmetic Contract

### 2.1. Inputs and outputs

For each output element/channel:

| Symbol | Type | Meaning |
|--------|------|---------|
| `acc` | signed `int32` | Systolic accumulator output. |
| `bias` | signed `int32` | Optional fused bias. Use `0` when disabled. |
| `multiplier` | signed `int32` | Fixed-point scale multiplier. Usually non-negative for normal quantization. |
| `shift` | unsigned `uint8` | Right shift amount after multiply. Valid range `0..31` for Phase 0. |
| `zero_point` | signed `int32` | Output zero-point. Use `0` for symmetric signed INT8. |
| `clamp_min` | signed `int32` | Final minimum output value. |
| `clamp_max` | signed `int32` | Final maximum output value. |
| `out` | `int8` or `uint8` byte | Final stored activation byte. |

The canonical formula is:

```text
biased      = int64(acc) + int64(bias)
scaled64    = biased * int64(multiplier)
rounded64   = round_shift_right(scaled64, shift)
with_zp     = rounded64 + int64(zero_point)
clamped     = clamp(with_zp, clamp_min, clamp_max)
out_byte    = low 8 bits of clamped after range guarantee
```

Phase 0 requires every implementation to match this formula exactly.

### 2.2. Rounding mode

Use **round-to-nearest, ties away from zero** for signed values:

```text
round_shift_right(x, 0) = x

round_shift_right(x, s) for s > 0:
  offset = 1 << (s - 1)
  if x >= 0: return (x + offset) >> s
  else:      return -(((-x) + offset) >> s)
```

Rationale:

- Deterministic and simple for RTL, Spatz, and C fallback.
- Avoids systematic negative bias from plain arithmetic shift.
- Handles negative accumulators symmetrically.

### 2.3. Saturation and output ranges

The final clamp range defines both activation and output dtype semantics:

| Use case | `clamp_min` | `clamp_max` | Notes |
|----------|-------------|-------------|-------|
| Signed INT8 | `-128` | `127` | Default Conv output. |
| ReLU signed INT8 | `0` | `127` | Fuse ReLU. |
| ReLU6 signed INT8 | quantized `0` | quantized `6` clipped to `127` | Requires model-provided scale. |
| Unsigned INT8 | `0` | `255` | Store byte as `uint8`; consumers must agree. |

Phase 0 default is signed INT8. Unsigned output is allowed only when the layer explicitly sets `clamp_min=0`, `clamp_max=255`, and the tensor dtype/layout marks the output as unsigned.

### 2.4. Per-tensor vs per-channel

Both are required in the spec, but implementation can stage them:

1. **Per-tensor:** one `bias/multiplier/shift/zero_point/clamp` tuple for all elements. This is enough for early graph bring-up.
2. **Per-channel:** one tuple per output channel `n` in `N=32`. This is required for good YOLO/CNN accuracy.

For systolic GEMM32 output layout `M × 32`, channel index is:

```text
channel = element_index % 32
```

For contiguous tensor storage, the qparam table should be stored as arrays of 32 entries:

```c
typedef struct {
    int32_t bias[32];
    int32_t multiplier[32];
    uint8_t shift[32];
    int32_t zero_point[32];
    int32_t clamp_min;
    int32_t clamp_max;
    uint32_t flags;
} npu_requant_qparams32_t;
```

`clamp_min/max` are shared in Phase 0. Per-channel clamp can be added later if a model requires it.

### 2.5. Overflow rules

- `acc + bias` must be evaluated in `int64`.
- `biased * multiplier` must be evaluated in `int64`.
- `shift > 31` is invalid in Phase 0 and must fail the test/operator rather than silently producing a value.
- `multiplier < 0` is not expected for normal quantization. The golden model supports it, but RTL may reject it unless a real model requires it.
- The final value must be clamped before byte packing; no wraparound is allowed before clamp.

---

## 3. Golden Model

The Python golden model below is the reference for every Spatz, C, and future RTL implementation.

```python
def round_shift_right(value: int, shift: int) -> int:
    if shift < 0 or shift > 31:
        raise ValueError("shift must be in 0..31")
    if shift == 0:
        return value
    offset = 1 << (shift - 1)
    if value >= 0:
        return (value + offset) >> shift
    return -(((-value) + offset) >> shift)


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def requant_one(acc: int, bias: int, multiplier: int, shift: int,
                zero_point: int, clamp_min: int, clamp_max: int) -> int:
    biased = int(acc) + int(bias)
    scaled = biased * int(multiplier)
    rounded = round_shift_right(scaled, shift)
    with_zp = rounded + int(zero_point)
    return clamp(with_zp, clamp_min, clamp_max)


def requant_mx32(acc_values, qparams, per_channel: bool = True):
    output = []
    for index, acc in enumerate(acc_values):
        channel = index % 32 if per_channel else 0
        output.append(requant_one(
            acc=acc,
            bias=qparams["bias"][channel],
            multiplier=qparams["multiplier"][channel],
            shift=qparams["shift"][channel],
            zero_point=qparams["zero_point"][channel],
            clamp_min=qparams["clamp_min"],
            clamp_max=qparams["clamp_max"],
        ))
    return output
```

### Required golden fixtures

Every implementation test must include:

1. **Zero path:** `acc=0`, `bias=0`, all zero-point/clamp variants.
2. **Positive rounding:** values just below/at/above half-LSB.
3. **Negative rounding:** values just below/at/above half-LSB with negative accumulators.
4. **Clamp edges:** outputs below min, exactly min/max, above max.
5. **Bias extremes:** large positive/negative bias that still fits int64 multiply.
6. **Per-channel variation:** 32 channels with different multipliers/shifts/bias.
7. **Unsigned output:** clamp `0..255`, byte compare as unsigned.
8. **Invalid shift:** `shift > 31` must fail explicitly.

---

## 4. Implemented RTL Contract

The first RTL implementation is integrated at the systolic controller output drain:

```text
systolic_array -> OFM FIFO INT32 row -> requant_pipeline -> packed INT8 TCDM row
                                      \-> raw INT32 4-port bypass
```

Implemented files:

- `hw/rtl/systolic/requant_pipeline.sv`: 32-lane combinational requant packer.
- `hw/rtl/systolic/systolic_controller.sv`: selects raw INT32 bypass or packed INT8 requant mode.
- `hw/rtl/cluster/cluster_ctrl_regs.sv`: exposes per-channel qparams through MMIO.
- `sw/lib/hal_systolic.c`: provides raw GEMM and requant GEMM HAL entry points.
- `sw/test/systolic_requant`: firmware test for packed output mode.
- `hw/rtl/cluster/tb/tests/test_systolic_requant.py`: cocotb byte-exact golden check.

### 4.1. MMIO register map

All offsets are relative to `cluster_ctrl_regs` at `0x2000_0000`.

| Offset | Name | Meaning |
|--------|------|---------|
| `0x0120` | `REG_RQ_CTRL` | Bit 0 enables RTL requant drain mode. |
| `0x0124` | `REG_RQ_CMIN` | Shared signed clamp minimum. |
| `0x0128` | `REG_RQ_CMAX` | Shared signed clamp maximum. |
| `0x0200 + 4*n` | `REG_RQ_BIAS_BASE[n]` | Per-channel signed bias. |
| `0x0280 + 4*n` | `REG_RQ_MULT_BASE[n]` | Per-channel signed multiplier. |
| `0x0300 + 4*n` | `REG_RQ_SHIFT_BASE[n]` | Per-channel right shift, valid `0..31`. |
| `0x0380 + 4*n` | `REG_RQ_ZP_BASE[n]` | Per-channel signed output zero-point. |

Reset defaults keep legacy behavior safe:

- Requant disabled.
- Clamp range `[-128, 127]`.
- Multiplier `1`, bias `0`, shift `0`, zero-point `0`.

### 4.2. Output layout

Raw bypass mode is unchanged:

```text
row output = 32 x INT32 = 128 bytes
write path = 4 x 256-bit OBI output ports
o_ptr increment = 128 bytes
```

Requant mode writes compact activation rows:

```text
row output = 32 x INT8 = 32 bytes
write path = output OBI port 0 only
o_ptr increment = 32 bytes
byte packing = channel n stored at byte n within the 256-bit row
```

This mode is intended for Conv/GEMM layers where the next consumer expects INT8 activations. Operators that need raw accumulators must keep requant disabled.

## 5. Follow-Up Roadmap

### Phase 1/P0 — RTL correctness and legacy bypass protection

Goal: prove the fused hardware path and protect existing INT32 behavior.

- Keep `systolic_gemm32()` explicitly disabling requant before raw GEMM.
- Use `systolic_gemm32_requant()` only when the caller wants packed INT8 rows.
- Compare RTL output byte-for-byte with the Python golden formula.
- Keep independent systolic and matmul tests comparing full INT32 output.

Acceptance:

- `test_systolic_requant` passes for multiple boundary `M` values.
- `test_independent_systolic` and `test_matmul` still pass in raw bypass mode.

### Phase 2/P0 — Spatz/C fallback parity

Goal: unblock graph correctness without new RTL risk.

- Extend `spatz_requant_i32_to_i8` to match the golden formula.
- Add bias and zero-point inputs.
- Add per-channel mode for `M × 32` systolic output.
- Add scalar C fallback using the same formula for debug and unsupported paths.
- Add dedicated firmware/cocotb output tests with exact byte comparison.

Acceptance:

- `sw/test/spatz_ops` covers per-tensor and per-channel requant.
- Python golden and firmware output match byte-for-byte.
- Graph scheduler only uses requant modes that have passed tests.

### Phase 3/P0 — Graph integration

Goal: make micro-model E2E correct before optimizing bandwidth.

- Use Spatz/C requant after `OP_SYSTOLIC_GEMM32`.
- Keep systolic output as INT32 in O-TCDM for now.
- Write INT8 activation to a separate TCDM buffer.
- Use exact golden model in `test_micro_yolo_e2e`.

Acceptance:

- Micro-YOLO output matches Python golden.
- Existing independent systolic tests still compare full INT32 output.

### Phase 4/P1 — Performance measurement

Goal: prove whether hardware requant is necessary.

Add counters or debug instrumentation:

- Systolic INT32 OFM write cycles.
- Spatz requant active cycles.
- TCDM arbitration stalls during requant.
- DMA writeback bytes and cycles with INT32 vs INT8 output.

Acceptance:

- Baseline document shows the cost of software/Spatz requant.
- Hardware requant is only approved if it removes a measured bottleneck.

### Phase 5/P2 — Hardware requant extensions

Recommended architecture if RTL is needed:

- Do **not** use an OBI wrapper that grants the controller after downstream writeback. That creates fragile handshake semantics.
- Prefer integrating requant at the systolic output drain path, after OFM FIFO pop and before TCDM write packing.
- The controller should know whether it writes INT32 rows or packed INT8 rows, so `o_ptr` increments and `done` semantics remain explicit.
- Keep bypass mode for debug and special layers that require raw INT32.

Potential RTL structure:

```text
systolic_array -> OFM FIFO INT32 row -> requant packer -> TCDM write port(s)
                                      \-> bypass INT32 4-port write
```

First RTL version already covers:

- Support `M × 32` only.
- Support per-channel qparams for 32 output channels.
- Support one packed 256-bit INT8 write per output row.
- Keep raw INT32 bypass path unchanged.
- Add independent `test_requant_pipeline` if block-level regressions are introduced.

---

## 6. Current Plan Assessment

The original requirement is valid, but the previous OBI Stream Interceptor plan should not be implemented as-is.

Issues corrected in this revision:

- Fixed broken accuracy section and missing arithmetic definition.
- Replaced ambiguous rounding with exact `round-to-nearest, ties away from zero`.
- Added bias, zero-point, signed/unsigned clamp, per-channel indexing, and overflow rules.
- Removed stale MMIO example outside the current `0x2000_xxxx` control aperture.
- Replaced the fragile “lie about grant” interceptor concept with an explicit future RTL recommendation.
- Switched to RTL-first implementation while retaining Spatz/C parity as a fallback and cross-check target.

---

## 7. Verification Gates

Minimum gates before scheduler dependency:

```bash
make -C sw/test/systolic_requant
make -C sw/test/independent_systolic
make -C sw/test/matmul
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_systolic_requant
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_systolic
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul
```

Fallback parity gates:

```bash
make -C sw/test/spatz_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```

Acceptance rule: every output byte must match the Python golden model exactly. Any unsupported qparam mode must fail explicitly and must not be used by the graph scheduler.
