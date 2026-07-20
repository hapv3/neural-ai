# Code Review Report: AFU and Requantization Blocks

This review covers the RTL source for `requant_pipeline.sv` and `afu_core.sv`, with optimization suggestions for performance and synthesis.

---

## 1. Module `requant_pipeline.sv`

> [!SUCCESS]
> **Strengths:**
> - The logic matches the standard TensorFlow Lite requantization flow: add bias, multiply by multiplier, right shift with rounding, add zero point, and clamp.
> - The `for` loop unrolls parallel hardware for all 32 Systolic Array columns.

> [!WARNING]
> **Timing and area risk:**
> The current source is written as a fully combinational `always_comb` block. In one clock cycle, the circuit must perform 32 64-bit additions, 32 64-bit multiplications, 32 barrel shifts, and 32 clamp comparison groups.
>
> **Synthesis consequences:**
> - High clocks such as 500 MHz are unlikely because the critical path is too long.
> - The expression `scaled = biased * 64'($signed(multiplier_i[i]))` pushes the synthesizer toward very large 64x64 multipliers, consuming excessive DSP resources.

> [!TIP]
> **Action items:**
> 1. **Pipeline the datapath:** Add at least 2 to 3 `always_ff` register stages between steps, for example Stage 1 (add bias + multiply) -> Stage 2 (shift + add zero point) -> Stage 3 (clamp).
> 2. **Constrain multiplication width:** `biased` needs roughly 33 bits and `multiplier` is 32 bits. Cast the multiply as `34-bit * 32-bit` instead of `64 * 64` to save DSP resources:
>
> ```verilog
> logic signed [33:0] biased_34;
> biased_34 = 34'($signed(acc_i[i])) + 34'($signed(bias_i[i]));
> scaled = 64'(biased_34) * 64'($signed(multiplier_i[i])); // Synthesis tools can optimize this better.
> ```

---

## 2. Module `afu_core.sv`

> [!SUCCESS]
> **Strengths:**
> - The FSM separates the two pipeline stages (`ST_PROCESS` and Stage 2) clearly.
> - The pipeline-hazard fix that stores SRAM `rdata` when S2 stalls is robust and correctly handles the 1-cycle `tc_sram` latency.
> - Data-unpacking modes `MODE_8BIT`, `MODE_16BIT`, and `MODE_32BIT` are supported cleanly.

> [!CAUTION]
> **Logic-area risk in input byte selection:**
> Line 86, `assign shift_in = in_buf_q >> {in_off_s1, 3'd0};`, asks synthesis to build a **256-bit barrel shifter** with a dynamic shift up to 31 bytes. This shifter consumes a large amount of mux logic and can significantly lengthen the Stage 1 critical path.

> [!TIP]
> **Action item:**
> The logic is functionally correct, but for edge silicon area/timing it is better to replace the 256-bit shifter with a small mux array that selects only the required `LUT_LANES` bytes, for example 4 bytes, instead of shifting all 32 bytes:
>
> ```verilog
> // Example direction: select the required bytes instead of shifting all 256 bits.
> for (genvar i = 0; i < LUT_LANES; i++) begin
>     logic [4:0] byte_idx;
>     assign byte_idx = in_off_s1 + 5'(i);
>     assign lut_idx_s1[i] = in_buf_q[byte_idx * 8 +: 8];
> end
> ```
>
> Some tools may require minor syntax changes for variable array indexing, but the core idea is to mux only the required bytes instead of shifting 32 bytes.

## Review Summary

The logic is structurally sound and no serious data-corruption bug is apparent from this review. The main silicon-readiness issue is timing: `requant_pipeline` should be split into additional pipeline stages before synthesis.
