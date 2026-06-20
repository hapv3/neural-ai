Plan: Spatz Operator Library + Static Graph Scheduler + Micro-YOLO E2E
Summary
Mục tiêu là đưa cluster từ mức test rời rạc sang chạy một micro-YOLO 32×32 end-to-end trên RTL hiện tại.
Compute partition cố định: systolic array xử lý Conv/GEMM INT8→INT32, Spatz xử lý mọi operator vectorizable mà config integer-only hiện tại support, Snitch chỉ làm control/scheduler và fallback biên nhỏ.
Model E2E đầu tiên: input 32×32×3 INT8, Conv3×3 padded K=32 → ReLU/requant → Conv1×1 head K=32,N=32 → requant → raw output tensor. Chưa làm NMS/decode/performance optimization.
Key Changes
Thêm Spatz operator library dạng C-callable RVV assembly/C wrappers:Memory/layout: vector copy/fill, e8/e16/e32 load-store, strided/indexed copy nếu Spatz test pass.
Arithmetic/activation: add/sub/mul/shift, min/max, compare/select, clamp, ReLU, LeakyReLU, requant INT32→INT8.
Reduction/pooling: max/sum primitive để dùng cho pooling hoặc postprocess sau này.
Rule: operator nào dùng RVV instruction mới thì phải có firmware+cocotb data-output test trước khi được scheduler dùng.

Thêm HAL/scheduler firmware tái sử dụng:hal_systolic: wrapper start/wait GEMM32 với M×32 * 32×32.
npu_graph: static graph table compiled vào firmware, không dynamic parser ở phase này.
npu_tensor_t: addr, shape, dtype, layout.
npu_layer_t: op type, input/output/weight/bias/qparam refs, op params.

Thêm micro-YOLO app:Testbench ghi input/weights/qparams vào L2.
Firmware DMA L2→TCDM, chạy graph, DMA output TCDM→L2.
Cocotb đọc output L2 và so từng phần tử với Python golden.

Implementation Details
Micro-YOLO graph:OP_DMA_IN: input + weights + qparams từ L2 vào TCDM.
OP_IM2COL3X3S1P1_C3_PAD32: tạo matrix 1024×32 INT8; 27 lane thật + 5 lane zero.
OP_SYSTOLIC_GEMM32: Conv0, M=1024, K=32, N=32, output 1024×32 INT32.
OP_SPATZ_REQUANT_RELU: INT32 accumulator → INT8 activation 1024×32.
OP_SYSTOLIC_GEMM32: Conv1/head 1×1, M=1024, K=32, N=32.
OP_SPATZ_REQUANT: output raw detection/features 32×32×32 INT8.
OP_DMA_OUT: ghi output về L2.

Memory layout mặc định:L2 input 0x80000000, weights0 0x80002000, weights1 0x80002400, qparams 0x80002800, output 0x80010000.
TCDM raw input 0x10100000, weights 0x10101000, im2col 0x10110000, activation ping 0x10120000, OFM INT32 0x10200000.

Failure/debug contract:D-TCM status 0x10008000: 0xDEADBEEF pass, 0xBADxxxxx fail.
D-TCM debug words record failing layer/op/index/got/expected where firmware self-check exists.

No performance work yet:No double-buffering, no overlap DMA/compute, no cycle optimization.
Only correctness, reuse, and operator coverage gates.

Test Plan
Keep existing regressions passing:make -C sw/spatz_vector
make -C sw/matmul_app
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul

Add Spatz operator tests:Memory width tests: vle8/16/32, vse8/16/32.
Arithmetic tests: add/sub/mul/shift/min/max/compare/select/clamp.
Requant test: INT32 vectors → clipped INT8 output with exact golden.
Optional negative FP test remains because current Spatz is integer-only.

Add scheduler/model tests:test_micro_yolo_e2e: randomized but bounded signed INT8 input/weights, Python golden mirrors firmware graph exactly.
Pass criteria: firmware signature pass, every output byte matches golden, no timeout.
Also test at least one deterministic fixture with hand-checkable small values.

Assumptions
Spatz config remains integer-only: N_FPU=0, RVF=0, RVD=0, VLEN=512, ELEN=32, N_IPU=2.
Micro-YOLO output là raw tensor, chưa gồm YOLO box decode/NMS.
Activation mặc định dùng ReLU/LeakyReLU; exact SiLU sẽ dùng LUT/integer approximation ở phase sau nếu cần model YOLO hiện đại.
Scalar fallback chỉ dùng cho graph control và boundary handling nhỏ; mọi tensor-wide operator phải đi qua Spatz nếu Spatz instruction support và đã pass data-output test.