# YOLO NPU Architecture Specification

## 1. Architectural Overview
The YOLO NPU is a heterogeneous compute architecture designed to achieve 10 TOPS of performance. It is deeply optimized for the YOLO family of models and supports quantized operations via TensorFlow Lite Micro (TFLM). 

To achieve high-performance matrix and vector operations, the architecture integrates 5 parallel **NPU Clusters**. Each cluster features:
1. **Matrix Engine**: A 32x32 Systolic Array optimized for Dense MatMul (Conv2D) operations.
2. **Vector Engine**: A Spatz Co-processor optimized for Depthwise Convolutions, Element-wise operations, and Non-linear Activations (e.g., SiLU, Mish, Softmax).
3. **Control Core**: A Snitch RISC-V scalar core responsible for controlling and dispatching commands to the Matrix and Vector engines.
4. **Shared Tightly-Coupled Data Memory (TCDM)**: A 256-bit wide L1 data memory shared across computing units to sustain high bandwidth requirements.

---

## 2. Cluster Micro-Architecture & Interfaces

Below is the micro-architecture and data-path diagram for a single `npu_cluster`.

```text
========================================================================================
|                                 NPU TOP LEVEL                                        |
|                                                                                      |
|   +------------------------------------------------------------------------------+   |
|   |                         NPU MANAGER SUBSYSTEM                                |   |
|   |  +--------+     +-------------------+     +--------+                         |   |
|   |  | I-TCM  |<--->|Manager Snitch Core|<--->| D-TCM  |                         |   |
|   |  +--------+     +-------------------+     +--------+                         |   |
|   |                           | (10) Orchestration Bus (APB)                     |   |
|   +---------------------------|--------------------------------------------------+   |
|                               v                                                      |
|   +------------------------------------------------------------------------------+   |
|   |                            NPU CLUSTER (x5)                                  |   |
|   |                                                                              |   |
|   |  +----------------+    (1) RVV Coprocessor I/F    +-----------------------+  |   |
|   |  |                |------------------------------>|                       |  |   |
|   |  |  Snitch Core   |                               |  Spatz Vector Engine  |  |   |
|   |  | (RISC-V Ctrl)  |    (2) Custom Matrix I/F      |  (Depthwise/Softmax)  |  |   |
|   |  |                |------------------------------>|                       |  |   |
|   |  +----------------+                               +-----------------------+  |   |
|   |   | ^   |    |                                                |              |   |
|   |   | |(9)|    | (3) OBI / TCDM Interface                       | (3) OBI      |   |
|   |   | |   |    v                                                v              |   |
|   |   | |   |(7) +=============================================================+ |   |
|   |   | |   |    |          Data TCDM Interconnect / Crossbar (256-bit)        | |   |
|   |   | |   |    +=============================================================+ |   |
|   |   | |   |I-Fetch ^                           ^                      ^        |   |
|   |   | |   |        | (3) OBI                   | (3) OBI              | (3) OBI|   |
|   |   v v   v        v                           v                      v        |   |
|   | +------+ +------+ +----------------+    +-------------------+  +-----------+ |   |
|   | |I-TCDM| |Snitch| |                |    |                   |  |           | |   |
|   | | (L1I)| |D-TCM | | Systolic Array |    |  Data TCDM (L1D)  |  |Cluster DMA| |   |
|   | |(Inst)| |(Data)| |   (32x32 PE)   |    |   (Shared SRAM)   |  |  Engine   | |   |
|   | +------+ +------+ +----------------+    +-------------------+  +-----------+ |   |
|   |   ^                                                              |           |   |
|   |   | (8) DMA Load Instructions                                    |           |   |
|   |   +--------------------------------------------------------------+           |   |
|   +------------------------------------------------------------------|-----------+   |
|           ^ (4) APB / CSR Interface                                  | (5) AXI4  |   |
|           |                                                          v Bus       |   |
========================================================================================
                                     |                              |
                             [ Host CPU ]                    [ L2 / DRAM ]
                                     |
                                     v (6) Interrupts
```

### Interface Roles & Specifications

1. **RVV Coprocessor Interface**: Connects the Snitch core to the Spatz Vector Engine. Snitch fetches and decodes RISC-V Vector (RVV) instructions, offloading them to Spatz for execution without stalling the main scalar pipeline.
2. **Custom Matrix Interface (RoCC / Accelerator I/F)**: A custom protocol used by Snitch to configure Systolic Array control registers (e.g., A/B matrix base addresses, M-K-N tile dimensions) and issue the `Start` command.
3. **OBI (Open Bus Interface) / TCDM Interface**: A high-bandwidth (256-bit), low-latency (1-cycle) interconnect for **Data**. It allows all masters (Snitch, Spatz, Systolic, DMA) to access the **D-TCDM** memory banks concurrently without blocking.
4. **APB (Advanced Peripheral Bus) / CSR Interface**: A low-speed bus used by the external Host CPU to configure cluster-level Control and Status Registers (CSRs), such as power gating and cluster enable signals.
5. **AXI4 Bus Interface (128/256-bit)**: A high-bandwidth protocol dedicated to the DMA Engine for fetching burst data (Weights, Feature Maps) from external memory (L2/DRAM).
6. **Interrupt Interface**: Hardware interrupt lines routed from the NPU back to the Host CPU to signal DMA completion or computation completion.
7. **Instruction Fetch Interface (I-Fetch)**: A direct connection from the Snitch Core to the isolated **I-TCDM** (Instruction Memory). This separation (Harvard architecture) ensures instruction fetching does not compete with heavy data traffic on the main crossbar.
8. **DMA Load Instructions Interface**: A dedicated datapath allowing the DMA to pre-load firmware into the I-TCDM before the NPU execution begins.
9. **Snitch Dedicated D-TCM Interface**: A private data connection to a small **Snitch D-TCM** (4KB-8KB). This stores the Snitch's stack and local scalar variables, guaranteeing a fixed 1-cycle access latency regardless of the massive bandwidth consumption by the Systolic Array and DMA on the shared D-TCDM.
10. **Orchestration Bus (Manager to Clusters)**: A control bus originating from the `NPU Manager Core` at the top level, routing down to the 5 NPU Clusters. The Manager Core uses this bus to wake up clusters, partition tasks, and allocate tile addresses (Tiling) dynamically, fully offloading the orchestration burden from the Host CPU.

---

## 3. SRAM Allocation Analysis (2.5 MB Budget)

The total on-chip SRAM budget is **2.5 MB (2560 KB)**. This budget is meticulously partitioned between the **NPU Manager Subsystem** at the top level and the **5 NPU Clusters** below it.

### A. NPU Manager Subsystem (Top-Level): 80 KB Total
Provides an independent brain for the Manager Snitch Core to execute global orchestration software without interference.
- **I-TCM (Instruction)**: `32 KB` for storing the scheduling, tiling logic, and external API handlers.
- **D-TCM (Data)**: `48 KB` dedicated to the stack and data structures describing the YOLO Graph Topology.

### B. NPU Clusters (5 Clusters x 496 KB = 2480 KB)
Each of the 5 clusters is allocated `496 KB`, sub-divided to optimize for INT8-heavy YOLO inference workflows:

#### 1. I-TCDM (Instruction Memory): 32 KB
Sufficient capacity to store the layer/operator execution firmware without dynamic reloading.

#### 2. Snitch D-TCM (Scalar Data Memory): 8 KB
A dedicated SRAM slice for the child Snitch core's stack and local configuration variables, ensuring constant 1-cycle response times for interrupts.

#### 3. D-TCDM (Shared Data for Weights & Feature Maps): 456 KB
The remaining 456 KB is dedicated to shared data processing, explicitly sub-divided for **Double Buffering (Ping-Pong)**:

* **Weight Buffer (256 KB)**: Split into 2 banks x 128 KB. 
  * *Rationale*: The Systolic Array employs a **Weight-Stationary** dataflow. Allocating over half of the D-TCDM to the Weight Buffer maximizes data reuse and drastically reduces memory fetch requests to the slow external DDR. A 128 KB bank can hold `131,072` INT8 weights.
* **Input Feature Map (IFM) Buffer (100 KB)**: Split into 2 banks x 50 KB.
  * *Rationale*: 50 KB per bank provides an optimal capacity to hold moderate spatial tiles, feeding the Systolic Array smoothly.
* **Output Feature Map (OFM) / Accumulator Buffer (100 KB)**: Split into 2 banks x 50 KB.
  * *Rationale*: The output of the INT8 MAC operations is accumulated as **INT32 (4 Bytes)**. Because an INT32 element takes 4x more space than an INT8 element, the OFM buffer needs substantial capacity (100 KB) to balance the spatial output matching the IFM.

**Memory Latency Hiding**: The dual-bank (Ping-Pong) nature of these buffers ensures that memory latency is effectively hidden. The Systolic array processes `Bank 0` while the DMA pre-fetches the subsequent tile into `Bank 1` in the background, keeping hardware utilization near ~100%.

---

## 4. Hardware Verification Plan

To ensure the reliability of this highly concurrent architecture, verification is executed using a Bottom-Up strategy:

### 1. Unit Testing (Block-Level)
- **Matrix Engine TB**: Verify the 32x32 Systolic Array using standalone testbenches. Validate Weight-Stationary loading, data streaming, and accumulator logic against a Python/C++ Golden Model.
- **Spatz Vector TB**: Independently test the Spatz core by executing fundamental RVV instructions (e.g., vector addition, vector multiplication).
- **DMA & AXI TB**: Stress-test the DMA with edge-cases, including varying burst sizes, unaligned memory addresses, and AXI protocol handshake (READY/VALID) compliance.

### 2. Cluster-Level Verification
- **TCDM Interconnect Arbitration**: Prove that the Snitch, Spatz, and Matrix Engine do not experience collisions or deadlocks when simultaneously accessing the shared 256-bit TCDM L1 memory.
- **Firmware Dispatching**: Run RTL simulations with basic C firmware to verify the instruction offload mechanism from the Snitch control core to the compute modules.

### 3. Top-Level Integration & System Testing
- **NPU Top TB (5 Clusters)**: Instantiate all 5 clusters concurrently. Verify that data distributed from the AXI bus via the DMA reaches all clusters at maximum throughput without causing interconnect bottlenecks.
- **Interrupt Handling**: Validate that computation completion and DMA error interrupts from all 5 clusters are correctly aggregated and routed to the Host CPU.
- **End-to-End Workload Simulation**: Simulate a full YOLO layer (e.g., 3x3 Conv2D) across the entire datapath: External Memory Allocation -> DMA to TCDM -> Computation on Matrix Engine -> DMA Write-back to External Memory.
- **Performance Profiling**: Measure the actual simulated cycle counts to calculate the realistic TOPS and hardware utilization rates against the 10 TOPS design target.
