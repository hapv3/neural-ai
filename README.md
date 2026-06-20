# Neural-AI NPU Cluster

Welcome to the **Neural-AI NPU Cluster**, a highly scalable, heterogeneous Neural Processing Unit architecture designed for accelerating Generative AI, YOLO, CNNs, and Vision Transformers at the edge.

## Architecture Highlights

The Neural-AI NPU Cluster follows a heterogeneous compute model where a lightweight RISC-V control core orchestrates highly specialized hardware engines. All components share a high-bandwidth, deterministic L1 Tightly-Coupled Data Memory (TCDM).

- **Control Core:** RISC-V RV32IMAC core handling control flow, tiling, and orchestrating the cluster.
- **Matrix Engine (Compute):** A 32x32 INT8 Systolic Array for accelerating Dense Matrix Multiplication and Convolutional operations with massive parallelism.
- **Vector Engine (Compute):** A highly capable vector co-processor for non-linear activations (SiLU, GELU, Softmax, Sigmoid) and element-wise operations.
- **Data Movement (iDMA):** A high-performance, modular Direct Memory Access engine responsible for background transfers between the global L2 (DRAM) and the local L1 TCDM.
- **TCDM Interconnect:** A grouped, hierarchical interconnect topology providing zero-starvation, fair access to shared SRAM banks for all engines.

---

## References & Acknowledgements

This project builds upon and integrates several state-of-the-art open-source hardware components. We deeply acknowledge the work of the **PULP (Parallel Ultra-Low Power) Platform** (ETH Zurich & University of Bologna) and the broader open-source hardware community.

The following IPs and concepts are utilized, referred to, or serve as architectural inspirations for the Neural-AI NPU Cluster:

### 1. Snitch Core Complex
- **Role:** Main control and coordination core.
- **Reference:** [pulp-platform/snitch](https://github.com/pulp-platform/snitch)
- **Citation:** Zaruba et al., *"Snitch: A Tiny Pseudo-Dual-Issue Processor for Area and Energy Efficient Execution of Floating-Point Intensive Workloads"*, IEEE Transactions on Computers, 2021.

### 2. Spatz Vector Engine
- **Role:** Vector Processing Unit for non-linear and Activation functions (AFU).
- **Reference:** [pulp-platform/spatz](https://github.com/pulp-platform/spatz)

### 3. iDMA (Modular Data Movement Accelerator)
- **Role:** High-bandwidth L2 <-> L1 data movement.
- **Reference:** [pulp-platform/iDMA](https://github.com/pulp-platform/idma)
- **Citation:** Benz et al., *"A high-performance, energy-efficient modular DMA engine architecture"*, IEEE Transactions on Computers, 2023.

### 4. MAGIA (Mesh Architecture for Generative Intelligence Acceleration)
- **Role:** Architectural inspiration. The grouped hierarchical TCDM interconnect and Memory-Mapped iDMA integration in our NPU are heavily inspired by MAGIA's local interconnect and Tile architecture.
- **Reference:** [pulp-platform/MAGIA](https://github.com/pulp-platform/MAGIA)

### 5. Standard Protocols
- **OBI (OpenBus Interface):** Used for low-latency intra-cluster communication. [OBI Spec](https://github.com/openhwgroup/programs/blob/master/TGs/cores-task-group/obi/OBI-v1.5.0.pdf).
- **AXI4:** Used for high-bandwidth global memory (L2/DRAM) access.

---

## License
*(To be added depending on the specific licensing terms of the integrated submodules).*