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

## Simulation Setup

This repository uses Verilator + cocotb for RTL simulation and RISC-V bare-metal
firmware for cluster tests. The top-level README only covers environment
preparation; detailed test flows live in [docs/test_flow.md](docs/test_flow.md)
and in the README files under `sw/test/*`, `hw/rtl/*/tb`, and the related source
directories.

### Required Tools

Install or provide these tools in `PATH`:

| Tool | Used for |
|---|---|
| `make` | Build firmware and launch RTL simulations |
| `verilator` | SystemVerilog simulation backend |
| `cocotb` | Cocotb |
| `hw/spatz/install/llvm` | RV32 + Spatz/RVV firmware tests |
| `hw/spatz/install/riscv-gcc` | Riscv toolchain for Snitch/Spatz/RVV tests |

The Spatz toolchains are expected at the default paths above. Most Spatz-related
Makefiles also accept `SPATZ_INSTALL=/path/to/spatz/install` if the toolchain is
installed elsewhere.

### Repository Checkout

Initialize bundled hardware dependencies before building:

```sh
git submodule update --init --recursive
```

or use the project helper:

```sh
./checkout_submodules.sh
```

### Python Environment

Use a virtual environment or your normal development environment, then install
the simulation-side Python packages:

```sh
python3 -m pip install --upgrade pip
python3 -m pip install cocotb cocotb-bus cocotb-coverage numpy pytest
```

Some vendored IPs also carry their own optional Python requirement files. Install
them only when working on those generators or vendor flows, for example:

```sh
python3 -m pip install -r hw/spatz/requirements.txt
python3 -m pip install -r hw/idma/requirements.txt
```

### Quick Environment Check

Run these checks before launching a long cluster regression:

```sh
verilator --version
cocotb-config --version
riscv64-unknown-elf-gcc --version
hw/spatz/install/llvm/bin/clang --version
hw/spatz/install/riscv-gcc/bin/riscv32-unknown-elf-objcopy --version
```

For faster incremental simulation builds:

```sh
export CCACHE_DIR=/tmp/ccache
export CCACHE_TEMPDIR=/tmp/ccache-tmp
export VERILATOR_JOBS=8
```

### Smoke Commands

Build one simple firmware image:

```sh
make -C sw/test/boot
```

Run the default cluster cocotb smoke test:

```sh
make -C hw/rtl/cluster COCOTB_TEST_MODULES=test_snitch_boot
```

For full regression order, pass/fail conventions, PMU reporting, and per-suite
commands, use [docs/test_flow.md](docs/test_flow.md). For operator-specific or
model-specific usage, use the README next to that source, for example
[sw/test/micro_yolo/README.md](sw/test/micro_yolo/README.md) and
[sw/test/spatz_ops/README.md](sw/test/spatz_ops/README.md).


## License
*(To be added depending on the specific licensing terms of the integrated submodules).*
