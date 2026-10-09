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

## Core and Accelerator Roadmap

The current cluster is optimized around a compact Snitch control core plus
dedicated matrix, vector, DMA, and AFU engines. The next architecture direction
is to mature the control and vector sides so the cluster can handle heavier
pre/post-processing, dynamic model scheduling, and production-grade vector
workloads without leaning on slow scalar loops.

### Control Core Upgrade: CVA6-Class Path

The Snitch-based control path remains useful for lightweight firmware bring-up,
deterministic tests, and simple descriptor dispatch. For mature scalar
performance, the roadmap adds a CVA6-class control-core option:

| Area | Planned improvement | Rationale |
|---|---|---|
| Scalar throughput | Replace or optionally pair Snitch with a CVA6-class RV64 application core for control-heavy kernels | Reduces bottlenecks in dynamic scheduling, decode/NMS, layout planning, and fallback operators. |
| Memory hierarchy | Add instruction/data cache options plus tighter AXI access to L2 | Lets scalar firmware process larger metadata and postprocess buffers without excessive TCM pressure. |
| Firmware model | Move from small bare-metal kernels toward a richer runtime that can launch graphs, manage descriptors, and handle interrupts robustly | Needed for multi-model and multi-cluster orchestration. |
| Interrupts and events | Use a real trap/WFI path for DMA, Systolic, AFU, and vector completion | Reduces polling overhead and improves power behavior. |

Snitch should remain the minimal baseline until the CVA6 path is integrated and
validated. The core interface contract should stay stable: command-control
MMIO, explicit accelerator descriptors, iDMA-owned bulk movement, and IRQ-based
completion.

### Pre/Post-Processing Acceleration

The NPU should avoid using scalar firmware for tensor-wide pre/post-processing.
Planned work:

- **Pre-processing:** resize, crop, pad, normalize, layout conversion, HWC to
  C32/ROW32 packing, quantization, and tiled input staging.
- **Post-processing:** YOLO DFL decode, sigmoid/class score filtering, top-k,
  NMS, bounding-box transform, transpose/reshape, and output compaction.
- **Data movement:** prefer iDMA 2D/3D transfers for regular layouts and vector
  gather/scatter for irregular tiles.
- **Operator fusion:** keep Conv requant/clamp fused in the systolic drain path;
  fuse activation, add, and reduction kernels when tensor lifetime and TCDM
  capacity allow it.
- **Runtime contract:** host/compiler emits descriptors for pre/post kernels
  just like Conv linebuffer descriptors, so firmware dispatches work instead of
  rebuilding schedules in scalar loops.

### Vector Engine Maturity: Ara-Class Upgrade Path

Spatz is the current lightweight RVV vector coprocessor for INT8 element-wise
and memory-oriented kernels. For mature vector performance, the roadmap adds an
Ara-class vector backend option:

| Area | Planned improvement | Rationale |
|---|---|---|
| Vector throughput | Evaluate an Ara-class RVV engine as the high-performance vector path | Provides a more mature target for wide vector processing, reductions, gathers/scatters, and postprocess kernels. |
| Memory bandwidth | Add enough TCDM/AXI bandwidth and banking to feed wider vector lanes | Vector performance is limited by memory if the interconnect remains sized only for the current Spatz path. |
| Kernel coverage | Expand vector kernels for strided/indexed loads, reductions, prefix/top-k helpers, NMS primitives, and layout transforms | Covers the parts of YOLO/CNN/ViT that are inefficient on the Systolic Array. |
| Software stack | Keep RVV intrinsics/assembly isolated behind C wrappers | Allows the same graph runtime to target Spatz first and Ara-class hardware later. |

The intended split is: Systolic Array for dense Conv/GEMM, AFU for LUT-style
nonlinear functions, iDMA for regular movement, and Spatz/Ara-class vector
hardware for irregular memory, reductions, element-wise math, and
pre/post-processing.

### Core-Level Improvements

Core and subsystem work that supports the roadmap:

- Stabilize the command-stream ABI so host/compiler output can drive CVA6,
  Snitch, or mixed control-core configurations.
- Introduce a scheduler policy that keeps TCDM allocation lifetime-aware and
  avoids unnecessary L2 save/reload for skip tensors.
- Preserve a small deterministic bare-metal test mode even when the richer
  CVA6 runtime is enabled.
- Keep core upgrades optional until regression coverage proves that current
  Snitch firmware behavior is preserved.

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

### 3. CVA6 and Ara Upgrade Targets
- **Role:** Future scalar and vector performance upgrade path for richer runtime
  control and mature RVV processing.
- **References:** [openhwgroup/cva6](https://github.com/openhwgroup/cva6),
  [pulp-platform/ara](https://github.com/pulp-platform/ara)

### 4. iDMA (Modular Data Movement Accelerator)
- **Role:** High-bandwidth L2 <-> L1 data movement.
- **Reference:** [pulp-platform/iDMA](https://github.com/pulp-platform/idma)
- **Citation:** Benz et al., *"A high-performance, energy-efficient modular DMA engine architecture"*, IEEE Transactions on Computers, 2023.

### 5. MAGIA (Mesh Architecture for Generative Intelligence Acceleration)
- **Role:** Architectural inspiration. The grouped hierarchical TCDM interconnect and Memory-Mapped iDMA integration in our NPU are heavily inspired by MAGIA's local interconnect and Tile architecture.
- **Reference:** [pulp-platform/MAGIA](https://github.com/pulp-platform/MAGIA)

### 6. Standard Protocols
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
| `yosys` with `read_slang` | Open-source RTL synthesis checks |
| `sta` / OpenSTA | Open-source static timing checks |
| SkyWater SKY130 PDK | Proxy Liberty library for local synthesis/STA |

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

### Synthesis Tool Setup

The repository includes Yosys/OpenSTA synthesis scripts for early RTL quality
checks.  They are SKY130 proxy flows, not TSMC 5 nm signoff flows, but they are
useful for catching synthesis parser issues, inferred long arithmetic paths,
large mux fanout, and block-level timing regressions before handing the RTL to a
commercial backend flow.

Default tool and PDK locations:

```sh
tools/yosys-install/bin/yosys
tools/opensta-install/bin/sta
tools/skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib
```

If your tools are installed elsewhere:

```sh
export YOSYS_BIN=/path/to/yosys
export STA_BIN=/path/to/sta
```

Quick checks:

```sh
tools/yosys-install/bin/yosys -V
tools/opensta-install/bin/sta -version
test -f tools/skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib
```

Block-level synthesis documentation lives in:

- [hw/rtl/systolic/synth/README.md](hw/rtl/systolic/synth/README.md)
- [hw/rtl/cluster/synth/README.md](hw/rtl/cluster/synth/README.md)

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
