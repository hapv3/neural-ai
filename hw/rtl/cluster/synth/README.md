# Cluster Synthesis Flow

This directory contains exploratory open-source synthesis and STA entry points
for the NPU cluster shell and selected cluster sub-blocks.  The flow targets
SkyWater SKY130 HD because that library is available locally and works with
Yosys/OpenSTA.  Treat the reports as RTL quality and trend checks, not as
commercial 5 nm signoff data.

## Required Tools

Expected default locations:

```sh
tools/yosys-install/bin/yosys
tools/opensta-install/bin/sta
tools/skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib
```

Yosys must support `read_slang` for the full cluster file lists.  Some small
sub-block scripts, such as `synth_sky130_pmu_tt.ys`, use plain `read_verilog`.

## File Lists And Blackboxes

- `cluster_shell_synth.f`
  File list for the cluster shell with real local NPU RTL and blackboxed vendor
  IP where needed.
- `cluster_synth.f`
  Broader cluster file list for whole-cluster experiments.
- `sram_blackboxes.sv`
  SRAM macro stand-ins for synthesis when `SYNTHESIS_SRAM_BLACKBOX` is defined.
- `cluster_ip_blackboxes.sv`
  Blackboxes for large external/vendor IPs that are not the target of the local
  timing experiment.
- `sky130_fd_sc_hd_yosys_gates.v`
  SKY130 helper cells used by selected Yosys mapping scripts.

## Common Commands

Run commands from the repository root:

```sh
tools/yosys-install/bin/yosys -s hw/rtl/cluster/synth/synth_sky130_pmu_tt.ys
tools/yosys-install/bin/yosys -s hw/rtl/cluster/synth/synth_sky130_afu_tt.ys
tools/yosys-install/bin/yosys -s hw/rtl/cluster/synth/synth_sky130_tcdm_tt.ys
tools/yosys-install/bin/yosys -s hw/rtl/cluster/synth/synth_sky130_shell_tt_sta_fast.ys
```

Primary outputs are written under `build/synth/`:

```text
build/synth/npu_pmu_sky130_tt.v
build/synth/afu_sky130_tt.v
build/synth/tcdm_interconnect_sky130_tt.v
build/synth/npu_cluster_shell_sky130_tt.v
```

`build/synth/` is generated output and should not be committed.

## OpenSTA

OpenSTA Tcl files consume mapped netlists under `build/synth/`:

```sh
PERIOD_NS=10.0 tools/opensta-install/bin/sta -no_init -no_splash -exit \
  hw/rtl/cluster/synth/sta_pmu_sky130.tcl

PERIOD_NS=10.0 tools/opensta-install/bin/sta -no_init -no_splash -exit \
  hw/rtl/cluster/synth/sta_shell_sky130.tcl
```

If an OpenSTA reader rejects a Yosys-emitted construct that is legal in Verilog,
create an STA-only copy of the netlist in `build/synth/` and leave the original
Yosys output unchanged.  The systolic Fmax helper shows this pattern by stripping
`signed` keywords from a copy before STA.

## Script Variants

Several `synth_sky130_shell_tt_sta_*` scripts are intentionally kept as
experiments for compile-time/timing-debug tradeoffs:

- `*_fast.ys`
- `*_fastmap.ys`
- `*_maxiter1.ys`
- `*_minimal.ys`
- `*_nocheck.ys`
- `*_skip_prune.ys`

Use `synth_sky130_shell_tt_sta_fast.ys` as the default cluster-shell smoke flow.
Use the other variants only when debugging a specific synthesis bottleneck.
