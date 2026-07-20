# Systolic Synthesis Flow

This directory contains open-source synthesis and STA entry points for the
systolic RTL blocks.  The flow uses SkyWater SKY130 HD as an available proxy
library so that structural synthesis issues, critical paths, and relative Fmax
trends can be checked before moving the design to a commercial 5 nm flow.

The SKY130 reports are not signoff results for TSMC 5 nm.  Use them to catch RTL
problems such as large combinational arithmetic, wide mux fanout, latch
inference, memory mapping mistakes, and synthesis parser issues.

## Required Tools

The scripts default to these repository-local paths:

```sh
tools/yosys-install/bin/yosys
tools/opensta-install/bin/sta
tools/skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib
```

Override them with environment variables when needed:

```sh
export YOSYS_BIN=/path/to/yosys
export STA_BIN=/path/to/sta
```

Yosys must include SystemVerilog frontend support for `read_slang`, because the
linebuffer/controller RTL uses SystemVerilog constructs that are not accepted by
the legacy Verilog frontend.

## Block Synthesis

Run individual synthesis scripts from the repository root:

```sh
tools/yosys-install/bin/yosys -s hw/rtl/systolic/synth/synth_sky130_array.ys
tools/yosys-install/bin/yosys -s hw/rtl/systolic/synth/synth_sky130_requant_pipeline_tt.ys
tools/yosys-install/bin/yosys -s hw/rtl/systolic/synth/synth_sky130_conv_linebuf_stream_packer_tt.ys
tools/yosys-install/bin/yosys -s hw/rtl/systolic/synth/synth_sky130_systolic_ctrl_regs_tt.ys
```

For controller-level synthesis with the mapped array and requant blocks, run the
bottom-up script after the dependent block netlists exist:

```sh
tools/yosys-install/bin/yosys -s hw/rtl/systolic/synth/synth_sky130_systolic_controller_bottomup_tt.ys
```

Main generated outputs are written under `build/synth/`, for example:

```text
build/synth/npu_systolic_array_sky130_tt.v
build/synth/conv_linebuf_stream_packer_sky130_tt.v
build/synth/systolic_controller_sky130_tt.v
```

`build/synth/` is generated output and should not be committed.

## Fmax Sweep

`run_sky130_fmax.sh` performs synthesis for the systolic array, creates an
OpenSTA-compatible netlist copy, then binary-searches the passing period for
either the PE or the full array timing setup:

```sh
TARGET=pe    hw/rtl/systolic/synth/run_sky130_fmax.sh
TARGET=array hw/rtl/systolic/synth/run_sky130_fmax.sh
```

Useful knobs:

```sh
LOW_NS=2.0 HIGH_NS=12.0 ITERATIONS=12 TARGET=array \
  hw/rtl/systolic/synth/run_sky130_fmax.sh

SKIP_SYNTH=1 TARGET=array \
  hw/rtl/systolic/synth/run_sky130_fmax.sh
```

Reports are written to:

```text
build/synth/reports/sky130_<target>_fmax_summary.txt
build/synth/reports/sky130_<target>_<period>ns.sta.log
```

## Wrapper Files

The wrapper modules provide stable top-level module names for STA or
bottom-up composition when a mapped implementation has been renamed by Yosys.
They are not behavioral alternatives to the RTL source.

`conv_linebuf_stream_packer` now fixes `K_MAX=5` and `STRIDE_MAX=2` internally.
Do not pass those as synthesis generics; only width and capacity parameters such
as `ADDR_WIDTH`, `DATA_WIDTH`, `ARRAY_DIM`, `INPUT_ELEM_WIDTH`, and
`MAX_INPUT_W` remain configurable.
