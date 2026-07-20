#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

YOSYS_BIN="${YOSYS_BIN:-tools/yosys-install/bin/yosys}"
STA_BIN="${STA_BIN:-tools/opensta-install/bin/sta}"
TARGET="${TARGET:-pe}"
LOW_NS="${LOW_NS:-5.0}"
HIGH_NS="${HIGH_NS:-12.0}"
ITERATIONS="${ITERATIONS:-10}"
SKIP_SYNTH="${SKIP_SYNTH:-0}"

BUILD_DIR="build/synth"
REPORT_DIR="$BUILD_DIR/reports"
RAW_NETLIST="$BUILD_DIR/npu_systolic_array_sky130_tt.v"
STA_NETLIST="$BUILD_DIR/npu_systolic_array_sky130_tt_sta.v"
SUMMARY="$REPORT_DIR/sky130_${TARGET}_fmax_summary.txt"

case "$TARGET" in
    pe)
        STA_TCL="hw/rtl/systolic/synth/sta_pe_sky130.tcl"
        ;;
    array)
        STA_TCL="hw/rtl/systolic/synth/sta_array_sky130.tcl"
        ;;
    *)
        echo "TARGET must be 'pe' or 'array', got '$TARGET'" >&2
        exit 2
        ;;
esac

mkdir -p "$BUILD_DIR" "$REPORT_DIR"

if [[ "$SKIP_SYNTH" != "1" ]]; then
    "$YOSYS_BIN" -s hw/rtl/systolic/synth/synth_sky130_array.ys \
        2>&1 | tee "$BUILD_DIR/npu_systolic_array_sky130_tt.log"
fi

# OpenSTA's Verilog reader does not accept the signed port syntax emitted by
# this Yosys backend version, so strip that syntax from the STA-only copy.
perl -pe 's/\bsigned\s+//g' "$RAW_NETLIST" > "$STA_NETLIST"

run_sta() {
    local period_ns="$1"
    local period_tag="${period_ns//./p}"
    local log="$REPORT_DIR/sky130_${TARGET}_${period_tag}ns.sta.log"

    PERIOD_NS="$period_ns" "$STA_BIN" -no_init -no_splash -exit "$STA_TCL" > "$log"

    local wns
    local tns
    wns="$(awk '/^wns max / {print $3}' "$log" | tail -n 1)"
    tns="$(awk '/^tns max / {print $3}' "$log" | tail -n 1)"
    if [[ -z "$wns" ]]; then
        echo "Could not parse WNS from $log" >&2
        return 2
    fi

    printf "%s %s %s\n" "$period_ns" "$wns" "${tns:-unknown}"
    awk -v wns="$wns" 'BEGIN { exit !(wns >= 0.0) }'
}

low="$LOW_NS"
high="$HIGH_NS"

if run_sta "$high" > "$REPORT_DIR/.high_check"; then
    :
else
    cat "$REPORT_DIR/.high_check"
    echo "HIGH_NS=$high does not meet timing; increase HIGH_NS." >&2
    exit 1
fi

if run_sta "$low" > "$REPORT_DIR/.low_check"; then
    cat "$REPORT_DIR/.low_check"
    freq="$(awk -v p="$low" 'BEGIN { printf "%.2f", 1000.0 / p }')"
    {
        echo "target=$TARGET"
        echo "corner=sky130_fd_sc_hd__tt_025C_1v80"
        echo "status=LOW_NS already passes"
        echo "period_ns<=${low}"
        echo "fmax_mhz>=${freq}"
    } | tee "$SUMMARY"
    exit 0
fi

for ((i = 0; i < ITERATIONS; i++)); do
    mid="$(awk -v lo="$low" -v hi="$high" 'BEGIN { printf "%.6f", (lo + hi) / 2.0 }')"
    if run_sta "$mid" > "$REPORT_DIR/.mid_check"; then
        high="$mid"
    else
        low="$mid"
    fi
done

period="$high"
freq="$(awk -v p="$period" 'BEGIN { printf "%.2f", 1000.0 / p }')"
final_line="$(run_sta "$period")"
wns="$(printf "%s\n" "$final_line" | awk '{print $2}')"
tns="$(printf "%s\n" "$final_line" | awk '{print $3}')"

{
    echo "target=$TARGET"
    echo "corner=sky130_fd_sc_hd__tt_025C_1v80"
    echo "period_ns=$period"
    echo "fmax_mhz=$freq"
    echo "wns_ns=$wns"
    echo "tns_ns=$tns"
    echo "sta_tcl=$STA_TCL"
    echo "netlist=$STA_NETLIST"
} | tee "$SUMMARY"
