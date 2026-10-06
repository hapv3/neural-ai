"""Artifact-pinned numerical checkpoints before the first YOLO cascade consumer."""

import hashlib
import json
import os
import struct
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

import test_compiled_model as runtime


def extract_c32(memory, address, rows, channels, row_stride, tile_cols):
    """Unpack one output-channel group with the ABI's spatial row pitch."""
    if not (0 < channels <= 32 and rows > 0 and tile_cols > 0):
        raise ValueError("invalid C32 checkpoint dimensions")
    if address < 0 or row_stride < tile_cols * 32:
        raise ValueError("invalid C32 checkpoint address/stride")
    offsets = address + np.arange(rows) // tile_cols * row_stride
    offsets += np.arange(rows) % tile_cols * 32
    if int(offsets[-1]) + channels > len(memory):
        raise ValueError("checkpoint exceeds captured memory")
    indices = offsets[:, None] + np.arange(channels)[None, :]
    return np.frombuffer(memory, dtype=np.int8)[indices].copy()


def compare_bytes(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("checkpoint shape/dtype mismatch")
    delta = actual.astype(np.int16) - expected.astype(np.int16)
    indices = np.flatnonzero(delta)
    first = int(indices[0]) if indices.size else None
    return {
        "bytes": int(actual.size), "mismatches": int(indices.size),
        "exact_percent": float(100 * (1 - indices.size / actual.size)),
        "mae": float(np.abs(delta).mean()), "max_abs": int(np.abs(delta).max()),
        "first_mismatch": first,
        "first_actual": int(actual.flat[first]) if first is not None else None,
        "first_expected": int(expected.flat[first]) if first is not None else None,
        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
    }


@cocotb.test()
async def test_yolo_stem_byte_exact(dut):
    import tensorflow as tf

    manifest = Path(os.environ["NAI_REPLAY_MANIFEST"]).resolve()
    folder = Path(os.environ["YOLO_BYTE_EXACT_DIR"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    full_model = runtime._load_pinned_range(manifest)
    commands, _ = runtime._logical_command_records(full_model)
    source = Path(os.environ["NEURAL_COMPILER_ROOT"]) / "test/model/yolov8n_320_int8.tflite"
    interpreter = tf.lite.Interpreter(
        model_path=str(source), experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    details = {t["index"]: t for t in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    conv_op = next(op for op in ops if op["op_name"] == "CONV_2D")
    conv_id = int(conv_op["outputs"][0])
    sigmoid = next(op for op in ops if op["op_name"] == "LOGISTIC" and conv_id in op["inputs"])
    mul = next(op for op in ops if op["op_name"] == "MUL" and
               set(map(int, op["inputs"])) == {conv_id, int(sigmoid["outputs"][0])})
    mul_id = int(mul["outputs"][0])
    input_detail = interpreter.get_input_details()[0]
    input_bytes = int(np.prod(input_detail["shape"]))
    input_data = ((np.arange(input_bytes, dtype=np.uint32) * 37 + 11) & 255).astype(np.uint8)
    interpreter.set_tensor(input_detail["index"], input_data.view(np.int8).reshape(input_detail["shape"]))
    interpreter.invoke()
    conv = interpreter.get_tensor(conv_id)
    activation = interpreter.get_tensor(mul_id)
    assert conv.ndim == 4 and conv.shape[0] == 1 and conv.shape == activation.shape
    channels = conv.shape[-1]
    assert channels <= 32
    reference = conv.reshape(-1, channels)
    activated_reference = activation.reshape(-1, channels)

    # Decode the actual command destinations; do not reuse historical addresses.
    checkpoints = {}
    first_address = None
    consumed_rows = 0
    for index, command in enumerate(commands):
        kind, _, _, layer, _ = struct.unpack_from("<HHIII", command)
        if kind == 9:
            assert layer == 0, "unexpected producer before the first activation"
            address, rows, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            assert accum == 0 and pitch == cols * 32 and cols == conv.shape[2]
            if first_address is None:
                first_address = address
            assert address == first_address + consumed_rows * 32
            checkpoints[index] = (conv_id, address, rows, pitch, cols, consumed_rows)
            consumed_rows += rows
        elif kind == 12 and checkpoints:
            src_region, _, src = struct.unpack_from("<HHI", command, 16)
            dst_region, _, dst = struct.unpack_from("<HHI", command, 24)
            count = struct.unpack_from("<I", command, 40)[0]
            assert src_region == dst_region == 6 and src == first_address and count % 32 == 0
            rows = count // 32
            assert rows <= consumed_rows
            checkpoints[index] = (mul_id, dst, rows, conv.shape[2] * 32, conv.shape[2], 0)
            end_command = index + 1
            break
    else:
        raise AssertionError("no supported stem Conv/LUT checkpoint in saved model")
    for index in checkpoints:
        runtime._require_restart_safe_boundary(full_model, index + 1)
    model = runtime._extract_selected_yolo320_command_range(full_model, 0, end_command)
    (folder / "prefix.nai").write_bytes(model)
    (folder / "input.bin").write_bytes(input_data.tobytes())
    report = {
        "model_sha256": hashlib.sha256(full_model).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "elf_sha256": hashlib.sha256(runtime._runtime_firmware_path().read_bytes()).hexdigest(),
        "prefix_commands": end_command, "checkpoints": [],
    }
    def persist_report():
        (folder / "byte_exact.json").write_text(json.dumps(report, indent=2))
    persist_report()

    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)
    await runtime.reset_dut(dut)
    bindings = [(1, 0, 0x80000000, input_bytes), (2, 0, 0x80100000, 176400),
                (3, 0, 0x80200000, 307200)]
    invocation, addresses = runtime.build_invocation_with_bindings(
        model, bindings, model_base=0x81000000, binding_table_base=0x81401000)
    for address, data in ((0x80000000, input_data.tobytes()), (0x80100000, bytes(176400)),
                          (0x80200000, bytes(307200)), (0x81000000, model),
                          (0x81401000, addresses), (0x81400000, invocation)):
        await runtime.write_l2_bytes(dut, address, data)
    records = runtime._selected_command_trace_records(model, 0, end_command)

    def capture(index, counters):
        runtime._write_command_pmu_csv(folder / "command_pmu.csv", [records[index]], [counters], index != 0)
        if index not in checkpoints:
            return
        tensor_id, address, rows, pitch, cols, first_row = checkpoints[index]
        memory = runtime.read_tcdm_bytes(dut)
        actual = extract_c32(memory, address, rows, channels, pitch, cols)
        expected = (reference if tensor_id == conv_id else activated_reference)[first_row:first_row + rows]
        stem = f"command-{index + 1}"
        (folder / f"{stem}-actual.bin").write_bytes(actual.tobytes())
        (folder / f"{stem}-expected.bin").write_bytes(expected.tobytes())
        item = {"command": index + 1, "tensor": tensor_id,
                "tensor_name": details[tensor_id]["name"], "tensor_shape": details[tensor_id]["shape"].tolist(),
                "tcdm_address": address, "rows": rows, "channels": int(channels),
                "row_stride": pitch, "tile_cols": cols, "reference_first_row": first_row,
                **compare_bytes(actual, expected)}
        report["checkpoints"].append(item)
        persist_report()
        dut._log.info("BYTE-EXACT command %d: %d/%d mismatches, first=%s max_abs=%d",
                      index + 1, item["mismatches"], item["bytes"], item["first_mismatch"], item["max_abs"])

    await runtime._load_and_run(dut, axi, invocation, timeout_cycles=1000000,
                               invocation_base=0x81400000, model=model,
                               command_trace_records=records, command_trace_callback=capture)
    snapshot = runtime._encode_yolo320_snapshot(
        full_model, end_command, runtime.read_tcdm_bytes(dut),
        bytes(await runtime.read_l2_bytes(dut, 0x80200000, 307200)),
        bytes(await runtime.read_l2_bytes(dut, 0x80100000, 176400)))
    (folder / f"command-{end_command}.snapshot").write_bytes(snapshot)
    runtime._verify_yolo320_snapshot(full_model, snapshot, expected_boundary=end_command)
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_STATUS) == runtime.NPU_CMD_STATUS_PASS
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_FAIL_CODE) == 0
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_DONE_COUNT) == end_command
    assert len(report["checkpoints"]) == len(checkpoints)
    failures = [r for r in report["checkpoints"] if r["mismatches"]]
    assert not failures, f"First mismatching checkpoint: {failures[0] if failures else None}"
