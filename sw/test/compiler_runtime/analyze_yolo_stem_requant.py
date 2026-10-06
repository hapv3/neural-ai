"""Explain saved stem checkpoints offline; never starts or changes a simulation."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def round_away(value, shift):
    if shift == 0:
        return value
    magnitude = (np.abs(value) + (1 << (shift - 1))) >> shift
    return np.where(value < 0, -magnitude, magnitude)


def quantize_scale(scale):
    mantissa, exponent = math.frexp(scale)
    multiplier = int(math.floor(mantissa * (1 << 31) + 0.5))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    return multiplier, exponent


def double_round(value, multiplier, exponent):
    product = (value * (1 << max(exponent, 0))) * multiplier
    # SaturatingRoundingDoublingHighMul: ties toward +infinity. INT32_MIN
    # times INT32_MIN cannot occur here (positive per-channel scale).
    high = (product + (1 << 30)) >> 31
    return round_away(high, max(-exponent, 0))


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    recorded = json.loads((args.checkpoints / "byte_exact.json").read_text())
    model = args.model.read_bytes()
    assert hashlib.sha256(model).hexdigest() == recorded["model_sha256"]
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == recorded["source_sha256"]
    sections = {}
    for index in range(struct.unpack_from("<I", model, 20)[0]):
        descriptor = struct.unpack_from("<8I", model, struct.unpack_from("<I", model, 24)[0] + index * 32)
        sections[descriptor[0]] = model[descriptor[2]:descriptor[2] + descriptor[3]]
    qparams = np.array([struct.unpack_from("<8i", sections[5], c * 32) for c in range(32)], dtype=np.int64)
    interpreter = tf.lite.Interpreter(
        model_path=str(args.source), experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    inp = interpreter.get_input_details()[0]
    raw_input = (args.checkpoints / "input.bin").read_bytes()
    interpreter.set_tensor(inp["index"], np.frombuffer(raw_input, np.int8).reshape(inp["shape"]))
    interpreter.invoke()
    op = next(op for op in interpreter._get_ops_details() if op["op_name"] == "CONV_2D")
    input_id, weight_id, bias_id = map(int, op["inputs"])
    output_id = int(op["outputs"][0])
    inputs = interpreter.get_tensor(input_id).astype(np.int64)[0]
    weights = interpreter.get_tensor(weight_id).astype(np.int64)
    bias = interpreter.get_tensor(bias_id).astype(np.int64)
    reference = interpreter.get_tensor(output_id)[0]
    channels = reference.shape[-1]
    checkpoints = [c for c in recorded["checkpoints"] if c["tensor"] == output_id]
    rows = sum(c["rows"] for c in checkpoints)
    width = reference.shape[1]
    assert rows % width == 0 and weights.shape[1:] == (3, 3, 3)
    height = rows // width
    assert np.all(details[weight_id]["quantization_parameters"]["zero_points"] == 0)
    zp = details[input_id]["quantization"][1]
    corrected_bias = bias - zp * weights.sum(axis=(1, 2, 3))
    assert np.array_equal(corrected_bias, qparams[:channels, 0]), "compiled bias differs from folded TFLite bias"
    accum = np.zeros((height, width, channels), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            accum += inputs[ky:ky + height * 2:2, kx:kx + width * 2:2] @ weights[:, ky, kx, :].T
    accum += corrected_bias
    accum = accum.reshape(rows, channels)
    actual = np.concatenate([
        np.frombuffer((args.checkpoints / f"command-{c['command']}-actual.bin").read_bytes(), np.int8).reshape(c["rows"], channels)
        for c in checkpoints])
    expected = reference[:height].reshape(rows, channels)
    hardware = np.empty_like(actual)
    tflite_math = np.empty_like(actual)
    precise_single = np.empty_like(actual)
    scales = details[weight_id]["quantization_parameters"]["scales"].astype(np.float64)
    assert scales.size in (1, channels)
    scales = np.broadcast_to(scales, (channels,)).copy()
    scales *= details[input_id]["quantization"][0] / details[output_id]["quantization"][0]
    multipliers = []
    for c in range(channels):
        _, multiplier, shift, output_zp, low, high, _, _ = map(int, qparams[c])
        q31, exponent = quantize_scale(float(scales[c]))
        hardware[:, c] = np.clip(round_away(accum[:, c] * multiplier, shift) + output_zp, low, high)
        tflite_math[:, c] = np.clip(double_round(accum[:, c], q31, exponent) + output_zp, low, high)
        assert exponent <= 31
        precise_single[:, c] = np.clip(round_away(accum[:, c] * q31, 31 - exponent) + output_zp, low, high)
        multipliers.append({"channel": c, "nai_multiplier": multiplier, "nai_shift": shift,
                            "tflite_multiplier": q31, "tflite_exponent": exponent})
    mismatches = np.argwhere(actual != expected)
    examples = []
    for row, channel in mismatches[:20]:
        r, c = int(row), int(channel)
        examples.append({"row": r, "channel": c, "accumulator_with_bias": int(accum[r, c]),
                         "actual": int(actual[r, c]), "reference": int(expected[r, c]),
                         "nai_math": int(hardware[r, c]), "tflite_math": int(tflite_math[r, c]),
                         **multipliers[c]})
    report = {"elements": int(actual.size),
              "rtl_vs_tflite_mismatches": int(np.count_nonzero(actual != expected)),
              "rtl_vs_nai_math_mismatches": int(np.count_nonzero(actual != hardware)),
              "double_round_vs_tflite_mismatches": int(np.count_nonzero(tflite_math != expected)),
              "precise_single_vs_tflite_mismatches": int(np.count_nonzero(precise_single != expected)),
              "nai_vs_precise_single_mismatches": int(np.count_nonzero(hardware != precise_single)),
              "channels": multipliers, "examples": examples}
    # Independently check the first fused activation using the *observed* Conv,
    # so propagation of earlier numerical error is not blamed on the LUT engine.
    lut_check = next(c for c in recorded["checkpoints"] if c["tensor"] != output_id)
    offset = 0
    for _ in range(lut_check["command"] - 1):
        offset += struct.unpack_from("<H", sections[1], offset + 2)[0]
    assert struct.unpack_from("<H", sections[1], offset)[0] == 12
    lut_offset = struct.unpack_from("<I", sections[1], offset + 36)[0]
    lut = np.frombuffer(sections[2][lut_offset:lut_offset + 256], np.int8)
    predicted_lut = lut[actual[:lut_check["rows"]].view(np.uint8)]
    observed_lut = np.frombuffer((args.checkpoints / f"command-{lut_check['command']}-actual.bin").read_bytes(), np.int8).reshape(predicted_lut.shape)
    report["afu_vs_lut_of_actual_conv_mismatches"] = int(np.count_nonzero(predicted_lut != observed_lut))
    destination = args.checkpoints / "requant_analysis.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
