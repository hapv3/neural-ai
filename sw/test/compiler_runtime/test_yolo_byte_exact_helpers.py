import unittest

import numpy as np

from analyze_yolo_stem_requant import double_round, quantize_scale, round_away
from test_yolo_stem_byte_exact import compare_bytes, extract_c32


class ByteExactHelpersTest(unittest.TestCase):
    def test_c32_spatial_pitch_and_padding(self):
        memory = bytearray(256)
        expected = np.arange(20, dtype=np.int8).reshape(4, 5)
        for row in range(4):
            address = 16 + row // 2 * 96 + row % 2 * 32
            memory[address:address + 5] = expected[row].tobytes()
        actual = extract_c32(bytes(memory), 16, 4, 5, 96, 2)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(compare_bytes(actual, expected)["mismatches"], 0)
        actual[1, 2] = -128
        report = compare_bytes(actual, expected)
        self.assertEqual((report["mismatches"], report["first_mismatch"], report["max_abs"]), (1, 7, 135))
        with self.assertRaises(ValueError):
            extract_c32(bytes(memory), 240, 4, 5, 96, 2)

    def test_rounding_ties(self):
        np.testing.assert_array_equal(round_away(np.array([-3, -1, 1, 3]), 1), [-2, -1, 1, 2])
        # High multiply rounds ties up; subsequent power-of-two division rounds away.
        np.testing.assert_array_equal(double_round(np.array([-3, -1, 1, 3]), 1 << 30, 0), [-1, 0, 1, 2])
        self.assertEqual(quantize_scale(0.5), (1 << 30, 0))

    def test_captured_stem_counterexamples(self):
        accum = np.array([4155, 5434, -29088], dtype=np.int64)
        np.testing.assert_array_equal(round_away(accum * 3359085, 31) + 5, [11, 13, -40])
        np.testing.assert_array_equal(double_round(accum, 1719851499, -9) + 5, [12, 14, -41])


if __name__ == "__main__":
    unittest.main()
