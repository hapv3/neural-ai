import struct
import unittest

from test_compiled_model import _restart_boundary_outstanding


def _command(command_type, size=32):
    return struct.pack("<HHIII", command_type, size, 0, 0, 0) + bytes(size - 16)


def _dma_submit(command_type, direction):
    command = bytearray(_command(command_type, 64))
    direction_offset = {24: 36, 25: 48, 26: 60}[command_type]
    struct.pack_into("<I", command, direction_offset, direction)
    return bytes(command)


def _dma_wait(direction):
    command = bytearray(_command(27))
    struct.pack_into("<I", command, 16, direction)
    return bytes(command)


class RestartBoundaryTest(unittest.TestCase):
    def test_tracks_each_async_engine_until_its_wait(self):
        commands = [
            _dma_submit(24, 0),
            _dma_submit(25, 1),
            _command(29, 160),
            _dma_wait(0),
            _command(30),
            _dma_wait(1),
        ]

        self.assertEqual(
            _restart_boundary_outstanding(commands, 3),
            {
                "dma_external_to_local": (1,),
                "dma_local_to_external": (2,),
                "systolic": 3,
            },
        )
        self.assertEqual(
            _restart_boundary_outstanding(commands, 6),
            {
                "dma_external_to_local": (),
                "dma_local_to_external": (),
                "systolic": None,
            },
        )

    def test_barrier_clears_all_async_work(self):
        commands = [
            _dma_submit(26, 0),
            _command(32, 224),
            _command(1),
        ]
        self.assertEqual(
            _restart_boundary_outstanding(commands, 3),
            {
                "dma_external_to_local": (),
                "dma_local_to_external": (),
                "systolic": None,
            },
        )

    def test_local_to_local_submit_is_synchronous(self):
        commands = [_dma_submit(26, 2)]
        self.assertEqual(
            _restart_boundary_outstanding(commands, 1),
            {
                "dma_external_to_local": (),
                "dma_local_to_external": (),
                "systolic": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
