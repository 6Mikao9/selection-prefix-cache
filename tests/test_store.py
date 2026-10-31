"""CPU format, integrity, and identity checks; no model or private data needed."""
import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from selection_cache.store import (
    CacheIdentity,
    DiskStore,
    MemoryStore,
    decode_chunk,
    encode_chunk,
    write_store,
)


IDENTITY = CacheIdentity("synthetic-model-v1", "prefix-digest", "selector-v1", "causal-position-v1")


class CodecTests(unittest.TestCase):
    def assert_roundtrip(self, rows, codec="auto", universe=65536):
        packet = encode_chunk(rows, universe=universe, codec=codec)
        actual = decode_chunk(packet)
        self.assertEqual(actual.dtype, np.dtype("uint16"))
        np.testing.assert_array_equal(actual, np.sort(rows, axis=1))
        return packet

    def test_boundary_ids_and_unsorted_input(self):
        rows = np.array([[65535, 0, 7], [32768, 65534, 1]], dtype=np.int32)
        for codec in ("raw_u16", "union_bitmap", "auto"):
            with self.subTest(codec=codec):
                self.assert_roundtrip(rows, codec=codec)

    def test_bitmap_with_partial_final_byte(self):
        rows = np.array([[0, 4, 8], [1, 5, 7], [2, 3, 6]], dtype=np.int64)
        packet = self.assert_roundtrip(rows, codec="union_bitmap", universe=9)
        self.assertEqual(packet.dictionary_size, 9)

    def test_auto_falls_back_for_disjoint_rows(self):
        rows = np.arange(32, dtype=np.int32).reshape(4, 8)
        packet = self.assert_roundtrip(rows)
        self.assertEqual(packet.codec, "raw_u16")

    def test_auto_compresses_shared_rows(self):
        rows = np.tile(np.arange(32, dtype=np.int32), (16, 1))
        packet = self.assert_roundtrip(rows)
        self.assertEqual(packet.codec, "union_bitmap")
        self.assertLess(len(packet.payload), rows.size * 2)

    def test_invalid_rows_are_rejected_before_narrowing(self):
        invalid = [
            np.array([[1.0, 2.0]]),
            np.array([[True, False]]),
            np.array([1, 2]),
            np.array([[1, 1]]),
            np.array([[-1, 2]]),
            np.array([[65536, 2]], dtype=np.uint64),
            np.empty((0, 2), dtype=np.int32),
            np.empty((2, 0), dtype=np.int32),
        ]
        for rows in invalid:
            with self.subTest(shape=rows.shape, dtype=str(rows.dtype)):
                with self.assertRaises(ValueError):
                    encode_chunk(rows)
        with self.assertRaises(ValueError):
            encode_chunk(np.array([[1, 8]]), universe=8)

    def test_truncated_packet_and_unknown_codec(self):
        for codec in ("raw_u16", "union_bitmap"):
            packet = encode_chunk(np.array([[3, 2, 1], [1, 2, 4]]), codec=codec)
            with self.subTest(codec=codec):
                with self.assertRaises(ValueError):
                    decode_chunk(dataclasses.replace(packet, payload=packet.payload[:-1]))
        with self.assertRaises(ValueError):
            decode_chunk(dataclasses.replace(packet, codec="unknown"))

    def test_bitmap_row_count_is_checked(self):
        packet = encode_chunk(np.array([[0, 1], [1, 2]]), codec="union_bitmap")
        payload = bytearray(packet.payload)
        payload[packet.dictionary_size * 2] = 0
        with self.assertRaises(ValueError):
            decode_chunk(dataclasses.replace(packet, payload=bytes(payload)))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "cache"
        self.expected = [
            np.array([[65535, 2, 1]], dtype=np.int32),
            np.tile(np.array([8, 4, 2], dtype=np.int32), (5, 1)),
            np.array([[20, 21, 22], [25, 24, 23]], dtype=np.int32),
        ]
        self.packets = [encode_chunk(rows) for rows in self.expected]

    def assert_store(self, store):
        self.assertEqual(len(store), len(self.expected))
        for i in (2, 0, 1, 2):
            np.testing.assert_array_equal(decode_chunk(store.read(i)), np.sort(self.expected[i], axis=1))

    def test_disk_and_memory_random_access(self):
        write_store(self.directory, IDENTITY, iter(self.packets))
        self.assert_store(DiskStore(self.directory, IDENTITY))
        self.assert_store(MemoryStore(self.packets, IDENTITY, IDENTITY))

    def test_identity_mismatch(self):
        write_store(self.directory, IDENTITY, self.packets)
        wrong = dataclasses.replace(IDENTITY, prefix_digest="another-prefix")
        with self.assertRaises(ValueError):
            DiskStore(self.directory, wrong)
        with self.assertRaises(ValueError):
            MemoryStore(self.packets, IDENTITY, wrong)

    def test_empty_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(IDENTITY, model_fingerprint="")

    def test_existing_directory_is_not_overwritten(self):
        write_store(self.directory, IDENTITY, self.packets)
        before = (self.directory / "data.bin").read_bytes()
        with self.assertRaises(FileExistsError):
            write_store(self.directory, IDENTITY, self.packets)
        self.assertEqual((self.directory / "data.bin").read_bytes(), before)

    def test_payload_corruption_is_rejected(self):
        write_store(self.directory, IDENTITY, self.packets)
        store = DiskStore(self.directory, IDENTITY)
        path = self.directory / "data.bin"
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
        with self.assertRaises(ValueError):
            store.read(0)

    def test_truncated_disk_file_is_rejected(self):
        write_store(self.directory, IDENTITY, self.packets)
        path = self.directory / "data.bin"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            store = DiskStore(self.directory, IDENTITY)
            store.read(len(self.packets) - 1)


if __name__ == "__main__":
    unittest.main()
