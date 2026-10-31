"""CUDA correctness under repeated buffer reuse; skipped without CUDA/Triton."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from selection_cache.store import CacheIdentity, DiskStore, MemoryStore, encode_chunk, write_store

try:
    import torch
except ImportError:
    torch = None

HAS_CUDA = bool(torch is not None and torch.cuda.is_available() and importlib.util.find_spec("triton"))


@unittest.skipUnless(HAS_CUDA, "CUDA and Triton are optional")
class CudaReplayTests(unittest.TestCase):
    def make_small_store(self):
        identity = CacheIdentity("synthetic-test", "prefix-test", "selector-test", "causal-test")
        chunks = [encode_chunk(np.array([[i, i + 1, 65535]], dtype=np.int32)) for i in range(5)]
        return MemoryStore(chunks, identity, identity)

    def test_nondefault_consumer_stream(self):
        from selection_cache.cuda import CudaReplay

        stream = torch.cuda.Stream()
        captured = []
        with CudaReplay(self.make_small_store(), buffers=2, prefetch=True) as replay:
            with torch.cuda.stream(stream):
                replay.run(lambda ids, ordinal: captured.append((ordinal, ids.clone())))
            self.assertTrue(stream.query(), "run must wait for its consumer stream before returning")
        for ordinal, ids in captured:
            np.testing.assert_array_equal(ids.cpu().numpy(), [[ordinal, ordinal + 1, 65535]])

    def test_callback_failure_closes_pipeline(self):
        from selection_cache.cuda import CudaReplay

        with CudaReplay(self.make_small_store(), buffers=2, prefetch=True) as replay:
            def fail(ids, ordinal):
                if ordinal == 1:
                    raise RuntimeError("intentional consumer failure")
            with self.assertRaisesRegex(RuntimeError, "intentional consumer failure"):
                replay.run(fail)
            with self.assertRaises(RuntimeError):
                replay.run(lambda ids, ordinal: None)
        with self.assertRaises(RuntimeError):
            replay.run(lambda ids, ordinal: None)

    def test_tiled_union_scan_and_mixed_k_output_stride(self):
        from selection_cache.cuda import CudaReplay

        identity = CacheIdentity("synthetic-test", "tiled-prefix", "selector-test", "causal-test")
        expected = [
            np.array([[9, 1, 5], [2, 8, 4], [7, 6, 3]], dtype=np.int32),
            np.arange(59000, 65144, dtype=np.int32).reshape(12, 512),
            np.arange(85, dtype=np.int32).reshape(5, 17)[:, ::-1].copy(),
            np.array([[65535, 0, 32768], [65534, 2, 1]], dtype=np.int32),
        ]
        formats = ("raw_u16", "union_bitmap", "union_bitmap", "raw_u16")
        packets = [encode_chunk(rows, codec=codec) for rows, codec in zip(expected, formats)]
        self.assertEqual(packets[1].dictionary_size, 6144)
        store = MemoryStore(packets, identity, identity)
        # The 6144-entry union spans several scan tiles; all smaller-k outputs
        # share a max-k=512 backing allocation and must honor its row stride.
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                with CudaReplay(store, buffers=2, prefetch=prefetch) as replay:
                    for _ in range(2):
                        captured = []
                        replay.run(lambda ids, ordinal: captured.append((ordinal, ids.clone())))
                        self.assertEqual([i for i, _ in captured], list(range(len(expected))))
                        for ordinal, ids in captured:
                            np.testing.assert_array_equal(
                                ids.cpu().numpy(), np.sort(expected[ordinal], axis=1)
                            )

    def test_reuse_mixed_formats_and_variable_rows(self):
        from selection_cache.cuda import CudaReplay

        identity = CacheIdentity("synthetic-test", "prefix-test", "selector-test", "causal-test")
        rng = np.random.default_rng(42)
        expected, packets = [], []
        for ordinal, rows in enumerate((1, 3, 2, 5, 1, 4, 2)):
            values = np.stack([rng.choice(65535, 17, replace=False) for _ in range(rows)])
            values[0, 0] = 65535
            expected.append(np.sort(values, axis=1))
            codec = ("raw_u16", "union_bitmap", "auto")[ordinal % 3]
            packets.append(encode_chunk(values, codec=codec))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache"
            write_store(path, identity, packets)
            stores = (MemoryStore(packets, identity, identity), DiskStore(path, identity))
            # Queue computation before reading indices to exercise consumer-completion events.
            matrix = torch.randn((512, 512), device="cuda", dtype=torch.float16)
            work = torch.empty_like(matrix)
            for store in stores:
                for prefetch in (False, True):
                    with self.subTest(store=type(store).__name__, prefetch=prefetch):
                        with CudaReplay(store, buffers=2, prefetch=prefetch) as replay:
                            for _ in range(2):
                                captured = []

                                def consume(indices, ordinal):
                                    self.assertEqual(indices.dtype, torch.int32)
                                    torch.mm(matrix, matrix, out=work)
                                    captured.append((ordinal, indices.clone()))

                                replay.run(consume)
                                self.assertEqual([i for i, _ in captured], list(range(len(expected))))
                                for ordinal, values in captured:
                                    np.testing.assert_array_equal(values.cpu().numpy(), expected[ordinal])


if __name__ == "__main__":
    unittest.main()
