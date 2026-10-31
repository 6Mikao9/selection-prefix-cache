"""CUDA head/tail equivalence, indexed tail reads, and useful head start."""
import importlib.util
import threading
import unittest

import numpy as np

from selection_cache.policy import SplitPlan
from selection_cache.store import CacheIdentity, MemoryStore, encode_chunk

try:
    import torch
except ImportError:
    torch = None

HAS_CUDA = bool(torch is not None and torch.cuda.is_available() and importlib.util.find_spec("triton"))


class RecordingStore:
    def __init__(self, store, gate=None):
        self.store, self.entries, self.gate = store, store.entries, gate
        self.reads = []

    def __len__(self):
        return len(self.store)

    def read(self, index):
        if self.gate is not None and not self.gate.wait(5):
            raise RuntimeError("tail read blocked the first head callback")
        self.reads.append(index)
        return self.store.read(index)


@unittest.skipUnless(HAS_CUDA, "CUDA and Triton are optional")
class HybridReplayTests(unittest.TestCase):
    def setUp(self):
        self.expected = [np.array([[i, i + 7, 65535], [i + 1, i + 8, 65534]], dtype=np.int32)
                         for i in range(7)]
        self.gpu = [torch.as_tensor(x, device="cuda") for x in self.expected]

    def make_store(self, head, gate=None):
        identity = CacheIdentity("synthetic-hybrid", "same-prefix", f"split={head}", "chunk-layout")
        packets = [encode_chunk(x, codec="raw_u16" if i % 2 else "union_bitmap")
                   for i, x in enumerate(self.expected[head:])]
        return RecordingStore(MemoryStore(packets, identity, identity), gate)

    def test_zero_partial_full_recompute_and_global_ordinals(self):
        from selection_cache.cuda import CudaReplay

        for head in (0, 1, 3, 6, 7):
            for prefetch in (False, True):
                with self.subTest(head=head, prefetch=prefetch):
                    store, recomputed = self.make_store(head), []
                    plan = SplitPlan(7, head)
                    with CudaReplay(store, buffers=2, prefetch=prefetch) as replay:
                        for _ in range(2):
                            captured = []
                            recomputed.clear()
                            store.reads.clear()

                            def recompute(ordinal):
                                recomputed.append(ordinal)
                                return self.gpu[ordinal] + 0

                            stats = replay.run_hybrid(
                                lambda ids, ordinal: captured.append((ordinal, ids.clone())), recompute, plan)
                            self.assertEqual(recomputed, list(range(head)))
                            self.assertEqual(store.reads, list(range(7 - head)))
                            self.assertEqual([i for i, _ in captured], list(range(7)))
                            self.assertEqual(stats["head_chunks"], head)
                            self.assertEqual(stats["tail_chunks"], 7 - head)
                            if head == 7:
                                self.assertEqual(stats["output_buffer_bytes"], 0)
                                self.assertEqual(stats["tail_payload_bytes"], 0)
                            for ordinal, actual in captured:
                                np.testing.assert_array_equal(actual.cpu().numpy(), self.expected[ordinal])

    def test_head_starts_without_waiting_for_tail_read(self):
        from selection_cache.cuda import CudaReplay

        gate = threading.Event()
        store = self.make_store(2, gate)
        captured = []

        def recompute(ordinal):
            gate.set()  # Read can only finish after useful head work was issued.
            return self.gpu[ordinal] + 0

        with CudaReplay(store, buffers=2, prefetch=True) as replay:
            replay.run_hybrid(lambda ids, i: captured.append((i, ids.clone())), recompute, SplitPlan(7, 2))
        self.assertTrue(gate.is_set())
        for ordinal, actual in captured:
            np.testing.assert_array_equal(actual.cpu().numpy(), self.expected[ordinal])

    def test_hybrid_on_nondefault_stream_and_plan_mismatch(self):
        from selection_cache.cuda import CudaReplay

        stream = torch.cuda.Stream()
        # Establish readiness for tensors constructed on the default stream.
        stream.wait_stream(torch.cuda.current_stream())
        with CudaReplay(self.make_store(3), buffers=2) as replay:
            with self.assertRaises(ValueError):
                replay.run_hybrid(lambda *_: None, lambda i: self.gpu[i], SplitPlan(7, 2))
            captured = []
            with torch.cuda.stream(stream):
                replay.run_hybrid(lambda ids, i: captured.append((i, ids.clone())),
                                  lambda i: self.gpu[i] + 0, SplitPlan(7, 3))
            self.assertTrue(stream.query())
        for ordinal, actual in captured:
            np.testing.assert_array_equal(actual.cpu().numpy(), self.expected[ordinal])


if __name__ == "__main__":
    unittest.main()
