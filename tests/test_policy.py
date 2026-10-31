"""CPU checks for split boundaries and cache-local ordinal mapping."""
import unittest

from selection_cache.policy import SplitPlan


class SplitPlanTests(unittest.TestCase):
    def test_chunk_aligned_round_down(self):
        plan = SplitPlan.from_fraction(17, 0.25)
        self.assertEqual(plan, SplitPlan(17, 4))
        self.assertEqual(plan.cached_chunks, 13)
        self.assertEqual(plan.cache_global_ordinal(0), 4)
        self.assertEqual(plan.cache_global_ordinal(12), 16)

    def test_zero_full_and_empty(self):
        self.assertEqual(SplitPlan.from_fraction(16, 0).cached_chunks, 16)
        self.assertEqual(SplitPlan.from_fraction(16, 1).cached_chunks, 0)
        self.assertEqual(SplitPlan(0, 0).cached_chunks, 0)
        with self.assertRaises(ValueError):
            SplitPlan(16, 16).cache_global_ordinal(0)

    def test_invalid_shapes(self):
        for total, head in [(-1, 0), (4, -1), (4, 5), (True, 0), (4, True), (4.0, 0), (4, 1.0)]:
            with self.subTest(total=total, head=head), self.assertRaises(ValueError):
                SplitPlan(total, head)

    def test_invalid_fraction(self):
        for fraction in (-0.1, 1.1, float("nan"), float("inf"), True, "0.25"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                SplitPlan.from_fraction(16, fraction)

    def test_invalid_local_ordinal(self):
        plan = SplitPlan(8, 3)
        for local in (-1, 5, 1.0, True):
            with self.subTest(local=local), self.assertRaises(ValueError):
                plan.cache_global_ordinal(local)


if __name__ == "__main__":
    unittest.main()
