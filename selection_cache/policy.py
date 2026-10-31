"""Static, chunk-aligned head-recompute / tail-cache selection policy."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class SplitPlan:
    """Cache only chunks [recompute_chunks, total_chunks).

    Chunk ordinals follow the caller's actual execution order. A plan does not
    infer token positions, layer dependencies, or cache identity. The caller
    must encode the split and chunk layout in its validated cache identity.
    """

    total_chunks: int
    recompute_chunks: int

    def __post_init__(self):
        if type(self.total_chunks) is not int or self.total_chunks < 0:
            raise ValueError("total_chunks must be a nonnegative integer")
        if (type(self.recompute_chunks) is not int
                or not 0 <= self.recompute_chunks <= self.total_chunks):
            raise ValueError("recompute_chunks must be an integer in [0, total_chunks]")

    @property
    def cached_chunks(self):
        return self.total_chunks - self.recompute_chunks

    def cache_global_ordinal(self, local):
        if type(local) is not int or not 0 <= local < self.cached_chunks:
            raise ValueError("cache-local ordinal is outside the retained tail")
        return self.recompute_chunks + local

    @classmethod
    def from_fraction(cls, total_chunks, fraction):
        """Round down to whole chunks: H = floor(total_chunks * fraction)."""
        if type(total_chunks) is not int or total_chunks < 0:
            raise ValueError("total_chunks must be a nonnegative integer")
        if (isinstance(fraction, bool) or not isinstance(fraction, Real)
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("fraction must be a finite number in [0, 1]")
        return cls(total_chunks, math.floor(total_chunks * fraction))
