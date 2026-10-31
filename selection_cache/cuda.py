"""Bounded selection replay with a disk reader and CUDA preparation stream.

The callback borrows an int32 tensor: enqueue all consumers on the calling
CUDA stream and do not retain it after the callback (clone if needed).
This is selection-set replay, not attention or KV-cache reconstruction.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _raw_decode(source, output, N: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(source + 2 * offset, offset < N, other=0).to(tl.int32)
    hi = tl.load(source + 2 * offset + 1, offset < N, other=0).to(tl.int32)
    tl.store(output + (offset // K) * STRIDE + offset % K,
             lo | (hi << 8), offset < N)


@triton.jit
def _union_decode(source, output, M: tl.constexpr, K: tl.constexpr,
                  MASK_BYTES: tl.constexpr, STRIDE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    within = tl.arange(0, BLOCK)
    count = tl.full((), 0, tl.int32)
    # Bounded tiles avoid one enormous scan when a forced union has M=65536.
    for tile in range(tl.cdiv(M, BLOCK)):
        offset = tile * BLOCK + within
        packed = tl.load(source + 2 * M + row * MASK_BYTES + offset // 8,
                         offset < M, other=0).to(tl.uint32)
        chosen = ((packed >> (offset % 8)) & 1).to(tl.int32)
        chosen = tl.where(offset < M, chosen, 0)
        rank = count + tl.cumsum(chosen, axis=0) - 1
        lo = tl.load(source + 2 * offset, offset < M, other=0).to(tl.int32)
        hi = tl.load(source + 2 * offset + 1, offset < M, other=0).to(tl.int32)
        tl.store(output + row * STRIDE + rank, lo | (hi << 8),
                 (chosen != 0) & (rank >= 0) & (rank < K))
        count += tl.sum(chosen, axis=0)


@dataclass
class _Slot:
    host: torch.Tensor
    host_array: np.ndarray
    device: torch.Tensor
    output: torch.Tensor
    ready: torch.cuda.Event
    copied: torch.cuda.Event
    consumed: torch.cuda.Event
    has_copy: bool = False
    has_consumer: bool = False


class CudaReplay:
    """Stream a validated store through a fixed number of reusable buffers.

    ``run(consume)`` calls ``consume(indices, ordinal)`` on the caller's current
    stream, in store order. The callback must enqueue its GPU work there (or
    join any other streams before returning). It may clone but must not retain
    the borrowed tensor. A callback must not recursively call ``run``.

    ``prefetch=False`` uses identical buffers/kernels but waits for each
    consumer before reading/preparing the next chunk. ``True`` overlaps a
    bounded CPU read-ahead queue and preparation with previous GPU consumers.
    No model integration, eviction policy, or automatic prefix matching is
    provided: the store must already have passed its identity validation.
    Only use stores produced by the validated writer: GPU decoding does not
    authenticate a manifest or revalidate every bitmap's population count.
    """

    def __init__(self, store, device="cuda", buffers=2, prefetch=True,
                 max_buffer_bytes=256 << 20):
        if isinstance(buffers, bool) or not isinstance(buffers, int) or not 1 <= buffers <= 8:
            raise ValueError("buffers must be an integer in [1, 8]")
        if (isinstance(max_buffer_bytes, bool) or not isinstance(max_buffer_bytes, int)
                or max_buffer_bytes <= 0):
            raise ValueError("max_buffer_bytes must be a positive integer")
        if not torch.cuda.is_available():
            raise RuntimeError("CudaReplay requires a CUDA GPU")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("device must be a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device()
                                   if requested.index is None else requested.index)
        self.store = store
        # Snapshot metadata; a store is immutable throughout this replay.
        self.entries = [dict(entry) for entry in store.entries]
        if len(self.entries) != len(store):
            raise ValueError("store length and metadata disagree")
        for entry in self.entries:
            self._validate_entry(entry)
        self.buffers = buffers
        self.prefetch = bool(prefetch)
        self.max_rows = max((e["rows"] for e in self.entries), default=1)
        self.max_k = max((e["k"] for e in self.entries), default=1)
        self.max_payload = max((e["nbytes"] for e in self.entries), default=1)
        self.output_bytes = buffers * self.max_rows * self.max_k * 4
        self.input_bytes = buffers * self.max_payload
        # Count both the pinned staging and device input, plus expanded output.
        if self.output_bytes + 2 * self.input_bytes > max_buffer_bytes:
            raise ValueError("replay buffers exceed max_buffer_bytes")
        self._closed = False
        self._running = False
        self._compute = None
        self._executor = None
        self._slots = []
        with torch.cuda.device(self.device):
            self._prepare = torch.cuda.Stream(device=self.device)
            for _ in range(buffers):
                host = torch.empty(self.max_payload, dtype=torch.uint8,
                                   pin_memory=True)
                self._slots.append(_Slot(
                    host, host.numpy(),
                    torch.empty(self.max_payload, dtype=torch.uint8,
                                device=self.device),
                    torch.empty((self.max_rows, self.max_k), dtype=torch.int32,
                                device=self.device),
                    torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()))
        if self.prefetch:
            self._executor = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="selection-read")

    @staticmethod
    def _validate_entry(entry):
        for name in ("rows", "k", "universe", "dictionary_size", "nbytes"):
            value = entry.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"invalid chunk metadata: {name}")
        rows, k, universe = entry["rows"], entry["k"], entry["universe"]
        m = entry["dictionary_size"]
        if not (rows >= 1 and 1 <= k <= universe <= 65536):
            raise ValueError("invalid chunk shape or universe")
        if entry.get("codec") == "raw_u16":
            expected = 2 * rows * k
            if m != 0:
                raise ValueError("raw chunks must have dictionary_size=0")
        elif entry.get("codec") == "union_bitmap":
            if not k <= m <= universe:
                raise ValueError("invalid union dictionary size")
            expected = 2 * m + rows * ((m + 7) // 8)
        else:
            raise ValueError("unsupported chunk codec")
        if expected != entry["nbytes"]:
            raise ValueError("chunk payload size does not match metadata")

    def _stage(self, index, chunk, stats):
        entry = self.entries[index]
        for name in ("codec", "rows", "k", "universe", "dictionary_size"):
            if getattr(chunk, name) != entry[name]:
                raise ValueError("store changed chunk metadata during replay")
        if len(chunk.payload) != entry["nbytes"]:
            raise ValueError("store returned an unexpected payload length")
        slot = self._slots[index % self.buffers]
        # The DMA may still be reading pinned memory after copy_ returns.
        # Waiting for this slot's copy only is not a device-wide barrier.
        if slot.has_copy:
            started = time.perf_counter()
            slot.copied.synchronize()
            stats["host_staging_reuse_wait_seconds"] += time.perf_counter() - started
        slot.host_array[:entry["nbytes"]] = np.frombuffer(chunk.payload, dtype=np.uint8)
        with torch.cuda.stream(self._prepare):
            if slot.has_consumer:
                self._prepare.wait_event(slot.consumed)
            slot.device[:entry["nbytes"]].copy_(
                slot.host[:entry["nbytes"]], non_blocking=True)
            slot.copied.record(self._prepare)
            if entry["codec"] == "raw_u16":
                n = entry["rows"] * entry["k"]
                _raw_decode[(triton.cdiv(n, 256),)](
                    slot.device, slot.output, n, entry["k"], self.max_k, 256)
            else:
                m = entry["dictionary_size"]
                _union_decode[(entry["rows"],)](
                    slot.device, slot.output, m, entry["k"], (m + 7) // 8,
                    self.max_k, 1024, num_warps=4)
            slot.ready.record(self._prepare)
        slot.has_copy = True
        return slot

    def run(self, consume):
        """Run one whole ordered pass; warm up separately before benchmarking.

        Wall time includes store reads, validation, preparation, callback host
        work, start-up and completion. First use may include Triton compilation.
        Host future/event wait times are diagnostic and are NOT GPU stall time.
        Only this replay's preparation and calling compute streams are synced.
        """
        if self._closed:
            raise RuntimeError("CudaReplay is closed")
        if self._running:
            raise RuntimeError("CudaReplay.run is not reentrant")
        self._running = True
        futures = {}
        stats = {"chunks": len(self.entries), "prefetch": self.prefetch,
                 "buffers": self.buffers, "output_buffer_bytes": self.output_bytes,
                 "device_input_buffer_bytes": self.input_bytes,
                 "pinned_buffer_bytes": self.input_bytes,
                 "cpu_read_ahead_payload_upper_bound_bytes":
                     self.buffers * self.max_payload if self.prefetch else self.max_payload,
                 "host_read_wait_seconds": 0.0,
                 "host_staging_reuse_wait_seconds": 0.0,
                 "host_serial_consumer_wait_seconds": 0.0}
        try:
            with torch.cuda.device(self.device):
                self._compute = torch.cuda.current_stream(self.device)
                self._compute.synchronize()
                self._prepare.synchronize()
                start = time.perf_counter()
                if self.prefetch:
                    for index in range(min(self.buffers, len(self.entries))):
                        futures[index] = self._executor.submit(self.store.read, index)
                for index, entry in enumerate(self.entries):
                    if not self.prefetch and index:
                        started = time.perf_counter()
                        self._slots[(index - 1) % self.buffers].consumed.synchronize()
                        stats["host_serial_consumer_wait_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    if self.prefetch:
                        chunk = futures.pop(index).result()
                    else:
                        chunk = self.store.read(index)
                    stats["host_read_wait_seconds"] += time.perf_counter() - started
                    slot = self._stage(index, chunk, stats)
                    del chunk
                    if self.prefetch and index + self.buffers < len(self.entries):
                        following = index + self.buffers
                        futures[following] = self._executor.submit(self.store.read, following)
                    self._compute.wait_event(slot.ready)
                    with torch.cuda.stream(self._compute):
                        consume(slot.output[:entry["rows"], :entry["k"]], index)
                        slot.consumed.record(self._compute)
                    slot.has_consumer = True
                self._compute.synchronize()
                self._prepare.synchronize()
                stats["wall_seconds"] = time.perf_counter() - start
                return stats
        except BaseException:
            for future in futures.values():
                future.cancel()
            self.close()
            raise
        finally:
            self._running = False

    def close(self):
        """Finish queued work before releasing buffers, including on exceptions."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._compute is not None:
                self._compute.synchronize()
            self._prepare.synchronize()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._slots.clear()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("CudaReplay is closed")
        return self

    def __exit__(self, *_):
        self.close()
