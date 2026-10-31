"""Synthetic index replay demo; no model weights or private trace required."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Keep host-side work bounded when run on a shared machine.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from selection_cache import (
    CacheIdentity, DiskStore, MemoryStore, decode_chunk, encode_chunk, write_store,
)


def synthetic(args):
    """Mix correlated and independent sets to exercise compressed/raw fallback."""
    rng = np.random.default_rng(args.seed)
    for ordinal in range(args.chunks):
        pool_size = min(args.universe, args.k + max(1, args.k // 8))
        pool = (rng.choice(args.universe, pool_size, replace=False)
                if ordinal % 3 != 2 else None)
        rows = [rng.choice(pool if pool is not None else args.universe,
                           args.k, replace=False) for _ in range(args.rows)]
        yield np.sort(np.asarray(rows, dtype=np.uint16), axis=1)


def cpu_replay(store, expected):
    start = time.perf_counter()
    for index in range(len(store)):
        actual = decode_chunk(store.read(index))
        np.testing.assert_array_equal(actual, expected[index])
    return {"wall_ms_including_validation": (time.perf_counter() - start) * 1000,
            "all_indices_exact": True, "chunks": len(store)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--directory", type=Path, required=True,
                        help="New directory; existing output is never overwritten")
    parser.add_argument("--source", choices=("disk", "memory"), default="disk")
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--universe", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--compute-size", type=int, default=1024)
    parser.add_argument("--compute-iters", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (1 <= args.k <= args.universe <= 65536):
        parser.error("require 1 <= k <= universe <= 65536")
    if not (1 <= args.chunks <= 512 and 1 <= args.rows <= 4096
            and 1 <= args.repeats <= 30 and 1 <= args.buffers <= 8
            and 16 <= args.compute_size <= 4096 and 0 <= args.compute_iters <= 32):
        parser.error("arguments exceed this bounded demo's supported limits")
    if args.chunks * args.rows * args.k * 2 > 128 * 1024**2:
        parser.error("synthetic reference data must fit within 128 MiB")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.directory.exists():
        parser.error("directory already exists; choose a new path")

    args.directory.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda":
        # Configure caches before CUDA initializes; keep them inside this run.
        os.environ.setdefault("TRITON_CACHE_DIR", str(args.directory.resolve() / "triton-cache"))
        os.environ.setdefault("CUDA_CACHE_PATH", str(args.directory.resolve() / "cuda-cache"))
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA not available; use an existing CUDA environment")
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("explicitly set CUDA_VISIBLE_DEVICES to an available GPU")
        if torch.cuda.device_count() != 1:
            parser.error("expose exactly one GPU for this bounded demo")
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            parser.error("less than 8 GiB free; select an idle GPU")
        torch.cuda.set_per_process_memory_fraction(min(0.03, (2 * 1024**3) / total))
    config = {name: getattr(args, name) for name in
              ("chunks", "rows", "k", "universe", "seed", "buffers",
               "compute_size", "compute_iters", "repeats", "source", "device")}
    identity = CacheIdentity(
        model_fingerprint="synthetic-selection-demo-v1",
        prefix_digest=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        selector_fingerprint=f"canonical-set-k={args.k}",
        position_mask_fingerprint="synthetic-noncausal-no-model",
    )
    expected = list(synthetic(args))
    stores, capacities = {}, {}
    for label, codec in (("raw", "raw_u16"), ("compressed", "auto")):
        start = time.perf_counter()
        write_store(args.directory / label, identity,
                    (encode_chunk(x, universe=args.universe, codec=codec) for x in expected))
        encoded_ms = (time.perf_counter() - start) * 1000
        disk = DiskStore(args.directory / label, expected_identity=identity)
        stores[label] = (disk if args.source == "disk" else
                         MemoryStore([disk.read(i) for i in range(len(disk))],
                                     identity, expected_identity=identity))
        capacities[label] = {
            "payload_bytes": sum(e["nbytes"] for e in disk.entries),
            "manifest_bytes": (args.directory / label / "manifest.json").stat().st_size,
            "codecs": {c: sum(e["codec"] == c for e in disk.entries)
                       for c in ("raw_u16", "union_bitmap")},
            "encode_validate_write_ms": encoded_ms,
        }
    report = {
        "kind": "synthetic-streaming-selection-replay",
        "config": config, "capacity": capacities,
        "limits": ["No model, selector or end-to-end prefill integration.",
                   "Disk reads may hit the OS page cache; not a cold SSD benchmark.",
                   "Selection sets are sorted; original top-k order is not preserved.",
                   "Only index buffers are bounded; this does not replace the main KV cache."],
        "cpu_roundtrip": {label: cpu_replay(store, expected) for label, store in stores.items()},
    }
    if args.device == "cuda":
        from selection_cache.cuda import CudaReplay
        torch.manual_seed(args.seed)
        a = torch.randn((args.compute_size, args.compute_size), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        checks = torch.empty(args.chunks, device="cuda", dtype=torch.int64)
        expected_sums = np.asarray([x.sum(dtype=np.int64) for x in expected])

        def consume(indices, ordinal):
            # A repeatable synthetic consumer, not attention. Each output is read
            # before its ring-buffer slot can be reused by the preparation stream.
            checks[ordinal].copy_(indices.sum(dtype=torch.int64))
            for _ in range(args.compute_iters):
                torch.mm(a, b, out=c)

        report["cuda"] = {}
        for label, store in stores.items():
            for prefetch in (False, True):
                mode = f"{label}_{'prefetch' if prefetch else 'serial'}"
                with CudaReplay(store, buffers=args.buffers, prefetch=prefetch) as replay:
                    # Full index validation outside the timed benchmark. Clones
                    # intentionally retain outputs only for this small demo check.
                    captured = []
                    replay.run(lambda indices, _: captured.append(indices.clone()))
                    for actual, reference in zip(captured, expected):
                        np.testing.assert_array_equal(actual.cpu().numpy(), reference)
                    del captured
                    replay.run(consume)  # warm kernel variants and the consumer
                    samples = []
                    for _ in range(args.repeats):
                        stats = replay.run(consume)
                        np.testing.assert_array_equal(checks.cpu().numpy(), expected_sums)
                        samples.append(stats)
                report["cuda"][mode] = {"all_indices_exact": True, "samples": samples}
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["torch_version"] = torch.__version__
        report["peak_torch_allocated_bytes_including_validation"] = torch.cuda.max_memory_allocated()
    output = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
