"""Sweep head recomputation / tail-only storage using real synthetic QK + top-k.

This is a selector and streaming microbenchmark, not an LLM or prefill result.
Each chunk has a growing key candidate set, beginning with an existing prefix;
rows in a chunk share that candidate count rather than a per-row causal mask.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from selection_cache import CacheIdentity, DiskStore, MemoryStore, encode_chunk, write_store
from selection_cache.policy import SplitPlan


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--universe", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--query-noise", type=float, default=0.05)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, 1/16, 1/8, 1/4, 1/2, 1])
    parser.add_argument("--sources", nargs="+", choices=["memory", "disk"], default=["memory", "disk"])
    parser.add_argument("--codecs", nargs="+", choices=["auto", "raw_u16"], default=["auto", "raw_u16"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if not (1 <= args.chunks <= 32 and 1 <= args.rows <= 128
            and 1 <= args.k <= min(args.universe, 512) and args.universe <= 65536
            and 16 <= args.head_dim <= 128 and 1 <= args.repeats <= 30
            and 1 <= args.buffers <= 4 and 0 <= args.query_noise <= 2):
        parser.error("arguments exceed this bounded benchmark's supported limits")
    if not all(math.isfinite(f) and 0 <= f <= 1 for f in args.fractions):
        parser.error("fractions must be finite and in [0, 1]")
    if args.directory.exists() or (args.output and args.output.exists()):
        parser.error("outputs must be new paths; existing results are never overwritten")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("set CUDA_VISIBLE_DEVICES to exactly one available GPU")
    return parser, args


def main():
    parser, args = parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.directory.resolve() / "triton-cache"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(args.directory.resolve() / "cuda-cache"))
    import torch
    from selection_cache.cuda import CudaReplay

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("one visible CUDA GPU is required")
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        parser.error("less than 8 GiB free; select an idle GPU")
    torch.cuda.set_per_process_memory_fraction(min(0.03, (2 * 1024**3) / total))
    torch.manual_seed(args.seed)
    keys = torch.randn((args.universe, args.head_dim), device="cuda", dtype=torch.bfloat16)
    base_queries = torch.randn((args.chunks, 1, args.head_dim), device="cuda", dtype=torch.bfloat16)
    queries = base_queries + args.query_noise * torch.randn(
        (args.chunks, args.rows, args.head_dim), device="cuda", dtype=torch.bfloat16)
    # Every row starts with >=k candidates. There is no implicit all-visible
    # early-prefix saving in this benchmark, and these are not model Q/K.
    candidates = [args.k + ((args.universe - args.k) * (i + 1)) // args.chunks
                  for i in range(args.chunks)]
    values = torch.randn((args.universe, 32), device="cuda", dtype=torch.bfloat16)
    checks = torch.empty(args.chunks, device="cuda", dtype=torch.int64)
    value_checks = torch.empty(args.chunks, device="cuda", dtype=torch.float32)

    def recompute(ordinal):
        score = torch.mm(queries[ordinal], keys[:candidates[ordinal]].T)
        # Canonical ordering is identical to the cache writer's ordering.
        return torch.sort(torch.topk(score, args.k, dim=-1, sorted=False).indices,
                          dim=-1).values.to(torch.int32)

    def consume(indices, ordinal):
        checks[ordinal].copy_(indices.sum(dtype=torch.int64))
        # Selected-value gather is actual index consumption, but is not a
        # complete attention computation, projection, MLP, GDN, or model.
        value_checks[ordinal].copy_(values[indices.long()].float().sum())

    torch.cuda.synchronize()
    capture_start = time.perf_counter()
    expected = [recompute(i).cpu().numpy() for i in range(args.chunks)]
    capture_seconds = time.perf_counter() - capture_start
    expected_sums = np.asarray([row.sum(dtype=np.int64) for row in expected])
    reference_value_sums = []
    for i, row in enumerate(expected):
        consume(torch.as_tensor(row, device="cuda"), i)
    torch.cuda.synchronize()
    reference_value_sums = value_checks.cpu().numpy().copy()
    config = {key: getattr(args, key) for key in
              ("chunks", "rows", "k", "universe", "head_dim", "query_noise",
               "fractions", "sources", "codecs", "repeats", "buffers", "seed")}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    heads = sorted({SplitPlan.from_fraction(args.chunks, f).recompute_chunks for f in args.fractions})
    modes = {}
    stores = {}
    for codec in dict.fromkeys(args.codecs):
        for head in heads:
            identity = CacheIdentity("synthetic-bf16-qk-v1", fingerprint,
                                     f"torch-topk-canonical-k={args.k};cache-start={head}",
                                     "chunk-candidates=" + ",".join(map(str, candidates)))
            folder = args.directory / f"{codec}-head{head}"
            start = time.perf_counter()
            # Only the tail is encoded and saved. Head indices exist solely
            # in the benchmark's reference oracle, not this cache artifact.
            write_store(folder, identity,
                        (encode_chunk(row, universe=args.universe, codec=codec)
                         for row in expected[head:]))
            encode_seconds = time.perf_counter() - start
            disk = DiskStore(folder, identity)
            payload_bytes = sum(e["nbytes"] for e in disk.entries)
            manifest_bytes = (folder / "manifest.json").stat().st_size
            for source in dict.fromkeys(args.sources):
                name = f"{source}:{codec}:head{head}"
                store = (disk if source == "disk" else MemoryStore(
                    (disk.read(i) for i in range(len(disk))), identity, identity))
                stores[name] = store
                modes[name] = {
                    "source": source, "codec_policy": codec, "head_chunks": head,
                    "tail_chunks": args.chunks - head, "head_fraction": head / args.chunks,
                    "candidate_counts": candidates,
                    "head_score_elements": args.rows * sum(candidates[:head]),
                    "head_score_work_fraction": sum(candidates[:head]) / sum(candidates),
                    "capacity": {"payload_bytes": payload_bytes, "manifest_bytes": manifest_bytes,
                                 "total_store_bytes": payload_bytes + manifest_bytes,
                                 "actual_codecs": {c: sum(e["codec"] == c for e in disk.entries)
                                                   for c in ("raw_u16", "union_bitmap")}},
                    "encode_validate_write_seconds": encode_seconds,
                    "all_indices_exact": False, "samples": [],
                }
    for mode in modes.values():
        baseline = modes.get(f"{mode['source']}:{mode['codec_policy']}:head0")
        if baseline:
            base = baseline["capacity"]["payload_bytes"]
            mode["capacity"]["payload_fraction_of_full_cache"] = mode["capacity"]["payload_bytes"] / base

    # Keep the same replay objects/buffers for every repeated sample. Their
    # aggregate allocation is benchmark harness memory, not per-request memory.
    round_orders = []
    with ExitStack() as stack:
        replays = {name: stack.enter_context(CudaReplay(store, buffers=args.buffers, prefetch=True))
                   for name, store in stores.items()}
        for name, replay in replays.items():
            plan = SplitPlan(args.chunks, modes[name]["head_chunks"])
            captured = []
            replay.run_hybrid(lambda ids, ordinal: captured.append((ordinal, ids.clone())), recompute, plan)
            if [i for i, _ in captured] != list(range(args.chunks)):
                raise AssertionError("chunk ordering differs from the unsplit reference")
            for ordinal, actual in captured:
                np.testing.assert_array_equal(actual.cpu().numpy(), expected[ordinal])
            del captured
            replay.run_hybrid(consume, recompute, plan)
            np.testing.assert_array_equal(checks.cpu().numpy(), expected_sums)
            np.testing.assert_array_equal(value_checks.cpu().numpy(), reference_value_sums)
            modes[name]["all_indices_exact"] = True
        rng = random.Random(args.seed + 1)
        for repeat in range(args.repeats):
            order = list(modes)
            rng.shuffle(order)
            round_orders.append(order)
            for name in order:
                mode = modes[name]
                stats = replays[name].run_hybrid(
                    consume, recompute, SplitPlan(args.chunks, mode["head_chunks"]))
                np.testing.assert_array_equal(checks.cpu().numpy(), expected_sums)
                np.testing.assert_array_equal(value_checks.cpu().numpy(), reference_value_sums)
                stats["repeat"] = repeat
                mode["samples"].append(stats)
    for mode in modes.values():
        wall = [s["wall_seconds"] for s in mode["samples"]]
        mode["wall_ms_median"] = statistics.median(wall) * 1000
        mode["wall_ms_min"] = min(wall) * 1000
        mode["wall_ms_max"] = max(wall) * 1000
    sources = ("split_demo.py", "selection_cache/cuda.py", "selection_cache/policy.py",
               "selection_cache/store.py")
    root = Path(__file__).resolve().parent
    report = {
        "kind": "synthetic-qk-topk-head-recompute-tail-cache", "config": config,
        "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "capture_reference_seconds_including_d2h": capture_seconds,
        "round_orders": round_orders, "modes": modes,
        "peak_torch_allocated_bytes_including_all_modes_and_validation": torch.cuda.max_memory_allocated(),
        "limits": [
            "Synthetic Q/K with real score+topk; no model or production QSA kernel.",
            "Candidate counts grow per chunk, not per-row causal masks; the warm prefix starts with >=k candidates.",
            "Wall time includes head score+topk/sort, tail reads/decode, all consumers, startup and completion.",
            "Encoding, reference capture, validation, JIT warmup and buffer construction are outside timed samples.",
            "File reads are warmed; no cold SSD result or artificial read delay is claimed.",
            "Exact sorted selection sets and consumer checks, not model logits or end-to-end TTFT.",
            "Per-mode buffers are reported in samples; process peak includes all modes and reference validation.",
            "Timing samples are interleaved in a seeded shuffled order; no confidence interval is claimed.",
        ],
    }
    result = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
