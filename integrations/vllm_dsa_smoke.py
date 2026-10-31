"""Native vLLM DSA selector capture/replay architecture smoke (TP=1).

Version-specific eager process-local hooks; no installed files are changed.
Ordered int32 logical indices including -1 padding are compressed with zlib.
This is an adapter baseline, NOT the sorted-uint16 selection_cache codec.
Only a single request, unchunked prefill, no speculative decoding is supported.
The original indexer K insertion and attention block-table mapping still run.
"""
import argparse
import hashlib
import json
import os
import time
import zlib
from pathlib import Path


class NativeSelectorReplay:
    def __init__(self, head_rows):
        import torch
        import vllm.model_executor.layers.sparse_attn_indexer as mod

        self.torch, self.mod = torch, mod
        self.mode = "off"
        self.head_rows = head_rows
        self.blobs = {}
        self.counts = {}
        self.events = []
        self.active = None
        self.pending = None
        original_forward = mod.SparseAttnIndexer.forward_cuda
        original_logits = mod.fp8_fp4_mqa_logits
        original_topk = mod.ops.top_k_per_row_prefill
        original_sparse = mod.sparse_attn_indexer
        owner = self

        def forward(indexer, *args, **kwargs):
            previous = owner.active
            owner.active = indexer.k_cache.prefix
            try:
                return original_forward(indexer, *args, **kwargs)
            finally:
                owner.active = previous

        def sparse(*args, **kwargs):
            # vLLM's optimized NVIDIA model calls the eager function directly,
            # bypassing SparseAttnIndexer.forward_cuda.
            previous = owner.active
            from vllm.utils.torch_utils import _resolve_layer_name
            owner.active = _resolve_layer_name(args[1])
            try:
                return original_sparse(*args, **kwargs)
            finally:
                owner.active = previous

        def logits(q, k, weights, starts, ends, **kwargs):
            n = q[0].shape[0]
            if owner.mode == "off":
                return original_logits(q, k, weights, starts, ends, **kwargs)
            assert owner.active is not None
            ordinal = owner.counts.get(owner.active, 0)
            owner.counts[owner.active] = ordinal + 1
            key = (owner.active, ordinal)
            h = min(n, owner.head_rows) if owner.mode == "replay" else n
            owner.pending = (key, n, h)
            if h:
                return original_logits(
                    (q[0][:h], None if q[1] is None else q[1][:h]),
                    k, weights[:h], starts[:h], ends[:h], **kwargs)
            return torch.empty((0, k[0].shape[0]), device=weights.device,
                               dtype=torch.float32)

        def topk(scores, starts, ends, output, num_rows, stride0, stride1, k):
            if owner.mode == "off":
                return original_topk(scores, starts, ends, output, num_rows,
                                     stride0, stride1, k)
            assert owner.pending is not None
            key, n, h = owner.pending
            owner.pending = None
            assert output.shape == (n, k)
            if h:
                original_topk(scores, starts[:h], ends[:h], output[:h], h,
                              scores.stride(0), scores.stride(1), k)
            if owner.mode == "capture":
                # Preserve rank order, sentinel padding, and exact logical values.
                a = output.detach().cpu().contiguous().numpy()
                assert a.dtype.name == "int32"
                split = min(n, owner.head_rows)
                owner.blobs[key] = (a.shape, zlib.compress(a[split:].tobytes()),
                                    hashlib.sha256(a[:split].tobytes()).hexdigest())
            else:
                import numpy as np
                shape, payload, head_hash = owner.blobs[key]
                assert shape == tuple(output.shape), (shape, output.shape)
                a = np.frombuffer(zlib.decompress(payload), dtype=np.int32).copy()
                cached = torch.from_numpy(a.reshape(n-h, k)).to(output.device)
                # Every cached index is logical; native attention subsequently
                # translates using the CURRENT request block table.
                output[h:].copy_(cached)
                actual_head = output[:h].detach().cpu().contiguous().numpy()
                assert hashlib.sha256(actual_head.tobytes()).hexdigest() == head_hash, \
                    "head selector mismatch"
            owner.events.append(dict(layer=key[0], chunk=key[1], rows=n,
                                     recomputed=h, skipped=n-h))

        mod.SparseAttnIndexer.forward_cuda = forward
        mod.fp8_fp4_mqa_logits = logits
        mod.ops.top_k_per_row_prefill = topk
        mod.sparse_attn_indexer = sparse
        import vllm.models.deepseek_v32.attention as optimized
        optimized.sparse_attn_indexer = sparse

    def begin(self, mode):
        self.mode = mode
        self.counts.clear()
        self.events.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--head", type=int, default=2048)
    p.add_argument("--output", default="native_dsa_smoke.json")
    a = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"), "bind one idle GPU"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    import torch

    adapter = NativeSelectorReplay(a.head)
    llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=max(a.lengths)+16,
              max_num_batched_tokens=max(a.lengths)+16, max_num_seqs=1,
              gpu_memory_utilization=0.45, dtype="bfloat16",
              disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    unit = tokenizer.encode("The library stores books. A reader finds a book. ",
                            add_special_tokens=False)
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=10,
                              ignore_eos=True, seed=0)
    results = []
    for n in a.lengths:
        prompt = (unit * (n // len(unit)+1))[:n]
        outputs = []
        timings = []
        events = []
        adapter.blobs.clear()
        for mode in ("capture", "replay"):
            llm.reset_prefix_cache()
            adapter.begin(mode)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = llm.generate([dict(prompt_token_ids=prompt)], sampling,
                               use_tqdm=False)[0].outputs[0]
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-t)
            outputs.append(out)
            events.append(list(adapter.events))
        assert events[0] and events[1], "No real DSA prefill selector calls intercepted"
        assert outputs[0].token_ids == outputs[1].token_ids
        lp = [{str(k): float(v.logprob) for k, v in o.logprobs[0].items()}
              for o in outputs]
        assert lp[0] == lp[1], (lp[0], lp[1])
        results.append(dict(tokens=n, head=a.head, token_ids=outputs[0].token_ids,
                            logprobs_exact=True, seconds=timings, events=events,
                            compressed_bytes=sum(len(x[1]) for x in adapter.blobs.values())))
        print(json.dumps(results[-1]), flush=True)
    Path(a.output).write_text(json.dumps(dict(
        caveat="Architecture smoke only; tiny model randomly initialized and toy finetuned. "
               "Ordered-int32 zlib adapter baseline; synchronized hooks invalidate latency claims.",
        prefix_caching=False, eager=True, results=results), indent=2))


if __name__ == "__main__":
    main()
