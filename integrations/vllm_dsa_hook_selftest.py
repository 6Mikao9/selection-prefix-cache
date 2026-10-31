"""CPU-only hook contract test; does not establish native GPU integration."""
import torch
import vllm.model_executor.layers.sparse_attn_indexer as mod
from vllm_dsa_smoke import NativeSelectorReplay


calls = []


def fake_logits(q, k, weights, starts, ends, **kwargs):
    calls.append(q[0].shape[0])
    return q[0].float().reshape(-1, 1).repeat(1, 10)


def fake_topk(scores, starts, ends, output, rows, stride0, stride1, k):
    for i in range(rows):
        valid = min(int(ends[i]), k)
        output[i].fill_(-1)
        # Deliberately unsorted, preserving selection order matters.
        output[i, :valid] = torch.arange(valid-1, -1, -1)


mod.fp8_fp4_mqa_logits = fake_logits
mod.ops.top_k_per_row_prefill = fake_topk
adapter = NativeSelectorReplay(4)
for head in (0, 4, 10, 20):
    adapter.head_rows = head
    adapter.blobs.clear()
    outputs = []
    calls.clear()
    for mode in ("capture", "replay"):
        adapter.begin(mode)
        adapter.active = "model.layers.0.indexer.k_cache"
        q = torch.arange(10).reshape(10, 1)
        starts = torch.zeros(10, dtype=torch.int32)
        ends = torch.arange(1, 11, dtype=torch.int32)
        output = torch.empty((10, 8), dtype=torch.int32)
        scores = mod.fp8_fp4_mqa_logits((q, None), (q, q), q, starts, ends)
        mod.ops.top_k_per_row_prefill(scores, starts, ends, output,
                                     scores.shape[0], *scores.stride(), 8)
        outputs.append(output)
    assert torch.equal(*outputs)
    assert calls == ([10] if head == 0 else [10, min(head, 10)]), calls
    assert adapter.events[0]["skipped"] == 10-min(head, 10)
print("PASS: CPU hook contract; head=0/4/10/20, ordered tail and -1 padding")
