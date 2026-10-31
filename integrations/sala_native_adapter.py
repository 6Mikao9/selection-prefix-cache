"""Experimental native MiniCPM-SALA logical-selection replay for SGLang 0.5.20.

Install in each model worker before Engine construction, e.g. via sitecustomize.
This adapter targets single-request, unchunked prefill and native top-k.
KV compression, linear state, and logical-to-physical translation still execute.
It does not implement the repository's CUDA codec/double-buffer pipeline.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import zlib


def install(control_path, log_path):
    import torch
    import numpy as np
    import sglang
    from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend
    from sglang.srt.layers.attention.minicpm.sparse_utils import (
        compressed_attention, compressed_attention_tilelang)

    if getattr(MiniCPMSparseBackend, "_selection_replay_installed", False):
        return
    original = MiniCPMSparseBackend.sparse_get_topk_impl
    original_outer = MiniCPMSparseBackend.get_topk_for_sparse
    records = {}

    def digest(tensor):
        raw=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def emit(data):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def outer(self, query_states, key_states, layer, forward_batch, is_prefill=True):
        self._selection_layer_id = layer.layer_id
        self._selection_prefill = is_prefill
        return original_outer(self, query_states, key_states, layer, forward_batch,
                              is_prefill=is_prefill)

    def hook(self, query_layer, cu_seqlens_q, cu_seqlens_k,
             max_seqlen_in_batch_q, max_seqlen_in_batch_k,
             compressed_k=None, compressed_cu_seqlens=None,
             compressed_k2=None, compressed_cu_seqlens2=None, fused_kernel=None):
        kwargs = dict(compressed_k=compressed_k,
                      compressed_cu_seqlens=compressed_cu_seqlens,
                      compressed_k2=compressed_k2,
                      compressed_cu_seqlens2=compressed_cu_seqlens2,
                      fused_kernel=fused_kernel)
        def native():
            return original(self, query_layer, cu_seqlens_q, cu_seqlens_k,
                            max_seqlen_in_batch_q, max_seqlen_in_batch_k, **kwargs)
        if not getattr(self, "_selection_prefill", False) or query_layer.shape[0] <= 1:
            return native()
        control = json.loads(Path(control_path).read_text(encoding="utf-8"))
        mode = control.get("mode", "baseline")
        if mode == "baseline":
            result = native()
            emit(dict(mode=mode, layer=self._selection_layer_id,
                      identity=control.get('identity'),run_id=control.get('run_id'),
                      block_size=self.block_size,
                      effective_head=query_layer.shape[0],
                      native_selector_queries=query_layer.shape[0],
                      query_tokens=query_layer.shape[0], native_selector_called=True))
            return result
        if (cu_seqlens_q.numel() != 2
                or query_layer.shape[0] != max_seqlen_in_batch_q
                or max_seqlen_in_batch_q != max_seqlen_in_batch_k):
            raise RuntimeError("Replay requires single-request, unchunked full prefill")
        identity = control.get("identity")
        if not identity:
            raise ValueError("A content-derived request identity is required")
        n = query_layer.shape[0]
        fraction = float(control.get("head_fraction", 0))
        head = n if fraction >= 1 else int(n*fraction)//self.block_size*self.block_size
        head = min(n, max(0, head))
        signature = (identity,id(self),self._selection_layer_id,tuple(query_layer.shape),
                     str(query_layer.dtype), self.block_size, self.sparse_topk,
                     self.kernel_topk, self.max_context_len, self.kernel_stride,
                     self.k1_kernel_size,self.k1_kernel_stride,
                     self.k2_kernel_size,self.k2_kernel_stride,
                     self.init_blocks,self.local_blocks,self.dense_len,
                     sglang.__version__,self.minicpm_fuse_topk,head)
        if mode == "capture":
            # Retain only one request/cut's CPU tail; no persistent GPU tensor.
            if records and (next(iter(records))[0],next(iter(records))[-1]) != (identity,head):
                records.clear()
            start=time.perf_counter()
            result = native()
            raw=result[:,head:].detach().contiguous().cpu().numpy().tobytes()
            blob=zlib.compress(raw,level=1)
            records[signature]=dict(shape=tuple(result.shape),tail=blob,
                head_digest=digest(result[:,:head]) if control.get('verify_indices') else None,
                query_digest=digest(query_layer) if control.get('verify_queries') else None)
            emit(dict(mode=mode, layer=self._selection_layer_id, shape=list(result.shape),
                      identity=identity,run_id=control.get('run_id'),effective_head=head,
                      block_size=self.block_size,
                      native_selector_queries=n,
                      query_tokens=n,recomputed_queries=n,replayed_queries=0,
                      planned_tail_queries=n-head,
                      tail_raw_bytes=len(raw),tail_compressed_bytes=len(blob),
                      full_raw_bytes=result.numel()*result.element_size(),
                      capture_seconds=time.perf_counter()-start,
                      persistent_gpu_bytes=0,native_selector_called=True))
            return result
        if mode != "replay" or signature not in records:
            raise RuntimeError("Missing matching native selection capture")
        record=records[signature]
        if record['query_digest'] is not None and digest(query_layer)!=record['query_digest']:
            raise RuntimeError("Replay query states differ from capture; refusing stale selections")
        result=torch.empty(record['shape'],device=query_layer.device,dtype=torch.int32)
        decode_start=time.perf_counter()
        tail_raw=zlib.decompress(record['tail'])
        tail_shape=(record['shape'][0],n-head,record['shape'][2])
        tail=torch.from_numpy(np.frombuffer(tail_raw,dtype=np.int32).copy().reshape(tail_shape))
        result[:,head:].copy_(tail,non_blocking=False)
        decode_seconds=time.perf_counter()-decode_start
        if head:
            cu_head = torch.tensor([0, head], device=query_layer.device, dtype=torch.int32)
            # Stage1 uses bottom-right causal alignment. A short Q against all
            # compressed keys would shift Q to the end of the original context.
            # Crop BOTH compression levels to precisely the head's native keys.
            k1n = max(0, (head-self.k1_kernel_size)//self.k1_kernel_stride+1)
            k2n = max(0, (head-self.k2_kernel_size)//self.k2_kernel_stride+1)
            k1 = compressed_k[:k1n]
            k2 = compressed_k2[:k2n]
            cu_k1 = torch.tensor([0,k1n], device=query_layer.device,dtype=torch.int32)
            cu_k2 = torch.tensor([0,k2n], device=query_layer.device,dtype=torch.int32)
            cache_lens = torch.zeros(1,device=query_layer.device,dtype=torch.int32)
            if self.minicpm_fuse_topk:
                selected_head = compressed_attention_tilelang(
                    query_layer[:head], k1, self.block_size, self.sparse_topk,
                    self.kernel_topk, cu_head, cu_k1, cache_lens=cache_lens,
                    fused_kernel=fused_kernel,max_cache_len=self.max_context_len)
            else:
                selected_head = compressed_attention(
                    query_layer[:head], k1, k2, self.kernel_stride,self.block_size,
                    self.sparse_topk,cu_head,cu_k1,cu_k2,head,self.max_context_len,
                    init_blocks=self.init_blocks,local_blocks=self.local_blocks,
                    cache_lens=cache_lens,
                    cu_seqlens_q_adjusted=cu_head*self.heads_per_group,
                    max_seqlen_q_adjusted=head*self.heads_per_group)
            result[:, :head] = selected_head
        equal = None
        if control.get("verify_indices"):
            equal = digest(result[:,:head])==record['head_digest']
            if not equal:
                raise RuntimeError("Head recomputation changes native sorted/padded top-k")
        emit(dict(mode=mode, layer=self._selection_layer_id, query_tokens=n,
                  identity=identity,run_id=control.get('run_id'),effective_head=head,
                  block_size=self.block_size,
                  native_selector_queries=head,
                  recomputed_queries=head, replayed_queries=n-head,
                  native_selector_called=bool(head), head_indices_equal=equal,
                  fused=self.minicpm_fuse_topk,tail_decode_copy_seconds=decode_seconds,
                  tail_raw_bytes=len(tail_raw),tail_compressed_bytes=len(record['tail']),
                  persistent_gpu_bytes=0))
        return result

    MiniCPMSparseBackend.get_topk_for_sparse = outer
    MiniCPMSparseBackend.sparse_get_topk_impl = hook
    MiniCPMSparseBackend._selection_replay_installed = True
