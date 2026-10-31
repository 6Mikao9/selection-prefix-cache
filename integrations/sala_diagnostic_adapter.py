"""Diagnostic only: full GPU selection retention trades memory for selector bypass.
Install via sitecustomize before worker Engine initialization. Not a cache policy.
"""
import hashlib
import json
from pathlib import Path


def install(control_path, log_path):
    import torch
    from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend as Backend
    if getattr(Backend, '_sala_diagnostic_installed', False):
        return
    if getattr(Backend, '_selection_replay_installed', False):
        raise RuntimeError('Diagnostic and production experiment hooks must not coexist')
    original, outer_original = Backend.sparse_get_topk_impl, Backend.get_topk_for_sparse
    records, pending, native_checks = {}, [], {}

    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def outer(self, query_states, key_states, layer, forward_batch, is_prefill=True):
        self._diagnostic_layer = layer.layer_id
        self._diagnostic_prefill = is_prefill
        return outer_original(self, query_states, key_states, layer, forward_batch, is_prefill=is_prefill)

    def hook(self, query, cu_q, cu_k, max_q, max_k, *args, **kwargs):
        if not getattr(self, '_diagnostic_prefill', False) or query.shape[0] <= 1:
            return original(self, query, cu_q, cu_k, max_q, max_k, *args, **kwargs)
        c = json.loads(Path(control_path).read_text())
        if pending and pending[0][0]['run_id'] != c['run_id']:
            drain()  # Previous generate completed before the next control was written.
        mode = c['mode']
        if cu_q.numel() != 2 or query.shape[0] != max_q or max_q != max_k:
            raise RuntimeError('Diagnostic requires TP1 single-request unchunked full prefill')
        key = (c['identity'], id(self), self._diagnostic_layer, tuple(query.shape), str(query.dtype))
        verify, profile = c.get('verify', False), c.get('profile', False)
        start = torch.cuda.Event(enable_timing=True) if profile else None
        end = torch.cuda.Event(enable_timing=True) if profile else None
        if start is not None: start.record()
        if mode in ('baseline', 'capture'):
            result = original(self, query, cu_q, cu_k, max_q, max_k, *args, **kwargs)
        elif mode in ('gpu', 'pinned'):
            if key not in records: raise RuntimeError('Missing matching capture')
            result = records[key]['gpu'] if mode == 'gpu' else records[key]['cpu'].to(query.device, non_blocking=True)
        else:
            raise ValueError('Unknown diagnostic mode')
        if end is not None: end.record()
        if mode == 'baseline' and verify:
            native_checks[key] = (digest(query), digest(result))
        if mode == 'capture':
            if key not in native_checks or native_checks[key] != (digest(query), digest(result)):
                raise RuntimeError('Capture differs from verified native query/ordered padded selection')
            # Retain both representations only when explicitly requested.
            records[key] = dict(gpu=result.detach().clone(), query=digest(query), indices=digest(result))
            if c.get('pinned'):
                records[key]['cpu'] = result.detach().to('cpu').pin_memory()
        record = records.get(key)
        if verify and mode in ('gpu', 'pinned'):
            if digest(query) != record['query'] or digest(result) != record['indices']:
                raise RuntimeError('Query or ordered padded selection differs from capture')
        pending.append((dict(run_id=c['run_id'], identity=c['identity'], mode=mode,
            layer=self._diagnostic_layer, query_tokens=query.shape[0],
            verified=bool(verify and mode in ('gpu','pinned')),
            native_selector_called=mode in ('baseline','capture'),
            persistent_gpu_bytes=sum(r['gpu'].numel()*r['gpu'].element_size() for r in records.values()),
            pinned_cpu_bytes=sum(r['cpu'].numel()*r['cpu'].element_size() for r in records.values() if 'cpu' in r)), start, end))
        return result

    # Each worker drains the previous completed request when its next request
    # reaches the hook; no individual selector layer synchronizes for profiling.
    def drain():
        if pending and any(end is not None for _, _, end in pending):
            torch.cuda.synchronize()
        with open(log_path, 'a', encoding='utf-8') as stream:
            for event, start, end in pending:
                if start is not None: event['selector_or_h2d_cuda_ms'] = start.elapsed_time(end)
                stream.write(json.dumps(event)+'\n')
        count = len(pending)
        pending.clear()
        return count
    Backend.get_topk_for_sparse = outer
    Backend.sparse_get_topk_impl = hook
    Backend._sala_diagnostic_installed = True
    Backend._sala_diagnostic_drain = staticmethod(drain)
