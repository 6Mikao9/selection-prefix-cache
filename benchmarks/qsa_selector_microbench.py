"""Random-input, real QSA selector kernels; NOT a full-model prefill benchmark.

The host control holds fully expanded, predecoded indices in pinned RAM. It is
an unbounded diagnostic, not a compressed cache implementation. No attention
consumer is timed. Output creation uses exclusive mode to protect prior results.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', nargs='+', type=int, default=[2048,8192,16384,32768])
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--block-topk', type=int, default=512)
    parser.add_argument('--compress-ratio', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if min(args.lengths + [args.heads,args.head_dim,args.block_topk,args.compress_ratio,args.repeats]) <= 0:
        parser.error('dimensions and repeats must be positive')
    import torch
    from sglang.srt.layers.attention.qsa import mqa, kernel, qsa_indexer
    if not torch.cuda.is_available() or not mqa.HAS_TILELANG:
        raise RuntimeError('CUDA and TileLang required; reference fallback prohibited')
    score_fn = mqa.tilelang_qsa_mqa_prefill
    topk_fn = kernel.qsa_fast_topk
    expand_fn = kernel.expand_qsa_block_indices
    def source_hash(module):
        return hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
    versions = {}
    for name in ['torch','sglang','tilelang','triton','sgl-kernel']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    result = dict(kind='synthetic_random_QK_real_QSA_selector_microbenchmark',
        limitations=['not full prefill or TTFT', 'not real model activations or trace',
                      'host control fully expanded and predecoded; no compression or bounded prefetch',
                      'no attention consumer; no headcuts'],
        parameters={k:v for k,v in vars(args).items() if k != 'output'},
        config_source=dict(url='https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json',
            sha256='889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b',
            indexer_n_heads=4,indexer_head_dim=128,indexer_budget=2048,indexer_compress_ratio=4),
        versions=versions, torch_cuda_runtime=torch.version.cuda,
        nvcc_version=subprocess.check_output([str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'--version'],text=True),
        source_sha256={m.__name__:source_hash(m) for m in [mqa,kernel,qsa_indexer]},
        gpu=dict(name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()),
        timing='Unprofiled rows have outer events only; separate profile rows pair stage events per chunk; sync only whole pipeline tail; wall time includes tail sync',
        records=[])
    try:
        topk_module = __import__('sglang.kernels.ops.elementwise.fast_topk', fromlist=['fast_topk'])
    except ImportError:
        topk_module = __import__('sglang.kernels.ops.attention.fast_topk', fromlist=['fast_topk'])
    result['source_sha256'][topk_module.__name__] = source_hash(topk_module)
    native_headers = list(Path(inspect.getfile(topk_module)).parents[2].glob('jit/**/fast_topk.cuh'))
    if not native_headers:
        raise RuntimeError('Cannot locate native fast_topk.cuh for provenance')
    for header in native_headers:
        result['source_sha256']['native/' + header.name] = hashlib.sha256(header.read_bytes()).hexdigest()
    result['limitations'].append('All CPU/GPU representations coexist; no deployment memory-saving claim; uint16 is dtype narrowing only')
    result['disabled_modes'] = []
    rng = random.Random(args.seed)
    for length in args.lengths:
        keys = length // args.compress_ratio
        rows = length
        chunk = qsa_indexer._qsa_prefill_row_chunk_size(rows,keys,args.heads)
        padded = ((chunk + max(1,128//args.heads)-1)//max(1,128//args.heads))*max(1,128//args.heads)
        if padded*keys*4 > 256*1024**2:
            raise RuntimeError('logits allocation exceeds hard 256 MiB limit')
        for repetition in range(args.repeats):
            torch.manual_seed(args.seed + repetition)
            q = torch.randn(rows,args.heads,args.head_dim,device='cuda',dtype=torch.bfloat16)
            k = torch.randn(keys,1,args.head_dim,device='cuda',dtype=torch.bfloat16)
            positions = torch.arange(rows,device='cuda',dtype=torch.int32)
            starts = torch.zeros_like(positions)
            ends = ((positions+1)//args.compress_ratio).clamp(max=keys)
            seq = torch.full_like(positions,length)
            width = args.block_topk*args.compress_ratio+args.compress_ratio-1
            def native(sample=False, save_blocks=None):
                events=[]
                output=torch.empty((rows,width),device='cuda',dtype=torch.int32)
                for begin in range(0,rows,chunk):
                    sl=slice(begin,min(rows,begin+chunk))
                    e=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if sample else None
                    if e: e[0].record()
                    logits=score_fn(q[sl],k,starts[sl],ends[sl])
                    if e: e[1].record()
                    indices=topk_fn(logits,starts[sl],ends[sl],topk=args.block_topk)
                    if e: e[2].record()
                    if save_blocks is not None:
                        save_blocks[sl].copy_(indices)
                    selected=expand_fn(indices,positions[sl],seq[sl],compress_ratio=args.compress_ratio,
                                       token_topk=args.block_topk*args.compress_ratio)
                    if e: e[3].record(); events.append(e)
                    output[sl].copy_(selected)
                    del logits, indices, selected
                return output,events
            # Compilation, allocator warmup, and capture are outside reported samples.
            warm_start=time.perf_counter()
            captured_blocks=torch.empty((rows,args.block_topk),device='cuda',dtype=torch.int32)
            reference,_=native(save_blocks=captured_blocks)
            native()
            torch.cuda.synchronize()
            warm_seconds=time.perf_counter()-warm_start
            host=reference.cpu().pin_memory()
            # Capture native blocks directly. Expansion packs the partial tail
            # before padding on short rows, so striding expanded tokens alone
            # cannot reliably recover the original padded block representation.
            host_blocks = captured_blocks.cpu().pin_memory()
            device_blocks = torch.empty_like(captured_blocks)
            host_u16 = device_u16 = None
            if keys <= 65535:
                encoded = torch.where(captured_blocks >= 0, captured_blocks, 65535).to(torch.uint16)
                host_u16 = encoded.cpu().pin_memory()
                device_u16 = torch.empty_like(encoded)
                del encoded
            else:
                result['disabled_modes'].append(dict(length=length,mode='host_blocks_u16_h2d_expand',reason='keys exceeds 65535; sentinel reserved'))
            def block_replay(narrow=False):
                if narrow:
                    device_u16.copy_(host_u16,non_blocking=True)
                    blocks=device_u16.to(torch.int32)
                    blocks.masked_fill_(blocks == 65535, -1)
                else:
                    device_blocks.copy_(host_blocks,non_blocking=True)
                    blocks=device_blocks
                return expand_fn(blocks,positions,seq,compress_ratio=args.compress_ratio,
                                 token_topk=args.block_topk*args.compress_ratio)
            if not torch.equal(block_replay(), reference):
                raise AssertionError('Captured block reconstruction did not preserve exact output')
            if host_u16 is not None and not torch.equal(block_replay(True), reference):
                raise AssertionError('uint16 block replay did not preserve exact output')
            sorted_reference=reference.sort(dim=-1).values
            def compare_native_or_replay(output, mode):
                exact=torch.equal(output,reference)
                if exact:
                    return dict(exact_order_and_padding=True,selected_multiset_and_padding_equal=True,
                                boundary_tie_rows=0,tie_equivalent_to_capture=True)
                if mode!='native':
                    raise AssertionError('Replay failed to preserve captured ordering/padding')
                sorted_output=output.sort(dim=-1).values
                changed=(sorted_output!=sorted_reference).any(dim=1).nonzero().flatten()
                if not changed.numel():
                    return dict(exact_order_and_padding=False,selected_multiset_and_padding_equal=True,
                                boundary_tie_rows=0,tie_equivalent_to_capture=True)
                # ReLU can make zero the k-th boundary for short causal rows.
                # Atomic tie-breaking can then change the selected SET as well
                # as its order. Independently require every exchanged block to
                # have exactly the k-th score; do not accept approximate ties.
                scores=score_fn(q[changed],k,starts[changed],ends[changed]).cpu().numpy()
                previous=reference[changed].cpu().numpy()
                current=output[changed].cpu().numpy()
                end_values=ends[changed].cpu().tolist()
                import numpy as np
                for i,(before,after,limit) in enumerate(zip(previous,current,end_values)):
                    a=before[:args.block_topk*args.compress_ratio:args.compress_ratio]//args.compress_ratio
                    b=after[:args.block_topk*args.compress_ratio:args.compress_ratio]//args.compress_ratio
                    aset,bset=set(a[a>=0].tolist()),set(b[b>=0].tolist())
                    width=min(limit,args.block_topk)
                    valid=(len(aset)==len(bset)==width and all(0<=x<limit for x in aset|bset)
                           and np.array_equal(before[args.block_topk*args.compress_ratio:],after[args.block_topk*args.compress_ratio:]))
                    threshold=np.partition(scores[i,:limit],limit-width)[limit-width] if width else float('inf')
                    exchanged=list(aset^bset)
                    valid=valid and all(scores[i,x]>=threshold for x in aset|bset)
                    valid=valid and all(scores[i,x]==threshold for x in exchanged)
                    if not valid:
                        raise AssertionError(f'Native selection differs beyond exact boundary ties: length={length}, query={changed[i].item()}')
                return dict(exact_order_and_padding=False,selected_multiset_and_padding_equal=False,
                            boundary_tie_rows=int(changed.numel()),tie_equivalent_to_capture=True)
            destination=torch.empty_like(reference)
            destination.copy_(host,non_blocking=True)
            torch.cuda.synchronize()
            modes=['native','gpu_resident','host_predecoded_h2d']
            modes.append('host_blocks_i32_h2d_expand')
            if host_u16 is not None:
                modes.append('host_blocks_u16_h2d_expand')
            rng.shuffle(modes)
            for mode in modes:
                events=[]
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                wall_start=time.perf_counter(); start.record()
                if mode=='native': output,events=native(False)
                elif mode=='gpu_resident': output=reference
                elif mode=='host_blocks_i32_h2d_expand': output=block_replay()
                elif mode=='host_blocks_u16_h2d_expand': output=block_replay(True)
                else:
                    destination.copy_(host,non_blocking=True); output=destination
                end.record(); end.synchronize()
                wall_ms=(time.perf_counter()-wall_start)*1000
                # Compare exact native ordering and -1 slots, outside timed interval.
                comparison=compare_native_or_replay(output,mode)
                paired=[dict(score_ms=e[0].elapsed_time(e[1]),topk_ms=e[1].elapsed_time(e[2]),
                             expand_ms=e[2].elapsed_time(e[3]),sum_ms=e[0].elapsed_time(e[3])) for e in events]
                result['records'].append(dict(length=length,repetition=repetition,mode=mode,phase='unprofiled',
                    transfer_bytes=(host.numel()*host.element_size() if mode=='host_predecoded_h2d' else
                                    host_blocks.numel()*4 if mode=='host_blocks_i32_h2d_expand' else
                                    host_u16.numel()*2 if mode=='host_blocks_u16_h2d_expand' else 0),
                    storage_bytes=(0 if mode=='native' else host_u16.numel()*2 if mode=='host_blocks_u16_h2d_expand' else
                                   host_blocks.numel()*4 if mode=='host_blocks_i32_h2d_expand' else reference.numel()*4),
                    chunk_rows=chunk,logits_max_bytes=padded*keys*4,expanded_bytes=reference.numel()*4,
                    cuda_ms=start.elapsed_time(end),wall_ms=wall_ms,**comparison,
                    paired_chunks=paired,warmup_seconds=warm_seconds))
            # Stage events are a separate measurement, never used for the
            # unprofiled replay comparison above.
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            wall_start=time.perf_counter(); start.record()
            output,events=native(True)
            end.record(); end.synchronize()
            wall_ms=(time.perf_counter()-wall_start)*1000
            comparison=compare_native_or_replay(output,'native')
            paired=[dict(score_ms=e[0].elapsed_time(e[1]),topk_ms=e[1].elapsed_time(e[2]),
                         expand_ms=e[2].elapsed_time(e[3]),sum_ms=e[0].elapsed_time(e[3])) for e in events]
            result['records'].append(dict(length=length,repetition=repetition,mode='native',phase='profile',
                chunk_rows=chunk,logits_max_bytes=padded*keys*4,expanded_bytes=reference.numel()*4,
                cuda_ms=start.elapsed_time(end),wall_ms=wall_ms,**comparison,
                paired_chunks=paired,warmup_seconds=warm_seconds))
            del q,k,reference,host,destination,output,sorted_reference
            del captured_blocks,host_blocks,device_blocks,host_u16,device_u16
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)


if __name__ == '__main__':
    main()
