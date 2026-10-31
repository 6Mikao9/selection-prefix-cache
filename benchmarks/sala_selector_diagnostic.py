"""TP1 offline selector-bypass diagnostic; no server TTFT or memory-saving claim.
Set worker sitecustomize to call integrations.sala_diagnostic_adapter.install
with SALA_DIAGNOSTIC_CONTROL and SALA_DIAGNOSTIC_LOG before Engine creation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.sala_validation import logprobs_equal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('model','control','adapter-log','output'): p.add_argument('--'+flag, required=True)
    p.add_argument('--tokens', type=int, default=16384)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--pinned', action='store_true')
    p.add_argument('--prefill-execution', choices=('eager','breakable'), default='eager')
    p.add_argument('--graph-events')
    p.add_argument('--mem-fraction-static', type=float, default=.5)
    a = p.parse_args()
    if a.tokens <= 8192 or a.repeats < 1 or not 0 < a.mem_fraction_static < 1: p.error('Invalid repeats or memory fraction')
    if any(Path(x).exists() for x in (a.output,a.adapter_log,a.control)):
        p.error('Output, adapter log and control must all be new paths')
    if os.environ.get('SALA_DIAGNOSTIC_CONTROL') != a.control or os.environ.get('SALA_DIAGNOSTIC_LOG') != a.adapter_log:
        p.error('Worker hook environment must match control/log arguments')
    if a.prefill_execution == 'breakable' and not a.graph_events:
        p.error('--graph-events required for breakable execution')
    if a.graph_events:
        if Path(a.graph_events).exists():p.error('--graph-events must be new')
        if os.environ.get('SALA_PREFILL_CONTROL') != a.control or os.environ.get('SALA_PREFILL_EVENTS') != a.graph_events:
            p.error('Prefill worker hook environment must match control/events')
    import sglang as sgl
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode('A quiet library preserves books and facts. ', add_special_tokens=False)
    ids = (seed*(a.tokens//len(seed)+1))[:a.tokens]
    identity = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    def control(mode, verify=False, profile=False):
        run = uuid.uuid4().hex
        Path(a.control).write_text(json.dumps(dict(mode=mode, run_id=run, identity=identity,
            verify=verify, profile=profile, pinned=a.pinned, tokens=a.tokens)))
        return run
    control('baseline')
    graph_args = ({'disable_cuda_graph':True} if a.prefill_execution == 'eager' else
        {'cuda_graph_config':{'decode':{'backend':'disabled'},'prefill':{'backend':'breakable','bs':[a.tokens],'max_bs':a.tokens}}})
    engine = sgl.Engine(model_path=a.model, trust_remote_code=True, tp_size=1,
        attention_backend='minicpm_flashattn', disable_radix_cache=True,
        **graph_args, mem_fraction_static=a.mem_fraction_static,
        max_running_requests=1, context_length=max(32768,a.tokens+256),
        max_total_tokens=max(32768,a.tokens+256), chunked_prefill_size=a.tokens, log_level='warning')
    results = []
    reference = None
    def request(mode, phase, verify=False, profile=False):
        nonlocal reference
        run = control(mode,verify,profile)
        engine.flush_cache()
        start = time.perf_counter()
        out = engine.generate(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=1),
                              return_logprob=True,top_logprobs_num=5)
        elapsed = time.perf_counter()-start
        if reference is None: reference=out
        equal = out.get('text') == reference.get('text') and logprobs_equal(out,reference)
        graph_events=[json.loads(line) for line in Path(a.graph_events).read_text().splitlines()] if a.graph_events and Path(a.graph_events).exists() else []
        graph_replays=[e for e in graph_events if e.get('run_id')==run and e.get('kind')=='actual_prefill_graph_replay']
        results.append(dict(prefill_execution=a.prefill_execution,
            shape_adapter=('restore eager 2D attention contract' if os.environ.get('SALA_PREFILL_SHAPE_FIX')=='1' else 'disabled'),
            actual_prefill_replays=len(graph_replays),run_id=run,mode=mode,phase=phase,wall_seconds=elapsed,
            output=out,output_equal=equal,profile=profile,
            timing_kind='offline first-token walltime with diagnostic hook; not server TTFT'))
        Path(a.output).write_text(json.dumps(results,indent=2))
        if a.prefill_execution=='breakable' and not graph_replays:
            raise RuntimeError('No actual prefill graph replay; refusing fallback result')
        if a.prefill_execution=='eager' and graph_replays:
            raise RuntimeError('Unexpected prefill graph replay in eager control')
        if not equal: raise RuntimeError('Output/returned-token logprobs differ')
    try:
        request('baseline','warmup')
        request('baseline','strict',verify=True)
        request('capture','capture')
        modes=['gpu']+(['pinned'] if a.pinned else [])
        for mode in modes: request(mode,'strict',verify=True)
        for mode in ['baseline']+modes: request(mode,'warmup')
        schedule=(['baseline']+modes)*a.repeats
        random.Random(42).shuffle(schedule)
        for mode in schedule: request(mode,'unprofiled')
        for mode in modes: request(mode,'strict',verify=True)
        for _ in range(max(3,a.repeats)):
            for mode in ['baseline']+modes: request(mode,'profile',profile=True)
        # A final request drains the previous request's events in subprocess workers.
        request('baseline','drain')
        events=[json.loads(line) for line in Path(a.adapter_log).read_text().splitlines()]
        expected=None
        for row in results[:-1]:
            layer_events=[e for e in events if e['run_id']==row['run_id']]
            layers=[e['layer'] for e in layer_events]
            if not layers or len(layers)!=len(set(layers)): raise RuntimeError('Hook missing or duplicate layers')
            if expected is None: expected=set(layers)
            if set(layers)!=expected: raise RuntimeError('Incomplete layer coverage')
            if any(e['query_tokens'] != a.tokens or e['native_selector_called'] != (row['mode'] in ('baseline','capture')) for e in layer_events):
                raise RuntimeError('Wrong selector query count or native path')
            if row['profile']:
                import math
                if any(not isinstance(e.get('selector_or_h2d_cuda_ms'), (int,float)) or not math.isfinite(e['selector_or_h2d_cuda_ms']) or e['selector_or_h2d_cuda_ms'] < 0 for e in layer_events):
                    raise RuntimeError('Missing or invalid CUDA event timing')
            if row['phase']=='strict' and row['mode']!='baseline' and not all(e['verified'] for e in layer_events):
                raise RuntimeError('Missing strict query/selection evidence')
        if a.graph_events:
            graph_events=[json.loads(line) for line in Path(a.graph_events).read_text().splitlines()]
            for row in results:
                if row['phase']=='profile':
                    intervals=[e for e in graph_events if e.get('run_id')==row['run_id'] and e.get('kind')=='prefill_forward_cuda_interval']
                    if len(intervals)!=1:raise RuntimeError('Missing or duplicate model-forward CUDA interval')
                    row['prefill_cuda_interval']=intervals[0]
            Path(a.output).write_text(json.dumps(results,indent=2))
        print('Completed. Compare only unprofiled rows; GPU cache is an optimistic memory-consuming reference.')
    finally:
        engine.shutdown()


if __name__ == '__main__': main()
