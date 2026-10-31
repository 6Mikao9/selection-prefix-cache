"""Native-only prefill graph compatibility probe; use separate engines per mode.
Worker sitecustomize must install integrations.sala_prefill_probe_adapter with
SALA_PREFILL_CONTROL and SALA_PREFILL_EVENTS. No cache-benefit claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.sala_validation import logprobs_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution',choices=('eager','breakable'),required=True)
    for name in ('model','output','events','control'):p.add_argument('--'+name,required=True)
    p.add_argument('--tokens',type=int,default=16384)
    p.add_argument('--repeats',type=int,default=7)
    a=p.parse_args()
    if a.tokens<=8192 or a.repeats<1:p.error('Require tokens >8192 and positive repeats')
    if any(Path(x).exists() for x in (a.output,a.events,a.control)):p.error('All artifacts must be new files')
    if os.environ.get('SALA_PREFILL_CONTROL')!=a.control or os.environ.get('SALA_PREFILL_EVENTS')!=a.events:
        p.error('Worker hook environment must match arguments')
    import sglang as sgl
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    seed=tokenizer.encode('A quiet library preserves books and facts. ',add_special_tokens=False)
    ids=(seed*(a.tokens//len(seed)+1))[:a.tokens]
    config={'decode':{'backend':'disabled'},'prefill':{'backend':'disabled' if a.execution=='eager' else 'breakable','bs':[a.tokens],'max_bs':a.tokens}}
    Path(a.control).write_text(json.dumps({'run_id':'initialization'}))
    report=dict(shape_adapter=('restore eager 2D attention contract' if os.environ.get('SALA_PREFILL_SHAPE_FIX') == '1' else 'disabled'),execution=a.execution,branch='native',status='running',requested_graph_config=config,
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),rows=[],
        limitation='Separate profile records current-stream model forward GPU timeline including launch gaps, not pure kernel sum; walltime is offline first-token request, not server TTFT.')
    def save():Path(a.output).write_text(json.dumps(report,indent=2))
    save()
    engine=None
    try:
        engine=sgl.Engine(model_path=a.model,trust_remote_code=True,tp_size=1,
            attention_backend='minicpm_flashattn',disable_radix_cache=True,
            cuda_graph_config=config,mem_fraction_static=.5,max_running_requests=1,
            context_length=max(32768,a.tokens+256),max_total_tokens=max(32768,a.tokens+256),
            chunked_prefill_size=a.tokens,log_level='info')
        reference=None
        for phase,count in [('warmup',7),('measurement',a.repeats),('profile',max(3,a.repeats)),('drain',1)]:
            for _ in range(count):
                run=uuid.uuid4().hex
                Path(a.control).write_text(json.dumps({'run_id':run,'tokens':a.tokens,'profile':phase=='profile'}))
                engine.flush_cache()
                start=time.perf_counter()
                out=engine.generate(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=1),return_logprob=True,top_logprobs_num=5)
                elapsed=time.perf_counter()-start
                if reference is None:reference=out
                equal=out.get('text')==reference.get('text') and logprobs_equal(out,reference)
                events=[json.loads(line) for line in Path(a.events).read_text().splitlines()] if Path(a.events).exists() else []
                actual=[e for e in events if e['run_id']==run and e.get('kind')=='actual_prefill_graph_replay']
                row=dict(run_id=run,phase=phase,output=out,output_equal=equal,
                    offline_wall_seconds=elapsed,actual_prefill_replays=len(actual),replay_evidence=actual)
                report['rows'].append(row);save()
                if not equal:raise RuntimeError('Native repeated output/logprob instability')
                if a.execution=='breakable' and (not actual or any(e.get('backend')!='breakable' for e in actual)):
                    report['status']='unsupported_or_eager_fallback';save()
                    raise RuntimeError('No proven breakable prefill replay; do not count as graph success')
                if a.execution=='eager' and actual:raise RuntimeError('Unexpected graph replay in eager control')
        final_events=[json.loads(line) for line in Path(a.events).read_text().splitlines()] if Path(a.events).exists() else []
        for row in report['rows']:
            if row['phase']=='profile':
                intervals=[e for e in final_events if e['run_id']==row['run_id'] and e.get('kind')=='prefill_forward_cuda_interval']
                if len(intervals)!=1:raise RuntimeError('Missing or duplicate prefill CUDA interval')
                row['prefill_cuda_interval']=intervals[0]
        report['status']='native_prefill_graph_verified' if a.execution=='breakable' else 'eager_control_completed';save()
    except Exception as exc:
        if report['status']=='running':report['status']='failed_or_unsupported'
        report['error_type']=type(exc).__name__;save()
        raise
    finally:
        if engine is not None:engine.shutdown()


if __name__=='__main__':main()
