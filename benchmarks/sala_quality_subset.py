"""SALA small-subset quality with official pretokenized benchmark records.

No benchmark source documents are embedded in this script or its output.
Input JSONL is prepared by integrations/benchmark_data.py or ruler_data.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.sala_validation import (
    log_offset, logprobs_equal, public_provenance, selector_evidence, validate_fraction,
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--control',required=True)
    p.add_argument('--adapter-log',help='The log passed to the worker adapter install()')
    p.add_argument('--output',required=True)
    p.add_argument('--backend',default='minicpm_flashattn')
    p.add_argument('--limit-per-task',type=int,default=4)
    p.add_argument('--fractions',default='0,0.25,0.5,1')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--chat-template',action='store_true')
    args=p.parse_args()
    if not 1 <= args.limit_per_task <= 64:
        p.error('--limit-per-task must be 1..64')
    if Path(args.output).exists():
        p.error('--output must be a new file')
    if not args.baseline_only and not args.adapter_log:
        p.error('--adapter-log is required for capture/replay evidence')
    fractions = [validate_fraction(float(h)) for h in args.fractions.split(',')]
    import sglang as sgl
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    from integrations.benchmark_data import score as longbench_score
    from integrations.ruler_data import score as ruler_score
    seen={}; samples=[]
    for line in Path(args.dataset).read_text(encoding='utf-8').splitlines():
        s=json.loads(line); task=s['task']; seen[task]=seen.get(task,0)+1
        if seen[task]<=args.limit_per_task:samples.append(s)
    if not samples:
        raise ValueError('No evaluation samples were selected')
    Path(args.control).write_text(json.dumps(dict(mode='baseline')),encoding='utf-8')
    engine=sgl.Engine(model_path=args.model,trust_remote_code=True,
        attention_backend=args.backend,disable_radix_cache=True,disable_cuda_graph=True,
        mem_fraction_static=0.78,max_running_requests=1,context_length=32768,
        chunked_prefill_size=32768,log_level='warning')
    try:
        with open(args.output,'w',encoding='utf-8') as f:
            for s in samples:
                ids=s['input_ids']
                expected_ids = tokenizer(s['prompt'], truncation=False,
                    add_special_tokens=not s['task'].startswith('niah_'))['input_ids']
                if ids != expected_ids:
                    raise ValueError('Prepared token IDs do not match the target model tokenizer')
                if args.chat_template:
                    ids=tokenizer.apply_chat_template([dict(role='user',content=s['prompt'])],
                        tokenize=True,add_generation_prompt=True,enable_thinking=False,
                        return_dict=False)
                identity=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                modes=[('baseline',1)]
                if not args.baseline_only:
                    for h in fractions:
                        modes += [('capture',h),('replay',h)]
                reference=None
                captured_layers=None
                baseline_layers=None
                for mode,h in modes:
                    run_id=uuid.uuid4().hex
                    Path(args.control).write_text(json.dumps(dict(mode=mode,identity=identity,
                        run_id=run_id,head_fraction=h,verify_queries=args.verify,verify_indices=args.verify)),encoding='utf-8')
                    engine.flush_cache()
                    offset=log_offset(args.adapter_log) if args.adapter_log else 0
                    start=time.perf_counter()
                    out=engine.generate(input_ids=ids,sampling_params=dict(
                        temperature=0,max_new_tokens=s['max_new_tokens']),return_logprob=True)
                    elapsed=time.perf_counter()-start
                    pred=out['text']; task=s['task']; answers=s['answers']
                    score=(ruler_score(task,pred,answers) if task.startswith('niah_')
                           else longbench_score(task,pred,answers))
                    if reference is None:reference=out
                    result={k:s[k] for k in ['id','task','source_row','raw_tokens','actual_tokens',
                        'truncated','max_new_tokens','provenance'] if k in s}
                    if 'provenance' in result:
                        result['provenance']=public_provenance(result['provenance'])
                    result.update(mode=mode,head_fraction=h,prediction=pred,answers=answers,
                        protocol='native_chat' if args.chat_template else 'official_raw',
                        body_actual_tokens=s['actual_tokens'],actual_tokens=len(ids),
                        score=score,prompt_sha256=identity,generation_seconds=elapsed,
                        timing_kind='whole generation, not TTFT; no timing comparison; hook CPU work included',
                        generated_tokens=out.get('meta_info',{}).get('completion_tokens'),
                        text_equal_baseline=out['text']==reference['text'],
                        output_token_logprobs_equal_baseline=logprobs_equal(out, reference),
                        output_token_logprobs=out.get('meta_info',{}).get('output_token_logprobs'))
                    if not args.baseline_only:
                        evidence=selector_evidence(args.adapter_log,offset,run_id,identity,
                            mode,len(ids),h,expected_layers=captured_layers if mode=='replay' else baseline_layers)
                        result.update(evidence)
                        if mode=='baseline':baseline_layers=evidence['layers']
                        elif mode=='capture':captured_layers=evidence['layers']
                    f.write(json.dumps(result)+'\n');f.flush()
                    print(s['id'],mode,h,score,elapsed,flush=True)
    finally:engine.shutdown()


if __name__=='__main__':main()
