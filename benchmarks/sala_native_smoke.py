"""Run native SALA capture/replay; outputs are a small subset, not leaderboard results.

Workers must import integrations.sala_native_adapter.install via sitecustomize.
All runtime paths are command-line inputs; no credentials or machine addresses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.sala_validation import log_offset, logprobs_equal, selector_evidence, validate_fraction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--adapter-log", required=True, help="The log passed to the worker adapter install()")
    p.add_argument("--output", required=True)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--backend", default="minicpm_flashattn")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument('--mem-fraction-static', type=float, default=0.78)
    p.add_argument('--max-total-tokens', type=int, default=32768)
    p.add_argument('--fractions', default='0,0.125,0.25,0.5,1')
    args = p.parse_args()
    if not 0 < args.mem_fraction_static < 1:
        p.error('--mem-fraction-static must be finite and strictly between 0 and 1')
    if args.tokens < 1 or args.max_new_tokens < 1 or not 1 <= args.repeats <= 30:
        p.error('Require positive token counts and 1..30 native/replay repeats')
    if args.max_total_tokens < args.tokens + args.max_new_tokens:
        p.error('--max-total-tokens must cover input and output tokens')
    if Path(args.output).exists():
        p.error('--output must be a new file')
    try:
        fractions = [validate_fraction(float(h)) for h in args.fractions.split(',')]
    except ValueError as exc:
        p.error(str(exc))
    import sglang as sgl
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = tokenizer.encode("A quiet library preserves books and facts. ", add_special_tokens=False)
    ids = (seed * (args.tokens // len(seed) + 1))[:args.tokens]
    identity = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    def control(mode, fraction=0):
        run_id = uuid.uuid4().hex
        Path(args.control).write_text(json.dumps(dict(mode=mode, identity=identity,
            run_id=run_id, head_fraction=fraction, verify_queries=args.verify,
            verify_indices=args.verify)), encoding="utf-8")
        return run_id
    control("baseline")
    engine = sgl.Engine(model_path=args.model, trust_remote_code=True,
        attention_backend=args.backend, disable_radix_cache=True,
        disable_cuda_graph=True, mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=1, context_length=max(32768, args.tokens+256),
        chunked_prefill_size=max(16384,args.tokens), log_level="warning")
    results = []
    try:
        # Exclude model startup and first native-kernel compilation from timing.
        engine.generate(input_ids=ids, sampling_params=dict(temperature=0,
            max_new_tokens=args.max_new_tokens))
        schedule=[('baseline',1)]*args.repeats
        captured_layers = None
        baseline_layers = None
        for fraction in fractions:
            schedule += [('capture',fraction)] + [('replay',fraction)]*args.repeats
        for mode, fraction in schedule:
            run_id = control(mode, fraction)
            engine.flush_cache()
            offset = log_offset(args.adapter_log)
            start=time.perf_counter()
            out=engine.generate(input_ids=ids, sampling_params=dict(temperature=0,
                max_new_tokens=args.max_new_tokens), return_logprob=True, top_logprobs_num=5)
            result=dict(mode=mode,head_fraction=fraction,wall_seconds=time.perf_counter()-start,
                        output=out, timing_kind=("offline first-token generation walltime, not server TTFT; hook synchronization and CPU zlib included" if args.max_new_tokens == 1 else "whole generation, not TTFT; hook synchronization and CPU zlib included"))
            evidence = selector_evidence(args.adapter_log, offset, run_id, identity,
                mode, len(ids), fraction,
                expected_layers=captured_layers if mode == 'replay' else baseline_layers)
            result.update(evidence)
            if mode == 'baseline' and baseline_layers is None:
                baseline_layers = evidence['layers']
            elif mode == 'capture':
                captured_layers = evidence['layers']
            results.append(result)
            Path(args.output).write_text(json.dumps(results,indent=2),encoding="utf-8")
            print(mode,fraction,result['wall_seconds'],out.get('text'),flush=True)
        reference=results[0]['output']
        for r in results[1:]:
            r['text_equal']=r['output'].get('text')==reference.get('text')
            r['output_token_logprobs_equal']=logprobs_equal(r['output'], reference)
        Path(args.output).write_text(json.dumps(results,indent=2),encoding="utf-8")
        if not all(r['text_equal'] and r['output_token_logprobs_equal'] for r in results[1:]):
            raise RuntimeError("Native capture/replay output equivalence failed; inspect saved results")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
