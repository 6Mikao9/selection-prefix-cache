"""Dense pretrained vLLM quality control on a pinned LongBench tiny subset.

No sparse selection or replay is introduced. Prompts and token IDs stay outside
the repository. Results contain IDs, predictions, references and metrics only.
Use an explicitly selected idle GPU; no dependencies are installed by this file.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-label", default="Qwen/Qwen3-4B")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--memutil", type=float, default=0.60)
    p.add_argument("--prepared-jsonl", type=Path,
                   help="Optional externally prepared RULER records")
    p.add_argument("--chat-template", action="store_true",
                   help="Wrap the official task text in this tokenizer's native user chat")
    p.add_argument("--disable-thinking", action="store_true",
                   help="Pass enable_thinking=False to the native chat template")
    a = p.parse_args()
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicit idle GPU required"
    assert 0 < a.memutil <= .6
    assert not a.disable_thinking or a.chat_template
    if a.output.exists():
        raise FileExistsError("Use a new output file")
    from transformers import AutoTokenizer
    from benchmark_data import (prepare, score, DATA_SHA256, CONFIG_SHA256)
    from sala_validation import public_provenance
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True,
                                              trust_remote_code=False)
    if a.prepared_jsonl:
        records = [json.loads(line) for line in a.prepared_jsonl.read_text().splitlines()]
        for r in records:
            # ruler_data.py has already rendered official input+answer_prefix
            # with this model's tokenizer; do not wrap or retokenize it.
            assert r["actual_tokens"] == len(r["input_ids"])
            assert r["input_ids"] == tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            r["prompt_sha256"] = hashlib.sha256(r["prompt"].encode()).hexdigest()
    else:
        archive, config = a.cache_dir / "data.zip", a.cache_dir / "dataset2prompt.json"
        assert digest(archive) == DATA_SHA256
        assert digest(config) == CONFIG_SHA256
        templates = json.loads(config.read_text(encoding="utf-8"))
        records = list(prepare(tokenizer, archive, templates, a.limit, a.max_tokens))
    for r in records:
        if a.chat_template:
            r["body_actual_tokens"] = r["actual_tokens"]
            options = {"enable_thinking": False} if a.disable_thinking else {}
            messages = [{"role": "user", "content": r["prompt"]}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True, **options)
            encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                add_generation_prompt=True, return_dict=False, **options)
            r["input_ids"] = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            assert all(isinstance(token, int) for token in r["input_ids"])
            r["actual_tokens"] = len(r["input_ids"])
            r["prompt_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
            if "provenance" in r:
                r["provenance"]["chat_wrapper"] = "native tokenizer user chat; generation prompt"
    assert records
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    import torch
    max_len = max(len(r["input_ids"])+r["max_new_tokens"] for r in records)
    llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=True,
              enable_prefix_caching=False, max_model_len=max_len,
              max_num_batched_tokens=4096, max_num_seqs=1,
              gpu_memory_utilization=a.memutil, dtype="bfloat16",
              disable_log_stats=True)
    summaries = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as target:
        for r in records:
            sampling = SamplingParams(temperature=0, max_tokens=r["max_new_tokens"], seed=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = llm.generate([dict(prompt_token_ids=r["input_ids"])], sampling,
                                  use_tqdm=False)[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            prediction = result.outputs[0].text
            metric = (score(r["task"], prediction, r["answers"])
                      if not a.prepared_jsonl else
                      sum(answer.lower() in prediction.lower() for answer in r["answers"])
                      / len(r["answers"]))
            row = {key: r[key] for key in ("id", "task", "raw_tokens", "actual_tokens",
                   "truncated", "max_new_tokens", "prompt_sha256")}
            for key in ("sample_id", "source_row", "provenance", "body_actual_tokens"):
                if key in r:
                    row[key] = public_provenance(r[key]) if key == 'provenance' else r[key]
            row.update(model=a.model_label, framework="vllm", mode="dense_full_attention",
                       prediction=prediction, answers=r["answers"], score=metric,
                       generated_tokens=len(result.outputs[0].token_ids),
                       finish_reason=result.outputs[0].finish_reason,
                       generation_seconds=seconds,
                       timing_kind="whole generation incl prefill; NOT TTFT; no warmup",
                       prefix_caching=False,
                       chat_wrapper="native user chat" if a.chat_template else "none",
                       enable_thinking=False if a.disable_thinking else "template default",
                       temperature=0,
                       flashinfer_sampler=os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "default"),
                       enable_chunked_prefill=True, max_num_batched_tokens=4096,
                       gpu_memory_utilization=a.memutil,
                       seed=0, tensor_parallel_size=1, eager=True,
                       input_token_cap=a.max_tokens,
                       versions={name: importlib.metadata.version(name)
                                 for name in ("torch", "vllm", "transformers")})
            target.write(json.dumps(row, ensure_ascii=False)+"\n")
            target.flush()
            summaries.append(row)
            print(json.dumps({key: row[key] for key in
                  ("id", "task", "actual_tokens", "score", "generation_seconds")}), flush=True)
    for task in sorted({r["task"] for r in summaries}):
        rows = [r for r in summaries if r["task"] == task]
        print(json.dumps(dict(task=task, n=len(rows),
                             mean_score=sum(r["score"] for r in rows)/len(rows))), flush=True)


if __name__ == "__main__":
    main()
