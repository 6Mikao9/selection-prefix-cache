"""Run the UNMODIFIED pinned NVIDIA RULER NIAH generator for tiny smoke tests.

Source: https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
scripts/{synthetic.yaml,data/synthetic/niah.py,data/synthetic/constants.py,
data/tokenizer.py,data/manifest_utils.py,eval/synthetic/constants.py}.
NVIDIA copyright 2024-2025, Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
Metric below follows official string_match_all, expressed per-example on 0..1.

Supply the pinned source tree via --ruler-root, and optional task-local dependency
directory via --dependencies. No automatic pip changes, no NLTK data download:
these two official tasks use noise/needle haystacks, not essay sentence splitting.
Install wonderwords==2.2.0, nltk==3.9.2, joblib==1.5.2, tenacity==9.1.2 into a
separate directory with pip --target --no-deps using predownloaded wheels.
Existing transformers, numpy, regex, tqdm and click are also required.
The base model template is {task_template}, as in official data/template.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

try:
    from integrations.benchmark_data import file_sha256
except ModuleNotFoundError:
    from benchmark_data import file_sha256

REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
SOURCE_HASHES = {
    "scripts/synthetic.yaml": "34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b",
    "scripts/data/synthetic/niah.py": "e9cada0a7660d274fe73a1338a90a7087e17b630169f1aaf14a8d3221c6805b5",
    "scripts/data/synthetic/constants.py": "6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde",
    "scripts/data/tokenizer.py": "c2e4bfab607eef87a86334558303c1811bc8e93f22a5c4b129f302726d2357a4",
    "scripts/data/manifest_utils.py": "ecac79322f28ce9a12388b12e35d560ecb7cdad6b9888467a5a13b0eff2db91e",
}
TASK_HAYSTACK = {"niah_single_1": "noise", "niah_multikey_2": "needle"}


def score(task, prediction, answers):
    if task not in TASK_HAYSTACK:
        raise ValueError("Unsupported RULER task")
    if not answers:
        raise ValueError("RULER references must not be empty")
    return sum(answer.lower() in prediction.lower() for answer in answers) / len(answers)


def validate_sources(root):
    for relative, digest in SOURCE_HASHES.items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f"Official source hash mismatch: {relative}")


def command(root, tokenizer, save_dir, task, max_tokens, limit, template):
    return [sys.executable, str(root / "scripts/data/synthetic/niah.py"),
            "--save_dir", str(save_dir), "--save_name", task,
            "--tokenizer_path", str(tokenizer), "--tokenizer_type", "hf",
            "--max_seq_length", str(max_tokens), "--tokens_to_generate", "128",
            "--num_samples", str(limit), "--random_seed", "42",
            "--type_haystack", TASK_HAYSTACK[task],
            "--type_needle_k", "words", "--type_needle_v", "numbers",
            "--num_needle_k", "1", "--num_needle_v", "1", "--num_needle_q", "1",
            "--template", template]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, choices=(8192, 16384), default=16384)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for path in (args.cache_dir.resolve(), args.output.resolve()):
        if path == repo or repo in path.parents:
            parser.error("Benchmark contexts and caches must remain outside repository")
    if args.output.exists() or not 1 <= args.limit <= 64:
        parser.error("Require new --output and --limit in 1..64")
    root = args.ruler_root.resolve()
    validate_sources(root)
    config = runpy.run_path(str(root / "scripts/data/synthetic/constants.py"))["TASKS"]["niah"]
    template = config["template"] + config["answer_prefix"]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    if args.dependencies:
        environment["PYTHONPATH"] = str(args.dependencies.resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    generated = Path(tempfile.mkdtemp(prefix=f"ruler-{args.max_tokens}-", dir=args.cache_dir))
    for task in TASK_HAYSTACK:
        cmd = command(root, args.tokenizer.resolve(), generated, task, args.max_tokens, args.limit, template)
        # No shell and no changes to upstream generator; its tokenizer and random
        # generation logic, sequence budget, answer-prefix split remain official.
        subprocess.run(cmd, env=environment, check=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for task in TASK_HAYSTACK:
            source = generated / task / "validation.jsonl"
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            if len(rows) != args.limit:
                raise ValueError("Official generator returned unexpected sample count")
            for row, sample in enumerate(rows):
                # Official inference concatenates input + answer_prefix.
                prompt = sample["input"] + sample["answer_prefix"]
                ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
                record = {
                    "id": f"ruler/{task}/{args.max_tokens}/seed42/{row}",
                    "sample_id": str(row), "source_row": row, "task": task,
                    "prompt": prompt, "input_ids": ids, "answers": sample["outputs"],
                    "raw_tokens": len(ids), "actual_tokens": len(ids), "truncated": False,
                    "input_token_cap": args.max_tokens - 128, "max_new_tokens": 128,
                    "official_length_with_generation": sample["length"],
                    "token_position_answer": sample["token_position_answer"],
                    "provenance": {
                        "benchmark": "RULER tiny smoke subset; unmodified official generator",
                        "code_revision": REVISION, "source_sha256": SOURCE_HASHES,
                        "source_url": f"https://github.com/NVIDIA/RULER/tree/{REVISION}",
                        "code_license": "Apache-2.0; Copyright NVIDIA 2024-2025",
                        "seed": 42, "model_template": "base", "metric": "string_match_all",
                        "dependency_versions": {"wonderwords": "2.2.0", "nltk": "3.9.2",
                                                "joblib": "1.5.2", "tenacity": "9.1.2"},
                        "tokenization": "HF tokenize, add_special_tokens=False, input+answer_prefix",
                    },
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps({k: record[k] for k in ("id", "actual_tokens", "truncated")}))


if __name__ == "__main__":
    main()
