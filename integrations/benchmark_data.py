"""Pinned official LongBench-v1 tiny smoke subset, not leaderboard evaluation.

Downloads remain in the REQUIRED external --cache-dir; never commit its contents
or the generated JSONL (which contains benchmark contexts). Only exact allowlisted
ZIP members are streamed, never extracted. TLS certificate checking is enabled.

Sources verified 2026-10-31:
  THUDM/LongBench commit 2e00731f8d0bff23dc4325161044d0ed8af94c1e,
  LongBench/{pred.py,eval.py,metrics.py,config/dataset2prompt.json}.
  https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench
  Data: https://huggingface.co/datasets/THUDM/LongBench at DATA_REVISION.
  Upstream code MIT, copyright (c) 2023 THU-KEG & Zhipu AI. Data card
  does not declare a dataset license; underlying source-data rights still apply.

The small metric adaptations below retain the upstream behavior (including zero
F1 for two empty normalized strings), maximum over references, score range 0..1.
Official pred.py truncates default-tokenized prompts in the middle, decodes each
half with skip_special_tokens=True, then RETOKENIZES. We preserve that operation,
and report the actual resulting count, which need not equal the requested cap.
No model-specific chat wrapper is added: MiniCPM is not in the original supported
model list. Send saved input_ids directly to inference to preserve this protocol.

MIT permission notice for adapted metric logic:
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import string
import urllib.request
import zipfile

CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
TASKS = ("passage_retrieval_en", "2wikimqa")
DATA_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
CONFIG_SHA256 = "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"
DATA_URL = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/{DATA_REVISION}/data.zip"
PROMPT_URL = f"https://raw.githubusercontent.com/THUDM/LongBench/{CODE_REVISION}/LongBench/config/dataset2prompt.json"
MAX_DOWNLOAD_BYTES = 160 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_provenance(tokenizer):
    """Identify tokenizer content without publishing its local directory."""
    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=True,
                            separators=(",", ":")).encode("utf-8")
    return {"tokenizer": "local-tokenizer",
            "tokenizer_vocab_sha256": hashlib.sha256(vocabulary).hexdigest()}


def cached_download(url, path, expected_sha256):
    """Bounded HTTPS download, verified by pinned SHA256; no insecure fallback."""
    path = Path(path)
    if path.exists():
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Cache SHA256 mismatch: {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    owns_partial = False
    try:
        # Exclusive creation also prevents colliding concurrent download writers.
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
            owns_partial = True
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download exceeds byte limit")
                output.write(chunk)
        if file_sha256(partial) != expected_sha256:
            raise ValueError(f"Downloaded SHA256 mismatch: {path.name}")
        partial.rename(path)
    except BaseException:
        if owns_partial and partial.exists():
            partial.unlink()
        raise
    return path


def read_samples(archive, task, limit):
    if task not in TASKS or not 1 <= limit <= 64:
        raise ValueError("Unsupported task or limit (allowed: 1..64)")
    member = f"data/{task}.jsonl"
    with zipfile.ZipFile(archive) as source:
        matches = [entry for entry in source.infolist() if entry.filename == member]
        if len(matches) != 1 or matches[0].file_size > MAX_MEMBER_BYTES:
            raise ValueError("Missing, duplicate or oversized allowlisted member")
        # Not extract()/extractall(): untrusted filenames cannot address disk paths.
        with source.open(matches[0]) as stream, io.TextIOWrapper(stream, encoding="utf-8") as text:
            for index, line in enumerate(text):
                if index >= limit:
                    break
                yield index, json.loads(line)


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_f1_score(prediction, ground_truth, **kwargs):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction, ground_truth, **kwargs):
    expected = re.findall(r"Paragraph (\d+)", ground_truth)[0]
    numbers = re.findall(r"\d+", prediction)
    return sum(number == expected for number in numbers) / len(numbers) if numbers else 0.0


def score(task, prediction, answers):
    metric = {"passage_retrieval_en": retrieval_score, "2wikimqa": qa_f1_score}[task]
    return max((metric(prediction, answer) for answer in answers), default=0.0)


def tokenize_prompt(tokenizer, prompt, max_tokens):
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    raw_ids = tokenizer(prompt, truncation=False)["input_ids"]
    truncated = len(raw_ids) > max_tokens
    if truncated:
        half = max_tokens // 2
        prompt = (tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                  + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True))
    ids = tokenizer(prompt, truncation=False)["input_ids"]
    return prompt, ids, len(raw_ids), truncated


def prepare(tokenizer, archive, templates, limit=8, max_tokens=16384):
    for task in TASKS:
        for row, sample in read_samples(archive, task, limit):
            prompt, ids, raw_tokens, truncated = tokenize_prompt(
                tokenizer, templates[task].format(**sample), max_tokens)
            yield {
                "id": f"longbench-v1/{task}/{sample['_id']}",
                "sample_id": sample["_id"], "task": task, "source_row": row,
                "prompt": prompt, "input_ids": ids, "answers": sample["answers"],
                "raw_tokens": raw_tokens, "actual_tokens": len(ids),
                "truncated": truncated, "input_token_cap": max_tokens,
                "source_length": sample["length"], "max_new_tokens": 32,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "provenance": {
                    "benchmark": "LongBench-v1 tiny smoke subset; first N source rows",
                    "code_revision": CODE_REVISION, "data_revision": DATA_REVISION,
                    "data_sha256": DATA_SHA256, "prompt_config_sha256": CONFIG_SHA256,
                    "data_url": DATA_URL, "prompt_url": PROMPT_URL,
                    "code_license": "MIT; Copyright 2023 THU-KEG & Zhipu AI",
                    "data_license": "Not declared in HF card; retain underlying source rights",
                    "tokenization": "default add_special_tokens; official middle truncation and retokenization",
                    "chat_wrapper": "none; MiniCPM not supported by original build_chat",
                    "metric": "retrieval_score" if task == "passage_retrieval_en" else "qa_f1_score",
                },
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL, outside repository")
    parser.add_argument("--cache-dir", type=Path, required=True, help="External download cache")
    parser.add_argument("--limit", type=int, default=8, help="First N rows per task (1..64)")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    for path in (args.cache_dir.resolve(), args.output.resolve()):
        if path == repository or repository in path.parents:
            parser.error("Cache and generated benchmark contexts must remain outside repository")
    if args.output.exists():
        parser.error("--output must be a new file")
    if not 1 <= args.limit <= 64 or args.max_tokens < 2:
        parser.error("--limit must be 1..64 and --max-tokens must be >= 2")
    archive = cached_download(DATA_URL, args.cache_dir / "data.zip", DATA_SHA256)
    config = cached_download(PROMPT_URL, args.cache_dir / "dataset2prompt.json", CONFIG_SHA256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    tokenizer_info = tokenizer_provenance(tokenizer)
    templates = json.loads(config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in prepare(tokenizer, archive, templates, args.limit, args.max_tokens):
            record["provenance"].update(tokenizer_info)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({key: record[key] for key in
                  ("id", "task", "raw_tokens", "actual_tokens", "truncated")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
