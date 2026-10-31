# Framework adapters and pretrained-model controls

Completed model runs use pretrained **Qwen3-4B with dense vLLM attention**. They
establish task/framework controls, not selection replay, cache savings or TTFT
speedup. Native sparse-model replay remains unverified. See the
[experiment report](../benchmarks/framework_results.md) and
[machine-readable summary](results/qwen_quality_summary.json).

## Completed Qwen controls

Each cell is a mean score on **four examples per task**, on a 0–1 scale.
These are tiny smoke subsets, not leaderboard evaluations.

| Input protocol | LongBench retrieval | LongBench 2wiki F1 | RULER single | RULER multikey |
|---|---:|---:|---:|---:|
| Official task text, no chat wrapper | 0.8333 | 0.0250 | 1.0000 | 1.0000 |
| Native Qwen chat, thinking disabled | 0.7500 | 0.2500 | incomplete | incomplete |

Raw and native-chat runs are separate input protocols. All raw outputs reached
the generation cap and often continued after their answers. Native-chat
LongBench answers used 2–8 tokens. RULER's official substring metric does not
penalize extra text. RULER native-chat stopped after five examples because of a
CUDA device failure; no complete score is reported for it.

- [LongBench raw, maxgen 32](results/qwen3_longbench4_raw32.jsonl)
- [LongBench native-chat nonthinking, maxgen 32](results/qwen3_longbench4_chat_nothink32.jsonl)
- [RULER raw, 16K sequence budget, maxgen 128](results/qwen3_ruler16k4_raw128.jsonl)

Results retain sample IDs, input counts, prompt hashes, predictions, references,
metrics and versions. Long benchmark contexts remain outside the repository.
`generation_seconds` covers whole generation including prefill, **not TTFT**;
cold/JIT and output-length effects prevent interpreting it as replay speedup.

## Reproduce dense controls

The recorded stack was vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.17.0,
BF16, TP=1, eager execution, prefix caching disabled, and FLASH_ATTN attention.
The installed FlashInfer sampler failed architecture detection; the documented
environment switch below selected the existing native sampler.

Run from the repository root using an existing compatible environment. `MODEL`
is a local checkpoint; `TASK` must be an absolute directory **outside this
repository**. Check utilization, memory and compute processes immediately before
each GPU launch, and bind only an idle GPU. GPU 0 is an example allocation.

```sh
export MODEL=/path/to/Qwen3-4B
export TASK=/path/to/external-experiment
mkdir -p "$TASK/cache" "$TASK/results"

# Download pinned official archive/config and verify SHA256 hashes.
python integrations/benchmark_data.py --tokenizer "$MODEL" \
  --cache-dir "$TASK/cache" --output "$TASK/longbench-qwen.jsonl" \
  --limit 4 --max-tokens 16384

nvidia-smi
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
python integrations/vllm_quality_baseline.py --model "$MODEL" \
  --cache-dir "$TASK/cache" --output "$TASK/results/longbench-raw.jsonl" \
  --limit 4 --max-tokens 16384

nvidia-smi
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
python integrations/vllm_quality_baseline.py --model "$MODEL" \
  --cache-dir "$TASK/cache" --output "$TASK/results/longbench-chat.jsonl" \
  --limit 4 --max-tokens 16384 --chat-template --disable-thinking
```

The dense runner prepares LongBench again in memory from the verified cache and
feeds exact `input_ids` to inference. The saved preparation JSONL makes the
protocol inspectable. Output filenames must be new.

For RULER, use the source at
[`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a).
Preparation validates source hashes and invokes the unmodified official
generator. Missing small CPU dependencies can be installed from pinned wheels
into a task-local target, as documented in [ruler_data.py](ruler_data.py), without
replacing framework packages.

```sh
export RULER_ROOT=/path/to/pinned-RULER-checkout
python integrations/ruler_data.py --tokenizer "$MODEL" \
  --ruler-root "$RULER_ROOT" --cache-dir "$TASK/ruler-generated" \
  --dependencies "$TASK/ruler-deps" --output "$TASK/ruler-qwen.jsonl" \
  --limit 4 --max-tokens 16384

nvidia-smi
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
python integrations/vllm_quality_baseline.py --model "$MODEL" \
  --cache-dir "$TASK/cache" --prepared-jsonl "$TASK/ruler-qwen.jsonl" \
  --output "$TASK/results/ruler-raw.jsonl"
```

## vLLM native DSA adapter: blocked before inference

[vllm_dsa_smoke.py](vllm_dsa_smoke.py) adds process-local eager hooks for vLLM
0.30's GLM/DeepSeek DSA prefill selector. It targets one TP=1 request, unchunked
prefill and one generated token. It preserves indexer K insertion and native
logical-to-physical block-table mapping. It recomputes head query rows and stores
only ordered int32 tail selections, including `-1` padding, with CPU zlib. This
is an adapter baseline, not the repository's sorted-uint16 CUDA transfer pipeline.

The real 0.62B GLM DSA architecture checkpoint was tried on an idle RTX 5090
(SM120), memory utilization 0.45. Model construction failed:

```text
No valid attention backend found ... use_mla=True ... use_sparse=True
TRITON_MLA: sparse not supported
FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's sparse MLA decode API
```

The installed FlashMLA sparse backend supports SM90/SM100, not SM120. No installed
packages were changed. **Native capture/replay has not passed.** The CPU-only
[hook contract test](vllm_dsa_hook_selftest.py) passed for head lengths 0/4/10/20
against ten query rows, checking ordered indices, skipped logits rows and `-1`
padding with mocked kernels. It is not a native GPU test. The checkpoint's
publisher describes random initialization plus toy-text finetuning, so it is an
architecture-smoke candidate rather than a representative quality model.

```sh
# Requires a supported native sparse-MLA environment; currently blocked.
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python integrations/vllm_dsa_smoke.py \
  --model /path/to/GLM-DSA-checkpoint --lengths 2048 4096 8192 --head 2048 \
  --output "$TASK/results/dsa-smoke.json"
```

The script would check head indices, generated token and returned top-10
logprobs, not full-vocabulary logits. CPU copies and synchronization make its
timings diagnostic rather than performance evidence.

## MiniCPM-SALA: operator probe only; model evaluation pending

SGLang 0.5.20 contains the MiniCPM-SALA model/backend. A **random operator-only**
16K nonfused selector probe found identical head selections for
H=8192/12288/16384 when both compressed-key levels were cropped using native
compression formulas. It did not load pretrained model weights.

The real MiniCPM-SALA model has **never loaded in these experiments**. Container
CUDA/device failures blocked further work on the RTX 5090 environment. Read-only
inspection of the alternate A100 environment confirmed a separate dependency
blocker: its SGLang 0.5.2 lacks SALA model classes, the MiniCPM sparse backend,
InfLLMv2 operators and TileLang. It cannot run these adapters directly. No
packages were installed or modified there; a compatible isolated environment
would be required.
The latest [CPU-zlib tail adapter](sala_native_adapter.py) has **not been GPU
tested**. No native model equivalence, quality, memory-saving or speed result
follows from the earlier random operator probe.

The adapter must be installed inside every model worker before Engine creation.
It targets one request with unchunked full prefill; native KV compression,
linear state and logical-to-physical conversion still run. This recipe is for
future validation on a compatible environment, not a completed model run.
Prepare benchmark records with **the SALA tokenizer**, not Qwen token IDs.
See the [SALA recovery and evaluation runbook](sala_runbook.md) for worker setup,
checkpoint revision and environment requirements.

```sh
export REPO=/path/to/this-checkout
export MODEL=/path/to/MiniCPM-SALA-checkpoint
export TASK=/path/to/external-sala-experiment
export SELECTION_CONTROL="$TASK/control.json"
export SELECTION_LOG="$TASK/adapter.jsonl"
mkdir -p "$TASK/hook" "$TASK/results"
cat > "$TASK/hook/sitecustomize.py" <<'PY'
import os
from integrations.sala_native_adapter import install
install(os.environ["SELECTION_CONTROL"], os.environ["SELECTION_LOG"])
PY

# Random operator probe only; no model weights or quality measurement.
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python benchmarks/sala_selector_probe.py --tokens 16384

# After preparing TASK/longbench-sala.jsonl with the SALA tokenizer:
nvidia-smi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$TASK/hook:$REPO" \
python -m benchmarks.sala_quality_subset --model "$MODEL" \
  --dataset "$TASK/longbench-sala.jsonl" --control "$SELECTION_CONTROL" \
  --adapter-log "$SELECTION_LOG" \
  --output "$TASK/results/sala-quality.jsonl" --limit-per-task 4 \
  --fractions 0,0.5,0.75,1 --chat-template --verify
```

The SALA runner's `--chat-template` sets `enable_thinking=False`. It disables
radix caching and flushes engine cache before each mode. Its current Engine
configuration uses memory fraction 0.78 and a 32K context; check capacity before
running. Both model drivers require `--adapter-log` for capture/replay (quality
baseline-only mode is the exception). It must point to the exact log passed to
worker `install()`. They validate fresh events by `run_id`, request identity,
mode, unique layer coverage, effective split and actual native selector query
count. Missing hooks, dense-only requests, stale events or differing sparse
layer coverage fail validation rather than counting as replay hits. Compare the
same input IDs under native/capture/replay. Returned output-token logprobs are
narrower evidence than full-vocabulary logits. CPU-zlib transfer and verification
costs are included in whole-generation timing.
