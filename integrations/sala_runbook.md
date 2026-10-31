# Native MiniCPM-SALA recovery and evaluation

This is an experimental SGLang 0.5.20 adapter for single-request, unchunked full
prefill. It caches only native logical block indices, before physical KV-page
translation. Native KV compression, attention, and Lightning Attention state
reconstruction still run. The persistent cache is a CPU zlib-compressed int32
tail, preserving the native index order and `-1` padding without additional
sorting. It does not use the CUDA codec or
double-buffer pipeline in the main prototype.

## Current evidence

- Public checkpoint: `openbmb/MiniCPM-SALA`, revision
  `9180fe1db74f71fb81bc7105efe88f6b19c959b0`.
- Four BF16 safetensors shards total 18,954,453,608 bytes; SHA256 checked on the
  download machine and the initial evaluation machine.
- Random-tensor native nonfused operator probe at 16,384 queries passed exact
  equality for head lengths 8,192, 12,288, and 16,384 when both compressed-key
  levels were cropped to their native head lengths. This is operator evidence,
  not a pretrained-model result.
- The fused operator produced differing indices for immediately repeated full
  calls on identical random inputs. Its active sparse suffix also differed.
  The planned model runs therefore use native `minicpm_flashattn` (nonfused).
- The initial pretrained-model launch failed before loading weights because
  CUDA driver initialization failed. Binding the intended card by UUID also
  failed. No SALA model-quality or model-performance result is available.
- The CPU-tail adapter revision has passed syntax checks; its GPU model
  execution has not yet been verified.

## Restore execution after the accelerator is available

Use a dedicated directory outside the source repository for weights, caches,
benchmark contexts, control files, and outputs. Set these variables to existing
local paths; no machine address or account is embedded in the commands:

Before restoring a previous remote task, synchronize the latest
`integrations/sala_native_adapter.py`, `integrations/sala_validation.py`, and both
`benchmarks/sala_native_smoke.py` and `benchmarks/sala_quality_subset.py` from this
repository. The earlier remote script copies and launcher predate the strict
hook-validation contract and must not be used to establish a successful replay.

```bash
export SALA_SOURCE_DIR=/absolute/path/to/selection-prefix-cache
export SALA_TASK_DIR=/absolute/path/to/dedicated-sala-task
export SALA_PYTHON=/absolute/path/to/compatible-environment/bin/python
export SALA_PHYSICAL_GPU=1
export CUDA_HOME=/absolute/path/to/compatible-cuda-toolkit
```

Immediately before each inference launch inspect the selected physical card's
memory, utilization, and compute-process ownership. A zero utilization reading
alone is insufficient. Do not launch if another workload is present or if the
driver cannot enumerate the accelerator. Do not reset other cards or processes.

```bash
nvidia-smi --id="$SALA_PHYSICAL_GPU" \
  --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
CUDA_VISIBLE_DEVICES="$SALA_PHYSICAL_GPU" "$SALA_PYTHON" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.device_count())'
```

Install the hook only for these processes using `sitecustomize.py`:

```bash
mkdir -p "$SALA_TASK_DIR/hooks" "$SALA_TASK_DIR/tmp" "$SALA_TASK_DIR/cache"
cat > "$SALA_TASK_DIR/hooks/sitecustomize.py" <<'PY'
import os
if os.environ.get("SALA_REPLAY_CONTROL"):
    from sala_native_adapter import install
    install(os.environ["SALA_REPLAY_CONTROL"], os.environ["SALA_REPLAY_LOG"])
PY
printf '%s\n' '{"mode":"baseline"}' > "$SALA_TASK_DIR/control.json"
export CUDA_VISIBLE_DEVICES="$SALA_PHYSICAL_GPU"
export PYTHONPATH="$SALA_TASK_DIR/hooks:$SALA_SOURCE_DIR/integrations:$SALA_SOURCE_DIR"
export SALA_REPLAY_CONTROL="$SALA_TASK_DIR/control.json"
export SALA_REPLAY_LOG="$SALA_TASK_DIR/selector-events.jsonl"
export HF_HOME="$SALA_TASK_DIR/hf"
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$SALA_TASK_DIR/cache"
export TMPDIR="$SALA_TASK_DIR/tmp"
export TRITON_CACHE_DIR="$SALA_TASK_DIR/triton"
export TILELANG_CACHE_DIR="$SALA_TASK_DIR/tilelang"
export TORCH_EXTENSIONS_DIR="$SALA_TASK_DIR/torch-extensions"
```

First run strict model capture/replay. The driver disables CUDA graphs and radix
prefix caching, flushes runtime caches between requests, and captures each cut
independently before three replay hits. H=0, .125N, .25N, .5N, N are included.
Intermediate cuts align to 64 query tokens; H=N includes the final partial block.

```bash
"$SALA_PYTHON" "$SALA_SOURCE_DIR/benchmarks/sala_native_smoke.py" \
  --model "$SALA_TASK_DIR/model" --control "$SALA_REPLAY_CONTROL" \
  --adapter-log "$SALA_REPLAY_LOG" \
  --output "$SALA_TASK_DIR/smoke-16k-strict.json" --tokens 16384 \
  --backend minicpm_flashattn --verify --repeats 3
```

Strict verification hashes queries and recomputed head indices through the CPU.
That synchronization changes timing. After strict verification passes, run again
without `--verify` and save a separate timing artifact. Capture calls include the
first native selection, CPU transfer, and tail compression and must be excluded
from replay-hit latency statistics. Report generation time as generation time,
not TTFT. Event logs report actual selector rows, raw/compressed tail bytes,
capture time, decode/copy time, and zero persistent GPU cache bytes. A complete
working output tensor is still allocated during replay.

`--adapter-log` is required for the smoke driver and for quality capture/replay.
The drivers require intercepted events for every capture/replay request and
validate its unique `run_id`, content identity, cut, native selector row count,
and layer set. A missing hook, missing events, or zero intercepted selector calls
fails validation instead of being accepted as a cache hit. Missing output-token
log probabilities also fail the equality check.

The checkpoint uses `dense_len=8192`; contexts below the native sparse threshold
can produce zero sparse-selector calls. Treat contexts at or below 8K as
baseline-only samples unless native sparse calls are actually demonstrated.
Quality evaluation of such pure-dense samples requires `--baseline-only`; using
them in capture/replay fails validation. They establish model behavior but cannot
establish selector-skipping speedups. Split a mixed dataset before replay
evaluation so pure-dense samples are not mistaken for successful replay cases.

## Small official quality subsets

Prepared JSONL files must use this model's tokenizer and contain official sample
IDs, prompts, answer lists, and token counts. Keep their benchmark text outside
Git. The supported preparation scripts are `integrations/benchmark_data.py` and
`integrations/ruler_data.py`; the latter invokes pinned official RULER source.

```bash
"$SALA_PYTHON" "$SALA_SOURCE_DIR/benchmarks/sala_quality_subset.py" \
  --model "$SALA_TASK_DIR/model" --dataset "$SALA_TASK_DIR/longbench-16k.jsonl" \
  --control "$SALA_REPLAY_CONTROL" --output "$SALA_TASK_DIR/longbench-raw-results.jsonl" \
  --adapter-log "$SALA_REPLAY_LOG" \
  --limit-per-task 4 --fractions 0,0.25,0.5,1 --verify
```

Repeat with `ruler-16k.jsonl`. LongBench uses the official 32-token cap; RULER uses
128 tokens. `--chat-template` is a separate model-native chat protocol and must
have its own output file. Raw official prompts and native-chat results must be
reported separately. These are small subsets, not full benchmark leaderboards.
Use a new output filename for each invocation; the drivers reject overwriting an
existing result. `--adapter-log` may be omitted only with the quality driver's
`--baseline-only` option. The quality equality fields are `text_equal_baseline`
and `output_token_logprobs_equal_baseline`; capture calls are recorded separately.

## Alternate environment requirements

The inspected alternate installation had SGLang 0.5.2 and Torch 2.8/cu128. Its
MiniCPM classes did not include SALA, and it had neither the MiniCPM sparse backend
nor InfLLMv2 operators. Existing Transformers/FLA/FlashAttention/FlashInfer do not
provide those missing native SGLang components. A separate compatible environment
is required; copying this adapter into that installation is insufficient. Do not
upgrade or replace its existing packages in place.
