# MiniCPM-SALA prefill graph source audit

Source audit and subsequent successful adapted 16K native/cache breakable validation for installed SGLang 0.5.20. See [prefill graph results](prefill_graph_results.md). Source locations below are relative to the installed package.

- `srt/arg_groups/cuda_graph_hook.py:60–62`: legacy SDK
  `disable_cuda_graph=True` disables both decode and prefill. Setting it false
  merely removes that override; it does not prove graph execution.
- `srt/model_executor/cuda_graph_config.py`: CUDA defaults are decode `full`
  and prefill `breakable`. The phase-aware interface replaces the old
  `enable_piecewise_cuda_graph` approach in this installation. Use
  `cuda_graph_config` or `cuda_graph_backend_prefill` explicitly.
- `srt/arg_groups/model_overrides/minicpm.py`: hybrid SALA forces radix cache
  off; it does not declare prefill graph support. MiniCPMSALA is absent from
  the architecture blacklist at `srt/configs/model_config.py:2181`.
  Absence from a blacklist is not a successful compatibility test.

Full prefill graphs are not the first candidate:
`srt/layers/attention/minicpm/backend.py:944–953` rejects non-decode/idle in
`init_forward_metadata_out_graph`. Full prefill replay calls that method at
`srt/model_executor/runner/prefill_cuda_graph_runner.py:1106`.
This restriction alone does **not** rule out breakable or TC-piecewise:
lines 1056–1070 and 1108–1114 use eager metadata initialization for their
ordinary backend contract.

Breakable prefill is the better candidate for a controlled compatibility test.
It captures the transformer body, leaving the LM head/logits tail eager
(prefill runner lines 548–559, 615–616). Native sparse selection and Lightning
Attention still require an actual capture/replay test; neither their Python
presence nor accepting a flag certifies graph-safe execution. Python diagnostic
hooks, control reads, CPU hashing/copies and mode changes cannot be assumed to
execute on every captured graph replay. A cached graph must not silently retain
a capture-time native/replay branch.

Two independent eager-fallback conditions matter:

1. Prefill token capture buckets must cover the entire full-prefill input.
   Default settings target a small token budget (the hook documents 2048).
   The runner rejects token counts above its maximum captured bucket at
   lines 1185–1190. For 16K/24K, explicitly provide a 16384/24576 bucket and
   matching max budget; do not infer execution from decode graphs.
2. With `return_logprob=True`, TC-piecewise fails the eligibility check at
   lines 1181–1182 because only breakable/full have the eager logits tail.
   Existing returned-logprob diagnostics would silently remain eager under TC.

Candidate SDK configuration, subject to real capture validation:

```python
cuda_graph_config={
    'decode': {'backend': 'disabled'},
    'prefill': {'backend': 'breakable', 'bs': [16384], 'max_bs': 16384},
}
```

For 24K replace both token values by 24576. Keep request concurrency one,
full unchunked prefill, radix off and sufficient runtime token capacity.
Collect resolved configuration, capture logs, actual graph replay evidence,
output/logprob equivalence and selector-path evidence before labeling a result
prefill-graph enabled. Use separate engines/captures per native/cache branch
unless branch switching under graph replay is explicitly established.
Only then compare native/cache crossed with eager/graph; keep profiling runs
separate from the unprofiled walltime comparison.

## Executed native 16K attempt: capture failed

The native breakable attempt loaded model weights, then failed during capture
warmup before measured requests. It produced zero measurement rows and no events
artifact. The process exit was -9; the log alone does not establish why the
parent was killed. The selected card subsequently had no active compute process.
See [sanitized failure record](../integrations/results/sala_prefill_bcg16k_failure.json).

Two separate problems were observed:

- The evidence hook incorrectly patched `PrefillCudaGraphRunner.replay`, which
  does not exist in this installation. `sitecustomize` logged AttributeError.
  The runner uses `execute` (line 1918) and delegates to backend `replay` at
  lines 1815/1865. Thus this attempt cannot supply actual-replay evidence.
- Capture warmup reached MiniCPM `self.o_norm(o)` at `models/minicpm.py:346`,
  then FlashInfer Cute RMSNorm rejected weight extent against expected 128.
  MiniCPM constructs output norm over `num_heads * head_dim` (line 278),
  whereas its per-head q/k norms use `head_dim`. This is consistent with an
  attention-output/dummy-shape contract mismatch, but runtime tensor shapes
  were not recorded, so its precise origin remains unproven. The traceback
  does not establish corrupted weights, a BF16 numerical problem, or general
  A100 FlashInfer incompatibility.

There is an official process-local dispatch alternative:
`FLASHINFER_USE_CUDA_NORM=1`. FlashInfer `norm/__init__.py:60,89` explicitly
supports it; `_rmsnorm_impl` then selects the CUDA norm module instead of Cute.
It is a candidate compatibility probe, not a demonstrated fix. A real shape
mismatch must still fail; skipping output norm, replacing its mathematical
meaning, or declaring capture successful without replay evidence is invalid.
Any follow-up must first repair the evidence hook, preserve strict output/logprob
checks, and separately establish actual prefill graph replay.

## Source-contract diagnosis and process-local probe adaptation

The failing class is specifically `MiniCPMLightningMixer` (not the dense
`MiniCPMAttention`). Its output norm width is `num_heads * head_dim`
(`models/minicpm.py:277–280`). Its RadixAttention receives three-dimensional
q/k/v. During BCG, `layers/radix_attention.py:226` allocates `empty_like(q)`;
the unified output path returns that three-dimensional tensor directly near
line 288. Ordinary MiniCPM backend returns flattened two-dimensional tensors
(`layers/attention/minicpm/backend.py:747,858`). This explains why output norm
sees a per-head 128 dimension instead of its full head-concatenated width.

The diagnostic adapter now offers explicit `SALA_PREFILL_SHAPE_FIX=1`, wrapping
only each output-normalized MiniCPM Lightning mixer's own attention instance.
Valid two-dimensional outputs pass unchanged. Three-dimensional `[T,H,D]`
outputs are reshaped only after checking H, D and the norm weight width;
unexpected shapes fail. It preserves tensor values and still executes output
norm. Enable it identically for eager and breakable controls. Metadata labels
it `restore eager 2D attention contract`. It does not modify installed packages.

This source-justified compatibility adaptation subsequently passed the native
and selection-cache 16K breakable probes described below. The initial failure
remains historical evidence; it is not the final compatibility status.

## Subsequent successful 16K breakable validation

After the evidence-hook repair and explicit process-local shape adaptation,
the native breakable probe completed successfully: all 22 requests had actual
prefill graph replay evidence and seven separate profile intervals were retained.
The full-selection cache diagnostic also completed: all 53 requests had one
actual graph replay each; all first 52 requests had eight selector-hook events.
The final request exists to drain deferred events and is excluded from selector
coverage claims.

Strict GPU-resident and pinned-CPU replay checks before and after timing passed
query-state and exact ordered, padded selection comparisons. Outputs matched
the native eager probe's text, returned-token logprobs and returned top-five
logprobs. The breakable implementation therefore demonstrably executes the
Python selection hook inside its attention break region for these requests:
dynamic native/capture/replay branches were observed, not inferred from flags.
This evidence does not extend to Full or TC-piecewise graphs.

The same shape adapter is enabled in the eager control, where valid two-
dimensional outputs pass unchanged. No norm or attention arithmetic is omitted;
GPU-resident selections remain an optimistic memory-consuming reference and
pinned CPU selections consume additional DRAM. Neither is a compressed CUDA
codec or production cache policy.

The matched eager/cache comparison is reported separately in
[prefill graph results](prefill_graph_results.md). No final timing numbers are
asserted here. Distinguish unprofiled offline first-token walltime from separate
current-stream forward CUDA intervals (which include launch gaps), and from
pure kernel sums or server TTFT. These synthetic single-request 16K probes do
not establish behavior at other sizes, multi-request serving, or graph backends.
