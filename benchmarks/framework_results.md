# Pretrained-model controls and native adapter status

Completed dense-model task controls and uncompleted native sparse-model
integration are separate evidence. None of the completed pretrained-model runs
measures selection-replay speedup, TTFT improvement or replay quality retention.
Commands are in [integrations/README.md](../integrations/README.md).

## Completed Qwen3-4B controls

All runs used pretrained Qwen3-4B, dense FLASH_ATTN attention, vLLM 0.30.0,
PyTorch 2.13.0, Transformers 5.17.0, BF16, TP=1, temperature 0 and seed 0.
Prefix caching and CUDA graphs were disabled. Chunked prefill used a 4096-token
batch budget, one running request and GPU memory utilization 0.60. The native
sampler switch bypassed an installed FlashInfer architecture-detection error.

Each task has **four examples**. Scores are on a **0–1 scale**:

| Benchmark/task | Raw task text | Native chat, nonthinking | Output budget |
|---|---:|---:|---:|
| LongBench `passage_retrieval_en` | 0.833333 | 0.750000 | 32 |
| LongBench `2wikimqa` | 0.025000 | 0.250000 | 32 |
| RULER `niah_single_1`, 16K | 1.000000 | incomplete | 128 |
| RULER `niah_multikey_2`, 16K | 1.000000 | incomplete | 128 |

Evidence:

- [Summary and protocol](../integrations/results/qwen_quality_summary.json)
- [LongBench raw predictions](../integrations/results/qwen3_longbench4_raw32.jsonl)
- [LongBench chat predictions](../integrations/results/qwen3_longbench4_chat_nothink32.jsonl)
- [RULER raw predictions](../integrations/results/qwen3_ruler16k4_raw128.jsonl)

JSONL records include sample IDs, source rows, prompt hashes, input counts,
predictions, references, scores, output lengths and versions. Source contexts
and prepared token IDs remain outside the repository. The weight revision was
not recorded in these control outputs; exact reproduction needs the same local
snapshot, and future evaluation should record its revision.

### LongBench protocol

The first four source rows per task use official prompts and metric behavior
from [LongBench code revision
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench).
The archive is pinned at [data revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`](https://huggingface.co/datasets/THUDM/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672).
Archive/config SHA256 hashes are checked and recorded. The input cap was 16384,
with official middle truncation and retokenization available; none of these
eight Qwen task bodies required truncation.

Retrieval body lengths were 11442/11883/13306/13021 tokens; 2wiki body lengths
were 7379/7124/6819/11609. The native Qwen user-chat generation template with
`enable_thinking=False` added twelve tokens to each body. Raw and chat are
therefore **different input protocols**, not repeated trials of one condition.

All raw predictions reached the 32-token cap, often continuing with explanation
or repeated answers. Chat answers used 2–8 generated tokens. Chat retrieval
predictions were `Paragraph 15`, `Paragraph 8`, `28`, and `Paragraph 17`; the
third reference was `Paragraph 18`. Chat 2wiki answered one of four examples
correctly. These observations show task behavior and format sensitivity; four
examples cannot establish general quality or a statistically significant
protocol difference.

### RULER protocol

The unmodified official generator from [NVIDIA/RULER revision
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)
generated `niah_single_1` and `niah_multikey_2`, four examples each, seed 42,
with a 16384-token total budget including 128 output tokens. The base template
uses official `input + answer_prefix`, HF tokenization with
`add_special_tokens=False` and no chat wrapper. Actual input lengths were
15731–16254 tokens without truncation. Source-file hashes and task-local CPU
dependency versions are retained in provenance.

All eight raw predictions contained the correct seven-digit number and scored
1.0 under official case-insensitive `string_match_all`. All reached the output
cap while continuing after the number. This substring-recall metric does not
penalize extra text; it is not exact-answer accuracy.

Native-chat nonthinking completed five rows; the sixth encountered
`CUDA error: unspecified launch failure`. Subsequent device inspection could
not obtain that GPU's handle. No reset or driver/package change was attempted.
Partial predictions and the original log were retained outside the repository;
this incomplete run has no complete-task score here.

### Timing boundary

`generation_seconds` surrounds synchronized whole generation, including prefill
and all output tokens. It is **not TTFT**. Cold/JIT effects were not controlled
as a performance experiment, and output lengths differ between protocols.
No latency comparison or selection-cache benefit follows. The separate
[synthetic split microbenchmark](split_results.md) has a different measurement
boundary and is not model TTFT either.

## Native sparse-model integration status

| Path | Evidence obtained | Missing evidence/status |
|---|---|---|
| vLLM GLM DSA | Real tiny checkpoint launch attempted; CPU mocked-hook contract passed | Sparse-MLA dependency blocked construction on SM120; no native capture/replay result |
| SGLang SALA operator | Random 16K nonfused selector, H=8192/12288/16384 matched the full-prefix selection | No weights loaded; does not test latest CPU-zlib adapter or all model states |
| SGLang SALA model | Adapter and benchmark runners prepared | Pretrained model never loaded; container CUDA failures blocked evaluation; latest tail adapter untested on GPU |
| A100 SALA dependency inspection | Read-only check confirmed SGLang 0.5.2 lacks SALA classes/backend, InfLLMv2 operators and TileLang | Cannot execute SALA directly; no packages installed or modified, no native model result |

The vLLM failure was `FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's sparse
MLA decode API`; Triton MLA did not support sparse MLA, and the installed
FlashMLA sparse path targeted SM90/SM100. The tiny GLM DSA checkpoint was randomly
initialized and toy-text finetuned according to its publisher, so it supports
architecture smoke rather than representative quality evaluation.

Both adapters preserve ordered logical selections and delegate current request
block-table conversion to the framework. CPU-zlib tail storage is an adapter
baseline, separate from the repository's CUDA codec/transfer pipeline. SALA head
recomputation crops both compressed-key levels to avoid bottom-right causal
alignment shifts. Random operator agreement supports that limited indexing
check, not model output equivalence. Native model validation still needs weight
loading, identical input IDs, fresh KV/linear state, actual selector-skip
evidence, matching outputs and specified logprob/logit checks before any
quality-retention claim.

The reviewed SALA drivers now require `--adapter-log` for capture/replay and
validate fresh events against the current `run_id`, input identity, mode,
unique sparse-layer coverage, effective head and actual native selector query
counts. Missing hooks, dense-only execution, stale logs or incomplete layer
coverage cannot count as successful replay. This is a driver safeguard, not
evidence of a completed GPU model run. See the
[SALA recovery and evaluation runbook](../integrations/sala_runbook.md) for
checkpoint pinning and setup; the alternate A100 installation remains blocked
by missing native dependencies rather than an untested compatibility assumption.
