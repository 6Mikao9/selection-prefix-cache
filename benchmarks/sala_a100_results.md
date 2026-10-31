# Native MiniCPM-SALA on A100

The strict 16K synthetic capture/replay and separate timing runs completed.
This CPU-tail adapter showed consistent outputs but no observed walltime
speedup on this synthetic input. An initial native-chat task baseline exposed
an output-budget limitation. Raw LongBench and RULER task smoke runs completed.

The checkpoint was `openbmb/MiniCPM-SALA`, revision
`9180fe1db74f71fb81bc7105efe88f6b19c959b0`, using the native nonfused
`minicpm_flashattn` backend. The device was A100-SXM4-80GB, with SGLang 0.5.20,
PyTorch 2.13.0+cu130 and Transformers 5.12.1. Static memory fraction was 0.5;
the runtime token budget was 32768. See [metadata](../integrations/results/sala_strict16k_a100_metadata.json).

## Consistency evidence

The input had 16384 tokens and the output budget was one token. Three native
baseline trials returned identical text (` books`), output token ID (6779),
returned-token logprob and returned top-five logprobs. All five captures and
15 replays matched the first native baseline on those same fields.

Each measured request intercepted the same eight selection layers:
0, 9, 16, 17, 22, 29, 30 and 31. Every replay had eight events (120 total),
with `head_indices_equal=true`; query verification was enabled and the run
completed without stale-query rejection. Effective heads were 0, 2048, 4096,
8192 and 16384, each with three replay trials. Native selector queries per
layer equaled the effective head, including zero at H=0. The compressed tail
preserved captured logical indices, their order and padding.

Evidence: [whitelisted request records](../integrations/results/sala_strict16k_a100.json)
and [whitelisted selector events](../integrations/results/sala_strict16k_a100_events.jsonl).
The public artifacts exclude server timestamps, request response IDs, original
logs, control files and private paths.

Repeated native baselines were checked for text and returned-token logprob
stability before capture/replay comparisons by effective head length and
layer coverage. Strict query-state hashes and ordered, padded head-index hashes
provide stronger evidence than matching output alone; returned-token logprobs
do not constitute an exhaustive comparison of vocabulary logits.

## Timing evidence

The separate run disabled query/index hash verification, warmed up before
measurement, and used the same 16384-token repeated-sentence input and one-token
output budget. Seven native baselines were stable; all five captures and 35
replays matched baseline text and returned-token logprobs. Times below are
seconds, with seven trials per baseline/replay and one capture per cut.

| Mode | H/N | Median | Min–max | Capture, single trial |
|---|---:|---:|---:|---:|
| Native | 1 | 1.900711 | 1.890335–1.903867 | — |
| Replay | 0 | 1.916129 | 1.904758–1.920187 | 2.138090 |
| Replay | 0.125 | 1.907548 | 1.905465–1.943910 | 2.111527 |
| Replay | 0.25 | 1.907445 | 1.905331–1.910251 | 2.091139 |
| Replay | 0.5 | 1.909521 | 1.908023–1.914510 | 2.042538 |
| Replay | 1 | 1.916424 | 1.913333–1.917380 | 1.913087 |

Replay medians were approximately 0.35–0.83% slower than the native median.
This describes the measured CPU zlib decode/copy adapter, including hook work;
it does not measure or rule out benefits from the prototype CUDA double-buffer
pipeline. Capture cost is separate and has not been amortized. Seven sequential
trials per condition do not establish a general performance effect.

Evidence: [all timing samples](../integrations/results/sala_timing16k_a100.json),
[selector events](../integrations/results/sala_timing16k_a100_events.jsonl), and
[environment metadata](../integrations/results/sala_timing16k_a100_metadata.json).

Strict verification transfers tensors to the CPU for hashing and must be kept
separate from timing without these checks. The public records retain all
timing samples, separating baseline, capture and replay.

Per-layer `tail_decode_copy_seconds` uses a CPU timer around inflate and a
synchronous copy, which may wait for earlier GPU work. Its layer sum is not an
isolated decompression or transfer cost. Likewise, `capture_seconds` includes
native queued GPU work and CPU serialization rather than isolated kernel time.
The performance comparison above uses whole-request walltime only.

Whole-generation walltime includes prefill and output generation. With a
one-token output budget it is offline first-token generation walltime, not
service TTFT. The repeated-sentence synthetic input measures this execution
path, not task quality.

## Selection-cache capacity

Strict-run capture totals over all eight layers are:

| H/N | Effective head | CPU compressed tail bytes | Saving versus H=0 | Raw tail bytes |
|---|---:|---:|---:|---:|
| 0 | 0 | 1,805,373 | 0% | 100,663,296 |
| 0.125 | 2048 | 1,703,358 | 5.7% | 88,080,384 |
| 0.25 | 4096 | 1,586,945 | 12.1% | 75,497,472 |
| 0.5 | 8192 | 1,182,432 | 34.5% | 50,331,648 |
| 1 | 16384 | 64 | 99.996% | 0 |

Full raw selection storage was 100,663,296 bytes (96 MiB) for every capture. Moving
from quarter to half head reduced the compressed tail by 404,513 bytes
(25.49%). H=N retains eight empty zlib streams, totaling 64 bytes. These
counts exclude Python/container overhead and are specific to this input.

Capacity totals sum `tail_compressed_bytes` over all eight observed selection
layers per capture. Raw tail and full raw selection sizes are recorded separately.
`persistent_gpu_bytes=0` refers only to the persistent selection cache; it says
nothing about the main KV cache, temporary tensors or peak GPU allocation.

The adapter stores native logical block indices as CPU zlib-compressed int32
tails and copies decoded tails back to the GPU. It does not implement the
prototype CUDA codec or its double-buffer pipeline. Main KV compression,
attention and Lightning Attention state reconstruction still execute.

## Task smoke results

The completed native-chat LongBench baseline used the first four source rows
for each task, the official task body plus the model's chat template, and a
32-token output budget. Both `passage_retrieval_en` and `2wikimqa` scored 0.0
(four examples each, scores on a 0–1 scale). Every prediction began with
`<think>` and reached 32 generated tokens before supplying a scored answer.
Input lengths were 12325–14395 and 7473–12505 tokens respectively.

This reveals a limitation of this chat protocol at this budget. The model's
template does not implement the `enable_thinking` switch, so this run is not
evidence for a guaranteed nonthinking protocol. It contained baseline requests
only and provides no evidence of cache-induced quality degradation. The
baseline self-comparison fields are also not repeated-run stability checks.

[All eight predictions, references and provenance](../integrations/results/sala_longbench_chat_baseline_a100.jsonl)
are retained, including the zero scores. Official raw-prompt baseline results
are reported separately below; neither the output budget nor
scoring rule has been changed to improve these chat scores. In the shared
provenance, `chat_wrapper=none` describes the prepared task body; the record's
`protocol=native_chat` identifies the wrapper actually applied for generation.

The separate official-raw protocol used the same first four task rows and
32-token output budget, without the chat wrapper. Retrieval scored 1.0 over
four examples: outputs identified paragraphs 15, 8, 18 and 17. The first used
7 generated tokens; the other three reached 32 tokens while continuing with
explanation. The metric therefore accepts answers with additional text.
Raw retrieval input lengths were 12316–14386 tokens.

Raw `2wikimqa` scored 0.0 over four examples (input lengths 7464–12496 tokens).
All four reached 32 generated tokens. Outputs included unfinished reasoning,
an intermediate family relation, a director name in response to a workplace
question, and director notes rather than a scored answer. These observations
show behavior under this specific prompt and budget; they do not establish
general model capability or determine what a larger budget would achieve.

[All eight raw predictions and references](../integrations/results/sala_longbench_raw_baseline_a100.jsonl)
are retained separately from chat results. These baseline-only results have
no replay comparison within that baseline run. The three shorter `2wikimqa`
inputs have baseline evidence only; the separate replay run is described below.

The raw replay evaluation completed on all five task bodies longer than 8192
tokens: four retrieval examples and `2wikimqa` source row 3. For each sample,
cuts 0, .25, .5 and 1 each had one capture and one replay (20 of each).
Every capture/replay matched its native baseline text, returned-token logprobs
and score. Retrieval remained 1.0 over these four examples; the single 2wiki
example remained 0.0. All requests intercepted eight layers; all 160 replay
layer events passed the head-index check. These are strict checks, not timing
measurements. The three shorter 2wiki examples have no replay claim.

For source rows retrieval 0/1/2/3 and 2wiki 3 respectively, effective quarter
heads were 3072/3136/3584/3456/3072 and half heads were
6144/6272/7168/6912/6208. These use actual 64-token alignment, not nominal
fractions. The native SALA `dense_len` is 8192, so these partial recomputation
heads, as well as the synthetic .125/.25/.5 heads, lie within the dense region.
They do not demonstrate a long sparse recomputation head. The event field
`native_selector_queries` counts query rows passed to the selector, not exact
scoring FLOPs.

Summing all eight capture layers separately by prompt identity and effective
head, natural-document CPU compressed tails ranged from 3,469,939–4,860,244
bytes at H=0, 3,306,447–4,666,652 at quarter head, and
3,106,730–4,271,083 at half head. H=N retained 64 bytes. These differ from the
repeated-sentence synthetic input; its compression ratio is not representative
of natural documents.

Evidence: [45 request records](../integrations/results/sala_longbench_raw_replay_a100.jsonl)
and [selector events](../integrations/results/sala_longbench_raw_replay_a100_events.jsonl).

RULER used the official raw task prompt for four `niah_single_1` and four
`niah_multikey_2` examples, with 16212–16239 input tokens and a 128-token output
budget. Each sample had a native baseline and capture/replay at cuts
0, .25, .5, .75 and 1: 88 requests total. Both tasks scored 1.0 in baseline
and at every capture and replay cut. All 88 requests reached 128 output tokens;
official case-insensitive substring recall accepts additional text, so these
scores do not imply concise or exact-answer generation.

All 40 captures and 40 replays matched their own baseline text, returned-token
logprobs and score. All 88 requests intercepted eight layers and all 320 replay
layer events passed the head-index check. At .75, effective heads were 12160
for all four single-key samples and for multikey rows 2/3; multikey rows 0/1
used 12096. Every .75 head therefore exceeded `dense_len=8192`, providing
strict partial-recomputation evidence across the dense-to-sparse threshold,
while preserving a replayed suffix. This is still a correctness smoke test,
not a task latency comparison or exhaustive sparse-model validation.

Evidence: [88 predictions and records](../integrations/results/sala_ruler_raw_replay_a100.jsonl),
[selector events](../integrations/results/sala_ruler_raw_replay_a100_events.jsonl),
and [metadata](../integrations/results/sala_ruler_raw_replay_a100_metadata.json).

Native baseline sample counts and scores are kept separate from selected
replay subsets and from the different chat protocol. These small LongBench and RULER subsets cannot
establish leaderboard performance or general quality retention. RULER substring
recall also does not establish exact-answer accuracy.
