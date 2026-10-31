# Native selector bypass diagnostic

The completed CPU-zlib SALA adapter experiment did not show an offline
first-token walltime speedup. That result combines selector bypass with CPU
inflate and a synchronous GPU copy; it cannot establish that selection reuse
itself has no benefit or predict a CUDA double-buffer implementation.

The 16K diagnostic completed and observed a small whole-request benefit from
selector bypass, supporting further investigation of replay transport cost.
The relevant comparison is the same native model's prefill with and without
selector execution. A dense-attention model is not required for that comparison.

## Historical motivation and evidence limits

A retained FlashNext summary reported selector-total P50 of 0.4235 ms and
sparse fused attention P50 of 0.1161 ms. Its separate stage P50 values were
0.0688 ms for score, 0.0388 ms for top-k and 0.0513 ms for block-index expand.
These marginal percentiles are not paired measurements: their sum or difference
does not identify an independently measured residual stage or an end-to-end
speedup ceiling.

That experiment used synchronized worker instrumentation, a 1024-token maximum
context, approximately 507–508 prompt tokens and 128 output tokens. It did not
separate prefill from decode, and synchronization perturbed execution. The
summary's link to the underlying worker report is unavailable; raw measurements
have not been recovered. The numbers motivate inspecting selection but do not
predict a long-context SALA result. No historical raw trace is republished here.

## Diagnostic design

`sala_selector_diagnostic.py` and `sala_diagnostic_adapter.py` compare:

| Mode | Work at the selection hook | Storage tradeoff |
|---|---|---|
| Native | Run the model's native selector | Normal native behavior |
| GPU resident | Return exact captured logical indices already on GPU | Full 16K selection retention is 96 MiB; not a memory-saving cache policy |
| Pinned CPU | Copy predecoded captured indices from pinned host memory | Full uncompressed host representation; timed requests exclude inflate |

Capture precedes all replay timing. The diagnostic retains both GPU and pinned
representations when pinned mode is enabled; host-mode measurements therefore
do not establish the footprint of a GPU-cache-free deployment. Main KV,
attention and other model work remain active. The input repeats a short sentence
and the generation budget is one token; this is not task-quality evidence.

Strict requests compare query states and complete ordered, padded selections
against capture, alongside returned text and token logprobs. Seven trials per
mode are randomly interleaved with seed 42 for unprofiled whole-request timing.
An independent profiling phase collects seven trials per mode using CUDA events
around the hook. Those intervals measure native selection or the replacement
path, not a decomposition of all model work. Profiling records are drained after
the request rather than synchronizing each layer. Strict hashes and profiled
walltimes are excluded from the main unprofiled timing comparison.

Whole-request times are offline first-token generation walltime, not service
TTFT. GPU-resident reuse is an optimistic memory-consuming diagnostic; pinned
reuse excludes decompression and is not the compressed production pipeline.
## Completed 16K results

The checkpoint was `openbmb/MiniCPM-SALA` revision
`9180fe1db74f71fb81bc7105efe88f6b19c959b0` on A100-SXM4-80GB, SGLang 0.5.20,
PyTorch 2.13.0+cu130 and Transformers 5.12.1, with static memory fraction .5.

| Mode | Unprofiled median seconds | Min–max seconds | Reduction versus native median |
|---|---:|---:|---:|
| Native | 1.913012 | 1.906894–1.916474 | — |
| GPU resident | 1.880705 | 1.873488–1.886031 | 1.69% |
| Pinned CPU | 1.885909 | 1.879566–1.890291 | 1.42% |

Each row has seven randomly interleaved trials. Separate CUDA-event profile
trials first sum the eight layer intervals within each request, then summarize
those seven request sums; these are not sums of marginal layer medians:

| Mode | Median eight-layer sum, ms | Min–max, ms |
|---|---:|---:|
| Native selector | 34.373632 | 33.546240–34.463745 |
| GPU resident hook | 0.038912 | 0.038912–0.041984 |
| Pinned H2D hook | 4.328800 | 4.308928–4.339392 |

All returned outputs and token logprobs matched the initial native output.
Both GPU and pinned strict checks, before and after unprofiled measurements,
covered all eight layers and verified identical query states and complete
ordered/padded captured indices. Capture also checked its query and indices
against the preceding verified native call. This is stronger than output
agreement alone, but does not compare full vocabulary logits.

The retained GPU representation was 100,663,296 bytes (96 MiB), and the pinned
host representation another 96 MiB. The experiment retained both in both replay
modes. It demonstrates a modest bypass opportunity at 16K under this protocol;
it does not demonstrate a compressed memory-saving implementation or isolate
score/top-k from all other native selector operations. The separate CPU-zlib
result cannot be generalized to this transport or a future CUDA pipeline.

Evidence: [all request records](../integrations/results/sala_selector_diagnostic16k_a100.json),
[layer events](../integrations/results/sala_selector_diagnostic16k_a100_events.jsonl),
and [metadata](../integrations/results/sala_selector_diagnostic16k_a100_metadata.json).
The original metadata contained an unrelated quality-prompt policy; the public
metadata instead records the synthetic raw-token input actually used by the runner.

## Completed 24K results and 32K admission limit

The same protocol and environment completed at 24576 input tokens, with seven
randomly interleaved unprofiled trials per mode:

| Mode | Median seconds | Min–max seconds | Reduction versus native median |
|---|---:|---:|---:|
| Native | 2.917551 | 2.915810–2.921560 | — |
| GPU resident | 2.864013 | 2.853425–2.869005 | 1.84% |
| Pinned CPU | 2.871700 | 2.864888–2.876640 | 1.57% |

Independent profile trials again sum all eight layer intervals per request
before computing the median and range over seven requests:

| Mode | Median eight-layer sum, ms | Min–max, ms |
|---|---:|---:|
| Native selector | 60.908545 | 59.447296–61.335552 |
| GPU resident hook | 0.041984 | 0.040960–0.044032 |
| Pinned H2D hook | 6.200352 | 6.187840–6.211200 |

All outputs and returned-token logprobs matched. Both strict GPU and pinned
checks before and after measurement covered eight layers with all query/index
checks passing. The retained GPU and pinned host representations were each
150,994,944 bytes (144 MiB), simultaneously present. The observed whole-request
benefit remains modest despite bypassing most measured selector work; this
does not establish a production memory-saving cache or a general scaling law.

Evidence: [request records](../integrations/results/sala_selector_diagnostic24k_a100.json),
[layer events](../integrations/results/sala_selector_diagnostic24k_a100_events.jsonl),
and [metadata](../integrations/results/sala_selector_diagnostic24k_a100_metadata.json).

A 32768-token attempt was rejected before inference: at static memory fraction
.5, SGLang admitted at most 30624 input tokens. A subsequent logprob-processing
`NoneType` exception followed that rejection. No selector events or valid
inference timings were produced. This is an incomplete configuration-limited
attempt, not a zero-time result or evidence of a hardware fault; the 24K run
kept the .5 memory setting rather than silently truncating the requested input.
