# Native SALA prefill graph probe

Breakable CUDA graph prefill executed successfully at 16384 input tokens.
The native-only probe observed nearly identical eager and graph timings.
A separate matched diagnostic below verified selector reuse inside actual
prefill graph execution and found modest hit-path benefits.

Both conditions used MiniCPM-SALA revision
`9180fe1db74f71fb81bc7105efe88f6b19c959b0`, the same repeated-sentence input,
one generated token, A100-SXM4-80GB and SGLang 0.5.20. Decode graphs were
disabled. The graph runner's successful `execute -> backend.replay` path was
recorded for every graph request; eager requests recorded no such replay.
Each condition had 22 requests: seven warmup, seven measurement, seven profile
and one drain. Across all 44 requests, text, output IDs, returned-token
logprobs and returned top-five logprobs were identical.

A process-local shape adapter on `MiniCPMLightningMixer` restores the eager
two-dimensional attention-output contract when the graph path returns
`[tokens, heads, head_dim]`. It reshapes to `[tokens, heads*head_dim]` before
the existing output norm. It does not replace the norm or modify the installed
environment. Both conditions used this adapter and recorded its identity.

| Execution | Offline first-token wall median, seconds | Min–max, seconds | Profile forward interval median, ms | Min–max, ms |
|---|---:|---:|---:|---:|
| Eager | 1.910064 | 1.904644–1.911157 | 1874.406372 | 1872.347168–1876.910034 |
| Breakable graph | 1.907527 | 1.906446–1.910192 | 1872.423950 | 1871.125488–1877.787598 |

Each summary uses seven requests. Whole-request walltime comes from the
unprofiled measurement phase; forward intervals come from separate profile
requests. The latter surrounds `ModelRunner.forward` on the current CUDA
stream and includes launch gaps; it is not a sum of kernel times. Walltime is
offline first-token generation, not server TTFT. The tiny median differences
(about .13% walltime and .11% forward interval) are within overlapping ranges
and do not establish a graph speedup.

The two engines ran separately in sequence rather than randomly interleaving
execution conditions. This native-only comparison is not evidence that dynamic
selection-cache hooks execute in a captured graph: that requires the separate
hook-coverage and exact selection checks reported below.

## Matched selector-cache diagnostic

The matching eager and breakable-graph engines each completed 53 requests.
Within each engine, seven unprofiled trials per mode were randomly interleaved;
seven separate profile requests per mode measured the current-stream forward
interval. Statistics below use Python `statistics.median`. Reductions compare
each mode with its own execution condition's native median, not across engines.

| Execution | Mode | Offline wall median s | Wall min–max s | Wall reduction | Prefill interval median ms | Interval min–max ms |
|---|---|---:|---:|---:|---:|---:|
| Eager | Native | 1.906998 | 1.904944–1.911188 | — | 1875.026978 | 1869.207520–1880.136719 |
| Eager | GPU resident | 1.878844 | 1.874486–1.884480 | 1.48% | 1844.688843 | 1842.463745–1850.212402 |
| Eager | Pinned CPU | 1.886336 | 1.876464–1.889369 | 1.08% | 1848.074219 | 1845.724121–1853.761597 |
| Breakable graph | Native | 1.912007 | 1.908803–1.912981 | — | 1876.062256 | 1869.148193–1881.912354 |
| Breakable graph | GPU resident | 1.884406 | 1.877342–1.886097 | 1.44% | 1846.576172 | 1840.159790–1849.147339 |
| Breakable graph | Pinned CPU | 1.888775 | 1.880144–1.889128 | 1.22% | 1848.980469 | 1842.696167–1852.314575 |

Across all 106 requests, text, token IDs, returned-token logprobs and top-five
logprobs were checked directly and matched each other and the native-only eager
probe. All 53 graph requests recorded one successful actual prefill replay;
all 53 eager requests recorded zero. Strict GPU/pinned checks before and after
unprofiled timing verified exact query states and complete ordered/padded
selection indices at all eight layers. This proves dynamic replay hooks ran
inside the measured graph condition rather than silently using native selection.

The 52 requests preceding the final drain in each engine contributed 416
matched selector events each, 832 total. The graph log additionally contains
24 initialization/capture events outside those request records; these are not
timing samples. Profile requests contributed 21 forward intervals per engine,
42 total. The native-only 22-request probes above are a separate experiment.

Each retained GPU selection representation is 96 MiB and its pinned CPU
predecoded copy another 96 MiB; both coexist. Capture, strict CPU hashes and
representation preparation are excluded from hit timing. Pinned timing does
not include inflate and neither path implements compressed double buffering.
The selector-cache opportunity persists with this prefill graph configuration,
but its observed whole-request benefit remains around one to one-and-a-half
percent. Separate sequential engines do not support a randomized paired claim
about graph versus eager execution. These are prefill-focused one-token probes,
not decode-graph results, task-quality tests or server TTFT.

Evidence: [graph cache requests](../integrations/results/sala_prefill_cache_bcg16k_a100.json),
[selector events](../integrations/results/sala_prefill_cache_bcg16k_a100_events.jsonl),
[graph/forward events](../integrations/results/sala_prefill_cache_bcg16k_a100_prefill_events.jsonl),
[metadata](../integrations/results/sala_prefill_cache_bcg16k_a100_metadata.json);
[eager cache requests](../integrations/results/sala_prefill_cache_eager16k_a100.json),
[selector events](../integrations/results/sala_prefill_cache_eager16k_a100_events.jsonl),
[forward events](../integrations/results/sala_prefill_cache_eager16k_a100_prefill_events.jsonl),
and [metadata](../integrations/results/sala_prefill_cache_eager16k_a100_metadata.json).

Evidence: [graph requests](../integrations/results/sala_prefill_bcg16k_a100.json),
[graph events](../integrations/results/sala_prefill_bcg16k_a100_events.jsonl),
[graph metadata](../integrations/results/sala_prefill_bcg16k_a100_metadata.json),
[eager requests](../integrations/results/sala_prefill_eager16k_a100.json),
[eager events](../integrations/results/sala_prefill_eager16k_a100_events.jsonl),
and [eager metadata](../integrations/results/sala_prefill_eager16k_a100_metadata.json).
