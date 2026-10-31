# QSA selector representation microbenchmark on A100

Caching block IDs rather than expanded token indices reduced hit-path time in
this native-kernel microbenchmark. This supports investigating small caches for
modest prefill benefits; it does not establish whole-model latency improvement.
The separate [SALA model diagnostic](selector_diagnostic_results.md) measures a
different model and boundary.

The successful run used random Q/K, SGLang 0.5.20 native QSA score/top-k/expand
kernels, four indexer heads of dimension 128, 512 selected blocks and expansion
ratio four. Configuration provenance and kernel hashes are retained in the
public artifact. A100-SXM4-80GB ran PyTorch 2.13.0 with CUDA runtime 13.0; the
process used NVCC 12.8 for JIT. No installation change is implied.

There are 168 records: at each of four lengths, five unprofiled modes each
have seven trials, plus seven separately profiled native trials. The following
values are median outer CUDA-event milliseconds from unprofiled trials:

| Query tokens | Native | Expanded int32 H2D | Block int32 H2D + expand | Block uint16 H2D + decode + expand | GPU resident |
|---|---:|---:|---:|---:|---:|
| 2048 | 0.297376 | 0.686816 | 0.209920 | 0.145760 | 0.012160 |
| 8192 | 0.933056 | 2.675488 | 0.760448 | 0.480736 | 0.013472 |
| 16384 | 2.605184 | 5.313760 | 1.496672 | 0.924992 | 0.011712 |
| 32768 | 8.111872 | 10.577440 | 2.899104 | 1.798848 | 0.011840 |

At 32K, the expanded representation has 2051 int32 entries per query and
occupies 256.375 MiB; 512 int32 block IDs occupy 64 MiB and uint16 block IDs
occupy 32 MiB. The uint16 path saves about 6.313 ms against native in this
selector-only boundary. It is dtype narrowing, not entropy compression.
The sentinel 65535 represents padding; this mode is disabled when the number
of keys exceeds 65535. Cache capture and representation preparation precede
hit timing. All CPU/GPU representations coexist, so the experiment does not
demonstrate deployment memory savings or bounded cache capacity.

The eightfold reduction above is relative to expanded token IDs, not main KV.
Using the [checkpoint configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json),
BF16 main KV at 32K is `32768 * 2(K/V) * 2(KV heads) * 256 * 2 bytes`
= 64 MiB per QSA layer. Thus 32 MiB of uint16 block IDs is still half that
baseline. This is configuration arithmetic, not a measured model allocation;
headers, indexer keys and recurrent state are excluded. Further trace-based
compression and selective retention remain necessary for a much smaller cache.

All replay modes reproduced the captured expanded indices byte for byte,
including original order and padding. Block IDs were captured directly from
native top-k, not inferred from expanded output: packed trailing entries make
that inverse reconstruction invalid. Repeated native calls can change atomic
output ordering and selected sets at exact boundary ties. The check separately
required legal cardinality/ranges, unchanged packed tail, scores at least the
kth threshold, and every exchanged block's score exactly equal to that
threshold. All checks passed; at most 140 query rows per record differed by
boundary ties. This is not approximate equality or model-quality equivalence.

There is no attention consumer, model activation trace, complete prefill,
decode measurement, graph-enabled model result, or transfer/compute overlap.
Wall milliseconds and individual trials remain available in the raw artifact;
CUDA milliseconds above are not server TTFT. Future prefill experiments should
compare retained bytes against milliseconds saved in the same model, while
accounting for capture/amortization and graph effects on total request time.

Earlier attempts with incompatible compiler/header versions, overly strict
native ordering/tie checks, or invalid inverse block reconstruction are failed
attempts, excluded from these 168 successful records.

Evidence: [complete successful run](../integrations/results/qsa_selector_microbench_a100.json)
and [metadata](../integrations/results/qsa_selector_microbench_a100_metadata.json).
