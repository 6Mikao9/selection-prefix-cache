# Native DSA backend status

No DSA benchmark was obtained in this round. The A100 source preflight is
[retained here](../integrations/results/dsa_preflight_a100.json). It is source
inspection, not a successful kernel execution or model-quality result.

The inspected SGLang 0.5.20 DSA ragged-prefill implementation uses native
DeepGEMM FP8 MQA logits on CUDA. In `dsa_indexer.py`, HIP/XPU branches occur
at lines 1151/1168, followed by `deep_gemm.fp8_mqa_logits` at line 1181;
the chunked CUDA branch calls the same primitive at line 1246. This path has
no BF16 score fallback for Ampere. Its source SHA256 is
`2f9634d275bde01c6eaffa0fabc339d592d2de67fa397b2e97af673f78ffbbfc`.

The installed native implementations inspected cover SM90, SM100 and SM120;
no SM80 MQA score implementation was found. The SM90 source includes Hopper
cluster and MMA descriptors at lines 6 and 8. Its SHA256 is
`d04c888c33951179e211c472e9bf575cd47efa9ee6a0d39423decff2e108f6d3`.
Consequently this installed native score path is unavailable on A100/SM80.
An executable top-k alone would not validate the complete score-to-top-k chain.

The initial report's `deep-gemm: not installed` field was a distribution-name
lookup error, contradicted by the source-present records. It is not evidence
that DeepGEMM was absent. The preflight now resolves the `deep_gemm` import's
actual distribution names using package metadata and reports unknown metadata
when no mapping exists; it never infers absence from a guessed distribution name.

The alternate RTX 5090 environment still reported an NVML error for one GPU.
No GPU task was started there after that incomplete health check. This does
not establish a hardware root cause.

Next, independently confirm a healthy idle supported card, bind it explicitly,
and run the preflight with `--check-device`. Capability >=9 is only a necessary
gate; actual SM120 support depends on the installed implementation. A subsequent
native microbenchmark must execute the real quantization/FP8 score/top-k path,
keep unprofiled and CUDA-event-profiled trials separate, and verify that replay
preserves the captured order and padding. Assess native tie-breaking separately.
No PyTorch score substitute or tiny-checkpoint quality claim
is presented here.
