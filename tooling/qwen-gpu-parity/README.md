# Qwen GPU parity diagnostic

A raw troubleshooting diagnostic for the qwen3-asr decoder on discrete GPUs.

> This tool is not a release gate. It consumes runner-local activation state
> and caller-provided labels, and its `*.diagnostic.*` output is not accepted by
> `gpu_correctness_gate.py`. Only the exact target/backend-bound common matrix
> and strict receipts can qualify an artifact.

## Why

qwen3-asr decode is GPU-kernel sensitive. The fused grouped-query-attention
(GQA) broadcast (`use_native_gqa`) is mis-computed by the ROCm/HIP flash kernel
on AMD RDNA4 / gfx1200: recognition degenerates into garbled, repeated tokens
(`languagelanguagele…`) on the GPU while the CPU output is correct. That class
of bug is invisible to the normal CI, which runs on Linux/ARM with no discrete
GPU, so it shipped unnoticed.

The runtime guard for this is a conservative default: native GQA is **off** on
the discrete-GPU lane and **on** for CPU/Metal (see
`qwen_llm_native_gqa_default_for_backend` in
`crates/openasr-core/src/models/qwen/llm_transformer.rs`). This diagnostic helps
reproduce that failure class; it cannot prove a target safe to activate. Only
the artifact-bound correctness
producer and final gate may prove a concrete execution cell safe to activate.

> A synthetic in-process numeric self-check was tried and **rejected**: a probe
> that exercises one op/shape can *false-pass* when the real decoder mis-computes
> a different op (e.g. the masked prefill `mul_mat` broadcast vs. an unmasked
> single-query flash). End-to-end transcript comparison is necessary, but it is
> not sufficient without runtime and artifact identity evidence.

## What it does

For each configured audio path the script compares CPU reference output with an
explicitly selected GPU provider/device. Its cold/reuse labels are separate
`openasr.exe` processes, and its `openasr.seq2seq-debug-trace.v1` header uses
caller-supplied provider/device labels. Missing GPU, fixture, pack, or debug
trace is a hard diagnostic failure; there is no CPU-only success path.

Consequently, this script does not prove same-process graph reuse, native
provider placement, runtime graph lifecycle, complete logits, or artifact
identity. Do not pass its JSONL files to `gpu_correctness_gate.py`.

## Run it locally

```pwsh
# on a gfx1200 / CUDA / Vulkan box, from the repo root
cargo build -p openasr-cli --release --features hip   # or --features cuda / vulkan
pwsh tooling/qwen-gpu-parity/run.ps1
```

Overrides (env):

| var | default |
|---|---|
| `OPENASR_QWEN_PARITY_EXE` | `target/release/openasr.exe` |
| `OPENASR_QWEN_PARITY_PACK` | resolved from `OPENASR_HOME/models/<id>/<quant>/<id>-<quant>.oasr` |
| `OPENASR_QWEN_PARITY_MODEL` | `qwen3-asr-0.6b` |
| `OPENASR_QWEN_PARITY_QUANT` | `q8_0` |
| `OPENASR_QWEN_PARITY_AUDIO` | `;`-separated audio paths |
| `OPENASR_QWEN_PARITY_EXPECTED_PROVIDER` | required operator assertion |
| `OPENASR_QWEN_PARITY_EXPECTED_DEVICE` | required operator assertion |
| `OPENASR_QWEN_PARITY_TRACE_DIR` | required output directory |

## Release evidence boundary

The workflow attests one CLI, binds one exact `.oasr` input, and records cold and
reuse output from the provider already active on the self-hosted runner. It
rejects missing or ambiguous files and CPU/other-device selection. The output is
intentionally diagnostic and is never consumed by release finalization.

Release qualification requires immutable candidate artifacts, an exact verified
`.oasr` pack and fixture, runtime-observed provider/device placement, same-
process cold/reuse requests, complete logits, graph lifecycle events, and
`openasr.short-audio-receipt.evidence.v1` bound to the projected matrix. This
diagnostic intentionally supplies none of those release-authorizing claims.
