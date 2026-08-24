# qwen GPU parity diagnostic

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
reproduce that failure class; it cannot prove a target safe to activate.

> A synthetic in-process numeric self-check was tried and **rejected**: a probe
> that exercises one op/shape can *false-pass* when the real decoder mis-computes
> a different op (e.g. the masked prefill `mul_mat` broadcast vs. an unmasked
> single-query flash). End-to-end comparison is necessary, but release
> correctness is gated by the common matrix rather than this convenience script.

## What it does

For each configured audio path the script compares CPU reference output with
an explicitly selected GPU provider/device and requires cold and same-process
reuse per-step diagnostic traces. Missing GPU, fixture, pack, or trace is a hard
failure; there is no CPU-only success path.

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

## CI

The workflow attests one CLI, binds one exact `.oasr` input, and records cold and
reuse output from the provider already active on the self-hosted runner. It
rejects missing or ambiguous files and CPU/other-device selection. The output is
intentionally diagnostic and is never consumed by release finalization.
