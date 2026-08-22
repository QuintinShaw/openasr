# qwen GPU parity gate

A correctness gate for the qwen3-asr decoder on discrete GPUs.

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
`crates/openasr-core/src/models/qwen/llm_transformer.rs`). This gate is what
proves a given GPU is safe to re-enable and catches any future regression of the
GPU decode path.

> A synthetic in-process numeric self-check was tried and **rejected**: a probe
> that exercises one op/shape can *false-pass* when the real decoder mis-computes
> a different op (e.g. the masked prefill `mul_mat` broadcast vs. an unmasked
> single-query flash). Only an end-to-end transcript comparison is complete by
> construction, so correctness is gated here, not by a runtime probe.

## What it does

For each configured audio path the producer compares CPU reference output with
an explicitly selected GPU provider/device and requires cold and same-process
reuse per-step traces. Missing GPU, fixture, pack, candidate identity, or trace
is a hard failure; there is no CPU-only success path.

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

## CI

The workflow requires immutable candidate CLI, staging catalog, exact `.oasr`
pack, correctness matrix/receipt, expected provider/device, and runtime-produced
cold/reuse per-step trace artifacts. It fails closed when any is absent,
unstable, mismatched, or when the host selects CPU/another device. A transcript
comparison without those identity-bound traces is not a passing run.
