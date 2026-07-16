# OpenASR — Performance

The committed performance "ruler" + regression guardrail: a fixed audio clip at
a fixed quantization, one machine-readable baseline per host profile, and a
one-command gate that fails closed on regression.

- **Suite config:** [`perf/suite.toml`](suite.toml)
- **Baselines (authoritative numbers):** [`perf/baselines/`](baselines/) — one
  JSON per host profile (e.g. `macos-aarch64.json`). The committed JSON is the
  source of truth for every measured value; this doc does not duplicate it.
- **Harness code:** `crates/openasr-core/src/{metrics,benchmark/suite}.rs` +
  `crates/openasr-cli/src/bench_suite_cli.rs`

## Running

```bash
# Gate the current build against the committed baseline (exits non-zero on regression):
cargo run --release -p openasr-cli -- bench-suite

# Re-baseline on the reference host after an intended perf change:
cargo run --release -p openasr-cli -- bench-suite --write-baseline perf/baselines/<host>.json

# One family / JSON output:
cargo run --release -p openasr-cli -- bench-suite --family whisper --format json
```

The harness drives the **real** transcription call path (`transcribe_with_backend`
→ `NativeBackend`), the same one `transcribe --benchmark` and `bench-suite` use — it measures
the production runtime, not a re-implementation. Each entry runs in a fresh
subprocess (`--run-single-entry`), so its peak RSS (a process high-water mark)
is uncontaminated by earlier entries.

## What is gated

| Metric | Type | Gates? | Default tolerance | Why |
| --- | --- | --- | --- | --- |
| **WER** | absolute | ✓ primary | +0.02 | Deterministic for a fixed model+audio — the reliable correctness guard. |
| **RTF** | relative | ✓ | +25% | Best-of-N wall clock (`--runs`, default 3) — keeps the fastest sample so background load doesn't gate. |
| **Peak RSS** | relative | ✓ (gating entries) | +20% | Gated on the stable gating entries, locking the zero-copy + gallocr-reuse memory wins. `compare_to_baseline` skips `gating = false`, so noisy fp16 entries are reported but not RSS-gated. |
| **Quant ordering** | ordinal | ✓ | +25% slack | Entries sharing an `ordering_group` (same model, different quant) must be RTF-ordered q4_k ≤ q8_0 ≤ fp16 — more compression must not be slower. |
| **vs whisper.cpp** | ratio | ✓ | +5% slack | Same-model openasr-vs-whisper.cpp best-of-N wall (goal 3, "beat comparable OSS"). Matched-thread (`-t 4`). The gate fires only if openasr regresses to slower than whisper.cpp beyond `cpp_slack`. |

A gating entry that regresses beyond tolerance — or is missing on either side —
fails the command. Non-gating entries (`gating = false`) are measured and shown
but never fail the gate. A quant-`ordering_group` is always checked regardless of
each member's `gating` flag.

The two gating entries in the committed baseline are `whisper-tiny-en-q8` and
`cohere-transcribe-q8`. The vs-whisper.cpp entry is `whisper-turbo-q8-vs-cpp`
(records both openasr and whisper.cpp best-of-N wall for the same turbo q8 pack).

## Fixed conditions

- **Clip:** a fixed LibriSpeech test-clean clip
  (`237-134500-0000.wav`, reference
  `"FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS DIVORCE CASE THE ANGRIER HE GREW"`).
  The same clip is used across families so RTF/memory are directly comparable.
  See `perf/suite.toml` for the per-entry `audio_path`/`reference`.
- **Families covered by the baseline:** whisper, cohere, qwen, parakeet-ctc,
  wav2vec2-ctc (incl. data2vec/hubert variants), moonshine, dolphin. The dolphin
  entry uses its own Chinese-dialect clip (`clip_sichuan.wav`, reference the
  model's golden `attention_rescoring` output `学校和底下好多那种野生枸杞`,
  CER 0.0000), not the shared English clip.
- **Quant per entry:** fp16 / q8_0 / q4_k (qwen also q3_k). The gating entries are
  at q8_0; other quants ride along as non-gating + ordering-group members.

## Dolphin: CPU vs Metal (AB-measured on M1)

Dolphin's Auto default is **Metal** on a GPU-capable Apple Silicon host (fail-closed
to CPU where no accelerator exists). This became the default once the
E-Branchformer encoder + CTC head weights moved into a WEIGHTS-usage static arena:
before that, the ggml scheduler only offloads an op whose weight `src` lives in a
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` buffer, so the whole ~1348-op encoder plus the
CTC projection stayed pinned to the CPU under an explicit Metal backend while only
the small decoder rescoring graphs ran on Metal -- a net GPU loss. With the weights
arena-resident, `GGML_SCHED_DEBUG=2` shows zero CPU compute splits: the encoder, the
CTC head, and all 10 decoder rescoring graphs run on Metal.

Warm best-of-6 compute, isolated host (M1), q8_0, identical golden transcript on
every cell:

| Audio | Backend | RTF (before arena) | RTF (after arena) |
| --- | --- | ---: | ---: |
| 3 s  | CPU   | 0.180 | 0.174 |
| 3 s  | Metal | 0.244 | **0.126** |
| 18 s | CPU   | 0.171 | 0.157 |
| 18 s | Metal | 0.334 | **0.081** |

Findings, all measured (not assumed):

1. **The arena reverses the GPU deficit.** Before, Metal lost to CPU (1.35x slower
   at 3 s, 2.07x at 18 s). After, Metal beats CPU on both lengths (1.38x faster at
   3 s, 1.27x at 18 s) and Metal's own compute drops 1.9x/2.6x. CPU is unchanged --
   weight placement does not alter the CPU path (`dolphin_encoder_parity` stays
   bit-exact).
2. **Margin is honest, not huge.** On a clean host the M1 CPU is strong, so the GPU
   win is ~1.3x rather than a landslide; it is consistently positive across both
   lengths. Peak RSS is also lower on Metal (quantized weights stay quantized in the
   WEIGHTS buffer rather than a dequantized graph-input blow-up).

**Default = Metal on Apple Silicon**, CPU elsewhere: the Auto gate
(`auto_gpu_enabled = true`) only ever selects an accelerator that is actually
present (`runtime_gpu_is_available`), and an explicit `--execution-target cpu`
always wins. CPU stays the bit-exact parity reference -- Metal fp16 reproduces the
golden transcript on the parity clip but is not itself the golden gate. Harness: the
`dolphin_perf_ab` ignored test (`OPENASR_DOLPHIN_AB_BACKEND`/`_REUSE`/`_RUNS`).

## Caveats

- **Peak RSS is a process high-water mark** — isolated per entry via the
  per-entry subprocess. With isolation in place, `gate_peak_rss = true` gates the
  stable gating entries. `gating = false` entries (incl. noisy fp16 packs) are
  skipped by `compare_to_baseline`, so they are reported but never RSS-gated.
- **Packs + audio live under `tmp/`** (gitignored, host-local), so `suite.toml`
  paths are machine-specific. The committed artifacts are the *schema*
  (`suite.toml`) and the *numbers* (`baselines/*.json`), not the model weights.
- **Single-run RTF** is noisy on short clips; the +25% default is intentional.
  Prefer a longer clip or multi-run medians before tightening.
- The vs-whisper.cpp gate establishes a same-model, matched-thread comparison
  harness with headroom; treat it as a measured comparison + regression guard,
  not a standing claim of having finally beaten OSS across the board.
