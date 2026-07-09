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

The committed `dolphin-cn-dialect-small-fp16` baseline is the **CPU** default (the
golden, parity-validated path). CPU-vs-Metal x with/without cross-request weight
reuse was AB-measured on the 2.38 s Sichuan clip (M1, best-of-5); all four configs
reproduce the golden transcript exactly (CER 0.0000):

| Backend | Weight reuse | RTF | Peak RSS |
| --- | --- | ---: | ---: |
| CPU | cold (reload/dequant each request) | 0.34 | ~1.88 GB |
| CPU | warm (pooled weights) | **0.29** | ~1.88 GB |
| Metal | cold | 0.32 | ~1.78 GB |
| Metal | warm (pooled weights) | **0.29** | ~1.78 GB |

**Re-measured post-#P5/#P6** (attention-rescoring build-once/run-many decoder
weights + gated encoder taps; release, best-of-5, same clip/host as above). The
previous rows in this table (CPU 0.89/0.72, Metal 0.67/0.48, ~3.4-3.7 GB) predate
both fixes: every one of the CTC n-best's `DOLPHIN_BEAM_SIZE=10` rescoring calls
used to rebuild the whole decoder graph and re-upload all ~200 decoder weight
tensors from scratch, and the encoder unconditionally materialized every
per-block hidden state as an extra f32 graph output. Removing that per-hypothesis
rebuild/re-upload (P5) plus gating the encoder's per-block taps off in production
(P6) cuts RTF by roughly half-to-2/3 and peak RSS by close to 2x across every
cell, dominating the older cross-request `DOLPHIN_WEIGHTS_POOL` reuse effect below.

Findings, both measured (not assumed):

1. **Reuse still helps, but far less than before.** The executor still pools the
   dequantized f32 weights per pack (`DOLPHIN_WEIGHTS_POOL`) across *requests*,
   so cold vs warm still shows a small gap (0.34 -> 0.29 CPU, 0.32 -> 0.29 Metal).
   That gap used to be much larger because the old per-hypothesis decoder rebuild
   dwarfed the one-time pack-load cost it was hiding; now that the rescore loop
   itself is build-once/run-many, cross-request reuse is a minor tail rather than
   the dominant lever.
2. **CPU and Metal are now close, not a clear Metal win.** The previous "Metal
   WINS here" conclusion was measured when the decoder rebuilt its whole graph
   (weights included) 10x per utterance -- wide enough per rebuild to amortize
   GPU dispatch. With the rebuild/re-upload gone, the two backends land within
   noise of each other on this clip (warm RTF 0.29 both). Re-validate on a longer
   clip before leaning on a backend recommendation from this table alone.

**Default = CPU** anyway: the parity gate is CPU bit-exact and Metal's fp16
numerics are not golden-validated (identical transcript on this clip is evidence,
not a guarantee across GPUs/audio). Metal remains an **opt-in** via
`--execution-target accelerated` / `OPENASR_GGML_BACKEND=metal`; the executor
fail-closes to CPU on the Auto default and engages Metal only on an explicit
accelerated request. Harness: the `dolphin_perf_ab` ignored test
(`OPENASR_DOLPHIN_AB_BACKEND`/`_REUSE`/`_RUNS`).

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
