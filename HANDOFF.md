# ReDimNet2-B6 embedder -- stage-4 handoff

Branch `feat/redimnet2-b6-embedder`. Stage-1/2 assets (specs, golden `.npy`,
weights) live under `/Volumes/QuintinDocument/openasr-dev/tmp/redimnet2-spike/`.
This handoff records exactly what is done and the precise remaining plan.
**Delete this file before the PR is marked ready** (it is a working note).

## Done (backbone, stage-3 session -- each step golden-pinned, all green)

See the stage-3 section of this file's prior revision (git history on this
branch) for the full backbone bring-up narrative and the two real ggml bugs
found via the parity harness (gallocr view-of-output corruption;
`to1d` vs the plain-reshape pre-pool flatten). Summary: full backbone graph
(`redimnet/backbone.rs` + `ops.rs`) reproduces the golden embeddings at
cosine 1.000000 for jfk/zh_sample/en_zh_mixed, all parity tests green.

## Done (SpeakerEmbedder wiring, this session -- all green)

- **`RedimNet2Model`** (`redimnet/backbone.rs`): owns the parsed `.oasr`
  `Weights` across calls; `forward(feats, frames)` rebuilds a fresh
  `GgmlCpuGraphRunner`/arena/graph per call (same shape as every parity
  test's `run_forward`) and returns the raw (pre-L2-normalize) 192-d
  embedding. Caching the arena/graph across calls is a later perf pass
  (plan item 5 below), not attempted here.
- **`RedimNet2Embedder`** (`embed/mod.rs`): implements `SpeakerEmbedder`
  (same trait as `WeSpeakerEmbedder`) -- `TFMelBanks` front end -> backbone
  -> `SpeakerEmbedding::l2_normalized`. `embedding_dim() == 192`.
  `embedding_space_version()` returns the pinned label
  `"redimnet2-b6-cn-v1"` (`pack::REDIMNET_EMBEDDING_SPACE_VERSION`) --
  documentation/audit metadata only; the real compatibility gate stays the
  pack content fingerprint (sha256) via `SpeakerEmbedderIdentity`.
- **`REDIMNET_CALIBRATION`** (`calibration.rs`): a distinct 192-dim cosine
  calibration profile, every threshold marked `TODO(voice-id-eval)` --
  conservative placeholders, not copied from `WESPEAKER_CALIBRATION`, pending
  a real LibriSpeech/AISHELL-4 calibration pass (separate in-flight task).
- **Runtime selection** (`embed/pack.rs`): `choose_embedder_pack` is the one
  pure selection rule -- ReDimNet2 wins whenever `OPENASR_REDIMNET_PACK` /
  the installed `redimnet*` dir resolves, WeSpeaker is used only as a
  fallback, neither present resolves to `None` (fail-closed, no panic).
  `shared_embedder`/`shared_embedder_identity`/`embedder_pack_installed` are
  unchanged call sites for every existing consumer (`enrollment.rs`,
  `streaming.rs`, `vbx/mod.rs`, `native_transcribe.rs`) -- only the pack
  resolved underneath changed. WeSpeaker is untouched and still fully
  functional as the fallback; removing it is plan item 6 below, not done.
- **Tests, all green**:
  - `pack::tests` -- 4-case selection matrix (both present / redimnet only /
    wespeaker only / neither) against the pure `choose_embedder_pack`, plus
    the `REDIMNET_EMBEDDING_SPACE_VERSION` pin.
  - `calibration::tests::redimnet_calibration_profile_is_pinned_and_distinct_from_wespeaker`
    -- pins the placeholder values and asserts they are not copies of
    WeSpeaker's tuned thresholds.
  - `enrollment::tests::old_wespeaker_profile_is_incompatible_with_new_redimnet_embedder`
    -- pins that a legacy 256-dim WeSpeaker profile is rejected by a 192-dim
    ReDimNet2 identity, with a readable `compatibility_status` reason, and
    that `VoiceprintStore::compatible_profiles` drops it rather than risk a
    cross-embedding-space comparison.
  - `embed::tests::redimnet_embedder_matches_python_reference_e2e_jfk`
    (`#[ignore]`, needs the local `redimnet2-spike` pack) -- the first test
    to exercise the full `SpeakerEmbedder` trait path end to end (raw
    `fixtures/jfk.wav` -> front end -> backbone -> L2-normalize), not just
    the backbone with a pre-dumped front-end tensor. Cosine vs
    `embeddings_b6/jfk.npy` = 1.00000024 (clamped near 1).
- Verification run: `cargo fmt --check`, `cargo clippy -p openasr-core --lib
  --tests -- -D warnings` (clean), `cargo test -p openasr-core --lib
  diarize::` (155 passed, 0 failed, up from 148 -- no regressions), plus the
  `#[ignore]`d redimnet suite against the real f32 pack (6 passed: frontend
  parity, all backbone stage/full-pipeline parity, and the new e2e trait
  test).

## Remaining plan (post-wiring; not started)

1. **Catalog/registry entry** + model card + `docs/model-audits/
   redimnet2-b6.md` (new-family release gate; `tooling/publish-model/scripts/
   _manifest.py --public` fails closed without a completed form). Out of
   scope for this stage per the task brief ("不接 dispatch 之外的产品面
   (catalog/上架另案)").
2. **Real calibration**: replace every `TODO(voice-id-eval)` threshold in
   `REDIMNET_CALIBRATION` once the LibriSpeech/AISHELL-4 same-speaker/
   cross-speaker cosine distribution is measured (separate task in flight).
3. **Shipping pack + quantization**: convert the final release pack at the
   chosen quant (f16/q8_0), re-run the parity gate against it (the f32 pack
   was the only fixture used so far), then benchmark RTF/RAM (separate,
   measurement-isolated session) and revisit whether `RedimNet2Model::forward`
   should cache its arena/graph across calls instead of rebuilding per embed.
4. **Enrollment UX for the WeSpeaker -> ReDimNet2 cutover**: once a user's
   active embedder identity flips from WeSpeaker to ReDimNet2 (installing the
   new pack), their old voiceprints become incompatible by design (see the
   `is_compatible_with` test above) and need re-registration. No UI/CLI
   surface for "your voiceprints are stale, re-register" exists yet --
   `compatibility_status`'s reason string is the only current signal,
   surfaced wherever a caller already reads it.
5. **Remove WeSpeaker** (explicit product decision, not started): once
   ReDimNet2 has shipped and been validated in the field, WeSpeaker's pack,
   pure-Rust model, and `WESPEAKER_CALIBRATION` can be deleted and
   `choose_embedder_pack`'s fallback branch removed. Not attempted in this
   PR by design.

## Reference pointers (still accurate)

- Design doc: `docs/design/redimnet2-b6-embedder.md` (decisions + architecture
  reference + risks). Read before making any structural change here.
- Upstream reference: `tmp/redimnet2-spike/repo/redimnet2/` (`redimnet2.py`,
  `layers/{blocks,resblocks,attention,convnext,poolings,redim_structural,
  layernorm}.py`).
- Structural facts: `tmp/redimnet2-spike/B6_STRUCTURE_SPEC.md` (per-stage
  dims, B3-vs-B6 diffs).
- Golden anchors: `tmp/redimnet2-spike/stage_dump_b6_jfk/` (13 `.npy`),
  `tmp/redimnet2-spike/embeddings_b6/` (3 fixture-sample final embeddings),
  `tmp/redimnet2-spike/frontend_dump/` (front-end intermediate tensors, used
  as the backbone's `spec` input for all three samples in the parity tests).
- f32 pack fixture: `tmp/redimnet2-spike/redimnet2-b6-f32.oasr`.
