# ReDimNet2-B6 embedder -- stage-3 handoff

Branch `feat/redimnet2-b6-embedder`. Stage-1/2 assets (specs, golden `.npy`,
weights) live under `/Volumes/QuintinDocument/openasr-dev/tmp/redimnet2-spike/`.
This handoff records exactly what is done and the precise remaining plan.
**Delete this file before the PR is marked ready** (it is a working note).

## Done (backbone, this session -- each step golden-pinned, all green)

- **Full backbone ggml graph** (`crates/openasr-core/src/diarize/embed/redimnet/
  backbone.rs` + `ops.rs`): `stem -> stage0..5 -> fin_wght1d -> head -> fin_to2d
  -> ASTP pool -> BatchNorm -> linear -> 192-d embedding`, built on the shared
  `GgmlCpuGraphRunner`/`GgmlStaticTensorArena` (same pattern as
  `models::dolphin::encoder_graph`). Loads the f32 `.oasr` pack via
  `diarize::embed::weights::Weights::from_oasr`; weight tensor names read
  verbatim from the pack (`backbone.*`/`pool.*`/`bn.*`/`linear.*`), with the
  down-conv's `groups` derived by introspecting the pack's own kernel shape
  (`cin_running / cin_per_group`) rather than recomputing `gcd` by hand.
- **Parity tests** (`backbone::tests`, all `#[ignore]`, gated on the local
  `redimnet2-spike` assets, **all green**):
  - `stem_parity_jfk` -- `01_outputs_1d` (stem: conv -> LN -> to1d -> gnorm).
  - `stage_parity_jfk` -- `02..07_outputs_1d` (stage0..5), each depending on
    every previously-verified stage via `weigth1d` aggregation.
  - `fin_and_head_parity_jfk` -- `fin_wght1d`, `99_backbone_2d_output`,
    `a0_pre_pool_flattened`.
  - `full_pipeline_cosine_gate` -- `a1_post_pool`/`a2_post_bn`/
    `a3_final_embedding`, plus the end-to-end cosine gate: **jfk/zh_sample/
    en_zh_mixed all show cosine 1.000000 vs the golden embeddings** (well
    above the > 0.9999 target).
  - `to1d_matches_hand_derived_frequency_major_formula` -- a synthetic
    (no-pack-needed) unit test pinning `to1d`'s `ne` derivation directly.
- Verification run: `cargo fmt --check`, `cargo clippy -p openasr-core --lib
  --tests -- -D warnings` (clean), `cargo test -p openasr-core --lib diarize::`
  (148 passed, 0 failed, no regressions in the rest of `diarize::`).

### Two real bugs found and fixed via the parity harness (read before touching `ops.rs`)

1. **Gallocr view-of-output-tensor corruption.** `to1d`/`to2d`/`group_norm_1d`
   (and any future op) must defensively `cont()` a tensor *before* taking any
   view of it (`permute`/`transpose`/`reshape` are all views in ggml) if that
   tensor might *also* be independently marked `set_output` (a parity tap, or
   any other long-lived read). The backend scheduler's gallocr does **not**
   protect a view's underlying source buffer purely because a view of it (or
   the tensor itself) carries the output flag -- it silently recycled the
   buffer once the view had been consumed, corrupting every later read of the
   supposedly-still-needed tensor. Root-caused by bisection: `run_stem` alone
   matched the golden bit-for-bit, but the identical computation, once more
   graph was built *after* it in the same call (even just `+stage0`), read
   back corrupted. Every 2D<->1D boundary op in `ops.rs` now `cont()`s its
   input defensively; keep doing this for any new op that takes a view of a
   caller-supplied tensor.
2. **`to1d` is not the final pre-pool flatten.** `ReDimNet2Wrap.forward`'s
   `out.reshape(bs, C*F, T)` (right before `pool`) is a **plain torch
   reshape** (`c*F+f`, C-major/F-minor), not `to1d`'s frequency-major merge
   (`f*C+c`) used everywhere else in the backbone. Reusing `to1d` there
   silently flattened in the wrong order (values fine, order wrong) -- fixed
   by adding a dedicated `ops::flatten_backbone_output` for exactly this one
   call site. If a new "flatten a 2D backbone output to 1D" call site shows up,
   check which torch op it actually mirrors before assuming it's `to1d`.

## Remaining plan (post-backbone; not started)

1. **`SpeakerEmbedder` impl** on a `RedimNet2Embedder` (`embedding_dim` 192),
   wiring `backbone::forward` behind the trait. L2-normalize the final
   embedding (not yet done in `backbone::forward` -- the golden comparison
   used raw pre-normalize vectors since cosine is scale-invariant; confirm
   whether the production trait needs an explicit normalize step or whether
   callers normalize downstream, matching `WeSpeaker`'s convention).
2. **`REDIMNET_CALIBRATION` profile** (192-dim cosine space, distinct from
   `WESPEAKER_CALIBRATION` -- must be measured against real enrollment data,
   not copied).
3. **Runtime pack resolution** (`OPENASR_REDIMNET_PACK` env + installed-dir
   hint), mirroring `pack.rs`'s existing WeSpeaker resolution path.
4. **Catalog/registry entry** + model card + `docs/model-audits/
   redimnet2-b6.md` (new-family release gate; `tooling/publish-model/scripts/
   _manifest.py --public` fails closed without a completed form).
5. **Shipping pack + quantization**: convert the final release pack at the
   chosen quant (f16/q8_0), re-run the parity gate against it (the f32 pack
   was the only fixture used so far), then benchmark RTF/RAM (separate,
   measurement-isolated session).
6. Not in scope for any of the above: `SpeakerEmbedder`/dispatch wiring must
   not touch `WeSpeaker` (untouched, remains the sole runtime embedder until
   this whole plan lands and is explicitly approved for cutover).

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
