//! ReDimNet2-B6 speaker embedder (192-d), ggml-graph backend.
//!
//! This is the Chinese-enhanced ReDimNet2-B6 embedder from PalabraAI/redimnet2
//! (MIT). Unlike the legacy pure-Rust WeSpeaker ResNet34 (`super::wespeaker`),
//! ReDimNet2 executes through a **ggml graph** (ggml-only invariant) fed from a
//! `.oasr` GGUF pack produced by `tooling/redimnet2/convert_redimnet2.py`.
//!
//! Bring-up status (staged, each step golden-pinned before the next):
//!   * [x] Front end (`frontend`): TFMelBanks port, parity vs `frontend_dump/`.
//!   * [x] Structural constants (`config`): per-stage dims from the checkpoint.
//!   * [x] Backbone ggml graph (`backbone`): stem -> 6 stages -> `fin_wght1d`
//!     -> head -> ASTP pool -> BN -> linear -> 192-d embedding. End-to-end
//!     cosine vs the golden embeddings is > 0.9999 for all three fixture
//!     samples (`#[ignore]`d parity tests in `backbone::tests`, gated on the
//!     local `redimnet2-spike` assets).
//!   * [ ] `SpeakerEmbedder` impl + runtime pack resolution + calibration.
//!
//! The `SpeakerEmbedder` trait is deliberately **not** implemented yet and no
//! runtime resolution path selects this embedder: wiring a half-built forward
//! pass into the fail-closed diarize dispatch would risk fabricating embeddings.
//! WeSpeaker stays the sole runtime embedder until this lands. See
//! `docs/design/redimnet2-b6-embedder.md` (backbone plan + golden anchors) and
//! `HANDOFF.md` (remaining plan: `SpeakerEmbedder`, calibration, catalog entry).

// The backbone graph is exercised by its own `#[cfg(test)]` parity harness,
// not yet by any runtime dispatch path (no `SpeakerEmbedder` impl exists
// yet), so its public surface is otherwise dead in a plain lib build.
#![allow(dead_code)]

pub(crate) mod backbone;
pub(crate) mod config;
pub(crate) mod frontend;
mod ops;
