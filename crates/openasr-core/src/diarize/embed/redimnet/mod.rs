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
//!   * [ ] Backbone ggml graph (stem -> 6 stages -> fin_wght1d -> head).
//!   * [ ] ASTP pool -> BN -> linear -> 192-d embedding.
//!   * [ ] `SpeakerEmbedder` impl + runtime pack resolution + calibration.
//!
//! The `SpeakerEmbedder` trait is deliberately **not** implemented yet and no
//! runtime resolution path selects this embedder: wiring a half-built forward
//! pass into the fail-closed diarize dispatch would risk fabricating embeddings.
//! WeSpeaker stays the sole runtime embedder until the backbone is golden-clean.
//! See `docs/design/redimnet2-b6-embedder.md` (backbone plan + golden anchors).

// Staged bring-up: the front end and structural constants are complete and
// golden-pinned by their own tests, but nothing in the runtime consumes them
// yet (the backbone that would is not built). Allow dead_code for this module
// tree until the `SpeakerEmbedder` impl lands and wires them in.
#![allow(dead_code)]

pub(crate) mod config;
pub(crate) mod frontend;
