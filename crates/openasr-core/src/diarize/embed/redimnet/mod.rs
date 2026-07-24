//! ReDimNet2-B6 speaker embedder (192-d), ggml-graph backend.
//!
//! This is the Chinese-enhanced ReDimNet2-B6 embedder from PalabraAI/redimnet2
//! (MIT). It is the **only** supported speaker-embedding path in OpenASR:
//! inference runs through a **ggml graph** (ggml-only invariant) fed from a
//! `.oasr` GGUF pack produced by `tooling/redimnet2/convert_redimnet2.py`. When
//! the pack is missing, diarization and Voice ID fail closed rather than
//! falling back to any other embedder.
//!
//! Bring-up status (staged, each step golden-pinned before the next):
//!   * [x] Front end (`frontend`): TFMelBanks port, parity vs `frontend_dump/`.
//!   * [x] Structural constants (`config`): per-stage dims from the checkpoint.
//!   * [x] Backbone ggml graph (`backbone`): stem -> 6 stages -> `fin_wght1d`
//!     -> head -> ASTP pool -> BN -> linear -> 192-d embedding. End-to-end
//!     cosine vs the golden embeddings is > 0.9999 for all three fixture
//!     samples (`#[ignore]`d parity tests in `backbone::tests`, gated on the
//!     local `redimnet2-spike` assets).
//!   * [x] `SpeakerEmbedder` impl (`backbone::RedimNet2Model` +
//!     `super::RedimNet2Embedder`), runtime pack resolution
//!     (`OPENASR_REDIMNET_PACK` / installed-dir hint, `super::pack`), and a
//!     dedicated calibration profile (`REDIMNET_CALIBRATION`).
//!
//! See `docs/design/redimnet2-b6-embedder.md` for the backbone plan and golden
//! anchors.

// A handful of `config`/`frontend` items (e.g. `StageConfig::{sf,st}`,
// `RedimNetFrontend::n_mels`) are checkpoint-structural fields kept for
// parity-table readability and cross-checking against the checkpoint (see
// `docs/design/redimnet2-b6-embedder.md` "per-stage hard-coded dims") even
// though the current graph builder derives its shapes from the pack's own
// tensor shapes instead of reading them back. `backbone` itself has no
// remaining dead code now that `RedimNet2Model` wires the full forward path.
#![allow(dead_code)]

pub(crate) mod backbone;
pub(crate) mod config;
pub(crate) mod frontend;
mod ops;
