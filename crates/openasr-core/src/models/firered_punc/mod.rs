//! FireRedPunc (`FireRedTeam/FireRedPunc`) punctuation-restoration model family.
//!
//! A BERT-style bidirectional encoder (initialised from `chinese-lert-base`)
//! with a token-classification head that predicts, for each subword, which of
//! five Chinese punctuation classes follows it (`<none>`, `，`, `。`, `？`,
//! `！`). It is a **text-in / labels-out post-processor**, not an ASR model:
//! no audio frontend, no autoregressive decode. It exists to punctuate the
//! output of unpunctuated ASR families (e.g. dolphin), gated by the catalog
//! `emits_punctuation` capability. Apache-2.0.
//!
//! The released label space is Chinese-only, so the integration is Chinese-only
//! by construction; the architecture cannot emit English half-width marks.
//!
//! Stage status:
//! - [`config`] / [`tensor_names`] / [`tokenizer`] define the pack geometry,
//!   GGUF tensor layout, and the offset-preserving WordPiece encoder.

// Later integration stages (runtime, package import, pull-contract dispatch,
// and the punctuation post-processing stage) consume the geometry, tensor
// names, label table, and tokenizer defined here; allow the not-yet-wired
// surface until those stages land.
#![allow(dead_code)]

pub(crate) mod config;
pub(crate) mod graph;
pub(crate) mod runtime;
pub(crate) mod runtime_contract;
pub(crate) mod tensor_names;
pub(crate) mod tokenizer;
pub(crate) mod weights;

// Pull-time contract validation for FireRedPunc punctuation packs
// (`general.architecture = "firered-punc"`) is dispatched through
// `crate::models::aux_pack_registry`, alongside the other auxiliary (non-ASR)
// families (diarization, translation) -- one table instead of a per-family
// function called from an ad hoc chain in `api::backend::native`. The contract
// itself is still the cheap metadata-only geometry validation in
// [`runtime_contract::parse_and_validate_firered_punc_metadata`] (no weight
// load).
