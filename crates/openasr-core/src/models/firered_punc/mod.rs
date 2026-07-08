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

/// Pull-time contract for FireRedPunc punctuation packs.
///
/// Returns `Some` only when the pack declares the `firered-punc` GGUF
/// architecture; ASR/translation packs fall through to their own adapters. The
/// check is the cheap metadata-only geometry validation (no weight load), so
/// `openasr pull` stays fail-closed for punctuation packs without paying a full
/// model build.
pub(crate) fn validate_punctuation_runtime_pack_contract(
    _path: &std::path::Path,
    metadata: &crate::GgufMetadata,
) -> Option<Result<(), String>> {
    if !runtime_contract::metadata_declares_firered_punc(metadata) {
        return None;
    }
    Some(
        runtime_contract::parse_and_validate_firered_punc_metadata(metadata)
            .map(|_| ())
            .map_err(|error| error.to_string()),
    )
}
