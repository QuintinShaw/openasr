//! FireRedPunc runtime: load an `.oasr` pack and punctuate text.
//!
//! Ties the pack contract, WordPiece tokenizer, and BERT graph together and
//! implements [`crate::punctuation::PunctuationClassifier`], so the punctuation
//! post-processing stage can drive it exactly like the unit-test mock. The
//! graph needs `&mut` for a forward, so it sits behind a `RefCell` to keep the
//! classifier trait's `&self`.

use std::cell::RefCell;
use std::path::Path;

use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
use crate::punctuation::{
    PunctuationClassifier, PunctuationError, PunctuationRestoreConfig, restore_punctuation,
};

use super::config::{FireRedPuncExecutionMetadata, TOKENIZER_GGML_TOKENS_KEY};
use super::graph::{FireRedPuncGraph, FireRedPuncGraphError, argmax_labels_per_position};
use super::runtime_contract::parse_and_validate_firered_punc_metadata;
use super::tokenizer::{FireRedPuncTokenizer, is_cjk_char};
use super::weights::load_firered_punc_weights;

/// Whether `text` contains at least one Han ideograph (per the tokenizer's
/// BERT `is_cjk_char` definition -- single source of truth, do not fork it).
///
/// This is the per-segment language gate for [`FireRedPuncRuntime::punctuate`]:
/// the checkpoint's label space is five full-width Chinese marks and its
/// training data is Chinese-only, so a segment with no Han ideograph is
/// outside its training domain. Skipping such segments keeps the stage from
/// planting Chinese punctuation into e.g. all-English FireRed output --
/// honest no-op over cross-language mislabeling.
pub(crate) fn segment_qualifies_for_chinese_punctuation(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedPuncRuntimeError {
    #[error("firered-punc pack read failed: {0}")]
    Read(String),
    #[error("firered-punc pack metadata invalid: {0}")]
    Metadata(String),
    #[error("firered-punc pack is missing '{0}'")]
    MissingMetadata(&'static str),
    #[error("firered-punc tokenizer build failed: {0}")]
    Tokenizer(String),
    #[error("firered-punc weight load failed: {0}")]
    Weights(String),
    #[error("firered-punc graph build failed: {0}")]
    Graph(#[from] FireRedPuncGraphError),
}

pub(crate) struct FireRedPuncRuntime {
    metadata: FireRedPuncExecutionMetadata,
    tokenizer: FireRedPuncTokenizer,
    graph: RefCell<FireRedPuncGraph>,
}

impl FireRedPuncRuntime {
    pub(crate) fn from_pack(path: &Path) -> Result<Self, FireRedPuncRuntimeError> {
        let reader = GgufTensorDataReader::from_path(path)
            .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
        let gguf = read_gguf_metadata(path)
            .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
        let metadata = parse_and_validate_firered_punc_metadata(&gguf)
            .map_err(|error| FireRedPuncRuntimeError::Metadata(error.to_string()))?;
        let tokens = gguf.get_string_array(TOKENIZER_GGML_TOKENS_KEY).ok_or(
            FireRedPuncRuntimeError::MissingMetadata(TOKENIZER_GGML_TOKENS_KEY),
        )?;
        let tokenizer = FireRedPuncTokenizer::new(tokens.to_vec())
            .map_err(|error| FireRedPuncRuntimeError::Tokenizer(error.to_string()))?;
        let weights = load_firered_punc_weights(&reader, &metadata)
            .map_err(|error| FireRedPuncRuntimeError::Weights(error.to_string()))?;
        let graph = FireRedPuncGraph::new(&weights, metadata)?;
        Ok(Self {
            metadata,
            tokenizer,
            graph: RefCell::new(graph),
        })
    }

    /// Restore punctuation on `text` (finalize-only, Chinese full-width marks).
    ///
    /// Segments with no Han ideograph are returned verbatim without running
    /// the classifier: see [`segment_qualifies_for_chinese_punctuation`].
    pub(crate) fn punctuate(&self, text: &str) -> Result<String, PunctuationError> {
        if !segment_qualifies_for_chinese_punctuation(text) {
            return Ok(text.to_string());
        }
        restore_punctuation(
            text,
            &self.tokenizer,
            self,
            PunctuationRestoreConfig::default(),
        )
    }
}

/// Exclusive-ownership `Send` wrapper around [`FireRedPuncRuntime`] for
/// callers that must hold the runtime inside a `Send` value (the streaming
/// session caches one per session, and `NativeAsrSession: Send`).
///
/// SAFETY rationale (same pattern as `xasr_zipformer::runtime`'s
/// `SendableRuntime`): the runtime owns ggml context/backend handles with no
/// thread-affine state; it is moved as an exclusive value and only accessed
/// through `&self` from one thread at a time. The wrapper is `Send`, NOT
/// `Sync` (the inner `RefCell` already forbids cross-thread sharing), so no
/// concurrent access to the graph can ever be observed.
pub(crate) struct SendableFireRedPuncRuntime(FireRedPuncRuntime);

unsafe impl Send for SendableFireRedPuncRuntime {}

impl SendableFireRedPuncRuntime {
    pub(crate) fn from_pack(path: &Path) -> Result<Self, FireRedPuncRuntimeError> {
        FireRedPuncRuntime::from_pack(path).map(Self)
    }

    /// See [`FireRedPuncRuntime::punctuate`] (including its per-segment Han
    /// gate).
    pub(crate) fn punctuate(&self, text: &str) -> Result<String, PunctuationError> {
        self.0.punctuate(text)
    }
}

impl PunctuationClassifier for FireRedPuncRuntime {
    fn predict_window_labels(
        &self,
        content_token_ids: &[u32],
    ) -> Result<Vec<usize>, PunctuationError> {
        if content_token_ids.is_empty() {
            return Ok(Vec::new());
        }
        // Wrap the content window in [CLS] ... [SEP] for the encoder.
        let mut ids = Vec::with_capacity(content_token_ids.len() + 2);
        ids.push(self.tokenizer.cls_id());
        ids.extend_from_slice(content_token_ids);
        ids.push(self.tokenizer.sep_id());

        let logits = self
            .graph
            .borrow_mut()
            .forward(&ids)
            .map_err(|error| PunctuationError::Classifier(error.to_string()))?;
        let per_position =
            argmax_labels_per_position(&logits, self.metadata.label_count, ids.len());
        // Drop the [CLS] (index 0) and [SEP] (last) predictions; return the
        // labels aligned to the content tokens.
        Ok(per_position[1..=content_token_ids.len()].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn han_gate_rejects_segments_without_han_ideographs() {
        // Regression guard for the FireRed English-transcript bug: FireRed is
        // FixedMultilingual (its `Transcription.language` is always `None`),
        // so an all-English segment reaches the punctuation stage and must be
        // filtered by content, not by language metadata.
        assert!(!segment_qualifies_for_chinese_punctuation(
            "THIS IS A LIBRIVOX RECORDING"
        ));
        assert!(!segment_qualifies_for_chinese_punctuation(""));
        // Half/full-width Latin, digits, and CJK punctuation without any Han
        // ideograph do not qualify either (is_cjk_char is Han-blocks only).
        assert!(!segment_qualifies_for_chinese_punctuation(
            "ｈｅｌｌｏ 123 。"
        ));
    }

    #[test]
    fn han_gate_accepts_chinese_and_mixed_segments() {
        assert!(segment_qualifies_for_chinese_punctuation("你好世界"));
        // One Han ideograph is enough: mixed zh/en segments must still be
        // punctuated (full-width marks throughout is the correct GB/T 15834
        // treatment of Chinese text with embedded Latin).
        assert!(segment_qualifies_for_chinese_punctuation("打开 hello 模式"));
    }

    /// Real-weights parity: only runs when `OPENASR_FIRERED_PUNC_REAL_PACK`
    /// points at a converted FireRedPunc `.oasr` pack. Left env-gated (like the
    /// hymt2 / qwen real-pack tests) so the default suite stays weight-free; the
    /// true upstream parity is exercised at publish time.
    #[test]
    fn real_pack_punctuates_readme_example() {
        let Some(path) = std::env::var_os("OPENASR_FIRERED_PUNC_REAL_PACK") else {
            return;
        };
        let runtime = FireRedPuncRuntime::from_pack(Path::new(&path)).expect("load real pack");
        let out = runtime.punctuate("你好世界").expect("punctuate");
        assert_eq!(out, "你好世界。", "upstream README golden");

        // The Han gate short-circuits before the classifier: an all-English
        // segment must come back byte-for-byte unchanged even with real
        // weights loaded.
        let english = "THIS IS A LIBRIVOX RECORDING";
        let out = runtime.punctuate(english).expect("punctuate english");
        assert_eq!(out, english, "no-Han segment passes through verbatim");
    }

    /// Converter golden gate: the engine's per-token argmax labels for the
    /// converted `.oasr` pack must exactly match the upstream PyTorch forward.
    /// Both env vars are dev-only (the pack is uncommitted; the JSON is emitted
    /// by `tmp/firered-punc-src/reference_forward.py`), so the default suite
    /// skips this -- it is the publish-time parity proof, mirroring the qwen
    /// forced-aligner reference convention. The JSON is a list of
    /// `{content_ids: [u32], ref_labels: [usize]}` entries; the same content
    /// ids are fed to both sides so this isolates the numeric forward from
    /// tokenization.
    #[test]
    fn real_pack_labels_match_pytorch_reference_golden() {
        let (Some(pack), Some(json)) = (
            std::env::var_os("OPENASR_FIRERED_PUNC_REAL_PACK"),
            std::env::var_os("OPENASR_FIRERED_PUNC_GOLDEN_JSON"),
        ) else {
            return;
        };
        let runtime = FireRedPuncRuntime::from_pack(Path::new(&pack)).expect("load real pack");
        let text = std::fs::read_to_string(Path::new(&json)).expect("read golden json");
        let entries: serde_json::Value = serde_json::from_str(&text).expect("parse golden json");
        let entries = entries.as_array().expect("golden json is a list");
        let mut checked = 0usize;
        for entry in entries {
            let content_ids: Vec<u32> = entry["content_ids"]
                .as_array()
                .expect("content_ids array")
                .iter()
                .map(|value| value.as_u64().expect("id is u64") as u32)
                .collect();
            let ref_labels: Vec<usize> = entry["ref_labels"]
                .as_array()
                .expect("ref_labels array")
                .iter()
                .map(|value| value.as_u64().expect("label is u64") as usize)
                .collect();
            let engine_labels = runtime
                .predict_window_labels(&content_ids)
                .expect("engine predict");
            assert_eq!(
                engine_labels,
                ref_labels,
                "label mismatch for sentence {:?}",
                entry.get("sentence")
            );
            checked += 1;
        }
        assert!(checked > 0, "golden json had no entries");
        eprintln!("firered-punc golden: {checked} sentences matched PyTorch reference");
    }
}
