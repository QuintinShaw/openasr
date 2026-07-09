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
use super::tokenizer::FireRedPuncTokenizer;
use super::weights::load_firered_punc_weights;

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
    pub(crate) fn punctuate(&self, text: &str) -> Result<String, PunctuationError> {
        restore_punctuation(
            text,
            &self.tokenizer,
            self,
            PunctuationRestoreConfig::default(),
        )
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
