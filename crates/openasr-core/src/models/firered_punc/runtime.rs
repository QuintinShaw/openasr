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
}
