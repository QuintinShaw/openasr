//! Shared HF `tokenizer.json` loading for FastConformer importers.
//!
//! Parakeet CTC and TDT carry the same BPE tokenizer shape: a model vocab
//! keyed by token text plus optional added tokens keyed by their final id.
//! This module owns only that source-file schema and dense id-order projection;
//! each importer keeps its own pack metadata and tensor conversion policy.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::models::local_source_import::{
    LocalSourceImportError, read_source_json_file, validate_error,
};

const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModelJson,
    #[serde(default)]
    added_tokens: Vec<TokenizerAddedToken>,
}

#[derive(Debug, Deserialize)]
struct TokenizerModelJson {
    vocab: BTreeMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerAddedToken {
    id: u32,
    content: String,
}

/// Read an HF tokenizer and project its token ids into the GGUF token array.
/// Base-vocab entries are filled first; added tokens intentionally overlay
/// them, matching Transformers' final tokenizer id assignment.
pub(crate) fn load_vocab_tokens(
    source_root: &Path,
    vocab_size: usize,
    family: &str,
) -> Result<Vec<String>, LocalSourceImportError> {
    let tokenizer: TokenizerJson = read_source_json_file(source_root, SOURCE_TOKENIZER_JSON)?;
    build_vocab_tokens(&tokenizer, vocab_size, family)
}

fn build_vocab_tokens(
    tokenizer: &TokenizerJson,
    vocab_size: usize,
    family: &str,
) -> Result<Vec<String>, LocalSourceImportError> {
    let mut tokens = vec![None::<String>; vocab_size];
    for (token, &id) in &tokenizer.model.vocab {
        if (id as usize) < vocab_size {
            tokens[id as usize] = Some(token.clone());
        }
    }
    for added in &tokenizer.added_tokens {
        if (added.id as usize) < vocab_size {
            tokens[added.id as usize] = Some(added.content.clone());
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                validate_error(format!("{family} tokenizer is missing token for id {id}"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_tokens_overlay_base_vocab_in_id_order() {
        let tokenizer: TokenizerJson = serde_json::from_str(
            r#"{
                "model": {"vocab": {"base": 0, "old": 1}},
                "added_tokens": [{"id": 1, "content": "<blank>"}]
            }"#,
        )
        .expect("tokenizer fixture");
        assert_eq!(
            build_vocab_tokens(&tokenizer, 2, "parakeet-ctc").expect("dense tokens"),
            vec!["base".to_string(), "<blank>".to_string()]
        );
    }

    #[test]
    fn missing_id_keeps_family_error_semantics() {
        let tokenizer: TokenizerJson =
            serde_json::from_str(r#"{"model":{"vocab":{"a":0}},"added_tokens":[]}"#)
                .expect("tokenizer fixture");
        let error =
            build_vocab_tokens(&tokenizer, 2, "parakeet-ctc").expect_err("missing id must fail");
        assert_eq!(
            error.to_string(),
            "parakeet-ctc tokenizer is missing token for id 1"
        );
    }
}
