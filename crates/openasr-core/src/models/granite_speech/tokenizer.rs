//! Granite Speech's byte-level GPT2-BPE tokenizer (`tokenizer.json` /
//! `vocab.json` + `merges.txt`, granite-4.0 vocabulary, `vocab_size=100353`).
//!
//! Reuses the shared byte-level BPE primitives in `models::gpt2_bpe` (the
//! same ones `qwen::tokenizer::Qwen3AsrTokenizer` wraps) verbatim -- this is a
//! stock GPT2-style tokenizer, no family-specific merge/byte-mapping logic.
//!
//! `from_source_files` (dev/test constructor, reads `vocab.json`/
//! `merges.txt` directly from an HF source tree) and `from_gguf_metadata`
//! (production constructor, reads the same `tokenizer.ggml.tokens`/
//! `tokenizer.ggml.merges` keys `package_import.rs` now writes into the
//! `.oasr` pack -- mirrors `Qwen3AsrTokenizer::from_gguf_metadata` exactly,
//! since both wrap the same stock GPT2-BPE shape) produce byte-identical
//! tokenizers for the same checkpoint.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::Path;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::oasr_metadata::{required_metadata_string, required_metadata_string_array};

const GRANITE_SPEECH_TOKENIZER_FAMILY: &str = "granite-speech";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechTokenizerError {
    #[error("granite-speech tokenizer failed to read '{path}': {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("granite-speech tokenizer failed to parse '{path}': {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("granite-speech tokenizer: {0}")]
    Validate(String),
    #[error("granite-speech tokenizer encode/decode failed: {0}")]
    Bpe(#[from] crate::NativeAsrError),
}

#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeechTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
}

impl GraniteSpeechTokenizer {
    /// Dev/test constructor: loads `vocab.json` (token -> id) + `merges.txt`
    /// (BPE merge rules, `#version: ...` header line skipped) directly from
    /// an HF source checkout. See module doc for why this is not yet the
    /// GGUF-backed path every other builtin tokenizer uses.
    pub(crate) fn from_source_files(
        source_root: &Path,
    ) -> Result<Self, GraniteSpeechTokenizerError> {
        let vocab_path = source_root.join("vocab.json");
        let vocab_bytes =
            std::fs::read(&vocab_path).map_err(|source| GraniteSpeechTokenizerError::Read {
                path: vocab_path.display().to_string(),
                source,
            })?;
        let vocab: BTreeMap<String, u32> =
            serde_json::from_slice(&vocab_bytes).map_err(|source| {
                GraniteSpeechTokenizerError::Parse {
                    path: vocab_path.display().to_string(),
                    source,
                }
            })?;
        let vocab_size = vocab
            .values()
            .copied()
            .max()
            .map(|max_id| max_id as usize + 1)
            .unwrap_or(0);
        let mut tokens = vec![None::<String>; vocab_size];
        for (token, id) in &vocab {
            if let Some(slot) = tokens.get_mut(*id as usize) {
                *slot = Some(token.clone());
            }
        }
        // Special tokens beyond the BPE vocab proper (e.g. the audio
        // placeholder at id 100352, `<|end_of_text|>` at 100257, etc.) are
        // absent from `vocab.json` but present in `added_tokens.json`; fill
        // any gaps with a placeholder id-string so `id_to_token` stays dense
        // (this is only needed for the encode/decode round-trip a text-only
        // decode-core test exercises -- the real pack conversion should carry
        // the full `tokenizer.json` special-token table instead).
        let added_path = source_root.join("added_tokens.json");
        let added_tokens: Option<BTreeMap<String, u32>> = std::fs::read(&added_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());
        if let Some(added) = added_tokens {
            for (token, id) in added {
                let id = id as usize;
                if id >= tokens.len() {
                    tokens.resize(id + 1, None);
                }
                tokens[id] = Some(token);
            }
        }

        let token_strings: Vec<String> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| token.clone().unwrap_or_else(|| format!("<|unused_{id}|>")))
            .collect();
        let token_to_id = build_token_to_id(&token_strings, "granite-speech")?;
        let id_to_token = token_strings.into_iter().map(Some).collect();

        let merges_path = source_root.join("merges.txt");
        let merges_text = std::fs::read_to_string(&merges_path).map_err(|source| {
            GraniteSpeechTokenizerError::Read {
                path: merges_path.display().to_string(),
                source,
            }
        })?;
        let merges: Vec<String> = merges_text
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(str::to_string)
            .collect();
        let merge_rank = build_merge_rank(&merges);

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
        })
    }

    /// Production constructor: reads `tokenizer.ggml.tokens`/
    /// `tokenizer.ggml.merges` from a `.oasr` pack's GGUF metadata (written
    /// by `package_import.rs`'s `granite_speech_runtime_gguf_metadata`).
    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(
            metadata,
            TOKENIZER_GGML_MODEL_KEY,
            GRANITE_SPEECH_TOKENIZER_FAMILY,
        )?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "granite-speech GGUF tokenizer key '{TOKENIZER_GGML_MODEL_KEY}' must be \
                     '{TOKENIZER_GGML_MODEL_VALUE_GPT2}', got '{tokenizer_model}'"
                ),
            });
        }
        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            GRANITE_SPEECH_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "granite-speech GGUF tokenizer key '{TOKENIZER_GGML_TOKENS_KEY}' cannot be empty"
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            GRANITE_SPEECH_TOKENIZER_FAMILY,
        )?;

        let token_to_id = build_token_to_id(tokens, GRANITE_SPEECH_TOKENIZER_FAMILY)?;
        let id_to_token = tokens.iter().map(|token| Some(token.clone())).collect();
        let merge_rank = build_merge_rank(merges);

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
        })
    }

    pub(crate) fn encode_prompt_text(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, GraniteSpeechTokenizerError> {
        Ok(encode_prompt_text(
            text,
            &self.token_to_id,
            &self.merge_rank,
            "granite-speech",
        )?)
    }

    pub(crate) fn decode_text_token_ids(
        &self,
        token_ids: &[u32],
    ) -> Result<String, GraniteSpeechTokenizerError> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let index = token_id as usize;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(GraniteSpeechTokenizerError::Validate(format!(
                    "granite-speech tokenizer id {token_id} is not in vocab"
                )));
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_ROOT_ENV: &str = "OPENASR_GRANITE_SPEECH_SOURCE_ROOT";

    fn source_root() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var_os(SOURCE_ROOT_ENV)?);
        path.join("vocab.json").exists().then_some(path)
    }

    #[test]
    #[ignore = "requires local granite-speech-4.1-2b tokenizer files under tmp/ (not committed)"]
    fn granite_speech_tokenizer_round_trips_plain_text() {
        let Some(root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let tokenizer = GraniteSpeechTokenizer::from_source_files(&root).expect("load tokenizer");
        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tokenizer.encode_prompt_text(text).expect("encode");
        assert!(!ids.is_empty());
        let decoded = tokenizer.decode_text_token_ids(&ids).expect("decode");
        assert_eq!(
            decoded, text,
            "BPE round-trip must be exact for plain ASCII text"
        );
    }
}
