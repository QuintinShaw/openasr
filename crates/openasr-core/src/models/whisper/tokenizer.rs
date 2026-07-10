use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde::Deserialize;

use crate::NativeAsrError;
use crate::TranscriptionTask;
use crate::arch::hparams::WHISPER_VOCAB_SIZE_KEY;
use crate::ggml_runtime::{
    GgmlRuntimeSource, GgufMetadata, read_gguf_metadata_from_runtime_source,
};
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::language::{language_control_token, normalize_language};
// Re-exported at this path (rather than imported privately) because
// `whisper::package_import` and `whisper::batched_decode` reach these three
// shared keys via `super::tokenizer::TOKENIZER_GGML_*`, matching how they
// already pull `TOKENIZER_GGML_MODEL_VALUE_GPT2` from this module.
pub(crate) use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
};
use crate::models::oasr_metadata::{
    optional_metadata_u32, required_metadata_string, required_metadata_string_array,
    required_metadata_u32, required_metadata_u32_array,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

const WHISPER_TOKENIZER_FAMILY: &str = "Whisper";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";
const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_MERGES_TXT: &str = "merges.txt";
const SOURCE_ADDED_TOKENS_JSON: &str = "added_tokens.json";
const NOTIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
const START_OF_TRANSCRIPT_TOKEN: &str = "<|startoftranscript|>";
const END_OF_TEXT_TOKEN: &str = "<|endoftext|>";
const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
const TRANSLATE_TOKEN: &str = "<|translate|>";
const ENGLISH_LANGUAGE_TOKEN: &str = "<|en|>";
pub(crate) const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

/// True when a Whisper pack carries the multilingual language-token block, using
/// the same `vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE` rule the decoder
/// applies. If the token list cannot be read, fail toward multilingual: the
/// decoder still validates the explicit `<|lang|>` token before use, so this only
/// affects whether an English-only pack rejects a foreign-language hint.
pub(crate) fn whisper_metadata_is_multilingual(metadata: &GgufMetadata) -> bool {
    match required_metadata_string_array(
        metadata,
        TOKENIZER_GGML_TOKENS_KEY,
        WHISPER_TOKENIZER_FAMILY,
    ) {
        Ok(tokens) => tokens.len() > super::ggml_executor::WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE,
        Err(_) => true,
    }
}
pub(crate) const TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY: &str = "tokenizer.ggml.special_token_ids";
pub(crate) const TOKENIZER_GGML_SOT_TOKEN_ID_KEY: &str = "tokenizer.ggml.sot_token_id";
pub(crate) const TOKENIZER_GGML_EOT_TOKEN_ID_KEY: &str = "tokenizer.ggml.eot_token_id";
pub(crate) const TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY: &str =
    "tokenizer.ggml.transcribe_token_id";
pub(crate) const TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY: &str =
    "tokenizer.ggml.no_timestamps_token_id";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperHfTokenizerImport {
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub special_token_ids: Vec<u32>,
    pub sot_token_id: u32,
    pub eot_token_id: u32,
    pub transcribe_token_id: u32,
    pub no_timestamps_token_id: u32,
}

impl WhisperHfTokenizerImport {
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// Inputs that select the Whisper decoder prefix: source `language` (raw request
/// hint, `None` for the model default), `task`, and whether the checkpoint is
/// multilingual. The default (`Transcribe` + unset/`en` language) reproduces the
/// legacy byte-identical prefix.
#[derive(Debug, Clone, Copy)]
pub struct WhisperPrefixSpec<'a> {
    pub language: Option<&'a str>,
    pub task: TranscriptionTask,
    pub is_multilingual: bool,
}

impl WhisperPrefixSpec<'_> {
    /// Legacy default: transcribe in the model's default (English) language.
    /// Guaranteed never to fail closed, so callers can `expect` the result.
    pub fn transcribe(is_multilingual: bool) -> Self {
        Self {
            language: None,
            task: TranscriptionTask::Transcribe,
            is_multilingual,
        }
    }
}

/// Fail-closed reasons the decoder prefix cannot be built for an explicitly
/// requested (non-default) language or task on a given pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperPrefixError {
    LanguageTokenMissing { language: String },
    TranslateTokenMissing,
}

#[derive(Debug, Clone)]
pub struct WhisperTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    special_token_ids: BTreeSet<u32>,
    sot_token_id: Option<u32>,
    eot_token_id: Option<u32>,
    transcribe_token_id: Option<u32>,
    notimestamps_token_id: Option<u32>,
}

impl WhisperTokenizer {
    pub(crate) fn from_ggml_runtime_source(
        runtime_source: &GgmlRuntimeSource,
    ) -> Result<Self, NativeAsrError> {
        let metadata = read_gguf_metadata_from_runtime_source(runtime_source).map_err(|error| {
            NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "could not read GGUF metadata from '{}': {error}",
                    runtime_source.path().display()
                ),
            }
        })?;
        Self::from_gguf_metadata(&metadata)
    }

    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model =
            required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY, WHISPER_TOKENIZER_FAMILY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Whisper GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }

        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Whisper GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_TOKENS_KEY
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        if merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Whisper GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_MERGES_KEY
                ),
            });
        }
        let special_token_ids = required_metadata_u32_array(
            metadata,
            TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        if special_token_ids.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Whisper GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY
                ),
            });
        }

        if let Some(vocab_size) =
            optional_metadata_u32(metadata, WHISPER_VOCAB_SIZE_KEY, WHISPER_TOKENIZER_FAMILY)?
        {
            let token_count =
                u32::try_from(tokens.len()).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Whisper GGUF tokenizer token count {} exceeds u32",
                        tokens.len()
                    ),
                })?;
            if token_count != vocab_size {
                return Err(NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Whisper GGUF tokenizer token count {} does not match '{}'={}",
                        token_count, WHISPER_VOCAB_SIZE_KEY, vocab_size
                    ),
                });
            }
        }

        let sot_token_id = required_metadata_u32(
            metadata,
            TOKENIZER_GGML_SOT_TOKEN_ID_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        let eot_token_id = required_metadata_u32(
            metadata,
            TOKENIZER_GGML_EOT_TOKEN_ID_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        let transcribe_token_id = required_metadata_u32(
            metadata,
            TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;
        let no_timestamps_token_id = required_metadata_u32(
            metadata,
            TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
            WHISPER_TOKENIZER_FAMILY,
        )?;

        let id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let token_to_id = build_token_to_id(tokens, "Whisper")?;
        let merge_rank = build_merge_rank(merges);
        validate_special_token_id(&id_to_token, sot_token_id, START_OF_TRANSCRIPT_TOKEN)?;
        validate_special_token_id(&id_to_token, eot_token_id, END_OF_TEXT_TOKEN)?;
        validate_special_token_id(&id_to_token, transcribe_token_id, TRANSCRIBE_TOKEN)?;
        validate_special_token_id(&id_to_token, no_timestamps_token_id, NOTIMESTAMPS_TOKEN)?;

        let mut special_ids = BTreeSet::new();
        for token_id in special_token_ids {
            let index =
                usize::try_from(*token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Whisper GGUF tokenizer special token id {token_id} does not fit usize"
                    ),
                })?;
            if index >= id_to_token.len() {
                return Err(NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Whisper GGUF tokenizer special token id {token_id} is out of range for vocab size {}",
                        id_to_token.len()
                    ),
                });
            }
            special_ids.insert(*token_id);
        }
        special_ids.insert(sot_token_id);
        special_ids.insert(eot_token_id);
        special_ids.insert(transcribe_token_id);
        special_ids.insert(no_timestamps_token_id);

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special_token_ids: special_ids,
            sot_token_id: Some(sot_token_id),
            eot_token_id: Some(eot_token_id),
            transcribe_token_id: Some(transcribe_token_id),
            notimestamps_token_id: Some(no_timestamps_token_id),
        })
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        // Multilingual Whisper tokenizers mark the timestamp tokens (`<|0.00|>`
        // ... `<|30.00|>`) as non-special, so they are absent from
        // `special_token_ids`. With a no-timestamps prompt the decoder should not
        // emit them, but large multilingual checkpoints can still leak a leading
        // `<|0.00|>`. Whisper's timestamp tokens form a contiguous range that
        // starts immediately after `<|notimestamps|>` and runs to the vocab end,
        // so drop any token id at or beyond that first timestamp id. The `.en`
        // no-timestamps decode never selects ids in this range, leaving those
        // transcripts byte-identical.
        let first_timestamp_token_id = self.notimestamps_token_id.and_then(|id| id.checked_add(1));
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if self.special_token_ids.contains(token_id) {
                continue;
            }
            if first_timestamp_token_id.is_some_and(|first| *token_id >= first) {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("Whisper tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Whisper tokenizer id {token_id} is not in vocab"),
                });
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Build the no-timestamps decoder prefix for a `language`/`task` selection.
    ///
    /// English-only (`.en`) checkpoints decode with just
    /// `<|startoftranscript|> <|notimestamps|>`, matching their training and
    /// keeping their byte-for-byte transcripts unchanged (language/task are
    /// neutralized — those tokens are not in the `.en` vocab). Multilingual
    /// checkpoints add the language + task tokens, decoding with
    /// `<|startoftranscript|> <|lang|> <|task|> <|notimestamps|>`. Without that
    /// pair a multilingual model is left in its timestamp-prediction state and
    /// either leaks a leading `<|0.00|>` (large checkpoints) or emits nothing
    /// but a timestamp (small/base checkpoints).
    ///
    /// Divergence from the legacy `<|en|> <|transcribe|>` default is gated on the
    /// request OPTION being explicitly non-default (a non-English language or
    /// `task=translate`), never on the resolved token, so a multilingual pack
    /// that happens to lack `<|en|>` keeps the legacy silent-omit on the default
    /// path. Only an explicit non-default request fails closed when its token is
    /// absent, rather than silently transcribing.
    pub fn decoder_prefix(
        &self,
        decoder_start_token_id: u32,
        spec: &WhisperPrefixSpec<'_>,
    ) -> Result<Vec<u32>, WhisperPrefixError> {
        let mut prefix = vec![decoder_start_token_id];
        if spec.is_multilingual {
            let explicit_language = spec
                .language
                .map(normalize_language)
                .filter(|language| !language.is_empty() && language != "en");
            if let Some(language) = explicit_language {
                let token = language_control_token(&language);
                let token_id = self
                    .token_id_by_content(&token)
                    .ok_or(WhisperPrefixError::LanguageTokenMissing { language })?;
                prefix.push(token_id);
            } else if let Some(language_token_id) = self.token_id_by_content(ENGLISH_LANGUAGE_TOKEN)
            {
                prefix.push(language_token_id);
            }
            match spec.task {
                TranscriptionTask::Transcribe => {
                    if let Some(transcribe_token_id) = self.transcribe_token_id {
                        prefix.push(transcribe_token_id);
                    }
                }
                TranscriptionTask::Translate => {
                    let token_id = self
                        .translate_token_id()
                        .ok_or(WhisperPrefixError::TranslateTokenMissing)?;
                    prefix.push(token_id);
                }
            }
        }
        if let Some(token_id) = self.notimestamps_token_id {
            prefix.push(token_id);
        }
        Ok(prefix)
    }

    pub fn token_id_by_content(&self, content: &str) -> Option<u32> {
        self.token_to_id.get(content).copied()
    }

    /// The `<|translate|>` task-token id, resolved by string content (its id
    /// shifts per checkpoint, so it has no dedicated metadata field unlike
    /// `transcribe_token_id`). `None` on a pack without the token.
    pub fn translate_token_id(&self) -> Option<u32> {
        self.token_id_by_content(TRANSLATE_TOKEN)
    }

    pub fn start_of_transcript_token_id(&self) -> Option<u32> {
        self.sot_token_id
    }

    pub fn end_of_text_token_id(&self) -> Option<u32> {
        self.eot_token_id
    }

    pub fn transcribe_token_id(&self) -> Option<u32> {
        self.transcribe_token_id
    }

    pub fn no_timestamps_token_id(&self) -> Option<u32> {
        self.notimestamps_token_id
    }

    pub fn initial_prompt_token_ids(&self) -> Result<Vec<u32>, NativeAsrError> {
        let mut tokens = vec![required_token_id(
            self.sot_token_id,
            START_OF_TRANSCRIPT_TOKEN,
        )?];
        tokens.push(required_token_id(
            self.notimestamps_token_id,
            NOTIMESTAMPS_TOKEN,
        )?);
        Ok(tokens)
    }

    pub fn encode_prompt_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        encode_prompt_text(text, &self.token_to_id, &self.merge_rank, "Whisper")
    }

    #[cfg(test)]
    fn from_tokenizer_payload_bytes(tokenizer_bytes: &[u8]) -> Result<Self, NativeAsrError> {
        let metadata = parse_tokenizer_json(tokenizer_bytes, SOURCE_TOKENIZER_JSON)?;
        let id_to_token = metadata.id_to_token_table()?;
        let token_to_id = metadata.token_to_id_map()?;
        let merge_rank = build_merge_rank(&metadata.merge_rules());
        let special_token_ids = metadata.special_token_ids();

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special_token_ids,
            sot_token_id: metadata.special_token_id(START_OF_TRANSCRIPT_TOKEN),
            eot_token_id: metadata.special_token_id(END_OF_TEXT_TOKEN),
            transcribe_token_id: metadata.special_token_id(TRANSCRIBE_TOKEN),
            notimestamps_token_id: metadata.special_token_id(NOTIMESTAMPS_TOKEN),
        })
    }
}

pub(crate) fn load_whisper_hf_tokenizer_import_v0(
    source_root: &Path,
) -> Result<WhisperHfTokenizerImport, NativeAsrError> {
    let tokenizer_json_path = source_root.join(SOURCE_TOKENIZER_JSON);
    if tokenizer_json_path.is_file() {
        let tokenizer_bytes = fs::read(&tokenizer_json_path).map_err(|error| {
            NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Whisper tokenizer import could not read '{}': {error}",
                    tokenizer_json_path.display()
                ),
            }
        })?;
        return tokenizer_import_from_tokenizer_json_bytes(&tokenizer_bytes);
    }

    let vocab_path = source_root.join(SOURCE_VOCAB_JSON);
    let merges_path = source_root.join(SOURCE_MERGES_TXT);
    let vocab: BTreeMap<String, u32> = read_json_file(&vocab_path, SOURCE_VOCAB_JSON)?;
    let merges_text =
        fs::read_to_string(&merges_path).map_err(|error| NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Whisper tokenizer import could not read '{}': {error}",
                merges_path.display()
            ),
        })?;
    let merges = merges_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut special_token_ids = BTreeSet::new();
    let added_tokens_path = source_root.join(SOURCE_ADDED_TOKENS_JSON);
    if added_tokens_path.is_file() {
        let added_tokens: WhisperAddedTokensFile =
            read_json_file(&added_tokens_path, SOURCE_ADDED_TOKENS_JSON)?;
        for token_id in added_tokens.into_token_ids() {
            special_token_ids.insert(token_id);
        }
    }

    build_import_from_vocab(vocab, merges, special_token_ids)
}

fn tokenizer_import_from_tokenizer_json_bytes(
    tokenizer_bytes: &[u8],
) -> Result<WhisperHfTokenizerImport, NativeAsrError> {
    let metadata = parse_tokenizer_json(tokenizer_bytes, SOURCE_TOKENIZER_JSON)?;
    let vocab = metadata.token_to_id_map()?;
    let merges = metadata.merge_rules();
    let special_token_ids = metadata.special_token_ids();
    build_import_from_vocab(vocab, merges, special_token_ids)
}

fn build_import_from_vocab(
    vocab: BTreeMap<String, u32>,
    merges: Vec<String>,
    mut special_token_ids: BTreeSet<u32>,
) -> Result<WhisperHfTokenizerImport, NativeAsrError> {
    if vocab.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer import requires non-empty vocab".to_string(),
        });
    }
    if merges.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer import requires non-empty merges".to_string(),
        });
    }
    if vocab.keys().any(|token| token.contains('\0')) {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer import found NUL byte in vocab token".to_string(),
        });
    }
    if merges.iter().any(|merge| merge.contains('\0')) {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer import found NUL byte in merges entry".to_string(),
        });
    }

    let tokens = dense_token_table_from_vocab(&vocab)?;
    let token_to_id = vocab;
    let sot_token_id =
        required_token_id_from_map(&token_to_id, START_OF_TRANSCRIPT_TOKEN, "tokenizer import")?;
    let eot_token_id =
        required_token_id_from_map(&token_to_id, END_OF_TEXT_TOKEN, "tokenizer import")?;
    let transcribe_token_id =
        required_token_id_from_map(&token_to_id, TRANSCRIBE_TOKEN, "tokenizer import")?;
    let no_timestamps_token_id =
        required_token_id_from_map(&token_to_id, NOTIMESTAMPS_TOKEN, "tokenizer import")?;

    special_token_ids.insert(sot_token_id);
    special_token_ids.insert(eot_token_id);
    special_token_ids.insert(transcribe_token_id);
    special_token_ids.insert(no_timestamps_token_id);
    if special_token_ids.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer import requires non-empty special token ids".to_string(),
        });
    }

    validate_special_token_id_from_tokens(&tokens, sot_token_id, START_OF_TRANSCRIPT_TOKEN)?;
    validate_special_token_id_from_tokens(&tokens, eot_token_id, END_OF_TEXT_TOKEN)?;
    validate_special_token_id_from_tokens(&tokens, transcribe_token_id, TRANSCRIBE_TOKEN)?;
    validate_special_token_id_from_tokens(&tokens, no_timestamps_token_id, NOTIMESTAMPS_TOKEN)?;

    Ok(WhisperHfTokenizerImport {
        tokens,
        merges,
        special_token_ids: special_token_ids.into_iter().collect(),
        sot_token_id,
        eot_token_id,
        transcribe_token_id,
        no_timestamps_token_id,
    })
}

fn dense_token_table_from_vocab(
    vocab: &BTreeMap<String, u32>,
) -> Result<Vec<String>, NativeAsrError> {
    let max_id =
        vocab
            .values()
            .copied()
            .max()
            .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
                reason: "Whisper tokenizer vocab is empty".to_string(),
            })?;
    let table_len = usize::try_from(max_id)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: "Whisper tokenizer vocab id exceeds supported size".to_string(),
        })?;
    let mut table = vec![None; table_len];
    for (token, id) in vocab {
        let index = usize::try_from(*id).map_err(|_| NativeAsrError::UnsupportedModelPack {
            reason: format!("Whisper tokenizer vocab id {id} does not fit into usize"),
        })?;
        match &table[index] {
            Some(existing) if existing != token => {
                return Err(NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Whisper tokenizer vocab id {id} maps to both '{existing}' and '{token}'"
                    ),
                });
            }
            _ => table[index] = Some(token.clone()),
        }
    }

    let mut dense = Vec::with_capacity(table_len);
    for (index, token) in table.into_iter().enumerate() {
        let token = token.ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Whisper tokenizer vocab id {index} is missing"),
        })?;
        dense.push(token);
    }
    Ok(dense)
}

fn validate_special_token_id(
    id_to_token: &[Option<String>],
    token_id: u32,
    expected_token: &'static str,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "Whisper tokenizer special token id {token_id} for '{expected_token}' does not fit usize"
        ),
    })?;
    let Some(Some(actual)) = id_to_token.get(index) else {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Whisper tokenizer special token id {token_id} for '{expected_token}' is out of range"
            ),
        });
    };
    if actual != expected_token {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Whisper tokenizer special token id {token_id} expected '{expected_token}', found '{actual}'"
            ),
        });
    }
    Ok(())
}

fn validate_special_token_id_from_tokens(
    tokens: &[String],
    token_id: u32,
    expected_token: &'static str,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "Whisper tokenizer special token id {token_id} for '{expected_token}' does not fit usize"
        ),
    })?;
    let Some(actual) = tokens.get(index) else {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Whisper tokenizer special token id {token_id} for '{expected_token}' is out of range"
            ),
        });
    };
    if actual != expected_token {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Whisper tokenizer special token id {token_id} expected '{expected_token}', found '{actual}'"
            ),
        });
    }
    Ok(())
}

fn required_token_id(token_id: Option<u32>, token: &'static str) -> Result<u32, NativeAsrError> {
    token_id.ok_or_else(|| NativeAsrError::UnsupportedModelPack {
        reason: format!("Whisper tokenizer is missing required special token '{token}'"),
    })
}

fn required_token_id_from_map(
    token_to_id: &BTreeMap<String, u32>,
    token: &'static str,
    source: &str,
) -> Result<u32, NativeAsrError> {
    token_to_id
        .get(token)
        .copied()
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Whisper {source} missing required special token '{token}'"),
        })
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for WhisperTokenizer {
    fn start_of_transcript_token_id(&self) -> Option<u32> {
        self.start_of_transcript_token_id()
    }

    fn transcribe_token_id(&self) -> Option<u32> {
        self.transcribe_token_id()
    }

    fn no_timestamps_token_id(&self) -> Option<u32> {
        self.no_timestamps_token_id()
    }

    fn token_id_by_content(&self, content: &str) -> Option<u32> {
        self.token_id_by_content(content)
    }
}

impl PhraseBiasTokenEncoder for WhisperTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        self.encode_prompt_text(phrase)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        // Byte-level BPE: also match the leading-space form the model emits
        // mid-sentence, not just the standalone tokenization.
        encode_bpe_phrase_bias_variants(phrase, |text| self.encode_prompt_text(text)).map(Some)
    }
}

fn read_json_file<T: for<'de> Deserialize<'de>>(
    path: &Path,
    label: &str,
) -> Result<T, NativeAsrError> {
    let bytes = fs::read(path).map_err(|error| NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "Whisper tokenizer import could not read '{}': {error}",
            path.display()
        ),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| NativeAsrError::UnsupportedModelPack {
        reason: format!("Whisper tokenizer import {label} is invalid JSON: {error}"),
    })
}

fn parse_tokenizer_json(
    tokenizer_bytes: &[u8],
    source_label: &str,
) -> Result<WhisperTokenizerJson, NativeAsrError> {
    serde_json::from_slice(tokenizer_bytes).map_err(|error| NativeAsrError::UnsupportedModelPack {
        reason: format!("Whisper {source_label} is invalid JSON: {error}"),
    })
}

#[derive(Debug, Deserialize)]
struct WhisperTokenizerJson {
    #[serde(default)]
    added_tokens: Vec<WhisperAddedToken>,
    model: WhisperTokenizerModel,
    #[serde(default)]
    post_processor: Option<WhisperPostProcessor>,
}

impl WhisperTokenizerJson {
    #[cfg(test)]
    fn special_token_id(&self, token: &str) -> Option<u32> {
        self.post_processor
            .as_ref()
            .and_then(|processor| processor.special_tokens.get(token))
            .and_then(|entry| entry.ids.first().copied())
            .or_else(|| {
                self.added_tokens
                    .iter()
                    .find(|entry| entry.content == token)
                    .map(|entry| entry.id)
            })
            .or_else(|| self.model.vocab.get(token).copied())
    }

    fn special_token_ids(&self) -> BTreeSet<u32> {
        let mut ids = self
            .added_tokens
            .iter()
            .filter(|entry| entry.special)
            .map(|entry| entry.id)
            .collect::<BTreeSet<_>>();
        if let Some(processor) = self.post_processor.as_ref() {
            for binding in processor.special_tokens.values() {
                ids.extend(binding.ids.iter().copied());
            }
        }
        ids
    }

    fn token_to_id_map(&self) -> Result<BTreeMap<String, u32>, NativeAsrError> {
        let mut token_to_id = self.model.vocab.clone();
        for entry in &self.added_tokens {
            match token_to_id.get(&entry.content) {
                Some(existing) if *existing != entry.id => {
                    return Err(NativeAsrError::UnsupportedModelPack {
                        reason: format!(
                            "Whisper tokenizer token '{}' maps to conflicting ids {} and {}",
                            entry.content, existing, entry.id
                        ),
                    });
                }
                _ => {
                    token_to_id.insert(entry.content.clone(), entry.id);
                }
            }
        }
        Ok(token_to_id)
    }

    fn merge_rules(&self) -> Vec<String> {
        self.model
            .merges
            .iter()
            .map(WhisperTokenizerMergeEntry::merge_rule)
            .collect()
    }

    #[cfg(test)]
    fn id_to_token_table(&self) -> Result<Vec<Option<String>>, NativeAsrError> {
        let token_to_id = self.token_to_id_map()?;
        let max_id = token_to_id.values().copied().max().ok_or_else(|| {
            NativeAsrError::UnsupportedModelPack {
                reason: "Whisper tokenizer vocab is empty".to_string(),
            }
        })?;
        let table_len = usize::try_from(max_id)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
                reason: "Whisper tokenizer vocab id exceeds supported size".to_string(),
            })?;
        let mut table = vec![None; table_len];
        for (token, id) in token_to_id {
            let index = usize::try_from(id).map_err(|_| NativeAsrError::UnsupportedModelPack {
                reason: format!("Whisper tokenizer vocab id {id} does not fit into usize"),
            })?;
            table[index] = Some(token);
        }
        Ok(table)
    }
}

#[derive(Debug, Deserialize)]
struct WhisperAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct WhisperTokenizerModel {
    #[serde(default)]
    vocab: BTreeMap<String, u32>,
    #[serde(default)]
    merges: Vec<WhisperTokenizerMergeEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WhisperTokenizerMergeEntry {
    Text(String),
    Pair([String; 2]),
}

impl WhisperTokenizerMergeEntry {
    fn merge_rule(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Pair(pair) => format!("{} {}", pair[0], pair[1]),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct WhisperPostProcessor {
    #[serde(default)]
    special_tokens: BTreeMap<String, WhisperSpecialTokenBinding>,
}

#[derive(Debug, Deserialize)]
struct WhisperSpecialTokenBinding {
    ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WhisperAddedTokensFile {
    Map(BTreeMap<String, u32>),
    List(Vec<WhisperAddedToken>),
}

impl WhisperAddedTokensFile {
    fn into_token_ids(self) -> Vec<u32> {
        match self {
            Self::Map(map) => map.into_values().collect(),
            Self::List(list) => list.into_iter().map(|entry| entry.id).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn special_added_token_json(id: u32, content: &str) -> String {
        format!(
            r#"{{"id":{id},"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#
        )
    }

    #[test]
    fn tokenizer_prefix_includes_decoder_start_and_notimestamps_when_present() {
        let notimestamps = special_added_token_json(42, "<|notimestamps|>");
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            format!(
                r#"{{
                    "version":"1.0",
                    "added_tokens":[{notimestamps}],
                    "decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},
                    "model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"a":0,"b":1}},"merges":[]}}
                }}"#
            )
            .as_bytes(),
        )
        .unwrap();

        assert_eq!(
            tokenizer
                .decoder_prefix(7, &WhisperPrefixSpec::transcribe(false))
                .expect("default prefix"),
            vec![7, 42]
        );
    }

    #[test]
    fn tokenizer_prefix_inserts_language_and_task_tokens_for_multilingual() {
        // Multilingual prefix must be `<|sot|> <|en|> <|transcribe|> <|notimestamps|>`.
        let sot = special_added_token_json(7, "<|startoftranscript|>");
        let en = special_added_token_json(8, "<|en|>");
        let transcribe = special_added_token_json(9, "<|transcribe|>");
        let notimestamps = special_added_token_json(10, "<|notimestamps|>");
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            format!(
                r#"{{
                    "version":"1.0",
                    "added_tokens":[{sot},{en},{transcribe},{notimestamps}],
                    "decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},
                    "model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"a":0,"<|startoftranscript|>":7,"<|en|>":8,"<|transcribe|>":9,"<|notimestamps|>":10}},"merges":[]}}
                }}"#
            )
            .as_bytes(),
        )
        .unwrap();

        // English-only path is unchanged (no `<|en|>`/`<|transcribe|>` inserted).
        assert_eq!(
            tokenizer
                .decoder_prefix(7, &WhisperPrefixSpec::transcribe(false))
                .expect("default prefix"),
            vec![7, 10]
        );
        // Multilingual path inserts the language + task tokens before notimestamps.
        assert_eq!(
            tokenizer
                .decoder_prefix(7, &WhisperPrefixSpec::transcribe(true))
                .expect("default prefix"),
            vec![7, 8, 9, 10]
        );
        // LID invariant: an unset language and an explicit "en" produce a
        // byte-identical multilingual prefix, so whisper auto-detecting English
        // is a no-op vs the legacy unset path. An explicit "fr" differs by
        // exactly the swapped language token.
        let unset = tokenizer
            .decoder_prefix(7, &WhisperPrefixSpec::transcribe(true))
            .expect("unset prefix");
        let explicit_en = tokenizer
            .decoder_prefix(
                7,
                &WhisperPrefixSpec {
                    language: Some("en"),
                    task: TranscriptionTask::Transcribe,
                    is_multilingual: true,
                },
            )
            .expect("explicit en prefix");
        assert_eq!(
            unset, explicit_en,
            "detecting en must be byte-identical to the unset path"
        );
        // Explicit non-English language swaps the language token; explicit
        // translate swaps the task token. Fail-closed when the token is absent.
        assert_eq!(
            tokenizer
                .decoder_prefix(
                    7,
                    &WhisperPrefixSpec {
                        language: Some("en"),
                        task: TranscriptionTask::Translate,
                        is_multilingual: true,
                    }
                )
                .unwrap_err(),
            WhisperPrefixError::TranslateTokenMissing
        );
        assert_eq!(
            tokenizer
                .decoder_prefix(
                    7,
                    &WhisperPrefixSpec {
                        language: Some("fr"),
                        task: TranscriptionTask::Transcribe,
                        is_multilingual: true,
                    }
                )
                .unwrap_err(),
            WhisperPrefixError::LanguageTokenMissing {
                language: "fr".to_string()
            }
        );
    }

    #[test]
    fn tokenizer_prefix_resolves_explicit_language_and_translate_tokens() {
        let sot = special_added_token_json(7, "<|startoftranscript|>");
        let en = special_added_token_json(8, "<|en|>");
        let fr = special_added_token_json(11, "<|fr|>");
        let transcribe = special_added_token_json(9, "<|transcribe|>");
        let translate = special_added_token_json(12, "<|translate|>");
        let notimestamps = special_added_token_json(10, "<|notimestamps|>");
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            format!(
                r#"{{
                    "version":"1.0",
                    "added_tokens":[{sot},{en},{fr},{transcribe},{translate},{notimestamps}],
                    "decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},
                    "model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"a":0,"<|startoftranscript|>":7,"<|en|>":8,"<|transcribe|>":9,"<|notimestamps|>":10,"<|fr|>":11,"<|translate|>":12}},"merges":[]}}
                }}"#
            )
            .as_bytes(),
        )
        .unwrap();

        // Non-English transcribe: `<|sot|> <|fr|> <|transcribe|> <|notimestamps|>`.
        assert_eq!(
            tokenizer
                .decoder_prefix(
                    7,
                    &WhisperPrefixSpec {
                        language: Some(" FR "),
                        task: TranscriptionTask::Transcribe,
                        is_multilingual: true,
                    }
                )
                .expect("prefix"),
            vec![7, 11, 9, 10]
        );
        // Translate (default language => en): `<|sot|> <|en|> <|translate|> <|notimestamps|>`.
        assert_eq!(
            tokenizer
                .decoder_prefix(
                    7,
                    &WhisperPrefixSpec {
                        language: None,
                        task: TranscriptionTask::Translate,
                        is_multilingual: true,
                    }
                )
                .expect("prefix"),
            vec![7, 8, 12, 10]
        );
    }

    #[test]
    fn tokenizer_prefix_default_keeps_silent_omit_when_en_token_absent() {
        // A multilingual pack missing `<|en|>` must keep the LEGACY silent-omit
        // on the default path (no <|en|>), not fail closed. Only an explicit
        // non-default request fails closed.
        let sot = special_added_token_json(7, "<|startoftranscript|>");
        let transcribe = special_added_token_json(9, "<|transcribe|>");
        let notimestamps = special_added_token_json(10, "<|notimestamps|>");
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            format!(
                r#"{{
                    "version":"1.0",
                    "added_tokens":[{sot},{transcribe},{notimestamps}],
                    "decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},
                    "model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"a":0,"<|startoftranscript|>":7,"<|transcribe|>":9,"<|notimestamps|>":10}},"merges":[]}}
                }}"#
            )
            .as_bytes(),
        )
        .unwrap();

        assert_eq!(
            tokenizer
                .decoder_prefix(7, &WhisperPrefixSpec::transcribe(true))
                .expect("default prefix"),
            vec![7, 9, 10]
        );
    }

    #[test]
    fn tokenizer_decode_drops_non_special_timestamp_tokens_beyond_notimestamps() {
        // Mirrors multilingual Whisper: `<|notimestamps|>` is special (id 3) and
        // the timestamp tokens at id >= 4 are NOT flagged special in the vocab, so
        // they would otherwise be rendered as literal text. They must be dropped.
        let notimestamps = special_added_token_json(3, "<|notimestamps|>");
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            format!(
                r#"{{
                    "version":"1.0",
                    "added_tokens":[{notimestamps}],
                    "decoder":{{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}},
                    "model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"h":0,"i":1,"<|notimestamps|>":3,"<|0.00|>":4,"<|0.02|>":5}},"merges":[]}}
                }}"#
            )
            .as_bytes(),
        )
        .unwrap();

        // Leading `<|0.00|>` (id 4) and trailing `<|0.02|>` (id 5) are dropped;
        // the `<|notimestamps|>` special (id 3) is dropped; only "hi" remains.
        let decoded = tokenizer.decode_text_token_ids(&[4, 0, 1, 5, 3]).unwrap();
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn tokenizer_decode_keeps_text_tokens_when_no_notimestamps_present() {
        // Without a `<|notimestamps|>` token there is no timestamp range to drop,
        // so all in-vocab text tokens render normally.
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            br#"{
                "version":"1.0",
                "added_tokens":[],
                "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"i":1},"merges":[]}
            }"#,
        )
        .unwrap();

        let decoded = tokenizer.decode_text_token_ids(&[0, 1]).unwrap();
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn tokenizer_decode_roundtrips_minimal_bpe_fixture() {
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            br#"{
                "version":"1.0",
                "added_tokens":[],
                "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"i":1},"merges":[]}
            }"#,
        )
        .unwrap();

        let decoded = tokenizer.decode_text_token_ids(&[0, 1]).unwrap();
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn tokenizer_decode_applies_gpt2_byte_level_mapping() {
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            br#"{
                "version":"1.0",
                "added_tokens":[],
                "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"\u0120hello":0,"\u010Aworld":1},"merges":[]}
            }"#,
        )
        .unwrap();

        let decoded = tokenizer.decode_text_token_ids(&[0, 1]).unwrap();
        assert_eq!(decoded, " hello\nworld");
    }

    #[test]
    fn tokenizer_encodes_prompt_text_with_special_tokens_and_gpt2_bytes() {
        let tokenizer = WhisperTokenizer::from_tokenizer_payload_bytes(
            br#"{
                "version":"1.0",
                "added_tokens":[
                    {"id":9,"content":"<|startoftranscript|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":10,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":11,"content":"<|transcribe|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":12,"content":"<|notimestamps|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
                ],
                "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"\u0120":0,"h":1,"i":2,"\u010A":3,"w":4,"o":5,"r":6,"l":7,"d":8,"<|startoftranscript|>":9,"<|endoftext|>":10,"<|transcribe|>":11,"<|notimestamps|>":12},"merges":[]}
            }"#,
        )
        .unwrap();

        let encoded = tokenizer
            .encode_prompt_text("<|startoftranscript|> hi\nworld")
            .unwrap();
        assert_eq!(encoded, vec![9, 0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn tokenizer_rejects_invalid_json() {
        let error = WhisperTokenizer::from_tokenizer_payload_bytes(b"{not-json}")
            .unwrap_err()
            .to_string();
        assert!(error.contains("tokenizer.json is invalid"), "{error}");
    }

    #[test]
    fn tokenizer_import_requires_merges() {
        let error = tokenizer_import_from_tokenizer_json_bytes(
            br#"{
                "added_tokens":[{"id":10,"content":"<|startoftranscript|>","special":true},{"id":11,"content":"<|endoftext|>","special":true},{"id":12,"content":"<|transcribe|>","special":true},{"id":13,"content":"<|notimestamps|>","special":true}],
                "model":{"vocab":{"a":0,"<|startoftranscript|>":10,"<|endoftext|>":11,"<|transcribe|>":12,"<|notimestamps|>":13},"merges":[]}
            }"#,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("requires non-empty merges"), "{error}");
    }

    #[test]
    fn tokenizer_from_gguf_metadata_requires_whisper_prompt_ids() {
        let mut values = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            crate::GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            crate::GgufMetadataValue::StringArray(vec![
                "<|endoftext|>".to_string(),
                "<|startoftranscript|>".to_string(),
            ]),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            crate::GgufMetadataValue::StringArray(vec!["a b".to_string()]),
        );
        values.insert(
            TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY.to_string(),
            crate::GgufMetadataValue::U32Array(vec![0, 1]),
        );

        let metadata = crate::GgufMetadata::from_values_for_test(values);
        let error = WhisperTokenizer::from_gguf_metadata(&metadata)
            .unwrap_err()
            .to_string();
        assert!(error.contains(TOKENIZER_GGML_SOT_TOKEN_ID_KEY), "{error}");
    }
}
