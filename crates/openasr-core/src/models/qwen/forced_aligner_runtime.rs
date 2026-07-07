//! Stage 3-6 NAR (non-autoregressive) execution pipeline for
//! Qwen3-ForcedAligner-0.6B: metadata parsing, prompt/token assembly, and the
//! end-to-end "mel -> audio encoder -> LLM prefill -> classify head at
//! `<timestamp>` positions -> fix_timestamp -> per-word spans" path.
//!
//! Stage 6 adds the real caller: [`refine_word_timestamps_with_forced_aligner`]
//! is invoked from `api::backend::native_transcribe` when a request opts into
//! `--word-timestamps=aligned` and the capability pack
//! (`models::qwen::forced_aligner_pack`) is installed. `align_forced` and
//! `load_forced_aligner_prepared_assets` stay `pub(crate)` (not `pub`): the
//! one in-crate caller does not need a wider surface, and every
//! execution-graph internal they touch (audio encoder weights, logits head,
//! layer projections, per-stage error enums) stays `pub(crate)` too.

use thiserror::Error;

use crate::ggml_runtime::{GgufMetadata, GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::gpt2_bpe::{build_merge_rank, build_token_to_id, encode_prompt_text};

use super::audio_encoder::{
    Qwen3AsrAudioEncoderError, Qwen3AsrAudioEncoderRuntime, Qwen3AsrAudioEncoderWeights,
    load_qwen3_audio_encoder_weights_from_reader,
};
use super::decode_prompt::Qwen3AsrDecodePrompt;
use super::forced_aligner_align_text::{
    Qwen3ForcedAlignerTextError, fix_timestamp, word_list_for_language,
};
use super::frontend::{
    Qwen3AsrMelFrontendError, load_qwen3_mel_frontend_plan_from_reader,
    qwen3_mel_features_from_prepared_audio,
};
use super::llm_prefill::{Qwen3AsrLlmPrefillInputError, build_qwen3_llm_prefill_input};
use super::llm_transformer::{
    Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmWholeDecoderGraphExecutor,
    load_qwen3_llm_attention_projections_from_reader,
};
use super::logits_head::{
    Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError,
    load_qwen3_llm_logits_head_from_reader_with_output_tensor,
};
use super::prompt_embedding::{
    Qwen3AsrPromptEmbeddingError, build_qwen3_prompt_embeddings_with_audio_splice,
};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::OUTPUT_WEIGHT;
use super::token_embedding::{
    Qwen3AsrTokenEmbeddingError, Qwen3AsrTokenEmbeddingTable,
    load_qwen3_token_embedding_table_from_reader,
};
use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;

/// Same rope theta as the shared qwen3-asr LLM stack (`QWEN_ROPE_THETA` in
/// `batched_decode.rs`); the forced aligner's LM shares that architecture
/// byte-for-byte (see `forced_aligner_import.rs`), so it uses the same value.
const FORCED_ALIGNER_ROPE_THETA: f32 = 1_000_000.0;
const DEFAULT_RMS_NORM_EPSILON: f32 = 1.0e-6;

const KEY_SAMPLE_RATE: &str = "qwen3_forced_aligner.audio.sample_rate_hz";
const KEY_N_MELS: &str = "qwen3_forced_aligner.audio.n_mels";
const KEY_N_FFT: &str = "qwen3_forced_aligner.audio.n_fft";
const KEY_WIN_LENGTH: &str = "qwen3_forced_aligner.audio.win_length";
const KEY_HOP_LENGTH: &str = "qwen3_forced_aligner.audio.hop_length";
const KEY_AUDIO_LAYERS: &str = "qwen3_forced_aligner.audio.n_layers";
const KEY_AUDIO_D_MODEL: &str = "qwen3_forced_aligner.audio.d_model";
const KEY_AUDIO_HEADS: &str = "qwen3_forced_aligner.audio.n_heads";
const KEY_LLM_LAYERS: &str = "qwen3_forced_aligner.llm.n_layers";
const KEY_LLM_D_MODEL: &str = "qwen3_forced_aligner.llm.d_model";
const KEY_LLM_HEADS: &str = "qwen3_forced_aligner.llm.n_heads";
const KEY_LLM_KV_HEADS: &str = "qwen3_forced_aligner.llm.n_kv_heads";
const KEY_LLM_HEAD_DIM: &str = "qwen3_forced_aligner.llm.head_dim";
const KEY_EMBED_VOCAB_SIZE: &str = "qwen3_forced_aligner.llm.embed_vocab_size";
const KEY_CLASSIFY_NUM: &str = "qwen3_forced_aligner.llm.classify_num";
const KEY_LLM_MAX_POSITIONS: &str = "qwen3_forced_aligner.llm.max_positions";
const KEY_AUDIO_START_TOKEN_ID: &str = "qwen3_forced_aligner.audio_start_token_id";
const KEY_AUDIO_END_TOKEN_ID: &str = "qwen3_forced_aligner.audio_end_token_id";
const KEY_AUDIO_PAD_TOKEN_ID: &str = "qwen3_forced_aligner.audio_pad_token_id";
const KEY_TIMESTAMP_TOKEN_ID: &str = "qwen3_forced_aligner.timestamp_token_id";
const KEY_TIMESTAMP_SEGMENT_TIME_MS: &str = "qwen3_forced_aligner.timestamp_segment_time_ms";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

#[derive(Debug, Error)]
pub(crate) enum Qwen3ForcedAlignerRuntimeError {
    #[error("qwen3-forced-aligner runtime is missing required GGUF metadata key '{key}'")]
    MissingMetadata { key: &'static str },
    #[error("qwen3-forced-aligner runtime GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadata { key: &'static str, reason: String },
    #[error("qwen3-forced-aligner tokenizer construction failed: {0}")]
    TokenizerFailed(#[from] crate::NativeAsrError),
    #[error("qwen3-forced-aligner text processing failed: {0}")]
    TextFailed(#[from] Qwen3ForcedAlignerTextError),
    #[error("qwen3-forced-aligner mel frontend failed: {0}")]
    MelFrontendFailed(#[from] Qwen3AsrMelFrontendError),
    #[error("qwen3-forced-aligner audio encoder failed: {0}")]
    AudioEncoderFailed(#[from] Qwen3AsrAudioEncoderError),
    #[error("qwen3-forced-aligner token embedding failed: {0}")]
    TokenEmbeddingFailed(#[from] Qwen3AsrTokenEmbeddingError),
    #[error("qwen3-forced-aligner prompt embedding failed: {0}")]
    PromptEmbeddingFailed(#[from] Qwen3AsrPromptEmbeddingError),
    #[error("qwen3-forced-aligner llm prefill input failed: {0}")]
    LlmPrefillInputFailed(#[from] Qwen3AsrLlmPrefillInputError),
    #[error("qwen3-forced-aligner llm graph failed: {reason}")]
    LlmGraphFailed { reason: String },
    #[error("qwen3-forced-aligner logits head failed: {0}")]
    LogitsHeadFailed(#[from] Qwen3AsrLlmLogitsHeadError),
    #[error(
        "qwen3-forced-aligner expected {expected} <timestamp> positions (2 per word x {word_count} words), found {found}"
    )]
    TimestampPositionCountMismatch {
        expected: usize,
        found: usize,
        word_count: usize,
    },
    #[error("qwen3-forced-aligner GGUF tensor read failed: {0}")]
    TensorRead(#[from] GgufTensorDataReadError),
    #[error("qwen3-forced-aligner llm layer projection load failed: {0}")]
    LlmTransformerFailed(#[from] super::llm_transformer::Qwen3AsrLlmTransformerError),
}

/// Parsed `qwen3_forced_aligner.*` GGUF metadata, with the embedding-table
/// vocab size and the classify-head width kept as two independent fields (see
/// the Stage 1 importer's `embed_vocab_size` / `classify_num` split -- the
/// forced aligner's output head is not tied to the token embedding table, so
/// a single `Qwen3AsrExecutionMetadata.vocab_size` cannot represent both).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3ForcedAlignerRuntimeMetadata {
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub audio_layers: usize,
    pub audio_d_model: usize,
    pub audio_heads: usize,
    pub llm_layers: usize,
    pub llm_d_model: usize,
    pub llm_heads: usize,
    pub llm_kv_heads: usize,
    pub llm_head_dim: usize,
    pub embed_vocab_size: usize,
    pub classify_num: usize,
    pub llm_max_positions: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
    pub timestamp_token_id: u32,
    pub timestamp_segment_time_ms: u32,
}

impl Qwen3ForcedAlignerRuntimeMetadata {
    /// A `Qwen3AsrExecutionMetadata` view sized for the shared token-embedding
    /// table / audio-encoder / LLM-layer loaders, which are architecture
    /// (layers/d_model/heads) driven and generic over `vocab_size` -- they
    /// never special-case the qwen3-asr tied head, so reusing them here with
    /// `vocab_size = embed_vocab_size` is exact, not an approximation.
    /// `eos_token_id`/`pad_token_id` are unused by every loader this view
    /// feeds (audio encoder, mel frontend, token embedding, LLM layer
    /// projections); the aligner's actual EOS-equivalent-free NAR decode
    /// never consults them, so any placeholder value is harmless.
    pub(crate) fn as_embedding_execution_metadata(&self) -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: self.sample_rate_hz,
            n_mels: self.n_mels,
            n_fft: self.n_fft,
            win_length: self.win_length,
            hop_length: self.hop_length,
            audio_layers: self.audio_layers,
            audio_d_model: self.audio_d_model,
            audio_heads: self.audio_heads,
            llm_layers: self.llm_layers,
            llm_d_model: self.llm_d_model,
            llm_heads: self.llm_heads,
            llm_kv_heads: self.llm_kv_heads,
            llm_head_dim: self.llm_head_dim,
            vocab_size: self.embed_vocab_size,
            llm_max_positions: self.llm_max_positions,
            audio_start_token_id: self.audio_start_token_id,
            audio_end_token_id: self.audio_end_token_id,
            audio_pad_token_id: self.audio_pad_token_id,
            eos_token_id: self.audio_end_token_id,
            pad_token_id: self.audio_pad_token_id,
        }
    }

    /// A `Qwen3AsrExecutionMetadata` view sized for the shared logits-head
    /// loader, with `vocab_size = classify_num` so it reads/matmuls against
    /// the 5000-wide `output.weight` classification head instead of a real
    /// vocabulary.
    pub(crate) fn as_classify_execution_metadata(&self) -> Qwen3AsrExecutionMetadata {
        let mut metadata = self.as_embedding_execution_metadata();
        metadata.vocab_size = self.classify_num;
        metadata
    }
}

pub(crate) fn parse_forced_aligner_runtime_metadata(
    metadata: &GgufMetadata,
) -> Result<Qwen3ForcedAlignerRuntimeMetadata, Qwen3ForcedAlignerRuntimeError> {
    Ok(Qwen3ForcedAlignerRuntimeMetadata {
        sample_rate_hz: required_u32(metadata, KEY_SAMPLE_RATE)?,
        n_mels: required_u32(metadata, KEY_N_MELS)? as usize,
        n_fft: required_u32(metadata, KEY_N_FFT)? as usize,
        win_length: required_u32(metadata, KEY_WIN_LENGTH)? as usize,
        hop_length: required_u32(metadata, KEY_HOP_LENGTH)? as usize,
        audio_layers: required_u32(metadata, KEY_AUDIO_LAYERS)? as usize,
        audio_d_model: required_u32(metadata, KEY_AUDIO_D_MODEL)? as usize,
        audio_heads: required_u32(metadata, KEY_AUDIO_HEADS)? as usize,
        llm_layers: required_u32(metadata, KEY_LLM_LAYERS)? as usize,
        llm_d_model: required_u32(metadata, KEY_LLM_D_MODEL)? as usize,
        llm_heads: required_u32(metadata, KEY_LLM_HEADS)? as usize,
        llm_kv_heads: required_u32(metadata, KEY_LLM_KV_HEADS)? as usize,
        llm_head_dim: required_u32(metadata, KEY_LLM_HEAD_DIM)? as usize,
        embed_vocab_size: required_u32(metadata, KEY_EMBED_VOCAB_SIZE)? as usize,
        classify_num: required_u32(metadata, KEY_CLASSIFY_NUM)? as usize,
        llm_max_positions: required_u32(metadata, KEY_LLM_MAX_POSITIONS)? as usize,
        audio_start_token_id: required_u32(metadata, KEY_AUDIO_START_TOKEN_ID)?,
        audio_end_token_id: required_u32(metadata, KEY_AUDIO_END_TOKEN_ID)?,
        audio_pad_token_id: required_u32(metadata, KEY_AUDIO_PAD_TOKEN_ID)?,
        timestamp_token_id: required_u32(metadata, KEY_TIMESTAMP_TOKEN_ID)?,
        timestamp_segment_time_ms: required_u32(metadata, KEY_TIMESTAMP_SEGMENT_TIME_MS)?,
    })
}

fn required_u32(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<u32, Qwen3ForcedAlignerRuntimeError> {
    metadata
        .get_u32(key)
        .ok_or(Qwen3ForcedAlignerRuntimeError::MissingMetadata { key })
}

/// One item of forced-alignment output: a word (or CJK character) span in
/// seconds. Mirrors the reference's `ForcedAlignItem`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ForcedAlignItem {
    pub text: String,
    pub start_time_s: f64,
    pub end_time_s: f64,
}

/// Assembles the aligner's decode prompt directly from BPE-encoded pieces
/// (rather than literally concatenating a Python-style string and
/// re-tokenizing it): the shared `encode_prompt_text` special-token matcher
/// only recognizes `<|...|>`-shaped tokens, not the bare `<timestamp>`
/// marker, so `<timestamp>` positions are injected as already-known token ids
/// instead of being round-tripped through text matching. This produces the
/// same token sequence as the reference for prose without embedded special
/// markers: the reference's literal string join places each word directly
/// adjacent to `<timestamp>` with no separating whitespace, so tokenizing
/// each word independently (as done here) reproduces the same "no leading
/// space" byte-level-BPE segmentation the reference gets from splitting on
/// added-token boundaries before re-tokenizing.
///
/// Returns the assembled `Qwen3AsrDecodePrompt` (for the shared audio-splice
/// prompt-embedding builder) plus the ordered list of `<timestamp>` token
/// positions (two per word: start, end).
pub(crate) fn build_forced_aligner_decode_prompt(
    metadata: &Qwen3ForcedAlignerRuntimeMetadata,
    token_to_id: &std::collections::BTreeMap<String, u32>,
    merge_rank: &std::collections::BTreeMap<String, usize>,
    word_list: &[String],
    audio_frame_count: usize,
) -> Result<(Qwen3AsrDecodePrompt, Vec<usize>), Qwen3ForcedAlignerRuntimeError> {
    let encode = |text: &str| -> Result<Vec<u32>, Qwen3ForcedAlignerRuntimeError> {
        encode_prompt_text(text, token_to_id, merge_rank, "Qwen3-ForcedAligner")
            .map_err(Qwen3ForcedAlignerRuntimeError::TokenizerFailed)
    };

    let mut token_ids = encode("<|audio_start|>")?;
    let audio_pad_start_index = token_ids.len();
    token_ids.extend(std::iter::repeat_n(
        metadata.audio_pad_token_id,
        audio_frame_count,
    ));
    token_ids.extend(encode("<|audio_end|>")?);

    let mut timestamp_positions = Vec::with_capacity(word_list.len() * 2);
    for word in word_list {
        token_ids.extend(encode(word)?);
        timestamp_positions.push(token_ids.len());
        token_ids.push(metadata.timestamp_token_id);
        timestamp_positions.push(token_ids.len());
        token_ids.push(metadata.timestamp_token_id);
    }

    Ok((
        Qwen3AsrDecodePrompt {
            token_ids,
            audio_pad_start_index,
            audio_pad_count: audio_frame_count,
        },
        timestamp_positions,
    ))
}

/// Everything read once from the `.oasr` pack, reusable across multiple
/// `align()` calls against the same pack (mirrors the qwen3-asr prepared
/// runtime's shape, but intentionally not merged with it -- the forced
/// aligner's asset set differs by exactly the classify head vs tied lm_head).
pub(crate) struct Qwen3ForcedAlignerPreparedAssets {
    pub metadata: Qwen3ForcedAlignerRuntimeMetadata,
    pub token_to_id: std::collections::BTreeMap<String, u32>,
    pub merge_rank: std::collections::BTreeMap<String, usize>,
    pub audio_encoder_weights: Qwen3AsrAudioEncoderWeights,
    pub token_embedding_table: Qwen3AsrTokenEmbeddingTable,
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub layer_attention_projections: Vec<Qwen3AsrLlmLayerAttentionProjection>,
}

pub(crate) fn load_forced_aligner_prepared_assets(
    pack_path: &std::path::Path,
) -> Result<Qwen3ForcedAlignerPreparedAssets, Qwen3ForcedAlignerRuntimeError> {
    let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(pack_path).map_err(|error| {
        Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
            key: "<gguf>",
            reason: error.to_string(),
        }
    })?;
    let metadata = parse_forced_aligner_runtime_metadata(&gguf_metadata)?;

    let tokens = gguf_metadata
        .get_string_array(TOKENIZER_GGML_TOKENS_KEY)
        .ok_or(Qwen3ForcedAlignerRuntimeError::MissingMetadata {
            key: TOKENIZER_GGML_TOKENS_KEY,
        })?;
    let merges = gguf_metadata
        .get_string_array(TOKENIZER_GGML_MERGES_KEY)
        .ok_or(Qwen3ForcedAlignerRuntimeError::MissingMetadata {
            key: TOKENIZER_GGML_MERGES_KEY,
        })?;
    let token_to_id = build_token_to_id(tokens, "Qwen3-ForcedAligner")?;
    let merge_rank = build_merge_rank(merges);

    let reader = GgufTensorDataReader::from_path(pack_path)?;
    let embedding_metadata = metadata.as_embedding_execution_metadata();
    let classify_metadata = metadata.as_classify_execution_metadata();

    let audio_encoder_weights =
        load_qwen3_audio_encoder_weights_from_reader(&reader, embedding_metadata)?;
    let token_embedding_table =
        load_qwen3_token_embedding_table_from_reader(&reader, embedding_metadata)?;
    let logits_head = load_qwen3_llm_logits_head_from_reader_with_output_tensor(
        &reader,
        classify_metadata,
        OUTPUT_WEIGHT,
        DEFAULT_RMS_NORM_EPSILON,
    )?;
    let layer_attention_projections =
        load_qwen3_llm_attention_projections_from_reader(&reader, embedding_metadata)?;

    Ok(Qwen3ForcedAlignerPreparedAssets {
        metadata,
        token_to_id,
        merge_rank,
        audio_encoder_weights,
        token_embedding_table,
        logits_head,
        layer_attention_projections,
    })
}

/// Runs the full NAR forced-alignment pipeline for one (audio, text,
/// language) sample against an already-loaded pack: mel -> audio encoder ->
/// prompt assembly -> token embedding + audio splice -> LLM prefill (single
/// forward pass, one row per prompt token) -> classify-head argmax at every
/// `<timestamp>` position -> `fix_timestamp` LIS repair -> per-word spans.
pub(crate) fn align_forced(
    pack_path: &std::path::Path,
    assets: &Qwen3ForcedAlignerPreparedAssets,
    audio_samples_16khz_mono: &[f32],
    text: &str,
    language: &str,
) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
    let word_list = word_list_for_language(text, language)?;

    let reader = GgufTensorDataReader::from_path(pack_path)?;
    let embedding_metadata = assets.metadata.as_embedding_execution_metadata();
    let mel_plan = load_qwen3_mel_frontend_plan_from_reader(&reader, embedding_metadata)?;
    let prepared_audio = GgmlAsrPreparedAudio::mono_16khz(audio_samples_16khz_mono.to_vec());
    let mel_features = qwen3_mel_features_from_prepared_audio(&prepared_audio, &mel_plan)?;

    let mut audio_runtime = Qwen3AsrAudioEncoderRuntime::new(Some(pack_path)).map_err(|error| {
        Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
            reason: format!("audio encoder runtime init failed: {error}"),
        }
    })?;
    let audio_embeddings = audio_runtime
        .encode(
            &assets.audio_encoder_weights,
            embedding_metadata,
            &mel_features,
        )
        .map_err(Qwen3ForcedAlignerRuntimeError::AudioEncoderFailed)?;

    let (decode_prompt, timestamp_positions) = build_forced_aligner_decode_prompt(
        &assets.metadata,
        &assets.token_to_id,
        &assets.merge_rank,
        &word_list,
        audio_embeddings.row_count,
    )?;

    let token_rows = assets
        .token_embedding_table
        .gather_rows(&decode_prompt.token_ids)?;
    let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
        &decode_prompt,
        assets.token_embedding_table.d_model(),
        &token_rows,
        &audio_embeddings.rows,
    )?;
    let prefill_input = build_qwen3_llm_prefill_input(&prompt_embeddings)?;

    let mut whole_decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
        &assets.layer_attention_projections,
        Some(pack_path),
    )
    .map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
        reason: error.to_string(),
    })?;
    let prefill_output = whole_decoder
        .run_prefill(
            &prefill_input.token_major_embeddings,
            prefill_input.token_count,
            FORCED_ALIGNER_ROPE_THETA,
        )
        .map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
            reason: error.to_string(),
        })?;

    let hidden_size = prompt_embeddings.hidden_size;
    let expected_timestamp_positions = word_list.len() * 2;
    if timestamp_positions.len() != expected_timestamp_positions {
        return Err(
            Qwen3ForcedAlignerRuntimeError::TimestampPositionCountMismatch {
                expected: expected_timestamp_positions,
                found: timestamp_positions.len(),
                word_count: word_list.len(),
            },
        );
    }

    let mut raw_timestamps_ms = Vec::with_capacity(timestamp_positions.len());
    for &position in &timestamp_positions {
        let start = position * hidden_size;
        let end = start + hidden_size;
        let hidden_row = &prefill_output.hidden[start..end];
        let bin = assets
            .logits_head
            .compute_top1_token_for_last_hidden(hidden_row)?;
        raw_timestamps_ms
            .push(i64::from(bin) * i64::from(assets.metadata.timestamp_segment_time_ms));
    }

    let fixed_ms = fix_timestamp(&raw_timestamps_ms)?;

    let mut items = Vec::with_capacity(word_list.len());
    for (index, word) in word_list.into_iter().enumerate() {
        let start_ms = fixed_ms[index * 2];
        let end_ms = fixed_ms[index * 2 + 1];
        items.push(ForcedAlignItem {
            text: word,
            start_time_s: round_to_millis(start_ms as f64 / 1000.0),
            end_time_s: round_to_millis(end_ms as f64 / 1000.0),
        });
    }
    Ok(items)
}

/// Matches Python's `round(x, 3)` (round-half-to-even on the underlying f64
/// representation is close enough here: timestamps are integer milliseconds
/// divided by 1000, i.e. always an exact multiple of 0.001, so there is no
/// rounding ambiguity to reproduce).
fn round_to_millis(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// One-shot entry point for the `--word-timestamps=aligned` opt-in refinement
/// tier: loads the pack fresh (no process-wide cache -- this runs at most once
/// per `transcribe` call, the same cost profile as loading the primary ASR
/// pack) and runs the full NAR pipeline. `language` accepts either an ISO
/// 639-1 code or a full name; unsupported languages (Japanese, Korean --
/// see `forced_aligner_align_text`) fail closed with
/// [`Qwen3ForcedAlignerRuntimeError::TextFailed`] rather than mis-tokenizing.
pub(crate) fn refine_word_timestamps_with_forced_aligner(
    pack_path: &std::path::Path,
    audio_samples_16khz_mono: &[f32],
    text: &str,
    language: &str,
) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
    let assets = load_forced_aligner_prepared_assets(pack_path)?;
    align_forced(pack_path, &assets, audio_samples_16khz_mono, text, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stage 5 gate: run the full NAR pipeline end-to-end against the real
    /// Qwen3-ForcedAligner-0.6B checkpoint for both fixtures (`jfk.wav`
    /// English, `zh_sample.wav` Chinese) and compare every word's start/end
    /// against the reference `qwen_asr.inference.qwen3_forced_aligner`
    /// output captured in `tmp/forced-aligner-ref/reference_output.json`
    /// (dev-machine only / gitignored -- see
    /// `tmp/forced-aligner-ref/run_reference.py`). Skips cleanly when the
    /// Stage 0 reference artifacts are absent (e.g. in ordinary CI).
    #[test]
    fn forced_aligner_end_to_end_matches_python_reference_for_jfk_and_zh_sample() {
        use std::path::PathBuf;

        use super::super::forced_aligner_import::{
            Qwen3ForcedAlignerLocalSourceImportRequest,
            convert_local_qwen_forced_aligner_source_to_runtime_pack,
        };
        use super::super::package_import::Qwen3AsrRuntimeQuantizationMode as ForcedAlignerQuantMode;
        use crate::api::audio_io::load_wav_16khz_mono_f32_v0;

        let source_root =
            PathBuf::from("/Volumes/QuintinDocument/hf-cache/qwen3-forced-aligner-0.6b");
        let ref_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tmp/forced-aligner-ref");
        let reference_output_path = ref_dir.join("reference_output.json");
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !source_root.exists() || !reference_output_path.exists() {
            eprintln!(
                "skipping: {} / {} not present (Stage 0 dev-machine reference artifacts)",
                source_root.display(),
                reference_output_path.display()
            );
            return;
        }

        let pack_dir = std::env::temp_dir().join("openasr-forced-aligner-stage5-test");
        let _ = std::fs::create_dir_all(&pack_dir);
        let pack_path = pack_dir.join("qwen3-forced-aligner-0.6b-fp16.oasr");
        let _ = std::fs::remove_file(&pack_path);
        let request = Qwen3ForcedAlignerLocalSourceImportRequest {
            source_root,
            output_root: pack_path.clone(),
            package_id: "qwen3-forced-aligner-0.6b".to_string(),
            package_variant: Some("fp16".to_string()),
            source_name: "Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            source_revision: "test".to_string(),
            license_name: "Apache-2.0".to_string(),
            license_source: "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            quantization: ForcedAlignerQuantMode::Fp16,
        };
        convert_local_qwen_forced_aligner_source_to_runtime_pack(&request)
            .expect("forced-aligner conversion must succeed");

        let assets = load_forced_aligner_prepared_assets(&pack_path).expect("prepared assets");

        let reference_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&reference_output_path).expect("read reference_output.json"),
        )
        .expect("parse reference_output.json");

        struct Case<'a> {
            key: &'a str,
            audio_relpath: &'a str,
            text: &'a str,
            language: &'a str,
        }
        let cases = [
            Case {
                key: "jfk",
                audio_relpath: "fixtures/jfk.wav",
                text: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
                language: "English",
            },
            Case {
                key: "zh_sample",
                audio_relpath: "fixtures/zh_sample.wav",
                text: "今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下",
                language: "Chinese",
            },
        ];

        for case in cases {
            let audio_path = repo_root.join(case.audio_relpath);
            let samples = load_wav_16khz_mono_f32_v0(
                &audio_path,
                "forced-aligner-stage5-test",
                "forced-aligner-stage5-test",
            )
            .expect("load wav");

            let items = align_forced(&pack_path, &assets, &samples, case.text, case.language)
                .expect("align_forced");

            let reference_items = reference_json[case.key]["items"]
                .as_array()
                .unwrap_or_else(|| panic!("reference items array for '{}'", case.key));
            assert_eq!(
                items.len(),
                reference_items.len(),
                "word count mismatch for '{}'",
                case.key
            );

            let mut diffs_ms = Vec::with_capacity(items.len() * 2);
            for (index, (item, reference_item)) in
                items.iter().zip(reference_items.iter()).enumerate()
            {
                let reference_text = reference_item["text"].as_str().unwrap_or_default();
                assert_eq!(
                    item.text, reference_text,
                    "word text mismatch at index {index} for '{}'",
                    case.key
                );
                let reference_start = reference_item["start_time"].as_f64().unwrap_or_default();
                let reference_end = reference_item["end_time"].as_f64().unwrap_or_default();
                diffs_ms.push(((item.start_time_s - reference_start) * 1000.0).abs());
                diffs_ms.push(((item.end_time_s - reference_end) * 1000.0).abs());
                eprintln!(
                    "{} word[{index}] {:?}: ours=({:.3},{:.3}) ref=({:.3},{:.3})",
                    case.key,
                    item.text,
                    item.start_time_s,
                    item.end_time_s,
                    reference_start,
                    reference_end
                );
            }
            diffs_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_diff_ms = diffs_ms[diffs_ms.len() / 2];
            eprintln!(
                "forced_aligner_end_to_end '{}': median start/end diff = {median_diff_ms:.3}ms (n={})",
                case.key,
                diffs_ms.len()
            );
            // Threshold: median per-word start/end diff under one 80ms
            // timestamp-segment bin (the classify head's own resolution), so
            // this catches wiring regressions without being brittle to
            // single-bin rounding differences from fp16 quantization.
            assert!(
                median_diff_ms < 80.0,
                "'{}' diverges from Python reference: median diff {median_diff_ms:.3}ms >= 80ms",
                case.key
            );
        }

        let _ = std::fs::remove_file(&pack_path);
    }
}
