//! Stage 3-6 NAR (non-autoregressive) execution pipeline for
//! Qwen3-ForcedAligner-0.6B: metadata parsing, prompt/token assembly, and the
//! end-to-end "mel -> audio encoder -> LLM prefill -> classify head at
//! `<timestamp>` positions -> fix_timestamp -> per-word spans" path.
//!
//! Stage 6 adds the request-scoped [`Qwen3ForcedAlignerSession`], invoked from
//! `api::backend::native_transcribe` when attribution needs real word anchors
//! or a request opts into `--word-timestamps=aligned` and the capability pack
//! (`models::qwen::forced_aligner_pack`) is installed. `align_forced` and
//! `load_forced_aligner_prepared_assets` stay `pub(crate)` (not `pub`): the
//! one in-crate caller does not need a wider surface, and every
//! execution-graph internal they touch (audio encoder weights, logits head,
//! layer projections, per-stage error enums) stays `pub(crate)` too.

use thiserror::Error;

#[cfg(test)]
use crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index_from_source;
use crate::ggml_runtime::{
    GgmlFlashAttentionPrecision, GgufMetadata, GgufRuntimeSourcePreflight, GgufTensorDataReadError,
    build_runtime_tensor_reader_from_preflight,
};
use crate::models::gpt2_bpe::{build_merge_rank, build_token_to_id, encode_prompt_text};
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_quant_audit::runtime_tensor_index_q8_floor_violations,
    pack_verifier::{PackCandidate, PackRoute, PackVerifier, VerifiedPack},
};

use super::audio_encoder::{
    Qwen3AsrAudioEncoderError, Qwen3AsrAudioEncoderRuntime, Qwen3AsrAudioEncoderWeights,
    load_qwen3_audio_encoder_weights_from_reader,
};
use super::decode_prompt::Qwen3AsrDecodePrompt;
use super::forced_aligner_align_text::{
    Qwen3ForcedAlignerTextError, fix_timestamp, word_list_for_language,
};
use super::frontend::{
    Qwen3AsrMelFrontendError, Qwen3AsrMelFrontendPlan, load_qwen3_mel_frontend_plan_from_reader,
    qwen3_mel_features_from_prepared_audio,
};
use super::llm_prefill::{Qwen3AsrLlmPrefillInputError, build_qwen3_llm_prefill_input};
use super::llm_transformer::{Qwen3AsrLlmWholeDecoderGraphExecutor, QwenWholeDecoderPlan};
use super::logits_head::{
    Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError,
    load_qwen3_llm_logits_head_from_reader_with_output_tensor,
};
use super::prompt_embedding::{
    Qwen3AsrPromptEmbeddingError, build_qwen3_prompt_embeddings_with_audio_splice,
};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::OUTPUT_WEIGHT;
use super::token_embedding::load_qwen3_token_embedding_table_from_reader;
use crate::models::ggml_asr_executor::GgmlAsrPreparedAudioView;
use crate::models::mapped_token_embedding::{MappedTokenEmbeddingError, MappedTokenEmbeddingTable};

/// Same rope theta as the shared qwen3-asr LLM stack (`QWEN_ROPE_THETA` in
/// `batched_decode.rs`); the forced aligner's LM shares that architecture
/// byte-for-byte (see `forced_aligner_import.rs`), so it uses the same value.
const FORCED_ALIGNER_ROPE_THETA: f32 = 1_000_000.0;
const DEFAULT_RMS_NORM_EPSILON: f32 = 1.0e-6;
/// Upper bound for one timestamp-classification graph. At the current 5k
/// vocabulary this keeps each f32 logits matrix at or below 1.25 MiB while
/// still amortizing graph construction and GPU submission across many words.
/// The bound is independent of transcript length.
const FORCED_ALIGNER_LOGITS_BATCH_ROWS: usize = 64;
// A <=2% odds advantage is below the cross-backend reduction-order envelope
// measured for this Q8 classification head. Treat such candidates as a
// numerical tie and choose the later timestamp bin deterministically. The
// logits themselves still run on the selected backend; this is only the
// ordered-class decision rule that prevents a near-tie from becoming a
// multi-bin word-boundary jump.
const FORCED_ALIGNER_TIMESTAMP_TIE_LOGIT_DELTA: f32 = 0.019_802_627;

fn stable_timestamp_bin(logits: &[f32]) -> Option<u32> {
    if logits.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let best = logits.iter().copied().reduce(f32::max)?;
    let floor = best - FORCED_ALIGNER_TIMESTAMP_TIE_LOGIT_DELTA;
    logits.iter().enumerate().rev().find_map(|(index, &value)| {
        (value >= floor)
            .then(|| u32::try_from(index).ok())
            .flatten()
    })
}

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
    TokenEmbeddingFailed(#[from] MappedTokenEmbeddingError),
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
    #[error("qwen3-forced-aligner timestamp hidden batch allocation overflowed")]
    TimestampHiddenBatchOverflow,
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

/// Cheap install-time contract probe for `aux_pack_registry`: proves the pack
/// carries every `qwen3_forced_aligner.*` scalar key
/// [`parse_forced_aligner_runtime_metadata`] requires, plus the BPE
/// tokenizer's `tokenizer.ggml.{tokens,merges}` arrays -- the two metadata
/// surfaces [`load_forced_aligner_prepared_assets`] reads before it ever
/// touches tensor data. Metadata-only (no tensor materialization), matching
/// every other builtin family's install-time `runtime_contract` parser
/// (whisper/moonshine/parakeet in `api::backend::native`, FireRedPunc here in
/// `aux_pack_registry`): a bare-bones pack that only carries generic
/// adapter-selection metadata must fail closed here rather than installing
/// successfully and only failing the first time `--word-timestamps=aligned`
/// actually loads it.
pub(crate) fn validate_forced_aligner_runtime_pack_contract(
    metadata: &GgufMetadata,
) -> Result<(), Qwen3ForcedAlignerRuntimeError> {
    parse_forced_aligner_runtime_metadata(metadata)?;
    if metadata
        .get_string_array(TOKENIZER_GGML_TOKENS_KEY)
        .is_none()
    {
        return Err(Qwen3ForcedAlignerRuntimeError::MissingMetadata {
            key: TOKENIZER_GGML_TOKENS_KEY,
        });
    }
    if metadata
        .get_string_array(TOKENIZER_GGML_MERGES_KEY)
        .is_none()
    {
        return Err(Qwen3ForcedAlignerRuntimeError::MissingMetadata {
            key: TOKENIZER_GGML_MERGES_KEY,
        });
    }
    Ok(())
}

/// One item of forced-alignment output: a word (or CJK character) span in
/// seconds. Mirrors the reference's `ForcedAlignItem`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ForcedAlignItem {
    pub text: String,
    pub start_time_s: f64,
    pub end_time_s: f64,
}

/// Honest request-local milestones exposed by the NAR alignment pipeline.
///
/// The audio encoder and decoder each execute as one backend graph, so there
/// is no truthful layer-level signal while either graph is in flight. These
/// events sit only on already-existing ownership boundaries and in the
/// timestamp-head loop; observers never enter model math or outlive a single
/// synchronous [`align_forced_with_progress`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForcedAlignerProgressEvent {
    MelReady,
    AudioEncodingStarted,
    AudioEncoded,
    PromptPrepared,
    DecoderPrefillStarted,
    DecoderPrefilled,
    TimestampLogitsStarted { total: usize },
    TimestampLogits { completed: usize, total: usize },
    Finalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ForcedAlignerStageBackends {
    audio: crate::ggml_runtime::GgmlCpuGraphBackend,
    decoder: crate::ggml_runtime::GgmlCpuGraphBackend,
    logits: crate::ggml_runtime::GgmlCpuGraphBackend,
}

impl ForcedAlignerStageBackends {
    const fn uniform(backend: crate::ggml_runtime::GgmlCpuGraphBackend) -> Self {
        Self {
            audio: backend,
            decoder: backend,
            logits: backend,
        }
    }

    const fn gpu_audio_hybrid() -> Self {
        Self {
            audio: crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            decoder: crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            logits: crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        }
    }
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
    pub mel_frontend_plan: Qwen3AsrMelFrontendPlan,
    pub audio_encoder_weights: Qwen3AsrAudioEncoderWeights,
    pub token_embedding_table: MappedTokenEmbeddingTable,
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub decoder_plan: QwenWholeDecoderPlan,
}

pub(crate) fn load_forced_aligner_prepared_assets(
    preflight: &GgufRuntimeSourcePreflight,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<Qwen3ForcedAlignerPreparedAssets, Qwen3ForcedAlignerRuntimeError> {
    let runtime_source = &preflight.runtime_source;
    let gguf_metadata = preflight.metadata.as_ref();
    let metadata = parse_forced_aligner_runtime_metadata(gguf_metadata)?;

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

    let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
            key: "<gguf preflight>",
            reason: error.to_string(),
        }
    })?;
    let embedding_metadata = metadata.as_embedding_execution_metadata();
    let classify_metadata = metadata.as_classify_execution_metadata();

    let mel_frontend_plan = load_qwen3_mel_frontend_plan_from_reader(&reader, embedding_metadata)?;
    let audio_encoder_weights =
        load_qwen3_audio_encoder_weights_from_reader(&reader, embedding_metadata)?;
    let token_embedding_table =
        load_qwen3_token_embedding_table_from_reader(&reader, embedding_metadata)?;
    let logits_head = load_qwen3_llm_logits_head_from_reader_with_output_tensor(
        &reader,
        runtime_source,
        classify_metadata,
        OUTPUT_WEIGHT,
        DEFAULT_RMS_NORM_EPSILON,
        backend,
    )?;
    let decoder_plan = QwenWholeDecoderPlan::for_qwen3_asr(&reader, embedding_metadata)?;

    Ok(Qwen3ForcedAlignerPreparedAssets {
        metadata,
        token_to_id,
        merge_rank,
        mel_frontend_plan,
        audio_encoder_weights,
        token_embedding_table,
        logits_head,
        decoder_plan,
    })
}

/// Runs the full NAR forced-alignment pipeline for one (audio, text,
/// language) sample against an already-loaded pack: mel -> audio encoder ->
/// prompt assembly -> token embedding + audio splice -> LLM prefill (single
/// forward pass, one row per prompt token) -> classify-head argmax at every
/// `<timestamp>` position -> `fix_timestamp` LIS repair -> per-word spans.
#[cfg(test)]
pub(crate) fn align_forced(
    preflight: &GgufRuntimeSourcePreflight,
    assets: &Qwen3ForcedAlignerPreparedAssets,
    audio_samples_16khz_mono: crate::PcmSlice,
    text: &str,
    language: &str,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
    align_forced_with_progress(
        preflight,
        assets,
        audio_samples_16khz_mono,
        text,
        language,
        backend,
        None,
    )
}

#[cfg(test)]
pub(crate) fn align_forced_with_progress(
    preflight: &GgufRuntimeSourcePreflight,
    assets: &Qwen3ForcedAlignerPreparedAssets,
    audio_samples_16khz_mono: crate::PcmSlice,
    text: &str,
    language: &str,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    observer: Option<&mut dyn FnMut(ForcedAlignerProgressEvent)>,
) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
    align_forced_with_stage_backends(
        preflight,
        assets,
        audio_samples_16khz_mono,
        text,
        language,
        ForcedAlignerStageBackends::uniform(backend),
        observer,
    )
}

fn align_forced_with_stage_backends(
    preflight: &GgufRuntimeSourcePreflight,
    assets: &Qwen3ForcedAlignerPreparedAssets,
    audio_samples_16khz_mono: crate::PcmSlice,
    text: &str,
    language: &str,
    backends: ForcedAlignerStageBackends,
    mut observer: Option<&mut dyn FnMut(ForcedAlignerProgressEvent)>,
) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
    let mut report = |event| {
        if let Some(observer) = observer.as_deref_mut() {
            observer(event);
        }
    };
    let word_list = word_list_for_language(text, language)?;

    let embedding_metadata = assets.metadata.as_embedding_execution_metadata();
    let prepared_audio = forced_aligner_prepared_audio(audio_samples_16khz_mono);
    let mel_features =
        qwen3_mel_features_from_prepared_audio(&prepared_audio, &assets.mel_frontend_plan)?;
    report(ForcedAlignerProgressEvent::MelReady);

    report(ForcedAlignerProgressEvent::AudioEncodingStarted);
    let audio_runtime_result = if matches!(
        backends.audio,
        crate::ggml_runtime::GgmlCpuGraphBackend::Gpu
    ) {
        Qwen3AsrAudioEncoderRuntime::new_from_preflight_with_flash_attention(
            preflight,
            backends.audio,
            false,
        )
    } else {
        Qwen3AsrAudioEncoderRuntime::new_from_preflight(preflight, backends.audio)
    };
    let mut audio_runtime =
        audio_runtime_result.map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
            reason: format!("audio encoder runtime init failed: {error}"),
        })?;
    let audio_embeddings = audio_runtime
        .encode(
            &assets.audio_encoder_weights,
            embedding_metadata,
            &mel_features,
        )
        .map_err(Qwen3ForcedAlignerRuntimeError::AudioEncoderFailed)?;
    report(ForcedAlignerProgressEvent::AudioEncoded);
    // The encoder graph and mel input are not needed by the LLM stage. Release
    // and drop them before constructing its much larger graph so the two
    // stages do not overlap in the request's peak working set.
    audio_runtime
        .release_transient_compute_memory()
        .map_err(Qwen3ForcedAlignerRuntimeError::AudioEncoderFailed)?;
    drop(audio_runtime);
    drop(mel_features);

    let (decode_prompt, timestamp_positions) = build_forced_aligner_decode_prompt(
        &assets.metadata,
        &assets.token_to_id,
        &assets.merge_rank,
        &word_list,
        audio_embeddings.row_count,
    )?;

    report(ForcedAlignerProgressEvent::DecoderPrefillStarted);
    let mut whole_decoder = if backends.decoder.is_gpu_class() {
        let device_embedding = assets
            .token_embedding_table
            .device_graph_spec()
            .ok_or_else(|| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                reason: "accelerated forced aligner requires a device-bindable token embedding"
                    .to_string(),
            })?;
        Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_token_embedding(
            &assets.decoder_plan,
            preflight,
            device_embedding,
            backends.decoder,
        )
    } else {
        Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight(
            &assets.decoder_plan,
            preflight,
            backends.decoder,
        )
    }
    .map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
        reason: error.to_string(),
    })?;
    if backends.decoder.is_gpu_class() {
        // Forced alignment turns a single near-tie attention argmax into an
        // absolute timestamp choice. Request ggml's precise attention contract
        // for this transient decoder on every GPU; ordinary Qwen-family decode
        // keeps the faster backend default.
        whole_decoder.set_flash_attention_precision(GgmlFlashAttentionPrecision::F32);
    }
    let hidden_size = assets.token_embedding_table.d_model();
    let audio_pad_end = decode_prompt
        .audio_pad_start_index
        .checked_add(decode_prompt.audio_pad_count)
        .ok_or_else(|| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
            reason: "forced aligner audio pad position overflowed".to_string(),
        })?;
    let audio_positions = (decode_prompt.audio_pad_start_index..audio_pad_end).collect::<Vec<_>>();
    let prefill_input = if backends.decoder.is_gpu_class() {
        let token_major_embeddings = whole_decoder
            .materialize_token_prompt_on_device(
                &decode_prompt.token_ids,
                &audio_embeddings.rows,
                &audio_positions,
            )
            .map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                reason: error.to_string(),
            })?
            .ok_or_else(|| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                reason: "accelerated forced aligner did not materialize token rows on device"
                    .to_string(),
            })?;
        super::llm_prefill::Qwen3AsrLlmPrefillInput {
            token_count: decode_prompt.token_ids.len(),
            hidden_size,
            token_major_embeddings,
        }
    } else {
        let token_rows = assets
            .token_embedding_table
            .gather_rows(&decode_prompt.token_ids)?;
        let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
            &decode_prompt,
            hidden_size,
            token_rows,
            &audio_embeddings.rows,
        )?;
        build_qwen3_llm_prefill_input(prompt_embeddings)?
    };
    drop(audio_embeddings);
    report(ForcedAlignerProgressEvent::PromptPrepared);
    let prefill_output = whole_decoder
        .run_stateless_prefill(
            &prefill_input.token_major_embeddings,
            prefill_input.token_count,
            FORCED_ALIGNER_ROPE_THETA,
        )
        .map_err(|error| Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
            reason: error.to_string(),
        })?;
    report(ForcedAlignerProgressEvent::DecoderPrefilled);

    // `prefill_output.hidden` is ordinary host-owned output. Drop the much
    // larger decoder graph/weight runtime before materializing the logits-head
    // runtime so their transient GPU allocations never overlap.
    drop(whole_decoder);
    drop(prefill_input);
    let mut logits_runtime = assets.logits_head.new_runtime(backends.logits)?;
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

    // Process bounded row batches instead of materializing a transcript-sized
    // [vocab, timestamp_count] logits graph. Every backend returns the bounded
    // logits rows so the same ordered near-tie rule is applied everywhere.
    // A device first-max is not sufficient here: a numerically insignificant
    // reduction-order difference can otherwise move a word boundary by many
    // timestamp bins.
    report(ForcedAlignerProgressEvent::TimestampLogitsStarted {
        total: timestamp_positions.len(),
    });
    let mut raw_timestamps_ms = Vec::with_capacity(timestamp_positions.len());
    let max_hidden_values = hidden_size
        .checked_mul(FORCED_ALIGNER_LOGITS_BATCH_ROWS)
        .ok_or(Qwen3ForcedAlignerRuntimeError::TimestampHiddenBatchOverflow)?;
    let mut timestamp_hidden = Vec::with_capacity(max_hidden_values);
    for positions in timestamp_positions.chunks(FORCED_ALIGNER_LOGITS_BATCH_ROWS) {
        timestamp_hidden.clear();
        for &position in positions {
            let start = position.checked_mul(hidden_size).ok_or(
                Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                    reason: "timestamp hidden-row offset overflow".to_string(),
                },
            )?;
            let end = start.checked_add(hidden_size).ok_or(
                Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                    reason: "timestamp hidden-row end overflow".to_string(),
                },
            )?;
            let row = prefill_output.hidden.get(start..end).ok_or(
                Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                    reason: "timestamp hidden-row position exceeds prefill output".to_string(),
                },
            )?;
            timestamp_hidden.extend_from_slice(row);
        }
        let logits = logits_runtime.compute_logits_for_hidden_rows(
            &assets.logits_head,
            &timestamp_hidden,
            positions.len(),
        )?;
        for row in logits.chunks_exact(assets.metadata.classify_num) {
            let bin = stable_timestamp_bin(row).ok_or_else(|| {
                Qwen3ForcedAlignerRuntimeError::LlmGraphFailed {
                    reason: "timestamp classification produced an empty logits row".to_string(),
                }
            })?;
            raw_timestamps_ms
                .push(i64::from(bin) * i64::from(assets.metadata.timestamp_segment_time_ms));
        }
        report(ForcedAlignerProgressEvent::TimestampLogits {
            completed: raw_timestamps_ms.len(),
            total: timestamp_positions.len(),
        });
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
    report(ForcedAlignerProgressEvent::Finalized);
    Ok(items)
}

fn forced_aligner_prepared_audio(
    audio_samples_16khz_mono: crate::PcmSlice,
) -> GgmlAsrPreparedAudioView<'static> {
    GgmlAsrPreparedAudioView::mono_16khz_shared(audio_samples_16khz_mono)
}

/// Matches Python's `round(x, 3)` (round-half-to-even on the underlying f64
/// representation is close enough here: timestamps are integer milliseconds
/// divided by 1000, i.e. always an exact multiple of 0.001, so there is no
/// rounding ambiguity to reproduce).
fn round_to_millis(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// One request-scoped forced-aligner session. Pack metadata and immutable
/// weights are loaded once, while each bounded audio/text item owns and drops
/// its graph runtime before the next item starts. That keeps peak graph memory
/// proportional to the largest ASR segment rather than the whole recording.
pub(crate) struct Qwen3ForcedAlignerSession {
    verified: VerifiedPack,
    assets: Qwen3ForcedAlignerPreparedAssets,
    backends: ForcedAlignerStageBackends,
}

pub(crate) fn validate_forced_aligner_quantization_contract(
    tensor_index: &crate::ggml_runtime::GgufTensorIndex,
) -> Result<(), Qwen3ForcedAlignerRuntimeError> {
    let violations = runtime_tensor_index_q8_floor_violations(
        super::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        tensor_index,
    )
    .map_err(|error| Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
        key: "<quantization floor>",
        reason: error.to_string(),
    })?;
    if let Some(first) = violations.first() {
        return Err(Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
            key: "<quantization floor>",
            reason: format!(
                "forced-aligner packs require Q8_0 or higher for every quantizable matrix; pack has {} violation(s), first tensor '{}' is {}",
                violations.len(),
                first.tensor,
                crate::models::pack_quant_audit::ggml_type_name(first.ggml_type),
            ),
        });
    }
    Ok(())
}

pub(crate) fn verify_forced_aligner_pack(
    pack_path: &std::path::Path,
) -> Result<VerifiedPack, Qwen3ForcedAlignerRuntimeError> {
    let verified = PackVerifier
        .verify_candidate(PackCandidate::new(pack_path))
        .map_err(|error| Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
            key: "<pack verifier>",
            reason: error.to_string(),
        })?;
    if !matches!(
        verified.route(),
        PackRoute::Aux {
            kind: AuxPackKind::ForcedAlignment,
            ..
        }
    ) {
        return Err(Qwen3ForcedAlignerRuntimeError::InvalidMetadata {
            key: "<pack route>",
            reason: format!(
                "expected auxiliary forced-alignment pack, got {:?}",
                verified.route()
            ),
        });
    }
    Ok(verified)
}

impl Qwen3ForcedAlignerSession {
    #[cfg(test)]
    pub(crate) fn load(
        pack_path: &std::path::Path,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, Qwen3ForcedAlignerRuntimeError> {
        Self::load_verified(verify_forced_aligner_pack(pack_path)?, backend)
    }

    pub(crate) fn load_verified(
        verified: VerifiedPack,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, Qwen3ForcedAlignerRuntimeError> {
        Self::load_verified_with_stage_backends(
            verified,
            ForcedAlignerStageBackends::uniform(backend),
        )
    }

    pub(crate) fn load_verified_gpu_audio_hybrid(
        verified: VerifiedPack,
    ) -> Result<Self, Qwen3ForcedAlignerRuntimeError> {
        Self::load_verified_with_stage_backends(
            verified,
            ForcedAlignerStageBackends::gpu_audio_hybrid(),
        )
    }

    fn load_verified_with_stage_backends(
        verified: VerifiedPack,
        backends: ForcedAlignerStageBackends,
    ) -> Result<Self, Qwen3ForcedAlignerRuntimeError> {
        let assets = load_forced_aligner_prepared_assets(verified.preflight(), backends.logits)?;
        Ok(Self {
            verified,
            assets,
            backends,
        })
    }

    pub(crate) fn align(
        &self,
        audio_samples_16khz_mono: crate::PcmSlice,
        text: &str,
        language: &str,
    ) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
        align_forced_with_stage_backends(
            self.verified.preflight(),
            &self.assets,
            audio_samples_16khz_mono,
            text,
            language,
            self.backends,
            None,
        )
    }

    pub(crate) fn align_with_progress(
        &self,
        audio_samples_16khz_mono: crate::PcmSlice,
        text: &str,
        language: &str,
        observer: &mut dyn FnMut(ForcedAlignerProgressEvent),
    ) -> Result<Vec<ForcedAlignItem>, Qwen3ForcedAlignerRuntimeError> {
        align_forced_with_stage_backends(
            self.verified.preflight(),
            &self.assets,
            audio_samples_16khz_mono,
            text,
            language,
            self.backends,
            Some(observer),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ForcedAlignerTestBackend {
        Cpu,
        Metal,
        GenericGpu,
        Cuda,
        Vulkan,
    }

    impl ForcedAlignerTestBackend {
        fn from_env() -> Self {
            match std::env::var("OPENASR_AUX_BENCH_BACKEND")
                .unwrap_or_else(|_| "cpu".to_string())
                .as_str()
            {
                "cpu" => Self::Cpu,
                "metal" => Self::Metal,
                "gpu" => Self::GenericGpu,
                "cuda" => Self::Cuda,
                "vulkan" => Self::Vulkan,
                value => panic!("unsupported OPENASR_AUX_BENCH_BACKEND '{value}'"),
            }
        }

        const fn graph_backend(self) -> crate::ggml_runtime::GgmlCpuGraphBackend {
            match self {
                Self::Cpu => crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
                Self::Metal => crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
                Self::GenericGpu | Self::Cuda | Self::Vulkan => {
                    crate::ggml_runtime::GgmlCpuGraphBackend::Gpu
                }
            }
        }

        const fn provider_label(self) -> &'static str {
            match self {
                Self::Cpu => "cpu",
                Self::Metal => "metal",
                Self::GenericGpu => "gpu-auto",
                Self::Cuda => "cuda",
                Self::Vulkan => "vulkan",
            }
        }

        const fn stage_topology(self) -> &'static str {
            match self {
                Self::Cpu | Self::Metal | Self::GenericGpu => "uniform",
                Self::Cuda => "cuda-audio-cpu-decoder-cpu-logits",
                Self::Vulkan => "vulkan-audio-cpu-decoder-cpu-logits",
            }
        }

        fn install_exact_route(self) -> Option<crate::ggml_runtime::RequestBackendOverrideGuard> {
            let provider = match self {
                Self::Cuda => crate::device::execution_route::ExecutionProvider::Cuda,
                Self::Vulkan => crate::device::execution_route::ExecutionProvider::Vulkan,
                Self::Cpu | Self::Metal | Self::GenericGpu => return None,
            };
            let route = crate::device::execution_route::enumerate_compute_devices_from_ggml(
                &crate::ggml_runtime::ggml_available_devices(),
            )
            .into_iter()
            .find(|device| device.provider == provider)
            .unwrap_or_else(|| {
                panic!("requested forced-aligner provider {provider:?} is unavailable")
            })
            .to_resolved_route();
            Some(crate::ggml_runtime::install_request_backend_override(Some(
                crate::ggml_runtime::RequestBackendPreference::Exact(route),
            )))
        }

        fn resolved_backend_name(self) -> String {
            crate::ggml_runtime::GgmlCpuGraphConfig::resolve_backend_name_for(self.graph_backend())
                .expect("resolve benchmark backend name")
        }

        fn load_session(
            self,
            pack: &std::path::Path,
        ) -> Result<Qwen3ForcedAlignerSession, Qwen3ForcedAlignerRuntimeError> {
            let verified = verify_forced_aligner_pack(pack)?;
            match self {
                Self::Cuda | Self::Vulkan => {
                    Qwen3ForcedAlignerSession::load_verified_gpu_audio_hybrid(verified)
                }
                Self::Cpu | Self::Metal | Self::GenericGpu => {
                    Qwen3ForcedAlignerSession::load_verified(verified, self.graph_backend())
                }
            }
        }
    }

    #[test]
    fn stable_timestamp_bin_keeps_clear_winner_and_resolves_near_tie_late() {
        assert_eq!(stable_timestamp_bin(&[]), None);
        assert_eq!(stable_timestamp_bin(&[1.0, 1.05, 1.0]), Some(1));
        assert_eq!(stable_timestamp_bin(&[1.0, 1.01, 1.0]), Some(2));
        assert_eq!(stable_timestamp_bin(&[1.0, 1.01, 1.01]), Some(2));
        assert_eq!(stable_timestamp_bin(&[1.0, f32::NAN, 1.0]), None);
        assert_eq!(stable_timestamp_bin(&[1.0, f32::INFINITY]), None);
    }

    #[test]
    fn discrete_gpu_hybrid_accelerates_only_the_audio_encoder() {
        let backends = ForcedAlignerStageBackends::gpu_audio_hybrid();
        assert_eq!(
            backends.audio,
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
        );
        assert_eq!(
            backends.decoder,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(
            backends.logits,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        );
    }

    fn quant_floor_index(
        output_type: i32,
        token_embedding_type: i32,
        decoder_type: i32,
    ) -> crate::ggml_runtime::GgufTensorIndex {
        crate::ggml_runtime::GgufTensorIndex::from_snapshot(
            crate::ggml_runtime::GgufTensorIndexSnapshot {
                path: "/nonexistent/forced-aligner-policy.oasr".into(),
                data_section_offset_bytes: 0,
                tensors: vec![
                    crate::ggml_runtime::GgufTensorMetadata {
                        name: "output.weight".to_string(),
                        dims: vec![1024, 5000],
                        ggml_type: output_type,
                        type_name: "synthetic".to_string(),
                        size_bytes: 0,
                        offset_bytes: 0,
                    },
                    crate::ggml_runtime::GgufTensorMetadata {
                        name: "token_embd.weight".to_string(),
                        dims: vec![1024, 152_064],
                        ggml_type: token_embedding_type,
                        type_name: "synthetic".to_string(),
                        size_bytes: 0,
                        offset_bytes: 0,
                    },
                    crate::ggml_runtime::GgufTensorMetadata {
                        name: "blk.0.ffn_gate.weight".to_string(),
                        dims: vec![1024, 3072],
                        ggml_type: decoder_type,
                        type_name: "synthetic".to_string(),
                        size_bytes: 0,
                        offset_bytes: 0,
                    },
                ],
            },
        )
        .expect("valid synthetic tensor index")
    }

    #[test]
    fn forced_aligner_pack_contract_requires_q8_for_every_matrix() {
        let q8 = crate::models::pack_quant_audit::GGML_TYPE_Q8_0 as i32;
        let q4 = crate::models::pack_quant_audit::GGML_TYPE_Q4_K as i32;
        validate_forced_aligner_quantization_contract(&quant_floor_index(q8, q8, q8))
            .expect("current Q8 contract must remain eligible on every backend");
        assert!(
            validate_forced_aligner_quantization_contract(&quant_floor_index(q4, q8, q8)).is_err()
        );
        assert!(
            validate_forced_aligner_quantization_contract(&quant_floor_index(q8, q4, q8)).is_err()
        );
        assert!(
            validate_forced_aligner_quantization_contract(&quant_floor_index(q8, q8, q4)).is_err()
        );
    }

    #[test]
    fn forced_aligner_frontend_keeps_the_callers_pcm_backing() {
        let audio = crate::PcmBuffer::from_vec(vec![0.0; 32_000]);
        let identity = audio.backing_identity();
        let prepared = forced_aligner_prepared_audio(audio.full_slice());

        assert_eq!(prepared.samples_f32.backing_identity(), identity);
        assert_eq!(prepared.samples_f32.as_ptr(), audio.as_ptr());
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_FORCED_ALIGNER_PACK, OPENASR_AUX_BENCH_AUDIO, and OPENASR_AUX_BENCH_TEXT"]
    fn forced_aligner_aux_audio_benchmark() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FORCED_ALIGNER_PACK",
            "Qwen3 forced-aligner runtime pack",
        )
        .expect("OPENASR_FORCED_ALIGNER_PACK");
        let audio = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_AUDIO",
            "private auxiliary-model benchmark audio",
        )
        .expect("OPENASR_AUX_BENCH_AUDIO");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model benchmark transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let text = std::fs::read_to_string(text_path).expect("read benchmark transcript");
        let text = text.trim();
        assert!(!text.is_empty(), "benchmark transcript must not be empty");
        let language =
            std::env::var("OPENASR_AUX_BENCH_LANGUAGE").unwrap_or_else(|_| "Chinese".to_string());
        let test_backend = ForcedAlignerTestBackend::from_env();
        let _route_guard = test_backend.install_exact_route();
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &audio,
            "forced-aligner auxiliary benchmark",
            "forced-aligner auxiliary benchmark",
        )
        .expect("load benchmark audio");
        let pcm = crate::PcmBuffer::from_vec(samples);
        let audio_seconds = pcm.len() as f64 / 16_000.0;
        let backend_name = test_backend.resolved_backend_name();
        let session = test_backend.load_session(&pack).expect("load aligner");
        let run = || {
            session
                .align(pcm.full_slice(), text, &language)
                .expect("align benchmark audio")
        };

        let mut items = run();
        let mut seconds = Vec::with_capacity(5);
        let mut phase_samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(5));
        for _ in 0..5 {
            let started = std::time::Instant::now();
            let mut phase_seconds = [None; 6];
            let mut observer = |event| {
                let phase = match event {
                    ForcedAlignerProgressEvent::MelReady => Some(0),
                    ForcedAlignerProgressEvent::AudioEncoded => Some(1),
                    ForcedAlignerProgressEvent::PromptPrepared => Some(2),
                    ForcedAlignerProgressEvent::DecoderPrefilled => Some(3),
                    ForcedAlignerProgressEvent::TimestampLogits { completed, total }
                        if completed == total =>
                    {
                        Some(4)
                    }
                    ForcedAlignerProgressEvent::Finalized => Some(5),
                    ForcedAlignerProgressEvent::AudioEncodingStarted
                    | ForcedAlignerProgressEvent::DecoderPrefillStarted
                    | ForcedAlignerProgressEvent::TimestampLogitsStarted { .. }
                    | ForcedAlignerProgressEvent::TimestampLogits { .. } => None,
                };
                if let Some(phase) = phase {
                    phase_seconds[phase] = Some(started.elapsed().as_secs_f64());
                }
            };
            items = session
                .align_with_progress(pcm.full_slice(), text, &language, &mut observer)
                .expect("align benchmark audio with progress");
            seconds.push(started.elapsed().as_secs_f64());
            for (sample, phase_seconds) in phase_samples.iter_mut().zip(phase_seconds) {
                sample.push(phase_seconds.expect("benchmark must report every phase"));
            }
        }
        assert!(!items.is_empty(), "benchmark alignment must emit items");
        let mut output_bytes = Vec::new();
        for item in &items {
            output_bytes.extend_from_slice(item.text.as_bytes());
            output_bytes.push(0);
            output_bytes.extend_from_slice(&item.start_time_s.to_le_bytes());
            output_bytes.extend_from_slice(&item.end_time_s.to_le_bytes());
        }
        let output_sha256 = crate::testing::benchmark_sha256_bytes([output_bytes]);
        let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
        let phase_cumulative_median_seconds =
            phase_samples.map(|samples| crate::testing::benchmark_median_seconds(samples).0);
        let phase_cumulative_fraction =
            phase_cumulative_median_seconds.map(|seconds| seconds / median_seconds);
        let memory = crate::metrics::process_memory_snapshot();
        eprintln!(
            "AUX_MODEL_BENCH model=qwen3-forced-aligner provider={} stage_topology={} backend_name={backend_name:?} audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={:.6} current_rss_bytes={:?} peak_rss_bytes={:?} current_phys_footprint_bytes={:?} peak_phys_footprint_bytes={:?} items={} output_sha256={output_sha256} runs={seconds:?} phase_cumulative_median_seconds={phase_cumulative_median_seconds:?} phase_cumulative_fraction={phase_cumulative_fraction:?}",
            test_backend.provider_label(),
            test_backend.stage_topology(),
            median_seconds / audio_seconds,
            memory.current_rss_bytes,
            memory.peak_rss_bytes,
            memory.current_phys_footprint_bytes,
            memory.peak_phys_footprint_bytes,
            items.len(),
        );
    }

    #[test]
    #[ignore = "host-local endurance gate: needs OPENASR_FORCED_ALIGNER_PACK, OPENASR_AUX_BENCH_AUDIO, and OPENASR_AUX_BENCH_TEXT"]
    fn forced_aligner_reuses_one_session_for_fifteen_minutes_of_segments() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FORCED_ALIGNER_PACK",
            "Qwen3 forced-aligner runtime pack",
        )
        .expect("OPENASR_FORCED_ALIGNER_PACK");
        let audio = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_AUDIO",
            "private auxiliary-model endurance audio",
        )
        .expect("OPENASR_AUX_BENCH_AUDIO");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model endurance transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let text = std::fs::read_to_string(text_path).expect("read endurance transcript");
        let text = text.trim();
        assert!(!text.is_empty(), "endurance transcript must not be empty");
        let language =
            std::env::var("OPENASR_AUX_BENCH_LANGUAGE").unwrap_or_else(|_| "Chinese".to_string());
        let test_backend = ForcedAlignerTestBackend::from_env();
        let _route_guard = test_backend.install_exact_route();
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &audio,
            "forced-aligner endurance gate",
            "forced-aligner endurance gate",
        )
        .expect("load endurance audio");
        let pcm = crate::PcmBuffer::from_vec(samples);
        let audio_seconds = pcm.len() as f64 / 16_000.0;
        assert!(audio_seconds > 0.0, "endurance audio must not be empty");
        let repetitions = (15.0 * 60.0 / audio_seconds).ceil() as usize;
        let represented_audio_seconds = audio_seconds * repetitions as f64;
        let backend_name = test_backend.resolved_backend_name();
        let session = test_backend.load_session(&pack).expect("load aligner");
        let run = || {
            session
                .align(pcm.full_slice(), text, &language)
                .expect("align endurance segment")
        };

        let expected = run();
        assert!(!expected.is_empty(), "endurance alignment must emit items");
        let started = std::time::Instant::now();
        for iteration in 0..repetitions {
            let actual = run();
            assert_eq!(
                actual, expected,
                "alignment output changed at endurance iteration {iteration}"
            );
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let mut output_bytes = Vec::new();
        for item in &expected {
            output_bytes.extend_from_slice(item.text.as_bytes());
            output_bytes.push(0);
            output_bytes.extend_from_slice(&item.start_time_s.to_le_bytes());
            output_bytes.extend_from_slice(&item.end_time_s.to_le_bytes());
        }
        let output_sha256 = crate::testing::benchmark_sha256_bytes([output_bytes]);
        let memory = crate::metrics::process_memory_snapshot();
        let peak_rss_bytes = memory.peak_rss_bytes.unwrap_or(0);
        let current_rss_bytes = memory.current_rss_bytes.unwrap_or(0);
        let phys_footprint_bytes = memory.current_phys_footprint_bytes.unwrap_or(0);
        let peak_phys_footprint_bytes = memory.peak_phys_footprint_bytes.unwrap_or(0);
        eprintln!(
            "AUX_MODEL_ENDURANCE model=qwen3-forced-aligner provider={} stage_topology={} backend_name={backend_name:?} segment_audio_seconds={audio_seconds:.6} repetitions={repetitions} represented_audio_seconds={represented_audio_seconds:.6} elapsed_seconds={elapsed_seconds:.6} rtf={:.6} peak_rss_bytes={peak_rss_bytes} current_rss_bytes={current_rss_bytes} phys_footprint_bytes={phys_footprint_bytes} peak_phys_footprint_bytes={peak_phys_footprint_bytes} items={} output_sha256={output_sha256}",
            test_backend.provider_label(),
            test_backend.stage_topology(),
            elapsed_seconds / represented_audio_seconds,
            expected.len(),
        );
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_FORCED_ALIGNER_PACK, OPENASR_AUX_BENCH_AUDIO, and OPENASR_AUX_BENCH_TEXT"]
    fn forced_aligner_cpu_and_metal_timestamps_stay_within_model_resolution() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FORCED_ALIGNER_PACK",
            "Qwen3 forced-aligner runtime pack",
        )
        .expect("OPENASR_FORCED_ALIGNER_PACK");
        let audio = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_AUDIO",
            "private auxiliary-model benchmark audio",
        )
        .expect("OPENASR_AUX_BENCH_AUDIO");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model benchmark transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let text = std::fs::read_to_string(text_path).expect("read parity transcript");
        let text = text.trim();
        assert!(!text.is_empty(), "parity transcript must not be empty");
        let language =
            std::env::var("OPENASR_AUX_BENCH_LANGUAGE").unwrap_or_else(|_| "Chinese".to_string());
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &audio,
            "forced-aligner backend parity",
            "forced-aligner backend parity",
        )
        .expect("load parity audio");
        let pcm = crate::PcmBuffer::from_vec(samples);
        let run = |backend| {
            let session = Qwen3ForcedAlignerSession::load(&pack, backend).expect("load aligner");
            session
                .align(pcm.full_slice(), text, &language)
                .expect("align parity audio")
        };
        let cpu = run(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu);
        let metal = run(crate::ggml_runtime::GgmlCpuGraphBackend::Metal);
        assert_eq!(cpu.len(), metal.len(), "CPU/Metal item count");
        let (median_ms, p95_ms, max_ms) = forced_aligner_timestamp_drift_stats(&cpu, &metal);
        eprintln!(
            "FORCED_ALIGNER_BACKEND_PARITY items={} endpoints={} median_ms={median_ms:.3} p95_ms={p95_ms:.3} max_ms={max_ms:.3}",
            cpu.len(),
            cpu.len() * 2,
        );

        assert!(
            median_ms < 80.0,
            "CPU/Metal median timestamp drift {median_ms:.3}ms exceeds one 80ms model bin"
        );
        assert!(
            p95_ms <= 160.0,
            "CPU/Metal p95 timestamp drift {p95_ms:.3}ms exceeds two 80ms model bins"
        );
        assert!(
            max_ms <= 320.0,
            "CPU/Metal maximum timestamp drift {max_ms:.3}ms exceeds four 80ms model bins"
        );
    }

    fn forced_aligner_timestamp_drift_stats(
        expected: &[ForcedAlignItem],
        actual: &[ForcedAlignItem],
    ) -> (f64, f64, f64) {
        assert_eq!(expected.len(), actual.len(), "forced-aligner item count");
        let mut differences_ms = Vec::with_capacity(expected.len() * 2);
        for (index, (expected_item, actual_item)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                expected_item.text, actual_item.text,
                "forced-aligner item text at {index}"
            );
            differences_ms
                .push((expected_item.start_time_s - actual_item.start_time_s).abs() * 1000.0);
            differences_ms.push((expected_item.end_time_s - actual_item.end_time_s).abs() * 1000.0);
        }
        differences_ms.sort_by(f64::total_cmp);
        (
            differences_ms[differences_ms.len() / 2],
            differences_ms[(differences_ms.len() - 1) * 95 / 100],
            differences_ms[differences_ms.len() - 1],
        )
    }

    #[test]
    #[ignore = "host-local: needs the forced-aligner pack, private audio/text, and official Python reference JSON"]
    fn forced_aligner_matches_official_reference_on_aux_audio() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FORCED_ALIGNER_PACK",
            "Qwen3 forced-aligner runtime pack",
        )
        .expect("OPENASR_FORCED_ALIGNER_PACK");
        let audio = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_AUDIO",
            "private auxiliary-model parity audio",
        )
        .expect("OPENASR_AUX_BENCH_AUDIO");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model parity transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let reference_path = crate::testing::external_test_fixture_path(
            "OPENASR_FORCED_ALIGNER_REFERENCE_JSON",
            "official Qwen3 forced-aligner reference JSON",
        )
        .expect("OPENASR_FORCED_ALIGNER_REFERENCE_JSON");
        let text = std::fs::read_to_string(text_path).expect("read parity transcript");
        let language =
            std::env::var("OPENASR_AUX_BENCH_LANGUAGE").unwrap_or_else(|_| "Chinese".to_string());
        let test_backend = ForcedAlignerTestBackend::from_env();
        let _route_guard = test_backend.install_exact_route();
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &audio,
            "forced-aligner official parity",
            "forced-aligner official parity",
        )
        .expect("load parity audio");
        let backend_name = test_backend.resolved_backend_name();
        let session = test_backend.load_session(&pack).expect("load aligner");
        let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
        let _execution_placement_guard = execution_placement.install();
        let items = session
            .align(
                crate::PcmBuffer::from_vec(samples).full_slice(),
                text.trim(),
                &language,
            )
            .expect("align parity audio");
        let observed = execution_placement.snapshot();
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(reference_path).expect("read official reference JSON"),
        )
        .expect("parse official reference JSON");
        let reference_items = reference["items"]
            .as_array()
            .expect("official reference items array");
        assert_eq!(items.len(), reference_items.len(), "reference item count");

        let mut differences_ms = Vec::with_capacity(items.len() * 2);
        for (index, (item, reference_item)) in items.iter().zip(reference_items.iter()).enumerate()
        {
            assert_eq!(
                item.text,
                reference_item["text"].as_str().unwrap_or_default(),
                "reference item text at {index}"
            );
            let reference_start = reference_item["start_time"].as_f64().unwrap_or_default();
            let reference_end = reference_item["end_time"].as_f64().unwrap_or_default();
            let start_difference_ms = (item.start_time_s - reference_start).abs() * 1000.0;
            let end_difference_ms = (item.end_time_s - reference_end).abs() * 1000.0;
            if start_difference_ms > 160.0 || end_difference_ms > 160.0 {
                eprintln!(
                    "FORCED_ALIGNER_OFFICIAL_OUTLIER index={index} ours=({:.3},{:.3}) reference=({reference_start:.3},{reference_end:.3}) differences_ms=({start_difference_ms:.3},{end_difference_ms:.3})",
                    item.start_time_s, item.end_time_s,
                );
            }
            differences_ms.push(start_difference_ms);
            differences_ms.push(end_difference_ms);
        }
        differences_ms.sort_by(f64::total_cmp);
        let median_ms = differences_ms[differences_ms.len() / 2];
        let p95_ms = differences_ms[(differences_ms.len() - 1) * 95 / 100];
        let max_ms = differences_ms[differences_ms.len() - 1];
        let memory = crate::metrics::process_memory_snapshot();
        eprintln!(
            "FORCED_ALIGNER_OFFICIAL_PARITY provider={} stage_topology={} backend_name={backend_name:?} items={} endpoints={} median_ms={median_ms:.3} p95_ms={p95_ms:.3} max_ms={max_ms:.3} observed_compute_nodes={:?} current_rss_bytes={:?} peak_rss_bytes={:?} current_phys_footprint_bytes={:?} peak_phys_footprint_bytes={:?}",
            test_backend.provider_label(),
            test_backend.stage_topology(),
            items.len(),
            differences_ms.len(),
            observed.observed_compute_nodes_by_backend,
            memory.current_rss_bytes,
            memory.peak_rss_bytes,
            memory.current_phys_footprint_bytes,
            memory.peak_phys_footprint_bytes,
        );
        assert!(median_ms < 80.0, "median drift {median_ms:.3}ms");
        assert!(p95_ms <= 160.0, "p95 drift {p95_ms:.3}ms");
        assert!(max_ms <= 320.0, "maximum drift {max_ms:.3}ms");
        let compute_nodes = &observed.observed_compute_nodes_by_backend;
        match test_backend {
            ForcedAlignerTestBackend::Metal => assert!(
                !compute_nodes.is_empty()
                    && compute_nodes.keys().all(|backend| {
                        let backend = backend.to_ascii_lowercase();
                        backend.starts_with("mtl") || backend.contains("metal")
                    }),
                "explicit Metal forced-aligner route observed non-Metal compute: {compute_nodes:?}",
            ),
            ForcedAlignerTestBackend::Cuda | ForcedAlignerTestBackend::Vulkan => {
                let provider = test_backend.provider_label();
                assert!(
                    compute_nodes.iter().any(|(backend, nodes)| {
                        backend.to_ascii_lowercase().contains(provider) && *nodes > 0
                    }),
                    "explicit {provider} Hybrid forced-aligner route did not execute its GPU stage: {compute_nodes:?}",
                );
                assert!(
                    compute_nodes.iter().any(|(backend, nodes)| {
                        backend.to_ascii_lowercase().contains("cpu") && *nodes > 0
                    }),
                    "explicit {provider} Hybrid forced-aligner route did not execute its CPU stages: {compute_nodes:?}",
                );
                assert!(
                    compute_nodes.keys().all(|backend| {
                        let backend = backend.to_ascii_lowercase();
                        backend.contains(provider) || backend.contains("cpu")
                    }),
                    "explicit {provider} Hybrid forced-aligner route observed an unrelated backend: {compute_nodes:?}",
                );
            }
            ForcedAlignerTestBackend::Cpu | ForcedAlignerTestBackend::GenericGpu => {}
        }
    }

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

        let source_root = match crate::testing::external_test_fixture_path(
            "OPENASR_QWEN_FORCED_ALIGNER_SOURCE",
            "Qwen forced-aligner source checkpoint directory",
        ) {
            Ok(path) => path,
            Err(skip) => {
                eprintln!("skipping: {skip}");
                return;
            }
        };
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

        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack_path).expect("runtime source");
        let preflight = load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .expect("runtime preflight");
        let assets = load_forced_aligner_prepared_assets(
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("prepared assets");

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

            let items = align_forced(
                &preflight,
                &assets,
                crate::PcmBuffer::from_vec(samples).full_slice(),
                case.text,
                case.language,
                crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
            )
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
