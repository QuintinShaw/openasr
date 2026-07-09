use std::sync::Arc;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::{Segment, Transcription};

use super::decoder_weights::{CohereDecoderLayerWeights, CohereTranscribeDecoderWeights};
use super::encoder_graph::CohereTranscribeEncoderOutput;
use super::graph_config::cohere_decoder_graph_config;
use super::greedy_decode::{
    CohereTranscribeGreedyDecodeError, CohereTranscribeGreedyDecodeResult,
    run_cohere_transcribe_greedy_decode_loop,
};
use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tokenizer::CohereTranscribeTokenizer;
use super::weights::{CohereMatrixLayout, CohereMatrixWeight, CohereVectorWeight};
use crate::PhraseBiasConfig;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
};
use crate::models::decode_token_history::context_window_budget;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::nn::decoder::{
    Seq2SeqReusableDecodeGraph, build_causal_mask_f16_bits, build_fixed_kv_attention_mask_bits,
    build_fixed_kv_attention_mask_bits_for_sequences, reusable_decode_graph_supported_for_runner,
    seq2seq_layer_stack,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

const COHERE_DECODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
/// Floor for the decoder's `no_alloc` metadata context node/tensor budget:
/// covers both the per-step decode cgraph AND every static tensor allocated
/// directly in the arena (weights, embeddings, cross-KV, self-KV -- see
/// `GgmlStaticTensorArena`, which is metadata-only: real tensor bytes land in
/// a backend buffer sized from the tensors' actual shapes, independent of
/// this context's size). Mirrors the encoder's proven `16_384` headroom
/// (`cohere_encoder_graph_config_with_overrides`) -- comfortably above the
/// realistic weight+KV tensor count for any decoder layer depth.
const COHERE_DECODER_GRAPH_SIZE_FLOOR: usize = 16_384;
const COHERE_DISABLE_INCREMENTAL_SELF_KV_ENV: &str = "OPENASR_COHERE_DISABLE_INCREMENTAL_SELF_KV";
const COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV: &str =
    "OPENASR_COHERE_MAX_GENERATED_TOKENS_OVERRIDE";

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereCrossAttentionLayerCache {
    pub frame_count: usize,
    pub hidden_size: usize,
    pub key_rows: Vec<f32>,
    pub value_rows: Vec<f32>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereCrossAttentionCache {
    pub frame_count: usize,
    pub hidden_size: usize,
    pub layers: Vec<CohereCrossAttentionLayerCache>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereDecoderGraphDecodeOutput {
    pub transcription: Transcription,
    pub generated_tokens: Vec<u32>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereDecoderGraphError {
    #[error("cohere-transcribe decoder graph input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[cfg_attr(not(test), allow(dead_code))]
    #[error("cohere-transcribe decoder graph weight projection is invalid: {reason}")]
    InvalidWeight { reason: String },
    #[error("cohere-transcribe decoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("cohere-transcribe decoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("cohere-transcribe decoder graph shape overflowed")]
    ShapeOverflow,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_cohere_cross_attention_cache_from_encoder_output(
    decoder_weights: &CohereTranscribeDecoderWeights,
    metadata: CohereTranscribeExecutionMetadata,
    encoder_output: &CohereTranscribeEncoderOutput,
) -> Result<CohereCrossAttentionCache, CohereDecoderGraphError> {
    if encoder_output.hidden_size != metadata.decoder_d_model {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "encoder hidden_size {} does not match decoder hidden size {}",
                encoder_output.hidden_size, metadata.decoder_d_model
            ),
        });
    }
    if encoder_output.frame_count == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "encoder output frame_count must be > 0".to_string(),
        });
    }
    let expected = encoder_output
        .frame_count
        .checked_mul(encoder_output.hidden_size)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    if encoder_output.rows.len() != expected {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "encoder rows length mismatch: got {}, expected {}",
                encoder_output.rows.len(),
                expected
            ),
        });
    }

    let mut layers = Vec::with_capacity(decoder_weights.layers.len());
    for layer in &decoder_weights.layers {
        let key_rows = project_hidden_sequence_with_bias(
            &layer.cross_k_weight,
            &layer.cross_k_bias,
            &encoder_output.rows,
            encoder_output.hidden_size,
            encoder_output.frame_count,
        )?;
        let value_rows = project_hidden_sequence_with_bias(
            &layer.cross_v_weight,
            &layer.cross_v_bias,
            &encoder_output.rows,
            encoder_output.hidden_size,
            encoder_output.frame_count,
        )?;
        layers.push(CohereCrossAttentionLayerCache {
            frame_count: encoder_output.frame_count,
            hidden_size: encoder_output.hidden_size,
            key_rows,
            value_rows,
        });
    }

    Ok(CohereCrossAttentionCache {
        frame_count: encoder_output.frame_count,
        hidden_size: encoder_output.hidden_size,
        layers,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_cohere_decoder_graph_short_form_with_runtime(
    decoder_runtime: &mut CohereDecoderGraphRuntime,
    tokenizer: &CohereTranscribeTokenizer,
    metadata: CohereTranscribeExecutionMetadata,
    prompt_tokens: &[u32],
    eos_token_id: u32,
    encoder_output: &CohereTranscribeEncoderOutput,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    audio_duration_seconds: f32,
) -> Result<CohereDecoderGraphDecodeOutput, CohereDecoderGraphError> {
    let mut step_executor =
        CohereDecoderGraphStepExecutor::from_runtime(decoder_runtime, encoder_output)?;
    let decode_text_token_ids = |token_ids: &[u32]| {
        tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
            CohereTranscribeGreedyDecodeError::TokenizerDecodeFailed {
                reason: error.to_string(),
            }
        })
    };
    let max_generated_tokens =
        decoder_max_generated_tokens_with_env(prompt_tokens, metadata, encoder_output.frame_count)?;
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: prompt_tokens.to_vec(),
        eot_token_id: eos_token_id,
        vocab_size: metadata.vocab_size,
        max_generated_tokens,
    };
    let decode = match run_cohere_transcribe_greedy_decode_loop(
        &config,
        tokenizer,
        phrase_bias,
        &mut step_executor,
        &decode_text_token_ids,
    ) {
        Ok(output) => output,
        Err(CohereTranscribeGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens,
            generated_probabilities,
            ..
        }) => CohereTranscribeGreedyDecodeResult {
            text: decode_text_token_ids(&generated_tokens).map_err(|error| {
                CohereDecoderGraphError::InvalidInput {
                    reason: error.to_string(),
                }
            })?,
            generated_tokens,
            generated_probabilities,
        },
        Err(error) => {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            });
        }
    };
    let text = decode.text.trim().to_string();
    let transcription = if request_diarization_from_prompt(prompt_tokens, tokenizer) {
        let segments = cohere_diarized_segments_from_generated_tokens(
            tokenizer,
            &decode.generated_tokens,
            audio_duration_seconds,
            &decode_text_token_ids,
        )?;
        if segments.is_empty() {
            cohere_plain_transcription_from_generated_tokens(
                text,
                &decode.generated_tokens,
                &decode.generated_probabilities,
                word_timestamps,
                audio_duration_seconds,
                &decode_text_token_ids,
            )?
        } else {
            let text = segments
                .iter()
                .map(|segment| segment.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            Transcription {
                text,
                segments,
                longform: None,
                language: None,
            }
        }
    } else {
        cohere_plain_transcription_from_generated_tokens(
            text,
            &decode.generated_tokens,
            &decode.generated_probabilities,
            word_timestamps,
            audio_duration_seconds,
            &decode_text_token_ids,
        )?
    };
    Ok(CohereDecoderGraphDecodeOutput {
        transcription,
        generated_tokens: decode.generated_tokens,
    })
}

fn request_diarization_from_prompt(
    prompt_tokens: &[u32],
    tokenizer: &CohereTranscribeTokenizer,
) -> bool {
    prompt_tokens
        .iter()
        .any(|token_id| tokenizer.token_content_by_id(*token_id) == Some("<|diarize|>"))
}

fn cohere_plain_transcription_from_generated_tokens(
    text: String,
    generated_tokens: &[u32],
    generated_probabilities: &[f32],
    word_timestamps: bool,
    audio_duration_seconds: f32,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, CohereTranscribeGreedyDecodeError>,
) -> Result<Transcription, CohereDecoderGraphError> {
    let words = if word_timestamps {
        seq2seq_word_timestamps_from_generated_tokens(
            generated_tokens,
            generated_probabilities,
            0.0,
            audio_duration_seconds,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            decode_text_token_ids,
        )
        .map_err(|error| CohereDecoderGraphError::InvalidInput {
            reason: error.to_string(),
        })?
    } else {
        Vec::new()
    };
    let segments = if words.is_empty() || text.is_empty() {
        Vec::new()
    } else {
        vec![Segment {
            start: 0.0,
            end: audio_duration_seconds.max(0.0),
            text: text.clone(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words,
        }]
    };
    Ok(Transcription {
        text,
        segments,
        longform: None,
        language: None,
    })
}

fn cohere_diarized_segments_from_generated_tokens(
    tokenizer: &CohereTranscribeTokenizer,
    generated_tokens: &[u32],
    audio_duration_seconds: f32,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, CohereTranscribeGreedyDecodeError>,
) -> Result<Vec<Segment>, CohereDecoderGraphError> {
    let mut segments = Vec::new();
    let mut speaker: Option<String> = None;
    let mut start = 0.0_f32;
    let mut last_timestamp = 0.0_f32;
    let mut text_tokens = Vec::new();
    let mut saw_speaker = false;

    for token_id in generated_tokens {
        let Some(token) = tokenizer.token_content_by_id(*token_id) else {
            text_tokens.push(*token_id);
            continue;
        };
        if let Some(next_speaker) = cohere_speaker_label_from_token(token) {
            flush_cohere_diarized_segment(
                &mut segments,
                &mut text_tokens,
                decode_text_token_ids,
                speaker.clone(),
                start,
                last_timestamp.max(start),
            )?;
            speaker = Some(next_speaker);
            saw_speaker = true;
            start = last_timestamp;
            continue;
        }
        if let Some(timestamp) = cohere_timestamp_seconds_from_token(token) {
            let timestamp = timestamp.max(0.0).min(audio_duration_seconds.max(0.0));
            if !text_tokens.is_empty() {
                flush_cohere_diarized_segment(
                    &mut segments,
                    &mut text_tokens,
                    decode_text_token_ids,
                    speaker.clone(),
                    start,
                    timestamp.max(start),
                )?;
                start = timestamp;
            } else {
                start = timestamp;
            }
            last_timestamp = timestamp;
            continue;
        }
        if token.starts_with("<|") && token.ends_with("|>") {
            continue;
        }
        text_tokens.push(*token_id);
    }

    flush_cohere_diarized_segment(
        &mut segments,
        &mut text_tokens,
        decode_text_token_ids,
        speaker,
        start,
        audio_duration_seconds.max(start),
    )?;

    if saw_speaker {
        Ok(segments)
    } else {
        Ok(Vec::new())
    }
}

fn flush_cohere_diarized_segment(
    segments: &mut Vec<Segment>,
    text_tokens: &mut Vec<u32>,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, CohereTranscribeGreedyDecodeError>,
    speaker: Option<String>,
    start: f32,
    end: f32,
) -> Result<(), CohereDecoderGraphError> {
    if text_tokens.is_empty() {
        return Ok(());
    }
    let text = decode_text_token_ids(text_tokens)
        .map_err(|error| CohereDecoderGraphError::InvalidInput {
            reason: error.to_string(),
        })?
        .trim()
        .to_string();
    text_tokens.clear();
    if text.is_empty() {
        return Ok(());
    }
    segments.push(Segment {
        start,
        end: end.max(start),
        text,
        speaker,
        speaker_label: None,
        speaker_profile_id: None,
        words: Vec::new(),
    });
    Ok(())
}

fn cohere_speaker_label_from_token(token: &str) -> Option<String> {
    let number = token
        .strip_prefix("<|spltoken")
        .and_then(|value| value.strip_suffix("|>"))?
        .parse::<usize>()
        .ok()?;
    Some(format!("SPEAKER_{number:02}"))
}

fn cohere_timestamp_seconds_from_token(token: &str) -> Option<f32> {
    token
        .strip_prefix("<|t:")
        .and_then(|value| value.strip_suffix("|>"))?
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite())
}

pub(crate) struct CohereDecoderGraphRuntime {
    // `reuse` holds raw pointers into `runner`, `arena`, and resident KV/cross
    // tensors, so it must be declared first and dropped first.
    reuse: Option<Seq2SeqReusableDecodeGraph>,
    metadata: CohereTranscribeExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    /// The `no_alloc` metadata context size used for `runner`'s own graph
    /// context and `arena`; reused verbatim for
    /// `start_persistent_graph_session` in [`Self::build_reusable_decode_graph`]
    /// so it does not have to be recomputed from a hardcoded constant.
    persistent_graph_context_bytes: usize,
    arena: GgmlStaticTensorArena,
    token_embedding: GgmlStaticTensor,
    positional_embedding: GgmlStaticTensor,
    emb_ln_weight: GgmlStaticTensor,
    emb_ln_bias: GgmlStaticTensor,
    out_ln_weight: GgmlStaticTensor,
    out_ln_bias: GgmlStaticTensor,
    output_head_weight: GgmlStaticTensor,
    output_head_bias: GgmlStaticTensor,
    layers: Vec<CohereDecoderLayerRuntime>,
    cross_layers: Vec<CohereDecoderCrossCacheLayerRuntime>,
    self_kv_layers: Vec<CohereDecoderSelfKvLayerRuntime>,
    cached_positions: usize,
    n_seq: usize,
}

#[derive(Clone, Copy)]
struct CoherePromptDebugTensors<'a> {
    token_state: GgmlCpuTensor<'a>,
    position_state: GgmlCpuTensor<'a>,
    emb_ln: GgmlCpuTensor<'a>,
    l0_attn_norm: GgmlCpuTensor<'a>,
    l0_q_proj: GgmlCpuTensor<'a>,
    l0_k_proj: GgmlCpuTensor<'a>,
    l0_v_proj: GgmlCpuTensor<'a>,
    h0_after_sa: GgmlCpuTensor<'a>,
    h0_after_ca: GgmlCpuTensor<'a>,
    h0_after_ffn: GgmlCpuTensor<'a>,
    final_state: GgmlCpuTensor<'a>,
}

#[derive(Clone, Copy)]
struct CohereDecoderLayerRuntime {
    attn_ln_weight: GgmlStaticTensor,
    attn_ln_bias: GgmlStaticTensor,
    attn_q_weight: GgmlStaticTensor,
    attn_q_bias: GgmlStaticTensor,
    attn_k_weight: GgmlStaticTensor,
    attn_k_bias: GgmlStaticTensor,
    attn_v_weight: GgmlStaticTensor,
    attn_v_bias: GgmlStaticTensor,
    attn_o_weight: GgmlStaticTensor,
    attn_o_bias: GgmlStaticTensor,
    cross_ln_weight: GgmlStaticTensor,
    cross_ln_bias: GgmlStaticTensor,
    cross_k_weight: GgmlStaticTensor,
    cross_k_bias: GgmlStaticTensor,
    cross_v_weight: GgmlStaticTensor,
    cross_v_bias: GgmlStaticTensor,
    cross_q_weight: GgmlStaticTensor,
    cross_q_bias: GgmlStaticTensor,
    cross_o_weight: GgmlStaticTensor,
    cross_o_bias: GgmlStaticTensor,
    ffn_ln_weight: GgmlStaticTensor,
    ffn_ln_bias: GgmlStaticTensor,
    ffn_up_weight: GgmlStaticTensor,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down_weight: GgmlStaticTensor,
    ffn_down_bias: GgmlStaticTensor,
}

#[derive(Clone, Copy)]
struct CohereDecoderSelfKvLayerRuntime {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
    max_positions: usize,
}

#[derive(Clone, Copy)]
struct CohereDecoderCrossCacheLayerRuntime {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
    frame_count: usize,
    hidden_size: usize,
}

struct CohereDecoderGraphStepExecutor<'a> {
    runtime: &'a mut CohereDecoderGraphRuntime,
}

impl<'a> CohereDecoderGraphStepExecutor<'a> {
    fn from_runtime(
        runtime: &'a mut CohereDecoderGraphRuntime,
        encoder_output: &CohereTranscribeEncoderOutput,
    ) -> Result<Self, CohereDecoderGraphError> {
        runtime.populate_cross_attention_cache(encoder_output)?;
        Ok(Self { runtime })
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for CohereDecoderGraphStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        let prefix = input
            .initial_prompt_tokens
            .iter()
            .copied()
            .chain(input.generated_tokens.iter().copied())
            .collect::<Vec<_>>();
        let logits = self.runtime.compute_step_logits(&prefix).map_err(|error| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

impl CohereDecoderGraphRuntime {
    pub(crate) fn new(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        cross_frame_count: usize,
        cross_hidden_size: usize,
        prefer_cpu_backend: bool,
    ) -> Result<Self, CohereDecoderGraphError> {
        Self::new_with_n_seq(
            decoder_weights,
            metadata,
            cross_frame_count,
            cross_hidden_size,
            prefer_cpu_backend,
            1,
        )
    }

    pub(crate) fn new_with_n_seq(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        cross_frame_count: usize,
        cross_hidden_size: usize,
        prefer_cpu_backend: bool,
        n_seq: usize,
    ) -> Result<Self, CohereDecoderGraphError> {
        if n_seq == 0 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "cohere decoder n_seq must be positive".to_string(),
            });
        }
        validate_decoder_runtime_shapes(decoder_weights, metadata)?;
        validate_encoder_cross_dimensions(
            cross_hidden_size,
            cross_frame_count,
            metadata,
            decoder_weights.layers.len(),
        )?;

        let mut config = cohere_decoder_graph_config(prefer_cpu_backend);
        config.graph_size = config.graph_size.max(COHERE_DECODER_GRAPH_SIZE_FLOOR);
        config.context_bytes =
            config
                .context_bytes
                .max(GgmlCpuGraphConfig::metadata_context_bytes(
                    config.graph_size,
                ));
        let persistent_graph_context_bytes = config.context_bytes;
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;

        let token_embedding =
            new_embedding_tensor_in_arena(&arena, &decoder_weights.token_embedding, "dec_emb")?;
        let positional_embedding = new_embedding_tensor_in_arena(
            &arena,
            &decoder_weights.positional_embedding,
            "dec_pos",
        )?;
        let emb_ln_weight =
            new_vector_tensor_in_arena(&arena, decoder_weights.emb_ln_weight.len, "dec_emb_ln_w")?;
        let emb_ln_bias =
            new_vector_tensor_in_arena(&arena, decoder_weights.emb_ln_bias.len, "dec_emb_ln_b")?;
        let out_ln_weight =
            new_vector_tensor_in_arena(&arena, decoder_weights.out_ln_weight.len, "dec_out_ln_w")?;
        let out_ln_bias =
            new_vector_tensor_in_arena(&arena, decoder_weights.out_ln_bias.len, "dec_out_ln_b")?;
        let output_head_weight = new_projection_tensor_in_arena(
            &arena,
            &decoder_weights.output_head_weight,
            "dec_head",
        )?;
        let output_head_bias =
            new_vector_tensor_in_arena(&arena, decoder_weights.output_head_bias.len, "dec_head_b")?;

        let mut layers = Vec::with_capacity(decoder_weights.layers.len());
        let mut cross_layers = Vec::with_capacity(decoder_weights.layers.len());
        let mut self_kv_layers = Vec::with_capacity(decoder_weights.layers.len());
        for layer in &decoder_weights.layers {
            let runtime = CohereDecoderLayerRuntime {
                attn_ln_weight: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_ln_weight.len,
                    "dec_attn_ln_w",
                )?,
                attn_ln_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_ln_bias.len,
                    "dec_attn_ln_b",
                )?,
                attn_q_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.attn_q_weight,
                    "dec_attn_q_w",
                )?,
                attn_q_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_q_bias.len,
                    "dec_attn_q_b",
                )?,
                attn_k_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.attn_k_weight,
                    "dec_attn_k_w",
                )?,
                attn_k_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_k_bias.len,
                    "dec_attn_k_b",
                )?,
                attn_v_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.attn_v_weight,
                    "dec_attn_v_w",
                )?,
                attn_v_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_v_bias.len,
                    "dec_attn_v_b",
                )?,
                attn_o_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.attn_o_weight,
                    "dec_attn_o_w",
                )?,
                attn_o_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.attn_o_bias.len,
                    "dec_attn_o_b",
                )?,
                cross_ln_weight: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_ln_weight.len,
                    "dec_cross_ln_w",
                )?,
                cross_ln_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_ln_bias.len,
                    "dec_cross_ln_b",
                )?,
                cross_k_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.cross_k_weight,
                    "dec_cross_k_w",
                )?,
                cross_k_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_k_bias.len,
                    "dec_cross_k_b",
                )?,
                cross_v_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.cross_v_weight,
                    "dec_cross_v_w",
                )?,
                cross_v_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_v_bias.len,
                    "dec_cross_v_b",
                )?,
                cross_q_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.cross_q_weight,
                    "dec_cross_q_w",
                )?,
                cross_q_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_q_bias.len,
                    "dec_cross_q_b",
                )?,
                cross_o_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.cross_o_weight,
                    "dec_cross_o_w",
                )?,
                cross_o_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.cross_o_bias.len,
                    "dec_cross_o_b",
                )?,
                ffn_ln_weight: new_vector_tensor_in_arena(
                    &arena,
                    layer.ffn_ln_weight.len,
                    "dec_ffn_ln_w",
                )?,
                ffn_ln_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.ffn_ln_bias.len,
                    "dec_ffn_ln_b",
                )?,
                ffn_up_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.ffn_up_weight,
                    "dec_ffn_up_w",
                )?,
                ffn_up_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.ffn_up_bias.len,
                    "dec_ffn_up_b",
                )?,
                ffn_down_weight: new_projection_tensor_in_arena(
                    &arena,
                    &layer.ffn_down_weight,
                    "dec_ffn_down_w",
                )?,
                ffn_down_bias: new_vector_tensor_in_arena(
                    &arena,
                    layer.ffn_down_bias.len,
                    "dec_ffn_down_b",
                )?,
            };
            layers.push(runtime);
            cross_layers.push(CohereDecoderCrossCacheLayerRuntime {
                key: new_persistent_cross_cache_tensor_in_arena(
                    &arena,
                    cross_hidden_size,
                    cross_frame_count,
                    n_seq,
                    "dec_cross_k_cache",
                )?,
                value: new_persistent_cross_cache_tensor_in_arena(
                    &arena,
                    cross_hidden_size,
                    cross_frame_count,
                    n_seq,
                    "dec_cross_v_cache",
                )?,
                frame_count: cross_frame_count,
                hidden_size: cross_hidden_size,
            });
            self_kv_layers.push(CohereDecoderSelfKvLayerRuntime {
                key: new_persistent_self_kv_tensor_in_arena(
                    &arena,
                    metadata.decoder_head_dim,
                    metadata.decoder_max_context,
                    metadata.decoder_heads,
                    n_seq,
                    "dec_self_k_cache",
                )?,
                value: new_persistent_self_kv_tensor_in_arena(
                    &arena,
                    metadata.decoder_head_dim,
                    metadata.decoder_max_context,
                    metadata.decoder_heads,
                    n_seq,
                    "dec_self_v_cache",
                )?,
                max_positions: metadata.decoder_max_context,
            });
        }

        upload_embedding_to_arena(
            &mut arena,
            token_embedding,
            &decoder_weights.token_embedding,
            "dec_emb",
        )?;
        upload_embedding_to_arena(
            &mut arena,
            positional_embedding,
            &decoder_weights.positional_embedding,
            "dec_pos",
        )?;
        upload_vector_to_arena(
            &mut arena,
            emb_ln_weight,
            &decoder_weights.emb_ln_weight,
            "dec_emb_ln_w",
        )?;
        upload_vector_to_arena(
            &mut arena,
            emb_ln_bias,
            &decoder_weights.emb_ln_bias,
            "dec_emb_ln_b",
        )?;
        upload_vector_to_arena(
            &mut arena,
            out_ln_weight,
            &decoder_weights.out_ln_weight,
            "dec_out_ln_w",
        )?;
        upload_vector_to_arena(
            &mut arena,
            out_ln_bias,
            &decoder_weights.out_ln_bias,
            "dec_out_ln_b",
        )?;
        upload_projection_to_arena(
            &mut arena,
            output_head_weight,
            &decoder_weights.output_head_weight,
            "dec_head",
        )?;
        upload_vector_to_arena(
            &mut arena,
            output_head_bias,
            &decoder_weights.output_head_bias,
            "dec_head_b",
        )?;
        for (layer_idx, (runtime, layer)) in layers.iter().zip(&decoder_weights.layers).enumerate()
        {
            upload_decoder_layer_to_arena(&mut arena, runtime, layer, layer_idx)?;
        }

        Ok(Self {
            reuse: None,
            metadata,
            runner,
            persistent_graph_context_bytes,
            arena,
            token_embedding,
            positional_embedding,
            emb_ln_weight,
            emb_ln_bias,
            out_ln_weight,
            out_ln_bias,
            output_head_weight,
            output_head_bias,
            layers,
            cross_layers,
            self_kv_layers,
            cached_positions: 0,
            n_seq,
        })
    }

    pub(super) fn populate_cross_attention_cache(
        &mut self,
        encoder_output: &CohereTranscribeEncoderOutput,
    ) -> Result<(), CohereDecoderGraphError> {
        if self.n_seq != 1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "single cohere cross-cache population requires n_seq=1".to_string(),
            });
        }
        self.populate_cross_attention_cache_slot(0, encoder_output)
    }

    pub(super) fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        encoder_output: &CohereTranscribeEncoderOutput,
    ) -> Result<(), CohereDecoderGraphError> {
        if slot_index >= self.n_seq {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "cohere cross-cache slot index {slot_index} out of range for n_seq {}",
                    self.n_seq
                ),
            });
        }
        self.cached_positions = 0;
        validate_encoder_cross_dimensions(
            encoder_output.hidden_size,
            encoder_output.frame_count,
            self.metadata,
            self.layers.len(),
        )?;
        let expected = encoder_output
            .frame_count
            .checked_mul(encoder_output.hidden_size)
            .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
        if encoder_output.rows.len() != expected {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "encoder rows length mismatch: got {}, expected {}",
                    encoder_output.rows.len(),
                    expected
                ),
            });
        }

        let mut graph = self.runner.start_graph();
        let encoder_rows = graph
            .new_tensor_2d_f32(
                encoder_output.hidden_size,
                encoder_output.frame_count,
                "cohere_encoder_rows",
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_2d(encoder_rows)",
                source,
            })?;
        graph.set_input(encoder_rows).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(encoder_rows)",
                source,
            }
        })?;

        let mut output_root = None;
        for (layer, cross_runtime) in self.layers.iter().zip(&self.cross_layers) {
            let key_rows = apply_linear_with_bias(
                &graph,
                encoder_rows,
                self.arena.graph_tensor(layer.cross_k_weight),
                self.arena.graph_tensor(layer.cross_k_bias),
                "decoder_cross_cache_k",
            )?;
            let key_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross_runtime.key),
                encoder_output.hidden_size,
                encoder_output.frame_count,
                self.n_seq,
                slot_index,
                "ggml_view_2d(dec_cross_k_cache_slot)",
            )?;
            let write_key = graph.cpy(key_rows, key_target).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_cpy(dec_cross_k_cache)",
                    source,
                }
            })?;
            graph.add_side_effect_root(write_key).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_build_forward_expand(dec_cross_k_cache)",
                    source,
                }
            })?;

            let value_rows = apply_linear_with_bias(
                &graph,
                encoder_rows,
                self.arena.graph_tensor(layer.cross_v_weight),
                self.arena.graph_tensor(layer.cross_v_bias),
                "decoder_cross_cache_v",
            )?;
            let value_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross_runtime.value),
                encoder_output.hidden_size,
                encoder_output.frame_count,
                self.n_seq,
                slot_index,
                "ggml_view_2d(dec_cross_v_cache_slot)",
            )?;
            let write_value = graph.cpy(value_rows, value_target).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_cpy(dec_cross_v_cache)",
                    source,
                }
            })?;
            graph.add_side_effect_root(write_value).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_build_forward_expand(dec_cross_v_cache)",
                    source,
                }
            })?;
            output_root = Some(value_rows);
        }

        let output_root = output_root.ok_or(CohereDecoderGraphError::InvalidInput {
            reason: "decoder runtime has no layers".to_string(),
        })?;
        graph.set_output(output_root).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(decoder_cross_cache)",
                source,
            }
        })?;
        graph
            .set_f32_slice(encoder_rows, &encoder_output.rows, "cohere_encoder_rows")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_f32_slice(encoder_rows)",
                source,
            })?;
        graph
            .compute_output_f32(output_root, expected)
            .map(|_| ())
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_compute(decoder_cross_cache)",
                source,
            })
    }

    pub(super) fn compute_step_logits(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, CohereDecoderGraphError> {
        if self.n_seq != 1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "single cohere decode step requires n_seq=1".to_string(),
            });
        }
        let total_prefix_tokens = decoder_tokens.len();
        if total_prefix_tokens == 0 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "decoder token_count must be > 0".to_string(),
            });
        }
        if total_prefix_tokens > self.metadata.decoder_max_context {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder token_count {} exceeds max context {}",
                    total_prefix_tokens, self.metadata.decoder_max_context
                ),
            });
        }
        let use_incremental_self_kv =
            std::env::var_os(COHERE_DISABLE_INCREMENTAL_SELF_KV_ENV).is_none();
        let position_offset = if use_incremental_self_kv {
            self.cached_positions
        } else {
            0
        };
        let single_token;
        let decode_tokens: &[u32] = if position_offset == 0 {
            decoder_tokens
        } else {
            if total_prefix_tokens != position_offset.saturating_add(1) {
                return Err(CohereDecoderGraphError::InvalidInput {
                    reason: format!(
                        "incremental decoder prefix mismatch: got {} tokens, expected {} cached + 1",
                        total_prefix_tokens, position_offset
                    ),
                });
            }
            single_token =
                [*decoder_tokens
                    .last()
                    .ok_or(CohereDecoderGraphError::InvalidInput {
                        reason: "incremental decoder step is missing last token".to_string(),
                    })?];
            &single_token
        };
        let token_count = decode_tokens.len();
        let total_token_count = position_offset
            .checked_add(token_count)
            .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
        if position_offset > 0 && token_count == 1 && self.supports_reusable_decode_graph() {
            return self.compute_reused_incremental_step_logits(decode_tokens[0], position_offset);
        }
        let mut graph = self.runner.start_graph();
        let hidden = self.metadata.decoder_d_model;
        let token_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "cohere_decoder_tokens")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(tokens)",
                source,
            })?;
        let position_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "cohere_decoder_positions")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(positions)",
                source,
            })?;
        let self_kv_row_indices = if token_count == 1 {
            let row_indices = graph
                .new_tensor_1d_i32(1, "cohere_decoder_self_kv_row")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_1d(self_kv_row)",
                    source,
                })?;
            graph.set_input(row_indices).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_set_input(self_kv_row)",
                    source,
                }
            })?;
            Some(row_indices)
        } else {
            None
        };
        graph.set_input(token_ids_tensor).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(tokens)",
                source,
            }
        })?;
        graph.set_input(position_ids_tensor).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(positions)",
                source,
            }
        })?;

        let mut uploads = Vec::new();
        uploads.push(DecoderUpload::I32(
            token_ids_tensor,
            Arc::<[i32]>::from(tokens_as_i32(decode_tokens)?.into_boxed_slice()),
            "cohere_decoder_tokens",
        ));
        uploads.push(DecoderUpload::I32(
            position_ids_tensor,
            Arc::<[i32]>::from(
                position_ids_i32_with_offset(position_offset, token_count)?.into_boxed_slice(),
            ),
            "cohere_decoder_positions",
        ));
        let self_attention_mask = if token_count > 1 {
            let mask = graph
                .new_tensor_3d_f16(token_count, token_count, 1, "cohere_decoder_self_mask")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_3d(self_mask)",
                    source,
                })?;
            graph
                .set_input(mask)
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_set_input(self_mask)",
                    source,
                })?;
            let bits = build_causal_mask_f16_bits(
                token_count,
                "cohere_decoder_self_mask",
                |step, source| CohereDecoderGraphError::GraphBuildFailed { step, source },
            )?;
            uploads.push(DecoderUpload::F16Bits(
                mask,
                bits,
                "cohere_decoder_self_mask",
            ));
            Some(mask)
        } else {
            None
        };
        if let Some(row_indices) = self_kv_row_indices {
            uploads.push(DecoderUpload::I32(
                row_indices,
                Arc::<[i32]>::from(
                    vec![i32::try_from(position_offset).map_err(|_| {
                        CohereDecoderGraphError::InvalidInput {
                            reason: format!("decoder position {position_offset} does not fit i32"),
                        }
                    })?]
                    .into_boxed_slice(),
                ),
                "cohere_decoder_self_kv_row",
            ));
        }

        let token_state = graph
            .get_rows(
                self.arena.graph_tensor(self.token_embedding),
                token_ids_tensor,
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(token)",
                source,
            })?;
        let position_state = graph
            .get_rows(
                self.arena.graph_tensor(self.positional_embedding),
                position_ids_tensor,
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(position)",
                source,
            })?;
        let mut state = graph.add(token_state, position_state).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(decoder_embedding)",
                source,
            }
        })?;
        state = apply_affine_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.emb_ln_weight),
            self.arena.graph_tensor(self.emb_ln_bias),
            "decoder_emb_norm",
        )?;
        let mut prompt_debug_tensors = Some(CoherePromptDebugTensors {
            token_state,
            position_state,
            emb_ln: state,
            l0_attn_norm: state,
            l0_q_proj: state,
            l0_k_proj: state,
            l0_v_proj: state,
            h0_after_sa: state,
            h0_after_ca: state,
            h0_after_ffn: state,
            final_state: state,
        });

        state = compose_seq2seq_decoder_layer_stack(
            &mut graph,
            state,
            hidden,
            token_count,
            total_token_count,
            position_offset,
            1,
            self.metadata.decoder_heads,
            &self.layers,
            &self.cross_layers,
            &self.self_kv_layers,
            self_kv_row_indices,
            self_attention_mask,
            &mut uploads,
            &mut prompt_debug_tensors,
        )?;

        state = apply_affine_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.out_ln_weight),
            self.arena.graph_tensor(self.out_ln_bias),
            "decoder_out_norm",
        )?;
        if position_offset == 0
            && let Some(debug) = prompt_debug_tensors.as_mut()
        {
            debug.final_state = state;
        }
        let last_state = view_last_token_state(&graph, state, hidden, token_count)?;
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.output_head_weight), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(output_head)",
                source,
            })?;
        let logits = graph
            .add(logits, self.arena.graph_tensor(self.output_head_bias))
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(output_head_bias)",
                source,
            })?;
        graph
            .set_output(logits)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(logits)",
                source,
            })?;
        let debug_prompt_step =
            std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_some() && position_offset == 0;
        // When the debug-token dump is enabled, the intermediate taps below must
        // also be marked as graph outputs (like logits) before
        // `prepare_outputs_for_upload` runs the gallocr scheduler -- otherwise the
        // scheduler's liveness-based buffer reuse would recycle their backend
        // storage as soon as the last consumer inside the first decoder layer
        // finishes, and the debug read-back below would return reused memory
        // instead of the tap's actual values.
        if debug_prompt_step {
            let debug = prompt_debug_tensors.ok_or(CohereDecoderGraphError::InvalidInput {
                reason: "missing prompt debug tensors for first decoder layer".to_string(),
            })?;
            for tap in [
                debug.token_state,
                debug.position_state,
                debug.emb_ln,
                debug.l0_attn_norm,
                debug.l0_q_proj,
                debug.l0_k_proj,
                debug.l0_v_proj,
                debug.h0_after_sa,
                debug.h0_after_ca,
                debug.h0_after_ffn,
                debug.final_state,
            ] {
                graph.set_output(tap).map_err(|source| {
                    CohereDecoderGraphError::GraphBuildFailed {
                        step: "ggml_set_output(debug_tap)",
                        source,
                    }
                })?;
            }
        }
        // Allocate the decode graph through the scheduler's gallocr for
        // liveness-based buffer reuse before uploading inputs, same ordering as
        // the sibling firered/moonshine decoders.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_prepare_outputs(logits)",
                source,
            })?;
        for upload in uploads {
            upload.apply(&mut graph)?;
        }
        let output = if debug_prompt_step {
            let debug = prompt_debug_tensors.ok_or(CohereDecoderGraphError::InvalidInput {
                reason: "missing prompt debug tensors for first decoder layer".to_string(),
            })?;
            let outputs = graph
                .compute_outputs_f32(&[
                    (logits, self.metadata.vocab_size),
                    (debug.token_state, hidden * token_count),
                    (debug.position_state, hidden * token_count),
                    (debug.emb_ln, hidden * token_count),
                    (debug.l0_attn_norm, hidden * token_count),
                    (debug.l0_q_proj, hidden * token_count),
                    (debug.l0_k_proj, hidden * token_count),
                    (debug.l0_v_proj, hidden * token_count),
                    (debug.h0_after_sa, hidden * token_count),
                    (debug.h0_after_ca, hidden * token_count),
                    (debug.h0_after_ffn, hidden * token_count),
                    (debug.final_state, hidden * token_count),
                ])
                .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?;
            emit_cohere_debug_prompt_intermediates_if_enabled(&outputs);
            outputs
                .into_iter()
                .next()
                .ok_or(CohereDecoderGraphError::InvalidInput {
                    reason: "missing logits output".to_string(),
                })?
        } else {
            graph
                .compute_output_f32(logits, self.metadata.vocab_size)
                .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?
        };
        emit_cohere_debug_step_logits_if_enabled(
            decode_tokens,
            position_offset,
            total_token_count,
            &output,
        );
        self.cached_positions = if use_incremental_self_kv {
            total_token_count
        } else {
            0
        };
        Ok(output)
    }

    fn supports_reusable_decode_graph(&self) -> bool {
        reusable_decode_graph_supported_for_runner(&self.runner)
    }

    fn compute_reused_incremental_step_logits(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, CohereDecoderGraphError> {
        if position >= self.metadata.decoder_max_context {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder position {position} exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }
        let token_id =
            i32::try_from(token_id).map_err(|_| CohereDecoderGraphError::InvalidInput {
                reason: format!("token id {token_id} does not fit i32"),
            })?;
        let position_i32 =
            i32::try_from(position).map_err(|_| CohereDecoderGraphError::InvalidInput {
                reason: format!("decoder position {position} does not fit i32"),
            })?;
        let total_tokens = position
            .checked_add(1)
            .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.max_positions != self.metadata.decoder_max_context || reuse.n_seq != 1
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
        }

        let reuse = self
            .reuse
            .as_mut()
            .expect("cohere reusable decode graph built above");
        let token_tensor = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let attention_mask = reuse.attention_mask;
        let logits = reuse.logits;
        let max_positions = reuse.max_positions;
        let graph = reuse.builder();

        graph
            .set_i32_slice(token_tensor, &[token_id], "cohere_reuse_token")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_token)",
                source,
            })?;
        graph
            .set_i32_slice(row_index, &[position_i32], "cohere_reuse_row")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_row)",
                source,
            })?;
        graph
            .set_i32_slice(position_tensor, &[position_i32], "cohere_reuse_position")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_position)",
                source,
            })?;
        let mask_bits =
            build_fixed_kv_attention_mask_bits(max_positions, total_tokens).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "cohere_reuse_self_mask",
                    source,
                }
            })?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "cohere_reuse_self_mask")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_f16_bits_slice(reuse_mask)",
                source,
            })?;

        let output = graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        self.cached_positions = total_tokens;
        Ok(output)
    }

    pub(super) fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        total_tokens_by_sequence: &[usize],
    ) -> Result<Vec<f32>, CohereDecoderGraphError> {
        if self.n_seq == 1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "batched cohere decode step requires n_seq > 1".to_string(),
            });
        }
        if token_ids.len() != self.n_seq
            || positions.len() != self.n_seq
            || total_tokens_by_sequence.len() != self.n_seq
        {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere decode inputs must have n_seq={} entries",
                    self.n_seq
                ),
            });
        }
        if positions
            .iter()
            .any(|&position| position >= self.metadata.decoder_max_context)
        {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere decoder position exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }
        if total_tokens_by_sequence.iter().any(|&total_tokens| {
            total_tokens == 0 || total_tokens > self.metadata.decoder_max_context
        }) {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere total token count must be in 1..={}",
                    self.metadata.decoder_max_context
                ),
            });
        }

        let token_ids = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| CohereDecoderGraphError::InvalidInput {
                    reason: format!("token id {token_id} does not fit i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positions = positions
            .iter()
            .map(|&position| {
                i32::try_from(position).map_err(|_| CohereDecoderGraphError::InvalidInput {
                    reason: format!("decoder position {position} does not fit i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.max_positions != self.metadata.decoder_max_context
                    || reuse.n_seq != self.n_seq
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
        }

        let reuse = self
            .reuse
            .as_mut()
            .expect("cohere batched reusable decode graph built above");
        let token_tensor = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let attention_mask = reuse.attention_mask;
        let logits = reuse.logits;
        let max_positions = reuse.max_positions;
        let graph = reuse.builder();

        graph
            .set_i32_slice(token_tensor, &token_ids, "cohere_reuse_batch_token")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_batch_token)",
                source,
            })?;
        graph
            .set_i32_slice(row_index, &positions, "cohere_reuse_batch_row")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_batch_row)",
                source,
            })?;
        graph
            .set_i32_slice(position_tensor, &positions, "cohere_reuse_batch_position")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(reuse_batch_position)",
                source,
            })?;
        let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
            max_positions,
            total_tokens_by_sequence,
        )
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: "cohere_reuse_batch_self_mask",
            source,
        })?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "cohere_reuse_batch_self_mask")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_f16_bits_slice(reuse_batch_mask)",
                source,
            })?;

        graph
            .compute_output_f32(logits, self.metadata.vocab_size * self.n_seq)
            .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })
    }

    pub(super) fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, CohereDecoderGraphError> {
        if self.n_seq == 1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "batched cohere prefill requires n_seq > 1".to_string(),
            });
        }
        let token_count = prompt_tokens.len();
        if token_count == 0 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "batched cohere prefill token_count must be > 0".to_string(),
            });
        }
        if token_count > self.metadata.decoder_max_context {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere prefill token_count {} exceeds max context {}",
                    token_count, self.metadata.decoder_max_context
                ),
            });
        }
        let output_tokens = token_count
            .checked_mul(self.n_seq)
            .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
        let prompt_tokens_i32 = tokens_as_i32(prompt_tokens)?;
        let mut token_ids = Vec::with_capacity(output_tokens);
        let mut positions = Vec::with_capacity(output_tokens);
        let mut row_indices = Vec::with_capacity(output_tokens);
        for _ in 0..self.n_seq {
            for (position, &token_id) in prompt_tokens_i32.iter().enumerate() {
                token_ids.push(token_id);
                let position_i32 =
                    i32::try_from(position).map_err(|_| CohereDecoderGraphError::InvalidInput {
                        reason: format!("decoder position {position} does not fit i32"),
                    })?;
                positions.push(position_i32);
                row_indices.push(position_i32);
            }
        }

        self.reuse = None;
        let mut graph = self.runner.start_graph();
        let hidden = self.metadata.decoder_d_model;
        let token_ids_tensor = graph
            .new_tensor_1d_i32(output_tokens, "cohere_prefill_token")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(prefill_token)",
                source,
            })?;
        let position_tensor = graph
            .new_tensor_1d_i32(output_tokens, "cohere_prefill_position")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(prefill_position)",
                source,
            })?;
        let row_index_tensor = graph
            .new_tensor_4d_i32(token_count, 1, self.n_seq, 1, "cohere_prefill_row")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_4d(prefill_row)",
                source,
            })?;
        let attention_mask = graph
            .new_tensor_3d_f16(token_count, token_count, 1, "cohere_prefill_self_mask")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_3d(prefill_mask)",
                source,
            })?;
        graph.set_input(token_ids_tensor).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(prefill_token)",
                source,
            }
        })?;
        graph.set_input(position_tensor).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(prefill_position)",
                source,
            }
        })?;
        graph.set_input(row_index_tensor).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(prefill_row)",
                source,
            }
        })?;
        graph.set_input(attention_mask).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(prefill_mask)",
                source,
            }
        })?;

        let token_state = graph
            .get_rows(
                self.arena.graph_tensor(self.token_embedding),
                token_ids_tensor,
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(prefill_token)",
                source,
            })?;
        let position_state = graph
            .get_rows(
                self.arena.graph_tensor(self.positional_embedding),
                position_tensor,
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(prefill_position)",
                source,
            })?;
        let mut state = graph.add(token_state, position_state).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(prefill_decoder_embedding)",
                source,
            }
        })?;
        state = apply_affine_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.emb_ln_weight),
            self.arena.graph_tensor(self.emb_ln_bias),
            "prefill_decoder_emb_norm",
        )?;
        let mut uploads = Vec::new();
        let mut prompt_debug_tensors = None;
        state = compose_seq2seq_decoder_layer_stack(
            &mut graph,
            state,
            hidden,
            token_count,
            token_count,
            0,
            self.n_seq,
            self.metadata.decoder_heads,
            &self.layers,
            &self.cross_layers,
            &self.self_kv_layers,
            Some(row_index_tensor),
            Some(attention_mask),
            &mut uploads,
            &mut prompt_debug_tensors,
        )?;
        debug_assert!(uploads.is_empty());

        state = apply_affine_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.out_ln_weight),
            self.arena.graph_tensor(self.out_ln_bias),
            "prefill_decoder_out_norm",
        )?;
        let last_state =
            view_batched_last_token_state(&graph, state, hidden, token_count, self.n_seq)?;
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.output_head_weight), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(prefill_output_head)",
                source,
            })?;
        let bias = graph
            .repeat(self.arena.graph_tensor(self.output_head_bias), logits)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_repeat(prefill_output_head_bias)",
                source,
            })?;
        let logits = graph.add(logits, bias).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(prefill_output_head_bias)",
                source,
            }
        })?;
        graph
            .set_output(logits)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(prefill_logits)",
                source,
            })?;
        // Allocate the batched prefill graph through the scheduler's gallocr
        // before uploading inputs, same ordering as the single-step decoder
        // above and the sibling firered/moonshine decoders. `uploads` is always
        // empty here (n_seq > 1 never emits cross-KV deferred writes; see the
        // debug_assert above), so there is nothing to defer past this point.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_prepare_outputs(prefill_logits)",
                source,
            })?;

        graph
            .set_i32_slice(token_ids_tensor, &token_ids, "cohere_prefill_token")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(prefill_token)",
                source,
            })?;
        graph
            .set_i32_slice(position_tensor, &positions, "cohere_prefill_position")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(prefill_position)",
                source,
            })?;
        graph
            .set_i32_slice(row_index_tensor, &row_indices, "cohere_prefill_row")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_i32_slice(prefill_row)",
                source,
            })?;
        let mask_bits =
            build_causal_mask_f16_bits(token_count, "cohere_prefill_self_mask", |step, source| {
                CohereDecoderGraphError::GraphBuildFailed { step, source }
            })?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "cohere_prefill_self_mask")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_f16_bits_slice(prefill_mask)",
                source,
            })?;

        let output = graph
            .compute_output_f32(logits, self.metadata.vocab_size * self.n_seq)
            .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        self.cached_positions = token_count;
        Ok(output)
    }

    fn build_reusable_decode_graph(&mut self) -> Result<(), CohereDecoderGraphError> {
        let hidden = self.metadata.decoder_d_model;
        let max_context = self.metadata.decoder_max_context;
        let n_seq = self.n_seq;
        let mut session = self
            .runner
            .start_persistent_graph_session(self.persistent_graph_context_bytes)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "cohere_reuse_session",
                source,
            })?;
        let graph = session.builder();
        let token_id = graph
            .new_tensor_1d_i32(n_seq, "cohere_reuse_token")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(reuse_token)",
                source,
            })?;
        let row_index = if n_seq == 1 {
            graph
                .new_tensor_1d_i32(1, "cohere_reuse_row")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_1d(reuse_row)",
                    source,
                })?
        } else {
            graph
                .new_tensor_4d_i32(1, 1, n_seq, 1, "cohere_reuse_row")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_4d(reuse_row)",
                    source,
                })?
        };
        let position = graph
            .new_tensor_1d_i32(n_seq, "cohere_reuse_position")
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_new_tensor_1d(reuse_position)",
                source,
            })?;
        let attention_mask = if n_seq == 1 {
            graph
                .new_tensor_3d_f16(max_context, 1, 1, "cohere_reuse_self_mask")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_3d(reuse_mask)",
                    source,
                })?
        } else {
            graph
                .new_tensor_4d_f16(max_context, 1, 1, n_seq, "cohere_reuse_self_mask")
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_new_tensor_4d(reuse_mask)",
                    source,
                })?
        };
        graph
            .set_input(token_id)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(reuse_token)",
                source,
            })?;
        graph
            .set_input(row_index)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(reuse_row)",
                source,
            })?;
        graph
            .set_input(position)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(reuse_position)",
                source,
            })?;
        graph.set_input(attention_mask).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_input(reuse_mask)",
                source,
            }
        })?;

        let token_state = graph
            .get_rows(self.arena.graph_tensor(self.token_embedding), token_id)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(reuse_token)",
                source,
            })?;
        let position_state = graph
            .get_rows(self.arena.graph_tensor(self.positional_embedding), position)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(reuse_position)",
                source,
            })?;
        let mut state = graph.add(token_state, position_state).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(reuse_decoder_embedding)",
                source,
            }
        })?;
        state = apply_affine_norm(
            graph,
            state,
            self.arena.graph_tensor(self.emb_ln_weight),
            self.arena.graph_tensor(self.emb_ln_bias),
            "reuse_decoder_emb_norm",
        )?;
        let mut uploads = Vec::new();
        let mut prompt_debug_tensors = None;
        state = compose_seq2seq_decoder_layer_stack(
            graph,
            state,
            hidden,
            1,
            max_context,
            0,
            self.n_seq,
            self.metadata.decoder_heads,
            &self.layers,
            &self.cross_layers,
            &self.self_kv_layers,
            Some(row_index),
            Some(attention_mask),
            &mut uploads,
            &mut prompt_debug_tensors,
        )?;
        debug_assert!(uploads.is_empty());

        state = apply_affine_norm(
            graph,
            state,
            self.arena.graph_tensor(self.out_ln_weight),
            self.arena.graph_tensor(self.out_ln_bias),
            "reuse_decoder_out_norm",
        )?;
        let last_state = if n_seq == 1 {
            view_last_token_state(graph, state, hidden, 1)?
        } else {
            state
        };
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.output_head_weight), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(reuse_output_head)",
                source,
            })?;
        let output_head_bias = self.arena.graph_tensor(self.output_head_bias);
        let logits = if n_seq == 1 {
            graph.add(logits, output_head_bias).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_add(reuse_output_head_bias)",
                    source,
                }
            })?
        } else {
            let bias = graph.repeat(output_head_bias, logits).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_repeat(reuse_output_head_bias)",
                    source,
                }
            })?;
            graph
                .add(logits, bias)
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_add(reuse_output_head_bias)",
                    source,
                })?
        };
        graph
            .set_output(logits)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(reuse_logits)",
                source,
            })?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_prepare_outputs(reuse_logits)",
                source,
            })?;

        self.reuse = Some(Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena(
            session,
            max_context,
            n_seq,
            token_id,
            row_index,
            position,
            attention_mask,
            logits,
        ));
        Ok(())
    }
}

#[derive(Clone)]
enum DecoderUpload<'a> {
    I32(
        crate::ggml_runtime::GgmlCpuTensor<'a>,
        Arc<[i32]>,
        &'static str,
    ),
    F16Bits(
        crate::ggml_runtime::GgmlCpuTensor<'a>,
        Arc<[u16]>,
        &'static str,
    ),
}

impl<'a> DecoderUpload<'a> {
    fn apply(
        self,
        graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    ) -> Result<(), CohereDecoderGraphError> {
        match self {
            Self::I32(tensor, values, step) => graph
                .set_i32_slice(tensor, &values, step)
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed { step, source }),
            Self::F16Bits(tensor, values, step) => graph
                .set_f16_bits_slice(tensor, &values, step)
                .map_err(|source| CohereDecoderGraphError::GraphBuildFailed { step, source }),
        }
    }
}

/// Cohere adapter around the shared `nn::decoder::seq2seq_layer_stack` driver.
/// The family-specific layer body stays in `apply_decoder_layer`, preserving the
/// exact op sequence, layer-0/prefill debug capture, and deferred upload order.
#[allow(clippy::too_many_arguments)]
fn compose_seq2seq_decoder_layer_stack<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    total_token_count: usize,
    position_offset: usize,
    n_seq: usize,
    attention_heads: usize,
    layers: &[CohereDecoderLayerRuntime],
    cross_layers: &[CohereDecoderCrossCacheLayerRuntime],
    self_kv_layers: &[CohereDecoderSelfKvLayerRuntime],
    self_kv_row_indices: Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    self_attention_mask: Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    prompt_debug_tensors: &mut Option<CoherePromptDebugTensors<'a>>,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    seq2seq_layer_stack(
        graph,
        state,
        layers,
        cross_layers,
        self_kv_layers,
        |length| CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "decoder layer-stack length mismatch: layers={}, cross_layers={}, self_kv_layers={}",
                length.layers, length.cross_layers, length.self_kv_layers
            ),
        },
        |graph, state, layer_idx, layer_runtime, cross_runtime, self_kv_runtime| {
            apply_decoder_layer(
                graph,
                state,
                hidden,
                token_count,
                total_token_count,
                position_offset,
                n_seq,
                attention_heads,
                layer_runtime,
                cross_runtime,
                self_kv_runtime,
                self_kv_row_indices,
                self_attention_mask,
                uploads,
                if layer_idx == 0 && position_offset == 0 {
                    Some(&mut *prompt_debug_tensors)
                } else {
                    None
                },
            )
        },
    )
}

fn apply_decoder_layer<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    total_token_count: usize,
    position_offset: usize,
    n_seq: usize,
    attention_heads: usize,
    layer: &CohereDecoderLayerRuntime,
    cross_runtime: &CohereDecoderCrossCacheLayerRuntime,
    self_kv: &CohereDecoderSelfKvLayerRuntime,
    self_kv_row_indices: Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    self_attention_mask: Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    mut prompt_debug_tensors: Option<&mut Option<CoherePromptDebugTensors<'a>>>,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    use crate::nn::decoder::{
        CrossKvHandle, SelfKvHandle, Seq2SeqLayerConfig, Seq2SeqLayerWeights, seq2seq_layer,
    };

    let self_attn_input = state;
    let head_dim = hidden
        .checked_div(attention_heads)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    validate_self_kv_step(
        self_kv,
        hidden,
        token_count,
        total_token_count,
        position_offset,
        attention_heads,
        self_kv_row_indices.is_some() && self_attention_mask.is_some(),
    )?;

    let config = Seq2SeqLayerConfig {
        hidden,
        attention_heads,
        head_dim,
        token_count,
        n_seq,
        total_token_count,
        position_offset,
        layer_norm_epsilon: COHERE_DECODER_LAYER_NORM_EPSILON,
        ffn_activation: crate::nn::ffn::FeedForwardActivation::Relu,
        self_kv_max_positions: self_kv.max_positions,
        cross_frame_count: cross_runtime.frame_count,
        cross_hidden_size: cross_runtime.hidden_size,
    };
    let weights = Seq2SeqLayerWeights {
        self_attn_norm_weight: layer.attn_ln_weight.as_graph_tensor(),
        self_attn_norm_bias: layer.attn_ln_bias.as_graph_tensor(),
        self_attn_q_weight: layer.attn_q_weight.as_graph_tensor(),
        self_attn_q_bias: layer.attn_q_bias.as_graph_tensor(),
        self_attn_k_weight: layer.attn_k_weight.as_graph_tensor(),
        self_attn_k_bias: layer.attn_k_bias.as_graph_tensor(),
        self_attn_v_weight: layer.attn_v_weight.as_graph_tensor(),
        self_attn_v_bias: layer.attn_v_bias.as_graph_tensor(),
        self_attn_o_weight: layer.attn_o_weight.as_graph_tensor(),
        self_attn_o_bias: layer.attn_o_bias.as_graph_tensor(),
        cross_attn_norm_weight: layer.cross_ln_weight.as_graph_tensor(),
        cross_attn_norm_bias: layer.cross_ln_bias.as_graph_tensor(),
        cross_attn_q_weight: layer.cross_q_weight.as_graph_tensor(),
        cross_attn_q_bias: layer.cross_q_bias.as_graph_tensor(),
        cross_attn_o_weight: layer.cross_o_weight.as_graph_tensor(),
        cross_attn_o_bias: layer.cross_o_bias.as_graph_tensor(),
        ffn_norm_weight: layer.ffn_ln_weight.as_graph_tensor(),
        ffn_norm_bias: layer.ffn_ln_bias.as_graph_tensor(),
        ffn_up_weight: layer.ffn_up_weight.as_graph_tensor(),
        ffn_up_bias: layer.ffn_up_bias.as_graph_tensor(),
        ffn_down_weight: layer.ffn_down_weight.as_graph_tensor(),
        ffn_down_bias: layer.ffn_down_bias.as_graph_tensor(),
    };
    let self_kv_handle = SelfKvHandle {
        key: self_kv.key.as_graph_tensor(),
        value: self_kv.value.as_graph_tensor(),
        row_indices: self_kv_row_indices,
        attention_mask: self_attention_mask,
    };
    let cross_kv_handle = CrossKvHandle {
        key: cross_runtime.key.as_graph_tensor(),
        value: cross_runtime.value.as_graph_tensor(),
    };

    let block = seq2seq_layer(
        graph,
        state,
        config,
        weights,
        self_kv_handle,
        cross_kv_handle,
        |step, source| CohereDecoderGraphError::GraphBuildFailed { step, source },
    )?;

    if let Some((mask, bits)) = block.deferred_self_mask {
        uploads.push(DecoderUpload::F16Bits(
            mask,
            bits,
            "cohere_decoder_layer_self_mask",
        ));
    }
    if let Some(slot) = prompt_debug_tensors.as_mut()
        && let Some(debug) = slot.as_mut()
    {
        debug.emb_ln = self_attn_input;
        debug.l0_attn_norm = block.taps.self_attn_norm;
        debug.l0_q_proj = block.taps.q_proj;
        debug.l0_k_proj = block.taps.k_proj;
        debug.l0_v_proj = block.taps.v_proj;
        debug.h0_after_sa = block.taps.after_self_attn;
        debug.h0_after_ca = block.taps.after_cross_attn;
        debug.h0_after_ffn = block.taps.after_ffn;
    }
    Ok(block.output)
}

fn emit_cohere_debug_prompt_intermediates_if_enabled(outputs: &[Vec<f32>]) {
    if std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_none() || outputs.len() < 12 {
        return;
    }
    let labels = [
        "token_state",
        "position_state",
        "emb_ln",
        "l0_attn_norm",
        "l0_q_proj",
        "l0_k_proj",
        "l0_v_proj",
        "h0_after_sa",
        "h0_after_ca",
        "h0_after_ffn",
        "final_state",
    ];
    for (label, values) in labels.iter().zip(outputs.iter().skip(1)) {
        let preview = values
            .iter()
            .take(5)
            .map(|value| format!("{value:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("openasr cohere step 0 (prompt) {label}: [{preview}]");
    }
}

fn validate_decoder_runtime_shapes(
    decoder_weights: &CohereTranscribeDecoderWeights,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<(), CohereDecoderGraphError> {
    if decoder_weights.layers.len() != metadata.decoder_layers {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "decoder layer count mismatch: weights={}, metadata={}",
                decoder_weights.layers.len(),
                metadata.decoder_layers
            ),
        });
    }
    Ok(())
}

fn validate_encoder_cross_dimensions(
    hidden_size: usize,
    frame_count: usize,
    metadata: CohereTranscribeExecutionMetadata,
    decoder_layers: usize,
) -> Result<(), CohereDecoderGraphError> {
    if hidden_size != metadata.decoder_d_model {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "cross-cache hidden size {} does not match decoder d_model {}",
                hidden_size, metadata.decoder_d_model
            ),
        });
    }
    if frame_count == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "cross-cache frame_count must be > 0".to_string(),
        });
    }
    if decoder_layers != metadata.decoder_layers {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "cross-cache layer count {} does not match decoder layer count {}",
                decoder_layers, metadata.decoder_layers,
            ),
        });
    }
    Ok(())
}

fn decoder_max_generated_tokens(
    prompt_tokens: &[u32],
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<usize, CohereDecoderGraphError> {
    if prompt_tokens.is_empty() {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "decode prompt must contain at least one token".to_string(),
        });
    }
    context_window_budget(metadata.decoder_max_context, prompt_tokens.len()).ok_or_else(|| {
        CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "decode prompt length {} exhausts decoder max context {}",
                prompt_tokens.len(),
                metadata.decoder_max_context
            ),
        }
    })
}

pub(super) fn decoder_max_generated_tokens_with_env(
    prompt_tokens: &[u32],
    metadata: CohereTranscribeExecutionMetadata,
    encoder_frame_count: usize,
) -> Result<usize, CohereDecoderGraphError> {
    let context_limited = decoder_max_generated_tokens(prompt_tokens, metadata)?;
    let heuristic_budget =
        decoder_max_generated_tokens_budget_from_encoder_frames(encoder_frame_count);
    let max_generated_tokens = context_limited.min(heuristic_budget);
    let Some(raw) = std::env::var_os(COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV) else {
        return Ok(max_generated_tokens);
    };
    let Some(raw) = raw
        .to_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(max_generated_tokens);
    };
    let override_value = raw.parse::<usize>().map_err(|_| {
        CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "{COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV} must be a positive integer, got '{raw}'"
            ),
        }
    })?;
    if override_value == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!("{COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV} must be > 0 when set"),
        });
    }
    Ok(max_generated_tokens.min(override_value))
}

fn decoder_max_generated_tokens_budget_from_encoder_frames(encoder_frame_count: usize) -> usize {
    // Guard against decoder runaway when EOT is never reached: short utterances
    // should not consume the full decoder context budget.
    let scaled = encoder_frame_count.saturating_mul(4);
    scaled.clamp(64, 512)
}

fn emit_cohere_debug_step_logits_if_enabled(
    decode_tokens: &[u32],
    position_offset: usize,
    total_token_count: usize,
    logits: &[f32],
) {
    if std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_none() || logits.is_empty() {
        return;
    }
    let mut top_token = 0usize;
    for token_id in 1..logits.len() {
        if logits[token_id] > logits[top_token] {
            top_token = token_id;
        }
    }
    eprintln!(
        "openasr cohere step logits: token_count={} position_offset={} total_token_count={} top_token={} top_logit={:.4} input_tokens={:?}",
        decode_tokens.len(),
        position_offset,
        total_token_count,
        top_token,
        logits[top_token],
        decode_tokens,
    );
}

#[cfg_attr(not(test), allow(dead_code))]
fn project_hidden_sequence_with_bias(
    weight: &CohereMatrixWeight,
    bias: &CohereVectorWeight,
    input_rows: &[f32],
    input_width: usize,
    row_count: usize,
) -> Result<Vec<f32>, CohereDecoderGraphError> {
    if bias.len != weight.rows {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "bias width {} does not match output width {} for {}",
                bias.len, weight.rows, weight.name
            ),
        });
    }
    let bias_values = vector_values_for_cpu(bias)?;
    let expected = input_width
        .checked_mul(row_count)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    if input_rows.len() != expected {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "projection input length mismatch: got {}, expected {}",
                input_rows.len(),
                expected
            ),
        });
    }
    if weight.cols != input_width {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "projection input width mismatch for {}: weight cols={} input={}",
                weight.name, weight.cols, input_width
            ),
        });
    }
    let mut out = vec![0.0_f32; row_count * weight.rows];
    match weight.layout {
        CohereMatrixLayout::RowsByColumns => {
            for row_idx in 0..row_count {
                let input = &input_rows[row_idx * input_width..(row_idx + 1) * input_width];
                let output = &mut out[row_idx * weight.rows..(row_idx + 1) * weight.rows];
                output.copy_from_slice(&bias_values);
                for (out_idx, out_value) in output.iter_mut().enumerate() {
                    let weight_row =
                        &weight.values[out_idx * input_width..(out_idx + 1) * input_width];
                    let mut acc = *out_value;
                    for input_idx in 0..input_width {
                        acc += input[input_idx] * weight_row[input_idx];
                    }
                    *out_value = acc;
                }
            }
        }
        CohereMatrixLayout::ColumnsByRows => {
            if weight.rows == weight.cols {
                for row_idx in 0..row_count {
                    let input = &input_rows[row_idx * input_width..(row_idx + 1) * input_width];
                    let output = &mut out[row_idx * weight.rows..(row_idx + 1) * weight.rows];
                    output.copy_from_slice(&bias_values);
                    for (out_idx, out_value) in output.iter_mut().enumerate().take(weight.rows) {
                        let weight_row =
                            &weight.values[out_idx * input_width..(out_idx + 1) * input_width];
                        let mut acc = *out_value;
                        for (input_idx, input_value) in input.iter().enumerate().take(input_width) {
                            acc += *input_value * weight_row[input_idx];
                        }
                        *out_value = acc;
                    }
                }
            } else {
                for row_idx in 0..row_count {
                    let input = &input_rows[row_idx * input_width..(row_idx + 1) * input_width];
                    let output = &mut out[row_idx * weight.rows..(row_idx + 1) * weight.rows];
                    output.copy_from_slice(&bias_values);
                    for (input_idx, input_value) in input.iter().enumerate().take(input_width) {
                        let weight_row =
                            &weight.values[input_idx * weight.rows..(input_idx + 1) * weight.rows];
                        for out_idx in 0..weight.rows {
                            output[out_idx] += *input_value * weight_row[out_idx];
                        }
                    }
                }
            }
        }
    }
    if out.iter().any(|value| !value.is_finite()) {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "projection output for '{}' contains non-finite values",
                weight.name
            ),
        });
    }
    Ok(out)
}

fn new_vector_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    len: usize,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, CohereDecoderGraphError> {
    arena.new_tensor_1d_f32(len, tensor_name).map_err(|source| {
        CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        }
    })
}

fn new_projection_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, CohereDecoderGraphError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.cols, weight.rows]
    {
        return arena
            .new_matmul_weight_2d_typed(weight.cols, weight.rows, raw.ggml_type, tensor_name)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: tensor_name,
                source,
            });
    }
    arena
        .new_tensor_2d_f32(weight.cols, weight.rows, tensor_name)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        })
}

fn new_embedding_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, CohereDecoderGraphError> {
    arena
        .new_tensor_2d_f32(weight.cols, weight.rows, tensor_name)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        })
}

fn new_persistent_cross_cache_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    hidden_size: usize,
    frame_count: usize,
    n_seq: usize,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, CohereDecoderGraphError> {
    let result = if n_seq == 1 {
        arena.new_tensor_2d_f32(hidden_size, frame_count, tensor_name)
    } else {
        arena.new_tensor_3d_f32(hidden_size, frame_count, n_seq, tensor_name)
    };
    result.map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
        step: tensor_name,
        source,
    })
}

fn new_persistent_self_kv_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    head_dim: usize,
    max_positions: usize,
    attention_heads: usize,
    n_seq: usize,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, CohereDecoderGraphError> {
    let result = if n_seq == 1 {
        arena.new_tensor_3d_f16(head_dim, max_positions, attention_heads, tensor_name)
    } else {
        arena.new_tensor_4d_f16(head_dim, max_positions, attention_heads, n_seq, tensor_name)
    };
    result.map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
        step: tensor_name,
        source,
    })
}

fn upload_vector_to_arena(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &CohereVectorWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.len]
        && arena
            .set_bytes_slice(tensor, raw.bytes(), tensor_name)
            .is_ok()
    {
        return Ok(());
    }
    let values = vector_values_for_cpu(weight)?;
    arena
        .set_f32_slice(tensor, &values, tensor_name)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        })
}

fn vector_values_for_cpu(
    weight: &CohereVectorWeight,
) -> Result<std::borrow::Cow<'_, [f32]>, CohereDecoderGraphError> {
    if !weight.values.is_empty() {
        return Ok(std::borrow::Cow::Borrowed(&weight.values));
    }
    let Some(raw) = &weight.raw_ggml else {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "vector {} has neither eager values nor raw payload",
                weight.name
            ),
        });
    };
    if raw.dims.as_slice() != [weight.len] {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "vector {} raw dims {:?} do not match expected len {}",
                weight.name, raw.dims, weight.len
            ),
        });
    }
    if raw.ggml_type != crate::ggml_runtime::GGML_TYPE_F32 {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "vector {} raw ggml type {} is not runtime f32",
                weight.name, raw.ggml_type
            ),
        });
    }
    let mut values = Vec::with_capacity(weight.len);
    for chunk in raw.bytes().chunks_exact(std::mem::size_of::<f32>()) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if values.len() != weight.len {
        return Err(CohereDecoderGraphError::InvalidWeight {
            reason: format!(
                "vector {} decoded len {} does not match expected {}",
                weight.name,
                values.len(),
                weight.len
            ),
        });
    }
    Ok(std::borrow::Cow::Owned(values))
}

fn upload_projection_to_arena(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.cols, weight.rows]
        && arena
            .set_bytes_slice(tensor, raw.bytes(), tensor_name)
            .is_ok()
    {
        return Ok(());
    }
    let values = projection_values_for_ggml(weight)?;
    arena
        .set_f32_slice(tensor, &values, tensor_name)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        })
}

fn upload_embedding_to_arena(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.rows, weight.cols]
        && arena
            .set_bytes_slice(tensor, raw.bytes(), tensor_name)
            .is_ok()
    {
        return Ok(());
    }
    arena
        .set_f32_slice(tensor, &weight.values, tensor_name)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: tensor_name,
            source,
        })
}

fn upload_decoder_layer_to_arena(
    arena: &mut GgmlStaticTensorArena,
    runtime: &CohereDecoderLayerRuntime,
    layer: &CohereDecoderLayerWeights,
    layer_idx: usize,
) -> Result<(), CohereDecoderGraphError> {
    let _ = layer_idx;
    upload_vector_to_arena(
        arena,
        runtime.attn_ln_weight,
        &layer.attn_ln_weight,
        "dec_attn_ln_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.attn_ln_bias,
        &layer.attn_ln_bias,
        "dec_attn_ln_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.attn_q_weight,
        &layer.attn_q_weight,
        "dec_attn_q_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.attn_q_bias,
        &layer.attn_q_bias,
        "dec_attn_q_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.attn_k_weight,
        &layer.attn_k_weight,
        "dec_attn_k_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.attn_k_bias,
        &layer.attn_k_bias,
        "dec_attn_k_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.attn_v_weight,
        &layer.attn_v_weight,
        "dec_attn_v_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.attn_v_bias,
        &layer.attn_v_bias,
        "dec_attn_v_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.attn_o_weight,
        &layer.attn_o_weight,
        "dec_attn_o_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.attn_o_bias,
        &layer.attn_o_bias,
        "dec_attn_o_b",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_ln_weight,
        &layer.cross_ln_weight,
        "dec_cross_ln_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_ln_bias,
        &layer.cross_ln_bias,
        "dec_cross_ln_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.cross_k_weight,
        &layer.cross_k_weight,
        "dec_cross_k_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_k_bias,
        &layer.cross_k_bias,
        "dec_cross_k_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.cross_v_weight,
        &layer.cross_v_weight,
        "dec_cross_v_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_v_bias,
        &layer.cross_v_bias,
        "dec_cross_v_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.cross_q_weight,
        &layer.cross_q_weight,
        "dec_cross_q_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_q_bias,
        &layer.cross_q_bias,
        "dec_cross_q_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.cross_o_weight,
        &layer.cross_o_weight,
        "dec_cross_o_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.cross_o_bias,
        &layer.cross_o_bias,
        "dec_cross_o_b",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.ffn_ln_weight,
        &layer.ffn_ln_weight,
        "dec_ffn_ln_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.ffn_ln_bias,
        &layer.ffn_ln_bias,
        "dec_ffn_ln_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.ffn_up_weight,
        &layer.ffn_up_weight,
        "dec_ffn_up_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.ffn_up_bias,
        &layer.ffn_up_bias,
        "dec_ffn_up_b",
    )?;
    upload_projection_to_arena(
        arena,
        runtime.ffn_down_weight,
        &layer.ffn_down_weight,
        "dec_ffn_down_w",
    )?;
    upload_vector_to_arena(
        arena,
        runtime.ffn_down_bias,
        &layer.ffn_down_bias,
        "dec_ffn_down_b",
    )
}

fn projection_values_for_ggml(
    weight: &CohereMatrixWeight,
) -> Result<Vec<f32>, CohereDecoderGraphError> {
    match weight.layout {
        CohereMatrixLayout::RowsByColumns => {
            transpose_matrix(&weight.values, weight.rows, weight.cols)
        }
        CohereMatrixLayout::ColumnsByRows => Ok(weight.values.clone()),
    }
}

fn transpose_matrix(
    values: &[f32],
    src_rows: usize,
    src_cols: usize,
) -> Result<Vec<f32>, CohereDecoderGraphError> {
    let expected = src_rows
        .checked_mul(src_cols)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    if values.len() != expected {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "matrix transpose expected {} values, got {}",
                expected,
                values.len()
            ),
        });
    }
    let mut out = vec![0.0_f32; expected];
    for row in 0..src_rows {
        for col in 0..src_cols {
            out[col * src_rows + row] = values[row * src_cols + col];
        }
    }
    Ok(out)
}

fn apply_affine_norm<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    input: crate::ggml_runtime::GgmlCpuTensor<'a>,
    weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    apply_affine_layer_norm(
        graph,
        input,
        COHERE_DECODER_LAYER_NORM_EPSILON,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: step,
            bias: step,
        },
        |step, source| CohereDecoderGraphError::GraphBuildFailed { step, source },
    )
}

fn apply_linear_with_bias<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    input: crate::ggml_runtime::GgmlCpuTensor<'a>,
    weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    let projected = graph
        .mul_mat(weight, input)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed { step, source })?;
    graph
        .add(projected, bias)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed { step, source })
}

fn cross_cache_slot_target<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    cache: crate::ggml_runtime::GgmlCpuTensor<'a>,
    hidden_size: usize,
    frame_count: usize,
    n_seq: usize,
    slot_index: usize,
    step: &'static str,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    if n_seq == 1 {
        return Ok(cache);
    }
    let row_stride = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    let slot_stride = hidden_size
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    let offset = slot_index
        .checked_mul(slot_stride)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(cache, hidden_size, frame_count, row_stride, offset)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed { step, source })
}

fn view_last_token_state<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    hidden: usize,
    prefix_len: usize,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    let contiguous_state =
        graph
            .cont(state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_cont(last_token_state)",
                source,
            })?;
    let row_stride = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    let offset = prefix_len
        .checked_sub(1)
        .and_then(|index| index.checked_mul(row_stride))
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(contiguous_state, hidden, 1, row_stride, offset)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: "ggml_view_2d(last_token_state)",
            source,
        })
}

fn view_batched_last_token_state<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    n_seq: usize,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    if token_count == 0 || n_seq == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "batched last-token view requires positive token_count and n_seq".to_string(),
        });
    }
    let contiguous_state =
        graph
            .cont(state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_cont(batched_last_token_state)",
                source,
            })?;
    let column_stride = hidden
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    let offset = token_count
        .checked_sub(1)
        .and_then(|index| index.checked_mul(hidden))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(contiguous_state, hidden, n_seq, column_stride, offset)
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: "ggml_view_2d(batched_last_token_state)",
            source,
        })
}

fn tokens_as_i32(tokens: &[u32]) -> Result<Vec<i32>, CohereDecoderGraphError> {
    tokens
        .iter()
        .copied()
        .map(|token| {
            i32::try_from(token).map_err(|_| CohereDecoderGraphError::InvalidInput {
                reason: format!("token id {token} does not fit i32"),
            })
        })
        .collect()
}

fn position_ids_i32_with_offset(
    position_offset: usize,
    token_count: usize,
) -> Result<Vec<i32>, CohereDecoderGraphError> {
    (position_offset..position_offset.saturating_add(token_count))
        .map(|index| {
            i32::try_from(index).map_err(|_| CohereDecoderGraphError::InvalidInput {
                reason: format!("position index {index} does not fit i32"),
            })
        })
        .collect()
}

fn validate_self_kv_step(
    self_kv: &CohereDecoderSelfKvLayerRuntime,
    hidden: usize,
    token_count: usize,
    total_token_count: usize,
    position_offset: usize,
    attention_heads: usize,
    allow_fixed_kv_span: bool,
) -> Result<(), CohereDecoderGraphError> {
    if token_count == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "self-KV step token_count must be > 0".to_string(),
        });
    }
    if attention_heads == 0 || !hidden.is_multiple_of(attention_heads) {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "self-KV requires hidden size {hidden} divisible by attention heads {attention_heads}"
            ),
        });
    }
    if total_token_count > self_kv.max_positions {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "self-KV total tokens {} exceed max positions {}",
                total_token_count, self_kv.max_positions
            ),
        });
    }
    if allow_fixed_kv_span {
        if token_count == 1 {
            return Ok(());
        }
        if position_offset == 0 {
            if token_count != total_token_count {
                return Err(CohereDecoderGraphError::InvalidInput {
                    reason: format!(
                        "self-KV fixed-span prefill mismatch: token_count={token_count} total_token_count={total_token_count}"
                    ),
                });
            }
            return Ok(());
        }
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "self-KV fixed-span path requires either one token or whole-prefix prefill at offset 0, got position_offset={} token_count={} total_token_count={}",
                position_offset, token_count, total_token_count
            ),
        });
    }
    if position_offset == 0 {
        if token_count != total_token_count {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "self-KV prefill mismatch: token_count={token_count} total_token_count={total_token_count}"
                ),
            });
        }
        return Ok(());
    }
    if token_count != 1 || total_token_count != position_offset.saturating_add(1) {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "self-KV incremental path requires one token at offset {}, got token_count={} total_token_count={}",
                position_offset, token_count, total_token_count
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::validate_ggml_runtime_source_path;
    use crate::{
        GgmlAsrExecutionOptions, GgmlAsrRuntimeSourcePreflight, GgufMetadata, GgufMetadataValue,
        read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
    };
    use tempfile::{NamedTempFile, TempPath};

    fn assert_logits_select_same_token(batched: &[f32], serial: &[f32], label: &str) {
        assert_eq!(
            batched.len(),
            serial.len(),
            "{label} logits length mismatch"
        );
        assert!(!batched.is_empty(), "{label} logits must not be empty");
        let mut batched_top = 0usize;
        let mut serial_top = 0usize;
        let mut dot = 0.0_f64;
        let mut batched_norm = 0.0_f64;
        let mut serial_norm = 0.0_f64;
        let mut max_abs_diff = 0.0_f32;
        for (index, (&batched_value, &serial_value)) in batched.iter().zip(serial).enumerate() {
            if batched_value > batched[batched_top] {
                batched_top = index;
            }
            if serial_value > serial[serial_top] {
                serial_top = index;
            }
            dot += f64::from(batched_value) * f64::from(serial_value);
            batched_norm += f64::from(batched_value) * f64::from(batched_value);
            serial_norm += f64::from(serial_value) * f64::from(serial_value);
            max_abs_diff = max_abs_diff.max((batched_value - serial_value).abs());
        }
        let cosine = dot / (batched_norm.sqrt() * serial_norm.sqrt());
        assert_eq!(
            batched_top, serial_top,
            "{label} top token mismatch: batched_top={batched_top} serial_top={serial_top} cosine={cosine:.6} max_abs_diff={max_abs_diff:.6}"
        );
        assert!(
            cosine > 0.95,
            "{label} logits drift too far: cosine={cosine:.6} max_abs_diff={max_abs_diff:.6}"
        );
    }

    fn write_runtime_ready_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata,
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    fn sample_encoder_output(
        metadata: CohereTranscribeExecutionMetadata,
    ) -> CohereTranscribeEncoderOutput {
        let frame_count = 4;
        let mut rows = Vec::with_capacity(frame_count * metadata.decoder_d_model);
        for frame_idx in 0..frame_count {
            for hidden_idx in 0..metadata.decoder_d_model {
                rows.push(
                    ((frame_idx * metadata.decoder_d_model + hidden_idx) as f32 * 0.03125).sin(),
                );
            }
        }
        CohereTranscribeEncoderOutput {
            frame_count,
            hidden_size: metadata.decoder_d_model,
            rows,
        }
    }

    fn diarization_tokenizer() -> CohereTranscribeTokenizer {
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|spltoken0|>".to_string(),
                "<|spltoken1|>".to_string(),
                "<|t:0.0|>".to_string(),
                "<|t:1.2|>".to_string(),
                "<|t:2.4|>".to_string(),
                "▁Hello".to_string(),
                "▁there".to_string(),
                "▁Thanks".to_string(),
            ]),
        );
        CohereTranscribeTokenizer::from_gguf_metadata(&GgufMetadata::from_values_for_test(values))
            .expect("tokenizer")
    }

    #[test]
    fn parses_cohere_diarization_token_stream_into_speaker_segments() {
        let tokenizer = diarization_tokenizer();
        let decode_text_token_ids = |token_ids: &[u32]| {
            tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                CohereTranscribeGreedyDecodeError::TokenizerDecodeFailed {
                    reason: error.to_string(),
                }
            })
        };

        let segments = cohere_diarized_segments_from_generated_tokens(
            &tokenizer,
            &[0, 2, 5, 6, 3, 1, 3, 7, 4],
            2.4,
            &decode_text_token_ids,
        )
        .expect("segments");

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 1.2);
        assert_eq!(segments[0].text, "Hello there");
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].start, 1.2);
        assert_eq!(segments[1].end, 2.4);
        assert_eq!(segments[1].text, "Thanks");
    }

    #[test]
    fn cohere_diarization_parser_does_not_invent_speakers() {
        let tokenizer = diarization_tokenizer();
        let decode_text_token_ids = |token_ids: &[u32]| {
            tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                CohereTranscribeGreedyDecodeError::TokenizerDecodeFailed {
                    reason: error.to_string(),
                }
            })
        };

        let segments = cohere_diarized_segments_from_generated_tokens(
            &tokenizer,
            &[5, 6],
            2.4,
            &decode_text_token_ids,
        )
        .expect("segments");

        assert!(segments.is_empty());
    }

    #[test]
    fn cross_cache_builds_finite_layer_rows() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let decoder_weights =
            super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                &reader, metadata,
            )
            .expect("decoder weights");
        let encoder_output = sample_encoder_output(metadata);

        let cache = build_cohere_cross_attention_cache_from_encoder_output(
            &decoder_weights,
            metadata,
            &encoder_output,
        )
        .expect("cross cache");

        assert_eq!(cache.layers.len(), metadata.decoder_layers);
        assert_eq!(cache.frame_count, encoder_output.frame_count);
        assert!(
            cache
                .layers
                .iter()
                .all(|layer| layer.key_rows.iter().all(|value| value.is_finite()))
        );
        assert!(
            cache
                .layers
                .iter()
                .all(|layer| layer.value_rows.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn decoder_runtime_emits_finite_step_logits() {
        with_forced_cpu_backend_for_test(|| {
            let (_runtime_path, preflight) = write_runtime_ready_preflight();
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
            let decoder_weights =
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights");
            let tokenizer = super::super::tokenizer::CohereTranscribeTokenizer::from_gguf_metadata(
                &preflight.metadata,
            )
            .expect("tokenizer");
            let prompt = super::super::prompt::build_cohere_transcribe_decode_prompt(
                &tokenizer,
                metadata.decoder_start_token_id,
                Some("en"),
                &GgmlAsrExecutionOptions::default(),
            )
            .expect("prompt");
            let encoder_output = sample_encoder_output(metadata);
            let mut runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                false,
            )
            .expect("decoder runtime");
            runtime
                .populate_cross_attention_cache(&encoder_output)
                .expect("populate cross cache");

            let logits = runtime
                .compute_step_logits(&prompt.token_ids)
                .expect("step logits");

            assert_eq!(logits.len(), metadata.vocab_size);
            assert!(logits.iter().all(|value| value.is_finite()));
        });
    }

    #[test]
    fn decoder_runtime_builds_batched_reusable_graph_shape() {
        with_forced_cpu_backend_for_test(|| {
            let (_runtime_path, preflight) = write_runtime_ready_preflight();
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
            let decoder_weights =
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights");
            let encoder_output = sample_encoder_output(metadata);
            let mut runtime = CohereDecoderGraphRuntime::new_with_n_seq(
                &decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                false,
                2,
            )
            .expect("batched decoder runtime");

            runtime
                .populate_cross_attention_cache_slot(0, &encoder_output)
                .expect("slot 0 cross cache should populate");
            runtime
                .populate_cross_attention_cache_slot(1, &encoder_output)
                .expect("slot 1 cross cache should populate");
            let slot_error = runtime
                .populate_cross_attention_cache_slot(2, &encoder_output)
                .expect_err("out-of-range slot must fail closed");
            assert!(matches!(
                slot_error,
                CohereDecoderGraphError::InvalidInput { .. }
            ));

            runtime
                .build_reusable_decode_graph()
                .expect("batched reusable graph should build");

            let reuse = runtime.reuse.as_ref().expect("reuse graph");
            assert_eq!(reuse.n_seq, 2);
            assert_eq!(reuse.max_positions, metadata.decoder_max_context);

            let logits = runtime
                .compute_reused_batched_step_logits(&[0, 1], &[0, 0], &[1, 1])
                .expect("batched reusable graph should compute");
            assert_eq!(logits.len(), metadata.vocab_size * 2);
            assert!(logits.iter().all(|value| value.is_finite()));
        });
    }

    #[test]
    fn decoder_runtime_batched_prefill_logits_match_serial_prefixes() {
        with_forced_cpu_backend_for_test(|| {
            let (_runtime_path, preflight) = write_runtime_ready_preflight();
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
            let decoder_weights =
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights");
            let tokenizer = super::super::tokenizer::CohereTranscribeTokenizer::from_gguf_metadata(
                &preflight.metadata,
            )
            .expect("tokenizer");
            let prompt = super::super::prompt::build_cohere_transcribe_decode_prompt(
                &tokenizer,
                metadata.decoder_start_token_id,
                Some("en"),
                &GgmlAsrExecutionOptions::default(),
            )
            .expect("prompt");
            let encoder_output_0 = sample_encoder_output(metadata);
            let mut encoder_output_1 = sample_encoder_output(metadata);
            for (index, value) in encoder_output_1.rows.iter_mut().enumerate() {
                *value = (*value + index as f32 * 0.0078125).cos();
            }

            let mut serial_runtime_0 = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output_0.frame_count,
                encoder_output_0.hidden_size,
                false,
            )
            .expect("serial runtime 0");
            serial_runtime_0
                .populate_cross_attention_cache(&encoder_output_0)
                .expect("serial cross cache 0");
            let serial_logits_0 = serial_runtime_0
                .compute_step_logits(&prompt.token_ids)
                .expect("serial logits 0");

            let mut serial_runtime_1 = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output_1.frame_count,
                encoder_output_1.hidden_size,
                false,
            )
            .expect("serial runtime 1");
            serial_runtime_1
                .populate_cross_attention_cache(&encoder_output_1)
                .expect("serial cross cache 1");
            let serial_logits_1 = serial_runtime_1
                .compute_step_logits(&prompt.token_ids)
                .expect("serial logits 1");

            let mut batched_runtime = CohereDecoderGraphRuntime::new_with_n_seq(
                &decoder_weights,
                metadata,
                encoder_output_0.frame_count,
                encoder_output_0.hidden_size,
                false,
                2,
            )
            .expect("batched runtime");
            batched_runtime
                .populate_cross_attention_cache_slot(0, &encoder_output_0)
                .expect("batched cross cache 0");
            batched_runtime
                .populate_cross_attention_cache_slot(1, &encoder_output_1)
                .expect("batched cross cache 1");
            let batched_logits = batched_runtime
                .compute_batched_prefill_logits(&prompt.token_ids)
                .expect("batched prefill logits");

            assert_eq!(batched_logits.len(), metadata.vocab_size * 2);
            assert_eq!(batched_runtime.cached_positions, prompt.token_ids.len());
            assert_logits_select_same_token(
                &batched_logits[0..metadata.vocab_size],
                &serial_logits_0,
                "slot 0",
            );
            assert_logits_select_same_token(
                &batched_logits[metadata.vocab_size..],
                &serial_logits_1,
                "slot 1",
            );
        });
    }

    #[test]
    fn decoder_runtime_reuses_persistent_self_kv_for_incremental_step() {
        with_forced_cpu_backend_for_test(|| {
            let (_runtime_path, preflight) = write_runtime_ready_preflight();
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
            let decoder_weights =
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights");
            let tokenizer = super::super::tokenizer::CohereTranscribeTokenizer::from_gguf_metadata(
                &preflight.metadata,
            )
            .expect("tokenizer");
            let prompt = super::super::prompt::build_cohere_transcribe_decode_prompt(
                &tokenizer,
                metadata.decoder_start_token_id,
                Some("en"),
                &GgmlAsrExecutionOptions::default(),
            )
            .expect("prompt");
            let encoder_output = sample_encoder_output(metadata);
            let mut runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                false,
            )
            .expect("decoder runtime");
            runtime
                .populate_cross_attention_cache(&encoder_output)
                .expect("populate cross cache");

            let prefill_logits = runtime
                .compute_step_logits(&prompt.token_ids)
                .expect("prefill logits");
            assert!(prefill_logits.iter().all(|value| value.is_finite()));
            assert_eq!(runtime.cached_positions, prompt.token_ids.len());

            let mut next_prefix = prompt.token_ids.clone();
            next_prefix.push(0);
            let incremental_logits = runtime
                .compute_step_logits(&next_prefix)
                .expect("incremental logits");

            assert_eq!(incremental_logits.len(), metadata.vocab_size);
            assert!(incremental_logits.iter().all(|value| value.is_finite()));
            assert_eq!(runtime.cached_positions, next_prefix.len());
        });
    }

    #[test]
    fn incremental_logits_match_full_prefix_recompute() {
        with_forced_cpu_backend_for_test(|| {
            let (_runtime_path, preflight) = write_runtime_ready_preflight();
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
            let decoder_weights =
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights");
            let tokenizer = super::super::tokenizer::CohereTranscribeTokenizer::from_gguf_metadata(
                &preflight.metadata,
            )
            .expect("tokenizer");
            let prompt = super::super::prompt::build_cohere_transcribe_decode_prompt(
                &tokenizer,
                metadata.decoder_start_token_id,
                Some("en"),
                &GgmlAsrExecutionOptions::default(),
            )
            .expect("prompt");
            let encoder_output = sample_encoder_output(metadata);
            let mut incremental_runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                false,
            )
            .expect("incremental runtime");
            incremental_runtime
                .populate_cross_attention_cache(&encoder_output)
                .expect("populate cross cache");
            incremental_runtime
                .compute_step_logits(&prompt.token_ids)
                .expect("prefill logits");
            let mut next_prefix = prompt.token_ids.clone();
            next_prefix.push(0);
            let incremental_logits = incremental_runtime
                .compute_step_logits(&next_prefix)
                .expect("incremental logits");

            let mut full_runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                false,
            )
            .expect("full runtime");
            full_runtime
                .populate_cross_attention_cache(&encoder_output)
                .expect("populate cross cache");
            let full_logits = full_runtime
                .compute_step_logits(&next_prefix)
                .expect("full-prefix logits");

            assert_eq!(incremental_logits.len(), full_logits.len());
            for (index, (incremental, full)) in
                incremental_logits.iter().zip(&full_logits).enumerate()
            {
                let diff = (incremental - full).abs();
                assert!(
                    diff < 1e-4,
                    "logit mismatch at vocab index {index}: incremental={incremental} full={full} diff={diff}"
                );
            }
        });
    }
}
