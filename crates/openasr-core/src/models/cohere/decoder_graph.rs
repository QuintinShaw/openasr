use std::sync::Arc;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedTensor, GgmlLoadedWeightBindingIdentity, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufRuntimeSourcePreflight,
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
use crate::api::backend::WordTimestamp;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
};
use crate::models::device_greedy_token::{
    DeviceGreedyStepOutputMode, first_max_argmax_reverse_indices,
    first_max_token_id_from_reversed_argmax,
};
use crate::models::seq2seq_decoder_state::Seq2SeqDecoderState;
use crate::models::seq2seq_dtw_alignment::{
    dtw_align_token_frames, speech_frame_bounds, token_text_carries_speech,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeStopReason,
};
use crate::models::seq2seq_word_timestamps::{
    Seq2SeqTokenSpan, Seq2SeqTokenTime, seq2seq_word_timestamps_from_generated_tokens,
    seq2seq_word_timestamps_from_token_spans, seq2seq_word_timestamps_from_token_times,
};
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
    /// How the shared driver ended this decode, carried through to the
    /// executor so a cut-short transcript is not returned as a complete one.
    pub stop_reason: Seq2SeqGreedyDecodeStopReason,
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
    control: &Arc<crate::TranscriptionControl>,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
    unstable_decode_text: Option<&crate::api::backend::UnstableDecodeTextObserver>,
) -> Result<CohereDecoderGraphDecodeOutput, CohereDecoderGraphError> {
    let decode_text_token_ids = |token_ids: &[u32]| {
        tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
            CohereTranscribeGreedyDecodeError::TokenizerDecodeFailed {
                reason: error.to_string(),
            }
        })
    };
    let max_generated_tokens =
        decoder_max_generated_tokens_with_env(prompt_tokens, metadata, encoder_output.frame_count)?;
    let planned_self_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
        prompt_tokens.len(),
        max_generated_tokens,
    )
    .map_err(|_| CohereDecoderGraphError::ShapeOverflow)?;
    decoder_runtime
        .decoder_state
        .self_attention
        .validate_exact_shape(
            crate::capacity::topology::StateKind::SelfAttentionKv,
            planned_self_positions,
        )
        .map_err(|error| CohereDecoderGraphError::InvalidInput {
            reason: error.to_string(),
        })?;
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: prompt_tokens.to_vec(),
        eot_token_id: eos_token_id,
        vocab_size: metadata.vocab_size,
        max_generated_tokens,
    };
    // Word-timestamp capture switches the last decoder layer to the unfused
    // f32 cross-attention for every incremental step so the per-token frame
    // row can be DTW-aligned. Off otherwise: the default fused path stays
    // byte-identical (and diarization-forced anchors stay post-hoc, like
    // whisper's `PostHocAnchors` mode, since `word_timestamps` is false then).
    decoder_runtime.collect_cross_attention = word_timestamps;
    let mut step_executor =
        CohereDecoderGraphStepExecutor::from_runtime(decoder_runtime, encoder_output)?;
    let decode = match run_cohere_transcribe_greedy_decode_loop(
        &config,
        tokenizer,
        phrase_bias,
        &mut step_executor,
        &decode_text_token_ids,
        control,
        decode_work_progress,
        unstable_decode_text,
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
            // Salvaging the prefix is not the same as completing the decode.
            stop_reason: Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
        },
        Err(error) => {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            });
        }
    };
    let text = decode.text.trim().to_string();
    // Align the captured per-token cross-attention rows to the audio timeline
    // with a monotone DTW pass, and use the resulting word spans in place of
    // the uniform post-hoc timestamps. If the DTW pass returns nothing (empty
    // / ragged input, all-punctuation decode), `cohere_plain_transcription...`
    // degrades to the center-of-mass path below.
    let dtw_words = (word_timestamps && !step_executor.token_alignments.is_empty())
        .then(|| {
            let alignments = std::mem::take(&mut step_executor.token_alignments);
            cohere_dtw_word_timestamps(
                &alignments,
                metadata,
                &decode.generated_probabilities,
                audio_duration_seconds,
                &decode_text_token_ids,
            )
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })
        })
        .transpose()?
        .filter(|words| !words.is_empty());
    let plain_words_override = dtw_words.as_ref();
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
                plain_words_override.map(|w| w.to_vec()),
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
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text,
                segments,
                longform: None,
                language: None,
                ..Default::default()
            }
        }
    } else {
        cohere_plain_transcription_from_generated_tokens(
            text,
            &decode.generated_tokens,
            &decode.generated_probabilities,
            plain_words_override.map(|w| w.to_vec()),
            word_timestamps,
            audio_duration_seconds,
            &decode_text_token_ids,
        )?
    };
    Ok(CohereDecoderGraphDecodeOutput {
        transcription,
        generated_tokens: decode.generated_tokens,
        stop_reason: decode.stop_reason,
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

/// Align per-token cross-attention frame rows to the audio timeline with a
/// monotone DTW pass and fold them into word timestamps, mirroring whisper's
/// no-timestamp DTW degrade. `token_alignments` pairs each generated (non-EOT)
/// token with its decoder's last-layer cross-attention frame row. Cohere
/// decodes `<|notimestamps|>` so there are no timestamp tokens: the DTW window
/// is bracketed on the content tokens' own attention peaks (leading/trailing
/// silence the model ignored is never bracketed by a peak, so it stays off the
/// timeline), and each content-token peak still owns its real audio span.
pub(crate) fn cohere_dtw_word_timestamps<E>(
    token_alignments: &[(u32, Vec<f32>)],
    metadata: CohereTranscribeExecutionMetadata,
    generated_probabilities: &[f32],
    duration: f32,
    decode_text: &dyn Fn(&[u32]) -> Result<String, E>,
) -> Result<Vec<WordTimestamp>, E> {
    let frame_count = token_alignments
        .first()
        .map(|alignment| alignment.1.len())
        .unwrap_or(0);
    if frame_count == 0 {
        return Ok(Vec::new());
    }
    // The three strided convs (k3,s2,p1) sub-sample the mel axis 8x, so one
    // encoder frame is 8 mel hops of audio; frames map to absolute wall-clock
    // time from clip start at that rate, not a fraction of `duration`.
    let hop = metadata.hop_length;
    let sample_rate = metadata.sample_rate_hz;
    if hop == 0 || sample_rate == 0 {
        return Ok(Vec::new());
    }
    let seconds_per_frame = 8.0 * hop as f32 / sample_rate as f32;
    if !seconds_per_frame.is_finite() || seconds_per_frame <= 0.0 {
        return Ok(Vec::new());
    }
    let duration = duration.max(0.0);
    let full_window = token_alignments
        .iter()
        .map(|alignment| alignment.1.clone())
        .collect::<Vec<Vec<f32>>>();
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        for (row_index, alignment) in token_alignments.iter().enumerate() {
            let text = decode_text(&[alignment.0]).unwrap_or_default();
            let (peak_frame, &peak_value) = alignment
                .1
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            let mut top_vec = alignment
                .1
                .iter()
                .enumerate()
                .map(|(frame, value)| (frame, *value))
                .collect::<Vec<_>>();
            top_vec.sort_by(|a, b| b.1.total_cmp(&a.1));
            top_vec.truncate(4);
            let top = top_vec
                .iter()
                .map(|(frame, value)| format!("{frame}@{value:.4}"))
                .collect::<Vec<_>>()
                .join(",");
            let peak_secs = peak_frame as f32 * seconds_per_frame;
            eprintln!(
                "cohere cross row {row_index} token={} text={text:?} peak@{peak_frame}({peak_secs:.2}s)={peak_value:.4} top=[{top}]",
                alignment.0
            );
        }
    }
    let is_content: Vec<bool> = token_alignments
        .iter()
        .map(|alignment| {
            decode_text(&[alignment.0]).is_ok_and(|text| token_text_carries_speech(&text))
        })
        .collect();
    // Cohere's last-layer cross-attention is diffuse and front-loaded on real
    // audio: several unrelated tokens share one early "priming" frame peak, so
    // the per-token peaks are not a clean monotone order and the DTW pass
    // over-spreads the first words (measured TempErr worse than the uniform
    // baseline on every clip available). Only trust the DTW word spans when the
    // content-token attention peaks are order-aligned; otherwise return empty
    // and let the caller keep the proven uniform post-hoc timestamps.
    //
    // `dtw_window` starts as the raw attention. When the raw peak order
    // zig-zags, one more chance is given after masking the dominant early
    // "sinks": an early frame that is the global max for a dominant share of
    // the rows is a shared diffuse-attention artifact, not evidence for where
    // any one token is spoken. Stripping it lets each masked row's
    // next-strongest frame (its real region) surface, which restores a
    // monotone peak order on clips where every non-artifact row already
    // pointed the right way. Rejected after both tests -> uniform fallback.
    let dtw_window: Option<Vec<Vec<f32>>> = if cross_attention_peaks_order_aligned(
        &full_window,
        &is_content,
    ) {
        None
    } else {
        match mask_dominant_early_sinks(&full_window, &is_content) {
            Some(window) if cross_attention_peaks_order_aligned(&window, &is_content) => {
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: order restored after sink strip, using DTW");
                }
                Some(window)
            }
            Some(window)
                if content_backward_fraction(&window, &is_content)
                    <= COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION
                    && band_duration_seconds(&full_window, seconds_per_frame)
                        >= COHERE_DTW_TOLERANT_MIN_BAND_SECONDS =>
            {
                // Not perfectly monotone, but the backward jumps are a
                // tiny minority of content pairs: the strip exposed a
                // mostly-clean left-to-right signal and the DTW path can be
                // trusted for it. This tier is strictly more permissive than
                // the strict re-test  above, applies only after the sink strip,
                // and is scoped to long windows (cohere's 30s long-form chunks)
                // where the pauses are actually measurable. Windows shorter than
                // the threshold follow the current strict path and fall
                // back to the provably-uniform baseline when they don't
                // reach the strict re-test, keeping short-clip behavior
                // unchanged.
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!(
                        "cohere cross gate: order restored after sink strip (tolerant), using DTW"
                    );
                }
                Some(window)
            }
            _ => {
                // The strict and tolerant DTW tests both rejected the window:
                // on a pause-heavy 30s chunk the per-word peak order zig-zags
                // too much for the DTW pass, and the monotone tiling the DTW
                // would emit (or the uniform fallback the caller would use)
                // would stretch a few seconds of speech across all 30s. On
                // LONG windows (where that stretch is large and measurable)
                // place each word at its own strongest attention frame instead:
                // the windows-are-diffuse-but-real case, where most tokens do
                // point the right way and only a few back-jump. Short windows
                // keep the provably-uniform baseline -- the stretch is small
                // there and the peak signal too noisy to be worth the risk.
                if band_duration_seconds(&full_window, seconds_per_frame)
                    >= COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS
                {
                    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                        eprintln!("cohere cross gate: peaks not aligned, using peak fallback");
                    }
                    return cohere_peak_fallback_word_timestamps(
                        &full_window,
                        &is_content,
                        token_alignments,
                        generated_probabilities,
                        seconds_per_frame,
                        decode_text,
                    );
                }
                if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
                    eprintln!("cohere cross gate: peaks not order-aligned, using uniform");
                }
                return Ok(Vec::new());
            }
        }
    };
    // The band is always derived from the unmasked raw attention: masking
    // zeroed frames would corrupt the earliest-peak bound, and the DTW cost
    // normalizes each row anyway, so the slice only needs the right range.
    let (band_start, band_end) = speech_frame_bounds(&full_window, &is_content).map_or_else(
        move || {
            (
                0usize,
                ((duration / seconds_per_frame).ceil() as usize).clamp(1, frame_count),
            )
        },
        |(start, end)| (start, end.clamp(start + 1, frame_count)),
    );
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!(
            "cohere cross band=({band_start},{band_end}) frames={frame_count} spf={seconds_per_frame}"
        );
    }
    let attention: Vec<Vec<f32>> = dtw_window
        .as_ref()
        .unwrap_or(&full_window)
        .iter()
        .map(|row| row[band_start.min(row.len())..band_end.min(row.len())].to_vec())
        .collect();
    let Some(spans) = dtw_align_token_frames(&attention) else {
        return Ok(Vec::new());
    };
    let probabilities_aligned = generated_probabilities.len() == token_alignments.len();
    let token_spans: Vec<Seq2SeqTokenSpan> = token_alignments
        .iter()
        .enumerate()
        .zip(spans.iter())
        .map(|((index, alignment), span)| Seq2SeqTokenSpan {
            token_id: alignment.0,
            frame_start: span.frame_start.saturating_add(band_start),
            frame_end: span.frame_end.saturating_add(band_start),
            probability: probabilities_aligned.then(|| generated_probabilities[index]),
        })
        .collect();
    seq2seq_word_timestamps_from_token_spans(
        &token_spans,
        0.0,
        duration,
        seconds_per_frame,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
    )
}

/// Order-gate fallback for long, pause-heavy windows: place each token at its
/// own strongest cross-attention frame and fold those into word timestamps.
///
/// Called only when the DTW order gate (strict and tolerant) has rejected the
/// window but the window is long enough that the stretch would hurt (see
/// [`COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS`]). Unlike the DTW tiling or the
/// uniform baseline -- both of which tile the window contiguously and cannot
/// express a gap -- each word lands where its attention is strongest, so the
/// midpoints between word centers fall where the attention falls and a short
/// utterance inside a long chunk stays short. The timeline is bounded at the
/// earliest frame (the clip start) and the latest *content-token* peak: a
/// trailing punctuation token's diffuse peak (attending into the ignored
/// padding) must not stretch the final word's end toward the far edge of the
/// chunk. Returns `Ok(Vec::new())` when no content-token peak exists, so the
/// caller keeps the uniform baseline rather than emitting a single degenerate
/// word.
fn cohere_peak_fallback_word_timestamps<E>(
    full_window: &[Vec<f32>],
    is_content: &[bool],
    token_alignments: &[(u32, Vec<f32>)],
    generated_probabilities: &[f32],
    seconds_per_frame: f32,
    decode_text: &dyn Fn(&[u32]) -> Result<String, E>,
) -> Result<Vec<WordTimestamp>, E> {
    let token_count = token_alignments.len();
    if token_count == 0 {
        return Ok(Vec::new());
    }
    let mut token_times = Vec::with_capacity(token_count);
    let mut last_content_peak_center: Option<f32> = None;
    for (index, alignment) in token_alignments.iter().enumerate() {
        let row = &full_window[index];
        let Some((peak_frame, &peak_value)) = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            return Ok(Vec::new());
        };
        let center_seconds = (peak_frame as f32) * seconds_per_frame;
        if is_content.get(index).copied().unwrap_or(false) && peak_value > 0.0 {
            last_content_peak_center = Some(match last_content_peak_center {
                Some(existing) => existing.max(center_seconds),
                None => center_seconds,
            });
        }
        token_times.push(Seq2SeqTokenTime {
            token_id: alignment.0,
            center_seconds,
            probability: generated_probabilities.get(index).copied(),
        });
    }
    let Some(segment_end) = last_content_peak_center else {
        return Ok(Vec::new());
    };
    seq2seq_word_timestamps_from_token_times(
        &token_times,
        0.0,
        segment_end,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        decode_text,
    )
}

fn cohere_plain_transcription_from_generated_tokens(
    text: String,
    generated_tokens: &[u32],
    generated_probabilities: &[f32],
    words_override: Option<Vec<WordTimestamp>>,
    word_timestamps: bool,
    audio_duration_seconds: f32,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, CohereTranscribeGreedyDecodeError>,
) -> Result<Transcription, CohereDecoderGraphError> {
    let words = if let Some(override_words) = words_override {
        override_words
    } else if word_timestamps {
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
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }]
    };
    Ok(Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text,
        segments,
        longform: None,
        language: None,
        ..Default::default()
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
    let speaker_label = speaker.clone();
    segments.push(Segment {
        start,
        end: end.max(start),
        text,
        speaker,
        speaker_label,
        speaker_person_id: None,
        speaker_snapshot_label: None,
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
    argmax_reverse_indices: Option<GgmlStaticTensor>,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
    metadata: CohereTranscribeExecutionMetadata,
    token_embedding: CohereDecoderWeightTensor,
    positional_embedding: CohereDecoderWeightTensor,
    emb_ln_weight: CohereDecoderWeightTensor,
    emb_ln_bias: CohereDecoderWeightTensor,
    out_ln_weight: CohereDecoderWeightTensor,
    out_ln_bias: CohereDecoderWeightTensor,
    output_head_weight: CohereDecoderWeightTensor,
    output_head_bias: CohereDecoderWeightTensor,
    layers: Vec<CohereDecoderLayerRuntime>,
    cross_layers: Vec<CohereDecoderCrossCacheLayerRuntime>,
    self_kv_layers: Vec<CohereDecoderSelfKvLayerRuntime>,
    decoder_state: Seq2SeqDecoderState,
    /// Stable planner-reserved column count of every cross-KV tensor. It is
    /// fixed for the runtime's lifetime and independent of the active
    /// utterance shape.
    cross_capacity_frames: usize,
    /// The active cross-frame-count [`Self::reuse`]'s persistent graph was
    /// last built for (0 if never built). `compute_reused_incremental_step_output`
    /// compares this against the current `cross_layers[0].frame_count` and
    /// rebuilds the (cheap, metadata-only) reusable graph whenever a
    /// differently-sized chunk swaps in, since `build_reusable_decode_graph`
    /// bakes the cross-attention view's frame count into the persistent
    /// graph's topology at build time.
    reuse_cross_frame_count: usize,
    cached_positions: usize,
    n_seq: usize,
    /// When true, every incremental step runs the unfused f32 cross-attention
    /// in the last decoder layer so the per-token frame-probability row can be
    /// captured for DTW word-timestamp alignment (see whisper's capture path).
    /// Word timestamps are the only consumer; off otherwise to keep the default
    /// fused-kernel decode path byte-identical.
    collect_cross_attention: bool,
    /// Head-averaged last-layer cross-attention frame row captured by the most
    /// recent incremental step, consumed (via `std::mem::take`) by the step
    /// executor to build the per-token alignment side channel.
    cross_attention_frame_probs: Option<Vec<f32>>,
    // Every graph-visible handle and persistent session above must drop before
    // its metadata/state arena, loaded roots, and backend runner.
    arena: GgmlStaticTensorArena,
    // Load-bearing for every `Loaded` weight handle above. In the unified
    // owner this upgrades the encoder's same-thread pack binding instead of
    // allocating a second decoder weight arena.
    loaded_weights: Option<GgmlLoadedWeightContext>,
    runner: GgmlCpuGraphRunner,
}

#[derive(Clone, Copy)]
enum CohereDecoderWeightTensor {
    Loaded(GgmlLoadedTensor),
    LoadedMatrixView(GgmlStaticTensor),
    Static(GgmlStaticTensor),
}

impl CohereDecoderWeightTensor {
    fn as_graph_tensor<'a>(self) -> GgmlCpuTensor<'a> {
        match self {
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
            Self::LoadedMatrixView(tensor) => tensor.as_graph_tensor(),
            Self::Static(tensor) => tensor.as_graph_tensor(),
        }
    }

    fn static_tensor(self) -> Option<GgmlStaticTensor> {
        match self {
            Self::Loaded(_) | Self::LoadedMatrixView(_) => None,
            Self::Static(tensor) => Some(tensor),
        }
    }
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
    attn_ln_weight: CohereDecoderWeightTensor,
    attn_ln_bias: CohereDecoderWeightTensor,
    attn_q_weight: CohereDecoderWeightTensor,
    attn_q_bias: CohereDecoderWeightTensor,
    attn_k_weight: CohereDecoderWeightTensor,
    attn_k_bias: CohereDecoderWeightTensor,
    attn_v_weight: CohereDecoderWeightTensor,
    attn_v_bias: CohereDecoderWeightTensor,
    attn_o_weight: CohereDecoderWeightTensor,
    attn_o_bias: CohereDecoderWeightTensor,
    cross_ln_weight: CohereDecoderWeightTensor,
    cross_ln_bias: CohereDecoderWeightTensor,
    cross_k_weight: CohereDecoderWeightTensor,
    cross_k_bias: CohereDecoderWeightTensor,
    cross_v_weight: CohereDecoderWeightTensor,
    cross_v_bias: CohereDecoderWeightTensor,
    cross_q_weight: CohereDecoderWeightTensor,
    cross_q_bias: CohereDecoderWeightTensor,
    cross_o_weight: CohereDecoderWeightTensor,
    cross_o_bias: CohereDecoderWeightTensor,
    ffn_ln_weight: CohereDecoderWeightTensor,
    ffn_ln_bias: CohereDecoderWeightTensor,
    ffn_up_weight: CohereDecoderWeightTensor,
    ffn_up_bias: CohereDecoderWeightTensor,
    ffn_down_weight: CohereDecoderWeightTensor,
    ffn_down_bias: CohereDecoderWeightTensor,
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
    /// The current utterance's planned encoder frame count, updated
    /// on every [`CohereDecoderGraphRuntime::populate_cross_attention_cache_slot`]
    /// call and always `<=` the stable resident column count.
    frame_count: usize,
    capacity_frames: usize,
    hidden_size: usize,
}

struct CohereDecoderGraphStepExecutor<'a> {
    runtime: &'a mut CohereDecoderGraphRuntime,
    /// When set (word-timestamp capture is on), every incremental step that
    /// emitted a frame-probability row is paired with the token that step
    /// generated, mirroring whisper's `token_alignments` side channel.
    token_alignments: Vec<(u32, Vec<f32>)>,
}

impl<'a> CohereDecoderGraphStepExecutor<'a> {
    fn from_runtime(
        runtime: &'a mut CohereDecoderGraphRuntime,
        encoder_output: &CohereTranscribeEncoderOutput,
    ) -> Result<Self, CohereDecoderGraphError> {
        runtime.populate_cross_attention_cache(encoder_output)?;
        Ok(Self {
            runtime,
            token_alignments: Vec::new(),
        })
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
        let output = self.runtime.compute_step_output(&prefix).map_err(|error| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            }
        })?;
        // Only incremental single-token steps produce a captured frame row;
        // the prompt step's cross-attention (token_count = prompt length) is
        // not per-token and is dropped.
        if let (Some(token_id), Some(frame_probs)) = (
            input.generated_tokens.last().copied(),
            std::mem::take(&mut self.runtime.cross_attention_frame_probs),
        ) {
            self.token_alignments.push((token_id, frame_probs));
        }
        Ok(output)
    }
}

impl CohereDecoderGraphRuntime {
    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: CohereTranscribeExecutionMetadata,
    ) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        for (size, label) in [
            (
                std::mem::size_of::<CohereDecoderLayerRuntime>(),
                "cohere decoder runtime layer handles",
            ),
            (
                std::mem::size_of::<CohereDecoderCrossCacheLayerRuntime>(),
                "cohere decoder cross-cache handles",
            ),
            (
                std::mem::size_of::<CohereDecoderSelfKvLayerRuntime>(),
                "cohere decoder self-KV handles",
            ),
        ] {
            bytes.add_usize(
                metadata
                    .decoder_layers
                    .checked_mul(size)
                    .ok_or_else(|| format!("{label} quote overflowed"))?,
                label,
            )?;
        }
        Ok(bytes.finish())
    }

    pub(crate) fn quoted_construction_peak_system_memory_bytes(
        metadata: CohereTranscribeExecutionMetadata,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<u64, String> {
        let retained = Self::quoted_retained_system_memory_bytes(metadata)?;
        retained
            .checked_add(device_top1_construction_transient_bytes(
                metadata.vocab_size,
                output_mode,
            )?)
            .ok_or_else(|| "cohere decoder construction peak overflowed".to_string())
    }

    pub(crate) fn new_from_preflight(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        cross_hidden_size: usize,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
        preflight: &GgufRuntimeSourcePreflight,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Self, CohereDecoderGraphError> {
        Self::new_with_n_seq_impl(
            decoder_weights,
            metadata,
            decoder_state,
            cross_hidden_size,
            backend,
            prefer_cpu_backend,
            1,
            Some(preflight),
            greedy_step_output_mode,
        )
    }

    pub(crate) fn new(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        cross_hidden_size: usize,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
    ) -> Result<Self, CohereDecoderGraphError> {
        Self::new_with_n_seq(
            decoder_weights,
            metadata,
            decoder_state,
            cross_hidden_size,
            backend,
            prefer_cpu_backend,
            1,
        )
    }

    pub(crate) fn new_with_n_seq(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        cross_hidden_size: usize,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
        n_seq: usize,
    ) -> Result<Self, CohereDecoderGraphError> {
        Self::new_with_n_seq_impl(
            decoder_weights,
            metadata,
            decoder_state,
            cross_hidden_size,
            backend,
            prefer_cpu_backend,
            n_seq,
            None,
            DeviceGreedyStepOutputMode::FullLogits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_n_seq_impl(
        decoder_weights: &CohereTranscribeDecoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        cross_hidden_size: usize,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
        n_seq: usize,
        runtime_preflight: Option<&GgufRuntimeSourcePreflight>,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Self, CohereDecoderGraphError> {
        decoder_state
            .validate()
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })?;
        if n_seq == 0 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "cohere decoder n_seq must be positive".to_string(),
            });
        }
        if n_seq != 1 && greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "cohere device top-1 requires n_seq=1".to_string(),
            });
        }
        validate_decoder_runtime_shapes(decoder_weights, metadata)?;
        validate_encoder_cross_dimensions(
            cross_hidden_size,
            decoder_state.cross_attention.logical_positions,
            metadata,
            decoder_weights.layers.len(),
        )?;
        decoder_state
            .self_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::SelfAttentionKv,
                metadata.decoder_max_context,
            )
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })?;
        let cross_alloc_frames = decoder_state.cross_attention.resident_positions;

        let mut config = cohere_decoder_graph_config(backend, prefer_cpu_backend);
        config.graph_size = config.graph_size.max(COHERE_DECODER_GRAPH_SIZE_FLOOR);
        config.context_bytes =
            config
                .context_bytes
                .max(GgmlCpuGraphConfig::metadata_context_bytes(
                    config.graph_size,
                ));
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights = runtime_preflight
            .map(|preflight| {
                runner
                    .load_gguf_weight_context_from_preflight(preflight)
                    .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                        step: "load_gguf_weight_context",
                        source,
                    })
            })
            .transpose()?;
        let arena_state = build_cohere_decoder_arena_state(
            &runner,
            loaded_weights.as_ref(),
            decoder_weights,
            metadata,
            cross_hidden_size,
            decoder_state.self_attention.resident_positions,
            cross_alloc_frames,
            n_seq,
            greedy_step_output_mode,
        )?;

        Ok(Self {
            reuse: None,
            argmax_reverse_indices: arena_state.argmax_reverse_indices,
            greedy_step_output_mode,
            loaded_weights,
            metadata,
            runner,
            arena: arena_state.arena,
            token_embedding: arena_state.token_embedding,
            positional_embedding: arena_state.positional_embedding,
            emb_ln_weight: arena_state.emb_ln_weight,
            emb_ln_bias: arena_state.emb_ln_bias,
            out_ln_weight: arena_state.out_ln_weight,
            out_ln_bias: arena_state.out_ln_bias,
            output_head_weight: arena_state.output_head_weight,
            output_head_bias: arena_state.output_head_bias,
            layers: arena_state.layers,
            cross_layers: arena_state.cross_layers,
            self_kv_layers: arena_state.self_kv_layers,
            decoder_state,
            cross_capacity_frames: cross_alloc_frames,
            reuse_cross_frame_count: 0,
            cached_positions: 0,
            n_seq,
            collect_cross_attention: false,
            cross_attention_frame_probs: None,
        })
    }

    pub(crate) fn graph_lane(&self) -> (GgmlCpuGraphBackend, bool) {
        (self.runner.backend_kind(), self.runner.uses_scheduler())
    }

    pub(crate) fn loaded_weight_binding_identity(&self) -> Option<GgmlLoadedWeightBindingIdentity> {
        self.loaded_weights
            .as_ref()
            .map(|loaded| self.runner.loaded_weight_binding_identity(loaded))
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.layers, "cohere decoder runtime layer handles")?;
        bytes.add_vec(&self.cross_layers, "cohere decoder cross-cache handles")?;
        bytes.add_vec(&self.self_kv_layers, "cohere decoder self-KV handles")?;
        Ok(bytes.finish())
    }

    pub(crate) fn construction_peak_system_memory_bytes(&self) -> Result<u64, String> {
        self.retained_system_memory_bytes()?
            .checked_add(device_top1_construction_transient_bytes(
                self.metadata.vocab_size,
                self.greedy_step_output_mode,
            )?)
            .ok_or_else(|| "cohere decoder construction peak overflowed".to_string())
    }

    pub(crate) fn activate_decoder_state(
        &mut self,
        decoder_state: Seq2SeqDecoderState,
    ) -> Result<(), CohereDecoderGraphError> {
        decoder_state
            .validate()
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })?;
        if decoder_state.self_attention.resident_positions
            != self.decoder_state.self_attention.resident_positions
            || decoder_state.cross_attention.resident_positions
                != self.decoder_state.cross_attention.resident_positions
        {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "cohere cached decoder resident capacity mismatch: cached self/cross={}/{}, requested={}/{}",
                    self.decoder_state.self_attention.resident_positions,
                    self.decoder_state.cross_attention.resident_positions,
                    decoder_state.self_attention.resident_positions,
                    decoder_state.cross_attention.resident_positions,
                ),
            });
        }
        decoder_state
            .self_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::SelfAttentionKv,
                self.metadata.decoder_max_context,
            )
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })?;
        if decoder_state.self_attention.logical_positions
            != self.decoder_state.self_attention.logical_positions
            || decoder_state.cross_attention.logical_positions
                != self.decoder_state.cross_attention.logical_positions
        {
            self.reuse = None;
            self.reuse_cross_frame_count = 0;
        }
        self.decoder_state = decoder_state;
        Ok(())
    }
}

/// Bundle of everything [`build_cohere_decoder_arena_state`] allocates
/// directly in a fresh arena: every decoder weight tensor plus the
/// cross-KV/self-KV caches. Factored out of runtime construction to keep the
/// tensor declaration and upload transaction cohesive.
struct CohereDecoderArenaState {
    arena: GgmlStaticTensorArena,
    argmax_reverse_indices: Option<GgmlStaticTensor>,
    token_embedding: CohereDecoderWeightTensor,
    positional_embedding: CohereDecoderWeightTensor,
    emb_ln_weight: CohereDecoderWeightTensor,
    emb_ln_bias: CohereDecoderWeightTensor,
    out_ln_weight: CohereDecoderWeightTensor,
    out_ln_bias: CohereDecoderWeightTensor,
    output_head_weight: CohereDecoderWeightTensor,
    output_head_bias: CohereDecoderWeightTensor,
    layers: Vec<CohereDecoderLayerRuntime>,
    cross_layers: Vec<CohereDecoderCrossCacheLayerRuntime>,
    self_kv_layers: Vec<CohereDecoderSelfKvLayerRuntime>,
}

#[allow(clippy::too_many_arguments)]
fn build_cohere_decoder_arena_state(
    runner: &GgmlCpuGraphRunner,
    loaded_weights: Option<&GgmlLoadedWeightContext>,
    decoder_weights: &CohereTranscribeDecoderWeights,
    metadata: CohereTranscribeExecutionMetadata,
    cross_hidden_size: usize,
    self_kv_alloc_positions: usize,
    cross_alloc_frames: usize,
    n_seq: usize,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
) -> Result<CohereDecoderArenaState, CohereDecoderGraphError> {
    let arena_tensor_count = decoder_weights
        .layers
        .len()
        .checked_mul(30)
        .and_then(|count| count.checked_add(8))
        .and_then(|count| {
            count.checked_add(
                (greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1) as usize,
            )
        })
        .ok_or_else(|| CohereDecoderGraphError::InvalidInput {
            reason: "cohere decoder static tensor count overflows usize".to_string(),
        })?;

    let mut arena = runner
        .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
            arena_tensor_count,
        ))
        .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
            step: "static_tensor_arena",
            source,
        })?;
    let argmax_reverse_indices =
        if greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            Some(
                arena
                    .new_tensor_1d_i32(metadata.vocab_size, "cohere_dec_argmax_reverse_indices")
                    .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                        step: "argmax_reverse_indices",
                        source,
                    })?,
            )
        } else {
            None
        };

    let token_embedding = decoder_embedding_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.token_embedding,
        "dec_emb",
    )?;
    let positional_embedding = decoder_embedding_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.positional_embedding,
        "dec_pos",
    )?;
    let emb_ln_weight = decoder_vector_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.emb_ln_weight,
        "dec_emb_ln_w",
    )?;
    let emb_ln_bias = decoder_vector_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.emb_ln_bias,
        "dec_emb_ln_b",
    )?;
    let out_ln_weight = decoder_vector_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.out_ln_weight,
        "dec_out_ln_w",
    )?;
    let out_ln_bias = decoder_vector_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.out_ln_bias,
        "dec_out_ln_b",
    )?;
    let output_head_weight = decoder_projection_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.output_head_weight,
        "dec_head",
    )?;
    let output_head_bias = decoder_vector_tensor(
        loaded_weights,
        &arena,
        &decoder_weights.output_head_bias,
        "dec_head_b",
    )?;

    let mut layers = Vec::with_capacity(decoder_weights.layers.len());
    let mut cross_layers = Vec::with_capacity(decoder_weights.layers.len());
    let mut self_kv_layers = Vec::with_capacity(decoder_weights.layers.len());
    for layer in &decoder_weights.layers {
        let runtime = CohereDecoderLayerRuntime {
            attn_ln_weight: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_ln_weight,
                "dec_attn_ln_w",
            )?,
            attn_ln_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_ln_bias,
                "dec_attn_ln_b",
            )?,
            attn_q_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.attn_q_weight,
                "dec_attn_q_w",
            )?,
            attn_q_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_q_bias,
                "dec_attn_q_b",
            )?,
            attn_k_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.attn_k_weight,
                "dec_attn_k_w",
            )?,
            attn_k_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_k_bias,
                "dec_attn_k_b",
            )?,
            attn_v_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.attn_v_weight,
                "dec_attn_v_w",
            )?,
            attn_v_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_v_bias,
                "dec_attn_v_b",
            )?,
            attn_o_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.attn_o_weight,
                "dec_attn_o_w",
            )?,
            attn_o_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.attn_o_bias,
                "dec_attn_o_b",
            )?,
            cross_ln_weight: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_ln_weight,
                "dec_cross_ln_w",
            )?,
            cross_ln_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_ln_bias,
                "dec_cross_ln_b",
            )?,
            cross_k_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.cross_k_weight,
                "dec_cross_k_w",
            )?,
            cross_k_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_k_bias,
                "dec_cross_k_b",
            )?,
            cross_v_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.cross_v_weight,
                "dec_cross_v_w",
            )?,
            cross_v_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_v_bias,
                "dec_cross_v_b",
            )?,
            cross_q_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.cross_q_weight,
                "dec_cross_q_w",
            )?,
            cross_q_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_q_bias,
                "dec_cross_q_b",
            )?,
            cross_o_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.cross_o_weight,
                "dec_cross_o_w",
            )?,
            cross_o_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.cross_o_bias,
                "dec_cross_o_b",
            )?,
            ffn_ln_weight: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_ln_weight,
                "dec_ffn_ln_w",
            )?,
            ffn_ln_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_ln_bias,
                "dec_ffn_ln_b",
            )?,
            ffn_up_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_up_weight,
                "dec_ffn_up_w",
            )?,
            ffn_up_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_up_bias,
                "dec_ffn_up_b",
            )?,
            ffn_down_weight: decoder_projection_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_down_weight,
                "dec_ffn_down_w",
            )?,
            ffn_down_bias: decoder_vector_tensor(
                loaded_weights,
                &arena,
                &layer.ffn_down_bias,
                "dec_ffn_down_b",
            )?,
        };
        layers.push(runtime);
        cross_layers.push(CohereDecoderCrossCacheLayerRuntime {
            key: new_persistent_cross_cache_tensor_in_arena(
                &arena,
                cross_hidden_size,
                cross_alloc_frames,
                n_seq,
                "dec_cross_k_cache",
            )?,
            value: new_persistent_cross_cache_tensor_in_arena(
                &arena,
                cross_hidden_size,
                cross_alloc_frames,
                n_seq,
                "dec_cross_v_cache",
            )?,
            // Placeholder until the first `populate_cross_attention_cache[_slot]`
            // call sets this to the actual current-utterance frame count;
            // never read before that (mirrors firered-aed's `0`-init, but
            // this arm keeps parity with the pre-existing invariant that
            // `frame_count` is always `<=` the tensor's real column count).
            frame_count: cross_alloc_frames,
            capacity_frames: cross_alloc_frames,
            hidden_size: cross_hidden_size,
        });
        self_kv_layers.push(CohereDecoderSelfKvLayerRuntime {
            key: new_persistent_self_kv_tensor_in_arena(
                &arena,
                metadata.decoder_head_dim,
                self_kv_alloc_positions,
                metadata.decoder_heads,
                n_seq,
                "dec_self_k_cache",
            )?,
            value: new_persistent_self_kv_tensor_in_arena(
                &arena,
                metadata.decoder_head_dim,
                self_kv_alloc_positions,
                metadata.decoder_heads,
                n_seq,
                "dec_self_v_cache",
            )?,
            max_positions: self_kv_alloc_positions,
        });
    }

    // The legacy Static path allocates this arena as a side effect of the
    // first weight upload below. A loaded-weight runtime skips every such
    // upload, but its persistent cross/self-KV roots still live in this
    // arena and must be bound before graph code creates views of them.
    if loaded_weights.is_some() {
        arena.allocate_backend_buffer().map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "allocate_decoder_state_arena",
                source,
            }
        })?;
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
    for (layer_idx, (runtime, layer)) in layers.iter().zip(&decoder_weights.layers).enumerate() {
        upload_decoder_layer_to_arena(&mut arena, runtime, layer, layer_idx)?;
    }
    if let Some(reverse_indices) = argmax_reverse_indices {
        arena
            .set_i32_slice(
                reverse_indices,
                &first_max_argmax_reverse_indices(metadata.vocab_size).map_err(|source| {
                    CohereDecoderGraphError::GraphBuildFailed {
                        step: "argmax_reverse_indices",
                        source,
                    }
                })?,
                "cohere_dec_argmax_reverse_indices",
            )
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "argmax_reverse_indices",
                source,
            })?;
    }

    Ok(CohereDecoderArenaState {
        arena,
        argmax_reverse_indices,
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
    })
}

impl CohereDecoderGraphRuntime {
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
        self.decoder_state
            .cross_attention
            .validate_exact_shape(
                crate::capacity::topology::StateKind::CrossAttentionKv,
                encoder_output.frame_count,
            )
            .map_err(|error| CohereDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            })?;
        if encoder_output.frame_count > self.cross_capacity_frames {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "cohere logical cross shape {} exceeds resident capacity {}",
                    encoder_output.frame_count, self.cross_capacity_frames
                ),
            });
        }
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
                layer.cross_k_weight.as_graph_tensor(),
                layer.cross_k_bias.as_graph_tensor(),
                "decoder_cross_cache_k",
            )?;
            let key_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross_runtime.key),
                encoder_output.hidden_size,
                encoder_output.frame_count,
                cross_runtime.capacity_frames,
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
                layer.cross_v_weight.as_graph_tensor(),
                layer.cross_v_bias.as_graph_tensor(),
                "decoder_cross_cache_v",
            )?;
            let value_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross_runtime.value),
                encoder_output.hidden_size,
                encoder_output.frame_count,
                cross_runtime.capacity_frames,
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
            })?;
        // Record the ACTUAL frame count this call just populated so the
        // cross-attention view in `compute_step_logits` / the persistent
        // reuse graph reads back exactly that many columns, not the (possibly
        // larger) allocated capacity. Every layer shares the same value by
        // construction (one `encoder_output` per call).
        for cross_runtime in &mut self.cross_layers {
            cross_runtime.frame_count = encoder_output.frame_count;
        }
        Ok(())
    }

    pub(super) fn compute_step_logits(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, CohereDecoderGraphError> {
        Ok(self
            .compute_step_output_with_mode(decoder_tokens, DeviceGreedyStepOutputMode::FullLogits)?
            .logits)
    }

    fn compute_step_output(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, CohereDecoderGraphError> {
        self.compute_step_output_with_mode(decoder_tokens, self.greedy_step_output_mode)
    }

    fn compute_step_output_with_mode(
        &mut self,
        decoder_tokens: &[u32],
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, CohereDecoderGraphError> {
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
        let logical_max_positions = self.decoder_state.self_attention.logical_positions;
        if total_prefix_tokens > logical_max_positions {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder token_count {} exceeds max context {}",
                    total_prefix_tokens, logical_max_positions
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
        if position_offset > 0
            && token_count == 1
            && !self.collect_cross_attention
            && self.supports_reusable_decode_graph()
        {
            // Word-timestamp capture needs the unfused f32 cross-attention in
            // the last layer; the reusable persistent graph bakes in the fused
            // flash path, so when capture is on we fall through to the
            // full-graph rebuild below.
            return self.compute_reused_incremental_step_output(
                decode_tokens[0],
                position_offset,
                output_mode,
            );
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
            .get_rows(self.token_embedding.as_graph_tensor(), token_ids_tensor)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(token)",
                source,
            })?;
        let position_state = graph
            .get_rows(
                self.positional_embedding.as_graph_tensor(),
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
            self.emb_ln_weight.as_graph_tensor(),
            self.emb_ln_bias.as_graph_tensor(),
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

        let (mut state, cross_frame_probs) = compose_seq2seq_decoder_layer_stack(
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
            self.collect_cross_attention && token_count == 1,
        )?;
        let cross_frame_probs = cross_frame_probs.filter(|_| self.collect_cross_attention);

        state = apply_affine_norm(
            &graph,
            state,
            self.out_ln_weight.as_graph_tensor(),
            self.out_ln_bias.as_graph_tensor(),
            "decoder_out_norm",
        )?;
        if position_offset == 0
            && let Some(debug) = prompt_debug_tensors.as_mut()
        {
            debug.final_state = state;
        }
        let last_state = view_last_token_state(&graph, state, hidden, token_count)?;
        let logits = graph
            .mul_mat(self.output_head_weight.as_graph_tensor(), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(output_head)",
                source,
            })?;
        let logits = graph
            .add(logits, self.output_head_bias.as_graph_tensor())
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_add(output_head_bias)",
                source,
            })?;
        let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            let reverse_indices = self.argmax_reverse_indices.ok_or_else(|| {
                CohereDecoderGraphError::InvalidInput {
                    reason: "cohere device top-1 reverse indices are unavailable".to_string(),
                }
            })?;
            Some(
                graph
                    .top1_argmax_first_max_reversed(
                        logits,
                        self.arena.graph_tensor(reverse_indices),
                    )
                    .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                        step: "ggml_argmax(top1)",
                        source,
                    })?,
            )
        } else {
            None
        };
        let output_root = top1.unwrap_or(logits);
        graph.set_output(output_root).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(logits)",
                source,
            }
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
        // the sibling firered/moonshine decoders. The captured cross-attention
        // tensor is a second root so its backend storage is not recycled.
        let mut prepare_roots = vec![output_root];
        if let Some(cross_frame_probs) = cross_frame_probs {
            prepare_roots.push(cross_frame_probs);
        }
        {
            let roots: &[_] = &prepare_roots;
            graph.prepare_outputs_for_upload(roots).map_err(|source| {
                CohereDecoderGraphError::GraphBuildFailed {
                    step: "ggml_prepare_outputs(logits)",
                    source,
                }
            })?;
        }
        for upload in uploads {
            upload.apply(&mut graph)?;
        }
        let output = if debug_prompt_step {
            if top1.is_some() {
                return Err(CohereDecoderGraphError::InvalidInput {
                    reason: "cohere debug-token output requires full logits".to_string(),
                });
            }
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
            Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: outputs.into_iter().next().ok_or(
                    CohereDecoderGraphError::InvalidInput {
                        reason: "missing logits output".to_string(),
                    },
                )?,
                greedy_token_hint: None,
            }
        } else if let Some(cross_frame_probs) = cross_frame_probs {
            // Word-timestamp capture: word timestamps always force FullLogits
            // (top1 is None), so read the logits row and the last layer's
            // cross-attention frame row back in one compute pass.
            let frame_count = self
                .cross_layers
                .first()
                .map(|layer| layer.frame_count)
                .unwrap_or(0);
            let heads = self.metadata.decoder_heads;
            let expected_probs = frame_count
                .checked_mul(heads)
                .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
            let outputs = graph
                .compute_outputs_f32(&[
                    (logits, self.metadata.vocab_size),
                    (cross_frame_probs, expected_probs),
                ])
                .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?;
            if outputs.len() != 2 {
                return Err(CohereDecoderGraphError::InvalidInput {
                    reason: "cross-attention capture expected two outputs".to_string(),
                });
            }
            let (logits_row, frame_row) = {
                let mut iter = outputs.into_iter();
                (
                    iter.next().expect("logits output present"),
                    iter.next().expect("cross-attention output present"),
                )
            };
            self.cross_attention_frame_probs = Some(average_cross_attention_frame_row(
                &frame_row,
                frame_count,
                heads,
            )?);
            Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: logits_row,
                greedy_token_hint: None,
            }
        } else {
            compute_greedy_step_output_for_graph(
                &mut graph,
                logits,
                top1,
                self.metadata.vocab_size,
            )?
        };
        emit_cohere_debug_step_logits_if_enabled(
            decode_tokens,
            position_offset,
            total_token_count,
            &output.logits,
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

    fn compute_reused_incremental_step_output(
        &mut self,
        token_id: u32,
        position: usize,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, CohereDecoderGraphError> {
        let logical_max_positions = self.decoder_state.self_attention.logical_positions;
        if position >= logical_max_positions {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder position {position} exceeds max context {}",
                    logical_max_positions
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
        // A cross-frame-count change also forces a rebuild: `build_reusable_decode_graph`
        // bakes the current `cross_layers[0].frame_count` into the persistent
        // graph's cross-attention view topology, so a same-shaped-otherwise
        // graph built for a different (earlier) chunk's frame count would
        // silently attend over the wrong span for this one.
        let active_cross_frame_count = self
            .cross_layers
            .first()
            .map(|layer| layer.frame_count)
            .unwrap_or(0);
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.is_poisoned()
                    || reuse.max_positions != logical_max_positions
                    || reuse.n_seq != 1
                    || self.reuse_cross_frame_count != active_cross_frame_count
                    || reuse.top1.is_some()
                        != (output_mode == DeviceGreedyStepOutputMode::DeviceTop1)
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph(output_mode)?;
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
        let top1 = reuse.top1;
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

        let output =
            compute_greedy_step_output_for_graph(graph, logits, top1, self.metadata.vocab_size)?;
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
        let logical_max_positions = self.decoder_state.self_attention.logical_positions;
        if positions
            .iter()
            .any(|&position| position >= logical_max_positions)
        {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere decoder position exceeds max context {}",
                    logical_max_positions
                ),
            });
        }
        if total_tokens_by_sequence
            .iter()
            .any(|&total_tokens| total_tokens == 0 || total_tokens > logical_max_positions)
        {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere total token count must be in 1..={}",
                    logical_max_positions
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
        let active_cross_frame_count = self
            .cross_layers
            .first()
            .map(|layer| layer.frame_count)
            .unwrap_or(0);
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.is_poisoned()
                    || reuse.max_positions != logical_max_positions
                    || reuse.n_seq != self.n_seq
                    || self.reuse_cross_frame_count != active_cross_frame_count
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph(DeviceGreedyStepOutputMode::FullLogits)?;
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
        let logical_max_positions = self.decoder_state.self_attention.logical_positions;
        if token_count > logical_max_positions {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched cohere prefill token_count {} exceeds max context {}",
                    token_count, logical_max_positions
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
            .get_rows(self.token_embedding.as_graph_tensor(), token_ids_tensor)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(prefill_token)",
                source,
            })?;
        let position_state = graph
            .get_rows(self.positional_embedding.as_graph_tensor(), position_tensor)
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
            self.emb_ln_weight.as_graph_tensor(),
            self.emb_ln_bias.as_graph_tensor(),
            "prefill_decoder_emb_norm",
        )?;
        let mut uploads = Vec::new();
        let mut prompt_debug_tensors = None;
        let (mut state, batched_probs) = compose_seq2seq_decoder_layer_stack(
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
            false,
        )?;
        debug_assert!(batched_probs.is_none());
        debug_assert!(uploads.is_empty());

        state = apply_affine_norm(
            &graph,
            state,
            self.out_ln_weight.as_graph_tensor(),
            self.out_ln_bias.as_graph_tensor(),
            "prefill_decoder_out_norm",
        )?;
        let last_state =
            view_batched_last_token_state(&graph, state, hidden, token_count, self.n_seq)?;
        let logits = graph
            .mul_mat(self.output_head_weight.as_graph_tensor(), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(prefill_output_head)",
                source,
            })?;
        let bias = graph
            .repeat(self.output_head_bias.as_graph_tensor(), logits)
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

    fn build_reusable_decode_graph(
        &mut self,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<(), CohereDecoderGraphError> {
        if self.n_seq != 1 && output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            return Err(CohereDecoderGraphError::InvalidInput {
                reason: "cohere device top-1 requires n_seq=1".to_string(),
            });
        }
        let hidden = self.metadata.decoder_d_model;
        let max_context = self.decoder_state.self_attention.logical_positions;
        let n_seq = self.n_seq;
        let active_cross_frame_count = self
            .cross_layers
            .first()
            .map(|layer| layer.frame_count)
            .unwrap_or(0);
        let mut session = self
            .runner
            .start_capacity_sized_persistent_graph_session()
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
            .get_rows(self.token_embedding.as_graph_tensor(), token_id)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_get_rows(reuse_token)",
                source,
            })?;
        let position_state = graph
            .get_rows(self.positional_embedding.as_graph_tensor(), position)
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
            self.emb_ln_weight.as_graph_tensor(),
            self.emb_ln_bias.as_graph_tensor(),
            "reuse_decoder_emb_norm",
        )?;
        let mut uploads = Vec::new();
        let mut prompt_debug_tensors = None;
        let (mut state, reuse_probs) = compose_seq2seq_decoder_layer_stack(
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
            false,
        )?;
        debug_assert!(reuse_probs.is_none());
        debug_assert!(uploads.is_empty());

        state = apply_affine_norm(
            graph,
            state,
            self.out_ln_weight.as_graph_tensor(),
            self.out_ln_bias.as_graph_tensor(),
            "reuse_decoder_out_norm",
        )?;
        let last_state = if n_seq == 1 {
            view_last_token_state(graph, state, hidden, 1)?
        } else {
            state
        };
        let logits = graph
            .mul_mat(self.output_head_weight.as_graph_tensor(), last_state)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_mul_mat(reuse_output_head)",
                source,
            })?;
        let output_head_bias = self.output_head_bias.as_graph_tensor();
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
        let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            let reverse_indices = self.argmax_reverse_indices.ok_or_else(|| {
                CohereDecoderGraphError::InvalidInput {
                    reason: "cohere device top-1 reverse indices are unavailable".to_string(),
                }
            })?;
            Some(
                graph
                    .top1_argmax_first_max_reversed(
                        logits,
                        self.arena.graph_tensor(reverse_indices),
                    )
                    .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                        step: "ggml_argmax(reuse_top1)",
                        source,
                    })?,
            )
        } else {
            None
        };
        let output_root = top1.unwrap_or(logits);
        graph.set_output(output_root).map_err(|source| {
            CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_set_output(reuse_logits)",
                source,
            }
        })?;
        graph
            .prepare_outputs_for_upload(&[output_root])
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: "ggml_prepare_outputs(reuse_logits)",
                source,
            })?;

        self.reuse = Some(
            Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena_and_optional_top1(
                session,
                max_context,
                n_seq,
                token_id,
                row_index,
                position,
                attention_mask,
                logits,
                top1,
            ),
        );
        self.reuse_cross_frame_count = active_cross_frame_count;
        Ok(())
    }
}

fn device_top1_construction_transient_bytes(
    vocab_size: usize,
    output_mode: DeviceGreedyStepOutputMode,
) -> Result<u64, String> {
    if output_mode == DeviceGreedyStepOutputMode::FullLogits {
        return Ok(0);
    }
    let bytes = vocab_size
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| "cohere device top-1 construction bytes overflowed".to_string())?;
    u64::try_from(bytes)
        .map_err(|_| "cohere device top-1 construction bytes exceed u64".to_string())
}

fn map_reversed_top1_token(
    reversed_token_id: i32,
    vocab_size: usize,
) -> Result<u32, CohereDecoderGraphError> {
    let token_id = first_max_token_id_from_reversed_argmax(reversed_token_id, vocab_size).map_err(
        |error| CohereDecoderGraphError::GraphExecutionFailed {
            reason: error.to_string(),
        },
    )?;
    u32::try_from(token_id).map_err(|_| CohereDecoderGraphError::GraphExecutionFailed {
        reason: "cohere device top-1 token id does not fit u32".to_string(),
    })
}

/// Head-average a captured `[frame_count, token_count, heads]` f32
/// cross-attention row into one length-`frame_count` probability vector.
/// Layout mirrors whisper's capture: element `(frame, token, head)` lives at
/// `frame + frame_count * (token + token_count * head)`, with the token axis
/// collapsing (`token_count == 1`) for the incremental steps.
fn average_cross_attention_frame_row(
    attention: &[f32],
    frame_count: usize,
    heads: usize,
) -> Result<Vec<f32>, CohereDecoderGraphError> {
    if frame_count == 0 || heads == 0 {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: "cross-attention capture produced an empty dimension".to_string(),
        });
    }
    let expected = frame_count
        .checked_mul(heads)
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    if attention.len() != expected {
        return Err(CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "cross-attention capture length mismatch: got {}, expected {expected}",
                attention.len()
            ),
        });
    }
    let inv_heads = 1.0 / heads as f32;
    Ok((0..frame_count)
        .map(|frame| {
            (0..heads)
                .map(|head| attention[frame + frame_count * head])
                .sum::<f32>()
                * inv_heads
        })
        .collect())
}

/// Whether the per-token cross-attention peaks, read in decode order, form a
/// monotone (non-decreasing) frame sequence over the content tokens, allowing
/// ties and single-frame jitter.
///
/// A clean alignment signal has each content token's strongest frame at or
/// after the previous content token's strongest frame (the speech is left to
/// right, so the attention follows it). Cohere's last-layer cross-attention is
/// instead diffuse and front-loaded: several unrelated tokens share one early
/// "priming" frame as their global max, so the peak sequence zig-zags and the
/// DTW pass over-spreads the first words past where they are spoken (measured
/// worse than the uniform post-hoc baseline on every available clip). Only a
/// non-zig-zag peak sequence is a trustworthy DTW input; anything else should
/// fall back to the uniform timestamps. Returns `true` (vacuously aligned) when
/// fewer than two content peaks can be formed, leaving the decision to the DTW
/// pass itself.
fn cross_attention_peaks_order_aligned(attention: &[Vec<f32>], is_content: &[bool]) -> bool {
    const TOLERANCE_FRAMES: usize = 1;
    let mut previous_peak: Option<usize> = None;
    for (index, row) in attention.iter().enumerate() {
        if !is_content.get(index).copied().unwrap_or(false) || row.is_empty() {
            continue;
        }
        let peak = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(frame, _)| frame);
        let Some(peak) = peak else {
            continue;
        };
        if let Some(previous) = previous_peak {
            // A backward jump of two or more frames is a zig-zag (ties and a
            // single-frame jitter are tolerated).
            if peak
                .checked_add(TOLERANCE_FRAMES)
                .is_some_and(|shifted| shifted < previous)
            {
                return false;
            }
        }
        previous_peak = Some(peak);
    }
    true
}

/// Search horizon, in frames, for the dominant-early-sink mask. ~0.8s at the
/// family's 0.08s/frame; the shared "priming" attention sits at the start of
/// the window, so only early frames are candidates. Late peaks are never
/// masked because they may carry the one token's real speech location.
const SINK_STRIP_SEARCH_FRAMES: usize = 10;

/// Fraction of forward-vs-backward content-token pairs tolerated in the
/// post-sink-strip "mostly-monotone" tier of the order gate. The strict
/// re-test rejects any backward jump of 2+ frames. The tolerant tier admits
/// windows where such jumps affect at most this fraction of content pairs.
/// Windows that fall short of the tolerant tier fall back to the uniform baseline.
const COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION: f32 = 0.10;

/// Minimum DTW band duration, in seconds, before the post-sink-strip
/// "mostly-monotone" tier of the order gate is allowed to admit a window.
/// The tolerant tier exists to catch long, pause-heavy windows (like
/// cohere's 30s long-form chunks) where the raw peak order zig-zags from the
/// shared early sink but the post-strip signal is still mostly left-to-right.
/// On shorter windows (clips of 15-25s), the same post-strip profile can
/// still carry enough accumulated drift for the DTW to land worse than the
/// uniform fallback; the guard keeps the short-window behavior unchanged and
/// restricts the newly-admitted region to where the pauses are actually measurable.
const COHERE_DTW_TOLERANT_MIN_BAND_SECONDS: f32 = 20.0;

/// Minimum window duration, in seconds, before the order-gate fallback switches
/// from "return empty -> uniform" to "place each word at its strongest
/// attention peak". On cohere's 30s long-form chunks the attention-peak order
/// on a pause-heavy decode is too zig-zaggy for the DTW pass, and neither the
/// DTW tiling nor the uniform fallback can leave a real gap: both stretch the
/// few seconds of speech across the whole 30s window (measured start-end span
/// error of up to ~14s on the worst windows). Placing each word at its own
/// strongest frame keeps the words where the attention is and lets the midpoints
/// between them fall naturally, so a 1s utterance in a 30s chunk stays 1s wide
/// instead of stretching to 30s. The threshold matches the tolerant tier's
/// band-length guard: below it the stretch is small enough that the plain
/// uniform baseline (which the caller emits on empty return) remains the safer
/// choice, keeping short-clip behavior unchanged.
const COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS: f32 = 20.0;

/// The wall-clock length of the window, given the per-row frame count and the
/// family's seconds-per-frame.
fn band_duration_seconds(window: &[Vec<f32>], seconds_per_frame: f32) -> f32 {
    let frame_count = window.first().map_or(0, |row| row.len());
    frame_count as f32 * seconds_per_frame
}

/// Returns the fraction of adjacent content-token peak pairs that backward-jump
/// by 2+ frames, in `[0.0, 1.0]`; `0.0` when there are no such pairs (fully
/// monotone after sinks). Non-content tokens are skipped.
fn content_backward_fraction(attention: &[Vec<f32>], is_content: &[bool]) -> f32 {
    let mut previous_peak: Option<usize> = None;
    let mut total_pairs = 0usize;
    let mut backward_pairs = 0usize;
    for (index, row) in attention.iter().enumerate() {
        if !is_content.get(index).copied().unwrap_or(false) || row.is_empty() {
            continue;
        }
        let peak = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(frame, _)| frame);
        let Some(peak) = peak else {
            continue;
        };
        if let Some(previous) = previous_peak {
            total_pairs += 1;
            if peak
                .checked_add(1)
                .is_some_and(|shifted| shifted < previous)
            {
                backward_pairs += 1;
            }
        }
        previous_peak = Some(peak);
    }
    if total_pairs == 0 {
        return 0.0;
    }
    backward_pairs as f32 / total_pairs as f32
}

/// Returns a copy of `window` with each dominant early sink frame zeroed in
/// every row, or `None` when no frame qualifies.
///
/// A dominant early sink is a frame within [`SINK_STRIP_SEARCH_FRAMES`] that
/// is the strict-majority global max of at least one of:
/// - all rows in the window (the strict, conservative reading), or
/// - the content rows only (the rows whose decodings actually carry speech;
///   the gate's order test ignores the non-content rows, so the sink check
///   matches the rows the gate cares about).
///
/// The two readings are a superset of each other in practice: a frame that is
/// a majority of all rows is almost always a majority of the (smaller) content
/// row set too, but the content reading can detect a shared early sink that
/// only dominates the meaningful rows. The union of the two is what this
/// function reports.
///
/// On diffuse, front-loaded decodes one such shared priming frame steals the
/// argmax from most tokens, and its removal is what exposes each row's
/// next-strongest frame (the token's real region) so the order gate can
/// re-test. The caller must still re-run `cross_attention_peaks_order_aligned`
/// on the result: this function only removes the artifact, the gate decides
/// whether the cleaned signal is trustworthy. A row whose only finite mass sat
/// on a masked frame has no valid peak after the strip, which the order gate
/// reads as a missing peak and ignores; when enough rows lose their peak the
/// order collapses and the gate rejects (the safe outcome).
fn mask_dominant_early_sinks(window: &[Vec<f32>], is_content: &[bool]) -> Option<Vec<Vec<f32>>> {
    let row_count = window.len();
    let frame_count = window.first()?.len();
    if frame_count == 0 {
        return None;
    }
    let search = SINK_STRIP_SEARCH_FRAMES.min(frame_count);
    let mut all_peak_counts = vec![0usize; search];
    let mut content_peak_counts = vec![0usize; search];
    let mut content_row_count = 0usize;
    for (index, row) in window.iter().enumerate() {
        let (peak, &value) = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        if value <= 0.0 || peak >= search {
            continue;
        }
        all_peak_counts[peak] += 1;
        if is_content.get(index).copied().unwrap_or(false) {
            content_peak_counts[peak] += 1;
            content_row_count += 1;
        }
    }
    let mut masked = vec![false; search];
    let mut sinks = Vec::new();
    for frame in 0..search {
        let is_all_majority = all_peak_counts[frame].saturating_mul(2) > row_count;
        let is_content_majority = content_row_count > 0
            && content_peak_counts[frame].saturating_mul(2) > content_row_count;
        if is_all_majority || is_content_majority {
            masked[frame] = true;
            sinks.push(frame);
        }
    }
    if sinks.is_empty() {
        return None;
    }
    if std::env::var_os("OPENASR_COHERE_DEBUG_CROSS").is_some() {
        eprintln!("cohere cross sink strip: masking frames {sinks:?}");
    }
    Some(
        window
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(frame, &value)| {
                        if frame < search && masked[frame] {
                            0.0
                        } else {
                            value
                        }
                    })
                    .collect()
            })
            .collect(),
    )
}

fn compute_greedy_step_output_for_graph<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    logits: GgmlCpuTensor<'a>,
    top1: Option<GgmlCpuTensor<'a>>,
    vocab_size: usize,
) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, CohereDecoderGraphError> {
    match top1 {
        Some(top1) => {
            let reversed_token_id = graph
                .compute_output_i32(top1, 1)
                .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?
                .into_iter()
                .next()
                .ok_or_else(|| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: "cohere device top-1 returned no token id".to_string(),
                })?;
            Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(map_reversed_top1_token(reversed_token_id, vocab_size)?),
            })
        }
        None => Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits: graph
                .compute_output_f32(logits, vocab_size)
                .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?,
            greedy_token_hint: None,
        }),
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
    collect_last_cross_attention: bool,
) -> Result<
    (
        crate::ggml_runtime::GgmlCpuTensor<'a>,
        Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    ),
    CohereDecoderGraphError,
> {
    let last_layer_index = layers.len().saturating_sub(1);
    let mut last_probs = None;
    let state = seq2seq_layer_stack(
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
            let (state, probs) = apply_decoder_layer(
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
                collect_last_cross_attention && layer_idx == last_layer_index,
            )?;
            if layer_idx == last_layer_index {
                last_probs = probs;
            }
            Ok(state)
        },
    )?;
    Ok((state, last_probs))
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
    collect_cross_attention: bool,
) -> Result<
    (
        crate::ggml_runtime::GgmlCpuTensor<'a>,
        Option<crate::ggml_runtime::GgmlCpuTensor<'a>>,
    ),
    CohereDecoderGraphError,
> {
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
        cross_kv_max_positions: cross_runtime.capacity_frames,
        cross_hidden_size: cross_runtime.hidden_size,
        collect_cross_attention,
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
    Ok((block.output, block.last_token_cross_attention))
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

pub(super) fn decoder_max_generated_tokens_with_env(
    prompt_tokens: &[u32],
    metadata: CohereTranscribeExecutionMetadata,
    encoder_frame_count: usize,
) -> Result<usize, CohereDecoderGraphError> {
    super::decode_budget::cohere_decode_budget(
        prompt_tokens.len(),
        encoder_frame_count,
        metadata.decoder_max_context,
    )
    .map(|budget| budget.max_generated_tokens)
    .map_err(|error| CohereDecoderGraphError::InvalidInput {
        reason: error.to_string(),
    })
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

fn loaded_decoder_tensor(
    loaded: &GgmlLoadedWeightContext,
    tensor_name: &str,
) -> Result<GgmlLoadedTensor, CohereDecoderGraphError> {
    loaded
        .tensor(tensor_name)
        .ok_or_else(|| CohereDecoderGraphError::InvalidInput {
            reason: format!(
                "cohere decoder verified loaded-weight context is missing tensor '{tensor_name}'"
            ),
        })
}

fn decoder_vector_tensor(
    loaded: Option<&GgmlLoadedWeightContext>,
    arena: &GgmlStaticTensorArena,
    weight: &CohereVectorWeight,
    tensor_name: &'static str,
) -> Result<CohereDecoderWeightTensor, CohereDecoderGraphError> {
    match loaded {
        Some(loaded) => {
            loaded_decoder_tensor(loaded, &weight.name).map(CohereDecoderWeightTensor::Loaded)
        }
        None => new_vector_tensor_in_arena(arena, weight.len, tensor_name)
            .map(CohereDecoderWeightTensor::Static),
    }
}

fn decoder_projection_tensor(
    loaded: Option<&GgmlLoadedWeightContext>,
    arena: &GgmlStaticTensorArena,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<CohereDecoderWeightTensor, CohereDecoderGraphError> {
    match loaded {
        Some(loaded) => arena
            .reshape_loaded_tensor_2d(
                loaded_decoder_tensor(loaded, &weight.name)?,
                weight.cols,
                weight.rows,
                tensor_name,
            )
            .map(CohereDecoderWeightTensor::LoadedMatrixView)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: tensor_name,
                source,
            }),
        None => new_projection_tensor_in_arena(arena, weight, tensor_name)
            .map(CohereDecoderWeightTensor::Static),
    }
}

fn decoder_embedding_tensor(
    loaded: Option<&GgmlLoadedWeightContext>,
    arena: &GgmlStaticTensorArena,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<CohereDecoderWeightTensor, CohereDecoderGraphError> {
    match loaded {
        Some(loaded) => arena
            .reshape_loaded_tensor_2d(
                loaded_decoder_tensor(loaded, &weight.name)?,
                weight.cols,
                weight.rows,
                tensor_name,
            )
            .map(CohereDecoderWeightTensor::LoadedMatrixView)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: tensor_name,
                source,
            }),
        None => new_embedding_tensor_in_arena(arena, weight, tensor_name)
            .map(CohereDecoderWeightTensor::Static),
    }
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
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.rows, weight.cols]
    {
        return arena
            .new_tensor_2d_typed(weight.cols, weight.rows, raw.ggml_type, tensor_name)
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
    tensor: CohereDecoderWeightTensor,
    weight: &CohereVectorWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    let Some(tensor) = tensor.static_tensor() else {
        return Ok(());
    };
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
    tensor: CohereDecoderWeightTensor,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    let Some(tensor) = tensor.static_tensor() else {
        return Ok(());
    };
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
    tensor: CohereDecoderWeightTensor,
    weight: &CohereMatrixWeight,
    tensor_name: &'static str,
) -> Result<(), CohereDecoderGraphError> {
    let Some(tensor) = tensor.static_tensor() else {
        return Ok(());
    };
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.rows, weight.cols]
    {
        return arena
            .set_bytes_slice(tensor, raw.bytes(), tensor_name)
            .map_err(|source| CohereDecoderGraphError::GraphBuildFailed {
                step: tensor_name,
                source,
            });
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
    capacity_frames: usize,
    slot_index: usize,
    step: &'static str,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, CohereDecoderGraphError> {
    // No `n_seq == 1` shortcut returning `cache` unchanged: the planner's
    // resident capacity may be larger than the active logical frame count,
    // so every write must target a
    // contiguous-prefix VIEW of exactly `frame_count` columns -- for `n_seq ==
    // 1` that is `slot_index == 0` with `offset == 0`, identical to the old
    // no-op shortcut whenever `frame_count` happens to equal the tensor's
    // full allocated size.
    let row_stride = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CohereDecoderGraphError::ShapeOverflow)?;
    let slot_stride = hidden_size
        .checked_mul(capacity_frames)
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
        GgmlAsrExecutionOptions, GgufMetadata, GgufMetadataValue, GgufRuntimeSourcePreflight,
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

    fn write_runtime_ready_preflight() -> (TempPath, GgufRuntimeSourcePreflight) {
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
            GgufRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
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

    fn decoder_state(
        metadata: CohereTranscribeExecutionMetadata,
        logical_cross_positions: usize,
        resident_cross_positions: usize,
    ) -> Seq2SeqDecoderState {
        use crate::models::seq2seq_decoder_state::Seq2SeqStateAxis;

        Seq2SeqDecoderState {
            self_attention: Seq2SeqStateAxis {
                logical_positions: metadata.decoder_max_context,
                resident_positions: metadata.decoder_max_context,
                hard_position_cap: metadata.decoder_max_context,
            },
            cross_attention: Seq2SeqStateAxis {
                logical_positions: logical_cross_positions,
                resident_positions: resident_cross_positions,
                hard_position_cap: resident_cross_positions,
            },
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
                decoder_state(
                    metadata,
                    encoder_output.frame_count,
                    encoder_output.frame_count,
                ),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
    fn decoder_runtime_captures_cross_attention_frame_row() {
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
            let frames = encoder_output.frame_count;
            let mut runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                decoder_state(metadata, frames, frames),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
                false,
            )
            .expect("decoder runtime");
            runtime
                .populate_cross_attention_cache(&encoder_output)
                .expect("populate cross cache");

            // Prompt step: no per-token capture (collect_cross_attention off).
            let logits = runtime
                .compute_step_logits(&prompt.token_ids)
                .expect("prompt step logits");
            assert_eq!(logits.len(), metadata.vocab_size);
            assert!(runtime.cross_attention_frame_probs.is_none());

            // Enable capture; an incremental (single new token) step must
            // switch to the unfused f32 cross-attention and capture a
            // head-averaged frame row of the right length, with finite
            // non-negative values summing to ~1 over frames.
            runtime.collect_cross_attention = true;
            let mut next_prefix = prompt.token_ids.clone();
            next_prefix.push(prompt.token_ids[0]);
            let step = runtime
                .compute_step_output(&next_prefix)
                .expect("incremental step output");
            let frame_rows = std::mem::take(&mut runtime.cross_attention_frame_probs)
                .expect("incremental step should capture cross-attention");
            assert_eq!(frame_rows.len(), frames);
            assert!(
                frame_rows
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            );
            let sum: f32 = frame_rows.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "softmax row should sum to ~1, got {sum}"
            );
            // Logits must still be produced for the argmax step.
            let _ = step.logits;
        });
    }

    #[test]
    fn average_cross_attention_frame_row_head_averages_and_validates() {
        // 3 frames, 2 heads -> layout [frame, token, head]; token_count == 1.
        // [frame, token, head] with token_count==1; element (frame, head) lives
        // at frame + frames*head, so a 3-frame/2-head row is 6 floats.
        let attention = vec![
            1.0, 0.0, 0.5, // frame 0, frame 1, frame 2 -- head 0
            0.5, 0.5, 1.0, // frame 0, frame 1, frame 2 -- head 1
        ];
        let out = average_cross_attention_frame_row(&attention, 3, 2).expect("row");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.75_f32).abs() < 1e-5);
        assert!((out[1] - 0.25_f32).abs() < 1e-5);
        assert!((out[2] - 0.75_f32).abs() < 1e-5);
        // Wrong length is rejected, not silently truncated.
        assert!(
            average_cross_attention_frame_row(&attention, 3, 3).is_err(),
            "mismatched length must be an error"
        );
        assert!(average_cross_attention_frame_row(&attention, 0, 2).is_err());
    }

    #[test]
    fn cohere_dtw_word_timestamps_returns_empty_without_alignments() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let decode_text = |_token_ids: &[u32]| Ok(String::new());
        let words = cohere_dtw_word_timestamps::<()>(&[], metadata, &[], 1.0, &decode_text)
            .expect("dtw words");
        assert!(words.is_empty(), "no alignments -> no words");
    }

    #[test]
    fn cohere_dtw_word_timestamps_places_word_at_earlier_attention() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 12;
        // Two "words": token 0 peaks early, token 1 peaks late. Each frame row
        // is a unit-spike (its attention concentrated on one frame).
        let mut row0 = vec![0.01f32; frames];
        row0[2] = 0.97;
        let mut row1 = vec![0.01f32; frames];
        row1[9] = 0.97;
        let alignments = vec![(5u32, row0), (6u32, row1)];
        let decode_text = |token_ids: &[u32]| {
            let mut decoded = String::new();
            for &token_id in token_ids {
                match token_id {
                    5 => decoded.push_str("hi"),
                    6 => decoded.push_str(" there"),
                    _ => {}
                }
            }
            Ok(decoded)
        };
        let duration = (frames as f32) * seconds_per_frame;
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &[0.99, 0.99],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert_eq!(words.len(), 2, "expected two words, got {words:?}");
        assert_eq!(words[0].word, "hi");
        assert_eq!(words[1].word, "there");
        // Monotone timeline and the words sit on their respective attention
        // frames rather than being smeared to the midpoint.
        assert!(words[0].start < words[1].end);
        assert!(words[0].start < 3.0 * seconds_per_frame + 1e-3);
        assert!(words[1].end > 8.0 * seconds_per_frame + 1e-3);
    }

    #[test]
    fn cross_attention_peaks_order_aligned_gate() {
        let spike = |frames: usize, peak: usize| {
            let mut row = vec![0.01f32; frames];
            row[peak] = 0.97;
            row
        };
        // Monotone non-decreasing peaks form a clean left-to-right order.
        assert!(cross_attention_peaks_order_aligned(
            &[spike(12, 2), spike(12, 2), spike(12, 5), spike(12, 9)],
            &[true; 4]
        ));
        // A backward jump of two or more frames is the diffuse front-loaded
        // zig-zag the gate must reject.
        assert!(!cross_attention_peaks_order_aligned(
            &[spike(12, 9), spike(12, 2)],
            &[true; 2]
        ));
        // Ties and a single frame of jitter are tolerated (not a zig-zag).
        assert!(cross_attention_peaks_order_aligned(
            &[spike(12, 5), spike(12, 5), spike(12, 4), spike(12, 5)],
            &[true; 4]
        ));
        // Non-content (punctuation) rows are skipped, so a diffuse peak in
        // them cannot break the order of the surrounding content peaks.
        assert!(cross_attention_peaks_order_aligned(
            &[spike(12, 9), spike(12, 0), spike(12, 10)],
            &[true, false, true]
        ));
        // Fewer than two content peaks -> vacuously aligned (left to the DTW).
        assert!(cross_attention_peaks_order_aligned(
            &[spike(12, 7)],
            &[true]
        ));
    }

    // Builds a "sink-dominated" row: an early sink frame at frame `sink` holds
    // the global max, with the token's real region (a lower value) at `real`.
    // After stripping `sink` this row's argmax becomes `real`.
    fn sink_row(frames: usize, sink: usize, real: usize) -> Vec<f32> {
        let mut row = vec![0.01f32; frames];
        row[sink] = 0.5;
        row[real] = 0.4;
        row
    }

    // A "right-pointing" row whose own speech location is already its global
    // max (it escaped the priming artifact).
    fn right_row(frames: usize, real: usize) -> Vec<f32> {
        let mut row = vec![0.01f32; frames];
        row[real] = 0.5;
        row
    }

    #[test]
    fn mask_dominant_early_sinks_detects_and_strips_the_shared_peak() {
        let frames = 12;
        // A dominant sink at frame 2 is the *global* max for a strict majority
        // (4 of 5) of the rows; the fifth row peaks elsewhere.
        let raw = vec![
            sink_row(frames, 2, 4),
            sink_row(frames, 2, 5),
            sink_row(frames, 2, 6),
            sink_row(frames, 2, 7),
            right_row(frames, 9),
        ];
        let all_content = vec![true; raw.len()];
        let striped = mask_dominant_early_sinks(&raw, &all_content).expect("a sink must be found");
        assert_eq!(striped.len(), raw.len());
        for (raw_row, striped_row) in raw.iter().zip(&striped) {
            assert_eq!(
                striped_row[2], 0.0,
                "sink frame must be zeroed in every row"
            );
            for (frame, (a, b)) in raw_row.iter().zip(striped_row).enumerate() {
                if frame != 2 {
                    assert_eq!(a, b, "non-sink frames stay untouched");
                }
            }
        }
        // Frame 2 is the argmax of only 2 of 4 rows: not a strict majority
        // (2*2 > 4 is false), so no sink qualifies.
        assert!(
            mask_dominant_early_sinks(
                &[
                    sink_row(frames, 2, 4),
                    sink_row(frames, 2, 5),
                    right_row(frames, 8),
                    right_row(frames, 9),
                ],
                &[true; 4],
            )
            .is_none()
        );
        // A shared peak beyond the 10-frame search horizon is out of scope
        // (late peaks carry a token's real region and must never be masked).
        assert!(
            mask_dominant_early_sinks(&[right_row(frames, 11), right_row(frames, 11),], &[true; 2])
                .is_none()
        );
        // Empty first row -> no frame count -> nothing to strip.
        assert!(mask_dominant_early_sinks(&[Vec::new(), vec![0.01; frames]], &[true; 2]).is_none());
    }

    #[test]
    fn sink_strip_restores_order_for_diffuse_front_loaded_decode() {
        let frames = 44;
        // The measured cohere artifact: one early sink (frame 3) steals the
        // argmax from most rows while each row's real region walks left to
        // right. One row sits *right* of another sink row, so the raw peak
        // order zig-zags (rejected); after the sink is stripped the reals are
        // monotone left to right (accepted).
        let rows = vec![
            sink_row(frames, 3, 10),
            sink_row(frames, 3, 13),
            right_row(frames, 16),
            sink_row(frames, 3, 19),
            sink_row(frames, 3, 22),
            sink_row(frames, 3, 25),
            sink_row(frames, 3, 28),
            right_row(frames, 40),
        ];
        let is_content = vec![true; rows.len()];
        assert!(
            !cross_attention_peaks_order_aligned(&rows, &is_content),
            "raw zig-zag must be rejected before stripping"
        );
        let striped =
            mask_dominant_early_sinks(&rows, &is_content).expect("sink frame 3 must be found");
        assert!(
            cross_attention_peaks_order_aligned(&striped, &is_content),
            "stripping the dominant sink must restore a monotone peak order"
        );
    }

    #[test]
    fn content_backward_fraction_counts_zigzag_pairs() {
        let frames = 20;
        // Fully monotone: no backward pairs -> fraction 0.0.
        let monotone = vec![
            right_row(frames, 4),
            right_row(frames, 8),
            right_row(frames, 12),
            right_row(frames, 16),
        ];
        assert_eq!(content_backward_fraction(&monotone, &[true; 4]), 0.0);
        // One backward jump of 8+ frames out of 3 pairs: 1/3.
        let zigzag = vec![
            right_row(frames, 4),
            right_row(frames, 12),
            right_row(frames, 5),
            right_row(frames, 16),
        ];
        assert!((content_backward_fraction(&zigzag, &[true; 4]) - 1.0 / 3.0).abs() < 1e-6);
        // Non-content rows are skipped entirely.
        assert_eq!(
            content_backward_fraction(
                &[
                    right_row(frames, 4),
                    right_row(frames, 0),
                    right_row(frames, 16)
                ],
                &[true, false, true],
            ),
            0.0
        );
        // No content pairs at all -> vacuously 0.0 (the caller still has the
        // strict re-test to fall through to).
        assert_eq!(
            content_backward_fraction(&[right_row(frames, 4)], &[true]),
            0.0
        );
    }

    #[test]
    fn band_duration_seconds_scales_with_row_length() {
        let window = vec![vec![0.0; 375], vec![0.0; 375]];
        assert!((band_duration_seconds(&window, 0.08) - 30.0).abs() < 1e-5);
        let short = vec![vec![0.0; 208], vec![0.0; 208]];
        assert!((band_duration_seconds(&short, 0.08) - 16.64).abs() < 1e-5);
        assert_eq!(band_duration_seconds(&[], 0.08), 0.0);
    }

    #[test]
    fn mask_dominant_early_sinks_strips_frame_dominating_content_rows_only() {
        let frames = 12;
        // 51 content rows peak at the early sink (frame 2); 105 non-content
        // rows peak at later distinct frames. Frame 2 is not a majority of
        // all rows (51*2=102 is not greater than 156) but is a strict
        // majority of the content rows (51*2=102 > 51). The gate only tests
        // content rows, so the sink must still be stripped.
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(156);
        let mut is_content: Vec<bool> = Vec::with_capacity(156);
        for _ in 0..51 {
            rows.push(sink_row(frames, 2, 9));
            is_content.push(true);
        }
        for i in 0..105 {
            let mut row = vec![0.01f32; frames];
            row[(i + 4) % (frames - 4) + 4] = 0.3;
            rows.push(row);
            is_content.push(false);
        }
        let stripped = mask_dominant_early_sinks(&rows, &is_content)
            .expect("sink must be found vs content rows");
        assert_eq!(
            stripped[0][2], 0.0,
            "sink frame must be zeroed in every row"
        );
        assert_eq!(
            stripped[100][2], 0.0,
            "sink strip applies to all rows, not just content"
        );
    }

    /// 20 content rows with the diffuse front-loaded artifact and a single
    /// residual dip: 18 rows share the early sink (frame 3) with their real
    /// region walking left to right around a gap at frame 20 (a right-pointing
    /// row), so the raw peak order zig-zags. After the sink is stripped the
    /// effective peaks are `10..15, 20, 15'..` with exactly one backward pair
    /// (~5% of content pairs): just under the tolerant 10% threshold but above
    /// the strict re-test (which allows zero).
    fn zigzag_after_strip_rows(frames: usize) -> (Vec<Vec<f32>>, Vec<bool>) {
        let reals: [usize; 20] = [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        ];
        let mut rows = Vec::with_capacity(20);
        // Row 5 is the right-pointing escapee at frame 20; everything before it
        // climbs 10..14, then row 5 jumps to 20, then the sink rows resume at
        // 15 and climb to 29. The single 20 -> 15 step is the lone dip.
        for (index, &real) in reals.iter().enumerate() {
            if index == 5 {
                rows.push(right_row(frames, 20));
            } else {
                rows.push(sink_row(frames, 3, real));
            }
        }
        let is_content = vec![true; rows.len()];
        (rows, is_content)
    }

    #[test]
    fn cohere_dtw_word_timestamps_uses_dtw_when_sink_strip_restores_order() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 44;
        // Eight content words with the diffuse front-loaded artifact (same
        // shape as the unit test above): the raw peak order zig-zags, so only
        // the sink-stripped re-test can admit the DTW pass and emit spans.
        let rows = vec![
            sink_row(frames, 3, 10),
            sink_row(frames, 3, 13),
            right_row(frames, 16),
            sink_row(frames, 3, 19),
            sink_row(frames, 3, 22),
            sink_row(frames, 3, 25),
            sink_row(frames, 3, 28),
            right_row(frames, 40),
        ];
        let token_ids = [10u32, 11, 12, 13, 14, 15, 16, 17];
        let alignments: Vec<(u32, Vec<f32>)> = token_ids
            .iter()
            .zip(rows.iter())
            .map(|(&token_id, row)| (token_id, row.clone()))
            .collect();
        let decode_text = |token_ids: &[u32]| {
            let mut decoded = String::new();
            for &token_id in token_ids {
                match token_id {
                    10 => decoded.push_str("And"),
                    11 => decoded.push_str(" so"),
                    12 => decoded.push_str(" my"),
                    13 => decoded.push_str(" fellow"),
                    14 => decoded.push_str(" country"),
                    15 => decoded.push_str(" can"),
                    16 => decoded.push_str(" for"),
                    17 => decoded.push_str(" you"),
                    _ => {}
                }
            }
            Ok(decoded)
        };
        let duration = (frames as f32) * seconds_per_frame;
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            !words.is_empty(),
            "sink-stripped monotone peaks must produce DTW word spans, got empty"
        );
        let actual_words: Vec<&str> = words.iter().map(|word| word.word.as_str()).collect();
        assert_eq!(
            actual_words,
            ["And", "so", "my", "fellow", "country", "can", "for", "you"],
            "word stream must match the transcript"
        );
        // DTW spans tile the band monotonically: non-decreasing, no overlap,
        // and within the clip (the uniform path would spread them evenly).
        for pair in words.windows(2) {
            assert!(pair[0].start <= pair[1].start);
            assert!(pair[0].end - 1e-6 <= pair[1].start);
        }
        assert!(words.last().is_some_and(|last| last.end <= duration + 1e-3));
    }

    #[test]
    fn cohere_dtw_word_timestamps_falls_back_when_sink_strip_cannot_save_order() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 44;
        // A dominant sink at frame 3 is present (3 of 4 rows peak there), but
        // the reals still zig-zag after it is stripped (a middle word's real
        // sits *after* the last one), so no amount of stripping makes the
        // signal trustworthy and the caller keeps the uniform timestamps.
        let rows = vec![
            sink_row(frames, 3, 12),
            right_row(frames, 16),
            sink_row(frames, 3, 30),
            sink_row(frames, 3, 20),
        ];
        let token_ids = [10u32, 11, 12, 13];
        let alignments: Vec<(u32, Vec<f32>)> = token_ids
            .iter()
            .zip(rows.iter())
            .map(|(&token_id, row)| (token_id, row.clone()))
            .collect();
        let decode_text = |token_ids: &[u32]| {
            let mut decoded = String::new();
            for &token_id in token_ids {
                match token_id {
                    10 => decoded.push_str("And"),
                    11 => decoded.push_str(" so"),
                    12 => decoded.push_str(" my"),
                    13 => decoded.push_str(" fellow"),
                    _ => {}
                }
            }
            Ok(decoded)
        };
        let duration = (frames as f32) * seconds_per_frame;
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            words.is_empty(),
            "zig-zag surviving the sink strip must fall back (empty), got {words:?}"
        );
    }

    fn tolerant_tier_window(
        frames: usize,
    ) -> (Vec<(u32, Vec<f32>)>, Vec<Vec<f32>>, Vec<bool>, f32) {
        let (rows, is_content) = zigzag_after_strip_rows(frames);
        let duration = (frames as f32) * (8.0 * 320.0 / 16000.0);
        let token_ids: Vec<u32> = (20..20 + rows.len() as u32).collect();
        let alignments = rows
            .iter()
            .zip(&token_ids)
            .map(|(row, &token_id)| (token_id, row.clone()))
            .collect();
        (alignments, rows, is_content, duration)
    }

    #[test]
    fn cohere_dtw_word_timestamps_uses_tolerant_dtw_when_strip_leaves_minor_zigzag_on_long_window()
    {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        // 260 frames * 0.08s = 20.8s, above the 20s threshold.
        let (alignments, rows, is_content, duration) = tolerant_tier_window(260);
        assert!(duration >= COHERE_DTW_TOLERANT_MIN_BAND_SECONDS);
        // Sanity-check the helper's contract: raw zig-zags (strict re-test fails),
        // but the fractional backward count is exactly 1/19 which is below the
        // 10% tolerant threshold.
        assert!(
            !cross_attention_peaks_order_aligned(&rows, &is_content),
            "the raw peak order must fail the strict monotone re-test"
        );
        let fraction = content_backward_fraction(&rows, &is_content);
        assert!(
            (fraction - 1.0 / 19.0).abs() < 1e-6,
            "expected exactly one backward pair out of 19, got {fraction}"
        );
        let token_ids: Vec<u32> = (20..20 + alignments.len() as u32).collect();
        let decode_text = |token_ids: &[u32]| {
            let mut s = String::new();
            for &token_id in token_ids {
                use std::fmt::Write;
                let _ = write!(s, "w{token_id} ");
            }
            Ok(s.trim_end().to_string())
        };
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            !words.is_empty(),
            "tolerant-tier long window must emit DTW word spans, got empty"
        );
        for (index, word) in words.iter().enumerate() {
            assert_eq!(word.word, format!("w{}", token_ids[index]));
        }
        for pair in words.windows(2) {
            assert!(pair[0].start <= pair[1].start, "timeline must be monotone");
            assert!(pair[0].end - 1e-6 <= pair[1].start, "no overlaps");
        }
        assert!(words.last().is_some_and(|last| last.end <= duration + 1e-3));
    }

    #[test]
    fn cohere_dtw_word_timestamps_falls_back_when_strip_zigzag_fits_short_window() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        // 44 frames * 0.08s = 3.52s, below the 20s threshold. Even though the
        // post-strip backward fraction would clear the tolerant tier, the
        // band-length guard must reject this short window so it falls back to
        // uniform baseline.
        let (alignments, rows, is_content, duration) = tolerant_tier_window(44);
        assert!(
            duration < COHERE_DTW_TOLERANT_MIN_BAND_SECONDS,
            "short-window test must stay below the tolerant band threshold"
        );
        assert!(
            !cross_attention_peaks_order_aligned(&rows, &is_content),
            "raw zig-zag must fail the strict re-test"
        );
        let fraction = content_backward_fraction(&rows, &is_content);
        assert!(
            fraction <= COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION,
            "fraction must be under the tolerant threshold so the only failing check is band length"
        );
        let token_ids: Vec<u32> = (20..20 + alignments.len() as u32).collect();
        let decode_text = |token_ids: &[u32]| {
            let mut s = String::new();
            for &token_id in token_ids {
                use std::fmt::Write;
                let _ = write!(s, "w{token_id} ");
            }
            Ok(s.trim_end().to_string())
        };
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            words.is_empty(),
            "short window below the tolerant band threshold must still fall back, got {words:?}"
        );
    }

    #[test]
    fn cohere_dtw_word_timestamps_falls_back_when_peaks_not_aligned() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let seconds_per_frame = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 12;
        // Two content words, but the second attends *earlier* than the first
        // (the diffuse front-loaded artifact). The gate rejects the DTW pass
        // and returns empty so the caller keeps the uniform post-hoc
        // timestamps instead of the over-spread spans the DTW would emit.
        let mut row0 = vec![0.01f32; frames];
        row0[9] = 0.97;
        let mut row1 = vec![0.01f32; frames];
        row1[2] = 0.97;
        let alignments = vec![(5u32, row0), (6u32, row1)];
        let decode_text = |token_ids: &[u32]| {
            let mut decoded = String::new();
            for &token_id in token_ids {
                match token_id {
                    5 => decoded.push_str("hi"),
                    6 => decoded.push_str(" there"),
                    _ => {}
                }
            }
            Ok(decoded)
        };
        let duration = (frames as f32) * seconds_per_frame;
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &[0.99, 0.99],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            words.is_empty(),
            "non-order-aligned peaks must fall back (empty), got {words:?}",
        );
    }

    /// 30 content rows on `frames` frames arranged so the raw peak order
    /// zig-zags immediately (row 0 peaks at ~100, row 1 at ~5) and ~50% of
    /// adjacent pairs are backward pairs (well above the 10% tolerant
    /// threshold), so BOTH the strict and the tolerant DTW tiers are
    /// rejected. No frame in [0, 10) is a dominant early sink (only one row
    /// peaks in that range), so `mask_dominant_early_sinks` returns `None`
    /// and the gate falls through to the catch-all branch.
    fn zigzag_no_sink_rows(frames: usize) -> Vec<Vec<f32>> {
        assert!(
            frames >= 110,
            "need at least 110 frames for the test pattern"
        );
        let n = 30usize;
        (0..n)
            .map(|i| {
                let step = (i / 2) as usize;
                let frame: usize = if i % 2 == 0 {
                    (frames / 2).saturating_sub(step.saturating_mul(3))
                } else {
                    (5usize + step.saturating_mul(3)).min(10)
                };
                let mut row = vec![0.01_f32; frames];
                row[frame] = 0.5;
                row
            })
            .collect()
    }

    #[test]
    fn cohere_dtw_word_timestamps_uses_peak_fallback_on_long_zigzag_window() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 250; // 20.0s >= PEAK_FALLBACK_MIN_SECONDS
        let duration = (frames as f32) * spf;
        assert!(duration >= COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS);
        let rows = zigzag_no_sink_rows(frames);
        let token_ids: Vec<u32> = (50..50 + rows.len() as u32).collect();
        let alignments: Vec<(u32, Vec<f32>)> = rows
            .iter()
            .zip(&token_ids)
            .map(|(row, &id)| (id, row.to_vec()))
            .collect();
        let decode_text = |token_ids: &[u32]| {
            let mut s = String::new();
            for &id in token_ids {
                use std::fmt::Write;
                let _ = write!(s, "w{id} ");
            }
            Ok(s.trim_end().to_string())
        };
        // Pre-check: raw peaks zigzag (strict fails), and backward fraction is
        // > 10% (tolerant also fails). No early sink is detected.
        let is_content = vec![true; rows.len()];
        assert!(
            !cross_attention_peaks_order_aligned(&rows, &is_content),
            "raw peaks must zigzag for the strict test to fail"
        );
        assert!(
            mask_dominant_early_sinks(&rows, &is_content).is_none(),
            "no dominant early sink is expected in this pattern"
        );
        assert!(
            content_backward_fraction(&rows, &is_content) > COHERE_DTW_MAX_BACKWARD_PAIR_FRACTION,
            "backward fraction must exceed the tolerant threshold"
        );
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("peak fallback words");
        assert!(
            !words.is_empty(),
            "long zigzag window must emit peak-fallback words, got empty"
        );
        // Each word must be at a distinct position (not spread uniformly).
        // The first word starts near the clip start; the last ends at or below
        // the last content-peak center (NOT at the full duration).
        assert!(
            words[0].start >= 0.0,
            "first word must not start before clip start"
        );
        let last_end = words.last().map(|w| w.end).unwrap();
        assert!(
            last_end < duration,
            "last word end ({last_end}) must be bounded by the last content-peak center, not the full duration ({duration})"
        );
        // Monotone and non-overlapping.
        for pair in words.windows(2) {
            assert!(
                pair[0].start <= pair[1].start + 1e-6,
                "timeline must be monotone"
            );
            assert!(pair[0].end - 1e-6 <= pair[1].start, "no overlaps");
        }
    }

    #[test]
    fn cohere_dtw_word_timestamps_falls_back_to_uniform_on_short_zigzag_window() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let spf = 8.0 * metadata.hop_length as f32 / metadata.sample_rate_hz as f32;
        let frames = 110; // 8.8s < PEAK_FALLBACK_MIN_SECONDS
        let duration = (frames as f32) * spf;
        assert!(duration < COHERE_DTW_PEAK_FALLBACK_MIN_SECONDS);
        let rows = zigzag_no_sink_rows(frames);
        let token_ids: Vec<u32> = (50..50 + rows.len() as u32).collect();
        let alignments: Vec<(u32, Vec<f32>)> = rows
            .iter()
            .zip(&token_ids)
            .map(|(row, &id)| (id, row.to_vec()))
            .collect();
        let decode_text = |token_ids: &[u32]| {
            let mut s = String::new();
            for &id in token_ids {
                use std::fmt::Write;
                let _ = write!(s, "w{id} ");
            }
            Ok(s.trim_end().to_string())
        };
        let words = cohere_dtw_word_timestamps::<()>(
            &alignments,
            metadata,
            &vec![0.99; token_ids.len()],
            duration,
            &decode_text,
        )
        .expect("dtw words");
        assert!(
            words.is_empty(),
            "short window below the peak-fallback threshold must fall back to uniform (empty), got {words:?}"
        );
    }

    #[test]
    fn device_top1_matches_full_logits_and_builds_reusable_output_root() {
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
            let state = decoder_state(
                metadata,
                encoder_output.frame_count,
                encoder_output.frame_count,
            );
            let mut full = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                state,
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
                false,
            )
            .expect("full-logits runtime");
            let mut top1 = CohereDecoderGraphRuntime::new_with_n_seq_impl(
                &decoder_weights,
                metadata,
                state,
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
                false,
                1,
                None,
                DeviceGreedyStepOutputMode::DeviceTop1,
            )
            .expect("device-top1 runtime");
            full.populate_cross_attention_cache(&encoder_output)
                .expect("populate full cross cache");
            top1.populate_cross_attention_cache(&encoder_output)
                .expect("populate top1 cross cache");

            let first_logits = full
                .compute_step_logits(&prompt.token_ids)
                .expect("fresh full logits");
            let first_expected = first_logits
                .iter()
                .enumerate()
                .fold(0, |best, (index, value)| {
                    if *value > first_logits[best] {
                        index
                    } else {
                        best
                    }
                }) as u32;
            let first_top1 = top1
                .compute_step_output(&prompt.token_ids)
                .expect("fresh device top1");
            assert!(first_top1.logits.is_empty());
            assert_eq!(first_top1.greedy_token_hint, Some(first_expected));

            let mut next_prefix = prompt.token_ids.clone();
            next_prefix.push(first_expected);
            let next_logits = full
                .compute_step_logits(&next_prefix)
                .expect("reused full logits");
            let next_expected = next_logits
                .iter()
                .enumerate()
                .fold(0, |best, (index, value)| {
                    if *value > next_logits[best] {
                        index
                    } else {
                        best
                    }
                }) as u32;
            let next_top1 = top1
                .compute_step_output(&next_prefix)
                .expect("second fresh device top1");
            assert!(next_top1.logits.is_empty());
            assert_eq!(next_top1.greedy_token_hint, Some(next_expected));

            top1.build_reusable_decode_graph(DeviceGreedyStepOutputMode::DeviceTop1)
                .expect("device top1 reusable graph");
            assert!(top1.reuse.as_ref().and_then(|reuse| reuse.top1).is_some());
        });
    }

    /// One resident arena serves multiple logical chunk shapes from the same
    /// envelope. Shape changes are explicit and never trigger arena growth.
    #[test]
    fn cross_cache_reuses_stable_resident_capacity_without_growing() {
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
            let small_encoder_output = sample_encoder_output(metadata);
            let big_frame_count = small_encoder_output.frame_count + 64;
            let mut rows = Vec::with_capacity(big_frame_count * metadata.decoder_d_model);
            for frame_idx in 0..big_frame_count {
                for hidden_idx in 0..metadata.decoder_d_model {
                    rows.push(
                        ((frame_idx * metadata.decoder_d_model + hidden_idx) as f32 * 0.03125)
                            .sin(),
                    );
                }
            }
            let big_encoder_output = CohereTranscribeEncoderOutput {
                frame_count: big_frame_count,
                hidden_size: metadata.decoder_d_model,
                rows,
            };
            let initial_state =
                decoder_state(metadata, small_encoder_output.frame_count, big_frame_count);
            let mut runtime = CohereDecoderGraphRuntime::new(
                &decoder_weights,
                metadata,
                initial_state,
                small_encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
                false,
            )
            .expect("decoder runtime");
            let resident_capacity = runtime.cross_capacity_frames;
            assert_eq!(resident_capacity, big_frame_count);
            runtime
                .populate_cross_attention_cache(&small_encoder_output)
                .expect("initial logical shape should populate");

            let mismatch = runtime
                .populate_cross_attention_cache(&big_encoder_output)
                .expect_err("unplanned logical shape must fail closed");
            assert!(matches!(
                mismatch,
                CohereDecoderGraphError::InvalidInput { .. }
            ));
            assert_eq!(runtime.cross_capacity_frames, resident_capacity);

            runtime
                .activate_decoder_state(decoder_state(metadata, big_frame_count, big_frame_count))
                .expect("activate larger logical shape inside the resident envelope");
            runtime
                .populate_cross_attention_cache(&big_encoder_output)
                .expect("larger planned logical shape should populate without reallocating");
            assert_eq!(runtime.cross_capacity_frames, resident_capacity);

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
            let logits = runtime
                .compute_step_logits(&prompt.token_ids)
                .expect("step logits after logical shape change");
            assert_eq!(logits.len(), metadata.vocab_size);
            assert!(logits.iter().all(|value| value.is_finite()));

            runtime
                .activate_decoder_state(initial_state)
                .expect("reactivate the smaller logical shape");
            runtime
                .populate_cross_attention_cache(&small_encoder_output)
                .expect("smaller shape should reuse the same resident arena");
            assert_eq!(runtime.cross_capacity_frames, resident_capacity);
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
                decoder_state(
                    metadata,
                    encoder_output.frame_count,
                    encoder_output.frame_count,
                ),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                .build_reusable_decode_graph(DeviceGreedyStepOutputMode::FullLogits)
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
                decoder_state(
                    metadata,
                    encoder_output_0.frame_count,
                    encoder_output_0.frame_count,
                ),
                encoder_output_0.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                decoder_state(
                    metadata,
                    encoder_output_1.frame_count,
                    encoder_output_1.frame_count,
                ),
                encoder_output_1.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                decoder_state(
                    metadata,
                    encoder_output_0.frame_count,
                    encoder_output_0.frame_count,
                ),
                encoder_output_0.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                decoder_state(
                    metadata,
                    encoder_output.frame_count,
                    encoder_output.frame_count,
                ),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                decoder_state(
                    metadata,
                    encoder_output.frame_count,
                    encoder_output.frame_count,
                ),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
                decoder_state(
                    metadata,
                    encoder_output.frame_count,
                    encoder_output.frame_count,
                ),
                encoder_output.hidden_size,
                GgmlCpuGraphBackend::Cpu,
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
