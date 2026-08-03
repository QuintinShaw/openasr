//! parakeet-ctc transcription core (goal-1 S4): tie the frontend → encoder graph
//! → CTC greedy decode → detokenize into a single `transcribe_pcm`. The WER gate
//! test exercises the whole pipeline on the fixed LibriSpeech clip (the
//! exit-signal acceptance: a brand-new CTC architecture transcribes correctly,
//! reusing the shared conformer_block + ctc decode + registries).

#![allow(dead_code)]

use std::fmt;
use std::sync::Arc;

use crate::PARAKEET_CTC_DECODE_POLICY_ID;
use crate::PhraseBiasConfig;
use crate::api::backend::WordTimestamp;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{OpenAsrArchitectureRegistry, PARAKEET_CTC_GGML_ARCHITECTURE_ID};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufMetadata, GgufTensorDataReader};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout,
};
use crate::models::ctc_greedy_decode::{CtcGreedyDecodeError, CtcGreedyDecodeResult};
use crate::models::ctc_streaming_driver::build_ctc_streaming_driver;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, run_builtin_ctc_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionViewRequest, GgmlAsrRuntimeSourcePreflight,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::ggml_streaming_session::GgmlAsrStreamingTranscriptSession;
use crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT;
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::parakeet_runtime_memory::{
    FastConformerMemoryTopology, FastConformerSystemMemoryPlan, checked_sum,
    plan_fastconformer_system_memory, tokenizer_quote_bytes,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner, SystemMemoryOwnerError,
};
use crate::{NativeAsrSession, PARAKEET_CTC_GGML_ADAPTER_ID};

use super::encoder_graph::ParakeetCtcEncoderGraph;
use super::encoder_weights::{ParakeetEncoderWeights, load_parakeet_ctc_encoder_weights};
use super::frontend::ParakeetFrontend;
use super::runtime_contract::{
    ParakeetCtcExecutionMetadata, parse_parakeet_ctc_execution_metadata,
};
use super::tokenizer::ParakeetTokenizer;

const PARAKEET_CTC_RUNTIME_MAX_IDLE_ENTRIES: usize = 4;
const PARAKEET_CTC_RUNTIME_MAX_INSTANCES_PER_KEY: usize = 4;

type ParakeetCtcRuntimeKey = (PackContentKey, ExecutionLaneKey);
type ParakeetCtcRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<ParakeetCtcRuntimeKey, ParakeetCtcPreparedRuntime>;
type ParakeetCtcRuntimeActor =
    PinnedRuntimeActorCheckout<ParakeetCtcRuntimeKey, ParakeetCtcPreparedRuntime>;

/// Resolves the parakeet block-stack `layer_count_hparam` against the parsed
/// metadata (HONESTY CONTRACT: reads the named hparam, NOT `weights.layers.len()`,
/// so a count drift between the descriptor and the materialized stack fails closed).
struct ParakeetLayerCountResolver {
    n_layers: usize,
}

impl LayerCountResolver for ParakeetLayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            "parakeet.n_layers" => Some(self.n_layers),
            _ => None,
        }
    }
}

fn ctc_err_to_string(error: CtcGreedyDecodeError) -> String {
    error.to_string()
}
fn registry_err_to_string(error: BuiltinDecodePolicyComponentRegistryError) -> String {
    error.to_string()
}

struct ParakeetCtcPreparedRuntime {
    metadata: ParakeetCtcExecutionMetadata,
    tokenizer: ParakeetTokenizer,
    graph: ParakeetCtcEncoderGraph,
}

const PARAKEET_CTC_STREAMING_EXECUTOR_ID: &str = "parakeet-ctc-ggml-snapshot-streaming-executor-v1";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParakeetCtcTranscription {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

/// Transcribe 16 kHz mono f32 PCM through the parakeet-ctc pipeline.
/// `runtime_source` is the same already-open pack `reader` was built from: the encoder graph
/// memory-maps it to bind the 2-D linears zero-copy (goals 7+8).
pub(crate) fn transcribe_parakeet_ctc_pcm(
    reader: &GgufTensorDataReader,
    gguf_metadata: &GgufMetadata,
    samples: &[f32],
    runtime_preflight: &GgmlAsrRuntimeSourcePreflight,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    backend: GgmlCpuGraphBackend,
) -> Result<ParakeetCtcTranscription, String> {
    let metadata =
        parse_parakeet_ctc_execution_metadata(gguf_metadata).map_err(|e| e.to_string())?;
    let tokenizer = ParakeetTokenizer::from_metadata(gguf_metadata)?;
    let frontend = ParakeetFrontend::new(&metadata);
    let features = frontend
        .features_from_samples(samples)
        .map_err(|e| e.to_string())?;
    let weights =
        load_parakeet_ctc_encoder_weights(reader, &metadata).map_err(|e| e.to_string())?;
    validate_parakeet_block_stack(metadata, &weights)?;
    let mut graph =
        ParakeetCtcEncoderGraph::new(&weights, metadata, Some(runtime_preflight), backend)
            .map_err(|e| e.to_string())?;
    let output = graph.encode(&features).map_err(|e| e.to_string())?;
    decode_parakeet_output(
        output,
        &tokenizer,
        phrase_bias,
        word_timestamps,
        samples.len() as f32 / 16_000.0_f32,
    )
}

fn transcribe_parakeet_ctc_pcm_cached(
    runtime_pool: &ParakeetCtcRuntimePool,
    samples: &[f32],
    preflight: &crate::GgmlAsrRuntimeSourcePreflight,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    backend: GgmlCpuGraphBackend,
) -> Result<ParakeetCtcTranscription, String> {
    let actor = checkout_parakeet_ctc_prepared_runtime(runtime_pool, preflight, backend)?;
    let samples = samples.to_vec();
    let phrase_bias = phrase_bias.cloned();
    actor
        .call_mut(move |runtime| {
            runtime.transcribe(&samples, phrase_bias.as_ref(), word_timestamps)
        })
        .map_err(|error| error.to_string())?
}

fn decode_parakeet_ctc_pcm_cached(
    runtime_pool: &ParakeetCtcRuntimePool,
    samples: &[f32],
    preflight: &crate::GgmlAsrRuntimeSourcePreflight,
    phrase_bias: Option<&PhraseBiasConfig>,
    backend: GgmlCpuGraphBackend,
) -> Result<CtcGreedyDecodeResult, String> {
    let actor = checkout_parakeet_ctc_prepared_runtime(runtime_pool, preflight, backend)?;
    let samples = samples.to_vec();
    let phrase_bias = phrase_bias.cloned();
    actor
        .call_mut(move |runtime| runtime.decode_result(&samples, phrase_bias.as_ref()))
        .map_err(|error| error.to_string())?
}

fn new_parakeet_ctc_runtime_pool() -> Arc<ParakeetCtcRuntimePool> {
    Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
        "openasr-parakeet-ctc-runtime-owner",
        AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            PARAKEET_CTC_RUNTIME_MAX_IDLE_ENTRIES,
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
            PARAKEET_CTC_RUNTIME_MAX_INSTANCES_PER_KEY,
        ),
    ))
}

fn checkout_parakeet_ctc_prepared_runtime(
    pool: &ParakeetCtcRuntimePool,
    preflight: &crate::GgmlAsrRuntimeSourcePreflight,
    resolved_backend: GgmlCpuGraphBackend,
) -> Result<ParakeetCtcRuntimeActor, String> {
    let backend = crate::models::parakeet_ctc::graph_config::parakeet_ctc_encoder_graph_config(
        resolved_backend,
    )
    .backend;
    let key = (
        PackContentKey::for_runtime_source(&preflight.runtime_source),
        current_execution_lane_key(backend),
    );
    let preflight = preflight.clone();
    let pack_content_id = preflight.runtime_source.content_id().to_string();
    pool.checkout_or_try_build_with(
        key,
        move || {
            let reader = build_runtime_tensor_reader_from_preflight(&preflight)
                .map_err(|error| error.to_string())?;
            let metadata = parse_parakeet_ctc_execution_metadata(&preflight.metadata)
                .map_err(|error| error.to_string())?;
            let (quote, plan) = parakeet_ctc_runtime_system_memory_quote(
                &preflight.metadata,
                &preflight.tensor_index,
                metadata,
                &pack_content_id,
            )
            .map_err(|error| error.to_string())?;
            Ok((
                quote.retained_bytes,
                (preflight, reader, metadata, quote, plan, backend),
            ))
        },
        |(preflight, reader, metadata, quote, plan, backend)| {
            match SystemMemoryOwner::try_allocate_transaction(quote, || {
                let tokenizer = ParakeetTokenizer::from_metadata(&preflight.metadata)?;
                let tokenizer_bytes = tokenizer.retained_system_memory_bytes()?;
                let weights = load_parakeet_ctc_encoder_weights(&reader, &metadata)
                    .map_err(|error| error.to_string())?;
                validate_parakeet_block_stack(metadata, &weights)?;
                let weights_bytes = weights.retained_system_memory_bytes()?;
                let graph =
                    ParakeetCtcEncoderGraph::new(&weights, metadata, Some(&preflight), backend)
                        .map_err(|error| error.to_string())?;
                let graph_bytes = graph.retained_system_memory_bytes()?;
                drop(weights);

                let retained = checked_sum(
                    [tokenizer_bytes, graph_bytes],
                    "parakeet-ctc measured runtime retained bytes",
                )
                .map_err(|error| error.to_string())?;
                let measured_build_peak = checked_sum(
                    [tokenizer_bytes, weights_bytes, graph_bytes],
                    "parakeet-ctc measured runtime build peak",
                )
                .map_err(|error| error.to_string())?;
                let planned_build_peak = tokenizer_bytes
                    .checked_add(plan.build_peak_bytes)
                    .ok_or_else(|| "parakeet-ctc measured build peak overflowed".to_string())?;
                let runtime = ParakeetCtcPreparedRuntime {
                    metadata,
                    tokenizer,
                    graph,
                };
                Ok(SystemMemoryAllocationOutcome::new(
                    runtime,
                    measured_build_peak.max(planned_build_peak),
                    retained,
                ))
            }) {
                Ok(owner) => Ok(owner),
                Err(SystemMemoryAllocationTransactionError::Allocation(reason)) => Err(reason),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(error.to_string())
                }
            }
        },
        |error| error.to_string(),
    )
}

fn parakeet_ctc_runtime_system_memory_quote(
    gguf_metadata: &GgufMetadata,
    tensor_index: &crate::GgufTensorIndex,
    metadata: ParakeetCtcExecutionMetadata,
    pack_content_id: &str,
) -> Result<(SystemMemoryAllocationQuote, FastConformerSystemMemoryPlan), SystemMemoryOwnerError> {
    let tokenizer_bytes = tokenizer_quote_bytes(gguf_metadata, "parakeet-ctc")?;
    let plan = plan_fastconformer_system_memory(
        tensor_index,
        FastConformerMemoryTopology {
            n_layers: metadata.n_layers,
            hidden_size: metadata.hidden_size,
            ffn_dim: metadata.ffn_dim,
            checkpoint_has_projection_biases: true,
            bound_tail_weight: "ctc.head.weight",
            retained_tail_bias: "ctc.head.bias",
        },
    )?;
    let retained_bytes = checked_sum(
        [tokenizer_bytes, plan.graph_retained_bytes],
        "parakeet-ctc quoted runtime retained bytes",
    )?;
    let peak_bytes = tokenizer_bytes
        .checked_add(plan.build_peak_bytes)
        .ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "parakeet_ctc_runtime_quote",
                "parakeet-ctc quoted runtime peak overflowed",
            )
        })?;
    let quote = SystemMemoryAllocationQuote::new(
        format!("parakeet-ctc-prepared-runtime:{pack_content_id}"),
        peak_bytes,
        retained_bytes,
    )?;
    Ok((quote, plan))
}

impl ParakeetCtcPreparedRuntime {
    fn transcribe(
        &mut self,
        samples: &[f32],
        phrase_bias: Option<&PhraseBiasConfig>,
        word_timestamps: bool,
    ) -> Result<ParakeetCtcTranscription, String> {
        let result = self.decode_result(samples, phrase_bias)?;
        parakeet_ctc_result_to_transcription(
            result,
            &self.tokenizer,
            word_timestamps,
            samples.len() as f32 / 16_000.0_f32,
        )
    }

    fn decode_result(
        &mut self,
        samples: &[f32],
        phrase_bias: Option<&PhraseBiasConfig>,
    ) -> Result<CtcGreedyDecodeResult, String> {
        let frontend = ParakeetFrontend::new(&self.metadata);
        let features = frontend
            .features_from_samples(samples)
            .map_err(|e| e.to_string())?;
        let output = self.graph.encode(&features).map_err(|e| e.to_string())?;
        decode_parakeet_ctc_result(&output, &self.tokenizer, phrase_bias)
    }
}

fn validate_parakeet_block_stack(
    metadata: ParakeetCtcExecutionMetadata,
    weights: &ParakeetEncoderWeights,
) -> Result<(), String> {
    // Make the block-stack descriptor LOAD-BEARING (P4 exit-signal honesty): the
    // Ctc encoder stack this pack materialized must agree with the parakeet
    // descriptor's declared shape / block kind / tensor scope / layer count, else
    // fail closed rather than silently build the wrong thing. Mirrors qwen+cohere.
    let parakeet_block_stack = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(PARAKEET_CTC_GGML_ARCHITECTURE_ID)
        .and_then(|descriptor| descriptor.block_stack);
    validate_stage_against_descriptor(
        PARAKEET_CTC_GGML_ARCHITECTURE_ID,
        parakeet_block_stack.as_ref(),
        OpenAsrStageRole::Encoder,
        OpenAsrOrchestrationShape::Ctc,
        StageBuildPlan {
            block_kind: OpenAsrBlockKind::ConformerBlock,
            tensor_name_scope: "enc.blk",
            family_layer_count: weights.layers.len(),
        },
        &ParakeetLayerCountResolver {
            n_layers: metadata.n_layers,
        },
    )
    .map_err(|error| format!("parakeet-ctc encoder block-stack descriptor mismatch: {error:?}"))?;
    Ok(())
}

fn decode_parakeet_output(
    output: super::encoder_graph::ParakeetCtcEncoderOutput,
    tokenizer: &ParakeetTokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    duration_seconds: f32,
) -> Result<ParakeetCtcTranscription, String> {
    let result = decode_parakeet_ctc_result(&output, tokenizer, phrase_bias)?;
    parakeet_ctc_result_to_transcription(result, tokenizer, word_timestamps, duration_seconds)
}

pub(crate) fn decode_parakeet_ctc_result(
    output: &super::encoder_graph::ParakeetCtcEncoderOutput,
    tokenizer: &ParakeetTokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<CtcGreedyDecodeResult, String> {
    let frame_logits: Vec<&[f32]> = (0..output.frame_count)
        .map(|f| &output.logits[f * output.vocab_size..(f + 1) * output.vocab_size])
        .collect();
    let detok = |ids: &[u32]| tokenizer.decode(ids);
    run_builtin_ctc_decode_policy(
        PARAKEET_CTC_DECODE_POLICY_ID,
        &frame_logits,
        output.vocab_size,
        phrase_bias,
        tokenizer,
        &detok,
        ctc_err_to_string,
        registry_err_to_string,
    )
}

pub(crate) fn parakeet_ctc_result_to_transcription(
    result: CtcGreedyDecodeResult,
    tokenizer: &ParakeetTokenizer,
    word_timestamps: bool,
    duration_seconds: f32,
) -> Result<ParakeetCtcTranscription, String> {
    let words = if word_timestamps {
        tokenizer.word_timestamps_from_token_spans(
            &result.token_spans,
            duration_seconds,
            result.frame_count,
        )?
    } else {
        Vec::new()
    };
    Ok(ParakeetCtcTranscription {
        text: result.text,
        words,
    })
}

/// Dedicated GgmlAsrViewExecutor for parakeet-ctc (DedicatedRuntimeExecutorV1).
/// Reuses a prepared runtime by `(canonical path, backend)`, runs the CTC
/// pipeline, returns a single-segment transcription.
#[derive(Clone)]
pub(crate) struct ParakeetCtcGgmlExecutor {
    runtime_pool: Arc<ParakeetCtcRuntimePool>,
}

impl fmt::Debug for ParakeetCtcGgmlExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParakeetCtcGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for ParakeetCtcGgmlExecutor {
    fn default() -> Self {
        Self {
            runtime_pool: new_parakeet_ctc_runtime_pool(),
        }
    }
}

impl ParakeetCtcGgmlExecutor {
    fn execute_ctc_result(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        decode_parakeet_ctc_pcm_cached(
            &self.runtime_pool,
            &request.prepared_audio.samples_f32,
            &preflight,
            request.request_options.phrase_bias.as_ref(),
            request.resolved_runtime.backend(),
        )
        .map_err(fail)
    }
}

impl GgmlAsrViewExecutor for ParakeetCtcGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::PARAKEET_CTC_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<
        crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
        crate::GgmlAsrExecutionError,
    > {
        Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
    }

    fn execute_view(
        &self,
        request: &crate::models::ggml_asr_executor::GgmlAsrExecutionViewRequest,
    ) -> Result<
        crate::models::ggml_asr_executor::GgmlAsrExecutionResult,
        crate::models::ggml_asr_executor::GgmlAsrExecutionError,
    > {
        use crate::api::backend::{Segment, Transcription};
        use crate::models::ggml_asr_executor::{GgmlAsrExecutionError, GgmlAsrExecutionResult};
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Fail-closed: validate the runtime source path before touching the pack
        // (Gate-0 preflight), then run the cached prepared-runtime path.
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let output = transcribe_parakeet_ctc_pcm_cached(
            &self.runtime_pool,
            &request.prepared_audio.samples_f32,
            &preflight,
            request.request_options.phrase_bias.as_ref(),
            request.request_options.word_timestamps,
            request.resolved_runtime.backend(),
        )
        .map_err(fail)?;
        let duration = request.prepared_audio.samples_f32.len() as f32 / 16_000.0_f32;
        let segments = if output.text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: duration,
                text: output.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: output.words,
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: output.text,
                segments,
                longform: None,
                language: None,
            },
            carry_context: None,
            decode_truncation: None,
        })
    }
}

impl GgmlAsrStreamingExecutor for ParakeetCtcGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        PARAKEET_CTC_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        if request.selected_family.adapter_id != PARAKEET_CTC_GGML_ADAPTER_ID {
            return Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: PARAKEET_CTC_STREAMING_EXECUTOR_ID,
                adapter_id: request.selected_family.adapter_id,
                reason: format!(
                    "parakeet-ctc streaming executor requires adapter '{PARAKEET_CTC_GGML_ADAPTER_ID}', got '{}'",
                    request.selected_family.adapter_id
                ),
            });
        }

        // Gate-off → snapshot driver; gate-on → incremental/windowed
        // driver. The FINAL is byte-identical either way; only partials differ.
        let driver = build_ctc_streaming_driver(
            self.clone(),
            PARAKEET_CTC_STREAMING_EXECUTOR_ID,
            PARAKEET_CTC_GGML_ADAPTER_ID,
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            ParakeetCtcGgmlExecutor::execute_ctc_result,
            <ParakeetCtcGgmlExecutor as GgmlAsrViewExecutor>::execute_view,
        );
        let session = GgmlAsrStreamingTranscriptSession::new(
            PARAKEET_CTC_STREAMING_EXECUTOR_ID,
            request,
            driver,
        )?;
        Ok(Box::new(session))
    }

    fn unload_idle_state(&self) {
        self.runtime_pool.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source;
    use std::path::Path;

    /// Minimal RIFF/WAVE reader for the fixed test clips (16 kHz mono 16-bit PCM).
    fn read_wav_mono_16k(path: &Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        // find the "data" subchunk.
        let mut i = 12; // skip RIFF....WAVE
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            if id == b"data" {
                let start = i + 8;
                let end = (start + size).min(bytes.len());
                return Some(
                    bytes[start..end]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                        .collect(),
                );
            }
            i += 8 + size + (size & 1);
        }
        None
    }

    /// Word-level WER (Levenshtein over uppercased, punctuation-free tokens).
    fn wer(reference: &str, hypothesis: &str) -> f64 {
        let norm = |s: &str| -> Vec<String> {
            s.to_uppercase()
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
                .split_whitespace()
                .map(str::to_string)
                .collect()
        };
        let r = norm(reference);
        let h = norm(hypothesis);
        if r.is_empty() {
            return if h.is_empty() { 0.0 } else { 1.0 };
        }
        let mut prev: Vec<usize> = (0..=h.len()).collect();
        let mut cur = vec![0usize; h.len() + 1];
        for (i, rw) in r.iter().enumerate() {
            cur[0] = i + 1;
            for (j, hw) in h.iter().enumerate() {
                let cost = usize::from(rw != hw);
                cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[h.len()] as f64 / r.len() as f64
    }

    /// The S4 exit-signal gate: parakeet-ctc-0.6b transcribes the fixed clip.
    /// Skipped when the pack/clip are absent. Prints the hypothesis + WER so the
    /// frontend (per-feature norm / log-guard / pad) can be bisected if WER is high.
    #[test]
    fn parakeet_ctc_transcribes_librispeech_clip_when_present() {
        // Resolve relative to the workspace root; tmp/ is gitignored, so the test
        // skips when the pack is absent.
        let roots = [Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")];
        let find = |rel: &str| roots.iter().map(|r| r.join(rel)).find(|p| p.exists());
        let (Some(pack), Some(clip)) = (
            find("tmp/models/parakeet-ctc-0.6b/openasr/parakeet-ctc-0.6b-fp16.oasr"),
            find("tmp/audio/librispeech/237-134500-0000.wav"),
        ) else {
            eprintln!("skipping: parakeet pack or clip absent");
            return;
        };
        let reference = "FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS \
                         DIVORCE CASE THE ANGRIER HE GREW";
        let samples = read_wav_mono_16k(&clip).expect("wav");
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("runtime source");
        let preflight = load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .expect("preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");

        let hypothesis = transcribe_parakeet_ctc_pcm(
            &reader,
            &preflight.metadata,
            &samples,
            &preflight,
            None,
            false,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("decode")
        .text;
        let wer = wer(reference, &hypothesis);
        eprintln!("parakeet-ctc hypothesis: {hypothesis:?}\nWER = {wer:.3}");
        // parakeet-ctc-0.6b is LibriSpeech-clean; the full pipeline (NeMo frontend
        // + FastConformer encoder + CTC greedy) transcribes this clip at WER 0.
        assert!(
            wer <= 0.05,
            "WER {wer:.3} too high; hypothesis: {hypothesis:?}"
        );
    }
}
