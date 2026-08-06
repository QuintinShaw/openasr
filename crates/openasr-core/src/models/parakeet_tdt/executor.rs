//! parakeet-tdt transcription core: frontend -> encoder graph (with in-graph
//! joint encoder projection) -> host TDT greedy decode -> detokenize.

use std::fmt;
use std::sync::Arc;

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufMetadata};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionViewRequest, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::parakeet_ctc::frontend::ParakeetFrontend;
use crate::models::parakeet_runtime_memory::{
    FastConformerMemoryTopology, FastConformerSystemMemoryPlan, checked_sum, element_bytes,
    named_tensor_quote_bytes, plan_fastconformer_system_memory, tokenizer_quote_bytes,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner, SystemMemoryOwnerError,
};
use crate::{NativeAsrSession, PARAKEET_TDT_GGML_ADAPTER_ID};

use super::encoder_graph::{ParakeetTdtEncoderGraph, ParakeetTdtMelFeatures};
use super::encoder_weights::{
    ParakeetTdtLstmLayerWeights, load_parakeet_tdt_encoder_weights,
    load_parakeet_tdt_joint_weights, load_parakeet_tdt_predictor_weights,
};
use super::greedy::{ParakeetTdtJoint, tdt_greedy_decode};
use super::predictor::ParakeetTdtPredictor;
use super::runtime_contract::{
    ParakeetTdtExecutionMetadata, parse_parakeet_tdt_execution_metadata,
};
use super::tokenizer::ParakeetTdtTokenizer;

type ParakeetTdtRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
type ParakeetTdtRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<ParakeetTdtRuntimeCacheKey, ParakeetTdtPreparedRuntime>;
type ParakeetTdtRuntimeActor =
    PinnedRuntimeActorCheckout<ParakeetTdtRuntimeCacheKey, ParakeetTdtPreparedRuntime>;

const PARAKEET_TDT_RUNTIME_MAX_IDLE_ENTRIES: usize = 4;
const PARAKEET_TDT_RUNTIME_MAX_INSTANCES_PER_KEY: usize = 4;

const PARAKEET_TDT_STREAMING_EXECUTOR_ID: &str = "parakeet-tdt-ggml-redecode-streaming-executor-v1";

struct ParakeetTdtPreparedRuntime {
    metadata: ParakeetTdtExecutionMetadata,
    tokenizer: ParakeetTdtTokenizer,
    graph: ParakeetTdtEncoderGraph,
    predictor: ParakeetTdtPredictor,
    joint: ParakeetTdtJoint,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParakeetTdtTranscription {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

fn new_parakeet_tdt_runtime_pool() -> Arc<ParakeetTdtRuntimePool> {
    Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
        "openasr-parakeet-tdt-runtime-owner",
        AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            PARAKEET_TDT_RUNTIME_MAX_IDLE_ENTRIES,
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
            PARAKEET_TDT_RUNTIME_MAX_INSTANCES_PER_KEY,
        ),
    ))
}

fn checkout_parakeet_tdt_prepared_runtime(
    pool: &ParakeetTdtRuntimePool,
    preflight: &crate::GgufRuntimeSourcePreflight,
    resolved_backend: GgmlCpuGraphBackend,
) -> Result<ParakeetTdtRuntimeActor, String> {
    let backend = crate::models::parakeet_tdt::graph_config::parakeet_tdt_encoder_graph_config(
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
            let metadata = parse_parakeet_tdt_execution_metadata(&preflight.metadata)
                .map_err(|error| error.to_string())?;
            let (quote, plan) = parakeet_tdt_runtime_system_memory_quote(
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
                let tokenizer = ParakeetTdtTokenizer::from_metadata(&preflight.metadata)?;
                let tokenizer_bytes = tokenizer.retained_system_memory_bytes()?;

                let encoder_weights = load_parakeet_tdt_encoder_weights(&reader, &metadata)
                    .map_err(|error| error.to_string())?;
                let encoder_weights_bytes = encoder_weights.retained_system_memory_bytes()?;
                let graph =
                    ParakeetTdtEncoderGraph::new(&encoder_weights, metadata, &preflight, backend)
                        .map_err(|error| error.to_string())?;
                let graph_bytes = graph.retained_system_memory_bytes()?;
                drop(encoder_weights);

                let predictor_weights = load_parakeet_tdt_predictor_weights(&reader, &metadata)
                    .map_err(|error| error.to_string())?;
                let predictor_bytes = predictor_weights.retained_system_memory_bytes()?;
                let predictor = ParakeetTdtPredictor::new(
                    predictor_weights,
                    metadata.pred_hidden,
                    metadata.vocab_size,
                );
                let joint_weights = load_parakeet_tdt_joint_weights(&reader, &metadata)
                    .map_err(|error| error.to_string())?;
                let joint_bytes = joint_weights.retained_system_memory_bytes()?;
                let joint = ParakeetTdtJoint::new(joint_weights, metadata.joint_hidden);

                let retained = checked_sum(
                    [tokenizer_bytes, graph_bytes, predictor_bytes, joint_bytes],
                    "parakeet-tdt measured runtime retained bytes",
                )
                .map_err(|error| error.to_string())?;
                let measured_encoder_peak = checked_sum(
                    [tokenizer_bytes, encoder_weights_bytes, graph_bytes],
                    "parakeet-tdt measured encoder build peak",
                )
                .map_err(|error| error.to_string())?;
                let planned_encoder_peak = tokenizer_bytes
                    .checked_add(plan.build_peak_bytes)
                    .ok_or_else(|| "parakeet-tdt measured build peak overflowed".to_string())?;
                let runtime = ParakeetTdtPreparedRuntime {
                    metadata,
                    tokenizer,
                    graph,
                    predictor,
                    joint,
                };
                Ok(SystemMemoryAllocationOutcome::new(
                    runtime,
                    retained
                        .max(measured_encoder_peak)
                        .max(planned_encoder_peak),
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

fn parakeet_tdt_runtime_system_memory_quote(
    gguf_metadata: &GgufMetadata,
    tensor_index: &crate::GgufTensorIndex,
    metadata: ParakeetTdtExecutionMetadata,
    pack_content_id: &str,
) -> Result<(SystemMemoryAllocationQuote, FastConformerSystemMemoryPlan), SystemMemoryOwnerError> {
    let tokenizer_bytes = tokenizer_quote_bytes(gguf_metadata, "parakeet-tdt")?;
    let plan = plan_fastconformer_system_memory(
        tensor_index,
        FastConformerMemoryTopology {
            n_layers: metadata.n_layers,
            hidden_size: metadata.hidden_size,
            ffn_dim: metadata.ffn_dim,
            checkpoint_has_projection_biases: false,
            bound_tail_weight: "enc.proj.weight",
            retained_tail_bias: "enc.proj.bias",
        },
    )?;
    let predictor_bytes = parakeet_tdt_predictor_quote_bytes(tensor_index, metadata.pred_layers)?;
    let joint_bytes = checked_sum(
        [
            named_tensor_quote_bytes(tensor_index, "joint.pred.weight", true)?,
            named_tensor_quote_bytes(tensor_index, "joint.pred.bias", true)?,
            named_tensor_quote_bytes(tensor_index, "joint.out.weight", true)?,
            named_tensor_quote_bytes(tensor_index, "joint.out.bias", true)?,
        ],
        "parakeet-tdt quoted joint bytes",
    )?;
    let retained_bytes = checked_sum(
        [
            tokenizer_bytes,
            plan.graph_retained_bytes,
            predictor_bytes,
            joint_bytes,
        ],
        "parakeet-tdt quoted runtime retained bytes",
    )?;
    let peak_bytes = tokenizer_bytes
        .checked_add(plan.build_peak_bytes)
        .ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "parakeet_tdt_runtime_quote",
                "parakeet-tdt quoted encoder build peak overflowed",
            )
        })?
        .max(retained_bytes);
    let quote = SystemMemoryAllocationQuote::new(
        format!("parakeet-tdt-prepared-runtime:{pack_content_id}"),
        peak_bytes,
        retained_bytes,
    )?;
    Ok((quote, plan))
}

fn parakeet_tdt_predictor_quote_bytes(
    tensor_index: &crate::GgufTensorIndex,
    pred_layers: usize,
) -> Result<u64, SystemMemoryOwnerError> {
    let mut values = vec![named_tensor_quote_bytes(
        tensor_index,
        "dec.embed.weight",
        true,
    )?];
    values.push(element_bytes::<ParakeetTdtLstmLayerWeights>(
        pred_layers,
        "parakeet-tdt predictor layer descriptors",
    )?);
    for layer in 0..pred_layers {
        for suffix in ["w_ih", "w_hh", "b_ih", "b_hh"] {
            values.push(named_tensor_quote_bytes(
                tensor_index,
                &format!("dec.lstm.{layer}.{suffix}"),
                true,
            )?);
        }
    }
    checked_sum(values, "parakeet-tdt quoted predictor bytes")
}

impl ParakeetTdtPreparedRuntime {
    fn transcribe(
        &mut self,
        samples: &[f32],
        word_timestamps: bool,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<ParakeetTdtTranscription, String> {
        let frontend = ParakeetFrontend::with_n_mels(self.metadata.n_mels);
        let features = frontend
            .features_from_samples(samples)
            .map_err(|e| e.to_string())?;
        let output = self
            .graph
            .encode(&ParakeetTdtMelFeatures {
                data: features.data,
                n_frames: features.n_frames,
                n_mels: features.n_mels,
            })
            .map_err(|e| e.to_string())?;
        if output.joint_hidden != self.metadata.joint_hidden {
            return Err(format!(
                "parakeet-tdt encoder emitted joint width {}, metadata declares {}",
                output.joint_hidden, self.metadata.joint_hidden
            ));
        }
        let emitted = tdt_greedy_decode(
            &output.features,
            output.frame_count,
            &self.metadata,
            &self.predictor,
            &self.joint,
            is_canceled,
        )?;
        let token_ids: Vec<u32> = emitted.iter().map(|token| token.token_id).collect();
        let text = self.tokenizer.decode(&token_ids)?;
        let words = if word_timestamps {
            self.tokenizer.word_timestamps_from_emitted(
                &emitted,
                samples.len() as f32 / 16_000.0_f32,
                output.frame_count,
            )?
        } else {
            Vec::new()
        };
        Ok(ParakeetTdtTranscription { text, words })
    }
}

/// Transcribe 16 kHz mono f32 PCM through a cached prepared runtime keyed by
/// `(pack content id, backend)`. The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a runtime built from the old
/// bytes.
fn transcribe_parakeet_tdt_pcm_cached(
    runtime_pool: &ParakeetTdtRuntimePool,
    samples: &[f32],
    preflight: &crate::GgufRuntimeSourcePreflight,
    word_timestamps: bool,
    backend: GgmlCpuGraphBackend,
    control: Arc<crate::api::backend::TranscriptionControl>,
) -> Result<ParakeetTdtTranscription, String> {
    let actor = checkout_parakeet_tdt_prepared_runtime(runtime_pool, preflight, backend)?;
    let samples = samples.to_vec();
    actor
        .call_mut(move |runtime| {
            runtime.transcribe(&samples, word_timestamps, &|| control.is_canceled())
        })
        .map_err(|error| error.to_string())?
}

/// Dedicated GgmlAsrViewExecutor for parakeet-tdt (DedicatedRuntimeExecutorV1).
#[derive(Clone)]
pub(crate) struct ParakeetTdtGgmlExecutor {
    runtime_pool: Arc<ParakeetTdtRuntimePool>,
}

impl fmt::Debug for ParakeetTdtGgmlExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParakeetTdtGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for ParakeetTdtGgmlExecutor {
    fn default() -> Self {
        Self {
            runtime_pool: new_parakeet_tdt_runtime_pool(),
        }
    }
}

impl ParakeetTdtGgmlExecutor {
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.runtime_pool
            .evict_where(|(key, _lane)| key.pack_content_id == pack_content_id);
    }
}

impl GgmlAsrViewExecutor for ParakeetTdtGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        ParakeetTdtGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        crate::arch::PARAKEET_TDT_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // The TDT greedy loop does not apply vocab-logit boosts yet (the
        // xasr transducer precedent); keep the capability honest.
        false
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<
        crate::models::ggml_asr_executor::GgmlAsrExecutionResult,
        crate::models::ggml_asr_executor::GgmlAsrExecutionError,
    > {
        use crate::api::backend::{Segment, Transcription};
        use crate::models::ggml_asr_executor::GgmlAsrExecutionResult;
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::PARAKEET_TDT_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Fail-closed: consume the already-admitted Gate-0 proof, then run the
        // cached prepared-runtime path against that same open source -- never
        // reopen or reparse a path inside the executor.
        let preflight = request.runtime_source_preflight();
        let output = transcribe_parakeet_tdt_pcm_cached(
            &self.runtime_pool,
            &request.prepared_audio.samples_f32,
            preflight,
            request.request_options.word_timestamps,
            request.resolved_runtime.backend(),
            Arc::clone(&request.execution_context.control),
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
                ..Default::default()
            },
            carry_context: None,
            decode_truncation: None,
        })
    }
}

impl GgmlAsrStreamingExecutor for ParakeetTdtGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        PARAKEET_TDT_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        // Partials re-decode the trailing window through the same offline
        // pipeline as the FINAL (the shared re-decode session every
        // non-frame-sync family uses); the FINAL stays byte-identical to
        // `execute()`. TDT's frame-synchronous decode makes a true
        // append-only frame-sync driver possible later; the offline re-decode
        // keeps v1 honest and simple.
        build_seq2seq_streaming_session(
            self.clone(),
            PARAKEET_TDT_STREAMING_EXECUTOR_ID,
            PARAKEET_TDT_GGML_ADAPTER_ID,
            "parakeet-tdt",
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            <ParakeetTdtGgmlExecutor as GgmlAsrViewExecutor>::execute_view,
        )
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

    fn read_wav_mono_16k(path: &Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        let mut i = 12;
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

    /// The exit-signal gate: parakeet-tdt-0.6b-v3 transcribes the bundled JFK
    /// clip coherently, with native word timestamps in order. Skipped when
    /// the pack is absent (tmp/ is host-local).
    #[test]
    fn parakeet_tdt_transcribes_jfk_clip_when_pack_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let pack = root
            .join("tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr");
        let clip = root.join("fixtures/jfk.wav");
        if !pack.exists() || !clip.exists() {
            eprintln!("skipping: parakeet-tdt pack or jfk clip absent");
            return;
        }
        let samples = read_wav_mono_16k(&clip).expect("wav");
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("validate runtime source");
        let preflight = load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .expect("preflight");
        let output = transcribe_parakeet_tdt_pcm_cached(
            &new_parakeet_tdt_runtime_pool(),
            &samples,
            &preflight,
            true,
            GgmlCpuGraphBackend::Cpu,
            Arc::new(crate::api::backend::TranscriptionControl::new()),
        )
        .expect("transcribe");
        eprintln!("parakeet-tdt hypothesis: {:?}", output.text);
        eprintln!("parakeet-tdt words: {:?}", output.words);
        let lowered = output.text.to_lowercase();
        assert!(
            lowered.contains("ask not what your country can do for you"),
            "unexpected transcript: {:?}",
            output.text
        );
        assert!(!output.words.is_empty(), "native word timestamps expected");
        for pair in output.words.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "word starts must be monotonic: {:?}",
                output.words
            );
        }
    }
}
