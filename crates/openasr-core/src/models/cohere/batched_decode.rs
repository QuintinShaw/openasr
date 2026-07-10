use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use thiserror::Error;

use super::decoder_graph::{
    CohereDecoderGraphDecodeOutput, CohereDecoderGraphRuntime,
    decoder_max_generated_tokens_with_env,
};
use super::decoder_weights::CohereTranscribeDecoderWeights;
use super::encoder_graph::CohereTranscribeEncoderOutput;
use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tokenizer::CohereTranscribeTokenizer;
use crate::PhraseBiasConfig;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, apply_seq2seq_text_postprocess,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, build_seq2seq_greedy_stop_token_ids,
};
#[cfg(test)]
use crate::models::seq2seq_serve_batch::{Envelope, OwnerThreadState};
use crate::models::seq2seq_serve_batch::{
    Seq2SeqServeBatchFamily, Seq2SeqServeRuntime, ServeBatchConfig, ServeBatchEngine,
    serve_batch_engine_for_key,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::serve_batch_env::{
    serve_batch_estimate_seq2seq_slot_bytes, serve_batch_select_and_apply_greedy_step,
};
use crate::{Segment, Transcription};

const COHERE_SERVE_BATCH_MAX_BATCH_LIMIT: usize = 8;

// Owner-fixture tests build a `CohereServeBatchConfig` struct-literal with these
// timings; the live defaults now live on the generic `ServeBatchConfig`.
#[cfg(test)]
const COHERE_SERVE_BATCH_QUEUE_CAPACITY: usize = 4;
#[cfg(test)]
const COHERE_SERVE_BATCH_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);
#[cfg(test)]
const COHERE_SERVE_BATCH_REPLY_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30 * 60);

static COHERE_SERVE_BATCH_ENGINES: OnceLock<
    Mutex<HashMap<CohereServeBatchEngineKey, Arc<ServeBatchEngine<CohereFamily>>>>,
> = OnceLock::new();

/// Field-identical alias onto the generic `ServeBatchConfig`. Preserved so
/// `ggml_executor`'s `CohereServeBatchConfig::from_env()` and the tests'
/// struct-literal construction keep compiling unchanged.
pub(super) type CohereServeBatchConfig = ServeBatchConfig;

/// Lets `CohereServeBatchConfig::from_env()` resolve to the generic
/// `ServeBatchConfig::from_env::<CohereFamily>()` without a turbofish at the
/// call site. Scoped per family module so the method name is unambiguous.
pub(super) trait CohereServeBatchConfigFromEnv: Sized {
    fn from_env() -> Result<Option<Self>, CohereServeBatchError>;
}

impl CohereServeBatchConfigFromEnv for CohereServeBatchConfig {
    fn from_env() -> Result<Option<Self>, CohereServeBatchError> {
        ServeBatchConfig::read_env::<CohereFamily>()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CohereServeBatchJob {
    pub runtime_cache_path: PathBuf,
    pub backend: GgmlCpuGraphBackend,
    pub uses_scheduler: bool,
    pub decoder_weights: Arc<CohereTranscribeDecoderWeights>,
    pub tokenizer: Arc<CohereTranscribeTokenizer>,
    pub metadata: CohereTranscribeExecutionMetadata,
    pub encoder_output: CohereTranscribeEncoderOutput,
    pub decode_config: Seq2SeqGreedyDecodeConfig,
    pub text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    pub word_timestamps: bool,
    pub audio_duration_seconds: f32,
    pub prefer_cpu_backend: bool,
}

#[derive(Debug, Error)]
pub(crate) enum CohereServeBatchError {
    #[error("cohere serve batch env {env} must be an integer in 0..={max}, got '{raw}'")]
    InvalidEnv {
        env: &'static str,
        raw: String,
        max: usize,
    },
    #[error("cohere serve batch requires max batch >= 2 when enabled, got {max_batch}")]
    InvalidEnabledBatch { max_batch: usize },
    #[error("cohere serve batch supports only gpu-class direct ggml backends, got {backend:?}")]
    UnsupportedBackend { backend: GgmlCpuGraphBackend },
    #[error("cohere serve batch engine registry mutex is poisoned")]
    RegistryPoisoned,
    #[error("cohere serve batch owner thread spawn failed: {reason}")]
    ThreadSpawnFailed { reason: String },
    #[error("cohere serve batch queue is full")]
    QueueFull,
    #[error("cohere serve batch owner thread is disconnected")]
    OwnerDisconnected,
    #[error("cohere serve batch owner reply timed out")]
    ReplyTimedOut,
    #[error("cohere serve batch owner failed: {reason}")]
    OwnerFailed { reason: String },
    #[error("cohere serve batch decode failed: {reason}")]
    DecodeFailed { reason: String },
}

impl CohereServeBatchError {
    /// Classifies the transient serve-batch failures that should surface as a
    /// retryable HTTP status. `Some(true)` => queue saturation (429 backpressure);
    /// `Some(false)` => owner gone / GPU step hung (503); `None` => every other
    /// variant keeps its existing (non-retryable) mapping.
    pub(crate) fn unavailable_retryable(&self) -> Option<bool> {
        match self {
            Self::QueueFull => Some(true),
            Self::OwnerDisconnected | Self::ReplyTimedOut => Some(false),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CohereServeBatchEngineKey {
    runtime_cache_path: PathBuf,
    backend: GgmlCpuGraphBackend,
    frame_count: usize,
    hidden_size: usize,
    max_batch: usize,
}

/// The cohere serve-batch ZST family wiring (`Seq2SeqServeBatchFamily`) that
/// drives the generic `OwnerThreadState` + generic `ServeBatchEngine`.
struct CohereFamily;

#[cfg(test)]
type CohereServeBatchEnvelope = Envelope<CohereFamily>;

pub(crate) struct CohereBatchSlot {
    job: CohereServeBatchJob,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    generated_probabilities: Vec<f32>,
    done: bool,
}

pub(super) fn submit_cohere_serve_batch_job(
    config: CohereServeBatchConfig,
    job: CohereServeBatchJob,
) -> Result<CohereDecoderGraphDecodeOutput, CohereServeBatchError> {
    let config = config.validate_for_job::<CohereFamily>(&job)?;
    let key = CohereFamily::engine_key(&job, config.max_batch);
    serve_batch_engine_for_key(&COHERE_SERVE_BATCH_ENGINES, key, config)?.submit(job)
}

fn cohere_serve_batch_vram_slot_bytes(job: &CohereServeBatchJob) -> usize {
    serve_batch_estimate_seq2seq_slot_bytes(
        job.metadata.decoder_layers,
        job.metadata.decoder_max_context,
        job.metadata.decoder_d_model,
        job.encoder_output.frame_count,
        job.encoder_output.hidden_size,
        std::mem::size_of::<u16>(),
        std::mem::size_of::<f32>(),
    )
}

impl Seq2SeqServeRuntime for CohereDecoderGraphRuntime {
    type Job = CohereServeBatchJob;
    type Error = CohereServeBatchError;

    fn build_serial(job: &Self::Job) -> Result<Self, Self::Error> {
        CohereDecoderGraphRuntime::new(
            &job.decoder_weights,
            job.metadata,
            job.encoder_output.frame_count,
            job.encoder_output.hidden_size,
            job.prefer_cpu_backend,
        )
        .map_err(map_decoder_error)
    }

    fn build_batched(job: &Self::Job, n_seq: usize) -> Result<Self, Self::Error> {
        CohereDecoderGraphRuntime::new_with_n_seq(
            &job.decoder_weights,
            job.metadata,
            job.encoder_output.frame_count,
            job.encoder_output.hidden_size,
            job.prefer_cpu_backend,
            n_seq,
        )
        .map_err(map_decoder_error)
    }

    fn populate_cross_attention_cache_serial(
        &mut self,
        job: &Self::Job,
    ) -> Result<(), Self::Error> {
        self.populate_cross_attention_cache(&job.encoder_output)
            .map_err(map_decoder_error)
    }

    fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        job: &Self::Job,
    ) -> Result<(), Self::Error> {
        CohereDecoderGraphRuntime::populate_cross_attention_cache_slot(
            self,
            slot_index,
            &job.encoder_output,
        )
        .map_err(map_decoder_error)
    }

    fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, Self::Error> {
        CohereDecoderGraphRuntime::compute_batched_prefill_logits(self, prompt_tokens)
            .map_err(map_decoder_error)
    }

    fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        totals: &[usize],
    ) -> Result<Vec<f32>, Self::Error> {
        CohereDecoderGraphRuntime::compute_reused_batched_step_logits(
            self, token_ids, positions, totals,
        )
        .map_err(map_decoder_error)
    }
}

impl Seq2SeqServeBatchFamily for CohereFamily {
    type Runtime = CohereDecoderGraphRuntime;
    type Job = CohereServeBatchJob;
    type Slot = CohereBatchSlot;
    type Output = CohereDecoderGraphDecodeOutput;
    type Error = CohereServeBatchError;
    type EngineKey = CohereServeBatchEngineKey;

    const THREAD_NAME_PREFIX: &'static str = "cohere";
    const MAX_BATCH_LIMIT: usize = COHERE_SERVE_BATCH_MAX_BATCH_LIMIT;

    fn engine_key(job: &Self::Job, max_batch: usize) -> Self::EngineKey {
        CohereServeBatchEngineKey {
            runtime_cache_path: job.runtime_cache_path.clone(),
            backend: job.backend,
            frame_count: job.encoder_output.frame_count,
            hidden_size: job.encoder_output.hidden_size,
            max_batch,
        }
    }

    fn engine_key_backend(key: &Self::EngineKey) -> GgmlCpuGraphBackend {
        key.backend
    }

    fn can_batch_with(a: &Self::Job, b: &Self::Job) -> bool {
        a.can_batch_with(b)
    }

    fn vram_slot_bytes(job: &Self::Job) -> usize {
        cohere_serve_batch_vram_slot_bytes(job)
    }

    fn backend(job: &Self::Job) -> GgmlCpuGraphBackend {
        job.backend
    }

    fn uses_scheduler(job: &Self::Job) -> bool {
        job.uses_scheduler
    }

    fn initial_prompt_tokens(job: &Self::Job) -> &[u32] {
        &job.decode_config.initial_prompt_tokens
    }

    fn vocab_size(job: &Self::Job) -> usize {
        job.decode_config.vocab_size
    }

    fn max_generated_tokens(job: &Self::Job) -> usize {
        job.decode_config.max_generated_tokens
    }

    fn decoder_max_context(job: &Self::Job) -> usize {
        job.metadata.decoder_max_context
    }

    fn slot_new(job: Self::Job) -> Result<Self::Slot, Self::Error> {
        CohereBatchSlot::new(job)
    }

    fn slot_job(slot: &Self::Slot) -> &Self::Job {
        &slot.job
    }

    fn slot_generated(slot: &Self::Slot) -> &[u32] {
        &slot.generated_tokens
    }

    fn slot_done(slot: &Self::Slot) -> bool {
        slot.done
    }

    fn slot_select_next_token(slot: &mut Self::Slot, logits: Vec<f32>) -> Result<(), Self::Error> {
        slot.select_next_token_from_logits(logits)
    }

    fn slot_finish(slot: Self::Slot) -> Result<Self::Output, Self::Error> {
        slot.finish()
    }

    fn decode_serial(
        serial_runtime: &mut Option<Self::Runtime>,
        job: Self::Job,
    ) -> Result<Self::Output, Self::Error> {
        if serial_runtime.is_none() {
            *serial_runtime = Some(CohereDecoderGraphRuntime::build_serial(&job)?);
        }
        let runtime =
            serial_runtime
                .as_mut()
                .ok_or_else(|| CohereServeBatchError::OwnerFailed {
                    reason: "cohere serve batch serial runtime cache is unexpectedly empty"
                        .to_string(),
                })?;
        runtime
            .populate_cross_attention_cache(&job.encoder_output)
            .map_err(map_decoder_error)?;
        let mut slot = CohereBatchSlot::new(job)?;
        loop {
            if slot.generated_tokens.len() >= slot.job.decode_config.max_generated_tokens {
                slot.done = true;
            }
            if slot.done {
                break;
            }
            let prefix = slot
                .job
                .decode_config
                .initial_prompt_tokens
                .iter()
                .copied()
                .chain(slot.generated_tokens.iter().copied())
                .collect::<Vec<_>>();
            let logits = runtime
                .compute_step_logits(&prefix)
                .map_err(map_decoder_error)?;
            slot.select_next_token_from_logits(logits)?;
        }
        slot.finish()
    }

    fn decode_failed(reason: String) -> Self::Error {
        CohereServeBatchError::DecodeFailed { reason }
    }

    fn owner_failed(reason: String) -> Self::Error {
        CohereServeBatchError::OwnerFailed { reason }
    }

    fn invalid_env(env: &'static str, raw: String, max: usize) -> Self::Error {
        CohereServeBatchError::InvalidEnv { env, raw, max }
    }

    fn invalid_enabled_batch(max_batch: usize) -> Self::Error {
        CohereServeBatchError::InvalidEnabledBatch { max_batch }
    }

    fn unsupported_backend(backend: GgmlCpuGraphBackend) -> Self::Error {
        CohereServeBatchError::UnsupportedBackend { backend }
    }

    fn registry_poisoned() -> Self::Error {
        CohereServeBatchError::RegistryPoisoned
    }

    fn thread_spawn_failed(reason: String) -> Self::Error {
        CohereServeBatchError::ThreadSpawnFailed { reason }
    }

    fn queue_full() -> Self::Error {
        CohereServeBatchError::QueueFull
    }

    fn owner_disconnected() -> Self::Error {
        CohereServeBatchError::OwnerDisconnected
    }

    fn reply_timed_out() -> Self::Error {
        CohereServeBatchError::ReplyTimedOut
    }
}

impl CohereServeBatchJob {
    fn can_batch_with(&self, other: &Self) -> bool {
        self.decode_config.initial_prompt_tokens == other.decode_config.initial_prompt_tokens
            && self.decode_config.eot_token_id == other.decode_config.eot_token_id
            && self.decode_config.vocab_size == other.decode_config.vocab_size
            && self.metadata.decoder_max_context == other.metadata.decoder_max_context
    }
}

impl CohereBatchSlot {
    fn new(job: CohereServeBatchJob) -> Result<Self, CohereServeBatchError> {
        if job.decode_config.initial_prompt_tokens.is_empty() {
            return Err(CohereServeBatchError::DecodeFailed {
                reason: "cohere serve batch requires at least one prompt token".to_string(),
            });
        }
        if job.decode_config.vocab_size == 0 {
            return Err(CohereServeBatchError::DecodeFailed {
                reason: "cohere serve batch requires vocab_size > 0".to_string(),
            });
        }
        if job.decode_config.max_generated_tokens == 0 {
            return Err(CohereServeBatchError::DecodeFailed {
                reason: "cohere serve batch requires max_generated_tokens > 0".to_string(),
            });
        }
        let stop_token_ids = build_seq2seq_greedy_stop_token_ids(&job.decode_config);
        Ok(Self {
            job,
            stop_token_ids,
            generated_tokens: Vec::new(),
            generated_probabilities: Vec::new(),
            done: false,
        })
    }

    fn select_next_token_from_logits(
        &mut self,
        logits: Vec<f32>,
    ) -> Result<(), CohereServeBatchError> {
        serve_batch_select_and_apply_greedy_step(
            &self.job.decode_config,
            &mut self.generated_tokens,
            &mut self.generated_probabilities,
            &mut self.done,
            self.stop_token_ids.as_slice(),
            logits,
        )
        .map_err(map_greedy_error)
    }

    fn finish(self) -> Result<CohereDecoderGraphDecodeOutput, CohereServeBatchError> {
        let raw_text = self
            .job
            .tokenizer
            .decode_text_token_ids(&self.generated_tokens)
            .map_err(|error| CohereServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        let text = apply_seq2seq_text_postprocess(self.job.text_postprocess_kind, &raw_text)
            .trim()
            .to_string();
        let words = if self.job.word_timestamps {
            seq2seq_word_timestamps_from_generated_tokens(
                &self.generated_tokens,
                &self.generated_probabilities,
                0.0,
                self.job.audio_duration_seconds,
                self.job.text_postprocess_kind,
                &|token_ids| {
                    self.job
                        .tokenizer
                        .decode_text_token_ids(token_ids)
                        .map_err(|error| error.to_string())
                },
            )
            .map_err(|error| CohereServeBatchError::DecodeFailed {
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
                end: self.job.audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words,
            }]
        };
        Ok(CohereDecoderGraphDecodeOutput {
            transcription: Transcription {
                text,
                segments,
                longform: None,
                language: None,
            },
            generated_tokens: self.generated_tokens,
        })
    }
}

pub(super) fn cohere_serve_batch_decode_config(
    prompt_tokens: &[u32],
    metadata: CohereTranscribeExecutionMetadata,
    encoder_frame_count: usize,
    eos_token_id: u32,
    tokenizer: &CohereTranscribeTokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<Seq2SeqGreedyDecodeConfig, CohereServeBatchError> {
    let max_generated_tokens =
        decoder_max_generated_tokens_with_env(prompt_tokens, metadata, encoder_frame_count)
            .map_err(map_decoder_error)?;
    let descriptor =
        crate::models::decode_policy_component_registry::resolve_builtin_decode_policy(
            crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID,
        )
        .map_err(|error| CohereServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })?;
    crate::models::decode_policy_component_registry::build_builtin_seq2seq_decode_policy_config(
        descriptor,
        &crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: prompt_tokens.to_vec(),
            eot_token_id: eos_token_id,
            vocab_size: metadata.vocab_size,
            max_generated_tokens,
        },
        tokenizer,
        phrase_bias,
    )
    .map_err(|error| CohereServeBatchError::DecodeFailed {
        reason: error.to_string(),
    })
}

pub(super) fn cohere_serve_batch_text_postprocess_kind()
-> Result<BuiltinDecodePolicySeq2SeqTextPostprocessKind, CohereServeBatchError> {
    crate::models::decode_policy_component_registry::resolve_builtin_decode_policy(
        crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    )
    .map(|descriptor| descriptor.seq2seq_text_postprocess_kind)
    .map_err(|error| CohereServeBatchError::DecodeFailed {
        reason: error.to_string(),
    })
}

fn map_decoder_error(
    error: super::decoder_graph::CohereDecoderGraphError,
) -> CohereServeBatchError {
    CohereServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

fn map_greedy_error(error: Seq2SeqGreedyDecodeError) -> CohereServeBatchError {
    CohereServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;

    #[test]
    fn serve_batch_error_classifies_transient_failures() {
        assert_eq!(
            CohereServeBatchError::QueueFull.unavailable_retryable(),
            Some(true)
        );
        assert_eq!(
            CohereServeBatchError::OwnerDisconnected.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            CohereServeBatchError::ReplyTimedOut.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            CohereServeBatchError::DecodeFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
        assert_eq!(
            CohereServeBatchError::OwnerFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
    }
    use crate::models::serve_batch_env::OPENASR_SERVE_BATCH_ENV;
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
    };
    use std::ffi::OsString;
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Duration;

    const COHERE_SERVE_BATCH_REAL_PACK_ENV: &str = "OPENASR_COHERE_SERVE_BATCH_REAL_PACK";

    /// Test-only static-batch driver: prefill + greedy step every slot to
    /// completion against a single shared batched runtime. Exercises the cohere
    /// runtime + slot wiring without the continuous-batching control flow.
    fn decode_batched_slots(
        state: &mut OwnerThreadState<CohereFamily>,
        slots: &mut [CohereBatchSlot],
    ) -> Result<(), CohereServeBatchError> {
        let n_seq = slots.len();
        let first_job = &slots
            .first()
            .ok_or_else(|| CohereServeBatchError::OwnerFailed {
                reason: "cohere serve batch received no slots".to_string(),
            })?
            .job;
        let prompt_len = first_job.decode_config.initial_prompt_tokens.len();
        if prompt_len == 0 {
            return Err(CohereServeBatchError::DecodeFailed {
                reason: "cohere serve batch prompt is empty".to_string(),
            });
        }
        let runtime = state.batched_runtime_for(first_job, n_seq)?;
        for (slot_index, slot) in slots.iter().enumerate() {
            runtime
                .populate_cross_attention_cache_slot(slot_index, &slot.job.encoder_output)
                .map_err(map_decoder_error)?;
        }

        let mut logits = Vec::new();
        for position in 0..prompt_len {
            let token_ids = slots
                .iter()
                .map(|slot| slot.job.decode_config.initial_prompt_tokens[position])
                .collect::<Vec<_>>();
            let positions = vec![position; n_seq];
            let totals = vec![position.saturating_add(1); n_seq];
            logits = runtime
                .compute_reused_batched_step_logits(&token_ids, &positions, &totals)
                .map_err(map_decoder_error)?;
        }
        scatter_and_select(slots, &logits)?;

        loop {
            for slot in slots.iter_mut().filter(|slot| !slot.done) {
                if slot.generated_tokens.len() >= slot.job.decode_config.max_generated_tokens {
                    slot.done = true;
                }
            }
            if slots.iter().all(|slot| slot.done) {
                break;
            }
            let mut token_ids = Vec::with_capacity(n_seq);
            let mut positions = Vec::with_capacity(n_seq);
            let mut totals = Vec::with_capacity(n_seq);
            for slot in slots.iter() {
                if slot.done {
                    token_ids.push(0);
                    positions.push(0);
                    totals.push(1);
                    continue;
                }
                let token_id = *slot.generated_tokens.last().ok_or_else(|| {
                    CohereServeBatchError::DecodeFailed {
                        reason: "cohere serve batch generated token history is empty".to_string(),
                    }
                })?;
                let total_tokens = prompt_len
                    .checked_add(slot.generated_tokens.len())
                    .ok_or_else(|| CohereServeBatchError::DecodeFailed {
                        reason: "cohere serve batch token count overflowed".to_string(),
                    })?;
                let position = total_tokens.checked_sub(1).ok_or_else(|| {
                    CohereServeBatchError::DecodeFailed {
                        reason: "cohere serve batch position underflowed".to_string(),
                    }
                })?;
                token_ids.push(token_id);
                positions.push(position);
                totals.push(total_tokens);
            }
            let logits = runtime
                .compute_reused_batched_step_logits(&token_ids, &positions, &totals)
                .map_err(map_decoder_error)?;
            scatter_and_select(slots, &logits)?;
        }
        Ok(())
    }

    fn scatter_and_select(
        slots: &mut [CohereBatchSlot],
        logits: &[f32],
    ) -> Result<(), CohereServeBatchError> {
        let vocab_size = slots
            .first()
            .map(|slot| slot.job.decode_config.vocab_size)
            .unwrap_or(0);
        let expected = vocab_size.checked_mul(slots.len()).ok_or_else(|| {
            CohereServeBatchError::DecodeFailed {
                reason: "cohere serve batch logits length overflowed".to_string(),
            }
        })?;
        if logits.len() != expected {
            return Err(CohereServeBatchError::DecodeFailed {
                reason: format!(
                    "cohere serve batch logits width mismatch: got {}, expected {}",
                    logits.len(),
                    expected
                ),
            });
        }
        for (slot_index, slot) in slots.iter_mut().enumerate() {
            if slot.done {
                continue;
            }
            let start = slot_index.checked_mul(vocab_size).ok_or_else(|| {
                CohereServeBatchError::DecodeFailed {
                    reason: "cohere serve batch logits offset overflowed".to_string(),
                }
            })?;
            let end = start.checked_add(vocab_size).ok_or_else(|| {
                CohereServeBatchError::DecodeFailed {
                    reason: "cohere serve batch logits end overflowed".to_string(),
                }
            })?;
            let slot_logits =
                logits
                    .get(start..end)
                    .ok_or_else(|| CohereServeBatchError::DecodeFailed {
                        reason: "cohere serve batch logits slice out of bounds".to_string(),
                    })?;
            slot.select_next_token_from_logits(slot_logits.to_vec())?;
        }
        Ok(())
    }

    fn with_serve_batch_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
        crate::models::serve_batch_env::with_serve_batch_env_lock(|| {
            let previous = std::env::var_os(OPENASR_SERVE_BATCH_ENV);
            set_serve_batch_env(value.map(OsString::from));
            let result = run();
            set_serve_batch_env(previous);
            result
        })
    }

    fn set_serve_batch_env(value: Option<OsString>) {
        match value {
            Some(value) => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::set_var(OPENASR_SERVE_BATCH_ENV, value);
                }
            }
            None => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::remove_var(OPENASR_SERVE_BATCH_ENV);
                }
            }
        }
    }

    fn read_runtime_source_preflight(runtime_path: &Path) -> GgmlAsrRuntimeSourcePreflight {
        let runtime_source =
            validate_ggml_runtime_source_path(runtime_path).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        }
    }

    fn write_runtime_ready_preflight() -> (tempfile::TempDir, PathBuf, GgmlAsrRuntimeSourcePreflight)
    {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let preflight = read_runtime_source_preflight(&runtime_path);
        (temp, runtime_path, preflight)
    }

    fn sample_encoder_output_with_frame_count(
        metadata: CohereTranscribeExecutionMetadata,
        phase: f32,
        frame_count: usize,
    ) -> CohereTranscribeEncoderOutput {
        let mut rows = Vec::with_capacity(frame_count * metadata.decoder_d_model);
        for frame_idx in 0..frame_count {
            for hidden_idx in 0..metadata.decoder_d_model {
                rows.push(
                    (((frame_idx * metadata.decoder_d_model + hidden_idx) as f32 * 0.03125)
                        + phase)
                        .sin(),
                );
            }
        }
        CohereTranscribeEncoderOutput {
            frame_count,
            hidden_size: metadata.decoder_d_model,
            rows,
        }
    }

    fn assert_logits_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
            let delta = (left_value - right_value).abs();
            assert!(
                delta <= 1.0e-4,
                "logit {index} differs: left={left_value}, right={right_value}, delta={delta}"
            );
        }
    }

    fn assert_argmax_matches(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .inspect(|(_, value)| assert!(value.is_finite()))
                .max_by(|(_, left), (_, right)| {
                    left.partial_cmp(right)
                        .expect("finite logits are comparable")
                })
                .map(|(index, _)| index)
                .expect("logits must be non-empty")
        };
        assert_eq!(argmax(left), argmax(right));
    }

    fn batch_job(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        decoder_weights: Arc<CohereTranscribeDecoderWeights>,
        tokenizer: Arc<CohereTranscribeTokenizer>,
        metadata: CohereTranscribeExecutionMetadata,
        encoder_output: CohereTranscribeEncoderOutput,
        prefer_cpu_backend: bool,
    ) -> CohereServeBatchJob {
        batch_job_with_max_generated_tokens(
            runtime_path,
            backend,
            uses_scheduler,
            decoder_weights,
            tokenizer,
            metadata,
            encoder_output,
            prefer_cpu_backend,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_job_with_max_generated_tokens(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        decoder_weights: Arc<CohereTranscribeDecoderWeights>,
        tokenizer: Arc<CohereTranscribeTokenizer>,
        metadata: CohereTranscribeExecutionMetadata,
        encoder_output: CohereTranscribeEncoderOutput,
        prefer_cpu_backend: bool,
        max_generated_tokens: usize,
    ) -> CohereServeBatchJob {
        CohereServeBatchJob {
            runtime_cache_path: runtime_path.to_path_buf(),
            backend,
            uses_scheduler,
            decoder_weights,
            tokenizer,
            metadata,
            encoder_output,
            decode_config: Seq2SeqGreedyDecodeConfig {
                initial_prompt_tokens: vec![0],
                eot_token_id: 2,
                stop_token_ids: Vec::new(),
                vocab_size: metadata.vocab_size,
                max_generated_tokens,
                suppress_first_step_token_ids: Vec::new(),
                suppress_token_ids: Vec::new(),
                phrase_biases: Vec::new(),
            },
            text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            word_timestamps: false,
            audio_duration_seconds: 1.0,
            prefer_cpu_backend,
        }
    }

    fn run_static_batch_fixture_with_preflight(
        runtime_path: &Path,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        prefer_cpu_backend: bool,
        encoder_frame_count: usize,
        strict_logit_parity: bool,
    ) {
        let runtime_config =
            super::super::graph_config::cohere_decoder_graph_config(prefer_cpu_backend);
        assert!(
            prefer_cpu_backend || !runtime_config.use_scheduler,
            "cohere static batch fixture validates the direct graph lane, got scheduler-backed {:?}",
            runtime_config.backend
        );
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let reader = build_runtime_tensor_reader_from_preflight(preflight).expect("reader");
        let decoder_weights = Arc::new(
            super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                &reader, metadata,
            )
            .expect("decoder weights"),
        );
        let tokenizer = Arc::new(
            CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata).expect("tokenizer"),
        );
        let encoder_output_0 =
            sample_encoder_output_with_frame_count(metadata, 0.0, encoder_frame_count);
        let encoder_output_1 =
            sample_encoder_output_with_frame_count(metadata, 0.25, encoder_frame_count);

        let mut serial_runtime_0 = CohereDecoderGraphRuntime::new(
            &decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            encoder_output_0.hidden_size,
            prefer_cpu_backend,
        )
        .expect("serial runtime 0");
        serial_runtime_0
            .populate_cross_attention_cache(&encoder_output_0)
            .expect("serial cross cache 0");
        let serial_logits_0 = serial_runtime_0
            .compute_step_logits(&[0])
            .expect("serial logits 0");

        let mut serial_runtime_1 = CohereDecoderGraphRuntime::new(
            &decoder_weights,
            metadata,
            encoder_output_1.frame_count,
            encoder_output_1.hidden_size,
            prefer_cpu_backend,
        )
        .expect("serial runtime 1");
        serial_runtime_1
            .populate_cross_attention_cache(&encoder_output_1)
            .expect("serial cross cache 1");
        let serial_logits_1 = serial_runtime_1
            .compute_step_logits(&[0])
            .expect("serial logits 1");

        let mut batched_runtime = CohereDecoderGraphRuntime::new_with_n_seq(
            &decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            encoder_output_0.hidden_size,
            prefer_cpu_backend,
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
            .compute_reused_batched_step_logits(&[0, 0], &[0, 0], &[1, 1])
            .expect("batched logits");
        let batched_logits_0 = &batched_logits[0..metadata.vocab_size];
        let batched_logits_1 = &batched_logits[metadata.vocab_size..];
        if strict_logit_parity {
            assert_logits_close(batched_logits_0, &serial_logits_0);
            assert_logits_close(batched_logits_1, &serial_logits_1);
        } else {
            assert_argmax_matches(batched_logits_0, &serial_logits_0);
            assert_argmax_matches(batched_logits_1, &serial_logits_1);
        }

        let mut slots = vec![
            CohereBatchSlot::new(batch_job(
                runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&decoder_weights),
                tokenizer.clone(),
                metadata,
                encoder_output_0.clone(),
                prefer_cpu_backend,
            ))
            .expect("slot 0"),
            CohereBatchSlot::new(batch_job(
                runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&decoder_weights),
                tokenizer,
                metadata,
                encoder_output_1,
                prefer_cpu_backend,
            ))
            .expect("slot 1"),
        ];
        let mut state = OwnerThreadState::<CohereFamily>::new();

        decode_batched_slots(&mut state, &mut slots).expect("batched decode");

        let outputs = slots
            .into_iter()
            .map(CohereBatchSlot::finish)
            .collect::<Result<Vec<_>, _>>()
            .expect("finish slots");
        assert_eq!(outputs.len(), 2);
        assert!(
            outputs
                .iter()
                .all(|output| output.generated_tokens.len() <= 1)
        );
    }

    fn run_refill_fixture_with_preflight(
        runtime_path: &Path,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        prefer_cpu_backend: bool,
        encoder_frame_count: usize,
    ) {
        let runtime_config =
            super::super::graph_config::cohere_decoder_graph_config(prefer_cpu_backend);
        assert!(
            prefer_cpu_backend || !runtime_config.use_scheduler,
            "cohere refill fixture validates the direct graph lane, got scheduler-backed {:?}",
            runtime_config.backend
        );
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let reader = build_runtime_tensor_reader_from_preflight(preflight).expect("tensor reader");
        let decoder_weights = Arc::new(
            super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                &reader, metadata,
            )
            .expect("decoder weights"),
        );
        let tokenizer = Arc::new(
            CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata).expect("tokenizer"),
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = batch_job_with_max_generated_tokens(
                runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&decoder_weights),
                tokenizer.clone(),
                metadata,
                sample_encoder_output_with_frame_count(
                    metadata,
                    encoder_phase,
                    encoder_frame_count,
                ),
                prefer_cpu_backend,
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            (CohereServeBatchEnvelope { job, reply }, reply_rx)
        };

        let (initial_fast, initial_fast_rx) = envelope(0.0, 1);
        let (initial_long, initial_long_rx) = envelope(0.25, 3);
        let (queued_refill, queued_refill_rx) = envelope(0.5, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(1);
        queued_tx.send(queued_refill).expect("queue refill job");

        let mut state = OwnerThreadState::<CohereFamily>::new();
        let config = CohereServeBatchConfig {
            max_batch: 2,
            queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: Duration::ZERO,
            send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: false,
        };
        let deferred = state.run_batch(
            vec![initial_fast, initial_long],
            &queued_rx,
            config.max_batch,
            config.trace_batches,
        );
        assert!(deferred.is_empty());

        let fast = initial_fast_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("fast reply")
            .expect("fast output");
        let long = initial_long_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("long reply")
            .expect("long output");
        let refill = queued_refill_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("refill reply")
            .expect("refill output");

        assert!(fast.generated_tokens.len() <= 1);
        assert!(long.generated_tokens.len() <= 3);
        assert!(refill.generated_tokens.len() <= 1);
    }

    fn run_tiny_static_batch_fixture(prefer_cpu_backend: bool) {
        let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
        run_static_batch_fixture_with_preflight(
            &runtime_path,
            &preflight,
            prefer_cpu_backend,
            4,
            true,
        );
    }

    #[test]
    fn cohere_owner_thread_refills_free_static_slot_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
            let runtime_config = super::super::graph_config::cohere_decoder_graph_config(true);
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader =
                build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
            let decoder_weights = Arc::new(
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights"),
            );
            let tokenizer = Arc::new(
                CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata)
                    .expect("tokenizer"),
            );

            let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
                let job = batch_job_with_max_generated_tokens(
                    &runtime_path,
                    runtime_config.backend,
                    runtime_config.use_scheduler,
                    Arc::clone(&decoder_weights),
                    tokenizer.clone(),
                    metadata,
                    sample_encoder_output_with_frame_count(metadata, encoder_phase, 4),
                    true,
                    max_generated_tokens,
                );
                let (reply, reply_rx) = mpsc::channel();
                (CohereServeBatchEnvelope { job, reply }, reply_rx)
            };

            let (initial_fast, initial_fast_rx) = envelope(0.0, 1);
            let (initial_long, initial_long_rx) = envelope(0.25, 3);
            let (queued_refill, queued_refill_rx) = envelope(0.5, 1);
            let (queued_tx, queued_rx) = mpsc::sync_channel(1);
            queued_tx.send(queued_refill).expect("queue refill job");

            let mut state = OwnerThreadState::<CohereFamily>::new();
            let config = CohereServeBatchConfig {
                max_batch: 2,
                queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
                collect_window: Duration::ZERO,
                send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
                reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
                trace_batches: false,
            };
            let deferred = state.run_batch(
                vec![initial_fast, initial_long],
                &queued_rx,
                config.max_batch,
                config.trace_batches,
            );
            assert!(deferred.is_empty());

            let fast = initial_fast_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast reply")
                .expect("fast output");
            let long = initial_long_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long reply")
                .expect("long output");
            let refill = queued_refill_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill reply")
                .expect("refill output");

            assert!(fast.generated_tokens.len() <= 1);
            assert!(long.generated_tokens.len() <= 3);
            assert!(refill.generated_tokens.len() <= 1);
        });
    }

    #[test]
    fn cohere_owner_thread_refills_padded_bucket_slot_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
            let runtime_config = super::super::graph_config::cohere_decoder_graph_config(true);
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader =
                build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
            let decoder_weights = Arc::new(
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights"),
            );
            let tokenizer = Arc::new(
                CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata)
                    .expect("tokenizer"),
            );

            let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
                let job = batch_job_with_max_generated_tokens(
                    &runtime_path,
                    runtime_config.backend,
                    runtime_config.use_scheduler,
                    Arc::clone(&decoder_weights),
                    tokenizer.clone(),
                    metadata,
                    sample_encoder_output_with_frame_count(metadata, encoder_phase, 4),
                    true,
                    max_generated_tokens,
                );
                let (reply, reply_rx) = mpsc::channel();
                (CohereServeBatchEnvelope { job, reply }, reply_rx)
            };

            let (initial_fast, initial_fast_rx) = envelope(0.0, 1);
            let (initial_long_a, initial_long_a_rx) = envelope(0.25, 3);
            let (initial_long_b, initial_long_b_rx) = envelope(0.5, 3);
            let (queued_refill, queued_refill_rx) = envelope(0.75, 1);
            let (queued_tx, queued_rx) = mpsc::sync_channel(1);
            queued_tx.send(queued_refill).expect("queue refill job");

            let mut state = OwnerThreadState::<CohereFamily>::new();
            let config = CohereServeBatchConfig {
                max_batch: 4,
                queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
                collect_window: Duration::ZERO,
                send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
                reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
                trace_batches: false,
            };
            let deferred = state.run_batch(
                vec![initial_fast, initial_long_a, initial_long_b],
                &queued_rx,
                config.max_batch,
                config.trace_batches,
            );
            assert!(deferred.is_empty());

            let fast = initial_fast_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast reply")
                .expect("fast output");
            let long_a = initial_long_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long a reply")
                .expect("long a output");
            let long_b = initial_long_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long b reply")
                .expect("long b output");
            let refill = queued_refill_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill reply")
                .expect("refill output");

            assert!(fast.generated_tokens.len() <= 1);
            assert!(long_a.generated_tokens.len() <= 3);
            assert!(long_b.generated_tokens.len() <= 3);
            assert!(refill.generated_tokens.len() <= 1);
            assert!(state.batched_runtimes.contains_key(&4));
        });
    }

    #[test]
    fn cohere_owner_thread_coalesces_multiple_refill_slots_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
            let runtime_config = super::super::graph_config::cohere_decoder_graph_config(true);
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader =
                build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
            let decoder_weights = Arc::new(
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights"),
            );
            let tokenizer = Arc::new(
                CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata)
                    .expect("tokenizer"),
            );

            let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
                let job = batch_job_with_max_generated_tokens(
                    &runtime_path,
                    runtime_config.backend,
                    runtime_config.use_scheduler,
                    Arc::clone(&decoder_weights),
                    tokenizer.clone(),
                    metadata,
                    sample_encoder_output_with_frame_count(metadata, encoder_phase, 4),
                    true,
                    max_generated_tokens,
                );
                let (reply, reply_rx) = mpsc::channel();
                (CohereServeBatchEnvelope { job, reply }, reply_rx)
            };

            let (initial_fast_a, initial_fast_a_rx) = envelope(0.0, 1);
            let (initial_fast_b, initial_fast_b_rx) = envelope(0.25, 1);
            let (initial_long, initial_long_rx) = envelope(0.5, 3);
            let (queued_refill_a, queued_refill_a_rx) = envelope(0.75, 1);
            let (queued_refill_b, queued_refill_b_rx) = envelope(1.0, 1);
            let (queued_tx, queued_rx) = mpsc::sync_channel(2);
            queued_tx.send(queued_refill_a).expect("queue refill job a");
            queued_tx.send(queued_refill_b).expect("queue refill job b");

            let mut state = OwnerThreadState::<CohereFamily>::new();
            let config = CohereServeBatchConfig {
                max_batch: 4,
                queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
                collect_window: Duration::ZERO,
                send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
                reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
                trace_batches: false,
            };
            let deferred = state.run_batch(
                vec![initial_fast_a, initial_fast_b, initial_long],
                &queued_rx,
                config.max_batch,
                config.trace_batches,
            );
            assert!(deferred.is_empty());

            let fast_a = initial_fast_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast a reply")
                .expect("fast a output");
            let fast_b = initial_fast_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast b reply")
                .expect("fast b output");
            let long = initial_long_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long reply")
                .expect("long output");
            let refill_a = queued_refill_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill a reply")
                .expect("refill a output");
            let refill_b = queued_refill_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill b reply")
                .expect("refill b output");

            assert!(fast_a.generated_tokens.len() <= 1);
            assert!(fast_b.generated_tokens.len() <= 1);
            assert!(long.generated_tokens.len() <= 3);
            assert!(refill_a.generated_tokens.len() <= 1);
            assert!(refill_b.generated_tokens.len() <= 1);
            assert!(state.batched_runtimes.contains_key(&4));
        });
    }

    #[test]
    fn cohere_owner_thread_rebuckets_full_static_bucket_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
            let runtime_config = super::super::graph_config::cohere_decoder_graph_config(true);
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader =
                build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
            let decoder_weights = Arc::new(
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights"),
            );
            let tokenizer = Arc::new(
                CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata)
                    .expect("tokenizer"),
            );

            let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
                let job = batch_job_with_max_generated_tokens(
                    &runtime_path,
                    runtime_config.backend,
                    runtime_config.use_scheduler,
                    Arc::clone(&decoder_weights),
                    tokenizer.clone(),
                    metadata,
                    sample_encoder_output_with_frame_count(metadata, encoder_phase, 4),
                    true,
                    max_generated_tokens,
                );
                let (reply, reply_rx) = mpsc::channel();
                (CohereServeBatchEnvelope { job, reply }, reply_rx)
            };

            let (initial_long_a, initial_long_a_rx) = envelope(0.0, 3);
            let (initial_long_b, initial_long_b_rx) = envelope(0.25, 3);
            let (queued_refill_a, queued_refill_a_rx) = envelope(0.5, 1);
            let (queued_refill_b, queued_refill_b_rx) = envelope(0.75, 1);
            let (queued_tx, queued_rx) = mpsc::sync_channel(2);
            queued_tx.send(queued_refill_a).expect("queue refill a");
            queued_tx.send(queued_refill_b).expect("queue refill b");

            let mut state = OwnerThreadState::<CohereFamily>::new();
            let config = CohereServeBatchConfig {
                max_batch: 4,
                queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
                collect_window: Duration::ZERO,
                send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
                reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
                trace_batches: false,
            };
            let deferred = state.run_batch(
                vec![initial_long_a, initial_long_b],
                &queued_rx,
                config.max_batch,
                config.trace_batches,
            );
            assert!(deferred.is_empty());

            let long_a = initial_long_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long a reply")
                .expect("long a output");
            let long_b = initial_long_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long b reply")
                .expect("long b output");
            let refill_a = queued_refill_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill a reply")
                .expect("refill a output");
            let refill_b = queued_refill_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("refill b reply")
                .expect("refill b output");

            assert!(long_a.generated_tokens.len() <= 3);
            assert!(long_b.generated_tokens.len() <= 3);
            assert!(refill_a.generated_tokens.len() <= 1);
            assert!(refill_b.generated_tokens.len() <= 1);
            assert!(state.batched_runtimes.contains_key(&2));
            assert!(state.batched_runtimes.contains_key(&4));
        });
    }

    #[test]
    fn cohere_owner_thread_shrinks_tail_static_bucket_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            let (_temp, runtime_path, preflight) = write_runtime_ready_preflight();
            let runtime_config = super::super::graph_config::cohere_decoder_graph_config(true);
            let metadata =
                super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
                    &preflight.metadata,
                )
                .expect("parse metadata");
            let reader =
                build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
            let decoder_weights = Arc::new(
                super::super::decoder_weights::load_cohere_transcribe_decoder_weights_from_reader(
                    &reader, metadata,
                )
                .expect("decoder weights"),
            );
            let tokenizer = Arc::new(
                CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata)
                    .expect("tokenizer"),
            );

            let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
                let job = batch_job_with_max_generated_tokens(
                    &runtime_path,
                    runtime_config.backend,
                    runtime_config.use_scheduler,
                    Arc::clone(&decoder_weights),
                    tokenizer.clone(),
                    metadata,
                    sample_encoder_output_with_frame_count(metadata, encoder_phase, 4),
                    true,
                    max_generated_tokens,
                );
                let (reply, reply_rx) = mpsc::channel();
                (CohereServeBatchEnvelope { job, reply }, reply_rx)
            };

            let (initial_fast_a, initial_fast_a_rx) = envelope(0.0, 1);
            let (initial_fast_b, initial_fast_b_rx) = envelope(0.25, 1);
            let (initial_long, initial_long_rx) = envelope(0.5, 3);
            let (_queued_tx, queued_rx) = mpsc::sync_channel(1);

            let mut state = OwnerThreadState::<CohereFamily>::new();
            let config = CohereServeBatchConfig {
                max_batch: 4,
                queue_capacity: COHERE_SERVE_BATCH_QUEUE_CAPACITY,
                collect_window: Duration::ZERO,
                send_timeout: COHERE_SERVE_BATCH_SEND_TIMEOUT,
                reply_timeout: COHERE_SERVE_BATCH_REPLY_TIMEOUT,
                trace_batches: false,
            };
            let deferred = state.run_batch(
                vec![initial_fast_a, initial_fast_b, initial_long],
                &queued_rx,
                config.max_batch,
                config.trace_batches,
            );
            assert!(deferred.is_empty());

            let fast_a = initial_fast_a_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast a reply")
                .expect("fast a output");
            let fast_b = initial_fast_b_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("fast b reply")
                .expect("fast b output");
            let long = initial_long_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("long reply")
                .expect("long output");

            assert!(fast_a.generated_tokens.len() <= 1);
            assert!(fast_b.generated_tokens.len() <= 1);
            assert!(long.generated_tokens.len() <= 3);
            assert!(state.batched_runtimes.contains_key(&4));
            assert!(state.batched_runtimes.contains_key(&2));
        });
    }

    #[test]
    fn cohere_serve_batch_env_defaults_off() {
        with_serve_batch_env(None, || {
            assert!(CohereServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn cohere_serve_batch_env_one_keeps_default_path() {
        with_serve_batch_env(Some("1"), || {
            assert!(CohereServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn cohere_serve_batch_env_accepts_two_to_eight() {
        with_serve_batch_env(Some("4"), || {
            let config = CohereServeBatchConfig::from_env()
                .unwrap()
                .expect("enabled");
            assert_eq!(config.max_batch, 4);
        });
    }

    #[test]
    fn cohere_serve_batch_env_rejects_out_of_range() {
        with_serve_batch_env(Some("9"), || {
            assert!(matches!(
                CohereServeBatchConfig::from_env(),
                Err(CohereServeBatchError::InvalidEnv { .. })
            ));
        });
    }

    #[test]
    fn cohere_owner_thread_decodes_static_cpu_batch() {
        with_forced_cpu_backend_for_test(|| {
            run_tiny_static_batch_fixture(true);
        });
    }

    #[test]
    #[ignore = "manual real-pack GPU harness: set OPENASR_COHERE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn cohere_owner_thread_decodes_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(COHERE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{COHERE_SERVE_BATCH_REAL_PACK_ENV} must point to a cohere .oasr model pack")
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        run_static_batch_fixture_with_preflight(&runtime_path, &preflight, false, 32, false);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_COHERE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn cohere_owner_thread_refills_free_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(COHERE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{COHERE_SERVE_BATCH_REAL_PACK_ENV} must point to a cohere .oasr model pack")
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        run_refill_fixture_with_preflight(&runtime_path, &preflight, false, 32);
    }
}
