use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use thiserror::Error;

use super::decoder_graph::{MoonshineDecodeOutput, MoonshineDecoderGraphRuntime};
use super::encoder_graph::MoonshineEncoderOutput;
use super::prepared_runtime::MoonshinePreparedRuntime;
use super::runtime_contract::MoonshineExecutionMetadata;
use super::tokenizer::MoonshineTokenizer;
use crate::PhraseBiasConfig;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::BuiltinDecodePolicySeq2SeqTextPostprocessKind;
use crate::models::phrase_bias_decode::build_token_phrase_biases;
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
    ServeBatchStepOutcome, serve_batch_estimate_seq2seq_slot_bytes, serve_batch_select_greedy_step,
};
use crate::{Segment, Transcription};

const MOONSHINE_SERVE_BATCH_MAX_BATCH_LIMIT: usize = 8;

// Owner-fixture tests build a `MoonshineServeBatchConfig` struct-literal with
// these timings; the live defaults now live on the generic `ServeBatchConfig`.
#[cfg(test)]
const MOONSHINE_SERVE_BATCH_QUEUE_CAPACITY: usize = 4;
#[cfg(test)]
const MOONSHINE_SERVE_BATCH_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);
#[cfg(test)]
const MOONSHINE_SERVE_BATCH_REPLY_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30 * 60);

static MOONSHINE_SERVE_BATCH_ENGINES: OnceLock<
    Mutex<HashMap<MoonshineServeBatchEngineKey, Arc<ServeBatchEngine<MoonshineFamily>>>>,
> = OnceLock::new();

/// Field-identical alias onto the generic `ServeBatchConfig`. Preserved so
/// `ggml_executor`'s `MoonshineServeBatchConfig::from_env()` and the tests'
/// struct-literal construction keep compiling unchanged.
pub(super) type MoonshineServeBatchConfig = ServeBatchConfig;

/// Lets `MoonshineServeBatchConfig::from_env()` resolve to the generic
/// `ServeBatchConfig::from_env::<MoonshineFamily>()` without a turbofish at the
/// call site. Scoped per family module so the method name is unambiguous.
pub(super) trait MoonshineServeBatchConfigFromEnv: Sized {
    fn from_env() -> Result<Option<Self>, MoonshineServeBatchError>;
}

impl MoonshineServeBatchConfigFromEnv for MoonshineServeBatchConfig {
    fn from_env() -> Result<Option<Self>, MoonshineServeBatchError> {
        ServeBatchConfig::read_env::<MoonshineFamily>()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MoonshineServeBatchJob {
    pub runtime_cache_path: PathBuf,
    pub backend: GgmlCpuGraphBackend,
    pub uses_scheduler: bool,
    pub prepared_runtime: Arc<MoonshinePreparedRuntime>,
    pub encoder_output: MoonshineEncoderOutput,
    pub decode_config: Seq2SeqGreedyDecodeConfig,
    pub word_timestamps: bool,
    pub audio_duration_seconds: f32,
}

#[derive(Debug, Error)]
pub(crate) enum MoonshineServeBatchError {
    #[error("moonshine serve batch env {env} must be an integer in 0..={max}, got '{raw}'")]
    InvalidEnv {
        env: &'static str,
        raw: String,
        max: usize,
    },
    #[error("moonshine serve batch requires max batch >= 2 when enabled, got {max_batch}")]
    InvalidEnabledBatch { max_batch: usize },
    #[error("moonshine serve batch supports only gpu-class direct ggml backends, got {backend:?}")]
    UnsupportedBackend { backend: GgmlCpuGraphBackend },
    #[error("moonshine serve batch engine registry mutex is poisoned")]
    RegistryPoisoned,
    #[error("moonshine serve batch owner thread spawn failed: {reason}")]
    ThreadSpawnFailed { reason: String },
    #[error("moonshine serve batch queue is full")]
    QueueFull,
    #[error("moonshine serve batch owner thread is disconnected")]
    OwnerDisconnected,
    #[error("moonshine serve batch owner reply timed out")]
    ReplyTimedOut,
    #[error("moonshine serve batch owner failed: {reason}")]
    OwnerFailed { reason: String },
    #[error("moonshine serve batch decode failed: {reason}")]
    DecodeFailed { reason: String },
}

impl MoonshineServeBatchError {
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
pub(crate) struct MoonshineServeBatchEngineKey {
    runtime_cache_path: PathBuf,
    backend: GgmlCpuGraphBackend,
    frame_count: usize,
    hidden_size: usize,
    max_batch: usize,
}

/// The moonshine serve-batch ZST family wiring (`Seq2SeqServeBatchFamily`) that
/// drives the generic `OwnerThreadState` + generic `ServeBatchEngine`.
struct MoonshineFamily;

#[cfg(test)]
type MoonshineServeBatchEnvelope = Envelope<MoonshineFamily>;

pub(crate) struct MoonshineBatchSlot {
    job: MoonshineServeBatchJob,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    generated_probabilities: Vec<f32>,
    done: bool,
}

pub(super) fn submit_moonshine_serve_batch_job(
    config: MoonshineServeBatchConfig,
    job: MoonshineServeBatchJob,
) -> Result<MoonshineDecodeOutput, MoonshineServeBatchError> {
    let config = config.validate_for_job::<MoonshineFamily>(&job)?;
    let key = MoonshineFamily::engine_key(&job, config.max_batch);
    serve_batch_engine_for_key(&MOONSHINE_SERVE_BATCH_ENGINES, key, config)?.submit(job)
}

fn moonshine_serve_batch_vram_slot_bytes(job: &MoonshineServeBatchJob) -> usize {
    serve_batch_estimate_seq2seq_slot_bytes(
        job.prepared_runtime.metadata.decoder_layers,
        job.prepared_runtime.metadata.decoder_max_context,
        job.prepared_runtime.metadata.d_model,
        job.encoder_output.frame_count,
        job.encoder_output.hidden_size,
        std::mem::size_of::<u16>(),
        std::mem::size_of::<u16>(),
    )
}

impl Seq2SeqServeRuntime for MoonshineDecoderGraphRuntime {
    type Job = MoonshineServeBatchJob;
    type Error = MoonshineServeBatchError;

    fn build_serial(job: &Self::Job) -> Result<Self, Self::Error> {
        // Serve-batch is never used with a dynamic adapter (the executor
        // forces the direct decode path when OPENASR_ADAPTER is active), so
        // worker runtimes are always adapter-free.
        MoonshineDecoderGraphRuntime::new(
            &job.prepared_runtime.decoder_weights,
            job.prepared_runtime.metadata,
            job.encoder_output.frame_count,
            false,
            Some(job.runtime_cache_path.as_path()),
            None,
        )
        .map_err(map_decoder_error)
    }

    fn build_batched(job: &Self::Job, n_seq: usize) -> Result<Self, Self::Error> {
        MoonshineDecoderGraphRuntime::new_with_n_seq(
            &job.prepared_runtime.decoder_weights,
            job.prepared_runtime.metadata,
            job.encoder_output.frame_count,
            false,
            Some(job.runtime_cache_path.as_path()),
            n_seq,
            None,
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
        MoonshineDecoderGraphRuntime::populate_cross_attention_cache_slot(
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
        MoonshineDecoderGraphRuntime::compute_batched_prefill_logits(self, prompt_tokens)
            .map_err(map_decoder_error)
    }

    fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        totals: &[usize],
    ) -> Result<Vec<f32>, Self::Error> {
        MoonshineDecoderGraphRuntime::compute_reused_batched_step_logits(
            self, token_ids, positions, totals,
        )
        .map_err(map_decoder_error)
    }
}

impl Seq2SeqServeBatchFamily for MoonshineFamily {
    type Runtime = MoonshineDecoderGraphRuntime;
    type Job = MoonshineServeBatchJob;
    type Slot = MoonshineBatchSlot;
    type Output = MoonshineDecodeOutput;
    type Error = MoonshineServeBatchError;
    type EngineKey = MoonshineServeBatchEngineKey;

    const THREAD_NAME_PREFIX: &'static str = "moonshine";
    const MAX_BATCH_LIMIT: usize = MOONSHINE_SERVE_BATCH_MAX_BATCH_LIMIT;

    fn engine_key(job: &Self::Job, max_batch: usize) -> Self::EngineKey {
        MoonshineServeBatchEngineKey {
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
        moonshine_serve_batch_vram_slot_bytes(job)
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
        job.prepared_runtime.metadata.decoder_max_context
    }

    fn slot_new(job: Self::Job) -> Result<Self::Slot, Self::Error> {
        MoonshineBatchSlot::new(job)
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
            *serial_runtime = Some(MoonshineDecoderGraphRuntime::build_serial(&job)?);
        }
        let runtime =
            serial_runtime
                .as_mut()
                .ok_or_else(|| MoonshineServeBatchError::OwnerFailed {
                    reason: "moonshine serve batch serial runtime cache is unexpectedly empty"
                        .to_string(),
                })?;
        runtime
            .populate_cross_attention_cache(&job.encoder_output)
            .map_err(map_decoder_error)?;
        let mut slot = MoonshineBatchSlot::new(job)?;
        loop {
            if slot.generated_tokens.len() >= slot.job.decode_config.max_generated_tokens {
                slot.done = true;
            }
            if slot.done {
                break;
            }
            let token_id = slot
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(slot.job.decode_config.initial_prompt_tokens[0]);
            let position = slot.generated_tokens.len();
            let logits = runtime
                .compute_incremental_step_logits(token_id, position)
                .map_err(map_decoder_error)?;
            slot.select_next_token_from_logits(logits)?;
        }
        slot.finish()
    }

    fn decode_failed(reason: String) -> Self::Error {
        MoonshineServeBatchError::DecodeFailed { reason }
    }

    fn owner_failed(reason: String) -> Self::Error {
        MoonshineServeBatchError::OwnerFailed { reason }
    }

    fn invalid_env(env: &'static str, raw: String, max: usize) -> Self::Error {
        MoonshineServeBatchError::InvalidEnv { env, raw, max }
    }

    fn invalid_enabled_batch(max_batch: usize) -> Self::Error {
        MoonshineServeBatchError::InvalidEnabledBatch { max_batch }
    }

    fn unsupported_backend(backend: GgmlCpuGraphBackend) -> Self::Error {
        MoonshineServeBatchError::UnsupportedBackend { backend }
    }

    fn registry_poisoned() -> Self::Error {
        MoonshineServeBatchError::RegistryPoisoned
    }

    fn thread_spawn_failed(reason: String) -> Self::Error {
        MoonshineServeBatchError::ThreadSpawnFailed { reason }
    }

    fn queue_full() -> Self::Error {
        MoonshineServeBatchError::QueueFull
    }

    fn owner_disconnected() -> Self::Error {
        MoonshineServeBatchError::OwnerDisconnected
    }

    fn reply_timed_out() -> Self::Error {
        MoonshineServeBatchError::ReplyTimedOut
    }
}

impl MoonshineServeBatchJob {
    fn can_batch_with(&self, other: &Self) -> bool {
        self.decode_config.initial_prompt_tokens == other.decode_config.initial_prompt_tokens
            && self.decode_config.eot_token_id == other.decode_config.eot_token_id
            && self.decode_config.vocab_size == other.decode_config.vocab_size
            && self.prepared_runtime.metadata.decoder_max_context
                == other.prepared_runtime.metadata.decoder_max_context
    }
}

impl MoonshineBatchSlot {
    fn new(job: MoonshineServeBatchJob) -> Result<Self, MoonshineServeBatchError> {
        if job.decode_config.initial_prompt_tokens.is_empty() {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch requires at least one prompt token".to_string(),
            });
        }
        if job.decode_config.vocab_size == 0 {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch requires vocab_size > 0".to_string(),
            });
        }
        if job.decode_config.max_generated_tokens == 0 {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch requires max_generated_tokens > 0".to_string(),
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
    ) -> Result<(), MoonshineServeBatchError> {
        match serve_batch_select_greedy_step(
            &self.job.decode_config,
            &self.generated_tokens,
            self.stop_token_ids.as_slice(),
            logits,
        )
        .map_err(map_greedy_error)?
        {
            ServeBatchStepOutcome::ReachedEot => self.done = true,
            ServeBatchStepOutcome::Token {
                token_id,
                probability,
            } => {
                self.generated_tokens.push(token_id);
                self.generated_probabilities.push(probability);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<MoonshineDecodeOutput, MoonshineServeBatchError> {
        let tokenizer = &self.job.prepared_runtime.tokenizer;
        let text = tokenizer
            .decode_text_token_ids(&self.generated_tokens)
            .map_err(|error| MoonshineServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?
            .trim()
            .to_string();
        let words = if self.job.word_timestamps {
            seq2seq_word_timestamps_from_generated_tokens(
                &self.generated_tokens,
                &self.generated_probabilities,
                0.0,
                self.job.audio_duration_seconds,
                BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
                &|token_ids| {
                    tokenizer
                        .decode_text_token_ids(token_ids)
                        .map_err(|error| error.to_string())
                },
            )
            .map_err(|error| MoonshineServeBatchError::DecodeFailed {
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
        Ok(MoonshineDecodeOutput {
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

pub(super) fn moonshine_serve_batch_decode_config(
    metadata: MoonshineExecutionMetadata,
    tokenizer: &MoonshineTokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<Seq2SeqGreedyDecodeConfig, MoonshineServeBatchError> {
    let initial_prompt_tokens = vec![metadata.bos_token_id];
    let max_generated_tokens = metadata
        .decoder_max_context
        .saturating_sub(initial_prompt_tokens.len())
        .max(1);
    Ok(Seq2SeqGreedyDecodeConfig {
        initial_prompt_tokens,
        eot_token_id: metadata.eos_token_id,
        stop_token_ids: Vec::new(),
        vocab_size: metadata.vocab_size,
        max_generated_tokens,
        suppress_first_step_token_ids: Vec::new(),
        suppress_token_ids: Vec::new(),
        phrase_biases: build_token_phrase_biases(phrase_bias, tokenizer).map_err(|error| {
            MoonshineServeBatchError::DecodeFailed {
                reason: format!("moonshine phrase-bias tokenization failed: {error}"),
            }
        })?,
    })
}

fn map_decoder_error(
    error: super::decoder_graph::MoonshineDecoderGraphError,
) -> MoonshineServeBatchError {
    MoonshineServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

fn map_greedy_error(error: Seq2SeqGreedyDecodeError) -> MoonshineServeBatchError {
    MoonshineServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::serve_batch_env::OPENASR_SERVE_BATCH_ENV;
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
    };
    use std::ffi::OsString;
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Duration;

    const MOONSHINE_SERVE_BATCH_REAL_PACK_ENV: &str = "OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK";

    #[test]
    fn serve_batch_error_classifies_transient_failures() {
        assert_eq!(
            MoonshineServeBatchError::QueueFull.unavailable_retryable(),
            Some(true)
        );
        assert_eq!(
            MoonshineServeBatchError::OwnerDisconnected.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            MoonshineServeBatchError::ReplyTimedOut.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            MoonshineServeBatchError::DecodeFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
        assert_eq!(
            MoonshineServeBatchError::OwnerFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
    }

    /// Test-only static-batch driver: prefill + greedy step every slot to
    /// completion against a single shared batched runtime. Exercises the
    /// moonshine runtime + slot wiring without the continuous-batching control
    /// flow.
    fn decode_batched_slots(
        state: &mut OwnerThreadState<MoonshineFamily>,
        slots: &mut [MoonshineBatchSlot],
    ) -> Result<(), MoonshineServeBatchError> {
        let n_seq = slots.len();
        let first_job = &slots
            .first()
            .ok_or_else(|| MoonshineServeBatchError::OwnerFailed {
                reason: "moonshine serve batch received no slots".to_string(),
            })?
            .job;
        let prompt_len = first_job.decode_config.initial_prompt_tokens.len();
        if prompt_len == 0 {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch prompt is empty".to_string(),
            });
        }
        let runtime = state.batched_runtime_for(first_job, n_seq)?;
        for (slot_index, slot) in slots.iter().enumerate() {
            runtime
                .populate_cross_attention_cache_slot(slot_index, &slot.job.encoder_output)
                .map_err(map_decoder_error)?;
        }

        let prompt_tokens = slots
            .first()
            .map(|slot| slot.job.decode_config.initial_prompt_tokens.as_slice())
            .ok_or_else(|| MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch received no slots".to_string(),
            })?;
        if prompt_tokens.len() != prompt_len {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch prompt length changed during test seed".to_string(),
            });
        }
        let logits = runtime
            .compute_batched_prefill_logits(prompt_tokens)
            .map_err(map_decoder_error)?;
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
                    MoonshineServeBatchError::DecodeFailed {
                        reason: "moonshine serve batch generated token history is empty"
                            .to_string(),
                    }
                })?;
                let total_tokens = prompt_len
                    .checked_add(slot.generated_tokens.len())
                    .ok_or_else(|| MoonshineServeBatchError::DecodeFailed {
                        reason: "moonshine serve batch token count overflowed".to_string(),
                    })?;
                let position = total_tokens.checked_sub(1).ok_or_else(|| {
                    MoonshineServeBatchError::DecodeFailed {
                        reason: "moonshine serve batch position underflowed".to_string(),
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
        slots: &mut [MoonshineBatchSlot],
        logits: &[f32],
    ) -> Result<(), MoonshineServeBatchError> {
        let vocab_size = slots
            .first()
            .map(|slot| slot.job.decode_config.vocab_size)
            .unwrap_or(0);
        let expected = vocab_size.checked_mul(slots.len()).ok_or_else(|| {
            MoonshineServeBatchError::DecodeFailed {
                reason: "moonshine serve batch logits length overflowed".to_string(),
            }
        })?;
        if logits.len() != expected {
            return Err(MoonshineServeBatchError::DecodeFailed {
                reason: format!(
                    "moonshine serve batch logits width mismatch: got {}, expected {}",
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
                MoonshineServeBatchError::DecodeFailed {
                    reason: "moonshine serve batch logits offset overflowed".to_string(),
                }
            })?;
            let end = start.checked_add(vocab_size).ok_or_else(|| {
                MoonshineServeBatchError::DecodeFailed {
                    reason: "moonshine serve batch logits end overflowed".to_string(),
                }
            })?;
            let slot_logits =
                logits
                    .get(start..end)
                    .ok_or_else(|| MoonshineServeBatchError::DecodeFailed {
                        reason: "moonshine serve batch logits slice out of bounds".to_string(),
                    })?;
            slot.select_next_token_from_logits(slot_logits.to_vec())?;
        }
        Ok(())
    }

    /// Test-only serial decode driver: runs the family serial path against a
    /// caller-owned lazily-built serial runtime (mirrors the previous
    /// `MoonshineOwnerThreadState::decode_serial_job`, which is now the generic
    /// owner's private hook).
    fn decode_serial_job(
        serial_runtime: &mut Option<MoonshineDecoderGraphRuntime>,
        job: MoonshineServeBatchJob,
    ) -> Result<MoonshineDecodeOutput, MoonshineServeBatchError> {
        MoonshineFamily::decode_serial(serial_runtime, job)
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
            metadata,
            tensor_index: Arc::new(tensor_index),
        }
    }

    fn sample_encoder_output(
        metadata: MoonshineExecutionMetadata,
        phase: f32,
        frame_count: usize,
    ) -> MoonshineEncoderOutput {
        let mut rows = Vec::with_capacity(frame_count * metadata.d_model);
        for frame_idx in 0..frame_count {
            for hidden_idx in 0..metadata.d_model {
                rows.push(
                    (((frame_idx * metadata.d_model + hidden_idx) as f32 * 0.03125) + phase).sin(),
                );
            }
        }
        MoonshineEncoderOutput {
            frame_count,
            hidden_size: metadata.d_model,
            rows,
        }
    }

    fn batch_job(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        prepared_runtime: Arc<MoonshinePreparedRuntime>,
        encoder_output: MoonshineEncoderOutput,
    ) -> MoonshineServeBatchJob {
        batch_job_with_max_generated_tokens(
            runtime_path,
            backend,
            uses_scheduler,
            prepared_runtime,
            encoder_output,
            1,
        )
    }

    fn batch_job_with_max_generated_tokens(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        prepared_runtime: Arc<MoonshinePreparedRuntime>,
        encoder_output: MoonshineEncoderOutput,
        max_generated_tokens: usize,
    ) -> MoonshineServeBatchJob {
        let decode_config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![prepared_runtime.metadata.bos_token_id],
            eot_token_id: prepared_runtime.metadata.eos_token_id,
            stop_token_ids: Vec::new(),
            vocab_size: prepared_runtime.metadata.vocab_size,
            max_generated_tokens,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        MoonshineServeBatchJob {
            runtime_cache_path: runtime_path.to_path_buf(),
            backend,
            uses_scheduler,
            prepared_runtime,
            encoder_output,
            decode_config,
            word_timestamps: false,
            audio_duration_seconds: 1.0,
        }
    }

    #[test]
    fn moonshine_serve_batch_env_defaults_off() {
        with_serve_batch_env(None, || {
            assert!(MoonshineServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn moonshine_serve_batch_env_one_keeps_default_path() {
        with_serve_batch_env(Some("1"), || {
            assert!(MoonshineServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn moonshine_serve_batch_env_accepts_two_to_eight() {
        with_serve_batch_env(Some("4"), || {
            let config = MoonshineServeBatchConfig::from_env()
                .unwrap()
                .expect("enabled");
            assert_eq!(config.max_batch, 4);
        });
    }

    #[test]
    fn moonshine_serve_batch_env_rejects_out_of_range() {
        with_serve_batch_env(Some("9"), || {
            assert!(matches!(
                MoonshineServeBatchConfig::from_env(),
                Err(MoonshineServeBatchError::InvalidEnv { .. })
            ));
        });
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_owner_thread_decodes_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(MOONSHINE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{MOONSHINE_SERVE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared_runtime = Arc::new(
            super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
                .expect("prepared runtime"),
        );
        let metadata = prepared_runtime.metadata;
        let encoder_output_0 = sample_encoder_output(metadata, 0.0, 32);
        let encoder_output_1 = sample_encoder_output(metadata, 0.25, 32);
        let runtime_config = super::super::graph_config::moonshine_decoder_graph_config(false);
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "moonshine static batch fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let mut slots = vec![
            MoonshineBatchSlot::new(batch_job(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&prepared_runtime),
                encoder_output_0,
            ))
            .expect("slot 0"),
            MoonshineBatchSlot::new(batch_job(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                prepared_runtime,
                encoder_output_1,
            ))
            .expect("slot 1"),
        ];
        let mut state = OwnerThreadState::<MoonshineFamily>::new();

        decode_batched_slots(&mut state, &mut slots).expect("batched decode");

        let outputs = slots
            .into_iter()
            .map(MoonshineBatchSlot::finish)
            .collect::<Result<Vec<_>, _>>()
            .expect("finish slots");
        assert_eq!(outputs.len(), 2);
        assert!(
            outputs
                .iter()
                .all(|output| output.generated_tokens.len() <= 1)
        );
    }

    /// Throughput-vs-N benchmark (issue #35 P2): measures aggregate decode
    /// tokens/s for batch widths N=1,2,4,8 on a real moonshine pack and the
    /// selected GGML backend. Each slot runs a fixed `steps` budget with EOS
    /// suppressed, so the workload is exactly N*steps tokens; we warm the
    /// per-N batched graph, then time `decode_batched_slots`. If the batched
    /// GEMM lever works, tokens/s rises with N (and wall stays ~flat). On a
    /// compute-bound backend (CPU) it stays ~flat, which is the contrast.
    /// Writes a CSV with an N column + tokens/s + speedup-vs-N=1.
    #[test]
    #[ignore = "manual throughput benchmark: set OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu|hip|vulkan; optional OPENASR_SERVE_BATCH_BENCH_STEPS, OPENASR_SERVE_BATCH_BENCH_OUT"]
    fn moonshine_serve_batch_throughput_vs_batch_width_real_pack() {
        let runtime_path = std::env::var_os(MOONSHINE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{MOONSHINE_SERVE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared_runtime = Arc::new(
            super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
                .expect("prepared runtime"),
        );
        let metadata = prepared_runtime.metadata;
        let runtime_config = super::super::graph_config::moonshine_decoder_graph_config(false);
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "throughput benchmark validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let steps = std::env::var("OPENASR_SERVE_BATCH_BENCH_STEPS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|steps| *steps > 0)
            .unwrap_or(48);
        let backend_label = std::env::var("OPENASR_GGML_BACKEND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("{:?}", runtime_config.backend));

        let build_job = |slot_idx: usize| -> MoonshineServeBatchJob {
            let encoder_output = sample_encoder_output(metadata, slot_idx as f32 * 0.125, 32);
            let decode_config = Seq2SeqGreedyDecodeConfig {
                initial_prompt_tokens: vec![metadata.bos_token_id],
                eot_token_id: metadata.eos_token_id,
                stop_token_ids: Vec::new(),
                vocab_size: metadata.vocab_size,
                max_generated_tokens: steps,
                suppress_first_step_token_ids: Vec::new(),
                // Suppress EOS so each slot runs the full fixed step budget ->
                // deterministic N*steps token workload to time.
                suppress_token_ids: vec![metadata.eos_token_id],
                phrase_biases: Vec::new(),
            };
            MoonshineServeBatchJob {
                runtime_cache_path: runtime_path.to_path_buf(),
                backend: runtime_config.backend,
                uses_scheduler: runtime_config.use_scheduler,
                prepared_runtime: Arc::clone(&prepared_runtime),
                encoder_output,
                decode_config,
                word_timestamps: false,
                audio_duration_seconds: 1.0,
            }
        };
        let build_slots = |n_seq: usize| -> Vec<MoonshineBatchSlot> {
            (0..n_seq)
                .map(|slot_idx| {
                    MoonshineBatchSlot::new(build_job(slot_idx)).expect("benchmark slot")
                })
                .collect()
        };

        let mut state = OwnerThreadState::<MoonshineFamily>::new();
        let mut serial_runtime: Option<MoonshineDecoderGraphRuntime> = None;

        let mut rows: Vec<String> =
            vec!["backend,n,steps,total_tokens,wall_ms,tokens_per_sec,speedup_vs_n1".to_string()];
        let mut baseline_tps: Option<f64> = None;

        for n_seq in [1usize, 2, 4, 8] {
            // N=1 is the real single-stream baseline: the batched prefill graph
            // is multi-sequence-only, and a lone request runs the serial path.
            // N>=2 runs the batched graph. Warm up first so the timed run
            // measures steady-state decode, not the one-time graph build.
            let (total_tokens, wall) = if n_seq == 1 {
                let _ = decode_serial_job(&mut serial_runtime, build_job(0))
                    .expect("warmup serial decode");
                let started = std::time::Instant::now();
                let output = decode_serial_job(&mut serial_runtime, build_job(0))
                    .expect("timed serial decode");
                (output.generated_tokens.len(), started.elapsed())
            } else {
                let mut warm = build_slots(n_seq);
                decode_batched_slots(&mut state, &mut warm).expect("warmup batched decode");
                let mut slots = build_slots(n_seq);
                let started = std::time::Instant::now();
                decode_batched_slots(&mut state, &mut slots).expect("timed batched decode");
                let total: usize = slots.iter().map(|slot| slot.generated_tokens.len()).sum();
                (total, started.elapsed())
            };

            let wall_secs = wall.as_secs_f64();
            let tokens_per_sec = if wall_secs > 0.0 {
                total_tokens as f64 / wall_secs
            } else {
                0.0
            };
            let baseline = *baseline_tps.get_or_insert(tokens_per_sec);
            let speedup = if baseline > 0.0 {
                tokens_per_sec / baseline
            } else {
                0.0
            };

            let row = format!(
                "{backend_label},{n_seq},{steps},{total_tokens},{:.1},{tokens_per_sec:.1},{speedup:.2}",
                wall_secs * 1000.0
            );
            println!("{row}");
            rows.push(row);
        }

        let csv = rows.join("\n") + "\n";
        let out_path = std::env::var_os("OPENASR_SERVE_BATCH_BENCH_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "moonshine-serve-batch-throughput-{backend_label}.csv"
                ))
            });
        std::fs::write(&out_path, csv).expect("write throughput csv");
        eprintln!("wrote throughput csv: {}", out_path.display());
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_owner_thread_refills_free_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(MOONSHINE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{MOONSHINE_SERVE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared_runtime = Arc::new(
            super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
                .expect("prepared runtime"),
        );
        let metadata = prepared_runtime.metadata;
        let runtime_config = super::super::graph_config::moonshine_decoder_graph_config(false);
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "moonshine refill fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&prepared_runtime),
                sample_encoder_output(metadata, encoder_phase, 32),
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            (MoonshineServeBatchEnvelope { job, reply }, reply_rx)
        };

        let (initial_fast, initial_fast_rx) = envelope(0.0, 1);
        let (initial_long, initial_long_rx) = envelope(0.25, 3);
        let (queued_refill, queued_refill_rx) = envelope(0.5, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(1);
        queued_tx.send(queued_refill).expect("queue refill job");

        let mut state = OwnerThreadState::<MoonshineFamily>::new();
        let config = MoonshineServeBatchConfig {
            max_batch: 2,
            queue_capacity: MOONSHINE_SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: Duration::ZERO,
            send_timeout: MOONSHINE_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: MOONSHINE_SERVE_BATCH_REPLY_TIMEOUT,
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

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_owner_thread_rebuckets_full_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(MOONSHINE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{MOONSHINE_SERVE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared_runtime = Arc::new(
            super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
                .expect("prepared runtime"),
        );
        let metadata = prepared_runtime.metadata;
        let runtime_config = super::super::graph_config::moonshine_decoder_graph_config(false);
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "moonshine rebucket fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&prepared_runtime),
                sample_encoder_output(metadata, encoder_phase, 32),
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            (MoonshineServeBatchEnvelope { job, reply }, reply_rx)
        };

        let (initial_long_a, initial_long_a_rx) = envelope(0.0, 3);
        let (initial_long_b, initial_long_b_rx) = envelope(0.25, 3);
        let (queued_refill_a, queued_refill_a_rx) = envelope(0.5, 1);
        let (queued_refill_b, queued_refill_b_rx) = envelope(0.75, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_refill_a).expect("queue refill a");
        queued_tx.send(queued_refill_b).expect("queue refill b");

        let mut state = OwnerThreadState::<MoonshineFamily>::new();
        let config = MoonshineServeBatchConfig {
            max_batch: 4,
            queue_capacity: MOONSHINE_SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: Duration::ZERO,
            send_timeout: MOONSHINE_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: MOONSHINE_SERVE_BATCH_REPLY_TIMEOUT,
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
            .recv_timeout(Duration::from_secs(30))
            .expect("long a reply")
            .expect("long a output");
        let long_b = initial_long_b_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("long b reply")
            .expect("long b output");
        let refill_a = queued_refill_a_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("refill a reply")
            .expect("refill a output");
        let refill_b = queued_refill_b_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("refill b reply")
            .expect("refill b output");

        assert!(long_a.generated_tokens.len() <= 3);
        assert!(long_b.generated_tokens.len() <= 3);
        assert!(refill_a.generated_tokens.len() <= 1);
        assert!(refill_b.generated_tokens.len() <= 1);
        assert!(state.batched_runtimes.contains_key(&2));
        assert!(state.batched_runtimes.contains_key(&4));
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_owner_thread_shrinks_tail_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(MOONSHINE_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{MOONSHINE_SERVE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared_runtime = Arc::new(
            super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
                .expect("prepared runtime"),
        );
        let metadata = prepared_runtime.metadata;
        let runtime_config = super::super::graph_config::moonshine_decoder_graph_config(false);
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "moonshine shrink fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                Arc::clone(&prepared_runtime),
                sample_encoder_output(metadata, encoder_phase, 32),
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            (MoonshineServeBatchEnvelope { job, reply }, reply_rx)
        };

        let (initial_fast_a, initial_fast_a_rx) = envelope(0.0, 1);
        let (initial_fast_b, initial_fast_b_rx) = envelope(0.25, 1);
        let (initial_long, initial_long_rx) = envelope(0.5, 3);
        let (_queued_tx, queued_rx) = mpsc::sync_channel(1);

        let mut state = OwnerThreadState::<MoonshineFamily>::new();
        let config = MoonshineServeBatchConfig {
            max_batch: 4,
            queue_capacity: MOONSHINE_SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: Duration::ZERO,
            send_timeout: MOONSHINE_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: MOONSHINE_SERVE_BATCH_REPLY_TIMEOUT,
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
            .recv_timeout(Duration::from_secs(30))
            .expect("fast a reply")
            .expect("fast a output");
        let fast_b = initial_fast_b_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("fast b reply")
            .expect("fast b output");
        let long = initial_long_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("long reply")
            .expect("long output");

        assert!(fast_a.generated_tokens.len() <= 1);
        assert!(fast_b.generated_tokens.len() <= 1);
        assert!(long.generated_tokens.len() <= 3);
        assert!(state.batched_runtimes.contains_key(&4));
        assert!(state.batched_runtimes.contains_key(&2));
    }
}
