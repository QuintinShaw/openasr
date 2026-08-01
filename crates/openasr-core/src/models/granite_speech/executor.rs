//! `GgmlAsrExecutor` implementation for granite-speech, wiring the already-
//! validated pipeline (`frontend` -> `encoder_graph` -> `qformer` -> `prompt`
//! -> `decode_executor` -> shared greedy-decode driver -> `tokenizer`)
//! against a real `.oasr` pack via `runtime_provider::load_tensors_from_oasr_pack`.
//!
//! Registry status: wired into `arch::mod`'s `OpenAsrArchitectureRegistry`
//! (architecture descriptor + component descriptors), `executor_component_registry`,
//! `decode_policy_component_registry`, and `runtime_tensor_contract_registry`
//! (see `runtime_contract.rs` for the metadata parsers those two validate
//! against). No dedicated `frontend_component_registry`/
//! `tokenizer_component_registry` entry exists in this codebase shape --
//! frontend/tokenizer selection for a dedicated (non-composed) executor
//! family is the executor's own job (this file constructs
//! `GraniteSpeechMelFrontend`/`GraniteSpeechTokenizer` directly), the same
//! precedent `firered_llm`/`mimo_asr`/`moss_transcribe_diarize` follow.
//!
//! Streaming: this pass was scoped "file-transcribe only, no streaming", but
//! `builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch`
//! has a fail-closed completeness gate that rejects its ENTIRE dispatch (for
//! every family, not just this one) if any registered architecture has no
//! streaming executor at all -- discovered by a workspace-wide test failure
//! across unrelated families after this one's architecture descriptor
//! landed, not a granite-speech-specific test. `GgmlAsrStreamingExecutor`
//! below is therefore a required registration, not scope creep: it reuses
//! the exact same offline `execute_inner` through the shared buffered
//! snapshot streaming driver (`build_seq2seq_streaming_session`, matching
//! moonshine/qwen's own precedent for a family with no incremental decode
//! session yet) -- no new streaming-specific logic, and no claim of
//! streaming-tuned latency.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;

use thiserror::Error;

use super::decode_executor::GraniteSpeechResidentAudioDecodeStepExecutor;
use super::decode_session::GraniteSpeechDecodeSession;
use super::decoder_graph::GraniteSpeechDecoderConfig;
use super::prompt::{GRANITE_SPEECH_AUDIO_TOKEN, build_audio_prompt_embeddings};
use super::runtime_contract::{
    parse_decoder_metadata, parse_encoder_metadata, parse_projector_metadata,
};
use super::runtime_provider::load_tensors_from_oasr_pack;
use super::tokenizer::GraniteSpeechTokenizer;
use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlRuntimeSource, read_gguf_metadata};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrPreparedAudio,
};
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError;
use crate::models::thread_local_runtime_cache::{
    PackContentKey, current_unload_generation, take_generation_tagged,
};

use crate::arch::{GRANITE_SPEECH_DECODE_POLICY_ID, GRANITE_SPEECH_GGML_ADAPTER_ID};

/// Everything granite-speech materializes from a pack that does NOT depend on
/// the per-request audio, kept resident across requests: the keep-quantized
/// decode session (its graph runner, the mmap'd loaded weight context, and the
/// decoder's zero-copy bound projection/norm/lm_head weights) plus the decoder
/// token-embedding table read once from the pack. Before this cache, the
/// keep-quantized decoder still rebuilt its loaded weight context on EVERY
/// `execute()` (a fresh runner init + `load_gguf_weight_context` +
/// `GraniteDecoderLoadedWeights::load`, ~4.2s measured) purely to re-derive
/// state that never changes between requests against the same pack. Mirrors
/// `firered_llm`'s resident-decoder cache (`FIRERED_LLM_DECODER_BY_KEY`) and
/// `mimo_asr`'s `MimoAsrPreparedRuntime`.
///
/// The single-runner invariant is preserved by construction: the session owns
/// its runner and the loaded context that was built ON that runner together as
/// one unit (`GraniteSpeechDecodeSession::new_keep_quantized`), and this struct
/// moves that whole session in and out of the cache without ever separating the
/// runner from its loaded context or re-binding weights onto a different runner.
/// The per-request K/V history is released
/// (`release_session_scoped_buffers`) before the session re-enters the cache,
/// so only the per-request-invariant resident state is carried across requests.
///
/// The embedding table is OWNED here (not a borrow of the request's transient
/// load) so the resident session carries no lifetime and can live in the
/// thread-local cache; it doubles as the `GraniteSpeechDecoderWeightProvider`
/// for prompt assembly and each generated token's per-step embedding lookup.
struct GraniteSpeechPreparedRuntime {
    session: GraniteSpeechDecodeSession,
    embed_table: HashMap<String, Vec<f32>>,
}

impl GraniteSpeechPreparedRuntime {
    /// Materialize the resident decode session + embedding table once for a
    /// given `(pack, backend)`. This is the whole ~4.2s cost the resident cache
    /// exists to pay exactly once instead of per request.
    fn build(
        source: &GgmlRuntimeSource,
        decoder_config: GraniteSpeechDecoderConfig,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechGgmlExecutorError> {
        let pack_path = source.path();
        // Only the decoder's token-embedding table on the host (dequantized to
        // f32) -- the projection/norm/lm_head weights are bound zero-copy inside
        // the session below (see `runtime_provider` / `decode_session`).
        let embed_table =
            load_tensors_from_oasr_pack(pack_path, "language_model.model.embed_tokens.weight")
                .map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
                    reason: error.to_string(),
                })?;
        let session =
            GraniteSpeechDecodeSession::new_keep_quantized(decoder_config, source, backend)
                .map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
                    reason: error.to_string(),
                })?;
        Ok(Self {
            session,
            embed_table,
        })
    }
}

/// Resident prepared-runtime cache keyed by (pack content id, backend), the
/// same design + idle-unload-generation tagging `firered_llm` / `mimo_asr` use.
/// The pack half is a [`PackContentKey`] from the request's already-open
/// source, so an in-place `.oasr` replacement at the same path resolves a
/// different id and the next lookup rebuilds instead of reusing a session whose
/// device-bound weights came from the old bytes. Entries carry the idle-unload
/// generation they were built under: `take_generation_tagged` discards any
/// runtime built before the last idle unload (the reaper cannot reach this
/// worker thread's TLS from its own thread), so the resident 2B decoder stays
/// evictable under memory pressure through the shared central unload clock -- no
/// bespoke policy. A plain `HashMap` (not the bounded LRU): the key does not
/// explode per audio chunk (one entry is built and reused across a whole
/// longform run for a given pack/backend), so there is no unbounded-growth
/// hazard to bound.
type GraniteSpeechPreparedRuntimeCacheKey = (PackContentKey, GgmlCpuGraphBackend);

thread_local! {
    static GRANITE_SPEECH_PREPARED_BY_KEY: RefCell<
        HashMap<GraniteSpeechPreparedRuntimeCacheKey, (u64, GraniteSpeechPreparedRuntime)>,
    > = RefCell::new(HashMap::new());
}

fn take_cached_prepared_runtime(
    key: &GraniteSpeechPreparedRuntimeCacheKey,
    unload_generation: u64,
) -> Option<GraniteSpeechPreparedRuntime> {
    GRANITE_SPEECH_PREPARED_BY_KEY
        .with(|cache| take_generation_tagged(&mut cache.borrow_mut(), key, unload_generation))
}

fn store_cached_prepared_runtime(
    key: GraniteSpeechPreparedRuntimeCacheKey,
    unload_generation: u64,
    prepared: GraniteSpeechPreparedRuntime,
) {
    GRANITE_SPEECH_PREPARED_BY_KEY.with(|cache| {
        cache
            .borrow_mut()
            .insert(key, (unload_generation, prepared));
    });
}

const GRANITE_SPEECH_EXECUTOR_ID: &str = "granite-speech-ggml-executor-v1";
const GRANITE_SPEECH_EOT_TOKEN_ID: u32 = 100_257;
const GRANITE_SPEECH_DEFAULT_QUESTION: &str =
    "can you transcribe the speech into a written format?";
const GRANITE_SPEECH_MAX_GENERATED_TOKENS: usize = 256;

#[derive(Debug, Error)]
enum GraniteSpeechGgmlExecutorError {
    #[error("granite-speech ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("granite-speech ggml executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("granite-speech ggml executor frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("granite-speech ggml executor metadata contract failed: {reason}")]
    MetadataFailed { reason: String },
    #[error("granite-speech ggml executor encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("granite-speech ggml executor projector failed: {reason}")]
    ProjectorFailed { reason: String },
    #[error("granite-speech ggml executor tokenizer failed: {reason}")]
    TokenizerFailed { reason: String },
    #[error("granite-speech ggml executor prompt assembly failed: {reason}")]
    PromptFailed { reason: String },
    #[error("granite-speech ggml executor decode failed: {reason}")]
    DecodeFailed { reason: String },
}

/// No-op phrase-bias shim: granite-speech applies keyword biasing through its
/// own prompt convention (the `Keywords:` suffix assembled above), never the
/// shared decode-time logit-boost path, so the registry-routed greedy loop is
/// handed a token source that contributes nothing -- mirrors
/// `mimo_asr::executor::NoPhraseBiasTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GraniteSpeechGgmlExecutor;

impl GraniteSpeechGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GraniteSpeechGgmlExecutorError> {
        if request.selected_family.adapter_id != GRANITE_SPEECH_GGML_ADAPTER_ID {
            return Err(GraniteSpeechGgmlExecutorError::AdapterMismatch {
                expected: GRANITE_SPEECH_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(
                |error| GraniteSpeechGgmlExecutorError::RuntimePreflightFailed {
                    reason: error.to_string(),
                },
            )?;
        let pack_path = preflight.runtime_source.path();

        let samples = downmix_prepared_audio(&request.prepared_audio);
        let frontend = super::frontend::GraniteSpeechMelFrontend::new();
        let (features, frames) = frontend.extract(&samples).map_err(|error| {
            GraniteSpeechGgmlExecutorError::FrontendFailed {
                reason: error.to_string(),
            }
        })?;

        let backend = request.resolved_runtime.backend();

        let metadata = read_gguf_metadata(pack_path).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let encoder_config = parse_encoder_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let projector_config = parse_projector_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = parse_decoder_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;

        let encoder_weights =
            load_tensors_from_oasr_pack(pack_path, "encoder.").map_err(|error| {
                GraniteSpeechGgmlExecutorError::EncoderFailed {
                    reason: error.to_string(),
                }
            })?;
        let encoder_output = super::encoder_graph::encode(
            &encoder_config,
            &encoder_weights,
            &features,
            frames,
            backend,
            false,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;

        let projector_weights =
            load_tensors_from_oasr_pack(pack_path, "projector.").map_err(|error| {
                GraniteSpeechGgmlExecutorError::ProjectorFailed {
                    reason: error.to_string(),
                }
            })?;
        let projector_output = super::qformer::project(
            &projector_config,
            &projector_weights,
            &encoder_output.encoder_out,
            encoder_output.frames,
            backend,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::ProjectorFailed {
            reason: error.to_string(),
        })?;

        let tokenizer = GraniteSpeechTokenizer::from_gguf_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::TokenizerFailed {
                reason: error.to_string(),
            }
        })?;

        // KWB (keyword-list biasing): the model's own documented prompt
        // convention -- "transcribe the speech to text. Keywords: <kw1>,
        // <kw2>, ..." -- not a decode-time logit bias (see the family's
        // end-to-end KWB test). `phrase_bias`'s configured phrases become
        // the `Keywords:` suffix when present.
        let question = match request.request_options.phrase_bias.as_ref() {
            Some(phrase_bias) if !phrase_bias.is_empty() => {
                let keywords = phrase_bias
                    .entries()
                    .iter()
                    .map(|entry| entry.phrase())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("transcribe the speech to text. Keywords: {keywords}")
            }
            _ => GRANITE_SPEECH_DEFAULT_QUESTION.to_string(),
        };
        let prompt_text = format!("USER: {GRANITE_SPEECH_AUDIO_TOKEN}{question}\n ASSISTANT:");

        // Cross-request resident runtime: the keep-quantized decode session
        // (runner + mmap'd loaded weight context + zero-copy decoder weights)
        // and the decoder embedding table are per-request-invariant for a given
        // (pack, backend), so take a resident one from this thread's cache (or
        // build it once on a cold miss) instead of paying the ~4.2s
        // loaded-context rebuild every transcription.
        let cache_key: GraniteSpeechPreparedRuntimeCacheKey = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            backend,
        );
        // Sampled before the take and reused for the store-back below: if the
        // idle-unload reaper bumps the generation while this decode is in
        // flight, the runtime goes back tagged with the pre-unload generation
        // and the *next* take discards it, so an unload is never lost to an
        // overlapping decode (mirrors firered_llm / mimo_asr).
        let unload_generation = current_unload_generation();
        let mut prepared = match take_cached_prepared_runtime(&cache_key, unload_generation) {
            Some(prepared) => prepared,
            None => GraniteSpeechPreparedRuntime::build(
                &preflight.runtime_source,
                decoder_config,
                backend,
            )?,
        };

        let (prompt_token_ids, prompt_embeddings) = build_audio_prompt_embeddings(
            &decoder_config,
            &prepared.embed_table,
            &tokenizer,
            &prompt_text,
            &projector_output.projected,
            projector_output.tokens,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::PromptFailed {
            reason: error.to_string(),
        })?;

        // Greedy decode rides the one shared driver via the decode-policy
        // registry (AGENTS.md single-driver invariant): the registered
        // `GRANITE_SPEECH_DECODE_POLICY_ID` descriptor supplies the
        // stop-token / suppression / postprocess policy, this executor only
        // hands over the config inputs, the step executor, and the token
        // decoder. Keyword biasing is done in the prompt above, so the token
        // source contributes nothing and `phrase_bias` stays `None`.
        let decode_config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: prompt_token_ids,
            eot_token_id: GRANITE_SPEECH_EOT_TOKEN_ID,
            vocab_size: decoder_config.vocab_size,
            max_generated_tokens: GRANITE_SPEECH_MAX_GENERATED_TOKENS,
        };
        let decode_text_token_ids =
            |token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> {
                tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                })
            };
        // Disjoint field borrows of the resident runtime: `&mut session` for the
        // decode graph and `&embed_table` for the per-step token embeds are
        // distinct fields, so both stay live for the whole decode.
        let mut step_executor = GraniteSpeechResidentAudioDecodeStepExecutor::new(
            &mut prepared.session,
            &prepared.embed_table,
            prompt_embeddings,
        );
        let decode_result = run_builtin_seq2seq_decode_policy(
            GRANITE_SPEECH_DECODE_POLICY_ID,
            &decode_config,
            &NoPhraseBiasTokenSource,
            None,
            &mut step_executor,
            &decode_text_token_ids,
            |error: Seq2SeqGreedyDecodeError| error,
            |error: Seq2SeqGreedyDecodeError| error,
            map_registry_error,
            &request.execution_context.control,
        );
        // Release this decode's per-request K/V buffers before the resident
        // session re-enters the cache (mirrors firered/mimo's
        // release_session_scoped_buffers): the heavy runner + loaded weights
        // stay resident, only the grown KV history is dropped -- regardless of
        // decode outcome, since a partial compute only poisons that per-request
        // state, not the uploaded weights. `step_executor`'s borrows of
        // `prepared` end here under NLL, so `prepared` is free to move into the
        // cache.
        prepared.session.release_session_scoped_buffers();
        store_cached_prepared_runtime(cache_key, unload_generation, prepared);
        let result =
            decode_result.map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
                reason: error.to_string(),
            })?;

        let audio_duration_seconds = request.prepared_audio.samples_f32.len() as f32
            / request.prepared_audio.sample_rate_hz.max(1) as f32;
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: result.text.clone(),
                segments: vec![Segment {
                    start: 0.0,
                    end: audio_duration_seconds,
                    text: result.text,
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                }],
                longform: None,
                language: None,
            },
            carry_context: None,
            // No intra-decode timestamps -- the single segment spans the whole
            // buffer -- so the cut point has no honest second to name, same as
            // mimo-asr / firered-aed.
            decode_truncation: result.stop_reason.into_decode_truncation(None),
        })
    }
}

fn downmix_prepared_audio(audio: &GgmlAsrPreparedAudio) -> Vec<f32> {
    if audio.channels <= 1 {
        return audio.samples_f32.clone();
    }
    let channels = audio.channels as usize;
    audio
        .samples_f32
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

impl GgmlAsrExecutor for GraniteSpeechGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        GRANITE_SPEECH_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // Native KWB via the prompt convention above -- not the shared
        // decode-time phrase_bias_decode logit-boost mechanism (unused here,
        // matching AGENTS.md's per-family explicit-declaration rule: a family
        // states its own true/false, it never inherits a default).
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| granite_speech_execute_error_to_ggml(self, error, request))
    }
}

fn granite_speech_execute_error_to_ggml(
    executor: &GraniteSpeechGgmlExecutor,
    error: GraniteSpeechGgmlExecutorError,
    request: &GgmlAsrExecutionRequest,
) -> GgmlAsrExecutionError {
    GgmlAsrExecutionError::ExecutorFailed {
        executor_id: GgmlAsrExecutor::executor_id(executor),
        adapter_id: request.selected_family.adapter_id,
        reason: error.to_string(),
    }
}

impl GraniteSpeechGgmlExecutor {
    /// Streaming decode: re-runs the SAME offline pipeline (`execute_inner`)
    /// against the growing/windowed audio buffer the shared streaming driver
    /// hands it -- there is no incremental KV-cache session to plug in yet
    /// (see `decode_executor.rs`'s O(n^2) recompute-per-step note), so every
    /// partial re-does frontend + encoder + Q-Former + a full prefill-style
    /// decode from scratch. This is registered to satisfy the codebase's
    /// fail-closed streaming-completeness gate
    /// (`builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch`
    /// rejects the WHOLE dispatch, for every family, if any registered
    /// architecture has no streaming executor at all) -- it is correctness-
    /// only, not a real-time-tuned streaming path. The FINAL transcript stays
    /// byte-identical to `execute()`.
    fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| granite_speech_execute_error_to_ggml(self, error, request))
    }
}

const GRANITE_SPEECH_STREAMING_EXECUTOR_ID: &str =
    "granite-speech-ggml-snapshot-streaming-executor-v1";

impl crate::models::ggml_asr_executor::GgmlAsrStreamingExecutor for GraniteSpeechGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        GRANITE_SPEECH_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &crate::models::ggml_asr_executor::GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
        crate::models::incremental_streaming_driver::build_seq2seq_streaming_session(
            *self,
            GRANITE_SPEECH_STREAMING_EXECUTOR_ID,
            GRANITE_SPEECH_GGML_ADAPTER_ID,
            "granite-speech",
            request,
            crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            GraniteSpeechGgmlExecutor::execute_streaming,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrBackendPreference;
    use crate::models::ggml_family_registry::granite_speech_runtime_descriptor_v1;

    /// Points at a real converted granite-speech `.oasr` pack via
    /// `OPENASR_GRANITE_SPEECH_PACK`. Loading it mmaps + touches a multi-GB
    /// file plus materializes the f16 token-embedding table -- a real memory
    /// commitment, not a network fetch -- so this stays `#[ignore]`d and skips
    /// silently when the env var is unset (same convention as firered-llm's
    /// own dev-pack test) rather than gating CI on a private multi-GB artifact.
    fn dev_pack_path() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_GRANITE_SPEECH_PACK",
            "granite-speech .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    /// Runs one full `execute()` against `pack_path` on the CPU (deterministic)
    /// backend and returns the transcript. Skips (returns `None`) when the pack
    /// is absent.
    fn transcribe_with_pack(pack_path: PathBuf, wav_path: PathBuf) -> Option<String> {
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "granite-speech e2e test",
            "granite-speech e2e test",
        )
        .expect("load wav fixture");

        let backend_preference = GgmlAsrBackendPreference::CpuOnly;
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
        );
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: granite_speech_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let _backend_guard = crate::ggml_runtime::install_request_backend_override(
            request.backend_preference.request_backend_override(),
        );

        let executor = GraniteSpeechGgmlExecutor;
        let result = executor
            .execute(&request)
            .expect("granite-speech transcribe");
        Some(result.transcription.text)
    }

    /// Resident prepared-runtime cache regression: calling `execute()` twice in
    /// a row on the same thread (same pack + backend) MUST hit the thread-local
    /// `GRANITE_SPEECH_PREPARED_BY_KEY` cache on the second call and still
    /// produce a byte-identical transcript to the first (cache-miss/build)
    /// call. This is the load-bearing correctness gate for the resident cache:
    /// the second decode reuses a session whose per-request K/V history was
    /// released before it re-entered the cache, so any leak of prior-request
    /// state into the reused session would diverge the two transcripts here.
    ///
    /// Transcript-vs-reference correctness (that the decode produces the RIGHT
    /// text) is covered separately by the llama.cpp-reference goldens in
    /// `decode_executor` (en/ja/kwb) and the bit-exact incremental-session gate
    /// in `decode_session`; this test isolates the "reuse == fresh" invariant
    /// the resident cache adds, which those do not exercise.
    #[test]
    #[ignore = "requires a private multi-GB granite-speech .oasr pack via \
                OPENASR_GRANITE_SPEECH_PACK; skips silently when unset"]
    fn resident_prepared_runtime_reuse_across_consecutive_calls_stays_byte_identical() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let Some(first_text) = transcribe_with_pack(pack_path.clone(), jfk_wav_path()) else {
            return;
        };
        let second_text =
            transcribe_with_pack(pack_path, jfk_wav_path()).expect("second transcribe");
        assert!(
            !first_text.trim().is_empty(),
            "first (cache-miss/build) transcript must be non-empty"
        );
        assert_eq!(
            first_text, second_text,
            "second execute() (a resident prepared-runtime cache hit) must match the first \
             (cache-miss/build) call byte-for-byte"
        );
    }
}
