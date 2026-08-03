//! funasr-nano dedicated executor: fbank+LFR frontend (`sensevoice` WavFrontend,
//! NO CMVN -- Fun-ASR-Nano runs directly on fbank+LFR) -> the SAN-M
//! [`encoder_graph`] (eps 1e-5, hidden-state output) -> the 2-layer transformer
//! [`adapter_graph`] -> low-frame-rate audio-token truncation -> ChatML+audio
//! splice ([`decode_prompt`] + `qwen::build_qwen3_prompt_embeddings_with_audio_splice`)
//! -> Qwen3-0.6B [`llm_transformer`] prefill/decode, driven through the ONE
//! shared greedy decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a [`Seq2SeqGreedyDecodeStepExecutor`]
//! impl below -- never a hand-rolled argmax loop (the repo's
//! `model-integration-shared-driver` invariant, see `AGENTS.md`).

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FUNASR_NANO_DECODE_POLICY_ID;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, Qwen3AsrPromptEmbeddings,
    build_qwen3_prompt_embeddings_with_audio_splice,
};
use crate::models::sensevoice::encoder_graph::build_sensevoice_encoder_input;
use crate::models::sensevoice::frontend::{SenseVoiceFbankFrontend, apply_lfr};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, PackContentKey, current_unload_generation,
    take_generation_tagged, with_thread_local_cached_mut_by_key,
};

use super::adapter_graph::FunasrNanoAdapterGraph;
use super::decode_prompt::{build_funasr_nano_decode_prompt, funasr_nano_audio_token_count};
use super::encoder_graph::FunasrNanoEncoderGraph;
use super::llm_transformer::FunasrNanoDecoderRuntime;
use super::runtime_contract::{
    FunasrNanoDecoderMetadata, parse_funasr_nano_adapter_metadata,
    parse_funasr_nano_decoder_metadata, parse_funasr_nano_encoder_metadata,
};
use super::tokenizer::FunasrNanoTokenizer;

const FUNASR_NANO_EXECUTOR_ID: &str = "funasr-nano-ggml-executor-v1";
const FUNASR_NANO_STREAMING_EXECUTOR_ID: &str = "funasr-nano-ggml-snapshot-streaming-executor-v1";
/// Upstream single-utterance hard cap (the official runtime warns that a single
/// clip beyond ~40s greedily repeats out of distribution; `--chunk 15` fixes
/// it). The executor fails closed rather than silently running an OOD
/// multi-minute prefill; longer audio is the shared longform slicing
/// orchestrator's job (see the `ConservativeSeq2SeqV1` longform profile).
/// Also the single-decode clamp host-memory admission reads
/// ([`super::capacity::FUNASR_NANO_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS`]).
pub(crate) const FUNASR_NANO_MAX_INPUT_SECONDS: f32 = 40.0;
/// Fail-closed backstop against a non-terminating decode -- greedy decode stops
/// at `<|im_end|>` well before this in practice. Admission charges the same
/// figure (`super::capacity::funasr_nano_admission_required_positions`).
pub(crate) const FUNASR_NANO_MAX_GENERATED_TOKENS: usize = 512;

type FunasrNanoRuntimeCacheKey = (PackContentKey, GgmlCpuGraphBackend);

/// Resident encoder-side runtime: the SAN-M encoder graph + transformer
/// adaptor with their weights already uploaded to (or bound zero-copy in)
/// backend memory. Cached per (pack content id, backend) below -- the
/// sensevoice `SenseVoicePreparedRuntime` / dolphin prepared-runtime pattern
/// -- so a repeat request rebuilds only the transient forward graph and
/// uploads only the utterance features instead of re-loading and re-uploading
/// every encoder + adaptor weight. Idle unload flows through the central
/// `bump_unload_generation` epoch that `BoundedRuntimeCache` syncs to.
struct FunasrNanoEncoderAdapterRuntime {
    encoder: FunasrNanoEncoderGraph,
    adapter: FunasrNanoAdapterGraph,
}

thread_local! {
    static FUNASR_NANO_DECODER_BY_KEY: RefCell<HashMap<FunasrNanoRuntimeCacheKey, (u64, FunasrNanoDecoderRuntime)>> =
        RefCell::new(HashMap::new());
    static FUNASR_NANO_ENCODER_ADAPTER_BY_KEY: RefCell<
        BoundedRuntimeCache<FunasrNanoRuntimeCacheKey, FunasrNanoEncoderAdapterRuntime>,
    > = RefCell::new(BoundedRuntimeCache::new());
}

fn take_cached_decoder_runtime(
    key: &FunasrNanoRuntimeCacheKey,
    unload_generation: u64,
) -> Option<FunasrNanoDecoderRuntime> {
    FUNASR_NANO_DECODER_BY_KEY
        .with(|cache| take_generation_tagged(&mut cache.borrow_mut(), key, unload_generation))
}

fn store_cached_decoder_runtime(
    key: FunasrNanoRuntimeCacheKey,
    unload_generation: u64,
    decoder: FunasrNanoDecoderRuntime,
) {
    FUNASR_NANO_DECODER_BY_KEY.with(|cache| {
        cache.borrow_mut().insert(key, (unload_generation, decoder));
    });
}

#[derive(Debug, Error)]
enum FunasrNanoExecutorError {
    #[error("funasr-nano executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("funasr-nano executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("funasr-nano runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("funasr-nano tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("funasr-nano audio duration {seconds:.1}s exceeds the upstream {limit:.0}s hard cap")]
    AudioTooLong { seconds: f32, limit: f32 },
    #[error("funasr-nano frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("funasr-nano encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("funasr-nano adapter failed: {reason}")]
    AdapterFailed { reason: String },
    #[error("funasr-nano decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("funasr-nano prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("funasr-nano decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("funasr-nano greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FunasrNanoGgmlExecutor;

/// No-op phrase-bias shim: funasr-nano's decode policy never consults these
/// (no phrase bias, single config-supplied eot token) -- mirrors
/// `firered_llm::executor::NoPhraseBiasTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

/// Drives `FunasrNanoDecoderRuntime` through the shared greedy loop: step 0
/// consumes the pre-built (audio-spliced) prompt embeddings via one prefill
/// pass; every later step embeds the last generated token and decodes
/// incrementally (device-side top-1 on the Metal reuse graph, full host logits
/// on CPU). Mirrors `moss_transcribe_diarize::executor::MossTdGreedyStepExecutor`.
struct FunasrNanoGreedyStepExecutor<'a> {
    decoder: &'a mut FunasrNanoDecoderRuntime,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    prompt_embeddings: Option<Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
    control: Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for FunasrNanoGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if let Some(prompt_embeddings) = self.prompt_embeddings.take() {
            self.cache_prompt_tokens = prompt_embeddings.token_count;
            let prefill = self
                .decoder
                .prefill(&prompt_embeddings, &mut self.layer_kv_caches, &self.control)
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: prefill.logits,
                greedy_token_hint: prefill.greedy_token_hint,
            });
        }
        let last_token = input.generated_tokens.last().copied().ok_or_else(|| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "funasr-nano generated token history is unexpectedly empty".to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "funasr-nano decode cache position underflowed".to_string(),
            })?;
        if let Some(token_id) = self
            .decoder
            .decode_step_reused_top1(last_token, cache_position, &self.layer_kv_caches)
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?
        {
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(token_id),
            });
        }
        let logits = self
            .decoder
            .decode_step(last_token, cache_position, &mut self.layer_kv_caches)
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

impl FunasrNanoGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, FunasrNanoExecutorError> {
        let expected_adapter = crate::arch::FUNASR_NANO_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(FunasrNanoExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| FunasrNanoExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;

        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let decoder_metadata =
            parse_funasr_nano_decoder_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokenizer = FunasrNanoTokenizer::from_gguf_metadata(&preflight.metadata).map_err(
            |error: NativeAsrError| FunasrNanoExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            },
        )?;

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > FUNASR_NANO_MAX_INPUT_SECONDS {
            return Err(FunasrNanoExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: FUNASR_NANO_MAX_INPUT_SECONDS,
            });
        }

        // Frontend: kaldi fbank + FunASR LFR stacking, NO CMVN (Fun-ASR-Nano's
        // config carries `cmvn_file: null`; the official runtime runs directly
        // on fbank+LFR).
        let fbank = SenseVoiceFbankFrontend::new()
            .compute(samples)
            .map_err(|error| FunasrNanoExecutorError::FrontendFailed {
                reason: error.to_string(),
            })?;
        let lfr = apply_lfr(&fbank.data, fbank.n_mels).map_err(|error| {
            FunasrNanoExecutorError::FrontendFailed {
                reason: error.to_string(),
            }
        })?;
        if lfr.feature_dim != encoder_metadata.feature_dim {
            return Err(FunasrNanoExecutorError::FrontendFailed {
                reason: format!(
                    "LFR feature dim {} does not match encoder feature dim {}",
                    lfr.feature_dim, encoder_metadata.feature_dim
                ),
            });
        }
        let encoder_input = build_sensevoice_encoder_input(
            &[],
            &lfr.data,
            encoder_metadata.feature_dim,
            encoder_metadata.d_model,
        )
        .map_err(|error| FunasrNanoExecutorError::FrontendFailed {
            reason: error.to_string(),
        })?;

        let runtime_source = &preflight.runtime_source;
        let backend = request.resolved_runtime.backend();
        let (speech_rows, audio_token_count) = run_encoder_and_adapter(
            runtime_source,
            encoder_metadata,
            adapter_metadata,
            &encoder_input.data,
            encoder_input.n_frames,
            encoder_input.feature_dim,
            backend,
        )?;

        let decode_prompt = build_funasr_nano_decode_prompt(&tokenizer, audio_token_count)
            .map_err(|error| FunasrNanoExecutorError::DecodePromptFailed {
                reason: error.to_string(),
            })?;

        let decoder_cache_key: FunasrNanoRuntimeCacheKey =
            (PackContentKey::for_runtime_source(runtime_source), backend);
        let unload_generation = current_unload_generation();
        let mut decoder = match take_cached_decoder_runtime(&decoder_cache_key, unload_generation) {
            Some(decoder) => decoder,
            None => FunasrNanoDecoderRuntime::new(runtime_source, decoder_metadata, backend)
                .map_err(|error| FunasrNanoExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?,
        };

        let decode_result = decode_with_decoder(
            &mut decoder,
            &decoder_metadata,
            &decode_prompt,
            &speech_rows,
            &tokenizer,
            &request.execution_context.control,
        );
        decoder.release_session_scoped_buffers();
        store_cached_decoder_runtime(decoder_cache_key, unload_generation, decoder);
        let result = decode_result?;
        let decode_truncation = result.stop_reason.into_decode_truncation(None);

        let text = result.text.trim().to_string();
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            text,
            longform: None,
            language: None,
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
            decode_truncation,
        })
    }
}

/// Run the resident SAN-M encoder + transformer adaptor for this pack+backend
/// over the prepared encoder input, and return the leading `n_aud` audio-token
/// rows (low-frame-rate truncation) plus that count. The encoder + adaptor
/// runtime comes out of the per-(pack content id, backend) thread-local cache
/// ([`FunasrNanoEncoderAdapterRuntime`]); only the transient forward graph and
/// the utterance features are per-request. Output is bit-identical to a fresh
/// build: residency changes where the weights live, never the forward math
/// (pinned by the `cached_encoder_adapter_matches_fresh_build_bit_for_bit`
/// golden test below).
#[allow(clippy::too_many_arguments)]
fn run_encoder_and_adapter(
    runtime_source: &crate::GgmlRuntimeSource,
    encoder_metadata: super::runtime_contract::FunasrNanoEncoderMetadata,
    adapter_metadata: super::runtime_contract::FunasrNanoAdapterMetadata,
    encoder_input: &[f32],
    n_frames: usize,
    feature_dim: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<(Vec<f32>, usize), FunasrNanoExecutorError> {
    let key: FunasrNanoRuntimeCacheKey =
        (PackContentKey::for_runtime_source(runtime_source), backend);
    with_thread_local_cached_mut_by_key(
        &FUNASR_NANO_ENCODER_ADAPTER_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            let encoder = FunasrNanoEncoderGraph::new(runtime_source, encoder_metadata, backend)
                .map_err(|error| FunasrNanoExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
            let adapter = FunasrNanoAdapterGraph::new(runtime_source, adapter_metadata, backend)
                .map_err(|error| FunasrNanoExecutorError::AdapterFailed {
                    reason: error.to_string(),
                })?;
            Ok(FunasrNanoEncoderAdapterRuntime { encoder, adapter })
        },
        |runtime| {
            let encoder_output = runtime
                .encoder
                .encode(encoder_input, n_frames, feature_dim)
                .map_err(|error| FunasrNanoExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
            let (adapter_rows, adapter_frames) = runtime
                .adapter
                .run(
                    &encoder_output.rows,
                    encoder_output.frame_count,
                    encoder_output.d_model,
                )
                .map_err(|error| FunasrNanoExecutorError::AdapterFailed {
                    reason: error.to_string(),
                })?;

            let n_aud =
                funasr_nano_audio_token_count(encoder_output.frame_count).min(adapter_frames);
            if n_aud == 0 {
                return Err(FunasrNanoExecutorError::AdapterFailed {
                    reason: "no audio tokens produced".to_string(),
                });
            }
            let llm_dim = adapter_metadata.llm_dim;
            let speech_rows = adapter_rows[..n_aud * llm_dim].to_vec();
            Ok((speech_rows, n_aud))
        },
    )
}

fn decode_with_decoder(
    decoder: &mut FunasrNanoDecoderRuntime,
    decoder_metadata: &FunasrNanoDecoderMetadata,
    decode_prompt: &crate::models::qwen::Qwen3AsrDecodePrompt,
    speech_rows: &[f32],
    tokenizer: &FunasrNanoTokenizer,
    control: &Arc<crate::api::backend::TranscriptionControl>,
) -> Result<Seq2SeqGreedyDecodeResult, FunasrNanoExecutorError> {
    let mut token_rows =
        Vec::with_capacity(decode_prompt.token_ids.len() * decoder_metadata.d_model);
    for &token_id in &decode_prompt.token_ids {
        let row = decoder.gather_token_embedding(token_id).map_err(|error| {
            FunasrNanoExecutorError::DecoderFailed {
                reason: error.to_string(),
            }
        })?;
        token_rows.extend_from_slice(&row);
    }
    let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
        decode_prompt,
        decoder_metadata.d_model,
        token_rows,
        speech_rows,
    )
    .map_err(|error| FunasrNanoExecutorError::PromptEmbeddingFailed {
        reason: error.to_string(),
    })?;

    let layer_kv_caches = decoder.new_kv_caches(
        decode_prompt
            .token_ids
            .len()
            .saturating_add(FUNASR_NANO_MAX_GENERATED_TOKENS),
    );
    let mut step_executor = FunasrNanoGreedyStepExecutor {
        decoder,
        layer_kv_caches,
        prompt_embeddings: Some(prompt_embeddings),
        cache_prompt_tokens: 0,
        control: Arc::clone(control),
    };
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: decode_prompt.token_ids.clone(),
        eot_token_id: tokenizer.chatml_im_end_token_id,
        vocab_size: decoder_metadata.vocab_size,
        max_generated_tokens: FUNASR_NANO_MAX_GENERATED_TOKENS,
    };
    let result = run_builtin_seq2seq_decode_policy(
        FUNASR_NANO_DECODE_POLICY_ID,
        &config,
        &NoPhraseBiasTokenSource,
        None,
        &mut step_executor,
        &|token_ids: &[u32]| {
            tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                    reason: error.to_string(),
                }
            })
        },
        |error: Seq2SeqGreedyDecodeError| error,
        |error: Seq2SeqGreedyDecodeError| error,
        map_registry_error,
        control,
    )
    .map_err(|error| FunasrNanoExecutorError::GreedyDecodeFailed {
        reason: error.to_string(),
    })?;
    Ok(result)
}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

impl GgmlAsrViewExecutor for FunasrNanoGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FUNASR_NANO_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

impl GgmlAsrStreamingExecutor for FunasrNanoGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FUNASR_NANO_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FUNASR_NANO_STREAMING_EXECUTOR_ID,
            crate::arch::FUNASR_NANO_GGML_ADAPTER_ID,
            "funasr-nano",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FunasrNanoGgmlExecutor::execute_view,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// Bring-up golden: reads the committed reference LFR features + adaptor
    /// output + reference transcript for the two clips the model.pt-derived
    /// oracle produced (`OPENASR_FUNASR_NANO_GOLDEN_DIR`), plus the fp16 `.oasr`
    /// pack (`OPENASR_FUNASR_NANO_PACK`, ~1.97GB dev-only artifact, NOT
    /// committed). Runs the SAN-M encoder + transformer adaptor against the
    /// reference LFR and asserts a near-1.0 cosine similarity vs the reference
    /// adaptor output, then drives the Qwen3-0.6B decoder through the shared
    /// greedy loop and asserts the decoded transcript matches the reference
    /// text. Stays `#[ignore]`d (multi-GB pack) like every other builtin
    /// family's real-weights golden.
    fn golden_dir() -> Option<PathBuf> {
        std::env::var_os("OPENASR_FUNASR_NANO_GOLDEN_DIR").map(PathBuf::from)
    }

    fn pack_path() -> Option<PathBuf> {
        std::env::var_os("OPENASR_FUNASR_NANO_PACK").map(PathBuf::from)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        (dot / (na * nb + 1e-12)) as f32
    }

    #[test]
    #[ignore = "requires OPENASR_FUNASR_NANO_GOLDEN_DIR + the ~1.97GB dev-only \
                OPENASR_FUNASR_NANO_PACK fp16 .oasr; runs encoder+adaptor cosine parity vs the \
                model.pt oracle and end-to-end greedy decode vs the reference transcript"]
    fn golden_encoder_adapter_cosine_and_end_to_end_text() {
        let (Some(dir), Some(pack)) = (golden_dir(), pack_path()) else {
            eprintln!("skipping: set OPENASR_FUNASR_NANO_GOLDEN_DIR and OPENASR_FUNASR_NANO_PACK");
            return;
        };
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("runtime source");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&gguf_metadata).expect("encoder metadata");
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&gguf_metadata).expect("adapter metadata");
        let decoder_metadata =
            parse_funasr_nano_decoder_metadata(&gguf_metadata).expect("decoder metadata");
        let tokenizer = FunasrNanoTokenizer::from_gguf_metadata(&gguf_metadata).expect("tokenizer");

        for (tag, expected_text) in [
            (
                "en",
                "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.",
            ),
            ("zh", "开饭时间早上九点至下午五点。"),
        ] {
            let lfr = read_f32(&dir.join(format!("lfr_{tag}.bin")));
            let ref_adp = read_f32(&dir.join(format!("adp_{tag}.bin")));
            let n_frames = lfr.len() / encoder_metadata.feature_dim;

            let encoder_input = build_sensevoice_encoder_input(
                &[],
                &lfr,
                encoder_metadata.feature_dim,
                encoder_metadata.d_model,
            )
            .expect("encoder input");
            let (speech_rows_full, _) = {
                let mut encoder = FunasrNanoEncoderGraph::new(
                    &runtime_source,
                    encoder_metadata,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("encoder");
                let out = encoder
                    .encode(
                        &encoder_input.data,
                        encoder_input.n_frames,
                        encoder_input.feature_dim,
                    )
                    .expect("encode");
                let mut adapter = FunasrNanoAdapterGraph::new(
                    &runtime_source,
                    adapter_metadata,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("adapter");
                adapter
                    .run(&out.rows, out.frame_count, out.d_model)
                    .expect("adapter")
            };
            assert_eq!(
                speech_rows_full.len(),
                ref_adp.len(),
                "[{tag}] adaptor shape"
            );
            let cos = cosine(&speech_rows_full, &ref_adp);
            eprintln!("[{tag}] adaptor cosine = {cos:.6} (frames={n_frames})");
            assert!(cos > 0.999, "[{tag}] adaptor cosine {cos} below 0.999");

            // End-to-end greedy decode from the reference-derived audio rows.
            let n_aud = funasr_nano_audio_token_count(n_frames);
            let speech_rows = speech_rows_full[..n_aud * adapter_metadata.llm_dim].to_vec();
            let decode_prompt =
                build_funasr_nano_decode_prompt(&tokenizer, n_aud).expect("decode prompt");
            let mut decoder = FunasrNanoDecoderRuntime::new(
                &runtime_source,
                decoder_metadata,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("decoder");
            let control = std::sync::Arc::new(crate::api::backend::TranscriptionControl::new());
            let result = decode_with_decoder(
                &mut decoder,
                &decoder_metadata,
                &decode_prompt,
                &speech_rows,
                &tokenizer,
                &control,
            )
            .expect("decode");
            eprintln!("[{tag}] text = {}", result.text);
            assert_eq!(
                result.text.trim(),
                expected_text,
                "[{tag}] transcript mismatch"
            );
        }
    }

    /// Residency must not change output: the resident cached encoder+adaptor
    /// path (`run_encoder_and_adapter` -- cache miss on the first call, then
    /// hits at both a different and a previously seen frame count) must
    /// produce bit-for-bit the same audio-token rows as a freshly built
    /// one-shot encoder + adaptor over the same reference LFR features
    /// (the dolphin prepared-runtime bit-identity pinning pattern).
    #[test]
    #[ignore = "requires OPENASR_FUNASR_NANO_GOLDEN_DIR + the ~1.97GB dev-only \
                OPENASR_FUNASR_NANO_PACK fp16 .oasr; pins bit-identity of the resident \
                cached encoder+adaptor runtime vs a fresh one-shot build"]
    fn cached_encoder_adapter_matches_fresh_build_bit_for_bit() {
        let (Some(dir), Some(pack)) = (golden_dir(), pack_path()) else {
            eprintln!("skipping: set OPENASR_FUNASR_NANO_GOLDEN_DIR and OPENASR_FUNASR_NANO_PACK");
            return;
        };
        let _generation_guard =
            crate::models::thread_local_runtime_cache::unload_generation_test_lock();
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("runtime source");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&gguf_metadata).expect("encoder metadata");
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&gguf_metadata).expect("adapter metadata");

        // en = cache miss (build + insert), zh = cache hit at a different
        // frame count, en again = cache hit at a previously seen frame count.
        for tag in ["en", "zh", "en"] {
            let lfr = read_f32(&dir.join(format!("lfr_{tag}.bin")));
            let encoder_input = build_sensevoice_encoder_input(
                &[],
                &lfr,
                encoder_metadata.feature_dim,
                encoder_metadata.d_model,
            )
            .expect("encoder input");

            // Fresh one-shot reference: a brand-new encoder + adaptor per call.
            let mut encoder = FunasrNanoEncoderGraph::new(
                &runtime_source,
                encoder_metadata,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("fresh encoder");
            let out = encoder
                .encode(
                    &encoder_input.data,
                    encoder_input.n_frames,
                    encoder_input.feature_dim,
                )
                .expect("fresh encode");
            let mut adapter = FunasrNanoAdapterGraph::new(
                &runtime_source,
                adapter_metadata,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("fresh adapter");
            let (full_rows, adapter_frames) = adapter
                .run(&out.rows, out.frame_count, out.d_model)
                .expect("fresh adapter run");
            let fresh_n_aud = funasr_nano_audio_token_count(out.frame_count).min(adapter_frames);
            let fresh_rows = &full_rows[..fresh_n_aud * adapter_metadata.llm_dim];

            // Resident cached path (what execute_inner runs).
            let (cached_rows, cached_n_aud) = run_encoder_and_adapter(
                &runtime_source,
                encoder_metadata,
                adapter_metadata,
                &encoder_input.data,
                encoder_input.n_frames,
                encoder_input.feature_dim,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("cached encoder+adapter");

            assert_eq!(cached_n_aud, fresh_n_aud, "[{tag}] audio token count");
            assert_eq!(cached_rows.len(), fresh_rows.len(), "[{tag}] row length");
            for (index, (cached, fresh)) in cached_rows.iter().zip(fresh_rows).enumerate() {
                assert_eq!(
                    cached.to_bits(),
                    fresh.to_bits(),
                    "[{tag}] audio-token value {index} differs: cached {cached} vs fresh {fresh}"
                );
            }
            eprintln!("[{tag}] cached == fresh bit-for-bit ({cached_n_aud} audio tokens)");
        }
    }
}
