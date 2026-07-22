//! firered-aed dedicated executor (Stage 4): fbank+CMVN [`frontend`] -> the
//! parity-verified Conformer [`encoder_graph`] -> greedy attention
//! [`decoder_graph`] -> char+SPM [`tokenizer`] detokenize. No CTC branch, no
//! phrase bias (pure autoregressive attention decode). The executor fails
//! closed with typed errors on a bad pack and never fabricates a transcript.
//!
//! Each call here encodes/decodes exactly one audio window ("single-segment"
//! in that sense -- there is no internal multi-slice batching, unlike
//! cohere's `batched_decode`). Long-file transcription is NOT single-shot,
//! though: the architecture-agnostic longform slicer in
//! `api::backend::native_transcribe` calls this executor once per slice for
//! every builtin family, firered-aed included, with each window pre-capped to
//! this architecture's `GlobalQuadratic` safety ceiling (issue #68) -- well
//! under the encoder's PE-table capacity. `execute_inner` still checks that
//! capacity itself and fails closed with a typed error if a window ever
//! arrives oversized (issue #158's defense-in-depth: a caller that bypasses
//! longform, or a future regression in the slicing wiring, must not reach an
//! opaque graph-allocation failure or a silently degraded transcript).
//!
//! [`frontend`]: super::frontend
//! [`encoder_graph`]: super::encoder_graph
//! [`decoder_graph`]: super::decoder_graph
//! [`tokenizer`]: super::tokenizer

#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_AED_GGML_ADAPTER_ID;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, canonical_runtime_cache_path,
    with_thread_local_cached_mut_by_key,
};

use super::decoder_graph::{
    FireRedDecoderGraphRuntime, run_firered_aed_decoder_greedy_with_runtime,
};
use super::encoder_graph::{
    FireRedEncoderGraphRuntime, FireRedEncoderOutput, predicted_encoder_time_frames,
};
use super::frontend::{FireRedFbankFrontend, apply_cmvn};
use super::graph_config::{firered_decoder_graph_config, firered_encoder_graph_config};
use super::runtime_contract::{FireRedAedExecutionMetadata, parse_firered_aed_execution_metadata};
use super::tokenizer::FireRedTokenizer;

const FIRERED_AED_EXECUTOR_ID: &str = "firered-aed-ggml-executor-v1";
const FIRERED_AED_STREAMING_EXECUTOR_ID: &str = "firered-aed-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

thread_local! {
    static FIRERED_AED_ENCODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<FireRedAedEncoderRuntimeCacheKey, FireRedEncoderGraphRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
    static FIRERED_AED_DECODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<FireRedAedDecoderRuntimeCacheKey, FireRedDecoderGraphRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
}

type FireRedAedEncoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);
/// (canonical pack path, backend, encoder frame count). The decoder's
/// cross-KV cache is allocated at a fixed size for the current utterance's
/// encoder frame count (see [`FireRedDecoderGraphRuntime::new`]), so a cached
/// runtime is only reusable across calls that share the same frame count --
/// mirrors cohere's `CohereDecoderRuntimeCacheKey` precedent.
type FireRedAedDecoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend, usize);

fn encode_with_cached_runtime(
    runtime_path: &Path,
    metadata: FireRedAedExecutionMetadata,
    cmvn_features: &[f32],
    n_frames: usize,
) -> Result<FireRedEncoderOutput, super::encoder_graph::FireRedEncoderError> {
    let key = (
        canonical_runtime_cache_path(runtime_path),
        firered_encoder_graph_config().backend,
    );
    with_thread_local_cached_mut_by_key(
        &FIRERED_AED_ENCODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || FireRedEncoderGraphRuntime::new(runtime_path, metadata),
        |runtime| runtime.encode(cmvn_features, n_frames),
    )
}

fn decode_with_cached_runtime(
    runtime_path: &Path,
    metadata: FireRedAedExecutionMetadata,
    encoder_rows: &[f32],
    encoder_frame_count: usize,
    decode_text: impl Fn(&[u32]) -> Result<String, String>,
) -> Result<
    super::decoder_graph::FireRedAedGreedyDecodeOutput,
    super::decoder_graph::FireRedDecoderError,
> {
    let key = (
        canonical_runtime_cache_path(runtime_path),
        firered_decoder_graph_config().backend,
        encoder_frame_count,
    );
    with_thread_local_cached_mut_by_key(
        &FIRERED_AED_DECODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || FireRedDecoderGraphRuntime::new(runtime_path, metadata, encoder_frame_count),
        |runtime| {
            run_firered_aed_decoder_greedy_with_runtime(
                runtime,
                metadata,
                encoder_rows,
                &decode_text,
            )
        },
    )
}

#[derive(Debug, Error)]
enum FireRedAedExecutorError {
    #[error("firered-aed executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("firered-aed executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("firered-aed runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("firered-aed tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("firered-aed cmvn vectors failed: {reason}")]
    CmvnBuildFailed { reason: String },
    #[error("firered-aed frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("firered-aed encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("firered-aed decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("firered-aed audio window ({window_seconds:.1}s) is too long for this pack: {reason}")]
    AudioWindowTooLong { window_seconds: f32, reason: String },
}

fn window_seconds(n_samples: usize, sample_rate_hz: u32) -> f32 {
    n_samples as f32 / sample_rate_hz.max(1) as f32
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FireRedAedGgmlExecutor;

impl FireRedAedGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedAedExecutorError> {
        if request.selected_family.adapter_id != FIRERED_AED_GGML_ADAPTER_ID {
            return Err(FireRedAedExecutorError::AdapterMismatch {
                expected: FIRERED_AED_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| FireRedAedExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;
        let metadata =
            parse_firered_aed_execution_metadata(&preflight.metadata).map_err(|error| {
                FireRedAedExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokens = preflight
            .metadata
            .get_string_array(TOKENIZER_TOKENS_KEY)
            .ok_or_else(|| FireRedAedExecutorError::TokenizerBuildFailed {
                reason: "pack missing tokenizer.ggml.tokens".to_string(),
            })?
            .to_vec();
        let tokenizer = FireRedTokenizer::new(tokens);

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
            FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            }
        })?;
        let feature_dim_shape = [metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;

        let samples = &request.prepared_audio.samples_f32;
        let frontend = FireRedFbankFrontend::new();
        let mut features =
            frontend
                .compute(samples)
                .map_err(|error| FireRedAedExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).map_err(
            |error| FireRedAedExecutorError::FrontendFailed {
                reason: error.to_string(),
            },
        )?;

        // Defense in depth (issue #158): the generic longform slicer in
        // `native_transcribe` already caps every window at this architecture's
        // declared `GlobalQuadratic` safety ceiling (issue #68), which is well
        // inside the encoder's baked rel-pos-table capacity below -- so this
        // should never trip in the normal request path. But this executor is
        // also reachable directly (a caller that skips longform, a future
        // regression in the slicing wiring, an oversized fixed/manual chunk
        // request), and a window past the PE table's capacity is a quality/
        // correctness problem even when it happens to fit in memory: reject it
        // with a typed, actionable error up front rather than let a caller
        // silently degrade or hit an opaque graph-allocation failure deep in
        // `encoder_graph`.
        let predicted_encoder_frames =
            predicted_encoder_time_frames(features.n_frames).map_err(|error| {
                FireRedAedExecutorError::AudioWindowTooLong {
                    window_seconds: window_seconds(
                        samples.len(),
                        request.prepared_audio.sample_rate_hz,
                    ),
                    reason: error.to_string(),
                }
            })?;
        let max_encoder_frames = metadata.encoder_max_frames();
        if predicted_encoder_frames > max_encoder_frames {
            return Err(FireRedAedExecutorError::AudioWindowTooLong {
                window_seconds: window_seconds(
                    samples.len(),
                    request.prepared_audio.sample_rate_hz,
                ),
                reason: format!(
                    "encoder frame count {predicted_encoder_frames} exceeds this pack's \
                     positional-encoding capacity of {max_encoder_frames} frames \
                     (~{:.0}s at 25 fps); this window should already be capped by \
                     longform slicing before reaching the executor",
                    max_encoder_frames as f32 / 25.0
                ),
            });
        }

        let runtime_path = preflight.runtime_source.path();
        let encoder_output =
            encode_with_cached_runtime(runtime_path, metadata, &features.data, features.n_frames)
                .map_err(|error| FireRedAedExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;

        let decode = decode_with_cached_runtime(
            runtime_path,
            metadata,
            &encoder_output.rows,
            encoder_output.frame_count,
            |ids| tokenizer.decode(ids).map_err(|error| error.to_string()),
        )
        .map_err(|error| FireRedAedExecutorError::DecoderFailed {
            reason: error.to_string(),
        })?;

        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        let text = decode.text.trim().to_string();
        let transcription = Transcription {
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            text,
            longform: None,
            language: None,
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
        })
    }
}

impl GgmlAsrExecutor for FireRedAedGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_AED_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

impl GgmlAsrStreamingExecutor for FireRedAedGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_AED_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FIRERED_AED_STREAMING_EXECUTOR_ID,
            FIRERED_AED_GGML_ADAPTER_ID,
            "firered-aed",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FireRedAedGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudio};
    use crate::models::ggml_family_registry::firered_aed_runtime_descriptor_v1;

    use super::*;

    // Pinned to the reference PyTorch decode captured by the dev-only
    // `tmp/firered-ref-src` harness (see the Stage 1-2 module docs); the
    // fp16 pack itself is a private, non-committed dev artifact.
    const GOLDEN_JFK_TEXT: &str = "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO \
         FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY";

    // Pinned to the reference PyTorch decode of `fixtures/zh_sample.wav` (a
    // macOS `say -v Tingting` synthesis of an original, non-copyrighted
    // Mandarin sentence written for this test) via the same
    // `tmp/firered-ref-src` harness. The reference tokenizer's `dict.txt` has
    // no punctuation/`<space>` entries, so the golden text is intentionally
    // punctuation-free.
    const GOLDEN_ZH_TEXT: &str = "今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的\
         川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下";

    fn dev_pack_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/firered-out/firered-aed-l-fp16.oasr")
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<String> {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "firered-aed golden test",
            "firered-aed golden test",
        )
        .expect("load wav fixture");

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: firered_aed_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };

        let executor = FireRedAedGgmlExecutor;
        let result = executor.execute(&request).expect("firered-aed transcribe");
        Some(result.transcription.text)
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_jfk_wav() {
        let Some(text) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_zh_sample_wav() {
        let Some(text) = transcribe_with_dev_pack(zh_wav_path()) else {
            return;
        };
        assert_eq!(text, GOLDEN_ZH_TEXT);
    }

    /// Demonstrates the thread-local encoder/decoder runtime cache: the
    /// second same-thread transcription of the same pack must be
    /// meaningfully faster than the first, because it skips re-loading the
    /// GGUF weight context (mmap + tensor-metadata construction) for both the
    /// encoder and the decoder. Not a strict regression gate (wall-clock,
    /// shared CI hardware) -- just an executable record of the speedup this
    /// module claims; skips silently without the dev-only pack.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn second_same_thread_transcribe_is_faster_than_first_due_to_runtime_cache() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            jfk_wav_path(),
            "firered-aed perf test",
            "firered-aed perf test",
        )
        .expect("load jfk.wav");

        let build_request = || GgmlAsrExecutionRequest {
            runtime_source_path: pack_path.clone(),
            runtime_source_preflight: None,
            selected_family: firered_aed_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples.clone()),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };
        let executor = FireRedAedGgmlExecutor;

        let first_start = std::time::Instant::now();
        let first = executor
            .execute(&build_request())
            .expect("firered-aed transcribe (first, cold runtime cache)");
        let first_elapsed = first_start.elapsed();

        let second_start = std::time::Instant::now();
        let second = executor
            .execute(&build_request())
            .expect("firered-aed transcribe (second, warm runtime cache)");
        let second_elapsed = second_start.elapsed();

        assert_eq!(first.transcription.text, GOLDEN_JFK_TEXT);
        assert_eq!(second.transcription.text, GOLDEN_JFK_TEXT);
        eprintln!("firered-aed runtime cache: first={first_elapsed:?} second={second_elapsed:?}");
        assert!(
            second_elapsed < first_elapsed,
            "expected cached (second) transcribe to be faster: first={first_elapsed:?} second={second_elapsed:?}"
        );
    }

    /// Issue #158 defense-in-depth: a single window past the pack's
    /// PE-table capacity (~200s for this dev pack's `pe_len=9999`) must fail
    /// closed with a typed `AudioWindowTooLong` error instead of reaching
    /// `encoder_graph`'s allocation and either OOM-ing or (on a machine with
    /// enough memory to actually build the graph) silently degrading past
    /// the model's trained/positional-encoding range. This should never
    /// happen on the real request path (the outer longform slicer already
    /// caps every window to the architecture's 30s safety ceiling), so this
    /// constructs an oversized window directly against the executor,
    /// bypassing the slicer, to prove the guard is load-bearing on its own.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn oversized_window_fails_closed_with_typed_error_instead_of_reaching_the_encoder_graph() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        // 210s of silence at 16 kHz: cheap to construct, and content does not
        // matter -- the guard trips on shape (predicted encoder frame count)
        // before any real encoding is attempted.
        let samples = vec![0.0_f32; 210 * 16_000];
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: firered_aed_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };
        let executor = FireRedAedGgmlExecutor;
        let error = executor
            .execute(&request)
            .expect_err("a 210s window must fail closed, not transcribe");
        let message = error.to_string();
        assert!(
            message.contains("too long") || message.contains("positional-encoding capacity"),
            "expected a PE-capacity typed error, got: {message}"
        );
    }
}
