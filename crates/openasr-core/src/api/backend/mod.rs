use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::language::LanguageMode;

mod mock;
mod native;

pub use mock::transcribe_with_mock_backend;
pub use native::{
    NativeBackend, NativeBackendExecutor, NativeRuntimeModelAdapter, NativeRuntimeModelIdSource,
    NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError,
    native_runtime_model_adapter_for_path, native_runtime_realtime_capabilities_for_path,
    native_runtime_transcription_capabilities_for_path, native_transcription_progress,
    resolve_local_native_runtime_model_identity, validate_local_native_model_pack_path,
    validate_native_runtime_model_pack_contract,
};

pub const NATIVE_RUNTIME_MODEL_ID_AUTO: &str = "__openasr_native_runtime_model_id_auto__";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Mock,
    Native,
}

impl BackendKind {
    pub const ALL: &'static [&'static str] = &["mock", "native"];
    pub const SELECTABLE: &'static [&'static str] = Self::ALL;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTarget {
    #[default]
    Auto,
    Cpu,
    Accelerated,
}

impl ExecutionTarget {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Accelerated => "accelerated",
        }
    }
}

/// Speech task selected per request. `Transcribe` keeps the audio's source
/// language; `Translate` is the Whisper-native X->English speech-translation
/// task. Family-neutral on purpose: every family flows through the same option
/// plumbing, but only whisper acts on `Translate` (others reject it explicitly).
/// Default is `Transcribe` so an omitted task is byte-identical to today.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionTask {
    #[default]
    Transcribe,
    Translate,
}

impl TranscriptionTask {
    pub const ALL: &'static [&'static str] = &["transcribe", "translate"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

impl fmt::Display for TranscriptionTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TranscriptionTask {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "transcribe" => Ok(Self::Transcribe),
            "translate" => Ok(Self::Translate),
            other => Err(format!(
                "Unsupported task '{other}'. Use one of: {}.",
                Self::ALL.join(", ")
            )),
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mock => "mock",
            Self::Native => "native",
        })
    }
}

impl FromStr for BackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "mock" => Ok(Self::Mock),
            "native" => Ok(Self::Native),
            other => Err(format!(
                "Unsupported backend '{other}'. Use one of: {}.",
                Self::SELECTABLE.join(", ")
            )),
        }
    }
}

use crate::{LongFormOptions, PhraseBiasConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendCapabilityBehavior {
    Supported,
    RejectRequest,
    MetadataOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendFeatureCapability {
    pub supported: bool,
    pub behavior: BackendCapabilityBehavior,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

impl BackendFeatureCapability {
    pub const fn supported() -> Self {
        Self {
            supported: true,
            behavior: BackendCapabilityBehavior::Supported,
            reason: None,
        }
    }

    pub const fn reject_request(reason: &'static str) -> Self {
        Self {
            supported: false,
            behavior: BackendCapabilityBehavior::RejectRequest,
            reason: Some(reason),
        }
    }

    pub const fn metadata_only(reason: &'static str) -> Self {
        Self {
            supported: false,
            behavior: BackendCapabilityBehavior::MetadataOnly,
            reason: Some(reason),
        }
    }
}

/// Per-pack source-language capability, derived from the resolved [`LanguageMode`].
/// Serialized into `/v1/capabilities` so clients present only the language
/// controls a given model actually honors. Drift-free by construction: it is
/// produced from the same mode the fail-closed gate dispatches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LanguageCapability {
    /// Stable machine tag: detect_and_specify | detect_implicit | specify_only |
    /// fixed_monolingual | fixed_multilingual.
    pub mode: &'static str,
    /// Whether omitting the language (auto) is honored. Always true.
    pub auto_supported: bool,
    /// Whether an explicit per-request language selection is honored.
    pub specify_supported: bool,
    /// The language used when none is requested (the conditioned default, or the
    /// intrinsically fixed single language).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_language: Option<&'static str>,
    /// Languages a fixed-multilingual model is built for (no per-request choice).
    /// Empty for the other modes.
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    pub fixed_languages: &'static [&'static str],
    /// Why an explicit selection is rejected, when `specify_supported` is false
    /// for a reason worth surfacing (e.g. not implemented yet).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

impl From<LanguageMode> for LanguageCapability {
    fn from(mode: LanguageMode) -> Self {
        match mode {
            LanguageMode::DetectAndSpecify => Self {
                mode: "detect_and_specify",
                auto_supported: true,
                specify_supported: true,
                default_language: None,
                fixed_languages: &[],
                reason: None,
            },
            LanguageMode::DetectImplicit { reject_reason } => Self {
                mode: "detect_implicit",
                auto_supported: true,
                specify_supported: false,
                default_language: None,
                fixed_languages: &[],
                reason: Some(reject_reason),
            },
            LanguageMode::SpecifyOnly { default_language } => Self {
                mode: "specify_only",
                auto_supported: true,
                specify_supported: true,
                default_language: Some(default_language),
                fixed_languages: &[],
                reason: None,
            },
            LanguageMode::FixedMonolingual { language } => Self {
                mode: "fixed_monolingual",
                auto_supported: true,
                specify_supported: false,
                default_language: Some(language),
                fixed_languages: &[],
                reason: None,
            },
            LanguageMode::FixedMultilingual { languages } => Self {
                mode: "fixed_multilingual",
                auto_supported: true,
                specify_supported: false,
                default_language: None,
                fixed_languages: languages,
                reason: None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TranscriptionBackendCapabilities {
    pub backend: BackendKind,
    pub segment_timestamps: BackendFeatureCapability,
    pub word_timestamps: BackendFeatureCapability,
    pub diarization: BackendFeatureCapability,
    pub phrase_bias: BackendFeatureCapability,
    pub inference_threads: BackendFeatureCapability,
    pub language: LanguageCapability,
}

impl TranscriptionBackendCapabilities {
    pub fn for_backend_kind(backend: BackendKind) -> Self {
        let unsupported_diarization = BackendFeatureCapability::reject_request(
            "Diarization is not implemented for this backend; requests with diarize=true are rejected.",
        );
        let unsupported_phrase_bias = BackendFeatureCapability::reject_request(
            "Phrase bias / hotword boosting is not implemented for this backend; requests with phrase_bias or hotword fields are rejected.",
        );
        let inference_threads = BackendFeatureCapability::supported();
        // Backend-level default; the native path overrides this per pack in
        // `native_runtime_transcription_capabilities_for_path`.
        let language = LanguageCapability::from(LanguageMode::DetectAndSpecify);

        match backend {
            BackendKind::Mock => Self {
                backend,
                segment_timestamps: BackendFeatureCapability::supported(),
                word_timestamps: BackendFeatureCapability::supported(),
                diarization: unsupported_diarization,
                phrase_bias: unsupported_phrase_bias,
                inference_threads,
                language,
            },
            BackendKind::Native => Self {
                backend,
                segment_timestamps: BackendFeatureCapability::supported(),
                word_timestamps: BackendFeatureCapability::supported(),
                diarization: unsupported_diarization,
                phrase_bias: BackendFeatureCapability::supported(),
                inference_threads,
                language,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub input_path: std::path::PathBuf,
    pub model_id: String,
    pub model_pack_path: Option<std::path::PathBuf>,
    /// OADP Phase 0: optional `.oadp` adapter pack to activate for this
    /// request (CLI `--adapter`). The native executor validates it fail-closed
    /// against the executing base pack; non-moonshine families hard-error.
    /// `None` leaves the server-side `OPENASR_ADAPTER` env surface in charge.
    pub adapter_path: Option<std::path::PathBuf>,
    pub language: Option<String>,
    pub task: Option<TranscriptionTask>,
    pub prompt: Option<String>,
    pub phrase_bias: Option<PhraseBiasConfig>,
    pub inference_threads: Option<u16>,
    pub execution_target: Option<ExecutionTarget>,
    pub word_timestamps: bool,
    /// Opt-in refinement tier (`--word-timestamps=aligned` / API
    /// `word_timestamps_mode=aligned`): after the family's own decode produces
    /// the transcript and its approximate per-word timestamps, re-run the
    /// installed Qwen3-ForcedAligner-0.6B capability pack over the finished
    /// text and full audio and replace each segment's words with the
    /// aligner-refined spans. Requires `word_timestamps` to also be `true`
    /// (checked fail-closed, not silently implied) and the capability pack to
    /// already be installed -- the native backend never downloads it.
    pub word_timestamps_refine: bool,
    pub longform: Option<LongFormOptions>,
    pub display_file_name: Option<String>,
    pub diarize: bool,
    /// Exact speaker count to force during diarization clustering (the
    /// `DiarizeHint::NumSpeakers` hint); `None` lets the threshold decide.
    pub diarize_speakers: Option<u8>,
}

impl TranscriptionRequest {
    pub fn new(input_path: impl Into<std::path::PathBuf>, model_id: impl Into<String>) -> Self {
        Self {
            input_path: input_path.into(),
            model_id: model_id.into(),
            model_pack_path: None,
            adapter_path: None,
            language: None,
            task: None,
            prompt: None,
            phrase_bias: None,
            inference_threads: None,
            execution_target: None,
            word_timestamps: false,
            word_timestamps_refine: false,
            longform: None,
            display_file_name: None,
            diarize: false,
            diarize_speakers: None,
        }
    }

    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    pub fn with_task(mut self, task: Option<TranscriptionTask>) -> Self {
        self.task = task;
        self
    }

    pub fn with_prompt(mut self, prompt: Option<String>) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_phrase_bias(mut self, phrase_bias: Option<PhraseBiasConfig>) -> Self {
        self.phrase_bias = phrase_bias;
        self
    }

    pub fn with_inference_threads(mut self, inference_threads: Option<u16>) -> Self {
        self.inference_threads = inference_threads;
        self
    }

    pub fn with_execution_target(mut self, execution_target: Option<ExecutionTarget>) -> Self {
        self.execution_target = execution_target;
        self
    }

    pub fn with_word_timestamps(mut self, word_timestamps: bool) -> Self {
        self.word_timestamps = word_timestamps;
        self
    }

    pub fn with_word_timestamps_refine(mut self, word_timestamps_refine: bool) -> Self {
        self.word_timestamps_refine = word_timestamps_refine;
        self
    }

    pub fn with_longform(mut self, longform: Option<LongFormOptions>) -> Self {
        self.longform = longform;
        self
    }

    pub fn with_model_pack_path(mut self, model_pack_path: Option<std::path::PathBuf>) -> Self {
        self.model_pack_path = model_pack_path;
        self
    }

    pub fn with_adapter_path(mut self, adapter_path: Option<std::path::PathBuf>) -> Self {
        self.adapter_path = adapter_path;
        self
    }

    pub fn with_display_file_name(mut self, display_file_name: Option<String>) -> Self {
        self.display_file_name = display_file_name;
        self
    }

    pub fn with_diarization(mut self, diarize: bool) -> Self {
        self.diarize = diarize;
        self
    }

    pub fn with_diarize_speakers(mut self, diarize_speakers: Option<u8>) -> Self {
        self.diarize_speakers = diarize_speakers;
        self
    }
}

// Serde shape is byte-for-byte the API's `JsonSegment`/`JsonWord` (see
// `format/json.rs`): same field order, same `skip_serializing_if`. This is what
// lets daemon history persist `segments_json` and hand it back to the desktop
// export UI without a second, drifting segment schema. `#[serde(default)]` on
// every skippable field makes the round-trip robust when those fields were
// omitted on write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
    /// Mean softmax probability of the decoded tokens forming this word
    /// (`0..=1`), when the family's decoder exposes per-token scores; `None`
    /// otherwise — never invented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionLongFormMetadata {
    pub chunk_count: usize,
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
    pub provenance: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<Segment>,
    pub longform: Option<TranscriptionLongFormMetadata>,
    /// Language the transcription is in (e.g. `en`). For whisper this is the
    /// auto-detected language (or the explicit `--language`); `None` for families
    /// that do not report a language.
    pub language: Option<String>,
}

pub fn add_segment_word_timestamps(transcription: &mut Transcription) {
    for segment in &mut transcription.segments {
        if !segment.words.is_empty() {
            continue;
        }
        segment.words = derive_segment_word_timestamps(segment);
    }
}

fn derive_segment_word_timestamps(segment: &Segment) -> Vec<WordTimestamp> {
    let words = segment.text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }
    let duration = (segment.end - segment.start).max(0.0);
    if duration == 0.0 {
        return words
            .into_iter()
            .map(|word| WordTimestamp {
                word: word.to_string(),
                start: segment.start,
                end: segment.start,
                confidence: None,
            })
            .collect();
    }

    let total_chars = words
        .iter()
        .map(|word| word.chars().count().max(1))
        .sum::<usize>() as f32;
    let mut cursor = segment.start;
    words
        .iter()
        .enumerate()
        .map(|(index, word)| {
            let start = cursor;
            let end = if index + 1 == words.len() {
                segment.end
            } else {
                cursor + duration * (word.chars().count().max(1) as f32 / total_chars)
            };
            cursor = end;
            WordTimestamp {
                word: (*word).to_string(),
                start,
                end,
                confidence: None,
            }
        })
        .collect()
}

pub trait TranscriptionBackend {
    fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcription, BackendError>;
}

pub(crate) fn reject_unsupported_diarization(
    request: &TranscriptionRequest,
    backend: &'static str,
) -> Result<(), BackendError> {
    if request.diarize {
        return Err(BackendError::DiarizationNotSupported { backend });
    }

    Ok(())
}

pub(crate) fn reject_unsupported_phrase_bias(
    request: &TranscriptionRequest,
    backend: &'static str,
) -> Result<(), BackendError> {
    if request
        .phrase_bias
        .as_ref()
        .is_some_and(|phrase_bias| !phrase_bias.is_empty())
    {
        return Err(BackendError::PhraseBiasNotSupported { backend });
    }

    Ok(())
}

pub(crate) fn reject_unsupported_phrase_bias_for_model(
    adapter: &'static str,
    model_family: &'static str,
    supported: bool,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<(), BackendError> {
    if supported || phrase_bias.is_none_or(PhraseBiasConfig::is_empty) {
        return Ok(());
    }

    Err(BackendError::PhraseBiasUnsupportedByModel {
        adapter: adapter.to_string(),
        model_family: model_family.to_string(),
    })
}

/// Only multilingual Whisper performs X->English speech translation.
fn family_supports_translation(adapter_id: &str) -> bool {
    adapter_id == crate::models::ggml_family_registry::WHISPER_GGML_ADAPTER_ID
}

/// A non-English source-language hint is honored only by Whisper (multilingual
/// packs) and the Cohere transcribe family; every other family — and any future
/// family until explicitly wired — fails closed.
fn family_supports_source_language(adapter_id: &str) -> bool {
    adapter_id == crate::models::ggml_family_registry::WHISPER_GGML_ADAPTER_ID
        || adapter_id == crate::models::ggml_family_registry::COHERE_TRANSCRIBE_GGML_ADAPTER_ID
}

/// True when this native family honors a non-English source-language decode
/// hint (multilingual Whisper, Cohere transcribe). The realtime server uses
/// this to decide whether the session-level translation source declaration
/// (`session.language="zh"`) may also be forwarded to the ASR session as a
/// decode hint; families that fail closed on hints they ignore must not
/// receive it.
pub fn native_adapter_supports_source_language_hint(adapter_id: &str) -> bool {
    family_supports_source_language(adapter_id)
}

/// Post-family-selection fail-closed gate: reject `task=translate` on a family
/// that cannot translate, or an explicit source language the resolved
/// [`LanguageMode`] cannot honor, naming the actual adapter. The default request
/// (`Transcribe` + unset/auto language) never trips this, so the WER-0 golden
/// path is untouched.
pub(crate) fn reject_unsupported_task_or_language(
    adapter_id: &'static str,
    language_mode: LanguageMode,
    task: TranscriptionTask,
    language: Option<&str>,
) -> Result<(), BackendError> {
    if task == TranscriptionTask::Translate && !family_supports_translation(adapter_id) {
        return Err(BackendError::RequestOptionUnsupportedByModel {
            adapter: adapter_id,
            option: "task=translate",
            reason: "Speech translation is only available on multilingual Whisper packs.",
        });
    }
    reject_unsupported_language(adapter_id, language_mode, language)
}

/// Fail-closed gate for an explicit source language, dispatched on the resolved
/// per-pack [`LanguageMode`]. An unset/empty (auto) language never trips it, so
/// the default decode path stays byte-identical.
pub(crate) fn reject_unsupported_language(
    adapter_id: &'static str,
    language_mode: LanguageMode,
    language: Option<&str>,
) -> Result<(), BackendError> {
    let requested = match language.map(str::trim) {
        None => return Ok(()),
        Some("") => return Ok(()),
        Some(language) => language,
    };
    match language_mode {
        // Explicit code accepted here; the family prompt builder validates that
        // the concrete `<|code|>` token exists and fails closed otherwise.
        LanguageMode::DetectAndSpecify | LanguageMode::SpecifyOnly { .. } => Ok(()),
        LanguageMode::DetectImplicit { reject_reason } => {
            Err(BackendError::RequestOptionUnsupportedByModel {
                adapter: adapter_id,
                option: "language",
                reason: reject_reason,
            })
        }
        LanguageMode::FixedMonolingual { language: fixed } => {
            if requested.eq_ignore_ascii_case(fixed) {
                Ok(())
            } else {
                Err(BackendError::RequestOptionUnsupportedByModel {
                    adapter: adapter_id,
                    option: "language",
                    reason: "This model transcribes a single fixed language and cannot be set to another.",
                })
            }
        }
        LanguageMode::FixedMultilingual { .. } => {
            Err(BackendError::RequestOptionUnsupportedByModel {
                adapter: adapter_id,
                option: "language",
                reason: "This model transcribes its built-in language set and does not accept a per-request language selection.",
            })
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error(
        "Diarization is not available for the {backend} backend in this setup.\nNative diarization needs the WeSpeaker speaker-embedder pack (wespeaker-voxceleb-resnet34-lm) or a self-diarizing model pack; install one, or omit --diarize / diarize=true."
    )]
    DiarizationNotSupported { backend: &'static str },
    #[error(
        "The speakers hint requires diarize=true.\nThe request was rejected instead of silently ignoring speakers."
    )]
    DiarizeSpeakersRequiresDiarization,
    #[error(
        "Phrase bias / hotword boosting is not supported by the {backend} backend yet.\nThe request was rejected instead of silently ignoring phrase_bias."
    )]
    PhraseBiasNotSupported { backend: &'static str },
    #[error(
        "Adapter packs (--adapter / .oadp) are not supported by the {backend} backend.\nThe request was rejected instead of silently ignoring the adapter."
    )]
    AdapterNotSupported { backend: &'static str },
    #[error(
        "Phrase bias / hotword boosting is not supported by the '{model_family}' native model family ({adapter}).\nThe request was rejected instead of silently ignoring phrase_bias."
    )]
    PhraseBiasUnsupportedByModel {
        adapter: String,
        model_family: String,
    },
    #[error(
        "The '{adapter}' model does not support the requested {option}.\n{reason}\nThe request was rejected instead of silently ignoring the option."
    )]
    RequestOptionUnsupportedByModel {
        adapter: &'static str,
        option: &'static str,
        reason: &'static str,
    },
    #[error(
        "Native ASR Core backend requires an explicit local runtime pack path.\nCurrent status: native stays fail-closed without a caller-provided runtime pack.\nRun with --backend native --model-pack /absolute/or/relative/path/to/model.gguf (or .oasr; active .oasr packs are GGUF-backed).\nNo remote URLs or downloads are allowed."
    )]
    NativeModelPackPathRequired,
    #[error(
        "Native ASR Core local runtime source path was rejected: {reason}\nNative execution is local-path-only and fail-closed (no remote URLs, no implicit downloads)."
    )]
    NativeModelPackPathRejected { reason: String },
    // Kept for compatibility with existing server/API error mapping while the
    // native graph execution path is still fail-closed.
    #[error(
        "Native ASR Core input format is unsupported for local inference: {reason}\nProvide 16 kHz mono PCM WAV input (or normalize before backend dispatch)."
    )]
    NativeUnsupportedInputFormat { reason: String },
    #[error(
        "Native ASR Core requested model '{requested}' does not match local runtime source model id '{local}'.\nUse the local runtime model id or omit the model override."
    )]
    NativeModelSelectionMismatch { requested: String, local: String },
    #[error(
        "Native ASR Core transcription stayed fail-closed after local runtime source validation/dispatch: {reason}\nNo partial transcript was emitted."
    )]
    NativeFailClosed { reason: String },
    #[error(
        "Native ASR Core serve-batch decode is temporarily unavailable: {reason}\nThis is a transient condition; retry the request."
    )]
    ServeBatchUnavailable { reason: String, retryable: bool },
    #[error(
        "word_timestamps_refine=true (--word-timestamps=aligned) requires word_timestamps=true.\nThe request was rejected instead of silently aligning without emitting words."
    )]
    WordTimestampAlignmentRequiresWordTimestamps,
    #[error(
        "Word-timestamp alignment refinement (--word-timestamps=aligned) is not available for the {backend} backend: the Qwen3-ForcedAligner-0.6B capability pack is not installed.\nInstall it, or use --word-timestamps for the model's own approximate timestamps."
    )]
    WordTimestampAlignmentPackMissing { backend: &'static str },
    #[error(
        "Word-timestamp alignment refinement failed: {reason}\nThe request was rejected instead of returning approximate timestamps silently relabeled as aligned."
    )]
    WordTimestampAlignmentFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_parser_accepts_mock() {
        assert_eq!("mock".parse(), Ok(BackendKind::Mock));
    }

    #[test]
    fn backend_kind_parser_accepts_native() {
        assert_eq!("native".parse(), Ok(BackendKind::Native));
    }

    #[test]
    fn backend_kind_parser_rejects_unknown_backend() {
        let error = "not-a-backend".parse::<BackendKind>().unwrap_err();
        assert!(error.contains("Unsupported backend 'not-a-backend'"));
        assert!(error.contains("mock, native"));
    }

    #[test]
    fn transcription_task_defaults_to_transcribe() {
        assert_eq!(TranscriptionTask::default(), TranscriptionTask::Transcribe);
        assert_eq!(TranscriptionTask::default().as_str(), "transcribe");
    }

    #[test]
    fn transcription_task_parser_accepts_both_tasks_case_insensitively() {
        assert_eq!("transcribe".parse(), Ok(TranscriptionTask::Transcribe));
        assert_eq!("translate".parse(), Ok(TranscriptionTask::Translate));
        assert_eq!("  Translate ".parse(), Ok(TranscriptionTask::Translate));
    }

    #[test]
    fn transcription_task_parser_rejects_unknown_task() {
        let error = "summarize".parse::<TranscriptionTask>().unwrap_err();
        assert!(error.contains("Unsupported task 'summarize'"));
        assert!(error.contains("transcribe, translate"));
    }

    #[test]
    fn transcription_task_serde_roundtrips_snake_case() {
        assert_eq!(
            serde_json::to_string(&TranscriptionTask::Translate).unwrap(),
            "\"translate\""
        );
        assert_eq!(
            serde_json::from_str::<TranscriptionTask>("\"transcribe\"").unwrap(),
            TranscriptionTask::Transcribe
        );
    }

    #[test]
    fn auto_language_never_trips_gate_for_any_mode() {
        use crate::models::ggml_family_registry::WHISPER_GGML_ADAPTER_ID;
        let modes = [
            LanguageMode::DetectAndSpecify,
            LanguageMode::DetectImplicit { reject_reason: "x" },
            LanguageMode::SpecifyOnly {
                default_language: "en",
            },
            LanguageMode::FixedMonolingual { language: "en" },
            LanguageMode::FixedMultilingual {
                languages: &["en", "zh"],
            },
        ];
        // The unset/empty/auto sentinel must never trip the gate on any mode -
        // this is the byte-identical golden-path invariant.
        for mode in modes {
            for language in [None, Some(""), Some("   ")] {
                assert!(
                    reject_unsupported_task_or_language(
                        WHISPER_GGML_ADAPTER_ID,
                        mode,
                        TranscriptionTask::Transcribe,
                        language,
                    )
                    .is_ok(),
                    "auto/unset language must never trip the gate (mode {mode:?}, language {language:?})"
                );
            }
        }
    }

    #[test]
    fn source_language_hint_capability_matches_language_gate() {
        use crate::models::ggml_family_registry::{
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID, QWEN3_ASR_GGML_ADAPTER_ID, WHISPER_GGML_ADAPTER_ID,
            XASR_ZIPFORMER_GGML_ADAPTER_ID,
        };
        // The realtime server uses this helper to decide whether the
        // translation source declaration may double as an ASR decode hint.
        assert!(native_adapter_supports_source_language_hint(
            WHISPER_GGML_ADAPTER_ID
        ));
        assert!(native_adapter_supports_source_language_hint(
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        ));
        assert!(!native_adapter_supports_source_language_hint(
            XASR_ZIPFORMER_GGML_ADAPTER_ID
        ));
        assert!(!native_adapter_supports_source_language_hint(
            QWEN3_ASR_GGML_ADAPTER_ID
        ));
    }

    #[test]
    fn language_gate_matrix_matches_decision_table() {
        use crate::models::ggml_family_registry::{
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID, MOONSHINE_GGML_ADAPTER_ID, WHISPER_GGML_ADAPTER_ID,
            XASR_ZIPFORMER_GGML_ADAPTER_ID,
        };
        let lang_ok = |adapter: &'static str, mode: LanguageMode, language: Option<&str>| {
            reject_unsupported_task_or_language(
                adapter,
                mode,
                TranscriptionTask::Transcribe,
                language,
            )
            .is_ok()
        };
        let lang_err = |adapter: &'static str, mode: LanguageMode, language: Option<&str>| {
            matches!(
                reject_unsupported_task_or_language(
                    adapter,
                    mode,
                    TranscriptionTask::Transcribe,
                    language,
                ),
                Err(BackendError::RequestOptionUnsupportedByModel {
                    option: "language",
                    ..
                })
            )
        };

        // DetectAndSpecify (multilingual whisper): any explicit code accepted at
        // the gate; the prompt builder validates the concrete token.
        assert!(lang_ok(
            WHISPER_GGML_ADAPTER_ID,
            LanguageMode::DetectAndSpecify,
            Some("fr")
        ));
        assert!(lang_ok(
            WHISPER_GGML_ADAPTER_ID,
            LanguageMode::DetectAndSpecify,
            Some("en")
        ));

        // SpecifyOnly (cohere): explicit code accepted at the gate.
        assert!(lang_ok(
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            LanguageMode::SpecifyOnly {
                default_language: "en"
            },
            Some("fr")
        ));

        // DetectImplicit (qwen): every explicit hint rejected, including en.
        let qwen = LanguageMode::DetectImplicit {
            reject_reason: "language-conditioned prompting is not implemented yet",
        };
        assert!(lang_err(WHISPER_GGML_ADAPTER_ID, qwen, Some("fr")));
        assert!(lang_err(WHISPER_GGML_ADAPTER_ID, qwen, Some("en")));

        // FixedMonolingual{en} (moonshine, whisper.en): only en accepted.
        let mono = LanguageMode::FixedMonolingual { language: "en" };
        assert!(lang_ok(MOONSHINE_GGML_ADAPTER_ID, mono, Some("en")));
        assert!(lang_ok(MOONSHINE_GGML_ADAPTER_ID, mono, Some("EN")));
        assert!(lang_err(MOONSHINE_GGML_ADAPTER_ID, mono, Some("fr")));

        // FixedMultilingual (xasr): every explicit hint rejected, even set members.
        let xasr = LanguageMode::FixedMultilingual {
            languages: &["en", "zh"],
        };
        assert!(lang_err(XASR_ZIPFORMER_GGML_ADAPTER_ID, xasr, Some("en")));
        assert!(lang_err(XASR_ZIPFORMER_GGML_ADAPTER_ID, xasr, Some("zh")));
        assert!(lang_err(XASR_ZIPFORMER_GGML_ADAPTER_ID, xasr, Some("fr")));
    }

    #[test]
    fn translate_gate_is_whisper_only() {
        use crate::models::ggml_family_registry::{
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID, WHISPER_GGML_ADAPTER_ID,
        };
        // Whisper honors translate.
        assert!(
            reject_unsupported_task_or_language(
                WHISPER_GGML_ADAPTER_ID,
                LanguageMode::DetectAndSpecify,
                TranscriptionTask::Translate,
                Some("fr"),
            )
            .is_ok()
        );
        // Cohere takes a source language but cannot translate.
        assert!(matches!(
            reject_unsupported_task_or_language(
                COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
                LanguageMode::SpecifyOnly {
                    default_language: "en"
                },
                TranscriptionTask::Translate,
                None,
            ),
            Err(BackendError::RequestOptionUnsupportedByModel {
                option: "task=translate",
                ..
            })
        ));
    }

    #[test]
    fn language_capability_does_not_drift_from_gate() {
        // The advertised capability is produced from the same mode the gate
        // dispatches on, so the two must never disagree.
        let modes = [
            LanguageMode::DetectAndSpecify,
            LanguageMode::DetectImplicit {
                reject_reason: "nope",
            },
            LanguageMode::SpecifyOnly {
                default_language: "en",
            },
            LanguageMode::FixedMonolingual { language: "en" },
            LanguageMode::FixedMultilingual {
                languages: &["en", "zh"],
            },
        ];
        let adapter = "test-adapter";
        for mode in modes {
            let cap = LanguageCapability::from(mode);
            // Auto is always honored and never trips the gate.
            assert!(cap.auto_supported);
            assert!(reject_unsupported_language(adapter, mode, None).is_ok());

            if cap.specify_supported {
                assert!(
                    reject_unsupported_language(adapter, mode, Some("fr")).is_ok(),
                    "{}: advertised specify_supported but gate rejected a code",
                    cap.mode
                );
            } else {
                assert!(
                    reject_unsupported_language(adapter, mode, Some("fr")).is_err(),
                    "{}: not specify_supported but gate accepted a foreign code",
                    cap.mode
                );
            }

            // The advertised default must itself pass the gate.
            if let Some(default) = cap.default_language {
                assert!(
                    reject_unsupported_language(adapter, mode, Some(default)).is_ok(),
                    "{}: advertised default '{default}' rejected by gate",
                    cap.mode
                );
            }
        }
    }

    #[test]
    fn current_backend_capabilities_expose_unsupported_options() {
        for backend in [BackendKind::Mock, BackendKind::Native] {
            let capabilities = TranscriptionBackendCapabilities::for_backend_kind(backend);
            assert_eq!(capabilities.backend, backend);
            assert!(capabilities.segment_timestamps.supported);
            assert!(capabilities.word_timestamps.supported);
            assert_eq!(
                capabilities.word_timestamps.behavior,
                BackendCapabilityBehavior::Supported
            );
            assert!(!capabilities.diarization.supported);
            assert_eq!(
                capabilities.phrase_bias.supported,
                backend == BackendKind::Native
            );
            assert!(capabilities.inference_threads.supported);
            assert_eq!(
                capabilities.inference_threads.behavior,
                BackendCapabilityBehavior::Supported
            );
        }
    }

    #[test]
    fn segment_word_timestamps_are_distributed_within_segment_bounds() {
        let mut transcription = Transcription {
            text: "hello world".to_string(),
            segments: vec![Segment {
                start: 1.0,
                end: 3.0,
                text: "hello world".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
        };

        add_segment_word_timestamps(&mut transcription);

        assert_eq!(transcription.segments[0].words.len(), 2);
        assert_eq!(transcription.segments[0].words[0].word, "hello");
        assert_eq!(transcription.segments[0].words[0].start, 1.0);
        assert!(transcription.segments[0].words[0].end <= 3.0);
        assert_eq!(transcription.segments[0].words[1].word, "world");
        assert_eq!(transcription.segments[0].words[1].end, 3.0);
    }
}
