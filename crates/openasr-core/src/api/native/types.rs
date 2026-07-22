use std::{fmt, path::PathBuf, sync::Arc};

use crate::realtime::RealtimeSessionId;
use crate::{LongFormOptions, PhraseBiasConfig, RequestSource, TranscriptionTask};

macro_rules! impl_enum_str_display {
    ($name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAsrCapabilityClass {
    Unsupported,
    NativeModelAdapter,
    FilePerUtteranceFallback,
}

impl_enum_str_display!(NativeAsrCapabilityClass {
    Unsupported => "unsupported",
    NativeModelAdapter => "native-model-adapter",
    FilePerUtteranceFallback => "file-per-utterance-fallback",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAsrBenchmarkStatus {
    NotBenchmarked,
    BenchmarkRequired,
    Benchmarked,
}

impl_enum_str_display!(NativeAsrBenchmarkStatus {
    NotBenchmarked => "not-benchmarked",
    BenchmarkRequired => "benchmark-required",
    Benchmarked => "benchmarked",
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrCapabilities {
    pub class: NativeAsrCapabilityClass,
    pub supports_true_streaming: bool,
    pub supports_partials: bool,
    /// True only when the true-streaming session uses the frame-sync
    /// append-only partial driver (fixed low-latency chunks, never revises
    /// already-emitted text). False for buffered/windowed re-decode
    /// streaming, even though that also reports `supports_partials`.
    pub supports_frame_sync_partials: bool,
    pub supports_timestamps: bool,
    pub supports_diarization: bool,
    pub supports_phrase_bias: bool,
    pub supports_quantized_models: bool,
    pub supports_hardware_acceleration: bool,
    pub benchmark_status: NativeAsrBenchmarkStatus,
}

impl NativeAsrCapabilities {
    const fn baseline(
        class: NativeAsrCapabilityClass,
        benchmark_status: NativeAsrBenchmarkStatus,
        supports_true_streaming: bool,
    ) -> Self {
        Self {
            class,
            supports_true_streaming,
            supports_partials: false,
            supports_frame_sync_partials: false,
            supports_timestamps: false,
            supports_diarization: false,
            supports_phrase_bias: false,
            supports_quantized_models: false,
            supports_hardware_acceleration: false,
            benchmark_status,
        }
    }

    pub const fn unsupported() -> Self {
        Self::baseline(
            NativeAsrCapabilityClass::Unsupported,
            NativeAsrBenchmarkStatus::NotBenchmarked,
            false,
        )
    }

    pub const fn file_per_utterance_fallback() -> Self {
        Self::baseline(
            NativeAsrCapabilityClass::FilePerUtteranceFallback,
            NativeAsrBenchmarkStatus::NotBenchmarked,
            false,
        )
    }

    pub const fn native_offline() -> Self {
        Self::baseline(
            NativeAsrCapabilityClass::NativeModelAdapter,
            NativeAsrBenchmarkStatus::BenchmarkRequired,
            false,
        )
    }

    pub const fn native_true_streaming() -> Self {
        Self::baseline(
            NativeAsrCapabilityClass::NativeModelAdapter,
            NativeAsrBenchmarkStatus::BenchmarkRequired,
            true,
        )
    }

    pub const fn with_partial_results(mut self, supported: bool) -> Self {
        self.supports_partials = supported;
        self
    }

    pub const fn with_frame_sync_partials(mut self, supported: bool) -> Self {
        self.supports_frame_sync_partials = supported;
        self
    }

    pub const fn with_timestamps(mut self, supported: bool) -> Self {
        self.supports_timestamps = supported;
        self
    }

    pub const fn with_diarization(mut self, supported: bool) -> Self {
        self.supports_diarization = supported;
        self
    }

    pub const fn with_phrase_bias(mut self, supported: bool) -> Self {
        self.supports_phrase_bias = supported;
        self
    }

    pub const fn with_quantized_models(mut self, supported: bool) -> Self {
        self.supports_quantized_models = supported;
        self
    }

    pub const fn with_hardware_acceleration(mut self, supported: bool) -> Self {
        self.supports_hardware_acceleration = supported;
        self
    }

    pub const fn with_benchmark_status(mut self, status: NativeAsrBenchmarkStatus) -> Self {
        self.benchmark_status = status;
        self
    }

    pub fn is_native_adapter(&self) -> bool {
        self.class == NativeAsrCapabilityClass::NativeModelAdapter
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrModelPackRef {
    pub id: String,
    pub family: String,
    pub variant: Option<String>,
    pub root: PathBuf,
    pub manifest_path: Option<PathBuf>,
}

impl NativeAsrModelPackRef {
    pub fn new(id: impl Into<String>, family: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            family: family.into(),
            variant: None,
            root: root.into(),
            manifest_path: None,
        }
    }

    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }

    pub fn with_manifest_path(mut self, manifest_path: impl Into<PathBuf>) -> Self {
        self.manifest_path = Some(manifest_path.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrTensorLayoutRef {
    pub name: String,
    pub format: String,
    pub quantization: Option<String>,
}

impl NativeAsrTensorLayoutRef {
    pub fn new(name: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            format: format.into(),
            quantization: None,
        }
    }

    pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
        self.quantization = Some(quantization.into());
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeAsrRequestOptions {
    pub language: Option<String>,
    pub task: Option<TranscriptionTask>,
    pub prompt: Option<String>,
    pub phrase_bias: Option<PhraseBiasConfig>,
    pub inference_threads: Option<u16>,
    pub diarize: bool,
    pub partial_results: bool,
    pub word_timestamps: bool,
    /// Opt-in `--word-timestamps=aligned` / `word_aligned` refinement tier;
    /// see `TranscriptionRequest::word_timestamps_refine`. Offline-only:
    /// streaming sessions never consult this field.
    pub word_timestamps_refine: bool,
}

impl NativeAsrRequestOptions {
    pub fn new() -> Self {
        Self::default()
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

    pub fn with_diarization(mut self, diarize: bool) -> Self {
        self.diarize = diarize;
        self
    }

    pub fn with_partial_results(mut self, partial_results: bool) -> Self {
        self.partial_results = partial_results;
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeAsrOfflineRequest {
    pub input_path: PathBuf,
    pub options: NativeAsrRequestOptions,
    pub longform: Option<LongFormOptions>,
    pub display_file_name: Option<String>,
    /// Which call path built this request -- carried through to
    /// [`crate::TranscriptionRequest::source`] by
    /// `native_offline_request_to_transcription_request` for the
    /// `stage=request_context` `daemon.log` line. Defaults to
    /// [`RequestSource::Unspecified`]; callers set it via [`Self::with_source`].
    pub source: RequestSource,
    /// The *source* audio's real sample rate/channel count, before this
    /// crate's normalization pipeline resamples/downmixes -- same
    /// "probed/known, never fabricated" contract as
    /// [`crate::TranscriptionRequest::source_sample_rate_hz`], which this
    /// carries through to.
    pub source_sample_rate_hz: Option<u32>,
    pub source_channels: Option<u16>,
    /// The source file's container/codec extension, same
    /// "probed/known, never fabricated" contract as
    /// [`crate::TranscriptionRequest::source_container`], which this carries
    /// through to.
    pub source_container: Option<String>,
    /// Ready-to-decode 16 kHz mono f32 samples already resident in memory --
    /// same "prefer this over re-reading `input_path`" contract as
    /// [`crate::TranscriptionRequest::prepared_samples`], which this carries
    /// through to via `native_offline_request_to_transcription_request`.
    pub prepared_samples: Option<Arc<Vec<f32>>>,
}

impl NativeAsrOfflineRequest {
    pub fn new(input_path: impl Into<PathBuf>) -> Self {
        Self {
            input_path: input_path.into(),
            options: NativeAsrRequestOptions::default(),
            longform: None,
            display_file_name: None,
            source: RequestSource::default(),
            source_sample_rate_hz: None,
            source_channels: None,
            source_container: None,
            prepared_samples: None,
        }
    }

    /// Attaches in-memory samples so the native backend can skip re-reading
    /// `input_path` from disk -- see the field's doc comment.
    pub fn with_prepared_samples(mut self, prepared_samples: Option<Arc<Vec<f32>>>) -> Self {
        self.prepared_samples = prepared_samples;
        self
    }

    pub fn with_options(mut self, options: NativeAsrRequestOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_longform(mut self, longform: Option<LongFormOptions>) -> Self {
        self.longform = longform;
        self
    }

    pub fn with_display_file_name(mut self, display_file_name: Option<String>) -> Self {
        self.display_file_name = display_file_name;
        self
    }

    pub fn with_source(mut self, source: RequestSource) -> Self {
        self.source = source;
        self
    }

    /// Sets the source audio's real sample rate/channel count. Pass `None`
    /// for either when it is genuinely unknown -- never a normalization
    /// constant; see this field's doc comment.
    pub fn with_source_audio_format(
        mut self,
        sample_rate_hz: Option<u32>,
        channels: Option<u16>,
    ) -> Self {
        self.source_sample_rate_hz = sample_rate_hz;
        self.source_channels = channels;
        self
    }

    /// Sets the source file's container/codec extension. Pass the raw
    /// extension (e.g. `"m4a"`) or `None` when genuinely unknown -- never
    /// the file name.
    pub fn with_source_container(mut self, container: Option<String>) -> Self {
        self.source_container = container;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrSessionContext {
    pub session_id: RealtimeSessionId,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
}

impl NativeAsrSessionContext {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: RealtimeSessionId(session_id.into()),
            trace_id: None,
            request_id: None,
        }
    }

    pub fn from_realtime_session_id(session_id: RealtimeSessionId) -> Self {
        Self {
            session_id,
            trace_id: None,
            request_id: None,
        }
    }

    pub fn with_trace_id(mut self, trace_id: Option<String>) -> Self {
        self.trace_id = trace_id;
        self
    }

    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAsrHardwareTarget {
    Auto,
    Cpu,
    Accelerated,
    AppleSilicon,
    NvidiaCuda,
    AmdGpu,
    IntelCpu,
    IntelGpu,
    IntelNpu,
}

impl_enum_str_display!(NativeAsrHardwareTarget {
    Auto => "auto",
    Cpu => "cpu",
    Accelerated => "accelerated",
    AppleSilicon => "apple-silicon",
    NvidiaCuda => "nvidia-cuda",
    AmdGpu => "amd-gpu",
    IntelCpu => "intel-cpu",
    IntelGpu => "intel-gpu",
    IntelNpu => "intel-npu",
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeAsrRuntimeReadiness {
    Ready,
    UnsupportedModelPack { reason: String },
    MissingLocalModelAsset { path: PathBuf },
    UnsupportedHardwareTarget { target: NativeAsrHardwareTarget },
    ProviderUnavailable { provider: String },
    BackendDoesNotSupportTrueStreaming { backend: String },
}

impl NativeAsrRuntimeReadiness {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}
