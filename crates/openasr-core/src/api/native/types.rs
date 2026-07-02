use std::{fmt, path::PathBuf};

use crate::realtime::RealtimeSessionId;
use crate::{LongFormOptions, PhraseBiasConfig, TranscriptionTask};

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeAsrOfflineRequest {
    pub input_path: PathBuf,
    pub options: NativeAsrRequestOptions,
    pub longform: Option<LongFormOptions>,
    pub display_file_name: Option<String>,
}

impl NativeAsrOfflineRequest {
    pub fn new(input_path: impl Into<PathBuf>) -> Self {
        Self {
            input_path: input_path.into(),
            options: NativeAsrRequestOptions::default(),
            longform: None,
            display_file_name: None,
        }
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
