mod errors;
mod streaming;
mod traits;
mod transcript_emitter;
mod types;

pub use super::audio_io::load_wav_16khz_mono_f32_v0 as load_native_wav_16khz_mono_f32_v0;
pub use errors::NativeAsrError;
pub use streaming::{NativeAsrBackpressurePolicy, NativeAsrStreamingSessionConfig};
pub use traits::{NativeAsrExecutor, NativeAsrModelAdapter, NativeAsrSession};
pub(crate) use transcript_emitter::NativeStreamingTranscriptEmitter;
pub use types::{
    NativeAsrBenchmarkStatus, NativeAsrCapabilities, NativeAsrCapabilityClass,
    NativeAsrHardwareTarget, NativeAsrModelPackRef, NativeAsrOfflineRequest,
    NativeAsrRequestOptions, NativeAsrRuntimeReadiness, NativeAsrSessionContext,
    NativeAsrTensorLayoutRef,
};

#[cfg(test)]
mod tests;
