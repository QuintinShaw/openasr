use crate::realtime::RealtimeAudioFormat;

use super::NativeAsrError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeAsrBackpressurePolicy {
    pub max_queued_audio_frames: usize,
    pub max_queued_events: usize,
}

const NATIVE_ASR_MIN_QUEUED_EVENTS: usize = 4;

impl Default for NativeAsrBackpressurePolicy {
    fn default() -> Self {
        Self {
            max_queued_audio_frames: 64,
            max_queued_events: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrStreamingSessionConfig {
    pub audio_format: RealtimeAudioFormat,
    pub backpressure: NativeAsrBackpressurePolicy,
    pub partial_results: bool,
    pub word_timestamps: bool,
    /// Minimum new audio (ms) between *partial* re-decodes, throttling live-caption
    /// emission so the engine does not re-decode the whole buffer on every 20 ms
    /// frame. `None` defers to the per-family default; `Some(0)` decodes every
    /// frame. Does not affect the FINAL transcript.
    pub min_partial_interval_ms: Option<u32>,
}

impl Default for NativeAsrStreamingSessionConfig {
    fn default() -> Self {
        Self {
            audio_format: RealtimeAudioFormat::pcm16_mono_16khz(),
            backpressure: NativeAsrBackpressurePolicy::default(),
            partial_results: false,
            word_timestamps: false,
            min_partial_interval_ms: None,
        }
    }
}

impl NativeAsrStreamingSessionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_audio_format(mut self, audio_format: RealtimeAudioFormat) -> Self {
        self.audio_format = audio_format;
        self
    }

    pub fn with_backpressure(mut self, backpressure: NativeAsrBackpressurePolicy) -> Self {
        self.backpressure = backpressure;
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

    pub fn with_min_partial_interval_ms(mut self, min_partial_interval_ms: Option<u32>) -> Self {
        self.min_partial_interval_ms = min_partial_interval_ms;
        self
    }

    pub fn validate(&self) -> Result<(), NativeAsrError> {
        self.audio_format
            .validate_normalized()
            .map_err(|error| NativeAsrError::invalid_streaming_session_config(error.to_string()))?;
        if self.backpressure.max_queued_audio_frames == 0 {
            return Err(NativeAsrError::invalid_streaming_session_config(
                "max_queued_audio_frames must be greater than 0",
            ));
        }
        if self.backpressure.max_queued_events < NATIVE_ASR_MIN_QUEUED_EVENTS {
            return Err(NativeAsrError::invalid_streaming_session_config(format!(
                "max_queued_events must be at least {NATIVE_ASR_MIN_QUEUED_EVENTS}"
            )));
        }
        Ok(())
    }
}
