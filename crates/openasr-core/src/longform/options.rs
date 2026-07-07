use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongFormMode {
    Off,
    Auto,
    Fixed,
    Energy,
    Vad,
}

/// Long-form (Stream-VAD) hysteresis parameters. Tuned for the "clean
/// segmentation" long-form use case (favor fewer, well-bounded slices over
/// low latency, since there is no live listener) rather than realtime
/// endpointing -- see `crate::diarize::vad`'s `DEFAULT_NEURAL_VAD_THRESHOLD` /
/// `DEFAULT_NEURAL_SPEECH_START_MS` / `SHORT_NEURAL_SPEECH_STOP_MS` for the
/// separately-tuned realtime defaults.
///
/// Values validated by a grid sweep (threshold in `0.2..=0.7`,
/// `min_silence_duration_ms` in `150..=1000`, `min_speech_duration_ms` in
/// `80..=250`) over Stream-VAD's per-frame probabilities on a real 5-minute
/// narration recording (`black_cat_poe_ty_5min.wav`): `min_silence_duration_ms`
/// dominates slice granularity (150ms fragments into 100+ sub-sentence spans;
/// 1000ms under-segments into spans up to ~60s that then get re-split by the
/// chunk-length cap anyway), while `threshold` and `min_speech_duration_ms`
/// have only marginal effect in the tested ranges. 450/250/0.5 -- the
/// pre-existing defaults -- sit in the flat, well-behaved part of that
/// surface (64 spans / 242.7s retained speech / 13.6s max span at th=0.5), so
/// they carry over unchanged from the pre-Stream-VAD engine.
#[derive(Debug, Clone, PartialEq)]
pub struct LongFormVadOptions {
    pub threshold: f32,
    pub min_speech_duration_ms: u32,
    pub min_silence_duration_ms: u32,
}

impl Default for LongFormVadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 450,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LongFormOptions {
    pub mode: LongFormMode,
    pub chunk_seconds: f32,
    pub overlap_seconds: f32,
    pub min_chunk_seconds: f32,
    pub max_chunk_seconds: f32,
    pub padding_seconds: f32,
    pub energy_silence_threshold_db: f32,
    pub energy_split_search_seconds: f32,
    pub suppress_silent_slices: bool,
    pub carry_prompt_across_slices: bool,
    pub max_context_chars: usize,
    pub fallback_to_energy_when_vad_unavailable: bool,
    pub fallback_to_energy_when_vad_empty: bool,
    pub vad: LongFormVadOptions,
}

impl Default for LongFormOptions {
    fn default() -> Self {
        Self {
            mode: LongFormMode::Auto,
            chunk_seconds: 30.0,
            overlap_seconds: 0.5,
            min_chunk_seconds: 1.0,
            max_chunk_seconds: 120.0,
            padding_seconds: 0.25,
            energy_silence_threshold_db: -38.0,
            energy_split_search_seconds: 5.0,
            suppress_silent_slices: false,
            carry_prompt_across_slices: true,
            max_context_chars: 512,
            fallback_to_energy_when_vad_unavailable: true,
            fallback_to_energy_when_vad_empty: true,
            vad: LongFormVadOptions::default(),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LongFormOptionsError {
    #[error("longform chunk_seconds must be finite and > 0, got {value}")]
    InvalidChunkSeconds { value: f32 },
    #[error("longform overlap_seconds must be finite and >= 0, got {value}")]
    InvalidOverlapSeconds { value: f32 },
    #[error("longform overlap_seconds {overlap_seconds} must be < chunk_seconds {chunk_seconds}")]
    OverlapExceedsChunk {
        overlap_seconds: f32,
        chunk_seconds: f32,
    },
    #[error("longform min_chunk_seconds must be finite and > 0, got {value}")]
    InvalidMinChunkSeconds { value: f32 },
    #[error("longform max_chunk_seconds must be finite and >= chunk_seconds, got {value}")]
    InvalidMaxChunkSeconds { value: f32 },
    #[error(
        "longform min_chunk_seconds {min_chunk_seconds} must be <= chunk_seconds {chunk_seconds}"
    )]
    MinChunkExceedsChunk {
        min_chunk_seconds: f32,
        chunk_seconds: f32,
    },
    #[error("longform padding_seconds must be finite and >= 0, got {value}")]
    InvalidPaddingSeconds { value: f32 },
    #[error("longform energy_split_search_seconds must be finite and > 0, got {value}")]
    InvalidEnergySearchSeconds { value: f32 },
    #[error("longform max_context_chars must be > 0")]
    InvalidMaxContextChars,
    #[error("longform vad.threshold must be finite and between 0 and 1, got {value}")]
    InvalidVadThreshold { value: f32 },
}

impl LongFormOptions {
    pub fn validate(&self) -> Result<(), LongFormOptionsError> {
        if !self.chunk_seconds.is_finite() || self.chunk_seconds <= 0.0 {
            return Err(LongFormOptionsError::InvalidChunkSeconds {
                value: self.chunk_seconds,
            });
        }
        if !self.overlap_seconds.is_finite() || self.overlap_seconds < 0.0 {
            return Err(LongFormOptionsError::InvalidOverlapSeconds {
                value: self.overlap_seconds,
            });
        }
        if self.overlap_seconds >= self.chunk_seconds {
            return Err(LongFormOptionsError::OverlapExceedsChunk {
                overlap_seconds: self.overlap_seconds,
                chunk_seconds: self.chunk_seconds,
            });
        }
        if !self.min_chunk_seconds.is_finite() || self.min_chunk_seconds <= 0.0 {
            return Err(LongFormOptionsError::InvalidMinChunkSeconds {
                value: self.min_chunk_seconds,
            });
        }
        if self.min_chunk_seconds > self.chunk_seconds {
            return Err(LongFormOptionsError::MinChunkExceedsChunk {
                min_chunk_seconds: self.min_chunk_seconds,
                chunk_seconds: self.chunk_seconds,
            });
        }
        if !self.max_chunk_seconds.is_finite() || self.max_chunk_seconds < self.chunk_seconds {
            return Err(LongFormOptionsError::InvalidMaxChunkSeconds {
                value: self.max_chunk_seconds,
            });
        }
        if !self.padding_seconds.is_finite() || self.padding_seconds < 0.0 {
            return Err(LongFormOptionsError::InvalidPaddingSeconds {
                value: self.padding_seconds,
            });
        }
        if !self.energy_split_search_seconds.is_finite() || self.energy_split_search_seconds <= 0.0
        {
            return Err(LongFormOptionsError::InvalidEnergySearchSeconds {
                value: self.energy_split_search_seconds,
            });
        }
        if self.max_context_chars == 0 {
            return Err(LongFormOptionsError::InvalidMaxContextChars);
        }
        if !self.vad.threshold.is_finite() || self.vad.threshold < 0.0 || self.vad.threshold > 1.0 {
            return Err(LongFormOptionsError::InvalidVadThreshold {
                value: self.vad.threshold,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_validate() {
        LongFormOptions::default().validate().unwrap();
    }

    #[test]
    fn overlap_must_be_less_than_chunk() {
        let options = LongFormOptions {
            overlap_seconds: 30.0,
            ..LongFormOptions::default()
        };
        let error = options.validate().unwrap_err();
        assert!(matches!(
            error,
            LongFormOptionsError::OverlapExceedsChunk { .. }
        ));
    }
}
