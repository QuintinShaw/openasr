//! [`LongFormVadProvider`] backed by the neural Silero model.
//!
//! The model emits one speech probability per 32 ms chunk; this layer turns that
//! sequence into speech spans using the same threshold / min-speech /
//! min-silence hysteresis the energy provider uses, so the long-form `Auto`
//! planner can weigh it against the other candidates on equal footing.

use thiserror::Error;

use super::silero::{CHUNK_SAMPLES, SAMPLE_RATE_HZ, SileroVadModel};
use super::weights::SileroWeightsError;
use crate::longform::{
    LongFormOptions, LongFormVadProvider, LongFormVadProviderKind, LongFormVadSlice,
};

/// Milliseconds of audio per probability (one 512-sample chunk at 16 kHz).
const CHUNK_MS: u32 = 32;

#[derive(Debug, Error)]
pub enum SileroVadError {
    #[error("silero VAD model is unavailable: {0}")]
    Unavailable(#[from] SileroWeightsError),
}

/// Neural VAD provider over the process-wide shared model. Cheap to construct
/// (it only borrows the model), so build one per request as needed.
pub struct SileroVadProvider {
    model: &'static SileroVadModel,
}

impl SileroVadProvider {
    /// Borrow the shared Silero model. Returns `None` when the vendored weights
    /// could not be loaded (callers fall back to the energy gate).
    pub fn shared() -> Option<Self> {
        super::shared_model().map(|model| Self { model })
    }

    /// Direct access to per-chunk probabilities, for diagnostics/tests.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        self.model.probabilities(samples)
    }
}

impl LongFormVadProvider for SileroVadProvider {
    fn provider_kind(&self) -> LongFormVadProviderKind {
        // Custom (not EnergyLike) so the Auto planner exercises it as a distinct
        // candidate against the energy gate.
        LongFormVadProviderKind::Custom
    }

    fn compute_speech_slices(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
    ) -> Result<Vec<LongFormVadSlice>, String> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(format!(
                "silero VAD requires {SAMPLE_RATE_HZ} Hz mono audio, got {sample_rate_hz} Hz"
            ));
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let probs = self.model.probabilities(samples);
        Ok(spans_from_probs(&probs, samples.len(), options))
    }
}

/// Convert per-chunk speech probabilities into sample-space speech spans with
/// threshold gating plus min-speech / min-silence hysteresis.
fn spans_from_probs(
    probs: &[f32],
    total_samples: usize,
    options: &LongFormOptions,
) -> Vec<LongFormVadSlice> {
    let threshold = options.vad.threshold.clamp(0.0, 1.0);
    let min_speech_chunks = ms_to_chunks(options.vad.min_speech_duration_ms);
    let min_silence_chunks = ms_to_chunks(options.vad.min_silence_duration_ms);

    let mut spans = Vec::new();
    let mut in_speech = false;
    let mut speech_start = 0usize;
    let mut trailing_silence = 0usize;

    for (idx, &prob) in probs.iter().enumerate() {
        if prob >= threshold {
            if !in_speech {
                in_speech = true;
                speech_start = idx;
            }
            trailing_silence = 0;
            continue;
        }
        if !in_speech {
            continue;
        }
        trailing_silence += 1;
        if trailing_silence < min_silence_chunks {
            continue;
        }
        let speech_end = idx + 1 - trailing_silence;
        push_span(
            &mut spans,
            speech_start,
            speech_end,
            min_speech_chunks,
            total_samples,
        );
        in_speech = false;
        trailing_silence = 0;
    }
    if in_speech {
        let speech_end = probs.len() - trailing_silence;
        push_span(
            &mut spans,
            speech_start,
            speech_end,
            min_speech_chunks,
            total_samples,
        );
    }
    spans
}

fn push_span(
    spans: &mut Vec<LongFormVadSlice>,
    start_chunk: usize,
    end_chunk: usize,
    min_speech_chunks: usize,
    total_samples: usize,
) {
    if end_chunk <= start_chunk || end_chunk - start_chunk < min_speech_chunks {
        return;
    }
    let start_sample = (start_chunk * CHUNK_SAMPLES).min(total_samples);
    let end_sample = (end_chunk * CHUNK_SAMPLES).min(total_samples);
    if end_sample > start_sample {
        spans.push(LongFormVadSlice {
            start_sample,
            end_sample,
        });
    }
}

fn ms_to_chunks(ms: u32) -> usize {
    (ms.div_ceil(CHUNK_MS)).max(1) as usize
}
