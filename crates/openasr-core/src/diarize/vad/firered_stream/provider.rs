//! [`LongFormVadProvider`] backed by the causal Stream-VAD DFSMN model: the
//! sole long-form VAD engine, run over the whole long-form utterance.

use thiserror::Error;

use super::frontend::SAMPLE_RATE_HZ;
use super::model::{FRAME_SHIFT_MS, FireRedStreamVadModel};
use super::streaming::FireRedStreamingVad;
use super::weights::FireRedStreamVadWeightsError;
use crate::longform::{
    LongFormOptions, LongFormVadProvider, LongFormVadProviderKind, LongFormVadSlice,
};

#[derive(Debug, Error)]
pub enum FireRedStreamVadError {
    #[error("firered Stream-VAD model is unavailable: {0}")]
    Unavailable(#[from] FireRedStreamVadWeightsError),
    #[error("firered Stream-VAD requires {expected} Hz mono audio, got {actual} Hz")]
    UnsupportedSampleRate { expected: u32, actual: u32 },
    #[error("firered Stream-VAD was canceled")]
    Canceled,
}

/// Neural VAD provider over the process-wide shared Stream-VAD model. Cheap
/// to construct (it only borrows the model), so build one per request.
pub struct FireRedStreamVadProvider {
    model: &'static FireRedStreamVadModel,
}

impl FireRedStreamVadProvider {
    /// Borrow the shared Stream-VAD model. Returns `None` when the vendored
    /// weights could not be loaded.
    pub fn shared() -> Option<Self> {
        super::shared_model().map(|model| Self { model })
    }

    /// Direct access to per-frame probabilities, for diagnostics/tests.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        self.model.probabilities(samples)
    }

    /// Offline speech slicing with bounded cancellation latency. One second
    /// of PCM is frontended and scored at a time while the causal DFSMN cache
    /// and the fbank overlap tail remain continuous, so output matches the
    /// batch model without making a long recording one uninterruptible call.
    pub(crate) fn compute_speech_slices_cancellable(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<LongFormVadSlice>, FireRedStreamVadError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(FireRedStreamVadError::UnsupportedSampleRate {
                expected: SAMPLE_RATE_HZ,
                actual: sample_rate_hz,
            });
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let mut streaming = FireRedStreamingVad::from_model(self.model);
        let mut probabilities = Vec::with_capacity(samples.len().div_ceil(FRAME_SAMPLES));
        for chunk in samples.chunks(SAMPLE_RATE_HZ as usize) {
            if canceled() {
                return Err(FireRedStreamVadError::Canceled);
            }
            probabilities.extend(streaming.accept_f32_chunk(chunk));
        }
        if canceled() {
            return Err(FireRedStreamVadError::Canceled);
        }
        Ok(spans_from_probs(&probabilities, samples.len(), options))
    }
}

impl LongFormVadProvider for FireRedStreamVadProvider {
    fn provider_kind(&self) -> LongFormVadProviderKind {
        LongFormVadProviderKind::Custom
    }

    fn compute_speech_slices(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
    ) -> Result<Vec<LongFormVadSlice>, String> {
        self.compute_speech_slices_cancellable(samples, sample_rate_hz, options, &|| false)
            .map_err(|error| error.to_string())
    }
}

/// Samples consumed per probability frame (10 ms at 16 kHz).
const FRAME_SAMPLES: usize = (SAMPLE_RATE_HZ as u64 * FRAME_SHIFT_MS as u64 / 1000) as usize;

/// Convert per-frame speech probabilities into sample-space speech spans with
/// threshold gating plus min-speech / min-silence hysteresis.
fn spans_from_probs(
    probs: &[f32],
    total_samples: usize,
    options: &LongFormOptions,
) -> Vec<LongFormVadSlice> {
    let threshold = options.vad.threshold.clamp(0.0, 1.0);
    let min_speech_frames = ms_to_frames(options.vad.min_speech_duration_ms);
    let min_silence_frames = ms_to_frames(options.vad.min_silence_duration_ms);

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
        if trailing_silence < min_silence_frames {
            continue;
        }
        let speech_end = idx + 1 - trailing_silence;
        push_span(
            &mut spans,
            speech_start,
            speech_end,
            min_speech_frames,
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
            min_speech_frames,
            total_samples,
        );
    }
    spans
}

fn push_span(
    spans: &mut Vec<LongFormVadSlice>,
    start_frame: usize,
    end_frame: usize,
    min_speech_frames: usize,
    total_samples: usize,
) {
    if end_frame <= start_frame || end_frame - start_frame < min_speech_frames {
        return;
    }
    let start_sample = (start_frame * FRAME_SAMPLES).min(total_samples);
    let end_sample = (end_frame * FRAME_SAMPLES).min(total_samples);
    if end_sample > start_sample {
        spans.push(LongFormVadSlice {
            start_sample,
            end_sample,
        });
    }
}

fn ms_to_frames(ms: u32) -> usize {
    (ms.div_ceil(FRAME_SHIFT_MS)).max(1) as usize
}
