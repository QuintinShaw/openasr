//! [`LongFormVadProvider`] backed by the causal Stream-VAD DFSMN model, used
//! to run Stream-VAD over a whole long-form utterance (mode "Stream-VAD as
//! longform" -- research question: does one causal-only checkpoint suffice
//! for both realtime and long-form, or do the two engines need to stay
//! separate). Not the default; see [`crate::longform::LongFormVadEngine::FireRedStream`].

use thiserror::Error;

use super::model::{FRAME_SHIFT_MS, FireRedStreamVadModel};
use super::weights::FireRedStreamVadWeightsError;
use crate::diarize::vad::firered::frontend::SAMPLE_RATE_HZ;
use crate::longform::{
    LongFormOptions, LongFormVadProvider, LongFormVadProviderKind, LongFormVadSlice,
};

#[derive(Debug, Error)]
pub enum FireRedStreamVadError {
    #[error("firered Stream-VAD model is unavailable: {0}")]
    Unavailable(#[from] FireRedStreamVadWeightsError),
}

/// Neural VAD provider over the process-wide shared Stream-VAD model. Cheap
/// to construct (it only borrows the model), so build one per request.
pub struct FireRedStreamVadProvider {
    model: &'static FireRedStreamVadModel,
}

impl FireRedStreamVadProvider {
    /// Borrow the shared Stream-VAD model. Returns `None` when the vendored
    /// weights could not be loaded (callers fall back to Silero/energy).
    pub fn shared() -> Option<Self> {
        super::shared_model().map(|model| Self { model })
    }

    /// Direct access to per-frame probabilities, for diagnostics/tests.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        self.model.probabilities(samples)
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
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(format!(
                "firered Stream-VAD requires {SAMPLE_RATE_HZ} Hz mono audio, got {sample_rate_hz} Hz"
            ));
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let probs = self.model.probabilities(samples);
        Ok(spans_from_probs(&probs, samples.len(), options))
    }
}

/// Samples consumed per probability frame (10 ms at 16 kHz).
const FRAME_SAMPLES: usize = (SAMPLE_RATE_HZ as u64 * FRAME_SHIFT_MS as u64 / 1000) as usize;

/// Convert per-frame speech probabilities into sample-space speech spans with
/// threshold gating plus min-speech / min-silence hysteresis. Identical logic
/// to `firered::provider::spans_from_probs` (kept as a small, family-local
/// copy rather than a shared helper -- the two providers' frame cadence
/// happens to match today but are independent checkpoints).
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
