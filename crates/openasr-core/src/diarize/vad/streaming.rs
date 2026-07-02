//! Streaming Silero VAD for the realtime path.
//!
//! Realtime frames arrive at an arbitrary, transport-chosen cadence (e.g. 20 ms
//! / 320 samples), while Silero runs on fixed 512-sample chunks with carried
//! state. This detector buffers incoming PCM into 512-sample chunks, advances
//! the model per completed chunk, and exposes the most recent speech
//! probability — which the [`crate::realtime::VadStateMachine`] consumes via
//! [`crate::realtime::VadDecision::Probability`]. Endpointing/hysteresis stays
//! entirely in the state machine; this layer only produces probabilities.

use std::fmt;

use super::silero::{CHUNK_SAMPLES, SileroVadModel, SileroVadState};

/// Buffered, stateful Silero detector for one realtime session.
pub struct SileroStreamingVad {
    model: &'static SileroVadModel,
    state: SileroVadState,
    buffer: Vec<f32>,
    last_prob: f32,
}

impl fmt::Debug for SileroStreamingVad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SileroStreamingVad")
            .field("buffered_samples", &self.buffer.len())
            .field("last_prob", &self.last_prob)
            .finish_non_exhaustive()
    }
}

impl SileroStreamingVad {
    /// Build a streaming detector over the shared model, or `None` if the model
    /// is unavailable (callers fall back to the energy gate).
    pub fn shared() -> Option<Self> {
        super::shared_model().map(|model| Self {
            model,
            state: SileroVadState::new(),
            buffer: Vec::with_capacity(CHUNK_SAMPLES * 2),
            last_prob: 0.0,
        })
    }

    /// Feed one frame of 16 kHz mono 16-bit PCM and return the current speech
    /// probability in `[0, 1]`. Samples accumulate until at least one 512-sample
    /// chunk is available; every completed chunk advances the LSTM state.
    ///
    /// The trailing `< 512` samples buffered between calls are intentionally not
    /// scored until they complete a chunk (unlike the batch path, which
    /// zero-pads the final partial). The endpointer hysteresis tolerates this
    /// ~32 ms granularity, so no `flush` is needed for correctness.
    pub fn accept_frame(&mut self, samples: &[i16]) -> f32 {
        self.buffer
            .extend(samples.iter().map(|sample| *sample as f32 / 32_768.0));
        let mut consumed = 0;
        while self.buffer.len() - consumed >= CHUNK_SAMPLES {
            let chunk = &self.buffer[consumed..consumed + CHUNK_SAMPLES];
            self.last_prob = self.model.process_chunk(chunk, &mut self.state);
            consumed += CHUNK_SAMPLES;
        }
        if consumed > 0 {
            self.buffer.drain(..consumed);
        }
        self.last_prob
    }

    /// Most recent probability without feeding new audio.
    pub fn last_probability(&self) -> f32 {
        self.last_prob
    }

    /// Clear all state for a new utterance/session.
    pub fn reset(&mut self) {
        self.state.reset();
        self.buffer.clear();
        self.last_prob = 0.0;
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn buffers_subchunk_frames_and_matches_batch_probabilities() {
        let Some(mut streaming) = SileroStreamingVad::shared() else {
            return;
        };
        let model = super::super::shared_model().unwrap();
        // 4 chunks of a 440 Hz tone (non-speech) fed in 160-sample frames.
        let total = CHUNK_SAMPLES * 4;
        let signal: Vec<f32> = (0..total)
            .map(|n| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16_000.0).sin())
            .collect();
        let pcm: Vec<i16> = signal.iter().map(|s| (s * 32_768.0) as i16).collect();

        let mut last = 0.0;
        for frame in pcm.chunks(160) {
            last = streaming.accept_frame(frame);
        }
        // The batch path over the same PCM-derived samples should agree on the
        // final chunk's probability (same model, same chunking).
        let batch =
            model.probabilities(&pcm.iter().map(|s| *s as f32 / 32_768.0).collect::<Vec<_>>());
        let expected = *batch.last().unwrap();
        assert!(
            (last - expected).abs() < 1e-4,
            "streaming {last} vs batch {expected}"
        );
    }

    #[test]
    fn detects_golden_speech_when_fed_in_realtime_frames() {
        let Some(mut streaming) = SileroStreamingVad::shared() else {
            return;
        };
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        // Feed as 20 ms (320-sample) frames, as a realtime transport would.
        let mut max_prob = 0.0f32;
        let mut silent_prefix = true;
        for (idx, frame) in pcm.chunks(320).enumerate() {
            let prob = streaming.accept_frame(frame);
            max_prob = max_prob.max(prob);
            // The clip opens with ~0.3 s of silence.
            if idx < 8 {
                silent_prefix &= prob < 0.5;
            }
        }
        assert!(silent_prefix, "leading silence misclassified as speech");
        assert!(max_prob > 0.5, "golden speech not detected: {max_prob}");
    }

    #[test]
    fn reset_clears_buffer_and_probability() {
        let Some(mut streaming) = SileroStreamingVad::shared() else {
            return;
        };
        streaming.accept_frame(&[1_000; 600]);
        streaming.reset();
        assert_eq!(streaming.last_probability(), 0.0);
    }
}
