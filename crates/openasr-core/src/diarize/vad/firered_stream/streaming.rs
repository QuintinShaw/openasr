//! Streaming Stream-VAD for the realtime path.
//!
//! Mirrors [`crate::diarize::vad::streaming::SileroStreamingVad`]'s buffering
//! contract (arbitrary-cadence PCM frames in, one probability out) but over
//! the causal Stream-VAD DFSMN instead of Silero's LSTM: incoming PCM
//! accumulates into a raw-sample buffer; whenever enough new samples have
//! arrived to complete one or more 10 ms fbank frames (25 ms window, 10 ms
//! hop), those frames are pushed through
//! [`FireRedStreamVadModel::forward_chunk`] with the carried
//! [`FireRedStreamVadCache`], and the buffer is trimmed back down to the
//! `< FRAME_LENGTH` remainder needed to complete the next frame. Because
//! Stream-VAD has no lookahead, this chunking never trades off against
//! accuracy -- the emitted probabilities are bit-identical to the
//! whole-utterance batch path over the same audio (see the `model` module
//! docs).

use std::fmt;

use super::model::{FRAME_SHIFT_MS, FireRedStreamVadCache, FireRedStreamVadModel};
use crate::diarize::vad::firered::frontend::FRAME_LENGTH;

/// Buffered, stateful Stream-VAD detector for one realtime session.
pub struct FireRedStreamingVad {
    model: &'static FireRedStreamVadModel,
    cache: FireRedStreamVadCache,
    raw_buffer: Vec<f32>,
    last_prob: f32,
}

impl fmt::Debug for FireRedStreamingVad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FireRedStreamingVad")
            .field("buffered_samples", &self.raw_buffer.len())
            .field("last_prob", &self.last_prob)
            .finish_non_exhaustive()
    }
}

impl FireRedStreamingVad {
    /// Build a streaming detector over the shared model, or `None` if the
    /// vendored weights are unavailable (callers fall back to Silero/energy).
    pub fn shared() -> Option<Self> {
        super::shared_model().map(|model| Self {
            model,
            cache: FireRedStreamVadCache::new(),
            raw_buffer: Vec::with_capacity(FRAME_LENGTH * 2),
            last_prob: 0.0,
        })
    }

    /// Feed one frame of 16 kHz mono 16-bit PCM and return the current speech
    /// probability in `[0, 1]`. Samples accumulate until at least one 10 ms
    /// fbank frame is available (25 ms of audio the first time, 10 ms of new
    /// audio thereafter); every completed frame advances the causal DFSMN
    /// cache. `10ms` granularity (`FRAME_SHIFT_MS`) is well within the
    /// endpointer's hysteresis tolerance, matching the Silero streaming
    /// contract.
    pub fn accept_frame(&mut self, samples: &[i16]) -> f32 {
        self.raw_buffer
            .extend(samples.iter().map(|sample| *sample as f32 / 32_768.0));
        if self.raw_buffer.len() < FRAME_LENGTH {
            return self.last_prob;
        }
        let (features, n_frames) = self.model.cmvn_features(&self.raw_buffer);
        if n_frames == 0 {
            return self.last_prob;
        }
        let probs = self
            .model
            .forward_chunk(&features, n_frames, &mut self.cache);
        self.last_prob = *probs.last().expect("n_frames > 0 implies non-empty probs");

        // `frontend.compute` uses snip_edges framing: frame i spans
        // `[i*FRAME_SHIFT, i*FRAME_SHIFT+FRAME_LENGTH)`. Once `n_frames` have
        // been scored, everything before `n_frames*FRAME_SHIFT` is fully
        // consumed; keep only the tail (< FRAME_LENGTH samples) needed to
        // complete the next frame once more audio arrives.
        let frame_shift_samples = (16_000u64 * FRAME_SHIFT_MS as u64 / 1000) as usize;
        let consumed = n_frames * frame_shift_samples;
        self.raw_buffer.drain(..consumed);
        self.last_prob
    }

    /// Most recent probability without feeding new audio.
    pub fn last_probability(&self) -> f32 {
        self.last_prob
    }

    /// Clear all state for a new utterance/session.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.raw_buffer.clear();
        self.last_prob = 0.0;
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn buffers_subchunk_frames_and_matches_batch_probabilities() {
        let Some(mut streaming) = FireRedStreamingVad::shared() else {
            return;
        };
        let model = super::super::shared_model().unwrap();
        // 4 s of a 440 Hz tone (non-speech) fed in 160-sample (10 ms) frames.
        let total = 16_000 * 4;
        let signal: Vec<f32> = (0..total)
            .map(|n| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16_000.0).sin())
            .collect();
        let pcm: Vec<i16> = signal.iter().map(|s| (s * 32_768.0) as i16).collect();

        let mut probs_streamed = Vec::new();
        for frame in pcm.chunks(160) {
            probs_streamed.push(streaming.accept_frame(frame));
        }
        let batch = model.probabilities(&signal);
        // The streaming detector only reports a new probability once a frame
        // completes; its last reported value should match the batch model's
        // last frame exactly (both are the same cached-forward code path).
        assert!(
            (probs_streamed.last().unwrap() - batch.last().unwrap()).abs() < 1e-4,
            "streamed {:?} vs batch {:?}",
            probs_streamed.last(),
            batch.last()
        );
    }

    #[test]
    fn detects_golden_speech_when_fed_in_realtime_frames() {
        let Some(mut streaming) = FireRedStreamingVad::shared() else {
            return;
        };
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        let mut max_prob = 0.0f32;
        let mut silent_prefix = true;
        for (idx, frame) in pcm.chunks(320).enumerate() {
            let prob = streaming.accept_frame(frame);
            max_prob = max_prob.max(prob);
            if idx < 8 {
                silent_prefix &= prob < 0.5;
            }
        }
        assert!(silent_prefix, "leading silence misclassified as speech");
        assert!(max_prob > 0.5, "golden speech not detected: {max_prob}");
    }

    #[test]
    fn reset_clears_buffer_and_probability() {
        let Some(mut streaming) = FireRedStreamingVad::shared() else {
            return;
        };
        streaming.accept_frame(&[1_000; 600]);
        streaming.reset();
        assert_eq!(streaming.last_probability(), 0.0);
    }

    #[test]
    fn chunking_granularity_does_not_change_the_result() {
        // Bit-identical regardless of how the same audio is split into
        // `accept_frame` calls -- the point of carrying an explicit cache
        // instead of recomputing over growing history.
        let Some(mut fine) = FireRedStreamingVad::shared() else {
            return;
        };
        let Some(mut coarse) = FireRedStreamingVad::shared() else {
            return;
        };
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();

        let mut fine_probs = Vec::new();
        for frame in pcm.chunks(80) {
            fine_probs.push(fine.accept_frame(frame));
        }
        let mut coarse_probs = Vec::new();
        for frame in pcm.chunks(3_200) {
            coarse_probs.push(coarse.accept_frame(frame));
        }
        assert_eq!(
            fine_probs.last().copied(),
            coarse_probs.last().copied(),
            "different chunk sizes over the same audio must converge to the same final probability"
        );
    }
}
