//! Decode-cadence policy for snapshot streaming **partials**.
//!
//! The snapshot streaming driver re-runs a full offline decode over the whole
//! accumulated audio buffer on every pushed frame. At a 20 ms frame cadence that
//! is far more often than a live caption needs to update, and the per-decode cost
//! grows with the buffer, so unthrottled it is the dominant source of live-caption
//! "吐字慢". This policy bounds how often a *partial* re-decode runs.
//!
//! The FINAL decode is **never** gated by this type — the driver runs it
//! unconditionally on finish — so the final transcript stays byte-identical to the
//! offline result and the golden-diff / WER-0 gate is untouched. Both levers below
//! only ever *delay* a partial decode (never add one), so they cannot change which
//! audio the final sees:
//!
//! 1. **Deterministic first-decode floor** (`first_decode_min_audio_ms`): require
//!    enough audio before the first PARTIAL so heavy encoder-decoder models do not
//!    waste a cold decode on a 20 ms fragment.
//! 2. **Deterministic steady-state floor** (`min_partial_audio_ms`): require at
//!    least this much *new* audio since the last partial decode before decoding
//!    again. Reproducible from the audio timeline alone.
//! 3. **Adaptive wall-time self-pacing**: also require the previous decode's
//!    wall-clock as new audio, so a heavy pack whose decode is slower than the floor
//!    self-throttles to ~one decode per decode-duration instead of backing up
//!    frame-by-frame. (The native server path also single-flights its Poll, so this
//!    is now a secondary guard; no extra headroom multiplier is applied, to keep
//!    partial spacing as tight as the decode allows.)
//!
//! The type is clock-free and pure: the caller (the driver) measures decode
//! wall-time and feeds it back via [`PartialDecodeCadence::record_decode`]. That
//! keeps the policy deterministic and unit-testable, and keeps `std::time` out of
//! the decode-policy logic.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PartialDecodeCadence {
    /// Steady-state floor: minimum new audio (ms) required between partial decodes.
    min_partial_audio_ms: u64,
    /// Cold-start floor: minimum accumulated audio (ms) before the first partial.
    first_decode_min_audio_ms: u64,
    /// Absolute end (ms) of the audio buffer at the last partial decode; `None`
    /// until the cold-start decode has run.
    last_decoded_end_ms: Option<u64>,
    /// Wall-clock cost (ms) of the last partial decode, for self-pacing.
    last_decode_duration_ms: u64,
}

impl PartialDecodeCadence {
    /// Cadence with a per-family audio-duration floor (ms of new audio required
    /// between partial decodes). `0` reproduces decode-on-every-frame.
    pub(crate) fn with_floor_ms(min_partial_audio_ms: u64) -> Self {
        Self {
            min_partial_audio_ms,
            first_decode_min_audio_ms: 0,
            last_decoded_end_ms: None,
            last_decode_duration_ms: 0,
        }
    }

    pub(crate) fn with_first_decode_min_audio_ms(mut self, first_decode_min_audio_ms: u64) -> Self {
        self.first_decode_min_audio_ms = first_decode_min_audio_ms;
        self
    }

    /// Decode on every pushed frame (no throttle). The driver default, preserving
    /// pre-cadence behaviour for callers that do not opt into a floor.
    pub(crate) fn every_frame() -> Self {
        Self::with_floor_ms(0)
    }

    /// Whether a partial decode should run now, given the absolute end (ms) of the
    /// accumulated audio buffer. The first decode waits for
    /// `first_decode_min_audio_ms`; after that it gates on
    /// `max(floor, last_decode_wall_ms)` of new audio.
    pub(crate) fn should_decode(&self, audio_end_ms: u64) -> bool {
        match self.last_decoded_end_ms {
            None => audio_end_ms >= self.first_decode_min_audio_ms,
            Some(last) => {
                let adaptive_ms = self.last_decode_duration_ms;
                let required_new_ms = self.min_partial_audio_ms.max(adaptive_ms);
                audio_end_ms.saturating_sub(last) >= required_new_ms
            }
        }
    }

    /// Record that a partial decode ran at `audio_end_ms`, taking
    /// `decode_duration_ms` of wall time. Call this whenever the decode runs — even
    /// if its partial was suppressed as an unchanged duplicate — so a run of
    /// suppressed partials does not re-trigger a decode on the very next frame.
    pub(crate) fn record_decode(&mut self, audio_end_ms: u64, decode_duration_ms: u64) {
        // The first (cold-start) decode pays a one-time, non-recurring weight-bind
        // cost; recording it as the self-pacing baseline would over-throttle the
        // second partial on heavy packs. Treat the cold start as zero-cost for
        // self-pacing — the steady-state floor still applies.
        let was_cold_start = self.last_decoded_end_ms.is_none();
        self.last_decoded_end_ms = Some(audio_end_ms);
        self.last_decode_duration_ms = if was_cold_start {
            0
        } else {
            decode_duration_ms
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frame_decodes_from_the_first_frame() {
        let mut cadence = PartialDecodeCadence::every_frame();
        assert!(cadence.should_decode(20)); // cold start
        cadence.record_decode(20, 0);
        // Floor 0 ⇒ any new audio decodes.
        assert!(cadence.should_decode(40));
        cadence.record_decode(40, 0);
        assert!(cadence.should_decode(60));
    }

    #[test]
    fn cold_start_decodes_then_floor_throttles() {
        let mut cadence = PartialDecodeCadence::with_floor_ms(300);
        // First frame always decodes.
        assert!(cadence.should_decode(20));
        cadence.record_decode(20, 0);

        // Then need 300 ms of new audio past the last decode (end 20).
        assert!(!cadence.should_decode(40));
        assert!(!cadence.should_decode(319));
        assert!(cadence.should_decode(320));
        cadence.record_decode(320, 0);
        assert!(!cadence.should_decode(600));
        assert!(cadence.should_decode(620));
    }

    #[test]
    fn cold_start_decode_cost_does_not_set_the_self_pacing_floor() {
        let mut cadence = PartialDecodeCadence::with_floor_ms(200);
        // A slow cold-start decode (one-time weight bind) must NOT throttle the
        // second partial — only the steady-state floor applies after it.
        cadence.record_decode(120, 999);
        assert!(!cadence.should_decode(300)); // 180 ms new < 200 floor
        assert!(cadence.should_decode(320)); // 200 ms new == floor
    }

    #[test]
    fn first_decode_waits_for_family_audio_floor() {
        let mut cadence =
            PartialDecodeCadence::with_floor_ms(200).with_first_decode_min_audio_ms(600);

        assert!(!cadence.should_decode(20));
        assert!(!cadence.should_decode(599));
        assert!(cadence.should_decode(600));
        cadence.record_decode(600, 999);

        assert!(!cadence.should_decode(799));
        assert!(cadence.should_decode(800));
    }

    #[test]
    fn self_pacing_raises_the_floor_when_decode_is_slow() {
        let mut cadence = PartialDecodeCadence::with_floor_ms(200);
        cadence.record_decode(20, 0); // cold start done, last decode ~0 ms

        // A slow decode (500 ms wall) now requires 500 ms of new audio, not 200.
        cadence.record_decode(520, 500);
        assert!(!cadence.should_decode(900)); // only 380 ms of new audio
        assert!(!cadence.should_decode(1019)); // 499 ms < 500
        assert!(cadence.should_decode(1020)); // 500 ms of new audio

        // A fast decode drops back to the floor.
        cadence.record_decode(1020, 10);
        assert!(cadence.should_decode(1220)); // 200 ms floor
    }
}
