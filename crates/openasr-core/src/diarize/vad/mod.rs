//! Neural Voice Activity Detection.
//!
//! OpenASR ships exactly one VAD engine: FireRedVAD's **Stream-VAD**
//! checkpoint ([`firered_stream`], causal DFSMN, Apache-2.0). It is vendored
//! (~2.3 MB, baked in via `include_bytes!`, no ggml/.oasr/catalog
//! involvement), so it is always available, and loaded once into a
//! process-wide [`shared_model`] shared by the batch provider and the
//! streaming detector. Because the checkpoint is strictly causal (no
//! lookahead), the same weights back:
//!
//! - realtime endpointing ([`crate::realtime::VadMode::ExternalProbability`],
//!   via [`FireRedStreamingVad`]),
//! - long-form speech slicing (the [`crate::longform::LongFormVadProvider`]
//!   seam, via [`FireRedStreamVadProvider`]), and
//! - diarization's speech-region resolution ([`crate::diarize::pipeline`]).
//!
//! The forward pass is validated bit-close (max abs prob error < 1e-3)
//! against a numpy reference reproduction of the upstream `DetectModel`
//! forward via a committed golden fixture (see [`firered_stream::tests`]),
//! and chunked streaming is verified bit-identical to the whole-utterance
//! batch path.
//!
//! The zero-dependency RMS energy gate (the long-form `EnergyLongFormVadProvider`,
//! [`crate::realtime::VadMode::Energy`] for realtime) remains available as an
//! explicit, independently useful mode -- not a VAD engine choice, but a
//! deliberately simpler alternative for callers that want no model
//! dependency at all. Selection is surface-specific: long-form picks the
//! energy gate only via the explicit `--segment-mode energy` slicing mode
//! (`resolve_longform_vad_provider` does not consult any env var and always
//! resolves Stream-VAD for VAD-based slicing); `OPENASR_VAD=energy` only
//! affects realtime (`crate::realtime::VadMode`). There is no other neural
//! engine and no runtime engine-selection mechanism between neural
//! implementations: Stream-VAD is not optional.

mod firered_stream;

#[cfg(test)]
mod tests;

pub use firered_stream::{FireRedStreamVadError, FireRedStreamVadProvider, FireRedStreamingVad};

/// The process-wide Stream-VAD model, loaded once (~2.3 MB). Returns `None`
/// only if the vendored weights blob fails to parse (a build-integrity
/// problem, since the blob is a fixed, committed asset).
pub fn shared_model() -> Option<&'static firered_stream::FireRedStreamVadModel> {
    firered_stream::shared_model()
}

/// Single source of truth for VAD-mode selection strings. `Some(true)` selects
/// the neural detector (Stream-VAD), `Some(false)` the energy gate, `None` is
/// unrecognized. Shared by the batch, server, and CLI surfaces (and the
/// `OPENASR_VAD` env) so the alias table never diverges.
pub fn parse_vad_engine(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "neural" | "firered-stream" | "firered_stream" | "fireredstream" => Some(true),
        "energy" | "rms" => Some(false),
        _ => None,
    }
}

/// The `OPENASR_VAD` environment override as a neural-vs-energy preference.
pub fn vad_engine_env_override() -> Option<bool> {
    std::env::var("OPENASR_VAD")
        .ok()
        .as_deref()
        .and_then(parse_vad_engine)
}

/// Probability threshold for the neural (Stream-VAD) detector when the caller
/// does not specify one. Shared by the server WS and CLI `live` surfaces so
/// the operating point never drifts between them. Validated (see
/// `SHORT_NEURAL_SPEECH_STOP_MS`'s doc for the sweep methodology) against
/// `0.4`/`0.6`: threshold barely moved endpointing behavior in the tested
/// range on real narration audio, so `0.5` (the conventional operating point)
/// carries over unchanged.
pub const DEFAULT_NEURAL_VAD_THRESHOLD: f32 = 0.5;

/// Default speech-start debounce (ms) under the neural detector. Lower than
/// the energy-gate default (`VadConfig::default().speech_start_ms`, 200)
/// because the neural probability stream is less noisy than an RMS gate, so
/// the server can emit `speech_started` earlier without using VAD as an audio
/// gate. A grid sweep (60/100/150ms) over Stream-VAD's per-frame
/// probabilities on a real 5-minute narration recording showed the start
/// debounce has no effect on utterance fragmentation (only on perceived
/// latency, linearly), so `100` stays the balance point between snappy
/// `speech_started` events and tolerance for a brief false-positive blip.
pub const DEFAULT_NEURAL_SPEECH_START_MS: u32 = 100;

/// Default speech-stop hangover (ms) under the neural detector. Shorter than
/// the energy-gate default (`VadConfig::default().speech_stop_ms`, 600)
/// because Stream-VAD is far more confident about silence than an RMS gate,
/// but still long enough to avoid clipping trailing syllables in live
/// captions. Shared by the server WS and CLI `live` surfaces (a client/flag
/// value always wins; energy sessions keep 600).
///
/// Unlike the long-form hysteresis (`crate::longform::LongFormVadOptions`,
/// tuned separately for clean batch segmentation), this value is tuned for
/// realtime responsiveness: a debounce sweep (300/400/500/600ms) over the
/// same real 5-minute narration recording's Stream-VAD probabilities showed
/// `min_silence`/hangover dominates utterance fragmentation (300ms: ~86-90
/// utterances, clearly over-segmenting mid-sentence pauses; 600ms: ~47-48,
/// under-responsive for a live transcript). `500` sits at the sweet spot
/// (~59-63 utterances) between fragmentation and turnaround latency.
pub const SHORT_NEURAL_SPEECH_STOP_MS: u32 = 500;

/// Whether a realtime session should use the neural detector, given an optional
/// explicit `engine` string. Single source of truth for the realtime default
/// across the server WS and CLI `live` surfaces: `OPENASR_VAD` wins, then the
/// explicit engine, else **default to neural** -- only an explicit `energy`/`rms`
/// opts out. Each surface maps the result onto its own `VadMode` (kept here as a
/// bool so this module does not depend on the realtime `VadMode`).
pub fn realtime_vad_prefers_neural(engine: Option<&str>) -> bool {
    vad_engine_env_override().or_else(|| engine.and_then(parse_vad_engine)) != Some(false)
}

/// Shared golden-clip fixture (16 kHz JFK: leading silence then speech) used by
/// VAD tests across this crate's modules.
#[cfg(test)]
pub(crate) mod test_fixtures {
    // Same clip as `firered_stream`'s own numerical-parity golden fixture;
    // only the raw samples are needed here (the reference probabilities are
    // exercised directly by `firered_stream::tests`).
    const GOLDEN: &[u8] = include_bytes!("assets/firered_stream_vad_16k_golden.bin");

    /// Decode the golden fixture's raw 16 kHz samples.
    fn golden_samples() -> Vec<f32> {
        assert_eq!(&GOLDEN[0..4], b"FRSG", "golden magic");
        let n_samples = u32::from_le_bytes(GOLDEN[4..8].try_into().unwrap()) as usize;
        GOLDEN[12..12 + n_samples * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Golden samples as 16-bit PCM, for realtime frame-based tests.
    pub(crate) fn golden_pcm() -> Vec<i16> {
        golden_samples()
            .iter()
            .map(|s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
            .collect()
    }
}
