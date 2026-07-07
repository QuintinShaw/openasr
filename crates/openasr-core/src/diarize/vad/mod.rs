//! Neural Voice Activity Detection.
//!
//! A pure-Rust forward pass of Silero VAD v6.2 (16 kHz, MIT) that drops into the
//! existing [`crate::longform::LongFormVadProvider`] seam with no new trait, so
//! it improves speech slicing for every ASR model, and feeds the realtime
//! [`crate::realtime::VadMode::ExternalProbability`] path for streaming
//! endpointing. The reference energy gate
//! ([`crate::longform::EnergyLongFormVadProvider`]) stays the zero-dependency
//! fallback.
//!
//! The model and weights are vendored (the model is ~1.2 MB infrastructure, not
//! a user-pulled ASR pack), so neural VAD is always available, and loaded once
//! into a process-wide [`shared_model`] shared by the batch provider and the
//! streaming detector. The forward pass is validated bit-close (max abs prob
//! error < 1e-3; the measured numpy-vs-ONNX delta is ~4e-6) against the upstream
//! ONNX reference via a committed golden fixture (see `tests`).
//!
//! [`firered`] vendors a second, alternative long-form-only neural engine
//! (`FireRedVAD`, Apache-2.0): a causal-FSMN `DetectModel`, selectable via
//! `OPENASR_VAD=firered` but not the default (see
//! [`crate::longform::LongFormVadEngine::FireRed`] for why). It is not wired
//! into realtime endpointing or diarization.

mod firered;
mod provider;
mod silero;
mod streaming;
mod weights;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

pub use firered::{FireRedVadError, FireRedVadProvider};
pub use provider::{SileroVadError, SileroVadProvider};
pub use silero::{SileroVadModel, SileroVadState};
pub use streaming::SileroStreamingVad;

static SHARED_MODEL: OnceLock<Option<SileroVadModel>> = OnceLock::new();

/// The process-wide Silero model, loaded once (~1.2 MB). Returns `None` if the
/// vendored weights fail to parse, so callers fall back to the energy gate.
pub fn shared_model() -> Option<&'static SileroVadModel> {
    SHARED_MODEL
        .get_or_init(|| SileroVadModel::embedded().ok())
        .as_ref()
}

/// Single source of truth for VAD-engine selection strings. `Some(true)` selects
/// the neural detector, `Some(false)` the energy gate, `None` is unrecognized.
/// Shared by the batch, server, and CLI surfaces (and the `OPENASR_VAD` env) so
/// the alias table never diverges.
pub fn parse_vad_engine(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "silero" | "neural" => Some(true),
        "energy" | "rms" => Some(false),
        _ => None,
    }
}

/// The `OPENASR_VAD` environment override as a neural-vs-energy preference.
/// Realtime-only (the `firered` engine is long-form-only; an
/// `OPENASR_VAD=firered` value is unrecognized here and falls through to the
/// caller's explicit engine / default, exactly like any other unknown string).
pub fn vad_engine_env_override() -> Option<bool> {
    std::env::var("OPENASR_VAD")
        .ok()
        .as_deref()
        .and_then(parse_vad_engine)
}

/// Long-form VAD-engine selection strings, three-way (Silero / Energy /
/// FireRed). Superset of [`parse_vad_engine`] for the long-form slicing path,
/// which -- unlike realtime endpointing -- supports the FireRedVAD engine.
pub fn parse_longform_vad_engine(value: &str) -> Option<crate::longform::LongFormVadEngine> {
    use crate::longform::LongFormVadEngine;
    match value.trim().to_ascii_lowercase().as_str() {
        "silero" | "neural" => Some(LongFormVadEngine::Silero),
        "energy" | "rms" => Some(LongFormVadEngine::Energy),
        "firered" | "fireredvad" => Some(LongFormVadEngine::FireRed),
        _ => None,
    }
}

/// The `OPENASR_VAD` environment override for long-form VAD-engine selection
/// (tri-state: Silero/Energy/FireRed). Single source of truth for the
/// `resolve_longform_vad_provider` env override so the alias table never
/// diverges from [`parse_longform_vad_engine`].
pub fn longform_vad_engine_env_override() -> Option<crate::longform::LongFormVadEngine> {
    std::env::var("OPENASR_VAD")
        .ok()
        .as_deref()
        .and_then(parse_longform_vad_engine)
}

/// Probability threshold for the neural (Silero) detector when the caller does not
/// specify one. 0.5 is the standard Silero operating point. Shared by the server WS
/// and CLI `live` surfaces so the operating point never drifts between them.
pub const DEFAULT_NEURAL_VAD_THRESHOLD: f32 = 0.5;

/// Default speech-start debounce (ms) under the neural detector. Lower than the
/// energy-gate default (`VadConfig::default().speech_start_ms`, 200) because the
/// neural probability stream is less noisy than an RMS gate, so the server can
/// emit `speech_started` earlier without using VAD as an audio gate.
pub const DEFAULT_NEURAL_SPEECH_START_MS: u32 = 100;

/// Default speech-stop hangover (ms) under the neural detector. Shorter than the
/// energy-gate default (`VadConfig::default().speech_stop_ms`, 600) because Silero
/// is far more confident about silence than an RMS gate, but still long enough to
/// avoid clipping trailing syllables in live captions. Shared by the server WS and
/// CLI `live` surfaces (a client/flag value always wins; energy sessions keep 600).
pub const SHORT_NEURAL_SPEECH_STOP_MS: u32 = 500;

/// Whether a realtime session should use the neural detector, given an optional
/// explicit `engine` string. Single source of truth for the realtime default
/// across the server WS and CLI `live` surfaces: `OPENASR_VAD` wins, then the
/// explicit engine, else **default to neural** — only an explicit `energy`/`rms`
/// opts out. Each surface maps the result onto its own `VadMode` (kept here as a
/// bool so this module does not depend on the realtime `VadMode`).
pub fn realtime_vad_prefers_neural(engine: Option<&str>) -> bool {
    vad_engine_env_override().or_else(|| engine.and_then(parse_vad_engine)) != Some(false)
}

/// Shared golden-clip fixture (16 kHz JFK: leading silence then speech) used by
/// VAD tests across this crate's modules.
#[cfg(test)]
pub(crate) mod test_fixtures {
    const GOLDEN: &[u8] = include_bytes!("assets/silero_v6_16k_golden.bin");

    /// Decode the golden fixture into `(samples, reference per-chunk probs)`.
    pub(crate) fn golden() -> (Vec<f32>, Vec<f32>) {
        assert_eq!(&GOLDEN[0..4], b"SLRG", "golden magic");
        let n_samples = u32::from_le_bytes(GOLDEN[4..8].try_into().unwrap()) as usize;
        let n_chunks = u32::from_le_bytes(GOLDEN[8..12].try_into().unwrap()) as usize;
        let mut off = 12;
        let mut read = |n: usize| -> Vec<f32> {
            let out = GOLDEN[off..off + n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += n * 4;
            out
        };
        let samples = read(n_samples);
        let probs = read(n_chunks);
        (samples, probs)
    }

    /// Golden samples as 16-bit PCM, for realtime frame-based tests.
    pub(crate) fn golden_pcm() -> Vec<i16> {
        golden()
            .0
            .iter()
            .map(|s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
            .collect()
    }
}
