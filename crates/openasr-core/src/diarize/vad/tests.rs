//! Numerical parity tests against the upstream Silero ONNX reference.
//!
//! `silero_v6_16k_golden.bin` holds a deterministic 16 kHz clip (64 chunks of
//! JFK speech: leading silence then sustained speech) plus the per-chunk speech
//! probabilities produced by the upstream `silero_vad_16k_op15.onnx` model. The
//! pure-Rust forward pass must reproduce those probabilities closely.

use super::provider::SileroVadProvider;
use super::silero::{CHUNK_SAMPLES, SileroVadModel};
use super::test_fixtures;
use crate::longform::{LongFormOptions, LongFormVadProvider};

struct Golden {
    samples: Vec<f32>,
    probs: Vec<f32>,
}

fn load_golden() -> Golden {
    let (samples, probs) = test_fixtures::golden();
    assert_eq!(samples.len(), probs.len() * CHUNK_SAMPLES);
    Golden { samples, probs }
}

#[test]
fn forward_matches_onnx_reference_probabilities() {
    let golden = load_golden();
    let model = SileroVadModel::embedded().expect("weights");
    let probs = model.probabilities(&golden.samples);
    assert_eq!(probs.len(), golden.probs.len());

    let mut max_err = 0.0f32;
    let mut disagreements = 0usize;
    for (mine, reference) in probs.iter().zip(golden.probs.iter()) {
        assert!((0.0..=1.0).contains(mine), "prob out of range: {mine}");
        max_err = max_err.max((mine - reference).abs());
        if (*mine >= 0.5) != (*reference >= 0.5) {
            disagreements += 1;
        }
    }
    // The numpy reference reproduced ONNX at ~4e-6; allow generous f32 slack.
    assert!(max_err < 1e-3, "max abs prob error {max_err} exceeds 1e-3");
    assert_eq!(
        disagreements, 0,
        "speech/non-speech decisions must all agree"
    );
}

#[test]
fn detects_speech_region_and_leading_silence() {
    let golden = load_golden();
    let model = SileroVadModel::embedded().expect("weights");
    let probs = model.probabilities(&golden.samples);
    // The clip opens with silence and then has sustained speech.
    assert!(
        probs[0] < 0.5,
        "leading chunk should be silence: {}",
        probs[0]
    );
    let speech = probs.iter().filter(|p| **p >= 0.5).count();
    assert!(
        speech > 30,
        "expected sustained speech, got {speech} chunks"
    );
}

#[test]
fn provider_emits_speech_span_over_speech_region() {
    let golden = load_golden();
    let provider = SileroVadProvider::shared().expect("provider");
    let options = LongFormOptions::default();
    let spans = provider
        .compute_speech_slices(&golden.samples, 16_000, &options)
        .expect("slices");
    assert!(!spans.is_empty(), "expected at least one speech span");
    // The speech run should start after the leading silence (~chunk 11) and
    // cover a large contiguous region.
    let longest = spans
        .iter()
        .map(|s| s.end_sample - s.start_sample)
        .max()
        .unwrap();
    assert!(
        longest > 20 * CHUNK_SAMPLES,
        "longest speech span too short: {longest} samples"
    );
    assert!(spans[0].start_sample > 0, "first span should skip silence");
}

#[test]
fn non_16k_sample_rate_is_rejected() {
    let provider = SileroVadProvider::shared().expect("provider");
    let err = provider
        .compute_speech_slices(&[0.0; 1024], 8_000, &LongFormOptions::default())
        .unwrap_err();
    assert!(err.contains("16000"), "unexpected error: {err}");
}

#[test]
fn silence_produces_no_speech_spans() {
    let provider = SileroVadProvider::shared().expect("provider");
    let spans = provider
        .compute_speech_slices(&vec![0.0; 16_000], 16_000, &LongFormOptions::default())
        .expect("slices");
    assert!(spans.is_empty(), "silence must not yield speech spans");
}

#[test]
fn realtime_vad_prefers_neural_defaults_to_neural_with_env_precedence() {
    let saved = std::env::var("OPENASR_VAD").ok();
    // SAFETY: only this test (within the openasr-core test binary) touches the
    // OPENASR_VAD env; mutations are sequential and the original is restored.
    unsafe { std::env::remove_var("OPENASR_VAD") };

    // Default (no engine, no env) is neural; only an explicit energy/rms opts out;
    // an unrecognized engine falls through to the neural default.
    assert!(super::realtime_vad_prefers_neural(None));
    assert!(super::realtime_vad_prefers_neural(Some("silero")));
    assert!(super::realtime_vad_prefers_neural(Some("neural")));
    assert!(super::realtime_vad_prefers_neural(Some(
        "definitely-not-an-engine"
    )));
    assert!(!super::realtime_vad_prefers_neural(Some("energy")));
    assert!(!super::realtime_vad_prefers_neural(Some("rms")));

    // OPENASR_VAD wins over the explicit engine in both directions.
    unsafe { std::env::set_var("OPENASR_VAD", "energy") };
    assert!(!super::realtime_vad_prefers_neural(Some("silero")));
    unsafe { std::env::set_var("OPENASR_VAD", "neural") };
    assert!(super::realtime_vad_prefers_neural(Some("energy")));

    match saved {
        Some(value) => unsafe { std::env::set_var("OPENASR_VAD", value) },
        None => unsafe { std::env::remove_var("OPENASR_VAD") },
    }
}
