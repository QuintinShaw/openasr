//! Numerical parity (Rust causal forward pass vs. a numpy reference
//! reproduction of the upstream torch `DetectModel` with `N2 = 0`) and
//! provider smoke tests for Stream-VAD.

use super::model::FireRedStreamVadModel;
use super::provider::FireRedStreamVadProvider;

/// Golden fixture: the same 3 s (48,000-sample) excerpt of `fixtures/jfk.wav`
/// as `firered_vad_16k_golden.bin`, plus reference per-10ms-frame speech
/// probabilities from a numpy reproduction of the upstream `DetectModel`
/// forward with `N2 = 0` (no lookahead) run against the vendored
/// `Stream-VAD/model.pth.tar` + `Stream-VAD/cmvn.ark` checkpoint (there is no
/// upstream Python streaming-VAD "batch" entrypoint to diff against directly;
/// the reference forward is the same math this module implements, checked
/// independently against the checkpoint's raw tensors). Binary layout: magic
/// `"FRSG"`, `u32 n_samples`, `u32 n_frames`, `f32[n_samples]` samples,
/// `f32[n_frames]` reference probs (all little-endian).
fn golden() -> (Vec<f32>, Vec<f32>) {
    const GOLDEN: &[u8] = include_bytes!("../assets/firered_stream_vad_16k_golden.bin");
    assert_eq!(&GOLDEN[0..4], b"FRSG", "golden magic");
    let n_samples = u32::from_le_bytes(GOLDEN[4..8].try_into().unwrap()) as usize;
    let n_frames = u32::from_le_bytes(GOLDEN[8..12].try_into().unwrap()) as usize;
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
    let probs = read(n_frames);
    (samples, probs)
}

fn max_abs_diff_with_location(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "frame count mismatch");
    let mut worst = 0.0f32;
    let mut worst_idx = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_idx = i;
        }
    }
    (worst, worst_idx)
}

#[test]
fn forward_pass_matches_reference_within_tolerance() {
    let (samples, reference_probs) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let probs = model.probabilities(&samples);

    assert_eq!(probs.len(), reference_probs.len());
    let (max_diff, at) = max_abs_diff_with_location(&probs, &reference_probs);
    assert!(
        max_diff < 1e-3,
        "max abs prob error {max_diff} at frame {at} (got {}, want {}) exceeds tolerance",
        probs[at],
        reference_probs[at],
    );
}

#[test]
fn chunked_streaming_forward_matches_batch_forward_bit_close() {
    // The load-bearing invariant this whole module exists for: chunking the
    // same audio through the cached streaming forward must reproduce the
    // whole-utterance batch forward, since Stream-VAD has no lookahead.
    use super::streaming::FireRedStreamingVad;

    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let batch_probs = model.probabilities(&samples);

    let pcm: Vec<i16> = samples
        .iter()
        .map(|s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect();
    let mut streaming = FireRedStreamingVad::shared().expect("shared Stream-VAD streaming model");
    let mut streamed_last = 0.0f32;
    // An odd, non-frame-aligned chunk size (37 samples) to stress the
    // raw-buffer bookkeeping.
    for frame in pcm.chunks(37) {
        streamed_last = streaming.accept_frame(frame);
    }
    let (max_diff, _) = max_abs_diff_with_location(
        &[streamed_last],
        &[*batch_probs.last().expect("non-empty batch probs")],
    );
    assert!(
        max_diff < 1e-4,
        "chunked streaming forward diverged from batch forward by {max_diff}"
    );
}

#[test]
fn probabilities_are_finite_and_in_unit_range() {
    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let probs = model.probabilities(&samples);
    assert!(!probs.is_empty());
    assert!(
        probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
    );
}

#[test]
fn empty_audio_returns_no_probabilities() {
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    assert!(model.probabilities(&[]).is_empty());
}

#[test]
fn shared_model_loads() {
    assert!(super::shared_model().is_some());
}

#[test]
fn provider_shared_computes_speech_slices_on_golden_clip() {
    use crate::longform::{LongFormOptions, LongFormVadProvider};

    let (samples, _) = golden();
    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let options = LongFormOptions::default();
    let slices = provider
        .compute_speech_slices(
            &samples,
            crate::diarize::vad::firered::frontend::SAMPLE_RATE_HZ,
            &options,
        )
        .expect("speech slices");
    assert!(!slices.is_empty(), "expected at least one speech span");
    for slice in &slices {
        assert!(slice.end_sample > slice.start_sample);
        assert!(slice.end_sample <= samples.len());
    }
}

#[test]
fn provider_rejects_wrong_sample_rate() {
    use crate::longform::{LongFormOptions, LongFormVadProvider};

    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let samples = vec![0.0f32; 8_000];
    let err = provider
        .compute_speech_slices(&samples, 8_000, &LongFormOptions::default())
        .expect_err("wrong sample rate must fail closed");
    assert!(err.contains("16000"));
}
