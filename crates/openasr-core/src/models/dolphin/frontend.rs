//! Dolphin frontends: kaldi-fbank (cn-dialect, WeNet format) and ESPnet
//! `DefaultFrontend` (multilingual) + global CMVN.
//!
//! [`DolphinFbankFrontend`] reproduces the reference feature pipeline used to fit
//! `small.cn`/`cn-dialect-base`: `torchaudio.compliance.kaldi.fbank` with the
//! `train.yaml` `fbank_conf` (25 ms window, 10 ms shift, 80 mel bins, `dither=0`)
//! over the waveform scaled by `1 << 15` (float `[-1, 1]` back to the int16
//! magnitude kaldi expects), followed by the checkpoint's `global_cmvn`
//! `(x - mean) * istd`.
//!
//! The fbank math itself is the shared [`crate::models::kaldi_fbank`] engine:
//! Povey window, `remove_dc_offset`, pre-emphasis 0.97, HTK mel scale over
//! 20-8000 Hz, `snip_edges=true`, and the `log(max(energy, eps))` floor -- the
//! firered_aed and sensevoice frontends run the identical computation over the
//! same config (see that module's doc). What's Dolphin-local is the CMVN affine
//! below (sign convention + tensor name) and the ESPnet frontend that follows.
//!
//! Parity: the pre-CMVN log-mel reproduces the golden `logmel_feats` fixture to a
//! max abs diff ~1e-4 (f32 FFT/mel noise), and `(x - mean) * istd` reproduces
//! `logmel_feats_cmvn` (the exact tensor fed to the encoder). Both are exercised
//! by the executor's end-to-end harness against the committed golden.
//!
//! [`DolphinEspnetFrontend`] is the multilingual (dolphin-small/dolphin-base)
//! counterpart: those checkpoints' `train.yaml` carries a `dataset_conf`
//! with *both* a `fbank_conf` and a `frontend_conf` key, and DataoceanAI's
//! own `dolphin/processor.py::extract_feats` picks the ESPnet `DefaultFrontend`
//! (`frontend_conf`) whenever it is present, ignoring `fbank_conf` entirely --
//! confirmed by reading `DataoceanAI/Dolphin`'s inference source directly
//! (`dolphin/model.py::Stft`/`LogMel`/`DefaultFrontend`), not assumed. That
//! frontend is a materially different pipeline from the kaldi one above: a
//! periodic Hann window (not Povey), `torch.stft`-style reflect-centered
//! framing (not kaldi `snip_edges`), no pre-emphasis/DC-removal/int16 scaling,
//! and a **Slaney**-normalized librosa mel scale/filterbank (not HTK
//! peak-normalized, and f64-precision throughout --
//! [`crate::models::audio_frontend::mel::MelScale::SlaneyF64Espnet`]). The
//! STFT framing/windowing/FFT and mel filterbank construction now go through
//! the shared [`crate::models::audio_frontend`] primitives (same arithmetic,
//! relocated); the log-energy floor placement (before `.ln()`, no CMVN here
//! since ESPnet checkpoints bake normalization into the encoder) stays
//! family-local.

#![allow(dead_code)]

use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale};
use crate::models::audio_frontend::{PadMode, StftFramer, hann_window_centered};
use crate::models::kaldi_fbank::{
    KaldiFbankConfig, KaldiFbankError, KaldiFbankFrontend, KaldiWindowKind,
};

/// 16 kHz, 25 ms window / 10 ms shift, 80 mel bins (train.yaml `fbank_conf`).
pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const NUM_MEL_BINS: usize = 80;

const FRONTEND_CONFIG: KaldiFbankConfig = KaldiFbankConfig {
    sample_rate_hz: SAMPLE_RATE_HZ,
    frame_length: 400, // 25 ms @ 16 kHz
    frame_shift: 160,  // 10 ms @ 16 kHz
    fft_size: 512,     // next pow2 >= 400 (kaldi rounds the window up)
    num_mel_bins: NUM_MEL_BINS,
    mel_low_hz: 20.0,
    mel_high_hz: 8_000.0, // high_freq <= 0 in kaldi => Nyquist (8 kHz)
    preemph_coeff: 0.97,
    input_scale: 32_768.0, // float [-1, 1] -> int16 magnitude
    log_energy_floor: 1.192_092_9e-7,
    window: KaldiWindowKind::Povey,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinFrontendError {
    #[error("dolphin frontend requires finite 16 kHz mono f32 audio")]
    UnsupportedAudio,
    #[error("dolphin frontend produced no fbank frames from {samples} samples")]
    NoFrames { samples: usize },
    #[error(
        "dolphin global_cmvn vector '{name}' has {actual} values, expected {expected} (feature dim)"
    )]
    CmvnLen {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin fbank produced {frames}x{stride} features, expected feature dim {expected}")]
    FeatureShape {
        frames: usize,
        stride: usize,
        expected: usize,
    },
}

impl From<KaldiFbankError> for DolphinFrontendError {
    fn from(error: KaldiFbankError) -> Self {
        match error {
            KaldiFbankError::UnsupportedAudio => Self::UnsupportedAudio,
            KaldiFbankError::NoFrames { samples } => Self::NoFrames { samples },
        }
    }
}

/// Pre-CMVN log-mel features, row-major `[frame][mel]` (mel innermost), matching
/// the golden `logmel_feats` layout and the tensor the encoder graph consumes.
pub(crate) type DolphinFbankFeatures = crate::models::kaldi_fbank::KaldiFbankFeatures;

pub(crate) struct DolphinFbankFrontend {
    inner: KaldiFbankFrontend,
}

impl std::fmt::Debug for DolphinFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DolphinFbankFrontend")
            .finish_non_exhaustive()
    }
}

impl Default for DolphinFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl DolphinFbankFrontend {
    pub(crate) fn new() -> Self {
        Self {
            inner: KaldiFbankFrontend::new(FRONTEND_CONFIG),
        }
    }

    /// Compute pre-CMVN kaldi log-mel features for 16 kHz mono `samples`
    /// (float in `[-1, 1]`). `snip_edges=true`: frame count is
    /// `1 + (len - FRAME_LENGTH) / FRAME_SHIFT`, frame `i` starts at
    /// `i * FRAME_SHIFT` with no edge reflection.
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<DolphinFbankFeatures, DolphinFrontendError> {
        Ok(self.inner.compute(samples)?)
    }
}

/// The inverse of `DolphinFbankFrontend::compute`'s frame-count formula
/// (`1 + (len - FRAME_LENGTH) / FRAME_SHIFT`): the fewest 16 kHz samples
/// guaranteed to produce at least `min_frames` snip-edges fbank frames. Lets a
/// caller (the streaming driver) reject/skip a too-short trailing window
/// before ever computing features, using the same framing constants
/// `compute` does so the two cannot drift apart.
pub(crate) fn kaldi_min_samples_for_frames(min_frames: usize) -> usize {
    if min_frames == 0 {
        return 0;
    }
    FRONTEND_CONFIG.frame_length + (min_frames - 1) * FRONTEND_CONFIG.frame_shift
}

/// Apply the checkpoint's `global_cmvn` `(x - mean) * istd` in place, per mel bin.
/// `features` is row-major `[frames, feature_dim]`; `mean`/`istd` are the
/// `encoder.global_cmvn.*` vectors baked in the pack (both length `feature_dim`).
pub(crate) fn apply_global_cmvn(
    features: &mut [f32],
    feature_dim: usize,
    mean: &[f32],
    istd: &[f32],
) -> Result<(), DolphinFrontendError> {
    if mean.len() != feature_dim {
        return Err(DolphinFrontendError::CmvnLen {
            name: "encoder.global_cmvn.mean",
            expected: feature_dim,
            actual: mean.len(),
        });
    }
    if istd.len() != feature_dim {
        return Err(DolphinFrontendError::CmvnLen {
            name: "encoder.global_cmvn.istd",
            expected: feature_dim,
            actual: istd.len(),
        });
    }
    if !features.len().is_multiple_of(feature_dim) {
        return Err(DolphinFrontendError::FeatureShape {
            frames: features.len() / feature_dim.max(1),
            stride: feature_dim,
            expected: feature_dim,
        });
    }
    for frame in features.chunks_exact_mut(feature_dim) {
        for ((value, m), s) in frame.iter_mut().zip(mean).zip(istd) {
            *value = (*value - *m) * *s;
        }
    }
    Ok(())
}

// --- ESPnet DefaultFrontend (multilingual dolphin-small/dolphin-base) -------

/// `frontend_conf` shared by dolphin-small and dolphin-base `train.yaml`:
/// `n_fft: 512, win_length: 400, hop_length: 160, fs: 16000`. `LogMel`'s own
/// defaults fill in the rest (unset in `frontend_conf`): `n_mels=80`,
/// `fmin=0`, `fmax=fs/2`, `htk=False` (librosa Slaney mel).
const ESPNET_N_FFT: usize = 512;
const ESPNET_WIN_LENGTH: usize = 400;
const ESPNET_HOP_LENGTH: usize = 160;
pub(crate) const ESPNET_NUM_MEL_BINS: usize = 80;
const ESPNET_FMIN_HZ: f64 = 0.0;
/// `LogMel.__init__`'s `fmax = fs / 2` default (16 kHz Nyquist).
const ESPNET_FMAX_HZ: f64 = 8_000.0;
/// `torch.clamp(mel_feat, min=1e-10)` before `.log()` in `LogMel.forward`.
const ESPNET_LOG_ENERGY_FLOOR: f32 = 1.0e-10;

pub(crate) struct DolphinEspnetFrontend {
    framer: StftFramer,
    /// Row-major `[n_mels][fft_bins]` Slaney mel filterbank
    /// (`MelScale::SlaneyF64Espnet`'s f64-precision edge placement).
    mel_filters: Vec<f32>,
    n_mels: usize,
    fft_bins: usize,
}

impl std::fmt::Debug for DolphinEspnetFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DolphinEspnetFrontend")
            .field("n_mels", &self.n_mels)
            .finish_non_exhaustive()
    }
}

impl Default for DolphinEspnetFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl DolphinEspnetFrontend {
    pub(crate) fn new() -> Self {
        // Periodic Hann window (`torch.hann_window(win_length)` default,
        // `periodic=True`), zero-padded/centered into the `n_fft`-length FFT
        // input buffer, mirroring `torch.stft`'s own window centering when
        // `win_length < n_fft`.
        let framer = StftFramer::new(
            ESPNET_N_FFT,
            ESPNET_WIN_LENGTH,
            ESPNET_HOP_LENGTH,
            PadMode::ReflectCenter,
            hann_window_centered(ESPNET_WIN_LENGTH, ESPNET_N_FFT),
        );
        let mel_filters = crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::SlaneyF64Espnet,
            sample_rate_hz: SAMPLE_RATE_HZ as f32,
            n_fft: ESPNET_N_FFT,
            n_mels: ESPNET_NUM_MEL_BINS,
            fmin: ESPNET_FMIN_HZ as f32,
            fmax: ESPNET_FMAX_HZ as f32,
            // `SlaneyF64Espnet` places edges with its own f64 formula and
            // ignores this field.
            mel_point_order: MelPointOrder::SpanTimesIndexFirst,
        });
        Self {
            framer,
            mel_filters,
            n_mels: ESPNET_NUM_MEL_BINS,
            fft_bins: ESPNET_N_FFT / 2 + 1,
        }
    }

    /// Compute pre-CMVN ESPnet log-mel features for 16 kHz mono `samples`
    /// (float in `[-1, 1]`). `torch.stft(..., center=True, pad_mode="reflect")`
    /// framing: the signal is conceptually reflect-padded by `n_fft/2` on each
    /// side (exactly `win_length`'s slack either side of `hop_length`), giving
    /// `n_frames = 1 + samples.len() / hop_length` (the padding cancels the
    /// `n_fft` term because `2 * (n_fft/2) == n_fft`). No pre-emphasis, DC
    /// removal, or dither: this frontend runs directly on the float PCM the
    /// caller already decoded (no int16 rescale, unlike the kaldi frontend).
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<DolphinFbankFeatures, DolphinFrontendError> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err(DolphinFrontendError::UnsupportedAudio);
        }
        let spectrogram = self
            .framer
            .power_spectrogram(samples)
            // `ReflectCenter` pads by `n_fft/2` both sides, so `TooShort`
            // never actually triggers for the nonempty input already
            // checked above (see `audio_frontend`'s own equivalent note).
            .map_err(|_| DolphinFrontendError::UnsupportedAudio)?;
        let n_frames = spectrogram.n_frames;
        let n_mels = self.n_mels;
        let fft_bins = self.fft_bins;

        let mut feats = vec![0.0f32; n_frames * n_mels];
        for fr in 0..n_frames {
            let power = &spectrogram.data[fr * fft_bins..(fr + 1) * fft_bins];
            for m in 0..n_mels {
                let row = &self.mel_filters[m * fft_bins..(m + 1) * fft_bins];
                let mut energy = 0.0f32;
                for bin in 0..fft_bins {
                    energy += row[bin] * power[bin];
                }
                feats[fr * n_mels + m] = energy.max(ESPNET_LOG_ENERGY_FLOOR).ln();
            }
        }
        Ok(DolphinFbankFeatures {
            data: feats,
            n_frames,
            n_mels,
        })
    }
}

/// The inverse of `DolphinEspnetFrontend::compute`'s frame-count formula
/// (`1 + samples.len() / HOP_LENGTH`, per that method's doc comment): the
/// fewest 16 kHz samples guaranteed to produce at least `min_frames`
/// reflect-centered STFT frames. See `kaldi_min_samples_for_frames` for the
/// `CnDialect` counterpart; the two schemes' framing differs (reflect-padded
/// vs. snip-edges) so each frontend owns its own inverse.
pub(crate) fn espnet_min_samples_for_frames(min_frames: usize) -> usize {
    if min_frames == 0 {
        return 0;
    }
    (min_frames - 1) * ESPNET_HOP_LENGTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_80_bin_snip_edges_fbank() {
        // 2380 ms of audio -> 236 frames (1 + (38080 - 400) / 160), matching the
        // golden clip's frame count.
        let samples = vec![0.01f32; 38_080];
        let features = DolphinFbankFrontend::new()
            .compute(&samples)
            .expect("fbank");
        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 236);
        assert_eq!(features.data.len(), 236 * 80);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_too_short_audio() {
        let samples = vec![0.0f32; 200];
        assert!(matches!(
            DolphinFbankFrontend::new().compute(&samples),
            Err(DolphinFrontendError::NoFrames { samples: 200 })
        ));
    }

    #[test]
    fn global_cmvn_applies_per_bin_affine() {
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        apply_global_cmvn(&mut feats, 2, &[0.5, 1.0], &[2.0, 0.5]).expect("cmvn");
        // row0: ((1-0.5)*2, (2-1)*0.5) = (1.0, 0.5); row1: ((3-0.5)*2, (4-1)*0.5).
        assert_eq!(feats, vec![1.0, 0.5, 5.0, 1.5]);
    }

    #[test]
    fn global_cmvn_rejects_mismatched_vector() {
        let mut feats = vec![0.0; 4];
        assert!(apply_global_cmvn(&mut feats, 2, &[0.0], &[1.0, 1.0]).is_err());
    }

    // --- ESPnet DefaultFrontend parity (dolphin-small/dolphin-base) --------
    //
    // Golden values below are from a direct port of
    // `dolphin/model.py::Stft`+`LogMel` (the exact classes DataoceanAI's own
    // inference code runs for these checkpoints), computed with
    // `torch.stft(n_fft=512, win_length=400, hop_length=160, window=hann_window(400),
    // center=True, pad_mode="reflect")` -> power -> `librosa.filters.mel(sr=16000,
    // n_fft=512, n_mels=80, fmin=0, fmax=8000, htk=False) @ power` -> `clamp(1e-10).log()`,
    // over `fixtures/jfk.wav` (16 kHz mono, 176000 samples).

    fn jfk_samples() -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
        crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &path,
            "dolphin_espnet_frontend_test",
            "jfk.wav",
        )
        .expect("load jfk.wav")
    }

    #[test]
    fn slaney_mel_filterbank_matches_librosa_reference() {
        let fft_bins = ESPNET_N_FFT / 2 + 1;
        let filters = crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::SlaneyF64Espnet,
            sample_rate_hz: SAMPLE_RATE_HZ as f32,
            n_fft: ESPNET_N_FFT,
            n_mels: ESPNET_NUM_MEL_BINS,
            fmin: 0.0,
            fmax: 8_000.0,
            mel_point_order: MelPointOrder::SpanTimesIndexFirst,
        });
        assert_eq!(filters.len(), 80 * fft_bins);

        // librosa.filters.mel(...)[0][:10]
        let expected_row0: [f32; 10] = [
            0.0,
            2.253_456e-2,
            8.637_710_5e-3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (bin, &expected) in expected_row0.iter().enumerate() {
            let actual = filters[bin];
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "mel row0 bin{bin}: actual {actual} expected {expected}"
            );
        }

        // librosa.filters.mel(...)[79][240:257]
        let expected_row79_tail: [f32; 17] = [
            1.066_234_2e-3,
            1.430_553_3e-3,
            1.794_872_1e-3,
            2.159_191_3e-3,
            2.523_510_3e-3,
            2.887_829_2e-3,
            3.252_148_4e-3,
            3.155_337e-3,
            2.804_744e-3,
            2.454_151e-3,
            2.103_558e-3,
            1.752_965e-3,
            1.402_372e-3,
            1.051_779e-3,
            7.011_86e-4,
            3.505_93e-4,
            0.0,
        ];
        for (offset, &expected) in expected_row79_tail.iter().enumerate() {
            let actual = filters[79 * fft_bins + 240 + offset];
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "mel row79 bin{}: actual {actual} expected {expected}",
                240 + offset
            );
        }
    }

    #[test]
    fn espnet_frontend_matches_reference_logmel_frame_count() {
        let samples = jfk_samples();
        assert_eq!(samples.len(), 176_000);
        let features = DolphinEspnetFrontend::new()
            .compute(&samples)
            .expect("espnet fbank");
        // 1 + 176000 / 160 = 1101 (torch.stft center=True reflect framing).
        assert_eq!(features.n_frames, 1_101);
        assert_eq!(features.n_mels, 80);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn espnet_frontend_matches_reference_logmel_values() {
        let samples = jfk_samples();
        let features = DolphinEspnetFrontend::new()
            .compute(&samples)
            .expect("espnet fbank");

        let frame_at = |frame: usize| &features.data[frame * 80..frame * 80 + 8];

        // frame 0 is fully inside the reflect-padded silence lead-in -> the
        // 1e-10 clamp floor, ln(1e-10) = -23.02585.
        for &value in frame_at(0) {
            assert!(
                (value - (-23.025_85)).abs() < 1.0e-3,
                "frame0 expected clamp floor, got {value}"
            );
        }

        // frame 500 first 8 mel bins (torch/librosa reference).
        let expected_frame500: [f32; 8] = [
            -6.403_16,
            -5.750_341_4,
            -5.296_384_3,
            -4.686_787,
            -4.055_321,
            -4.389_283,
            -3.961_102_7,
            -3.866_433_4,
        ];
        for (bin, &expected) in expected_frame500.iter().enumerate() {
            let actual = frame_at(500)[bin];
            assert!(
                (actual - expected).abs() < 1.0e-2,
                "frame500 bin{bin}: actual {actual} expected {expected}"
            );
        }

        // last frame (1100) first 8 mel bins.
        let expected_frame_last: [f32; 8] = [
            -9.255_437,
            -8.575_424,
            -7.676_475,
            -6.291_418,
            -3.404_137_6,
            -3.624_057_8,
            -3.642_896_4,
            -2.373_729_7,
        ];
        for (bin, &expected) in expected_frame_last.iter().enumerate() {
            let actual = frame_at(1_100)[bin];
            assert!(
                (actual - expected).abs() < 1.0e-2,
                "frame1100 bin{bin}: actual {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn espnet_frontend_rejects_empty_audio() {
        assert!(matches!(
            DolphinEspnetFrontend::new().compute(&[]),
            Err(DolphinFrontendError::UnsupportedAudio)
        ));
    }

    // --- min-samples-for-frames inverses (streaming short-tail guard) ------

    #[test]
    fn kaldi_min_samples_for_frames_round_trips_through_compute() {
        for min_frames in 1..=8usize {
            let min_samples = kaldi_min_samples_for_frames(min_frames);
            let features = DolphinFbankFrontend::new()
                .compute(&vec![0.01f32; min_samples])
                .expect("fbank");
            assert!(
                features.n_frames >= min_frames,
                "min_frames={min_frames} min_samples={min_samples} produced only {} frames",
                features.n_frames
            );
            // One sample short must not still reach `min_frames` (the bound is
            // tight, not just sufficient), except at min_frames=1 where even 0
            // samples is a distinct, separately-rejected edge case.
            if min_samples > 0 {
                let short = &vec![0.01f32; min_samples - 1];
                let frames_short = DolphinFbankFrontend::new()
                    .compute(short)
                    .map(|f| f.n_frames)
                    .unwrap_or(0);
                assert!(
                    frames_short < min_frames,
                    "min_frames={min_frames}: {} samples unexpectedly reached {frames_short} frames",
                    min_samples - 1
                );
            }
        }
    }

    #[test]
    fn espnet_min_samples_for_frames_round_trips_through_compute() {
        for min_frames in 1..=8usize {
            let min_samples = espnet_min_samples_for_frames(min_frames);
            let features = DolphinEspnetFrontend::new()
                .compute(&vec![0.01f32; min_samples.max(1)])
                .expect("espnet fbank");
            assert!(
                features.n_frames >= min_frames,
                "min_frames={min_frames} min_samples={min_samples} produced only {} frames",
                features.n_frames
            );
        }
    }
}
