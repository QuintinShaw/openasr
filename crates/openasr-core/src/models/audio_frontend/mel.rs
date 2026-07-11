//! Triangular mel filterbank construction, shared across model families.
//!
//! Every family's mel filterbank is a set of triangular filters spanning
//! `[fmin, fmax]` on *some* mel-like scale over the `n_fft / 2 + 1` power
//! bins of an `n_fft`-point real FFT. Families differ along exactly two
//! axes, both captured by [`MelScale`]:
//!
//! - the hz<->mel warping formula ([`hz_to_mel`] / [`mel_to_hz`]), and
//! - whether filter edges are compared in the Hz domain (with the classic
//!   librosa/`torchaudio` triangular-ramp construction, optionally
//!   Slaney-area-normalized) or entirely in the mel domain (the
//!   `kaldi`/`torchaudio.compliance.kaldi.fbank` construction, which never
//!   converts filter edges back to Hz).
//!
//! [`filterbank`] returns a dense row-major `[n_mels, fft_bins]` matrix for
//! every scale. `Kaldi`-scale callers that want the sparse
//! first-nonzero-bin-plus-run representation
//! [`crate::models::kaldi_fbank`] uses for its hot loop can derive it from
//! this matrix (leading/trailing zeros per row); the numeric filter weights
//! are identical either way.

/// Which hz<->mel warping (and filter-construction convention) to use.
///
/// `Htk`/`Kaldi` are not yet consumed by any switched-over family (only
/// `Slaney` is, via `parakeet_ctc`) -- they're declared now per this
/// module's target API and covered by their own unit tests below, so the
/// family PRs that migrate `xasr_zipformer` / `kaldi_fbank`'s consumers
/// onto this module can wire them in directly instead of inventing the
/// scale math a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum MelScale {
    /// librosa `norm="slaney"`, `htk=False`: linear below 1 kHz, log above;
    /// filter edges are converted back to Hz and the resulting triangle is
    /// area-normalized (`2 / (right_hz - left_hz)`). Used by
    /// `parakeet_ctc`/`parakeet_tdt` and `whisper`.
    Slaney,
    /// Classic HTK mel scale (`1127 * ln(1 + hz / 700)`); filter edges are
    /// converted back to Hz, triangles are *not* area-normalized. Used by
    /// `xasr_zipformer`.
    Htk,
    /// Same hz<->mel formula as [`MelScale::Htk`], but filter weights are
    /// computed by mapping each FFT bin's *own* frequency to the mel domain
    /// and comparing it against the (already mel-domain) filter edges --
    /// never converting edges back to Hz. This is
    /// `torchaudio.compliance.kaldi.fbank`'s construction, used by
    /// [`crate::models::kaldi_fbank`] (`firered_aed`/`dolphin`/`sensevoice`).
    Kaldi,
}

/// Geometry for [`filterbank`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterbankConfig {
    pub scale: MelScale,
    pub sample_rate_hz: f32,
    pub n_fft: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
}

/// `hz -> mel` for `scale`.
pub(crate) fn hz_to_mel(scale: MelScale, hz: f32) -> f32 {
    match scale {
        MelScale::Slaney => hz_to_mel_slaney(hz),
        MelScale::Htk | MelScale::Kaldi => hz_to_mel_htk(hz),
    }
}

/// `mel -> hz` for `scale` (the inverse of [`hz_to_mel`]).
pub(crate) fn mel_to_hz(scale: MelScale, mel: f32) -> f32 {
    match scale {
        MelScale::Slaney => mel_to_hz_slaney(mel),
        MelScale::Htk | MelScale::Kaldi => mel_to_hz_htk(mel),
    }
}

/// Dense row-major `[n_mels, n_fft/2+1]` triangular mel filterbank.
pub(crate) fn filterbank(config: FilterbankConfig) -> Vec<f32> {
    let fft_bins = config.n_fft / 2 + 1;
    match config.scale {
        MelScale::Slaney => hz_domain_filterbank(config, fft_bins, true),
        MelScale::Htk => hz_domain_filterbank(config, fft_bins, false),
        MelScale::Kaldi => mel_domain_filterbank(config, fft_bins),
    }
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    // librosa slaney: linear below 1000 Hz, log above.
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        (hz - F_MIN) / F_SP
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

fn hz_to_mel_htk(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// librosa-style construction: `n_mels + 2` edges equally spaced in the mel
/// domain, converted back to Hz, triangles built from Hz-domain distances to
/// `(left, center, right)`; optionally Slaney-area-normalized
/// (`2 / (right - left)`).
fn hz_domain_filterbank(config: FilterbankConfig, fft_bins: usize, slaney_norm: bool) -> Vec<f32> {
    let FilterbankConfig {
        scale,
        sample_rate_hz,
        n_fft,
        n_mels,
        fmin,
        fmax,
    } = config;
    let fft_freqs: Vec<f32> = (0..fft_bins)
        .map(|i| i as f32 * sample_rate_hz / n_fft as f32)
        .collect();
    let mel_min = hz_to_mel(scale, fmin);
    let mel_max = hz_to_mel(scale, fmax);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(scale, m)).collect();

    let mut filters = vec![0.0f32; n_mels * fft_bins];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let enorm = if slaney_norm {
            2.0 / (right - left)
        } else {
            1.0
        };
        for (bin, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - left) / (center - left);
            let upper = (right - f) / (right - center);
            let weight = lower.min(upper).max(0.0);
            filters[m * fft_bins + bin] = weight * enorm;
        }
    }
    filters
}

/// `torchaudio.compliance.kaldi.fbank` construction: `n_mels + 2` edges
/// equally spaced in the mel domain (never converted back to Hz); each FFT
/// bin's own frequency is mapped to the mel domain and compared against
/// `(left, center, right)` there. Filters are zero outside a contiguous run
/// of bins, but this returns the same dense `[n_mels, fft_bins]` shape as
/// the Hz-domain construction -- callers that want the sparse
/// first-bin-plus-run representation derive it from the zero runs.
fn mel_domain_filterbank(config: FilterbankConfig, fft_bins: usize) -> Vec<f32> {
    let FilterbankConfig {
        scale,
        sample_rate_hz,
        n_fft,
        n_mels,
        fmin,
        fmax,
    } = config;
    let fft_bin_width = sample_rate_hz / n_fft as f32;
    let mel_low = hz_to_mel(scale, fmin);
    let mel_high = hz_to_mel(scale, fmax);
    let mel_delta = (mel_high - mel_low) / (n_mels as f32 + 1.0);

    let mut filters = vec![0.0f32; n_mels * fft_bins];
    for m in 0..n_mels {
        let left = mel_low + m as f32 * mel_delta;
        let center = mel_low + (m as f32 + 1.0) * mel_delta;
        let right = mel_low + (m as f32 + 2.0) * mel_delta;
        for k in 0..fft_bins {
            let mel = hz_to_mel(scale, fft_bin_width * k as f32);
            if mel > left && mel < right {
                let weight = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
                filters[m * fft_bins + k] = weight;
            }
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact reimplementation of the pre-refactor
    /// `parakeet_ctc::frontend::slaney_mel_filterbank`, kept only in this
    /// test to pin `MelScale::Slaney` to the values the parakeet family
    /// shipped before this module existed.
    fn reference_parakeet_slaney_filterbank(
        n_mels: usize,
        n_fft: usize,
        fft_bins: usize,
    ) -> Vec<f32> {
        const SAMPLE_RATE: f32 = 16_000.0;
        const MEL_FMIN: f32 = 0.0;
        const MEL_FMAX: f32 = 8_000.0;
        let fft_freqs: Vec<f32> = (0..fft_bins)
            .map(|i| i as f32 * SAMPLE_RATE / n_fft as f32)
            .collect();
        let mel_min = hz_to_mel_slaney(MEL_FMIN);
        let mel_max = hz_to_mel_slaney(MEL_FMAX);
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

        let mut filters = vec![0.0f32; n_mels * fft_bins];
        for m in 0..n_mels {
            let left = hz_points[m];
            let center = hz_points[m + 1];
            let right = hz_points[m + 2];
            let enorm = 2.0 / (right - left);
            for (bin, &f) in fft_freqs.iter().enumerate() {
                let lower = (f - left) / (center - left);
                let upper = (right - f) / (right - center);
                let weight = lower.min(upper).max(0.0);
                filters[m * fft_bins + bin] = weight * enorm;
            }
        }
        filters
    }

    #[test]
    fn slaney_filterbank_is_byte_identical_to_pre_refactor_parakeet_impl() {
        let (n_mels, n_fft, fft_bins) = (80, 512, 257);
        let expected = reference_parakeet_slaney_filterbank(n_mels, n_fft, fft_bins);
        let actual = filterbank(FilterbankConfig {
            scale: MelScale::Slaney,
            sample_rate_hz: 16_000.0,
            n_fft,
            n_mels,
            fmin: 0.0,
            fmax: 8_000.0,
        });
        assert_eq!(expected, actual);

        let (n_mels128, _, _) = (128, 512, 257);
        let expected128 = reference_parakeet_slaney_filterbank(n_mels128, n_fft, fft_bins);
        let actual128 = filterbank(FilterbankConfig {
            scale: MelScale::Slaney,
            sample_rate_hz: 16_000.0,
            n_fft,
            n_mels: n_mels128,
            fmin: 0.0,
            fmax: 8_000.0,
        });
        assert_eq!(expected128, actual128);
    }

    #[test]
    fn htk_filterbank_matches_xasr_zipformer_reference_points() {
        // xasr_zipformer::frontend's htk_mel_filterbank uses the same
        // hz-domain construction without area normalization; cross-check a
        // couple of known htk hz<->mel round-trip points independently of
        // any live family module.
        for hz in [20.0f32, 300.0, 1000.0, 4000.0, 7600.0] {
            let mel = hz_to_mel(MelScale::Htk, hz);
            let back = mel_to_hz(MelScale::Htk, mel);
            assert!((back - hz).abs() < 1e-2, "hz={hz} round-trip={back}");
        }
        let fb = filterbank(FilterbankConfig {
            scale: MelScale::Htk,
            sample_rate_hz: 16_000.0,
            n_fft: 512,
            n_mels: 80,
            fmin: 20.0,
            fmax: 7_600.0,
        });
        for m in 0..80 {
            let row = &fb[m * 257..(m + 1) * 257];
            assert!(row.iter().sum::<f32>() > 0.0, "row {m} empty");
            assert!(row.iter().all(|w| w.is_finite() && *w >= 0.0));
        }
    }

    #[test]
    fn kaldi_scale_filterbank_matches_kaldi_fbank_module_shape() {
        // Same mel-domain construction as crate::models::kaldi_fbank's
        // build_mel_filters (HTK mel scale, weight computed in mel space):
        // every row is a single contiguous nonzero run, peak-normalized
        // triangles (peak weight 1.0 at the center bin), gated strictly
        // inside (left, right).
        let fb = filterbank(FilterbankConfig {
            scale: MelScale::Kaldi,
            sample_rate_hz: 16_000.0,
            n_fft: 512,
            n_mels: 80,
            fmin: 20.0,
            fmax: 8_000.0,
        });
        for m in 0..80 {
            let row = &fb[m * 257..(m + 1) * 257];
            let nonzero: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|&(_, &w)| w > 0.0)
                .map(|(i, _)| i)
                .collect();
            if nonzero.is_empty() {
                continue;
            }
            let first = *nonzero.first().unwrap();
            let last = *nonzero.last().unwrap();
            assert_eq!(
                nonzero.len(),
                last - first + 1,
                "row {m} weights are not one contiguous run"
            );
            assert!(row.iter().all(|w| w.is_finite() && *w >= 0.0 && *w <= 1.0));
        }
    }

    #[test]
    fn hz_to_mel_and_mel_to_hz_round_trip_for_every_scale() {
        for scale in [MelScale::Slaney, MelScale::Htk, MelScale::Kaldi] {
            for hz in [0.0f32, 20.0, 440.0, 1000.0, 4000.0, 8000.0] {
                let mel = hz_to_mel(scale, hz);
                let back = mel_to_hz(scale, mel);
                assert!(
                    (back - hz).abs() < 1e-2,
                    "scale={scale:?} hz={hz} round_trip={back}"
                );
            }
        }
    }
}
