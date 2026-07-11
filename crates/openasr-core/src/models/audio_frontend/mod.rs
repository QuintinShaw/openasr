//! Model-agnostic STFT/mel DSP primitives, shared across ASR frontend
//! modules.
//!
//! This is the base layer [`crate::models::kaldi_fbank`] already carved out
//! for the batch kaldi-fbank engine (`firered_aed`/`dolphin`/`sensevoice`);
//! this module holds the lower-level primitives that engine -- and every
//! other family's frontend -- is built from: framing + windowing + FFT
//! ([`StftFramer`]), mel filterbank construction ([`mel`]), and per-feature
//! (mean/std) normalization ([`per_feature_normalize`]).
//!
//! Each family still owns its own frontend module, error type, and any
//! family-specific pre/post-processing (pre-emphasis, DC removal, log-guard
//! style, LFR stacking, CMVN affine, ...); this module intentionally stops
//! at the shared numeric primitives so a config difference between families
//! stays a config difference instead of a second copy of the math. See
//! `parakeet_ctc/frontend.rs` for the first family built directly on these
//! primitives; other families migrate in their own follow-up changes (each
//! independently golden-verified) rather than all at once here.
//!
//! Numeric behavior is carried over byte-for-byte from the pre-refactor
//! per-family copies -- nothing here is a "cleanup" of the math itself, only
//! of where it lives.

pub(crate) mod mel;

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// How a [`StftFramer`] turns raw samples into fixed-length frames before
/// windowing + FFT. Each variant fixes both the boundary-padding behavior
/// *and* the window-buffer alignment convention real frontends pair it
/// with; see the field docs on [`StftFramer::new`] for why those two are
/// coupled per engine rather than independent axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PadMode {
    /// `torch.stft(center=True, pad_mode="reflect")`: reflect-pad `n_fft/2`
    /// samples on both ends (no edge repeat) before framing at `hop`
    /// stride, i.e. `n_frames = (padded_len - n_fft) / hop + 1`. Used by
    /// `parakeet_ctc`/`parakeet_tdt`, `whisper`, `cohere`, `qwen`, and
    /// `dolphin`'s ESPnet-variant frontend (`DolphinEspnetFrontend`).
    ReflectCenter,
    /// Kaldi/sherpa-onnx `snip_edges=false`: frame `i`'s window starts at
    /// absolute sample index `i*hop - (win/2 - hop/2)` (the asymmetric
    /// kaldi offset, distinct from `ReflectCenter`'s symmetric `n_fft/2`
    /// centering), reflected (no edge repeat) against the *total* stream
    /// length rather than a once-materialized padded buffer. There is no
    /// whole-buffer [`StftFramer::power_spectrogram`] entry point for this
    /// mode -- drive it exclusively through
    /// [`StftFramer::power_spectrogram_kaldi_snip_edges_false_range`], which
    /// takes an explicit frame range and an optional dropped-prefix offset
    /// so streaming callers can hold only a tail of the signal (O(1) memory)
    /// instead of the whole utterance. Used by `xasr_zipformer`'s streaming
    /// fbank.
    KaldiSnipEdgesFalse,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum StftError {
    #[error("stft framer produced no frames from {samples} samples")]
    TooShort { samples: usize },
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum KaldiFrameRangeError {
    #[error(
        "kaldi snip_edges=false frame range needs a sample earlier than the retained prefix (frame {frame})"
    )]
    MissingPrefix { frame: usize },
}

/// Power spectrogram, row-major `[frame][fft_bin]` (bin innermost).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PowerSpectrogram {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_fft_bins: usize,
}

/// Frames raw audio, applies an analysis window, and runs a real FFT to
/// produce a power spectrogram. Owns the FFT plan (built once per instance)
/// since planning dominates the per-call cost for short frames. `Clone`
/// clones the `Arc`-held FFT plan (cheap, shares the twiddle tables) --
/// needed so `xasr_zipformer::frontend::XasrFbankFrontend` (which embeds
/// this) can keep deriving `Clone` itself.
#[derive(Clone)]
pub(crate) struct StftFramer {
    n_fft: usize,
    /// Analysis window length before any `n_fft` zero-pad (documentation +
    /// `Debug` only; alignment is baked into `window` by the caller).
    win: usize,
    hop: usize,
    pad_mode: PadMode,
    /// Pre-built analysis window, already embedded in an `n_fft`-length
    /// buffer at whatever alignment the caller's family needs (e.g.
    /// [`hann_window_centered`] centers a `win < n_fft` window; a future
    /// left-aligned builder for the kaldi/snip-edges convention would zero
    /// the trailing `n_fft - win` samples instead). `StftFramer` itself is
    /// agnostic to that choice -- it only ever indexes `window[0..n_fft]`.
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for StftFramer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StftFramer")
            .field("n_fft", &self.n_fft)
            .field("win", &self.win)
            .field("hop", &self.hop)
            .field("pad_mode", &self.pad_mode)
            .finish_non_exhaustive()
    }
}

impl StftFramer {
    /// `window` must have length `n_fft` (see the field doc above for why
    /// alignment is the caller's responsibility).
    pub(crate) fn new(
        n_fft: usize,
        win: usize,
        hop: usize,
        pad_mode: PadMode,
        window: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(window.len(), n_fft, "StftFramer window must be n_fft long");
        Self {
            n_fft,
            win,
            hop,
            pad_mode,
            window,
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(n_fft),
        }
    }

    pub(crate) fn n_fft_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Compute the power spectrogram for mono `samples` (float, any scale --
    /// scaling to the engine's native domain, e.g. int16 magnitude, is the
    /// caller's job before calling this).
    pub(crate) fn power_spectrogram(&self, samples: &[f32]) -> Result<PowerSpectrogram, StftError> {
        match self.pad_mode {
            PadMode::ReflectCenter => self.power_spectrogram_reflect_center(samples),
            PadMode::KaldiSnipEdgesFalse => panic!(
                "StftFramer::power_spectrogram: PadMode::KaldiSnipEdgesFalse has no whole-buffer \
                 entry point -- use power_spectrogram_kaldi_snip_edges_false_range"
            ),
        }
    }

    /// Kaldi `snip_edges=false` incremental framing (see
    /// [`PadMode::KaldiSnipEdgesFalse`]): computes power-spectrogram rows for
    /// frames `[start_frame, end_frame)` of a stream whose total length is
    /// `total_len`, reading only `samples[sample_offset..]` (streaming
    /// callers may have already dropped the earlier prefix). Returns
    /// [`KaldiFrameRangeError::MissingPrefix`] instead of panicking when a
    /// frame needs a sample before `sample_offset`.
    pub(crate) fn power_spectrogram_kaldi_snip_edges_false_range(
        &self,
        samples: &[f32],
        sample_offset: usize,
        total_len: usize,
        start_frame: usize,
        end_frame: usize,
    ) -> Result<PowerSpectrogram, KaldiFrameRangeError> {
        debug_assert_eq!(self.pad_mode, PadMode::KaldiSnipEdgesFalse);
        // Kaldi's asymmetric offset: half the window, minus half the hop.
        let frame_offset = (self.win / 2) as i64 - (self.hop / 2) as i64;
        let fft_bins = self.n_fft_bins();
        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        let n_frames = end_frame.saturating_sub(start_frame);
        let mut data = vec![0.0f32; n_frames * fft_bins];
        for frame in start_frame..end_frame {
            for value in fft_in.iter_mut() {
                *value = 0.0;
            }
            let start = frame as i64 * self.hop as i64 - frame_offset;
            for (i, (slot, &window_value)) in fft_in
                .iter_mut()
                .zip(self.window.iter())
                .enumerate()
                .take(self.win)
            {
                let absolute_idx = reflect_index_no_repeat(total_len, start + i as i64);
                let sample_idx = absolute_idx
                    .checked_sub(sample_offset)
                    .ok_or(KaldiFrameRangeError::MissingPrefix { frame })?;
                *slot = samples[sample_idx] * window_value;
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("stft framer kaldi-range rfft");
            let row_offset = (frame - start_frame) * fft_bins;
            for (bin, c) in fft_out.iter().enumerate() {
                data[row_offset + bin] = c.norm_sqr();
            }
        }
        Ok(PowerSpectrogram {
            data,
            n_frames,
            n_fft_bins: fft_bins,
        })
    }

    fn power_spectrogram_reflect_center(
        &self,
        samples: &[f32],
    ) -> Result<PowerSpectrogram, StftError> {
        let pad = self.n_fft / 2;
        let padded = reflect_pad(samples, pad);
        if padded.len() < self.n_fft {
            return Err(StftError::TooShort {
                samples: samples.len(),
            });
        }
        let n_frames = (padded.len() - self.n_fft) / self.hop + 1;
        let fft_bins = self.n_fft_bins();

        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        let mut data = vec![0.0f32; n_frames * fft_bins];
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop;
            for i in 0..self.n_fft {
                fft_in[i] = padded[start + i] * self.window[i];
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("stft framer rfft");
            let row = &mut data[frame_idx * fft_bins..(frame_idx + 1) * fft_bins];
            for (bin, c) in fft_out.iter().enumerate() {
                row[bin] = c.norm_sqr();
            }
        }
        Ok(PowerSpectrogram {
            data,
            n_frames,
            n_fft_bins: fft_bins,
        })
    }
}

/// Periodic Hann window of `win_length`, zero-padded symmetrically into a
/// buffer of `n_fft` (`torch.stft` pads a shorter `win_length` to `n_fft`,
/// centered).
pub(crate) fn hann_window_centered(win_length: usize, n_fft: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    let offset = (n_fft - win_length) / 2;
    for i in 0..win_length {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos();
        window[offset + i] = w;
    }
    window
}

/// Periodic Hann window of `length`, computed by pre-dividing the angular
/// step (`2*pi / length`) once and multiplying by the sample index, rather
/// than [`hann_window_centered`]'s per-sample `(2*pi*i) / length` operation
/// order. The two are the exact same real-valued formula but round
/// differently in f32 for some lengths (confirmed by exhaustive bit
/// comparison at length 400); this is `whisper`'s own bit-exact
/// construction, used only where `win_length == n_fft` (no centering slack,
/// hence no `n_fft` parameter).
pub(crate) fn hann_window_periodic_scale_first(length: usize) -> Vec<f32> {
    let scale = std::f32::consts::PI * 2.0 / length as f32;
    (0..length)
        .map(|i| 0.5 - 0.5 * (scale * i as f32).cos())
        .collect()
}

/// Kaldi "Povey" window (`hann(i)^0.85`, `torch.hann_window`'s
/// `periodic=True` formula raised to the 0.85 power), zero-padded on the
/// right into an `n_fft`-length buffer (`window[win_length..]` stays zero).
/// This is the kaldi `snip_edges=false` left-aligned convention -- the
/// `win_length < n_fft` slack goes entirely on the right, as opposed to
/// [`hann_window_centered`]'s symmetric centering (see
/// [`PadMode::KaldiSnipEdgesFalse`]). Used by `xasr_zipformer`'s streaming
/// fbank; `kaldi_fbank.rs`'s batch engine builds its own copy of this same
/// formula independently (unifying that is a separate follow-up, out of
/// scope here).
pub(crate) fn povey_window_left_aligned(win_length: usize, n_fft: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    for (i, slot) in window.iter_mut().enumerate().take(win_length) {
        let hann =
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32).cos();
        *slot = hann.powf(0.85);
    }
    window
}

/// numpy/`torch.stft`-style reflect padding (no edge repeat): pads `pad`
/// samples of mirrored signal onto both ends of `samples`.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(samples[(pad - i).min(n.saturating_sub(1))]);
    }
    out.extend_from_slice(samples);
    for i in 0..pad {
        let idx = n.saturating_sub(2 + i);
        out.push(samples[idx.min(n.saturating_sub(1))]);
    }
    out
}

/// `numpy`/`torch` `mode="reflect"` sample lookup at a possibly
/// out-of-range signal index (may be negative or `>= samples_len`): mirrors
/// the signal about each edge without repeating the edge sample, matching
/// `torch.stft`'s `pad_mode="reflect"` -- the per-index equivalent of
/// [`reflect_pad`], usable when materializing the whole padded buffer up
/// front isn't practical (kaldi `snip_edges=false` incremental streaming
/// framing, `xasr_zipformer`; ESPnet per-frame framing, `dolphin`). Single-
/// or zero-sample signals fold to index 0.
pub(crate) fn reflect_index_no_repeat(samples_len: usize, index: i64) -> usize {
    if samples_len <= 1 {
        return 0;
    }
    let last = samples_len as i64 - 1;
    let period = 2 * last;
    let mut folded = index % period;
    if folded < 0 {
        folded += period;
    }
    if folded > last {
        (period - folded) as usize
    } else {
        folded as usize
    }
}

/// Per-feature (column) mean/std normalization of a row-major
/// `[n_frames, n_features]` matrix (feature innermost), in place.
///
/// `ddof` is the variance denominator's delta degrees of freedom
/// (`denom = max(n_frames - ddof, 1)`; `ddof = 0.0` is the population
/// variance every current family uses). `eps` is added to the standard
/// deviation *after* the square root (not inside it), matching every
/// current family's normalization epsilon placement.
pub(crate) fn per_feature_normalize(
    data: &mut [f32],
    n_frames: usize,
    n_features: usize,
    eps: f32,
    ddof: f64,
) {
    debug_assert_eq!(data.len(), n_frames * n_features);
    let mean_denom = n_frames.max(1) as f64;
    for feat in 0..n_features {
        let mut mean = 0.0f64;
        for fr in 0..n_frames {
            mean += data[fr * n_features + feat] as f64;
        }
        mean /= mean_denom;

        let mut var = 0.0f64;
        for fr in 0..n_frames {
            let d = data[fr * n_features + feat] as f64 - mean;
            var += d * d;
        }
        let var_denom = (n_frames as f64 - ddof).max(1.0);
        var /= var_denom;
        let std = (var.sqrt() as f32) + eps;

        for fr in 0..n_frames {
            let v = &mut data[fr * n_features + feat];
            *v = (*v - mean as f32) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parakeet_style_framer() -> StftFramer {
        StftFramer::new(
            512,
            400,
            160,
            PadMode::ReflectCenter,
            hann_window_centered(400, 512),
        )
    }

    #[test]
    fn reflect_center_frame_count_matches_torch_stft_formula() {
        let framer = parakeet_style_framer();
        // 1 s @ 16 kHz -> padded len = 16000 + 512, n_frames = (padded -
        // 512) / 160 + 1.
        let samples = vec![0.01f32; 16_000];
        let spectrogram = framer.power_spectrogram(&samples).expect("spectrogram");
        let expected_frames = (16_000 + 512 - 512) / 160 + 1;
        assert_eq!(spectrogram.n_frames, expected_frames);
        assert_eq!(spectrogram.n_fft_bins, 257);
        assert_eq!(
            spectrogram.data.len(),
            spectrogram.n_frames * spectrogram.n_fft_bins
        );
        assert!(spectrogram.data.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn reflect_center_handles_audio_shorter_than_n_fft_without_panicking() {
        // padded len = samples.len() + n_fft always >= n_fft for nonempty
        // input, so `ReflectCenter` never actually returns `TooShort` for
        // this pad amount (n_fft/2 both sides) -- it's a defensive variant
        // carried over from the pre-refactor per-family error type, kept
        // for parity. What must hold is: short-but-nonempty audio still
        // produces one valid (finite) frame, not a panic.
        let framer = parakeet_style_framer();
        let samples = vec![0.01f32; 4];
        let spectrogram = framer.power_spectrogram(&samples).expect("spectrogram");
        assert_eq!(spectrogram.n_frames, 1);
        assert!(spectrogram.data.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn hann_window_centered_is_zero_padded_symmetrically() {
        let window = hann_window_centered(400, 512);
        assert_eq!(window.len(), 512);
        let offset = (512 - 400) / 2;
        assert!(window[..offset].iter().all(|&v| v == 0.0));
        assert!(window[offset + 400..].iter().all(|&v| v == 0.0));
        // periodic Hann starts and (nearly) ends at 0, peaks at the center.
        assert_eq!(window[offset], 0.0);
        assert!(window[offset + 200] > 0.99);
    }

    #[test]
    fn per_feature_normalize_ddof_zero_matches_population_stats() {
        // 3 frames, 2 features: feature 0 = [1,2,3], feature 1 = [10,10,10].
        let mut data = vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0];
        per_feature_normalize(&mut data, 3, 2, 1.0e-5, 0.0);
        // feature 1 has zero variance: normalized to (x - mean) / (0 + eps).
        for fr in 0..3 {
            assert!((data[fr * 2 + 1]).abs() < 1.0e-3);
        }
        // feature 0: mean=2, population var=2/3, std=sqrt(2/3).
        let std = (2.0f64 / 3.0).sqrt() as f32 + 1.0e-5;
        assert!((data[0] - (1.0 - 2.0) / std).abs() < 1.0e-5);
        assert!((data[2] - (2.0 - 2.0) / std).abs() < 1.0e-5);
        assert!((data[4] - (3.0 - 2.0) / std).abs() < 1.0e-5);
    }

    #[test]
    fn per_feature_normalize_ddof_one_uses_sample_variance() {
        let mut ddof0 = vec![1.0f32, 2.0, 3.0];
        let mut ddof1 = ddof0.clone();
        per_feature_normalize(&mut ddof0, 3, 1, 0.0, 0.0);
        per_feature_normalize(&mut ddof1, 3, 1, 0.0, 1.0);
        // sample variance (ddof=1) is larger than population variance
        // (ddof=0), so the ddof=1 std is larger and its normalized
        // magnitudes are smaller.
        assert!(ddof1[0].abs() < ddof0[0].abs());
    }

    /// Exact reimplementation of the pre-refactor `whisper::mel::hann_window`,
    /// kept only to pin [`hann_window_periodic_scale_first`] to the values
    /// `whisper` shipped before this module existed.
    fn reference_whisper_hann_window(length: usize) -> Vec<f32> {
        let scale = std::f32::consts::PI * 2.0 / length as f32;
        (0..length)
            .map(|index| 0.5 - 0.5 * (scale * index as f32).cos())
            .collect()
    }

    #[test]
    fn hann_window_periodic_scale_first_is_byte_identical_to_pre_refactor_whisper_impl() {
        for length in [400usize, 512usize] {
            assert_eq!(
                reference_whisper_hann_window(length),
                hann_window_periodic_scale_first(length),
                "length={length}"
            );
        }
    }

    /// Exact reimplementation of the pre-refactor
    /// `xasr_zipformer::frontend::povey_window`, kept only to pin
    /// [`povey_window_left_aligned`] to the values `xasr_zipformer` shipped
    /// before this module existed.
    fn reference_xasr_povey_window(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| {
                let hann =
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos();
                hann.powf(0.85)
            })
            .collect()
    }

    #[test]
    fn povey_window_left_aligned_is_byte_identical_to_pre_refactor_xasr_impl() {
        let expected = reference_xasr_povey_window(400);
        let actual = povey_window_left_aligned(400, 512);
        assert_eq!(&actual[..400], expected.as_slice());
        assert!(actual[400..].iter().all(|&v| v == 0.0));
    }

    /// Exact reimplementation of the pre-refactor
    /// `xasr_zipformer::frontend::reflect_sample_index`, kept only to pin
    /// [`reflect_index_no_repeat`] to the values `xasr_zipformer` shipped
    /// before this module existed. (`dolphin::frontend::reflect_sample`'s
    /// per-index math is the same formula independently derived; both
    /// families' pre-refactor golden coverage exercises this indirectly.)
    fn reference_xasr_reflect_sample_index(index: i64, len: usize) -> usize {
        if len <= 1 {
            return 0;
        }
        let last = len as i64 - 1;
        let period = 2 * last;
        let mut folded = index % period;
        if folded < 0 {
            folded += period;
        }
        if folded > last {
            (period - folded) as usize
        } else {
            folded as usize
        }
    }

    #[test]
    fn reflect_index_no_repeat_is_byte_identical_to_pre_refactor_xasr_impl() {
        for len in [1usize, 2, 5, 400] {
            for index in -20i64..40 {
                assert_eq!(
                    reference_xasr_reflect_sample_index(index, len),
                    reflect_index_no_repeat(len, index),
                    "len={len} index={index}"
                );
            }
        }
    }

    #[test]
    fn reflect_index_no_repeat_stays_in_bounds() {
        for &i in &(-20..20).collect::<Vec<i64>>() {
            assert!(reflect_index_no_repeat(5, i) < 5);
        }
        assert_eq!(reflect_index_no_repeat(5, -1), 1);
        assert_eq!(reflect_index_no_repeat(5, 5), 3);
    }

    fn xasr_style_framer() -> StftFramer {
        StftFramer::new(
            512,
            400,
            160,
            PadMode::KaldiSnipEdgesFalse,
            povey_window_left_aligned(400, 512),
        )
    }

    #[test]
    fn kaldi_snip_edges_false_range_matches_full_computation_when_split() {
        let framer = xasr_style_framer();
        let samples: Vec<f32> = (0..8_000)
            .map(|i| (2.0 * std::f32::consts::PI * 313.0 * i as f32 / 16_000.0).sin() * 0.1)
            .collect();
        let total_len = samples.len();
        // 7 frames covers the whole clip at this length; split the range to
        // confirm the incremental range API is order/slice independent.
        let full = framer
            .power_spectrogram_kaldi_snip_edges_false_range(&samples, 0, total_len, 0, 7)
            .expect("full range");
        let head = framer
            .power_spectrogram_kaldi_snip_edges_false_range(&samples, 0, total_len, 0, 3)
            .expect("head range");
        let tail = framer
            .power_spectrogram_kaldi_snip_edges_false_range(&samples, 0, total_len, 3, 7)
            .expect("tail range");
        assert_eq!(head.data, full.data[..3 * full.n_fft_bins]);
        assert_eq!(tail.data, full.data[3 * full.n_fft_bins..]);
    }

    #[test]
    fn kaldi_snip_edges_false_range_rejects_dropped_prefix() {
        let framer = xasr_style_framer();
        let samples = vec![0.01f32; 8_000];
        // Frame 0 needs samples starting at 0*160 - (400/2 - 160/2) = -120,
        // reflected into [0, total_len); asking for frame 0 while
        // `sample_offset` has already dropped everything before absolute
        // sample 4000 must fail closed, not panic or read out of bounds.
        let tail = &samples[4_000..];
        let error = framer
            .power_spectrogram_kaldi_snip_edges_false_range(tail, 4_000, 8_000, 0, 1)
            .unwrap_err();
        assert!(matches!(
            error,
            KaldiFrameRangeError::MissingPrefix { frame: 0 }
        ));
    }
}
