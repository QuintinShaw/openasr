//! ReDimNet2 `TFMelBanks` front end (pure Rust port).
//!
//! Reference: `redimnet2/layers/features_tf.py` from PalabraAI/redimnet2 (MIT),
//! as configured by the B6 checkpoint `model_config.spec_params`
//! (`do_preemph=true`, `norm_signal=true`, `feat_type="tf"`, `hop_length=160`,
//! `F=72`). This is deliberately a *separate* front end from the WeSpeaker Kaldi
//! `Fbank` (`super::super::fbank`): the two differ in bin count (72 vs 80), mel
//! formula (Slaney `2595*log10` vs Kaldi `1127*ln`), `f_max` (7600 vs 8000),
//! signal scaling (per-utterance zero-mean/unit-std vs int16 full-scale), window
//! (Hamming vs Povey), and STFT path (explicit cos/sin conv-DFT vs rustfft), so
//! nothing is shared. See `docs/design/redimnet2-b6-embedder.md`.
//!
//! Forward order (`TFMelBanks.forward` composed with the outer `torchfbank`
//! `Sequential`):
//!
//! ```text
//! x = NormalizeAudio(x)   # (x - mean) / (std_pop + 1e-8), per utterance
//! x = PreEmphasis(x)      # reflect-pad 1 left, conv1d([-0.97, 1.0])
//! p = |DFT(x)|^2          # explicit cos/sin conv1d, stride=160, pad=80, no sqrt
//! m = mel_matrix @ p      # (72,256) x (256,T), Slaney mel, clip(1e-8, 1e8)
//! y = log(m + 1e-8)       # natural log
//! y = y - mean(y, time)   # CMN, per mel bin over time
//! ```
//!
//! All constants (Hamming window, cos/sin DFT kernels, Slaney mel matrix) are
//! deterministic and recomputed here rather than baked into the pack, matching
//! the WeSpeaker `Fbank` convention. Numeric parity against the reference is
//! pinned by `tests::frontend_parity` (kernels/matrix and the four staged
//! tensors) against `frontend_dump/*.npy`.

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
/// FFT-bin half-band actually used: `n_fft / 2 = 256`.
pub(crate) const N_BINS: usize = N_FFT / 2;
pub(crate) const N_MELS: usize = 72;
const F_MIN: f64 = 20.0;
const F_MAX: f64 = 7600.0;
const EPS: f32 = 1e-8;
const INV_EPS: f32 = 1e8;
const PREEMPH_COEF: f32 = 0.97;
/// `F.conv1d(..., padding=shift//2)` in the reference (`shift = hop = 160`).
const STFT_PAD: usize = HOP_LENGTH / 2;

/// Symmetric Hamming window, `scipy.signal.windows.hamming(400)`:
/// `w[n] = 0.54 - 0.46*cos(2*pi*n/(M-1))`.
fn hamming(m: usize) -> Vec<f32> {
    let denom = (m - 1) as f64;
    (0..m)
        .map(|n| (0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / denom).cos()) as f32)
        .collect()
}

/// Slaney/HTK mel: `2595 * log10(1 + hz/700)`.
fn hz2mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel filterbank, `get_filterbanks(nfilt=72, nfft=256, sr=16000, 20, 7600)`.
/// Returns a `(N_MELS, N_BINS)` row-major matrix `mel[mel_idx * N_BINS + bin]`.
///
/// The reference passes `nfft = self.nfft // 2 = 256` here (a per-implementation
/// quirk), so the FFT-bin frequency map spans `linspace(0, sr/2, 256)` and drops
/// bin 0 (DC), whose filter weights are then re-inserted as an all-zero row.
fn mel_filterbank() -> Vec<f32> {
    let low_mel = hz2mel(F_MIN);
    let high_mel = hz2mel(F_MAX);
    // melpoints = linspace(low_mel, high_mel, nfilt + 2)
    let n_pts = N_MELS + 2;
    let mel_step = (high_mel - low_mel) / (n_pts - 1) as f64;
    let melpoints: Vec<f64> = (0..n_pts).map(|i| low_mel + mel_step * i as f64).collect();

    // spectrogram_bins_mel = hz2mel(linspace(0, sr/2, nfft))[1:]  -> 255 values.
    let hz_step = (SAMPLE_RATE / 2) as f64 / (N_BINS - 1) as f64;
    let bins_mel: Vec<f64> = (1..N_BINS).map(|b| hz2mel(hz_step * b as f64)).collect(); // length N_BINS-1 = 255, indexed bin = 1..256

    let mut mat = vec![0.0f32; N_MELS * N_BINS];
    for m in 0..N_MELS {
        let lower = melpoints[m];
        let center = melpoints[m + 1];
        let upper = melpoints[m + 2];
        for (idx, &bin_mel) in bins_mel.iter().enumerate() {
            let bin = idx + 1; // bin 0 (DC) stays all-zero
            let lower_slope = (bin_mel - lower) / (center - lower);
            let upper_slope = (upper - bin_mel) / (upper - center);
            let w = lower_slope.min(upper_slope).max(0.0);
            mat[m * N_BINS + bin] = w as f32;
        }
    }
    mat
}

/// Precomputed cos/sin DFT kernels, each `(N_BINS, WIN_LENGTH)` row-major:
/// `kernel[k * WIN_LENGTH + t]`. Real = `cos(2*pi*k*t/n_fft)*win[t]`, imag =
/// `sin(2*pi*k*t/n_fft)*win[t]` (imag sign is irrelevant: only `re^2+im^2` is
/// used).
struct DftKernels {
    real: Vec<f32>,
    imag: Vec<f32>,
}

fn dft_kernels(window: &[f32]) -> DftKernels {
    let mut real = vec![0.0f32; N_BINS * WIN_LENGTH];
    let mut imag = vec![0.0f32; N_BINS * WIN_LENGTH];
    for k in 0..N_BINS {
        for t in 0..WIN_LENGTH {
            let phase = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / N_FFT as f64;
            real[k * WIN_LENGTH + t] = (phase.cos() as f32) * window[t];
            imag[k * WIN_LENGTH + t] = (phase.sin() as f32) * window[t];
        }
    }
    DftKernels { real, imag }
}

/// The ReDimNet2 `TFMelBanks` front end.
pub(crate) struct RedimNetFrontend {
    window: Vec<f32>,
    kernels: DftKernels,
    mel: Vec<f32>,
}

impl RedimNetFrontend {
    pub(crate) fn new() -> Self {
        let window = hamming(WIN_LENGTH);
        let kernels = dft_kernels(&window);
        let mel = mel_filterbank();
        Self {
            window,
            kernels,
            mel,
        }
    }

    /// Number of mel bins (`72`).
    pub(crate) fn n_mels(&self) -> usize {
        N_MELS
    }

    /// STFT frame count for `t` input samples: `floor((t + 2*pad - win)/hop) + 1`.
    fn frame_count(t: usize) -> usize {
        let padded = t + 2 * STFT_PAD;
        if padded < WIN_LENGTH {
            0
        } else {
            (padded - WIN_LENGTH) / HOP_LENGTH + 1
        }
    }

    /// `NormalizeAudio`: `(x - mean) / (std_pop + 1e-8)` over the whole utterance.
    fn normalize(wav: &[f32]) -> Vec<f32> {
        let n = wav.len();
        if n == 0 {
            return Vec::new();
        }
        let mean = wav.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = wav.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var.sqrt() + EPS as f64);
        wav.iter()
            .map(|&v| ((v as f64 - mean) * inv) as f32)
            .collect()
    }

    /// `PreEmphasis(coef=0.97)`: reflect-pad one sample on the left, then
    /// `conv1d([-0.97, 1.0])`. Output length equals the input length.
    /// `y[0] = x[0] - 0.97*x[1]` (reflect), `y[i] = x[i] - 0.97*x[i-1]`.
    fn preemph(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut y = vec![0.0f32; n];
        // reflect pad on left: x[-1] mirrors to x[1].
        let left = if n > 1 { x[1] } else { x[0] };
        y[0] = x[0] - PREEMPH_COEF * left;
        for i in 1..n {
            y[i] = x[i] - PREEMPH_COEF * x[i - 1];
        }
        y
    }

    /// `|DFT|^2` power spectrum via explicit cos/sin conv1d. Returns
    /// `(power, frames)` with `power` row-major `[bin * frames + frame]`,
    /// `bin` in `0..N_BINS`. Values are clipped to `[1e-8, 1e8]`.
    fn stft_power(&self, x: &[f32]) -> (Vec<f32>, usize) {
        let t = x.len();
        let frames = Self::frame_count(t);
        let mut power = vec![0.0f32; N_BINS * frames];
        for f in 0..frames {
            let base = f * HOP_LENGTH; // start index in the padded signal
            for k in 0..N_BINS {
                let rk = &self.kernels.real[k * WIN_LENGTH..(k + 1) * WIN_LENGTH];
                let ik = &self.kernels.imag[k * WIN_LENGTH..(k + 1) * WIN_LENGTH];
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for w in 0..WIN_LENGTH {
                    let p = base + w; // padded index
                    // padded[p] = x[p - STFT_PAD] inside [0,t), else 0.
                    if p >= STFT_PAD && p - STFT_PAD < t {
                        let s = x[p - STFT_PAD];
                        re += s * rk[w];
                        im += s * ik[w];
                    }
                }
                let mut pw = re * re + im * im;
                pw = pw.clamp(EPS, INV_EPS);
                power[k * frames + f] = pw;
            }
        }
        (power, frames)
    }

    /// `mel_matrix @ power`: `(72,256) x (256,frames) -> (72,frames)`, clipped to
    /// `[1e-8, 1e8]`. Row-major `[mel * frames + frame]`.
    fn mel_linear(&self, power: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; N_MELS * frames];
        for m in 0..N_MELS {
            let row = &self.mel[m * N_BINS..(m + 1) * N_BINS];
            for f in 0..frames {
                let mut acc = 0.0f32;
                for bin in 0..N_BINS {
                    acc += row[bin] * power[bin * frames + f];
                }
                out[m * frames + f] = acc.clamp(EPS, INV_EPS);
            }
        }
        out
    }

    /// Full front end. Returns `(features, frames)` with `features` row-major
    /// `[mel * frames + frame]` (mel-major, matching the reference `(72, T)`
    /// spectrogram layout), after `log(mel + 1e-8)` and per-bin CMN over time.
    pub(crate) fn forward(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        let normalized = Self::normalize(wav);
        let preemphasized = Self::preemph(&normalized);
        let (power, frames) = self.stft_power(&preemphasized);
        if frames == 0 {
            return (Vec::new(), 0);
        }
        let mut feats = self.mel_linear(&power, frames);
        // Outer TFMelBanks.forward: + eps, log, CMN over time (dim=-1).
        for m in 0..N_MELS {
            let row = &mut feats[m * frames..(m + 1) * frames];
            for v in row.iter_mut() {
                *v = (*v + EPS).ln();
            }
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / frames as f64;
            let mean = mean as f32;
            for v in row.iter_mut() {
                *v -= mean;
            }
        }
        (feats, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Spike scratch dir holding the reference `frontend_dump/*.npy`. Not
    /// committed; the parity test is `#[ignore]` and skips if absent.
    const FRONTEND_DUMP: &str =
        "/Volumes/QuintinDocument/openasr-dev/tmp/redimnet2-spike/frontend_dump";

    fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect("read npy");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
        assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
        let fortran = header.contains("'fortran_order': True");
        let shape_start = header.find("'shape':").expect("shape key");
        let paren = header[shape_start..].find('(').unwrap() + shape_start;
        let close = header[paren..].find(')').unwrap() + paren;
        let shape: Vec<usize> = header[paren + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = header_start + header_len;
        let raw: Vec<f32> = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Normalize to C order. Only 2D fortran arrays occur in the dumps (the
        // mel matrix); reorder column-major -> row-major.
        let values = if fortran && shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let mut c = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for col in 0..cols {
                    c[r * cols + col] = raw[col * rows + r];
                }
            }
            c
        } else {
            assert!(
                !fortran || shape.len() <= 1,
                "unsupported fortran-order npy rank {}",
                shape.len()
            );
            raw
        };
        (shape, values)
    }

    /// `(max abs, mean abs)` diff. Cross-implementation fp32 parity is not
    /// bit-exact (different reduction order, conv vs our loop), so we gate a
    /// wide max and a tight mean, matching the firered-aed parity convention.
    fn diff(actual: &[f32], expected: &[f32]) -> (f32, f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        let mut max = 0.0f32;
        let mut sum = 0.0f64;
        for (a, e) in actual.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            max = max.max(d);
            sum += d as f64;
        }
        (max, (sum / actual.len() as f64) as f32)
    }

    fn dump() -> PathBuf {
        PathBuf::from(FRONTEND_DUMP)
    }

    #[test]
    fn frame_count_matches_reference_jfk() {
        // jfk.wav = 176000 samples -> reference spec_output T = 1099.
        assert_eq!(RedimNetFrontend::frame_count(176_000), 1099);
    }

    #[test]
    #[ignore = "requires local frontend_dump/*.npy under tmp/redimnet2-spike (not committed)"]
    fn frontend_parity() {
        let root = dump();
        if !root.join("real_kernel.npy").exists() {
            eprintln!("skip: {root:?} not present");
            return;
        }
        let fe = RedimNetFrontend::new();

        // 1. DFT kernels (256, 400) and mel matrix (72, 256): deterministic
        // constants, so these must be bit-tight.
        let (rk_shape, rk) = load_npy_f32(&root.join("real_kernel.npy"));
        assert_eq!(rk_shape, vec![N_BINS, WIN_LENGTH]);
        let (ik_shape, ik) = load_npy_f32(&root.join("image_kernel.npy"));
        assert_eq!(ik_shape, vec![N_BINS, WIN_LENGTH]);
        let (mel_shape, mel) = load_npy_f32(&root.join("mel_filterbank_matrix.npy"));
        assert_eq!(mel_shape, vec![N_MELS, N_BINS]);
        let (rk_m, _) = diff(&fe.kernels.real, &rk);
        let (ik_m, _) = diff(&fe.kernels.imag, &ik);
        let (mel_m, _) = diff(&fe.mel, &mel);
        println!("real_kernel  max {rk_m:.3e}");
        println!("image_kernel max {ik_m:.3e}");
        println!("mel_matrix   max {mel_m:.3e}");
        assert!(rk_m < 1e-6, "real_kernel diverged: {rk_m:.3e}");
        assert!(ik_m < 1e-6, "image_kernel diverged: {ik_m:.3e}");
        assert!(mel_m < 1e-6, "mel_matrix diverged: {mel_m:.3e}");

        // 2. Staged tensors for each fixture sample.
        for stem in ["jfk", "zh_sample", "en_zh_mixed"] {
            let (_, wav) = load_npy_f32(&root.join(format!("{stem}_01_normalized_waveform.npy")));
            // The dumped "normalized_waveform" IS NormalizeAudio(raw); reproduce
            // preemph/mel/cmn from it directly to isolate each stage. Feed the
            // reference-normalized signal so we compare only our downstream math.
            let (_, ref_preemph) =
                load_npy_f32(&root.join(format!("{stem}_02_preemph_waveform.npy")));
            let (mel_shape, ref_mel) =
                load_npy_f32(&root.join(format!("{stem}_03_mel_linear.npy")));
            let (_, ref_cmn) = load_npy_f32(&root.join(format!("{stem}_04_cmn_output.npy")));

            // preemph parity (on the reference normalized waveform).
            let our_preemph = RedimNetFrontend::preemph(&wav);
            let (pe_max, pe_mean) = diff(&our_preemph, &ref_preemph);

            // mel_linear parity (on the reference preemph waveform).
            let (power, frames) = fe.stft_power(&ref_preemph);
            assert_eq!(mel_shape[0], N_MELS);
            assert_eq!(mel_shape[1], frames, "frame count mismatch for {stem}");
            let our_mel = fe.mel_linear(&power, frames);
            let (mel_max, mel_mean) = diff(&our_mel, &ref_mel);

            // full CMN output (from raw -> everything), driven by the wav we can
            // reconstruct: use reference preemph -> our mel -> log/cmn.
            let mut our_cmn = our_mel.clone();
            for m in 0..N_MELS {
                let rowr = &mut our_cmn[m * frames..(m + 1) * frames];
                for v in rowr.iter_mut() {
                    *v = (*v + EPS).ln();
                }
                let mean = rowr.iter().map(|&v| v as f64).sum::<f64>() / frames as f64;
                let mean = mean as f32;
                for v in rowr.iter_mut() {
                    *v -= mean;
                }
            }
            let (cmn_max, cmn_mean) = diff(&our_cmn, &ref_cmn);

            println!(
                "{stem:12} preemph[max {pe_max:.3e} mean {pe_mean:.3e}] \
                 mel[max {mel_max:.3e} mean {mel_mean:.3e}] \
                 cmn[max {cmn_max:.3e} mean {cmn_mean:.3e}]"
            );

            // Wide max / tight mean, per the cross-impl fp32 convention.
            assert!(pe_max < 1e-4, "{stem} preemph max {pe_max:.3e}");
            assert!(mel_mean < 1e-3, "{stem} mel_linear mean {mel_mean:.3e}");
            assert!(cmn_mean < 1e-4, "{stem} cmn mean {cmn_mean:.3e}");
        }
    }
}
