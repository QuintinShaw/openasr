//! Pure-Rust forward pass of pyannote segmentation-3.0 (PyanNet, MIT).
//!
//! ```text
//! waveform [n] -> wav InstanceNorm
//!   -> SincNet: conv(1->80,k251,s10) abs, maxpool3, InstanceNorm, leakyReLU
//!               conv(80->60,k5), maxpool3, InstanceNorm, leakyReLU
//!               conv(60->60,k5), maxpool3, InstanceNorm, leakyReLU   -> [60, frames]
//!   -> 4x BiLSTM (hidden 128)                                        -> [frames, 256]
//!   -> linear 256->128 (leakyReLU) -> 128->128 (leakyReLU) -> 128->7
//!   -> log-softmax                                                   -> [frames, 7]
//! ```
//!
//! The 7 classes are the powerset of up to 3 concurrent speakers (`∅, {1}, {2},
//! {3}, {1,2}, {1,3}, {2,3}`). The sinc filters are materialized into a plain
//! conv weight in the exported model, so no sinc parametrization is needed here.

use super::ops::{
    abs_inplace, instance_norm_inplace, leaky_relu_inplace, linear, log_softmax,
    lstm_bidirectional, maxpool1d,
};
use crate::diarize::embed::ops::conv1d;
use crate::diarize::embed::weights::{Weights, WeightsError};

const EPS: f32 = 1e-5;
pub(super) const ALPHA: f32 = 0.01;
pub(super) const HIDDEN: usize = 128;
/// Minimum input samples that still form one output frame through the SincNet
/// conv/pool chain (conv0 k251 s10 → 3× maxpool/3 + 2× conv k5). Below this the
/// segmenter returns no frames instead of underflowing.
const MIN_SAMPLES: usize = 911;
/// Number of powerset classes in the segmentation output.
pub(crate) const NUM_CLASSES: usize = 7;

/// Exact SincNet output-frame geometry used by the segmentation adapter and
/// its request-memory estimator. Keeping the valid conv/pool chain here with
/// the model prevents admission from guessing a frame count.
pub(crate) const fn output_frame_count(samples: usize) -> usize {
    let mut frames = valid_output_count(samples, 251, 10);
    frames = valid_output_count(frames, 3, 3);
    frames = valid_output_count(frames, 5, 1);
    frames = valid_output_count(frames, 3, 3);
    frames = valid_output_count(frames, 5, 1);
    valid_output_count(frames, 3, 3)
}

/// Exact shape-derived upper bound for f32 payloads simultaneously owned by
/// one pure-Rust PyanNet forward. The input waveform belongs to the caller and
/// is excluded. This mirrors the actual conv/pool, transpose, four BiLSTM, and
/// classifier lifetimes; no nominal frames-per-second approximation is used.
pub(crate) const fn quoted_forward_peak_bytes(samples: usize) -> u64 {
    let conv0 = valid_output_count(samples, 251, 10);
    let pool0 = valid_output_count(conv0, 3, 3);
    let conv1 = valid_output_count(pool0, 5, 1);
    let pool1 = valid_output_count(conv1, 3, 3);
    let conv2 = valid_output_count(pool1, 5, 1);
    let frames = valid_output_count(conv2, 3, 3);

    let sinc0 = samples
        .saturating_add(80usize.saturating_mul(conv0))
        .saturating_add(80usize.saturating_mul(pool0));
    let sinc1 = samples
        .saturating_add(80usize.saturating_mul(pool0))
        .saturating_add(60usize.saturating_mul(conv1))
        .saturating_add(60usize.saturating_mul(pool1));
    let sinc2 = samples
        .saturating_add(60usize.saturating_mul(pool1))
        .saturating_add(60usize.saturating_mul(conv2))
        .saturating_add(60usize.saturating_mul(frames));

    // `h` from SincNet remains live while it is transposed. An LSTM call
    // overlaps the previous and next feature rows plus output, h/c, and
    // new_h/new_c. Classifier shadow bindings overlap both 128-wide hidden
    // rows, logits, and log-softmax output.
    let recurrent = 60usize
        .saturating_mul(frames)
        .saturating_add(256usize.saturating_mul(frames))
        .saturating_add(256usize.saturating_mul(frames))
        .saturating_add(4usize.saturating_mul(HIDDEN));
    let classifier = 60usize
        .saturating_mul(frames)
        .saturating_add(256usize.saturating_mul(frames))
        .saturating_add(128usize.saturating_mul(frames).saturating_mul(2))
        .saturating_add(NUM_CLASSES.saturating_mul(frames).saturating_mul(2));

    let elements = max_usize(
        max_usize(max_usize(sinc0, sinc1), max_usize(sinc2, recurrent)),
        classifier,
    );
    (elements as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
}

const fn valid_output_count(input: usize, kernel: usize, stride: usize) -> usize {
    if input < kernel {
        0
    } else {
        (input - kernel) / stride + 1
    }
}

const fn max_usize(lhs: usize, rhs: usize) -> usize {
    if lhs > rhs { lhs } else { rhs }
}

/// Per-layer LSTM weight names (W input, R recurrent, B bias) in the exported
/// ONNX graph.
pub(super) const LSTM_WEIGHTS: [(&str, &str, &str); 4] = [
    ("onnx::LSTM_784", "onnx::LSTM_785", "onnx::LSTM_783"),
    ("onnx::LSTM_827", "onnx::LSTM_828", "onnx::LSTM_826"),
    ("onnx::LSTM_870", "onnx::LSTM_871", "onnx::LSTM_869"),
    ("onnx::LSTM_913", "onnx::LSTM_914", "onnx::LSTM_912"),
];

pub(crate) struct PyannetModel {
    w: Weights,
}

impl PyannetModel {
    #[cfg(test)]
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        Self::from_weights(Weights::from_safetensors(bytes)?)
    }

    /// Load from a diarization `.oasr` (GGUF-v0) pack.
    #[cfg(test)]
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        Self::from_weights(Weights::from_oasr(path)?)
    }

    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, WeightsError> {
        Self::from_weights(Weights::from_preflight(preflight)?)
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, WeightsError> {
        Weights::quoted_persistent_host_commitment_bytes(tensor_index)
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, WeightsError> {
        self.w.persistent_host_commitment_bytes()
    }

    fn from_weights(w: Weights) -> Result<Self, WeightsError> {
        let model = Self { w };
        model.validate_weight_contract()?;
        Ok(model)
    }

    /// Family-owned, validated logical weights. Accelerated implementations
    /// consume this view only after the same constructor contract has closed.
    pub(super) const fn weights(&self) -> &Weights {
        &self.w
    }

    /// Run the network on `samples` (16 kHz mono) and return the per-frame
    /// log-probabilities (`[frames, 7]` row-major) plus the frame count.
    pub(crate) fn forward(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), WeightsError> {
        let (h, frames) = self.sincnet(samples)?;
        // transpose [60, frames] -> [frames, 60] for the recurrent stack.
        let feat = transpose(&h, 60, frames);
        Ok((self.recurrent_classifier(&feat, frames)?, frames))
    }

    /// Recurrent stack plus classifier on row-major `[frames, 60]` features.
    pub(super) fn recurrent_classifier(
        &self,
        features: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, WeightsError> {
        let expected = frames.saturating_mul(60);
        if features.len() != expected {
            return Err(WeightsError::InvalidInput(format!(
                "PyanNet recurrent input has {} values, expected {expected}",
                features.len()
            )));
        }
        let mut feat = features.to_vec();
        let mut in_size = 60;
        for (w_name, r_name, b_name) in LSTM_WEIGHTS {
            let w = self.w.get(w_name)?;
            let r = self.w.get(r_name)?;
            let b = self.w.get(b_name)?;
            feat = lstm_bidirectional(&feat, frames, in_size, w, r, b, HIDDEN);
            in_size = 2 * HIDDEN;
        }
        self.classifier(&feat, frames)
    }

    /// SincNet front-end: returns `([60, frames]` channel-major, `frames)`.
    fn sincnet(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), WeightsError> {
        // Below the receptive field the conv/pool chain cannot form one output
        // frame; bail to empty rather than underflow the length arithmetic.
        let n = samples.len();
        if n < MIN_SAMPLES {
            return Ok((Vec::new(), 0));
        }
        // wav_norm1d: instance-norm the raw waveform with a scalar affine.
        let mut x = samples.to_vec();
        let wav_scale = self.w.get("sincnet.wav_norm1d.weight")?;
        let wav_bias = self.w.get("sincnet.wav_norm1d.bias")?;
        instance_norm_inplace(&mut x, 1, n, Some(wav_scale), Some(wav_bias), EPS);

        // SincNet block 0: sinc conv (materialized), abs, maxpool, norm, leakyReLU.
        let sinc_w = self.w.get("/sincnet/conv1d.0/Concat_2_output_0")?;
        let (mut h, mut l) = conv1d(&x, 1, n, sinc_w, None, 80, 251, 10, 0, 1);
        abs_inplace(&mut h);
        let pooled = maxpool1d(&h, 80, l, 3, 3);
        h = pooled.0;
        l = pooled.1;
        self.norm(&mut h, 80, l, "sincnet.norm1d.0")?;
        leaky_relu_inplace(&mut h, ALPHA);

        // SincNet block 1 + 2: conv(k5), maxpool, norm, leakyReLU.
        for (idx, (c_in, c_out)) in [(80usize, 60usize), (60, 60)].into_iter().enumerate() {
            let conv = format!("sincnet.conv1d.{}", idx + 1);
            let weight = self.w.get(&format!("{conv}.weight"))?;
            let bias = self.w.get(&format!("{conv}.bias"))?;
            let (out, lo) = conv1d(&h, c_in, l, weight, Some(bias), c_out, 5, 1, 0, 1);
            let pooled = maxpool1d(&out, c_out, lo, 3, 3);
            h = pooled.0;
            l = pooled.1;
            self.norm(&mut h, c_out, l, &format!("sincnet.norm1d.{}", idx + 1))?;
            leaky_relu_inplace(&mut h, ALPHA);
        }
        debug_assert_eq!(l, output_frame_count(n));
        Ok((h, l))
    }

    fn classifier(&self, feat: &[f32], frames: usize) -> Result<Vec<f32>, WeightsError> {
        // classifier: 256 -> 128 -> 128 -> 7, leakyReLU between the linears.
        let mut h = linear(
            feat,
            frames,
            256,
            self.w.get("onnx::MatMul_915")?,
            128,
            Some(self.w.get("linear.0.bias")?),
        );
        leaky_relu_inplace(&mut h, ALPHA);
        let mut h = linear(
            &h,
            frames,
            128,
            self.w.get("onnx::MatMul_916")?,
            128,
            Some(self.w.get("linear.1.bias")?),
        );
        leaky_relu_inplace(&mut h, ALPHA);
        let logits = linear(
            &h,
            frames,
            128,
            self.w.get("onnx::MatMul_917")?,
            NUM_CLASSES,
            Some(self.w.get("classifier.bias")?),
        );
        Ok(log_softmax(&logits, frames, NUM_CLASSES))
    }

    /// Test hook: return the SincNet output `[60, frames]` and the layer-1 LSTM
    /// output `[frames, 256]` for stage-by-stage validation against ONNX.
    #[cfg(test)]
    pub(crate) fn stages(
        &self,
        samples: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, usize), WeightsError> {
        let (h, frames) = self.sincnet(samples)?;
        let feat = transpose(&h, 60, frames);
        let (w, r, b) = LSTM_WEIGHTS[0];
        let lstm1 = lstm_bidirectional(
            &feat,
            frames,
            60,
            self.w.get(w)?,
            self.w.get(r)?,
            self.w.get(b)?,
            HIDDEN,
        );
        Ok((h, lstm1, frames))
    }

    fn norm(&self, x: &mut [f32], c: usize, l: usize, name: &str) -> Result<(), WeightsError> {
        let gamma = self.w.get(&format!("{name}.weight"))?;
        let beta = self.w.get(&format!("{name}.bias"))?;
        instance_norm_inplace(x, c, l, Some(gamma), Some(beta), EPS);
        Ok(())
    }

    fn validate_weight_contract(&self) -> Result<(), WeightsError> {
        for (name, shape) in [
            ("sincnet.wav_norm1d.weight", &[1][..]),
            ("sincnet.wav_norm1d.bias", &[1][..]),
            ("/sincnet/conv1d.0/Concat_2_output_0", &[80, 1, 251][..]),
            ("sincnet.norm1d.0.weight", &[80][..]),
            ("sincnet.norm1d.0.bias", &[80][..]),
            ("sincnet.conv1d.1.weight", &[60, 80, 5][..]),
            ("sincnet.conv1d.1.bias", &[60][..]),
            ("sincnet.norm1d.1.weight", &[60][..]),
            ("sincnet.norm1d.1.bias", &[60][..]),
            ("sincnet.conv1d.2.weight", &[60, 60, 5][..]),
            ("sincnet.conv1d.2.bias", &[60][..]),
            ("sincnet.norm1d.2.weight", &[60][..]),
            ("sincnet.norm1d.2.bias", &[60][..]),
            ("onnx::MatMul_915", &[256, 128][..]),
            ("linear.0.bias", &[128][..]),
            ("onnx::MatMul_916", &[128, 128][..]),
            ("linear.1.bias", &[128][..]),
            ("onnx::MatMul_917", &[128, NUM_CLASSES][..]),
            ("classifier.bias", &[NUM_CLASSES][..]),
        ] {
            self.expect_shape(name, shape)?;
        }
        for (layer, (w_name, r_name, b_name)) in LSTM_WEIGHTS.into_iter().enumerate() {
            let input = if layer == 0 { 60 } else { 2 * HIDDEN };
            self.expect_shape(w_name, &[2, 4 * HIDDEN, input])?;
            self.expect_shape(r_name, &[2, 4 * HIDDEN, HIDDEN])?;
            self.expect_shape(b_name, &[2, 8 * HIDDEN])?;
        }
        Ok(())
    }

    fn expect_shape(&self, name: &str, want: &[usize]) -> Result<(), WeightsError> {
        let got = self.w.shape(name)?;
        if got == want {
            Ok(())
        } else {
            Err(WeightsError::ShapeMismatch {
                name: name.to_string(),
                got: got.to_vec(),
                want: want.to_vec(),
            })
        }
    }
}

/// Transpose a `[rows, cols]` channel-major buffer to `[cols, rows]`.
pub(super) fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}
