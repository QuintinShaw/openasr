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
const ALPHA: f32 = 0.01;
const HIDDEN: usize = 128;
/// Minimum input samples that still form one output frame through the SincNet
/// conv/pool chain (conv0 k251 s10 → 3× maxpool/3 + 2× conv k5). Below this the
/// segmenter returns no frames instead of underflowing.
const MIN_SAMPLES: usize = 911;
/// Number of powerset classes in the segmentation output.
pub(crate) const NUM_CLASSES: usize = 7;

/// Per-layer LSTM weight names (W input, R recurrent, B bias) in the exported
/// ONNX graph.
const LSTM_WEIGHTS: [(&str, &str, &str); 4] = [
    ("onnx::LSTM_784", "onnx::LSTM_785", "onnx::LSTM_783"),
    ("onnx::LSTM_827", "onnx::LSTM_828", "onnx::LSTM_826"),
    ("onnx::LSTM_870", "onnx::LSTM_871", "onnx::LSTM_869"),
    ("onnx::LSTM_913", "onnx::LSTM_914", "onnx::LSTM_912"),
];

pub(crate) struct PyannetModel {
    w: Weights,
}

impl PyannetModel {
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        Ok(Self {
            w: Weights::from_safetensors(bytes)?,
        })
    }

    /// Load from a diarization `.oasr` (GGUF-v0) pack.
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        Ok(Self {
            w: Weights::from_oasr(path)?,
        })
    }

    /// Run the network on `samples` (16 kHz mono) and return the per-frame
    /// log-probabilities (`[frames, 7]` row-major) plus the frame count.
    pub(crate) fn forward(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), WeightsError> {
        let (h, frames) = self.sincnet(samples)?;
        // transpose [60, frames] -> [frames, 60] for the recurrent stack.
        let mut feat = transpose(&h, 60, frames);
        let mut in_size = 60;
        for (w_name, r_name, b_name) in LSTM_WEIGHTS {
            let w = self.w.get(w_name)?;
            let r = self.w.get(r_name)?;
            let b = self.w.get(b_name)?;
            feat = lstm_bidirectional(&feat, frames, in_size, w, r, b, HIDDEN);
            in_size = 2 * HIDDEN;
        }
        let logp = self.classifier(&feat, frames)?;
        Ok((logp, frames))
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
}

/// Transpose a `[rows, cols]` channel-major buffer to `[cols, rows]`.
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}
