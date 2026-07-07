//! Pure-Rust forward pass of FireRedVAD's `DetectModel` (a causal-FSMN DNN,
//! `DFSMN config: 8x[256-128(20,20,1,1)]-1x256`, ~0.6 M params, 10 ms frames).
//!
//! Reproduces `fireredvad/core/detect_model.py`'s `DetectModel` exactly for
//! the non-streaming (whole-utterance, no cache) forward pass used by
//! long-form batch VAD:
//!
//! ```text
//! feat [T, 80] --CMVN--> x0
//!   -> fc1 (80->256) + ReLU -> fc2 (256->128) + ReLU        = p0
//!   -> fsmn1(p0)  [p0 + causal-lookback(p0) + lookahead(p0)] = memory
//!   -> x7 DFSMNBlock:
//!        h = fc1(memory) + ReLU        (128->256)
//!        p = fc2(h)                    (256->128, no bias)
//!        memory = fsmn(p) + memory     [p + lookback(p) + lookahead(p) + residual]
//!   -> dnn: fc(128->256) + ReLU
//!   -> out: fc(256->1) -> sigmoid                            = per-frame speech prob
//! ```
//!
//! The FSMN "filter" is a depthwise (per-channel) 1-D FIR: `lookback` sums the
//! current frame and the previous `LOOKBACK_ORDER - 1` frames (causal,
//! zero-padded at the start); `lookahead` sums the next `LOOKAHEAD_ORDER`
//! frames (zero-padded at the end, and forced to zero on the very last
//! frame -- matching upstream's `F.pad(..., (0, S2))` on the shifted
//! lookahead output). Both use unit dilation (`S1 = S2 = 1`) in the vendored
//! checkpoint, asserted at load time (see `weights::FireRedVadWeights`).
//!
//! All ops are small (~0.6 M MACs/frame), so a naive per-frame/per-channel
//! loop runs far under real time for long-form batch use and stays
//! numerically faithful to the upstream torch model (validated bit-close
//! against a committed golden fixture; see `tests`).

use super::frontend::{FbankFeatures, FireRedVadFbankFrontend, NUM_MEL_BINS, apply_cmvn};
use super::weights::{BlockWeights, FireRedVadWeights, FireRedVadWeightsError};

/// Frame shift of the upstream fbank frontend (10 ms), i.e. the cadence of
/// one speech probability.
pub const FRAME_SHIFT_MS: u32 = 10;
/// Depthwise FSMN "lookback" (causal) filter length: `N1` in the upstream
/// `DetectModel` args.
pub(crate) const LOOKBACK_ORDER: usize = 20;
/// Depthwise FSMN "lookahead" filter length: `N2` in the upstream args.
pub(crate) const LOOKAHEAD_ORDER: usize = 20;
/// DNN hidden width (`H` in the upstream args).
pub(crate) const HIDDEN: usize = 256;
/// FSMN projection width (`P` in the upstream args).
pub(crate) const PROJ: usize = 128;
/// Repeated `DFSMNBlock` count (`R - 1`; `R = 8` total FSMN blocks, the first
/// being the input-connecting `fsmn1` with no external skip connection).
pub(crate) const NUM_BLOCKS: usize = 7;

pub struct FireRedVadModel {
    weights: FireRedVadWeights,
    frontend: FireRedVadFbankFrontend,
}

impl FireRedVadModel {
    /// Load the vendored, validated weights.
    pub fn embedded() -> Result<Self, FireRedVadWeightsError> {
        Ok(Self {
            weights: FireRedVadWeights::embedded()?,
            frontend: FireRedVadFbankFrontend::new(),
        })
    }

    /// Compute one speech probability per 10 ms frame over the whole
    /// `samples` buffer (16 kHz mono `f32` in `[-1, 1]`). Whole-utterance
    /// (non-streaming, no cache) forward pass -- matches
    /// `FireRedVad.detect(..., do_postprocess=False)`. Returns an empty
    /// vector for audio shorter than one 25 ms frame.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        let FbankFeatures { mut data, n_frames } = self.frontend.compute(samples);
        if n_frames == 0 {
            return Vec::new();
        }
        apply_cmvn(
            &mut data,
            &self.weights.cmvn_mean,
            &self.weights.cmvn_inv_stddev,
        );
        self.forward(&data, n_frames)
    }

    fn forward(&self, cmvn_feat: &[f32], t: usize) -> Vec<f32> {
        let w = &self.weights;

        // fc1 (80 -> 256) + ReLU, fc2 (256 -> 128) + ReLU.
        let h0 = linear_relu(cmvn_feat, t, NUM_MEL_BINS, &w.fc1_w, &w.fc1_b, HIDDEN);
        let p0 = linear_relu(&h0, t, HIDDEN, &w.fc2_w, &w.fc2_b, PROJ);

        // fsmn1: no external skip, but FSMN itself is residual over its input.
        let mut memory = fsmn_apply(&p0, t, PROJ, &w.fsmn1_lookback, &w.fsmn1_lookahead);

        for block in &w.blocks {
            memory = dfsmn_block(&memory, t, block);
        }

        // dnn: fc (128 -> 256) + ReLU, then out: fc (256 -> 1) + sigmoid.
        let dnn_out = linear_relu(&memory, t, PROJ, &w.dnn_w, &w.dnn_b, HIDDEN);
        let mut probs = Vec::with_capacity(t);
        for frame in dnn_out.chunks_exact(HIDDEN) {
            let mut logit = w.out_b;
            for (v, wt) in frame.iter().zip(w.out_w.iter()) {
                logit += v * wt;
            }
            probs.push(sigmoid(logit));
        }
        probs
    }
}

/// One `DFSMNBlock`: `memory = FSMN(fc2(ReLU(fc1(inputs)))) + inputs`.
fn dfsmn_block(inputs: &[f32], t: usize, block: &BlockWeights) -> Vec<f32> {
    let h = linear_relu(inputs, t, PROJ, &block.fc1_w, &block.fc1_b, HIDDEN);
    let p = linear_no_bias(&h, t, HIDDEN, &block.fc2_w, PROJ);
    let fsmn_out = fsmn_apply(&p, t, PROJ, &block.lookback, &block.lookahead);
    let mut out = fsmn_out;
    for (o, residual) in out.iter_mut().zip(inputs) {
        *o += residual;
    }
    out
}

/// `y[t, c] = x[t, c] + lookback(x)[t, c] + lookahead(x)[t, c]` (the FSMN
/// layer's own internal residual; upstream's `FSMN.forward`).
///
/// `lookback(x)[t] = sum_{k=0}^{K-1} w_lb[c, k] * x[t - (K-1) + k]` (causal,
/// zero for negative indices; `K = LOOKBACK_ORDER`).
/// `lookahead(x)[t] = sum_{k=0}^{M-1} w_la[c, k] * x[t + 1 + k]` for
/// `t < T - 1` (zero for out-of-range indices; `M = LOOKAHEAD_ORDER`),
/// and exactly `0` for the last frame `t == T - 1` (upstream's
/// `F.pad(lookahead[..., N2*S2:], (0, S2))` right-pads the shifted output by
/// one zero frame).
fn fsmn_apply(
    x: &[f32],
    t: usize,
    channels: usize,
    lookback_w: &[f32],
    lookahead_w: &[f32],
) -> Vec<f32> {
    let mut out = x.to_vec();
    for c in 0..channels {
        let lb_w = &lookback_w[c * LOOKBACK_ORDER..(c + 1) * LOOKBACK_ORDER];
        let la_w = &lookahead_w[c * LOOKAHEAD_ORDER..(c + 1) * LOOKAHEAD_ORDER];
        for ti in 0..t {
            let mut lb = 0.0f32;
            for (k, wt) in lb_w.iter().enumerate() {
                let src = ti as isize - (LOOKBACK_ORDER as isize - 1) + k as isize;
                if src >= 0 {
                    lb += wt * x[src as usize * channels + c];
                }
            }
            let mut la = 0.0f32;
            if ti + 1 < t {
                for (k, wt) in la_w.iter().enumerate() {
                    let src = ti + 1 + k;
                    if src < t {
                        la += wt * x[src * channels + c];
                    }
                }
            }
            out[ti * channels + c] += lb + la;
        }
    }
    out
}

/// `y[t, o] = ReLU(bias[o] + sum_i x[t, i] * weight[o, i])`. `weight` is
/// PyTorch `nn.Linear` layout, `[out_dim, in_dim]` row-major.
fn linear_relu(
    x: &[f32],
    t: usize,
    in_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; t * out_dim];
    for ti in 0..t {
        let row = &x[ti * in_dim..(ti + 1) * in_dim];
        for o in 0..out_dim {
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            let mut acc = bias[o];
            for (xi, wi) in row.iter().zip(w_row) {
                acc += xi * wi;
            }
            out[ti * out_dim + o] = acc.max(0.0);
        }
    }
    out
}

/// Same as [`linear_relu`] but without a bias or the trailing ReLU (the
/// `DFSMNBlock`'s `fc2`, which upstream declares `bias=False`).
fn linear_no_bias(x: &[f32], t: usize, in_dim: usize, weight: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * out_dim];
    for ti in 0..t {
        let row = &x[ti * in_dim..(ti + 1) * in_dim];
        for o in 0..out_dim {
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (xi, wi) in row.iter().zip(w_row) {
                acc += xi * wi;
            }
            out[ti * out_dim + o] = acc;
        }
    }
    out
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
