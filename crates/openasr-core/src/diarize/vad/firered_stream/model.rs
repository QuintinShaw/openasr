//! Causal, cache-carrying forward pass of FireRedVAD's **Stream-VAD**
//! (`Stream-VAD/model.pth.tar`): a `DetectModel` DFSMN architecture exported
//! with `N2 = 0` (no lookahead), i.e. strictly causal at every layer.
//!
//! Because there is no lookahead term, a single [`forward_chunk`] call is
//! *the entire model* -- the "whole-utterance" batch path
//! ([`FireRedStreamVadModel::probabilities`]) is defined as exactly one call
//! to [`FireRedStreamVadModel::forward_chunk`] over all frames with a fresh
//! [`FireRedStreamVadCache`]. Feeding the same audio through
//! [`super::streaming::FireRedStreamingVad`] in many small chunks (any
//! chunking) produces bit-identical per-frame probabilities to that batch
//! call, because the only cross-frame state ("lookback", a depthwise causal
//! FIR over the previous `LOOKBACK_ORDER - 1` frames at each of the 8 FSMN
//! layers) is carried exactly in `cache` -- there is no other source of
//! cross-chunk-boundary error, unlike models with true recurrent state
//! (LSTM/attention) where chunking changes numerics.
//!
//! [`forward_chunk`]: FireRedStreamVadModel::forward_chunk

use super::frontend::{FbankFeatures, FireRedVadFbankFrontend, NUM_MEL_BINS, apply_cmvn};
use super::weights::{FireRedStreamVadWeights, FireRedStreamVadWeightsError};

/// Frame shift of the shared fbank frontend (10 ms).
pub const FRAME_SHIFT_MS: u32 = 10;
/// Depthwise FSMN "lookback" (causal) filter length (`N1` upstream).
pub(crate) const LOOKBACK_ORDER: usize = 20;
/// DNN hidden width (`H` upstream).
pub(crate) const HIDDEN: usize = 256;
/// FSMN projection width (`P` upstream).
pub(crate) const PROJ: usize = 128;
/// Repeated `DFSMNBlock` count (`R - 1`; `R = 8` upstream).
pub(crate) const NUM_BLOCKS: usize = 7;
/// Frames of lookback history that must be carried across a chunk boundary
/// (the FIR needs the previous `LOOKBACK_ORDER - 1` frames).
const CACHE_FRAMES: usize = LOOKBACK_ORDER - 1;

/// Per-session lookback state for the causal DFSMN stack: the trailing
/// `<= LOOKBACK_ORDER - 1` frames of each of the 8 FSMN layers' inputs
/// (`fsmn1` plus the 7 `DFSMNBlock`s), carried across [`forward_chunk`] calls.
///
/// [`forward_chunk`]: FireRedStreamVadModel::forward_chunk
#[derive(Clone)]
pub struct FireRedStreamVadCache {
    fsmn1: Vec<f32>,
    blocks: Vec<Vec<f32>>,
}

impl FireRedStreamVadCache {
    pub fn new() -> Self {
        Self {
            fsmn1: Vec::new(),
            blocks: vec![Vec::new(); NUM_BLOCKS],
        }
    }

    /// Clear all carried history for a new utterance/session.
    pub fn reset(&mut self) {
        self.fsmn1.clear();
        for block in &mut self.blocks {
            block.clear();
        }
    }
}

impl Default for FireRedStreamVadCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FireRedStreamVadModel {
    weights: FireRedStreamVadWeights,
    frontend: FireRedVadFbankFrontend,
}

impl FireRedStreamVadModel {
    /// Load the vendored, validated weights.
    pub fn embedded() -> Result<Self, FireRedStreamVadWeightsError> {
        Ok(Self {
            weights: FireRedStreamVadWeights::embedded()?,
            frontend: FireRedVadFbankFrontend::new(),
        })
    }

    /// Compute one speech probability per 10 ms frame over the whole
    /// `samples` buffer (16 kHz mono `f32` in `[-1, 1]`). Defined as a single
    /// [`forward_chunk`] call with a fresh cache -- see the module docs for
    /// why this is bit-identical to any chunking of the same audio through
    /// the streaming detector.
    ///
    /// [`forward_chunk`]: Self::forward_chunk
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
        let mut cache = FireRedStreamVadCache::new();
        self.forward_chunk(&data, n_frames, &mut cache)
    }

    /// Compute CMVN'd log-mel features for a raw-sample buffer, without
    /// running the DFSMN stack. Used by [`super::streaming::FireRedStreamingVad`]
    /// to turn each newly-completed window of raw PCM into the feature frames
    /// [`forward_chunk`] expects. Returns `(features, n_frames)`;
    /// `features.len() == n_frames * NUM_MEL_BINS`.
    ///
    /// [`forward_chunk`]: Self::forward_chunk
    pub(crate) fn cmvn_features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let FbankFeatures { mut data, n_frames } = self.frontend.compute(samples);
        if n_frames > 0 {
            apply_cmvn(
                &mut data,
                &self.weights.cmvn_mean,
                &self.weights.cmvn_inv_stddev,
            );
        }
        (data, n_frames)
    }

    /// Causal forward over one chunk of `t` CMVN'd feature frames
    /// (`[t, NUM_MEL_BINS]`, row-major), updating `cache` in place so a
    /// following call continues the lookback history exactly (no future
    /// context is ever read, so chunk boundaries introduce no error).
    pub fn forward_chunk(
        &self,
        cmvn_feat: &[f32],
        t: usize,
        cache: &mut FireRedStreamVadCache,
    ) -> Vec<f32> {
        let w = &self.weights;

        let h0 = linear_relu(cmvn_feat, t, NUM_MEL_BINS, &w.fc1_w, &w.fc1_b, HIDDEN);
        let p0 = linear_relu(&h0, t, HIDDEN, &w.fc2_w, &w.fc2_b, PROJ);
        let mut memory =
            fsmn_apply_causal_cached(&p0, t, PROJ, &w.fsmn1_lookback, &mut cache.fsmn1);

        for (block, block_cache) in w.blocks.iter().zip(cache.blocks.iter_mut()) {
            let h = linear_relu(&memory, t, PROJ, &block.fc1_w, &block.fc1_b, HIDDEN);
            let p = linear_no_bias(&h, t, HIDDEN, &block.fc2_w, PROJ);
            let mut out = fsmn_apply_causal_cached(&p, t, PROJ, &block.lookback, block_cache);
            for (o, residual) in out.iter_mut().zip(memory.iter()) {
                *o += residual;
            }
            memory = out;
        }

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

/// `y[t, c] = x[t, c] + lookback(x)[t, c]` where
/// `lookback(x)[t] = sum_{k=0}^{K-1} w_lb[c, k] * x[t - (K-1) + k]`, `x`
/// spanning `cache ++ this chunk` so the FIR sees exactly the same history it
/// would in a single whole-utterance call. `cache` is replaced with the
/// trailing `<= K - 1` frames of `cache ++ x` for the next call.
///
/// The naive form of this loop (channel-outer, tap-innermost) reads
/// `combined` with a `channels`-wide stride per tap -- cache-hostile and not
/// auto-vectorizable. Instead this transposes `cache ++ x` to channel-major
/// with a `K - 1` zero-pad up front, so each channel's causal FIR becomes a
/// branch-free contiguous [`dot`] over a sliding window (no `src >= 0`
/// bounds check in the hot loop; the zero-pad supplies the "no history yet"
/// case at the very start of a session).
fn fsmn_apply_causal_cached(
    x: &[f32],
    t: usize,
    channels: usize,
    lookback_w: &[f32],
    cache: &mut Vec<f32>,
) -> Vec<f32> {
    let cache_frames = cache.len() / channels;
    let total = cache_frames + t;
    let mut combined = Vec::with_capacity(total * channels);
    combined.extend_from_slice(cache);
    combined.extend_from_slice(x);

    let mut out = x.to_vec();

    let pad = LOOKBACK_ORDER - 1;
    let row_len = pad + total;
    let mut padded = vec![0.0f32; channels * row_len];
    for (frame, src_frame) in combined.chunks_exact(channels).enumerate() {
        for (c, &v) in src_frame.iter().enumerate() {
            padded[c * row_len + pad + frame] = v;
        }
    }

    for c in 0..channels {
        let lb_w = &lookback_w[c * LOOKBACK_ORDER..(c + 1) * LOOKBACK_ORDER];
        let chan_row = &padded[c * row_len..(c + 1) * row_len];
        for ti_new in 0..t {
            let ti = cache_frames + ti_new;
            let window = &chan_row[ti..ti + LOOKBACK_ORDER];
            out[ti_new * channels + c] += dot(window, lb_w);
        }
    }

    let keep = total.min(CACHE_FRAMES);
    let start = total - keep;
    *cache = combined[start * channels..].to_vec();
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
            out[ti * out_dim + o] = (bias[o] + dot(row, w_row)).max(0.0);
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
            out[ti * out_dim + o] = dot(row, w_row);
        }
    }
    out
}

/// Dot product with 8 independent accumulator lanes. A single running
/// accumulator (`acc += a[i]*b[i]`) creates a serial dependency chain that
/// blocks auto-vectorization of the float reduction; splitting into 8 lanes
/// gives the compiler independent accumulations it can pack into SIMD
/// registers (NEON/AVX) before the final horizontal reduction.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let ai = &a[i * LANES..i * LANES + LANES];
        let bi = &b[i * LANES..i * LANES + LANES];
        for lane in 0..LANES {
            acc[lane] += ai[lane] * bi[lane];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in chunks * LANES..a.len() {
        sum += a[i] * b[i];
    }
    sum
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
