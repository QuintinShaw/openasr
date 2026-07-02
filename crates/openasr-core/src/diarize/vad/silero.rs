//! Pure-Rust forward pass of Silero VAD v6.2 (16 kHz).
//!
//! Per 32 ms / 512-sample chunk the model emits one speech probability and
//! carries an LSTM state plus a 64-sample audio context across chunks:
//!
//! ```text
//! window = [context(64) | chunk(512)]            (576 samples)
//!   -> reflect-pad right by 64 -> 640
//!   -> learned-STFT conv (k=256, stride=128)     -> [258, 4]
//!   -> magnitude(real[0..129], imag[129..258])   -> [129, 4]
//!   -> conv1 (129->128, k3, s1, p1) + ReLU       -> [128, 4]
//!   -> conv2 (128-> 64, k3, s2, p1) + ReLU       -> [ 64, 2]
//!   -> conv3 ( 64-> 64, k3, s2, p1) + ReLU       -> [ 64, 1]
//!   -> conv4 ( 64->128, k3, s1, p1) + ReLU       -> [128, 1]
//!   -> LSTM cell (single step, state carried)    -> h, c
//!   -> ReLU(h) -> final 1x1 conv -> sigmoid       -> probability
//! ```
//!
//! All ops are tiny (~0.3 M MACs/chunk), so naive dependency-free loops run far
//! under real time and stay numerically faithful to the upstream ONNX model.

use super::weights::{
    FREQ_BINS, HIDDEN, STFT_FILTERS, STFT_KERNEL, SileroWeights, SileroWeightsError,
};

/// New audio samples consumed per inference step (32 ms at 16 kHz).
pub const CHUNK_SAMPLES: usize = 512;
/// Audio context carried from the previous chunk into the STFT window.
const CONTEXT_SAMPLES: usize = 64;
/// STFT window = context + chunk.
const WINDOW_SAMPLES: usize = CONTEXT_SAMPLES + CHUNK_SAMPLES; // 576
/// Right reflection padding applied before the STFT convolution.
const STFT_PAD_RIGHT: usize = 64;
/// The only sample rate this model supports.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Recurrent state carried between chunks: LSTM hidden/cell and audio context.
#[derive(Debug, Clone)]
pub struct SileroVadState {
    hidden: [f32; HIDDEN],
    cell: [f32; HIDDEN],
    context: [f32; CONTEXT_SAMPLES],
}

impl Default for SileroVadState {
    fn default() -> Self {
        Self {
            hidden: [0.0; HIDDEN],
            cell: [0.0; HIDDEN],
            context: [0.0; CONTEXT_SAMPLES],
        }
    }
}

impl SileroVadState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Loaded Silero VAD model. Cheap to clone is not implemented on purpose; load
/// once and reuse across chunks/streams.
pub struct SileroVadModel {
    weights: SileroWeights,
}

impl SileroVadModel {
    /// Load the vendored, validated weights.
    pub fn embedded() -> Result<Self, SileroWeightsError> {
        Ok(Self {
            weights: SileroWeights::embedded()?,
        })
    }

    /// Run one 512-sample chunk and return its speech probability in `[0, 1]`,
    /// advancing `state`. Chunks shorter than 512 are zero-padded on the right.
    pub fn process_chunk(&self, chunk: &[f32], state: &mut SileroVadState) -> f32 {
        let mut window = [0.0f32; WINDOW_SAMPLES];
        window[..CONTEXT_SAMPLES].copy_from_slice(&state.context);
        let take = chunk.len().min(CHUNK_SAMPLES);
        window[CONTEXT_SAMPLES..CONTEXT_SAMPLES + take].copy_from_slice(&chunk[..take]);

        let prob = self.forward(&window, state);

        // Context for the next step is the last 64 samples of the window, i.e.
        // the tail of the new chunk.
        state
            .context
            .copy_from_slice(&window[WINDOW_SAMPLES - CONTEXT_SAMPLES..]);
        prob
    }

    /// Compute one speech probability per 512-sample chunk over `samples`,
    /// starting from a fresh state. The trailing partial chunk (if any) is
    /// zero-padded and still scored.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        let mut state = SileroVadState::new();
        let chunk_count = samples.len().div_ceil(CHUNK_SAMPLES);
        let mut probs = Vec::with_capacity(chunk_count);
        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + CHUNK_SAMPLES).min(samples.len());
            probs.push(self.process_chunk(&samples[offset..end], &mut state));
            offset = end;
        }
        probs
    }

    fn forward(&self, window: &[f32; WINDOW_SAMPLES], state: &mut SileroVadState) -> f32 {
        let (mag, frames) = self.stft_magnitude(window);
        let (e1, f1) = conv1d_relu(
            &mag,
            FREQ_BINS,
            frames,
            &self.weights.conv1_w,
            &self.weights.conv1_b,
            HIDDEN,
            1,
            1,
        );
        let (e2, f2) = conv1d_relu(
            &e1,
            HIDDEN,
            f1,
            &self.weights.conv2_w,
            &self.weights.conv2_b,
            64,
            2,
            1,
        );
        let (e3, f3) = conv1d_relu(
            &e2,
            64,
            f2,
            &self.weights.conv3_w,
            &self.weights.conv3_b,
            64,
            2,
            1,
        );
        let (e4, f4) = conv1d_relu(
            &e3,
            64,
            f3,
            &self.weights.conv4_w,
            &self.weights.conv4_b,
            HIDDEN,
            1,
            1,
        );
        debug_assert_eq!(f4, 1, "encoder must collapse to a single frame per chunk");

        // Encoder output is [HIDDEN, 1] -> the LSTM input vector.
        self.lstm_step(&e4, state);
        self.decode(state)
    }

    /// Learned STFT: reflect-pad the window right by 64, convolve with the fixed
    /// DFT basis (stride 128), then take per-bin magnitude of the real/imag
    /// halves. Returns the `[FREQ_BINS, frames]` magnitude (row-major) + frames.
    fn stft_magnitude(&self, window: &[f32; WINDOW_SAMPLES]) -> (Vec<f32>, usize) {
        const PADDED: usize = WINDOW_SAMPLES + STFT_PAD_RIGHT; // 640
        let mut padded = [0.0f32; PADDED];
        padded[..WINDOW_SAMPLES].copy_from_slice(window);
        // numpy 'reflect': padded[L+j] = window[L-2-j] (edge sample not repeated).
        for (j, slot) in padded[WINDOW_SAMPLES..].iter_mut().enumerate() {
            *slot = window[WINDOW_SAMPLES - 2 - j];
        }

        let frames = (PADDED - STFT_KERNEL) / STFT_STRIDE + 1;
        // conv: 1 input channel, STFT_FILTERS outputs, kernel 256, stride 128.
        let mut filtered = vec![0.0f32; STFT_FILTERS * frames];
        for (out_ch, filter) in self
            .weights
            .stft_basis
            .chunks_exact(STFT_KERNEL)
            .enumerate()
        {
            for frame in 0..frames {
                let base = frame * STFT_STRIDE;
                let mut acc = 0.0f32;
                for (k, w) in filter.iter().enumerate() {
                    acc += w * padded[base + k];
                }
                filtered[out_ch * frames + frame] = acc;
            }
        }

        let mut mag = vec![0.0f32; FREQ_BINS * frames];
        for bin in 0..FREQ_BINS {
            for frame in 0..frames {
                let re = filtered[bin * frames + frame];
                let im = filtered[(FREQ_BINS + bin) * frames + frame];
                mag[bin * frames + frame] = (re * re + im * im).sqrt();
            }
        }
        (mag, frames)
    }

    fn lstm_step(&self, input: &[f32], state: &mut SileroVadState) {
        let w_ih = &self.weights.lstm_w_ih;
        let w_hh = &self.weights.lstm_w_hh;
        let b_ih = &self.weights.lstm_b_ih;
        let b_hh = &self.weights.lstm_b_hh;
        // PyTorch LSTMCell gate order: input, forget, cell, output.
        let mut gates = [0.0f32; 4 * HIDDEN];
        for (row, gate) in gates.iter_mut().enumerate() {
            let wi = &w_ih[row * HIDDEN..row * HIDDEN + HIDDEN];
            let wh = &w_hh[row * HIDDEN..row * HIDDEN + HIDDEN];
            let mut acc = b_ih[row] + b_hh[row];
            for k in 0..HIDDEN {
                acc += wi[k] * input[k] + wh[k] * state.hidden[k];
            }
            *gate = acc;
        }
        for n in 0..HIDDEN {
            let i = sigmoid(gates[n]);
            let f = sigmoid(gates[HIDDEN + n]);
            let g = gates[2 * HIDDEN + n].tanh();
            let o = sigmoid(gates[3 * HIDDEN + n]);
            let c = f * state.cell[n] + i * g;
            state.cell[n] = c;
            state.hidden[n] = o * c.tanh();
        }
    }

    fn decode(&self, state: &SileroVadState) -> f32 {
        // ReLU(hidden) -> 1x1 conv (dot product) -> sigmoid.
        let mut logit = self.weights.final_b;
        for n in 0..HIDDEN {
            logit += self.weights.final_w[n] * state.hidden[n].max(0.0);
        }
        sigmoid(logit)
    }
}

const STFT_STRIDE: usize = 128;

/// 1-D cross-correlation with zero padding, bias, and a fused ReLU.
///
/// `input` is `[in_ch, len]` row-major; `weight` is `[out_ch, in_ch, kernel]`.
/// Returns `([out_ch, out_len] row-major, out_len)`.
fn conv1d_relu(
    input: &[f32],
    in_ch: usize,
    len: usize,
    weight: &[f32],
    bias: &[f32],
    out_ch: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize) {
    let kernel = weight.len() / (out_ch * in_ch);
    let out_len = (len + 2 * pad - kernel) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_len];
    for oc in 0..out_ch {
        let w_oc = &weight[oc * in_ch * kernel..(oc + 1) * in_ch * kernel];
        for ot in 0..out_len {
            let start = ot * stride; // position in the (virtually) padded input
            let mut acc = bias[oc];
            for ic in 0..in_ch {
                let w_ic = &w_oc[ic * kernel..ic * kernel + kernel];
                let row = &input[ic * len..ic * len + len];
                for (k, w) in w_ic.iter().enumerate() {
                    let pos = start + k;
                    if pos < pad {
                        continue;
                    }
                    let t = pos - pad;
                    if t < len {
                        acc += w * row[t];
                    }
                }
            }
            out[oc * out_len + ot] = acc.max(0.0);
        }
    }
    (out, out_len)
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
