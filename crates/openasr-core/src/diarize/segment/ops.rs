//! Generic primitives for the pyannote PyanNet forward pass: instance norm,
//! 1-D max pool, leaky ReLU, an ONNX-semantics bidirectional LSTM, linear, and
//! log-softmax. All tensors are flat row-major `f32` with explicit shapes;
//! `[c, l]` layouts are channel-major (`row = c * l + t`).

/// Per-channel instance normalization over the length axis (biased variance),
/// then optional per-channel affine. `x` is `[c, l]`.
pub(crate) fn instance_norm_inplace(
    x: &mut [f32],
    c: usize,
    l: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) {
    for ch in 0..c {
        let row = &mut x[ch * l..(ch + 1) * l];
        let mean = row.iter().sum::<f32>() / l as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let g = gamma.map(|w| w[ch]).unwrap_or(1.0);
        let b = beta.map(|w| w[ch]).unwrap_or(0.0);
        for v in row {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

/// 1-D max pool over `[c, l]` with `kernel`/`stride`, no padding. Returns an
/// empty result if `l < kernel` (no valid window) rather than underflowing.
pub(crate) fn maxpool1d(
    x: &[f32],
    c: usize,
    l: usize,
    kernel: usize,
    stride: usize,
) -> (Vec<f32>, usize) {
    if l < kernel {
        return (Vec::new(), 0);
    }
    let l_out = (l - kernel) / stride + 1;
    let mut out = vec![f32::NEG_INFINITY; c * l_out];
    for ch in 0..c {
        for t in 0..l_out {
            let start = t * stride;
            let mut m = f32::NEG_INFINITY;
            for k in 0..kernel {
                m = m.max(x[ch * l + start + k]);
            }
            out[ch * l_out + t] = m;
        }
    }
    (out, l_out)
}

pub(crate) fn leaky_relu_inplace(x: &mut [f32], alpha: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= alpha;
        }
    }
}

pub(crate) fn abs_inplace(x: &mut [f32]) {
    for v in x {
        *v = v.abs();
    }
}

/// Linear `y = x · wᵀ + b`. `x` is `[rows, in]`, `weight` is `[in, out]`
/// (row-major, the ONNX MatMul layout), `bias` is `[out]`.
pub(crate) fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = bias.map(|b| b[o]).unwrap_or(0.0);
            for i in 0..in_dim {
                acc += x[r * in_dim + i] * weight[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

/// Row-wise log-softmax over `[rows, cols]`.
pub(crate) fn log_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        let log_sum = max + sum.ln();
        for c in 0..cols {
            out[r * cols + c] = row[c] - log_sum;
        }
    }
    out
}

use crate::diarize::embed::ops::sigmoid;

/// ONNX-semantics bidirectional LSTM.
///
/// `input` is `[seq, in_size]`; `w` is `[2, 4*hidden, in_size]`, `r` is
/// `[2, 4*hidden, hidden]`, `bias` is `[2, 8*hidden]` (`[Wb(4h) | Rb(4h)]`).
/// Gate order is ONNX `i, o, f, c`. Output is `[seq, 2*hidden]` with the forward
/// direction in `[0, hidden)` and the backward in `[hidden, 2*hidden)`.
pub(crate) fn lstm_bidirectional(
    input: &[f32],
    seq: usize,
    in_size: usize,
    w: &[f32],
    r: &[f32],
    bias: &[f32],
    hidden: usize,
) -> Vec<f32> {
    let gate = 4 * hidden;
    let mut output = vec![0.0f32; seq * 2 * hidden];
    for dir in 0..2 {
        let w_dir = &w[dir * gate * in_size..(dir + 1) * gate * in_size];
        let r_dir = &r[dir * gate * hidden..(dir + 1) * gate * hidden];
        let b_dir = &bias[dir * 8 * hidden..(dir + 1) * 8 * hidden];
        let mut h = vec![0.0f32; hidden];
        let mut c = vec![0.0f32; hidden];
        // gate slice helpers: order i=0, o=1, f=2, c=3.
        for step in 0..seq {
            let t = if dir == 0 { step } else { seq - 1 - step };
            let x = &input[t * in_size..(t + 1) * in_size];
            let mut new_h = vec![0.0f32; hidden];
            let mut new_c = vec![0.0f32; hidden];
            for unit in 0..hidden {
                let mut g = [0.0f32; 4];
                for (gi, gv) in g.iter_mut().enumerate() {
                    let row = gi * hidden + unit;
                    let mut acc = b_dir[row] + b_dir[4 * hidden + row]; // Wb + Rb
                    let w_row = &w_dir[row * in_size..(row + 1) * in_size];
                    for k in 0..in_size {
                        acc += w_row[k] * x[k];
                    }
                    let r_row = &r_dir[row * hidden..(row + 1) * hidden];
                    for k in 0..hidden {
                        acc += r_row[k] * h[k];
                    }
                    *gv = acc;
                }
                let it = sigmoid(g[0]);
                let ot = sigmoid(g[1]);
                let ft = sigmoid(g[2]);
                let ct = g[3].tanh();
                let cell = ft * c[unit] + it * ct;
                new_c[unit] = cell;
                new_h[unit] = ot * cell.tanh();
            }
            h = new_h;
            c = new_c;
            let base = t * 2 * hidden + dir * hidden;
            output[base..base + hidden].copy_from_slice(&h);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool_reduces_length() {
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0];
        let (out, l) = maxpool1d(&x, 1, 6, 3, 3);
        assert_eq!(l, 2);
        assert_eq!(out, vec![3.0, 5.0]);
    }

    #[test]
    fn maxpool_too_short_is_empty_not_underflow() {
        let (out, l) = maxpool1d(&[1.0, 2.0], 1, 2, 3, 3);
        assert_eq!(l, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn instance_norm_zero_mean_unit_var() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        instance_norm_inplace(&mut x, 1, 4, None, None, 1e-5);
        let mean = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn leaky_relu_scales_negatives() {
        let mut x = vec![-2.0, 3.0];
        leaky_relu_inplace(&mut x, 0.01);
        assert!((x[0] + 0.02).abs() < 1e-6 && (x[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn log_softmax_rows_sum_to_one_in_prob() {
        let out = log_softmax(&[1.0, 2.0, 3.0], 1, 3);
        let p: f32 = out.iter().map(|v| v.exp()).sum();
        assert!((p - 1.0).abs() < 1e-5);
    }
}
