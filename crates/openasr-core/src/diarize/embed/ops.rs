//! Shared neural primitives used by the pure-Rust pyannote segmenter.
//!
//! ReDimNet2-B6 runs through a ggml graph and does not use this module. What
//! remains here is the 1-D conv stack (and `sigmoid`) that `segment/pyannet`
//! still needs. Tensors are flat row-major `Vec<f32>` with explicit shapes.

use rayon::prelude::*;

/// Multiply-accumulate count below which a conv runs single-threaded; tiny ops
/// are cheaper than a rayon fork-join.
const PAR_MIN_MACS: usize = 1 << 16;

/// 1-D cross-correlation with zero padding, optional bias, and dilation.
///
/// `input` is `[c_in, l]` row-major; `weight` is `[c_out, c_in, k]`. Returns
/// `([c_out, l_out] row-major, l_out)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d(
    input: &[f32],
    c_in: usize,
    l: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> (Vec<f32>, usize) {
    let effective = dilation * (k - 1) + 1;
    let l_out = (l + 2 * pad).saturating_sub(effective) / stride + 1;
    let mut out = vec![0.0f32; c_out * l_out];
    if k == 1 && stride == 1 && pad == 0 {
        matmul_1x1(input, c_in, l, weight, bias, c_out, &mut out);
        return (out, l_out);
    }
    let row = |oc: usize, dst: &mut [f32]| {
        conv1d_row(
            input, c_in, l, weight, bias, oc, k, stride, pad, dilation, dst,
        );
    };
    if c_out * l_out * c_in * k >= PAR_MIN_MACS {
        out.par_chunks_mut(l_out)
            .enumerate()
            .for_each(|(oc, dst)| row(oc, dst));
    } else {
        out.chunks_mut(l_out)
            .enumerate()
            .for_each(|(oc, dst)| row(oc, dst));
    }
    (out, l_out)
}

#[allow(clippy::too_many_arguments)]
fn conv1d_row(
    input: &[f32],
    c_in: usize,
    l: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    dst: &mut [f32],
) {
    let l_out = dst.len();
    dst.fill(bias.map_or(0.0, |bb| bb[oc]));
    for ic in 0..c_in {
        let in_row = &input[ic * l..ic * l + l];
        let w_base = (oc * c_in + ic) * k;
        for kk in 0..k {
            let wgt = weight[w_base + kk];
            let off = kk * dilation;
            // ot*stride + off - pad must land in [0, l).
            let ot_start = if pad > off {
                (pad - off).div_ceil(stride)
            } else {
                0
            };
            let ot_end = if l + pad > off {
                ((l + pad - off - 1) / stride + 1).min(l_out)
            } else {
                0
            };
            if ot_start >= ot_end {
                continue;
            }
            if stride == 1 {
                let t0 = ot_start + off - pad;
                let src = &in_row[t0..t0 + (ot_end - ot_start)];
                for (d, s) in dst[ot_start..ot_end].iter_mut().zip(src) {
                    *d += wgt * *s;
                }
            } else {
                let mut t = ot_start * stride + off - pad;
                for d in dst[ot_start..ot_end].iter_mut() {
                    *d += wgt * in_row[t];
                    t += stride;
                }
            }
        }
    }
}

/// `out [c_out, l] = weight [c_out, c_in] · input [c_in, l] (+ bias)` — the
/// k=1 stride-1 conv1d fast path.
fn matmul_1x1(
    input: &[f32],
    c_in: usize,
    l: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    out: &mut [f32],
) {
    if l < 8 {
        // Degenerate widths: plain dot products.
        let row = |oc: usize, dst: &mut [f32]| {
            for (t, d) in dst.iter_mut().enumerate() {
                let mut acc = bias.map_or(0.0, |bb| bb[oc]);
                for ic in 0..c_in {
                    acc += weight[oc * c_in + ic] * input[ic * l + t];
                }
                *d = acc;
            }
        };
        if c_out * c_in * l >= PAR_MIN_MACS {
            out.par_chunks_mut(l)
                .enumerate()
                .for_each(|(oc, dst)| row(oc, dst));
        } else {
            out.chunks_mut(l)
                .enumerate()
                .for_each(|(oc, dst)| row(oc, dst));
        }
        return;
    }
    let block = |bi: usize, rows: &mut [f32]| {
        matmul_1x1_rows(input, c_in, l, weight, bias, bi * 4, rows);
    };
    if c_out * c_in * l >= PAR_MIN_MACS {
        out.par_chunks_mut(4 * l)
            .enumerate()
            .for_each(|(bi, rows)| block(bi, rows));
    } else {
        out.chunks_mut(4 * l)
            .enumerate()
            .for_each(|(bi, rows)| block(bi, rows));
    }
}

fn matmul_1x1_rows(
    input: &[f32],
    c_in: usize,
    l: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc0: usize,
    rows: &mut [f32],
) {
    let n = rows.len() / l;
    for (r, dst) in rows.chunks_mut(l).enumerate() {
        dst.fill(bias.map_or(0.0, |bb| bb[oc0 + r]));
    }
    if n == 4 {
        let (r0, rest) = rows.split_at_mut(l);
        let (r1, rest) = rest.split_at_mut(l);
        let (r2, r3) = rest.split_at_mut(l);
        for ic in 0..c_in {
            let x = &input[ic * l..ic * l + l];
            let w0 = weight[oc0 * c_in + ic];
            let w1 = weight[(oc0 + 1) * c_in + ic];
            let w2 = weight[(oc0 + 2) * c_in + ic];
            let w3 = weight[(oc0 + 3) * c_in + ic];
            for i in 0..l {
                r0[i] += w0 * x[i];
                r1[i] += w1 * x[i];
                r2[i] += w2 * x[i];
                r3[i] += w3 * x[i];
            }
        }
    } else {
        for (r, dst) in rows.chunks_mut(l).enumerate() {
            for ic in 0..c_in {
                let x = &input[ic * l..ic * l + l];
                let wgt = weight[(oc0 + r) * c_in + ic];
                for (d, s) in dst.iter_mut().zip(x) {
                    *d += wgt * *s;
                }
            }
        }
    }
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_identity_kernel() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0];
        let (out, l_out) = conv1d(&input, 1, 4, &weight, None, 1, 1, 1, 0, 1);
        assert_eq!(l_out, 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv1d_dilation_and_pad() {
        let input = [0.0, 1.0, 0.0, 0.0, 0.0];
        let weight = [1.0, 1.0, 1.0];
        let (out, l_out) = conv1d(&input, 1, 5, &weight, None, 1, 3, 1, 2, 2);
        assert_eq!(l_out, 5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn conv1d_strided_matches_naive() {
        let c_in = 3;
        let c_out = 2;
        let l = 17;
        let (k, stride, pad, dil) = (5, 2, 2, 1);
        let input: Vec<f32> = (0..c_in * l).map(|i| (i as f32 * 0.7).sin()).collect();
        let weight: Vec<f32> = (0..c_out * c_in * k)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let bias = [0.5f32, -0.25];
        let (out, l_out) = conv1d(
            &input,
            c_in,
            l,
            &weight,
            Some(&bias),
            c_out,
            k,
            stride,
            pad,
            dil,
        );
        for oc in 0..c_out {
            for ot in 0..l_out {
                let mut acc = bias[oc];
                for ic in 0..c_in {
                    for kk in 0..k {
                        let pos = ot * stride + kk * dil;
                        if pos >= pad && pos - pad < l {
                            acc += weight[(oc * c_in + ic) * k + kk] * input[ic * l + pos - pad];
                        }
                    }
                }
                assert_eq!(out[oc * l_out + ot], acc, "oc={oc} ot={ot}");
            }
        }
    }

    #[test]
    fn conv1d_k1_matches_naive() {
        let c_in = 5;
        let c_out = 6;
        let l = 11;
        let input: Vec<f32> = (0..c_in * l).map(|i| (i as f32 * 0.11).sin()).collect();
        let weight: Vec<f32> = (0..c_out * c_in).map(|i| (i as f32 * 0.17).cos()).collect();
        let (out, l_out) = conv1d(&input, c_in, l, &weight, None, c_out, 1, 1, 0, 1);
        assert_eq!(l_out, l);
        for oc in 0..c_out {
            for t in 0..l {
                let mut acc = 0.0f32;
                for ic in 0..c_in {
                    acc += weight[oc * c_in + ic] * input[ic * l + t];
                }
                assert_eq!(out[oc * l + t], acc, "oc={oc} t={t}");
            }
        }
    }

    #[test]
    fn sigmoid_is_monotone_and_bounded() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-20.0) < 1e-6);
        assert!((sigmoid(20.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-1.0) < sigmoid(0.0) && sigmoid(0.0) < sigmoid(1.0));
    }
}
