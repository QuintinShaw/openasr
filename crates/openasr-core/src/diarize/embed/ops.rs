//! Generic, dependency-free neural ops for the speaker-embedding networks.
//!
//! Tensors are flat row-major `Vec<f32>` with explicit shapes (time length is
//! dynamic), mirroring the VAD forward pass. These are the load-bearing
//! primitives for the WeSpeaker port: 2-D conv (ResNet head), inference
//! batch-norm (folded from running stats), and statistics pooling
//! (mean + Bessel-corrected std).
//!
//! The conv kernels use small hand-written loops for tiny cases and an
//! im2col+SGEMM path for large bias-free 3x3 ResNet blocks. The SGEMM path has a
//! different accumulation order than the naive plane loop, so parity is guarded
//! with bounded max-abs/RMS error tests instead of bit-exact assertions.

use rayon::prelude::*;

/// Multiply-accumulate count below which a conv runs single-threaded; tiny ops
/// (e.g. the CAM mask MLPs) are cheaper than a rayon fork-join.
const PAR_MIN_MACS: usize = 1 << 16;

/// 2-D cross-correlation with zero padding and optional bias.
///
/// `input` is `[c_in, h, w]` row-major; `weight` is `[c_out, c_in, kh, kw]`.
/// Returns `([c_out, h_out, w_out], h_out, w_out)`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn conv2d(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    let plane = |oc: usize, dst: &mut [f32]| {
        conv2d_plane(
            input, c_in, h, w, weight, bias, oc, kh, kw, stride_h, stride_w, pad_h, pad_w, h_out,
            w_out, dst,
        );
    };
    if c_out * h_out * w_out * c_in * kh * kw >= PAR_MIN_MACS {
        out.par_chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(oc, dst)| plane(oc, dst));
    } else {
        out.chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(oc, dst)| plane(oc, dst));
    }
    (out, h_out, w_out)
}

/// 2-D cross-correlation followed by per-output-channel batch-norm affine and
/// optional ReLU. Applying the affine inside the output-plane worker avoids a
/// second full-tensor pass for ResNet-style conv+BN(+ReLU) blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_batchnorm(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    relu: bool,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;
    if bias.is_none() && kh == 3 && kw == 3 && c_in >= 16 && h_out * w_out >= 512 {
        return conv2d_batchnorm_im2col_sgemm(
            input, c_in, h, w, weight, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w, h_out,
            w_out, gamma, beta, mean, var, eps, relu,
        );
    }
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    let plane_len = h_out * w_out;
    let group = |bi: usize, rows: &mut [f32]| {
        let oc0 = bi * 4;
        if rows.len() == 4 * plane_len {
            conv2d_batchnorm_planes4(
                input, c_in, h, w, weight, bias, oc0, kh, kw, stride_h, stride_w, pad_h, pad_w,
                h_out, w_out, gamma, beta, mean, var, eps, relu, rows,
            );
        } else {
            rows.chunks_mut(plane_len)
                .enumerate()
                .for_each(|(offset, dst)| {
                    conv2d_batchnorm_plane(
                        input,
                        c_in,
                        h,
                        w,
                        weight,
                        bias,
                        oc0 + offset,
                        kh,
                        kw,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        h_out,
                        w_out,
                        gamma,
                        beta,
                        mean,
                        var,
                        eps,
                        relu,
                        dst,
                    );
                });
        }
    };
    if c_out * h_out * w_out * c_in * kh * kw >= PAR_MIN_MACS {
        out.par_chunks_mut(4 * plane_len)
            .enumerate()
            .for_each(|(bi, rows)| group(bi, rows));
    } else {
        out.chunks_mut(4 * plane_len)
            .enumerate()
            .for_each(|(bi, rows)| group(bi, rows));
    }
    (out, h_out, w_out)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_batchnorm_im2col_sgemm(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    relu: bool,
) -> (Vec<f32>, usize, usize) {
    let k = c_in * kh * kw;
    let n = h_out * w_out;
    let mut cols = vec![0.0f32; k * n];
    let fill_col_row = |k_idx: usize, row: &mut [f32]| {
        let ic = k_idx / (kh * kw);
        let rem = k_idx % (kh * kw);
        let ky = rem / kw;
        let kx = rem % kw;
        for oy in 0..h_out {
            let iy = oy * stride_h + ky;
            let dst_row = &mut row[oy * w_out..(oy + 1) * w_out];
            if iy < pad_h || iy >= pad_h + h {
                dst_row.fill(0.0);
                continue;
            }
            let y = iy - pad_h;
            let in_row = &input[ic * h * w + y * w..ic * h * w + y * w + w];
            if stride_w == 1 {
                let ox_start = pad_w.saturating_sub(kx);
                let ox_end = (w + pad_w).saturating_sub(kx).min(w_out);
                dst_row[..ox_start].fill(0.0);
                dst_row[ox_end..].fill(0.0);
                if ox_start < ox_end {
                    let t0 = ox_start + kx - pad_w;
                    dst_row[ox_start..ox_end]
                        .copy_from_slice(&in_row[t0..t0 + (ox_end - ox_start)]);
                }
            } else {
                for (ox, dst) in dst_row.iter_mut().enumerate() {
                    let ix = ox * stride_w + kx;
                    *dst = if ix >= pad_w && ix < pad_w + w {
                        in_row[ix - pad_w]
                    } else {
                        0.0
                    };
                }
            }
        }
    };
    if k * n >= PAR_MIN_MACS {
        cols.par_chunks_mut(n)
            .enumerate()
            .for_each(|(k_idx, row)| fill_col_row(k_idx, row));
    } else {
        cols.chunks_mut(n)
            .enumerate()
            .for_each(|(k_idx, row)| fill_col_row(k_idx, row));
    }

    let mut out = vec![0.0f32; c_out * n];
    const GEMM_ROW_BLOCK: usize = 16;
    out.par_chunks_mut(GEMM_ROW_BLOCK * n)
        .enumerate()
        .for_each(|(block_idx, rows)| {
            let oc0 = block_idx * GEMM_ROW_BLOCK;
            let rows_m = rows.len() / n;
            let weight_block = &weight[oc0 * k..(oc0 + rows_m) * k];
            // SAFETY: weight_block is row-major [rows_m, k], cols is row-major
            // [k, n], and rows is row-major [rows_m, n] with non-overlapping
            // output rows from par_chunks_mut.
            unsafe {
                matrixmultiply::sgemm(
                    rows_m,
                    k,
                    n,
                    1.0,
                    weight_block.as_ptr(),
                    k as isize,
                    1,
                    cols.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    rows.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
            rows.chunks_mut(n).enumerate().for_each(|(offset, row)| {
                let oc = oc0 + offset;
                apply_batchnorm_plane(row, gamma[oc], beta[oc], mean[oc], var[oc], eps, relu);
            });
        });
    (out, h_out, w_out)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_batchnorm_plane(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    relu: bool,
    dst: &mut [f32],
) {
    conv2d_plane(
        input, c_in, h, w, weight, bias, oc, kh, kw, stride_h, stride_w, pad_h, pad_w, h_out,
        w_out, dst,
    );
    apply_batchnorm_plane(dst, gamma[oc], beta[oc], mean[oc], var[oc], eps, relu);
}

#[allow(clippy::too_many_arguments)]
fn conv2d_batchnorm_planes4(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc0: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    relu: bool,
    rows: &mut [f32],
) {
    let plane_len = h_out * w_out;
    let (r0, rest) = rows.split_at_mut(plane_len);
    let (r1, rest) = rest.split_at_mut(plane_len);
    let (r2, r3) = rest.split_at_mut(plane_len);
    r0.fill(bias.map_or(0.0, |bb| bb[oc0]));
    r1.fill(bias.map_or(0.0, |bb| bb[oc0 + 1]));
    r2.fill(bias.map_or(0.0, |bb| bb[oc0 + 2]));
    r3.fill(bias.map_or(0.0, |bb| bb[oc0 + 3]));

    for oy in 0..h_out {
        let row0 = &mut r0[oy * w_out..(oy + 1) * w_out];
        let row1 = &mut r1[oy * w_out..(oy + 1) * w_out];
        let row2 = &mut r2[oy * w_out..(oy + 1) * w_out];
        let row3 = &mut r3[oy * w_out..(oy + 1) * w_out];
        for ic in 0..c_in {
            for ky in 0..kh {
                let iy = oy * stride_h + ky;
                if iy < pad_h || iy >= pad_h + h {
                    continue;
                }
                let y = iy - pad_h;
                let in_row = &input[ic * h * w + y * w..ic * h * w + y * w + w];
                let w_base0 = ((oc0 * c_in + ic) * kh + ky) * kw;
                let w_base1 = ((((oc0 + 1) * c_in) + ic) * kh + ky) * kw;
                let w_base2 = ((((oc0 + 2) * c_in) + ic) * kh + ky) * kw;
                let w_base3 = ((((oc0 + 3) * c_in) + ic) * kh + ky) * kw;
                for kx in 0..kw {
                    let w0 = weight[w_base0 + kx];
                    let w1 = weight[w_base1 + kx];
                    let w2 = weight[w_base2 + kx];
                    let w3 = weight[w_base3 + kx];
                    if stride_w == 1 {
                        let ox_start = pad_w.saturating_sub(kx);
                        let ox_end = (w + pad_w).saturating_sub(kx).min(w_out);
                        if ox_start >= ox_end {
                            continue;
                        }
                        let t0 = ox_start + kx - pad_w;
                        let src = &in_row[t0..t0 + (ox_end - ox_start)];
                        for (i, s) in src.iter().enumerate() {
                            let ox = ox_start + i;
                            let s = *s;
                            row0[ox] += w0 * s;
                            row1[ox] += w1 * s;
                            row2[ox] += w2 * s;
                            row3[ox] += w3 * s;
                        }
                    } else {
                        for ox in 0..w_out {
                            let ix = ox * stride_w + kx;
                            if ix >= pad_w && ix < pad_w + w {
                                let s = in_row[ix - pad_w];
                                row0[ox] += w0 * s;
                                row1[ox] += w1 * s;
                                row2[ox] += w2 * s;
                                row3[ox] += w3 * s;
                            }
                        }
                    }
                }
            }
        }
    }

    apply_batchnorm_plane(r0, gamma[oc0], beta[oc0], mean[oc0], var[oc0], eps, relu);
    apply_batchnorm_plane(
        r1,
        gamma[oc0 + 1],
        beta[oc0 + 1],
        mean[oc0 + 1],
        var[oc0 + 1],
        eps,
        relu,
    );
    apply_batchnorm_plane(
        r2,
        gamma[oc0 + 2],
        beta[oc0 + 2],
        mean[oc0 + 2],
        var[oc0 + 2],
        eps,
        relu,
    );
    apply_batchnorm_plane(
        r3,
        gamma[oc0 + 3],
        beta[oc0 + 3],
        mean[oc0 + 3],
        var[oc0 + 3],
        eps,
        relu,
    );
}

fn apply_batchnorm_plane(
    dst: &mut [f32],
    gamma: f32,
    beta: f32,
    mean: f32,
    var: f32,
    eps: f32,
    relu: bool,
) {
    let scale = gamma / (var + eps).sqrt();
    let shift = beta - mean * scale;
    if relu {
        for v in dst {
            *v = (*v * scale + shift).max(0.0);
        }
    } else {
        for v in dst {
            *v = *v * scale + shift;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d_plane(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    dst: &mut [f32],
) {
    dst.fill(bias.map_or(0.0, |bb| bb[oc]));
    for oy in 0..h_out {
        let row_out = &mut dst[oy * w_out..(oy + 1) * w_out];
        for ic in 0..c_in {
            for ky in 0..kh {
                let iy = oy * stride_h + ky;
                if iy < pad_h || iy >= pad_h + h {
                    continue;
                }
                let y = iy - pad_h;
                let in_row = &input[ic * h * w + y * w..ic * h * w + y * w + w];
                let w_base = ((oc * c_in + ic) * kh + ky) * kw;
                for kx in 0..kw {
                    let wgt = weight[w_base + kx];
                    if stride_w == 1 {
                        // ox + kx - pad_w must land in [0, w).
                        let ox_start = pad_w.saturating_sub(kx);
                        let ox_end = (w + pad_w).saturating_sub(kx).min(w_out);
                        if ox_start >= ox_end {
                            continue;
                        }
                        let t0 = ox_start + kx - pad_w;
                        let src = &in_row[t0..t0 + (ox_end - ox_start)];
                        for (d, s) in row_out[ox_start..ox_end].iter_mut().zip(src) {
                            *d += wgt * *s;
                        }
                    } else {
                        for (ox, d) in row_out.iter_mut().enumerate() {
                            let ix = ox * stride_w + kx;
                            if ix >= pad_w && ix < pad_w + w {
                                *d += wgt * in_row[ix - pad_w];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 1-D dilated cross-correlation with zero padding and optional bias.
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
/// k=1 stride-1 conv1d, i.e. the D-TDNN bottleneck matmuls that dominate the
/// forward pass. Register-blocked over 4 output rows so one pass over the
/// input feeds 4 independent FMA streams; each row still accumulates `ic` in
/// order, so every element matches the naive loop bit-for-bit.
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
        // Degenerate widths (e.g. the pooled [2c, 1] dense): plain dot products.
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

/// Inference batch-norm over channels of a `[c, l]` (or `[c, h*w]`) tensor:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`, applied per channel.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn batchnorm_inplace(
    x: &mut [f32],
    c: usize,
    l: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) {
    let apply = |ch: usize, row: &mut [f32]| {
        let scale = gamma[ch] / (var[ch] + eps).sqrt();
        let shift = beta[ch] - mean[ch] * scale;
        for v in row {
            *v = *v * scale + shift;
        }
    };
    if c * l >= PAR_MIN_MACS {
        x.par_chunks_mut(l)
            .enumerate()
            .for_each(|(ch, row)| apply(ch, row));
    } else {
        x.chunks_mut(l)
            .enumerate()
            .for_each(|(ch, row)| apply(ch, row));
    }
}

/// Fused inference batch-norm + ReLU into a fresh buffer (one read, one write
/// instead of copy + normalize + clamp):
/// `y = max(0, (x - mean) / sqrt(var + eps) * gamma + beta)` per channel.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn batchnorm_relu(
    x: &[f32],
    c: usize,
    l: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; c * l];
    out.chunks_mut(l).enumerate().for_each(|(ch, dst)| {
        let scale = gamma[ch] / (var[ch] + eps).sqrt();
        let shift = beta[ch] - mean[ch] * scale;
        let src = &x[ch * l..ch * l + l];
        for (d, s) in dst.iter_mut().zip(src) {
            *d = (*s * scale + shift).max(0.0);
        }
    });
    out
}

#[cfg(test)]
pub(crate) fn relu_inplace(x: &mut [f32]) {
    let apply = |v: &mut f32| {
        if *v < 0.0 {
            *v = 0.0;
        }
    };
    if x.len() >= PAR_MIN_MACS {
        x.par_iter_mut().for_each(apply);
    } else {
        x.iter_mut().for_each(apply);
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

/// Statistics pooling over time of a `[c, t]` tensor: concatenates the per-
/// channel mean and the **Bessel-corrected** standard deviation, yielding `2c`.
/// Matches the ONNX tail (`std = sqrt(mean((x-mean)^2) * t/(t-1))`).
pub(crate) fn stats_pool(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 2 * c];
    if t == 0 {
        return out;
    }
    let bessel = if t > 1 {
        t as f32 / (t as f32 - 1.0)
    } else {
        1.0
    };
    let (mean_out, std_out) = out.split_at_mut(c);
    let compute = |ch: usize, mean_slot: &mut f32, std_slot: &mut f32| {
        let row = &x[ch * t..ch * t + t];
        let mean = row.iter().sum::<f32>() / t as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / t as f32;
        *mean_slot = mean;
        *std_slot = (var * bessel).sqrt();
    };
    if c * t >= PAR_MIN_MACS {
        mean_out
            .par_iter_mut()
            .zip(std_out.par_iter_mut())
            .enumerate()
            .for_each(|(ch, (mean_slot, std_slot))| compute(ch, mean_slot, std_slot));
    } else {
        mean_out
            .iter_mut()
            .zip(std_out.iter_mut())
            .enumerate()
            .for_each(|(ch, (mean_slot, std_slot))| compute(ch, mean_slot, std_slot));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_identity_kernel() {
        // c_in=1, l=4, identity 1-tap kernel, no pad.
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0]; // [c_out=1, c_in=1, k=1]
        let (out, l_out) = conv1d(&input, 1, 4, &weight, None, 1, 1, 1, 0, 1);
        assert_eq!(l_out, 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv1d_dilation_and_pad() {
        // k=3, dilation=2, pad=2 -> same length; symmetric kernel sums neighbors.
        let input = [0.0, 1.0, 0.0, 0.0, 0.0];
        let weight = [1.0, 1.0, 1.0]; // sums taps at t-2, t, t+2
        let (out, l_out) = conv1d(&input, 1, 5, &weight, None, 1, 3, 1, 2, 2);
        assert_eq!(l_out, 5);
        // position 1 has the 1.0 at its center tap; positions 3 picks t-2=1.
        assert_eq!(out[1], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn conv1d_strided_matches_naive() {
        // Strided + padded conv (the tdnn shape, scaled down) against a naive
        // reference, exercising the trimmed-range axpy path.
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
        // The matmul fast path (k=1) against a naive reference, wide enough to
        // hit the 4-row blocked kernel and a remainder block.
        let c_in = 5;
        let c_out = 6;
        let l = 11;
        let input: Vec<f32> = (0..c_in * l).map(|i| (i as f32 * 0.11).sin()).collect();
        let weight: Vec<f32> = (0..c_out * c_in).map(|i| (i as f32 * 0.23).cos()).collect();
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
    fn conv2d_3x3_same_pad_sum() {
        // 1 channel, 3x3 ones kernel, pad 1 -> each output is the 3x3 neighborhood sum.
        let input = [
            1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0,
        ];
        let weight = vec![1.0; 9];
        let (out, h, w) = conv2d(&input, 1, 3, 3, &weight, None, 1, 3, 3, 1, 1, 1, 1);
        assert_eq!((h, w), (3, 3));
        assert_eq!(out[4], 9.0); // center sees all 9 ones
        assert_eq!(out[0], 4.0); // corner sees 4 (rest zero-padded)
    }

    #[test]
    fn conv2d_freq_stride_downsamples_height() {
        // height stride 2 halves the freq axis (rounding up via the conv formula).
        let input = vec![1.0; 4 * 3]; // [1, h=4, w=3]
        let weight = vec![0.0; 9];
        let (_, h, w) = conv2d(&input, 1, 4, 3, &weight, None, 1, 3, 3, 2, 1, 1, 1);
        assert_eq!((h, w), (2, 3));
    }

    #[test]
    fn conv2d_batchnorm_im2col_sgemm_matches_naive_with_bounded_error() {
        let c_in = 16;
        let c_out = 7;
        let h = 23;
        let w = 23;
        let kh = 3;
        let kw = 3;
        let stride_h = 1;
        let stride_w = 1;
        let pad_h = 1;
        let pad_w = 1;
        let h_out = h;
        let w_out = w;
        let input: Vec<f32> = (0..c_in * h * w)
            .map(|i| ((i as f32) * 0.013).sin() * 0.7)
            .collect();
        let weight: Vec<f32> = (0..c_out * c_in * kh * kw)
            .map(|i| ((i as f32) * 0.017).cos() * 0.11)
            .collect();
        let gamma: Vec<f32> = (0..c_out).map(|i| 0.8 + i as f32 * 0.03).collect();
        let beta: Vec<f32> = (0..c_out).map(|i| -0.2 + i as f32 * 0.02).collect();
        let mean: Vec<f32> = (0..c_out).map(|i| -0.1 + i as f32 * 0.01).collect();
        let var: Vec<f32> = (0..c_out).map(|i| 0.5 + i as f32 * 0.04).collect();

        let (fast, fast_h, fast_w) = conv2d_batchnorm(
            &input, c_in, h, w, &weight, None, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w,
            &gamma, &beta, &mean, &var, 1e-5, true,
        );
        assert_eq!((fast_h, fast_w), (h_out, w_out));

        let plane_len = h_out * w_out;
        let mut reference = vec![0.0; c_out * plane_len];
        for oc in 0..c_out {
            conv2d_batchnorm_plane(
                &input,
                c_in,
                h,
                w,
                &weight,
                None,
                oc,
                kh,
                kw,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                h_out,
                w_out,
                &gamma,
                &beta,
                &mean,
                &var,
                1e-5,
                true,
                &mut reference[oc * plane_len..(oc + 1) * plane_len],
            );
        }

        let mut max_abs = 0.0_f32;
        let mut sum_sq = 0.0_f64;
        for (actual, expected) in fast.iter().zip(&reference) {
            let diff = (actual - expected).abs();
            max_abs = max_abs.max(diff);
            sum_sq += f64::from(diff * diff);
        }
        let rms = (sum_sq / fast.len() as f64).sqrt() as f32;
        println!("conv2d im2col sgemm parity max_abs={max_abs:.9} rms={rms:.9}");
        const MAX_ABS_TOLERANCE: f32 = 2.0e-6;
        const RMS_TOLERANCE: f32 = 5.0e-7;
        assert!(max_abs <= MAX_ABS_TOLERANCE, "max_abs {max_abs}");
        assert!(rms <= RMS_TOLERANCE, "rms {rms}");
    }

    #[test]
    fn batchnorm_normalizes_to_unit() {
        let mut x = vec![1.0, 3.0]; // c=1, l=2, mean 2 var 1
        batchnorm_inplace(&mut x, 1, 2, &[1.0], &[0.0], &[2.0], &[1.0], 0.0);
        assert!((x[0] + 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batchnorm_relu_matches_separate_passes() {
        let x = vec![1.0, 3.0, -4.0, 0.5]; // c=2, l=2
        let gamma = [1.5, 2.0];
        let beta = [0.1, -0.2];
        let mean = [2.0, -1.0];
        let var = [1.0, 4.0];
        let mut reference = x.clone();
        batchnorm_inplace(&mut reference, 2, 2, &gamma, &beta, &mean, &var, 1e-5);
        relu_inplace(&mut reference);
        let fused = batchnorm_relu(&x, 2, 2, &gamma, &beta, &mean, &var, 1e-5);
        assert_eq!(fused, reference);
    }

    #[test]
    fn stats_pool_mean_and_bessel_std() {
        // c=1, t=4, values 1,2,3,4 -> mean 2.5, biased var 1.25, bessel*4/3.
        let x = [1.0, 2.0, 3.0, 4.0];
        let out = stats_pool(&x, 1, 4);
        assert!((out[0] - 2.5).abs() < 1e-6);
        let expected_std = (1.25f32 * 4.0 / 3.0).sqrt();
        assert!((out[1] - expected_std).abs() < 1e-6);
    }
}
