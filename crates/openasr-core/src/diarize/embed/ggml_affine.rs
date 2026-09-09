//! Host BatchNorm fold and ggml channel affine shared by speaker-embedder graphs.
//!
//! ReDimNet2 and WeSpeaker ResNet both eval-fold BN on the host, then apply
//! `y = x * scale + shift` on a `[T,F,C,N]` activation. Keep that math in one
//! place so a third embedder graph does not copy it.

use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

pub(super) type AffineResult<'a> = Result<GgmlCpuTensor<'a>, GgmlCpuGraphError>;

/// Precomputed per-channel affine for eval-mode BatchNorm:
/// `y = gamma*(x-mean)/sqrt(var+eps) + beta = x*scale + shift`.
pub(super) fn batchnorm_affine(
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = gamma.len();
    let mut scale = vec![0.0f32; n];
    let mut shift = vec![0.0f32; n];
    for i in 0..n {
        let s = gamma[i] / (running_var[i] + eps).sqrt();
        scale[i] = s;
        shift[i] = beta[i] - running_mean[i] * s;
    }
    (scale, shift)
}

/// Apply a precomputed per-channel affine to a 2D tensor (`ne=[T,F,C,N]`,
/// `scale`/`shift` are `ne=[C]`, broadcast via a `[1,1,C,1]` reshape).
pub(super) fn apply_channel_affine_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    scale: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> AffineResult<'a> {
    let scale4d = graph.reshape_4d(scale, 1, 1, c, 1)?;
    let shift4d = graph.reshape_4d(shift, 1, 1, c, 1)?;
    let scaled = graph.mul(x2d, scale4d)?;
    graph.add(scaled, shift4d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_affine_matches_eval_formula() {
        let gamma = [2.0f32];
        let beta = [0.5f32];
        let mean = [1.0f32];
        let var = [3.0f32];
        let (scale, shift) = batchnorm_affine(&gamma, &beta, &mean, &var, 1.0);
        let s = 2.0 / 2.0;
        assert!((scale[0] - s).abs() < 1e-6);
        assert!((shift[0] - (0.5 - 1.0 * s)).abs() < 1e-6);
    }
}
