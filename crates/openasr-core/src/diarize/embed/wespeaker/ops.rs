//! Shared ggml building blocks for the WeSpeaker ResNet backbone.
//!
//! Layout: 2D activations use ggml `ne = [T, F, C, N]`, forced by `conv_2d`
//! (`[W, H, Cin, N]`, torch `(N,C,F,T)` reversed). BatchNorm is folded on the
//! host into scale/shift and applied as mul+add.

use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

pub(super) type OpResult<'a> = Result<GgmlCpuTensor<'a>, GgmlCpuGraphError>;

fn map_err(_stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GgmlCpuGraphError + Copy {
    |source| source
}

pub(super) use super::super::ggml_affine::{apply_channel_affine_2d, batchnorm_affine};

pub(super) fn conv2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    x2d: GgmlCpuTensor<'a>,
    stride: usize,
    padding: usize,
) -> OpResult<'a> {
    graph
        .conv_2d(kernel, x2d, stride, stride, padding, padding, 1, 1)
        .map_err(map_err("conv2d"))
}

pub(super) fn conv_bn<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    x2d: GgmlCpuTensor<'a>,
    c_out: usize,
    stride: usize,
    padding: usize,
    scale: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
    relu: bool,
) -> OpResult<'a> {
    let m = map_err("conv_bn");
    let conv = conv2d(graph, kernel, x2d, stride, padding)?;
    let out = apply_channel_affine_2d(graph, conv, c_out, scale, shift)?;
    if relu {
        graph.relu(out).map_err(m)
    } else {
        Ok(out)
    }
}

pub(super) struct BasicBlockWeights<'a> {
    pub conv1: GgmlCpuTensor<'a>,
    pub bn1_scale: GgmlCpuTensor<'a>,
    pub bn1_shift: GgmlCpuTensor<'a>,
    pub conv2: GgmlCpuTensor<'a>,
    pub bn2_scale: GgmlCpuTensor<'a>,
    pub bn2_shift: GgmlCpuTensor<'a>,
    pub shortcut_conv: Option<GgmlCpuTensor<'a>>,
    pub shortcut_scale: Option<GgmlCpuTensor<'a>>,
    pub shortcut_shift: Option<GgmlCpuTensor<'a>>,
}

pub(super) fn basic_block<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c_out: usize,
    stride: usize,
    w: &BasicBlockWeights<'a>,
) -> OpResult<'a> {
    let m = map_err("basic_block");
    let out = conv_bn(
        graph,
        w.conv1,
        x2d,
        c_out,
        stride,
        1,
        w.bn1_scale,
        w.bn1_shift,
        true,
    )?;
    let out = conv_bn(
        graph,
        w.conv2,
        out,
        c_out,
        1,
        1,
        w.bn2_scale,
        w.bn2_shift,
        false,
    )?;
    let shortcut = match (w.shortcut_conv, w.shortcut_scale, w.shortcut_shift) {
        (Some(conv), Some(scale), Some(shift)) => {
            conv_bn(graph, conv, x2d, c_out, stride, 0, scale, shift, false)?
        }
        _ => x2d,
    };
    let added = graph.add(out, shortcut).map_err(m)?;
    graph.relu(added).map_err(m)
}

pub(super) struct BottleneckWeights<'a> {
    pub conv1: GgmlCpuTensor<'a>,
    pub bn1_scale: GgmlCpuTensor<'a>,
    pub bn1_shift: GgmlCpuTensor<'a>,
    pub conv2: GgmlCpuTensor<'a>,
    pub bn2_scale: GgmlCpuTensor<'a>,
    pub bn2_shift: GgmlCpuTensor<'a>,
    pub conv3: GgmlCpuTensor<'a>,
    pub bn3_scale: GgmlCpuTensor<'a>,
    pub bn3_shift: GgmlCpuTensor<'a>,
    pub shortcut_conv: Option<GgmlCpuTensor<'a>>,
    pub shortcut_scale: Option<GgmlCpuTensor<'a>>,
    pub shortcut_shift: Option<GgmlCpuTensor<'a>>,
    pub planes: usize,
    pub c_out: usize,
}

pub(super) fn bottleneck_block<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    stride: usize,
    w: &BottleneckWeights<'a>,
) -> OpResult<'a> {
    let m = map_err("bottleneck_block");
    let out = conv_bn(
        graph,
        w.conv1,
        x2d,
        w.planes,
        1,
        0,
        w.bn1_scale,
        w.bn1_shift,
        true,
    )?;
    let out = conv_bn(
        graph,
        w.conv2,
        out,
        w.planes,
        stride,
        1,
        w.bn2_scale,
        w.bn2_shift,
        true,
    )?;
    let out = conv_bn(
        graph,
        w.conv3,
        out,
        w.c_out,
        1,
        0,
        w.bn3_scale,
        w.bn3_shift,
        false,
    )?;
    let shortcut = match (w.shortcut_conv, w.shortcut_scale, w.shortcut_shift) {
        (Some(conv), Some(scale), Some(shift)) => {
            conv_bn(graph, conv, x2d, w.c_out, stride, 0, scale, shift, false)?
        }
        _ => x2d,
    };
    let added = graph.add(out, shortcut).map_err(m)?;
    graph.relu(added).map_err(m)
}

/// Flatten `[C, F, T']` as torch `reshape(C*F, T')`: ggml `ne=[T, F, C, 1]`
/// merges F (fast) and C (slow) into `ne=[T, C*F]`.
pub(super) fn flatten_cft<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    f: usize,
    t: usize,
) -> OpResult<'a> {
    let m = map_err("flatten_cft");
    let x2d = graph.cont(x2d).map_err(m)?;
    let merged = graph.reshape_3d(x2d, t, f * c, 1).map_err(m)?;
    let flat = graph.reshape_2d(merged, t, f * c).map_err(m)?;
    graph.cont(flat).map_err(m)
}

/// Official TSTP: `cat(mean, sqrt(unbiased_var + 1e-7))` over time (`ne0`).
pub(super) fn tstp<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_tc: GgmlCpuTensor<'a>,
    t: usize,
    cf: usize,
    eps_1e7: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    let m = map_err("tstp");
    let x_tc = graph.cont(x_tc).map_err(m)?;
    let mean_row = graph.mean_rows(x_tc).map_err(m)?;
    let mean_bc = graph.repeat_4d(mean_row, t, cf, 1, 1).map_err(m)?;
    let mean_bc = graph
        .reshape_2d(graph.cont(mean_bc).map_err(m)?, t, cf)
        .map_err(m)?;
    let centered = graph.sub(x_tc, mean_bc).map_err(m)?;
    let sq = graph.sqr(centered).map_err(m)?;
    let sum_sq = graph.sum_rows(sq).map_err(m)?;
    let var = graph.scale(sum_sq, 1.0 / ((t - 1) as f32)).map_err(m)?;
    let var_eps = graph.add(var, eps_1e7).map_err(m)?;
    let std_row = graph.sqrt(var_eps).map_err(m)?;
    let mean_flat = graph.reshape_1d(mean_row, cf).map_err(m)?;
    let std_flat = graph.reshape_1d(std_row, cf).map_err(m)?;
    graph.concat(mean_flat, std_flat, 0).map_err(m)
}

pub(super) fn linear_1d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    x: GgmlCpuTensor<'a>,
    in_dim: usize,
    out_dim: usize,
) -> OpResult<'a> {
    let m = map_err("linear_1d");
    let x2 = graph.reshape_2d(x, in_dim, 1).map_err(m)?;
    let projected = graph.mul_mat(weight, x2).map_err(m)?;
    let biased = graph.add(projected, bias).map_err(m)?;
    graph.reshape_1d(biased, out_dim).map_err(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    #[test]
    fn bottleneck_output_channels_match_expansion() {
        let mut runner = GgmlCpuGraphRunner::new(
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Cpu),
        )
        .expect("runner");
        let mut graph = runner.start_graph();
        let t = 8;
        let f = 10;
        let c_in = 32;
        let planes = 32;
        let c_out = 128;
        let x = graph.new_tensor_4d_f32(t, f, c_in, 1, "bn_in").expect("x");
        let conv1 = graph
            .new_tensor_4d_f32(1, 1, c_in, planes, "c1")
            .expect("c1");
        let conv2 = graph
            .new_tensor_4d_f32(3, 3, planes, planes, "c2")
            .expect("c2");
        let conv3 = graph
            .new_tensor_4d_f32(1, 1, planes, c_out, "c3")
            .expect("c3");
        let sc = graph
            .new_tensor_4d_f32(1, 1, c_in, c_out, "sc")
            .expect("sc");
        let ones = |len, name| graph.new_tensor_1d_f32(len, name).expect("bn");
        let w = BottleneckWeights {
            conv1,
            bn1_scale: ones(planes, "s1"),
            bn1_shift: ones(planes, "h1"),
            conv2,
            bn2_scale: ones(planes, "s2"),
            bn2_shift: ones(planes, "h2"),
            conv3,
            bn3_scale: ones(c_out, "s3"),
            bn3_shift: ones(c_out, "h3"),
            shortcut_conv: Some(sc),
            shortcut_scale: Some(ones(c_out, "ss")),
            shortcut_shift: Some(ones(c_out, "sh")),
            planes,
            c_out,
        };
        let out = bottleneck_block(&graph, x, 1, &w).expect("bottleneck");
        graph.set_output(out).expect("set_output");
    }
}
