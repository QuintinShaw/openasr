use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

/// Shared conv activation vocabulary for `nn::conv`. The full set is the
/// reusable `nn/` building-block surface; not every variant is exercised by a
/// current model, but each is a supported option for new architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum ConvActivation {
    Gelu,
    GeluErf,
    Relu,
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Conv1dParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Conv2dParams {
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding_x: usize,
    pub padding_y: usize,
    pub dilation_x: usize,
    pub dilation_y: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvBlockSteps {
    pub conv: &'static str,
    pub bias: &'static str,
    pub activation: &'static str,
}

pub(crate) fn reshape_bias_4d<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    bias: GgmlCpuTensor<'a>,
    out_channels: usize,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    graph
        .reshape_4d(bias, 1, 1, out_channels, 1)
        .map_err(|source| map_err(step, source))
}

pub(crate) fn apply_conv_1d_bias_activation<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    params: Conv1dParams,
    activation: ConvActivation,
    steps: ConvBlockSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let conv = graph
        .conv_1d(
            weight,
            input,
            params.stride,
            params.padding,
            params.dilation,
        )
        .map_err(|source| map_err(steps.conv, source))?;
    let conv = graph
        .add(conv, bias)
        .map_err(|source| map_err(steps.bias, source))?;
    apply_activation(graph, conv, activation, steps.activation, map_err)
}

pub(crate) fn apply_conv_2d_bias_activation<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    params: Conv2dParams,
    activation: ConvActivation,
    steps: ConvBlockSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let conv = graph
        .conv_2d(
            weight,
            input,
            params.stride_x,
            params.stride_y,
            params.padding_x,
            params.padding_y,
            params.dilation_x,
            params.dilation_y,
        )
        .map_err(|source| map_err(steps.conv, source))?;
    let conv = graph
        .add(conv, bias)
        .map_err(|source| map_err(steps.bias, source))?;
    apply_activation(graph, conv, activation, steps.activation, map_err)
}

pub(crate) fn apply_conv_2d_depthwise_bias_activation<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    params: Conv2dParams,
    activation: Option<ConvActivation>,
    steps: ConvBlockSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let conv = graph
        .depthwise_conv_2d(
            weight,
            input,
            params.stride_x,
            params.stride_y,
            params.padding_x,
            params.padding_y,
            params.dilation_x,
            params.dilation_y,
        )
        .map_err(|source| map_err(steps.conv, source))?;
    let conv = graph
        .add(conv, bias)
        .map_err(|source| map_err(steps.bias, source))?;
    if let Some(activation) = activation {
        apply_activation(graph, conv, activation, steps.activation, map_err)
    } else {
        Ok(conv)
    }
}

fn apply_activation<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    activation: ConvActivation,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    match activation {
        ConvActivation::Gelu => graph.gelu(input),
        ConvActivation::GeluErf => graph.gelu_erf(input),
        ConvActivation::Relu => graph.relu(input),
        ConvActivation::Silu => graph.silu(input),
    }
    .map_err(|source| map_err(step, source))
}
