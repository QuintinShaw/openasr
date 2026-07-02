use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AffineLayerNormSteps {
    pub norm: &'static str,
    pub scale: &'static str,
    pub bias: &'static str,
}

pub(crate) fn apply_affine_layer_norm<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    epsilon: f32,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    steps: AffineLayerNormSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let normalized = graph
        .norm(input, epsilon)
        .map_err(|source| map_err(steps.norm, source))?;
    let scaled = graph
        .mul(normalized, weight)
        .map_err(|source| map_err(steps.scale, source))?;
    graph
        .add(scaled, bias)
        .map_err(|source| map_err(steps.bias, source))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RmsNormSteps {
    pub norm: &'static str,
    pub scale: &'static str,
}

pub(crate) fn apply_rms_norm<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    epsilon: f32,
    weight: GgmlCpuTensor<'a>,
    steps: RmsNormSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let normalized = graph
        .rms_norm(input, epsilon)
        .map_err(|source| map_err(steps.norm, source))?;
    graph
        .mul(normalized, weight)
        .map_err(|source| map_err(steps.scale, source))
}
