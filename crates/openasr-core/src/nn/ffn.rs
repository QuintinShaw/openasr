use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

/// Shared FFN activation vocabulary for `nn::ffn`. The full set is the reusable
/// `nn/` building-block surface; not every variant is exercised by a current
/// model, but each is a supported option for new architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum FeedForwardActivation {
    Gelu,
    GeluErf,
    Relu,
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FeedForwardResidualSteps {
    pub activation: &'static str,
    pub scale: Option<&'static str>,
    pub residual: &'static str,
}

pub(crate) fn apply_feed_forward_residual<'a, E, FUp, FDown, FMap>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    residual: GgmlCpuTensor<'a>,
    activation: FeedForwardActivation,
    output_scale: Option<f32>,
    steps: FeedForwardResidualSteps,
    project_up: FUp,
    project_down: FDown,
    map_err: FMap,
) -> Result<GgmlCpuTensor<'a>, E>
where
    FUp: FnOnce(&mut GgmlCpuGraphBuilder<'a>, GgmlCpuTensor<'a>) -> Result<GgmlCpuTensor<'a>, E>,
    FDown: FnOnce(&mut GgmlCpuGraphBuilder<'a>, GgmlCpuTensor<'a>) -> Result<GgmlCpuTensor<'a>, E>,
    FMap: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let up = project_up(graph, input)?;
    let activated = apply_activation(graph, up, activation, steps.activation, map_err)?;
    let mut down = project_down(graph, activated)?;
    if let Some(scale) = output_scale {
        down = graph
            .scale(down, scale)
            .map_err(|source| map_err(steps.scale.expect("scale step required"), source))?;
    }
    graph
        .add(down, residual)
        .map_err(|source| map_err(steps.residual, source))
}

/// Steps labels for `apply_gated_feed_forward_residual`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GatedFeedForwardResidualSteps {
    pub gate_activation: &'static str,
    pub gate_mul: &'static str,
    pub residual: &'static str,
}

/// SwiGLU / gated FFN: `down(act(gate(x)) * up(x)) + residual`.
///
/// `project_gate` and `project_up` are separate projections; their outputs are
/// element-wise multiplied after activating the gate. `project_down` reduces
/// back to the hidden size. Matches the Qwen3-ASR LLM decoder FFN pattern.
pub(crate) fn apply_gated_feed_forward_residual<'a, E, FGate, FUp, FDown, FMap>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    residual: GgmlCpuTensor<'a>,
    activation: FeedForwardActivation,
    steps: GatedFeedForwardResidualSteps,
    project_gate: FGate,
    project_up: FUp,
    project_down: FDown,
    map_err: FMap,
) -> Result<GgmlCpuTensor<'a>, E>
where
    FGate: FnOnce(&mut GgmlCpuGraphBuilder<'a>, GgmlCpuTensor<'a>) -> Result<GgmlCpuTensor<'a>, E>,
    FUp: FnOnce(&mut GgmlCpuGraphBuilder<'a>, GgmlCpuTensor<'a>) -> Result<GgmlCpuTensor<'a>, E>,
    FDown: FnOnce(&mut GgmlCpuGraphBuilder<'a>, GgmlCpuTensor<'a>) -> Result<GgmlCpuTensor<'a>, E>,
    FMap: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let gate = project_gate(graph, input)?;
    let gate = apply_activation(graph, gate, activation, steps.gate_activation, map_err)?;
    let up = project_up(graph, input)?;
    let gated = graph
        .mul(gate, up)
        .map_err(|source| map_err(steps.gate_mul, source))?;
    let down = project_down(graph, gated)?;
    graph
        .add(down, residual)
        .map_err(|source| map_err(steps.residual, source))
}

fn apply_activation<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    activation: FeedForwardActivation,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    match activation {
        FeedForwardActivation::Gelu => graph.gelu(input),
        FeedForwardActivation::GeluErf => graph.gelu_erf(input),
        FeedForwardActivation::Relu => graph.relu(input),
        FeedForwardActivation::Silu => graph.silu(input),
    }
    .map_err(|source| map_err(step, source))
}
