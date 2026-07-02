use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};

pub(crate) const STANDARD_HEAD_PERMUTE_AXES: [i32; 4] = [0, 2, 1, 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionHeadLayout {
    pub head_dim: usize,
    pub attention_heads: usize,
    pub sequence_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionReshapeSteps {
    pub reshape: &'static str,
    pub permute: &'static str,
    pub cont: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionValueMergeSteps {
    pub value_permute: &'static str,
    pub value_cont: &'static str,
    pub context_mul: &'static str,
    pub context_merge_permute: &'static str,
    pub context_merge_cont: &'static str,
    pub context_merge_reshape: &'static str,
}

pub(crate) fn reshape_projection_to_attention_heads<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    layout: AttentionHeadLayout,
    permute_axes: [i32; 4],
    contiguous: bool,
    steps: AttentionReshapeSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let reshaped = graph
        .reshape_3d(
            projection,
            layout.head_dim,
            layout.attention_heads,
            layout.sequence_len,
        )
        .map_err(|source| map_err(steps.reshape, source))?;
    let permuted = graph
        .permute(
            reshaped,
            permute_axes[0],
            permute_axes[1],
            permute_axes[2],
            permute_axes[3],
        )
        .map_err(|source| map_err(steps.permute, source))?;
    if contiguous {
        graph
            .cont(permuted)
            .map_err(|source| map_err(steps.cont, source))
    } else {
        Ok(permuted)
    }
}

pub(crate) fn attention_context_from_probs<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    value_heads: GgmlCpuTensor<'a>,
    attention_probs: GgmlCpuTensor<'a>,
    layout: AttentionHeadLayout,
    steps: AttentionValueMergeSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let value_t = graph
        .permute(value_heads, 1, 0, 2, 3)
        .map_err(|source| map_err(steps.value_permute, source))?;
    let value_t = graph
        .cont(value_t)
        .map_err(|source| map_err(steps.value_cont, source))?;
    let context = graph
        .mul_mat(value_t, attention_probs)
        .map_err(|source| map_err(steps.context_mul, source))?;
    merge_attention_heads_to_hidden(graph, context, layout, steps, map_err)
}

fn merge_attention_heads_to_hidden<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    context: GgmlCpuTensor<'a>,
    layout: AttentionHeadLayout,
    steps: AttentionValueMergeSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(|source| map_err(steps.context_merge_permute, source))?;
    let merged = graph
        .cont(merged)
        .map_err(|source| map_err(steps.context_merge_cont, source))?;
    graph
        .reshape_2d(
            merged,
            layout.head_dim * layout.attention_heads,
            layout.sequence_len,
        )
        .map_err(|source| map_err(steps.context_merge_reshape, source))
}
