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

/// Shared labels for the Transformer-XL AC+BD relative-position score path so
/// family graphs (Cohere / FastConformer / FireRed-AED) do not fork step names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelativePositionAttentionSteps {
    pub cont_k: &'static str,
    pub cont_r: &'static str,
    pub cont_q_u: &'static str,
    pub cont_q_v: &'static str,
    pub cont_v: &'static str,
    pub mul_ac: &'static str,
    pub mul_bd_raw: &'static str,
    pub relative_shift: &'static str,
    pub add_scores: &'static str,
    pub scale_scores: &'static str,
    pub add_mask: &'static str,
    pub soft_max: &'static str,
    pub fused: &'static str,
    pub fused_merge: &'static str,
}

impl RelativePositionAttentionSteps {
    pub(crate) const DEFAULT: Self = Self {
        cont_k: "ggml_cont(attn_k)",
        cont_r: "ggml_cont(attn_r)",
        cont_q_u: "ggml_cont(attn_q_u)",
        cont_q_v: "ggml_cont(attn_q_v)",
        cont_v: "ggml_cont(attn_v)",
        mul_ac: "ggml_mul_mat(attn_ac)",
        mul_bd_raw: "ggml_mul_mat(attn_bd_raw)",
        relative_shift: "ggml_view_3d(relative_shift)",
        add_scores: "ggml_add(attn_scores)",
        scale_scores: "ggml_scale(attn_scores)",
        add_mask: "ggml_add(attn_key_mask)",
        soft_max: "ggml_soft_max(attn_scores)",
        fused: "ggml_flash_attn_rel_pos(attn)",
        fused_merge: "ggml_reshape_2d(attn_rel_pos_merge)",
    };
}

/// Inputs to Transformer-XL relative-position self-attention after Q/K/V/R have
/// already been projected and reshaped into attention-head layout
/// (`[head_dim, seq, heads]` after the standard permute).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RelativePositionAttentionInputs<'a> {
    pub q_u: GgmlCpuTensor<'a>,
    pub q_v: GgmlCpuTensor<'a>,
    pub k: GgmlCpuTensor<'a>,
    pub r: GgmlCpuTensor<'a>,
    pub v: GgmlCpuTensor<'a>,
    /// Optional additive F32 mask broadcast over scores (FireRed pad mask).
    pub mask: Option<GgmlCpuTensor<'a>>,
    pub layout: AttentionHeadLayout,
    pub scale: f32,
    /// Byte strides for the naive `rel_shift` view of `bd_raw`. Only used on
    /// the non-CPU fallback path.
    pub rel_shift_nb1: usize,
    pub rel_shift_nb2: usize,
    pub rel_shift_offset: usize,
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

/// Shared Transformer-XL relative-position self-attention.
///
/// On the plain CPU backend this emits the fused
/// `ggml_flash_attn_rel_pos` op (tile-local content + relative scores with
/// online softmax; never materializes T x T). Non-CPU backends keep the
/// existing mul_mat + rel_shift + soft_max fallback and never claim the fused
/// op is supported.
pub(crate) fn relative_position_attention_context<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    inputs: RelativePositionAttentionInputs<'a>,
    rel_steps: RelativePositionAttentionSteps,
    merge_steps: AttentionValueMergeSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    if graph.supports_flash_attn_rel_pos() {
        relative_position_attention_fused(graph, inputs, rel_steps, map_err)
    } else {
        relative_position_attention_naive(graph, inputs, rel_steps, merge_steps, map_err)
    }
}

fn relative_position_attention_fused<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    inputs: RelativePositionAttentionInputs<'a>,
    rel_steps: RelativePositionAttentionSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let q_u = graph
        .cont(inputs.q_u)
        .map_err(|source| map_err(rel_steps.cont_q_u, source))?;
    let q_v = graph
        .cont(inputs.q_v)
        .map_err(|source| map_err(rel_steps.cont_q_v, source))?;
    let k = graph
        .cont(inputs.k)
        .map_err(|source| map_err(rel_steps.cont_k, source))?;
    let r = graph
        .cont(inputs.r)
        .map_err(|source| map_err(rel_steps.cont_r, source))?;
    let v = graph
        .cont(inputs.v)
        .map_err(|source| map_err(rel_steps.cont_v, source))?;
    let flash = graph
        .flash_attn_rel_pos(q_u, q_v, k, r, v, inputs.mask, inputs.scale)
        .map_err(|source| map_err(rel_steps.fused, source))?;
    graph
        .reshape_2d(
            flash,
            inputs.layout.head_dim * inputs.layout.attention_heads,
            inputs.layout.sequence_len,
        )
        .map_err(|source| map_err(rel_steps.fused_merge, source))
}

fn relative_position_attention_naive<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    inputs: RelativePositionAttentionInputs<'a>,
    rel_steps: RelativePositionAttentionSteps,
    merge_steps: AttentionValueMergeSteps,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let frame_count = inputs.layout.sequence_len;
    let heads = inputs.layout.attention_heads;
    let ac = graph
        .mul_mat(
            graph
                .cont(inputs.k)
                .map_err(|source| map_err(rel_steps.cont_k, source))?,
            inputs.q_u,
        )
        .map_err(|source| map_err(rel_steps.mul_ac, source))?;
    let bd_raw = graph
        .mul_mat(
            graph
                .cont(inputs.r)
                .map_err(|source| map_err(rel_steps.cont_r, source))?,
            inputs.q_v,
        )
        .map_err(|source| map_err(rel_steps.mul_bd_raw, source))?;
    let bd = graph
        .view_3d(
            bd_raw,
            frame_count,
            frame_count,
            heads,
            inputs.rel_shift_nb1,
            inputs.rel_shift_nb2,
            inputs.rel_shift_offset,
        )
        .map_err(|source| map_err(rel_steps.relative_shift, source))?;
    let mut scores = graph
        .add(ac, bd)
        .map_err(|source| map_err(rel_steps.add_scores, source))?;
    scores = graph
        .scale(scores, inputs.scale)
        .map_err(|source| map_err(rel_steps.scale_scores, source))?;
    if let Some(mask) = inputs.mask {
        scores = graph
            .add(scores, mask)
            .map_err(|source| map_err(rel_steps.add_mask, source))?;
    }
    let scores = graph
        .soft_max(scores)
        .map_err(|source| map_err(rel_steps.soft_max, source))?;
    // Match historical FireRed/cohere layout: value heads may already be
    // contiguous or strided; `attention_context_from_probs` re-permutes them.
    attention_context_from_probs(graph, inputs.v, scores, inputs.layout, merge_steps, map_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    fn identity_map_err(step: &'static str, source: GgmlCpuGraphError) -> GgmlCpuGraphError {
        let _ = step;
        source
    }

    fn run_rel_pos_once(
        force_naive: bool,
        head_dim: usize,
        heads: usize,
        frames: usize,
        seed: u32,
    ) -> Vec<f32> {
        let d_model = head_dim * heads;
        let nr = 2 * frames - 1;
        let config = GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Cpu,
            ..GgmlCpuGraphConfig::conservative_default()
        };
        let mut runner = GgmlCpuGraphRunner::new(config).expect("runner");
        let mut graph = runner.start_graph();

        let q_u = graph
            .new_tensor_3d_f32(head_dim, frames, heads, "q_u")
            .expect("q_u");
        let q_v = graph
            .new_tensor_3d_f32(head_dim, frames, heads, "q_v")
            .expect("q_v");
        let k = graph
            .new_tensor_3d_f32(head_dim, frames, heads, "k")
            .expect("k");
        let r = graph
            .new_tensor_3d_f32(head_dim, nr, heads, "r")
            .expect("r");
        let v = graph
            .new_tensor_3d_f32(head_dim, frames, heads, "v")
            .expect("v");
        for tensor in [q_u, q_v, k, r, v] {
            graph.set_input(tensor).expect("set_input");
        }

        let element = std::mem::size_of::<f32>();
        let inputs = RelativePositionAttentionInputs {
            q_u,
            q_v,
            k,
            r,
            v,
            mask: None,
            layout: AttentionHeadLayout {
                head_dim,
                attention_heads: heads,
                sequence_len: frames,
            },
            scale: 1.0 / (head_dim as f32).sqrt(),
            rel_shift_nb1: (2 * frames - 2) * element,
            rel_shift_nb2: (2 * frames - 1) * frames * element,
            rel_shift_offset: (frames - 1) * element,
        };
        let merge = AttentionValueMergeSteps {
            value_permute: "value_permute",
            value_cont: "value_cont",
            context_mul: "context_mul",
            context_merge_permute: "merge_permute",
            context_merge_cont: "merge_cont",
            context_merge_reshape: "merge_reshape",
        };
        let out = if force_naive {
            relative_position_attention_naive(
                &graph,
                inputs,
                RelativePositionAttentionSteps::DEFAULT,
                merge,
                identity_map_err,
            )
            .expect("naive")
        } else {
            relative_position_attention_context(
                &graph,
                inputs,
                RelativePositionAttentionSteps::DEFAULT,
                merge,
                identity_map_err,
            )
            .expect("fused_or_naive")
        };
        graph.set_output(out).expect("output");

        let fill = |base: f32| -> Vec<f32> {
            let mut values = vec![0.0f32; head_dim * frames * heads];
            for (idx, slot) in values.iter_mut().enumerate() {
                let t = (seed as f32 + base + idx as f32) * 0.017;
                *slot = t.sin();
            }
            values
        };
        let r_fill = {
            let mut values = vec![0.0f32; head_dim * nr * heads];
            for (idx, slot) in values.iter_mut().enumerate() {
                let t = (seed as f32 + 9.0 + idx as f32) * 0.013;
                *slot = t.cos();
            }
            values
        };

        graph.set_f32_slice(q_u, &fill(0.0), "q_u").expect("q_u");
        graph.set_f32_slice(q_v, &fill(1.0), "q_v").expect("q_v");
        graph.set_f32_slice(k, &fill(2.0), "k").expect("k");
        graph.set_f32_slice(r, &r_fill, "r").expect("r");
        graph.set_f32_slice(v, &fill(4.0), "v").expect("v");
        graph
            .compute_output_f32(out, d_model * frames)
            .expect("compute")
    }

    #[test]
    fn fused_relative_position_attention_matches_naive_on_cpu() {
        assert!(
            GgmlCpuGraphRunner::new(GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Cpu,
                ..GgmlCpuGraphConfig::conservative_default()
            })
            .expect("runner")
            .start_graph()
            .supports_flash_attn_rel_pos(),
            "CPU backend must advertise fused relative-position attention"
        );
        let fused = run_rel_pos_once(false, 8, 2, 5, 3);
        let naive = run_rel_pos_once(true, 8, 2, 5, 3);
        assert_eq!(fused.len(), naive.len());
        let mut max_abs = 0.0f32;
        for (a, b) in fused.iter().zip(naive.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 2e-5,
            "fused vs naive max_abs_diff={max_abs} exceeds tolerance"
        );
    }
}
