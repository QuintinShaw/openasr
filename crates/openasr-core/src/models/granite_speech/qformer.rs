//! Granite Speech 4.1 `GraniteSpeechEncoderProjector`: a BLIP-2 Q-Former window
//! projector. New component (not shared with any other family yet): the
//! encoder output is windowed into non-overlapping `window_size=15`-frame
//! blocks, a fixed `num_queries = window_size / downsample_rate = 3` learned
//! query set attends (self-attention, then cross-attention into the window)
//! through 2 Q-Former layers, and the resulting 3 tokens per window are
//! linearly projected to the LLM's hidden size -- a 5x temporal downsample on
//! top of the front-end's 2x frame-stacking, matching the model card's "10x
//! total downsampling" claim.
//!
//! Faithful port of HF `transformers.models.blip_2.modeling_blip_2` (the
//! `Blip2QFormerModel`/`Blip2QFormerLayer` the projector instantiates via
//! `AutoModel.from_config(projector_config)`) plus
//! `GraniteSpeechEncoderProjector.forward` itself, cross-checked against
//! upstream llama.cpp's `granite-speech.cpp` QFormer graph section (reference
//! only, not an OpenASR upstream). `use_qformer_text_input=false` in this
//! checkpoint, so only the query-token path (`intermediate_query`/
//! `output_query`, never `intermediate`/`output`) is implemented -- the BERT
//! text-embedding path some Blip2Qformer configs carry is unreachable here and
//! intentionally not built.
//!
//! Placement note (infra-vs-family split, see `AGENTS.md`): this Q-Former
//! variant is granite-speech-only today, so it lives under this family
//! directory rather than `nn/`. If a second family adopts the same BLIP-2
//! Q-Former shape, the block-level attention/FFN plumbing here is the
//! candidate to hoist into a shared `nn::qformer` builder -- do not
//! pre-emptively generalize it before that second consumer exists.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechProjectorError {
    #[error("granite-speech projector shape error: {reason}")]
    Shape { reason: String },
    #[error("granite-speech projector missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("granite-speech projector weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("granite-speech projector weight '{name}' could not be read: {reason}")]
    WeightRead { name: String, reason: String },
    #[error("granite-speech projector GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(
    stage: &'static str,
) -> impl Fn(GgmlCpuGraphError) -> GraniteSpeechProjectorError + Copy {
    move |source| GraniteSpeechProjectorError::Ggml { stage, source }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraniteSpeechProjectorConfig {
    pub encoder_hidden_size: usize,
    pub llm_hidden_size: usize,
    pub window_size: usize,
    pub downsample_rate: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
}

impl GraniteSpeechProjectorConfig {
    pub(crate) fn granite_speech_4_1_2b() -> Self {
        Self {
            encoder_hidden_size: 1024,
            llm_hidden_size: 2048,
            window_size: 15,
            downsample_rate: 5,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            intermediate_size: 4096,
            layer_norm_eps: 1.0e-12,
        }
    }

    fn num_queries(&self) -> usize {
        self.window_size / self.downsample_rate
    }

    fn head_dim(&self) -> usize {
        self.encoder_hidden_size / self.num_attention_heads
    }
}

fn projector_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    GgmlCpuGraphConfig {
        context_bytes: 128 * 1024 * 1024,
        graph_size: 8192,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: true,
    }
}

/// Request-invariant Q-Former state. Native matrix weights remain mmap-bound;
/// the checkpoint's tiny f16 learned query is converted to resident f32 once
/// because the affine LayerNorm graph consumes f32 operands.
pub(crate) struct GraniteSpeechProjectorRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: GgmlLoadedWeightContext,
    query_arena: GgmlStaticTensorArena,
    query: GgmlStaticTensor,
}

impl GraniteSpeechProjectorRuntime {
    pub(crate) fn quoted_system_memory_bytes(
        config: &GraniteSpeechProjectorConfig,
    ) -> Result<(u64, u64), String> {
        let transient = config
            .num_queries()
            .checked_mul(config.encoder_hidden_size)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "granite projector query quote overflowed".to_string())?;
        Ok((
            u64::try_from(transient)
                .map_err(|_| "granite projector query quote exceeds u64".to_string())?,
            0,
        ))
    }

    pub(crate) const fn retained_system_memory_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        config: &GraniteSpeechProjectorConfig,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechProjectorError> {
        let graph_config = projector_graph_config(backend);
        let runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(ggml_err("load_gguf_weight_context"))?;
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| GraniteSpeechProjectorError::WeightRead {
                    name: "projector.query".to_string(),
                    reason: error.to_string(),
                })?;
        let num_queries = config.num_queries();
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(
                "projector.query",
                &[1, num_queries as u64, config.encoder_hidden_size as u64],
            )
            .map_err(|error| GraniteSpeechProjectorError::WeightRead {
                name: "projector.query".to_string(),
                reason: error.to_string(),
            })?;
        let mut query_arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(8))
            .map_err(ggml_err("query_static_tensor_arena"))?;
        let query = query_arena
            .new_tensor_2d_f32(
                config.encoder_hidden_size,
                num_queries,
                "granite_speech_projector_query",
            )
            .map_err(ggml_err("query_alloc"))?;
        query_arena
            .set_f32_slice(query, &values, "granite_speech_projector_query")
            .map_err(ggml_err("query_upload"))?;
        Ok(Self {
            runner,
            loaded,
            query_arena,
            query,
        })
    }

    pub(crate) fn project(
        &mut self,
        config: &GraniteSpeechProjectorConfig,
        encoder_out: &[f32],
        frames: usize,
    ) -> Result<GraniteSpeechProjectorOutput, GraniteSpeechProjectorError> {
        let weights =
            build_loaded_projector_weights(&self.loaded, &self.query_arena, self.query, config)?;
        run_projector_graph(&mut self.runner, config, &weights, encoder_out, frames)
    }

    pub(crate) fn release_transient_compute_memory(
        &mut self,
    ) -> Result<(), GraniteSpeechProjectorError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(ggml_err("release_transient_scheduler_working_set"))
    }
}

pub(crate) trait GraniteSpeechProjectorWeightProvider {
    fn tensor(&self, name: &str) -> Option<&[f32]>;
}

impl GraniteSpeechProjectorWeightProvider for HashMap<String, Vec<f32>> {
    fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.get(name).map(Vec::as_slice)
    }
}

struct QformerLayerWeights<'a> {
    self_q_w: GgmlCpuTensor<'a>,
    self_q_b: GgmlCpuTensor<'a>,
    self_k_w: GgmlCpuTensor<'a>,
    self_k_b: GgmlCpuTensor<'a>,
    self_v_w: GgmlCpuTensor<'a>,
    self_v_b: GgmlCpuTensor<'a>,
    self_out_w: GgmlCpuTensor<'a>,
    self_out_b: GgmlCpuTensor<'a>,
    self_out_norm_w: GgmlCpuTensor<'a>,
    self_out_norm_b: GgmlCpuTensor<'a>,
    cross_q_w: GgmlCpuTensor<'a>,
    cross_q_b: GgmlCpuTensor<'a>,
    cross_k_w: GgmlCpuTensor<'a>,
    cross_k_b: GgmlCpuTensor<'a>,
    cross_v_w: GgmlCpuTensor<'a>,
    cross_v_b: GgmlCpuTensor<'a>,
    cross_out_w: GgmlCpuTensor<'a>,
    cross_out_b: GgmlCpuTensor<'a>,
    cross_out_norm_w: GgmlCpuTensor<'a>,
    cross_out_norm_b: GgmlCpuTensor<'a>,
    ffn_up_w: GgmlCpuTensor<'a>,
    ffn_up_b: GgmlCpuTensor<'a>,
    ffn_down_w: GgmlCpuTensor<'a>,
    ffn_down_b: GgmlCpuTensor<'a>,
    ffn_out_norm_w: GgmlCpuTensor<'a>,
    ffn_out_norm_b: GgmlCpuTensor<'a>,
}

struct ProjectorWeights<'a> {
    query: GgmlCpuTensor<'a>,
    qformer_layernorm_w: GgmlCpuTensor<'a>,
    qformer_layernorm_b: GgmlCpuTensor<'a>,
    layers: Vec<QformerLayerWeights<'a>>,
    linear_w: GgmlCpuTensor<'a>,
    linear_b: GgmlCpuTensor<'a>,
}

struct WeightBuilder<'p> {
    provider: &'p dyn GraniteSpeechProjectorWeightProvider,
    uploads: Vec<(GgmlStaticTensor, &'p [f32], &'static str)>,
}

impl<'p> WeightBuilder<'p> {
    fn new(provider: &'p dyn GraniteSpeechProjectorWeightProvider) -> Self {
        Self {
            provider,
            uploads: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expected: usize) -> Result<&'p [f32], GraniteSpeechProjectorError> {
        let data = self.provider.tensor(name).ok_or_else(|| {
            GraniteSpeechProjectorError::MissingWeight {
                name: name.to_string(),
            }
        })?;
        if data.len() != expected {
            return Err(GraniteSpeechProjectorError::WeightLen {
                name: name.to_string(),
                expected,
                actual: data.len(),
            });
        }
        Ok(data)
    }

    fn w1<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
        let data = self.fetch(name, len)?;
        let handle = arena
            .new_tensor_1d_f32(len, "granite_speech_proj_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads
            .push((handle, data, "granite_speech_proj_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn w2<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
        let data = self.fetch(name, ne0 * ne1)?;
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, "granite_speech_proj_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads
            .push((handle, data, "granite_speech_proj_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), GraniteSpeechProjectorError> {
        for (handle, data, name) in &self.uploads {
            arena
                .set_f32_slice(*handle, data, name)
                .map_err(ggml_err("upload_weight"))?;
        }
        Ok(())
    }
}

fn build_layer_weights<'a, 'p>(
    arena: &GgmlStaticTensorArena,
    builder: &mut WeightBuilder<'p>,
    config: &GraniteSpeechProjectorConfig,
    index: usize,
) -> Result<QformerLayerWeights<'a>, GraniteSpeechProjectorError> {
    let d = config.encoder_hidden_size;
    let inter = config.intermediate_size;
    let p = |suffix: &str| format!("projector.qformer.encoder.layer.{index}.{suffix}");
    Ok(QformerLayerWeights {
        self_q_w: builder.w2(arena, &p("attention.attention.query.weight"), d, d)?,
        self_q_b: builder.w1(arena, &p("attention.attention.query.bias"), d)?,
        self_k_w: builder.w2(arena, &p("attention.attention.key.weight"), d, d)?,
        self_k_b: builder.w1(arena, &p("attention.attention.key.bias"), d)?,
        self_v_w: builder.w2(arena, &p("attention.attention.value.weight"), d, d)?,
        self_v_b: builder.w1(arena, &p("attention.attention.value.bias"), d)?,
        self_out_w: builder.w2(arena, &p("attention.output.dense.weight"), d, d)?,
        self_out_b: builder.w1(arena, &p("attention.output.dense.bias"), d)?,
        self_out_norm_w: builder.w1(arena, &p("attention.output.LayerNorm.weight"), d)?,
        self_out_norm_b: builder.w1(arena, &p("attention.output.LayerNorm.bias"), d)?,
        cross_q_w: builder.w2(arena, &p("crossattention.attention.query.weight"), d, d)?,
        cross_q_b: builder.w1(arena, &p("crossattention.attention.query.bias"), d)?,
        cross_k_w: builder.w2(arena, &p("crossattention.attention.key.weight"), d, d)?,
        cross_k_b: builder.w1(arena, &p("crossattention.attention.key.bias"), d)?,
        cross_v_w: builder.w2(arena, &p("crossattention.attention.value.weight"), d, d)?,
        cross_v_b: builder.w1(arena, &p("crossattention.attention.value.bias"), d)?,
        cross_out_w: builder.w2(arena, &p("crossattention.output.dense.weight"), d, d)?,
        cross_out_b: builder.w1(arena, &p("crossattention.output.dense.bias"), d)?,
        cross_out_norm_w: builder.w1(arena, &p("crossattention.output.LayerNorm.weight"), d)?,
        cross_out_norm_b: builder.w1(arena, &p("crossattention.output.LayerNorm.bias"), d)?,
        ffn_up_w: builder.w2(arena, &p("intermediate_query.dense.weight"), d, inter)?,
        ffn_up_b: builder.w1(arena, &p("intermediate_query.dense.bias"), inter)?,
        ffn_down_w: builder.w2(arena, &p("output_query.dense.weight"), inter, d)?,
        ffn_down_b: builder.w1(arena, &p("output_query.dense.bias"), d)?,
        ffn_out_norm_w: builder.w1(arena, &p("output_query.LayerNorm.weight"), d)?,
        ffn_out_norm_b: builder.w1(arena, &p("output_query.LayerNorm.bias"), d)?,
    })
}

fn packed_projector_tensor_name(name: &str) -> String {
    const LONG_PREFIX: &str = "projector.qformer.encoder.layer.";
    match name.strip_prefix(LONG_PREFIX) {
        Some(rest) => format!("projector.qf.{rest}"),
        None => name.to_string(),
    }
}

fn loaded_tensor<'a>(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
    let packed_name = packed_projector_tensor_name(name);
    loaded
        .tensor(&packed_name)
        .map(GgmlLoadedTensor::as_graph_tensor)
        .ok_or(GraniteSpeechProjectorError::MissingWeight { name: packed_name })
}

fn build_loaded_layer_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
    index: usize,
) -> Result<QformerLayerWeights<'a>, GraniteSpeechProjectorError> {
    let p = |suffix: &str| format!("projector.qformer.encoder.layer.{index}.{suffix}");
    Ok(QformerLayerWeights {
        self_q_w: loaded_tensor(loaded, &p("attention.attention.query.weight"))?,
        self_q_b: loaded_tensor(loaded, &p("attention.attention.query.bias"))?,
        self_k_w: loaded_tensor(loaded, &p("attention.attention.key.weight"))?,
        self_k_b: loaded_tensor(loaded, &p("attention.attention.key.bias"))?,
        self_v_w: loaded_tensor(loaded, &p("attention.attention.value.weight"))?,
        self_v_b: loaded_tensor(loaded, &p("attention.attention.value.bias"))?,
        self_out_w: loaded_tensor(loaded, &p("attention.output.dense.weight"))?,
        self_out_b: loaded_tensor(loaded, &p("attention.output.dense.bias"))?,
        self_out_norm_w: loaded_tensor(loaded, &p("attention.output.LayerNorm.weight"))?,
        self_out_norm_b: loaded_tensor(loaded, &p("attention.output.LayerNorm.bias"))?,
        cross_q_w: loaded_tensor(loaded, &p("crossattention.attention.query.weight"))?,
        cross_q_b: loaded_tensor(loaded, &p("crossattention.attention.query.bias"))?,
        cross_k_w: loaded_tensor(loaded, &p("crossattention.attention.key.weight"))?,
        cross_k_b: loaded_tensor(loaded, &p("crossattention.attention.key.bias"))?,
        cross_v_w: loaded_tensor(loaded, &p("crossattention.attention.value.weight"))?,
        cross_v_b: loaded_tensor(loaded, &p("crossattention.attention.value.bias"))?,
        cross_out_w: loaded_tensor(loaded, &p("crossattention.output.dense.weight"))?,
        cross_out_b: loaded_tensor(loaded, &p("crossattention.output.dense.bias"))?,
        cross_out_norm_w: loaded_tensor(loaded, &p("crossattention.output.LayerNorm.weight"))?,
        cross_out_norm_b: loaded_tensor(loaded, &p("crossattention.output.LayerNorm.bias"))?,
        ffn_up_w: loaded_tensor(loaded, &p("intermediate_query.dense.weight"))?,
        ffn_up_b: loaded_tensor(loaded, &p("intermediate_query.dense.bias"))?,
        ffn_down_w: loaded_tensor(loaded, &p("output_query.dense.weight"))?,
        ffn_down_b: loaded_tensor(loaded, &p("output_query.dense.bias"))?,
        ffn_out_norm_w: loaded_tensor(loaded, &p("output_query.LayerNorm.weight"))?,
        ffn_out_norm_b: loaded_tensor(loaded, &p("output_query.LayerNorm.bias"))?,
    })
}

fn build_loaded_projector_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
    query_arena: &GgmlStaticTensorArena,
    query: GgmlStaticTensor,
    config: &GraniteSpeechProjectorConfig,
) -> Result<ProjectorWeights<'a>, GraniteSpeechProjectorError> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        layers.push(build_loaded_layer_weights(loaded, index)?);
    }
    Ok(ProjectorWeights {
        query: query_arena.graph_tensor(query),
        qformer_layernorm_w: loaded_tensor(loaded, "projector.qformer.layernorm.weight")?,
        qformer_layernorm_b: loaded_tensor(loaded, "projector.qformer.layernorm.bias")?,
        layers,
        linear_w: loaded_tensor(loaded, "projector.linear.weight")?,
        linear_b: loaded_tensor(loaded, "projector.linear.bias")?,
    })
}

fn affine_ln<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
    apply_affine_layer_norm(
        graph,
        input,
        eps,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: stage,
            scale: stage,
            bias: stage,
        },
        |s, source| GraniteSpeechProjectorError::Ggml { stage: s, source },
    )
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
    let projected = graph.mul_mat(weight, input).map_err(ggml_err(stage))?;
    graph.add(projected, bias).map_err(ggml_err(stage))
}

/// Multi-head attention shared by the Q-Former's self- and cross-attention
/// sublayers (`Blip2QFormerMultiHeadAttention`): `q_input`/`kv_input` are
/// `[d_model, q_len/kv_len, nblocks]`; no mask (the projector never passes an
/// `encoder_attention_mask`, so even the zero-padded tail of the last window
/// is attended over unmasked -- matches the HF reference exactly, not a bug).
#[allow(clippy::too_many_arguments)]
fn qformer_mha<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    q_input: GgmlCpuTensor<'a>,
    kv_input: GgmlCpuTensor<'a>,
    q_len: usize,
    kv_len: usize,
    nblocks: usize,
    config: &GraniteSpeechProjectorConfig,
    q_w: GgmlCpuTensor<'a>,
    q_b: GgmlCpuTensor<'a>,
    k_w: GgmlCpuTensor<'a>,
    k_b: GgmlCpuTensor<'a>,
    v_w: GgmlCpuTensor<'a>,
    v_b: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
    let map = ggml_err(stage);
    let d_model = config.encoder_hidden_size;
    let n_head = config.num_attention_heads;
    let d_head = config.head_dim();

    let q = linear(graph, q_w, q_input, q_b, stage)?;
    let k = linear(graph, k_w, kv_input, k_b, stage)?;
    let v = linear(graph, v_w, kv_input, v_b, stage)?;

    let q4 = graph
        .reshape_4d(q, d_head, n_head, q_len, nblocks)
        .map_err(map)?;
    let k4 = graph
        .reshape_4d(k, d_head, n_head, kv_len, nblocks)
        .map_err(map)?;
    let v4 = graph
        .reshape_4d(v, d_head, n_head, kv_len, nblocks)
        .map_err(map)?;

    let q_perm = graph
        .cont(graph.permute(q4, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let k_perm = graph
        .cont(graph.permute(k4, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    let v_perm = graph
        .cont(graph.permute(v4, 1, 2, 0, 3).map_err(map)?)
        .map_err(map)?;

    let kq = graph.mul_mat(k_perm, q_perm).map_err(map)?;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let probs = graph.soft_max_ext(kq, None, scale, 0.0).map_err(map)?;

    let attn_out = graph.mul_mat(v_perm, probs).map_err(map)?;
    let attn_out = graph
        .cont(graph.permute(attn_out, 0, 2, 1, 3).map_err(map)?)
        .map_err(map)?;
    graph
        .reshape_3d(attn_out, d_model, q_len, nblocks)
        .map_err(map)
}

fn qformer_layer<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    queries: GgmlCpuTensor<'a>,
    encoder_windows: GgmlCpuTensor<'a>,
    num_queries: usize,
    window_size: usize,
    nblocks: usize,
    config: &GraniteSpeechProjectorConfig,
    weights: &QformerLayerWeights<'a>,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechProjectorError> {
    let eps = config.layer_norm_eps;

    let self_attn = qformer_mha(
        graph,
        queries,
        queries,
        num_queries,
        num_queries,
        nblocks,
        config,
        weights.self_q_w,
        weights.self_q_b,
        weights.self_k_w,
        weights.self_k_b,
        weights.self_v_w,
        weights.self_v_b,
        "qformer_self_attn",
    )?;
    let self_attn = linear(
        graph,
        weights.self_out_w,
        self_attn,
        weights.self_out_b,
        "qformer_self_out",
    )?;
    let self_attn = graph
        .add(self_attn, queries)
        .map_err(ggml_err("qformer_self_residual"))?;
    let queries = affine_ln(
        graph,
        self_attn,
        eps,
        weights.self_out_norm_w,
        weights.self_out_norm_b,
        "qformer_self_ln",
    )?;

    let cross_attn = qformer_mha(
        graph,
        queries,
        encoder_windows,
        num_queries,
        window_size,
        nblocks,
        config,
        weights.cross_q_w,
        weights.cross_q_b,
        weights.cross_k_w,
        weights.cross_k_b,
        weights.cross_v_w,
        weights.cross_v_b,
        "qformer_cross_attn",
    )?;
    let cross_attn = linear(
        graph,
        weights.cross_out_w,
        cross_attn,
        weights.cross_out_b,
        "qformer_cross_out",
    )?;
    let cross_attn = graph
        .add(cross_attn, queries)
        .map_err(ggml_err("qformer_cross_residual"))?;
    let queries = affine_ln(
        graph,
        cross_attn,
        eps,
        weights.cross_out_norm_w,
        weights.cross_out_norm_b,
        "qformer_cross_ln",
    )?;

    let ffn = linear(
        graph,
        weights.ffn_up_w,
        queries,
        weights.ffn_up_b,
        "qformer_ffn_up",
    )?;
    let ffn = graph.gelu(ffn).map_err(ggml_err("qformer_ffn_gelu"))?;
    let ffn = linear(
        graph,
        weights.ffn_down_w,
        ffn,
        weights.ffn_down_b,
        "qformer_ffn_down",
    )?;
    let ffn = graph
        .add(ffn, queries)
        .map_err(ggml_err("qformer_ffn_residual"))?;
    affine_ln(
        graph,
        ffn,
        eps,
        weights.ffn_out_norm_w,
        weights.ffn_out_norm_b,
        "qformer_ffn_ln",
    )
}

pub(crate) struct GraniteSpeechProjectorOutput {
    pub tokens: usize,
    pub dim: usize,
    pub projected: Vec<f32>,
}

/// Build and run the full Q-Former projector graph on the CPU backend.
/// `encoder_out` is the Conformer encoder's `[frames, encoder_hidden_size]`
/// row-major output (see `encoder_graph::encode`'s `encoder_out`).
pub(crate) fn project(
    config: &GraniteSpeechProjectorConfig,
    provider: &dyn GraniteSpeechProjectorWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<GraniteSpeechProjectorOutput, GraniteSpeechProjectorError> {
    let d_model = config.encoder_hidden_size;
    let num_queries = config.num_queries();
    let graph_config = projector_graph_config(backend);
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let tensor_count = 32 + 48 * config.num_hidden_layers;
    let arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(tensor_count);
    let arena = runner
        .start_static_tensor_arena(arena_bytes)
        .map_err(ggml_err("static_tensor_arena"))?;
    let mut builder = WeightBuilder::new(provider);
    let query = builder.w2(&arena, "projector.query", d_model, num_queries)?;
    let qformer_layernorm_w = builder.w1(&arena, "projector.qformer.layernorm.weight", d_model)?;
    let qformer_layernorm_b = builder.w1(&arena, "projector.qformer.layernorm.bias", d_model)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        layers.push(build_layer_weights(&arena, &mut builder, config, index)?);
    }
    let linear_w = builder.w2(
        &arena,
        "projector.linear.weight",
        d_model,
        config.llm_hidden_size,
    )?;
    let linear_b = builder.w1(&arena, "projector.linear.bias", config.llm_hidden_size)?;
    let mut arena = arena;
    builder.upload(&mut arena)?;
    let weights = ProjectorWeights {
        query,
        qformer_layernorm_w,
        qformer_layernorm_b,
        layers,
        linear_w,
        linear_b,
    };
    run_projector_graph(&mut runner, config, &weights, encoder_out, frames)
}

fn run_projector_graph<'a>(
    runner: &'a mut GgmlCpuGraphRunner,
    config: &GraniteSpeechProjectorConfig,
    weights: &ProjectorWeights<'a>,
    encoder_out: &[f32],
    frames: usize,
) -> Result<GraniteSpeechProjectorOutput, GraniteSpeechProjectorError> {
    let d_model = config.encoder_hidden_size;
    if encoder_out.len() != frames * d_model {
        return Err(GraniteSpeechProjectorError::Shape {
            reason: format!(
                "encoder_out has {} values, expected {frames}x{d_model}",
                encoder_out.len()
            ),
        });
    }
    let window_size = config.window_size;
    let num_queries = config.num_queries();
    let nblocks = frames.div_ceil(window_size);
    let padded_len = nblocks * window_size;
    let pad_amount = padded_len - frames;

    let dynamic_arena = runner
        .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(8))
        .map_err(ggml_err("dynamic_static_tensor_arena"))?;
    let zero_pad_handle = if pad_amount > 0 {
        Some(
            dynamic_arena
                .new_tensor_2d_f32(d_model, pad_amount, "granite_speech_proj_zero_pad")
                .map_err(ggml_err("weight_alloc_zero_pad"))?,
        )
    } else {
        None
    };
    let mut dynamic_arena = dynamic_arena;
    if let Some(handle) = zero_pad_handle {
        let zeros = vec![0.0f32; d_model * pad_amount];
        dynamic_arena
            .set_f32_slice(handle, &zeros, "granite_speech_proj_zero_pad")
            .map_err(ggml_err("upload_zero_pad"))?;
    }

    let mut graph = runner.start_graph();
    let input = graph
        .new_tensor_2d_f32(d_model, frames, "granite_speech_proj_encoder_out")
        .map_err(ggml_err("input_alloc"))?;
    let zero_pad = zero_pad_handle.map(|h| dynamic_arena.graph_tensor(h));

    let padded = match zero_pad {
        Some(pad) => graph
            .concat(input, pad, 1)
            .map_err(ggml_err("pad_concat"))?,
        None => input,
    };
    let encoder_windows = graph
        .reshape_3d(padded, d_model, window_size, nblocks)
        .map_err(ggml_err("window_reshape"))?;

    let normed_query = affine_ln(
        &graph,
        weights.query,
        config.layer_norm_eps,
        weights.qformer_layernorm_w,
        weights.qformer_layernorm_b,
        "qformer_query_ln",
    )?;
    let normed_query_3d = graph
        .reshape_3d(normed_query, d_model, num_queries, 1)
        .map_err(ggml_err("query_reshape"))?;
    let target = graph
        .new_tensor_3d_f32(
            d_model,
            num_queries,
            nblocks,
            "granite_speech_proj_query_bcast",
        )
        .map_err(ggml_err("query_bcast_alloc"))?;
    let mut queries = graph
        .repeat(normed_query_3d, target)
        .map_err(ggml_err("query_repeat"))?;

    for layer in &weights.layers {
        queries = qformer_layer(
            &graph,
            queries,
            encoder_windows,
            num_queries,
            window_size,
            nblocks,
            config,
            layer,
        )?;
    }

    let flattened = graph
        .reshape_2d(queries, d_model, num_queries * nblocks)
        .map_err(ggml_err("flatten"))?;
    let projected = linear(
        &graph,
        weights.linear_w,
        flattened,
        weights.linear_b,
        "proj_linear",
    )?;

    graph
        .set_output(projected)
        .map_err(ggml_err("set_output"))?;
    graph.set_input(input).map_err(ggml_err("mark_input"))?;
    graph
        .prepare_outputs_for_upload(&[projected])
        .map_err(ggml_err("prepare_outputs"))?;
    graph
        .set_f32_slice(input, encoder_out, "granite_speech_proj_encoder_out")
        .map_err(ggml_err("upload_encoder_out"))?;

    let expected = num_queries * nblocks * config.llm_hidden_size;
    let mut outputs = graph
        .compute_outputs_f32(&[(projected, expected)])
        .map_err(ggml_err("compute"))?;
    let out = outputs.pop().expect("projected tap");

    Ok(GraniteSpeechProjectorOutput {
        tokens: num_queries * nblocks,
        dim: config.llm_hidden_size,
        projected: out,
    })
}
