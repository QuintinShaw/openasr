//! wav2vec2-ctc encoder graph: raw-waveform conv feature extractor → feature
//! projection → grouped positional-conv embedding (+SamePad crop) → encoder
//! layer-norm → post-norm transformer layers × N → CTC head → `[vocab, frames]`.
//!
//! Layout convention (ggml): `[ne0, ne1, ...]`. Conv stages use `[T, channels]`
//! (time fastest); the transformer + norms use `[features, T]` (features fastest).
//! Per-stage diff taps (feature-extractor out / pos-embed out / post-encoder out)
//! are exposed via `OPENASR_WAV2VEC2_TRACE` for bisecting against an HF reference.

#![allow(dead_code)]

use std::path::Path;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::models::wav2vec2_ctc::graph_config::wav2vec2_ctc_encoder_graph_config;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};
use crate::nn::wav2vec2::{
    GroupedConv1dParams, Wav2Vec2EncoderLayerConfig, Wav2Vec2EncoderLayerWeights, grouped_conv_1d,
    wav2vec2_post_norm_encoder_layer, wav2vec2_stable_layer_norm_encoder_layer,
};

use super::encoder_weights::{
    NamedTensor, Wav2Vec2EncoderLayerWeights as HostLayerWeights, Wav2Vec2EncoderWeights,
};
use super::runtime_contract::{
    FEATURE_EXTRACTOR_CONV_KERNEL, FEATURE_EXTRACTOR_CONV_STRIDE, FeatExtractNorm,
    Wav2Vec2CtcExecutionMetadata,
};

const GGML_TYPE_F16: i32 = 1;
/// Base context budget; scaled up for the larger 24-layer / 1024-hidden variants
/// (hubert/lv60/data2vec) so the arena + graph fit.
const WAV2VEC2_ENCODER_GRAPH_CONTEXT_BYTES: usize = 1536 * 1024 * 1024;
const LAYER_NORM_EPSILON: f32 = 1.0e-5;
const GROUP_NORM_EPSILON: f32 = 1.0e-5;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Wav2Vec2EncoderError {
    #[error("wav2vec2-ctc encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("wav2vec2-ctc encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("wav2vec2-ctc encoder shape error: {reason}")]
    Shape { reason: String },
}

/// Per-frame CTC logits: `logits[frame * vocab_size + v]` is the logit for token
/// `v` at output frame `frame` (ggml frame-major buffer).
#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2CtcEncoderOutput {
    pub frame_count: usize,
    pub vocab_size: usize,
    pub logits: Vec<f32>,
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> Wav2Vec2EncoderError {
    move |source| Wav2Vec2EncoderError::GraphBuildFailed { step, source }
}

fn bf2(step: &'static str, source: GgmlCpuGraphError) -> Wav2Vec2EncoderError {
    Wav2Vec2EncoderError::GraphBuildFailed { step, source }
}

/// A 2-D linear weight: arena (f32-uploaded) or zero-copy leaf bound to the
/// mmap'd pack (native q4_K/f16/f32).
#[derive(Clone, Copy)]
enum WeightSlot {
    Arena(GgmlStaticTensor),
    Loaded(GgmlLoadedTensor),
}

impl WeightSlot {
    fn graph<'a>(self, arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(handle) => arena.graph_tensor(handle),
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
        }
    }
}

fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, Wav2Vec2EncoderError> {
    match loaded.and_then(|ctx| ctx.tensor(name)) {
        Some(tensor) => Ok(WeightSlot::Loaded(tensor)),
        None => Err(Wav2Vec2EncoderError::Shape {
            reason: format!(
                "2-D linear '{name}' could not be bound zero-copy from the runtime pack \
                 (loaded weight context missing or tensor absent); host payload was dropped"
            ),
        }),
    }
}

fn alloc_static_f32(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, Wav2Vec2EncoderError> {
    let map = |source| Wav2Vec2EncoderError::GraphBuildFailed { step, source };
    match weight.dims.as_slice() {
        [] | [_] => arena
            .new_tensor_1d_f32(weight.values.len(), step)
            .map_err(map),
        [ne0, ne1] => arena.new_tensor_2d_f32(*ne0, *ne1, step).map_err(map),
        [ne0, ne1, ne2] => arena.new_tensor_3d_f32(*ne0, *ne1, *ne2, step).map_err(map),
        _ => Err(Wav2Vec2EncoderError::Shape {
            reason: format!(
                "tensor '{}' has unsupported rank {:?}",
                weight.name, weight.dims
            ),
        }),
    }
}

fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, Wav2Vec2EncoderError> {
    let map = |source| Wav2Vec2EncoderError::GraphBuildFailed { step, source };
    match weight.dims.as_slice() {
        [ne0, ne1, ne2] => arena
            .new_tensor_3d_typed(*ne0, *ne1, *ne2, GGML_TYPE_F16, step)
            .map_err(map),
        _ => Err(Wav2Vec2EncoderError::Shape {
            reason: format!("f16 conv '{}' rank {:?}", weight.name, weight.dims),
        }),
    }
}

fn upload_static(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), Wav2Vec2EncoderError> {
    arena
        .set_f32_slice(tensor, &weight.values, step)
        .map_err(|source| Wav2Vec2EncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), Wav2Vec2EncoderError> {
    let bits: Vec<u16> = weight.values.iter().copied().map(f32_to_f16_bits).collect();
    arena
        .set_f16_bits_slice(tensor, &bits, step)
        .map_err(|source| Wav2Vec2EncoderError::GraphBuildFailed { step, source })
}

/// Arena handles for one feature-extractor conv layer.
struct FeArena {
    conv_weight: GgmlStaticTensor,
    /// Output channels (ggml conv kernel dims `[K, IC, OC]` -> OC).
    out_channels: usize,
    /// Optional conv bias `[out_channels]` (hubert/lv60).
    conv_bias: Option<GgmlStaticTensor>,
    /// Channel-norm gamma/beta (group: layer 0 only; layer: every layer).
    norm_weight: Option<GgmlStaticTensor>,
    norm_bias: Option<GgmlStaticTensor>,
}

/// Arena handles + bound slots for one transformer layer.
struct LayerArena {
    q_weight: WeightSlot,
    q_bias: GgmlStaticTensor,
    k_weight: WeightSlot,
    k_bias: GgmlStaticTensor,
    v_weight: WeightSlot,
    v_bias: GgmlStaticTensor,
    out_weight: WeightSlot,
    out_bias: GgmlStaticTensor,
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    ffn_up_weight: WeightSlot,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down_weight: WeightSlot,
    ffn_down_bias: GgmlStaticTensor,
    final_norm_weight: GgmlStaticTensor,
    final_norm_bias: GgmlStaticTensor,
}

pub(crate) struct Wav2Vec2CtcEncoderGraph {
    metadata: Wav2Vec2CtcExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    feature_extractor: Vec<FeArena>,
    fp_norm_weight: GgmlStaticTensor,
    fp_norm_bias: GgmlStaticTensor,
    fp_proj_weight: WeightSlot,
    fp_proj_bias: GgmlStaticTensor,
    /// Positional-conv stack: 1 entry (wav2vec2/hubert folded weight-norm conv)
    /// or N entries (data2vec stacked plain convs). Each = (kernel, bias).
    pos_conv: Vec<(GgmlStaticTensor, GgmlStaticTensor)>,
    encoder_norm_weight: GgmlStaticTensor,
    encoder_norm_bias: GgmlStaticTensor,
    layers: Vec<LayerArena>,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
}

impl Wav2Vec2CtcEncoderGraph {
    pub(crate) fn new(
        weights: &Wav2Vec2EncoderWeights,
        metadata: Wav2Vec2CtcExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, Wav2Vec2EncoderError> {
        let mut config = wav2vec2_ctc_encoder_graph_config();
        config.context_bytes = WAV2VEC2_ENCODER_GRAPH_CONTEXT_BYTES;
        // Larger variants (e.g. hubert-xlarge, ~48 transformer layers vs large's
        // 24) build more graph nodes than the default 4096-node cap, tripping
        // `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`. Size the cgraph to the
        // data-driven layer count with generous per-layer headroom. Capacity only
        // — the built graph and op order are unchanged, so existing models stay
        // byte-identical.
        config.graph_size = config.graph_size.max(weights.layers.len() * 256 + 2048);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            Wav2Vec2EncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(WAV2VEC2_ENCODER_GRAPH_CONTEXT_BYTES)
            .map_err(|source| Wav2Vec2EncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;

        // ----- allocate all arena tensors first (first upload freezes) -----
        let mut fe_handles = Vec::with_capacity(weights.feature_extractor.len());
        for (idx, fe) in weights.feature_extractor.iter().enumerate() {
            let conv_weight = alloc_static_f16(&arena, &fe.conv_weight, "fe_conv")?;
            let conv_bias = match &fe.conv_bias {
                Some(w) => Some(alloc_static_f32(&arena, w, "fe_conv_b")?),
                None => None,
            };
            let norm_weight = match &fe.norm_weight {
                Some(w) => Some(alloc_static_f32(&arena, w, "fe_gn_w")?),
                None => None,
            };
            let norm_bias = match &fe.norm_bias {
                Some(w) => Some(alloc_static_f32(&arena, w, "fe_gn_b")?),
                None => None,
            };
            let _ = idx;
            let out_channels =
                *fe.conv_weight
                    .dims
                    .get(2)
                    .ok_or_else(|| Wav2Vec2EncoderError::Shape {
                        reason: format!(
                            "feature-extractor conv '{}' rank {:?} != 3",
                            fe.conv_weight.name, fe.conv_weight.dims
                        ),
                    })?;
            fe_handles.push(FeArena {
                conv_weight,
                out_channels,
                conv_bias,
                norm_weight,
                norm_bias,
            });
        }
        let fp_norm_weight = alloc_static_f32(&arena, &weights.fp_norm_weight, "fp_norm_w")?;
        let fp_norm_bias = alloc_static_f32(&arena, &weights.fp_norm_bias, "fp_norm_b")?;
        let fp_proj_weight = bind_loaded(loaded, &weights.fp_proj_weight.name)?;
        let fp_proj_bias = alloc_static_f32(&arena, &weights.fp_proj_bias, "fp_proj_b")?;
        let mut pos_conv = Vec::with_capacity(weights.pos_conv_layers.len());
        for (i, layer) in weights.pos_conv_layers.iter().enumerate() {
            let w = alloc_static_f16(&arena, &layer.weight, "posconv_w")?;
            let b = alloc_static_f32(&arena, &layer.bias, "posconv_b")?;
            pos_conv.push((w, b));
            let _ = i;
        }
        let encoder_norm_weight =
            alloc_static_f32(&arena, &weights.encoder_norm_weight, "enc_norm_w")?;
        let encoder_norm_bias = alloc_static_f32(&arena, &weights.encoder_norm_bias, "enc_norm_b")?;

        let mut layer_handles = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layer_handles.push(alloc_layer(&arena, loaded, layer)?);
        }
        let ctc_head_weight = bind_loaded(loaded, &weights.ctc_head_weight.name)?;
        let ctc_head_bias = alloc_static_f32(&arena, &weights.ctc_head_bias, "ctc_head_b")?;

        // ----- upload all values (arena freezes on first set) -----
        for (fe, handle) in weights.feature_extractor.iter().zip(&fe_handles) {
            upload_static_f16(&mut arena, handle.conv_weight, &fe.conv_weight, "fe_conv")?;
            if let (Some(t), Some(w)) = (handle.conv_bias, &fe.conv_bias) {
                upload_static(&mut arena, t, w, "fe_conv_b")?;
            }
            if let (Some(t), Some(w)) = (handle.norm_weight, &fe.norm_weight) {
                upload_static(&mut arena, t, w, "fe_gn_w")?;
            }
            if let (Some(t), Some(w)) = (handle.norm_bias, &fe.norm_bias) {
                upload_static(&mut arena, t, w, "fe_gn_b")?;
            }
        }
        upload_static(
            &mut arena,
            fp_norm_weight,
            &weights.fp_norm_weight,
            "fp_norm_w",
        )?;
        upload_static(&mut arena, fp_norm_bias, &weights.fp_norm_bias, "fp_norm_b")?;
        upload_static(&mut arena, fp_proj_bias, &weights.fp_proj_bias, "fp_proj_b")?;
        for ((w, b), layer) in pos_conv.iter().zip(&weights.pos_conv_layers) {
            upload_static_f16(&mut arena, *w, &layer.weight, "posconv_w")?;
            upload_static(&mut arena, *b, &layer.bias, "posconv_b")?;
        }
        upload_static(
            &mut arena,
            encoder_norm_weight,
            &weights.encoder_norm_weight,
            "enc_norm_w",
        )?;
        upload_static(
            &mut arena,
            encoder_norm_bias,
            &weights.encoder_norm_bias,
            "enc_norm_b",
        )?;
        for (layer, handle) in weights.layers.iter().zip(&layer_handles) {
            upload_layer(&mut arena, layer, handle)?;
        }
        upload_static(
            &mut arena,
            ctc_head_bias,
            &weights.ctc_head_bias,
            "ctc_head_b",
        )?;

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            feature_extractor: fe_handles,
            fp_norm_weight,
            fp_norm_bias,
            fp_proj_weight,
            fp_proj_bias,
            pos_conv,
            encoder_norm_weight,
            encoder_norm_bias,
            layers: layer_handles,
            ctc_head_weight,
            ctc_head_bias,
        })
    }

    pub(crate) fn encode(
        &mut self,
        samples: &[f32],
    ) -> Result<Wav2Vec2CtcEncoderOutput, Wav2Vec2EncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.hidden_size;
        let frames = feature_extractor_frame_count(samples.len());
        if frames == 0 {
            return Err(Wav2Vec2EncoderError::Shape {
                reason: "feature extractor produced zero frames".into(),
            });
        }
        let trace = std::env::var_os("OPENASR_WAV2VEC2_TRACE").is_some();

        let mut graph = self.runner.start_graph();
        let element = std::mem::size_of::<f32>();

        // ----- raw waveform input as [n_samples, 1] (time, in_channel) -----
        let input = graph
            .new_tensor_2d_f32(samples.len(), 1, "w2v2_pcm")
            .map_err(bf("new_pcm"))?;
        graph.set_input(input).map_err(bf("set_input_pcm"))?;

        // ----- 7-layer strided conv feature extractor -----
        let layer_norm_fe = matches!(metadata.feat_extract_norm, FeatExtractNorm::Layer);
        let mut state = input; // [T, channels]
        let mut cur_time = samples.len();
        for (idx, fe) in self.feature_extractor.iter().enumerate() {
            let stride = FEATURE_EXTRACTOR_CONV_STRIDE[idx];
            let kernel = FEATURE_EXTRACTOR_CONV_KERNEL[idx];
            let out_channels = fe.out_channels;
            let mut conv = graph
                .conv_1d(self.arena.graph_tensor(fe.conv_weight), state, stride, 0, 1)
                .map_err(bf("fe_conv"))?;
            cur_time = (cur_time - kernel) / stride + 1;
            // Optional conv bias `[out_channels]` (hubert/lv60 conv_bias=true):
            // broadcast over the time axis via a transpose dance.
            if let Some(cb) = fe.conv_bias {
                let feat_major = transpose_to_feature_major(&graph, conv)?; // [C, T]
                let biased = graph
                    .add(feat_major, self.arena.graph_tensor(cb))
                    .map_err(bf("fe_conv_bias"))?;
                let back = graph.transpose(biased).map_err(bf("fe_conv_bias_t"))?;
                conv = graph.cont(back).map_err(bf("fe_conv_bias_cont"))?;
            }
            // Channel norm: GroupNorm (per-channel) for the base "group" variant
            // (layer 0 only), LayerNorm-over-channels for the "layer" variant
            // (every layer). Both consume gamma/beta `[C]`.
            let normed = match (fe.norm_weight, fe.norm_bias) {
                (Some(gn_w), Some(gn_b)) if layer_norm_fe => apply_feature_extractor_layer_norm(
                    &graph,
                    conv,
                    self.arena.graph_tensor(gn_w),
                    self.arena.graph_tensor(gn_b),
                )?,
                (Some(gn_w), Some(gn_b)) => apply_feature_extractor_group_norm(
                    &graph,
                    conv,
                    self.arena.graph_tensor(gn_w),
                    self.arena.graph_tensor(gn_b),
                    cur_time,
                    out_channels,
                )?,
                _ => conv,
            };
            state = graph.gelu_erf(normed).map_err(bf("fe_gelu"))?;
        }
        // state is [T_frames, 512] (time, channel).
        if trace {
            trace_tensor(&mut graph, state, "feature_extractor_out");
        }

        // ----- transpose to [512, T] (channel-fastest) for projection -----
        let mut hidden = transpose_to_feature_major(&graph, state)?; // [512, T]
        // feature_projection: layer_norm over 512, then Linear 512 -> 768.
        hidden = apply_affine_layer_norm(
            &graph,
            hidden,
            LAYER_NORM_EPSILON,
            self.arena.graph_tensor(self.fp_norm_weight),
            self.arena.graph_tensor(self.fp_norm_bias),
            AffineLayerNormSteps {
                norm: "fp_norm",
                scale: "fp_norm_scale",
                bias: "fp_norm_bias",
            },
            bf2,
        )?;
        hidden = graph
            .mul_mat(self.fp_proj_weight.graph(&self.arena), hidden)
            .map_err(bf("fp_proj"))?;
        hidden = graph
            .add(hidden, self.arena.graph_tensor(self.fp_proj_bias))
            .map_err(bf("fp_proj_bias"))?;
        // hidden is [768, T].

        // ----- grouped positional conv embedding -----
        // wav2vec2/hubert: one folded weight-norm conv (even kernel + SamePad crop).
        // data2vec: a stack of plain grouped convs (odd kernel, no crop, gelu each,
        // applied sequentially) — both return position_embeddings added to hidden.
        let pos = if self.pos_conv.len() <= 1 {
            let (w, b) = self.pos_conv[0];
            positional_conv_embedding(
                &graph,
                &self.arena,
                w,
                b,
                hidden,
                frames,
                d_model,
                metadata.num_conv_pos_embedding_groups,
                metadata.num_conv_pos_embeddings,
            )?
        } else {
            data2vec_positional_conv_embedding(
                &graph,
                &self.arena,
                &self.pos_conv,
                hidden,
                frames,
                d_model,
                metadata.num_conv_pos_embedding_groups,
                metadata.num_conv_pos_embeddings,
            )?
        };
        if trace {
            trace_tensor(&mut graph, pos, "pos_conv_out");
        }
        let mut state = graph.add(hidden, pos).map_err(bf("pos_add"))?;

        let stable = metadata.do_stable_layer_norm;
        let encoder_norm_step = AffineLayerNormSteps {
            norm: "enc_norm",
            scale: "enc_norm_scale",
            bias: "enc_norm_bias",
        };

        // Post-norm encoder (base-960h): layer_norm BEFORE the stack.
        if !stable {
            state = apply_affine_layer_norm(
                &graph,
                state,
                LAYER_NORM_EPSILON,
                self.arena.graph_tensor(self.encoder_norm_weight),
                self.arena.graph_tensor(self.encoder_norm_bias),
                encoder_norm_step,
                bf2,
            )?;
        }

        // ----- transformer layers (post-norm or stable/pre-norm) -----
        let layer_config = Wav2Vec2EncoderLayerConfig {
            d_model,
            attention_heads: metadata.n_heads,
            head_dim: metadata.head_dim,
            sequence_len: frames,
            layer_norm_epsilon: LAYER_NORM_EPSILON,
        };
        for handle in &self.layers {
            let layer_weights = layer_weights(&self.arena, handle);
            state = if stable {
                wav2vec2_stable_layer_norm_encoder_layer(
                    &mut graph,
                    state,
                    layer_config,
                    &layer_weights,
                    bf2,
                )?
            } else {
                wav2vec2_post_norm_encoder_layer(
                    &mut graph,
                    state,
                    layer_config,
                    &layer_weights,
                    bf2,
                )?
            };
        }

        // Stable encoder (large variants): final layer_norm AFTER the stack.
        if stable {
            state = apply_affine_layer_norm(
                &graph,
                state,
                LAYER_NORM_EPSILON,
                self.arena.graph_tensor(self.encoder_norm_weight),
                self.arena.graph_tensor(self.encoder_norm_bias),
                encoder_norm_step,
                bf2,
            )?;
        }
        if trace {
            trace_tensor(&mut graph, state, "encoder_out");
        }

        // ----- CTC head -----
        let head = self.ctc_head_weight.graph(&self.arena);
        let mut logits = graph.mul_mat(head, state).map_err(bf("ctc_head"))?;
        logits = graph
            .add(logits, self.arena.graph_tensor(self.ctc_head_bias))
            .map_err(bf("ctc_head_bias"))?;
        graph.set_output(logits).map_err(bf("set_output"))?;

        let _ = element;
        // Peak-RSS lever: allocate the compute graph with the scheduler's gallocr
        // (liveness-based buffer REUSE) instead of the default per-tensor
        // alloc_ctx_tensors that set_f32_slice would otherwise force. Without this
        // every per-layer intermediate gets its own buffer (RSS = sum over layers,
        // ~85MB/layer); with reuse it collapses to the working-set peak. The full
        // graph (incl. the CTC-head output) is built above, so prepare here.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(input, samples, "w2v2_pcm")
            .map_err(bf("upload_pcm"))?;

        let want =
            metadata
                .vocab_size
                .checked_mul(frames)
                .ok_or_else(|| Wav2Vec2EncoderError::Shape {
                    reason: "logits overflow".into(),
                })?;
        let logits = graph.compute_output_f32(logits, want).map_err(|error| {
            Wav2Vec2EncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(Wav2Vec2CtcEncoderOutput {
            frame_count: frames,
            vocab_size: metadata.vocab_size,
            logits,
        })
    }
}

/// Grouped positional-conv embedding: hidden `[768, T]` → transpose to
/// `[T, 768]` → grouped conv (16 groups, kernel 128, pad 64) → `[T+1, 768]`
/// → transpose to `[768, T+1]` → add bias → crop last frame (SamePad) →
/// GELU. Returns `[768, T]`.
#[allow(clippy::too_many_arguments)]
fn positional_conv_embedding<'a>(
    graph: &GgmlCpuGraphBuilderRef<'a>,
    arena: &GgmlStaticTensorArena,
    pos_conv_weight: GgmlStaticTensor,
    pos_conv_bias: GgmlStaticTensor,
    hidden: GgmlCpuTensor<'a>,
    frames: usize,
    d_model: usize,
    groups: usize,
    kernel: usize,
) -> Result<GgmlCpuTensor<'a>, Wav2Vec2EncoderError> {
    let in_per_group = d_model / groups;
    let out_per_group = d_model / groups;
    let padding = kernel / 2;

    // hidden [768, T] -> [T, 768] (time fastest) for conv_1d data.
    let data = graph
        .transpose(hidden)
        .map_err(bf("posconv_transpose_in"))?;
    let data = graph.cont(data).map_err(bf("posconv_cont_in"))?;

    // Per-group kernel views: enc.posconv.weight is [K, in/g, out] f16; group
    // g uses the out-channel slice [g*out/g : (g+1)*out/g] = ne2 slice.
    let kernel_tensor = arena.graph_tensor(pos_conv_weight);
    let f16_width = std::mem::size_of::<u16>();
    let mut group_kernels = Vec::with_capacity(groups);
    for g in 0..groups {
        // [K, in/g, out/g] view of [K, in/g, out] at ne2 offset g*out/g.
        let view = graph
            .view_3d(
                kernel_tensor,
                kernel,
                in_per_group,
                out_per_group,
                kernel * f16_width,
                kernel * in_per_group * f16_width,
                g * out_per_group * kernel * in_per_group * f16_width,
            )
            .map_err(bf("posconv_kernel_view"))?;
        let view = graph.cont(view).map_err(bf("posconv_kernel_cont"))?;
        group_kernels.push(view);
    }
    let params = GroupedConv1dParams {
        groups,
        time: frames,
        in_per_group,
        out_per_group,
        stride: 1,
        padding,
        dilation: 1,
    };
    // grouped conv output [T+1, 768].
    let conv = grouped_conv_1d(graph, data, &group_kernels, &params, "posconv", bf2)?;
    // transpose to [768, T+1] (feature fastest), add bias, crop last frame.
    let conv = graph.transpose(conv).map_err(bf("posconv_transpose_out"))?;
    let conv = graph.cont(conv).map_err(bf("posconv_cont_out"))?;
    let conv = graph
        .add(conv, arena.graph_tensor(pos_conv_bias))
        .map_err(bf("posconv_bias"))?;
    // SamePad crop: drop the LAST time frame (even kernel) -> [768, T].
    let element = std::mem::size_of::<f32>();
    let cropped = graph
        .view_2d(conv, d_model, frames, d_model * element, 0)
        .map_err(bf("posconv_crop"))?;
    let cropped = graph.cont(cropped).map_err(bf("posconv_crop_cont"))?;
    graph.gelu_erf(cropped).map_err(bf("posconv_gelu"))
}

/// data2vec positional-conv embedding: a stack of plain grouped convs applied
/// SEQUENTIALLY (each = transpose → grouped conv(groups, kernel, pad=kernel/2) →
/// transpose → +bias → GELU). The odd kernel (19) with symmetric padding keeps
/// the time dimension at `frames` (output = T + 2*(K/2) - K + 1 = T for odd K), so
/// there is NO SamePad crop. Returns `[d_model, T]` position embeddings to add to
/// hidden (mirrors HF `Data2VecAudioPositionalConvEmbedding`).
#[allow(clippy::too_many_arguments)]
fn data2vec_positional_conv_embedding<'a>(
    graph: &GgmlCpuGraphBuilderRef<'a>,
    arena: &GgmlStaticTensorArena,
    pos_conv: &[(GgmlStaticTensor, GgmlStaticTensor)],
    hidden: GgmlCpuTensor<'a>,
    frames: usize,
    d_model: usize,
    groups: usize,
    kernel: usize,
) -> Result<GgmlCpuTensor<'a>, Wav2Vec2EncoderError> {
    let in_per_group = d_model / groups;
    let out_per_group = d_model / groups;
    let padding = kernel / 2;
    let f16_width = std::mem::size_of::<u16>();
    let mut current = hidden;
    for (weight, bias) in pos_conv {
        // [d_model, T] -> [T, d_model] for conv_1d data.
        let data = graph
            .transpose(current)
            .map_err(bf("d2v_posconv_transpose_in"))?;
        let data = graph.cont(data).map_err(bf("d2v_posconv_cont_in"))?;
        // Per-group kernel views of [K, in/g, out] f16.
        let kernel_tensor = arena.graph_tensor(*weight);
        let mut group_kernels = Vec::with_capacity(groups);
        for g in 0..groups {
            let view = graph
                .view_3d(
                    kernel_tensor,
                    kernel,
                    in_per_group,
                    out_per_group,
                    kernel * f16_width,
                    kernel * in_per_group * f16_width,
                    g * out_per_group * kernel * in_per_group * f16_width,
                )
                .map_err(bf("d2v_posconv_kernel_view"))?;
            let view = graph.cont(view).map_err(bf("d2v_posconv_kernel_cont"))?;
            group_kernels.push(view);
        }
        let params = GroupedConv1dParams {
            groups,
            time: frames,
            in_per_group,
            out_per_group,
            stride: 1,
            padding,
            dilation: 1,
        };
        // odd kernel + symmetric pad => output time == frames, no crop.
        let conv = grouped_conv_1d(graph, data, &group_kernels, &params, "d2v_posconv", bf2)?;
        let conv = graph
            .transpose(conv)
            .map_err(bf("d2v_posconv_transpose_out"))?;
        let conv = graph.cont(conv).map_err(bf("d2v_posconv_cont_out"))?;
        let conv = graph
            .add(conv, arena.graph_tensor(*bias))
            .map_err(bf("d2v_posconv_bias"))?;
        current = graph.gelu_erf(conv).map_err(bf("d2v_posconv_gelu"))?;
    }
    Ok(current)
}

type GgmlCpuGraphBuilderRef<'a> = crate::ggml_runtime::GgmlCpuGraphBuilder<'a>;

/// Feature-extractor group norm (n_groups == n_channels = per-channel). The conv
/// output is `[T, C]`; reshape to `[T, 1, C, 1]` so each channel is one group
/// over ne0=T, group_norm, then the affine gamma/beta `[C]` are applied across
/// the channel axis after transposing to `[C, T]`. We keep everything in `[T, C]`
/// by applying gamma/beta via a transpose dance.
fn apply_feature_extractor_group_norm<'a>(
    graph: &GgmlCpuGraphBuilderRef<'a>,
    conv: GgmlCpuTensor<'a>,  // [T, C]
    gamma: GgmlCpuTensor<'a>, // [C]
    beta: GgmlCpuTensor<'a>,  // [C]
    t: usize,
    c: usize,
) -> Result<GgmlCpuTensor<'a>, Wav2Vec2EncoderError> {
    // Reshape [T, C] -> [T, 1, C, 1] so ggml group_norm (which normalizes over
    // ne0*ne1 within each of the n_groups channel groups along ne2) treats each
    // channel as one group spanning T (ne0). n_groups == c (per-channel).
    let reshaped = graph
        .reshape_4d(conv, t, 1, c, 1)
        .map_err(bf("fe_gn_reshape"))?;
    let normed = graph
        .group_norm(reshaped, c, GROUP_NORM_EPSILON)
        .map_err(bf("fe_gn"))?;
    // back to [T, C].
    let normed = graph
        .reshape_2d(normed, t, c)
        .map_err(bf("fe_gn_reshape_back"))?;
    // affine: scale/shift per channel. Channels are ne1, so transpose to [C, T],
    // mul by gamma [C] (broadcast over ne1=T), add beta [C], transpose back.
    let feat_major = transpose_to_feature_major(graph, normed)?; // [C, T]
    let scaled = graph.mul(feat_major, gamma).map_err(bf("fe_gn_scale"))?;
    let shifted = graph.add(scaled, beta).map_err(bf("fe_gn_shift"))?;
    // back to [T, C].
    let back = graph
        .transpose(shifted)
        .map_err(bf("fe_gn_transpose_back"))?;
    graph.cont(back).map_err(bf("fe_gn_cont_back"))
}

/// Feature-extractor LayerNorm OVER CHANNELS (`feat_extract_norm=="layer"`).
/// The conv output is `[T, C]`; HF transposes to put channels last and applies
/// LayerNorm over the channel dim per time step, then an affine gamma/beta `[C]`.
/// We transpose to `[C, T]` (channels-fastest = ne0), `norm` over ne0, scale by
/// gamma `[C]` (broadcast over T=ne1), add beta `[C]`, transpose back to `[T, C]`.
fn apply_feature_extractor_layer_norm<'a>(
    graph: &GgmlCpuGraphBuilderRef<'a>,
    conv: GgmlCpuTensor<'a>,  // [T, C]
    gamma: GgmlCpuTensor<'a>, // [C]
    beta: GgmlCpuTensor<'a>,  // [C]
) -> Result<GgmlCpuTensor<'a>, Wav2Vec2EncoderError> {
    let feat_major = transpose_to_feature_major(graph, conv)?; // [C, T]
    let normed = graph
        .norm(feat_major, LAYER_NORM_EPSILON)
        .map_err(bf("fe_ln"))?;
    let scaled = graph.mul(normed, gamma).map_err(bf("fe_ln_scale"))?;
    let shifted = graph.add(scaled, beta).map_err(bf("fe_ln_shift"))?;
    let back = graph
        .transpose(shifted)
        .map_err(bf("fe_ln_transpose_back"))?;
    graph.cont(back).map_err(bf("fe_ln_cont_back"))
}

/// Transpose a `[ne0, ne1]` tensor to `[ne1, ne0]` (contiguous).
fn transpose_to_feature_major<'a>(
    graph: &GgmlCpuGraphBuilderRef<'a>,
    t: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, Wav2Vec2EncoderError> {
    let transposed = graph.transpose(t).map_err(bf("transpose"))?;
    graph.cont(transposed).map_err(bf("transpose_cont"))
}

fn trace_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilderRef<'a>,
    _tensor: GgmlCpuTensor<'a>,
    _label: &str,
) {
    let _ = graph;
}

/// Feature-extractor output frame count for `n_samples` raw samples.
pub(crate) fn feature_extractor_frame_count(n_samples: usize) -> usize {
    let mut t = n_samples;
    for (k, s) in FEATURE_EXTRACTOR_CONV_KERNEL
        .iter()
        .zip(FEATURE_EXTRACTOR_CONV_STRIDE.iter())
    {
        if t < *k {
            return 0;
        }
        t = (t - k) / s + 1;
    }
    t
}

fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &HostLayerWeights,
) -> Result<LayerArena, Wav2Vec2EncoderError> {
    Ok(LayerArena {
        q_weight: bind_loaded(loaded, &layer.attn_q_weight.name)?,
        q_bias: alloc_static_f32(arena, &layer.attn_q_bias, "q_b")?,
        k_weight: bind_loaded(loaded, &layer.attn_k_weight.name)?,
        k_bias: alloc_static_f32(arena, &layer.attn_k_bias, "k_b")?,
        v_weight: bind_loaded(loaded, &layer.attn_v_weight.name)?,
        v_bias: alloc_static_f32(arena, &layer.attn_v_bias, "v_b")?,
        out_weight: bind_loaded(loaded, &layer.attn_out_weight.name)?,
        out_bias: alloc_static_f32(arena, &layer.attn_out_bias, "out_b")?,
        attn_norm_weight: alloc_static_f32(arena, &layer.attn_norm_weight, "attn_norm_w")?,
        attn_norm_bias: alloc_static_f32(arena, &layer.attn_norm_bias, "attn_norm_b")?,
        ffn_up_weight: bind_loaded(loaded, &layer.ffn_up_weight.name)?,
        ffn_up_bias: alloc_static_f32(arena, &layer.ffn_up_bias, "ffn_up_b")?,
        ffn_down_weight: bind_loaded(loaded, &layer.ffn_down_weight.name)?,
        ffn_down_bias: alloc_static_f32(arena, &layer.ffn_down_bias, "ffn_down_b")?,
        final_norm_weight: alloc_static_f32(arena, &layer.final_norm_weight, "final_norm_w")?,
        final_norm_bias: alloc_static_f32(arena, &layer.final_norm_bias, "final_norm_b")?,
    })
}

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    layer: &HostLayerWeights,
    h: &LayerArena,
) -> Result<(), Wav2Vec2EncoderError> {
    let pairs: [(GgmlStaticTensor, &NamedTensor); 10] = [
        (h.q_bias, &layer.attn_q_bias),
        (h.k_bias, &layer.attn_k_bias),
        (h.v_bias, &layer.attn_v_bias),
        (h.out_bias, &layer.attn_out_bias),
        (h.attn_norm_weight, &layer.attn_norm_weight),
        (h.attn_norm_bias, &layer.attn_norm_bias),
        (h.ffn_up_bias, &layer.ffn_up_bias),
        (h.ffn_down_bias, &layer.ffn_down_bias),
        (h.final_norm_weight, &layer.final_norm_weight),
        (h.final_norm_bias, &layer.final_norm_bias),
    ];
    for (tensor, weight) in pairs {
        upload_static(arena, tensor, weight, "layer_weight")?;
    }
    Ok(())
}

fn layer_weights<'a>(
    arena: &'a GgmlStaticTensorArena,
    h: &LayerArena,
) -> Wav2Vec2EncoderLayerWeights<'a> {
    let g = |t: GgmlStaticTensor| arena.graph_tensor(t);
    let b = |slot: WeightSlot| slot.graph(arena);
    Wav2Vec2EncoderLayerWeights {
        q_weight: b(h.q_weight),
        q_bias: g(h.q_bias),
        k_weight: b(h.k_weight),
        k_bias: g(h.k_bias),
        v_weight: b(h.v_weight),
        v_bias: g(h.v_bias),
        out_weight: b(h.out_weight),
        out_bias: g(h.out_bias),
        layer_norm_weight: g(h.attn_norm_weight),
        layer_norm_bias: g(h.attn_norm_bias),
        ff_intermediate_weight: b(h.ffn_up_weight),
        ff_intermediate_bias: g(h.ffn_up_bias),
        ff_output_weight: b(h.ffn_down_weight),
        ff_output_bias: g(h.ffn_down_bias),
        final_layer_norm_weight: g(h.final_norm_weight),
        final_layer_norm_bias: g(h.final_norm_bias),
    }
}

/// Round-to-nearest f32 -> f16 bit pattern (mirrors the cohere importer).
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let half_mantissa = (mantissa_with_hidden >> shift) as u16;
        return sign | half_mantissa;
    }
    let half_mantissa = (mantissa >> 13) as u16;
    sign | ((half_exponent as u16) << 10) | half_mantissa
}
