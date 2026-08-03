//! Granite Speech 4.1 Conformer CTC encoder ggml graph.
//!
//! Faithful port of HF `transformers.models.granite_speech.modeling_granite_speech`
//! (`GraniteSpeechCTCEncoder` / `GraniteSpeechConformerBlock` / `*Attention` /
//! `*ConvModule`), cross-checked against upstream llama.cpp's
//! `tools/mtmd/models/granite-speech.cpp` ggml graph builder for the op-level
//! translation (llama.cpp is a reference implementation only, not an OpenASR
//! upstream). Differences from the shared `nn::encoder::conformer_block`
//! (used by `fastconformer`/parakeet) are real, not incidental: Granite uses
//! Shaw's relative-position *embedding* (a learned `[2*max_pos_emb+1, dim_head]`
//! table indexed by clamped position distance, added directly to the raw
//! attention scores) rather than the Transformer-XL content/position bias
//! decomposition, and attention is local to non-overlapping `context_size`-frame
//! blocks (the paper's "4-second" block-attention: `context_size=200` frames at
//! the encoder's ~20ms effective frame rate after the front-end's `x2` stacking).
//! FF1/FF2 (macaron halves) and LayerNorm do reuse the shared `nn::ffn`/`nn::norm`
//! helpers, since that part of the math is identical.
//!
//! Numeric parity against an HF `transformers` fp32 reference lives in the
//! `parity` dev harness (`cargo test -p openasr-core granite_speech_encoder_parity
//! -- --ignored --nocapture`).
//!
//! Long-audio note (context-window bound, tracked for the decoder pass): this
//! encoder's block-local attention scales to arbitrary audio length on its own
//! (each `context_size`-frame block only attends within itself), but the
//! downstream Granite LLM decoder's native training context is 4096 tokens and
//! the encoder emits roughly one projector token per 100ms of audio, so a long
//! recording must be chunked before it reaches the decoder. OpenASR already has
//! a VAD-driven long-form chunker (`longform/`) for exactly this shape of
//! problem; the decoder pass should route through it instead of growing
//! `n_ctx` past the trained window (upstream llama.cpp's own experiments show
//! forcing a larger context on this architecture produces incomplete/reordered
//! transcripts rather than a clean truncation -- see the granite-speech
//! integration research notes). Acceptance for long audio is therefore judged
//! post-chunking (each VAD segment stays well under the trained context), not
//! by feeding hours of raw audio through a single decode call.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechEncoderError {
    #[error("granite-speech encoder shape error: {reason}")]
    Shape { reason: String },
    #[error("granite-speech encoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("granite-speech encoder weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("granite-speech encoder weight '{name}' could not be read: {reason}")]
    WeightRead { name: String, reason: String },
    #[error("granite-speech encoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GraniteSpeechEncoderError + Copy {
    move |source| GraniteSpeechEncoderError::Ggml { stage, source }
}

/// Scalar/shape knobs for the granite-speech-4.1-2b Conformer CTC encoder
/// (`encoder_config` in the HF `config.json`). Values below are the shipped
/// checkpoint's; a future variant with a different encoder size would carry
/// its own config value, not a code change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraniteSpeechEncoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dim_head: usize,
    pub feedforward_mult: usize,
    pub conv_kernel_size: usize,
    pub conv_expansion_factor: usize,
    pub context_size: usize,
    pub max_pos_emb: usize,
    /// Self-conditioned CTC tap output width (dual char+BPE vocab used only for
    /// the mid-layer conditioning signal, not exposed as a final encoder output).
    pub output_dim: usize,
    pub layer_norm_eps: f32,
    pub batch_norm_eps: f32,
}

impl GraniteSpeechEncoderConfig {
    /// `ibm-granite/granite-speech-4.1-2b` (identical `encoder_config` to its
    /// `granite-4.0-1b-speech` predecessor).
    pub(crate) fn granite_speech_4_1_2b() -> Self {
        Self {
            input_dim: 160,
            hidden_dim: 1024,
            num_layers: 16,
            num_heads: 8,
            dim_head: 128,
            feedforward_mult: 4,
            conv_kernel_size: 15,
            conv_expansion_factor: 2,
            context_size: 200,
            max_pos_emb: 512,
            output_dim: 348,
            layer_norm_eps: 1.0e-5,
            batch_norm_eps: 1.0e-5,
        }
    }

    fn inner_attn_dim(&self) -> usize {
        self.num_heads * self.dim_head
    }

    fn conv_inner_dim(&self) -> usize {
        self.hidden_dim * self.conv_expansion_factor
    }
}

fn encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    GgmlCpuGraphConfig {
        context_bytes: 256 * 1024 * 1024,
        graph_size: 16384,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: true,
    }
}

/// Request-invariant Granite encoder state. Large matrix weights stay in their
/// native GGUF type and are bound from the pack mapping once; only the small
/// derived BatchNorm affine tensors are materialized as resident f32 values.
/// Per-audio graphs remain shape-specific and are rebuilt on this runner.
pub(crate) struct GraniteSpeechEncoderRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: GgmlLoadedWeightContext,
    bn_arena: GgmlStaticTensorArena,
    bn_affines: Vec<(GgmlStaticTensor, GgmlStaticTensor)>,
}

impl GraniteSpeechEncoderRuntime {
    pub(crate) fn quoted_system_memory_bytes(
        config: &GraniteSpeechEncoderConfig,
    ) -> Result<(u64, u64), String> {
        let retained = config
            .num_layers
            .checked_mul(std::mem::size_of::<(GgmlStaticTensor, GgmlStaticTensor)>())
            .ok_or_else(|| "granite encoder handle quote overflowed".to_string())?;
        let upload_entries = config
            .num_layers
            .checked_mul(2)
            .and_then(|count| {
                count.checked_mul(std::mem::size_of::<(
                    GgmlStaticTensor,
                    Vec<f32>,
                    &'static str,
                )>())
            })
            .ok_or_else(|| "granite encoder upload descriptor quote overflowed".to_string())?;
        // At the final layer, all prior folded scale/bias vectors remain in
        // `uploads` while gamma/beta/mean/variance and the new scale/bias are
        // simultaneously live: 2*N + 4 vectors in total.
        let value_vectors = config
            .num_layers
            .checked_mul(2)
            .and_then(|count| count.checked_add(4))
            .and_then(|count| count.checked_mul(config.conv_inner_dim()))
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "granite encoder BN construction quote overflowed".to_string())?;
        let peak = retained
            .checked_add(upload_entries)
            .and_then(|bytes| bytes.checked_add(value_vectors))
            .ok_or_else(|| "granite encoder construction peak quote overflowed".to_string())?;
        Ok((
            u64::try_from(peak)
                .map_err(|_| "granite encoder peak quote exceeds u64".to_string())?,
            u64::try_from(retained)
                .map_err(|_| "granite encoder retained quote exceeds u64".to_string())?,
        ))
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.bn_affines, "granite encoder BN handle pairs")?;
        Ok(bytes.finish())
    }

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        config: &GraniteSpeechEncoderConfig,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechEncoderError> {
        let graph_config = encoder_graph_config(backend);
        let runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(ggml_err("load_gguf_weight_context"))?;
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| GraniteSpeechEncoderError::WeightRead {
                    name: "encoder.*".to_string(),
                    reason: error.to_string(),
                })?;
        let arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(2 * config.num_layers + 16);
        let mut bn_arena = runner
            .start_static_tensor_arena(arena_bytes)
            .map_err(ggml_err("batch_norm_static_tensor_arena"))?;
        let conv_inner = config.conv_inner_dim();
        let expected_shape = [conv_inner as u64];
        let mut bn_affines = Vec::with_capacity(config.num_layers);
        let mut uploads = Vec::with_capacity(2 * config.num_layers);
        for index in 0..config.num_layers {
            let prefix = format!("encoder.layers.{index}.conv.batch_norm");
            let read = |suffix: &str| -> Result<Vec<f32>, GraniteSpeechEncoderError> {
                let name = format!("{prefix}.{suffix}");
                reader
                    .host_tensor_f32_copy_dequantized_by_name(&name, &expected_shape)
                    .map_err(|error| GraniteSpeechEncoderError::WeightRead {
                        name,
                        reason: error.to_string(),
                    })
            };
            let gamma = read("weight")?;
            let beta = read("bias")?;
            let running_mean = read("running_mean")?;
            let running_var = read("running_var")?;
            let (scale, bias) = fold_batch_norm(
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                config.batch_norm_eps,
            );
            let scale_handle = bn_arena
                .new_tensor_1d_f32(conv_inner, "granite_speech_bn_scale")
                .map_err(ggml_err("batch_norm_scale_alloc"))?;
            let bias_handle = bn_arena
                .new_tensor_1d_f32(conv_inner, "granite_speech_bn_bias")
                .map_err(ggml_err("batch_norm_bias_alloc"))?;
            bn_affines.push((scale_handle, bias_handle));
            uploads.push((scale_handle, scale, "granite_speech_bn_scale"));
            uploads.push((bias_handle, bias, "granite_speech_bn_bias"));
        }
        for (handle, values, name) in uploads {
            bn_arena
                .set_f32_slice(handle, &values, name)
                .map_err(ggml_err("batch_norm_upload"))?;
        }
        Ok(Self {
            runner,
            loaded,
            bn_arena,
            bn_affines,
        })
    }

    pub(crate) fn encode(
        &mut self,
        config: &GraniteSpeechEncoderConfig,
        features: &[f32],
        frames_in: usize,
        capture_mid_tap: bool,
    ) -> Result<GraniteSpeechEncoderOutput, GraniteSpeechEncoderError> {
        let weights =
            build_loaded_encoder_weights(&self.loaded, &self.bn_arena, &self.bn_affines, config)?;
        run_encoder_graph(
            &mut self.runner,
            config,
            &weights,
            features,
            frames_in,
            capture_mid_tap,
        )
    }
}

/// Weight source for the encoder graph: dequantized f32 tensors keyed by their
/// HF safetensors name (`encoder.layers.{i}.attn.to_q.weight`, ...). The dev
/// parity harness implements this directly over a `HashMap` loaded from the
/// original `.safetensors`; the `.oasr` runtime path (a future pass) will
/// implement it over the mmap'd GGUF tensor index instead, same as every other
/// family's `*WeightProvider`.
pub(crate) trait GraniteSpeechEncoderWeightProvider {
    fn tensor(&self, name: &str) -> Option<&[f32]>;
}

impl GraniteSpeechEncoderWeightProvider for HashMap<String, Vec<f32>> {
    fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.get(name).map(Vec::as_slice)
    }
}

struct ConformerLayerWeights<'a> {
    ff1_norm_w: GgmlCpuTensor<'a>,
    ff1_norm_b: GgmlCpuTensor<'a>,
    ff1_up_w: GgmlCpuTensor<'a>,
    ff1_up_b: GgmlCpuTensor<'a>,
    ff1_down_w: GgmlCpuTensor<'a>,
    ff1_down_b: GgmlCpuTensor<'a>,
    attn_norm_w: GgmlCpuTensor<'a>,
    attn_norm_b: GgmlCpuTensor<'a>,
    attn_to_q_w: GgmlCpuTensor<'a>,
    attn_to_kv_w: GgmlCpuTensor<'a>,
    attn_to_out_w: GgmlCpuTensor<'a>,
    attn_to_out_b: GgmlCpuTensor<'a>,
    attn_rel_pos_emb_w: GgmlCpuTensor<'a>,
    conv_norm_w: GgmlCpuTensor<'a>,
    conv_norm_b: GgmlCpuTensor<'a>,
    conv_up_w: GgmlCpuTensor<'a>,
    conv_up_b: GgmlCpuTensor<'a>,
    conv_dw_w: GgmlCpuTensor<'a>,
    /// Folded `BatchNorm1d` scale (`gamma / sqrt(var + eps)`), computed on the
    /// host from `{weight, bias, running_mean, running_var}` so the graph never
    /// needs a batchnorm op (mirrors every other family's BN-fold convention).
    conv_bn_scale: GgmlCpuTensor<'a>,
    /// Folded `BatchNorm1d` bias (`beta - mean * scale`).
    conv_bn_bias: GgmlCpuTensor<'a>,
    conv_down_w: GgmlCpuTensor<'a>,
    conv_down_b: GgmlCpuTensor<'a>,
    ff2_norm_w: GgmlCpuTensor<'a>,
    ff2_norm_b: GgmlCpuTensor<'a>,
    ff2_up_w: GgmlCpuTensor<'a>,
    ff2_up_b: GgmlCpuTensor<'a>,
    ff2_down_w: GgmlCpuTensor<'a>,
    ff2_down_b: GgmlCpuTensor<'a>,
    post_norm_w: GgmlCpuTensor<'a>,
    post_norm_b: GgmlCpuTensor<'a>,
}

struct EncoderWeights<'a> {
    input_linear_w: GgmlCpuTensor<'a>,
    input_linear_b: GgmlCpuTensor<'a>,
    layers: Vec<ConformerLayerWeights<'a>>,
    ctc_out_w: GgmlCpuTensor<'a>,
    ctc_out_b: GgmlCpuTensor<'a>,
    ctc_out_mid_w: GgmlCpuTensor<'a>,
    ctc_out_mid_b: GgmlCpuTensor<'a>,
}

struct WeightBuilder<'p> {
    provider: &'p dyn GraniteSpeechEncoderWeightProvider,
    uploads: Vec<(GgmlStaticTensor, &'p [f32], &'static str)>,
    /// Host-computed values with no single provider tensor backing them (the
    /// folded BN scale/bias); owned so the upload phase can borrow them.
    owned_uploads: Vec<(GgmlStaticTensor, Vec<f32>, &'static str)>,
}

impl<'p> WeightBuilder<'p> {
    fn new(provider: &'p dyn GraniteSpeechEncoderWeightProvider) -> Self {
        Self {
            provider,
            uploads: Vec::new(),
            owned_uploads: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expected: usize) -> Result<&'p [f32], GraniteSpeechEncoderError> {
        let data =
            self.provider
                .tensor(name)
                .ok_or_else(|| GraniteSpeechEncoderError::MissingWeight {
                    name: name.to_string(),
                })?;
        if data.len() != expected {
            return Err(GraniteSpeechEncoderError::WeightLen {
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
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
        let data = self.fetch(name, len)?;
        let handle = arena
            .new_tensor_1d_f32(len, "granite_speech_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads.push((handle, data, "granite_speech_weight"));
        Ok(arena.graph_tensor(handle))
    }

    /// A 2-D `.weight` matmul operand bound as ggml `[ne0=in, ne1=out]` for
    /// `mul_mat(w, x)` (same convention as every other family: PyTorch's
    /// row-major `[out, in]` flat layout is byte-identical to ggml's
    /// `ne=[in, out]`, so no transpose is needed on load).
    fn w2<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
        let data = self.fetch(name, ne0 * ne1)?;
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, "granite_speech_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads.push((handle, data, "granite_speech_weight"));
        Ok(arena.graph_tensor(handle))
    }

    /// Depthwise conv kernel: PyTorch `Conv1d(groups=channels)` weight shape
    /// `[channels, 1, kernel]` (row-major, kernel innermost) is byte-identical
    /// to ggml `ne=[kernel, 1, 1, channels]`.
    fn w_dw_kernel<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        kernel: usize,
        channels: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
        let data = self.fetch(name, kernel * channels)?;
        let handle = arena
            .new_tensor_4d_f32(kernel, 1, 1, channels, "granite_speech_weight")
            .map_err(ggml_err("weight_alloc_dw"))?;
        self.uploads.push((handle, data, "granite_speech_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn owned_1d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        len: usize,
        values: Vec<f32>,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
        debug_assert_eq!(values.len(), len);
        let handle = arena
            .new_tensor_1d_f32(len, "granite_speech_weight")
            .map_err(ggml_err("weight_alloc_1d_owned"))?;
        self.owned_uploads
            .push((handle, values, "granite_speech_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), GraniteSpeechEncoderError> {
        for (handle, data, name) in &self.uploads {
            arena
                .set_f32_slice(*handle, data, name)
                .map_err(ggml_err("upload_weight"))?;
        }
        for (handle, data, name) in &self.owned_uploads {
            arena
                .set_f32_slice(*handle, data, name)
                .map_err(ggml_err("upload_weight_owned"))?;
        }
        Ok(())
    }
}

/// Fold `BatchNorm1d(x) = (x - mean) / sqrt(var + eps) * gamma + beta` into a
/// single per-channel affine `x * scale + bias`, computed once on the host so
/// the graph applies it as a `mul` + `add` (no batchnorm op needed).
fn fold_batch_norm(
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = gamma.len();
    let mut scale = vec![0.0f32; n];
    let mut bias = vec![0.0f32; n];
    for i in 0..n {
        let s = gamma[i] / (running_var[i] + eps).sqrt();
        scale[i] = s;
        bias[i] = beta[i] - running_mean[i] * s;
    }
    (scale, bias)
}

fn build_layer_weights<'a, 'p>(
    arena: &GgmlStaticTensorArena,
    builder: &mut WeightBuilder<'p>,
    config: &GraniteSpeechEncoderConfig,
    index: usize,
) -> Result<ConformerLayerWeights<'a>, GraniteSpeechEncoderError> {
    let d = config.hidden_dim;
    let inner = config.inner_attn_dim();
    let ffn = d * config.feedforward_mult;
    let conv_inner = config.conv_inner_dim();
    let kernel = config.conv_kernel_size;
    let n_pos = 2 * config.max_pos_emb + 1;
    let p = |suffix: &str| format!("encoder.layers.{index}.{suffix}");

    let bn_gamma = builder.fetch(&p("conv.batch_norm.weight"), conv_inner)?;
    let bn_beta = builder.fetch(&p("conv.batch_norm.bias"), conv_inner)?;
    let bn_mean = builder.fetch(&p("conv.batch_norm.running_mean"), conv_inner)?;
    let bn_var = builder.fetch(&p("conv.batch_norm.running_var"), conv_inner)?;
    let (bn_scale, bn_bias) =
        fold_batch_norm(bn_gamma, bn_beta, bn_mean, bn_var, config.batch_norm_eps);

    Ok(ConformerLayerWeights {
        ff1_norm_w: builder.w1(arena, &p("ff1.pre_norm.weight"), d)?,
        ff1_norm_b: builder.w1(arena, &p("ff1.pre_norm.bias"), d)?,
        ff1_up_w: builder.w2(arena, &p("ff1.up_proj.weight"), d, ffn)?,
        ff1_up_b: builder.w1(arena, &p("ff1.up_proj.bias"), ffn)?,
        ff1_down_w: builder.w2(arena, &p("ff1.down_proj.weight"), ffn, d)?,
        ff1_down_b: builder.w1(arena, &p("ff1.down_proj.bias"), d)?,
        attn_norm_w: builder.w1(arena, &p("attn.pre_norm.weight"), d)?,
        attn_norm_b: builder.w1(arena, &p("attn.pre_norm.bias"), d)?,
        attn_to_q_w: builder.w2(arena, &p("attn.to_q.weight"), d, inner)?,
        attn_to_kv_w: builder.w2(arena, &p("attn.to_kv.weight"), d, 2 * inner)?,
        attn_to_out_w: builder.w2(arena, &p("attn.to_out.weight"), inner, d)?,
        attn_to_out_b: builder.w1(arena, &p("attn.to_out.bias"), d)?,
        attn_rel_pos_emb_w: builder.w2(
            arena,
            &p("attn.rel_pos_emb.weight"),
            config.dim_head,
            n_pos,
        )?,
        conv_norm_w: builder.w1(arena, &p("conv.norm.weight"), d)?,
        conv_norm_b: builder.w1(arena, &p("conv.norm.bias"), d)?,
        conv_up_w: builder.w2(arena, &p("conv.up_conv.weight"), d, 2 * conv_inner)?,
        conv_up_b: builder.w1(arena, &p("conv.up_conv.bias"), 2 * conv_inner)?,
        conv_dw_w: builder.w_dw_kernel(
            arena,
            &p("conv.depth_conv.conv.weight"),
            kernel,
            conv_inner,
        )?,
        conv_bn_scale: builder.owned_1d(arena, conv_inner, bn_scale)?,
        conv_bn_bias: builder.owned_1d(arena, conv_inner, bn_bias)?,
        conv_down_w: builder.w2(arena, &p("conv.down_conv.weight"), conv_inner, d)?,
        conv_down_b: builder.w1(arena, &p("conv.down_conv.bias"), d)?,
        ff2_norm_w: builder.w1(arena, &p("ff2.pre_norm.weight"), d)?,
        ff2_norm_b: builder.w1(arena, &p("ff2.pre_norm.bias"), d)?,
        ff2_up_w: builder.w2(arena, &p("ff2.up_proj.weight"), d, ffn)?,
        ff2_up_b: builder.w1(arena, &p("ff2.up_proj.bias"), ffn)?,
        ff2_down_w: builder.w2(arena, &p("ff2.down_proj.weight"), ffn, d)?,
        ff2_down_b: builder.w1(arena, &p("ff2.down_proj.bias"), d)?,
        post_norm_w: builder.w1(arena, &p("post_norm.weight"), d)?,
        post_norm_b: builder.w1(arena, &p("post_norm.bias"), d)?,
    })
}

fn loaded_tensor<'a>(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    loaded
        .tensor(name)
        .map(GgmlLoadedTensor::as_graph_tensor)
        .ok_or_else(|| GraniteSpeechEncoderError::MissingWeight {
            name: name.to_string(),
        })
}

fn build_loaded_layer_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
    bn_arena: &GgmlStaticTensorArena,
    bn_affine: (GgmlStaticTensor, GgmlStaticTensor),
    index: usize,
) -> Result<ConformerLayerWeights<'a>, GraniteSpeechEncoderError> {
    let p = |suffix: &str| format!("encoder.layers.{index}.{suffix}");
    Ok(ConformerLayerWeights {
        ff1_norm_w: loaded_tensor(loaded, &p("ff1.pre_norm.weight"))?,
        ff1_norm_b: loaded_tensor(loaded, &p("ff1.pre_norm.bias"))?,
        ff1_up_w: loaded_tensor(loaded, &p("ff1.up_proj.weight"))?,
        ff1_up_b: loaded_tensor(loaded, &p("ff1.up_proj.bias"))?,
        ff1_down_w: loaded_tensor(loaded, &p("ff1.down_proj.weight"))?,
        ff1_down_b: loaded_tensor(loaded, &p("ff1.down_proj.bias"))?,
        attn_norm_w: loaded_tensor(loaded, &p("attn.pre_norm.weight"))?,
        attn_norm_b: loaded_tensor(loaded, &p("attn.pre_norm.bias"))?,
        attn_to_q_w: loaded_tensor(loaded, &p("attn.to_q.weight"))?,
        attn_to_kv_w: loaded_tensor(loaded, &p("attn.to_kv.weight"))?,
        attn_to_out_w: loaded_tensor(loaded, &p("attn.to_out.weight"))?,
        attn_to_out_b: loaded_tensor(loaded, &p("attn.to_out.bias"))?,
        attn_rel_pos_emb_w: loaded_tensor(loaded, &p("attn.rel_pos_emb.weight"))?,
        conv_norm_w: loaded_tensor(loaded, &p("conv.norm.weight"))?,
        conv_norm_b: loaded_tensor(loaded, &p("conv.norm.bias"))?,
        conv_up_w: loaded_tensor(loaded, &p("conv.up_conv.weight"))?,
        conv_up_b: loaded_tensor(loaded, &p("conv.up_conv.bias"))?,
        conv_dw_w: loaded_tensor(loaded, &p("conv.depth_conv.conv.weight"))?,
        conv_bn_scale: bn_arena.graph_tensor(bn_affine.0),
        conv_bn_bias: bn_arena.graph_tensor(bn_affine.1),
        conv_down_w: loaded_tensor(loaded, &p("conv.down_conv.weight"))?,
        conv_down_b: loaded_tensor(loaded, &p("conv.down_conv.bias"))?,
        ff2_norm_w: loaded_tensor(loaded, &p("ff2.pre_norm.weight"))?,
        ff2_norm_b: loaded_tensor(loaded, &p("ff2.pre_norm.bias"))?,
        ff2_up_w: loaded_tensor(loaded, &p("ff2.up_proj.weight"))?,
        ff2_up_b: loaded_tensor(loaded, &p("ff2.up_proj.bias"))?,
        ff2_down_w: loaded_tensor(loaded, &p("ff2.down_proj.weight"))?,
        ff2_down_b: loaded_tensor(loaded, &p("ff2.down_proj.bias"))?,
        post_norm_w: loaded_tensor(loaded, &p("post_norm.weight"))?,
        post_norm_b: loaded_tensor(loaded, &p("post_norm.bias"))?,
    })
}

fn build_loaded_encoder_weights<'a>(
    loaded: &GgmlLoadedWeightContext,
    bn_arena: &GgmlStaticTensorArena,
    bn_affines: &[(GgmlStaticTensor, GgmlStaticTensor)],
    config: &GraniteSpeechEncoderConfig,
) -> Result<EncoderWeights<'a>, GraniteSpeechEncoderError> {
    if bn_affines.len() != config.num_layers {
        return Err(GraniteSpeechEncoderError::Shape {
            reason: format!(
                "resident BatchNorm layer count is {}, expected {}",
                bn_affines.len(),
                config.num_layers
            ),
        });
    }
    let mut layers = Vec::with_capacity(config.num_layers);
    for (index, affine) in bn_affines.iter().copied().enumerate() {
        layers.push(build_loaded_layer_weights(loaded, bn_arena, affine, index)?);
    }
    Ok(EncoderWeights {
        input_linear_w: loaded_tensor(loaded, "encoder.input_linear.weight")?,
        input_linear_b: loaded_tensor(loaded, "encoder.input_linear.bias")?,
        layers,
        ctc_out_w: loaded_tensor(loaded, "encoder.out.weight")?,
        ctc_out_b: loaded_tensor(loaded, "encoder.out.bias")?,
        ctc_out_mid_w: loaded_tensor(loaded, "encoder.out_mid.weight")?,
        ctc_out_mid_b: loaded_tensor(loaded, "encoder.out_mid.bias")?,
    })
}

fn affine_ln<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
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
        |s, source| GraniteSpeechEncoderError::Ggml { stage: s, source },
    )
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    let projected = graph.mul_mat(weight, input).map_err(ggml_err(stage))?;
    graph.add(projected, bias).map_err(ggml_err(stage))
}

/// Macaron feed-forward half: `0.5 * FF(pre_norm(x)) + x`, `SiLU` activation
/// (matches `GraniteSpeechConformerFeedForward` + the `0.5 *` in
/// `GraniteSpeechConformerBlock.forward`).
fn feed_forward_half<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    norm_w: GgmlCpuTensor<'a>,
    norm_b: GgmlCpuTensor<'a>,
    up_w: GgmlCpuTensor<'a>,
    up_b: GgmlCpuTensor<'a>,
    down_w: GgmlCpuTensor<'a>,
    down_b: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    let normed = affine_ln(graph, input, eps, norm_w, norm_b, stage)?;
    apply_feed_forward_residual(
        graph,
        normed,
        input,
        FeedForwardActivation::Silu,
        Some(0.5),
        FeedForwardResidualSteps {
            activation: stage,
            scale: Some(stage),
            residual: stage,
        },
        |graph, value| linear(graph, up_w, value, up_b, stage),
        |graph, value| linear(graph, down_w, value, down_b, stage),
        |s, source| GraniteSpeechEncoderError::Ggml { stage: s, source },
    )
}

/// Shaw relative-position self-attention, block-local to non-overlapping
/// `context_size`-frame windows. `attn_dists` is the precomputed
/// `[context_size * context_size]` clamped-distance index table (constant for
/// a given config, independent of the input); `attn_mask` is `Some` only when
/// the last block is zero-padded (`frames % context_size != 0`), a
/// `[context_size, context_size, 1, num_blocks]` additive mask that is all-zero
/// except in the last block plane, where positions outside `[0, remainder)` on
/// either axis are `f32::MIN` (mirrors `-torch.finfo(dtype).max`).
#[allow(clippy::too_many_arguments)]
fn block_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    attn_dists: GgmlCpuTensor<'a>,
    attn_mask: Option<GgmlCpuTensor<'a>>,
    zero_pad: Option<GgmlCpuTensor<'a>>,
    weights: &ConformerLayerWeights<'a>,
    config: &GraniteSpeechEncoderConfig,
    frames: usize,
    padded_len: usize,
    num_blocks: usize,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    let stage = "attn";
    let map = ggml_err(stage);
    let d_head = config.dim_head;
    let n_head = config.num_heads;
    let context_size = config.context_size;
    let n_embd = config.inner_attn_dim();

    let normed = affine_ln(
        graph,
        input,
        config.layer_norm_eps,
        weights.attn_norm_w,
        weights.attn_norm_b,
        stage,
    )?;
    let normed_padded = match zero_pad {
        Some(pad) => graph.concat(normed, pad, 1).map_err(map)?,
        None => normed,
    };

    let q = graph
        .mul_mat(weights.attn_to_q_w, normed_padded)
        .map_err(map)?;
    let kv = graph
        .mul_mat(weights.attn_to_kv_w, normed_padded)
        .map_err(map)?;
    let f32_size = std::mem::size_of::<f32>();
    let kv_nb1 = 2 * n_embd * f32_size;
    let k = graph
        .view_2d(kv, n_embd, padded_len, kv_nb1, 0)
        .map_err(map)?;
    let k = graph.cont(k).map_err(map)?;
    let v = graph
        .view_2d(kv, n_embd, padded_len, kv_nb1, n_embd * f32_size)
        .map_err(map)?;
    let v = graph.cont(v).map_err(map)?;

    let q4 = graph
        .reshape_4d(q, d_head, n_head, context_size, num_blocks)
        .map_err(map)?;
    let k4 = graph
        .reshape_4d(k, d_head, n_head, context_size, num_blocks)
        .map_err(map)?;
    let v4 = graph
        .reshape_4d(v, d_head, n_head, context_size, num_blocks)
        .map_err(map)?;

    let q_perm = graph.permute(q4, 0, 2, 1, 3).map_err(map)?;
    let q_perm = graph.cont(q_perm).map_err(map)?;
    let k_perm = graph.permute(k4, 0, 2, 1, 3).map_err(map)?;
    let k_perm = graph.cont(k_perm).map_err(map)?;

    let kq = graph.mul_mat(k_perm, q_perm).map_err(map)?;

    // Shaw's relative-position embedding: one learned `dim_head` vector per
    // (query, key) clamped-distance pair, contracted against the (unpermuted)
    // query vector for that query position. See `attn_dists` doc comment.
    // The pack preserves the source tensor's `[dim_head, n_pos]` row-major
    // bytes but GGUF records its logical extents as `[n_pos, dim_head]`.
    // Reinterpret the contiguous storage to the graph's `[dim_head, n_pos]`
    // convention before row lookup. This is a no-op for the f32 parity path,
    // whose arena tensor already has that shape.
    let rel_pos_emb = graph
        .reshape_2d(
            weights.attn_rel_pos_emb_w,
            d_head,
            2 * config.max_pos_emb + 1,
        )
        .map_err(map)?;
    let pos_emb = graph.get_rows(rel_pos_emb, attn_dists).map_err(map)?;
    let pos_emb = graph
        .reshape_3d(pos_emb, d_head, context_size, context_size)
        .map_err(map)?;
    let pos_emb = graph
        .reshape_4d(pos_emb, d_head, context_size, 1, context_size)
        .map_err(map)?;
    let q_shaw = graph.permute(q4, 0, 1, 3, 2).map_err(map)?;
    let q_shaw = graph.cont(q_shaw).map_err(map)?;
    let pos_attn = graph.mul_mat(pos_emb, q_shaw).map_err(map)?;
    let pos_attn = graph.permute(pos_attn, 0, 2, 3, 1).map_err(map)?;
    let pos_attn = graph.cont(pos_attn).map_err(map)?;

    let scores = graph.add(kq, pos_attn).map_err(map)?;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let probs = graph
        .soft_max_ext(scores, attn_mask, scale, 0.0)
        .map_err(map)?;

    let v_perm = graph.permute(v4, 1, 2, 0, 3).map_err(map)?;
    let v_perm = graph.cont(v_perm).map_err(map)?;
    let attn_out = graph.mul_mat(v_perm, probs).map_err(map)?;
    let attn_out = graph.permute(attn_out, 0, 2, 1, 3).map_err(map)?;
    let attn_out = graph.cont(attn_out).map_err(map)?;
    let attn_out = graph
        .reshape_2d(attn_out, n_embd, padded_len)
        .map_err(map)?;
    let attn_out = if frames < padded_len {
        graph
            .view_2d(attn_out, n_embd, frames, n_embd * f32_size, 0)
            .map_err(map)?
    } else {
        attn_out
    };
    let attn_out = graph.cont(attn_out).map_err(map)?;

    linear(
        graph,
        weights.attn_to_out_w,
        attn_out,
        weights.attn_to_out_b,
        stage,
    )
}

/// GLU + depthwise-conv module: `norm -> up_conv(2x, GLU) -> depthwise conv
/// (folded BatchNorm + SiLU) -> down_conv`. The two pointwise (`kernel_size=1`)
/// convs are algebraically identical to a per-frame linear layer over the
/// channel axis, so they are implemented as `mul_mat`, not `conv_1d`; only the
/// genuinely temporal depthwise conv uses `depthwise_conv_2d` (2D op with the
/// unused spatial axis fixed at 1), same idiom as the `dolphin`/`wav2vec2`
/// conv modules elsewhere in this crate.
fn conv_module<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weights: &ConformerLayerWeights<'a>,
    config: &GraniteSpeechEncoderConfig,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    let stage = "conv";
    let map = ggml_err(stage);
    let conv_inner = config.conv_inner_dim();
    let kernel = config.conv_kernel_size;
    let padding = kernel / 2;

    let normed = affine_ln(
        graph,
        input,
        config.layer_norm_eps,
        weights.conv_norm_w,
        weights.conv_norm_b,
        stage,
    )?;
    let up = linear(graph, weights.conv_up_w, normed, weights.conv_up_b, stage)?;

    // GLU(dim=channel): first half is the value branch, second half the gate.
    let f32_size = std::mem::size_of::<f32>();
    let up_nb1 = 2 * conv_inner * f32_size;
    let value = graph
        .view_2d(up, conv_inner, frames, up_nb1, 0)
        .map_err(map)?;
    let value = graph.cont(value).map_err(map)?;
    let gate = graph
        .view_2d(up, conv_inner, frames, up_nb1, conv_inner * f32_size)
        .map_err(map)?;
    let gate = graph.cont(gate).map_err(map)?;
    let gate = graph.sigmoid(gate).map_err(map)?;
    let glu_out = graph.mul(value, gate).map_err(map)?;

    // Depthwise temporal conv: transpose to [frames, channels] so the conv's
    // spatial (width) axis is time, then back.
    let transposed = graph.transpose(glu_out).map_err(map)?;
    let transposed = graph.cont(transposed).map_err(map)?;
    let as_4d = graph
        .reshape_4d(transposed, frames, 1, conv_inner, 1)
        .map_err(map)?;
    // As with the relative-position table, the pack preserves the source
    // `[channels, 1, kernel]` extent order while the direct ggml depthwise op
    // consumes `[kernel, 1, 1, channels]`; contiguous bytes are identical.
    let conv_dw_w = graph
        .reshape_4d(weights.conv_dw_w, kernel, 1, 1, conv_inner)
        .map_err(map)?;
    let conv = graph
        .depthwise_conv_2d(conv_dw_w, as_4d, 1, 1, padding, 0, 1, 1)
        .map_err(map)?;
    let conv = graph.permute(conv, 1, 2, 0, 3).map_err(map)?;
    let conv = graph.cont(conv).map_err(map)?;
    let conv = graph.reshape_2d(conv, conv_inner, frames).map_err(map)?;

    // Folded BatchNorm1d + SiLU.
    let conv = graph.mul(conv, weights.conv_bn_scale).map_err(map)?;
    let conv = graph.add(conv, weights.conv_bn_bias).map_err(map)?;
    let conv = graph.silu(conv).map_err(map)?;

    linear(graph, weights.conv_down_w, conv, weights.conv_down_b, stage)
}

#[allow(clippy::too_many_arguments)]
fn conformer_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    attn_dists: GgmlCpuTensor<'a>,
    attn_mask: Option<GgmlCpuTensor<'a>>,
    zero_pad: Option<GgmlCpuTensor<'a>>,
    weights: &ConformerLayerWeights<'a>,
    config: &GraniteSpeechEncoderConfig,
    frames: usize,
    padded_len: usize,
    num_blocks: usize,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechEncoderError> {
    let hidden = feed_forward_half(
        graph,
        input,
        config.layer_norm_eps,
        weights.ff1_norm_w,
        weights.ff1_norm_b,
        weights.ff1_up_w,
        weights.ff1_up_b,
        weights.ff1_down_w,
        weights.ff1_down_b,
        "ff1",
    )?;
    let attn_out = block_attention(
        graph, hidden, attn_dists, attn_mask, zero_pad, weights, config, frames, padded_len,
        num_blocks,
    )?;
    let hidden = graph
        .add(attn_out, hidden)
        .map_err(ggml_err("attn_residual"))?;
    let conv_out = conv_module(graph, hidden, weights, config, frames)?;
    let hidden = graph
        .add(conv_out, hidden)
        .map_err(ggml_err("conv_residual"))?;
    let hidden = feed_forward_half(
        graph,
        hidden,
        config.layer_norm_eps,
        weights.ff2_norm_w,
        weights.ff2_norm_b,
        weights.ff2_up_w,
        weights.ff2_up_b,
        weights.ff2_down_w,
        weights.ff2_down_b,
        "ff2",
    )?;
    affine_ln(
        graph,
        hidden,
        config.layer_norm_eps,
        weights.post_norm_w,
        weights.post_norm_b,
        "post_norm",
    )
}

/// Clamped relative-position distance table: `attention_dists[q, k] =
/// clamp(q - k, -context_size, context_size) + max_pos_emb`, row-major
/// `[query][key]` flattened, matching `GraniteSpeechCTCEncoder`'s
/// `attention_dists` buffer exactly.
fn attention_dists_table(context_size: usize, max_pos_emb: usize) -> Vec<i32> {
    let cs = context_size as i64;
    let mpe = max_pos_emb as i64;
    let mut table = vec![0i32; context_size * context_size];
    for q in 0..context_size {
        for k in 0..context_size {
            let dist = (q as i64) - (k as i64);
            let clamped = dist.clamp(-cs, cs) + mpe;
            table[q * context_size + k] = clamped as i32;
        }
    }
    table
}

/// The `[context_size, context_size]` additive mask for the last (possibly
/// zero-padded) block: `0.0` where both query and key fall inside
/// `[0, remainder)`, `f32::MIN` otherwise.
fn last_block_mask(context_size: usize, remainder: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; context_size * context_size];
    for q in 0..context_size {
        for k in 0..context_size {
            if q >= remainder || k >= remainder {
                mask[q * context_size + k] = f32::MIN;
            }
        }
    }
    mask
}

pub(crate) struct GraniteSpeechEncoderOutput {
    pub frames: usize,
    pub dim: usize,
    /// Hidden state immediately after layer `num_layers/2` completes, BEFORE
    /// the self-conditioning CTC tap is added back in (present only when
    /// `capture_taps` is true). Useful for parity bisection: if this matches
    /// but `encoder_out` does not, the bug is in the CTC tap, not the blocks.
    pub mid_block_out: Vec<f32>,
    pub encoder_out: Vec<f32>,
}

/// Build and run the full 16-layer Conformer CTC encoder graph on the CPU
/// backend. `features` is the front-end's `[frames, input_dim]` row-major
/// mel/frame-stacked input (`input_dim=160` = 80 log-mel bins x2 stacking);
/// this function does not itself compute mel features (see the module doc's
/// parity-harness note: numeric validation starts from precomputed features,
/// same convention as `dolphin`'s parity harness).
pub(crate) fn encode(
    config: &GraniteSpeechEncoderConfig,
    provider: &dyn GraniteSpeechEncoderWeightProvider,
    features: &[f32],
    frames_in: usize,
    backend: GgmlCpuGraphBackend,
    capture_mid_tap: bool,
) -> Result<GraniteSpeechEncoderOutput, GraniteSpeechEncoderError> {
    let input_dim = config.input_dim;
    let graph_config = encoder_graph_config(backend);
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let tensor_count = 64 + 96 * config.num_layers;
    let arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(tensor_count);
    let arena = runner
        .start_static_tensor_arena(arena_bytes)
        .map_err(ggml_err("static_tensor_arena"))?;
    let mut builder = WeightBuilder::new(provider);
    let input_linear_w = builder.w2(
        &arena,
        "encoder.input_linear.weight",
        input_dim,
        config.hidden_dim,
    )?;
    let input_linear_b = builder.w1(&arena, "encoder.input_linear.bias", config.hidden_dim)?;
    let mut layers = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        layers.push(build_layer_weights(&arena, &mut builder, config, index)?);
    }
    let ctc_out_w = builder.w2(
        &arena,
        "encoder.out.weight",
        config.hidden_dim,
        config.output_dim,
    )?;
    let ctc_out_b = builder.w1(&arena, "encoder.out.bias", config.output_dim)?;
    let ctc_out_mid_w = builder.w2(
        &arena,
        "encoder.out_mid.weight",
        config.output_dim,
        config.hidden_dim,
    )?;
    let ctc_out_mid_b = builder.w1(&arena, "encoder.out_mid.bias", config.hidden_dim)?;
    let mut arena = arena;
    builder.upload(&mut arena)?;
    let weights = EncoderWeights {
        input_linear_w,
        input_linear_b,
        layers,
        ctc_out_w,
        ctc_out_b,
        ctc_out_mid_w,
        ctc_out_mid_b,
    };
    run_encoder_graph(
        &mut runner,
        config,
        &weights,
        features,
        frames_in,
        capture_mid_tap,
    )
}

fn run_encoder_graph<'a>(
    runner: &'a mut GgmlCpuGraphRunner,
    config: &GraniteSpeechEncoderConfig,
    weights: &EncoderWeights<'a>,
    features: &[f32],
    frames_in: usize,
    capture_mid_tap: bool,
) -> Result<GraniteSpeechEncoderOutput, GraniteSpeechEncoderError> {
    let input_dim = config.input_dim;
    if features.len() != frames_in * input_dim {
        return Err(GraniteSpeechEncoderError::Shape {
            reason: format!(
                "features has {} values, expected {frames_in}x{input_dim}",
                features.len()
            ),
        });
    }
    if frames_in == 0 {
        return Err(GraniteSpeechEncoderError::Shape {
            reason: "frames_in must be > 0".to_string(),
        });
    }

    let context_size = config.context_size;
    let num_blocks = frames_in.div_ceil(context_size);
    let padded_len = num_blocks * context_size;
    let remainder = frames_in % context_size;
    let pad_amount = padded_len - frames_in;

    let dynamic_arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(16);
    let dynamic_arena = runner
        .start_static_tensor_arena(dynamic_arena_bytes)
        .map_err(ggml_err("dynamic_static_tensor_arena"))?;
    let attn_dists_table = attention_dists_table(context_size, config.max_pos_emb);
    let attn_dists_handle = dynamic_arena
        .new_tensor_1d_i32(context_size * context_size, "granite_speech_attn_dists")
        .map_err(ggml_err("weight_alloc_attn_dists"))?;
    let mask_table = (remainder > 0).then(|| last_block_mask(context_size, remainder));
    let mask_handle = if mask_table.is_some() {
        Some(
            dynamic_arena
                .new_tensor_4d_f32(
                    context_size,
                    context_size,
                    1,
                    num_blocks,
                    "granite_speech_attn_mask",
                )
                .map_err(ggml_err("weight_alloc_attn_mask"))?,
        )
    } else {
        None
    };
    let zero_pad_handle = if pad_amount > 0 {
        Some(
            dynamic_arena
                .new_tensor_2d_f32(config.hidden_dim, pad_amount, "granite_speech_zero_pad")
                .map_err(ggml_err("weight_alloc_zero_pad"))?,
        )
    } else {
        None
    };
    let mut dynamic_arena = dynamic_arena;
    dynamic_arena
        .set_i32_slice(
            attn_dists_handle,
            &attn_dists_table,
            "granite_speech_attn_dists",
        )
        .map_err(ggml_err("upload_attn_dists"))?;
    if let (Some(handle), Some(table)) = (mask_handle, &mask_table) {
        let mut full = vec![0.0f32; context_size * context_size * num_blocks];
        let last_block_start = (num_blocks - 1) * context_size * context_size;
        full[last_block_start..last_block_start + context_size * context_size]
            .copy_from_slice(table);
        dynamic_arena
            .set_f32_slice(handle, &full, "granite_speech_attn_mask")
            .map_err(ggml_err("upload_attn_mask"))?;
    }
    if let Some(handle) = zero_pad_handle {
        let zeros = vec![0.0f32; config.hidden_dim * pad_amount];
        dynamic_arena
            .set_f32_slice(handle, &zeros, "granite_speech_zero_pad")
            .map_err(ggml_err("upload_zero_pad"))?;
    }

    let mut graph = runner.start_graph();
    let input = graph
        .new_tensor_2d_f32(input_dim, frames_in, "granite_speech_features")
        .map_err(ggml_err("input_alloc"))?;
    let attn_dists = dynamic_arena.graph_tensor(attn_dists_handle);
    let attn_mask = mask_handle.map(|h| dynamic_arena.graph_tensor(h));
    let zero_pad = zero_pad_handle.map(|h| dynamic_arena.graph_tensor(h));

    let mut hidden = linear(
        &graph,
        weights.input_linear_w,
        input,
        weights.input_linear_b,
        "input_linear",
    )?;

    let mid_tap_layer = config.num_layers / 2;
    // `encoder.out_mid.weight` is stored with source extents
    // `[hidden_dim, output_dim]`; the original f32 builder intentionally
    // reinterpreted its flat bytes as ggml `[output_dim, hidden_dim]`.
    let ctc_out_mid_w = graph
        .reshape_2d(weights.ctc_out_mid_w, config.output_dim, config.hidden_dim)
        .map_err(ggml_err("ctc_out_mid_weight_reshape"))?;
    let mut mid_tap: Option<GgmlCpuTensor> = None;
    for (index, layer) in weights.layers.iter().enumerate() {
        hidden = conformer_block(
            &mut graph, hidden, attn_dists, attn_mask, zero_pad, layer, config, frames_in,
            padded_len, num_blocks,
        )?;
        if index + 1 == mid_tap_layer {
            if capture_mid_tap {
                mid_tap = Some(hidden);
            }
            let mid = linear(
                &graph,
                weights.ctc_out_w,
                hidden,
                weights.ctc_out_b,
                "ctc_out",
            )?;
            let mid = graph.soft_max(mid).map_err(ggml_err("ctc_softmax"))?;
            let mid = linear(
                &graph,
                ctc_out_mid_w,
                mid,
                weights.ctc_out_mid_b,
                "ctc_out_mid",
            )?;
            hidden = graph.add(hidden, mid).map_err(ggml_err("ctc_residual"))?;
        }
    }
    let encoder_out = hidden;

    let mut taps = Vec::with_capacity(2);
    if let Some(mid) = mid_tap {
        taps.push(mid);
    }
    taps.push(encoder_out);
    for tap in &taps {
        graph.set_output(*tap).map_err(ggml_err("set_output"))?;
    }
    graph
        .set_input(input)
        .map_err(ggml_err("mark_input(features)"))?;
    graph
        .prepare_outputs_for_upload(&taps)
        .map_err(ggml_err("prepare_outputs"))?;
    graph
        .set_f32_slice(input, features, "granite_speech_features")
        .map_err(ggml_err("upload_features"))?;

    let expected = frames_in * config.hidden_dim;
    let output_specs: Vec<(GgmlCpuTensor, usize)> =
        taps.iter().map(|tap| (*tap, expected)).collect();
    let mut outputs = graph
        .compute_outputs_f32(&output_specs)
        .map_err(ggml_err("compute"))?;

    let encoder_out = outputs.pop().expect("encoder_out tap");
    let mid_block_out = if capture_mid_tap {
        outputs.pop().expect("mid tap")
    } else {
        Vec::new()
    };

    Ok(GraniteSpeechEncoderOutput {
        frames: frames_in,
        dim: config.hidden_dim,
        mid_block_out,
        encoder_out,
    })
}
