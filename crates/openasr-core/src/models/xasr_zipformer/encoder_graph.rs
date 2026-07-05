//! X-ASR Zipformer2 encoder execution boundary.
//!
//! Hands-off: single-responsibility ggml graph transcription, guarded by
//! golden/parity tests. Do not split this module for "tidiness" -- the tensor
//! wiring is validated as a whole and refactoring here risks silent numeric
//! drift.
//!
//! This module is intentionally shaped like the other model `encoder_graph`
//! modules, but starts with a correctness-preserving reference backend. The
//! Zipformer-specific ops (Swoosh, BiasNorm, NonlinAttention, and chunkwise conv
//! scale) must not be approximated in GGML; this boundary lets us replace stages
//! with exact GGML kernels/op compositions one at a time while keeping the
//! executor-facing API stable.

use super::encoder_ops::{
    SWOOSH_L_OFFSET, SWOOSH_L_SHIFT, SWOOSH_LINEAR_SCALE, SWOOSH_R_OFFSET, SWOOSH_R_SHIFT,
};
use super::encoder_reference::{
    XasrZipformerLayerReferenceCaches, bypass_reference, downsample_streaming_reference,
    encode_embed_reference, streaming_key_padding_mask, upsample_streaming_reference,
    zipformer_layer_streaming_reference,
};
use super::encoder_weights::{
    XasrConv2dWeights, XasrConvolutionModuleWeights, XasrEncoderEmbedWeights,
    XasrEncoderLayerWeights, XasrEncoderStackWeights, XasrEncoderWeights, XasrLinearPairWeights,
    XasrLinearWithBias,
};
use super::runtime_contract::XasrZipformerExecutionMetadata;
use super::weights::StoredLinear;
use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlPersistentGraphSession,
};
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

const XASR_PROFILE_ENV: &str = "OPENASR_XASR_PROFILE";

#[derive(Debug, thiserror::Error)]
pub(crate) enum XasrEncoderGraphError {
    #[error("xasr encoder graph shape error: {reason}")]
    Shape { reason: String },
    #[error("xasr encoder reference backend failed at {stage}: {reason}")]
    Reference { stage: &'static str, reason: String },
    #[error("xasr encoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum XasrEncoderGraphBackend {
    /// Exact Rust reference backend. Used until each Zipformer2 custom op has an
    /// exact GGML implementation and parity gate.
    Reference,
    /// Exact GGML graph for stack0 only. Multiscale downsample/upsample and the
    /// remaining stacks stay outside this facade until their parity gates land.
    GgmlCpuStack0,
    /// Exact GGML graph from encoder-embed rows through the full pre-joiner
    /// encoder output. Encoder embed itself can still be provided by the Rust
    /// frontend path until the production executor wires feature-to-embed cache.
    GgmlCpuFullEncoder,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderFeatureInput {
    /// Number of fbank frames.
    pub frames: usize,
    /// Feature dimension; X-ASR expects 80.
    pub feature_dim: usize,
    /// Frame-major `[frames, feature_dim]` fbank rows.
    pub rows: Vec<f32>,
}

impl XasrEncoderFeatureInput {
    pub(crate) fn new(
        frames: usize,
        feature_dim: usize,
        rows: Vec<f32>,
    ) -> Result<Self, XasrEncoderGraphError> {
        let expected =
            frames
                .checked_mul(feature_dim)
                .ok_or_else(|| XasrEncoderGraphError::Shape {
                    reason: "feature input shape overflows".to_string(),
                })?;
        if rows.len() != expected {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "feature input has {} values, expected {frames}x{feature_dim}={expected}",
                    rows.len()
                ),
            });
        }
        Ok(Self {
            frames,
            feature_dim,
            rows,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderGraphOutput {
    pub frames: usize,
    pub dim: usize,
    /// Frame-major `[frames, dim]` rows.
    pub rows: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct XasrEncoderLayerCache {
    pub cached_key: Vec<f32>,
    pub cached_nonlin_attention: Vec<f32>,
    pub cached_val1: Vec<f32>,
    pub cached_val2: Vec<f32>,
    pub cached_conv1: Vec<f32>,
    pub cached_conv2: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct XasrEncoderChunkState {
    pub embed_states: Vec<f32>,
    pub layer_caches: Vec<XasrEncoderLayerCache>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderChunkOutput {
    pub output: XasrEncoderGraphOutput,
    pub state: XasrEncoderChunkState,
}

#[derive(Debug, Clone, PartialEq)]
struct XasrEncoderEmbedOwnedOutput {
    frames: usize,
    dim: usize,
    rows: Vec<f32>,
    new_embed_states: Vec<f32>,
}

pub(crate) struct XasrZipformerEncoderGraph {
    metadata: XasrZipformerExecutionMetadata,
    weights: XasrEncoderWeights,
    backend: XasrEncoderGraphBackend,
    ggml_config: Option<GgmlCpuGraphConfig>,
    // Drop order is load-bearing: reusable sessions hold raw backend pointers
    // into `ggml_runner`, so this field must be declared before `ggml_runner`.
    full_encoder_reuse: RefCell<Option<XasrFullEncoderReusableGraph>>,
    ggml_runner: RefCell<Option<GgmlCpuGraphRunner>>,
    // The encoder-embed graph rebuilds its forward graph every chunk; it must run
    // on a SEPARATE backend from `ggml_runner` so it does not stomp the
    // full-encoder persistent session's prepared (frozen) graph buffer between
    // chunk computes.
    embed_ggml_runner: RefCell<Option<GgmlCpuGraphRunner>>,
}

impl std::fmt::Debug for XasrZipformerEncoderGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XasrZipformerEncoderGraph")
            .field("metadata", &self.metadata)
            .field("weights", &self.weights)
            .field("backend", &self.backend)
            .field("ggml_config", &self.ggml_config)
            .finish_non_exhaustive()
    }
}

impl XasrZipformerEncoderGraph {
    pub(crate) fn new_reference(
        metadata: XasrZipformerExecutionMetadata,
        weights: XasrEncoderWeights,
    ) -> Result<Self, XasrEncoderGraphError> {
        validate_metadata_and_weights(&metadata, &weights)?;
        Ok(Self {
            metadata,
            weights,
            backend: XasrEncoderGraphBackend::Reference,
            ggml_config: None,
            full_encoder_reuse: RefCell::new(None),
            ggml_runner: RefCell::new(None),
            embed_ggml_runner: RefCell::new(None),
        })
    }

    pub(crate) fn new_ggml_cpu_stack0(
        metadata: XasrZipformerExecutionMetadata,
        weights: XasrEncoderWeights,
        config: GgmlCpuGraphConfig,
    ) -> Result<Self, XasrEncoderGraphError> {
        validate_metadata_and_weights(&metadata, &weights)?;
        Ok(Self {
            metadata,
            weights,
            backend: XasrEncoderGraphBackend::GgmlCpuStack0,
            ggml_config: Some(config),
            full_encoder_reuse: RefCell::new(None),
            ggml_runner: RefCell::new(None),
            embed_ggml_runner: RefCell::new(None),
        })
    }

    pub(crate) fn new_ggml_cpu_full_encoder(
        metadata: XasrZipformerExecutionMetadata,
        weights: XasrEncoderWeights,
        config: GgmlCpuGraphConfig,
    ) -> Result<Self, XasrEncoderGraphError> {
        validate_metadata_and_weights(&metadata, &weights)?;
        Ok(Self {
            metadata,
            weights,
            backend: XasrEncoderGraphBackend::GgmlCpuFullEncoder,
            ggml_config: Some(config),
            full_encoder_reuse: RefCell::new(None),
            ggml_runner: RefCell::new(None),
            embed_ggml_runner: RefCell::new(None),
        })
    }

    pub(crate) fn backend(&self) -> XasrEncoderGraphBackend {
        self.backend
    }

    /// First production-facing slice: fbank -> encoder embed -> stack0.
    ///
    /// This intentionally stops before multiscale downsample/upsample. It gives
    /// the future executor a stable input/output boundary while we land exact
    /// GGML implementations for the Zipformer custom operators.
    pub(crate) fn encode_stack0_from_features(
        &self,
        input: &XasrEncoderFeatureInput,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        if input.feature_dim != self.metadata.feature_dim {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "feature_dim {} does not match metadata {}",
                    input.feature_dim, self.metadata.feature_dim
                ),
            });
        }
        let embed = encode_embed_reference(
            &self.weights.embed,
            &input.rows,
            input.frames,
            input.feature_dim,
            None,
        )
        .map_err(|reason| XasrEncoderGraphError::Reference {
            stage: "encoder_embed",
            reason,
        })?;
        self.encode_stack0_from_embed_rows(&embed.rows, embed.frames, embed.dim, input.frames)
    }

    /// Full pre-joiner encoder slice. Output rows stay in the Zipformer encoder
    /// dimension (`max(encoder_dims)`, 768 for X-ASR); the Rust joiner owns the
    /// later encoder projection to joiner dim.
    pub(crate) fn encode_from_features(
        &self,
        input: &XasrEncoderFeatureInput,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        if input.feature_dim != self.metadata.feature_dim {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "feature_dim {} does not match metadata {}",
                    input.feature_dim, self.metadata.feature_dim
                ),
            });
        }
        let embed = encode_embed_reference(
            &self.weights.embed,
            &input.rows,
            input.frames,
            input.feature_dim,
            None,
        )
        .map_err(|reason| XasrEncoderGraphError::Reference {
            stage: "encoder_embed",
            reason,
        })?;
        self.encode_from_embed_rows(&embed.rows, embed.frames, embed.dim, input.frames)
    }

    pub(crate) fn encode_streaming_chunk_from_features(
        &self,
        input: &XasrEncoderFeatureInput,
        state: Option<&XasrEncoderChunkState>,
    ) -> Result<XasrEncoderChunkOutput, XasrEncoderGraphError> {
        if self.backend != XasrEncoderGraphBackend::GgmlCpuFullEncoder {
            return Err(XasrEncoderGraphError::Shape {
                reason: "streaming chunk execution requires GgmlCpuFullEncoder backend".to_string(),
            });
        }
        if input.feature_dim != self.metadata.feature_dim {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "feature_dim {} does not match metadata {}",
                    input.feature_dim, self.metadata.feature_dim
                ),
            });
        }
        let embed = self.encode_embed_from_features_ggml(
            input,
            state.map(|state| state.embed_states.as_slice()),
        )?;
        let chunk = self.encode_from_embed_rows_ggml_full_with_cache_capture(
            &embed.rows,
            embed.frames,
            embed.dim,
            input.frames,
            state.map(|state| state.layer_caches.as_slice()),
        )?;
        Ok(XasrEncoderChunkOutput {
            output: chunk.output,
            state: XasrEncoderChunkState {
                embed_states: embed.new_embed_states,
                layer_caches: chunk.layer_caches,
            },
        })
    }

    fn encode_embed_from_features_ggml(
        &self,
        input: &XasrEncoderFeatureInput,
        embed_states: Option<&[f32]>,
    ) -> Result<XasrEncoderEmbedOwnedOutput, XasrEncoderGraphError> {
        let config = self
            .ggml_config
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML encoder embed backend requires a graph config".to_string(),
            })?;
        validate_rows_len(
            &input.rows,
            input.frames,
            input.feature_dim,
            "encoder embed feature input",
        )?;
        let total_profile = xasr_encoder_profile_start();
        let runner_profile = xasr_encoder_profile_start();
        // Dedicated embed backend: keep this rebuild-every-chunk graph off the
        // full-encoder persistent session's runner so it cannot invalidate the
        // prepared graph's frozen buffers.
        let mut runner_slot = self.embed_ggml_runner.borrow_mut();
        if runner_slot.is_none() {
            *runner_slot = Some(map_ggml_stage(
                "encoder_embed_runner_init",
                GgmlCpuGraphRunner::new(config),
            )?);
            xasr_encoder_profile_log(
                "encoder_graph_runner_init",
                runner_profile,
                format_args!(
                    "frames={} dim={} capture_caches=true backend={:?}",
                    input.frames, input.feature_dim, config.backend
                ),
            );
        } else {
            xasr_encoder_profile_log(
                "encoder_graph_runner_reuse",
                runner_profile,
                format_args!(
                    "frames={} dim={} capture_caches=true backend={:?}",
                    input.frames, input.feature_dim, config.backend
                ),
            );
        }
        let runner = runner_slot
            .as_mut()
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML encoder embed runner cache was not initialized".to_string(),
            })?;
        let build_profile = xasr_encoder_profile_start();
        let mut graph = runner.start_graph();
        let binding = map_ggml_stage(
            "encoder_embed_alloc",
            XasrEncoderEmbedGraphBinding::new(
                &mut graph,
                &self.weights.embed,
                input.frames,
                input.feature_dim,
            ),
        )?;
        binding.set_inputs(&mut graph)?;
        let features = binding.tensors.features;
        let output = map_ggml_stage(
            "encoder_embed_graph",
            apply_encoder_embed_graph(&graph, features, binding.tensors.embed, binding.shape),
        )?;
        map_ggml_stage("encoder_embed_set_output", graph.set_output(output.rows))?;
        map_ggml_stage(
            "encoder_embed_set_cache_output",
            graph.set_output(output.new_embed_states),
        )?;
        map_ggml_stage(
            "encoder_embed_upload_features",
            graph.set_f32_slice(features, &input.rows, "xasr_encoder_embed_features"),
        )?;
        binding.upload(&mut graph, &self.weights.embed, embed_states)?;
        let expected_rows = binding
            .shape
            .embed_frames
            .checked_mul(binding.shape.output_dim)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr encoder embed output length overflows".to_string(),
            })?;
        let expected_cache = binding
            .shape
            .channels
            .checked_mul(binding.shape.cache_frames)
            .and_then(|value| value.checked_mul(binding.shape.embed_width))
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr encoder embed cache length overflows".to_string(),
            })?;
        xasr_encoder_profile_log(
            "encoder_embed_graph_build",
            build_profile,
            format_args!(
                "input_frames={} embed_frames={} output_dim={}",
                input.frames, binding.shape.embed_frames, binding.shape.output_dim
            ),
        );
        let compute_profile = xasr_encoder_profile_start();
        let mut outputs = map_ggml_stage(
            "encoder_embed_compute",
            graph.compute_outputs_f32(&[
                (output.rows, expected_rows),
                (output.new_embed_states, expected_cache),
            ]),
        )?;
        xasr_encoder_profile_log(
            "encoder_embed_graph_compute",
            compute_profile,
            format_args!(
                "input_frames={} embed_frames={} output_dim={}",
                input.frames, binding.shape.embed_frames, binding.shape.output_dim
            ),
        );
        let rows = outputs.remove(0);
        let new_embed_states = outputs.remove(0);
        xasr_encoder_profile_log(
            "encoder_embed_graph_total",
            total_profile,
            format_args!(
                "input_frames={} embed_frames={} output_dim={}",
                input.frames, binding.shape.embed_frames, binding.shape.output_dim
            ),
        );
        Ok(XasrEncoderEmbedOwnedOutput {
            frames: binding.shape.embed_frames,
            dim: binding.shape.output_dim,
            rows,
            new_embed_states,
        })
    }

    pub(crate) fn encode_from_embed_rows(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        if self.backend == XasrEncoderGraphBackend::GgmlCpuFullEncoder {
            return self.encode_from_embed_rows_ggml_full(rows, frames, dim, valid_left_context);
        }
        let stack0 = self.encode_stack0_from_embed_rows(rows, frames, dim, valid_left_context)?;
        self.encode_multiscale_tail_from_stack0_rows(
            &stack0.rows,
            stack0.frames,
            stack0.dim,
            valid_left_context,
        )
    }

    /// Stack0 slice from already-computed embed rows. `valid_left_context` is the
    /// original feature-frame span represented by this chunk; for the first
    /// 480ms X-ASR oracle chunk this is 61 while stack0 sees 24 embed frames.
    pub(crate) fn encode_stack0_from_embed_rows(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        let stack = self.stack0()?;
        if dim != stack.dim {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "stack0 input dim {dim} does not match stack dim {}",
                    stack.dim
                ),
            });
        }
        let expected = frames
            .checked_mul(dim)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "stack0 input shape overflows".to_string(),
            })?;
        if rows.len() != expected {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "stack0 input has {} values, expected {frames}x{dim}={expected}",
                    rows.len()
                ),
            });
        }
        if valid_left_context == 0 {
            return Err(XasrEncoderGraphError::Shape {
                reason: "stack0 valid_left_context must be > 0".to_string(),
            });
        }

        match self.backend {
            XasrEncoderGraphBackend::Reference => self.encode_stack0_from_embed_rows_reference(
                stack,
                rows,
                frames,
                dim,
                valid_left_context,
            ),
            XasrEncoderGraphBackend::GgmlCpuStack0
            | XasrEncoderGraphBackend::GgmlCpuFullEncoder => self
                .encode_stack0_from_embed_rows_ggml(stack, rows, frames, dim, valid_left_context),
        }
    }

    fn encode_stack0_from_embed_rows_reference(
        &self,
        stack: &XasrEncoderStackWeights,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        let mut state = rows.to_vec();
        for (layer_index, layer) in stack.layers.iter().enumerate() {
            let output = zipformer_layer_streaming_reference(
                layer,
                &state,
                frames,
                dim,
                self.metadata.num_heads[0],
                self.metadata.query_head_dims[0],
                self.metadata.left_context_len[0],
                valid_left_context,
                XasrZipformerLayerReferenceCaches::default(),
            )
            .map_err(|reason| XasrEncoderGraphError::Reference {
                stage: "stack0_layer",
                reason: format!("layer {layer_index}: {reason}"),
            })?;
            state = output.rows;
        }

        Ok(XasrEncoderGraphOutput {
            frames,
            dim,
            rows: state,
        })
    }

    fn encode_stack0_from_embed_rows_ggml(
        &self,
        stack: &XasrEncoderStackWeights,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        let config = self
            .ggml_config
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML stack0 backend requires a graph config".to_string(),
            })?;
        let mut runner_slot = self.ggml_runner.borrow_mut();
        if runner_slot.is_none() {
            *runner_slot = Some(map_ggml_stage(
                "stack0_runner_init",
                GgmlCpuGraphRunner::new(config),
            )?);
        }
        let runner = runner_slot
            .as_mut()
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML stack0 runner cache was not initialized".to_string(),
            })?;
        let mut graph = runner.start_graph();
        let input = map_ggml_stage(
            "stack0_input_alloc",
            graph.new_tensor_2d_f32(dim, frames, "stack0_input"),
        )?;
        map_ggml_stage("stack0_set_input", graph.set_input(input))?;

        let mut state = input;
        let mut bindings = Vec::with_capacity(stack.layers.len());
        for layer in &stack.layers {
            let binding = map_ggml_stage(
                "stack0_layer_alloc",
                XasrZipformerLayerGraphBinding::new(
                    &mut graph,
                    layer,
                    dim,
                    frames,
                    self.metadata.left_context_len[0],
                    self.metadata.num_heads[0],
                    self.metadata.query_head_dims[0],
                ),
            )?;
            binding.set_inputs(&mut graph)?;
            let output = map_ggml_stage(
                "stack0_layer_graph",
                apply_zipformer_layer_graph(&graph, state, binding.tensors(), binding.shape()),
            )?;
            state = output.rows;
            bindings.push(binding);
        }

        map_ggml_stage("stack0_set_output", graph.set_output(state))?;
        map_ggml_stage(
            "stack0_upload_input",
            graph.set_f32_slice(input, rows, "stack0_input"),
        )?;
        for (binding, layer) in bindings.iter().zip(&stack.layers) {
            binding.upload(&mut graph, layer, valid_left_context, None)?;
        }

        let expected = frames
            .checked_mul(dim)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "stack0 output shape overflows".to_string(),
            })?;
        let mut outputs = map_ggml_stage(
            "stack0_compute",
            graph.compute_outputs_f32(&[(state, expected)]),
        )?;
        Ok(XasrEncoderGraphOutput {
            frames,
            dim,
            rows: outputs.remove(0),
        })
    }

    fn encode_from_embed_rows_ggml_full(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        self.encode_from_embed_rows_ggml_full_internal(
            rows,
            frames,
            dim,
            valid_left_context,
            None,
            false,
        )
        .map(|chunk| chunk.output)
    }

    fn encode_from_embed_rows_ggml_full_with_cache_capture(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
        layer_caches: Option<&[XasrEncoderLayerCache]>,
    ) -> Result<XasrEncoderChunkGraphOutput, XasrEncoderGraphError> {
        self.encode_from_embed_rows_ggml_full_internal(
            rows,
            frames,
            dim,
            valid_left_context,
            layer_caches,
            true,
        )
    }

    fn encode_from_embed_rows_ggml_full_internal(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
        layer_caches: Option<&[XasrEncoderLayerCache]>,
        capture_caches: bool,
    ) -> Result<XasrEncoderChunkGraphOutput, XasrEncoderGraphError> {
        let stack0 = self.stack0()?;
        if dim != stack0.dim {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "full encoder input dim {dim} does not match stack0 dim {}",
                    stack0.dim
                ),
            });
        }
        validate_rows_len(rows, frames, dim, "full encoder input")?;
        if valid_left_context == 0 {
            return Err(XasrEncoderGraphError::Shape {
                reason: "full encoder valid_left_context must be > 0".to_string(),
            });
        }
        if let Some(caches) = layer_caches
            && caches.len() != self.metadata.total_encoder_layers()
        {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "full encoder got {} layer cache(s), expected {}",
                    caches.len(),
                    self.metadata.total_encoder_layers()
                ),
            });
        }
        if capture_caches {
            return self.encode_from_embed_rows_ggml_full_reused(
                rows,
                frames,
                dim,
                valid_left_context,
                layer_caches,
            );
        }

        let config = self
            .ggml_config
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML full encoder backend requires a graph config".to_string(),
            })?;
        let total_profile = xasr_encoder_profile_start();
        let runner_profile = xasr_encoder_profile_start();
        let mut runner_slot = self.ggml_runner.borrow_mut();
        if runner_slot.is_none() {
            *runner_slot = Some(map_ggml_stage(
                "full_encoder_runner_init",
                GgmlCpuGraphRunner::new(config),
            )?);
            xasr_encoder_profile_log(
                "encoder_graph_runner_init",
                runner_profile,
                format_args!(
                    "frames={frames} dim={dim} capture_caches={capture_caches} backend={:?}",
                    config.backend
                ),
            );
        } else {
            xasr_encoder_profile_log(
                "encoder_graph_runner_reuse",
                runner_profile,
                format_args!(
                    "frames={frames} dim={dim} capture_caches={capture_caches} backend={:?}",
                    config.backend
                ),
            );
        }
        let runner = runner_slot
            .as_mut()
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML full encoder runner cache was not initialized".to_string(),
            })?;
        let build_profile = xasr_encoder_profile_start();
        let mut graph = runner.start_graph();
        let input = map_ggml_stage(
            "full_encoder_input_alloc",
            graph.new_tensor_2d_f32(dim, frames, "xasr_full_input"),
        )?;
        map_ggml_stage("full_encoder_set_input", graph.set_input(input))?;

        let mut state = input;
        let mut trunk_frames = frames;
        let mut trunk_dim = dim;
        let mut stack3_tail = None;
        let mut stack4_tail = None;
        let mut layer_uploads = Vec::new();
        let mut downsample_uploads = Vec::new();
        let mut out_combiner_uploads = Vec::new();
        let mut cache_index = 0usize;

        for stack_index in 0..self.metadata.num_stacks {
            let stack = self.weights.stacks.get(stack_index).ok_or_else(|| {
                XasrEncoderGraphError::Shape {
                    reason: format!("encoder weights missing stack{stack_index}"),
                }
            })?;
            let factor = stack.downsampling_factor;
            if factor == 0 {
                return Err(XasrEncoderGraphError::Shape {
                    reason: format!("stack{stack_index} downsampling factor must be > 0"),
                });
            }
            let (mut branch_rows, branch_frames, branch_valid_left_context, padded_rows) =
                if stack_index == 0 {
                    if factor != 1 {
                        return Err(XasrEncoderGraphError::Shape {
                            reason: format!("stack0 contract drift: downsampling_factor={factor}"),
                        });
                    }
                    (state, trunk_frames, valid_left_context, None)
                } else {
                    let downsample_bias = stack.downsample_bias.as_ref().ok_or_else(|| {
                        XasrEncoderGraphError::Shape {
                            reason: format!("stack{stack_index} missing downsample bias"),
                        }
                    })?;
                    let binding = map_ggml_stage(
                        "full_encoder_downsample_alloc",
                        XasrDownsampleGraphBinding::new(
                            &mut graph,
                            trunk_frames,
                            trunk_dim,
                            stack.dim,
                            factor,
                        ),
                    )?;
                    binding.set_inputs(&mut graph)?;
                    let downsample = map_ggml_stage(
                        "full_encoder_downsample_graph",
                        apply_downsample_graph(
                            &graph,
                            state,
                            downsample_bias,
                            binding.tensors,
                            binding.shape,
                        ),
                    )?;
                    downsample_uploads.push(binding);
                    (
                        downsample.rows,
                        downsample.frames,
                        valid_left_context.div_ceil(factor),
                        Some(downsample.padded_rows),
                    )
                };

            for (layer_index, layer) in stack.layers.iter().enumerate() {
                let binding = map_ggml_stage(
                    "full_encoder_layer_alloc",
                    XasrZipformerLayerGraphBinding::new(
                        &mut graph,
                        layer,
                        stack.dim,
                        branch_frames,
                        self.metadata.left_context_len[stack_index],
                        self.metadata.num_heads[stack_index],
                        self.metadata.query_head_dims[stack_index],
                    ),
                )?;
                binding.set_inputs(&mut graph)?;
                let output = map_ggml_stage(
                    "full_encoder_layer_graph",
                    apply_zipformer_layer_graph(
                        &graph,
                        branch_rows,
                        binding.tensors(),
                        binding.shape(),
                    ),
                )?;
                branch_rows = output.rows;
                layer_uploads.push(XasrZipformerLayerGraphUpload {
                    cache_index,
                    stack_index,
                    layer_index,
                    valid_left_context: branch_valid_left_context,
                    binding,
                    output,
                });
                cache_index += 1;
            }

            if let Some(padded_rows) = padded_rows {
                let padded_frames = branch_frames.checked_mul(factor).ok_or_else(|| {
                    XasrEncoderGraphError::Shape {
                        reason: format!("stack{stack_index} padded frame count overflows"),
                    }
                })?;
                let upsample = map_ggml_stage(
                    "full_encoder_upsample_graph",
                    apply_upsample_graph(
                        &graph,
                        branch_rows,
                        XasrUpsampleGraphShape {
                            frames: branch_frames,
                            dim: stack.dim,
                            factor,
                            target_frames: padded_frames,
                        },
                    ),
                )?;
                let scale = map_ggml_stage(
                    "full_encoder_out_combiner_alloc",
                    graph.new_tensor_1d_f32(stack.dim, "xasr_out_combiner_scale"),
                )?;
                map_ggml_stage(
                    "full_encoder_out_combiner_set_input",
                    graph.set_input(scale),
                )?;
                state = map_ggml_stage(
                    "full_encoder_out_combiner_graph",
                    apply_bypass_graph(&graph, padded_rows, upsample, scale),
                )?;
                out_combiner_uploads.push(XasrOutCombinerGraphUpload { stack_index, scale });
                trunk_frames = padded_frames;
                trunk_dim = stack.dim;
                if stack_index == 3 {
                    stack3_tail = Some(map_ggml_stage(
                        "full_encoder_stack3_tail_graph",
                        slice_frame_rows_graph(&graph, state, trunk_frames, trunk_dim, 512, 768),
                    )?);
                } else if stack_index == 4 {
                    stack4_tail = Some(map_ggml_stage(
                        "full_encoder_stack4_tail_graph",
                        slice_frame_rows_graph(&graph, state, trunk_frames, trunk_dim, 256, 512),
                    )?);
                }
            } else {
                state = branch_rows;
                trunk_frames = branch_frames;
                trunk_dim = stack.dim;
            }
        }

        let output_dim = self.metadata.encoder_output_dim();
        if output_dim == 768 && trunk_dim == 256 {
            let stack4_tail = stack4_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack4 tail".to_string(),
            })?;
            let stack3_tail = stack3_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack3 tail".to_string(),
            })?;
            state = map_ggml_stage(
                "full_encoder_final_concat_stack4",
                graph.concat(state, stack4_tail, 0),
            )?;
            state = map_ggml_stage(
                "full_encoder_final_concat_stack3",
                graph.concat(state, stack3_tail, 0),
            )?;
            trunk_dim = output_dim;
        }

        let output_factor = self.weights.downsample_output_bias.len();
        let output_downsample = map_ggml_stage(
            "full_encoder_output_downsample_alloc",
            XasrDownsampleGraphBinding::new(
                &mut graph,
                trunk_frames,
                trunk_dim,
                output_dim,
                output_factor,
            ),
        )?;
        output_downsample.set_inputs(&mut graph)?;
        let output = map_ggml_stage(
            "full_encoder_output_downsample_graph",
            apply_downsample_graph(
                &graph,
                state,
                &self.weights.downsample_output_bias,
                output_downsample.tensors,
                output_downsample.shape,
            ),
        )?;
        downsample_uploads.push(output_downsample);

        map_ggml_stage("full_encoder_set_output", graph.set_output(output.rows))?;
        map_ggml_stage(
            "full_encoder_upload_input",
            graph.set_f32_slice(input, rows, "xasr_full_input"),
        )?;
        for upload in &downsample_uploads {
            upload.upload(&mut graph)?;
        }
        for upload in &layer_uploads {
            let layer = &self.weights.stacks[upload.stack_index].layers[upload.layer_index];
            let caches = layer_caches.and_then(|caches| caches.get(upload.cache_index));
            upload
                .binding
                .upload(&mut graph, layer, upload.valid_left_context, caches)?;
        }
        for upload in &out_combiner_uploads {
            let scale = self.weights.stacks[upload.stack_index]
                .out_combiner_bypass_scale
                .as_ref()
                .ok_or_else(|| XasrEncoderGraphError::Shape {
                    reason: format!(
                        "stack{} missing out_combiner bypass scale",
                        upload.stack_index
                    ),
                })?;
            upload_f32(&mut graph, upload.scale, scale, "xasr_out_combiner_scale")?;
        }

        let expected =
            output
                .frames
                .checked_mul(output.dim)
                .ok_or_else(|| XasrEncoderGraphError::Shape {
                    reason: "full encoder output shape overflows".to_string(),
                })?;
        let mut output_specs = vec![(output.rows, expected)];
        if capture_caches {
            for upload in &layer_uploads {
                let lengths = layer_cache_lengths(upload.binding.shape())?;
                output_specs.extend_from_slice(&[
                    (upload.output.new_cached_key, lengths.cached_key),
                    (
                        upload.output.new_cached_nonlin_attention,
                        lengths.cached_nonlin_attention,
                    ),
                    (upload.output.new_cached_val1, lengths.cached_val1),
                    (upload.output.new_cached_val2, lengths.cached_val2),
                    (upload.output.new_cached_conv1, lengths.cached_conv1),
                    (upload.output.new_cached_conv2, lengths.cached_conv2),
                ]);
            }
        }
        xasr_encoder_profile_log(
            "encoder_graph_build",
            build_profile,
            format_args!(
                "frames={frames} output_frames={} output_specs={} layers={}",
                output.frames,
                output_specs.len(),
                layer_uploads.len()
            ),
        );
        let compute_profile = xasr_encoder_profile_start();
        let mut outputs = map_ggml_stage(
            "full_encoder_compute",
            graph.compute_outputs_f32(&output_specs),
        )?;
        xasr_encoder_profile_log(
            "encoder_graph_compute",
            compute_profile,
            format_args!(
                "frames={frames} output_frames={} output_specs={}",
                output.frames,
                output_specs.len()
            ),
        );
        let rows = outputs.remove(0);
        let mut new_layer_caches = Vec::new();
        if capture_caches {
            new_layer_caches.reserve(layer_uploads.len());
            for _ in &layer_uploads {
                new_layer_caches.push(XasrEncoderLayerCache {
                    cached_key: outputs.remove(0),
                    cached_nonlin_attention: outputs.remove(0),
                    cached_val1: outputs.remove(0),
                    cached_val2: outputs.remove(0),
                    cached_conv1: outputs.remove(0),
                    cached_conv2: outputs.remove(0),
                });
            }
        }
        xasr_encoder_profile_log(
            "encoder_graph_total",
            total_profile,
            format_args!(
                "frames={frames} output_frames={} capture_caches={capture_caches}",
                output.frames
            ),
        );
        Ok(XasrEncoderChunkGraphOutput {
            output: XasrEncoderGraphOutput {
                frames: output.frames,
                dim: output.dim,
                rows,
            },
            layer_caches: new_layer_caches,
        })
    }

    fn encode_from_embed_rows_ggml_full_reused(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
        layer_caches: Option<&[XasrEncoderLayerCache]>,
    ) -> Result<XasrEncoderChunkGraphOutput, XasrEncoderGraphError> {
        let config = self
            .ggml_config
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML full encoder backend requires a graph config".to_string(),
            })?;
        let total_profile = xasr_encoder_profile_start();
        {
            let mut reuse_slot = self.full_encoder_reuse.borrow_mut();
            let needs_rebuild = !reuse_slot
                .as_ref()
                .is_some_and(|reuse| reuse.matches(frames, dim, valid_left_context));
            if needs_rebuild {
                let runner_profile = xasr_encoder_profile_start();
                let mut runner_slot = self.ggml_runner.borrow_mut();
                if runner_slot.is_none() {
                    *runner_slot = Some(map_ggml_stage(
                        "full_encoder_runner_init",
                        GgmlCpuGraphRunner::new(config),
                    )?);
                    xasr_encoder_profile_log(
                        "encoder_graph_runner_init",
                        runner_profile,
                        format_args!(
                            "frames={frames} dim={dim} capture_caches=true backend={:?} scheduler={}",
                            config.backend, config.use_scheduler
                        ),
                    );
                } else {
                    xasr_encoder_profile_log(
                        "encoder_graph_runner_reuse",
                        runner_profile,
                        format_args!(
                            "frames={frames} dim={dim} capture_caches=true backend={:?} scheduler={}",
                            config.backend, config.use_scheduler
                        ),
                    );
                }
                let runner = runner_slot
                    .as_mut()
                    .ok_or_else(|| XasrEncoderGraphError::Shape {
                        reason: "GGML full encoder runner cache was not initialized".to_string(),
                    })?;
                let build_profile = xasr_encoder_profile_start();
                let mut session = map_ggml_stage(
                    "full_encoder_persistent_session",
                    runner.start_persistent_graph_session(config.context_bytes),
                )?;
                let graph = session.builder();
                let input = map_ggml_stage(
                    "full_encoder_input_alloc",
                    graph.new_tensor_2d_f32(dim, frames, "xasr_full_input"),
                )?;
                map_ggml_stage("full_encoder_set_input", graph.set_input(input))?;
                let plan = self.build_full_encoder_graph_plan(
                    graph,
                    input,
                    frames,
                    dim,
                    valid_left_context,
                )?;
                let output_specs = full_encoder_output_specs(&plan, true)?;
                // Build the forward cgraph and allocate the backend buffer ONCE here.
                // This flips the builder onto its prepared-graph fast path, so every
                // later compute_outputs_f32 skips build_forward_graph (which would
                // otherwise allocate a fresh cgraph into the persistent 2GB context
                // every chunk and eventually overflow it). With the cgraph frozen,
                // the session is reusable for the whole stream and per-chunk uploads
                // write in place into the allocated buffer.
                let output_tensors: Vec<_> =
                    output_specs.iter().map(|(tensor, _)| *tensor).collect();
                map_ggml_stage(
                    "full_encoder_prepare_outputs",
                    graph.prepare_outputs_for_upload(&output_tensors),
                )?;
                xasr_encoder_profile_log(
                    "encoder_graph_build",
                    build_profile,
                    format_args!(
                        "frames={frames} output_frames={} output_specs={} layers={} persistent_ops=true prepared=true",
                        plan.output.frames,
                        output_specs.len(),
                        plan.layer_uploads.len()
                    ),
                );
                *reuse_slot = Some(XasrFullEncoderReusableGraph {
                    session,
                    frames,
                    dim,
                    valid_left_context,
                    input,
                    output_frames: plan.output.frames,
                    output_dim: plan.output.dim,
                    static_uploaded: false,
                    output_specs,
                    layer_uploads: plan.layer_uploads,
                    downsample_uploads: plan.downsample_uploads,
                    out_combiner_uploads: plan.out_combiner_uploads,
                });
            }
        }

        let upload_profile = xasr_encoder_profile_start();
        let mut reuse_slot = self.full_encoder_reuse.borrow_mut();
        let reuse = reuse_slot
            .as_mut()
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "GGML full encoder reusable graph was not initialized".to_string(),
            })?;
        let input = reuse.input;
        let output_specs = reuse.output_specs.clone();
        let output_frames = reuse.output_frames;
        let output_dim = reuse.output_dim;
        let needs_static_upload = !reuse.static_uploaded;
        if needs_static_upload {
            reuse.static_uploaded = true;
        }
        let layer_uploads = reuse.layer_uploads.clone();
        let downsample_uploads = reuse.downsample_uploads.clone();
        let out_combiner_uploads = reuse.out_combiner_uploads.clone();
        let graph = reuse.builder();
        map_ggml_stage(
            "full_encoder_upload_input",
            graph.set_f32_slice(input, rows, "xasr_full_input"),
        )?;
        if needs_static_upload {
            for upload in &downsample_uploads {
                upload.upload(graph)?;
            }
            for upload in &layer_uploads {
                let layer = &self.weights.stacks[upload.stack_index].layers[upload.layer_index];
                let caches = layer_caches.and_then(|caches| caches.get(upload.cache_index));
                upload
                    .binding
                    .upload(graph, layer, upload.valid_left_context, caches)?;
            }
            for upload in &out_combiner_uploads {
                let scale = self.weights.stacks[upload.stack_index]
                    .out_combiner_bypass_scale
                    .as_ref()
                    .ok_or_else(|| XasrEncoderGraphError::Shape {
                        reason: format!(
                            "stack{} missing out_combiner bypass scale",
                            upload.stack_index
                        ),
                    })?;
                upload_f32(graph, upload.scale, scale, "xasr_out_combiner_scale")?;
            }
        } else {
            for upload in &layer_uploads {
                let caches = layer_caches.and_then(|caches| caches.get(upload.cache_index));
                upload.binding.upload_dynamic_caches(graph, caches)?;
            }
        }
        xasr_encoder_profile_log(
            "encoder_graph_upload",
            upload_profile,
            format_args!(
                "frames={frames} output_frames={output_frames} output_specs={} layers={} static={needs_static_upload}",
                output_specs.len(),
                layer_uploads.len()
            ),
        );
        let compute_profile = xasr_encoder_profile_start();
        let mut outputs = map_ggml_stage(
            "full_encoder_compute",
            graph.compute_outputs_f32(&output_specs),
        )?;
        xasr_encoder_profile_log(
            "encoder_graph_compute",
            compute_profile,
            format_args!(
                "frames={frames} output_frames={output_frames} output_specs={}",
                output_specs.len()
            ),
        );
        let rows = outputs.remove(0);
        let mut new_layer_caches = Vec::with_capacity(layer_uploads.len());
        for _ in &layer_uploads {
            new_layer_caches.push(XasrEncoderLayerCache {
                cached_key: outputs.remove(0),
                cached_nonlin_attention: outputs.remove(0),
                cached_val1: outputs.remove(0),
                cached_val2: outputs.remove(0),
                cached_conv1: outputs.remove(0),
                cached_conv2: outputs.remove(0),
            });
        }
        xasr_encoder_profile_log(
            "encoder_graph_total",
            total_profile,
            format_args!("frames={frames} output_frames={output_frames} capture_caches=true"),
        );
        Ok(XasrEncoderChunkGraphOutput {
            output: XasrEncoderGraphOutput {
                frames: output_frames,
                dim: output_dim,
                rows,
            },
            layer_caches: new_layer_caches,
        })
    }

    fn build_full_encoder_graph_plan<'a>(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        mut state: GgmlCpuTensor<'a>,
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrFullEncoderGraphPlan<'a>, XasrEncoderGraphError> {
        let mut trunk_frames = frames;
        let mut trunk_dim = dim;
        let mut stack3_tail = None;
        let mut stack4_tail = None;
        let mut layer_uploads = Vec::new();
        let mut downsample_uploads = Vec::new();
        let mut out_combiner_uploads = Vec::new();
        let mut cache_index = 0usize;

        for stack_index in 0..self.metadata.num_stacks {
            let stack = self.weights.stacks.get(stack_index).ok_or_else(|| {
                XasrEncoderGraphError::Shape {
                    reason: format!("encoder weights missing stack{stack_index}"),
                }
            })?;
            let factor = stack.downsampling_factor;
            if factor == 0 {
                return Err(XasrEncoderGraphError::Shape {
                    reason: format!("stack{stack_index} downsampling factor must be > 0"),
                });
            }
            let (mut branch_rows, branch_frames, branch_valid_left_context, padded_rows) =
                if stack_index == 0 {
                    if factor != 1 {
                        return Err(XasrEncoderGraphError::Shape {
                            reason: format!("stack0 contract drift: downsampling_factor={factor}"),
                        });
                    }
                    (state, trunk_frames, valid_left_context, None)
                } else {
                    let downsample_bias = stack.downsample_bias.as_ref().ok_or_else(|| {
                        XasrEncoderGraphError::Shape {
                            reason: format!("stack{stack_index} missing downsample bias"),
                        }
                    })?;
                    let binding = map_ggml_stage(
                        "full_encoder_downsample_alloc",
                        XasrDownsampleGraphBinding::new(
                            graph,
                            trunk_frames,
                            trunk_dim,
                            stack.dim,
                            factor,
                        ),
                    )?;
                    binding.set_persistent_inputs(graph)?;
                    let downsample = map_ggml_stage(
                        "full_encoder_downsample_graph",
                        apply_downsample_graph(
                            graph,
                            state,
                            downsample_bias,
                            binding.tensors,
                            binding.shape,
                        ),
                    )?;
                    downsample_uploads.push(binding);
                    (
                        downsample.rows,
                        downsample.frames,
                        valid_left_context.div_ceil(factor),
                        Some(downsample.padded_rows),
                    )
                };

            for (layer_index, layer) in stack.layers.iter().enumerate() {
                let binding = map_ggml_stage(
                    "full_encoder_layer_alloc",
                    XasrZipformerLayerGraphBinding::new(
                        graph,
                        layer,
                        stack.dim,
                        branch_frames,
                        self.metadata.left_context_len[stack_index],
                        self.metadata.num_heads[stack_index],
                        self.metadata.query_head_dims[stack_index],
                    ),
                )?;
                binding.set_persistent_inputs(graph)?;
                let output = map_ggml_stage(
                    "full_encoder_layer_graph",
                    apply_zipformer_layer_graph(
                        graph,
                        branch_rows,
                        binding.tensors(),
                        binding.shape(),
                    ),
                )?;
                branch_rows = output.rows;
                layer_uploads.push(XasrZipformerLayerGraphUpload {
                    cache_index,
                    stack_index,
                    layer_index,
                    valid_left_context: branch_valid_left_context,
                    binding,
                    output,
                });
                cache_index += 1;
            }

            if let Some(padded_rows) = padded_rows {
                let padded_frames = branch_frames.checked_mul(factor).ok_or_else(|| {
                    XasrEncoderGraphError::Shape {
                        reason: format!("stack{stack_index} padded frame count overflows"),
                    }
                })?;
                let upsample = map_ggml_stage(
                    "full_encoder_upsample_graph",
                    apply_upsample_graph(
                        graph,
                        branch_rows,
                        XasrUpsampleGraphShape {
                            frames: branch_frames,
                            dim: stack.dim,
                            factor,
                            target_frames: padded_frames,
                        },
                    ),
                )?;
                let scale = map_ggml_stage(
                    "full_encoder_out_combiner_alloc",
                    graph.new_tensor_1d_f32(stack.dim, "xasr_out_combiner_scale"),
                )?;
                map_ggml_stage(
                    "full_encoder_out_combiner_set_input",
                    graph.set_input(scale),
                )?;
                map_ggml_stage(
                    "full_encoder_out_combiner_keep_alive",
                    graph.set_output(scale),
                )?;
                state = map_ggml_stage(
                    "full_encoder_out_combiner_graph",
                    apply_bypass_graph(graph, padded_rows, upsample, scale),
                )?;
                out_combiner_uploads.push(XasrOutCombinerGraphUpload { stack_index, scale });
                trunk_frames = padded_frames;
                trunk_dim = stack.dim;
                if stack_index == 3 {
                    stack3_tail = Some(map_ggml_stage(
                        "full_encoder_stack3_tail_graph",
                        slice_frame_rows_graph(graph, state, trunk_frames, trunk_dim, 512, 768),
                    )?);
                } else if stack_index == 4 {
                    stack4_tail = Some(map_ggml_stage(
                        "full_encoder_stack4_tail_graph",
                        slice_frame_rows_graph(graph, state, trunk_frames, trunk_dim, 256, 512),
                    )?);
                }
            } else {
                state = branch_rows;
                trunk_frames = branch_frames;
                trunk_dim = stack.dim;
            }
        }

        let output_dim = self.metadata.encoder_output_dim();
        if output_dim == 768 && trunk_dim == 256 {
            let stack4_tail = stack4_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack4 tail".to_string(),
            })?;
            let stack3_tail = stack3_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack3 tail".to_string(),
            })?;
            state = map_ggml_stage(
                "full_encoder_final_concat_stack4",
                graph.concat(state, stack4_tail, 0),
            )?;
            state = map_ggml_stage(
                "full_encoder_final_concat_stack3",
                graph.concat(state, stack3_tail, 0),
            )?;
            trunk_dim = output_dim;
        }

        let output_factor = self.weights.downsample_output_bias.len();
        let output_downsample = map_ggml_stage(
            "full_encoder_output_downsample_alloc",
            XasrDownsampleGraphBinding::new(
                graph,
                trunk_frames,
                trunk_dim,
                output_dim,
                output_factor,
            ),
        )?;
        output_downsample.set_persistent_inputs(graph)?;
        let output = map_ggml_stage(
            "full_encoder_output_downsample_graph",
            apply_downsample_graph(
                graph,
                state,
                &self.weights.downsample_output_bias,
                output_downsample.tensors,
                output_downsample.shape,
            ),
        )?;
        downsample_uploads.push(output_downsample);
        map_ggml_stage("full_encoder_set_output", graph.set_output(output.rows))?;

        Ok(XasrFullEncoderGraphPlan {
            output,
            layer_uploads,
            downsample_uploads,
            out_combiner_uploads,
        })
    }

    fn stack0(&self) -> Result<&XasrEncoderStackWeights, XasrEncoderGraphError> {
        let stack = self
            .weights
            .stacks
            .first()
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "encoder weights do not include stack0".to_string(),
            })?;
        if stack.stack != 0 || stack.downsampling_factor != 1 {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "stack0 contract drift: stack={}, downsampling_factor={}",
                    stack.stack, stack.downsampling_factor
                ),
            });
        }
        Ok(stack)
    }

    fn encode_multiscale_tail_from_stack0_rows(
        &self,
        rows: &[f32],
        frames: usize,
        dim: usize,
        valid_left_context: usize,
    ) -> Result<XasrEncoderGraphOutput, XasrEncoderGraphError> {
        validate_rows_len(rows, frames, dim, "stack0 output")?;
        let mut trunk_rows = rows.to_vec();
        let mut trunk_frames = frames;
        let mut trunk_dim = dim;
        let mut stack3_tail = None;
        let mut stack4_tail = None;

        for stack_index in 1..self.metadata.num_stacks {
            let stack = self.weights.stacks.get(stack_index).ok_or_else(|| {
                XasrEncoderGraphError::Shape {
                    reason: format!("encoder weights missing stack{stack_index}"),
                }
            })?;
            let factor = stack.downsampling_factor;
            if factor == 0 {
                return Err(XasrEncoderGraphError::Shape {
                    reason: format!("stack{stack_index} downsampling factor must be > 0"),
                });
            }
            let downsample_bias =
                stack
                    .downsample_bias
                    .as_ref()
                    .ok_or_else(|| XasrEncoderGraphError::Shape {
                        reason: format!("stack{stack_index} missing downsample bias"),
                    })?;
            let downsample = downsample_streaming_reference(
                &trunk_rows,
                trunk_frames,
                trunk_dim,
                stack.dim,
                downsample_bias,
            )
            .map_err(|reason| XasrEncoderGraphError::Reference {
                stage: "stack_downsample",
                reason: format!("stack{stack_index}: {reason}"),
            })?;

            let mut branch_rows = downsample.rows;
            let branch_frames = downsample.frames;
            let branch_valid_left_context = valid_left_context.div_ceil(factor);
            for (layer_index, layer) in stack.layers.iter().enumerate() {
                let output = zipformer_layer_streaming_reference(
                    layer,
                    &branch_rows,
                    branch_frames,
                    stack.dim,
                    self.metadata.num_heads[stack_index],
                    self.metadata.query_head_dims[stack_index],
                    self.metadata.left_context_len[stack_index],
                    branch_valid_left_context,
                    XasrZipformerLayerReferenceCaches::default(),
                )
                .map_err(|reason| XasrEncoderGraphError::Reference {
                    stage: "stack_layer",
                    reason: format!("stack{stack_index}.layer{layer_index}: {reason}"),
                })?;
                branch_rows = output.rows;
            }

            let padded_frames = downsample
                .padded_rows
                .len()
                .checked_div(stack.dim)
                .ok_or_else(|| XasrEncoderGraphError::Shape {
                    reason: format!("stack{stack_index} padded frame count is invalid"),
                })?;
            let upsample = upsample_streaming_reference(
                &branch_rows,
                branch_frames,
                stack.dim,
                factor,
                padded_frames,
            )
            .map_err(|reason| XasrEncoderGraphError::Reference {
                stage: "stack_upsample",
                reason: format!("stack{stack_index}: {reason}"),
            })?;
            let out_combiner = stack.out_combiner_bypass_scale.as_ref().ok_or_else(|| {
                XasrEncoderGraphError::Shape {
                    reason: format!("stack{stack_index} missing out_combiner bypass scale"),
                }
            })?;
            trunk_rows = bypass_reference(
                &downsample.padded_rows,
                &upsample,
                out_combiner,
                padded_frames,
                stack.dim,
            )
            .map_err(|reason| XasrEncoderGraphError::Reference {
                stage: "stack_out_combiner",
                reason: format!("stack{stack_index}: {reason}"),
            })?;
            trunk_frames = padded_frames;
            trunk_dim = stack.dim;
            if stack_index == 3 {
                stack3_tail = Some(slice_frame_rows_reference(
                    &trunk_rows,
                    trunk_frames,
                    trunk_dim,
                    512,
                    768,
                )?);
            } else if stack_index == 4 {
                stack4_tail = Some(slice_frame_rows_reference(
                    &trunk_rows,
                    trunk_frames,
                    trunk_dim,
                    256,
                    512,
                )?);
            }
        }

        let output_dim = self.metadata.encoder_output_dim();
        if output_dim == 768 && trunk_dim == 256 {
            let stack4_tail = stack4_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack4 tail".to_string(),
            })?;
            let stack3_tail = stack3_tail.ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "full encoder final concat missing stack3 tail".to_string(),
            })?;
            trunk_rows = concat_frame_rows_reference(
                trunk_frames,
                &[(&trunk_rows, 256), (&stack4_tail, 256), (&stack3_tail, 256)],
            )?;
            trunk_dim = output_dim;
        }
        let output = downsample_streaming_reference(
            &trunk_rows,
            trunk_frames,
            trunk_dim,
            output_dim,
            &self.weights.downsample_output_bias,
        )
        .map_err(|reason| XasrEncoderGraphError::Reference {
            stage: "downsample_output",
            reason,
        })?;
        Ok(XasrEncoderGraphOutput {
            frames: output.frames,
            dim: output.dim,
            rows: output.rows,
        })
    }
}

fn validate_metadata_and_weights(
    metadata: &XasrZipformerExecutionMetadata,
    weights: &XasrEncoderWeights,
) -> Result<(), XasrEncoderGraphError> {
    if metadata.num_stacks == 0 {
        return Err(XasrEncoderGraphError::Shape {
            reason: "metadata num_stacks must include stack0".to_string(),
        });
    }
    for (name, len) in [
        ("num_encoder_layers", metadata.num_encoder_layers.len()),
        ("encoder_dims", metadata.encoder_dims.len()),
        ("query_head_dims", metadata.query_head_dims.len()),
        ("num_heads", metadata.num_heads.len()),
        ("left_context_len", metadata.left_context_len.len()),
        ("downsampling_factors", metadata.downsampling_factors.len()),
    ] {
        if len < metadata.num_stacks {
            return Err(XasrEncoderGraphError::Shape {
                reason: format!(
                    "metadata {name} has {len} entries, expected at least {}",
                    metadata.num_stacks
                ),
            });
        }
    }
    if weights.stacks.len() < metadata.num_stacks {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "weights include {} stack(s), metadata requires {}",
                weights.stacks.len(),
                metadata.num_stacks
            ),
        });
    }
    let stack0 = weights
        .stacks
        .first()
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "encoder weights do not include stack0".to_string(),
        })?;
    if stack0.dim != metadata.encoder_dims[0] {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "stack0 dim {} does not match metadata {}",
                stack0.dim, metadata.encoder_dims[0]
            ),
        });
    }
    if stack0.layers.len() != metadata.num_encoder_layers[0] {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "stack0 has {} layer(s), metadata declares {}",
                stack0.layers.len(),
                metadata.num_encoder_layers[0]
            ),
        });
    }
    Ok(())
}

fn slice_frame_rows_reference(
    rows: &[f32],
    frames: usize,
    dim: usize,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, XasrEncoderGraphError> {
    validate_rows_len(rows, frames, dim, "slice input")?;
    if start > end || end > dim {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!("slice range {start}..{end} exceeds dim {dim}"),
        });
    }
    let out_dim = end - start;
    let mut output = vec![0.0_f32; frames * out_dim];
    for frame in 0..frames {
        let src = frame * dim + start;
        let dst = frame * out_dim;
        output[dst..dst + out_dim].copy_from_slice(&rows[src..src + out_dim]);
    }
    Ok(output)
}

fn slice_frame_rows_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    rows: GgmlCpuTensor<'a>,
    frames: usize,
    dim: usize,
    start: usize,
    end: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if start > end || end > dim {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr graph slice range exceeds input dimension",
        });
    }
    let out_dim = end - start;
    if out_dim == 0 || frames == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr graph slice dimensions must be positive",
        });
    }
    let row_stride = dim.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr graph slice row stride overflows",
        },
    )?;
    let offset = start.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr graph slice offset overflows",
        },
    )?;
    let sliced = graph.view_2d(rows, out_dim, frames, row_stride, offset)?;
    graph.cont(sliced)
}

fn concat_frame_rows_reference(
    frames: usize,
    parts: &[(&[f32], usize)],
) -> Result<Vec<f32>, XasrEncoderGraphError> {
    let total_dim = parts
        .iter()
        .try_fold(0usize, |acc, (_, dim)| acc.checked_add(*dim))
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "concat output dim overflows".to_string(),
        })?;
    let mut output = vec![
        0.0_f32;
        frames.checked_mul(total_dim).ok_or_else(|| {
            XasrEncoderGraphError::Shape {
                reason: "concat output length overflows".to_string(),
            }
        })?
    ];
    for (rows, dim) in parts {
        validate_rows_len(rows, frames, *dim, "concat input")?;
    }
    for frame in 0..frames {
        let mut dst = frame * total_dim;
        for (rows, dim) in parts {
            let src = frame * *dim;
            output[dst..dst + *dim].copy_from_slice(&rows[src..src + *dim]);
            dst += *dim;
        }
    }
    Ok(output)
}

fn validate_rows_len(
    rows: &[f32],
    frames: usize,
    dim: usize,
    name: &'static str,
) -> Result<(), XasrEncoderGraphError> {
    let expected = frames
        .checked_mul(dim)
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: format!("{name} shape overflows"),
        })?;
    if rows.len() != expected {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "{name} has {} values, expected {frames}x{dim}={expected}",
                rows.len()
            ),
        });
    }
    Ok(())
}

fn full_encoder_output_specs<'a>(
    plan: &XasrFullEncoderGraphPlan<'a>,
    capture_caches: bool,
) -> Result<Vec<(GgmlCpuTensor<'a>, usize)>, XasrEncoderGraphError> {
    let expected = plan
        .output
        .frames
        .checked_mul(plan.output.dim)
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "full encoder output shape overflows".to_string(),
        })?;
    let mut output_specs = vec![(plan.output.rows, expected)];
    if capture_caches {
        for upload in &plan.layer_uploads {
            let lengths = layer_cache_lengths(upload.binding.shape())?;
            output_specs.extend_from_slice(&[
                (upload.output.new_cached_key, lengths.cached_key),
                (
                    upload.output.new_cached_nonlin_attention,
                    lengths.cached_nonlin_attention,
                ),
                (upload.output.new_cached_val1, lengths.cached_val1),
                (upload.output.new_cached_val2, lengths.cached_val2),
                (upload.output.new_cached_conv1, lengths.cached_conv1),
                (upload.output.new_cached_conv2, lengths.cached_conv2),
            ]);
        }
    }
    Ok(output_specs)
}

fn xasr_encoder_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(XASR_PROFILE_ENV)
            .and_then(|value| value.into_string().ok())
            .is_some_and(|value| {
                let value = value.trim();
                !value.is_empty()
                    && !value.eq_ignore_ascii_case("0")
                    && !value.eq_ignore_ascii_case("false")
            })
    })
}

fn xasr_encoder_profile_start() -> Option<Instant> {
    xasr_encoder_profile_enabled().then(Instant::now)
}

fn xasr_encoder_profile_log(
    stage: &str,
    started_at: Option<Instant>,
    detail: std::fmt::Arguments<'_>,
) {
    if let Some(started_at) = started_at {
        eprintln!(
            "openasr_xasr_profile stage={stage} elapsed_ms={:.3} {detail}",
            started_at.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn map_ggml_stage<T>(
    stage: &'static str,
    result: Result<T, GgmlCpuGraphError>,
) -> Result<T, XasrEncoderGraphError> {
    result.map_err(|source| XasrEncoderGraphError::Ggml { stage, source })
}

#[derive(Debug, Clone, Copy)]
struct XasrZipformerLayerGraphUpload<'a> {
    cache_index: usize,
    stack_index: usize,
    layer_index: usize,
    valid_left_context: usize,
    binding: XasrZipformerLayerGraphBinding<'a>,
    output: XasrZipformerLayerGraphOutput<'a>,
}

#[derive(Debug)]
struct XasrFullEncoderGraphPlan<'a> {
    output: XasrDownsampleGraphOutput<'a>,
    layer_uploads: Vec<XasrZipformerLayerGraphUpload<'a>>,
    downsample_uploads: Vec<XasrDownsampleGraphBinding<'a>>,
    out_combiner_uploads: Vec<XasrOutCombinerGraphUpload<'a>>,
}

struct XasrFullEncoderReusableGraph {
    session: GgmlPersistentGraphSession,
    frames: usize,
    dim: usize,
    valid_left_context: usize,
    input: GgmlCpuTensor<'static>,
    output_frames: usize,
    output_dim: usize,
    static_uploaded: bool,
    output_specs: Vec<(GgmlCpuTensor<'static>, usize)>,
    layer_uploads: Vec<XasrZipformerLayerGraphUpload<'static>>,
    downsample_uploads: Vec<XasrDownsampleGraphBinding<'static>>,
    out_combiner_uploads: Vec<XasrOutCombinerGraphUpload<'static>>,
}

impl XasrFullEncoderReusableGraph {
    fn matches(&self, frames: usize, dim: usize, valid_left_context: usize) -> bool {
        // The prepared cgraph is frozen and the persistent context no longer grows
        // per chunk, so one session that matches the chunk geometry is reused for
        // the entire stream (no rebuild cap).
        self.frames == frames && self.dim == dim && self.valid_left_context == valid_left_context
    }

    fn builder(&mut self) -> &mut GgmlCpuGraphBuilder<'static> {
        self.session.builder()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct XasrEncoderChunkGraphOutput {
    output: XasrEncoderGraphOutput,
    layer_caches: Vec<XasrEncoderLayerCache>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct XasrLayerCacheLengths {
    cached_key: usize,
    cached_nonlin_attention: usize,
    cached_val1: usize,
    cached_val2: usize,
    cached_conv1: usize,
    cached_conv2: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrEncoderEmbedGraphBinding<'a> {
    tensors: XasrEncoderEmbedGraphBindingTensors<'a>,
    shape: XasrEncoderEmbedGraphShape,
}

#[derive(Debug, Clone, Copy)]
struct XasrEncoderEmbedGraphBindingTensors<'a> {
    features: GgmlCpuTensor<'a>,
    embed: XasrEncoderEmbedGraphTensors<'a>,
}

impl<'a> XasrEncoderEmbedGraphBinding<'a> {
    fn new(
        graph: &mut GgmlCpuGraphBuilder<'a>,
        weights: &XasrEncoderEmbedWeights,
        input_frames: usize,
        feature_dim: usize,
    ) -> Result<Self, GgmlCpuGraphError> {
        let shape = encoder_embed_graph_shape(weights, input_frames, feature_dim)?;
        let features = graph.new_tensor_2d_f32(feature_dim, input_frames, "xasr_embed_features")?;
        let tensors = XasrEncoderEmbedGraphTensors {
            conv0_weight: allocate_conv2d_weight_tensor(graph, &weights.conv0, "xasr_embed_c0_w")?,
            conv0_bias: graph.new_tensor_1d_f32(weights.conv0.bias.len(), "xasr_embed_c0_b")?,
            conv4_weight: allocate_conv2d_weight_tensor(graph, &weights.conv4, "xasr_embed_c4_w")?,
            conv4_bias: graph.new_tensor_1d_f32(weights.conv4.bias.len(), "xasr_embed_c4_b")?,
            conv7_weight: allocate_conv2d_weight_tensor(graph, &weights.conv7, "xasr_embed_c7_w")?,
            conv7_bias: graph.new_tensor_1d_f32(weights.conv7.bias.len(), "xasr_embed_c7_b")?,
            embed_cache: graph.new_tensor_4d_f32(
                shape.embed_width,
                shape.cache_frames,
                shape.channels,
                1,
                "xasr_embed_cache",
            )?,
            convnext_depthwise_weight: allocate_conv2d_weight_tensor(
                graph,
                &weights.convnext_depthwise,
                "xasr_embed_cnx_dw_w",
            )?,
            convnext_depthwise_bias: graph
                .new_tensor_1d_f32(weights.convnext_depthwise.bias.len(), "xasr_embed_cnx_dw_b")?,
            convnext_pointwise1_weight: allocate_conv2d_weight_tensor(
                graph,
                &weights.convnext_pointwise1,
                "xasr_embed_cnx_p1_w",
            )?,
            convnext_pointwise1_bias: graph.new_tensor_1d_f32(
                weights.convnext_pointwise1.bias.len(),
                "xasr_embed_cnx_p1_b",
            )?,
            convnext_pointwise2_weight: allocate_conv2d_weight_tensor(
                graph,
                &weights.convnext_pointwise2,
                "xasr_embed_cnx_p2_w",
            )?,
            convnext_pointwise2_bias: graph.new_tensor_1d_f32(
                weights.convnext_pointwise2.bias.len(),
                "xasr_embed_cnx_p2_b",
            )?,
            out_weight: allocate_stored_linear_weight_tensor(
                graph,
                &weights.out.weight,
                "xasr_embed_out_w",
            )?,
            out_bias: graph.new_tensor_1d_f32(weights.out.bias.len(), "xasr_embed_out_b")?,
            out_norm_bias: graph
                .new_tensor_1d_f32(weights.out_norm_bias.len(), "xasr_embed_out_norm_b")?,
            swoosh_r_offset: graph.new_tensor_1d_f32(1, "xasr_embed_swoosh_r_offset")?,
            swoosh_r_shift: graph.new_tensor_1d_f32(1, "xasr_embed_swoosh_r_shift")?,
            swoosh_l_offset: graph.new_tensor_1d_f32(1, "xasr_embed_swoosh_l_offset")?,
            swoosh_l_shift: graph.new_tensor_1d_f32(1, "xasr_embed_swoosh_l_shift")?,
        };
        Ok(Self {
            tensors: XasrEncoderEmbedGraphBindingTensors {
                features,
                embed: tensors,
            },
            shape,
        })
    }

    fn set_inputs(&self, graph: &mut GgmlCpuGraphBuilder<'a>) -> Result<(), XasrEncoderGraphError> {
        let embed = self.tensors.embed;
        for tensor in [
            self.tensors.features,
            embed.conv0_weight,
            embed.conv0_bias,
            embed.conv4_weight,
            embed.conv4_bias,
            embed.conv7_weight,
            embed.conv7_bias,
            embed.embed_cache,
            embed.convnext_depthwise_weight,
            embed.convnext_depthwise_bias,
            embed.convnext_pointwise1_weight,
            embed.convnext_pointwise1_bias,
            embed.convnext_pointwise2_weight,
            embed.convnext_pointwise2_bias,
            embed.out_weight,
            embed.out_bias,
            embed.out_norm_bias,
            embed.swoosh_r_offset,
            embed.swoosh_r_shift,
            embed.swoosh_l_offset,
            embed.swoosh_l_shift,
        ] {
            map_ggml_stage("encoder_embed_set_input", graph.set_input(tensor))?;
        }
        Ok(())
    }

    fn upload(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        weights: &XasrEncoderEmbedWeights,
        embed_states: Option<&[f32]>,
    ) -> Result<(), XasrEncoderGraphError> {
        let tensors = self.tensors.embed;
        upload_conv2d_tensors(
            graph,
            tensors.conv0_weight,
            tensors.conv0_bias,
            &weights.conv0,
        )?;
        upload_conv2d_tensors(
            graph,
            tensors.conv4_weight,
            tensors.conv4_bias,
            &weights.conv4,
        )?;
        upload_conv2d_tensors(
            graph,
            tensors.conv7_weight,
            tensors.conv7_bias,
            &weights.conv7,
        )?;
        let expected_cache = self
            .shape
            .channels
            .checked_mul(self.shape.cache_frames)
            .and_then(|value| value.checked_mul(self.shape.embed_width))
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr encoder embed cache length overflows".to_string(),
            })?;
        upload_cache_or_zero(
            graph,
            tensors.embed_cache,
            expected_cache,
            embed_states,
            "xasr_embed_cache",
        )?;
        upload_conv2d_tensors(
            graph,
            tensors.convnext_depthwise_weight,
            tensors.convnext_depthwise_bias,
            &weights.convnext_depthwise,
        )?;
        upload_conv2d_tensors(
            graph,
            tensors.convnext_pointwise1_weight,
            tensors.convnext_pointwise1_bias,
            &weights.convnext_pointwise1,
        )?;
        upload_conv2d_tensors(
            graph,
            tensors.convnext_pointwise2_weight,
            tensors.convnext_pointwise2_bias,
            &weights.convnext_pointwise2,
        )?;
        upload_linear_with_bias_tensors(graph, tensors.out_weight, tensors.out_bias, &weights.out)?;
        upload_f32(
            graph,
            tensors.out_norm_bias,
            &weights.out_norm_bias,
            "xasr_embed_out_norm_b",
        )?;
        upload_f32(
            graph,
            tensors.swoosh_r_offset,
            &[SWOOSH_R_OFFSET],
            "xasr_embed_swoosh_r_offset",
        )?;
        upload_f32(
            graph,
            tensors.swoosh_r_shift,
            &[SWOOSH_R_SHIFT],
            "xasr_embed_swoosh_r_shift",
        )?;
        upload_f32(
            graph,
            tensors.swoosh_l_offset,
            &[SWOOSH_L_OFFSET],
            "xasr_embed_swoosh_l_offset",
        )?;
        upload_f32(
            graph,
            tensors.swoosh_l_shift,
            &[SWOOSH_L_SHIFT],
            "xasr_embed_swoosh_l_shift",
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct XasrDownsampleGraphBinding<'a> {
    tensors: XasrDownsampleGraphTensors<'a>,
    shape: XasrDownsampleGraphShape,
}

impl<'a> XasrDownsampleGraphBinding<'a> {
    fn new(
        graph: &mut GgmlCpuGraphBuilder<'a>,
        frames: usize,
        input_dim: usize,
        target_dim: usize,
        factor: usize,
    ) -> Result<Self, GgmlCpuGraphError> {
        if frames == 0 || input_dim == 0 || target_dim == 0 || factor == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample binding dimensions must be positive",
            });
        }
        let out_frames = frames.div_ceil(factor);
        let padded_frames =
            out_frames
                .checked_mul(factor)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr downsample binding padded frame count overflows",
                })?;
        let channel_pad = if target_dim > input_dim {
            Some(graph.new_tensor_2d_f32(target_dim - input_dim, frames, "xasr_down_ch_pad")?)
        } else {
            None
        };
        let frame_pad = if padded_frames > frames {
            Some(graph.new_tensor_2d_f32(target_dim, padded_frames - frames, "xasr_down_fr_pad")?)
        } else {
            None
        };
        Ok(Self {
            tensors: XasrDownsampleGraphTensors {
                channel_pad,
                frame_pad,
            },
            shape: XasrDownsampleGraphShape {
                frames,
                input_dim,
                target_dim,
                factor,
            },
        })
    }

    fn set_inputs(&self, graph: &mut GgmlCpuGraphBuilder<'a>) -> Result<(), XasrEncoderGraphError> {
        if let Some(tensor) = self.tensors.channel_pad {
            map_ggml_stage("downsample_channel_pad_set_input", graph.set_input(tensor))?;
        }
        if let Some(tensor) = self.tensors.frame_pad {
            map_ggml_stage("downsample_frame_pad_set_input", graph.set_input(tensor))?;
        }
        Ok(())
    }

    fn set_persistent_inputs(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
    ) -> Result<(), XasrEncoderGraphError> {
        self.set_inputs(graph)?;
        if let Some(tensor) = self.tensors.channel_pad {
            map_ggml_stage(
                "downsample_channel_pad_keep_alive",
                graph.set_output(tensor),
            )?;
        }
        if let Some(tensor) = self.tensors.frame_pad {
            map_ggml_stage("downsample_frame_pad_keep_alive", graph.set_output(tensor))?;
        }
        Ok(())
    }

    fn upload(&self, graph: &mut GgmlCpuGraphBuilder<'a>) -> Result<(), XasrEncoderGraphError> {
        if let Some(tensor) = self.tensors.channel_pad {
            upload_zero_cache(
                graph,
                tensor,
                self.shape.target_dim - self.shape.input_dim,
                self.shape.frames,
                "xasr_down_ch_pad",
            )?;
        }
        if let Some(tensor) = self.tensors.frame_pad {
            let out_frames = self.shape.frames.div_ceil(self.shape.factor);
            let padded_frames = out_frames.checked_mul(self.shape.factor).ok_or_else(|| {
                XasrEncoderGraphError::Shape {
                    reason: "xasr downsample binding padded frame count overflows".to_string(),
                }
            })?;
            upload_zero_cache(
                graph,
                tensor,
                self.shape.target_dim,
                padded_frames - self.shape.frames,
                "xasr_down_fr_pad",
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct XasrOutCombinerGraphUpload<'a> {
    stack_index: usize,
    scale: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrZipformerLayerGraphBinding<'a> {
    tensors: XasrZipformerLayerGraphTensors<'a>,
    shape: XasrZipformerLayerGraphShape,
}

impl<'a> XasrZipformerLayerGraphBinding<'a> {
    fn new(
        graph: &mut GgmlCpuGraphBuilder<'a>,
        layer: &XasrEncoderLayerWeights,
        dim: usize,
        frames: usize,
        left_context_len: usize,
        num_heads: usize,
        query_head_dim: usize,
    ) -> Result<Self, GgmlCpuGraphError> {
        let query_dim =
            num_heads
                .checked_mul(query_head_dim)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr stack0 query dimension overflows",
                })?;
        let k_len =
            left_context_len
                .checked_add(frames)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr stack0 key length overflows",
                })?;
        let attn = &layer.self_attn_weights;
        let pos_dim = attn.linear_pos.input_dim;
        let pos_output_dim = attn.linear_pos.output_dim;
        let rel_len = left_context_len
            .checked_add(
                frames
                    .checked_mul(2)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "xasr stack0 relative position length overflows",
                    })?,
            )
            .and_then(|value| value.checked_sub(1))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr stack0 relative position length overflows",
            })?;
        let nonlin_hidden_dim = layer.nonlin_attention.out_proj.weight.input_dim;
        let self1_value_dim = layer.self_attn1.in_proj.weight.output_dim;
        let self2_value_dim = layer.self_attn2.in_proj.weight.output_dim;
        let conv1_causal_kernel_len = conv1d_kernel_len(&layer.conv_module1)?;
        let conv1_chunk_kernel_len = chunk_conv1d_kernel_len(&layer.conv_module1)?;
        let conv1_cache_len = conv1_chunk_kernel_len / 2;
        let conv2_causal_kernel_len = conv1d_kernel_len(&layer.conv_module2)?;
        let conv2_chunk_kernel_len = chunk_conv1d_kernel_len(&layer.conv_module2)?;
        let conv2_cache_len = conv2_chunk_kernel_len / 2;

        let tensors = XasrZipformerLayerGraphTensors {
            layer_head: XasrLayerHeadGraphTensors {
                feed_forward1: allocate_feed_forward_tensors(graph, &layer.feed_forward1)?,
                attention_weights: XasrSelfAttentionWeightsGraphTensors {
                    cache: graph.new_tensor_2d_f32(
                        query_dim,
                        left_context_len,
                        "xasr_stack0_attn_cache",
                    )?,
                    mask: graph.new_tensor_2d_f32(k_len, frames, "xasr_stack0_attn_mask")?,
                    pos_embedding: graph.new_tensor_2d_f32(
                        pos_dim,
                        rel_len,
                        "xasr_stack0_attn_pos",
                    )?,
                    in_proj_weight: allocate_linear_weight_tensor(
                        graph,
                        &attn.in_proj,
                        "xasr_stack0_attn_in_w",
                    )?,
                    in_proj_bias: allocate_linear_bias_tensor(
                        graph,
                        &attn.in_proj,
                        "xasr_stack0_attn_in_b",
                    )?,
                    linear_pos_weight: allocate_stored_linear_weight_tensor(
                        graph,
                        &attn.linear_pos,
                        "xasr_stack0_attn_pos_w",
                    )?,
                },
                nonlin_cache: graph.new_tensor_2d_f32(
                    nonlin_hidden_dim,
                    left_context_len,
                    "xasr_stack0_nonlin_cache",
                )?,
                nonlin_in_proj_weight: allocate_linear_weight_tensor(
                    graph,
                    &layer.nonlin_attention.in_proj,
                    "xasr_stack0_nonlin_in_w",
                )?,
                nonlin_in_proj_bias: allocate_linear_bias_tensor(
                    graph,
                    &layer.nonlin_attention.in_proj,
                    "xasr_stack0_nonlin_in_b",
                )?,
                nonlin_out_proj_weight: allocate_linear_weight_tensor(
                    graph,
                    &layer.nonlin_attention.out_proj,
                    "xasr_stack0_nonlin_out_w",
                )?,
                nonlin_out_proj_bias: allocate_linear_bias_tensor(
                    graph,
                    &layer.nonlin_attention.out_proj,
                    "xasr_stack0_nonlin_out_b",
                )?,
                self1_cache: graph.new_tensor_2d_f32(
                    self1_value_dim,
                    left_context_len,
                    "xasr_stack0_self1_cache",
                )?,
                self1_in_proj_weight: allocate_linear_weight_tensor(
                    graph,
                    &layer.self_attn1.in_proj,
                    "xasr_stack0_self1_in_w",
                )?,
                self1_in_proj_bias: allocate_linear_bias_tensor(
                    graph,
                    &layer.self_attn1.in_proj,
                    "xasr_stack0_self1_in_b",
                )?,
                self1_out_proj_weight: allocate_linear_weight_tensor(
                    graph,
                    &layer.self_attn1.out_proj,
                    "xasr_stack0_self1_out_w",
                )?,
                self1_out_proj_bias: allocate_linear_bias_tensor(
                    graph,
                    &layer.self_attn1.out_proj,
                    "xasr_stack0_self1_out_b",
                )?,
                conv_module1: allocate_convolution_module_tensors(
                    graph,
                    &layer.conv_module1,
                    dim,
                    frames,
                    conv1_cache_len,
                )?,
                feed_forward2: allocate_feed_forward_tensors(graph, &layer.feed_forward2)?,
                bypass_mid_scale: graph.new_tensor_1d_f32(dim, "xasr_stack0_bypass_mid_scale")?,
            },
            self2_cache: graph.new_tensor_2d_f32(
                self2_value_dim,
                left_context_len,
                "xasr_stack0_self2_cache",
            )?,
            self2_in_proj_weight: allocate_linear_weight_tensor(
                graph,
                &layer.self_attn2.in_proj,
                "xasr_stack0_self2_in_w",
            )?,
            self2_in_proj_bias: allocate_linear_bias_tensor(
                graph,
                &layer.self_attn2.in_proj,
                "xasr_stack0_self2_in_b",
            )?,
            self2_out_proj_weight: allocate_linear_weight_tensor(
                graph,
                &layer.self_attn2.out_proj,
                "xasr_stack0_self2_out_w",
            )?,
            self2_out_proj_bias: allocate_linear_bias_tensor(
                graph,
                &layer.self_attn2.out_proj,
                "xasr_stack0_self2_out_b",
            )?,
            layer_tail: XasrLayerTailGraphTensors {
                conv_module2: allocate_convolution_module_tensors(
                    graph,
                    &layer.conv_module2,
                    dim,
                    frames,
                    conv2_cache_len,
                )?,
                feed_forward3: allocate_feed_forward_tensors(graph, &layer.feed_forward3)?,
                norm_bias: graph.new_tensor_1d_f32(dim, "xasr_stack0_norm_bias")?,
                bypass_scale: graph.new_tensor_1d_f32(dim, "xasr_stack0_bypass_scale")?,
            },
        };
        let shape = XasrZipformerLayerGraphShape {
            layer_head: XasrLayerHeadGraphShape {
                dim,
                frames,
                left_context_len,
                num_heads,
                query_head_dim,
                pos_dim,
                pos_output_dim,
                nonlin_hidden_dim,
                self1_value_dim,
                conv1_cache_len,
                conv1_causal_kernel_len,
                conv1_chunk_kernel_len,
            },
            self2_value_dim,
            layer_tail: XasrLayerTailGraphShape {
                dim,
                frames,
                conv_cache_len: conv2_cache_len,
                conv_causal_kernel_len: conv2_causal_kernel_len,
                conv_chunk_kernel_len: conv2_chunk_kernel_len,
                norm_log_scale: layer.norm_log_scale[0],
            },
        };
        Ok(Self { tensors, shape })
    }

    fn tensors(&self) -> XasrZipformerLayerGraphTensors<'a> {
        self.tensors
    }

    fn shape(&self) -> XasrZipformerLayerGraphShape {
        self.shape
    }

    fn set_inputs(&self, graph: &mut GgmlCpuGraphBuilder<'a>) -> Result<(), XasrEncoderGraphError> {
        self.for_each_input_tensor(|tensor| {
            map_ggml_stage("stack0_layer_set_input", graph.set_input(tensor))
        })
    }

    fn set_persistent_inputs(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
    ) -> Result<(), XasrEncoderGraphError> {
        self.set_inputs(graph)?;
        self.for_each_static_input_tensor(|tensor| {
            map_ggml_stage("stack0_layer_keep_alive", graph.set_output(tensor))
        })
    }

    fn for_each_input_tensor(
        &self,
        mut f: impl FnMut(GgmlCpuTensor<'a>) -> Result<(), XasrEncoderGraphError>,
    ) -> Result<(), XasrEncoderGraphError> {
        let tensors = self.tensors;
        let head = tensors.layer_head;
        let attn = head.attention_weights;
        let conv1 = head.conv_module1;
        let tail = tensors.layer_tail;
        let conv2 = tail.conv_module2;
        for tensor in [
            head.feed_forward1.in_proj_weight,
            head.feed_forward1.in_proj_bias,
            head.feed_forward1.out_proj_weight,
            head.feed_forward1.out_proj_bias,
            head.feed_forward1.swoosh_l_offset,
            head.feed_forward1.swoosh_l_shift,
            attn.cache,
            attn.mask,
            attn.pos_embedding,
            attn.in_proj_weight,
            attn.in_proj_bias,
            attn.linear_pos_weight,
            head.nonlin_cache,
            head.nonlin_in_proj_weight,
            head.nonlin_in_proj_bias,
            head.nonlin_out_proj_weight,
            head.nonlin_out_proj_bias,
            head.self1_cache,
            head.self1_in_proj_weight,
            head.self1_in_proj_bias,
            head.self1_out_proj_weight,
            head.self1_out_proj_bias,
            conv1.cache,
            conv1.in_proj_weight,
            conv1.in_proj_bias,
            conv1.causal_kernel,
            conv1.causal_bias,
            conv1.chunk_kernel,
            conv1.chunk_bias,
            conv1.chunk_scale,
            conv1.out_proj_weight,
            conv1.out_proj_bias,
            conv1.swoosh_r_offset,
            conv1.swoosh_r_shift,
            head.feed_forward2.in_proj_weight,
            head.feed_forward2.in_proj_bias,
            head.feed_forward2.out_proj_weight,
            head.feed_forward2.out_proj_bias,
            head.feed_forward2.swoosh_l_offset,
            head.feed_forward2.swoosh_l_shift,
            head.bypass_mid_scale,
            tensors.self2_cache,
            tensors.self2_in_proj_weight,
            tensors.self2_in_proj_bias,
            tensors.self2_out_proj_weight,
            tensors.self2_out_proj_bias,
            conv2.cache,
            conv2.in_proj_weight,
            conv2.in_proj_bias,
            conv2.causal_kernel,
            conv2.causal_bias,
            conv2.chunk_kernel,
            conv2.chunk_bias,
            conv2.chunk_scale,
            conv2.out_proj_weight,
            conv2.out_proj_bias,
            conv2.swoosh_r_offset,
            conv2.swoosh_r_shift,
            tail.feed_forward3.in_proj_weight,
            tail.feed_forward3.in_proj_bias,
            tail.feed_forward3.out_proj_weight,
            tail.feed_forward3.out_proj_bias,
            tail.feed_forward3.swoosh_l_offset,
            tail.feed_forward3.swoosh_l_shift,
            tail.norm_bias,
            tail.bypass_scale,
        ] {
            f(tensor)?;
        }
        Ok(())
    }

    fn for_each_static_input_tensor(
        &self,
        mut f: impl FnMut(GgmlCpuTensor<'a>) -> Result<(), XasrEncoderGraphError>,
    ) -> Result<(), XasrEncoderGraphError> {
        let tensors = self.tensors;
        let head = tensors.layer_head;
        let attn = head.attention_weights;
        let conv1 = head.conv_module1;
        let tail = tensors.layer_tail;
        let conv2 = tail.conv_module2;
        for tensor in [
            head.feed_forward1.in_proj_weight,
            head.feed_forward1.in_proj_bias,
            head.feed_forward1.out_proj_weight,
            head.feed_forward1.out_proj_bias,
            head.feed_forward1.swoosh_l_offset,
            head.feed_forward1.swoosh_l_shift,
            attn.mask,
            attn.pos_embedding,
            attn.in_proj_weight,
            attn.in_proj_bias,
            attn.linear_pos_weight,
            head.nonlin_in_proj_weight,
            head.nonlin_in_proj_bias,
            head.nonlin_out_proj_weight,
            head.nonlin_out_proj_bias,
            head.self1_in_proj_weight,
            head.self1_in_proj_bias,
            head.self1_out_proj_weight,
            head.self1_out_proj_bias,
            conv1.in_proj_weight,
            conv1.in_proj_bias,
            conv1.causal_kernel,
            conv1.causal_bias,
            conv1.chunk_kernel,
            conv1.chunk_bias,
            conv1.chunk_scale,
            conv1.out_proj_weight,
            conv1.out_proj_bias,
            conv1.swoosh_r_offset,
            conv1.swoosh_r_shift,
            head.feed_forward2.in_proj_weight,
            head.feed_forward2.in_proj_bias,
            head.feed_forward2.out_proj_weight,
            head.feed_forward2.out_proj_bias,
            head.feed_forward2.swoosh_l_offset,
            head.feed_forward2.swoosh_l_shift,
            head.bypass_mid_scale,
            tensors.self2_in_proj_weight,
            tensors.self2_in_proj_bias,
            tensors.self2_out_proj_weight,
            tensors.self2_out_proj_bias,
            conv2.in_proj_weight,
            conv2.in_proj_bias,
            conv2.causal_kernel,
            conv2.causal_bias,
            conv2.chunk_kernel,
            conv2.chunk_bias,
            conv2.chunk_scale,
            conv2.out_proj_weight,
            conv2.out_proj_bias,
            conv2.swoosh_r_offset,
            conv2.swoosh_r_shift,
            tail.feed_forward3.in_proj_weight,
            tail.feed_forward3.in_proj_bias,
            tail.feed_forward3.out_proj_weight,
            tail.feed_forward3.out_proj_bias,
            tail.feed_forward3.swoosh_l_offset,
            tail.feed_forward3.swoosh_l_shift,
            tail.norm_bias,
            tail.bypass_scale,
        ] {
            f(tensor)?;
        }
        Ok(())
    }

    fn upload(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        layer: &XasrEncoderLayerWeights,
        valid_left_context: usize,
        cache: Option<&XasrEncoderLayerCache>,
    ) -> Result<(), XasrEncoderGraphError> {
        let tensors = self.tensors;
        let shape = self.shape;
        let head = tensors.layer_head;
        let lengths = layer_cache_lengths(shape)?;
        upload_feed_forward_tensors(graph, head.feed_forward1, &layer.feed_forward1)?;
        upload_attention_weights_tensors(
            graph,
            head.attention_weights,
            &layer.self_attn_weights,
            shape.layer_head,
            valid_left_context,
            cache.map(|cache| cache.cached_key.as_slice()),
        )?;
        upload_cache_or_zero(
            graph,
            head.nonlin_cache,
            lengths.cached_nonlin_attention,
            cache.map(|cache| cache.cached_nonlin_attention.as_slice()),
            "xasr_stack0_nonlin_cache",
        )?;
        upload_linear_with_bias_tensors(
            graph,
            head.nonlin_in_proj_weight,
            head.nonlin_in_proj_bias,
            &layer.nonlin_attention.in_proj,
        )?;
        upload_linear_with_bias_tensors(
            graph,
            head.nonlin_out_proj_weight,
            head.nonlin_out_proj_bias,
            &layer.nonlin_attention.out_proj,
        )?;
        upload_cache_or_zero(
            graph,
            head.self1_cache,
            lengths.cached_val1,
            cache.map(|cache| cache.cached_val1.as_slice()),
            "xasr_stack0_self1_cache",
        )?;
        upload_linear_with_bias_tensors(
            graph,
            head.self1_in_proj_weight,
            head.self1_in_proj_bias,
            &layer.self_attn1.in_proj,
        )?;
        upload_linear_with_bias_tensors(
            graph,
            head.self1_out_proj_weight,
            head.self1_out_proj_bias,
            &layer.self_attn1.out_proj,
        )?;
        upload_convolution_module_tensors(
            graph,
            head.conv_module1,
            &layer.conv_module1,
            shape.layer_head.dim,
            shape.layer_head.frames,
            shape.layer_head.conv1_cache_len,
            cache.map(|cache| cache.cached_conv1.as_slice()),
        )?;
        upload_feed_forward_tensors(graph, head.feed_forward2, &layer.feed_forward2)?;
        upload_f32(
            graph,
            head.bypass_mid_scale,
            &layer.bypass_mid_scale,
            "xasr_stack0_bypass_mid_scale",
        )?;
        upload_cache_or_zero(
            graph,
            tensors.self2_cache,
            lengths.cached_val2,
            cache.map(|cache| cache.cached_val2.as_slice()),
            "xasr_stack0_self2_cache",
        )?;
        upload_linear_with_bias_tensors(
            graph,
            tensors.self2_in_proj_weight,
            tensors.self2_in_proj_bias,
            &layer.self_attn2.in_proj,
        )?;
        upload_linear_with_bias_tensors(
            graph,
            tensors.self2_out_proj_weight,
            tensors.self2_out_proj_bias,
            &layer.self_attn2.out_proj,
        )?;
        upload_convolution_module_tensors(
            graph,
            tensors.layer_tail.conv_module2,
            &layer.conv_module2,
            shape.layer_tail.dim,
            shape.layer_tail.frames,
            shape.layer_tail.conv_cache_len,
            cache.map(|cache| cache.cached_conv2.as_slice()),
        )?;
        upload_feed_forward_tensors(
            graph,
            tensors.layer_tail.feed_forward3,
            &layer.feed_forward3,
        )?;
        upload_f32(
            graph,
            tensors.layer_tail.norm_bias,
            &layer.norm_bias,
            "xasr_stack0_norm_bias",
        )?;
        upload_f32(
            graph,
            tensors.layer_tail.bypass_scale,
            &layer.bypass_scale,
            "xasr_stack0_bypass_scale",
        )?;
        Ok(())
    }

    fn upload_dynamic_caches(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        cache: Option<&XasrEncoderLayerCache>,
    ) -> Result<(), XasrEncoderGraphError> {
        let tensors = self.tensors;
        let head = tensors.layer_head;
        let lengths = layer_cache_lengths(self.shape)?;
        upload_cache_or_zero(
            graph,
            head.attention_weights.cache,
            lengths.cached_key,
            cache.map(|cache| cache.cached_key.as_slice()),
            "xasr_stack0_attn_cache",
        )?;
        upload_cache_or_zero(
            graph,
            head.nonlin_cache,
            lengths.cached_nonlin_attention,
            cache.map(|cache| cache.cached_nonlin_attention.as_slice()),
            "xasr_stack0_nonlin_cache",
        )?;
        upload_cache_or_zero(
            graph,
            head.self1_cache,
            lengths.cached_val1,
            cache.map(|cache| cache.cached_val1.as_slice()),
            "xasr_stack0_self1_cache",
        )?;
        upload_cache_or_zero(
            graph,
            head.conv_module1.cache,
            lengths.cached_conv1,
            cache.map(|cache| cache.cached_conv1.as_slice()),
            "xasr_conv_cache",
        )?;
        upload_cache_or_zero(
            graph,
            tensors.self2_cache,
            lengths.cached_val2,
            cache.map(|cache| cache.cached_val2.as_slice()),
            "xasr_stack0_self2_cache",
        )?;
        upload_cache_or_zero(
            graph,
            tensors.layer_tail.conv_module2.cache,
            lengths.cached_conv2,
            cache.map(|cache| cache.cached_conv2.as_slice()),
            "xasr_conv_cache",
        )
    }
}

fn allocate_feed_forward_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &XasrLinearPairWeights,
) -> Result<XasrFeedForwardGraphTensors<'a>, GgmlCpuGraphError> {
    Ok(XasrFeedForwardGraphTensors {
        in_proj_weight: allocate_linear_weight_tensor(graph, &weights.in_proj, "xasr_ff_in_w")?,
        in_proj_bias: allocate_linear_bias_tensor(graph, &weights.in_proj, "xasr_ff_in_b")?,
        out_proj_weight: allocate_linear_weight_tensor(graph, &weights.out_proj, "xasr_ff_out_w")?,
        out_proj_bias: allocate_linear_bias_tensor(graph, &weights.out_proj, "xasr_ff_out_b")?,
        swoosh_l_offset: graph.new_tensor_1d_f32(1, "xasr_ff_swoosh_l_offset")?,
        swoosh_l_shift: graph.new_tensor_1d_f32(1, "xasr_ff_swoosh_l_shift")?,
    })
}

fn allocate_convolution_module_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &XasrConvolutionModuleWeights,
    dim: usize,
    frames: usize,
    cache_len: usize,
) -> Result<XasrConvolutionModuleGraphTensors<'a>, GgmlCpuGraphError> {
    Ok(XasrConvolutionModuleGraphTensors {
        cache: graph.new_tensor_2d_f32(cache_len, dim, "xasr_conv_cache")?,
        in_proj_weight: allocate_linear_weight_tensor(graph, &weights.in_proj, "xasr_conv_in_w")?,
        in_proj_bias: allocate_linear_bias_tensor(graph, &weights.in_proj, "xasr_conv_in_b")?,
        causal_kernel: graph.new_tensor_3d_f32(
            conv1d_kernel_len(weights)?,
            1,
            dim,
            "xasr_conv_causal_w",
        )?,
        causal_bias: graph.new_tensor_1d_f32(dim, "xasr_conv_causal_b")?,
        chunk_kernel: graph.new_tensor_3d_f32(
            chunk_conv1d_kernel_len(weights)?,
            1,
            dim,
            "xasr_conv_chunk_w",
        )?,
        chunk_bias: graph.new_tensor_1d_f32(dim, "xasr_conv_chunk_b")?,
        chunk_scale: graph.new_tensor_2d_f32(frames, dim, "xasr_conv_chunk_scale")?,
        out_proj_weight: allocate_linear_weight_tensor(
            graph,
            &weights.out_proj,
            "xasr_conv_out_w",
        )?,
        out_proj_bias: allocate_linear_bias_tensor(graph, &weights.out_proj, "xasr_conv_out_b")?,
        swoosh_r_offset: graph.new_tensor_1d_f32(1, "xasr_conv_swoosh_r_offset")?,
        swoosh_r_shift: graph.new_tensor_1d_f32(1, "xasr_conv_swoosh_r_shift")?,
    })
}

fn allocate_linear_weight_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &XasrLinearWithBias,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    allocate_stored_linear_weight_tensor(graph, &weights.weight, name)
}

/// Allocate the ggml tensor for a rank-2 `.weight` `mul_mat` operand. When the
/// weight carries a native (quantized / f16) payload it is allocated at that
/// stored ggml type via `new_matmul_weight_2d_typed` -- gated to CPU-supported
/// types for a direct GPU backend -- so the weight stays quantized in the backend
/// buffer and feeds `mul_mat`'s quantized/f16 lhs path directly. Without a native
/// payload (f32 test graphs / dequantized providers) it falls back to an f32
/// tensor. The stored layout is `[ne0=in, ne1=out]` (the importer already reversed
/// HF `[out, in]`), so the raw block bytes upload order-preserving.
fn allocate_stored_linear_weight_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weight: &StoredLinear,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    match &weight.native {
        Some(payload) => graph.new_matmul_weight_2d_typed(
            weight.input_dim,
            weight.output_dim,
            payload.element_type.ggml_type(),
            name,
        ),
        None => graph.new_tensor_2d_f32(weight.input_dim, weight.output_dim, name),
    }
}

/// Upload a rank-2 `.weight` `mul_mat` operand. A native (quantized / f16) weight
/// uploads its raw ggml block bytes verbatim (kept quantized in the backend
/// buffer, no dequant-to-f32 blow-up); a dequantized weight uploads its f32
/// `values`.
fn upload_stored_linear_weight_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    weight: &StoredLinear,
    name: &'static str,
) -> Result<(), XasrEncoderGraphError> {
    match &weight.native {
        Some(payload) => map_ggml_stage(
            "stack0_layer_upload",
            graph.set_matmul_weight_bytes(
                tensor,
                payload.bytes(),
                payload.element_type.ggml_type(),
                name,
            ),
        ),
        None => upload_f32(graph, tensor, &weight.values, name),
    }
}

fn allocate_linear_bias_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &XasrLinearWithBias,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    graph.new_tensor_1d_f32(weights.bias.len(), name)
}

fn allocate_conv2d_weight_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weights: &XasrConv2dWeights,
    name: &'static str,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let [width, height, input, output]: [usize; 4] =
        weights.weight.dims.as_slice().try_into().map_err(|_| {
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr conv2d weight must have rank 4",
            }
        })?;
    graph.new_tensor_4d_f32(width, height, input, output, name)
}

fn encoder_embed_graph_shape(
    weights: &XasrEncoderEmbedWeights,
    input_frames: usize,
    feature_dim: usize,
) -> Result<XasrEncoderEmbedGraphShape, GgmlCpuGraphError> {
    if weights.out_norm_log_scale.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed out_norm_log_scale must not be empty",
        });
    }
    let conv0 = conv2d_dims(&weights.conv0)?;
    let conv4 = conv2d_dims(&weights.conv4)?;
    let conv7 = conv2d_dims(&weights.conv7)?;
    let conv0_h = conv2d_output_dim(input_frames, conv0.1, 1, 0, 0)?;
    let conv4_h = conv2d_output_dim(conv0_h, conv4.1, 2, 0, 0)?;
    let subsampled_frames = conv2d_output_dim(conv4_h, conv7.1, 1, 0, 0)?;
    let conv0_w = conv2d_output_dim(feature_dim, conv0.0, 1, 1, 1)?;
    let conv4_w = conv2d_output_dim(conv0_w, conv4.0, 2, 0, 0)?;
    let embed_width = conv2d_output_dim(conv4_w, conv7.0, 2, 0, 0)?;
    let cache_frames = 3usize;
    let embed_frames = subsampled_frames.checked_sub(cache_frames).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed subsampled frames must exceed cache frames",
        },
    )?;
    Ok(XasrEncoderEmbedGraphShape {
        input_frames,
        feature_dim,
        embed_width,
        subsampled_frames,
        embed_frames,
        cache_frames,
        channels: conv7.3,
        output_dim: weights.out.weight.output_dim,
        out_norm_log_scale: weights.out_norm_log_scale[0],
    })
}

fn conv2d_dims(
    weights: &XasrConv2dWeights,
) -> Result<(usize, usize, usize, usize), GgmlCpuGraphError> {
    let [width, height, input, output]: [usize; 4] =
        weights.weight.dims.as_slice().try_into().map_err(|_| {
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr conv2d weight must have rank 4",
            }
        })?;
    Ok((width, height, input, output))
}

fn conv2d_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    pad_before: usize,
    pad_after: usize,
) -> Result<usize, GgmlCpuGraphError> {
    if input == 0 || kernel == 0 || stride == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr conv2d dimensions must be positive",
        });
    }
    let padded = input
        .checked_add(pad_before)
        .and_then(|value| value.checked_add(pad_after))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr conv2d padded dimension overflows",
        })?;
    if padded < kernel {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr conv2d kernel exceeds padded input",
        });
    }
    Ok((padded - kernel) / stride + 1)
}

fn conv1d_kernel_len(weights: &XasrConvolutionModuleWeights) -> Result<usize, GgmlCpuGraphError> {
    weights
        .depthwise_causal_conv
        .weight
        .dims
        .first()
        .copied()
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution causal kernel must have rank >= 1",
        })
}

fn chunk_conv1d_kernel_len(
    weights: &XasrConvolutionModuleWeights,
) -> Result<usize, GgmlCpuGraphError> {
    weights
        .depthwise_chunkwise_conv
        .weight
        .dims
        .first()
        .copied()
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution chunk kernel must have rank >= 1",
        })
}

fn upload_attention_weights_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensors: XasrSelfAttentionWeightsGraphTensors<'a>,
    weights: &super::encoder_weights::XasrSelfAttentionWeightsWeights,
    shape: XasrLayerHeadGraphShape,
    valid_left_context: usize,
    cached_key: Option<&[f32]>,
) -> Result<(), XasrEncoderGraphError> {
    let query_dim = shape
        .num_heads
        .checked_mul(shape.query_head_dim)
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "xasr stack0 query dimension overflows".to_string(),
        })?;
    upload_cache_or_zero(
        graph,
        tensors.cache,
        shape
            .left_context_len
            .checked_mul(query_dim)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr attention key cache length overflows".to_string(),
            })?,
        cached_key,
        "xasr_stack0_attn_cache",
    )?;
    let mask_values =
        attention_mask_values_for_graph(shape.left_context_len, shape.frames, valid_left_context)?;
    upload_f32(graph, tensors.mask, &mask_values, "xasr_stack0_attn_mask")?;
    let pos_embedding = compact_relative_positional_encoding_for_graph(
        shape.frames,
        shape.left_context_len,
        shape.pos_dim,
    )?;
    upload_f32(
        graph,
        tensors.pos_embedding,
        &pos_embedding,
        "xasr_stack0_attn_pos",
    )?;
    upload_linear_with_bias_tensors(
        graph,
        tensors.in_proj_weight,
        tensors.in_proj_bias,
        &weights.in_proj,
    )?;
    upload_stored_linear_weight_tensor(
        graph,
        tensors.linear_pos_weight,
        &weights.linear_pos,
        "xasr_stack0_attn_pos_w",
    )
}

fn upload_convolution_module_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensors: XasrConvolutionModuleGraphTensors<'a>,
    weights: &XasrConvolutionModuleWeights,
    dim: usize,
    frames: usize,
    cache_len: usize,
    cache: Option<&[f32]>,
) -> Result<(), XasrEncoderGraphError> {
    let expected_cache =
        dim.checked_mul(cache_len)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr convolution cache length overflows".to_string(),
            })?;
    upload_cache_or_zero(
        graph,
        tensors.cache,
        expected_cache,
        cache,
        "xasr_conv_cache",
    )?;
    upload_linear_with_bias_tensors(
        graph,
        tensors.in_proj_weight,
        tensors.in_proj_bias,
        &weights.in_proj,
    )?;
    upload_f32(
        graph,
        tensors.causal_kernel,
        &weights.depthwise_causal_conv.weight.values,
        "xasr_conv_causal_w",
    )?;
    upload_f32(
        graph,
        tensors.causal_bias,
        &weights.depthwise_causal_conv.bias,
        "xasr_conv_causal_b",
    )?;
    upload_f32(
        graph,
        tensors.chunk_kernel,
        &weights.depthwise_chunkwise_conv.weight.values,
        "xasr_conv_chunk_w",
    )?;
    upload_f32(
        graph,
        tensors.chunk_bias,
        &weights.depthwise_chunkwise_conv.bias,
        "xasr_conv_chunk_b",
    )?;
    let chunk_scale = chunkwise_conv_scale_values_for_graph(weights, dim, frames)?;
    upload_f32(
        graph,
        tensors.chunk_scale,
        &chunk_scale,
        "xasr_conv_chunk_scale",
    )?;
    upload_linear_with_bias_tensors(
        graph,
        tensors.out_proj_weight,
        tensors.out_proj_bias,
        &weights.out_proj,
    )?;
    upload_f32(
        graph,
        tensors.swoosh_r_offset,
        &[SWOOSH_R_OFFSET],
        "xasr_conv_swoosh_r_offset",
    )?;
    upload_f32(
        graph,
        tensors.swoosh_r_shift,
        &[SWOOSH_R_SHIFT],
        "xasr_conv_swoosh_r_shift",
    )
}

fn upload_feed_forward_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensors: XasrFeedForwardGraphTensors<'a>,
    weights: &XasrLinearPairWeights,
) -> Result<(), XasrEncoderGraphError> {
    upload_linear_with_bias_tensors(
        graph,
        tensors.in_proj_weight,
        tensors.in_proj_bias,
        &weights.in_proj,
    )?;
    upload_linear_with_bias_tensors(
        graph,
        tensors.out_proj_weight,
        tensors.out_proj_bias,
        &weights.out_proj,
    )?;
    upload_f32(
        graph,
        tensors.swoosh_l_offset,
        &[SWOOSH_L_OFFSET],
        "xasr_ff_swoosh_l_offset",
    )?;
    upload_f32(
        graph,
        tensors.swoosh_l_shift,
        &[SWOOSH_L_SHIFT],
        "xasr_ff_swoosh_l_shift",
    )
}

fn upload_linear_with_bias_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weight_tensor: GgmlCpuTensor<'a>,
    bias_tensor: GgmlCpuTensor<'a>,
    weights: &XasrLinearWithBias,
) -> Result<(), XasrEncoderGraphError> {
    upload_stored_linear_weight_tensor(
        graph,
        weight_tensor,
        &weights.weight,
        "xasr_linear_weight",
    )?;
    upload_f32(graph, bias_tensor, &weights.bias, "xasr_linear_bias")
}

fn upload_conv2d_tensors<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weight_tensor: GgmlCpuTensor<'a>,
    bias_tensor: GgmlCpuTensor<'a>,
    weights: &XasrConv2dWeights,
) -> Result<(), XasrEncoderGraphError> {
    upload_f32(
        graph,
        weight_tensor,
        &weights.weight.values,
        "xasr_conv2d_weight",
    )?;
    upload_f32(graph, bias_tensor, &weights.bias, "xasr_conv2d_bias")
}

fn layer_cache_lengths(
    shape: XasrZipformerLayerGraphShape,
) -> Result<XasrLayerCacheLengths, XasrEncoderGraphError> {
    let head = shape.layer_head;
    let query_dim = head
        .num_heads
        .checked_mul(head.query_head_dim)
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "xasr layer cache query dimension overflows".to_string(),
        })?;
    let checked = |left: usize, right: usize, name: &'static str| {
        left.checked_mul(right)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: format!("xasr layer cache {name} length overflows"),
            })
    };
    Ok(XasrLayerCacheLengths {
        cached_key: checked(head.left_context_len, query_dim, "key")?,
        cached_nonlin_attention: checked(head.left_context_len, head.nonlin_hidden_dim, "nonlin")?,
        cached_val1: checked(head.left_context_len, head.self1_value_dim, "val1")?,
        cached_val2: checked(head.left_context_len, shape.self2_value_dim, "val2")?,
        cached_conv1: checked(head.dim, head.conv1_cache_len, "conv1")?,
        cached_conv2: checked(
            shape.layer_tail.dim,
            shape.layer_tail.conv_cache_len,
            "conv2",
        )?,
    })
}

fn upload_cache_or_zero<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    expected_len: usize,
    values: Option<&[f32]>,
    tensor_name: &'static str,
) -> Result<(), XasrEncoderGraphError> {
    match values {
        Some(values) => {
            if values.len() != expected_len {
                return Err(XasrEncoderGraphError::Shape {
                    reason: format!(
                        "{tensor_name} has {} cache values, expected {expected_len}",
                        values.len()
                    ),
                });
            }
            upload_f32(graph, tensor, values, tensor_name)
        }
        None => upload_f32(graph, tensor, &vec![0.0_f32; expected_len], tensor_name),
    }
}

fn upload_zero_cache<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    rows: usize,
    cols: usize,
    tensor_name: &'static str,
) -> Result<(), XasrEncoderGraphError> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "xasr stack0 cache length overflows".to_string(),
        })?;
    upload_f32(graph, tensor, &vec![0.0_f32; len], tensor_name)
}

fn upload_f32<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    values: &[f32],
    tensor_name: &'static str,
) -> Result<(), XasrEncoderGraphError> {
    map_ggml_stage(
        "stack0_layer_upload",
        graph.set_f32_slice(tensor, values, tensor_name),
    )
}

fn attention_mask_values_for_graph(
    left_context_len: usize,
    frames: usize,
    valid_left_context: usize,
) -> Result<Vec<f32>, XasrEncoderGraphError> {
    let key_padding_mask = streaming_key_padding_mask(left_context_len, frames, valid_left_context)
        .map_err(|reason| XasrEncoderGraphError::Shape { reason })?;
    let k_len =
        left_context_len
            .checked_add(frames)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr stack0 key length overflows".to_string(),
            })?;
    let mut mask_values = vec![
        0.0_f32;
        frames.checked_mul(k_len).ok_or_else(|| {
            XasrEncoderGraphError::Shape {
                reason: "xasr stack0 attention mask length overflows".to_string(),
            }
        })?
    ];
    for target in 0..frames {
        for source in 0..k_len {
            if key_padding_mask[source] {
                mask_values[target * k_len + source] = -1000.0;
            }
        }
    }
    Ok(mask_values)
}

fn compact_relative_positional_encoding_for_graph(
    frames: usize,
    left_context_len: usize,
    embed_dim: usize,
) -> Result<Vec<f32>, XasrEncoderGraphError> {
    if !embed_dim.is_multiple_of(2) {
        return Err(XasrEncoderGraphError::Shape {
            reason: "xasr relative positional encoding dim must be even".to_string(),
        });
    }
    let total_context =
        frames
            .checked_add(left_context_len)
            .ok_or_else(|| XasrEncoderGraphError::Shape {
                reason: "xasr relative positional context overflows".to_string(),
            })?;
    let seq_len = left_context_len
        .checked_add(
            frames
                .checked_mul(2)
                .ok_or_else(|| XasrEncoderGraphError::Shape {
                    reason: "xasr relative positional sequence length overflows".to_string(),
                })?,
        )
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "xasr relative positional sequence length overflows".to_string(),
        })?;
    let compression_length = (embed_dim as f32).sqrt();
    let length_scale = embed_dim as f32 / (2.0 * std::f32::consts::PI);
    let mut output = vec![
        0.0_f32;
        seq_len.checked_mul(embed_dim).ok_or_else(|| {
            XasrEncoderGraphError::Shape {
                reason: "xasr relative positional output length overflows".to_string(),
            }
        })?
    ];
    for row in 0..seq_len {
        let offset = row as isize - (total_context as isize - 1);
        let sign = (offset as f32).signum();
        let abs = (offset as f32).abs();
        let compressed =
            compression_length * sign * ((abs + compression_length).ln() - compression_length.ln());
        let atan = (compressed / length_scale).atan();
        for i in 0..embed_dim / 2 {
            let value = atan * (i + 1) as f32;
            output[row * embed_dim + 2 * i] = value.cos();
            output[row * embed_dim + 2 * i + 1] = value.sin();
        }
        output[row * embed_dim + embed_dim - 1] = 1.0;
    }
    Ok(output)
}

fn chunkwise_conv_scale_values_for_graph(
    weights: &XasrConvolutionModuleWeights,
    channels: usize,
    frames: usize,
) -> Result<Vec<f32>, XasrEncoderGraphError> {
    let mut output = vec![
        0.0_f32;
        channels.checked_mul(frames).ok_or_else(|| {
            XasrEncoderGraphError::Shape {
                reason: "xasr chunkwise conv scale length overflows".to_string(),
            }
        })?
    ];
    for c in 0..channels {
        for t in 0..frames {
            output[c * frames + t] = chunkwise_conv_scale_for_graph(weights, c, t, frames)?;
        }
    }
    Ok(output)
}

fn chunkwise_conv_scale_for_graph(
    weights: &XasrConvolutionModuleWeights,
    channel: usize,
    frame: usize,
    chunk_size: usize,
) -> Result<f32, XasrEncoderGraphError> {
    let dims = &weights.chunkwise_conv_scale.dims;
    let chunk_dims = &weights.depthwise_chunkwise_conv.weight.dims;
    if chunk_dims.len() < 3 {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "xasr depthwise_chunkwise_conv has rank {}, expected at least 3",
                chunk_dims.len()
            ),
        });
    }
    let expected = [2, chunk_dims[2], chunk_dims[0]];
    if dims != &expected {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "xasr chunkwise_conv_scale has dims {:?}, expected {:?}",
                dims, expected
            ),
        });
    }
    let channels = dims[1];
    let kernel = dims[2];
    if channel >= channels {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!("xasr chunkwise conv channel {channel} exceeds {channels}"),
        });
    }
    let values = &weights.chunkwise_conv_scale.values;
    let expected_values = 2usize
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(kernel))
        .ok_or_else(|| XasrEncoderGraphError::Shape {
            reason: "xasr chunkwise conv scale value count overflows".to_string(),
        })?;
    if values.len() != expected_values {
        return Err(XasrEncoderGraphError::Shape {
            reason: format!(
                "xasr chunkwise_conv_scale has {} values, expected {expected_values}",
                values.len()
            ),
        });
    }
    let left = if frame < kernel {
        values[channel * kernel + frame]
    } else {
        0.0
    };
    let right_base = channels * kernel;
    let right = if chunk_size < kernel {
        values[right_base + channel * kernel + kernel - chunk_size + frame]
    } else {
        let pad = chunk_size - kernel;
        if frame >= pad {
            values[right_base + channel * kernel + frame - pad]
        } else {
            0.0
        }
    };
    Ok(1.0 + left + right)
}

fn apply_swoosh_r_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    offset: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    apply_swoosh_graph(graph, input, offset, shift)
}

fn apply_swoosh_l_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    offset: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    apply_swoosh_graph(graph, input, offset, shift)
}

fn apply_swoosh_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    offset: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let shifted = graph.sub(input, offset)?;
    let softplus = graph.softplus(shifted)?;
    let linear = graph.scale(input, SWOOSH_LINEAR_SCALE)?;
    let without_linear = graph.sub(softplus, linear)?;
    graph.sub(without_linear, shift)
}

fn apply_bias_norm_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    log_scale: f32,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if !log_scale.is_finite() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr BiasNorm log_scale must be finite",
        });
    }
    let centered = graph.sub(input, bias)?;
    let mean_square = graph.mean_rows(graph.sqr(centered)?)?;
    let denom = graph.sqrt(mean_square)?;
    let normalized = graph.div(input, denom)?;
    graph.scale(normalized, log_scale.exp())
}

#[derive(Debug, Clone, Copy)]
struct XasrEncoderEmbedGraphTensors<'a> {
    conv0_weight: GgmlCpuTensor<'a>,
    conv0_bias: GgmlCpuTensor<'a>,
    conv4_weight: GgmlCpuTensor<'a>,
    conv4_bias: GgmlCpuTensor<'a>,
    conv7_weight: GgmlCpuTensor<'a>,
    conv7_bias: GgmlCpuTensor<'a>,
    embed_cache: GgmlCpuTensor<'a>,
    convnext_depthwise_weight: GgmlCpuTensor<'a>,
    convnext_depthwise_bias: GgmlCpuTensor<'a>,
    convnext_pointwise1_weight: GgmlCpuTensor<'a>,
    convnext_pointwise1_bias: GgmlCpuTensor<'a>,
    convnext_pointwise2_weight: GgmlCpuTensor<'a>,
    convnext_pointwise2_bias: GgmlCpuTensor<'a>,
    out_weight: GgmlCpuTensor<'a>,
    out_bias: GgmlCpuTensor<'a>,
    out_norm_bias: GgmlCpuTensor<'a>,
    swoosh_r_offset: GgmlCpuTensor<'a>,
    swoosh_r_shift: GgmlCpuTensor<'a>,
    swoosh_l_offset: GgmlCpuTensor<'a>,
    swoosh_l_shift: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrEncoderEmbedGraphShape {
    input_frames: usize,
    feature_dim: usize,
    embed_width: usize,
    subsampled_frames: usize,
    embed_frames: usize,
    cache_frames: usize,
    channels: usize,
    output_dim: usize,
    out_norm_log_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct XasrEncoderEmbedGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_embed_states: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrDownsampleGraphTensors<'a> {
    channel_pad: Option<GgmlCpuTensor<'a>>,
    frame_pad: Option<GgmlCpuTensor<'a>>,
}

#[derive(Debug, Clone, Copy)]
struct XasrDownsampleGraphShape {
    frames: usize,
    input_dim: usize,
    target_dim: usize,
    factor: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrDownsampleGraphOutput<'a> {
    padded_rows: GgmlCpuTensor<'a>,
    rows: GgmlCpuTensor<'a>,
    frames: usize,
    dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrUpsampleGraphShape {
    frames: usize,
    dim: usize,
    factor: usize,
    target_frames: usize,
}

fn apply_upsample_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    rows: GgmlCpuTensor<'a>,
    shape: XasrUpsampleGraphShape,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if shape.frames == 0 || shape.dim == 0 || shape.factor == 0 || shape.target_frames == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr upsample dimensions must be positive",
        });
    }
    let frame_stride = shape.dim.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr upsample frame stride overflows",
        },
    )?;
    let mut acc = None;
    for out_frame in 0..shape.target_frames {
        let src_frame = out_frame / shape.factor;
        if src_frame >= shape.frames {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr upsample target frame exceeds available source frames",
            });
        }
        let offset =
            src_frame
                .checked_mul(frame_stride)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr upsample frame offset overflows",
                })?;
        let frame = graph.view_2d(rows, shape.dim, 1, frame_stride, offset)?;
        let frame = graph.cont(frame)?;
        acc = Some(match acc {
            Some(current) => graph.concat(current, frame, 1)?,
            None => frame,
        });
    }
    acc.ok_or(GgmlCpuGraphError::UnsupportedInputs {
        reason: "xasr upsample target frames must not be empty",
    })
}

fn apply_downsample_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    rows: GgmlCpuTensor<'a>,
    weights_logits: &[f32],
    tensors: XasrDownsampleGraphTensors<'a>,
    shape: XasrDownsampleGraphShape,
) -> Result<XasrDownsampleGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.frames == 0 || shape.input_dim == 0 || shape.target_dim == 0 || shape.factor == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr downsample dimensions must be positive",
        });
    }
    if weights_logits.len() != shape.factor {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr downsample weights length must match factor",
        });
    }
    let out_frames = shape.frames.div_ceil(shape.factor);
    let padded_frames =
        out_frames
            .checked_mul(shape.factor)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample padded frame count overflows",
            })?;
    let resized = if shape.target_dim > shape.input_dim {
        let pad = tensors
            .channel_pad
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample channel pad tensor is required",
            })?;
        graph.concat(rows, pad, 0)?
    } else if shape.target_dim < shape.input_dim {
        let row_stride = shape
            .input_dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample resize row stride overflows",
            })?;
        let sliced = graph.view_2d(rows, shape.target_dim, shape.frames, row_stride, 0)?;
        graph.cont(sliced)?
    } else {
        rows
    };
    let padded_rows = if padded_frames > shape.frames {
        let pad = tensors
            .frame_pad
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample frame pad tensor is required",
            })?;
        graph.concat(resized, pad, 1)?
    } else {
        graph.cont(resized)?
    };
    let weights = softmax_for_graph(weights_logits)?;
    let frame_stride = shape
        .target_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr downsample frame stride overflows",
        })?;
    let grouped_frame_stride =
        frame_stride
            .checked_mul(shape.factor)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr downsample grouped frame stride overflows",
            })?;
    let mut acc = None;
    for (factor_index, &weight) in weights.iter().enumerate() {
        let offset =
            factor_index
                .checked_mul(frame_stride)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr downsample factor offset overflows",
                })?;
        let selected = graph.view_2d(
            padded_rows,
            shape.target_dim,
            out_frames,
            grouped_frame_stride,
            offset,
        )?;
        let selected = graph.cont(selected)?;
        let weighted = graph.scale(selected, weight)?;
        acc = Some(match acc {
            Some(current) => graph.add(current, weighted)?,
            None => weighted,
        });
    }
    Ok(XasrDownsampleGraphOutput {
        padded_rows,
        rows: acc.ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr downsample weights must not be empty",
        })?,
        frames: out_frames,
        dim: shape.target_dim,
    })
}

fn softmax_for_graph(values: &[f32]) -> Result<Vec<f32>, GgmlCpuGraphError> {
    if values.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr softmax input must not be empty",
        });
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0_f32;
    let mut output = Vec::with_capacity(values.len());
    for &value in values {
        let exp = (value - max).exp();
        denom += exp;
        output.push(exp);
    }
    if !(denom.is_finite() && denom > 0.0) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr softmax denominator must be finite and positive",
        });
    }
    for value in &mut output {
        *value /= denom;
    }
    Ok(output)
}

fn apply_encoder_embed_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    features: GgmlCpuTensor<'a>,
    tensors: XasrEncoderEmbedGraphTensors<'a>,
    shape: XasrEncoderEmbedGraphShape,
) -> Result<XasrEncoderEmbedGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.feature_dim == 0
        || shape.input_frames == 0
        || shape.embed_width == 0
        || shape.subsampled_frames == 0
        || shape.embed_frames == 0
        || shape.cache_frames == 0
        || shape.channels == 0
        || shape.output_dim == 0
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed dimensions must be positive",
        });
    }
    if shape.subsampled_frames != shape.embed_frames + shape.cache_frames {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed subsampled_frames must equal embed_frames + cache_frames",
        });
    }

    let input = graph.reshape_4d(features, shape.feature_dim, shape.input_frames, 1, 1)?;
    let mut state = apply_conv2d_with_bias_graph(
        graph,
        tensors.conv0_weight,
        input,
        tensors.conv0_bias,
        8,
        XasrConv2dGraphParams {
            stride_w: 1,
            stride_h: 1,
            pad_w: 1,
            pad_h: 0,
        },
    )?;
    state = apply_swoosh_r_graph(
        graph,
        state,
        tensors.swoosh_r_offset,
        tensors.swoosh_r_shift,
    )?;
    state = apply_conv2d_with_bias_graph(
        graph,
        tensors.conv4_weight,
        state,
        tensors.conv4_bias,
        32,
        XasrConv2dGraphParams {
            stride_w: 2,
            stride_h: 2,
            pad_w: 0,
            pad_h: 0,
        },
    )?;
    state = apply_swoosh_r_graph(
        graph,
        state,
        tensors.swoosh_r_offset,
        tensors.swoosh_r_shift,
    )?;
    state = apply_conv2d_with_bias_graph(
        graph,
        tensors.conv7_weight,
        state,
        tensors.conv7_bias,
        shape.channels,
        XasrConv2dGraphParams {
            stride_w: 2,
            stride_h: 1,
            pad_w: 0,
            pad_h: 0,
        },
    )?;
    state = apply_swoosh_r_graph(
        graph,
        state,
        tensors.swoosh_r_offset,
        tensors.swoosh_r_shift,
    )?;
    let state = graph.cont(state)?;

    let element_size = std::mem::size_of::<f32>();
    let state_nb1 = shape.embed_width.checked_mul(element_size).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed state row stride overflows",
        },
    )?;
    let state_nb2 = shape.subsampled_frames.checked_mul(state_nb1).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed state plane stride overflows",
        },
    )?;
    let state_nb3 =
        shape
            .channels
            .checked_mul(state_nb2)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr encoder embed state hyperplane stride overflows",
            })?;
    let residual = graph.view_4d(
        state,
        shape.embed_width,
        shape.embed_frames,
        shape.channels,
        1,
        state_nb1,
        state_nb2,
        state_nb3,
        0,
    )?;
    let residual = graph.cont(residual)?;
    let concat = graph.concat(tensors.embed_cache, state, 1)?;
    let concat_h = shape
        .cache_frames
        .checked_add(shape.subsampled_frames)
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed concat height overflows",
        })?;
    let concat_nb2 =
        concat_h
            .checked_mul(state_nb1)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr encoder embed concat plane stride overflows",
            })?;
    let concat_nb3 =
        shape
            .channels
            .checked_mul(concat_nb2)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr encoder embed concat hyperplane stride overflows",
            })?;
    let cache_offset = shape
        .subsampled_frames
        .checked_sub(shape.cache_frames)
        .and_then(|height| height.checked_mul(state_nb1))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr encoder embed cache offset overflows",
        })?;
    let new_embed_states = graph.view_4d(
        concat,
        shape.embed_width,
        shape.cache_frames,
        shape.channels,
        1,
        state_nb1,
        concat_nb2,
        concat_nb3,
        cache_offset,
    )?;
    let new_embed_states = graph.cont(new_embed_states)?;

    let mut convnext = apply_depthwise_conv2d_with_bias_graph(
        graph,
        tensors.convnext_depthwise_weight,
        concat,
        tensors.convnext_depthwise_bias,
        shape.channels,
        XasrConv2dGraphParams {
            stride_w: 1,
            stride_h: 1,
            pad_w: 3,
            pad_h: 0,
        },
    )?;
    convnext = apply_conv2d_with_bias_graph(
        graph,
        tensors.convnext_pointwise1_weight,
        convnext,
        tensors.convnext_pointwise1_bias,
        shape.channels * 3,
        XasrConv2dGraphParams {
            stride_w: 1,
            stride_h: 1,
            pad_w: 0,
            pad_h: 0,
        },
    )?;
    convnext = apply_swoosh_l_graph(
        graph,
        convnext,
        tensors.swoosh_l_offset,
        tensors.swoosh_l_shift,
    )?;
    convnext = apply_conv2d_with_bias_graph(
        graph,
        tensors.convnext_pointwise2_weight,
        convnext,
        tensors.convnext_pointwise2_bias,
        shape.channels,
        XasrConv2dGraphParams {
            stride_w: 1,
            stride_h: 1,
            pad_w: 0,
            pad_h: 0,
        },
    )?;
    let state = graph.add(residual, convnext)?;
    let state = graph.permute(state, 0, 2, 1, 3)?;
    let state = graph.cont(state)?;
    let projection_input = graph.reshape_2d(
        state,
        shape.embed_width * shape.channels,
        shape.embed_frames,
    )?;
    let rows = apply_linear_graph(
        graph,
        tensors.out_weight,
        projection_input,
        tensors.out_bias,
    )?;
    let rows = apply_bias_norm_graph(graph, rows, tensors.out_norm_bias, shape.out_norm_log_scale)?;
    Ok(XasrEncoderEmbedGraphOutput {
        rows,
        new_embed_states,
    })
}

#[derive(Debug, Clone, Copy)]
struct XasrConv2dGraphParams {
    stride_w: usize,
    stride_h: usize,
    pad_w: usize,
    pad_h: usize,
}

fn apply_conv2d_with_bias_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    out_channels: usize,
    params: XasrConv2dGraphParams,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let output = graph.conv_2d(
        kernel,
        input,
        params.stride_w,
        params.stride_h,
        params.pad_w,
        params.pad_h,
        1,
        1,
    )?;
    let bias = graph.reshape_4d(bias, 1, 1, out_channels, 1)?;
    graph.add(output, bias)
}

fn apply_depthwise_conv2d_with_bias_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    channels: usize,
    params: XasrConv2dGraphParams,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let output = graph.depthwise_conv_2d(
        kernel,
        input,
        params.stride_w,
        params.stride_h,
        params.pad_w,
        params.pad_h,
        1,
        1,
    )?;
    let bias = graph.reshape_4d(bias, 1, 1, channels, 1)?;
    graph.add(output, bias)
}

#[derive(Debug, Clone, Copy)]
struct XasrFeedForwardGraphTensors<'a> {
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    out_proj_weight: GgmlCpuTensor<'a>,
    out_proj_bias: GgmlCpuTensor<'a>,
    swoosh_l_offset: GgmlCpuTensor<'a>,
    swoosh_l_shift: GgmlCpuTensor<'a>,
}

fn apply_feed_forward_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weights: XasrFeedForwardGraphTensors<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let hidden = apply_linear_graph(graph, weights.in_proj_weight, input, weights.in_proj_bias)?;
    let hidden = apply_swoosh_l_graph(
        graph,
        hidden,
        weights.swoosh_l_offset,
        weights.swoosh_l_shift,
    )?;
    apply_linear_graph(
        graph,
        weights.out_proj_weight,
        hidden,
        weights.out_proj_bias,
    )
}

fn apply_conv_in_proj_glu_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    dim: usize,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if dim == 0 || frames == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution GLU dim and frames must be positive",
        });
    }
    let projected = apply_linear_graph(graph, in_proj_weight, input, in_proj_bias)?;
    let row_stride_bytes = dim
        .checked_mul(2)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution GLU row stride overflows",
        })?;
    let gate_offset_bytes = dim.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution GLU gate offset overflows",
        },
    )?;
    let main = graph.view_2d(projected, dim, frames, row_stride_bytes, 0)?;
    let main = graph.cont(main)?;
    let gate = graph.view_2d(projected, dim, frames, row_stride_bytes, gate_offset_bytes)?;
    let gate = graph.cont(gate)?;
    let gate = graph.sigmoid(gate)?;
    graph.mul(main, gate)
}

fn apply_depthwise_conv1d_channel_major_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    channels: usize,
    input_len: usize,
    kernel_len: usize,
    padding: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if channels == 0 || input_len == 0 || kernel_len == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr depthwise conv dimensions must be positive",
        });
    }
    let input = graph.reshape_4d(input, input_len, 1, channels, 1)?;
    let kernel = graph.reshape_4d(kernel, kernel_len, 1, 1, channels)?;
    let output = graph.depthwise_conv_2d(kernel, input, 1, 1, padding, 0, 1, 1)?;
    let bias = graph.reshape_4d(bias, 1, 1, channels, 1)?;
    graph.add(output, bias)
}

#[derive(Debug, Clone, Copy)]
struct XasrDepthwiseMixGraphTensors<'a> {
    cached_input: GgmlCpuTensor<'a>,
    chunk_input: GgmlCpuTensor<'a>,
    causal_kernel: GgmlCpuTensor<'a>,
    causal_bias: GgmlCpuTensor<'a>,
    chunk_kernel: GgmlCpuTensor<'a>,
    chunk_bias: GgmlCpuTensor<'a>,
    chunk_scale: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrDepthwiseMixShape {
    channels: usize,
    frames: usize,
    cached_len: usize,
    causal_kernel_len: usize,
    chunk_kernel_len: usize,
}

fn apply_depthwise_mix_channel_major_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    tensors: XasrDepthwiseMixGraphTensors<'a>,
    shape: XasrDepthwiseMixShape,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let causal = apply_depthwise_conv1d_channel_major_graph(
        graph,
        tensors.cached_input,
        tensors.causal_kernel,
        tensors.causal_bias,
        shape.channels,
        shape.cached_len,
        shape.causal_kernel_len,
        0,
    )?;
    let chunk = apply_depthwise_conv1d_channel_major_graph(
        graph,
        tensors.chunk_input,
        tensors.chunk_kernel,
        tensors.chunk_bias,
        shape.channels,
        shape.frames,
        shape.chunk_kernel_len,
        shape.chunk_kernel_len / 2,
    )?;
    let scale = graph.reshape_4d(tensors.chunk_scale, shape.frames, 1, shape.channels, 1)?;
    let scaled_chunk = graph.mul(chunk, scale)?;
    graph.add(causal, scaled_chunk)
}

#[derive(Debug, Clone, Copy)]
struct XasrConvolutionModuleGraphTensors<'a> {
    cache: GgmlCpuTensor<'a>,
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    causal_kernel: GgmlCpuTensor<'a>,
    causal_bias: GgmlCpuTensor<'a>,
    chunk_kernel: GgmlCpuTensor<'a>,
    chunk_bias: GgmlCpuTensor<'a>,
    chunk_scale: GgmlCpuTensor<'a>,
    out_proj_weight: GgmlCpuTensor<'a>,
    out_proj_bias: GgmlCpuTensor<'a>,
    swoosh_r_offset: GgmlCpuTensor<'a>,
    swoosh_r_shift: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrConvolutionModuleGraphShape {
    dim: usize,
    frames: usize,
    cache_len: usize,
    causal_kernel_len: usize,
    chunk_kernel_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrConvolutionModuleGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_cache: GgmlCpuTensor<'a>,
}

fn apply_convolution_module_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrConvolutionModuleGraphTensors<'a>,
    shape: XasrConvolutionModuleGraphShape,
) -> Result<XasrConvolutionModuleGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.dim == 0 || shape.frames == 0 || shape.cache_len == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution module dimensions must be positive",
        });
    }
    if shape.cache_len.checked_add(1) != Some(shape.causal_kernel_len) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution module cache_len must equal causal_kernel_len - 1",
        });
    }
    if shape.cache_len != shape.chunk_kernel_len / 2 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution module cache_len must equal chunk_kernel_len / 2",
        });
    }

    let gated = apply_conv_in_proj_glu_graph(
        graph,
        input,
        tensors.in_proj_weight,
        tensors.in_proj_bias,
        shape.dim,
        shape.frames,
    )?;
    let channel_major = graph.transpose(gated)?;
    let channel_major = graph.cont(channel_major)?;
    let cached_input = graph.concat(tensors.cache, channel_major, 0)?;

    let cached_len =
        shape
            .cache_len
            .checked_add(shape.frames)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr convolution module cached length overflows",
            })?;
    let row_stride_bytes = cached_len.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution module cache row stride overflows",
        },
    )?;
    let new_cache_offset_bytes = shape.frames.checked_mul(std::mem::size_of::<f32>()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr convolution module cache offset overflows",
        },
    )?;
    let new_cache = graph.view_2d(
        cached_input,
        shape.cache_len,
        shape.dim,
        row_stride_bytes,
        new_cache_offset_bytes,
    )?;
    let new_cache = graph.cont(new_cache)?;

    let depthwise = apply_depthwise_mix_channel_major_graph(
        graph,
        XasrDepthwiseMixGraphTensors {
            cached_input,
            chunk_input: channel_major,
            causal_kernel: tensors.causal_kernel,
            causal_bias: tensors.causal_bias,
            chunk_kernel: tensors.chunk_kernel,
            chunk_bias: tensors.chunk_bias,
            chunk_scale: tensors.chunk_scale,
        },
        XasrDepthwiseMixShape {
            channels: shape.dim,
            frames: shape.frames,
            cached_len,
            causal_kernel_len: shape.causal_kernel_len,
            chunk_kernel_len: shape.chunk_kernel_len,
        },
    )?;
    let depthwise = graph.cont(depthwise)?;
    let depthwise = graph.reshape_2d(depthwise, shape.frames, shape.dim)?;
    let frame_major = graph.transpose(depthwise)?;
    let frame_major = graph.cont(frame_major)?;
    let activated = apply_swoosh_r_graph(
        graph,
        frame_major,
        tensors.swoosh_r_offset,
        tensors.swoosh_r_shift,
    )?;
    let rows = apply_linear_graph(
        graph,
        tensors.out_proj_weight,
        activated,
        tensors.out_proj_bias,
    )?;
    Ok(XasrConvolutionModuleGraphOutput { rows, new_cache })
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerTailGraphTensors<'a> {
    conv_module2: XasrConvolutionModuleGraphTensors<'a>,
    feed_forward3: XasrFeedForwardGraphTensors<'a>,
    norm_bias: GgmlCpuTensor<'a>,
    bypass_scale: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerTailGraphShape {
    dim: usize,
    frames: usize,
    conv_cache_len: usize,
    conv_causal_kernel_len: usize,
    conv_chunk_kernel_len: usize,
    norm_log_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerTailGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_conv2_cache: GgmlCpuTensor<'a>,
}

fn apply_layer_tail_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    original_rows: GgmlCpuTensor<'a>,
    add18_rows: GgmlCpuTensor<'a>,
    tensors: XasrLayerTailGraphTensors<'a>,
    shape: XasrLayerTailGraphShape,
) -> Result<XasrLayerTailGraphOutput<'a>, GgmlCpuGraphError> {
    let conv2 = apply_convolution_module_graph(
        graph,
        add18_rows,
        tensors.conv_module2,
        XasrConvolutionModuleGraphShape {
            dim: shape.dim,
            frames: shape.frames,
            cache_len: shape.conv_cache_len,
            causal_kernel_len: shape.conv_causal_kernel_len,
            chunk_kernel_len: shape.conv_chunk_kernel_len,
        },
    )?;
    let add23 = graph.add(add18_rows, conv2.rows)?;
    let ff3 = apply_feed_forward_graph(graph, add23, tensors.feed_forward3)?;
    let add24 = graph.add(add23, ff3)?;
    let norm = apply_bias_norm_graph(graph, add24, tensors.norm_bias, shape.norm_log_scale)?;
    let rows = apply_bypass_graph(graph, original_rows, norm, tensors.bypass_scale)?;
    Ok(XasrLayerTailGraphOutput {
        rows,
        new_conv2_cache: conv2.new_cache,
    })
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionGraphTensors<'a> {
    cache: GgmlCpuTensor<'a>,
    attention_weights: GgmlCpuTensor<'a>,
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    out_proj_weight: GgmlCpuTensor<'a>,
    out_proj_bias: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionGraphShape {
    dim: usize,
    frames: usize,
    left_context_len: usize,
    num_heads: usize,
    value_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_cache: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrAttentionWeightedValuesGraphShape {
    frames: usize,
    left_context_len: usize,
    num_heads: usize,
    value_dim: usize,
}

fn apply_attention_weighted_values_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    values: GgmlCpuTensor<'a>,
    attention_weights: GgmlCpuTensor<'a>,
    shape: XasrAttentionWeightedValuesGraphShape,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if shape.frames == 0
        || shape.left_context_len == 0
        || shape.num_heads == 0
        || shape.value_dim == 0
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr attention weighted values dimensions must be positive",
        });
    }
    if !shape.value_dim.is_multiple_of(shape.num_heads) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr attention weighted values value_dim must be divisible by num_heads",
        });
    }
    let k_len = shape.left_context_len.checked_add(shape.frames).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr attention weighted values key length overflows",
        },
    )?;
    let value_head_dim = shape.value_dim / shape.num_heads;
    let element_size = std::mem::size_of::<f32>();
    let value_row_stride =
        shape
            .value_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr attention weighted values row stride overflows",
            })?;
    let attn_head_stride =
        k_len
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr attention weighted values weight row stride overflows",
            })?;
    let attn_head_bytes = shape
        .frames
        .checked_mul(k_len)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr attention weighted values weight head span overflows",
        })?;
    let mut head_outputs = Vec::with_capacity(shape.num_heads);
    for head in 0..shape.num_heads {
        let value_offset = head
            .checked_mul(value_head_dim)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr attention weighted values value head offset overflows",
            })?;
        let values_head = graph.view_2d(
            values,
            value_head_dim,
            k_len,
            value_row_stride,
            value_offset,
        )?;
        let values_head = graph.cont(values_head)?;
        let values_head = graph.transpose(values_head)?;
        let values_head = graph.cont(values_head)?;
        let attn_offset =
            head.checked_mul(attn_head_bytes)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr attention weighted values weight head offset overflows",
                })?;
        let attn_head = graph.view_2d(
            attention_weights,
            k_len,
            shape.frames,
            attn_head_stride,
            attn_offset,
        )?;
        let attended = graph.mul_mat(values_head, attn_head)?;
        head_outputs.push(graph.cont(attended)?);
    }

    let mut combined = head_outputs[0];
    for &next in &head_outputs[1..] {
        combined = graph.concat(combined, next, 0)?;
    }
    Ok(combined)
}

fn apply_self_attention_value_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrSelfAttentionGraphTensors<'a>,
    shape: XasrSelfAttentionGraphShape,
) -> Result<XasrSelfAttentionGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.dim == 0
        || shape.frames == 0
        || shape.left_context_len == 0
        || shape.num_heads == 0
        || shape.value_dim == 0
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention dimensions must be positive",
        });
    }
    if !shape.value_dim.is_multiple_of(shape.num_heads) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention value_dim must be divisible by num_heads",
        });
    }
    let element_size = std::mem::size_of::<f32>();

    let current = apply_linear_graph(graph, tensors.in_proj_weight, input, tensors.in_proj_bias)?;
    let values = graph.concat(tensors.cache, current, 1)?;
    let new_cache_offset = shape
        .frames
        .checked_mul(shape.value_dim)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention cache offset overflows",
        })?;
    let value_row_stride =
        shape
            .value_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention value row stride overflows",
            })?;
    let new_cache = graph.view_2d(
        values,
        shape.value_dim,
        shape.left_context_len,
        value_row_stride,
        new_cache_offset,
    )?;
    let new_cache = graph.cont(new_cache)?;

    let combined = apply_attention_weighted_values_graph(
        graph,
        values,
        tensors.attention_weights,
        XasrAttentionWeightedValuesGraphShape {
            frames: shape.frames,
            left_context_len: shape.left_context_len,
            num_heads: shape.num_heads,
            value_dim: shape.value_dim,
        },
    )?;
    let rows = apply_linear_graph(
        graph,
        tensors.out_proj_weight,
        combined,
        tensors.out_proj_bias,
    )?;
    Ok(XasrSelfAttentionGraphOutput { rows, new_cache })
}

#[derive(Debug, Clone, Copy)]
struct XasrNonlinAttentionGraphTensors<'a> {
    cache: GgmlCpuTensor<'a>,
    attention_weights: GgmlCpuTensor<'a>,
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    out_proj_weight: GgmlCpuTensor<'a>,
    out_proj_bias: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrNonlinAttentionGraphShape {
    dim: usize,
    frames: usize,
    left_context_len: usize,
    num_heads: usize,
    hidden_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrNonlinAttentionGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_cache: GgmlCpuTensor<'a>,
}

fn apply_nonlin_attention_value_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrNonlinAttentionGraphTensors<'a>,
    shape: XasrNonlinAttentionGraphShape,
) -> Result<XasrNonlinAttentionGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.dim == 0
        || shape.frames == 0
        || shape.left_context_len == 0
        || shape.num_heads == 0
        || shape.hidden_dim == 0
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr nonlin attention dimensions must be positive",
        });
    }
    if !shape.hidden_dim.is_multiple_of(shape.num_heads) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr nonlin attention hidden_dim must be divisible by num_heads",
        });
    }
    let element_size = std::mem::size_of::<f32>();
    let projected_dim =
        shape
            .hidden_dim
            .checked_mul(3)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr nonlin attention projected dim overflows",
            })?;
    let projected_row_stride =
        projected_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr nonlin attention projected row stride overflows",
            })?;

    let projected = apply_linear_graph(graph, tensors.in_proj_weight, input, tensors.in_proj_bias)?;
    let gate = graph.view_2d(
        projected,
        shape.hidden_dim,
        shape.frames,
        projected_row_stride,
        0,
    )?;
    let gate = graph.cont(gate)?;
    let gate = graph.tanh(gate)?;
    let x_offset =
        shape
            .hidden_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr nonlin attention x offset overflows",
            })?;
    let x = graph.view_2d(
        projected,
        shape.hidden_dim,
        shape.frames,
        projected_row_stride,
        x_offset,
    )?;
    let x = graph.cont(x)?;
    let x = graph.mul(x, gate)?;
    let y_offset = x_offset
        .checked_mul(2)
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr nonlin attention y offset overflows",
        })?;
    let y = graph.view_2d(
        projected,
        shape.hidden_dim,
        shape.frames,
        projected_row_stride,
        y_offset,
    )?;
    let y = graph.cont(y)?;

    let values = graph.concat(tensors.cache, x, 1)?;
    let value_row_stride =
        shape
            .hidden_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr nonlin attention value row stride overflows",
            })?;
    let new_cache_offset = shape
        .frames
        .checked_mul(shape.hidden_dim)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr nonlin attention cache offset overflows",
        })?;
    let new_cache = graph.view_2d(
        values,
        shape.hidden_dim,
        shape.left_context_len,
        value_row_stride,
        new_cache_offset,
    )?;
    let new_cache = graph.cont(new_cache)?;
    let attended = apply_attention_weighted_values_graph(
        graph,
        values,
        tensors.attention_weights,
        XasrAttentionWeightedValuesGraphShape {
            frames: shape.frames,
            left_context_len: shape.left_context_len,
            num_heads: shape.num_heads,
            value_dim: shape.hidden_dim,
        },
    )?;
    let attended = graph.mul(attended, y)?;
    let rows = apply_linear_graph(
        graph,
        tensors.out_proj_weight,
        attended,
        tensors.out_proj_bias,
    )?;
    Ok(XasrNonlinAttentionGraphOutput { rows, new_cache })
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionWeightsGraphTensors<'a> {
    cache: GgmlCpuTensor<'a>,
    mask: GgmlCpuTensor<'a>,
    pos_embedding: GgmlCpuTensor<'a>,
    in_proj_weight: GgmlCpuTensor<'a>,
    in_proj_bias: GgmlCpuTensor<'a>,
    linear_pos_weight: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionWeightsGraphShape {
    dim: usize,
    frames: usize,
    left_context_len: usize,
    num_heads: usize,
    query_head_dim: usize,
    pos_dim: usize,
    pos_output_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrSelfAttentionWeightsGraphOutput<'a> {
    weights: GgmlCpuTensor<'a>,
    new_cache: GgmlCpuTensor<'a>,
}

fn apply_self_attention_weights_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrSelfAttentionWeightsGraphTensors<'a>,
    shape: XasrSelfAttentionWeightsGraphShape,
) -> Result<XasrSelfAttentionWeightsGraphOutput<'a>, GgmlCpuGraphError> {
    if shape.dim == 0
        || shape.frames == 0
        || shape.left_context_len == 0
        || shape.num_heads == 0
        || shape.query_head_dim == 0
        || shape.pos_dim == 0
        || shape.pos_output_dim == 0
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention weights dimensions must be positive",
        });
    }
    if !shape.pos_output_dim.is_multiple_of(shape.num_heads) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention pos output dim must be divisible by num_heads",
        });
    }
    let query_dim = shape.num_heads.checked_mul(shape.query_head_dim).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention query dim overflows",
        },
    )?;
    let projected_dim = query_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(shape.pos_output_dim))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention projected dim overflows",
        })?;
    let k_len = shape.left_context_len.checked_add(shape.frames).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention key length overflows",
        },
    )?;
    let rel_len =
        shape
            .left_context_len
            .checked_add(shape.frames.checked_mul(2).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr self attention relative length overflows",
                },
            )?)
            .and_then(|value| value.checked_sub(1))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention relative length overflows",
            })?;
    let pos_head_dim = shape.pos_output_dim / shape.num_heads;
    let element_size = std::mem::size_of::<f32>();
    let projected_row_stride =
        projected_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention projected row stride overflows",
            })?;
    let query_row_stride =
        query_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention query row stride overflows",
            })?;
    let pos_row_stride = shape.pos_output_dim.checked_mul(element_size).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention pos row stride overflows",
        },
    )?;

    let projected = apply_linear_graph(graph, tensors.in_proj_weight, input, tensors.in_proj_bias)?;
    let q = graph.view_2d(projected, query_dim, shape.frames, projected_row_stride, 0)?;
    let q = graph.cont(q)?;
    let current_k_offset =
        query_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention key offset overflows",
            })?;
    let current_k = graph.view_2d(
        projected,
        query_dim,
        shape.frames,
        projected_row_stride,
        current_k_offset,
    )?;
    let current_k = graph.cont(current_k)?;
    let p_offset = current_k_offset
        .checked_mul(2)
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention positional projection offset overflows",
        })?;
    let p = graph.view_2d(
        projected,
        shape.pos_output_dim,
        shape.frames,
        projected_row_stride,
        p_offset,
    )?;
    let p = graph.cont(p)?;

    let all_keys = graph.concat(tensors.cache, current_k, 1)?;
    let new_cache_offset = shape
        .frames
        .checked_mul(query_dim)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "xasr self attention new cache offset overflows",
        })?;
    let new_cache = graph.view_2d(
        all_keys,
        query_dim,
        shape.left_context_len,
        query_row_stride,
        new_cache_offset,
    )?;
    let new_cache = graph.cont(new_cache)?;

    let projected_pos = graph.mul_mat(tensors.linear_pos_weight, tensors.pos_embedding)?;
    let projected_pos = graph.cont(projected_pos)?;
    let pos_head_span_bytes =
        pos_head_dim
            .checked_mul(element_size)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention pos head span overflows",
            })?;

    let mut head_outputs = Vec::with_capacity(shape.num_heads);
    for head in 0..shape.num_heads {
        let query_head_offset = head
            .checked_mul(shape.query_head_dim)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention query head offset overflows",
            })?;
        let q_head = graph.view_2d(
            q,
            shape.query_head_dim,
            shape.frames,
            query_row_stride,
            query_head_offset,
        )?;
        let q_head = graph.cont(q_head)?;
        let k_head = graph.view_2d(
            all_keys,
            shape.query_head_dim,
            k_len,
            query_row_stride,
            query_head_offset,
        )?;
        let k_head = graph.cont(k_head)?;
        let qk_scores = graph.mul_mat(k_head, q_head)?;

        let pos_head_offset = head
            .checked_mul(pos_head_dim)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "xasr self attention pos head offset overflows",
            })?;
        let p_head = graph.view_2d(
            p,
            pos_head_dim,
            shape.frames,
            pos_row_stride,
            pos_head_offset,
        )?;
        let p_head = graph.cont(p_head)?;

        let mut pos_columns = Vec::with_capacity(shape.frames);
        for target in 0..shape.frames {
            let relative_start = shape.frames - 1 - target;
            if relative_start + k_len > rel_len {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr self attention relative position view exceeds embedding length",
                });
            }
            let pos_offset = relative_start
                .checked_mul(shape.pos_output_dim)
                .and_then(|value| value.checked_add(head * pos_head_dim))
                .and_then(|value| value.checked_mul(element_size))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr self attention relative position offset overflows",
                })?;
            let pos_segment = graph.view_2d(
                projected_pos,
                pos_head_dim,
                k_len,
                pos_row_stride,
                pos_offset,
            )?;
            let pos_segment = graph.cont(pos_segment)?;
            let p_target_offset = target.checked_mul(pos_head_span_bytes).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "xasr self attention p target offset overflows",
                },
            )?;
            let p_target = graph.view_2d(
                p_head,
                pos_head_dim,
                1,
                pos_head_span_bytes,
                p_target_offset,
            )?;
            let p_target = graph.cont(p_target)?;
            let pos_score = graph.mul_mat(pos_segment, p_target)?;
            pos_columns.push(graph.cont(pos_score)?);
        }
        let mut pos_scores = pos_columns[0];
        for &next in &pos_columns[1..] {
            pos_scores = graph.concat(pos_scores, next, 1)?;
        }

        let scores = graph.add(qk_scores, pos_scores)?;
        let scores = graph.cont(scores)?;
        let probs = graph.soft_max_ext(scores, Some(tensors.mask), 1.0, 0.0)?;
        head_outputs.push(graph.cont(probs)?);
    }

    let mut weights = head_outputs[0];
    for &next in &head_outputs[1..] {
        weights = graph.concat(weights, next, 2)?;
    }
    Ok(XasrSelfAttentionWeightsGraphOutput { weights, new_cache })
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerHeadGraphTensors<'a> {
    feed_forward1: XasrFeedForwardGraphTensors<'a>,
    attention_weights: XasrSelfAttentionWeightsGraphTensors<'a>,
    nonlin_cache: GgmlCpuTensor<'a>,
    nonlin_in_proj_weight: GgmlCpuTensor<'a>,
    nonlin_in_proj_bias: GgmlCpuTensor<'a>,
    nonlin_out_proj_weight: GgmlCpuTensor<'a>,
    nonlin_out_proj_bias: GgmlCpuTensor<'a>,
    self1_cache: GgmlCpuTensor<'a>,
    self1_in_proj_weight: GgmlCpuTensor<'a>,
    self1_in_proj_bias: GgmlCpuTensor<'a>,
    self1_out_proj_weight: GgmlCpuTensor<'a>,
    self1_out_proj_bias: GgmlCpuTensor<'a>,
    conv_module1: XasrConvolutionModuleGraphTensors<'a>,
    feed_forward2: XasrFeedForwardGraphTensors<'a>,
    bypass_mid_scale: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerHeadGraphShape {
    dim: usize,
    frames: usize,
    left_context_len: usize,
    num_heads: usize,
    query_head_dim: usize,
    pos_dim: usize,
    pos_output_dim: usize,
    nonlin_hidden_dim: usize,
    self1_value_dim: usize,
    conv1_cache_len: usize,
    conv1_causal_kernel_len: usize,
    conv1_chunk_kernel_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct XasrLayerHeadGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    attention_weights: GgmlCpuTensor<'a>,
    new_cached_key: GgmlCpuTensor<'a>,
    new_cached_nonlin_attention: GgmlCpuTensor<'a>,
    new_cached_val1: GgmlCpuTensor<'a>,
    new_cached_conv1: GgmlCpuTensor<'a>,
}

fn apply_layer_head_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrLayerHeadGraphTensors<'a>,
    shape: XasrLayerHeadGraphShape,
) -> Result<XasrLayerHeadGraphOutput<'a>, GgmlCpuGraphError> {
    let ff1 = apply_feed_forward_graph(graph, input, tensors.feed_forward1)?;
    let add6 = graph.add(input, ff1)?;
    let attention = apply_self_attention_weights_graph(
        graph,
        input,
        tensors.attention_weights,
        XasrSelfAttentionWeightsGraphShape {
            dim: shape.dim,
            frames: shape.frames,
            left_context_len: shape.left_context_len,
            num_heads: shape.num_heads,
            query_head_dim: shape.query_head_dim,
            pos_dim: shape.pos_dim,
            pos_output_dim: shape.pos_output_dim,
        },
    )?;
    let nonlin = apply_nonlin_attention_value_graph(
        graph,
        add6,
        XasrNonlinAttentionGraphTensors {
            cache: tensors.nonlin_cache,
            attention_weights: attention.weights,
            in_proj_weight: tensors.nonlin_in_proj_weight,
            in_proj_bias: tensors.nonlin_in_proj_bias,
            out_proj_weight: tensors.nonlin_out_proj_weight,
            out_proj_bias: tensors.nonlin_out_proj_bias,
        },
        XasrNonlinAttentionGraphShape {
            dim: shape.dim,
            frames: shape.frames,
            left_context_len: shape.left_context_len,
            num_heads: 1,
            hidden_dim: shape.nonlin_hidden_dim,
        },
    )?;
    let add8 = graph.add(add6, nonlin.rows)?;
    let self1 = apply_self_attention_value_graph(
        graph,
        add8,
        XasrSelfAttentionGraphTensors {
            cache: tensors.self1_cache,
            attention_weights: attention.weights,
            in_proj_weight: tensors.self1_in_proj_weight,
            in_proj_bias: tensors.self1_in_proj_bias,
            out_proj_weight: tensors.self1_out_proj_weight,
            out_proj_bias: tensors.self1_out_proj_bias,
        },
        XasrSelfAttentionGraphShape {
            dim: shape.dim,
            frames: shape.frames,
            left_context_len: shape.left_context_len,
            num_heads: shape.num_heads,
            value_dim: shape.self1_value_dim,
        },
    )?;
    let add10 = graph.add(add8, self1.rows)?;
    let conv1 = apply_convolution_module_graph(
        graph,
        add10,
        tensors.conv_module1,
        XasrConvolutionModuleGraphShape {
            dim: shape.dim,
            frames: shape.frames,
            cache_len: shape.conv1_cache_len,
            causal_kernel_len: shape.conv1_causal_kernel_len,
            chunk_kernel_len: shape.conv1_chunk_kernel_len,
        },
    )?;
    let add15 = graph.add(add10, conv1.rows)?;
    let ff2 = apply_feed_forward_graph(graph, add15, tensors.feed_forward2)?;
    let add16 = graph.add(add15, ff2)?;
    let rows = apply_bypass_graph(graph, input, add16, tensors.bypass_mid_scale)?;
    Ok(XasrLayerHeadGraphOutput {
        rows,
        attention_weights: attention.weights,
        new_cached_key: attention.new_cache,
        new_cached_nonlin_attention: nonlin.new_cache,
        new_cached_val1: self1.new_cache,
        new_cached_conv1: conv1.new_cache,
    })
}

#[derive(Debug, Clone, Copy)]
struct XasrZipformerLayerGraphTensors<'a> {
    layer_head: XasrLayerHeadGraphTensors<'a>,
    self2_cache: GgmlCpuTensor<'a>,
    self2_in_proj_weight: GgmlCpuTensor<'a>,
    self2_in_proj_bias: GgmlCpuTensor<'a>,
    self2_out_proj_weight: GgmlCpuTensor<'a>,
    self2_out_proj_bias: GgmlCpuTensor<'a>,
    layer_tail: XasrLayerTailGraphTensors<'a>,
}

#[derive(Debug, Clone, Copy)]
struct XasrZipformerLayerGraphShape {
    layer_head: XasrLayerHeadGraphShape,
    self2_value_dim: usize,
    layer_tail: XasrLayerTailGraphShape,
}

#[derive(Debug, Clone, Copy)]
struct XasrZipformerLayerGraphOutput<'a> {
    rows: GgmlCpuTensor<'a>,
    new_cached_key: GgmlCpuTensor<'a>,
    new_cached_nonlin_attention: GgmlCpuTensor<'a>,
    new_cached_val1: GgmlCpuTensor<'a>,
    new_cached_val2: GgmlCpuTensor<'a>,
    new_cached_conv1: GgmlCpuTensor<'a>,
    new_cached_conv2: GgmlCpuTensor<'a>,
}

fn apply_zipformer_layer_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    tensors: XasrZipformerLayerGraphTensors<'a>,
    shape: XasrZipformerLayerGraphShape,
) -> Result<XasrZipformerLayerGraphOutput<'a>, GgmlCpuGraphError> {
    let head = apply_layer_head_graph(graph, input, tensors.layer_head, shape.layer_head)?;
    let self2 = apply_self_attention_value_graph(
        graph,
        head.rows,
        XasrSelfAttentionGraphTensors {
            cache: tensors.self2_cache,
            attention_weights: head.attention_weights,
            in_proj_weight: tensors.self2_in_proj_weight,
            in_proj_bias: tensors.self2_in_proj_bias,
            out_proj_weight: tensors.self2_out_proj_weight,
            out_proj_bias: tensors.self2_out_proj_bias,
        },
        XasrSelfAttentionGraphShape {
            dim: shape.layer_head.dim,
            frames: shape.layer_head.frames,
            left_context_len: shape.layer_head.left_context_len,
            num_heads: shape.layer_head.num_heads,
            value_dim: shape.self2_value_dim,
        },
    )?;
    let add18 = graph.add(head.rows, self2.rows)?;
    let tail = apply_layer_tail_graph(graph, input, add18, tensors.layer_tail, shape.layer_tail)?;
    Ok(XasrZipformerLayerGraphOutput {
        rows: tail.rows,
        new_cached_key: head.new_cached_key,
        new_cached_nonlin_attention: head.new_cached_nonlin_attention,
        new_cached_val1: head.new_cached_val1,
        new_cached_val2: self2.new_cache,
        new_cached_conv1: head.new_cached_conv1,
        new_cached_conv2: tail.new_conv2_cache,
    })
}

fn apply_bypass_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    original: GgmlCpuTensor<'a>,
    current: GgmlCpuTensor<'a>,
    scale: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let delta = graph.sub(current, original)?;
    let scaled_delta = graph.mul(delta, scale)?;
    graph.add(original, scaled_delta)
}

fn apply_linear_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let projected = graph.mul_mat(weight, input)?;
    graph.add(projected, bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{
        GgmlCpuGraphConfig, GgmlCpuGraphRunner, GgufTensorDataReader, read_gguf_metadata,
    };
    use crate::models::xasr_zipformer::encoder_ops::{
        SWOOSH_L_OFFSET, SWOOSH_L_SHIFT, SWOOSH_R_OFFSET, SWOOSH_R_SHIFT, bias_norm_last_dim,
        swoosh_l, swoosh_r,
    };
    use crate::models::xasr_zipformer::encoder_reference::{
        XasrZipformerLayerReferenceCaches, bypass_reference,
        convolution_module_streaming_reference, downsample_streaming_reference,
        encode_embed_reference, feed_forward_reference, nonlin_attention_streaming_reference,
        self_attention_streaming_reference, self_attention_weights_streaming_reference,
        streaming_key_padding_mask, upsample_streaming_reference,
        zipformer_layer_streaming_reference,
    };
    use crate::models::xasr_zipformer::encoder_weights::{
        XasrConv2dWeights, XasrConvolutionModuleWeights, XasrEncoderEmbedWeights,
        XasrLinearPairWeights, XasrLinearWithBias, load_xasr_encoder_weights,
    };
    use crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata;
    use crate::models::xasr_zipformer::weights::{NamedTensor, StoredLinear};
    use std::fs;
    use std::path::Path;

    fn metadata() -> XasrZipformerExecutionMetadata {
        XasrZipformerExecutionMetadata {
            num_stacks: 1,
            num_encoder_layers: vec![2],
            encoder_dims: vec![192],
            query_head_dims: vec![32],
            value_head_dims: vec![12],
            num_heads: vec![4],
            cnn_module_kernels: vec![31],
            left_context_len: vec![256],
            downsampling_factors: vec![1],
            feature_dim: 80,
            decode_chunk_len: 48,
            joiner_dim: 512,
            decoder_context_size: 2,
            vocab_size: 5000,
            blank_id: 0,
        }
    }

    fn tensor(name: &str, dims: Vec<usize>) -> NamedTensor {
        let len = dims.iter().product();
        NamedTensor {
            name: name.to_string(),
            dims,
            values: vec![0.0; len],
        }
    }

    fn conv2d(name: &str, dims: Vec<usize>) -> XasrConv2dWeights {
        let out = dims[3];
        XasrConv2dWeights {
            weight: tensor(&format!("{name}.weight"), dims),
            bias: vec![0.0; out],
        }
    }

    fn dummy_embed() -> XasrEncoderEmbedWeights {
        XasrEncoderEmbedWeights {
            conv0: conv2d("conv0", vec![3, 3, 1, 8]),
            conv4: conv2d("conv4", vec![3, 3, 8, 32]),
            conv7: conv2d("conv7", vec![3, 3, 32, 128]),
            convnext_depthwise: conv2d("convnext_dw", vec![7, 7, 1, 128]),
            convnext_pointwise1: conv2d("convnext_pw1", vec![1, 1, 128, 384]),
            convnext_pointwise2: conv2d("convnext_pw2", vec![1, 1, 384, 128]),
            out: super::super::encoder_weights::XasrLinearWithBias {
                weight: StoredLinear {
                    name: "out.weight".to_string(),
                    input_dim: 2432,
                    output_dim: 192,
                    values: vec![0.0; 2432 * 192],
                    native: None,
                },
                bias: vec![0.0; 192],
            },
            out_norm_bias: vec![0.0; 192],
            out_norm_log_scale: vec![0.0],
        }
    }

    #[test]
    fn new_reference_rejects_missing_stack0() {
        let weights = XasrEncoderWeights {
            embed: dummy_embed(),
            stacks: Vec::new(),
            downsample_output_bias: vec![0.0, 0.0],
        };

        let error = XasrZipformerEncoderGraph::new_reference(metadata(), weights)
            .expect_err("missing stack0 must fail");

        assert!(error.to_string().contains("weights include 0 stack"));
    }

    #[test]
    fn feature_input_rejects_len_mismatch() {
        let error = XasrEncoderFeatureInput::new(2, 80, vec![0.0; 159])
            .expect_err("shape mismatch must fail");

        assert!(error.to_string().contains("expected 2x80=160"));
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML encoder embed helper with exported ONNX debug tensors"]
    fn ggml_encoder_embed_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-encoder-debug-480ms.x.f32");
        let expected_path = root.join("oracle-encoder-debug-480ms.out_norm.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing encoder embed oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let embed = &weights.embed;
        let input_frames = 61;
        let feature_dim = metadata.feature_dim;
        let embed_width = 19;
        let subsampled_frames = 27;
        let embed_frames = 24;
        let cache_frames = 3;
        let channels = 128;
        let output_dim = metadata.encoder_dims[0];
        assert_eq!(input_values.len(), input_frames * feature_dim);
        assert_eq!(expected.len(), embed_frames * output_dim);
        let reference =
            encode_embed_reference(embed, &input_values, input_frames, feature_dim, None)
                .expect("embed reference");
        assert_max_abs_diff(
            "encoder embed rust reference",
            &reference.rows,
            &expected,
            6.0e-3,
        );

        let cache_values = vec![0.0_f32; channels * cache_frames * embed_width];
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 64 * 1024 * 1024;
        config.graph_size = 16_384;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(feature_dim, input_frames, "embed_input")
            .expect("input allocation should succeed");
        let conv0_w = graph
            .new_tensor_4d_f32(3, 3, 1, 8, "embed_conv0_w")
            .expect("conv0 weight allocation should succeed");
        let conv0_b = graph
            .new_tensor_1d_f32(8, "embed_conv0_b")
            .expect("conv0 bias allocation should succeed");
        let conv4_w = graph
            .new_tensor_4d_f32(3, 3, 8, 32, "embed_conv4_w")
            .expect("conv4 weight allocation should succeed");
        let conv4_b = graph
            .new_tensor_1d_f32(32, "embed_conv4_b")
            .expect("conv4 bias allocation should succeed");
        let conv7_w = graph
            .new_tensor_4d_f32(3, 3, 32, channels, "embed_conv7_w")
            .expect("conv7 weight allocation should succeed");
        let conv7_b = graph
            .new_tensor_1d_f32(channels, "embed_conv7_b")
            .expect("conv7 bias allocation should succeed");
        let embed_cache = graph
            .new_tensor_4d_f32(embed_width, cache_frames, channels, 1, "embed_cache")
            .expect("embed cache allocation should succeed");
        let convnext_dw_w = graph
            .new_tensor_4d_f32(7, 7, 1, channels, "embed_convnext_dw_w")
            .expect("convnext depthwise weight allocation should succeed");
        let convnext_dw_b = graph
            .new_tensor_1d_f32(channels, "embed_convnext_dw_b")
            .expect("convnext depthwise bias allocation should succeed");
        let convnext_pw1_w = graph
            .new_tensor_4d_f32(1, 1, channels, channels * 3, "embed_convnext_pw1_w")
            .expect("convnext pointwise1 weight allocation should succeed");
        let convnext_pw1_b = graph
            .new_tensor_1d_f32(channels * 3, "embed_convnext_pw1_b")
            .expect("convnext pointwise1 bias allocation should succeed");
        let convnext_pw2_w = graph
            .new_tensor_4d_f32(1, 1, channels * 3, channels, "embed_convnext_pw2_w")
            .expect("convnext pointwise2 weight allocation should succeed");
        let convnext_pw2_b = graph
            .new_tensor_1d_f32(channels, "embed_convnext_pw2_b")
            .expect("convnext pointwise2 bias allocation should succeed");
        let out_w = graph
            .new_tensor_2d_f32(embed_width * channels, output_dim, "embed_out_w")
            .expect("out weight allocation should succeed");
        let out_b = graph
            .new_tensor_1d_f32(output_dim, "embed_out_b")
            .expect("out bias allocation should succeed");
        let out_norm_b = graph
            .new_tensor_1d_f32(output_dim, "embed_out_norm_b")
            .expect("out norm bias allocation should succeed");
        let swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "embed_swoosh_r_offset")
            .expect("swoosh r offset allocation should succeed");
        let swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "embed_swoosh_r_shift")
            .expect("swoosh r shift allocation should succeed");
        let swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "embed_swoosh_l_offset")
            .expect("swoosh l offset allocation should succeed");
        let swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "embed_swoosh_l_shift")
            .expect("swoosh l shift allocation should succeed");
        for tensor in [
            input,
            conv0_w,
            conv0_b,
            conv4_w,
            conv4_b,
            conv7_w,
            conv7_b,
            embed_cache,
            convnext_dw_w,
            convnext_dw_b,
            convnext_pw1_w,
            convnext_pw1_b,
            convnext_pw2_w,
            convnext_pw2_b,
            out_w,
            out_b,
            out_norm_b,
            swoosh_r_offset,
            swoosh_r_shift,
            swoosh_l_offset,
            swoosh_l_shift,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }

        let output = apply_encoder_embed_graph(
            &graph,
            input,
            XasrEncoderEmbedGraphTensors {
                conv0_weight: conv0_w,
                conv0_bias: conv0_b,
                conv4_weight: conv4_w,
                conv4_bias: conv4_b,
                conv7_weight: conv7_w,
                conv7_bias: conv7_b,
                embed_cache,
                convnext_depthwise_weight: convnext_dw_w,
                convnext_depthwise_bias: convnext_dw_b,
                convnext_pointwise1_weight: convnext_pw1_w,
                convnext_pointwise1_bias: convnext_pw1_b,
                convnext_pointwise2_weight: convnext_pw2_w,
                convnext_pointwise2_bias: convnext_pw2_b,
                out_weight: out_w,
                out_bias: out_b,
                out_norm_bias: out_norm_b,
                swoosh_r_offset,
                swoosh_r_shift,
                swoosh_l_offset,
                swoosh_l_shift,
            },
            XasrEncoderEmbedGraphShape {
                input_frames,
                feature_dim,
                embed_width,
                subsampled_frames,
                embed_frames,
                cache_frames,
                channels,
                output_dim,
                out_norm_log_scale: embed.out_norm_log_scale[0],
            },
        )
        .expect("encoder embed graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_embed_states)
            .expect("embed cache output should set");

        graph
            .set_f32_slice(input, &input_values, "embed_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(conv0_w, &embed.conv0.weight.values, "embed_conv0_w")
            .expect("conv0 weight upload should succeed");
        graph
            .set_f32_slice(conv0_b, &embed.conv0.bias, "embed_conv0_b")
            .expect("conv0 bias upload should succeed");
        graph
            .set_f32_slice(conv4_w, &embed.conv4.weight.values, "embed_conv4_w")
            .expect("conv4 weight upload should succeed");
        graph
            .set_f32_slice(conv4_b, &embed.conv4.bias, "embed_conv4_b")
            .expect("conv4 bias upload should succeed");
        graph
            .set_f32_slice(conv7_w, &embed.conv7.weight.values, "embed_conv7_w")
            .expect("conv7 weight upload should succeed");
        graph
            .set_f32_slice(conv7_b, &embed.conv7.bias, "embed_conv7_b")
            .expect("conv7 bias upload should succeed");
        graph
            .set_f32_slice(embed_cache, &cache_values, "embed_cache")
            .expect("embed cache upload should succeed");
        graph
            .set_f32_slice(
                convnext_dw_w,
                &embed.convnext_depthwise.weight.values,
                "embed_convnext_dw_w",
            )
            .expect("convnext depthwise weight upload should succeed");
        graph
            .set_f32_slice(
                convnext_dw_b,
                &embed.convnext_depthwise.bias,
                "embed_convnext_dw_b",
            )
            .expect("convnext depthwise bias upload should succeed");
        graph
            .set_f32_slice(
                convnext_pw1_w,
                &embed.convnext_pointwise1.weight.values,
                "embed_convnext_pw1_w",
            )
            .expect("convnext pointwise1 weight upload should succeed");
        graph
            .set_f32_slice(
                convnext_pw1_b,
                &embed.convnext_pointwise1.bias,
                "embed_convnext_pw1_b",
            )
            .expect("convnext pointwise1 bias upload should succeed");
        graph
            .set_f32_slice(
                convnext_pw2_w,
                &embed.convnext_pointwise2.weight.values,
                "embed_convnext_pw2_w",
            )
            .expect("convnext pointwise2 weight upload should succeed");
        graph
            .set_f32_slice(
                convnext_pw2_b,
                &embed.convnext_pointwise2.bias,
                "embed_convnext_pw2_b",
            )
            .expect("convnext pointwise2 bias upload should succeed");
        graph
            .set_f32_slice(out_w, &embed.out.weight.values, "embed_out_w")
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_b, &embed.out.bias, "embed_out_b")
            .expect("out bias upload should succeed");
        graph
            .set_f32_slice(out_norm_b, &embed.out_norm_bias, "embed_out_norm_b")
            .expect("out norm bias upload should succeed");
        graph
            .set_f32_slice(swoosh_r_offset, &[SWOOSH_R_OFFSET], "embed_swoosh_r_offset")
            .expect("swoosh r offset upload should succeed");
        graph
            .set_f32_slice(swoosh_r_shift, &[SWOOSH_R_SHIFT], "embed_swoosh_r_shift")
            .expect("swoosh r shift upload should succeed");
        graph
            .set_f32_slice(swoosh_l_offset, &[SWOOSH_L_OFFSET], "embed_swoosh_l_offset")
            .expect("swoosh l offset upload should succeed");
        graph
            .set_f32_slice(swoosh_l_shift, &[SWOOSH_L_SHIFT], "embed_swoosh_l_shift")
            .expect("swoosh l shift upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, embed_frames * output_dim),
                (
                    output.new_embed_states,
                    channels * cache_frames * embed_width,
                ),
            ])
            .expect("encoder embed graph should compute");
        assert_max_abs_diff(
            "encoder embed graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "encoder embed graph cache",
            &actual[1],
            &reference.new_embed_states,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML downsample helper with exported ONNX debug tensors"]
    fn ggml_downsample_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-stack0-debug-480ms.layer1.f32");
        let padded_path = root.join("oracle-downsample-stack1-debug-480ms.padded.f32");
        let reshaped_path = root.join("oracle-downsample-stack1-debug-480ms.reshaped.f32");
        let output_path = root.join("oracle-downsample-stack1-debug-480ms.output.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists()
            || !padded_path.exists()
            || !reshaped_path.exists()
            || !output_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing downsample oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected_padded = read_f32_file(&padded_path);
        let expected_reshaped = read_f32_file(&reshaped_path);
        let expected_output = read_f32_file(&output_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let stack1 = &weights.stacks[1];
        let frames = 24;
        let input_dim = metadata.encoder_dims[0];
        let target_dim = metadata.encoder_dims[1];
        let factor = metadata.downsampling_factors[1];
        let output_frames = frames / factor;
        assert_eq!(input_values.len(), frames * input_dim);
        assert_eq!(expected_padded.len(), frames * target_dim);
        assert_eq!(expected_reshaped.len(), frames * target_dim);
        assert_eq!(expected_output.len(), output_frames * target_dim);
        let bias_logits = stack1.downsample_bias.as_ref().expect("stack1 downsample");
        let reference = downsample_streaming_reference(
            &input_values,
            frames,
            input_dim,
            target_dim,
            bias_logits,
        )
        .expect("downsample reference");
        assert_max_abs_diff(
            "downsample rust reference output",
            &reference.rows,
            &expected_output,
            2.0e-2,
        );

        let channel_pad_values = vec![0.0_f32; (target_dim - input_dim) * frames];
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(input_dim, frames, "downsample_input")
            .expect("input allocation should succeed");
        let channel_pad = graph
            .new_tensor_2d_f32(target_dim - input_dim, frames, "downsample_channel_pad")
            .expect("channel pad allocation should succeed");
        for tensor in [input, channel_pad] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_downsample_graph(
            &graph,
            input,
            bias_logits,
            XasrDownsampleGraphTensors {
                channel_pad: Some(channel_pad),
                frame_pad: None,
            },
            XasrDownsampleGraphShape {
                frames,
                input_dim,
                target_dim,
                factor,
            },
        )
        .expect("downsample graph");
        assert_eq!(output.frames, output_frames);
        assert_eq!(output.dim, target_dim);
        graph
            .set_output(output.padded_rows)
            .expect("padded rows output should set");
        graph
            .set_output(output.rows)
            .expect("rows output should set");

        graph
            .set_f32_slice(input, &input_values, "downsample_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(channel_pad, &channel_pad_values, "downsample_channel_pad")
            .expect("channel pad upload should succeed");
        let actual = graph
            .compute_outputs_f32(&[
                (output.padded_rows, frames * target_dim),
                (output.rows, output_frames * target_dim),
            ])
            .expect("downsample graph should compute");
        assert_max_abs_diff(
            "downsample graph padded",
            &actual[0],
            &expected_padded,
            2.0e-2,
        );
        assert_max_abs_diff(
            "downsample graph reshaped",
            &actual[0],
            &expected_reshaped,
            2.0e-2,
        );
        assert_max_abs_diff(
            "downsample graph output",
            &actual[1],
            &expected_output,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML upsample/out-combiner helpers with exported ONNX debug tensors"]
    fn ggml_upsample_out_combiner_helpers_match_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let original_path = root.join("oracle-downsample-stack1-debug-480ms.padded.f32");
        let stack1_path = root.join("oracle-stack1-debug-480ms.layer1.f32");
        let upsample_path = root.join("oracle-stack1-combine-debug-480ms.upsample.f32");
        let out_combiner_path = root.join("oracle-stack1-combine-debug-480ms.out_combiner.f32");
        let padded_path = root.join("oracle-stack1-combine-debug-480ms.padded.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !original_path.exists()
            || !stack1_path.exists()
            || !upsample_path.exists()
            || !out_combiner_path.exists()
            || !padded_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing upsample/out-combiner oracle files");
            return;
        }

        let original_values = read_f32_file(&original_path);
        let stack1_values = read_f32_file(&stack1_path);
        let expected_upsample = read_f32_file(&upsample_path);
        let expected_out_combiner = read_f32_file(&out_combiner_path);
        let expected_padded = read_f32_file(&padded_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let stack = &weights.stacks[1];
        let frames = 12;
        let target_frames = 24;
        let dim = metadata.encoder_dims[1];
        let next_dim = metadata.encoder_dims[2];
        let factor = metadata.downsampling_factors[1];
        assert_eq!(original_values.len(), target_frames * dim);
        assert_eq!(stack1_values.len(), frames * dim);
        assert_eq!(expected_upsample.len(), target_frames * dim);
        assert_eq!(expected_out_combiner.len(), target_frames * dim);
        assert_eq!(expected_padded.len(), target_frames * next_dim);
        let scale = stack
            .out_combiner_bypass_scale
            .as_ref()
            .expect("stack1 out combiner");
        let upsample =
            upsample_streaming_reference(&stack1_values, frames, dim, factor, target_frames)
                .expect("upsample reference");
        let out_combiner = bypass_reference(&original_values, &upsample, scale, target_frames, dim)
            .expect("out combiner reference");
        assert_max_abs_diff(
            "upsample rust reference",
            &upsample,
            &expected_upsample,
            2.0e-2,
        );
        assert_max_abs_diff(
            "out combiner rust reference",
            &out_combiner,
            &expected_out_combiner,
            2.0e-2,
        );

        let channel_pad_values = vec![0.0_f32; (next_dim - dim) * target_frames];
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let original = graph
            .new_tensor_2d_f32(dim, target_frames, "combine_original")
            .expect("original allocation should succeed");
        let stack_rows = graph
            .new_tensor_2d_f32(dim, frames, "combine_stack")
            .expect("stack rows allocation should succeed");
        let bypass_scale = graph
            .new_tensor_1d_f32(dim, "combine_bypass_scale")
            .expect("bypass scale allocation should succeed");
        let channel_pad = graph
            .new_tensor_2d_f32(next_dim - dim, target_frames, "combine_channel_pad")
            .expect("channel pad allocation should succeed");
        for tensor in [original, stack_rows, bypass_scale, channel_pad] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let upsample = apply_upsample_graph(
            &graph,
            stack_rows,
            XasrUpsampleGraphShape {
                frames,
                dim,
                factor,
                target_frames,
            },
        )
        .expect("upsample graph");
        let out_combiner = apply_bypass_graph(&graph, original, upsample, bypass_scale)
            .expect("out combiner graph");
        let padded = graph
            .concat(out_combiner, channel_pad, 0)
            .expect("next-stack channel pad should build");
        graph
            .set_output(upsample)
            .expect("upsample output should set");
        graph
            .set_output(out_combiner)
            .expect("out combiner output should set");
        graph.set_output(padded).expect("padded output should set");

        graph
            .set_f32_slice(original, &original_values, "combine_original")
            .expect("original upload should succeed");
        graph
            .set_f32_slice(stack_rows, &stack1_values, "combine_stack")
            .expect("stack rows upload should succeed");
        graph
            .set_f32_slice(bypass_scale, scale, "combine_bypass_scale")
            .expect("bypass scale upload should succeed");
        graph
            .set_f32_slice(channel_pad, &channel_pad_values, "combine_channel_pad")
            .expect("channel pad upload should succeed");
        let actual = graph
            .compute_outputs_f32(&[
                (upsample, target_frames * dim),
                (out_combiner, target_frames * dim),
                (padded, target_frames * next_dim),
            ])
            .expect("upsample/out-combiner graph should compute");
        assert_max_abs_diff(
            "upsample graph ONNX parity",
            &actual[0],
            &expected_upsample,
            2.0e-2,
        );
        assert_max_abs_diff(
            "out combiner graph ONNX parity",
            &actual[1],
            &expected_out_combiner,
            2.0e-2,
        );
        assert_max_abs_diff(
            "combiner padded graph ONNX parity",
            &actual[2],
            &expected_padded,
            2.0e-2,
        );
    }

    #[test]
    fn ggml_swoosh_helpers_match_rust_reference() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(3, 2, "swoosh_input")
            .expect("input allocation should succeed");
        let r_offset = graph
            .new_tensor_1d_f32(1, "swoosh_r_offset")
            .expect("r offset allocation should succeed");
        let r_shift = graph
            .new_tensor_1d_f32(1, "swoosh_r_shift")
            .expect("r shift allocation should succeed");
        let l_offset = graph
            .new_tensor_1d_f32(1, "swoosh_l_offset")
            .expect("l offset allocation should succeed");
        let l_shift = graph
            .new_tensor_1d_f32(1, "swoosh_l_shift")
            .expect("l shift allocation should succeed");
        for tensor in [input, r_offset, r_shift, l_offset, l_shift] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }

        let swoosh_r_output =
            apply_swoosh_r_graph(&graph, input, r_offset, r_shift).expect("swoosh r graph");
        let swoosh_l_output =
            apply_swoosh_l_graph(&graph, input, l_offset, l_shift).expect("swoosh l graph");
        graph
            .set_output(swoosh_r_output)
            .expect("swoosh r output should set");
        graph
            .set_output(swoosh_l_output)
            .expect("swoosh l output should set");

        let values = [-2.0_f32, -0.5, 0.0, 1.0, 2.0, 4.0];
        graph
            .set_f32_slice(input, &values, "swoosh_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(r_offset, &[SWOOSH_R_OFFSET], "swoosh_r_offset")
            .expect("r offset upload should succeed");
        graph
            .set_f32_slice(r_shift, &[SWOOSH_R_SHIFT], "swoosh_r_shift")
            .expect("r shift upload should succeed");
        graph
            .set_f32_slice(l_offset, &[SWOOSH_L_OFFSET], "swoosh_l_offset")
            .expect("l offset upload should succeed");
        graph
            .set_f32_slice(l_shift, &[SWOOSH_L_SHIFT], "swoosh_l_shift")
            .expect("l shift upload should succeed");

        let outputs = graph
            .compute_outputs_f32(&[
                (swoosh_r_output, values.len()),
                (swoosh_l_output, values.len()),
            ])
            .expect("swoosh graph should compute");

        assert_max_abs_diff("swoosh r graph", &outputs[0], &values.map(swoosh_r), 1.0e-5);
        assert_max_abs_diff("swoosh l graph", &outputs[1], &values.map(swoosh_l), 1.0e-5);
    }

    #[test]
    fn ggml_bias_norm_helper_matches_rust_reference() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(3, 2, "bias_norm_input")
            .expect("input allocation should succeed");
        let bias_tensor = graph
            .new_tensor_1d_f32(3, "bias_norm_bias")
            .expect("bias allocation should succeed");
        graph
            .set_input(input)
            .expect("input set_input should succeed");
        graph
            .set_input(bias_tensor)
            .expect("bias set_input should succeed");

        let log_scale = 0.2_f32;
        let output =
            apply_bias_norm_graph(&graph, input, bias_tensor, log_scale).expect("bias norm graph");
        graph
            .set_output(output)
            .expect("bias norm output should set");

        let values = [2.0_f32, 4.0, 6.0, 3.0, 3.0, 3.0];
        let bias = [1.0_f32, 2.0, 3.0];
        graph
            .set_f32_slice(input, &values, "bias_norm_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(bias_tensor, &bias, "bias_norm_bias")
            .expect("bias upload should succeed");
        let actual = graph
            .compute_output_f32(output, values.len())
            .expect("bias norm graph should compute");

        let mut expected = values.to_vec();
        bias_norm_last_dim(&mut expected, 3, &bias, log_scale).expect("reference bias norm");
        assert_max_abs_diff("bias norm graph", &actual, &expected, 1.0e-5);
    }

    #[test]
    fn ggml_feed_forward_helper_matches_rust_reference() {
        const DIM: usize = 3;
        const HIDDEN: usize = 4;
        const FRAMES: usize = 2;

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(DIM, FRAMES, "ff_input")
            .expect("input allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(DIM, HIDDEN, "ff_in_weight")
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(HIDDEN, "ff_in_bias")
            .expect("in bias allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(HIDDEN, DIM, "ff_out_weight")
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(DIM, "ff_out_bias")
            .expect("out bias allocation should succeed");
        let swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "ff_swoosh_l_offset")
            .expect("swoosh offset allocation should succeed");
        let swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "ff_swoosh_l_shift")
            .expect("swoosh shift allocation should succeed");
        for tensor in [
            input,
            in_weight,
            in_bias,
            out_weight,
            out_bias,
            swoosh_l_offset,
            swoosh_l_shift,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }

        let output = apply_feed_forward_graph(
            &graph,
            input,
            XasrFeedForwardGraphTensors {
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
                swoosh_l_offset,
                swoosh_l_shift,
            },
        )
        .expect("feed-forward graph");
        graph
            .set_output(output)
            .expect("feed-forward output should set");

        let input_values = [0.2_f32, -0.5, 1.0, 1.5, -1.0, 0.25];
        let in_weight_values = [
            0.1_f32, -0.2, 0.3, //
            0.4, 0.0, -0.1, //
            -0.3, 0.2, 0.5, //
            0.6, -0.4, 0.1,
        ];
        let in_bias_values = [0.01_f32, -0.02, 0.03, -0.04];
        let out_weight_values = [
            0.2_f32, -0.1, 0.3, -0.2, //
            -0.4, 0.5, 0.1, 0.2, //
            0.3, 0.2, -0.5, 0.4,
        ];
        let out_bias_values = [0.05_f32, -0.03, 0.02];
        graph
            .set_f32_slice(input, &input_values, "ff_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(in_weight, &in_weight_values, "ff_in_weight")
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &in_bias_values, "ff_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(out_weight, &out_weight_values, "ff_out_weight")
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &out_bias_values, "ff_out_bias")
            .expect("out bias upload should succeed");
        graph
            .set_f32_slice(swoosh_l_offset, &[SWOOSH_L_OFFSET], "ff_swoosh_l_offset")
            .expect("swoosh offset upload should succeed");
        graph
            .set_f32_slice(swoosh_l_shift, &[SWOOSH_L_SHIFT], "ff_swoosh_l_shift")
            .expect("swoosh shift upload should succeed");
        let actual = graph
            .compute_output_f32(output, DIM * FRAMES)
            .expect("feed-forward graph should compute");

        let weights = XasrLinearPairWeights {
            in_proj: XasrLinearWithBias {
                weight: StoredLinear {
                    name: "ff.in_proj.weight".to_string(),
                    input_dim: DIM,
                    output_dim: HIDDEN,
                    values: in_weight_values.to_vec(),
                    native: None,
                },
                bias: in_bias_values.to_vec(),
            },
            out_proj: XasrLinearWithBias {
                weight: StoredLinear {
                    name: "ff.out_proj.weight".to_string(),
                    input_dim: HIDDEN,
                    output_dim: DIM,
                    values: out_weight_values.to_vec(),
                    native: None,
                },
                bias: out_bias_values.to_vec(),
            },
        };
        let expected =
            feed_forward_reference(&weights, &input_values, FRAMES, DIM).expect("reference ff");
        assert_max_abs_diff("feed-forward graph", &actual, &expected, 1.0e-5);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML feed-forward helper with exported ONNX debug tensors"]
    fn ggml_feed_forward_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-layer0-ff-debug-480ms.ff1.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing feed-forward graph oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let ff = &weights.stacks[0].layers[0].feed_forward1;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "ff1_input")
            .expect("input allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(
                ff.in_proj.weight.input_dim,
                ff.in_proj.weight.output_dim,
                "ff1_in_weight",
            )
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(ff.in_proj.bias.len(), "ff1_in_bias")
            .expect("in bias allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(
                ff.out_proj.weight.input_dim,
                ff.out_proj.weight.output_dim,
                "ff1_out_weight",
            )
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(ff.out_proj.bias.len(), "ff1_out_bias")
            .expect("out bias allocation should succeed");
        let swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "ff1_swoosh_l_offset")
            .expect("swoosh offset allocation should succeed");
        let swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "ff1_swoosh_l_shift")
            .expect("swoosh shift allocation should succeed");
        for tensor in [
            input,
            in_weight,
            in_bias,
            out_weight,
            out_bias,
            swoosh_l_offset,
            swoosh_l_shift,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_feed_forward_graph(
            &graph,
            input,
            XasrFeedForwardGraphTensors {
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
                swoosh_l_offset,
                swoosh_l_shift,
            },
        )
        .expect("feed-forward graph");
        graph
            .set_output(output)
            .expect("feed-forward output should set");

        graph
            .set_f32_slice(input, &input_values, "ff1_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(in_weight, &ff.in_proj.weight.values, "ff1_in_weight")
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &ff.in_proj.bias, "ff1_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(out_weight, &ff.out_proj.weight.values, "ff1_out_weight")
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &ff.out_proj.bias, "ff1_out_bias")
            .expect("out bias upload should succeed");
        graph
            .set_f32_slice(swoosh_l_offset, &[SWOOSH_L_OFFSET], "ff1_swoosh_l_offset")
            .expect("swoosh offset upload should succeed");
        graph
            .set_f32_slice(swoosh_l_shift, &[SWOOSH_L_SHIFT], "ff1_swoosh_l_shift")
            .expect("swoosh shift upload should succeed");
        let actual = graph
            .compute_output_f32(output, frames * dim)
            .expect("feed-forward graph should compute");

        assert_max_abs_diff("feed-forward graph ONNX parity", &actual, &expected, 2.0e-2);
    }

    #[test]
    fn ggml_bypass_helper_matches_rust_reference() {
        const DIM: usize = 3;
        const FRAMES: usize = 2;

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let original = graph
            .new_tensor_2d_f32(DIM, FRAMES, "bypass_original")
            .expect("original allocation should succeed");
        let current = graph
            .new_tensor_2d_f32(DIM, FRAMES, "bypass_current")
            .expect("current allocation should succeed");
        let scale = graph
            .new_tensor_1d_f32(DIM, "bypass_scale")
            .expect("scale allocation should succeed");
        for tensor in [original, current, scale] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_bypass_graph(&graph, original, current, scale)
            .expect("bypass graph should build");
        graph.set_output(output).expect("bypass output should set");

        let original_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let current_values = [2.0_f32, -2.0, 7.0, 0.0, 8.0, 9.0];
        let scale_values = [0.0_f32, 0.25, 1.0];
        graph
            .set_f32_slice(original, &original_values, "bypass_original")
            .expect("original upload should succeed");
        graph
            .set_f32_slice(current, &current_values, "bypass_current")
            .expect("current upload should succeed");
        graph
            .set_f32_slice(scale, &scale_values, "bypass_scale")
            .expect("scale upload should succeed");
        let actual = graph
            .compute_output_f32(output, DIM * FRAMES)
            .expect("bypass graph should compute");
        let expected = bypass_reference(
            &original_values,
            &current_values,
            &scale_values,
            FRAMES,
            DIM,
        )
        .expect("reference bypass");
        assert_max_abs_diff("bypass graph", &actual, &expected, 1.0e-6);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML bypass helper with exported ONNX debug tensors"]
    fn ggml_bypass_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let add16_path = root.join("oracle-layer0-ff-debug-480ms.add16.f32");
        let norm_path = root.join("oracle-layer0-second-half-debug-480ms.norm.f32");
        let expected_mid_path = root.join("oracle-layer0-second-half-debug-480ms.bypass_mid.f32");
        let expected_final_path = root.join("oracle-layer0-second-half-debug-480ms.bypass.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists()
            || !add16_path.exists()
            || !norm_path.exists()
            || !expected_mid_path.exists()
            || !expected_final_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing bypass graph oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let add16 = read_f32_file(&add16_path);
        let norm = read_f32_file(&norm_path);
        let expected_mid = read_f32_file(&expected_mid_path);
        let expected_final = read_f32_file(&expected_final_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let layer = &weights.stacks[0].layers[0];
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        for values in [&input_values, &add16, &norm, &expected_mid, &expected_final] {
            assert_eq!(values.len(), frames * dim);
        }

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let original = graph
            .new_tensor_2d_f32(dim, frames, "bypass_original")
            .expect("original allocation should succeed");
        let mid_current = graph
            .new_tensor_2d_f32(dim, frames, "bypass_mid_current")
            .expect("mid current allocation should succeed");
        let final_current = graph
            .new_tensor_2d_f32(dim, frames, "bypass_final_current")
            .expect("final current allocation should succeed");
        let mid_scale = graph
            .new_tensor_1d_f32(dim, "bypass_mid_scale")
            .expect("mid scale allocation should succeed");
        let final_scale = graph
            .new_tensor_1d_f32(dim, "bypass_final_scale")
            .expect("final scale allocation should succeed");
        for tensor in [original, mid_current, final_current, mid_scale, final_scale] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let mid_output =
            apply_bypass_graph(&graph, original, mid_current, mid_scale).expect("mid bypass");
        let final_output =
            apply_bypass_graph(&graph, original, final_current, final_scale).expect("final bypass");
        graph.set_output(mid_output).expect("mid output should set");
        graph
            .set_output(final_output)
            .expect("final output should set");

        graph
            .set_f32_slice(original, &input_values, "bypass_original")
            .expect("original upload should succeed");
        graph
            .set_f32_slice(mid_current, &add16, "bypass_mid_current")
            .expect("mid current upload should succeed");
        graph
            .set_f32_slice(final_current, &norm, "bypass_final_current")
            .expect("final current upload should succeed");
        graph
            .set_f32_slice(mid_scale, &layer.bypass_mid_scale, "bypass_mid_scale")
            .expect("mid scale upload should succeed");
        graph
            .set_f32_slice(final_scale, &layer.bypass_scale, "bypass_final_scale")
            .expect("final scale upload should succeed");

        let outputs = graph
            .compute_outputs_f32(&[(mid_output, frames * dim), (final_output, frames * dim)])
            .expect("bypass graph should compute");
        assert_max_abs_diff(
            "bypass_mid graph ONNX parity",
            &outputs[0],
            &expected_mid,
            2.0e-2,
        );
        assert_max_abs_diff(
            "bypass final graph ONNX parity",
            &outputs[1],
            &expected_final,
            2.0e-2,
        );
    }

    #[test]
    fn ggml_conv_in_proj_glu_helper_matches_manual_reference() {
        const DIM: usize = 3;
        const FRAMES: usize = 2;
        const OUT: usize = DIM * 2;

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(DIM, FRAMES, "conv_glu_input")
            .expect("input allocation should succeed");
        let weight = graph
            .new_tensor_2d_f32(DIM, OUT, "conv_glu_weight")
            .expect("weight allocation should succeed");
        let bias = graph
            .new_tensor_1d_f32(OUT, "conv_glu_bias")
            .expect("bias allocation should succeed");
        for tensor in [input, weight, bias] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_conv_in_proj_glu_graph(&graph, input, weight, bias, DIM, FRAMES)
            .expect("conv GLU graph should build");
        graph
            .set_output(output)
            .expect("conv GLU output should set");

        let input_values = [0.2_f32, -0.5, 1.0, 1.5, -1.0, 0.25];
        let weight_values = [
            0.1_f32, -0.2, 0.3, //
            0.4, 0.0, -0.1, //
            -0.3, 0.2, 0.5, //
            0.6, -0.4, 0.1, //
            0.2, 0.7, -0.3, //
            -0.5, 0.1, 0.4,
        ];
        let bias_values = [0.01_f32, -0.02, 0.03, -0.04, 0.05, -0.06];
        graph
            .set_f32_slice(input, &input_values, "conv_glu_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(weight, &weight_values, "conv_glu_weight")
            .expect("weight upload should succeed");
        graph
            .set_f32_slice(bias, &bias_values, "conv_glu_bias")
            .expect("bias upload should succeed");
        let actual = graph
            .compute_output_f32(output, DIM * FRAMES)
            .expect("conv GLU graph should compute");

        let linear = StoredLinear {
            name: "conv.in_proj.weight".to_string(),
            input_dim: DIM,
            output_dim: OUT,
            values: weight_values.to_vec(),
            native: None,
        };
        let mut expected = Vec::with_capacity(DIM * FRAMES);
        for frame in input_values.chunks_exact(DIM) {
            let projected = linear.apply(frame, Some(&bias_values)).expect("linear");
            for c in 0..DIM {
                expected.push(projected[c] * sigmoid_reference(projected[DIM + c]));
            }
        }
        assert_max_abs_diff("conv in_proj GLU graph", &actual, &expected, 1.0e-6);
    }

    #[test]
    fn ggml_depthwise_conv1d_channel_major_helper_matches_manual_reference() {
        const CHANNELS: usize = 2;
        const INPUT_LEN: usize = 4;
        const KERNEL: usize = 3;
        const VALID_OUT: usize = INPUT_LEN - KERNEL + 1;

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(INPUT_LEN, CHANNELS, "dw_input")
            .expect("input allocation should succeed");
        let kernel = graph
            .new_tensor_3d_f32(KERNEL, 1, CHANNELS, "dw_kernel")
            .expect("kernel allocation should succeed");
        let bias = graph
            .new_tensor_1d_f32(CHANNELS, "dw_bias")
            .expect("bias allocation should succeed");
        for tensor in [input, kernel, bias] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let valid = apply_depthwise_conv1d_channel_major_graph(
            &graph, input, kernel, bias, CHANNELS, INPUT_LEN, KERNEL, 0,
        )
        .expect("valid depthwise conv graph should build");
        let same = apply_depthwise_conv1d_channel_major_graph(
            &graph,
            input,
            kernel,
            bias,
            CHANNELS,
            INPUT_LEN,
            KERNEL,
            KERNEL / 2,
        )
        .expect("same depthwise conv graph should build");
        graph.set_output(valid).expect("valid output should set");
        graph.set_output(same).expect("same output should set");

        let input_values = [
            1.0_f32, 2.0, 3.0, 4.0, //
            -1.0, 0.5, 2.0, -0.5,
        ];
        let kernel_values = [
            0.25_f32, -0.5, 0.75, //
            -0.3, 0.1, 0.2,
        ];
        let bias_values = [0.05_f32, -0.1];
        graph
            .set_f32_slice(input, &input_values, "dw_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(kernel, &kernel_values, "dw_kernel")
            .expect("kernel upload should succeed");
        graph
            .set_f32_slice(bias, &bias_values, "dw_bias")
            .expect("bias upload should succeed");
        let outputs = graph
            .compute_outputs_f32(&[(valid, CHANNELS * VALID_OUT), (same, CHANNELS * INPUT_LEN)])
            .expect("depthwise conv graph should compute");

        let expected_valid = depthwise_conv1d_channel_major_reference(
            &input_values,
            &kernel_values,
            &bias_values,
            CHANNELS,
            INPUT_LEN,
            KERNEL,
            0,
        );
        let expected_same = depthwise_conv1d_channel_major_reference(
            &input_values,
            &kernel_values,
            &bias_values,
            CHANNELS,
            INPUT_LEN,
            KERNEL,
            KERNEL / 2,
        );
        assert_max_abs_diff(
            "depthwise valid graph",
            &outputs[0],
            &expected_valid,
            1.0e-6,
        );
        assert_max_abs_diff("depthwise same graph", &outputs[1], &expected_same, 1.0e-6);
    }

    #[test]
    fn ggml_depthwise_mix_channel_major_helper_matches_manual_reference() {
        const CHANNELS: usize = 2;
        const FRAMES: usize = 4;
        const CACHED_LEN: usize = 5;
        const CAUSAL_KERNEL: usize = 2;
        const CHUNK_KERNEL: usize = 3;

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let cached_input = graph
            .new_tensor_2d_f32(CACHED_LEN, CHANNELS, "mix_cached_input")
            .expect("cached input allocation should succeed");
        let chunk_input = graph
            .new_tensor_2d_f32(FRAMES, CHANNELS, "mix_chunk_input")
            .expect("chunk input allocation should succeed");
        let causal_kernel = graph
            .new_tensor_3d_f32(CAUSAL_KERNEL, 1, CHANNELS, "mix_causal_kernel")
            .expect("causal kernel allocation should succeed");
        let causal_bias = graph
            .new_tensor_1d_f32(CHANNELS, "mix_causal_bias")
            .expect("causal bias allocation should succeed");
        let chunk_kernel = graph
            .new_tensor_3d_f32(CHUNK_KERNEL, 1, CHANNELS, "mix_chunk_kernel")
            .expect("chunk kernel allocation should succeed");
        let chunk_bias = graph
            .new_tensor_1d_f32(CHANNELS, "mix_chunk_bias")
            .expect("chunk bias allocation should succeed");
        let chunk_scale = graph
            .new_tensor_2d_f32(FRAMES, CHANNELS, "mix_chunk_scale")
            .expect("chunk scale allocation should succeed");
        for tensor in [
            cached_input,
            chunk_input,
            causal_kernel,
            causal_bias,
            chunk_kernel,
            chunk_bias,
            chunk_scale,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_depthwise_mix_channel_major_graph(
            &graph,
            XasrDepthwiseMixGraphTensors {
                cached_input,
                chunk_input,
                causal_kernel,
                causal_bias,
                chunk_kernel,
                chunk_bias,
                chunk_scale,
            },
            XasrDepthwiseMixShape {
                channels: CHANNELS,
                frames: FRAMES,
                cached_len: CACHED_LEN,
                causal_kernel_len: CAUSAL_KERNEL,
                chunk_kernel_len: CHUNK_KERNEL,
            },
        )
        .expect("depthwise mix graph should build");
        graph.set_output(output).expect("mix output should set");

        let cached_values = [
            0.25_f32, 1.0, 2.0, 3.0, 4.0, //
            -0.75, -1.0, 0.5, 2.0, -0.5,
        ];
        let chunk_values = [
            1.0_f32, 2.0, 3.0, 4.0, //
            -1.0, 0.5, 2.0, -0.5,
        ];
        let causal_kernel_values = [
            0.5_f32, -0.25, //
            -0.2, 0.4,
        ];
        let causal_bias_values = [0.05_f32, -0.1];
        let chunk_kernel_values = [
            0.25_f32, -0.5, 0.75, //
            -0.3, 0.1, 0.2,
        ];
        let chunk_bias_values = [0.01_f32, -0.02];
        let chunk_scale_values = [
            1.0_f32, 0.5, 1.5, 0.25, //
            0.75, 1.25, 0.8, 1.1,
        ];
        graph
            .set_f32_slice(cached_input, &cached_values, "mix_cached_input")
            .expect("cached input upload should succeed");
        graph
            .set_f32_slice(chunk_input, &chunk_values, "mix_chunk_input")
            .expect("chunk input upload should succeed");
        graph
            .set_f32_slice(causal_kernel, &causal_kernel_values, "mix_causal_kernel")
            .expect("causal kernel upload should succeed");
        graph
            .set_f32_slice(causal_bias, &causal_bias_values, "mix_causal_bias")
            .expect("causal bias upload should succeed");
        graph
            .set_f32_slice(chunk_kernel, &chunk_kernel_values, "mix_chunk_kernel")
            .expect("chunk kernel upload should succeed");
        graph
            .set_f32_slice(chunk_bias, &chunk_bias_values, "mix_chunk_bias")
            .expect("chunk bias upload should succeed");
        graph
            .set_f32_slice(chunk_scale, &chunk_scale_values, "mix_chunk_scale")
            .expect("chunk scale upload should succeed");

        let actual = graph
            .compute_output_f32(output, CHANNELS * FRAMES)
            .expect("depthwise mix graph should compute");
        let expected = depthwise_mix_channel_major_reference(
            &cached_values,
            &chunk_values,
            &causal_kernel_values,
            &causal_bias_values,
            &chunk_kernel_values,
            &chunk_bias_values,
            &chunk_scale_values,
            XasrDepthwiseMixShape {
                channels: CHANNELS,
                frames: FRAMES,
                cached_len: CACHED_LEN,
                causal_kernel_len: CAUSAL_KERNEL,
                chunk_kernel_len: CHUNK_KERNEL,
            },
        );
        assert_max_abs_diff("depthwise mix graph", &actual, &expected, 1.0e-6);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML depthwise mix helper with exported ONNX debug tensors"]
    fn ggml_depthwise_mix_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-second-half-debug-480ms.add18.f32");
        let expected_path = root.join("oracle-layer0-second-half-debug-480ms.conv2_depthwise.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing depthwise mix oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let conv = &weights.stacks[0].layers[0].conv_module2;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);

        let prepared =
            prepare_depthwise_mix_inputs_for_test(conv, &input_values, frames, dim).expect("prep");
        let causal_kernel = conv.depthwise_causal_conv.weight.dims[0];
        let chunk_kernel = conv.depthwise_chunkwise_conv.weight.dims[0];
        let cached_len = chunk_kernel / 2 + frames;
        assert_eq!(prepared.cached_input.len(), dim * cached_len);
        assert_eq!(prepared.channel_major.len(), dim * frames);
        assert_eq!(prepared.chunk_scale.len(), dim * frames);

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let cached_input = graph
            .new_tensor_2d_f32(cached_len, dim, "conv2_cached_input")
            .expect("cached input allocation should succeed");
        let chunk_input = graph
            .new_tensor_2d_f32(frames, dim, "conv2_chunk_input")
            .expect("chunk input allocation should succeed");
        let causal_kernel_t = graph
            .new_tensor_3d_f32(causal_kernel, 1, dim, "conv2_causal_kernel")
            .expect("causal kernel allocation should succeed");
        let causal_bias = graph
            .new_tensor_1d_f32(dim, "conv2_causal_bias")
            .expect("causal bias allocation should succeed");
        let chunk_kernel_t = graph
            .new_tensor_3d_f32(chunk_kernel, 1, dim, "conv2_chunk_kernel")
            .expect("chunk kernel allocation should succeed");
        let chunk_bias = graph
            .new_tensor_1d_f32(dim, "conv2_chunk_bias")
            .expect("chunk bias allocation should succeed");
        let chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "conv2_chunk_scale")
            .expect("chunk scale allocation should succeed");
        for tensor in [
            cached_input,
            chunk_input,
            causal_kernel_t,
            causal_bias,
            chunk_kernel_t,
            chunk_bias,
            chunk_scale,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_depthwise_mix_channel_major_graph(
            &graph,
            XasrDepthwiseMixGraphTensors {
                cached_input,
                chunk_input,
                causal_kernel: causal_kernel_t,
                causal_bias,
                chunk_kernel: chunk_kernel_t,
                chunk_bias,
                chunk_scale,
            },
            XasrDepthwiseMixShape {
                channels: dim,
                frames,
                cached_len,
                causal_kernel_len: causal_kernel,
                chunk_kernel_len: chunk_kernel,
            },
        )
        .expect("depthwise mix graph");
        graph.set_output(output).expect("output should set");

        graph
            .set_f32_slice(cached_input, &prepared.cached_input, "conv2_cached_input")
            .expect("cached input upload should succeed");
        graph
            .set_f32_slice(chunk_input, &prepared.channel_major, "conv2_chunk_input")
            .expect("chunk input upload should succeed");
        graph
            .set_f32_slice(
                causal_kernel_t,
                &conv.depthwise_causal_conv.weight.values,
                "conv2_causal_kernel",
            )
            .expect("causal kernel upload should succeed");
        graph
            .set_f32_slice(
                causal_bias,
                &conv.depthwise_causal_conv.bias,
                "conv2_causal_bias",
            )
            .expect("causal bias upload should succeed");
        graph
            .set_f32_slice(
                chunk_kernel_t,
                &conv.depthwise_chunkwise_conv.weight.values,
                "conv2_chunk_kernel",
            )
            .expect("chunk kernel upload should succeed");
        graph
            .set_f32_slice(
                chunk_bias,
                &conv.depthwise_chunkwise_conv.bias,
                "conv2_chunk_bias",
            )
            .expect("chunk bias upload should succeed");
        graph
            .set_f32_slice(chunk_scale, &prepared.chunk_scale, "conv2_chunk_scale")
            .expect("chunk scale upload should succeed");

        let actual = graph
            .compute_output_f32(output, dim * frames)
            .expect("depthwise mix graph should compute");
        assert_max_abs_diff(
            "conv2 depthwise mix ONNX parity",
            &actual,
            &expected,
            2.0e-2,
        );
    }

    #[test]
    fn ggml_convolution_module_helper_matches_rust_reference() {
        const DIM: usize = 2;
        const FRAMES: usize = 3;
        const CACHE_LEN: usize = 1;
        const CAUSAL_KERNEL: usize = 2;
        const CHUNK_KERNEL: usize = 3;
        const PROJECTED: usize = 2 * DIM;

        let input_values = [0.2_f32, -0.5, 1.0, 1.5, -1.0, 0.25];
        let cache_values = [0.125_f32, -0.25];
        let in_weight_values = [
            0.1_f32, -0.2, //
            0.3, 0.4, //
            -0.5, 0.2, //
            0.6, -0.1,
        ];
        let in_bias_values = [0.01_f32, -0.02, 0.03, -0.04];
        let causal_kernel_values = [
            0.5_f32, -0.25, //
            -0.2, 0.4,
        ];
        let causal_bias_values = [0.05_f32, -0.1];
        let chunk_kernel_values = [
            0.25_f32, -0.5, 0.75, //
            -0.3, 0.1, 0.2,
        ];
        let chunk_bias_values = [0.01_f32, -0.02];
        let chunk_scale_source_values = [
            0.02_f32, -0.03, 0.01, //
            -0.04, 0.05, 0.03, //
            0.01, 0.02, -0.02, //
            0.03, -0.01, 0.04,
        ];
        let out_weight_values = [
            0.2_f32, -0.1, //
            -0.4, 0.5,
        ];
        let out_bias_values = [0.05_f32, -0.03];
        let weights = XasrConvolutionModuleWeights {
            in_proj: XasrLinearWithBias {
                weight: StoredLinear {
                    name: "conv.in_proj.weight".to_string(),
                    input_dim: DIM,
                    output_dim: PROJECTED,
                    values: in_weight_values.to_vec(),
                    native: None,
                },
                bias: in_bias_values.to_vec(),
            },
            depthwise_causal_conv: super::super::encoder_weights::XasrConv1dWeights {
                weight: NamedTensor {
                    name: "conv.depthwise.causal.weight".to_string(),
                    dims: vec![CAUSAL_KERNEL, 1, DIM],
                    values: causal_kernel_values.to_vec(),
                },
                bias: causal_bias_values.to_vec(),
            },
            depthwise_chunkwise_conv: super::super::encoder_weights::XasrConv1dWeights {
                weight: NamedTensor {
                    name: "conv.depthwise.chunk.weight".to_string(),
                    dims: vec![CHUNK_KERNEL, 1, DIM],
                    values: chunk_kernel_values.to_vec(),
                },
                bias: chunk_bias_values.to_vec(),
            },
            chunkwise_conv_scale: NamedTensor {
                name: "conv.depthwise.chunk_scale".to_string(),
                dims: vec![2, DIM, CHUNK_KERNEL],
                values: chunk_scale_source_values.to_vec(),
            },
            out_proj: XasrLinearWithBias {
                weight: StoredLinear {
                    name: "conv.out_proj.weight".to_string(),
                    input_dim: DIM,
                    output_dim: DIM,
                    values: out_weight_values.to_vec(),
                    native: None,
                },
                bias: out_bias_values.to_vec(),
            },
        };
        let mut chunk_scale_values = vec![0.0_f32; DIM * FRAMES];
        for c in 0..DIM {
            for t in 0..FRAMES {
                chunk_scale_values[c * FRAMES + t] =
                    chunkwise_conv_scale_for_test(&weights, c, t, FRAMES).expect("chunkwise scale");
            }
        }

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(DIM, FRAMES, "conv_input")
            .expect("input allocation should succeed");
        let cache = graph
            .new_tensor_2d_f32(CACHE_LEN, DIM, "conv_cache")
            .expect("cache allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(DIM, PROJECTED, "conv_in_weight")
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(PROJECTED, "conv_in_bias")
            .expect("in bias allocation should succeed");
        let causal_kernel = graph
            .new_tensor_3d_f32(CAUSAL_KERNEL, 1, DIM, "conv_causal_kernel")
            .expect("causal kernel allocation should succeed");
        let causal_bias = graph
            .new_tensor_1d_f32(DIM, "conv_causal_bias")
            .expect("causal bias allocation should succeed");
        let chunk_kernel = graph
            .new_tensor_3d_f32(CHUNK_KERNEL, 1, DIM, "conv_chunk_kernel")
            .expect("chunk kernel allocation should succeed");
        let chunk_bias = graph
            .new_tensor_1d_f32(DIM, "conv_chunk_bias")
            .expect("chunk bias allocation should succeed");
        let chunk_scale = graph
            .new_tensor_2d_f32(FRAMES, DIM, "conv_chunk_scale")
            .expect("chunk scale allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(DIM, DIM, "conv_out_weight")
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(DIM, "conv_out_bias")
            .expect("out bias allocation should succeed");
        let swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "conv_swoosh_r_offset")
            .expect("swoosh offset allocation should succeed");
        let swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "conv_swoosh_r_shift")
            .expect("swoosh shift allocation should succeed");
        for tensor in [
            input,
            cache,
            in_weight,
            in_bias,
            causal_kernel,
            causal_bias,
            chunk_kernel,
            chunk_bias,
            chunk_scale,
            out_weight,
            out_bias,
            swoosh_r_offset,
            swoosh_r_shift,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_convolution_module_graph(
            &graph,
            input,
            XasrConvolutionModuleGraphTensors {
                cache,
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                causal_kernel,
                causal_bias,
                chunk_kernel,
                chunk_bias,
                chunk_scale,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
                swoosh_r_offset,
                swoosh_r_shift,
            },
            XasrConvolutionModuleGraphShape {
                dim: DIM,
                frames: FRAMES,
                cache_len: CACHE_LEN,
                causal_kernel_len: CAUSAL_KERNEL,
                chunk_kernel_len: CHUNK_KERNEL,
            },
        )
        .expect("convolution module graph should build");
        graph
            .set_output(output.rows)
            .expect("conv rows output should set");
        graph
            .set_output(output.new_cache)
            .expect("conv cache output should set");

        graph
            .set_f32_slice(input, &input_values, "conv_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(cache, &cache_values, "conv_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(in_weight, &in_weight_values, "conv_in_weight")
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &in_bias_values, "conv_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(causal_kernel, &causal_kernel_values, "conv_causal_kernel")
            .expect("causal kernel upload should succeed");
        graph
            .set_f32_slice(causal_bias, &causal_bias_values, "conv_causal_bias")
            .expect("causal bias upload should succeed");
        graph
            .set_f32_slice(chunk_kernel, &chunk_kernel_values, "conv_chunk_kernel")
            .expect("chunk kernel upload should succeed");
        graph
            .set_f32_slice(chunk_bias, &chunk_bias_values, "conv_chunk_bias")
            .expect("chunk bias upload should succeed");
        graph
            .set_f32_slice(chunk_scale, &chunk_scale_values, "conv_chunk_scale")
            .expect("chunk scale upload should succeed");
        graph
            .set_f32_slice(out_weight, &out_weight_values, "conv_out_weight")
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &out_bias_values, "conv_out_bias")
            .expect("out bias upload should succeed");
        graph
            .set_f32_slice(swoosh_r_offset, &[SWOOSH_R_OFFSET], "conv_swoosh_r_offset")
            .expect("swoosh offset upload should succeed");
        graph
            .set_f32_slice(swoosh_r_shift, &[SWOOSH_R_SHIFT], "conv_swoosh_r_shift")
            .expect("swoosh shift upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, DIM * FRAMES),
                (output.new_cache, DIM * CACHE_LEN),
            ])
            .expect("convolution module graph should compute");
        let expected = convolution_module_streaming_reference(
            &weights,
            &input_values,
            FRAMES,
            DIM,
            Some(&cache_values),
            None,
        )
        .expect("reference convolution module");
        assert_max_abs_diff(
            "convolution module rows",
            &actual[0],
            &expected.rows,
            1.0e-5,
        );
        assert_max_abs_diff(
            "convolution module new cache",
            &actual[1],
            &expected.new_cache,
            1.0e-5,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML convolution helper with exported ONNX debug tensors"]
    fn ggml_convolution_module_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-second-half-debug-480ms.add18.f32");
        let expected_path = root.join("oracle-layer0-second-half-debug-480ms.conv2.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing convolution module oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let conv = &weights.stacks[0].layers[0].conv_module2;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);

        let causal_kernel = conv.depthwise_causal_conv.weight.dims[0];
        let chunk_kernel = conv.depthwise_chunkwise_conv.weight.dims[0];
        let cache_len = chunk_kernel / 2;
        let cache_values = vec![0.0_f32; dim * cache_len];
        let mut chunk_scale_values = vec![0.0_f32; dim * frames];
        for c in 0..dim {
            for t in 0..frames {
                chunk_scale_values[c * frames + t] =
                    chunkwise_conv_scale_for_test(conv, c, t, frames).expect("chunkwise scale");
            }
        }
        let reference = convolution_module_streaming_reference(
            conv,
            &input_values,
            frames,
            dim,
            Some(&cache_values),
            None,
        )
        .expect("reference convolution module");

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "conv2_input")
            .expect("input allocation should succeed");
        let cache = graph
            .new_tensor_2d_f32(cache_len, dim, "conv2_cache")
            .expect("cache allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(
                conv.in_proj.weight.input_dim,
                conv.in_proj.weight.output_dim,
                "conv2_in_weight",
            )
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(conv.in_proj.bias.len(), "conv2_in_bias")
            .expect("in bias allocation should succeed");
        let causal_kernel_t = graph
            .new_tensor_3d_f32(causal_kernel, 1, dim, "conv2_causal_kernel")
            .expect("causal kernel allocation should succeed");
        let causal_bias = graph
            .new_tensor_1d_f32(dim, "conv2_causal_bias")
            .expect("causal bias allocation should succeed");
        let chunk_kernel_t = graph
            .new_tensor_3d_f32(chunk_kernel, 1, dim, "conv2_chunk_kernel")
            .expect("chunk kernel allocation should succeed");
        let chunk_bias = graph
            .new_tensor_1d_f32(dim, "conv2_chunk_bias")
            .expect("chunk bias allocation should succeed");
        let chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "conv2_chunk_scale")
            .expect("chunk scale allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(
                conv.out_proj.weight.input_dim,
                conv.out_proj.weight.output_dim,
                "conv2_out_weight",
            )
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(conv.out_proj.bias.len(), "conv2_out_bias")
            .expect("out bias allocation should succeed");
        let swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "conv2_swoosh_r_offset")
            .expect("swoosh offset allocation should succeed");
        let swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "conv2_swoosh_r_shift")
            .expect("swoosh shift allocation should succeed");
        for tensor in [
            input,
            cache,
            in_weight,
            in_bias,
            causal_kernel_t,
            causal_bias,
            chunk_kernel_t,
            chunk_bias,
            chunk_scale,
            out_weight,
            out_bias,
            swoosh_r_offset,
            swoosh_r_shift,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_convolution_module_graph(
            &graph,
            input,
            XasrConvolutionModuleGraphTensors {
                cache,
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                causal_kernel: causal_kernel_t,
                causal_bias,
                chunk_kernel: chunk_kernel_t,
                chunk_bias,
                chunk_scale,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
                swoosh_r_offset,
                swoosh_r_shift,
            },
            XasrConvolutionModuleGraphShape {
                dim,
                frames,
                cache_len,
                causal_kernel_len: causal_kernel,
                chunk_kernel_len: chunk_kernel,
            },
        )
        .expect("convolution module graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_cache)
            .expect("cache output should set");

        graph
            .set_f32_slice(input, &input_values, "conv2_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(cache, &cache_values, "conv2_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(in_weight, &conv.in_proj.weight.values, "conv2_in_weight")
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &conv.in_proj.bias, "conv2_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(
                causal_kernel_t,
                &conv.depthwise_causal_conv.weight.values,
                "conv2_causal_kernel",
            )
            .expect("causal kernel upload should succeed");
        graph
            .set_f32_slice(
                causal_bias,
                &conv.depthwise_causal_conv.bias,
                "conv2_causal_bias",
            )
            .expect("causal bias upload should succeed");
        graph
            .set_f32_slice(
                chunk_kernel_t,
                &conv.depthwise_chunkwise_conv.weight.values,
                "conv2_chunk_kernel",
            )
            .expect("chunk kernel upload should succeed");
        graph
            .set_f32_slice(
                chunk_bias,
                &conv.depthwise_chunkwise_conv.bias,
                "conv2_chunk_bias",
            )
            .expect("chunk bias upload should succeed");
        graph
            .set_f32_slice(chunk_scale, &chunk_scale_values, "conv2_chunk_scale")
            .expect("chunk scale upload should succeed");
        graph
            .set_f32_slice(out_weight, &conv.out_proj.weight.values, "conv2_out_weight")
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &conv.out_proj.bias, "conv2_out_bias")
            .expect("out bias upload should succeed");
        graph
            .set_f32_slice(swoosh_r_offset, &[SWOOSH_R_OFFSET], "conv2_swoosh_r_offset")
            .expect("swoosh offset upload should succeed");
        graph
            .set_f32_slice(swoosh_r_shift, &[SWOOSH_R_SHIFT], "conv2_swoosh_r_shift")
            .expect("swoosh shift upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, dim * frames),
                (output.new_cache, dim * cache_len),
            ])
            .expect("convolution module graph should compute");
        assert_max_abs_diff("conv2 graph ONNX parity", &actual[0], &expected, 2.0e-2);
        assert_max_abs_diff(
            "conv2 graph new cache",
            &actual[1],
            &reference.new_cache,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML layer tail helper with exported ONNX debug tensors"]
    fn ggml_layer_tail_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let original_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let add18_path = root.join("oracle-layer0-second-half-debug-480ms.add18.f32");
        let expected_path = root.join("oracle-layer0-second-half-debug-480ms.bypass.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !original_path.exists()
            || !add18_path.exists()
            || !expected_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing layer tail oracle files");
            return;
        }

        let original_values = read_f32_file(&original_path);
        let add18_values = read_f32_file(&add18_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let layer = &weights.stacks[0].layers[0];
        let conv = &layer.conv_module2;
        let ff3 = &layer.feed_forward3;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        for values in [&original_values, &add18_values, &expected] {
            assert_eq!(values.len(), frames * dim);
        }

        let causal_kernel = conv.depthwise_causal_conv.weight.dims[0];
        let chunk_kernel = conv.depthwise_chunkwise_conv.weight.dims[0];
        let cache_len = chunk_kernel / 2;
        let cache_values = vec![0.0_f32; dim * cache_len];
        let mut chunk_scale_values = vec![0.0_f32; dim * frames];
        for c in 0..dim {
            for t in 0..frames {
                chunk_scale_values[c * frames + t] =
                    chunkwise_conv_scale_for_test(conv, c, t, frames).expect("chunkwise scale");
            }
        }
        let conv_reference = convolution_module_streaming_reference(
            conv,
            &add18_values,
            frames,
            dim,
            Some(&cache_values),
            None,
        )
        .expect("reference conv2");

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let original = graph
            .new_tensor_2d_f32(dim, frames, "tail_original")
            .expect("original allocation should succeed");
        let add18 = graph
            .new_tensor_2d_f32(dim, frames, "tail_add18")
            .expect("add18 allocation should succeed");
        let conv_cache = graph
            .new_tensor_2d_f32(cache_len, dim, "tail_conv2_cache")
            .expect("cache allocation should succeed");
        let conv_in_weight = graph
            .new_tensor_2d_f32(
                conv.in_proj.weight.input_dim,
                conv.in_proj.weight.output_dim,
                "tail_conv2_in_weight",
            )
            .expect("conv in weight allocation should succeed");
        let conv_in_bias = graph
            .new_tensor_1d_f32(conv.in_proj.bias.len(), "tail_conv2_in_bias")
            .expect("conv in bias allocation should succeed");
        let conv_causal_kernel = graph
            .new_tensor_3d_f32(causal_kernel, 1, dim, "tail_conv2_causal_kernel")
            .expect("causal kernel allocation should succeed");
        let conv_causal_bias = graph
            .new_tensor_1d_f32(dim, "tail_conv2_causal_bias")
            .expect("causal bias allocation should succeed");
        let conv_chunk_kernel = graph
            .new_tensor_3d_f32(chunk_kernel, 1, dim, "tail_conv2_chunk_kernel")
            .expect("chunk kernel allocation should succeed");
        let conv_chunk_bias = graph
            .new_tensor_1d_f32(dim, "tail_conv2_chunk_bias")
            .expect("chunk bias allocation should succeed");
        let conv_chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "tail_conv2_chunk_scale")
            .expect("chunk scale allocation should succeed");
        let conv_out_weight = graph
            .new_tensor_2d_f32(
                conv.out_proj.weight.input_dim,
                conv.out_proj.weight.output_dim,
                "tail_conv2_out_weight",
            )
            .expect("conv out weight allocation should succeed");
        let conv_out_bias = graph
            .new_tensor_1d_f32(conv.out_proj.bias.len(), "tail_conv2_out_bias")
            .expect("conv out bias allocation should succeed");
        let conv_swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "tail_conv2_swoosh_r_offset")
            .expect("conv swoosh offset allocation should succeed");
        let conv_swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "tail_conv2_swoosh_r_shift")
            .expect("conv swoosh shift allocation should succeed");
        let ff3_in_weight = graph
            .new_tensor_2d_f32(
                ff3.in_proj.weight.input_dim,
                ff3.in_proj.weight.output_dim,
                "tail_ff3_in_weight",
            )
            .expect("ff3 in weight allocation should succeed");
        let ff3_in_bias = graph
            .new_tensor_1d_f32(ff3.in_proj.bias.len(), "tail_ff3_in_bias")
            .expect("ff3 in bias allocation should succeed");
        let ff3_out_weight = graph
            .new_tensor_2d_f32(
                ff3.out_proj.weight.input_dim,
                ff3.out_proj.weight.output_dim,
                "tail_ff3_out_weight",
            )
            .expect("ff3 out weight allocation should succeed");
        let ff3_out_bias = graph
            .new_tensor_1d_f32(ff3.out_proj.bias.len(), "tail_ff3_out_bias")
            .expect("ff3 out bias allocation should succeed");
        let ff3_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "tail_ff3_swoosh_l_offset")
            .expect("ff3 swoosh offset allocation should succeed");
        let ff3_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "tail_ff3_swoosh_l_shift")
            .expect("ff3 swoosh shift allocation should succeed");
        let norm_bias = graph
            .new_tensor_1d_f32(dim, "tail_norm_bias")
            .expect("norm bias allocation should succeed");
        let bypass_scale = graph
            .new_tensor_1d_f32(dim, "tail_bypass_scale")
            .expect("bypass scale allocation should succeed");
        for tensor in [
            original,
            add18,
            conv_cache,
            conv_in_weight,
            conv_in_bias,
            conv_causal_kernel,
            conv_causal_bias,
            conv_chunk_kernel,
            conv_chunk_bias,
            conv_chunk_scale,
            conv_out_weight,
            conv_out_bias,
            conv_swoosh_r_offset,
            conv_swoosh_r_shift,
            ff3_in_weight,
            ff3_in_bias,
            ff3_out_weight,
            ff3_out_bias,
            ff3_swoosh_l_offset,
            ff3_swoosh_l_shift,
            norm_bias,
            bypass_scale,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_layer_tail_graph(
            &graph,
            original,
            add18,
            XasrLayerTailGraphTensors {
                conv_module2: XasrConvolutionModuleGraphTensors {
                    cache: conv_cache,
                    in_proj_weight: conv_in_weight,
                    in_proj_bias: conv_in_bias,
                    causal_kernel: conv_causal_kernel,
                    causal_bias: conv_causal_bias,
                    chunk_kernel: conv_chunk_kernel,
                    chunk_bias: conv_chunk_bias,
                    chunk_scale: conv_chunk_scale,
                    out_proj_weight: conv_out_weight,
                    out_proj_bias: conv_out_bias,
                    swoosh_r_offset: conv_swoosh_r_offset,
                    swoosh_r_shift: conv_swoosh_r_shift,
                },
                feed_forward3: XasrFeedForwardGraphTensors {
                    in_proj_weight: ff3_in_weight,
                    in_proj_bias: ff3_in_bias,
                    out_proj_weight: ff3_out_weight,
                    out_proj_bias: ff3_out_bias,
                    swoosh_l_offset: ff3_swoosh_l_offset,
                    swoosh_l_shift: ff3_swoosh_l_shift,
                },
                norm_bias,
                bypass_scale,
            },
            XasrLayerTailGraphShape {
                dim,
                frames,
                conv_cache_len: cache_len,
                conv_causal_kernel_len: causal_kernel,
                conv_chunk_kernel_len: chunk_kernel,
                norm_log_scale: layer.norm_log_scale[0],
            },
        )
        .expect("layer tail graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_conv2_cache)
            .expect("conv cache output should set");

        graph
            .set_f32_slice(original, &original_values, "tail_original")
            .expect("original upload should succeed");
        graph
            .set_f32_slice(add18, &add18_values, "tail_add18")
            .expect("add18 upload should succeed");
        graph
            .set_f32_slice(conv_cache, &cache_values, "tail_conv2_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(
                conv_in_weight,
                &conv.in_proj.weight.values,
                "tail_conv2_in_weight",
            )
            .expect("conv in weight upload should succeed");
        graph
            .set_f32_slice(conv_in_bias, &conv.in_proj.bias, "tail_conv2_in_bias")
            .expect("conv in bias upload should succeed");
        graph
            .set_f32_slice(
                conv_causal_kernel,
                &conv.depthwise_causal_conv.weight.values,
                "tail_conv2_causal_kernel",
            )
            .expect("conv causal kernel upload should succeed");
        graph
            .set_f32_slice(
                conv_causal_bias,
                &conv.depthwise_causal_conv.bias,
                "tail_conv2_causal_bias",
            )
            .expect("conv causal bias upload should succeed");
        graph
            .set_f32_slice(
                conv_chunk_kernel,
                &conv.depthwise_chunkwise_conv.weight.values,
                "tail_conv2_chunk_kernel",
            )
            .expect("conv chunk kernel upload should succeed");
        graph
            .set_f32_slice(
                conv_chunk_bias,
                &conv.depthwise_chunkwise_conv.bias,
                "tail_conv2_chunk_bias",
            )
            .expect("conv chunk bias upload should succeed");
        graph
            .set_f32_slice(
                conv_chunk_scale,
                &chunk_scale_values,
                "tail_conv2_chunk_scale",
            )
            .expect("conv chunk scale upload should succeed");
        graph
            .set_f32_slice(
                conv_out_weight,
                &conv.out_proj.weight.values,
                "tail_conv2_out_weight",
            )
            .expect("conv out weight upload should succeed");
        graph
            .set_f32_slice(conv_out_bias, &conv.out_proj.bias, "tail_conv2_out_bias")
            .expect("conv out bias upload should succeed");
        graph
            .set_f32_slice(
                conv_swoosh_r_offset,
                &[SWOOSH_R_OFFSET],
                "tail_conv2_swoosh_r_offset",
            )
            .expect("conv swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                conv_swoosh_r_shift,
                &[SWOOSH_R_SHIFT],
                "tail_conv2_swoosh_r_shift",
            )
            .expect("conv swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                ff3_in_weight,
                &ff3.in_proj.weight.values,
                "tail_ff3_in_weight",
            )
            .expect("ff3 in weight upload should succeed");
        graph
            .set_f32_slice(ff3_in_bias, &ff3.in_proj.bias, "tail_ff3_in_bias")
            .expect("ff3 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff3_out_weight,
                &ff3.out_proj.weight.values,
                "tail_ff3_out_weight",
            )
            .expect("ff3 out weight upload should succeed");
        graph
            .set_f32_slice(ff3_out_bias, &ff3.out_proj.bias, "tail_ff3_out_bias")
            .expect("ff3 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff3_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "tail_ff3_swoosh_l_offset",
            )
            .expect("ff3 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff3_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "tail_ff3_swoosh_l_shift",
            )
            .expect("ff3 swoosh shift upload should succeed");
        graph
            .set_f32_slice(norm_bias, &layer.norm_bias, "tail_norm_bias")
            .expect("norm bias upload should succeed");
        graph
            .set_f32_slice(bypass_scale, &layer.bypass_scale, "tail_bypass_scale")
            .expect("bypass scale upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, dim * frames),
                (output.new_conv2_cache, dim * cache_len),
            ])
            .expect("layer tail graph should compute");
        assert_max_abs_diff(
            "layer tail graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "layer tail graph conv2 cache",
            &actual[1],
            &conv_reference.new_cache,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML attention-weights helper with exported ONNX debug tensors"]
    fn ggml_attention_weights_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-layer0-debug-480ms.softmax.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing attention-weights oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let attn = &weights.stacks[0].layers[0].self_attn_weights;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        let left_context_len = metadata.left_context_len[0];
        let valid_left_context_len = 61;
        let num_heads = metadata.num_heads[0];
        let query_head_dim = metadata.query_head_dims[0];
        let query_dim = num_heads * query_head_dim;
        let k_len = left_context_len + frames;
        let pos_dim = attn.linear_pos.input_dim;
        let pos_output_dim = attn.linear_pos.output_dim;
        let rel_len = left_context_len + 2 * frames - 1;
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), num_heads * frames * k_len);
        let key_padding_mask =
            streaming_key_padding_mask(left_context_len, frames, valid_left_context_len)
                .expect("key padding mask");
        let reference = self_attention_weights_streaming_reference(
            attn,
            &input_values,
            frames,
            dim,
            num_heads,
            query_head_dim,
            left_context_len,
            None,
            Some(&key_padding_mask),
        )
        .expect("reference attention weights");
        let cache_values = vec![0.0_f32; left_context_len * query_dim];
        let pos_embedding_values =
            compact_relative_positional_encoding_for_test(frames, left_context_len, pos_dim);
        let mut mask_values = vec![0.0_f32; frames * k_len];
        for target in 0..frames {
            for source in 0..k_len {
                if key_padding_mask[source] {
                    mask_values[target * k_len + source] = -1000.0;
                }
            }
        }

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "attn_weights_input")
            .expect("input allocation should succeed");
        let cache = graph
            .new_tensor_2d_f32(query_dim, left_context_len, "attn_weights_cache")
            .expect("cache allocation should succeed");
        let mask = graph
            .new_tensor_2d_f32(k_len, frames, "attn_weights_mask")
            .expect("mask allocation should succeed");
        let pos_embedding = graph
            .new_tensor_2d_f32(pos_dim, rel_len, "attn_weights_pos_embedding")
            .expect("pos embedding allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(
                attn.in_proj.weight.input_dim,
                attn.in_proj.weight.output_dim,
                "attn_weights_in_weight",
            )
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(attn.in_proj.bias.len(), "attn_weights_in_bias")
            .expect("in bias allocation should succeed");
        let linear_pos_weight = graph
            .new_tensor_2d_f32(
                attn.linear_pos.input_dim,
                attn.linear_pos.output_dim,
                "attn_weights_linear_pos",
            )
            .expect("linear pos allocation should succeed");
        for tensor in [
            input,
            cache,
            mask,
            pos_embedding,
            in_weight,
            in_bias,
            linear_pos_weight,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_self_attention_weights_graph(
            &graph,
            input,
            XasrSelfAttentionWeightsGraphTensors {
                cache,
                mask,
                pos_embedding,
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                linear_pos_weight,
            },
            XasrSelfAttentionWeightsGraphShape {
                dim,
                frames,
                left_context_len,
                num_heads,
                query_head_dim,
                pos_dim,
                pos_output_dim,
            },
        )
        .expect("attention weights graph");
        graph
            .set_output(output.weights)
            .expect("weights output should set");
        graph
            .set_output(output.new_cache)
            .expect("cache output should set");

        graph
            .set_f32_slice(input, &input_values, "attn_weights_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(cache, &cache_values, "attn_weights_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(mask, &mask_values, "attn_weights_mask")
            .expect("mask upload should succeed");
        graph
            .set_f32_slice(
                pos_embedding,
                &pos_embedding_values,
                "attn_weights_pos_embedding",
            )
            .expect("pos embedding upload should succeed");
        graph
            .set_f32_slice(
                in_weight,
                &attn.in_proj.weight.values,
                "attn_weights_in_weight",
            )
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &attn.in_proj.bias, "attn_weights_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(
                linear_pos_weight,
                &attn.linear_pos.values,
                "attn_weights_linear_pos",
            )
            .expect("linear pos upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.weights, num_heads * frames * k_len),
                (output.new_cache, left_context_len * query_dim),
            ])
            .expect("attention weights graph should compute");
        assert_max_abs_diff(
            "attention weights graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "attention weights graph new cache",
            &actual[1],
            &reference.new_cached_key,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML layer-head helper with exported ONNX debug tensors"]
    fn ggml_layer_head_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-layer0-second-half-debug-480ms.bypass_mid.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing layer-head oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let layer = &weights.stacks[0].layers[0];
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        let left_context_len = metadata.left_context_len[0];
        let valid_left_context_len = 61;
        let num_heads = metadata.num_heads[0];
        let query_head_dim = metadata.query_head_dims[0];
        let query_dim = num_heads * query_head_dim;
        let k_len = left_context_len + frames;
        let attn = &layer.self_attn_weights;
        let pos_dim = attn.linear_pos.input_dim;
        let pos_output_dim = attn.linear_pos.output_dim;
        let rel_len = left_context_len + 2 * frames - 1;
        let nonlin = &layer.nonlin_attention;
        let nonlin_hidden_dim = nonlin.out_proj.weight.input_dim;
        let self1 = &layer.self_attn1;
        let self1_value_dim = self1.in_proj.weight.output_dim;
        let conv1 = &layer.conv_module1;
        let conv1_causal_kernel = conv1.depthwise_causal_conv.weight.dims[0];
        let conv1_chunk_kernel = conv1.depthwise_chunkwise_conv.weight.dims[0];
        let conv1_cache_len = conv1_chunk_kernel / 2;
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);

        let key_padding_mask =
            streaming_key_padding_mask(left_context_len, frames, valid_left_context_len)
                .expect("key padding mask");
        let attention_reference = self_attention_weights_streaming_reference(
            attn,
            &input_values,
            frames,
            dim,
            num_heads,
            query_head_dim,
            left_context_len,
            None,
            Some(&key_padding_mask),
        )
        .expect("attention weights reference");
        let ff1 = feed_forward_reference(&layer.feed_forward1, &input_values, frames, dim)
            .expect("ff1 reference");
        let add6 = add_same_shape_for_test(&input_values, &ff1);
        let nonlin_reference = nonlin_attention_streaming_reference(
            nonlin,
            &add6,
            &attention_reference.weights[..frames * k_len],
            frames,
            dim,
            1,
            left_context_len,
            None,
        )
        .expect("nonlin reference");
        let add8 = add_same_shape_for_test(&add6, &nonlin_reference.rows);
        let self1_reference = self_attention_streaming_reference(
            self1,
            &add8,
            &attention_reference.weights,
            frames,
            dim,
            num_heads,
            left_context_len,
            None,
        )
        .expect("self1 reference");
        let add10 = add_same_shape_for_test(&add8, &self1_reference.rows);
        let conv1_reference =
            convolution_module_streaming_reference(conv1, &add10, frames, dim, None, None)
                .expect("conv1 reference");
        let add15 = add_same_shape_for_test(&add10, &conv1_reference.rows);
        let ff2 = feed_forward_reference(&layer.feed_forward2, &add15, frames, dim)
            .expect("ff2 reference");
        let add16 = add_same_shape_for_test(&add15, &ff2);
        let bypass_mid =
            bypass_reference(&input_values, &add16, &layer.bypass_mid_scale, frames, dim)
                .expect("bypass_mid reference");
        assert_max_abs_diff("layer-head rust reference", &bypass_mid, &expected, 2.0e-2);

        let attn_cache_values = vec![0.0_f32; left_context_len * query_dim];
        let nonlin_cache_values = vec![0.0_f32; left_context_len * nonlin_hidden_dim];
        let self1_cache_values = vec![0.0_f32; left_context_len * self1_value_dim];
        let conv1_cache_values = vec![0.0_f32; dim * conv1_cache_len];
        let pos_embedding_values =
            compact_relative_positional_encoding_for_test(frames, left_context_len, pos_dim);
        let mut mask_values = vec![0.0_f32; frames * k_len];
        for target in 0..frames {
            for source in 0..k_len {
                if key_padding_mask[source] {
                    mask_values[target * k_len + source] = -1000.0;
                }
            }
        }
        let mut conv1_chunk_scale_values = vec![0.0_f32; dim * frames];
        for c in 0..dim {
            for t in 0..frames {
                conv1_chunk_scale_values[c * frames + t] =
                    chunkwise_conv_scale_for_test(conv1, c, t, frames).expect("chunkwise scale");
            }
        }

        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 16 * 1024 * 1024;
        config.graph_size = 16_384;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "head_input")
            .expect("input allocation should succeed");
        let ff1_in_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward1.in_proj.weight.input_dim,
                layer.feed_forward1.in_proj.weight.output_dim,
                "head_ff1_in_w",
            )
            .expect("ff1 in weight allocation should succeed");
        let ff1_in_b = graph
            .new_tensor_1d_f32(layer.feed_forward1.in_proj.bias.len(), "head_ff1_in_b")
            .expect("ff1 in bias allocation should succeed");
        let ff1_out_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward1.out_proj.weight.input_dim,
                layer.feed_forward1.out_proj.weight.output_dim,
                "head_ff1_out_w",
            )
            .expect("ff1 out weight allocation should succeed");
        let ff1_out_b = graph
            .new_tensor_1d_f32(layer.feed_forward1.out_proj.bias.len(), "head_ff1_out_b")
            .expect("ff1 out bias allocation should succeed");
        let ff1_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "head_ff1_swoosh_l_offset")
            .expect("ff1 swoosh offset allocation should succeed");
        let ff1_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "head_ff1_swoosh_l_shift")
            .expect("ff1 swoosh shift allocation should succeed");
        let attn_cache = graph
            .new_tensor_2d_f32(query_dim, left_context_len, "head_attn_cache")
            .expect("attention cache allocation should succeed");
        let attn_mask = graph
            .new_tensor_2d_f32(k_len, frames, "head_attn_mask")
            .expect("attention mask allocation should succeed");
        let attn_pos = graph
            .new_tensor_2d_f32(pos_dim, rel_len, "head_attn_pos")
            .expect("attention position allocation should succeed");
        let attn_in_w = graph
            .new_tensor_2d_f32(
                attn.in_proj.weight.input_dim,
                attn.in_proj.weight.output_dim,
                "head_attn_in_w",
            )
            .expect("attention in weight allocation should succeed");
        let attn_in_b = graph
            .new_tensor_1d_f32(attn.in_proj.bias.len(), "head_attn_in_b")
            .expect("attention in bias allocation should succeed");
        let attn_pos_w = graph
            .new_tensor_2d_f32(
                attn.linear_pos.input_dim,
                attn.linear_pos.output_dim,
                "head_attn_pos_w",
            )
            .expect("attention pos weight allocation should succeed");
        let nonlin_cache = graph
            .new_tensor_2d_f32(nonlin_hidden_dim, left_context_len, "head_nonlin_cache")
            .expect("nonlin cache allocation should succeed");
        let nonlin_in_w = graph
            .new_tensor_2d_f32(
                nonlin.in_proj.weight.input_dim,
                nonlin.in_proj.weight.output_dim,
                "head_nonlin_in_w",
            )
            .expect("nonlin in weight allocation should succeed");
        let nonlin_in_b = graph
            .new_tensor_1d_f32(nonlin.in_proj.bias.len(), "head_nonlin_in_b")
            .expect("nonlin in bias allocation should succeed");
        let nonlin_out_w = graph
            .new_tensor_2d_f32(
                nonlin.out_proj.weight.input_dim,
                nonlin.out_proj.weight.output_dim,
                "head_nonlin_out_w",
            )
            .expect("nonlin out weight allocation should succeed");
        let nonlin_out_b = graph
            .new_tensor_1d_f32(nonlin.out_proj.bias.len(), "head_nonlin_out_b")
            .expect("nonlin out bias allocation should succeed");
        let self1_cache = graph
            .new_tensor_2d_f32(self1_value_dim, left_context_len, "head_self1_cache")
            .expect("self1 cache allocation should succeed");
        let self1_in_w = graph
            .new_tensor_2d_f32(
                self1.in_proj.weight.input_dim,
                self1.in_proj.weight.output_dim,
                "head_self1_in_w",
            )
            .expect("self1 in weight allocation should succeed");
        let self1_in_b = graph
            .new_tensor_1d_f32(self1.in_proj.bias.len(), "head_self1_in_b")
            .expect("self1 in bias allocation should succeed");
        let self1_out_w = graph
            .new_tensor_2d_f32(
                self1.out_proj.weight.input_dim,
                self1.out_proj.weight.output_dim,
                "head_self1_out_w",
            )
            .expect("self1 out weight allocation should succeed");
        let self1_out_b = graph
            .new_tensor_1d_f32(self1.out_proj.bias.len(), "head_self1_out_b")
            .expect("self1 out bias allocation should succeed");
        let conv1_cache = graph
            .new_tensor_2d_f32(conv1_cache_len, dim, "head_conv1_cache")
            .expect("conv1 cache allocation should succeed");
        let conv1_in_w = graph
            .new_tensor_2d_f32(
                conv1.in_proj.weight.input_dim,
                conv1.in_proj.weight.output_dim,
                "head_conv1_in_w",
            )
            .expect("conv1 in weight allocation should succeed");
        let conv1_in_b = graph
            .new_tensor_1d_f32(conv1.in_proj.bias.len(), "head_conv1_in_b")
            .expect("conv1 in bias allocation should succeed");
        let conv1_causal_w = graph
            .new_tensor_3d_f32(conv1_causal_kernel, 1, dim, "head_conv1_causal_w")
            .expect("conv1 causal weight allocation should succeed");
        let conv1_causal_b = graph
            .new_tensor_1d_f32(dim, "head_conv1_causal_b")
            .expect("conv1 causal bias allocation should succeed");
        let conv1_chunk_w = graph
            .new_tensor_3d_f32(conv1_chunk_kernel, 1, dim, "head_conv1_chunk_w")
            .expect("conv1 chunk weight allocation should succeed");
        let conv1_chunk_b = graph
            .new_tensor_1d_f32(dim, "head_conv1_chunk_b")
            .expect("conv1 chunk bias allocation should succeed");
        let conv1_chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "head_conv1_chunk_scale")
            .expect("conv1 chunk scale allocation should succeed");
        let conv1_out_w = graph
            .new_tensor_2d_f32(
                conv1.out_proj.weight.input_dim,
                conv1.out_proj.weight.output_dim,
                "head_conv1_out_w",
            )
            .expect("conv1 out weight allocation should succeed");
        let conv1_out_b = graph
            .new_tensor_1d_f32(conv1.out_proj.bias.len(), "head_conv1_out_b")
            .expect("conv1 out bias allocation should succeed");
        let conv1_swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "head_conv1_swoosh_r_offset")
            .expect("conv1 swoosh offset allocation should succeed");
        let conv1_swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "head_conv1_swoosh_r_shift")
            .expect("conv1 swoosh shift allocation should succeed");
        let ff2_in_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward2.in_proj.weight.input_dim,
                layer.feed_forward2.in_proj.weight.output_dim,
                "head_ff2_in_w",
            )
            .expect("ff2 in weight allocation should succeed");
        let ff2_in_b = graph
            .new_tensor_1d_f32(layer.feed_forward2.in_proj.bias.len(), "head_ff2_in_b")
            .expect("ff2 in bias allocation should succeed");
        let ff2_out_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward2.out_proj.weight.input_dim,
                layer.feed_forward2.out_proj.weight.output_dim,
                "head_ff2_out_w",
            )
            .expect("ff2 out weight allocation should succeed");
        let ff2_out_b = graph
            .new_tensor_1d_f32(layer.feed_forward2.out_proj.bias.len(), "head_ff2_out_b")
            .expect("ff2 out bias allocation should succeed");
        let ff2_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "head_ff2_swoosh_l_offset")
            .expect("ff2 swoosh offset allocation should succeed");
        let ff2_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "head_ff2_swoosh_l_shift")
            .expect("ff2 swoosh shift allocation should succeed");
        let bypass_mid_scale = graph
            .new_tensor_1d_f32(dim, "head_bypass_mid_scale")
            .expect("bypass scale allocation should succeed");
        for tensor in [
            input,
            ff1_in_w,
            ff1_in_b,
            ff1_out_w,
            ff1_out_b,
            ff1_swoosh_l_offset,
            ff1_swoosh_l_shift,
            attn_cache,
            attn_mask,
            attn_pos,
            attn_in_w,
            attn_in_b,
            attn_pos_w,
            nonlin_cache,
            nonlin_in_w,
            nonlin_in_b,
            nonlin_out_w,
            nonlin_out_b,
            self1_cache,
            self1_in_w,
            self1_in_b,
            self1_out_w,
            self1_out_b,
            conv1_cache,
            conv1_in_w,
            conv1_in_b,
            conv1_causal_w,
            conv1_causal_b,
            conv1_chunk_w,
            conv1_chunk_b,
            conv1_chunk_scale,
            conv1_out_w,
            conv1_out_b,
            conv1_swoosh_r_offset,
            conv1_swoosh_r_shift,
            ff2_in_w,
            ff2_in_b,
            ff2_out_w,
            ff2_out_b,
            ff2_swoosh_l_offset,
            ff2_swoosh_l_shift,
            bypass_mid_scale,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }

        let output = apply_layer_head_graph(
            &graph,
            input,
            XasrLayerHeadGraphTensors {
                feed_forward1: XasrFeedForwardGraphTensors {
                    in_proj_weight: ff1_in_w,
                    in_proj_bias: ff1_in_b,
                    out_proj_weight: ff1_out_w,
                    out_proj_bias: ff1_out_b,
                    swoosh_l_offset: ff1_swoosh_l_offset,
                    swoosh_l_shift: ff1_swoosh_l_shift,
                },
                attention_weights: XasrSelfAttentionWeightsGraphTensors {
                    cache: attn_cache,
                    mask: attn_mask,
                    pos_embedding: attn_pos,
                    in_proj_weight: attn_in_w,
                    in_proj_bias: attn_in_b,
                    linear_pos_weight: attn_pos_w,
                },
                nonlin_cache,
                nonlin_in_proj_weight: nonlin_in_w,
                nonlin_in_proj_bias: nonlin_in_b,
                nonlin_out_proj_weight: nonlin_out_w,
                nonlin_out_proj_bias: nonlin_out_b,
                self1_cache,
                self1_in_proj_weight: self1_in_w,
                self1_in_proj_bias: self1_in_b,
                self1_out_proj_weight: self1_out_w,
                self1_out_proj_bias: self1_out_b,
                conv_module1: XasrConvolutionModuleGraphTensors {
                    cache: conv1_cache,
                    in_proj_weight: conv1_in_w,
                    in_proj_bias: conv1_in_b,
                    causal_kernel: conv1_causal_w,
                    causal_bias: conv1_causal_b,
                    chunk_kernel: conv1_chunk_w,
                    chunk_bias: conv1_chunk_b,
                    chunk_scale: conv1_chunk_scale,
                    out_proj_weight: conv1_out_w,
                    out_proj_bias: conv1_out_b,
                    swoosh_r_offset: conv1_swoosh_r_offset,
                    swoosh_r_shift: conv1_swoosh_r_shift,
                },
                feed_forward2: XasrFeedForwardGraphTensors {
                    in_proj_weight: ff2_in_w,
                    in_proj_bias: ff2_in_b,
                    out_proj_weight: ff2_out_w,
                    out_proj_bias: ff2_out_b,
                    swoosh_l_offset: ff2_swoosh_l_offset,
                    swoosh_l_shift: ff2_swoosh_l_shift,
                },
                bypass_mid_scale,
            },
            XasrLayerHeadGraphShape {
                dim,
                frames,
                left_context_len,
                num_heads,
                query_head_dim,
                pos_dim,
                pos_output_dim,
                nonlin_hidden_dim,
                self1_value_dim,
                conv1_cache_len,
                conv1_causal_kernel_len: conv1_causal_kernel,
                conv1_chunk_kernel_len: conv1_chunk_kernel,
            },
        )
        .expect("layer head graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_cached_key)
            .expect("key cache output should set");
        graph
            .set_output(output.new_cached_nonlin_attention)
            .expect("nonlin cache output should set");
        graph
            .set_output(output.new_cached_val1)
            .expect("val1 cache output should set");
        graph
            .set_output(output.new_cached_conv1)
            .expect("conv1 cache output should set");

        graph
            .set_f32_slice(input, &input_values, "head_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(
                ff1_in_w,
                &layer.feed_forward1.in_proj.weight.values,
                "head_ff1_in_w",
            )
            .expect("ff1 in weight upload should succeed");
        graph
            .set_f32_slice(ff1_in_b, &layer.feed_forward1.in_proj.bias, "head_ff1_in_b")
            .expect("ff1 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff1_out_w,
                &layer.feed_forward1.out_proj.weight.values,
                "head_ff1_out_w",
            )
            .expect("ff1 out weight upload should succeed");
        graph
            .set_f32_slice(
                ff1_out_b,
                &layer.feed_forward1.out_proj.bias,
                "head_ff1_out_b",
            )
            .expect("ff1 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff1_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "head_ff1_swoosh_l_offset",
            )
            .expect("ff1 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff1_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "head_ff1_swoosh_l_shift",
            )
            .expect("ff1 swoosh shift upload should succeed");
        graph
            .set_f32_slice(attn_cache, &attn_cache_values, "head_attn_cache")
            .expect("attn cache upload should succeed");
        graph
            .set_f32_slice(attn_mask, &mask_values, "head_attn_mask")
            .expect("attn mask upload should succeed");
        graph
            .set_f32_slice(attn_pos, &pos_embedding_values, "head_attn_pos")
            .expect("attn pos upload should succeed");
        graph
            .set_f32_slice(attn_in_w, &attn.in_proj.weight.values, "head_attn_in_w")
            .expect("attn in weight upload should succeed");
        graph
            .set_f32_slice(attn_in_b, &attn.in_proj.bias, "head_attn_in_b")
            .expect("attn in bias upload should succeed");
        graph
            .set_f32_slice(attn_pos_w, &attn.linear_pos.values, "head_attn_pos_w")
            .expect("attn pos weight upload should succeed");
        graph
            .set_f32_slice(nonlin_cache, &nonlin_cache_values, "head_nonlin_cache")
            .expect("nonlin cache upload should succeed");
        graph
            .set_f32_slice(
                nonlin_in_w,
                &nonlin.in_proj.weight.values,
                "head_nonlin_in_w",
            )
            .expect("nonlin in weight upload should succeed");
        graph
            .set_f32_slice(nonlin_in_b, &nonlin.in_proj.bias, "head_nonlin_in_b")
            .expect("nonlin in bias upload should succeed");
        graph
            .set_f32_slice(
                nonlin_out_w,
                &nonlin.out_proj.weight.values,
                "head_nonlin_out_w",
            )
            .expect("nonlin out weight upload should succeed");
        graph
            .set_f32_slice(nonlin_out_b, &nonlin.out_proj.bias, "head_nonlin_out_b")
            .expect("nonlin out bias upload should succeed");
        graph
            .set_f32_slice(self1_cache, &self1_cache_values, "head_self1_cache")
            .expect("self1 cache upload should succeed");
        graph
            .set_f32_slice(self1_in_w, &self1.in_proj.weight.values, "head_self1_in_w")
            .expect("self1 in weight upload should succeed");
        graph
            .set_f32_slice(self1_in_b, &self1.in_proj.bias, "head_self1_in_b")
            .expect("self1 in bias upload should succeed");
        graph
            .set_f32_slice(
                self1_out_w,
                &self1.out_proj.weight.values,
                "head_self1_out_w",
            )
            .expect("self1 out weight upload should succeed");
        graph
            .set_f32_slice(self1_out_b, &self1.out_proj.bias, "head_self1_out_b")
            .expect("self1 out bias upload should succeed");
        graph
            .set_f32_slice(conv1_cache, &conv1_cache_values, "head_conv1_cache")
            .expect("conv1 cache upload should succeed");
        graph
            .set_f32_slice(conv1_in_w, &conv1.in_proj.weight.values, "head_conv1_in_w")
            .expect("conv1 in weight upload should succeed");
        graph
            .set_f32_slice(conv1_in_b, &conv1.in_proj.bias, "head_conv1_in_b")
            .expect("conv1 in bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_causal_w,
                &conv1.depthwise_causal_conv.weight.values,
                "head_conv1_causal_w",
            )
            .expect("conv1 causal weight upload should succeed");
        graph
            .set_f32_slice(
                conv1_causal_b,
                &conv1.depthwise_causal_conv.bias,
                "head_conv1_causal_b",
            )
            .expect("conv1 causal bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_w,
                &conv1.depthwise_chunkwise_conv.weight.values,
                "head_conv1_chunk_w",
            )
            .expect("conv1 chunk weight upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_b,
                &conv1.depthwise_chunkwise_conv.bias,
                "head_conv1_chunk_b",
            )
            .expect("conv1 chunk bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_scale,
                &conv1_chunk_scale_values,
                "head_conv1_chunk_scale",
            )
            .expect("conv1 chunk scale upload should succeed");
        graph
            .set_f32_slice(
                conv1_out_w,
                &conv1.out_proj.weight.values,
                "head_conv1_out_w",
            )
            .expect("conv1 out weight upload should succeed");
        graph
            .set_f32_slice(conv1_out_b, &conv1.out_proj.bias, "head_conv1_out_b")
            .expect("conv1 out bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_swoosh_r_offset,
                &[SWOOSH_R_OFFSET],
                "head_conv1_swoosh_r_offset",
            )
            .expect("conv1 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                conv1_swoosh_r_shift,
                &[SWOOSH_R_SHIFT],
                "head_conv1_swoosh_r_shift",
            )
            .expect("conv1 swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                ff2_in_w,
                &layer.feed_forward2.in_proj.weight.values,
                "head_ff2_in_w",
            )
            .expect("ff2 in weight upload should succeed");
        graph
            .set_f32_slice(ff2_in_b, &layer.feed_forward2.in_proj.bias, "head_ff2_in_b")
            .expect("ff2 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff2_out_w,
                &layer.feed_forward2.out_proj.weight.values,
                "head_ff2_out_w",
            )
            .expect("ff2 out weight upload should succeed");
        graph
            .set_f32_slice(
                ff2_out_b,
                &layer.feed_forward2.out_proj.bias,
                "head_ff2_out_b",
            )
            .expect("ff2 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff2_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "head_ff2_swoosh_l_offset",
            )
            .expect("ff2 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff2_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "head_ff2_swoosh_l_shift",
            )
            .expect("ff2 swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                bypass_mid_scale,
                &layer.bypass_mid_scale,
                "head_bypass_mid_scale",
            )
            .expect("bypass scale upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, frames * dim),
                (output.new_cached_key, left_context_len * query_dim),
                (
                    output.new_cached_nonlin_attention,
                    left_context_len * nonlin_hidden_dim,
                ),
                (output.new_cached_val1, left_context_len * self1_value_dim),
                (output.new_cached_conv1, dim * conv1_cache_len),
            ])
            .expect("layer head graph should compute");
        assert_max_abs_diff(
            "layer head graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "layer head graph key cache",
            &actual[1],
            &attention_reference.new_cached_key,
            2.0e-2,
        );
        assert_max_abs_diff(
            "layer head graph nonlin cache",
            &actual[2],
            &nonlin_reference.new_cache,
            2.0e-2,
        );
        assert_max_abs_diff(
            "layer head graph self1 cache",
            &actual[3],
            &self1_reference.new_cache,
            2.0e-2,
        );
        assert_max_abs_diff(
            "layer head graph conv1 cache",
            &actual[4],
            &conv1_reference.new_cache,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML full layer helper with exported ONNX debug tensors"]
    fn ggml_zipformer_layer_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-layer0-second-half-debug-480ms.bypass.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing full-layer oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let layer = &weights.stacks[0].layers[0];
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        let left_context_len = metadata.left_context_len[0];
        let valid_left_context_len = 61;
        let num_heads = metadata.num_heads[0];
        let query_head_dim = metadata.query_head_dims[0];
        let query_dim = num_heads * query_head_dim;
        let k_len = left_context_len + frames;
        let attn = &layer.self_attn_weights;
        let pos_dim = attn.linear_pos.input_dim;
        let pos_output_dim = attn.linear_pos.output_dim;
        let rel_len = left_context_len + 2 * frames - 1;
        let nonlin = &layer.nonlin_attention;
        let nonlin_hidden_dim = nonlin.out_proj.weight.input_dim;
        let self1 = &layer.self_attn1;
        let self1_value_dim = self1.in_proj.weight.output_dim;
        let self2 = &layer.self_attn2;
        let self2_value_dim = self2.in_proj.weight.output_dim;
        let conv1 = &layer.conv_module1;
        let conv1_causal_kernel = conv1.depthwise_causal_conv.weight.dims[0];
        let conv1_chunk_kernel = conv1.depthwise_chunkwise_conv.weight.dims[0];
        let conv1_cache_len = conv1_chunk_kernel / 2;
        let conv2 = &layer.conv_module2;
        let conv2_causal_kernel = conv2.depthwise_causal_conv.weight.dims[0];
        let conv2_chunk_kernel = conv2.depthwise_chunkwise_conv.weight.dims[0];
        let conv2_cache_len = conv2_chunk_kernel / 2;
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);

        let reference = zipformer_layer_streaming_reference(
            layer,
            &input_values,
            frames,
            dim,
            num_heads,
            query_head_dim,
            left_context_len,
            valid_left_context_len,
            XasrZipformerLayerReferenceCaches {
                cached_key: None,
                cached_nonlin_attention: None,
                cached_val1: None,
                cached_val2: None,
                cached_conv1: None,
                cached_conv2: None,
            },
        )
        .expect("zipformer layer reference");
        assert_max_abs_diff(
            "full layer rust reference",
            &reference.rows,
            &expected,
            2.0e-2,
        );

        let key_padding_mask =
            streaming_key_padding_mask(left_context_len, frames, valid_left_context_len)
                .expect("key padding mask");
        let attn_cache_values = vec![0.0_f32; left_context_len * query_dim];
        let nonlin_cache_values = vec![0.0_f32; left_context_len * nonlin_hidden_dim];
        let self1_cache_values = vec![0.0_f32; left_context_len * self1_value_dim];
        let self2_cache_values = vec![0.0_f32; left_context_len * self2_value_dim];
        let conv1_cache_values = vec![0.0_f32; dim * conv1_cache_len];
        let conv2_cache_values = vec![0.0_f32; dim * conv2_cache_len];
        let pos_embedding_values =
            compact_relative_positional_encoding_for_test(frames, left_context_len, pos_dim);
        let mut mask_values = vec![0.0_f32; frames * k_len];
        for target in 0..frames {
            for source in 0..k_len {
                if key_padding_mask[source] {
                    mask_values[target * k_len + source] = -1000.0;
                }
            }
        }
        let mut conv1_chunk_scale_values = vec![0.0_f32; dim * frames];
        let mut conv2_chunk_scale_values = vec![0.0_f32; dim * frames];
        for c in 0..dim {
            for t in 0..frames {
                conv1_chunk_scale_values[c * frames + t] =
                    chunkwise_conv_scale_for_test(conv1, c, t, frames).expect("conv1 scale");
                conv2_chunk_scale_values[c * frames + t] =
                    chunkwise_conv_scale_for_test(conv2, c, t, frames).expect("conv2 scale");
            }
        }

        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 64 * 1024 * 1024;
        config.graph_size = 65_536;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "layer_input")
            .expect("input allocation should succeed");
        let ff1_in_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward1.in_proj.weight.input_dim,
                layer.feed_forward1.in_proj.weight.output_dim,
                "layer_ff1_in_w",
            )
            .expect("ff1 in weight allocation should succeed");
        let ff1_in_b = graph
            .new_tensor_1d_f32(layer.feed_forward1.in_proj.bias.len(), "layer_ff1_in_b")
            .expect("ff1 in bias allocation should succeed");
        let ff1_out_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward1.out_proj.weight.input_dim,
                layer.feed_forward1.out_proj.weight.output_dim,
                "layer_ff1_out_w",
            )
            .expect("ff1 out weight allocation should succeed");
        let ff1_out_b = graph
            .new_tensor_1d_f32(layer.feed_forward1.out_proj.bias.len(), "layer_ff1_out_b")
            .expect("ff1 out bias allocation should succeed");
        let ff1_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "layer_ff1_swoosh_l_offset")
            .expect("ff1 swoosh offset allocation should succeed");
        let ff1_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "layer_ff1_swoosh_l_shift")
            .expect("ff1 swoosh shift allocation should succeed");
        let attn_cache = graph
            .new_tensor_2d_f32(query_dim, left_context_len, "layer_attn_cache")
            .expect("attention cache allocation should succeed");
        let attn_mask = graph
            .new_tensor_2d_f32(k_len, frames, "layer_attn_mask")
            .expect("attention mask allocation should succeed");
        let attn_pos = graph
            .new_tensor_2d_f32(pos_dim, rel_len, "layer_attn_pos")
            .expect("attention position allocation should succeed");
        let attn_in_w = graph
            .new_tensor_2d_f32(
                attn.in_proj.weight.input_dim,
                attn.in_proj.weight.output_dim,
                "layer_attn_in_w",
            )
            .expect("attention in weight allocation should succeed");
        let attn_in_b = graph
            .new_tensor_1d_f32(attn.in_proj.bias.len(), "layer_attn_in_b")
            .expect("attention in bias allocation should succeed");
        let attn_pos_w = graph
            .new_tensor_2d_f32(
                attn.linear_pos.input_dim,
                attn.linear_pos.output_dim,
                "layer_attn_pos_w",
            )
            .expect("attention pos weight allocation should succeed");
        let nonlin_cache = graph
            .new_tensor_2d_f32(nonlin_hidden_dim, left_context_len, "layer_nonlin_cache")
            .expect("nonlin cache allocation should succeed");
        let nonlin_in_w = graph
            .new_tensor_2d_f32(
                nonlin.in_proj.weight.input_dim,
                nonlin.in_proj.weight.output_dim,
                "layer_nonlin_in_w",
            )
            .expect("nonlin in weight allocation should succeed");
        let nonlin_in_b = graph
            .new_tensor_1d_f32(nonlin.in_proj.bias.len(), "layer_nonlin_in_b")
            .expect("nonlin in bias allocation should succeed");
        let nonlin_out_w = graph
            .new_tensor_2d_f32(
                nonlin.out_proj.weight.input_dim,
                nonlin.out_proj.weight.output_dim,
                "layer_nonlin_out_w",
            )
            .expect("nonlin out weight allocation should succeed");
        let nonlin_out_b = graph
            .new_tensor_1d_f32(nonlin.out_proj.bias.len(), "layer_nonlin_out_b")
            .expect("nonlin out bias allocation should succeed");
        let self1_cache = graph
            .new_tensor_2d_f32(self1_value_dim, left_context_len, "layer_self1_cache")
            .expect("self1 cache allocation should succeed");
        let self1_in_w = graph
            .new_tensor_2d_f32(
                self1.in_proj.weight.input_dim,
                self1.in_proj.weight.output_dim,
                "layer_self1_in_w",
            )
            .expect("self1 in weight allocation should succeed");
        let self1_in_b = graph
            .new_tensor_1d_f32(self1.in_proj.bias.len(), "layer_self1_in_b")
            .expect("self1 in bias allocation should succeed");
        let self1_out_w = graph
            .new_tensor_2d_f32(
                self1.out_proj.weight.input_dim,
                self1.out_proj.weight.output_dim,
                "layer_self1_out_w",
            )
            .expect("self1 out weight allocation should succeed");
        let self1_out_b = graph
            .new_tensor_1d_f32(self1.out_proj.bias.len(), "layer_self1_out_b")
            .expect("self1 out bias allocation should succeed");
        let conv1_cache = graph
            .new_tensor_2d_f32(conv1_cache_len, dim, "layer_conv1_cache")
            .expect("conv1 cache allocation should succeed");
        let conv1_in_w = graph
            .new_tensor_2d_f32(
                conv1.in_proj.weight.input_dim,
                conv1.in_proj.weight.output_dim,
                "layer_conv1_in_w",
            )
            .expect("conv1 in weight allocation should succeed");
        let conv1_in_b = graph
            .new_tensor_1d_f32(conv1.in_proj.bias.len(), "layer_conv1_in_b")
            .expect("conv1 in bias allocation should succeed");
        let conv1_causal_w = graph
            .new_tensor_3d_f32(conv1_causal_kernel, 1, dim, "layer_conv1_causal_w")
            .expect("conv1 causal weight allocation should succeed");
        let conv1_causal_b = graph
            .new_tensor_1d_f32(dim, "layer_conv1_causal_b")
            .expect("conv1 causal bias allocation should succeed");
        let conv1_chunk_w = graph
            .new_tensor_3d_f32(conv1_chunk_kernel, 1, dim, "layer_conv1_chunk_w")
            .expect("conv1 chunk weight allocation should succeed");
        let conv1_chunk_b = graph
            .new_tensor_1d_f32(dim, "layer_conv1_chunk_b")
            .expect("conv1 chunk bias allocation should succeed");
        let conv1_chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "layer_conv1_chunk_scale")
            .expect("conv1 chunk scale allocation should succeed");
        let conv1_out_w = graph
            .new_tensor_2d_f32(
                conv1.out_proj.weight.input_dim,
                conv1.out_proj.weight.output_dim,
                "layer_conv1_out_w",
            )
            .expect("conv1 out weight allocation should succeed");
        let conv1_out_b = graph
            .new_tensor_1d_f32(conv1.out_proj.bias.len(), "layer_conv1_out_b")
            .expect("conv1 out bias allocation should succeed");
        let conv1_swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "layer_conv1_swoosh_r_offset")
            .expect("conv1 swoosh offset allocation should succeed");
        let conv1_swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "layer_conv1_swoosh_r_shift")
            .expect("conv1 swoosh shift allocation should succeed");
        let ff2_in_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward2.in_proj.weight.input_dim,
                layer.feed_forward2.in_proj.weight.output_dim,
                "layer_ff2_in_w",
            )
            .expect("ff2 in weight allocation should succeed");
        let ff2_in_b = graph
            .new_tensor_1d_f32(layer.feed_forward2.in_proj.bias.len(), "layer_ff2_in_b")
            .expect("ff2 in bias allocation should succeed");
        let ff2_out_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward2.out_proj.weight.input_dim,
                layer.feed_forward2.out_proj.weight.output_dim,
                "layer_ff2_out_w",
            )
            .expect("ff2 out weight allocation should succeed");
        let ff2_out_b = graph
            .new_tensor_1d_f32(layer.feed_forward2.out_proj.bias.len(), "layer_ff2_out_b")
            .expect("ff2 out bias allocation should succeed");
        let ff2_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "layer_ff2_swoosh_l_offset")
            .expect("ff2 swoosh offset allocation should succeed");
        let ff2_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "layer_ff2_swoosh_l_shift")
            .expect("ff2 swoosh shift allocation should succeed");
        let bypass_mid_scale = graph
            .new_tensor_1d_f32(dim, "layer_bypass_mid_scale")
            .expect("bypass mid scale allocation should succeed");
        let self2_cache = graph
            .new_tensor_2d_f32(self2_value_dim, left_context_len, "layer_self2_cache")
            .expect("self2 cache allocation should succeed");
        let self2_in_w = graph
            .new_tensor_2d_f32(
                self2.in_proj.weight.input_dim,
                self2.in_proj.weight.output_dim,
                "layer_self2_in_w",
            )
            .expect("self2 in weight allocation should succeed");
        let self2_in_b = graph
            .new_tensor_1d_f32(self2.in_proj.bias.len(), "layer_self2_in_b")
            .expect("self2 in bias allocation should succeed");
        let self2_out_w = graph
            .new_tensor_2d_f32(
                self2.out_proj.weight.input_dim,
                self2.out_proj.weight.output_dim,
                "layer_self2_out_w",
            )
            .expect("self2 out weight allocation should succeed");
        let self2_out_b = graph
            .new_tensor_1d_f32(self2.out_proj.bias.len(), "layer_self2_out_b")
            .expect("self2 out bias allocation should succeed");
        let conv2_cache = graph
            .new_tensor_2d_f32(conv2_cache_len, dim, "layer_conv2_cache")
            .expect("conv2 cache allocation should succeed");
        let conv2_in_w = graph
            .new_tensor_2d_f32(
                conv2.in_proj.weight.input_dim,
                conv2.in_proj.weight.output_dim,
                "layer_conv2_in_w",
            )
            .expect("conv2 in weight allocation should succeed");
        let conv2_in_b = graph
            .new_tensor_1d_f32(conv2.in_proj.bias.len(), "layer_conv2_in_b")
            .expect("conv2 in bias allocation should succeed");
        let conv2_causal_w = graph
            .new_tensor_3d_f32(conv2_causal_kernel, 1, dim, "layer_conv2_causal_w")
            .expect("conv2 causal weight allocation should succeed");
        let conv2_causal_b = graph
            .new_tensor_1d_f32(dim, "layer_conv2_causal_b")
            .expect("conv2 causal bias allocation should succeed");
        let conv2_chunk_w = graph
            .new_tensor_3d_f32(conv2_chunk_kernel, 1, dim, "layer_conv2_chunk_w")
            .expect("conv2 chunk weight allocation should succeed");
        let conv2_chunk_b = graph
            .new_tensor_1d_f32(dim, "layer_conv2_chunk_b")
            .expect("conv2 chunk bias allocation should succeed");
        let conv2_chunk_scale = graph
            .new_tensor_2d_f32(frames, dim, "layer_conv2_chunk_scale")
            .expect("conv2 chunk scale allocation should succeed");
        let conv2_out_w = graph
            .new_tensor_2d_f32(
                conv2.out_proj.weight.input_dim,
                conv2.out_proj.weight.output_dim,
                "layer_conv2_out_w",
            )
            .expect("conv2 out weight allocation should succeed");
        let conv2_out_b = graph
            .new_tensor_1d_f32(conv2.out_proj.bias.len(), "layer_conv2_out_b")
            .expect("conv2 out bias allocation should succeed");
        let conv2_swoosh_r_offset = graph
            .new_tensor_1d_f32(1, "layer_conv2_swoosh_r_offset")
            .expect("conv2 swoosh offset allocation should succeed");
        let conv2_swoosh_r_shift = graph
            .new_tensor_1d_f32(1, "layer_conv2_swoosh_r_shift")
            .expect("conv2 swoosh shift allocation should succeed");
        let ff3_in_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward3.in_proj.weight.input_dim,
                layer.feed_forward3.in_proj.weight.output_dim,
                "layer_ff3_in_w",
            )
            .expect("ff3 in weight allocation should succeed");
        let ff3_in_b = graph
            .new_tensor_1d_f32(layer.feed_forward3.in_proj.bias.len(), "layer_ff3_in_b")
            .expect("ff3 in bias allocation should succeed");
        let ff3_out_w = graph
            .new_tensor_2d_f32(
                layer.feed_forward3.out_proj.weight.input_dim,
                layer.feed_forward3.out_proj.weight.output_dim,
                "layer_ff3_out_w",
            )
            .expect("ff3 out weight allocation should succeed");
        let ff3_out_b = graph
            .new_tensor_1d_f32(layer.feed_forward3.out_proj.bias.len(), "layer_ff3_out_b")
            .expect("ff3 out bias allocation should succeed");
        let ff3_swoosh_l_offset = graph
            .new_tensor_1d_f32(1, "layer_ff3_swoosh_l_offset")
            .expect("ff3 swoosh offset allocation should succeed");
        let ff3_swoosh_l_shift = graph
            .new_tensor_1d_f32(1, "layer_ff3_swoosh_l_shift")
            .expect("ff3 swoosh shift allocation should succeed");
        let norm_bias = graph
            .new_tensor_1d_f32(dim, "layer_norm_bias")
            .expect("norm bias allocation should succeed");
        let bypass_scale = graph
            .new_tensor_1d_f32(dim, "layer_bypass_scale")
            .expect("bypass scale allocation should succeed");

        for tensor in [
            input,
            ff1_in_w,
            ff1_in_b,
            ff1_out_w,
            ff1_out_b,
            ff1_swoosh_l_offset,
            ff1_swoosh_l_shift,
            attn_cache,
            attn_mask,
            attn_pos,
            attn_in_w,
            attn_in_b,
            attn_pos_w,
            nonlin_cache,
            nonlin_in_w,
            nonlin_in_b,
            nonlin_out_w,
            nonlin_out_b,
            self1_cache,
            self1_in_w,
            self1_in_b,
            self1_out_w,
            self1_out_b,
            conv1_cache,
            conv1_in_w,
            conv1_in_b,
            conv1_causal_w,
            conv1_causal_b,
            conv1_chunk_w,
            conv1_chunk_b,
            conv1_chunk_scale,
            conv1_out_w,
            conv1_out_b,
            conv1_swoosh_r_offset,
            conv1_swoosh_r_shift,
            ff2_in_w,
            ff2_in_b,
            ff2_out_w,
            ff2_out_b,
            ff2_swoosh_l_offset,
            ff2_swoosh_l_shift,
            bypass_mid_scale,
            self2_cache,
            self2_in_w,
            self2_in_b,
            self2_out_w,
            self2_out_b,
            conv2_cache,
            conv2_in_w,
            conv2_in_b,
            conv2_causal_w,
            conv2_causal_b,
            conv2_chunk_w,
            conv2_chunk_b,
            conv2_chunk_scale,
            conv2_out_w,
            conv2_out_b,
            conv2_swoosh_r_offset,
            conv2_swoosh_r_shift,
            ff3_in_w,
            ff3_in_b,
            ff3_out_w,
            ff3_out_b,
            ff3_swoosh_l_offset,
            ff3_swoosh_l_shift,
            norm_bias,
            bypass_scale,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }

        let output = apply_zipformer_layer_graph(
            &graph,
            input,
            XasrZipformerLayerGraphTensors {
                layer_head: XasrLayerHeadGraphTensors {
                    feed_forward1: XasrFeedForwardGraphTensors {
                        in_proj_weight: ff1_in_w,
                        in_proj_bias: ff1_in_b,
                        out_proj_weight: ff1_out_w,
                        out_proj_bias: ff1_out_b,
                        swoosh_l_offset: ff1_swoosh_l_offset,
                        swoosh_l_shift: ff1_swoosh_l_shift,
                    },
                    attention_weights: XasrSelfAttentionWeightsGraphTensors {
                        cache: attn_cache,
                        mask: attn_mask,
                        pos_embedding: attn_pos,
                        in_proj_weight: attn_in_w,
                        in_proj_bias: attn_in_b,
                        linear_pos_weight: attn_pos_w,
                    },
                    nonlin_cache,
                    nonlin_in_proj_weight: nonlin_in_w,
                    nonlin_in_proj_bias: nonlin_in_b,
                    nonlin_out_proj_weight: nonlin_out_w,
                    nonlin_out_proj_bias: nonlin_out_b,
                    self1_cache,
                    self1_in_proj_weight: self1_in_w,
                    self1_in_proj_bias: self1_in_b,
                    self1_out_proj_weight: self1_out_w,
                    self1_out_proj_bias: self1_out_b,
                    conv_module1: XasrConvolutionModuleGraphTensors {
                        cache: conv1_cache,
                        in_proj_weight: conv1_in_w,
                        in_proj_bias: conv1_in_b,
                        causal_kernel: conv1_causal_w,
                        causal_bias: conv1_causal_b,
                        chunk_kernel: conv1_chunk_w,
                        chunk_bias: conv1_chunk_b,
                        chunk_scale: conv1_chunk_scale,
                        out_proj_weight: conv1_out_w,
                        out_proj_bias: conv1_out_b,
                        swoosh_r_offset: conv1_swoosh_r_offset,
                        swoosh_r_shift: conv1_swoosh_r_shift,
                    },
                    feed_forward2: XasrFeedForwardGraphTensors {
                        in_proj_weight: ff2_in_w,
                        in_proj_bias: ff2_in_b,
                        out_proj_weight: ff2_out_w,
                        out_proj_bias: ff2_out_b,
                        swoosh_l_offset: ff2_swoosh_l_offset,
                        swoosh_l_shift: ff2_swoosh_l_shift,
                    },
                    bypass_mid_scale,
                },
                self2_cache,
                self2_in_proj_weight: self2_in_w,
                self2_in_proj_bias: self2_in_b,
                self2_out_proj_weight: self2_out_w,
                self2_out_proj_bias: self2_out_b,
                layer_tail: XasrLayerTailGraphTensors {
                    conv_module2: XasrConvolutionModuleGraphTensors {
                        cache: conv2_cache,
                        in_proj_weight: conv2_in_w,
                        in_proj_bias: conv2_in_b,
                        causal_kernel: conv2_causal_w,
                        causal_bias: conv2_causal_b,
                        chunk_kernel: conv2_chunk_w,
                        chunk_bias: conv2_chunk_b,
                        chunk_scale: conv2_chunk_scale,
                        out_proj_weight: conv2_out_w,
                        out_proj_bias: conv2_out_b,
                        swoosh_r_offset: conv2_swoosh_r_offset,
                        swoosh_r_shift: conv2_swoosh_r_shift,
                    },
                    feed_forward3: XasrFeedForwardGraphTensors {
                        in_proj_weight: ff3_in_w,
                        in_proj_bias: ff3_in_b,
                        out_proj_weight: ff3_out_w,
                        out_proj_bias: ff3_out_b,
                        swoosh_l_offset: ff3_swoosh_l_offset,
                        swoosh_l_shift: ff3_swoosh_l_shift,
                    },
                    norm_bias,
                    bypass_scale,
                },
            },
            XasrZipformerLayerGraphShape {
                layer_head: XasrLayerHeadGraphShape {
                    dim,
                    frames,
                    left_context_len,
                    num_heads,
                    query_head_dim,
                    pos_dim,
                    pos_output_dim,
                    nonlin_hidden_dim,
                    self1_value_dim,
                    conv1_cache_len,
                    conv1_causal_kernel_len: conv1_causal_kernel,
                    conv1_chunk_kernel_len: conv1_chunk_kernel,
                },
                self2_value_dim,
                layer_tail: XasrLayerTailGraphShape {
                    dim,
                    frames,
                    conv_cache_len: conv2_cache_len,
                    conv_causal_kernel_len: conv2_causal_kernel,
                    conv_chunk_kernel_len: conv2_chunk_kernel,
                    norm_log_scale: layer.norm_log_scale[0],
                },
            },
        )
        .expect("zipformer layer graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_cached_key)
            .expect("key cache output should set");
        graph
            .set_output(output.new_cached_nonlin_attention)
            .expect("nonlin cache output should set");
        graph
            .set_output(output.new_cached_val1)
            .expect("val1 cache output should set");
        graph
            .set_output(output.new_cached_val2)
            .expect("val2 cache output should set");
        graph
            .set_output(output.new_cached_conv1)
            .expect("conv1 cache output should set");
        graph
            .set_output(output.new_cached_conv2)
            .expect("conv2 cache output should set");

        graph
            .set_f32_slice(input, &input_values, "layer_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(
                ff1_in_w,
                &layer.feed_forward1.in_proj.weight.values,
                "layer_ff1_in_w",
            )
            .expect("ff1 in weight upload should succeed");
        graph
            .set_f32_slice(
                ff1_in_b,
                &layer.feed_forward1.in_proj.bias,
                "layer_ff1_in_b",
            )
            .expect("ff1 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff1_out_w,
                &layer.feed_forward1.out_proj.weight.values,
                "layer_ff1_out_w",
            )
            .expect("ff1 out weight upload should succeed");
        graph
            .set_f32_slice(
                ff1_out_b,
                &layer.feed_forward1.out_proj.bias,
                "layer_ff1_out_b",
            )
            .expect("ff1 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff1_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "layer_ff1_swoosh_l_offset",
            )
            .expect("ff1 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff1_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "layer_ff1_swoosh_l_shift",
            )
            .expect("ff1 swoosh shift upload should succeed");
        graph
            .set_f32_slice(attn_cache, &attn_cache_values, "layer_attn_cache")
            .expect("attn cache upload should succeed");
        graph
            .set_f32_slice(attn_mask, &mask_values, "layer_attn_mask")
            .expect("attn mask upload should succeed");
        graph
            .set_f32_slice(attn_pos, &pos_embedding_values, "layer_attn_pos")
            .expect("attn pos upload should succeed");
        graph
            .set_f32_slice(attn_in_w, &attn.in_proj.weight.values, "layer_attn_in_w")
            .expect("attn in weight upload should succeed");
        graph
            .set_f32_slice(attn_in_b, &attn.in_proj.bias, "layer_attn_in_b")
            .expect("attn in bias upload should succeed");
        graph
            .set_f32_slice(attn_pos_w, &attn.linear_pos.values, "layer_attn_pos_w")
            .expect("attn pos weight upload should succeed");
        graph
            .set_f32_slice(nonlin_cache, &nonlin_cache_values, "layer_nonlin_cache")
            .expect("nonlin cache upload should succeed");
        graph
            .set_f32_slice(
                nonlin_in_w,
                &nonlin.in_proj.weight.values,
                "layer_nonlin_in_w",
            )
            .expect("nonlin in weight upload should succeed");
        graph
            .set_f32_slice(nonlin_in_b, &nonlin.in_proj.bias, "layer_nonlin_in_b")
            .expect("nonlin in bias upload should succeed");
        graph
            .set_f32_slice(
                nonlin_out_w,
                &nonlin.out_proj.weight.values,
                "layer_nonlin_out_w",
            )
            .expect("nonlin out weight upload should succeed");
        graph
            .set_f32_slice(nonlin_out_b, &nonlin.out_proj.bias, "layer_nonlin_out_b")
            .expect("nonlin out bias upload should succeed");
        graph
            .set_f32_slice(self1_cache, &self1_cache_values, "layer_self1_cache")
            .expect("self1 cache upload should succeed");
        graph
            .set_f32_slice(self1_in_w, &self1.in_proj.weight.values, "layer_self1_in_w")
            .expect("self1 in weight upload should succeed");
        graph
            .set_f32_slice(self1_in_b, &self1.in_proj.bias, "layer_self1_in_b")
            .expect("self1 in bias upload should succeed");
        graph
            .set_f32_slice(
                self1_out_w,
                &self1.out_proj.weight.values,
                "layer_self1_out_w",
            )
            .expect("self1 out weight upload should succeed");
        graph
            .set_f32_slice(self1_out_b, &self1.out_proj.bias, "layer_self1_out_b")
            .expect("self1 out bias upload should succeed");
        graph
            .set_f32_slice(conv1_cache, &conv1_cache_values, "layer_conv1_cache")
            .expect("conv1 cache upload should succeed");
        graph
            .set_f32_slice(conv1_in_w, &conv1.in_proj.weight.values, "layer_conv1_in_w")
            .expect("conv1 in weight upload should succeed");
        graph
            .set_f32_slice(conv1_in_b, &conv1.in_proj.bias, "layer_conv1_in_b")
            .expect("conv1 in bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_causal_w,
                &conv1.depthwise_causal_conv.weight.values,
                "layer_conv1_causal_w",
            )
            .expect("conv1 causal weight upload should succeed");
        graph
            .set_f32_slice(
                conv1_causal_b,
                &conv1.depthwise_causal_conv.bias,
                "layer_conv1_causal_b",
            )
            .expect("conv1 causal bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_w,
                &conv1.depthwise_chunkwise_conv.weight.values,
                "layer_conv1_chunk_w",
            )
            .expect("conv1 chunk weight upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_b,
                &conv1.depthwise_chunkwise_conv.bias,
                "layer_conv1_chunk_b",
            )
            .expect("conv1 chunk bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_chunk_scale,
                &conv1_chunk_scale_values,
                "layer_conv1_chunk_scale",
            )
            .expect("conv1 chunk scale upload should succeed");
        graph
            .set_f32_slice(
                conv1_out_w,
                &conv1.out_proj.weight.values,
                "layer_conv1_out_w",
            )
            .expect("conv1 out weight upload should succeed");
        graph
            .set_f32_slice(conv1_out_b, &conv1.out_proj.bias, "layer_conv1_out_b")
            .expect("conv1 out bias upload should succeed");
        graph
            .set_f32_slice(
                conv1_swoosh_r_offset,
                &[SWOOSH_R_OFFSET],
                "layer_conv1_swoosh_r_offset",
            )
            .expect("conv1 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                conv1_swoosh_r_shift,
                &[SWOOSH_R_SHIFT],
                "layer_conv1_swoosh_r_shift",
            )
            .expect("conv1 swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                ff2_in_w,
                &layer.feed_forward2.in_proj.weight.values,
                "layer_ff2_in_w",
            )
            .expect("ff2 in weight upload should succeed");
        graph
            .set_f32_slice(
                ff2_in_b,
                &layer.feed_forward2.in_proj.bias,
                "layer_ff2_in_b",
            )
            .expect("ff2 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff2_out_w,
                &layer.feed_forward2.out_proj.weight.values,
                "layer_ff2_out_w",
            )
            .expect("ff2 out weight upload should succeed");
        graph
            .set_f32_slice(
                ff2_out_b,
                &layer.feed_forward2.out_proj.bias,
                "layer_ff2_out_b",
            )
            .expect("ff2 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff2_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "layer_ff2_swoosh_l_offset",
            )
            .expect("ff2 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff2_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "layer_ff2_swoosh_l_shift",
            )
            .expect("ff2 swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                bypass_mid_scale,
                &layer.bypass_mid_scale,
                "layer_bypass_mid_scale",
            )
            .expect("bypass mid scale upload should succeed");
        graph
            .set_f32_slice(self2_cache, &self2_cache_values, "layer_self2_cache")
            .expect("self2 cache upload should succeed");
        graph
            .set_f32_slice(self2_in_w, &self2.in_proj.weight.values, "layer_self2_in_w")
            .expect("self2 in weight upload should succeed");
        graph
            .set_f32_slice(self2_in_b, &self2.in_proj.bias, "layer_self2_in_b")
            .expect("self2 in bias upload should succeed");
        graph
            .set_f32_slice(
                self2_out_w,
                &self2.out_proj.weight.values,
                "layer_self2_out_w",
            )
            .expect("self2 out weight upload should succeed");
        graph
            .set_f32_slice(self2_out_b, &self2.out_proj.bias, "layer_self2_out_b")
            .expect("self2 out bias upload should succeed");
        graph
            .set_f32_slice(conv2_cache, &conv2_cache_values, "layer_conv2_cache")
            .expect("conv2 cache upload should succeed");
        graph
            .set_f32_slice(conv2_in_w, &conv2.in_proj.weight.values, "layer_conv2_in_w")
            .expect("conv2 in weight upload should succeed");
        graph
            .set_f32_slice(conv2_in_b, &conv2.in_proj.bias, "layer_conv2_in_b")
            .expect("conv2 in bias upload should succeed");
        graph
            .set_f32_slice(
                conv2_causal_w,
                &conv2.depthwise_causal_conv.weight.values,
                "layer_conv2_causal_w",
            )
            .expect("conv2 causal weight upload should succeed");
        graph
            .set_f32_slice(
                conv2_causal_b,
                &conv2.depthwise_causal_conv.bias,
                "layer_conv2_causal_b",
            )
            .expect("conv2 causal bias upload should succeed");
        graph
            .set_f32_slice(
                conv2_chunk_w,
                &conv2.depthwise_chunkwise_conv.weight.values,
                "layer_conv2_chunk_w",
            )
            .expect("conv2 chunk weight upload should succeed");
        graph
            .set_f32_slice(
                conv2_chunk_b,
                &conv2.depthwise_chunkwise_conv.bias,
                "layer_conv2_chunk_b",
            )
            .expect("conv2 chunk bias upload should succeed");
        graph
            .set_f32_slice(
                conv2_chunk_scale,
                &conv2_chunk_scale_values,
                "layer_conv2_chunk_scale",
            )
            .expect("conv2 chunk scale upload should succeed");
        graph
            .set_f32_slice(
                conv2_out_w,
                &conv2.out_proj.weight.values,
                "layer_conv2_out_w",
            )
            .expect("conv2 out weight upload should succeed");
        graph
            .set_f32_slice(conv2_out_b, &conv2.out_proj.bias, "layer_conv2_out_b")
            .expect("conv2 out bias upload should succeed");
        graph
            .set_f32_slice(
                conv2_swoosh_r_offset,
                &[SWOOSH_R_OFFSET],
                "layer_conv2_swoosh_r_offset",
            )
            .expect("conv2 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                conv2_swoosh_r_shift,
                &[SWOOSH_R_SHIFT],
                "layer_conv2_swoosh_r_shift",
            )
            .expect("conv2 swoosh shift upload should succeed");
        graph
            .set_f32_slice(
                ff3_in_w,
                &layer.feed_forward3.in_proj.weight.values,
                "layer_ff3_in_w",
            )
            .expect("ff3 in weight upload should succeed");
        graph
            .set_f32_slice(
                ff3_in_b,
                &layer.feed_forward3.in_proj.bias,
                "layer_ff3_in_b",
            )
            .expect("ff3 in bias upload should succeed");
        graph
            .set_f32_slice(
                ff3_out_w,
                &layer.feed_forward3.out_proj.weight.values,
                "layer_ff3_out_w",
            )
            .expect("ff3 out weight upload should succeed");
        graph
            .set_f32_slice(
                ff3_out_b,
                &layer.feed_forward3.out_proj.bias,
                "layer_ff3_out_b",
            )
            .expect("ff3 out bias upload should succeed");
        graph
            .set_f32_slice(
                ff3_swoosh_l_offset,
                &[SWOOSH_L_OFFSET],
                "layer_ff3_swoosh_l_offset",
            )
            .expect("ff3 swoosh offset upload should succeed");
        graph
            .set_f32_slice(
                ff3_swoosh_l_shift,
                &[SWOOSH_L_SHIFT],
                "layer_ff3_swoosh_l_shift",
            )
            .expect("ff3 swoosh shift upload should succeed");
        graph
            .set_f32_slice(norm_bias, &layer.norm_bias, "layer_norm_bias")
            .expect("norm bias upload should succeed");
        graph
            .set_f32_slice(bypass_scale, &layer.bypass_scale, "layer_bypass_scale")
            .expect("bypass scale upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, frames * dim),
                (output.new_cached_key, left_context_len * query_dim),
                (
                    output.new_cached_nonlin_attention,
                    left_context_len * nonlin_hidden_dim,
                ),
                (output.new_cached_val1, left_context_len * self1_value_dim),
                (output.new_cached_val2, left_context_len * self2_value_dim),
                (output.new_cached_conv1, dim * conv1_cache_len),
                (output.new_cached_conv2, dim * conv2_cache_len),
            ])
            .expect("zipformer layer graph should compute");
        assert_max_abs_diff(
            "zipformer layer graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph key cache",
            &actual[1],
            &reference.new_cached_key,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph nonlin cache",
            &actual[2],
            &reference.new_cached_nonlin_attention,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph self1 cache",
            &actual[3],
            &reference.new_cached_val1,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph self2 cache",
            &actual[4],
            &reference.new_cached_val2,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph conv1 cache",
            &actual[5],
            &reference.new_cached_conv1,
            2.0e-2,
        );
        assert_max_abs_diff(
            "zipformer layer graph conv2 cache",
            &actual[6],
            &reference.new_cached_conv2,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML self-attention value helper with exported ONNX debug tensors"]
    fn ggml_self_attention_value_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-attn-values-debug-480ms.add8.f32");
        let softmax_path = root.join("oracle-layer0-debug-480ms.softmax.f32");
        let expected_path = root.join("oracle-layer0-attn-values-debug-480ms.self1.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists()
            || !softmax_path.exists()
            || !expected_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing self-attention value oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let attention_values = read_f32_file(&softmax_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let self_attn = &weights.stacks[0].layers[0].self_attn1;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        let left_context_len = metadata.left_context_len[0];
        let num_heads = metadata.num_heads[0];
        let value_dim = self_attn.in_proj.weight.output_dim;
        let k_len = left_context_len + frames;
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);
        assert_eq!(attention_values.len(), num_heads * frames * k_len);
        let cache_values = vec![0.0_f32; left_context_len * value_dim];
        let reference = self_attention_streaming_reference(
            self_attn,
            &input_values,
            &attention_values,
            frames,
            dim,
            num_heads,
            left_context_len,
            Some(&cache_values),
        )
        .expect("reference self attention");

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "self1_input")
            .expect("input allocation should succeed");
        let cache = graph
            .new_tensor_2d_f32(value_dim, left_context_len, "self1_cache")
            .expect("cache allocation should succeed");
        let attention = graph
            .new_tensor_3d_f32(k_len, frames, num_heads, "self1_attention")
            .expect("attention allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(
                self_attn.in_proj.weight.input_dim,
                self_attn.in_proj.weight.output_dim,
                "self1_in_weight",
            )
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(self_attn.in_proj.bias.len(), "self1_in_bias")
            .expect("in bias allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(
                self_attn.out_proj.weight.input_dim,
                self_attn.out_proj.weight.output_dim,
                "self1_out_weight",
            )
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(self_attn.out_proj.bias.len(), "self1_out_bias")
            .expect("out bias allocation should succeed");
        for tensor in [
            input, cache, attention, in_weight, in_bias, out_weight, out_bias,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_self_attention_value_graph(
            &graph,
            input,
            XasrSelfAttentionGraphTensors {
                cache,
                attention_weights: attention,
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
            },
            XasrSelfAttentionGraphShape {
                dim,
                frames,
                left_context_len,
                num_heads,
                value_dim,
            },
        )
        .expect("self attention graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_cache)
            .expect("cache output should set");

        graph
            .set_f32_slice(input, &input_values, "self1_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(cache, &cache_values, "self1_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(attention, &attention_values, "self1_attention")
            .expect("attention upload should succeed");
        graph
            .set_f32_slice(
                in_weight,
                &self_attn.in_proj.weight.values,
                "self1_in_weight",
            )
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &self_attn.in_proj.bias, "self1_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(
                out_weight,
                &self_attn.out_proj.weight.values,
                "self1_out_weight",
            )
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &self_attn.out_proj.bias, "self1_out_bias")
            .expect("out bias upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, dim * frames),
                (output.new_cache, left_context_len * value_dim),
            ])
            .expect("self attention graph should compute");
        assert_max_abs_diff(
            "self1 value graph ONNX parity",
            &actual[0],
            &expected,
            2.0e-2,
        );
        assert_max_abs_diff(
            "self1 value graph new cache",
            &actual[1],
            &reference.new_cache,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML nonlin-attention helper with exported ONNX debug tensors"]
    fn ggml_nonlin_attention_helper_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.add6.f32");
        let softmax_path = root.join("oracle-layer0-debug-480ms.softmax.f32");
        let expected_path = root.join("oracle-layer0-attn-values-debug-480ms.nonlin.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists()
            || !softmax_path.exists()
            || !expected_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing nonlin-attention oracle files");
            return;
        }

        let input_values = read_f32_file(&input_path);
        let softmax = read_f32_file(&softmax_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let nonlin = &weights.stacks[0].layers[0].nonlin_attention;
        let frames = 24;
        let dim = metadata.encoder_dims[0];
        let left_context_len = metadata.left_context_len[0];
        let num_heads = 1;
        let hidden_dim = nonlin.out_proj.weight.input_dim;
        let k_len = left_context_len + frames;
        assert_eq!(input_values.len(), frames * dim);
        assert_eq!(expected.len(), frames * dim);
        assert!(softmax.len() >= frames * k_len);
        let attention_values = &softmax[..frames * k_len];
        let cache_values = vec![0.0_f32; left_context_len * hidden_dim];
        let reference = nonlin_attention_streaming_reference(
            nonlin,
            &input_values,
            attention_values,
            frames,
            dim,
            num_heads,
            left_context_len,
            Some(&cache_values),
        )
        .expect("reference nonlin attention");

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(dim, frames, "nonlin_input")
            .expect("input allocation should succeed");
        let cache = graph
            .new_tensor_2d_f32(hidden_dim, left_context_len, "nonlin_cache")
            .expect("cache allocation should succeed");
        let attention = graph
            .new_tensor_3d_f32(k_len, frames, num_heads, "nonlin_attention")
            .expect("attention allocation should succeed");
        let in_weight = graph
            .new_tensor_2d_f32(
                nonlin.in_proj.weight.input_dim,
                nonlin.in_proj.weight.output_dim,
                "nonlin_in_weight",
            )
            .expect("in weight allocation should succeed");
        let in_bias = graph
            .new_tensor_1d_f32(nonlin.in_proj.bias.len(), "nonlin_in_bias")
            .expect("in bias allocation should succeed");
        let out_weight = graph
            .new_tensor_2d_f32(
                nonlin.out_proj.weight.input_dim,
                nonlin.out_proj.weight.output_dim,
                "nonlin_out_weight",
            )
            .expect("out weight allocation should succeed");
        let out_bias = graph
            .new_tensor_1d_f32(nonlin.out_proj.bias.len(), "nonlin_out_bias")
            .expect("out bias allocation should succeed");
        for tensor in [
            input, cache, attention, in_weight, in_bias, out_weight, out_bias,
        ] {
            graph
                .set_input(tensor)
                .expect("set_input should succeed before allocation");
        }
        let output = apply_nonlin_attention_value_graph(
            &graph,
            input,
            XasrNonlinAttentionGraphTensors {
                cache,
                attention_weights: attention,
                in_proj_weight: in_weight,
                in_proj_bias: in_bias,
                out_proj_weight: out_weight,
                out_proj_bias: out_bias,
            },
            XasrNonlinAttentionGraphShape {
                dim,
                frames,
                left_context_len,
                num_heads,
                hidden_dim,
            },
        )
        .expect("nonlin attention graph");
        graph
            .set_output(output.rows)
            .expect("rows output should set");
        graph
            .set_output(output.new_cache)
            .expect("cache output should set");

        graph
            .set_f32_slice(input, &input_values, "nonlin_input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(cache, &cache_values, "nonlin_cache")
            .expect("cache upload should succeed");
        graph
            .set_f32_slice(attention, attention_values, "nonlin_attention")
            .expect("attention upload should succeed");
        graph
            .set_f32_slice(in_weight, &nonlin.in_proj.weight.values, "nonlin_in_weight")
            .expect("in weight upload should succeed");
        graph
            .set_f32_slice(in_bias, &nonlin.in_proj.bias, "nonlin_in_bias")
            .expect("in bias upload should succeed");
        graph
            .set_f32_slice(
                out_weight,
                &nonlin.out_proj.weight.values,
                "nonlin_out_weight",
            )
            .expect("out weight upload should succeed");
        graph
            .set_f32_slice(out_bias, &nonlin.out_proj.bias, "nonlin_out_bias")
            .expect("out bias upload should succeed");

        let actual = graph
            .compute_outputs_f32(&[
                (output.rows, dim * frames),
                (output.new_cache, left_context_len * hidden_dim),
            ])
            .expect("nonlin attention graph should compute");
        assert_max_abs_diff("nonlin graph ONNX parity", &actual[0], &expected, 2.0e-2);
        assert_max_abs_diff(
            "nonlin graph new cache",
            &actual[1],
            &reference.new_cache,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR encoder_graph stack0 facade with exported ONNX debug tensors"]
    fn stack0_graph_facade_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-stack0-debug-480ms.layer1.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing stack0 encoder_graph oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let graph = XasrZipformerEncoderGraph::new_reference(metadata, weights).expect("graph");

        assert_eq!(graph.backend(), XasrEncoderGraphBackend::Reference);
        let output = graph
            .encode_stack0_from_embed_rows(&input, 24, 192, 61)
            .expect("stack0 graph facade");

        assert_eq!(output.frames, 24);
        assert_eq!(output.dim, 192);
        assert_max_abs_diff("stack0 graph facade", &output.rows, &expected, 2.0e-2);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML stack0 facade with exported ONNX debug tensors"]
    fn ggml_stack0_graph_facade_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-layer0-ff-debug-480ms.input.f32");
        let expected_path = root.join("oracle-stack0-debug-480ms.layer1.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing GGML stack0 encoder_graph oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 256 * 1024 * 1024;
        config.graph_size = 262_144;
        let graph = XasrZipformerEncoderGraph::new_ggml_cpu_stack0(metadata, weights, config)
            .expect("graph");

        assert_eq!(graph.backend(), XasrEncoderGraphBackend::GgmlCpuStack0);
        let output = graph
            .encode_stack0_from_embed_rows(&input, 24, 192, 61)
            .expect("GGML stack0 graph facade");

        assert_eq!(output.frames, 24);
        assert_eq!(output.dim, 192);
        assert_max_abs_diff("GGML stack0 graph facade", &output.rows, &expected, 2.0e-2);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR full encoder reference facade with exported ONNX debug tensors"]
    fn full_encoder_reference_facade_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-encoder-debug-480ms.out_norm.f32");
        let expected_path = root.join("oracle-encoder-debug-480ms.pre_joiner.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing full encoder facade oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let graph = XasrZipformerEncoderGraph::new_reference(metadata, weights).expect("graph");

        let output = graph
            .encode_from_embed_rows(&input, 24, 192, 61)
            .expect("full encoder graph facade");

        assert_eq!(output.frames, 12);
        assert_eq!(output.dim, 768);
        assert_max_abs_diff("full encoder graph facade", &output.rows, &expected, 3.0e-2);
    }

    #[test]
    #[ignore = "host-local: compares X-ASR GGML full encoder facade with exported ONNX debug tensors"]
    fn ggml_full_encoder_graph_facade_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-encoder-debug-480ms.out_norm.f32");
        let expected_path = root.join("oracle-encoder-debug-480ms.pre_joiner.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing GGML full encoder facade oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 2 * 1024 * 1024 * 1024;
        config.graph_size = 2_000_000;
        let graph = XasrZipformerEncoderGraph::new_ggml_cpu_full_encoder(metadata, weights, config)
            .expect("graph");

        assert_eq!(graph.backend(), XasrEncoderGraphBackend::GgmlCpuFullEncoder);
        let output = graph
            .encode_from_embed_rows(&input, 24, 192, 61)
            .expect("GGML full encoder graph facade");

        assert_eq!(output.frames, 12);
        assert_eq!(output.dim, 768);
        assert_max_abs_diff(
            "GGML full encoder graph facade",
            &output.rows,
            &expected,
            3.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR cache-aware chunk facade with exported ONNX debug tensors"]
    fn ggml_streaming_chunk_facade_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-encoder-debug-480ms.x.f32");
        let expected_path = root.join("oracle-encoder-debug-480ms.pre_joiner.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists() || !expected_path.exists() || !pack.exists() {
            eprintln!("skipping: missing GGML streaming chunk oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected = read_f32_file(&expected_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let total_layers = metadata.total_encoder_layers();
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.context_bytes = 2 * 1024 * 1024 * 1024;
        config.graph_size = 2_000_000;
        let graph = XasrZipformerEncoderGraph::new_ggml_cpu_full_encoder(metadata, weights, config)
            .expect("graph");
        let features = XasrEncoderFeatureInput::new(61, 80, input).expect("features");

        let chunk = graph
            .encode_streaming_chunk_from_features(&features, None)
            .expect("GGML streaming chunk facade");

        assert_eq!(chunk.output.frames, 12);
        assert_eq!(chunk.output.dim, 768);
        assert_eq!(chunk.state.embed_states.len(), 128 * 3 * 19);
        assert_eq!(chunk.state.layer_caches.len(), total_layers);
        assert!(
            chunk
                .state
                .layer_caches
                .iter()
                .all(|cache| !cache.cached_key.is_empty()
                    && !cache.cached_nonlin_attention.is_empty()
                    && !cache.cached_val1.is_empty()
                    && !cache.cached_val2.is_empty()
                    && !cache.cached_conv1.is_empty()
                    && !cache.cached_conv2.is_empty())
        );
        assert_max_abs_diff(
            "GGML streaming chunk facade",
            &chunk.output.rows,
            &expected,
            3.0e-2,
        );
    }

    fn read_f32_file(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).unwrap_or_else(|error| {
            panic!("read {}: {error}", path.display());
        });
        assert!(
            bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
            "{} byte length is not f32-aligned",
            path.display()
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn assert_max_abs_diff(name: &str, lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len(), "{name} length mismatch");
        let diff = lhs
            .iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            diff <= tolerance,
            "{name} max abs diff {diff} exceeds tolerance {tolerance}"
        );
    }

    fn add_same_shape_for_test(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "test add length mismatch");
        lhs.iter().zip(rhs).map(|(&lhs, &rhs)| lhs + rhs).collect()
    }

    fn sigmoid_reference(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn compact_relative_positional_encoding_for_test(
        frames: usize,
        left_context_len: usize,
        embed_dim: usize,
    ) -> Vec<f32> {
        assert!(embed_dim.is_multiple_of(2));
        let total_context = frames + left_context_len;
        let seq_len = left_context_len + 2 * frames - 1;
        let compression_length = (embed_dim as f32).sqrt();
        let length_scale = embed_dim as f32 / (2.0 * std::f32::consts::PI);
        let mut output = vec![0.0_f32; seq_len * embed_dim];
        for row in 0..seq_len {
            let offset = row as isize - (total_context as isize - 1);
            let sign = (offset as f32).signum();
            let abs = (offset as f32).abs();
            let compressed = compression_length
                * sign
                * ((abs + compression_length).ln() - compression_length.ln());
            let atan = (compressed / length_scale).atan();
            for i in 0..embed_dim / 2 {
                let value = atan * (i + 1) as f32;
                output[row * embed_dim + 2 * i] = value.cos();
                output[row * embed_dim + 2 * i + 1] = value.sin();
            }
            output[row * embed_dim + embed_dim - 1] = 1.0;
        }
        output
    }

    fn depthwise_conv1d_channel_major_reference(
        input: &[f32],
        kernel: &[f32],
        bias: &[f32],
        channels: usize,
        input_len: usize,
        kernel_len: usize,
        padding: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), channels * input_len);
        assert_eq!(kernel.len(), channels * kernel_len);
        assert_eq!(bias.len(), channels);
        let output_len = input_len + 2 * padding - kernel_len + 1;
        let mut output = vec![0.0_f32; channels * output_len];
        for c in 0..channels {
            for t in 0..output_len {
                let mut sum = bias[c];
                for k in 0..kernel_len {
                    let Some(input_t) = (t + k).checked_sub(padding) else {
                        continue;
                    };
                    if input_t >= input_len {
                        continue;
                    }
                    sum += input[c * input_len + input_t] * kernel[c * kernel_len + k];
                }
                output[c * output_len + t] = sum;
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn depthwise_mix_channel_major_reference(
        cached_input: &[f32],
        chunk_input: &[f32],
        causal_kernel: &[f32],
        causal_bias: &[f32],
        chunk_kernel: &[f32],
        chunk_bias: &[f32],
        chunk_scale: &[f32],
        shape: XasrDepthwiseMixShape,
    ) -> Vec<f32> {
        let causal = depthwise_conv1d_channel_major_reference(
            cached_input,
            causal_kernel,
            causal_bias,
            shape.channels,
            shape.cached_len,
            shape.causal_kernel_len,
            0,
        );
        let chunk = depthwise_conv1d_channel_major_reference(
            chunk_input,
            chunk_kernel,
            chunk_bias,
            shape.channels,
            shape.frames,
            shape.chunk_kernel_len,
            shape.chunk_kernel_len / 2,
        );
        assert_eq!(causal.len(), shape.channels * shape.frames);
        assert_eq!(chunk.len(), shape.channels * shape.frames);
        assert_eq!(chunk_scale.len(), shape.channels * shape.frames);
        causal
            .iter()
            .zip(chunk.iter())
            .zip(chunk_scale.iter())
            .map(|((&causal, &chunk), &scale)| causal + chunk * scale)
            .collect()
    }

    struct PreparedDepthwiseMixInputs {
        cached_input: Vec<f32>,
        channel_major: Vec<f32>,
        chunk_scale: Vec<f32>,
    }

    fn prepare_depthwise_mix_inputs_for_test(
        weights: &XasrConvolutionModuleWeights,
        rows: &[f32],
        frames: usize,
        dim: usize,
    ) -> Result<PreparedDepthwiseMixInputs, String> {
        assert_eq!(rows.len(), frames * dim);
        let mut gated = vec![0.0_f32; frames * dim];
        for (t, frame) in rows.chunks_exact(dim).enumerate() {
            let projected = weights
                .in_proj
                .weight
                .apply(frame, Some(&weights.in_proj.bias))?;
            assert_eq!(projected.len(), 2 * dim);
            for c in 0..dim {
                gated[t * dim + c] = projected[c] * sigmoid_reference(projected[dim + c]);
            }
        }

        let mut channel_major = vec![0.0_f32; dim * frames];
        for t in 0..frames {
            for c in 0..dim {
                channel_major[c * frames + t] = gated[t * dim + c];
            }
        }

        let left_pad = weights.depthwise_chunkwise_conv.weight.dims[0] / 2;
        let mut cached_input = vec![0.0_f32; dim * (left_pad + frames)];
        for c in 0..dim {
            let dst = c * (left_pad + frames) + left_pad;
            let src = c * frames;
            cached_input[dst..dst + frames].copy_from_slice(&channel_major[src..src + frames]);
        }

        let mut chunk_scale = vec![0.0_f32; dim * frames];
        for c in 0..dim {
            for t in 0..frames {
                chunk_scale[c * frames + t] = chunkwise_conv_scale_for_test(weights, c, t, frames)?;
            }
        }

        Ok(PreparedDepthwiseMixInputs {
            cached_input,
            channel_major,
            chunk_scale,
        })
    }

    fn chunkwise_conv_scale_for_test(
        weights: &XasrConvolutionModuleWeights,
        channel: usize,
        frame: usize,
        chunk_size: usize,
    ) -> Result<f32, String> {
        let dims = &weights.chunkwise_conv_scale.dims;
        if dims
            != &[
                2,
                weights.depthwise_chunkwise_conv.weight.dims[2],
                weights.depthwise_chunkwise_conv.weight.dims[0],
            ]
        {
            return Err(format!("unexpected chunkwise conv scale dims: {dims:?}"));
        }
        let channels = dims[1];
        let kernel = dims[2];
        let values = &weights.chunkwise_conv_scale.values;
        let left = if frame < kernel {
            values[channel * kernel + frame]
        } else {
            0.0
        };
        let right_base = channels * kernel;
        let right = if chunk_size < kernel {
            values[right_base + channel * kernel + kernel - chunk_size + frame]
        } else {
            let pad = chunk_size - kernel;
            if frame >= pad {
                values[right_base + channel * kernel + frame - pad]
            } else {
                0.0
            }
        };
        Ok(1.0 + left + right)
    }
}
