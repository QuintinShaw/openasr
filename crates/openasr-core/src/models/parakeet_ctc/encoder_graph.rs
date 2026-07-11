//! parakeet-ctc encoder graph (goal-1 S3c): the shared FastConformer
//! subsampling + conformer stack (`models::fastconformer`) with a CTC-head
//! tail, producing `[vocab, frames]` logits.
//!
//! The subsampling op sequence is a verbatim clone of cohere `encode()` (same
//! NeMo dw_striding: regular conv0+ReLU, depthwise conv2, pointwise conv3+ReLU,
//! depthwise conv5, pointwise conv6+ReLU), so the #1 frame-count risk is borrowed
//! from a proven impl. The conformer layers reuse `nn::encoder::conformer_block`
//! and the rel-pos table reuses `nn::encoder::build_relative_positional_encoding`;
//! the shared arena/subsampling/conformer-loop skeleton is
//! `models::fastconformer`, which `parakeet_tdt` builds on identically save
//! for its tail (joint encoder projection instead of a CTC head).

#![allow(dead_code)]

use std::path::Path;

use crate::ggml_runtime::{GgmlCpuGraphError, GgmlStaticTensor, WeightSlot};
use crate::models::fastconformer::{
    self, FastConformerEncoderCore, FastConformerGraphError, FastConformerStackConfig,
};
use crate::models::parakeet_ctc::graph_config::parakeet_ctc_encoder_graph_config;

use super::encoder_weights::ParakeetEncoderWeights;
use super::runtime_contract::ParakeetCtcExecutionMetadata;

const PARAKEET_ENCODER_GRAPH_CONTEXT_BYTES: usize = 768 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetEncoderError {
    #[error("parakeet-ctc encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("parakeet-ctc encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("parakeet-ctc encoder shape error: {reason}")]
    Shape { reason: String },
}

impl FastConformerGraphError for ParakeetEncoderError {
    fn graph_build_failed(step: &'static str, source: GgmlCpuGraphError) -> Self {
        Self::GraphBuildFailed { step, source }
    }
    fn shape(reason: String) -> Self {
        Self::Shape { reason }
    }
}

/// 80-bin log-mel features for one (single-chunk) utterance: `data` is row-major
/// `[n_mels, n_frames]` (n_mels fastest), matching the encoder's mel input tensor.
#[derive(Debug, Clone)]
pub(crate) struct ParakeetMelFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// Encoder output: per-frame CTC logits. `logits[frame * vocab_size + v]` is the
/// logit for token `v` at output frame `frame` (ggml frame-major buffer).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetCtcEncoderOutput {
    pub frame_count: usize,
    pub vocab_size: usize,
    pub logits: Vec<f32>,
}

pub(crate) struct ParakeetCtcEncoderGraph {
    metadata: ParakeetCtcExecutionMetadata,
    core: FastConformerEncoderCore,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
}

impl ParakeetCtcEncoderGraph {
    pub(crate) fn new(
        weights: &ParakeetEncoderWeights,
        metadata: ParakeetCtcExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, ParakeetEncoderError> {
        let config = parakeet_ctc_encoder_graph_config();
        // CTC head: `ctc.head.weight` is f16 `[1, d_model, vocab]` on disk (the
        // packer's reversed-dims convention) -- bound zero-copy + reshaped to
        // `[d_model, vocab]` for the head matmul in `encode`. Its bias stays
        // arena (1-D f32). Declared/uploaded through `build`'s tail closures so
        // it participates in the shared "declare everything, then upload
        // everything" arena pass alongside the subsampling + conformer layers.
        let (core, (ctc_head_weight, ctc_head_bias)) = FastConformerEncoderCore::build(
            config,
            PARAKEET_ENCODER_GRAPH_CONTEXT_BYTES,
            runtime_path,
            &weights.subsampling,
            &weights.layers,
            |arena, loaded| {
                let weight = fastconformer::bind_loaded(loaded, &weights.ctc_head_weight.name)?;
                let bias =
                    fastconformer::alloc_static(arena, &weights.ctc_head_bias, "ctc_head_b")?;
                Ok((weight, bias))
            },
            |arena, &(_, bias)| {
                fastconformer::upload_static(arena, bias, &weights.ctc_head_bias, "ctc_head_b")
            },
        )?;

        Ok(Self {
            metadata,
            core,
            ctc_head_weight,
            ctc_head_bias,
        })
    }

    pub(crate) fn encode(
        &mut self,
        mel: &ParakeetMelFeatures,
    ) -> Result<ParakeetCtcEncoderOutput, ParakeetEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.hidden_size;
        let mut graph = self.core.runner.start_graph();
        let stack = fastconformer::build_conformer_stack::<ParakeetEncoderError>(
            &mut graph,
            &self.core.arena,
            &self.core.sub,
            &self.core.layers,
            FastConformerStackConfig {
                hidden_size: d_model,
                n_heads: metadata.n_heads,
                head_dim: metadata.head_dim,
                conv_kernel: metadata.conv_kernel,
                subsampling_channels: metadata.subsampling_channels,
                scale_input: true,
            },
            mel.n_mels,
            mel.n_frames,
            "parakeet_mel",
            "parakeet_rel_pos",
        )?;

        // ----- CTC head -----
        // No extra final norm: conformer_block already applies its per-layer
        // out_norm as its last step, and parakeet has no separate encoder-level
        // final-norm tensor -- the CTC head consumes the last block's output.
        let head = graph
            .reshape_2d(
                self.ctc_head_weight.graph(&self.core.arena),
                d_model,
                metadata.vocab_size,
            )
            .map_err(|source| ParakeetEncoderError::GraphBuildFailed {
                step: "ctc_head_reshape",
                source,
            })?;
        let mut logits = graph.mul_mat(head, stack.state).map_err(|source| {
            ParakeetEncoderError::GraphBuildFailed {
                step: "ctc_head_matmul",
                source,
            }
        })?;
        logits = graph
            .add(logits, self.core.arena.graph_tensor(self.ctc_head_bias))
            .map_err(|source| ParakeetEncoderError::GraphBuildFailed {
                step: "ctc_head_bias",
                source,
            })?;
        graph
            .set_output(logits)
            .map_err(|source| ParakeetEncoderError::GraphBuildFailed {
                step: "set_output_logits",
                source,
            })?;

        // Peak-RSS lever: allocate the compute graph via the scheduler's gallocr
        // (liveness-based buffer REUSE) before uploading inputs, instead of the
        // per-tensor alloc_ctx_tensors that the uploads would otherwise force
        // (RSS = sum over conformer layers). Collapses peak RSS to the working set.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|source| ParakeetEncoderError::GraphBuildFailed {
                step: "prepare_outputs",
                source,
            })?;
        fastconformer::upload_graph_f32(&mut graph, stack.mel_t, &mel.data, "upload_mel")?;
        fastconformer::upload_graph_f32(&mut graph, stack.pos_t, &stack.positional, "upload_pos")?;

        let want = metadata
            .vocab_size
            .checked_mul(stack.subsampled_frames)
            .ok_or_else(|| ParakeetEncoderError::Shape {
                reason: "logits overflow".into(),
            })?;
        let logits = graph.compute_output_f32(logits, want).map_err(|error| {
            ParakeetEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(ParakeetCtcEncoderOutput {
            frame_count: stack.subsampled_frames,
            vocab_size: metadata.vocab_size,
            logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;
    use crate::models::fastconformer::graph::conv_out_dim;
    use crate::models::parakeet_ctc::encoder_weights::load_parakeet_ctc_encoder_weights;
    use crate::models::parakeet_ctc::runtime_contract::parse_parakeet_ctc_execution_metadata;
    use std::path::Path;

    /// S3c gate: build the full encoder graph from the real pack + run it on a
    /// dummy mel; assert finite [vocab, frames] logits with no ggml shape error
    /// (validates subsampling frame-count + conformer + head wiring). WER
    /// correctness is the S4 gate. Skipped when the pack is absent.
    #[test]
    fn encoder_graph_smoke_produces_finite_logits_when_pack_present() {
        let candidates = [Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-ctc-0.6b/openasr/parakeet-ctc-0.6b-fp16.oasr")];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            eprintln!("skipping: parakeet-ctc-0.6b pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_parakeet_ctc_execution_metadata(&gguf_metadata).expect("metadata");
        let weights = load_parakeet_ctc_encoder_weights(&reader, &metadata).expect("weights");

        let mut graph =
            ParakeetCtcEncoderGraph::new(&weights, metadata, Some(path.as_path())).expect("graph");
        let n_frames = 128usize;
        let mel = ParakeetMelFeatures {
            data: (0..metadata.n_mels * n_frames)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                .collect(),
            n_frames,
            n_mels: metadata.n_mels,
        };
        let out = graph.encode(&mel).expect("encode");
        let expected_frames = conv_out_dim(conv_out_dim(conv_out_dim(n_frames)));
        assert_eq!(out.frame_count, expected_frames);
        assert_eq!(out.vocab_size, metadata.vocab_size);
        assert_eq!(out.logits.len(), metadata.vocab_size * expected_frames);
        assert!(
            out.logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
    }
}
