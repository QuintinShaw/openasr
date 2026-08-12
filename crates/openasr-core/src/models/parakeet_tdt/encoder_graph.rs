//! parakeet-tdt encoder graph: the shared FastConformer subsampling +
//! conformer stack (`models::fastconformer`, also used by `parakeet_ctc`)
//! with the TDT differences: `scale_input` honored from pack metadata (false
//! for v3) and the joint ENCODER PROJECTION (`enc.proj`, d_model -> joint
//! hidden) applied in-graph instead of a CTC head. Output is the per-frame
//! projected encoder representation the shared TDT greedy control loop
//! consumes. CPU evaluates predictor/joint with its scalar oracle; accelerated
//! routes keep those neural stages in persistent device graphs.

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlStaticTensor, GgufRuntimeSourcePreflight, WeightSlot,
};
use crate::models::fastconformer::{
    self, FastConformerEncoderCore, FastConformerGraphError, FastConformerStackConfig,
};
use crate::models::parakeet_tdt::graph_config::parakeet_tdt_encoder_graph_config;

use super::device_decoder_graph::ParakeetTdtDeviceDecoder;
use super::encoder_weights::ParakeetTdtEncoderWeights;
use super::runtime_contract::ParakeetTdtExecutionMetadata;

const PARAKEET_TDT_TAIL_STATIC_TENSORS: usize = 1;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetTdtEncoderError {
    #[error("parakeet-tdt encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("parakeet-tdt encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("parakeet-tdt encoder shape error: {reason}")]
    Shape { reason: String },
}

impl FastConformerGraphError for ParakeetTdtEncoderError {
    fn graph_build_failed(step: &'static str, source: GgmlCpuGraphError) -> Self {
        Self::GraphBuildFailed { step, source }
    }
    fn shape(reason: String) -> Self {
        Self::Shape { reason }
    }
}

/// 128-bin log-mel features for one utterance: `data` is row-major
/// `[n_mels, n_frames]` (n_mels fastest).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtMelFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// Encoder output: per-frame joint-projected encoder features.
/// `features[frame * joint_hidden + j]` (frame-major).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtEncoderOutput {
    pub frame_count: usize,
    pub joint_hidden: usize,
    pub features: Vec<f32>,
}

pub(crate) struct ParakeetTdtEncoderGraph {
    metadata: ParakeetTdtExecutionMetadata,
    // Must drop before `core`: both persistent sessions reference the core's
    // runner/backend and mmap-backed loaded tensors.
    device_decoder: Option<ParakeetTdtDeviceDecoder>,
    core: FastConformerEncoderCore,
    enc_proj_weight: WeightSlot,
    enc_proj_bias: GgmlStaticTensor,
}

impl ParakeetTdtEncoderGraph {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let core = crate::models::parakeet_runtime_memory::graph_retained_bytes(&self.core)?;
        let decoder = self
            .device_decoder
            .as_ref()
            .map(ParakeetTdtDeviceDecoder::retained_system_memory_bytes)
            .transpose()?
            .unwrap_or(0);
        core.checked_add(decoder)
            .ok_or_else(|| "parakeet-tdt graph retained bytes overflowed".to_string())
    }

    pub(crate) fn new(
        weights: &ParakeetTdtEncoderWeights,
        metadata: ParakeetTdtExecutionMetadata,
        runtime_preflight: &GgufRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, ParakeetTdtEncoderError> {
        let config = parakeet_tdt_encoder_graph_config(backend);
        let (mut core, (enc_proj_weight, enc_proj_bias)) = FastConformerEncoderCore::build(
            config,
            PARAKEET_TDT_TAIL_STATIC_TENSORS,
            metadata.hidden_size,
            runtime_preflight,
            &weights.subsampling,
            &weights.layers,
            |arena, loaded| {
                let weight = fastconformer::bind_loaded(loaded, &weights.enc_proj_weight.name)?;
                let bias =
                    fastconformer::alloc_static(arena, &weights.enc_proj_bias, "enc_proj_b")?;
                Ok((weight, bias))
            },
            |arena, &(_, bias)| {
                fastconformer::upload_static(arena, bias, &weights.enc_proj_bias, "enc_proj_b")
            },
        )?;

        let device_decoder =
            if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Cpu {
                None
            } else {
                let loaded = core.loaded_weights.as_ref().ok_or_else(|| {
                    ParakeetTdtEncoderError::Shape {
                    reason:
                        "accelerated predictor/joint requires the verified loaded-weight context"
                            .to_string(),
                }
                })?;
                Some(
                    ParakeetTdtDeviceDecoder::new(&mut core.runner, loaded, metadata).map_err(
                        |source| ParakeetTdtEncoderError::GraphBuildFailed {
                            step: "device_predictor_joint",
                            source,
                        },
                    )?,
                )
            };

        Ok(Self {
            metadata,
            device_decoder,
            core,
            enc_proj_weight,
            enc_proj_bias,
        })
    }

    pub(crate) fn device_decoder_mut(&mut self) -> Option<&mut ParakeetTdtDeviceDecoder> {
        self.device_decoder.as_mut()
    }

    pub(crate) fn encode(
        &mut self,
        mel: &ParakeetTdtMelFeatures,
    ) -> Result<ParakeetTdtEncoderOutput, ParakeetTdtEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.hidden_size;
        let mut graph = self.core.runner.start_graph();
        let stack = fastconformer::build_conformer_stack::<ParakeetTdtEncoderError>(
            &mut graph,
            &self.core.arena,
            self.core.relative_position_inverse_timescales,
            &self.core.sub,
            &self.core.layers,
            FastConformerStackConfig {
                hidden_size: d_model,
                n_heads: metadata.n_heads,
                head_dim: metadata.head_dim,
                conv_kernel: metadata.conv_kernel,
                subsampling_channels: metadata.subsampling_channels,
                // Metadata-driven: parakeet-ctc's HF checkpoint scales, v3's
                // does NOT (the HF conversion folded NeMo's xscaling away).
                scale_input: metadata.scale_input,
            },
            metadata.n_mels,
            mel.n_mels,
            mel.n_frames,
            "parakeet_tdt_mel",
            "parakeet_tdt_rel_pos",
        )?;

        // ----- joint encoder projection: d_model -> joint_hidden -----
        // (In place of parakeet-ctc's CTC head; the last conformer block
        // already applied its out_norm, and there is no separate encoder-level
        // final norm in the checkpoint.)
        let proj = graph
            .reshape_2d(
                self.enc_proj_weight.graph(&self.core.arena),
                d_model,
                metadata.joint_hidden,
            )
            .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed {
                step: "enc_proj_reshape",
                source,
            })?;
        let mut features = graph.mul_mat(proj, stack.state).map_err(|source| {
            ParakeetTdtEncoderError::GraphBuildFailed {
                step: "enc_proj_matmul",
                source,
            }
        })?;
        features = graph
            .add(features, self.core.arena.graph_tensor(self.enc_proj_bias))
            .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed {
                step: "enc_proj_bias",
                source,
            })?;
        graph
            .set_output(features)
            .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed {
                step: "set_output_features",
                source,
            })?;

        graph
            .prepare_outputs_for_upload(&[features])
            .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed {
                step: "prepare_outputs",
                source,
            })?;
        fastconformer::upload_graph_f32(&mut graph, stack.mel_t, &mel.data, "upload_mel")?;
        fastconformer::upload_graph_f32(
            &mut graph,
            stack.positions_t,
            &stack.positions,
            "upload_positions",
        )?;

        let want = metadata
            .joint_hidden
            .checked_mul(stack.subsampled_frames)
            .ok_or_else(|| ParakeetTdtEncoderError::Shape {
                reason: "features overflow".into(),
            })?;
        let features = graph.compute_output_f32(features, want).map_err(|error| {
            ParakeetTdtEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(ParakeetTdtEncoderOutput {
            frame_count: stack.subsampled_frames,
            joint_hidden: metadata.joint_hidden,
            features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{
        build_runtime_tensor_reader_from_preflight, load_runtime_source_metadata_and_tensor_index,
    };
    use crate::models::fastconformer::graph::conv_out_dim;
    use crate::models::parakeet_tdt::encoder_weights::load_parakeet_tdt_encoder_weights;
    use crate::models::parakeet_tdt::runtime_contract::parse_parakeet_tdt_execution_metadata;
    use std::path::Path;

    /// Encoder smoke gate: build the full graph from the real pack + run it on
    /// a dummy mel; assert finite [joint_hidden, frames] features with no ggml
    /// shape error. Transcript correctness is the executor-stage gate.
    #[test]
    fn encoder_graph_smoke_produces_finite_features_when_pack_present() {
        let candidates = [Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr",
        )];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            eprintln!("skipping: parakeet-tdt-0.6b-v3 pack not present");
            return;
        };
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&path).expect("runtime preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let gguf_metadata = preflight.metadata.as_ref();
        let metadata = parse_parakeet_tdt_execution_metadata(gguf_metadata).expect("metadata");
        let weights = load_parakeet_tdt_encoder_weights(&reader, &metadata).expect("weights");

        let mut graph = ParakeetTdtEncoderGraph::new(
            &weights,
            metadata,
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect("graph");
        let n_frames = 128usize;
        let mel = ParakeetTdtMelFeatures {
            data: (0..metadata.n_mels * n_frames)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                .collect(),
            n_frames,
            n_mels: metadata.n_mels,
        };
        let out = graph.encode(&mel).expect("encode");
        let expected_frames = conv_out_dim(conv_out_dim(conv_out_dim(n_frames)));
        assert_eq!(out.frame_count, expected_frames);
        assert_eq!(out.joint_hidden, metadata.joint_hidden);
        assert_eq!(out.features.len(), metadata.joint_hidden * expected_frames);
        assert!(
            out.features.iter().all(|v| v.is_finite()),
            "features must be finite"
        );
    }

    #[test]
    fn encode_rejects_mel_n_mels_that_drifts_from_metadata() {
        let candidates = [Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr",
        )];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            eprintln!("skipping: parakeet-tdt-0.6b-v3 pack not present");
            return;
        };
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&path).expect("runtime preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let gguf_metadata = preflight.metadata.as_ref();
        let metadata = parse_parakeet_tdt_execution_metadata(gguf_metadata).expect("metadata");
        let weights = load_parakeet_tdt_encoder_weights(&reader, &metadata).expect("weights");

        let mut graph = ParakeetTdtEncoderGraph::new(
            &weights,
            metadata,
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect("graph");
        let n_frames = 128usize;
        let wrong_n_mels = metadata.n_mels.saturating_add(1).max(1);
        let mel = ParakeetTdtMelFeatures {
            data: vec![0.0; wrong_n_mels * n_frames],
            n_frames,
            n_mels: wrong_n_mels,
        };
        let err = graph
            .encode(&mel)
            .expect_err("encode must reject mel.n_mels that drifts from metadata");
        match err {
            ParakeetTdtEncoderError::Shape { reason } => {
                assert!(
                    reason.contains(&wrong_n_mels.to_string())
                        && reason.contains(&metadata.n_mels.to_string()),
                    "shape error must report both values, got {reason:?}"
                );
            }
            other => panic!("expected Shape error, got {other:?}"),
        }
    }
}
