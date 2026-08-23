//! Device-resident stateless predictor and RNN-T joiner for X-ASR.
//!
//! The encoder frame, decoder context, and complete joiner logits cross the
//! device boundary. Token selection and probability calculation stay on the
//! host so every provider shares the XASR last-max oracle. Quantized joiner
//! matrices remain in their stored ggml type instead of acquiring a host-f32
//! copy.

use crate::ggml_runtime::{
    GgmlCpuGraphConfig, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlPersistentGraphSession,
    GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReader, GgufWeightTensorPayload,
};

use super::graph_config::{DEVICE_HEAD_GRAPH_SIZE, xasr_zipformer_device_head_graph_config};
use super::greedy::{XasrGreedyDecodeBackend, argmax};
use super::package_import::compact_xasr_name;
use super::runtime_contract::{
    XASR_DECODER_CONV_GROUPS, XASR_OUTPUT_DOWNSAMPLING_FACTOR, XasrRuntimeTensorContract,
    XasrZipformerExecutionMetadata,
};

const STATIC_TENSOR_COUNT: usize = 9;
struct ProjectionGraph {
    session: GgmlPersistentGraphSession,
    input: GgmlCpuTensor<'static>,
}

struct JointGraph {
    session: GgmlPersistentGraphSession,
    encoder_frame: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
}

struct SpeculativeBlankGraph {
    session: GgmlPersistentGraphSession,
    encoder_frames: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
    frames: usize,
}

struct HeadWeights {
    decoder_embedding: GgmlStaticTensor,
    decoder_conv: GgmlStaticTensor,
    encoder_proj_weight: GgmlStaticTensor,
    encoder_proj_bias: GgmlStaticTensor,
    decoder_proj_weight: GgmlStaticTensor,
    decoder_proj_bias: GgmlStaticTensor,
    output_weight: GgmlStaticTensor,
    output_bias: GgmlStaticTensor,
    decoder_projection: GgmlStaticTensor,
}

/// Field order is intentional: the persistent sessions contain raw references
/// into both the runner and arena, so all graphs must drop first.
pub(crate) struct XasrDeviceHead {
    decoder_projection: ProjectionGraph,
    joint: JointGraph,
    speculative_blank: Option<SpeculativeBlankGraph>,
    runner: GgmlCpuGraphRunner,
    arena: GgmlStaticTensorArena,
    context_size: usize,
    decoder_dim: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: u32,
    last_token: Option<u32>,
    last_probability: f32,
}

fn checked_payload<'a>(
    reader: &'a GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    upstream_name: &str,
) -> Result<GgufWeightTensorPayload<'a>, String> {
    let name = compact_xasr_name(upstream_name);
    let shape = contract.shape(&name).ok_or_else(|| {
        format!("tensor '{name}' is not part of the xasr-zipformer runtime contract")
    })?;
    let payload = reader
        .weight_tensor_payload_by_name(&name)
        .map_err(|error| error.to_string())?;
    if !shape.matches(&payload.dims) {
        return Err(format!(
            "xasr device head tensor '{name}' has dims {:?}: {}",
            payload.dims,
            shape.describe()
        ));
    }
    Ok(payload)
}

fn new_matmul_weight(
    arena: &GgmlStaticTensorArena,
    payload: &GgufWeightTensorPayload<'_>,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, String> {
    let [ne0, ne1]: [usize; 2] = payload.dims.as_slice().try_into().map_err(|_| {
        format!(
            "xasr device head tensor '{}' must be rank 2",
            payload.metadata.name
        )
    })?;
    arena
        .new_matmul_weight_2d_typed(ne0, ne1, payload.element_type.ggml_type(), tensor_name)
        .map_err(|error| error.to_string())
}

fn upload_weight(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    payload: &GgufWeightTensorPayload<'_>,
    tensor_name: &'static str,
) -> Result<(), String> {
    arena
        .set_bytes_slice(tensor, payload.bytes, tensor_name)
        .map_err(|error| error.to_string())
}

impl XasrDeviceHead {
    pub(crate) fn new(
        reader: &GgufTensorDataReader,
        metadata: &XasrZipformerExecutionMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        speculative_blank_batch: bool,
    ) -> Result<Self, String> {
        let contract = XasrRuntimeTensorContract::for_metadata(metadata);
        let decoder_embedding = checked_payload(reader, &contract, "decoder.embedding.weight")?;
        let decoder_conv = checked_payload(reader, &contract, "decoder.conv.weight")?;
        let encoder_proj_weight = checked_payload(reader, &contract, "joiner.encoder_proj.weight")?;
        let encoder_proj_bias = checked_payload(reader, &contract, "joiner.encoder_proj.bias")?;
        let decoder_proj_weight = checked_payload(reader, &contract, "joiner.decoder_proj.weight")?;
        let decoder_proj_bias = checked_payload(reader, &contract, "joiner.decoder_proj.bias")?;
        let output_weight = checked_payload(reader, &contract, "joiner.output_linear.weight")?;
        let output_bias = checked_payload(reader, &contract, "joiner.output_linear.bias")?;

        let mut runner = GgmlCpuGraphRunner::new(xasr_zipformer_device_head_graph_config(backend))
            .map_err(|error| error.to_string())?;
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
                STATIC_TENSOR_COUNT,
            ))
            .map_err(|error| error.to_string())?;

        // Allocate every tensor before the first upload freezes the arena.
        let weights = HeadWeights {
            decoder_embedding: arena
                .new_tensor_from_weight_payload(&decoder_embedding)
                .map_err(|error| error.to_string())?,
            decoder_conv: arena
                .new_tensor_from_weight_payload(&decoder_conv)
                .map_err(|error| error.to_string())?,
            encoder_proj_weight: new_matmul_weight(
                &arena,
                &encoder_proj_weight,
                "xasr_head_encoder_proj_weight",
            )?,
            encoder_proj_bias: arena
                .new_tensor_from_weight_payload(&encoder_proj_bias)
                .map_err(|error| error.to_string())?,
            decoder_proj_weight: new_matmul_weight(
                &arena,
                &decoder_proj_weight,
                "xasr_head_decoder_proj_weight",
            )?,
            decoder_proj_bias: arena
                .new_tensor_from_weight_payload(&decoder_proj_bias)
                .map_err(|error| error.to_string())?,
            output_weight: new_matmul_weight(&arena, &output_weight, "xasr_head_output_weight")?,
            output_bias: arena
                .new_tensor_from_weight_payload(&output_bias)
                .map_err(|error| error.to_string())?,
            decoder_projection: arena
                .new_tensor_1d_f32(metadata.joiner_dim, "xasr_head_decoder_projection")
                .map_err(|error| error.to_string())?,
        };

        for (tensor, payload, name) in [
            (
                weights.decoder_embedding,
                &decoder_embedding,
                "xasr_head_decoder_embedding",
            ),
            (
                weights.decoder_conv,
                &decoder_conv,
                "xasr_head_decoder_conv",
            ),
            (
                weights.encoder_proj_weight,
                &encoder_proj_weight,
                "xasr_head_encoder_proj_weight",
            ),
            (
                weights.encoder_proj_bias,
                &encoder_proj_bias,
                "xasr_head_encoder_proj_bias",
            ),
            (
                weights.decoder_proj_weight,
                &decoder_proj_weight,
                "xasr_head_decoder_proj_weight",
            ),
            (
                weights.decoder_proj_bias,
                &decoder_proj_bias,
                "xasr_head_decoder_proj_bias",
            ),
            (
                weights.output_weight,
                &output_weight,
                "xasr_head_output_weight",
            ),
            (weights.output_bias, &output_bias, "xasr_head_output_bias"),
        ] {
            upload_weight(&mut arena, tensor, payload, name)?;
        }
        arena
            .set_f32_slice(
                weights.decoder_projection,
                &vec![0.0_f32; metadata.joiner_dim],
                "xasr_head_decoder_projection",
            )
            .map_err(|error| error.to_string())?;

        let decoder_projection =
            Self::build_decoder_projection(&mut runner, &arena, &weights, metadata)?;
        let joint = Self::build_joint(&mut runner, &arena, &weights, metadata)?;
        let speculative_blank = if speculative_blank_batch {
            if !metadata
                .decode_chunk_len
                .is_multiple_of(XASR_OUTPUT_DOWNSAMPLING_FACTOR)
            {
                return Err(format!(
                    "xasr decode chunk length {} is not divisible by output downsampling factor {}",
                    metadata.decode_chunk_len, XASR_OUTPUT_DOWNSAMPLING_FACTOR
                ));
            }
            let frames = metadata.decode_chunk_len / XASR_OUTPUT_DOWNSAMPLING_FACTOR;
            if frames == 0 {
                return Err("xasr speculative blank batch requires at least one frame".to_string());
            }
            Some(Self::build_speculative_blank(
                &mut runner,
                &arena,
                &weights,
                metadata,
                frames,
            )?)
        } else {
            None
        };

        Ok(Self {
            decoder_projection,
            joint,
            speculative_blank,
            runner,
            arena,
            context_size: metadata.decoder_context_size,
            decoder_dim: metadata.decoder_dim(),
            encoder_dim: metadata.encoder_output_dim(),
            vocab_size: metadata.vocab_size,
            blank_id: metadata.blank_id,
            last_token: None,
            last_probability: 0.0,
        })
    }

    fn build_decoder_projection(
        runner: &mut GgmlCpuGraphRunner,
        arena: &GgmlStaticTensorArena,
        weights: &HeadWeights,
        metadata: &XasrZipformerExecutionMetadata,
    ) -> Result<ProjectionGraph, String> {
        let mut session = runner
            .start_persistent_graph_session_with_node_capacity(DEVICE_HEAD_GRAPH_SIZE)
            .map_err(|error| error.to_string())?;
        let graph = session.builder();
        let decoder_context = graph
            .new_tensor_1d_i32(metadata.decoder_context_size, "xasr_head_context")
            .map_err(|error| error.to_string())?;
        graph
            .set_input(decoder_context)
            .map_err(|error| error.to_string())?;
        let embedded = graph
            .get_rows(weights.decoder_embedding.as_graph_tensor(), decoder_context)
            .and_then(|value| graph.transpose(value))
            .and_then(|value| graph.cont(value))
            .map_err(|error| error.to_string())?;
        let in_per_group = metadata.decoder_dim() / XASR_DECODER_CONV_GROUPS;
        let packed_width = metadata
            .decoder_context_size
            .checked_mul(in_per_group)
            .ok_or_else(|| "xasr decoder packed group width overflowed".to_string())?;
        let input = graph
            .reshape_3d(embedded, packed_width, 1, XASR_DECODER_CONV_GROUPS)
            .map_err(|error| error.to_string())?;
        let conv = graph
            .reshape_3d(
                weights.decoder_conv.as_graph_tensor(),
                packed_width,
                in_per_group,
                XASR_DECODER_CONV_GROUPS,
            )
            .and_then(|kernel| graph.mul_mat(kernel, input))
            .and_then(|value| graph.reshape_1d(value, metadata.decoder_dim()))
            .and_then(|value| graph.relu(value))
            .map_err(|error| error.to_string())?;
        let decoder_projection = graph
            .mul_mat(weights.decoder_proj_weight.as_graph_tensor(), conv)
            .and_then(|value| graph.add(value, weights.decoder_proj_bias.as_graph_tensor()))
            .map_err(|error| error.to_string())?;
        let write = graph
            .cpy(
                decoder_projection,
                arena.graph_tensor(weights.decoder_projection),
            )
            .map_err(|error| error.to_string())?;
        graph
            .add_side_effect_root(write)
            .and_then(|()| graph.prepare_side_effects_for_upload())
            .map_err(|error| error.to_string())?;
        Ok(ProjectionGraph {
            session,
            input: decoder_context,
        })
    }

    fn build_joint(
        runner: &mut GgmlCpuGraphRunner,
        arena: &GgmlStaticTensorArena,
        weights: &HeadWeights,
        metadata: &XasrZipformerExecutionMetadata,
    ) -> Result<JointGraph, String> {
        let mut session = runner
            .start_persistent_graph_session_with_node_capacity(DEVICE_HEAD_GRAPH_SIZE)
            .map_err(|error| error.to_string())?;
        let graph = session.builder();
        let encoder_frame = graph
            .new_tensor_1d_f32(metadata.encoder_output_dim(), "xasr_head_encoder_frame")
            .map_err(|error| error.to_string())?;
        graph
            .set_input(encoder_frame)
            .map_err(|error| error.to_string())?;
        let encoder_projection = graph
            .mul_mat(weights.encoder_proj_weight.as_graph_tensor(), encoder_frame)
            .and_then(|value| graph.add(value, weights.encoder_proj_bias.as_graph_tensor()))
            .map_err(|error| error.to_string())?;
        let joined = graph
            .add(
                encoder_projection,
                arena.graph_tensor(weights.decoder_projection),
            )
            .and_then(|value| graph.tanh(value))
            .map_err(|error| error.to_string())?;
        let logits = graph
            .mul_mat(weights.output_weight.as_graph_tensor(), joined)
            .and_then(|value| graph.add(value, weights.output_bias.as_graph_tensor()))
            .map_err(|error| error.to_string())?;
        // Keep token selection on the host until the shared runtime exposes a
        // named, provider-specific last-max capability. Reading the complete
        // row preserves the XASR host oracle's exact tie behavior on every
        // backend while the joiner itself remains device-resident.
        graph
            .set_output(logits)
            .map_err(|error| error.to_string())?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|error| error.to_string())?;
        Ok(JointGraph {
            session,
            encoder_frame,
            logits,
        })
    }

    fn build_speculative_blank(
        runner: &mut GgmlCpuGraphRunner,
        arena: &GgmlStaticTensorArena,
        weights: &HeadWeights,
        metadata: &XasrZipformerExecutionMetadata,
        frames: usize,
    ) -> Result<SpeculativeBlankGraph, String> {
        let mut session = runner
            .start_persistent_graph_session_with_node_capacity(DEVICE_HEAD_GRAPH_SIZE)
            .map_err(|error| error.to_string())?;
        let graph = session.builder();
        let encoder_frames = graph
            .new_tensor_2d_f32(
                metadata.encoder_output_dim(),
                frames,
                "xasr_head_speculative_encoder_frames",
            )
            .map_err(|error| error.to_string())?;
        graph
            .set_input(encoder_frames)
            .map_err(|error| error.to_string())?;
        let encoder_projection = graph
            .mul_mat(
                weights.encoder_proj_weight.as_graph_tensor(),
                encoder_frames,
            )
            .and_then(|value| graph.add(value, weights.encoder_proj_bias.as_graph_tensor()))
            .map_err(|error| error.to_string())?;
        let joined = graph
            .add(
                encoder_projection,
                arena.graph_tensor(weights.decoder_projection),
            )
            .and_then(|value| graph.tanh(value))
            .map_err(|error| error.to_string())?;
        let logits = graph
            .mul_mat(weights.output_weight.as_graph_tensor(), joined)
            .and_then(|value| graph.add(value, weights.output_bias.as_graph_tensor()))
            .map_err(|error| error.to_string())?;
        // The speculative path uses the same host last-max oracle as scalar
        // decode. A compact device top1 would authorize a tie policy that is
        // not part of the named runtime capability contract.
        graph
            .set_output(logits)
            .map_err(|error| error.to_string())?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|error| error.to_string())?;
        Ok(SpeculativeBlankGraph {
            session,
            encoder_frames,
            logits,
            frames,
        })
    }
}

impl XasrGreedyDecodeBackend for XasrDeviceHead {
    fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String> {
        if frame.len() != self.encoder_dim {
            return Err(format!(
                "xasr device head encoder frame has {} values, expected {}",
                frame.len(),
                self.encoder_dim
            ));
        }
        let graph = self.joint.session.builder();
        graph
            .set_f32_slice(self.joint.encoder_frame, frame, "xasr_head_encoder_frame")
            .map_err(|error| error.to_string())
    }

    fn project_decoder_context(&mut self, context: &[u32]) -> Result<(), String> {
        if context.len() != self.context_size {
            return Err(format!(
                "xasr device head context has {} tokens, expected {}",
                context.len(),
                self.context_size
            ));
        }
        let token_ids = context
            .iter()
            .map(|&token| {
                if token as usize >= self.vocab_size {
                    return Err(format!(
                        "xasr device head token {token} exceeds vocab {}",
                        self.vocab_size
                    ));
                }
                i32::try_from(token)
                    .map_err(|_| format!("xasr device head token {token} exceeds i32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let graph = self.decoder_projection.session.builder();
        graph
            .set_i32_slice(
                self.decoder_projection.input,
                &token_ids,
                "xasr_head_context",
            )
            .and_then(|()| graph.compute_side_effects())
            .map_err(|error| error.to_string())
    }

    fn next_token(&mut self) -> Result<u32, String> {
        let graph = self.joint.session.builder();
        let logits = graph
            .compute_output_f32(self.joint.logits, self.vocab_size)
            .map_err(|error| error.to_string())?;
        let token = argmax(&logits)
            .ok_or_else(|| "xasr device head produced no finite logits".to_string())?;
        let probability = crate::models::seq2seq_greedy_decode::token_softmax_probability(
            &logits,
            token as usize,
        );
        if !probability.is_finite() {
            return Err("xasr device head selected a non-finite probability".to_string());
        }
        self.last_token = Some(token);
        self.last_probability = probability;
        Ok(token)
    }

    fn token_probability(&self, token: u32) -> Result<f32, String> {
        if self.last_token != Some(token) {
            return Err(format!(
                "xasr device head probability requested for token {token} before selection"
            ));
        }
        Ok(self.last_probability)
    }

    fn speculative_blank_prefix_len(
        &mut self,
        context: Option<&[u32]>,
        encoder_frames: &[f32],
        frame_count: usize,
        encoder_dim: usize,
    ) -> Result<Option<usize>, String> {
        let Some(speculative_frames) = self
            .speculative_blank
            .as_ref()
            .map(|speculative| speculative.frames)
        else {
            return Ok(None);
        };
        if frame_count < speculative_frames || encoder_dim != self.encoder_dim {
            return Ok(None);
        }
        if let Some(context) = context {
            self.project_decoder_context(context)?;
        }
        let Some(speculative) = self.speculative_blank.as_mut() else {
            return Err("xasr speculative graph disappeared during context projection".to_string());
        };
        let value_count = speculative
            .frames
            .checked_mul(encoder_dim)
            .ok_or_else(|| "xasr speculative frame shape overflowed".to_string())?;
        if self.vocab_size == 0 {
            return Err("xasr speculative device head requires a non-empty vocabulary".to_string());
        }
        let logits_count = speculative
            .frames
            .checked_mul(self.vocab_size)
            .ok_or_else(|| "xasr speculative logits shape overflowed".to_string())?;
        let graph = speculative.session.builder();
        graph
            .set_f32_slice(
                speculative.encoder_frames,
                &encoder_frames[..value_count],
                "xasr_head_speculative_encoder_frames",
            )
            .map_err(|error| error.to_string())?;
        let logits = graph
            .compute_output_f32(speculative.logits, logits_count)
            .map_err(|error| error.to_string())?;
        let mut first_non_blank = speculative.frames;
        for (frame, frame_logits) in logits.chunks_exact(self.vocab_size).enumerate() {
            let token = argmax(frame_logits).ok_or_else(|| {
                format!("xasr speculative device head produced no finite logits for frame {frame}")
            })?;
            if token != self.blank_id {
                first_non_blank = frame;
                break;
            }
        }
        Ok(Some(first_non_blank))
    }
}

impl XasrDeviceHead {
    pub(crate) fn initial_context(&self) -> Vec<u32> {
        vec![self.blank_id; self.context_size]
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> u64 {
        // This owner keeps no variable-size Rust backing. Native metadata
        // contexts, graph allocations, and the WEIGHTS arena are admitted by
        // the shared ggml layer and carry their own leases.
        let _keep_alive = (&self.runner, &self.arena, self.decoder_dim);
        0
    }

    #[cfg(test)]
    pub(super) fn persistent_graph_node_counts_for_test(&self) -> Vec<Option<usize>> {
        let mut counts = vec![
            self.decoder_projection
                .session
                .prepared_native_node_count_for_test(),
            self.joint.session.prepared_native_node_count_for_test(),
        ];
        if let Some(speculative) = &self.speculative_blank {
            counts.push(speculative.session.prepared_native_node_count_for_test());
        }
        counts
    }
}
