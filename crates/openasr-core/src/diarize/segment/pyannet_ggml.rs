//! Full-device ggml execution for PyanNet segmentation-3.0.
//!
//! SincNet, the four bidirectional LSTMs, and the classifier all execute on the
//! selected backend. Host code only supplies waveform samples and performs the
//! reversible layout/reordering between recurrent direction graphs. All model
//! weights are uploaded once, then the construction-only host materialization
//! is dropped so accelerated execution does not retain a second f32 copy.

use thiserror::Error;

use super::pyannet::{ALPHA, HIDDEN, LSTM_WEIGHTS, NUM_CLASSES, PyannetModel, output_frame_count};
use crate::{
    device::execution_policy::ExecutionPlacement,
    diarize::embed::weights::{Weights, WeightsError},
    ggml_runtime::{
        GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
        GgmlCpuGraphRunner, GgmlCpuTensor, GgmlPersistentGraphSession, GgmlStaticTensor,
        GgmlStaticTensorArena,
    },
};

const INPUT_FEATURES: usize = 60;
const GATE_COUNT: usize = 4;
const SINC_EPSILON: f32 = 1.0e-5;
const SINC_GRAPH_SIZE: usize = 1 << 9;
// One reusable direction graph contains one full recurrent scan. Keeping the
// two input-width variants separate avoids padding the first layer from 60 to
// 256 channels, while reducing live graph metadata by more than 5x versus one
// monolithic four-layer bidirectional graph.
const DIRECTION_GRAPH_SIZE: usize = 1 << 14;
const MAX_DIRECTION_FRAMES: usize = 800;
// 13 SincNet handles + 6 stacked LSTM handles + 6 classifier handles + zero.
const ARENA_TENSORS: usize = 26;
const ACCELERATED_HOST_COMMITMENT_BYTES: u64 = 64 * 1024;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const LSTM_GROUP_SPECS: [(usize, usize); 2] = [(INPUT_FEATURES, 2), (2 * HIDDEN, 6)];

#[derive(Debug, Error)]
pub(crate) enum PyannetGgmlError {
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Graph(#[from] GgmlCpuGraphError),
    #[error("PyanNet recurrent input has {got} values, expected {expected}")]
    InvalidFeaturePayload { got: usize, expected: usize },
    #[error("PyanNet recurrent input has {frames} frames, maximum supported is {maximum}")]
    TooManyFrames { frames: usize, maximum: usize },
    #[error("PyanNet SincNet output has {got} values, expected {expected}")]
    InvalidSincOutput { got: usize, expected: usize },
}

#[derive(Clone, Copy)]
struct SincConvHandles {
    weight: GgmlStaticTensor,
    bias: Option<GgmlStaticTensor>,
    norm_weight: GgmlStaticTensor,
    norm_bias: GgmlStaticTensor,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
}

#[derive(Clone, Copy)]
struct SincHandles {
    waveform_norm_weight: GgmlStaticTensor,
    waveform_norm_bias: GgmlStaticTensor,
    convs: [SincConvHandles; 3],
}

#[derive(Clone, Copy)]
struct LstmStackHandles {
    input_features: usize,
    slots: usize,
    input_weights: GgmlStaticTensor,
    recurrent_weights: GgmlStaticTensor,
    combined_biases: GgmlStaticTensor,
}

#[derive(Clone, Copy)]
struct LinearHandles {
    weight: GgmlStaticTensor,
    bias: GgmlStaticTensor,
}

struct WeightHandles {
    sinc: SincHandles,
    lstm_groups: [LstmStackHandles; 2],
    classifier: [LinearHandles; 3],
    zero_state: GgmlStaticTensor,
}

impl WeightHandles {
    fn allocate(arena: &GgmlStaticTensorArena) -> Result<Self, GgmlCpuGraphError> {
        Ok(Self {
            sinc: SincHandles {
                waveform_norm_weight: arena.new_tensor_1d_f32(1, "pyannet_waveform_norm_weight")?,
                waveform_norm_bias: arena.new_tensor_1d_f32(1, "pyannet_waveform_norm_bias")?,
                convs: [
                    allocate_sinc_conv(arena, 1, 80, 251, 10, false)?,
                    allocate_sinc_conv(arena, 80, 60, 5, 1, true)?,
                    allocate_sinc_conv(arena, 60, 60, 5, 1, true)?,
                ],
            },
            lstm_groups: [
                allocate_lstm_stack(arena, INPUT_FEATURES, 2)?,
                allocate_lstm_stack(arena, 2 * HIDDEN, 6)?,
            ],
            classifier: [
                allocate_linear(arena, 2 * HIDDEN, HIDDEN)?,
                allocate_linear(arena, HIDDEN, HIDDEN)?,
                allocate_linear(arena, HIDDEN, NUM_CLASSES)?,
            ],
            zero_state: arena.new_tensor_1d_f32(HIDDEN, "pyannet_zero_state")?,
        })
    }

    fn upload(
        &self,
        arena: &mut GgmlStaticTensorArena,
        weights: &Weights,
    ) -> Result<(), PyannetGgmlError> {
        arena.set_f32_slice(
            self.sinc.waveform_norm_weight,
            weights.get("sincnet.wav_norm1d.weight")?,
            "pyannet_waveform_norm_weight",
        )?;
        arena.set_f32_slice(
            self.sinc.waveform_norm_bias,
            weights.get("sincnet.wav_norm1d.bias")?,
            "pyannet_waveform_norm_bias",
        )?;
        for (index, handles) in self.sinc.convs.iter().copied().enumerate() {
            let (weight_name, bias_name) = if index == 0 {
                ("/sincnet/conv1d.0/Concat_2_output_0".to_string(), None)
            } else {
                (
                    format!("sincnet.conv1d.{index}.weight"),
                    Some(format!("sincnet.conv1d.{index}.bias")),
                )
            };
            arena.set_f32_slice(
                handles.weight,
                weights.get(&weight_name)?,
                "pyannet_sinc_conv_weight",
            )?;
            if let (Some(handle), Some(name)) = (handles.bias, bias_name) {
                arena.set_f32_slice(handle, weights.get(&name)?, "pyannet_sinc_conv_bias")?;
            }
            arena.set_f32_slice(
                handles.norm_weight,
                weights.get(&format!("sincnet.norm1d.{index}.weight"))?,
                "pyannet_sinc_norm_weight",
            )?;
            arena.set_f32_slice(
                handles.norm_bias,
                weights.get(&format!("sincnet.norm1d.{index}.bias"))?,
                "pyannet_sinc_norm_bias",
            )?;
        }
        let gate = GATE_COUNT * HIDDEN;
        let mut input_weight_stacks = LSTM_GROUP_SPECS
            .map(|(input, slots)| vec![0.0f32; input.saturating_mul(gate).saturating_mul(slots)]);
        let mut recurrent_weight_stacks = LSTM_GROUP_SPECS
            .map(|(_, slots)| vec![0.0f32; HIDDEN.saturating_mul(gate).saturating_mul(slots)]);
        let mut combined_bias_stacks =
            LSTM_GROUP_SPECS.map(|(_, slots)| vec![0.0f32; gate.saturating_mul(slots)]);

        for (layer, (w_name, r_name, b_name)) in LSTM_WEIGHTS.into_iter().enumerate() {
            let input = if layer == 0 {
                INPUT_FEATURES
            } else {
                2 * HIDDEN
            };
            let group = usize::from(layer != 0);
            let w = weights.get(w_name)?;
            let r = weights.get(r_name)?;
            let bias = weights.get(b_name)?;
            for direction in 0..2 {
                let slot = if layer == 0 {
                    direction
                } else {
                    (layer - 1) * 2 + direction
                };
                let w_start = direction * gate * input;
                let r_start = direction * gate * HIDDEN;
                let b_start = direction * 2 * gate;
                let stacked_w_start = slot * gate * input;
                input_weight_stacks[group][stacked_w_start..stacked_w_start + gate * input]
                    .copy_from_slice(&w[w_start..w_start + gate * input]);
                let stacked_r_start = slot * gate * HIDDEN;
                recurrent_weight_stacks[group][stacked_r_start..stacked_r_start + gate * HIDDEN]
                    .copy_from_slice(&r[r_start..r_start + gate * HIDDEN]);
                let stacked_b_start = slot * gate;
                for index in 0..gate {
                    combined_bias_stacks[group][stacked_b_start + index] =
                        bias[b_start + index] + bias[b_start + gate + index];
                }
            }
        }
        for (group, handles) in self.lstm_groups.iter().enumerate() {
            arena.set_f32_slice(
                handles.input_weights,
                &input_weight_stacks[group],
                "pyannet_lstm_input_weights",
            )?;
            arena.set_f32_slice(
                handles.recurrent_weights,
                &recurrent_weight_stacks[group],
                "pyannet_lstm_recurrent_weights",
            )?;
            arena.set_f32_slice(
                handles.combined_biases,
                &combined_bias_stacks[group],
                "pyannet_lstm_combined_biases",
            )?;
        }
        for (handles, (weight_name, bias_name, input, output)) in self.classifier.iter().zip([
            ("onnx::MatMul_915", "linear.0.bias", 2 * HIDDEN, HIDDEN),
            ("onnx::MatMul_916", "linear.1.bias", HIDDEN, HIDDEN),
            ("onnx::MatMul_917", "classifier.bias", HIDDEN, NUM_CLASSES),
        ]) {
            let transposed = transpose_onnx_matmul_weight(weights.get(weight_name)?, input, output);
            arena.set_f32_slice(handles.weight, &transposed, "pyannet_linear_weight")?;
            arena.set_f32_slice(handles.bias, weights.get(bias_name)?, "pyannet_linear_bias")?;
        }
        arena.set_f32_slice(self.zero_state, &vec![0.0f32; HIDDEN], "pyannet_zero_state")?;
        Ok(())
    }
}

fn allocate_sinc_conv(
    arena: &GgmlStaticTensorArena,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    with_bias: bool,
) -> Result<SincConvHandles, GgmlCpuGraphError> {
    Ok(SincConvHandles {
        weight: arena.new_tensor_4d_f32(
            kernel,
            1,
            input_channels,
            output_channels,
            "pyannet_sinc_conv_weight",
        )?,
        bias: with_bias
            .then(|| arena.new_tensor_1d_f32(output_channels, "pyannet_sinc_conv_bias"))
            .transpose()?,
        norm_weight: arena.new_tensor_1d_f32(output_channels, "pyannet_sinc_norm_weight")?,
        norm_bias: arena.new_tensor_1d_f32(output_channels, "pyannet_sinc_norm_bias")?,
        input_channels,
        output_channels,
        kernel,
        stride,
    })
}

fn allocate_lstm_stack(
    arena: &GgmlStaticTensorArena,
    input: usize,
    slots: usize,
) -> Result<LstmStackHandles, GgmlCpuGraphError> {
    let gate = GATE_COUNT * HIDDEN;
    Ok(LstmStackHandles {
        input_features: input,
        slots,
        input_weights: arena.new_tensor_2d_f32(
            input,
            gate * slots,
            "pyannet_lstm_input_weights",
        )?,
        recurrent_weights: arena.new_tensor_2d_f32(
            HIDDEN,
            gate * slots,
            "pyannet_lstm_recurrent_weights",
        )?,
        combined_biases: arena.new_tensor_2d_f32(
            1,
            gate * slots,
            "pyannet_lstm_combined_biases",
        )?,
    })
}

fn allocate_linear(
    arena: &GgmlStaticTensorArena,
    input: usize,
    output: usize,
) -> Result<LinearHandles, GgmlCpuGraphError> {
    Ok(LinearHandles {
        weight: arena.new_tensor_2d_f32(input, output, "pyannet_linear_weight")?,
        bias: arena.new_tensor_1d_f32(output, "pyannet_linear_bias")?,
    })
}

/// ONNX MatMul stores `[input, output]` row-major, while ggml's matrix leaf
/// for `mul_mat(weight, states)` stores output rows with input contiguous.
fn transpose_onnx_matmul_weight(values: &[f32], input: usize, output: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; input * output];
    for input_index in 0..input {
        for output_index in 0..output {
            transposed[output_index * input + input_index] =
                values[input_index * output + output_index];
        }
    }
    transposed
}

struct ResidentWeights {
    handles: WeightHandles,
    arena: GgmlStaticTensorArena,
}

struct PersistentSincGraph {
    session: GgmlPersistentGraphSession,
    input: GgmlCpuTensor<'static>,
    output: GgmlCpuTensor<'static>,
    samples: usize,
    frames: usize,
}

struct PersistentDirectionGraph {
    session: GgmlPersistentGraphSession,
    input: GgmlCpuTensor<'static>,
    row_indices: GgmlCpuTensor<'static>,
    output: GgmlCpuTensor<'static>,
    frames: usize,
}

/// Thread-confined full-device PyanNet runtime.
///
/// Field order is load-bearing: graph tensors drop before their static arena,
/// and the arena drops before the backend runner that allocated it.
pub(crate) struct PyannetGgmlRuntime {
    sinc_graph: Option<PersistentSincGraph>,
    direction_graphs: [Option<PersistentDirectionGraph>; 2],
    resident: ResidentWeights,
    runner: GgmlCpuGraphRunner,
}

impl PyannetGgmlRuntime {
    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Result<Self, PyannetGgmlError> {
        Self::new(PyannetModel::from_preflight(preflight)?, backend, placement)
    }

    pub(crate) fn new(
        model: PyannetModel,
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Result<Self, PyannetGgmlError> {
        let mut config = GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend);
        config.graph_size = DIRECTION_GRAPH_SIZE;
        config.context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(DIRECTION_GRAPH_SIZE);
        // A GPU-backed instance owns the complete neural graph. Enabling the
        // multi-backend scheduler here would permit an implicit CPU fallback
        // and retain scheduler high-water for an otherwise device-complete
        // runtime.
        let graph_placement = device_graph_placement(backend, placement);
        let config =
            crate::models::graph_runtime_config::apply_execution_placement(config, graph_placement);
        let runner = GgmlCpuGraphRunner::new(config)?;
        let arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(ARENA_TENSORS))?;
        let handles = WeightHandles::allocate(&arena)?;
        let mut arena = arena;
        handles.upload(&mut arena, model.weights())?;
        drop(model);
        Ok(Self {
            sinc_graph: None,
            direction_graphs: std::array::from_fn(|_| None),
            resident: ResidentWeights { handles, arena },
            runner,
        })
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, PyannetGgmlError> {
        Ok(ACCELERATED_HOST_COMMITMENT_BYTES)
    }

    pub(crate) const fn quoted_persistent_host_commitment_bytes() -> u64 {
        ACCELERATED_HOST_COMMITMENT_BYTES
    }

    #[cfg(test)]
    fn prepared_graph_allocation_bytes(&self) -> Option<u64> {
        self.direction_graphs
            .iter()
            .map(|graph| {
                graph
                    .as_ref()
                    .and_then(|graph| graph.session.prepared_allocation_bytes())
            })
            .try_fold(0_u64, |total, bytes| Some(total.saturating_add(bytes?)))
    }

    pub(crate) fn forward(
        &mut self,
        samples: &[f32],
    ) -> Result<(Vec<f32>, usize), PyannetGgmlError> {
        let frames = output_frame_count(samples.len());
        if frames == 0 {
            return Ok((Vec::new(), 0));
        }
        let features = self.run_sincnet(samples, frames)?;
        let output = self.forward_features(&features, frames)?;
        Ok((output, frames))
    }

    fn run_sincnet(
        &mut self,
        samples: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, PyannetGgmlError> {
        if self.sinc_graph.as_ref().is_none_or(|graph| {
            graph.samples != samples.len() || graph.frames != frames || graph.session.is_poisoned()
        }) {
            self.sinc_graph = None;
            self.sinc_graph = Some(self.build_sinc_graph(samples.len(), frames)?);
        }
        let persistent = self
            .sinc_graph
            .as_mut()
            .expect("PyanNet SincNet graph built");
        let graph = persistent.session.builder();
        graph.set_f32_slice(persistent.input, samples, "pyannet_waveform")?;
        let output = graph.compute_output_f32(persistent.output, frames * INPUT_FEATURES)?;
        let expected = frames.saturating_mul(INPUT_FEATURES);
        if output.len() != expected {
            return Err(PyannetGgmlError::InvalidSincOutput {
                got: output.len(),
                expected,
            });
        }
        Ok(output)
    }

    fn build_sinc_graph(
        &mut self,
        samples: usize,
        expected_frames: usize,
    ) -> Result<PersistentSincGraph, GgmlCpuGraphError> {
        let mut session = self
            .runner
            .start_persistent_graph_session_with_node_capacity(SINC_GRAPH_SIZE)?;
        let graph = session.builder();
        let input = graph.new_tensor_2d_f32(samples, 1, "pyannet_waveform")?;
        let sinc = self.resident.handles.sinc;
        let normalized = graph.norm(input, SINC_EPSILON)?;
        let normalized = graph.mul(
            normalized,
            self.resident.arena.graph_tensor(sinc.waveform_norm_weight),
        )?;
        let mut state = graph.add(
            normalized,
            self.resident.arena.graph_tensor(sinc.waveform_norm_bias),
        )?;
        let mut time = samples;
        for (index, handles) in sinc.convs.iter().copied().enumerate() {
            debug_assert_eq!(
                handles.input_channels,
                if index == 0 {
                    1
                } else {
                    sinc.convs[index - 1].output_channels
                }
            );
            let state_4d = graph.reshape_4d(state, time, 1, handles.input_channels, 1)?;
            state = graph.conv_2d_direct(
                self.resident.arena.graph_tensor(handles.weight),
                state_4d,
                handles.stride,
                1,
                0,
                0,
                1,
                1,
            )?;
            time = valid_output_count(time, handles.kernel, handles.stride);
            state = graph.reshape_2d(state, time, handles.output_channels)?;
            if index == 0 {
                state = graph.abs(state)?;
            } else if let Some(bias) = handles.bias {
                state = apply_channel_affine(
                    graph,
                    state,
                    None,
                    Some(self.resident.arena.graph_tensor(bias)),
                )?;
            }
            state = graph.max_pool_1d(state, 3, 3, 0)?;
            time = valid_output_count(time, 3, 3);
            state = apply_channel_instance_norm(
                graph,
                state,
                time,
                handles.output_channels,
                self.resident.arena.graph_tensor(handles.norm_weight),
                self.resident.arena.graph_tensor(handles.norm_bias),
            )?;
            state = graph.leaky_relu(state, ALPHA)?;
        }
        if time != expected_frames {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "PyanNet SincNet graph geometry does not match family contract",
            });
        }
        // ggml conv tensors are `[time, channels]` with time contiguous. The
        // recurrent graph consumes `[frames, channels]` row-major, so make the
        // transposed view contiguous before readback.
        let output = graph.cont(graph.transpose(state)?)?;
        graph.set_input(input)?;
        graph.set_output(output)?;
        graph.prepare_outputs_for_upload(&[output])?;
        Ok(PersistentSincGraph {
            session,
            input,
            output,
            samples,
            frames: expected_frames,
        })
    }

    #[cfg(test)]
    fn run_sincnet_first_block_probe(
        &mut self,
        samples: &[f32],
        stage: usize,
    ) -> Result<Vec<f32>, PyannetGgmlError> {
        let mut graph = self.runner.start_graph();
        let input = graph.new_tensor_2d_f32(samples.len(), 1, "pyannet_probe_input")?;
        let sinc = self.resident.handles.sinc;
        let normalized = graph.norm(input, SINC_EPSILON)?;
        let normalized = graph.mul(
            normalized,
            self.resident.arena.graph_tensor(sinc.waveform_norm_weight),
        )?;
        let waveform = graph.add(
            normalized,
            self.resident.arena.graph_tensor(sinc.waveform_norm_bias),
        )?;
        let conv_handles = sinc.convs[0];
        let waveform_4d = graph.reshape_4d(waveform, samples.len(), 1, 1, 1)?;
        let conv = graph.conv_2d_direct(
            self.resident.arena.graph_tensor(conv_handles.weight),
            waveform_4d,
            conv_handles.stride,
            1,
            0,
            0,
            1,
            1,
        )?;
        let conv_time = valid_output_count(samples.len(), conv_handles.kernel, conv_handles.stride);
        let conv = graph.reshape_2d(conv, conv_time, conv_handles.output_channels)?;
        let absolute = graph.abs(conv)?;
        let pooled = graph.max_pool_1d(absolute, 3, 3, 0)?;
        let pooled_time = valid_output_count(conv_time, 3, 3);
        let normalized = apply_channel_instance_norm(
            &graph,
            pooled,
            pooled_time,
            conv_handles.output_channels,
            self.resident.arena.graph_tensor(conv_handles.norm_weight),
            self.resident.arena.graph_tensor(conv_handles.norm_bias),
        )?;
        let activated = graph.leaky_relu(normalized, ALPHA)?;
        let (output, len) = match stage {
            0 => (waveform, samples.len()),
            1 => (conv, conv_time * conv_handles.output_channels),
            2 => (absolute, conv_time * conv_handles.output_channels),
            3 => (pooled, pooled_time * conv_handles.output_channels),
            4 => (normalized, pooled_time * conv_handles.output_channels),
            5 => (activated, pooled_time * conv_handles.output_channels),
            _ => {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "PyanNet first-block probe stage out of range",
                }
                .into());
            }
        };
        graph.set_input(input)?;
        graph.set_f32_slice(input, samples, "pyannet_probe_input")?;
        Ok(graph.compute_output_f32(output, len)?)
    }

    fn forward_features(
        &mut self,
        features: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, PyannetGgmlError> {
        let expected = frames.saturating_mul(INPUT_FEATURES);
        if features.len() != expected {
            return Err(PyannetGgmlError::InvalidFeaturePayload {
                got: features.len(),
                expected,
            });
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        if frames > MAX_DIRECTION_FRAMES {
            return Err(PyannetGgmlError::TooManyFrames {
                frames,
                maximum: MAX_DIRECTION_FRAMES,
            });
        }
        let mut states = features.to_vec();
        for layer in 0..LSTM_WEIGHTS.len() {
            let group = usize::from(layer != 0);
            let slot_base = if layer == 0 { 0 } else { (layer - 1) * 2 };
            let forward = self.run_lstm_direction(&states, frames, group, slot_base, false)?;
            let backward = self.run_lstm_direction(&states, frames, group, slot_base + 1, true)?;
            states = interleave_bidirectional_states(&forward, &backward, frames);
        }
        self.run_classifier(&states, frames)
    }

    fn run_lstm_direction(
        &mut self,
        states: &[f32],
        frames: usize,
        group: usize,
        slot: usize,
        reverse: bool,
    ) -> Result<Vec<f32>, PyannetGgmlError> {
        let handles = self.resident.handles.lstm_groups[group];
        debug_assert!(slot < handles.slots);
        let expected = frames.saturating_mul(handles.input_features);
        if states.len() != expected {
            return Err(PyannetGgmlError::InvalidFeaturePayload {
                got: states.len(),
                expected,
            });
        }
        if self.direction_graphs[group]
            .as_ref()
            .is_none_or(|graph| graph.frames != frames || graph.session.is_poisoned())
        {
            self.direction_graphs[group] = None;
            self.direction_graphs[group] = Some(self.build_direction_graph(group, frames)?);
        }

        let input = if reverse {
            reverse_frame_major(states, frames, handles.input_features)
        } else {
            states.to_vec()
        };
        let gate = GATE_COUNT * HIDDEN;
        let first_row = slot.saturating_mul(gate);
        let rows = (first_row..first_row + gate)
            .map(|row| i32::try_from(row).expect("PyanNet stacked LSTM row fits i32"))
            .collect::<Vec<_>>();
        let persistent = self.direction_graphs[group]
            .as_mut()
            .expect("PyanNet direction graph built");
        let graph = persistent.session.builder();
        graph.set_f32_slice(persistent.input, &input, "pyannet_recurrent_input")?;
        graph.set_i32_slice(persistent.row_indices, &rows, "pyannet_lstm_weight_rows")?;
        let output = graph.compute_output_f32(persistent.output, frames * HIDDEN)?;
        Ok(if reverse {
            reverse_frame_major(&output, frames, HIDDEN)
        } else {
            output
        })
    }

    fn build_direction_graph(
        &mut self,
        group: usize,
        frames: usize,
    ) -> Result<PersistentDirectionGraph, GgmlCpuGraphError> {
        let handles = self.resident.handles.lstm_groups[group];
        let mut session = self.runner.start_persistent_graph_session(
            GgmlCpuGraphConfig::metadata_context_bytes(DIRECTION_GRAPH_SIZE),
        )?;
        let graph = session.builder();
        let input =
            graph.new_tensor_2d_f32(handles.input_features, frames, "pyannet_recurrent_input")?;
        let row_indices =
            graph.new_tensor_1d_i32(GATE_COUNT * HIDDEN, "pyannet_lstm_weight_rows")?;
        let input_weight = graph.get_rows(
            self.resident.arena.graph_tensor(handles.input_weights),
            row_indices,
        )?;
        let recurrent_weight = graph.get_rows(
            self.resident.arena.graph_tensor(handles.recurrent_weights),
            row_indices,
        )?;
        let combined_bias = graph.reshape_1d(
            graph.get_rows(
                self.resident.arena.graph_tensor(handles.combined_biases),
                row_indices,
            )?,
            GATE_COUNT * HIDDEN,
        )?;
        let zero = self
            .resident
            .arena
            .graph_tensor(self.resident.handles.zero_state);
        let output = build_lstm_direction(
            graph,
            input_weight,
            recurrent_weight,
            combined_bias,
            input,
            zero,
            frames,
            HIDDEN,
        )?;
        graph.set_input(input)?;
        graph.set_input(row_indices)?;
        graph.set_output(output)?;
        graph.prepare_outputs_for_upload(&[output])?;
        Ok(PersistentDirectionGraph {
            session,
            input,
            row_indices,
            output,
            frames,
        })
    }

    fn run_classifier(
        &mut self,
        states: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, PyannetGgmlError> {
        let mut graph = self.runner.start_graph();
        let input = graph.new_tensor_2d_f32(2 * HIDDEN, frames, "pyannet_classifier_input")?;
        let mut output = input;
        for (index, handles) in self.resident.handles.classifier.iter().copied().enumerate() {
            output = linear(&graph, &self.resident.arena, handles, output)?;
            if index + 1 != self.resident.handles.classifier.len() {
                output = graph.leaky_relu(output, ALPHA)?;
            }
        }
        output = graph.log(graph.soft_max(output)?)?;
        graph.set_input(input)?;
        graph.set_f32_slice(input, states, "pyannet_classifier_input")?;
        Ok(graph.compute_output_f32(output, frames * NUM_CLASSES)?)
    }
}

const fn valid_output_count(input: usize, kernel: usize, stride: usize) -> usize {
    if input < kernel {
        0
    } else {
        (input - kernel) / stride + 1
    }
}

fn apply_channel_affine<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    scale: Option<GgmlCpuTensor<'a>>,
    bias: Option<GgmlCpuTensor<'a>>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let mut feature_major = graph.cont(graph.transpose(input)?)?;
    if let Some(scale) = scale {
        feature_major = graph.mul(feature_major, scale)?;
    }
    if let Some(bias) = bias {
        feature_major = graph.add(feature_major, bias)?;
    }
    graph.cont(graph.transpose(feature_major)?)
}

fn apply_channel_instance_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    _time: usize,
    _channels: usize,
    scale: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    // Conv output is `[time, channels]`, so each channel is one ggml row and
    // `norm` applies the exact per-channel InstanceNorm reduction over time.
    // This also avoids backend-specific GroupNorm channel partition kernels.
    let normalized = graph.norm(input, SINC_EPSILON)?;
    apply_channel_affine(graph, normalized, Some(scale), Some(bias))
}

fn device_graph_placement(
    backend: GgmlCpuGraphBackend,
    stage_placement: ExecutionPlacement,
) -> ExecutionPlacement {
    if backend.is_gpu_class() {
        ExecutionPlacement::FullDevice
    } else {
        stage_placement
    }
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    handles: LinearHandles,
    input: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    graph.add(
        graph.mul_mat(arena.graph_tensor(handles.weight), input)?,
        arena.graph_tensor(handles.bias),
    )
}

fn build_lstm_direction<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input_weight: GgmlCpuTensor<'a>,
    recurrent_weight: GgmlCpuTensor<'a>,
    combined_bias: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    zero: GgmlCpuTensor<'a>,
    frames: usize,
    hidden: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let gate = GATE_COUNT * hidden;
    let projected = graph.mul_mat(input_weight, input)?;
    let mut h = zero;
    let mut c = zero;
    let mut chronological = vec![zero; frames];
    for (frame, state) in chronological.iter_mut().enumerate() {
        let input_gates = graph.view_1d(projected, gate, frame * gate * F32_BYTES)?;
        let recurrent_gates = graph.mul_mat(recurrent_weight, h)?;
        let gates = graph.add(graph.add(input_gates, recurrent_gates)?, combined_bias)?;
        let input_gate = graph.sigmoid(graph.view_1d(gates, hidden, 0)?)?;
        let output_gate = graph.sigmoid(graph.view_1d(gates, hidden, hidden * F32_BYTES)?)?;
        let forget_gate = graph.sigmoid(graph.view_1d(gates, hidden, 2 * hidden * F32_BYTES)?)?;
        let cell_gate = graph.tanh(graph.view_1d(gates, hidden, 3 * hidden * F32_BYTES)?)?;
        c = graph.add(
            graph.mul(forget_gate, c)?,
            graph.mul(input_gate, cell_gate)?,
        )?;
        h = graph.mul(output_gate, graph.tanh(c)?)?;
        *state = h;
    }
    balanced_time_concat(graph, chronological, hidden)
}

fn reverse_frame_major(values: &[f32], frames: usize, width: usize) -> Vec<f32> {
    let mut reversed = vec![0.0f32; values.len()];
    for frame in 0..frames {
        let source = frame * width;
        let target = (frames - 1 - frame) * width;
        reversed[target..target + width].copy_from_slice(&values[source..source + width]);
    }
    reversed
}

fn interleave_bidirectional_states(forward: &[f32], backward: &[f32], frames: usize) -> Vec<f32> {
    debug_assert_eq!(forward.len(), frames * HIDDEN);
    debug_assert_eq!(backward.len(), frames * HIDDEN);
    let mut combined = vec![0.0f32; frames * 2 * HIDDEN];
    for frame in 0..frames {
        let source = frame * HIDDEN;
        let target = frame * 2 * HIDDEN;
        combined[target..target + HIDDEN].copy_from_slice(&forward[source..source + HIDDEN]);
        combined[target + HIDDEN..target + 2 * HIDDEN]
            .copy_from_slice(&backward[source..source + HIDDEN]);
    }
    combined
}

/// Assemble chronological states with logarithmic concat depth. A left-fold
/// would repeatedly copy a growing prefix and turn output assembly quadratic.
fn balanced_time_concat<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    states: Vec<GgmlCpuTensor<'a>>,
    hidden: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let mut level = states
        .into_iter()
        .map(|state| graph.reshape_2d(state, hidden, 1))
        .collect::<Result<Vec<_>, _>>()?;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut pairs = level.chunks_exact(2);
        for pair in &mut pairs {
            next.push(graph.concat(pair[0], pair[1], 1)?);
        }
        if let [tail] = pairs.remainder() {
            next.push(*tail);
        }
        level = next;
    }
    level.pop().ok_or(GgmlCpuGraphError::UnsupportedInputs {
        reason: "PyanNet LSTM sequence must contain at least one frame",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_linear_weight_transpose_preserves_matmul_contract() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_onnx_matmul_weight(&values, 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn legacy_hybrid_input_is_normalized_to_the_complete_device_graph() {
        assert_eq!(
            device_graph_placement(GgmlCpuGraphBackend::Metal, ExecutionPlacement::Hybrid,),
            ExecutionPlacement::FullDevice,
        );
        assert_eq!(
            device_graph_placement(GgmlCpuGraphBackend::Cpu, ExecutionPlacement::CpuOnly,),
            ExecutionPlacement::CpuOnly,
        );
    }

    #[test]
    #[ignore = "requires OPENASR_PYANNOTE_F32_PACK and a representative Metal device"]
    fn recurrent_graph_matches_family_math_on_cpu_and_metal() {
        let pack = std::env::var_os("OPENASR_PYANNOTE_F32_PACK")
            .map(std::path::PathBuf::from)
            .expect("OPENASR_PYANNOTE_F32_PACK");
        let samples: Vec<f32> = (0..160_000)
            .map(|index| {
                let time = index as f32 / 16_000.0;
                0.13 * (time * 311.0 * std::f32::consts::TAU).sin()
                    + 0.07 * (time * 877.0 * std::f32::consts::TAU).cos()
            })
            .collect();
        let reference_model = PyannetModel::from_oasr(&pack).expect("reference model");
        let (reference, frames) = reference_model.forward(&samples).expect("family forward");
        let (reference_sinc, _, stage_frames) =
            reference_model.stages(&samples).expect("family stages");
        assert_eq!(stage_frames, frames);
        let reference_sinc =
            crate::diarize::segment::pyannet::transpose(&reference_sinc, INPUT_FEATURES, frames);
        assert_eq!(frames, 589);
        let mut cpu_first_block_probes: Option<Vec<Vec<f32>>> = None;
        for (backend, sinc_tolerance, tolerance) in [
            (GgmlCpuGraphBackend::Cpu, 1.0e-3f32, 5.0e-4f32),
            (GgmlCpuGraphBackend::Metal, 5.0e-3f32, 1.0e-2f32),
        ] {
            let model = PyannetModel::from_oasr(&pack).expect("runtime model");
            let placement = if backend == GgmlCpuGraphBackend::Cpu {
                ExecutionPlacement::CpuOnly
            } else {
                ExecutionPlacement::Hybrid
            };
            let mut runtime =
                PyannetGgmlRuntime::new(model, backend, placement).expect("ggml runtime");
            if backend == GgmlCpuGraphBackend::Metal {
                assert!(!runtime.runner.uses_scheduler());
            }
            let probes = [1usize, 4]
                .into_iter()
                .map(|stage| {
                    runtime
                        .run_sincnet_first_block_probe(&samples, stage)
                        .expect("first-block probe")
                })
                .collect::<Vec<_>>();
            if backend == GgmlCpuGraphBackend::Cpu {
                cpu_first_block_probes = Some(probes);
            } else {
                for ((stage, tolerance), (actual, expected)) in
                    [(1usize, 2.0e-6), (4, 1.0e-3)].into_iter().zip(
                        probes.iter().zip(
                            cpu_first_block_probes
                                .as_ref()
                                .expect("CPU first-block probes"),
                        ),
                    )
                {
                    let max_abs = actual
                        .iter()
                        .zip(expected)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_abs < tolerance,
                        "PyanNet SincNet stage {stage} Metal/CPU max abs {max_abs} exceeds {tolerance}"
                    );
                }
            }
            let sinc = runtime
                .run_sincnet(&samples, frames)
                .expect("ggml SincNet forward");
            let sinc_max_abs = sinc
                .iter()
                .zip(&reference_sinc)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            eprintln!("PYANNET_SINC backend={backend:?} max_abs={sinc_max_abs:.9}");
            assert!(
                sinc_max_abs < sinc_tolerance,
                "{backend:?} PyanNet SincNet max abs {sinc_max_abs} exceeds {sinc_tolerance}"
            );
            let actual = runtime
                .forward_features(&sinc, frames)
                .expect("ggml recurrent forward");
            let actual_frames = frames;
            if backend == GgmlCpuGraphBackend::Metal {
                let prepared_bytes = runtime
                    .prepared_graph_allocation_bytes()
                    .expect("direct Metal graph allocation");
                eprintln!("PYANNET_METAL_GRAPH prepared_allocation_bytes={prepared_bytes}",);
                assert!(
                    prepared_bytes <= 5 * 1024 * 1024,
                    "PyanNet reusable direction graph allocations regressed to {prepared_bytes} bytes"
                );
            }
            assert_eq!(actual_frames, frames);
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < tolerance,
                "{backend:?} PyanNet full graph max abs {max_abs} exceeds {tolerance}"
            );
        }
    }
}
