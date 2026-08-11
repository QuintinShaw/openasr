//! Hybrid ggml execution for PyanNet segmentation-3.0.
//!
//! SincNet stays on the host because its small valid convolutions, pooling and
//! instance normalization are already a compact family-owned implementation.
//! The four bidirectional LSTMs and classifier are the compute-dominant block;
//! this module uploads those weights once and owns a persistent backend graph.
//! An explicit Metal request therefore reaches real Metal kernels without
//! changing PyanNet's window geometry or ONNX gate semantics.

use thiserror::Error;

use super::pyannet::{ALPHA, HIDDEN, LSTM_WEIGHTS, NUM_CLASSES, PyannetModel, transpose};
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
const GRAPH_SIZE: usize = 1 << 17;
const ARENA_TENSORS: usize = 1 << 8;
const F32_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Debug, Error)]
pub(crate) enum PyannetGgmlError {
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Graph(#[from] GgmlCpuGraphError),
    #[error("PyanNet recurrent input has {got} values, expected {expected}")]
    InvalidFeaturePayload { got: usize, expected: usize },
}

#[derive(Clone, Copy)]
struct LstmDirectionHandles {
    input_weight: GgmlStaticTensor,
    recurrent_weight: GgmlStaticTensor,
    combined_bias: GgmlStaticTensor,
}

struct LstmLayerHandles {
    directions: [LstmDirectionHandles; 2],
}

#[derive(Clone, Copy)]
struct LinearHandles {
    weight: GgmlStaticTensor,
    bias: GgmlStaticTensor,
}

struct WeightHandles {
    lstm: Vec<LstmLayerHandles>,
    classifier: [LinearHandles; 3],
    zero_state: GgmlStaticTensor,
}

impl WeightHandles {
    fn allocate(arena: &GgmlStaticTensorArena) -> Result<Self, GgmlCpuGraphError> {
        let mut lstm = Vec::with_capacity(LSTM_WEIGHTS.len());
        for layer in 0..LSTM_WEIGHTS.len() {
            let input = if layer == 0 {
                INPUT_FEATURES
            } else {
                2 * HIDDEN
            };
            lstm.push(LstmLayerHandles {
                directions: [
                    allocate_lstm_direction(arena, input, HIDDEN)?,
                    allocate_lstm_direction(arena, input, HIDDEN)?,
                ],
            });
        }
        Ok(Self {
            lstm,
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
        for (layer, ((w_name, r_name, b_name), handles)) in
            LSTM_WEIGHTS.into_iter().zip(&self.lstm).enumerate()
        {
            let input = if layer == 0 {
                INPUT_FEATURES
            } else {
                2 * HIDDEN
            };
            let gate = GATE_COUNT * HIDDEN;
            let w = weights.get(w_name)?;
            let r = weights.get(r_name)?;
            let bias = weights.get(b_name)?;
            for (direction, direction_handles) in handles.directions.iter().enumerate() {
                let w_start = direction * gate * input;
                let r_start = direction * gate * HIDDEN;
                let b_start = direction * 2 * gate;
                arena.set_f32_slice(
                    direction_handles.input_weight,
                    &w[w_start..w_start + gate * input],
                    "pyannet_lstm_input_weight",
                )?;
                arena.set_f32_slice(
                    direction_handles.recurrent_weight,
                    &r[r_start..r_start + gate * HIDDEN],
                    "pyannet_lstm_recurrent_weight",
                )?;
                let mut combined_bias = vec![0.0f32; gate];
                for index in 0..gate {
                    combined_bias[index] = bias[b_start + index] + bias[b_start + gate + index];
                }
                arena.set_f32_slice(
                    direction_handles.combined_bias,
                    &combined_bias,
                    "pyannet_lstm_combined_bias",
                )?;
            }
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

fn allocate_lstm_direction(
    arena: &GgmlStaticTensorArena,
    input: usize,
    hidden: usize,
) -> Result<LstmDirectionHandles, GgmlCpuGraphError> {
    let gate = GATE_COUNT * hidden;
    Ok(LstmDirectionHandles {
        input_weight: arena.new_tensor_2d_f32(input, gate, "pyannet_lstm_input_weight")?,
        recurrent_weight: arena.new_tensor_2d_f32(hidden, gate, "pyannet_lstm_recurrent_weight")?,
        combined_bias: arena.new_tensor_1d_f32(gate, "pyannet_lstm_combined_bias")?,
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

struct PersistentGraph {
    session: GgmlPersistentGraphSession,
    input: GgmlCpuTensor<'static>,
    output: GgmlCpuTensor<'static>,
    frames: usize,
}

/// Thread-confined hybrid PyanNet runtime.
///
/// Field order is load-bearing: graph tensors drop before their static arena,
/// and the arena drops before the backend runner that allocated it.
pub(crate) struct PyannetGgmlRuntime {
    graph: Option<PersistentGraph>,
    resident: ResidentWeights,
    model: PyannetModel,
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
        config.graph_size = GRAPH_SIZE;
        config.context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE);
        let config =
            crate::models::graph_runtime_config::apply_execution_placement(config, placement);
        let runner = GgmlCpuGraphRunner::new(config)?;
        let arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(ARENA_TENSORS))?;
        let handles = WeightHandles::allocate(&arena)?;
        let mut arena = arena;
        handles.upload(&mut arena, model.weights())?;
        Ok(Self {
            graph: None,
            resident: ResidentWeights { handles, arena },
            model,
            runner,
        })
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, PyannetGgmlError> {
        Ok(self.model.persistent_host_commitment_bytes()?)
    }

    pub(crate) fn forward(
        &mut self,
        samples: &[f32],
    ) -> Result<(Vec<f32>, usize), PyannetGgmlError> {
        let (channel_major, frames) = self.model.sincnet_features(samples)?;
        if frames == 0 {
            return Ok((Vec::new(), 0));
        }
        let features = transpose(&channel_major, INPUT_FEATURES, frames);
        let output = self.forward_features(&features, frames)?;
        Ok((output, frames))
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
        if self
            .graph
            .as_ref()
            .is_none_or(|graph| graph.frames != frames || graph.session.is_poisoned())
        {
            self.graph = None;
            self.graph = Some(self.build_graph(frames)?);
        }
        let persistent = self.graph.as_mut().expect("PyanNet graph built");
        let graph = persistent.session.builder();
        graph.set_f32_slice(persistent.input, features, "pyannet_recurrent_input")?;
        Ok(graph.compute_output_f32(persistent.output, frames * NUM_CLASSES)?)
    }

    fn build_graph(&mut self, frames: usize) -> Result<PersistentGraph, GgmlCpuGraphError> {
        let mut session = self.runner.start_persistent_graph_session(
            GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE),
        )?;
        let graph = session.builder();
        let input = graph.new_tensor_2d_f32(INPUT_FEATURES, frames, "pyannet_recurrent_input")?;
        let zero = self
            .resident
            .arena
            .graph_tensor(self.resident.handles.zero_state);
        let mut states = input;
        for layer in &self.resident.handles.lstm {
            states = build_bidirectional_lstm(
                graph,
                &self.resident.arena,
                layer,
                states,
                zero,
                frames,
                HIDDEN,
            )?;
        }
        states = graph.leaky_relu(
            linear(
                graph,
                &self.resident.arena,
                self.resident.handles.classifier[0],
                states,
            )?,
            ALPHA,
        )?;
        states = graph.leaky_relu(
            linear(
                graph,
                &self.resident.arena,
                self.resident.handles.classifier[1],
                states,
            )?,
            ALPHA,
        )?;
        let logits = linear(
            graph,
            &self.resident.arena,
            self.resident.handles.classifier[2],
            states,
        )?;
        let output = graph.log(graph.soft_max(logits)?)?;
        graph.set_input(input)?;
        graph.set_output(output)?;
        graph.prepare_outputs_for_upload(&[output])?;
        Ok(PersistentGraph {
            session,
            input,
            output,
            frames,
        })
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

fn build_bidirectional_lstm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    handles: &LstmLayerHandles,
    input: GgmlCpuTensor<'a>,
    zero: GgmlCpuTensor<'a>,
    frames: usize,
    hidden: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let forward = build_lstm_direction(
        graph,
        arena,
        handles.directions[0],
        input,
        zero,
        frames,
        hidden,
        false,
    )?;
    let backward = build_lstm_direction(
        graph,
        arena,
        handles.directions[1],
        input,
        zero,
        frames,
        hidden,
        true,
    )?;
    graph.concat(forward, backward, 0)
}

#[allow(clippy::too_many_arguments)]
fn build_lstm_direction<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    handles: LstmDirectionHandles,
    input: GgmlCpuTensor<'a>,
    zero: GgmlCpuTensor<'a>,
    frames: usize,
    hidden: usize,
    reverse: bool,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    let gate = GATE_COUNT * hidden;
    let projected = graph.mul_mat(arena.graph_tensor(handles.input_weight), input)?;
    let mut h = zero;
    let mut c = zero;
    let mut chronological = vec![zero; frames];
    for step in 0..frames {
        let frame = if reverse { frames - 1 - step } else { step };
        let input_gates = graph.view_1d(projected, gate, frame * gate * F32_BYTES)?;
        let recurrent_gates = graph.mul_mat(arena.graph_tensor(handles.recurrent_weight), h)?;
        let gates = graph.add(
            graph.add(input_gates, recurrent_gates)?,
            arena.graph_tensor(handles.combined_bias),
        )?;
        let input_gate = graph.sigmoid(graph.view_1d(gates, hidden, 0)?)?;
        let output_gate = graph.sigmoid(graph.view_1d(gates, hidden, hidden * F32_BYTES)?)?;
        let forget_gate = graph.sigmoid(graph.view_1d(gates, hidden, 2 * hidden * F32_BYTES)?)?;
        let cell_gate = graph.tanh(graph.view_1d(gates, hidden, 3 * hidden * F32_BYTES)?)?;
        c = graph.add(
            graph.mul(forget_gate, c)?,
            graph.mul(input_gate, cell_gate)?,
        )?;
        h = graph.mul(output_gate, graph.tanh(c)?)?;
        chronological[frame] = h;
    }
    balanced_time_concat(graph, chronological, hidden)
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
        assert_eq!(frames, 589);
        for (backend, tolerance) in [
            (GgmlCpuGraphBackend::Cpu, 2.0e-4f32),
            (GgmlCpuGraphBackend::Metal, 1.0e-2f32),
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
                assert!(runtime.runner.uses_scheduler());
            }
            let (actual, actual_frames) = runtime.forward(&samples).expect("ggml forward");
            assert_eq!(actual_frames, frames);
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < tolerance,
                "{backend:?} PyanNet recurrent max abs {max_abs} exceeds {tolerance}"
            );
        }
    }
}
