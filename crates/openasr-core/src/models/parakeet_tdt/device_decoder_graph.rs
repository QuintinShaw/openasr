//! Device-resident Parakeet-TDT prediction network and joint head.
//!
//! The FastConformer encoder, predictor LSTM, and joint head share one runner
//! and one mmap-backed loaded-weight context. Only recurrent state, one encoder
//! frame, and logits cross the device boundary; predictor/joint weights never
//! acquire a second host-f32 representation on accelerated routes.

use std::mem::size_of;

use crate::ggml_runtime::{
    GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedWeightContext, GgmlPersistentGraphSession, GgmlStaticTensor, GgmlStaticTensorArena,
};

use super::greedy::ParakeetTdtDecodeBackend;
use super::runtime_contract::ParakeetTdtExecutionMetadata;

struct PredictorStepGraph {
    session: GgmlPersistentGraphSession,
    token: GgmlCpuTensor<'static>,
    h_inputs: Vec<GgmlCpuTensor<'static>>,
    c_inputs: Vec<GgmlCpuTensor<'static>>,
    pred_proj_output: Option<GgmlCpuTensor<'static>>,
    h_outputs: Vec<GgmlCpuTensor<'static>>,
    c_outputs: Vec<GgmlCpuTensor<'static>>,
}

struct JointStepGraph {
    session: GgmlPersistentGraphSession,
    encoder_frame: GgmlCpuTensor<'static>,
    pred_proj: Option<GgmlCpuTensor<'static>>,
    logits: GgmlCpuTensor<'static>,
}

enum PredictorState {
    Host {
        h: Vec<Vec<f32>>,
        c: Vec<Vec<f32>>,
        pred_proj: Vec<f32>,
    },
    Resident {
        arena: Box<GgmlStaticTensorArena>,
        h: Vec<GgmlStaticTensor>,
        c: Vec<GgmlStaticTensor>,
        reset_zeros: Vec<f32>,
        state_width: usize,
    },
}

/// Stateful accelerated decoder. Field order is intentional: both graph
/// sessions must drop before the encoder core frees the shared runner, loaded
/// weight context, or static arena that owns their raw tensor dependencies.
pub(crate) struct ParakeetTdtDeviceDecoder {
    predictor: PredictorStepGraph,
    joint: JointStepGraph,
    state: PredictorState,
    logits: Vec<f32>,
    joint_hidden: usize,
}

fn loaded_tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'static>, GgmlCpuGraphError> {
    loaded
        .tensor(name)
        .map(|tensor| tensor.as_graph_tensor())
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "parakeet-tdt device decoder is missing a verified loaded tensor",
        })
}

fn checked_payload_bytes(
    metadata: ParakeetTdtExecutionMetadata,
    resident_predictor_state: bool,
) -> Result<u64, String> {
    // SystemMemory quotes engine-requested Rust heap capacity, not inline
    // struct storage. Session/context/backend allocations carry their own
    // shared-layer leases; only Vec backing owned by this decoder is counted
    // here, matching `retained_system_memory_bytes` after construction.
    let pred = metadata.pred_hidden as u64;
    let layers = metadata.pred_layers as u64;
    let joint = metadata.joint_hidden as u64;
    let out = metadata
        .vocab_size
        .checked_add(metadata.n_durations)
        .ok_or_else(|| "parakeet-tdt device decoder output width overflowed".to_string())?
        as u64;
    if resident_predictor_state {
        // Only the logits row and one reset page remain in host memory. The
        // mutable h/c/projection payload is admitted by the backend-memory
        // broker through the state arena and must not be double-counted as
        // SystemMemory.
        let state_values = out
            .checked_add(pred.max(joint))
            .ok_or_else(|| "parakeet-tdt resident host state size overflowed".to_string())?;
        let state_bytes = state_values
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| "parakeet-tdt resident host state bytes overflowed".to_string())?;
        let descriptor_bytes = layers
            .checked_mul((2 * size_of::<GgmlStaticTensor>()) as u64)
            .ok_or_else(|| "parakeet-tdt resident state descriptor bytes overflowed".to_string())?;
        state_bytes.checked_add(descriptor_bytes).ok_or_else(|| {
            "parakeet-tdt resident device decoder retained bytes overflowed".to_string()
        })
    } else {
        let state_values = pred
            .checked_mul(layers)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(joint))
            .and_then(|value| value.checked_add(out))
            .ok_or_else(|| "parakeet-tdt device decoder state size overflowed".to_string())?;
        let state_bytes = state_values
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| "parakeet-tdt device decoder state bytes overflowed".to_string())?;
        // h/c outer Vec descriptors plus four per-layer graph-handle vectors.
        let descriptor_bytes = layers
            .checked_mul(
                (2 * size_of::<Vec<f32>>() + 4 * size_of::<GgmlCpuTensor<'static>>()) as u64,
            )
            .ok_or_else(|| "parakeet-tdt device decoder descriptor bytes overflowed".to_string())?;
        state_bytes
            .checked_add(descriptor_bytes)
            .ok_or_else(|| "parakeet-tdt device decoder retained bytes overflowed".to_string())
    }
}

pub(crate) fn planned_retained_system_memory_bytes(
    metadata: ParakeetTdtExecutionMetadata,
    resident_predictor_state: bool,
) -> Result<u64, String> {
    checked_payload_bytes(metadata, resident_predictor_state)
}

impl ParakeetTdtDeviceDecoder {
    pub(crate) fn new(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        metadata: ParakeetTdtExecutionMetadata,
        resident_predictor_state: bool,
    ) -> Result<Self, GgmlCpuGraphError> {
        let (predictor, joint, state) = if resident_predictor_state {
            let tensor_count = metadata
                .pred_layers
                .checked_mul(2)
                .and_then(|count| count.checked_add(1))
                .ok_or(GgmlCpuGraphError::InvalidGraphSize)?;
            let mut arena = runner.start_state_tensor_arena(
                GgmlCpuGraphConfig::metadata_context_bytes(tensor_count),
            )?;
            let mut h = Vec::with_capacity(metadata.pred_layers);
            let mut c = Vec::with_capacity(metadata.pred_layers);
            for _ in 0..metadata.pred_layers {
                h.push(arena.new_tensor_1d_f32(
                    metadata.pred_hidden,
                    "parakeet_tdt_resident_predictor_h",
                )?);
                c.push(arena.new_tensor_1d_f32(
                    metadata.pred_hidden,
                    "parakeet_tdt_resident_predictor_c",
                )?);
            }
            let pred_proj = arena.new_tensor_1d_f32(
                metadata.joint_hidden,
                "parakeet_tdt_resident_predictor_projection",
            )?;
            arena.allocate_backend_buffer()?;
            let reset_zeros = vec![0.0; metadata.pred_hidden.max(metadata.joint_hidden)];
            for tensor in h.iter().chain(&c) {
                arena.set_f32_slice(
                    *tensor,
                    &reset_zeros[..metadata.pred_hidden],
                    "parakeet_tdt_resident_predictor_state",
                )?;
            }
            arena.set_f32_slice(
                pred_proj,
                &reset_zeros[..metadata.joint_hidden],
                "parakeet_tdt_resident_predictor_projection",
            )?;
            let predictor = Self::build_resident_predictor(
                runner, loaded, &arena, &h, &c, pred_proj, metadata,
            )?;
            let joint = Self::build_resident_joint(runner, loaded, &arena, pred_proj, metadata)?;
            (
                predictor,
                joint,
                PredictorState::Resident {
                    arena: Box::new(arena),
                    h,
                    c,
                    reset_zeros,
                    state_width: metadata.pred_hidden,
                },
            )
        } else {
            (
                Self::build_host_predictor(runner, loaded, metadata)?,
                Self::build_host_joint(runner, loaded, metadata)?,
                PredictorState::Host {
                    h: vec![vec![0.0; metadata.pred_hidden]; metadata.pred_layers],
                    c: vec![vec![0.0; metadata.pred_hidden]; metadata.pred_layers],
                    pred_proj: vec![0.0; metadata.joint_hidden],
                },
            )
        };
        let out_rows = metadata
            .vocab_size
            .checked_add(metadata.n_durations)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "parakeet-tdt device decoder output width overflowed",
            })?;
        Ok(Self {
            predictor,
            joint,
            state,
            logits: vec![0.0; out_rows],
            joint_hidden: metadata.joint_hidden,
        })
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let logits_bytes = (self.logits.capacity() * size_of::<f32>()) as u64;
        let state_bytes = match &self.state {
            PredictorState::Host { h, c, pred_proj } => {
                let payload = h.iter().chain(c).try_fold(0_u64, |total, values| {
                    total
                        .checked_add((values.capacity() * size_of::<f32>()) as u64)
                        .ok_or_else(|| "parakeet-tdt device state bytes overflowed".to_string())
                })?;
                let outer = ((h.capacity() + c.capacity()) * size_of::<Vec<f32>>()) as u64;
                payload
                    .checked_add((pred_proj.capacity() * size_of::<f32>()) as u64)
                    .and_then(|value| value.checked_add(outer))
                    .ok_or_else(|| {
                        "parakeet-tdt host predictor retained bytes overflowed".to_string()
                    })?
            }
            PredictorState::Resident {
                h, c, reset_zeros, ..
            } => {
                let handles = h.capacity().checked_add(c.capacity()).ok_or_else(|| {
                    "parakeet-tdt resident state handle count overflowed".to_string()
                })?;
                ((handles * size_of::<GgmlStaticTensor>())
                    + reset_zeros.capacity() * size_of::<f32>()) as u64
            }
        };
        let graph_handle_count = self
            .predictor
            .h_inputs
            .capacity()
            .checked_add(self.predictor.c_inputs.capacity())
            .and_then(|value| value.checked_add(self.predictor.h_outputs.capacity()))
            .and_then(|value| value.checked_add(self.predictor.c_outputs.capacity()))
            .ok_or_else(|| "parakeet-tdt device handle count overflowed".to_string())?;
        let graph_handle_bytes = (graph_handle_count * size_of::<GgmlCpuTensor<'static>>()) as u64;
        state_bytes
            .checked_add(logits_bytes)
            .and_then(|value| value.checked_add(graph_handle_bytes))
            .ok_or_else(|| "parakeet-tdt device retained bytes overflowed".to_string())
    }

    fn build_host_predictor(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        metadata: ParakeetTdtExecutionMetadata,
    ) -> Result<PredictorStepGraph, GgmlCpuGraphError> {
        let hidden = metadata.pred_hidden;
        hidden
            .checked_mul(4)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "parakeet-tdt predictor gate width overflowed",
            })?;
        // The graph uses fewer than 24 nodes per recurrent layer today. Keep
        // an explicit margin for ggml view/materialization nodes while avoiding
        // the FastConformer encoder's 8k-node metadata budget for this tiny
        // token-step graph.
        let graph_size = metadata
            .pred_layers
            .checked_mul(32)
            .and_then(|nodes| nodes.checked_add(32))
            .ok_or(GgmlCpuGraphError::InvalidGraphSize)?;
        let mut session = runner.start_persistent_graph_session_with_node_capacity(graph_size)?;
        let graph = session.builder();
        let token = graph.new_tensor_1d_i32(1, "parakeet_tdt_predictor_token")?;
        graph.set_input(token)?;
        let mut input = graph.get_rows(loaded_tensor(loaded, "dec.embed.weight")?, token)?;
        let mut h_inputs = Vec::with_capacity(metadata.pred_layers);
        let mut c_inputs = Vec::with_capacity(metadata.pred_layers);
        let mut h_outputs = Vec::with_capacity(metadata.pred_layers);
        let mut c_outputs = Vec::with_capacity(metadata.pred_layers);
        for layer in 0..metadata.pred_layers {
            let h = graph.new_tensor_1d_f32(hidden, "parakeet_tdt_predictor_h")?;
            let c = graph.new_tensor_1d_f32(hidden, "parakeet_tdt_predictor_c")?;
            graph.set_input(h)?;
            graph.set_input(c)?;
            h_inputs.push(h);
            c_inputs.push(c);
            let prefix = format!("dec.lstm.{layer}");
            let mut packed =
                graph.mul_mat(loaded_tensor(loaded, &format!("{prefix}.w_ih"))?, input)?;
            packed = graph.add(packed, loaded_tensor(loaded, &format!("{prefix}.b_ih"))?)?;
            let recurrent = graph.mul_mat(loaded_tensor(loaded, &format!("{prefix}.w_hh"))?, h)?;
            packed = graph.add(packed, recurrent)?;
            packed = graph.add(packed, loaded_tensor(loaded, &format!("{prefix}.b_hh"))?)?;
            let bytes = size_of::<f32>();
            let input_gate = graph.sigmoid(graph.view_1d(packed, hidden, 0)?)?;
            let forget_gate = graph.sigmoid(graph.view_1d(packed, hidden, hidden * bytes)?)?;
            let cell_gate = graph.tanh(graph.view_1d(packed, hidden, 2 * hidden * bytes)?)?;
            let output_gate =
                graph.sigmoid(graph.view_1d(packed, hidden, 3 * hidden * bytes)?)?;
            let new_c = graph.add(
                graph.mul(forget_gate, c)?,
                graph.mul(input_gate, cell_gate)?,
            )?;
            let new_h = graph.mul(output_gate, graph.tanh(new_c)?)?;
            graph.set_output(new_h)?;
            graph.set_output(new_c)?;
            h_outputs.push(new_h);
            c_outputs.push(new_c);
            input = new_h;
        }
        let mut pred_proj = graph.mul_mat(loaded_tensor(loaded, "joint.pred.weight")?, input)?;
        pred_proj = graph.add(pred_proj, loaded_tensor(loaded, "joint.pred.bias")?)?;
        graph.set_output(pred_proj)?;
        let mut outputs = Vec::with_capacity(1 + 2 * metadata.pred_layers);
        outputs.push(pred_proj);
        outputs.extend(h_outputs.iter().copied());
        outputs.extend(c_outputs.iter().copied());
        graph.prepare_outputs_for_upload(&outputs)?;
        Ok(PredictorStepGraph {
            session,
            token,
            h_inputs,
            c_inputs,
            pred_proj_output: Some(pred_proj),
            h_outputs,
            c_outputs,
        })
    }

    fn build_resident_predictor(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        state_arena: &GgmlStaticTensorArena,
        h_state: &[GgmlStaticTensor],
        c_state: &[GgmlStaticTensor],
        pred_proj_state: GgmlStaticTensor,
        metadata: ParakeetTdtExecutionMetadata,
    ) -> Result<PredictorStepGraph, GgmlCpuGraphError> {
        if h_state.len() != metadata.pred_layers || c_state.len() != metadata.pred_layers {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "parakeet-tdt resident predictor state layer count mismatch",
            });
        }
        let hidden = metadata.pred_hidden;
        hidden
            .checked_mul(4)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "parakeet-tdt predictor gate width overflowed",
            })?;
        let graph_size = metadata
            .pred_layers
            .checked_mul(40)
            .and_then(|nodes| nodes.checked_add(32))
            .ok_or(GgmlCpuGraphError::InvalidGraphSize)?;
        let mut session = runner.start_persistent_graph_session_with_node_capacity(graph_size)?;
        let graph = session.builder();
        graph.reserve_side_effect_roots(
            metadata
                .pred_layers
                .checked_mul(2)
                .and_then(|roots| roots.checked_add(1))
                .ok_or(GgmlCpuGraphError::InvalidGraphSize)?,
        )?;
        let token = graph.new_tensor_1d_i32(1, "parakeet_tdt_predictor_token")?;
        graph.set_input(token)?;
        let mut input = graph.get_rows(loaded_tensor(loaded, "dec.embed.weight")?, token)?;
        for layer in 0..metadata.pred_layers {
            let h = state_arena.graph_tensor(h_state[layer]);
            let c = state_arena.graph_tensor(c_state[layer]);
            let prefix = format!("dec.lstm.{layer}");
            let mut packed =
                graph.mul_mat(loaded_tensor(loaded, &format!("{prefix}.w_ih"))?, input)?;
            packed = graph.add(packed, loaded_tensor(loaded, &format!("{prefix}.b_ih"))?)?;
            let recurrent = graph.mul_mat(loaded_tensor(loaded, &format!("{prefix}.w_hh"))?, h)?;
            packed = graph.add(packed, recurrent)?;
            packed = graph.add(packed, loaded_tensor(loaded, &format!("{prefix}.b_hh"))?)?;
            let bytes = size_of::<f32>();
            let input_gate = graph.sigmoid(graph.view_1d(packed, hidden, 0)?)?;
            let forget_gate = graph.sigmoid(graph.view_1d(packed, hidden, hidden * bytes)?)?;
            let cell_gate = graph.tanh(graph.view_1d(packed, hidden, 2 * hidden * bytes)?)?;
            let output_gate =
                graph.sigmoid(graph.view_1d(packed, hidden, 3 * hidden * bytes)?)?;
            let new_c = graph.add(
                graph.mul(forget_gate, c)?,
                graph.mul(input_gate, cell_gate)?,
            )?;
            let new_h = graph.mul(output_gate, graph.tanh(new_c)?)?;
            let write_h = graph.cpy(new_h, h)?;
            let write_c = graph.cpy(new_c, c)?;
            graph.add_side_effect_root(write_h)?;
            graph.add_side_effect_root(write_c)?;
            input = new_h;
        }
        let mut pred_proj = graph.mul_mat(loaded_tensor(loaded, "joint.pred.weight")?, input)?;
        pred_proj = graph.add(pred_proj, loaded_tensor(loaded, "joint.pred.bias")?)?;
        let write_pred_proj = graph.cpy(pred_proj, state_arena.graph_tensor(pred_proj_state))?;
        graph.add_side_effect_root(write_pred_proj)?;
        graph.prepare_side_effects_for_upload()?;
        Ok(PredictorStepGraph {
            session,
            token,
            h_inputs: Vec::new(),
            c_inputs: Vec::new(),
            pred_proj_output: None,
            h_outputs: Vec::new(),
            c_outputs: Vec::new(),
        })
    }

    fn build_host_joint(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        metadata: ParakeetTdtExecutionMetadata,
    ) -> Result<JointStepGraph, GgmlCpuGraphError> {
        // add -> ReLU -> projection -> bias is four compute nodes. Thirty-two
        // leaves ample graph bookkeeping headroom without inheriting the
        // encoder's resident metadata allocation.
        let mut session = runner.start_persistent_graph_session_with_node_capacity(32)?;
        let graph = session.builder();
        let encoder_frame =
            graph.new_tensor_1d_f32(metadata.joint_hidden, "parakeet_tdt_joint_encoder")?;
        let pred_proj =
            graph.new_tensor_1d_f32(metadata.joint_hidden, "parakeet_tdt_joint_predictor")?;
        graph.set_input(encoder_frame)?;
        graph.set_input(pred_proj)?;
        let mid = graph.relu(graph.add(encoder_frame, pred_proj)?)?;
        let mut logits = graph.mul_mat(loaded_tensor(loaded, "joint.out.weight")?, mid)?;
        logits = graph.add(logits, loaded_tensor(loaded, "joint.out.bias")?)?;
        graph.set_output(logits)?;
        graph.prepare_outputs_for_upload(&[logits])?;
        Ok(JointStepGraph {
            session,
            encoder_frame,
            pred_proj: Some(pred_proj),
            logits,
        })
    }

    fn build_resident_joint(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        state_arena: &GgmlStaticTensorArena,
        pred_proj_state: GgmlStaticTensor,
        metadata: ParakeetTdtExecutionMetadata,
    ) -> Result<JointStepGraph, GgmlCpuGraphError> {
        let mut session = runner.start_persistent_graph_session_with_node_capacity(32)?;
        let graph = session.builder();
        let encoder_frame =
            graph.new_tensor_1d_f32(metadata.joint_hidden, "parakeet_tdt_joint_encoder")?;
        graph.set_input(encoder_frame)?;
        let pred_proj = state_arena.graph_tensor(pred_proj_state);
        let mid = graph.relu(graph.add(encoder_frame, pred_proj)?)?;
        let mut logits = graph.mul_mat(loaded_tensor(loaded, "joint.out.weight")?, mid)?;
        logits = graph.add(logits, loaded_tensor(loaded, "joint.out.bias")?)?;
        graph.set_output(logits)?;
        graph.prepare_outputs_for_upload(&[logits])?;
        Ok(JointStepGraph {
            session,
            encoder_frame,
            pred_proj: None,
            logits,
        })
    }

    fn predictor_step(&mut self, token_id: u32) -> Result<(), String> {
        let token = i32::try_from(token_id)
            .map_err(|_| format!("parakeet-tdt predictor token {token_id} exceeds i32"))?;
        let Self {
            predictor, state, ..
        } = self;
        let graph = predictor.session.builder();
        graph
            .set_i32_slice(predictor.token, &[token], "parakeet_tdt_predictor_token")
            .map_err(|error| error.to_string())?;
        match state {
            PredictorState::Host { h, c, pred_proj } => {
                for (tensor, values) in predictor.h_inputs.iter().copied().zip(h.iter()) {
                    graph
                        .set_f32_slice(tensor, values, "parakeet_tdt_predictor_h")
                        .map_err(|error| error.to_string())?;
                }
                for (tensor, values) in predictor.c_inputs.iter().copied().zip(c.iter()) {
                    graph
                        .set_f32_slice(tensor, values, "parakeet_tdt_predictor_c")
                        .map_err(|error| error.to_string())?;
                }
                let mut targets = Vec::with_capacity(1 + h.len() + c.len());
                targets.push((
                    predictor.pred_proj_output.ok_or_else(|| {
                        "parakeet-tdt host predictor is missing projection output".to_string()
                    })?,
                    pred_proj.as_mut_slice(),
                ));
                targets.extend(
                    predictor
                        .h_outputs
                        .iter()
                        .copied()
                        .zip(h.iter_mut().map(Vec::as_mut_slice)),
                );
                targets.extend(
                    predictor
                        .c_outputs
                        .iter()
                        .copied()
                        .zip(c.iter_mut().map(Vec::as_mut_slice)),
                );
                graph
                    .compute_outputs_into_f32(&mut targets)
                    .map_err(|error| error.to_string())
            }
            PredictorState::Resident { .. } => graph
                .compute_side_effects()
                .map_err(|error| error.to_string()),
        }
    }
}

impl ParakeetTdtDecodeBackend for ParakeetTdtDeviceDecoder {
    fn output_rows(&self) -> usize {
        self.logits.len()
    }

    fn begin(&mut self, blank_token_id: u32) -> Result<(), String> {
        match &mut self.state {
            PredictorState::Host { h, c, .. } => {
                for values in h.iter_mut().chain(c) {
                    values.fill(0.0);
                }
            }
            PredictorState::Resident {
                arena,
                h,
                c,
                reset_zeros,
                state_width,
            } => {
                for tensor in h.iter().chain(c.iter()) {
                    arena
                        .set_f32_slice(
                            *tensor,
                            &reset_zeros[..*state_width],
                            "parakeet_tdt_resident_predictor_reset",
                        )
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        self.predictor_step(blank_token_id)
    }

    fn logits<'a>(&'a mut self, encoder_frame: &[f32]) -> Result<&'a [f32], String> {
        let Self {
            joint,
            state,
            logits,
            joint_hidden,
            ..
        } = self;
        if encoder_frame.len() != *joint_hidden {
            return Err(format!(
                "parakeet-tdt joint encoder width {}, expected {}",
                encoder_frame.len(),
                joint_hidden
            ));
        }
        let graph = joint.session.builder();
        graph
            .set_f32_slice(
                joint.encoder_frame,
                encoder_frame,
                "parakeet_tdt_joint_encoder",
            )
            .map_err(|error| error.to_string())?;
        if let Some(pred_proj_tensor) = joint.pred_proj {
            let PredictorState::Host { pred_proj, .. } = state else {
                return Err(
                    "parakeet-tdt resident predictor unexpectedly used a host joint input"
                        .to_string(),
                );
            };
            graph
                .set_f32_slice(pred_proj_tensor, pred_proj, "parakeet_tdt_joint_predictor")
                .map_err(|error| error.to_string())?;
        } else if !matches!(state, PredictorState::Resident { .. }) {
            return Err(
                "parakeet-tdt host predictor unexpectedly used resident joint state".to_string(),
            );
        }
        graph
            .compute_outputs_into_f32(&mut [(joint.logits, logits.as_mut_slice())])
            .map_err(|error| error.to_string())?;
        Ok(logits)
    }

    fn accept_token(&mut self, token_id: u32) -> Result<(), String> {
        self.predictor_step(token_id)
    }
}
