//! Device-resident Parakeet-TDT prediction network and joint head.
//!
//! The FastConformer encoder, predictor LSTM, and joint head share one runner
//! and one mmap-backed loaded-weight context. Only recurrent state, one encoder
//! frame, and logits cross the device boundary; predictor/joint weights never
//! acquire a second host-f32 representation on accelerated routes.

use std::mem::size_of;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedWeightContext,
    GgmlPersistentGraphSession,
};

use super::greedy::ParakeetTdtDecodeBackend;
use super::runtime_contract::ParakeetTdtExecutionMetadata;

struct PredictorStepGraph {
    session: GgmlPersistentGraphSession,
    token: GgmlCpuTensor<'static>,
    h_inputs: Vec<GgmlCpuTensor<'static>>,
    c_inputs: Vec<GgmlCpuTensor<'static>>,
    pred_proj_output: GgmlCpuTensor<'static>,
    h_outputs: Vec<GgmlCpuTensor<'static>>,
    c_outputs: Vec<GgmlCpuTensor<'static>>,
}

struct JointStepGraph {
    session: GgmlPersistentGraphSession,
    encoder_frame: GgmlCpuTensor<'static>,
    pred_proj: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
}

/// Stateful accelerated decoder. Field order is intentional: both graph
/// sessions must drop before the encoder core frees the shared runner, loaded
/// weight context, or static arena that owns their raw tensor dependencies.
pub(crate) struct ParakeetTdtDeviceDecoder {
    predictor: PredictorStepGraph,
    joint: JointStepGraph,
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    pred_proj: Vec<f32>,
    logits: Vec<f32>,
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

fn checked_payload_bytes(metadata: ParakeetTdtExecutionMetadata) -> Result<u64, String> {
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
        .checked_mul((2 * size_of::<Vec<f32>>() + 4 * size_of::<GgmlCpuTensor<'static>>()) as u64)
        .ok_or_else(|| "parakeet-tdt device decoder descriptor bytes overflowed".to_string())?;
    state_bytes
        .checked_add(descriptor_bytes)
        .ok_or_else(|| "parakeet-tdt device decoder retained bytes overflowed".to_string())
}

pub(crate) fn planned_retained_system_memory_bytes(
    metadata: ParakeetTdtExecutionMetadata,
) -> Result<u64, String> {
    checked_payload_bytes(metadata)
}

impl ParakeetTdtDeviceDecoder {
    pub(crate) fn new(
        runner: &mut GgmlCpuGraphRunner,
        loaded: &GgmlLoadedWeightContext,
        metadata: ParakeetTdtExecutionMetadata,
    ) -> Result<Self, GgmlCpuGraphError> {
        let predictor = Self::build_predictor(runner, loaded, metadata)?;
        let joint = Self::build_joint(runner, loaded, metadata)?;
        let out_rows = metadata
            .vocab_size
            .checked_add(metadata.n_durations)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "parakeet-tdt device decoder output width overflowed",
            })?;
        Ok(Self {
            predictor,
            joint,
            h: vec![vec![0.0; metadata.pred_hidden]; metadata.pred_layers],
            c: vec![vec![0.0; metadata.pred_hidden]; metadata.pred_layers],
            pred_proj: vec![0.0; metadata.joint_hidden],
            logits: vec![0.0; out_rows],
        })
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let state_bytes = self
            .h
            .iter()
            .chain(&self.c)
            .try_fold(0_u64, |total, values| {
                total
                    .checked_add((values.capacity() * size_of::<f32>()) as u64)
                    .ok_or_else(|| "parakeet-tdt device state bytes overflowed".to_string())
            })?;
        let flat_bytes = [self.pred_proj.capacity(), self.logits.capacity()]
            .into_iter()
            .try_fold(0_u64, |total, capacity| {
                total
                    .checked_add((capacity * size_of::<f32>()) as u64)
                    .ok_or_else(|| "parakeet-tdt device output bytes overflowed".to_string())
            })?;
        let outer_bytes = ((self.h.capacity() + self.c.capacity()) * size_of::<Vec<f32>>()) as u64;
        let handle_count = self
            .predictor
            .h_inputs
            .capacity()
            .checked_add(self.predictor.c_inputs.capacity())
            .and_then(|value| value.checked_add(self.predictor.h_outputs.capacity()))
            .and_then(|value| value.checked_add(self.predictor.c_outputs.capacity()))
            .ok_or_else(|| "parakeet-tdt device handle count overflowed".to_string())?;
        let handle_bytes = (handle_count * size_of::<GgmlCpuTensor<'static>>()) as u64;
        state_bytes
            .checked_add(flat_bytes)
            .and_then(|value| value.checked_add(outer_bytes))
            .and_then(|value| value.checked_add(handle_bytes))
            .ok_or_else(|| "parakeet-tdt device retained bytes overflowed".to_string())
    }

    fn build_predictor(
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
            pred_proj_output: pred_proj,
            h_outputs,
            c_outputs,
        })
    }

    fn build_joint(
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
            pred_proj,
            logits,
        })
    }

    fn predictor_step(&mut self, token_id: u32) -> Result<(), String> {
        let token = i32::try_from(token_id)
            .map_err(|_| format!("parakeet-tdt predictor token {token_id} exceeds i32"))?;
        let Self {
            predictor,
            h,
            c,
            pred_proj,
            ..
        } = self;
        let graph = predictor.session.builder();
        graph
            .set_i32_slice(predictor.token, &[token], "parakeet_tdt_predictor_token")
            .map_err(|error| error.to_string())?;
        for ((tensor, values), name) in predictor
            .h_inputs
            .iter()
            .copied()
            .zip(h.iter())
            .zip(std::iter::repeat("parakeet_tdt_predictor_h"))
        {
            graph
                .set_f32_slice(tensor, values, name)
                .map_err(|error| error.to_string())?;
        }
        for (tensor, values) in predictor.c_inputs.iter().copied().zip(c.iter()) {
            graph
                .set_f32_slice(tensor, values, "parakeet_tdt_predictor_c")
                .map_err(|error| error.to_string())?;
        }
        let mut targets = Vec::with_capacity(1 + h.len() + c.len());
        targets.push((predictor.pred_proj_output, pred_proj.as_mut_slice()));
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
}

impl ParakeetTdtDecodeBackend for ParakeetTdtDeviceDecoder {
    fn output_rows(&self) -> usize {
        self.logits.len()
    }

    fn begin(&mut self, blank_token_id: u32) -> Result<(), String> {
        for values in self.h.iter_mut().chain(&mut self.c) {
            values.fill(0.0);
        }
        self.predictor_step(blank_token_id)
    }

    fn logits<'a>(&'a mut self, encoder_frame: &[f32]) -> Result<&'a [f32], String> {
        let Self {
            joint,
            pred_proj,
            logits,
            ..
        } = self;
        if encoder_frame.len() != pred_proj.len() {
            return Err(format!(
                "parakeet-tdt joint encoder width {}, expected {}",
                encoder_frame.len(),
                pred_proj.len()
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
        graph
            .set_f32_slice(joint.pred_proj, pred_proj, "parakeet_tdt_joint_predictor")
            .map_err(|error| error.to_string())?;
        graph
            .compute_outputs_into_f32(&mut [(joint.logits, logits.as_mut_slice())])
            .map_err(|error| error.to_string())?;
        Ok(logits)
    }

    fn accept_token(&mut self, token_id: u32) -> Result<(), String> {
        self.predictor_step(token_id)
    }
}
