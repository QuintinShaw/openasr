//! Persistent ggml execution for FireRed Stream-VAD's causal DFSMN stack.
//!
//! The Kaldi-compatible frontend and the per-session 19-frame causal caches
//! remain family-owned host state. This module owns one uploaded copy of the
//! vendored weights and one persistent graph per frame geometry; selecting
//! Metal therefore changes only the implementation of the same mathematical
//! forward pass, not the streaming state machine or its chunking contract.

use thiserror::Error;

use super::{
    frontend::NUM_MEL_BINS,
    model::{
        CACHE_FRAMES, FireRedStreamVadCache, FireRedStreamVadModel, HIDDEN, LOOKBACK_ORDER,
        NUM_BLOCKS, PROJ,
    },
    weights::{BlockWeights, FireRedStreamVadWeights},
};
use crate::device::execution_policy::ExecutionPlacement;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlPersistentGraphSession, GgmlStaticTensor,
    GgmlStaticTensorArena,
};

const GRAPH_SIZE: usize = 1 << 12;
const ARENA_TENSORS: usize = 1 << 8;
const F32_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Debug, Error)]
pub(crate) enum FireRedStreamVadGgmlError {
    #[error(transparent)]
    Graph(#[from] GgmlCpuGraphError),
    #[error("firered Stream-VAD feature payload has {got} values, expected {expected}")]
    InvalidFeaturePayload { got: usize, expected: usize },
}

#[derive(Clone, Copy)]
struct LinearHandles {
    weight: GgmlStaticTensor,
    bias: Option<GgmlStaticTensor>,
}

#[derive(Clone, Copy)]
struct LayerHandles {
    fc1: LinearHandles,
    fc2: LinearHandles,
    lookback_tap_major: GgmlStaticTensor,
}

struct WeightHandles {
    first: LayerHandles,
    blocks: Vec<LayerHandles>,
    dnn: LinearHandles,
    output: LinearHandles,
}

impl WeightHandles {
    fn allocate(
        arena: &GgmlStaticTensorArena,
        weights: &FireRedStreamVadWeights,
    ) -> Result<Self, GgmlCpuGraphError> {
        let first = LayerHandles {
            fc1: allocate_linear(arena, NUM_MEL_BINS, HIDDEN, true)?,
            fc2: allocate_linear(arena, HIDDEN, PROJ, true)?,
            lookback_tap_major: arena.new_tensor_2d_f32(
                PROJ,
                LOOKBACK_ORDER,
                "firered_vad_lookback",
            )?,
        };
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for _ in &weights.blocks {
            blocks.push(LayerHandles {
                fc1: allocate_linear(arena, PROJ, HIDDEN, true)?,
                fc2: allocate_linear(arena, HIDDEN, PROJ, false)?,
                lookback_tap_major: arena.new_tensor_2d_f32(
                    PROJ,
                    LOOKBACK_ORDER,
                    "firered_vad_lookback",
                )?,
            });
        }
        Ok(Self {
            first,
            blocks,
            dnn: allocate_linear(arena, PROJ, HIDDEN, true)?,
            output: allocate_linear(arena, HIDDEN, 1, true)?,
        })
    }

    fn upload(
        &self,
        arena: &mut GgmlStaticTensorArena,
        weights: &FireRedStreamVadWeights,
    ) -> Result<(), GgmlCpuGraphError> {
        upload_linear(arena, self.first.fc1, &weights.fc1_w, Some(&weights.fc1_b))?;
        upload_linear(arena, self.first.fc2, &weights.fc2_w, Some(&weights.fc2_b))?;
        upload_lookback(
            arena,
            self.first.lookback_tap_major,
            &weights.fsmn1_lookback,
        )?;
        for (handles, block) in self.blocks.iter().zip(&weights.blocks) {
            upload_block(arena, *handles, block)?;
        }
        upload_linear(arena, self.dnn, &weights.dnn_w, Some(&weights.dnn_b))?;
        upload_linear(
            arena,
            self.output,
            &weights.out_w,
            Some(std::slice::from_ref(&weights.out_b)),
        )?;
        Ok(())
    }
}

fn allocate_linear(
    arena: &GgmlStaticTensorArena,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<LinearHandles, GgmlCpuGraphError> {
    Ok(LinearHandles {
        weight: arena.new_tensor_2d_f32(input, output, "firered_vad_linear_weight")?,
        bias: bias
            .then(|| arena.new_tensor_1d_f32(output, "firered_vad_linear_bias"))
            .transpose()?,
    })
}

fn upload_linear(
    arena: &mut GgmlStaticTensorArena,
    handles: LinearHandles,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<(), GgmlCpuGraphError> {
    arena.set_f32_slice(handles.weight, weight, "firered_vad_linear_weight")?;
    match (handles.bias, bias) {
        (Some(handle), Some(values)) => {
            arena.set_f32_slice(handle, values, "firered_vad_linear_bias")?;
        }
        (None, None) => {}
        _ => {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "FireRedVAD linear bias contract mismatch",
            });
        }
    }
    Ok(())
}

fn upload_block(
    arena: &mut GgmlStaticTensorArena,
    handles: LayerHandles,
    block: &BlockWeights,
) -> Result<(), GgmlCpuGraphError> {
    upload_linear(arena, handles.fc1, &block.fc1_w, Some(&block.fc1_b))?;
    upload_linear(arena, handles.fc2, &block.fc2_w, None)?;
    upload_lookback(arena, handles.lookback_tap_major, &block.lookback)
}

fn upload_lookback(
    arena: &mut GgmlStaticTensorArena,
    handle: GgmlStaticTensor,
    channel_major: &[f32],
) -> Result<(), GgmlCpuGraphError> {
    let mut tap_major = vec![0.0f32; PROJ * LOOKBACK_ORDER];
    for channel in 0..PROJ {
        for tap in 0..LOOKBACK_ORDER {
            tap_major[tap * PROJ + channel] = channel_major[channel * LOOKBACK_ORDER + tap];
        }
    }
    arena.set_f32_slice(handle, &tap_major, "firered_vad_lookback")
}

struct ResidentWeights {
    handles: WeightHandles,
    arena: GgmlStaticTensorArena,
}

struct PersistentGraph {
    session: GgmlPersistentGraphSession,
    features: GgmlCpuTensor<'static>,
    cache_inputs: Vec<GgmlCpuTensor<'static>>,
    probabilities: GgmlCpuTensor<'static>,
    cache_outputs: Vec<GgmlCpuTensor<'static>>,
    frames: usize,
}

/// Thread-confined, resident FireRedVAD graph runtime.
///
/// Field order is load-bearing: graph tensors drop before their static arena,
/// and the arena drops before the backend runner that allocated it.
pub(crate) struct FireRedStreamVadGgmlRuntime {
    graph: Option<PersistentGraph>,
    resident: ResidentWeights,
    runner: GgmlCpuGraphRunner,
}

impl FireRedStreamVadGgmlRuntime {
    pub(crate) fn new(
        model: &FireRedStreamVadModel,
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Result<Self, FireRedStreamVadGgmlError> {
        let mut config = GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend);
        config.graph_size = GRAPH_SIZE;
        config.context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE);
        let config =
            crate::models::graph_runtime_config::apply_execution_placement(config, placement);
        let runner = GgmlCpuGraphRunner::new(config)?;
        let arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(ARENA_TENSORS))?;
        let handles = WeightHandles::allocate(&arena, model.weights())?;
        let mut arena = arena;
        handles.upload(&mut arena, model.weights())?;
        Ok(Self {
            graph: None,
            resident: ResidentWeights { handles, arena },
            runner,
        })
    }

    pub(crate) fn forward_chunk(
        &mut self,
        cmvn_feat: &[f32],
        frames: usize,
        cache: &mut FireRedStreamVadCache,
    ) -> Result<Vec<f32>, FireRedStreamVadGgmlError> {
        let expected = frames.saturating_mul(NUM_MEL_BINS);
        if cmvn_feat.len() != expected {
            return Err(FireRedStreamVadGgmlError::InvalidFeaturePayload {
                got: cmvn_feat.len(),
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
        let persistent = self.graph.as_mut().expect("FireRedVAD graph built");
        let graph = persistent.session.builder();
        graph.set_f32_slice(persistent.features, cmvn_feat, "firered_vad_features")?;
        let mut padded_cache = vec![0.0f32; CACHE_FRAMES * PROJ];
        for (layer, input) in persistent.cache_inputs.iter().enumerate() {
            padded_cache.fill(0.0);
            let values = cache.layer(layer);
            let copy_len = values.len().min(padded_cache.len());
            let src_start = values.len() - copy_len;
            let dst_start = padded_cache.len() - copy_len;
            padded_cache[dst_start..].copy_from_slice(&values[src_start..]);
            graph.set_f32_slice(*input, &padded_cache, "firered_vad_cache")?;
        }
        let mut outputs = Vec::with_capacity(NUM_BLOCKS + 2);
        outputs.push((persistent.probabilities, frames));
        outputs.extend(
            persistent
                .cache_outputs
                .iter()
                .copied()
                .map(|output| (output, CACHE_FRAMES * PROJ)),
        );
        let mut values = graph.compute_outputs_f32(&outputs)?;
        let probabilities = values.remove(0);
        for (layer, layer_cache) in values.into_iter().enumerate() {
            cache.replace_layer(layer, layer_cache);
        }
        Ok(probabilities)
    }

    fn build_graph(&mut self, frames: usize) -> Result<PersistentGraph, GgmlCpuGraphError> {
        let mut session = self.runner.start_persistent_graph_session(
            GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE),
        )?;
        let graph = session.builder();
        let features = graph.new_tensor_2d_f32(NUM_MEL_BINS, frames, "firered_vad_features")?;
        let mut cache_inputs = Vec::with_capacity(NUM_BLOCKS + 1);
        for _ in 0..=NUM_BLOCKS {
            cache_inputs.push(graph.new_tensor_2d_f32(PROJ, CACHE_FRAMES, "firered_vad_cache")?);
        }

        let handles = &self.resident.handles;
        let hidden = linear_relu(graph, &self.resident.arena, handles.first.fc1, features)?;
        let projected = linear_relu(graph, &self.resident.arena, handles.first.fc2, hidden)?;
        let (mut memory, first_cache) = causal_fsmn(
            graph,
            &self.resident.arena,
            projected,
            cache_inputs[0],
            handles.first.lookback_tap_major,
            frames,
        )?;
        let mut cache_outputs = vec![first_cache];
        for (layer, layer_handles) in handles.blocks.iter().enumerate() {
            let hidden = linear_relu(graph, &self.resident.arena, layer_handles.fc1, memory)?;
            let projected = linear(graph, &self.resident.arena, layer_handles.fc2, hidden)?;
            let (filtered, next_cache) = causal_fsmn(
                graph,
                &self.resident.arena,
                projected,
                cache_inputs[layer + 1],
                layer_handles.lookback_tap_major,
                frames,
            )?;
            memory = graph.add(filtered, memory)?;
            cache_outputs.push(next_cache);
        }
        let dnn = linear_relu(graph, &self.resident.arena, handles.dnn, memory)?;
        let logits = linear(graph, &self.resident.arena, handles.output, dnn)?;
        let probabilities = graph.sigmoid(logits)?;
        graph.set_input(features)?;
        for input in &cache_inputs {
            graph.set_input(*input)?;
        }
        graph.set_output(probabilities)?;
        for output in &cache_outputs {
            graph.set_output(*output)?;
        }
        let mut outputs = Vec::with_capacity(cache_outputs.len() + 1);
        outputs.push(probabilities);
        outputs.extend(cache_outputs.iter().copied());
        graph.prepare_outputs_for_upload(&outputs)?;
        Ok(PersistentGraph {
            session,
            features,
            cache_inputs,
            probabilities,
            cache_outputs,
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
    let projected = graph.mul_mat(arena.graph_tensor(handles.weight), input)?;
    handles.bias.map_or(Ok(projected), |bias| {
        graph.add(projected, arena.graph_tensor(bias))
    })
}

fn linear_relu<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    handles: LinearHandles,
    input: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    graph.relu(linear(graph, arena, handles, input)?)
}

fn causal_fsmn<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    projected: GgmlCpuTensor<'a>,
    cache: GgmlCpuTensor<'a>,
    lookback_tap_major: GgmlStaticTensor,
    frames: usize,
) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), GgmlCpuGraphError> {
    let combined = graph.cont(graph.concat(cache, projected, 1)?)?;
    let lookback = arena.graph_tensor(lookback_tap_major);
    let row_bytes = PROJ * F32_BYTES;
    let mut sum = None;
    for tap in 0..LOOKBACK_ORDER {
        let samples = graph.view_2d(combined, PROJ, frames, row_bytes, tap * row_bytes)?;
        let tap_weight = graph.view_1d(lookback, PROJ, tap * row_bytes)?;
        let weighted = graph.mul(samples, tap_weight)?;
        sum = Some(match sum {
            Some(accumulator) => graph.add(accumulator, weighted)?,
            None => weighted,
        });
    }
    let filtered = graph.add(projected, sum.expect("LOOKBACK_ORDER is non-zero"))?;
    let next_cache = graph.cont(graph.view_2d(
        combined,
        PROJ,
        CACHE_FRAMES,
        row_bytes,
        frames * row_bytes,
    )?)?;
    Ok((filtered, next_cache))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_features(frames: usize) -> Vec<f32> {
        (0..frames * NUM_MEL_BINS)
            .map(|index| ((index as f32 * 0.017).sin() + (index as f32 * 0.003).cos()) * 0.25)
            .collect()
    }

    #[test]
    fn ggml_cpu_matches_family_forward_across_chunk_boundaries() {
        let model = FireRedStreamVadModel::embedded().expect("embedded FireRedVAD");
        let mut runtime = FireRedStreamVadGgmlRuntime::new(
            &model,
            GgmlCpuGraphBackend::Cpu,
            ExecutionPlacement::CpuOnly,
        )
        .expect("CPU ggml runtime");
        let mut reference_cache = FireRedStreamVadCache::new();
        let mut runtime_cache = FireRedStreamVadCache::new();
        for frames in [1, 7, 23, 7] {
            let features = synthetic_features(frames);
            let reference = model.forward_chunk(&features, frames, &mut reference_cache);
            let actual = runtime
                .forward_chunk(&features, frames, &mut runtime_cache)
                .expect("ggml forward");
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 2e-4, "CPU ggml max abs {max_abs}");
        }
    }

    #[test]
    fn ggml_runtime_rejects_mismatched_feature_payload() {
        let model = FireRedStreamVadModel::embedded().expect("embedded FireRedVAD");
        let mut runtime = FireRedStreamVadGgmlRuntime::new(
            &model,
            GgmlCpuGraphBackend::Cpu,
            ExecutionPlacement::CpuOnly,
        )
        .expect("CPU ggml runtime");
        let error = runtime
            .forward_chunk(
                &[0.0; NUM_MEL_BINS - 1],
                1,
                &mut FireRedStreamVadCache::new(),
            )
            .expect_err("mismatched features must fail closed");
        assert!(matches!(
            error,
            FireRedStreamVadGgmlError::InvalidFeaturePayload { .. }
        ));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn ggml_metal_matches_family_forward_across_chunk_boundaries() {
        let model = FireRedStreamVadModel::embedded().expect("embedded FireRedVAD");
        let mut runtime = FireRedStreamVadGgmlRuntime::new(
            &model,
            GgmlCpuGraphBackend::Metal,
            ExecutionPlacement::FullDevice,
        )
        .expect("Apple Silicon must construct the FireRedVAD Metal graph");
        assert!(!runtime.runner.uses_scheduler());
        let mut reference_cache = FireRedStreamVadCache::new();
        let mut runtime_cache = FireRedStreamVadCache::new();
        for frames in [1, 7, 23, 7] {
            let features = synthetic_features(frames);
            let reference = model.forward_chunk(&features, frames, &mut reference_cache);
            let actual = runtime
                .forward_chunk(&features, frames, &mut runtime_cache)
                .expect("Metal ggml forward");
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 5e-3, "Metal ggml max abs {max_abs}");
        }
    }
}
