//! Parameterized WeSpeaker ResNet ggml graph.
//!
//! Stem → four stages (BasicBlock or Bottleneck from the size table) → flatten
//! → TSTP → `seg_1` Linear. Arena upload, persistent graph, and CPU/Metal
//! backends follow the ReDimNet resident pattern without copying that graph.

use std::sync::Arc;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlPersistentGraphSession, GgmlStaticTensor, GgmlStaticTensorArena,
};

use super::super::weights::{Weights, WeightsError};
use super::config::{
    self, BN_EPS, BlockKind, EMBED_DIM, N_MELS, ResNetConfig, STAGE_STRIDES, TSTP_EPS,
};
use super::ops;

#[derive(Debug, thiserror::Error)]
pub(crate) enum WeSpeakerBackboneError {
    #[error("wespeaker backbone weight error: {0}")]
    Weights(#[from] WeightsError),
    #[error("wespeaker backbone shape error: {reason}")]
    Shape { reason: String },
    #[error("wespeaker backbone ggml error: {0}")]
    Ggml(#[from] GgmlCpuGraphError),
}

impl WeSpeakerBackboneError {
    pub(crate) fn is_canceled(&self) -> bool {
        matches!(
            self,
            Self::Ggml(GgmlCpuGraphError::Aborted | GgmlCpuGraphError::Canceled)
        )
    }

    pub(crate) fn is_terminal_backend_failure(&self) -> bool {
        matches!(
            self,
            Self::Ggml(GgmlCpuGraphError::DeviceLost | GgmlCpuGraphError::BackendPoisoned)
        )
    }
}

fn shape_err(reason: impl Into<String>) -> WeSpeakerBackboneError {
    WeSpeakerBackboneError::Shape {
        reason: reason.into(),
    }
}

enum PendingData<'p> {
    Borrowed(&'p [f32]),
    Owned(Vec<f32>),
}

impl PendingData<'_> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(data) => data,
            Self::Owned(data) => data,
        }
    }
}

struct Pending<'p> {
    handle: GgmlStaticTensor,
    data: PendingData<'p>,
}

struct WBuilder<'p> {
    weights: &'p Weights,
    pending: Vec<Pending<'p>>,
}

impl<'p> WBuilder<'p> {
    fn new(weights: &'p Weights) -> Self {
        Self {
            weights,
            pending: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expect_ne: &[usize]) -> Result<&'p [f32], WeSpeakerBackboneError> {
        let shape = self.weights.shape(name)?;
        if shape != expect_ne {
            return Err(shape_err(format!(
                "tensor '{name}' has pack shape {shape:?}, expected ne {expect_ne:?}"
            )));
        }
        Ok(self.weights.get(name)?)
    }

    fn tensor_1d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, WeSpeakerBackboneError> {
        let data = self.fetch(name, &[len])?;
        let handle = arena.new_tensor_1d_f32(len, "wespeaker_weight")?;
        self.pending.push(Pending {
            handle,
            data: PendingData::Borrowed(data),
        });
        Ok(arena.graph_tensor(handle))
    }

    fn tensor_2d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, WeSpeakerBackboneError> {
        let data = self.fetch(name, &[ne0, ne1])?;
        let handle = arena.new_tensor_2d_f32(ne0, ne1, "wespeaker_weight")?;
        self.pending.push(Pending {
            handle,
            data: PendingData::Borrowed(data),
        });
        Ok(arena.graph_tensor(handle))
    }

    fn tensor_4d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, WeSpeakerBackboneError> {
        let data = self.fetch(name, &[ne0, ne1, ne2, ne3])?;
        let handle = arena.new_tensor_4d_f32(ne0, ne1, ne2, ne3, "wespeaker_weight")?;
        self.pending.push(Pending {
            handle,
            data: PendingData::Borrowed(data),
        });
        Ok(arena.graph_tensor(handle))
    }

    fn batchnorm_affine<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        prefix: &str,
        channels: usize,
    ) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), WeSpeakerBackboneError> {
        let gamma = self.fetch(&format!("{prefix}.weight"), &[channels])?;
        let beta = self.fetch(&format!("{prefix}.bias"), &[channels])?;
        let mean = self.fetch(&format!("{prefix}.running_mean"), &[channels])?;
        let var = self.fetch(&format!("{prefix}.running_var"), &[channels])?;
        let (scale, shift) = ops::batchnorm_affine(gamma, beta, mean, var, BN_EPS);
        let scale_handle = arena.new_tensor_1d_f32(channels, "wespeaker_bn_scale")?;
        let shift_handle = arena.new_tensor_1d_f32(channels, "wespeaker_bn_shift")?;
        self.pending.push(Pending {
            handle: scale_handle,
            data: PendingData::Owned(scale),
        });
        self.pending.push(Pending {
            handle: shift_handle,
            data: PendingData::Owned(shift),
        });
        Ok((
            arena.graph_tensor(scale_handle),
            arena.graph_tensor(shift_handle),
        ))
    }

    fn scalar<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        value: f32,
    ) -> Result<GgmlCpuTensor<'a>, WeSpeakerBackboneError> {
        let handle = arena.new_tensor_2d_f32(1, 1, "wespeaker_scalar")?;
        self.pending.push(Pending {
            handle,
            data: PendingData::Owned(vec![value]),
        });
        Ok(arena.graph_tensor(handle))
    }

    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), WeSpeakerBackboneError> {
        for pending in &self.pending {
            arena.set_f32_slice(pending.handle, pending.data.as_slice(), "wespeaker_weight")?;
        }
        Ok(())
    }
}

enum BlockW<'a> {
    Basic(ops::BasicBlockWeights<'a>),
    Bottleneck(ops::BottleneckWeights<'a>),
}

struct WeSpeakerBackboneWeights<'a> {
    config: ResNetConfig,
    stem_conv: GgmlCpuTensor<'a>,
    stem_bn_scale: GgmlCpuTensor<'a>,
    stem_bn_shift: GgmlCpuTensor<'a>,
    blocks: Vec<BlockW<'a>>,
    seg_w: GgmlCpuTensor<'a>,
    seg_b: GgmlCpuTensor<'a>,
    eps_1e7: GgmlCpuTensor<'a>,
}

fn load_basic_block<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    prefix: &str,
    c_in: usize,
    c_out: usize,
    stride: usize,
    shortcut: bool,
) -> Result<ops::BasicBlockWeights<'a>, WeSpeakerBackboneError> {
    let conv1 = b.tensor_4d(arena, &format!("{prefix}.conv1.weight"), 3, 3, c_in, c_out)?;
    let (bn1_scale, bn1_shift) = b.batchnorm_affine(arena, &format!("{prefix}.bn1"), c_out)?;
    let conv2 = b.tensor_4d(arena, &format!("{prefix}.conv2.weight"), 3, 3, c_out, c_out)?;
    let (bn2_scale, bn2_shift) = b.batchnorm_affine(arena, &format!("{prefix}.bn2"), c_out)?;
    let (shortcut_conv, shortcut_scale, shortcut_shift) = if shortcut {
        let conv = b.tensor_4d(
            arena,
            &format!("{prefix}.shortcut.0.weight"),
            1,
            1,
            c_in,
            c_out,
        )?;
        let (scale, shift) = b.batchnorm_affine(arena, &format!("{prefix}.shortcut.1"), c_out)?;
        (Some(conv), Some(scale), Some(shift))
    } else {
        let _ = stride;
        (None, None, None)
    };
    Ok(ops::BasicBlockWeights {
        conv1,
        bn1_scale,
        bn1_shift,
        conv2,
        bn2_scale,
        bn2_shift,
        shortcut_conv,
        shortcut_scale,
        shortcut_shift,
    })
}

fn load_bottleneck_block<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    prefix: &str,
    c_in: usize,
    planes: usize,
    c_out: usize,
    stride: usize,
    shortcut: bool,
) -> Result<ops::BottleneckWeights<'a>, WeSpeakerBackboneError> {
    let conv1 = b.tensor_4d(arena, &format!("{prefix}.conv1.weight"), 1, 1, c_in, planes)?;
    let (bn1_scale, bn1_shift) = b.batchnorm_affine(arena, &format!("{prefix}.bn1"), planes)?;
    let conv2 = b.tensor_4d(
        arena,
        &format!("{prefix}.conv2.weight"),
        3,
        3,
        planes,
        planes,
    )?;
    let (bn2_scale, bn2_shift) = b.batchnorm_affine(arena, &format!("{prefix}.bn2"), planes)?;
    let conv3 = b.tensor_4d(
        arena,
        &format!("{prefix}.conv3.weight"),
        1,
        1,
        planes,
        c_out,
    )?;
    let (bn3_scale, bn3_shift) = b.batchnorm_affine(arena, &format!("{prefix}.bn3"), c_out)?;
    let (shortcut_conv, shortcut_scale, shortcut_shift) = if shortcut {
        let conv = b.tensor_4d(
            arena,
            &format!("{prefix}.shortcut.0.weight"),
            1,
            1,
            c_in,
            c_out,
        )?;
        let (scale, shift) = b.batchnorm_affine(arena, &format!("{prefix}.shortcut.1"), c_out)?;
        (Some(conv), Some(scale), Some(shift))
    } else {
        let _ = stride;
        (None, None, None)
    };
    Ok(ops::BottleneckWeights {
        conv1,
        bn1_scale,
        bn1_shift,
        conv2,
        bn2_scale,
        bn2_shift,
        conv3,
        bn3_scale,
        bn3_shift,
        shortcut_conv,
        shortcut_scale,
        shortcut_shift,
        planes,
        c_out,
    })
}

fn load_weights<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    config: ResNetConfig,
) -> Result<WeSpeakerBackboneWeights<'a>, WeSpeakerBackboneError> {
    let stem_c = config.stem_channels();
    let stem_conv = b.tensor_4d(arena, "conv1.weight", 3, 3, 1, stem_c)?;
    let (stem_bn_scale, stem_bn_shift) = b.batchnorm_affine(arena, "bn1", stem_c)?;
    let mut blocks = Vec::new();
    let mut in_channels = stem_c;
    for (stage, &n_blocks) in config.num_blocks.iter().enumerate() {
        let planes = config.stage_planes[stage];
        let out_channels = config.stage_out_channels(stage);
        for block_idx in 0..n_blocks {
            let stride = if block_idx == 0 {
                STAGE_STRIDES[stage]
            } else {
                1
            };
            let shortcut = config.shortcut_required(in_channels, stage, stride);
            let prefix = format!("layer{}.{}", stage + 1, block_idx);
            let block = match config.block_kind {
                BlockKind::Basic => BlockW::Basic(load_basic_block(
                    b,
                    arena,
                    &prefix,
                    in_channels,
                    out_channels,
                    stride,
                    shortcut,
                )?),
                BlockKind::Bottleneck => BlockW::Bottleneck(load_bottleneck_block(
                    b,
                    arena,
                    &prefix,
                    in_channels,
                    planes,
                    out_channels,
                    stride,
                    shortcut,
                )?),
            };
            blocks.push(block);
            in_channels = out_channels;
        }
    }
    let seg_w = b.tensor_2d(arena, "seg_1.weight", config.tstp_out(), EMBED_DIM)?;
    let seg_b = b.tensor_1d(arena, "seg_1.bias", EMBED_DIM)?;
    let eps_1e7 = b.scalar(arena, TSTP_EPS)?;
    Ok(WeSpeakerBackboneWeights {
        config,
        stem_conv,
        stem_bn_scale,
        stem_bn_shift,
        blocks,
        seg_w,
        seg_b,
        eps_1e7,
    })
}

fn expected_static_tensor_count(config: ResNetConfig) -> usize {
    let mut count = 0usize;
    // stem conv + stem BN scale/shift
    count += 1 + 2;
    let mut in_channels = config.stem_channels();
    for (stage, &n_blocks) in config.num_blocks.iter().enumerate() {
        let planes = config.stage_planes[stage];
        let out_channels = config.stage_out_channels(stage);
        for block_idx in 0..n_blocks {
            let stride = if block_idx == 0 {
                STAGE_STRIDES[stage]
            } else {
                1
            };
            let shortcut = config.shortcut_required(in_channels, stage, stride);
            match config.block_kind {
                BlockKind::Basic => {
                    count += 2; // conv1, conv2
                    count += 4; // two BN scale/shift pairs
                }
                BlockKind::Bottleneck => {
                    count += 3; // conv1/2/3
                    count += 6; // three BN pairs
                    let _ = planes;
                }
            }
            if shortcut {
                count += 1 + 2; // 1x1 conv + BN scale/shift
            }
            in_channels = out_channels;
        }
    }
    count += 2; // seg_1 weight/bias
    count += 1; // TSTP eps
    count
}

fn forward<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    spec: GgmlCpuTensor<'a>,
    frames: usize,
    w: &WeSpeakerBackboneWeights<'a>,
) -> Result<GgmlCpuTensor<'a>, WeSpeakerBackboneError> {
    let t_out = config::post_stride_time_len(frames);
    if t_out < 2 {
        return Err(shape_err(format!(
            "WeSpeaker post-stride time length is {t_out}; need at least 2 for unbiased TSTP"
        )));
    }
    let spec4d = graph.reshape_4d(spec, frames, N_MELS, 1, 1)?;
    let stem_c = w.config.stem_channels();
    let mut x = ops::conv_bn(
        graph,
        w.stem_conv,
        spec4d,
        stem_c,
        1,
        1,
        w.stem_bn_scale,
        w.stem_bn_shift,
        true,
    )?;
    let mut block_iter = w.blocks.iter();
    for (stage, &n_blocks) in w.config.num_blocks.iter().enumerate() {
        let out_c = w.config.stage_out_channels(stage);
        for block_idx in 0..n_blocks {
            let stride = if block_idx == 0 {
                STAGE_STRIDES[stage]
            } else {
                1
            };
            let block = block_iter
                .next()
                .ok_or_else(|| shape_err("block weight list shorter than topology"))?;
            x = match block {
                BlockW::Basic(weights) => ops::basic_block(graph, x, out_c, stride, weights)?,
                BlockW::Bottleneck(weights) => ops::bottleneck_block(graph, x, stride, weights)?,
            };
        }
    }
    let last_c = w.config.last_channels();
    let f_out = N_MELS / config::FREQ_STRIDE;
    let flat = ops::flatten_cft(graph, x, last_c, f_out, t_out)?;
    let pooled = ops::tstp(graph, flat, t_out, w.config.tstp_in(), w.eps_1e7)?;
    Ok(ops::linear_1d(
        graph,
        w.seg_w,
        w.seg_b,
        pooled,
        w.config.tstp_out(),
        EMBED_DIM,
    )?)
}

const WESPEAKER_GRAPH_NODE_CAPACITY: usize = 1usize << 12;

fn runner_config_with_threads(
    n_threads: Option<usize>,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
) -> GgmlCpuGraphConfig {
    let graph_size = WESPEAKER_GRAPH_NODE_CAPACITY;
    let mut config = GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend);
    config.context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(graph_size);
    config.graph_size = graph_size;
    if n_threads.is_some() {
        config.n_threads = n_threads;
    }
    crate::models::graph_runtime_config::apply_execution_placement(config, placement)
}

fn graph_context_bytes() -> usize {
    GgmlCpuGraphConfig::metadata_context_bytes(WESPEAKER_GRAPH_NODE_CAPACITY)
}

fn arena_context_bytes(static_tensors: usize) -> usize {
    GgmlCpuGraphConfig::metadata_context_bytes(static_tensors.next_power_of_two().max(256))
}

struct WeSpeakerResidentWeights {
    weights: WeSpeakerBackboneWeights<'static>,
    // Drop order: handles, then arena, then runner in the outer struct.
    #[allow(dead_code)]
    arena: GgmlStaticTensorArena,
}

pub(crate) struct WeSpeakerResidentRuntime {
    graph: Option<WeSpeakerPersistentGraph>,
    resident: WeSpeakerResidentWeights,
    runner: GgmlCpuGraphRunner,
    n_threads: Option<usize>,
    _parsed_weights: Arc<Weights>,
}

struct WeSpeakerPersistentGraph {
    session: GgmlPersistentGraphSession,
    input: GgmlCpuTensor<'static>,
    output: GgmlCpuTensor<'static>,
    frames: usize,
}

impl WeSpeakerResidentRuntime {
    pub(crate) fn new(
        weights: Arc<Weights>,
        config: ResNetConfig,
        n_threads: Option<usize>,
        backend: GgmlCpuGraphBackend,
        placement: crate::device::execution_policy::ExecutionPlacement,
    ) -> Result<Self, WeSpeakerBackboneError> {
        let expected = expected_static_tensor_count(config);
        let runner =
            GgmlCpuGraphRunner::new(runner_config_with_threads(n_threads, backend, placement))?;
        let arena = runner.start_static_tensor_arena(arena_context_bytes(expected))?;
        let mut builder = WBuilder::new(&weights);
        let loaded = load_weights(&mut builder, &arena, config)?;
        if builder.pending.len() != expected {
            return Err(shape_err(format!(
                "WeSpeaker static tensor topology declared {} handles, expected {expected}",
                builder.pending.len()
            )));
        }
        let mut arena = arena;
        builder.upload(&mut arena)?;
        let loaded = unsafe {
            std::mem::transmute::<WeSpeakerBackboneWeights<'_>, WeSpeakerBackboneWeights<'static>>(
                loaded,
            )
        };
        Ok(Self {
            graph: None,
            resident: WeSpeakerResidentWeights {
                weights: loaded,
                arena,
            },
            runner,
            n_threads,
            _parsed_weights: weights,
        })
    }

    pub(crate) fn forward(
        &mut self,
        feats: &[f32],
        frames: usize,
        n_threads: Option<usize>,
    ) -> Result<Vec<f32>, WeSpeakerBackboneError> {
        if config::post_stride_time_len(frames) < 2 {
            return Err(shape_err(format!(
                "WeSpeaker post-stride time length is {}; need at least 2 for unbiased TSTP",
                config::post_stride_time_len(frames)
            )));
        }
        if let Some(n_threads) = n_threads
            && self.n_threads != Some(n_threads)
        {
            self.runner.reconfigure_cpu_thread_count(n_threads)?;
            self.n_threads = Some(n_threads);
        }
        let must_rebuild = self
            .graph
            .as_ref()
            .is_none_or(|graph| graph.frames != frames || graph.session.is_poisoned());
        if must_rebuild {
            self.graph = None;
            let mut session = self
                .runner
                .start_persistent_graph_session(graph_context_bytes())?;
            let graph = session.builder();
            let input = graph.new_tensor_2d_f32(frames, N_MELS, "wespeaker_fbank_input")?;
            let output = forward(graph, input, frames, &self.resident.weights)?;
            graph.set_input(input)?;
            graph.set_output(output)?;
            graph.prepare_outputs_for_upload(&[output])?;
            self.graph = Some(WeSpeakerPersistentGraph {
                session,
                input,
                output,
                frames,
            });
        }
        let graph = self.graph.as_mut().expect("persistent graph built");
        let builder = graph.session.builder();
        builder.set_f32_slice(graph.input, feats, "wespeaker_fbank_input")?;
        Ok(builder.compute_output_f32(graph.output, EMBED_DIM)?)
    }
}

pub(crate) struct WeSpeakerResNetModel {
    weights: Arc<Weights>,
    config: ResNetConfig,
}

impl WeSpeakerResNetModel {
    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, WeSpeakerBackboneError> {
        let config = config::config_from_metadata(preflight.metadata()).map_err(shape_err)?;
        Ok(Self {
            weights: Arc::new(Weights::from_preflight(preflight)?),
            config,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, WeSpeakerBackboneError> {
        use crate::models::{
            aux_pack_registry::AuxPackKind,
            pack_verifier::{PackCandidate, PackRoute, PackVerifier},
        };
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(path))
            .map_err(|error| WeightsError::Gguf(error.to_string()))?;
        if !matches!(
            verified_pack.route(),
            PackRoute::Aux {
                kind: AuxPackKind::Diarization,
                ..
            }
        ) {
            return Err(WeightsError::Gguf(format!(
                "WeSpeaker pack route is not auxiliary diarization: {:?}",
                verified_pack.route()
            ))
            .into());
        }
        Self::from_preflight(verified_pack.preflight())
    }

    pub(crate) fn config(&self) -> ResNetConfig {
        self.config
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, WeSpeakerBackboneError> {
        self.weights
            .persistent_host_commitment_bytes()
            .map_err(WeSpeakerBackboneError::from)
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, WeSpeakerBackboneError> {
        Weights::quoted_persistent_host_commitment_bytes(tensor_index)
            .map_err(WeSpeakerBackboneError::from)
    }

    pub(crate) fn shared_weights(&self) -> Arc<Weights> {
        Arc::clone(&self.weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resnet34_static_tensor_count_is_stable() {
        assert_eq!(expected_static_tensor_count(config::RESNET34), 111);
    }
}
