#[cfg(test)]
use std::path::Path;

#[cfg(test)]
use crate::ggml_runtime::validate_ggml_runtime_source_path;
use crate::ggml_runtime::{
    AutoGpuPolicy, GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedWeightContext, GgmlPersistentGraphSession,
    GgmlRuntimeSource, GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReader,
    read_gguf_metadata_from_runtime_source,
};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};
use crate::nn::wav2vec2::{GroupedConv1dParams, grouped_conv_1d};

use super::config::{
    CONFORMER_DIM, CONFORMER_HEADS, CONFORMER_KERNEL, CONFORMER_LAYERS, CONV_CHANNELS,
    CONV_KERNELS, CONV_STRIDES, HEAD_DIM, HIDDEN_SIZE, LAYER_REPRESENTATIONS, POWERSET_CLASSES,
    RELATIVE_POSITION_BUCKETS, RELATIVE_POSITION_MAX_DISTANCE, REMAINING_HEADS, TOTAL_HEADS,
    TRANSFORMER_LAYERS, output_frames,
};
use super::weights::{read_tensor_f32, runtime_tensor_name, validate_tensor_contract};
use super::{DiariZenSegmenterError, DiariZenWindowOutput, postprocess_logits};

const GRAPH_SIZE: usize = 1 << 16;
const STATIC_GRAPH_SIZE: usize = 1 << 10;
const LAYER_NORM_EPSILON: f32 = 1.0e-5;
const BATCH_NORM_EPSILON: f32 = 1.0e-5;

const CONFORMER_TRACE_NAMES: [&str; 4] = [
    "conformer_layer_00",
    "conformer_layer_01",
    "conformer_layer_02",
    "conformer_layer_03",
];

fn graph_error(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> DiariZenSegmenterError + Copy {
    move |source| DiariZenSegmenterError::graph(step, source)
}

#[derive(Debug, Clone, Copy)]
struct BatchNormAffine {
    scale: GgmlStaticTensor,
    shift: GgmlStaticTensor,
}

#[derive(Clone, Copy)]
struct StaticHandles {
    relative_bias: GgmlStaticTensor,
    two: GgmlStaticTensor,
    batch_norm: [BatchNormAffine; CONFORMER_LAYERS],
}

#[derive(Clone)]
#[cfg_attr(not(test), allow(dead_code))]
struct TraceTensor {
    name: String,
    tensor: GgmlCpuTensor<'static>,
    len: usize,
}

/// Field order is a lifetime invariant: the persistent session owns graph
/// nodes pointing into both arenas and into the runner backend, so it must be
/// dropped before those owners.
pub(super) struct DiariZenRuntime {
    graph: Option<DiariZenPersistentGraph>,
    static_handles: StaticHandles,
    layer_sum: Vec<f32>,
    positional_conv_element_width: usize,
    trace: bool,
    samples: usize,
    frames: usize,
    _static_arena: GgmlStaticTensorArena,
    _loaded_weights: GgmlLoadedWeightContext,
    _runner: GgmlCpuGraphRunner,
}

struct DiariZenPersistentGraph {
    session: GgmlPersistentGraphSession,
    pcm: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
    #[cfg_attr(not(test), allow(dead_code))]
    traces: Vec<TraceTensor>,
}

impl DiariZenRuntime {
    #[cfg(test)]
    pub(super) fn new(
        path: &Path,
        samples: usize,
        trace: bool,
        backend: Option<GgmlCpuGraphBackend>,
    ) -> Result<Self, DiariZenSegmenterError> {
        let source = validate_ggml_runtime_source_path(path)
            .map_err(|error| DiariZenSegmenterError::PackSource(error.to_string()))?;
        Self::from_runtime_source(&source, samples, trace, backend)
    }

    /// Build from the same already-open mapping used to derive the cache key.
    /// This prevents an in-place path replacement from pairing the old key
    /// with weights re-opened from the new file (or vice versa).
    pub(super) fn from_runtime_source(
        source: &GgmlRuntimeSource,
        samples: usize,
        trace: bool,
        backend: Option<GgmlCpuGraphBackend>,
    ) -> Result<Self, DiariZenSegmenterError> {
        let metadata = read_gguf_metadata_from_runtime_source(source)
            .map_err(|error| DiariZenSegmenterError::PackRead(error.to_string()))?;
        super::config::validate_metadata(&metadata)?;
        let reader = GgufTensorDataReader::from_runtime_source(source)
            .map_err(|error| DiariZenSegmenterError::PackRead(error.to_string()))?;
        validate_tensor_contract(reader.tensor_index())?;

        let frames = output_frames(samples);
        if frames == 0 {
            return Err(DiariZenSegmenterError::WindowSize {
                expected: super::config::WINDOW_SAMPLES,
                actual: samples,
            });
        }

        let mut runner_config =
            GgmlCpuGraphConfig::gated_runtime_default(AutoGpuPolicy::AllBackends);
        runner_config.graph_size = GRAPH_SIZE;
        runner_config.context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE);
        runner_config.use_scheduler = backend.is_none();
        if let Some(backend) = backend {
            runner_config.backend = backend;
            runner_config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
                backend,
                crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
            );
        }

        let mut runner =
            GgmlCpuGraphRunner::new(runner_config).map_err(graph_error("create_runtime"))?;
        let loaded_weights = runner
            .load_gguf_weight_context(source)
            .map_err(graph_error("bind_resident_weights"))?;
        let mut static_arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
                STATIC_GRAPH_SIZE,
            ))
            .map_err(graph_error("create_static_arena"))?;
        let static_handles = build_static_handles(&reader, &static_arena, frames)?;
        upload_static_handles(&reader, &mut static_arena, static_handles, frames)?;

        let layer_sum = read_weight_f32(&reader, "weight_sum.weight")?;
        if layer_sum.len() != LAYER_REPRESENTATIONS {
            return Err(DiariZenSegmenterError::TensorShapeMismatch {
                name: "weight_sum.weight".to_string(),
                expected: vec![LAYER_REPRESENTATIONS as u64],
                actual: vec![layer_sum.len() as u64],
            });
        }
        let positional_conv_weight =
            runtime_tensor_name("wavlm_model.encoder.transformer.pos_conv_embed.conv.weight");
        let positional_conv_element_width = match reader
            .tensor_index()
            .get(&positional_conv_weight)
            .expect("the strict tensor contract already proved this tensor exists")
            .type_name
            .as_str()
        {
            "f32" => std::mem::size_of::<f32>(),
            "f16" => std::mem::size_of::<u16>(),
            tensor_type => {
                return Err(DiariZenSegmenterError::UnsupportedTensorType {
                    name: positional_conv_weight,
                    tensor_type: tensor_type.to_string(),
                });
            }
        };

        let graph = build_persistent_graph(
            &mut runner,
            &loaded_weights,
            &static_arena,
            static_handles,
            &layer_sum,
            positional_conv_element_width,
            samples,
            frames,
            trace,
        )?;

        Ok(Self {
            graph: Some(graph),
            static_handles,
            layer_sum,
            positional_conv_element_width,
            trace,
            samples,
            frames,
            _static_arena: static_arena,
            _loaded_weights: loaded_weights,
            _runner: runner,
        })
    }

    fn ensure_healthy_graph(&mut self) -> Result<(), DiariZenSegmenterError> {
        let must_rebuild = self
            .graph
            .as_ref()
            .is_none_or(|graph| graph.session.is_poisoned());
        if must_rebuild {
            self.graph = None;
            self.graph = Some(build_persistent_graph(
                &mut self._runner,
                &self._loaded_weights,
                &self._static_arena,
                self.static_handles,
                &self.layer_sum,
                self.positional_conv_element_width,
                self.samples,
                self.frames,
                self.trace,
            )?);
        }
        Ok(())
    }

    pub(super) fn infer(
        &mut self,
        samples: &[f32],
    ) -> Result<DiariZenWindowOutput, DiariZenSegmenterError> {
        if samples.len() != self.samples {
            return Err(DiariZenSegmenterError::WindowSize {
                expected: self.samples,
                actual: samples.len(),
            });
        }
        self.ensure_healthy_graph()?;
        let graph = self.graph.as_mut().expect("healthy graph ensured");
        graph
            .session
            .builder()
            .set_f32_slice(graph.pcm, samples, "diarizen_pcm")
            .map_err(graph_error("upload_pcm"))?;
        let logits = graph
            .session
            .builder()
            .compute_output_f32(graph.logits, self.frames * POWERSET_CLASSES)
            .map_err(graph_error("compute_logits"))?;
        let (powerset_class, activity) = postprocess_logits(&logits, self.frames);
        Ok(DiariZenWindowOutput {
            frame_count: self.frames,
            logits,
            powerset_class,
            activity,
        })
    }

    #[cfg(test)]
    pub(super) fn infer_trace(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<(String, Vec<f32>)>, DiariZenSegmenterError> {
        if samples.len() != self.samples {
            return Err(DiariZenSegmenterError::WindowSize {
                expected: self.samples,
                actual: samples.len(),
            });
        }
        self.ensure_healthy_graph()?;
        let graph = self.graph.as_mut().expect("healthy graph ensured");
        graph
            .session
            .builder()
            .set_f32_slice(graph.pcm, samples, "diarizen_pcm")
            .map_err(graph_error("upload_pcm_trace"))?;
        let mut specs = graph
            .traces
            .iter()
            .map(|tap| (tap.tensor, tap.len))
            .collect::<Vec<_>>();
        specs.push((graph.logits, self.frames * POWERSET_CLASSES));
        let values = graph
            .session
            .builder()
            .compute_outputs_f32(&specs)
            .map_err(graph_error("compute_trace"))?;
        let mut named = graph
            .traces
            .iter()
            .zip(values.iter())
            .map(|(tap, values)| (tap.name.clone(), values.clone()))
            .collect::<Vec<_>>();
        named.push((
            "logits".to_string(),
            values.last().cloned().unwrap_or_default(),
        ));
        Ok(named)
    }
}

fn build_persistent_graph(
    runner: &mut GgmlCpuGraphRunner,
    loaded_weights: &GgmlLoadedWeightContext,
    static_arena: &GgmlStaticTensorArena,
    static_handles: StaticHandles,
    layer_sum: &[f32],
    positional_conv_element_width: usize,
    samples: usize,
    frames: usize,
    trace: bool,
) -> Result<DiariZenPersistentGraph, DiariZenSegmenterError> {
    let mut session = runner
        .start_persistent_graph_session(GgmlCpuGraphConfig::metadata_context_bytes(GRAPH_SIZE))
        .map_err(graph_error("create_persistent_graph"))?;
    let (pcm, logits, traces) = build_graph(
        session.builder(),
        loaded_weights,
        static_arena,
        static_handles,
        layer_sum,
        positional_conv_element_width,
        samples,
        frames,
        trace,
    )?;

    let mut outputs = traces.iter().map(|tap| tap.tensor).collect::<Vec<_>>();
    outputs.push(logits);
    session
        .builder()
        .set_input(pcm)
        .map_err(graph_error("mark_pcm_input"))?;
    for output in &outputs {
        session
            .builder()
            .set_output(*output)
            .map_err(graph_error("mark_graph_output"))?;
    }
    session
        .builder()
        .prepare_outputs_for_upload(&outputs)
        .map_err(graph_error("prepare_persistent_graph"))?;

    Ok(DiariZenPersistentGraph {
        session,
        pcm,
        logits,
        traces,
    })
}

fn build_static_handles(
    _reader: &GgufTensorDataReader,
    arena: &GgmlStaticTensorArena,
    frames: usize,
) -> Result<StaticHandles, DiariZenSegmenterError> {
    let relative_bias = arena
        .new_tensor_3d_f32(frames, frames, TOTAL_HEADS, "diarizen_relative_bias")
        .map_err(graph_error("allocate_relative_bias"))?;
    let two = arena
        .new_tensor_1d_f32(1, "diarizen_two")
        .map_err(graph_error("allocate_two"))?;
    let mut batch_norm = Vec::with_capacity(CONFORMER_LAYERS);
    for _ in 0..CONFORMER_LAYERS {
        let scale = arena
            .new_tensor_1d_f32(CONFORMER_DIM, "diarizen_bn_scale")
            .map_err(graph_error("allocate_bn_scale"))?;
        let shift = arena
            .new_tensor_1d_f32(CONFORMER_DIM, "diarizen_bn_shift")
            .map_err(graph_error("allocate_bn_shift"))?;
        batch_norm.push(BatchNormAffine { scale, shift });
    }
    Ok(StaticHandles {
        relative_bias,
        two,
        batch_norm: batch_norm
            .try_into()
            .expect("CONFORMER_LAYERS controls the exact static BN count"),
    })
}

fn upload_static_handles(
    reader: &GgufTensorDataReader,
    arena: &mut GgmlStaticTensorArena,
    handles: StaticHandles,
    frames: usize,
) -> Result<(), DiariZenSegmenterError> {
    let rel = read_weight_f32(
        reader,
        "wavlm_model.encoder.transformer.layers.0.attention.rel_attn_embed.weight",
    )?;
    let mut bias = vec![0.0_f32; frames * frames * TOTAL_HEADS];
    for head in 0..TOTAL_HEADS {
        for query in 0..frames {
            for key in 0..frames {
                let bucket = relative_position_bucket(key as isize - query as isize);
                bias[key + query * frames + head * frames * frames] =
                    rel[head + bucket * TOTAL_HEADS];
            }
        }
    }
    arena
        .set_f32_slice(handles.relative_bias, &bias, "diarizen_relative_bias")
        .map_err(graph_error("upload_relative_bias"))?;
    arena
        .set_f32_slice(handles.two, &[2.0], "diarizen_two")
        .map_err(graph_error("upload_two"))?;

    for (layer, affine) in handles.batch_norm.iter().enumerate() {
        let prefix = format!("conformer.conformer_layer.{layer}.conv.bn_norm");
        let gamma = read_weight_f32(reader, &format!("{prefix}.weight"))?;
        let beta = read_weight_f32(reader, &format!("{prefix}.bias"))?;
        let mean = read_weight_f32(reader, &format!("{prefix}.running_mean"))?;
        let variance = read_weight_f32(reader, &format!("{prefix}.running_var"))?;
        let mut scale = vec![0.0_f32; CONFORMER_DIM];
        let mut shift = vec![0.0_f32; CONFORMER_DIM];
        for channel in 0..CONFORMER_DIM {
            scale[channel] = gamma[channel] / (variance[channel] + BATCH_NORM_EPSILON).sqrt();
            shift[channel] = beta[channel] - mean[channel] * scale[channel];
        }
        arena
            .set_f32_slice(affine.scale, &scale, "diarizen_bn_scale")
            .map_err(graph_error("upload_bn_scale"))?;
        arena
            .set_f32_slice(affine.shift, &shift, "diarizen_bn_shift")
            .map_err(graph_error("upload_bn_shift"))?;
    }
    Ok(())
}

fn relative_position_bucket(relative_position: isize) -> usize {
    let half = RELATIVE_POSITION_BUCKETS / 2;
    let sign = usize::from(relative_position > 0) * half;
    let distance = relative_position.unsigned_abs();
    let max_exact = half / 2;
    let bucket = if distance < max_exact {
        distance
    } else {
        let logarithmic = max_exact as f64
            + ((distance as f64 / max_exact as f64).ln()
                / (RELATIVE_POSITION_MAX_DISTANCE as f64 / max_exact as f64).ln()
                * (half - max_exact) as f64);
        (logarithmic as usize).min(half - 1)
    };
    sign + bucket
}

#[allow(clippy::too_many_arguments)]
fn build_graph(
    graph: &mut GgmlCpuGraphBuilder<'static>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    static_handles: StaticHandles,
    layer_sum: &[f32],
    positional_conv_element_width: usize,
    samples: usize,
    frames: usize,
    trace: bool,
) -> Result<
    (
        GgmlCpuTensor<'static>,
        GgmlCpuTensor<'static>,
        Vec<TraceTensor>,
    ),
    DiariZenSegmenterError,
> {
    let pcm = graph
        .new_tensor_2d_f32(samples, 1, "diarizen_pcm")
        .map_err(graph_error("allocate_pcm"))?;
    let mut traces = Vec::new();

    let mut state = graph
        .norm(pcm, LAYER_NORM_EPSILON)
        .map_err(graph_error("wavlm_waveform_norm"))?;
    let mut current_frames = samples;
    for layer in 0..CONV_CHANNELS.len() {
        let kernel = weight(
            weights,
            &format!("wavlm_model.feature_extractor.conv_layers.{layer}.conv.weight"),
        )?;
        state = graph
            .conv_1d(kernel, state, CONV_STRIDES[layer], 0, 1)
            .map_err(graph_error("wavlm_feature_conv"))?;
        current_frames = (current_frames - CONV_KERNELS[layer]) / CONV_STRIDES[layer] + 1;
        state = feature_layer_norm(
            graph,
            state,
            weight(
                weights,
                &format!("wavlm_model.feature_extractor.conv_layers.{layer}.layer_norm.weight"),
            )?,
            weight(
                weights,
                &format!("wavlm_model.feature_extractor.conv_layers.{layer}.layer_norm.bias"),
            )?,
        )?;
        state = graph
            .gelu_erf(state)
            .map_err(graph_error("wavlm_feature_gelu"))?;
    }
    debug_assert_eq!(current_frames, frames);
    state = graph
        .transpose(state)
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("wavlm_feature_transpose"))?;
    state = graph
        .mul(
            state,
            weight(weights, "wavlm_model.feature_extractor.dummy_weight")?,
        )
        .map_err(graph_error("wavlm_feature_mask"))?;
    if trace {
        traces.push(TraceTensor {
            name: "wavlm_feature_extractor".to_string(),
            tensor: state,
            len: frames * CONV_CHANNELS[6],
        });
    }

    state = affine_layer_norm(
        graph,
        state,
        weight(
            weights,
            "wavlm_model.encoder.feature_projection.layer_norm.weight",
        )?,
        weight(
            weights,
            "wavlm_model.encoder.feature_projection.layer_norm.bias",
        )?,
        "wavlm_feature_projection_norm",
    )?;
    state = linear(
        graph,
        state,
        weight(
            weights,
            "wavlm_model.encoder.feature_projection.projection.weight",
        )?,
        weight(
            weights,
            "wavlm_model.encoder.feature_projection.projection.bias",
        )?,
        "wavlm_feature_projection",
    )?;
    if trace {
        traces.push(TraceTensor {
            name: "wavlm_feature_projection".to_string(),
            tensor: state,
            len: frames * HIDDEN_SIZE,
        });
    }

    let positional =
        positional_convolution(graph, weights, state, frames, positional_conv_element_width)?;
    if trace {
        traces.push(TraceTensor {
            name: "wavlm_positional_conv".to_string(),
            tensor: positional,
            len: frames * HIDDEN_SIZE,
        });
    }
    state = graph
        .add(state, positional)
        .map_err(graph_error("wavlm_positional_add"))?;
    // DiariZen constructs the outer Transformer with
    // `layer_norm_first = !encoder_layer_norm_first`. This checkpoint uses
    // pre-norm encoder layers, so extract_features returns this positional sum
    // directly and never applies the outer transformer's final LayerNorm.
    if trace {
        traces.push(TraceTensor {
            name: "wavlm_transformer_preprocessed".to_string(),
            tensor: state,
            len: frames * HIDDEN_SIZE,
        });
    }
    let mut mixed = graph
        .cont(state)
        .and_then(|value| graph.scale(value, layer_sum[0]))
        .map_err(graph_error("wavlm_layer_sum_0"))?;

    for layer in 0..TRANSFORMER_LAYERS {
        state = wavlm_layer(graph, weights, arena, static_handles, state, frames, layer)?;
        if trace {
            traces.push(TraceTensor {
                name: format!("wavlm_layer_{layer:02}"),
                tensor: state,
                len: frames * HIDDEN_SIZE,
            });
        }
        let weighted = graph
            .cont(state)
            .and_then(|value| graph.scale(value, layer_sum[layer + 1]))
            .map_err(graph_error("wavlm_layer_sum"))?;
        mixed = graph
            .add(mixed, weighted)
            .map_err(graph_error("wavlm_layer_sum_add"))?;
    }
    if trace {
        traces.push(TraceTensor {
            name: "weighted_layer_sum_raw".to_string(),
            tensor: mixed,
            len: frames * HIDDEN_SIZE,
        });
    }

    let projection_raw = linear(
        graph,
        mixed,
        weight(weights, "proj.weight")?,
        weight(weights, "proj.bias")?,
        "diarizen_projection",
    )?;
    if trace {
        traces.push(TraceTensor {
            name: "projection_raw".to_string(),
            tensor: projection_raw,
            len: frames * CONFORMER_DIM,
        });
    }
    state = affine_layer_norm(
        graph,
        projection_raw,
        weight(weights, "lnorm.weight")?,
        weight(weights, "lnorm.bias")?,
        "diarizen_projection_norm",
    )?;
    if trace {
        traces.push(TraceTensor {
            name: "projection_norm".to_string(),
            tensor: state,
            len: frames * CONFORMER_DIM,
        });
    }

    for (layer, &trace_name) in CONFORMER_TRACE_NAMES.iter().enumerate() {
        state = conformer_layer(
            graph,
            weights,
            arena,
            static_handles.batch_norm[layer],
            state,
            frames,
            layer,
        )?;
        if trace {
            traces.push(TraceTensor {
                name: trace_name.to_string(),
                tensor: state,
                len: frames * CONFORMER_DIM,
            });
        }
    }

    let logits = linear(
        graph,
        state,
        weight(weights, "classifier.weight")?,
        weight(weights, "classifier.bias")?,
        "diarizen_classifier",
    )?;
    Ok((pcm, logits, traces))
}

fn weight<'a>(
    weights: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let runtime_name = runtime_tensor_name(name);
    weights
        .tensor(&runtime_name)
        .map(|tensor| tensor.as_graph_tensor())
        .ok_or(DiariZenSegmenterError::MissingTensor(runtime_name))
}

fn read_weight_f32(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<Vec<f32>, DiariZenSegmenterError> {
    read_tensor_f32(reader, &runtime_tensor_name(name))
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    matrix: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    graph
        .mul_mat(matrix, input)
        .and_then(|value| graph.add(value, bias))
        .map_err(graph_error(step))
}

fn affine_layer_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    apply_affine_layer_norm(
        graph,
        input,
        LAYER_NORM_EPSILON,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: step,
            scale: step,
            bias: step,
        },
        DiariZenSegmenterError::graph,
    )
}

fn feature_layer_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let feature_major = graph
        .transpose(input)
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("feature_layer_norm_transpose"))?;
    let normalized = graph
        .norm(feature_major, LAYER_NORM_EPSILON)
        .and_then(|value| graph.mul(value, weight))
        .and_then(|value| graph.add(value, bias))
        .map_err(graph_error("feature_layer_norm_affine"))?;
    let time_major = graph
        .transpose(normalized)
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("feature_layer_norm_restore"))?;
    Ok(time_major)
}

fn positional_convolution<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    hidden: GgmlCpuTensor<'a>,
    frames: usize,
    element_width: usize,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    const GROUPS: usize = 16;
    const KERNEL: usize = 128;
    let in_per_group = HIDDEN_SIZE / GROUPS;
    let out_per_group = HIDDEN_SIZE / GROUPS;
    let data = graph
        .transpose(hidden)
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("wavlm_pos_conv_input"))?;
    let kernel = weight(
        weights,
        "wavlm_model.encoder.transformer.pos_conv_embed.conv.weight",
    )?;
    let mut kernels = Vec::with_capacity(GROUPS);
    for group in 0..GROUPS {
        let view = graph
            .view_3d(
                kernel,
                KERNEL,
                in_per_group,
                out_per_group,
                KERNEL * element_width,
                KERNEL * in_per_group * element_width,
                group * out_per_group * KERNEL * in_per_group * element_width,
            )
            .and_then(|value| graph.cont(value))
            .map_err(graph_error("wavlm_pos_conv_kernel"))?;
        kernels.push(view);
    }
    let conv = grouped_conv_1d(
        graph,
        data,
        &kernels,
        &GroupedConv1dParams {
            groups: GROUPS,
            time: frames,
            in_per_group,
            out_per_group,
            stride: 1,
            padding: KERNEL / 2,
            dilation: 1,
        },
        "wavlm_pos_conv",
        DiariZenSegmenterError::graph,
    )?;
    let bias = weight(
        weights,
        "wavlm_model.encoder.transformer.pos_conv_embed.conv.bias",
    )?;
    let conv = graph
        .transpose(conv)
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.add(value, bias))
        .map_err(graph_error("wavlm_pos_conv_bias"))?;
    let cropped = graph
        .view_2d(
            conv,
            HIDDEN_SIZE,
            frames,
            HIDDEN_SIZE * std::mem::size_of::<f32>(),
            0,
        )
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.gelu_erf(value))
        .map_err(graph_error("wavlm_pos_conv_crop_gelu"))?;
    Ok(cropped)
}

#[allow(clippy::too_many_arguments)]
fn wavlm_layer<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    static_handles: StaticHandles,
    mut state: GgmlCpuTensor<'a>,
    frames: usize,
    layer: usize,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let prefix = format!("wavlm_model.encoder.transformer.layers.{layer}");
    if !REMAINING_HEADS[layer].is_empty() {
        let residual = state;
        let normalized = affine_layer_norm(
            graph,
            state,
            weight(weights, &format!("{prefix}.layer_norm.weight"))?,
            weight(weights, &format!("{prefix}.layer_norm.bias"))?,
            "wavlm_attention_pre_norm",
        )?;
        let attention = wavlm_attention(
            graph,
            weights,
            arena,
            static_handles,
            normalized,
            frames,
            layer,
        )?;
        state = graph
            .add(attention, residual)
            .map_err(graph_error("wavlm_attention_residual"))?;
    }
    let residual = state;
    let normalized = affine_layer_norm(
        graph,
        state,
        weight(weights, &format!("{prefix}.final_layer_norm.weight"))?,
        weight(weights, &format!("{prefix}.final_layer_norm.bias"))?,
        "wavlm_ffn_pre_norm",
    )?;
    let up = linear(
        graph,
        normalized,
        weight(
            weights,
            &format!("{prefix}.feed_forward.intermediate_dense.weight"),
        )?,
        weight(
            weights,
            &format!("{prefix}.feed_forward.intermediate_dense.bias"),
        )?,
        "wavlm_ffn_up",
    )?;
    let activated = graph.gelu_erf(up).map_err(graph_error("wavlm_ffn_gelu"))?;
    let down = linear(
        graph,
        activated,
        weight(
            weights,
            &format!("{prefix}.feed_forward.output_dense.weight"),
        )?,
        weight(weights, &format!("{prefix}.feed_forward.output_dense.bias"))?,
        "wavlm_ffn_down",
    )?;
    graph
        .add(residual, down)
        .map_err(graph_error("wavlm_ffn_residual"))
}

#[allow(clippy::too_many_arguments)]
fn wavlm_attention<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    static_handles: StaticHandles,
    state: GgmlCpuTensor<'a>,
    frames: usize,
    layer: usize,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let heads = REMAINING_HEADS[layer].len();
    let prefix = format!("wavlm_model.encoder.transformer.layers.{layer}.attention");
    let project = |name: &str| -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
        linear(
            graph,
            state,
            weight(weights, &format!("{prefix}.{name}.weight"))?,
            weight(weights, &format!("{prefix}.{name}.bias"))?,
            "wavlm_attention_projection",
        )
    };
    let layout = AttentionHeadLayout {
        head_dim: HEAD_DIM,
        attention_heads: heads,
        sequence_len: frames,
    };
    let reshape = |projection| {
        reshape_projection_to_attention_heads(
            graph,
            projection,
            layout,
            STANDARD_HEAD_PERMUTE_AXES,
            true,
            AttentionReshapeSteps {
                reshape: "wavlm_attention_reshape",
                permute: "wavlm_attention_permute",
                cont: "wavlm_attention_cont",
            },
            DiariZenSegmenterError::graph,
        )
    };
    let q = reshape(project("q_proj")?)?;
    let k = reshape(project("k_proj")?)?;
    let v = reshape(project("v_proj")?)?;
    let scores = graph
        .mul_mat(k, q)
        .map_err(graph_error("wavlm_attention_scores"))?;
    let mask = wavlm_relative_mask(graph, weights, arena, static_handles, state, frames, layer)?;
    let probs = graph
        .cont(scores)
        .and_then(|scores| {
            graph.soft_max_ext(scores, Some(mask), 1.0 / (HEAD_DIM as f32).sqrt(), 0.0)
        })
        .map_err(graph_error("wavlm_attention_softmax"))?;
    let context = attention_context_from_probs(
        graph,
        v,
        probs,
        layout,
        AttentionValueMergeSteps {
            value_permute: "wavlm_value_permute",
            value_cont: "wavlm_value_cont",
            context_mul: "wavlm_context_mul",
            context_merge_permute: "wavlm_context_merge_permute",
            context_merge_cont: "wavlm_context_merge_cont",
            context_merge_reshape: "wavlm_context_merge_reshape",
        },
        DiariZenSegmenterError::graph,
    )?;
    linear(
        graph,
        context,
        weight(weights, &format!("{prefix}.out_proj.weight"))?,
        weight(weights, &format!("{prefix}.out_proj.bias"))?,
        "wavlm_attention_output",
    )
}

#[allow(clippy::too_many_arguments)]
fn wavlm_relative_mask<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    static_handles: StaticHandles,
    state: GgmlCpuTensor<'a>,
    frames: usize,
    layer: usize,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let prefix = format!("wavlm_model.encoder.transformer.layers.{layer}.attention");
    let query_heads = graph
        .reshape_3d(state, HEAD_DIM, TOTAL_HEADS, frames)
        .and_then(|value| graph.permute(value, 0, 2, 1, 3))
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("wavlm_gate_query_reshape"))?;
    let gate = linear(
        graph,
        query_heads,
        weight(weights, &format!("{prefix}.gru_rel_pos_linear.weight"))?,
        weight(weights, &format!("{prefix}.gru_rel_pos_linear.bias"))?,
        "wavlm_gate_linear",
    )?;
    let gate = graph
        .reshape_4d(gate, 4, 2, frames, TOTAL_HEADS)
        .and_then(|value| graph.sum_rows(value))
        .and_then(|value| graph.sigmoid(value))
        .map_err(graph_error("wavlm_gate_reduce"))?;
    let gate_a = graph
        .view_3d(
            gate,
            1,
            frames,
            TOTAL_HEADS,
            2 * std::mem::size_of::<f32>(),
            2 * frames * std::mem::size_of::<f32>(),
            0,
        )
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("wavlm_gate_a"))?;
    let gate_b = graph
        .view_3d(
            gate,
            1,
            frames,
            TOTAL_HEADS,
            2 * std::mem::size_of::<f32>(),
            2 * frames * std::mem::size_of::<f32>(),
            std::mem::size_of::<f32>(),
        )
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("wavlm_gate_b"))?;
    let constant = graph
        .reshape_3d(
            weight(weights, &format!("{prefix}.gru_rel_pos_const"))?,
            1,
            1,
            TOTAL_HEADS,
        )
        .map_err(graph_error("wavlm_gate_const"))?;
    let factor = graph
        .mul(gate_b, constant)
        .and_then(|value| graph.mul(value, gate_a))
        .and_then(|value| graph.sub(value, gate_a))
        .and_then(|value| graph.add(value, arena.graph_tensor(static_handles.two)))
        .map_err(graph_error("wavlm_gate_factor"))?;
    let gated = graph
        .mul(arena.graph_tensor(static_handles.relative_bias), factor)
        .map_err(graph_error("wavlm_gate_relative_bias"))?;

    let plane_bytes = frames * frames * std::mem::size_of::<f32>();
    let mut selected = None;
    for &head in REMAINING_HEADS[layer] {
        let plane = graph
            .view_3d(
                gated,
                frames,
                frames,
                1,
                frames * std::mem::size_of::<f32>(),
                plane_bytes,
                head * plane_bytes,
            )
            .and_then(|value| graph.cont(value))
            .map_err(graph_error("wavlm_select_relative_head"))?;
        selected = Some(match selected {
            None => plane,
            Some(previous) => graph
                .concat(previous, plane, 2)
                .map_err(graph_error("wavlm_concat_relative_heads"))?,
        });
    }
    selected.ok_or(DiariZenSegmenterError::MissingTensor(prefix))
}

fn conformer_layer<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    batch_norm: BatchNormAffine,
    mut state: GgmlCpuTensor<'a>,
    frames: usize,
    layer: usize,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let prefix = format!("conformer.conformer_layer.{layer}");
    state = conformer_ffn(graph, weights, state, &format!("{prefix}.ffn1"))?;

    let residual = state;
    let normalized = affine_layer_norm(
        graph,
        state,
        weight(weights, &format!("{prefix}.mha.ln_norm.weight"))?,
        weight(weights, &format!("{prefix}.mha.ln_norm.bias"))?,
        "conformer_attention_norm",
    )?;
    let layout = AttentionHeadLayout {
        head_dim: CONFORMER_DIM / CONFORMER_HEADS,
        attention_heads: CONFORMER_HEADS,
        sequence_len: frames,
    };
    let project = |name: &str| {
        linear(
            graph,
            normalized,
            weight(weights, &format!("{prefix}.mha.mha.{name}.weight"))?,
            weight(weights, &format!("{prefix}.mha.mha.{name}.bias"))?,
            "conformer_attention_projection",
        )
    };
    let reshape = |projection| {
        reshape_projection_to_attention_heads(
            graph,
            projection,
            layout,
            STANDARD_HEAD_PERMUTE_AXES,
            true,
            AttentionReshapeSteps {
                reshape: "conformer_attention_reshape",
                permute: "conformer_attention_permute",
                cont: "conformer_attention_cont",
            },
            DiariZenSegmenterError::graph,
        )
    };
    let q = reshape(project("linearQ")?)?;
    let k = reshape(project("linearK")?)?;
    let v = reshape(project("linearV")?)?;
    let scores = graph
        .mul_mat(k, q)
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.soft_max_ext(value, None, 1.0 / 8.0, 0.0))
        .map_err(graph_error("conformer_attention_softmax"))?;
    let context = attention_context_from_probs(
        graph,
        v,
        scores,
        layout,
        AttentionValueMergeSteps {
            value_permute: "conformer_value_permute",
            value_cont: "conformer_value_cont",
            context_mul: "conformer_context_mul",
            context_merge_permute: "conformer_context_merge_permute",
            context_merge_cont: "conformer_context_merge_cont",
            context_merge_reshape: "conformer_context_merge_reshape",
        },
        DiariZenSegmenterError::graph,
    )?;
    let attention = linear(
        graph,
        context,
        weight(weights, &format!("{prefix}.mha.mha.linearO.weight"))?,
        weight(weights, &format!("{prefix}.mha.mha.linearO.bias"))?,
        "conformer_attention_output",
    )?;
    state = graph
        .add(residual, attention)
        .map_err(graph_error("conformer_attention_residual"))?;

    state = conformer_convolution(graph, weights, arena, batch_norm, state, frames, &prefix)?;
    state = conformer_ffn(graph, weights, state, &format!("{prefix}.ffn2"))?;
    affine_layer_norm(
        graph,
        state,
        weight(weights, &format!("{prefix}.ln_norm.weight"))?,
        weight(weights, &format!("{prefix}.ln_norm.bias"))?,
        "conformer_final_norm",
    )
}

fn conformer_ffn<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    state: GgmlCpuTensor<'a>,
    prefix: &str,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let normalized = affine_layer_norm(
        graph,
        state,
        weight(weights, &format!("{prefix}.ln_norm.weight"))?,
        weight(weights, &format!("{prefix}.ln_norm.bias"))?,
        "conformer_ffn_norm",
    )?;
    let up = linear(
        graph,
        normalized,
        weight(weights, &format!("{prefix}.w_1.weight"))?,
        weight(weights, &format!("{prefix}.w_1.bias"))?,
        "conformer_ffn_up",
    )?;
    let activated = graph.silu(up).map_err(graph_error("conformer_ffn_silu"))?;
    let down = linear(
        graph,
        activated,
        weight(weights, &format!("{prefix}.w_2.weight"))?,
        weight(weights, &format!("{prefix}.w_2.bias"))?,
        "conformer_ffn_down",
    )?;
    let down = graph
        .cont(down)
        .and_then(|value| graph.scale(value, 0.5))
        .map_err(graph_error("conformer_ffn_scale"))?;
    graph
        .add(state, down)
        .map_err(graph_error("conformer_ffn_residual"))
}

#[allow(clippy::too_many_arguments)]
fn conformer_convolution<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &GgmlLoadedWeightContext,
    arena: &GgmlStaticTensorArena,
    batch_norm: BatchNormAffine,
    state: GgmlCpuTensor<'a>,
    frames: usize,
    prefix: &str,
) -> Result<GgmlCpuTensor<'a>, DiariZenSegmenterError> {
    let normalized = affine_layer_norm(
        graph,
        state,
        weight(weights, &format!("{prefix}.conv.ln_norm.weight"))?,
        weight(weights, &format!("{prefix}.conv.ln_norm.bias"))?,
        "conformer_conv_norm",
    )?;
    let pointwise_weight = graph
        .reshape_2d(
            weight(weights, &format!("{prefix}.conv.pointwise_conv1.weight"))?,
            CONFORMER_DIM,
            2 * CONFORMER_DIM,
        )
        .map_err(graph_error("conformer_pw1_weight"))?;
    let pointwise = linear(
        graph,
        normalized,
        pointwise_weight,
        weight(weights, &format!("{prefix}.conv.pointwise_conv1.bias"))?,
        "conformer_pw1",
    )?;
    let row_bytes = 2 * CONFORMER_DIM * std::mem::size_of::<f32>();
    let value = graph
        .view_2d(pointwise, CONFORMER_DIM, frames, row_bytes, 0)
        .and_then(|value| graph.cont(value))
        .map_err(graph_error("conformer_glu_value"))?;
    let gate = graph
        .view_2d(
            pointwise,
            CONFORMER_DIM,
            frames,
            row_bytes,
            CONFORMER_DIM * std::mem::size_of::<f32>(),
        )
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.sigmoid(value))
        .map_err(graph_error("conformer_glu_gate"))?;
    let glu = graph
        .mul(value, gate)
        .map_err(graph_error("conformer_glu"))?;

    let kernel = graph
        .reshape_4d(
            weight(weights, &format!("{prefix}.conv.depthwise_conv.weight"))?,
            CONFORMER_KERNEL,
            1,
            1,
            CONFORMER_DIM,
        )
        .map_err(graph_error("conformer_depthwise_kernel"))?;
    let conv_input = graph
        .transpose(glu)
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.reshape_4d(value, frames, 1, CONFORMER_DIM, 1))
        .map_err(graph_error("conformer_depthwise_input"))?;
    let depthwise_bias = weight(weights, &format!("{prefix}.conv.depthwise_conv.bias"))?;
    let conv = graph
        .depthwise_conv_2d(
            kernel,
            conv_input,
            1,
            1,
            (CONFORMER_KERNEL - 1) / 2,
            0,
            1,
            1,
        )
        .and_then(|value| graph.permute(value, 1, 2, 0, 3))
        .and_then(|value| graph.cont(value))
        .and_then(|value| graph.add(value, depthwise_bias))
        .map_err(graph_error("conformer_depthwise"))?;
    let normalized = graph
        .mul(conv, arena.graph_tensor(batch_norm.scale))
        .and_then(|value| graph.add(value, arena.graph_tensor(batch_norm.shift)))
        .and_then(|value| graph.silu(value))
        .map_err(graph_error("conformer_batch_norm_silu"))?;
    let pointwise_weight = graph
        .reshape_2d(
            weight(weights, &format!("{prefix}.conv.pointwise_conv2.weight"))?,
            CONFORMER_DIM,
            CONFORMER_DIM,
        )
        .map_err(graph_error("conformer_pw2_weight"))?;
    let output = linear(
        graph,
        normalized,
        pointwise_weight,
        weight(weights, &format!("{prefix}.conv.pointwise_conv2.bias"))?,
        "conformer_pw2",
    )?;
    graph
        .add(state, output)
        .map_err(graph_error("conformer_conv_residual"))
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn relative_bucket_matches_wavlm_boundaries() {
        assert_eq!(relative_position_bucket(0), 0);
        assert_eq!(relative_position_bucket(-79), 79);
        assert_eq!(relative_position_bucket(-80), 80);
        assert_eq!(relative_position_bucket(1), 161);
        assert_eq!(relative_position_bucket(79), 239);
        assert_eq!(relative_position_bucket(80), 240);
        assert_eq!(relative_position_bucket(-800), 159);
        assert_eq!(relative_position_bucket(800), 319);
        assert_eq!(relative_position_bucket(8_000), 319);
    }
}
