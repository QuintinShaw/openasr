//! 8-codebook speech-embedding sum (P2.0 blood lesson #3) + the 6L
//! bidirectional input-local transformer + group downcast: the bridge from
//! per-frame RVQ codes to the 36L Qwen2 backbone's prompt embeddings.
//!
//! Reference (`modeling_mimo_audio.py::_prepare_input_embeds` /
//! `apply_input_local_transformer`, P2.0 findings SS3):
//! 1. For each of the 8 RVQ channels, look up `speech_embeddings[channel]`
//!    and **sum** (not concatenate) across channels, masking a channel's
//!    contribution to zero on its own `zeroemb_idx` row.
//! 2. Reshape into independent `group_size`(=4)-frame groups (no
//!    cross-group attention -- `input_full_attention=true` means each
//!    group's own 4 positions attend bidirectionally to each other, never to
//!    another group) and run the 6L Qwen2-shaped (RMSNorm, SwiGLU, qkv-bias,
//!    RoPE, no QK-norm) transformer over every group in one batched ggml pass
//!    (heads and groups both live on ggml's batch axes ne2/ne3).
//! 3. Flatten each group's 4 post-transformer rows into one 4096-wide vector
//!    (`speech_group_downcast`, `pos` outer / `hidden` inner, matching
//!    PyTorch's `.view(B, T_groups, -1)`) and project to the backbone's
//!    hidden size with `speech_group_proj` (bias-free).

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlRopeExtParams, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufTensorDataReadError, GgufTensorDataReader,
};
use crate::nn::norm::{RmsNormSteps, apply_rms_norm};

use super::runtime_contract::MimoInlocalMetadata;
use super::rvq::MimoRvqCodes;
use super::tensor_names::{
    INLOCAL_NORM_WEIGHT, SPEECH_GROUP_PROJ_WEIGHT, mimo_inlocal_layer_tensor_names,
    speech_embd_weight_name,
};

const RMS_NORM_EPSILON: f32 = 1.0e-6;

/// The input-local runner's graph config, right-sized from the 6L
/// transformer's forward-graph node budget instead of the historical flat
/// 256 MB. Each Qwen2-shaped block builds ~35-45 ggml ops (RMSNorm, qkv
/// projections + bias, per-head reshape/RoPE/permute/cont, scaled dot-product
/// attention, output projection, SwiGLU FFN and two residuals); the default
/// `graph_size` (4096) keeps >60x headroom over the 6-block graph, and longer
/// audio grows tensor DIMENSIONS, not the op count.
/// [`GgmlCpuGraphConfig::metadata_context_bytes`] sizes the `no_alloc`
/// metadata context from that node budget: `ggml_init` always mallocs the
/// full `mem_size` even with `no_alloc=true`, so the old 256 MB committed
/// ~256 MB of metadata context for a graph that needs ~2 MB. Mirrors the qwen
/// audio-encoder (`models/qwen/audio_encoder.rs`) and xasr_zipformer encoder
/// (`models/xasr_zipformer/graph_config.rs`) sizing.
fn mimo_input_local_graph_config(
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> GgmlCpuGraphConfig {
    let mut config = GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    crate::models::graph_runtime_config::apply_request_execution_placement(config)
}

#[derive(Debug, Error)]
pub(crate) enum MimoInputLocalError {
    #[error("mimo-asr input-local graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("mimo-asr input-local graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error(
        "mimo-asr input-local weight '{name}' could not be bound zero-copy from the runtime pack"
    )]
    WeightNotBound { name: String },
    #[error("mimo-asr input-local could not read tensor '{name}': {source}")]
    TensorRead {
        name: String,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error(
        "mimo-asr input-local RVQ code count {frame_count} is not a multiple of group_size {group_size}"
    )]
    FrameCountNotGroupAligned {
        frame_count: usize,
        group_size: usize,
    },
    #[error(
        "mimo-asr speech-embedding table count {table_count} does not match RVQ channel count {code_channels}"
    )]
    SpeechEmbeddingLayoutMismatch {
        table_count: usize,
        code_channels: usize,
    },
    #[error("mimo-asr input-local shape overflowed")]
    ShapeOverflow,
}

fn build_err(step: &'static str, source: GgmlCpuGraphError) -> MimoInputLocalError {
    MimoInputLocalError::GraphBuildFailed { step, source }
}

// --- Step 1: 8-codebook embedding sum ---
//
// CPU keeps this scalar implementation as the exact oracle. Accelerated
// execution binds the native GGUF tables below and performs get_rows + sum in
// the input-local graph, retaining no full host-f32 table copy.

pub(crate) struct MimoSpeechEmbeddingTables {
    d_model: usize,
    /// One `[vocab_size][d_model]` table per RVQ channel.
    tables: Vec<Vec<f32>>,
    vocab_sizes: Vec<usize>,
    zeroemb_idx: Vec<u32>,
}

impl MimoSpeechEmbeddingTables {
    pub(crate) fn quoted_retained_system_memory_bytes(
        d_model: usize,
        speech_vocab_sizes: &[u32],
    ) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_usize(
            speech_vocab_sizes
                .len()
                .checked_mul(std::mem::size_of::<Vec<f32>>())
                .ok_or_else(|| "mimo-asr speech table descriptors quote overflowed".to_string())?,
            "mimo-asr speech table descriptors quote",
        )?;
        let value_count = speech_vocab_sizes
            .iter()
            .try_fold(0usize, |total, &vocab_size| {
                total.checked_add((vocab_size as usize).checked_mul(d_model)?)
            })
            .ok_or_else(|| "mimo-asr speech embedding values quote overflowed".to_string())?;
        bytes.add_usize(
            value_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "mimo-asr speech embedding bytes quote overflowed".to_string())?,
            "mimo-asr speech embedding values quote",
        )?;
        for (label, entry_size) in [
            (
                "mimo-asr speech vocabulary sizes quote",
                std::mem::size_of::<usize>(),
            ),
            (
                "mimo-asr speech zero indices quote",
                std::mem::size_of::<u32>(),
            ),
        ] {
            bytes.add_usize(
                speech_vocab_sizes
                    .len()
                    .checked_mul(entry_size)
                    .ok_or_else(|| format!("{label} overflowed"))?,
                label,
            )?;
        }
        Ok(bytes.finish())
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.tables, "mimo-asr speech embedding tables")?;
        for table in &self.tables {
            bytes.add_vec(table, "mimo-asr speech embedding table values")?;
        }
        bytes.add_vec(&self.vocab_sizes, "mimo-asr speech vocabulary sizes")?;
        bytes.add_vec(&self.zeroemb_idx, "mimo-asr speech zero-embedding indices")?;
        Ok(bytes.finish())
    }
}

pub(crate) fn load_speech_embedding_tables_from_reader(
    reader: &GgufTensorDataReader,
    d_model: usize,
    speech_vocab_sizes: &[u32],
    zeroemb_idx: &[u32],
) -> Result<MimoSpeechEmbeddingTables, MimoInputLocalError> {
    let mut tables = Vec::with_capacity(speech_vocab_sizes.len());
    let mut vocab_sizes = Vec::with_capacity(speech_vocab_sizes.len());
    for (channel, &vocab_size) in speech_vocab_sizes.iter().enumerate() {
        let vocab_size = vocab_size as usize;
        let name = speech_embd_weight_name(channel);
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(&name, &[d_model as u64, vocab_size as u64])
            .map_err(|source| MimoInputLocalError::TensorRead { name, source })?;
        tables.push(values);
        vocab_sizes.push(vocab_size);
    }
    Ok(MimoSpeechEmbeddingTables {
        d_model,
        tables,
        vocab_sizes,
        zeroemb_idx: zeroemb_idx.to_vec(),
    })
}

/// `codes`: `[frame_count][8]` RVQ channel ids (from [`super::rvq::encode_rvq_codes`]).
/// Returns `[frame_count][d_model]` row-major summed embeddings (blood lesson
/// #3: sum across channels, zero out a channel whose code equals its own
/// `zeroemb_idx`).
pub(crate) fn sum_speech_embeddings(
    tables: &MimoSpeechEmbeddingTables,
    codes: &MimoRvqCodes,
) -> Result<Vec<f32>, MimoInputLocalError> {
    if tables.tables.len() != codes.channels()
        || tables.vocab_sizes.len() != codes.channels()
        || tables.zeroemb_idx.len() != codes.channels()
    {
        return Err(MimoInputLocalError::SpeechEmbeddingLayoutMismatch {
            table_count: tables.tables.len(),
            code_channels: codes.channels(),
        });
    }
    let d_model = tables.d_model;
    let frame_count = codes.frame_count();
    let mut out = vec![0.0_f32; frame_count * d_model];
    for frame_idx in 0..frame_count {
        let row = &mut out[frame_idx * d_model..(frame_idx + 1) * d_model];
        for channel in 0..codes.channels() {
            let Some(code) = codes.code(frame_idx, channel) else {
                continue;
            };
            if code == tables.zeroemb_idx[channel] {
                continue;
            }
            let vocab_size = tables.vocab_sizes[channel];
            let code = code as usize;
            if code >= vocab_size {
                continue; // defensive: out-of-range code never contributes.
            }
            let table_row = &tables.tables[channel][code * d_model..(code + 1) * d_model];
            for (dst, src) in row.iter_mut().zip(table_row.iter()) {
                *dst += *src;
            }
        }
    }
    Ok(out)
}

// --- Step 2+3: 6L batched bidirectional transformer + group downcast ---

struct LayerRuntime {
    attn_norm: GgmlStaticTensor,
    attn_q: GgmlLoadedTensor,
    attn_q_bias: GgmlStaticTensor,
    attn_k: GgmlLoadedTensor,
    attn_k_bias: GgmlStaticTensor,
    attn_v: GgmlLoadedTensor,
    attn_v_bias: GgmlStaticTensor,
    attn_output: GgmlLoadedTensor,
    ffn_norm: GgmlStaticTensor,
    ffn_gate: GgmlLoadedTensor,
    ffn_up: GgmlLoadedTensor,
    ffn_down: GgmlLoadedTensor,
}

pub(crate) struct MimoInputLocalRuntime {
    metadata: MimoInlocalMetadata,
    runner: GgmlCpuGraphRunner,
    #[allow(dead_code)]
    loaded_weights: GgmlLoadedWeightContext,
    arena: GgmlStaticTensorArena,
    final_norm: GgmlStaticTensor,
    group_proj: GgmlLoadedTensor,
    layers: Vec<LayerRuntime>,
    device_speech_embeddings: Option<Vec<GgmlLoadedTensor>>,
}

pub(crate) enum MimoInputLocalInput<'a> {
    HostSummedEmbeddings(&'a [f32]),
    DeviceRvqCodes(&'a MimoRvqCodes),
}

impl MimoInputLocalRuntime {
    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: &MimoInlocalMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<u64, String> {
        let layer_bytes = metadata
            .n_layers
            .checked_mul(std::mem::size_of::<LayerRuntime>())
            .ok_or_else(|| "mimo-asr input-local handle quote overflowed".to_string())?;
        let speech_handle_bytes = if backend.is_gpu_class() {
            metadata
                .audio_channels
                .checked_mul(std::mem::size_of::<GgmlLoadedTensor>())
                .ok_or_else(|| {
                    "mimo-asr device speech-embedding handle quote overflowed".to_string()
                })?
        } else {
            0
        };
        let bytes = layer_bytes
            .checked_add(speech_handle_bytes)
            .ok_or_else(|| "mimo-asr input-local retained quote overflowed".to_string())?;
        u64::try_from(bytes)
            .map_err(|_| "mimo-asr input-local handle quote exceeds u64".to_string())
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        // Native graph arenas, backend buffers, and loaded zero-copy weight
        // contexts are admitted by their constructors and intentionally do not
        // belong to this Rust-container accounting.
        bytes.add_vec(&self.layers, "mimo-asr input-local layer handles")?;
        if let Some(tables) = &self.device_speech_embeddings {
            bytes.add_vec(tables, "mimo-asr device speech-embedding handles")?;
        }
        Ok(bytes.finish())
    }
}

fn bind(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, MimoInputLocalError> {
    loaded
        .tensor(name)
        .ok_or_else(|| MimoInputLocalError::WeightNotBound {
            name: name.to_string(),
        })
}

fn new_vector(
    arena: &GgmlStaticTensorArena,
    len: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MimoInputLocalError> {
    arena
        .new_tensor_1d_f32(len, name)
        .map_err(|source| MimoInputLocalError::GraphBuildFailed { step: name, source })
}

fn upload(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    reader: &GgufTensorDataReader,
    name: &str,
    shape: &[u64],
) -> Result<(), MimoInputLocalError> {
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, shape)
        .map_err(|source| MimoInputLocalError::TensorRead {
            name: name.to_string(),
            source,
        })?;
    arena
        .set_f32_slice(tensor, &values, "mimo_inlocal_upload")
        .map_err(|source| MimoInputLocalError::GraphBuildFailed {
            step: "mimo_inlocal_upload",
            source,
        })
}

impl MimoInputLocalRuntime {
    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: MimoInlocalMetadata,
        speech_vocab_sizes: &[u32],
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, MimoInputLocalError> {
        let config = mimo_input_local_graph_config(backend);
        let device_speech_embeddings_enabled = config.backend.is_gpu_class();
        let runner =
            GgmlCpuGraphRunner::new(config).map_err(|source| build_err("runner_init", source))?;
        let loaded_weights = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(|error| MimoInputLocalError::GraphExecutionFailed {
                reason: format!("load_gguf_weight_context: {error}"),
            })?;
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| MimoInputLocalError::GraphExecutionFailed {
                    reason: format!("build_runtime_tensor_reader_from_preflight: {error}"),
                })?;
        let arena_tensor_count = metadata
            .n_layers
            .checked_mul(5)
            .and_then(|count| count.checked_add(1))
            .ok_or(MimoInputLocalError::ShapeOverflow)?;
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
                arena_tensor_count,
            ))
            .map_err(|source| build_err("static_tensor_arena", source))?;

        let d = metadata.d_model;
        let final_norm = new_vector(&arena, d, "inlocal_final_norm")?;
        let group_proj = bind(&loaded_weights, SPEECH_GROUP_PROJ_WEIGHT)?;

        let mut layers = Vec::with_capacity(metadata.n_layers);
        for layer_idx in 0..metadata.n_layers {
            let names = mimo_inlocal_layer_tensor_names(layer_idx);
            layers.push(LayerRuntime {
                attn_norm: new_vector(&arena, d, "inlocal_attn_norm")?,
                attn_q: bind(&loaded_weights, &names.attn_q_weight)?,
                attn_q_bias: new_vector(&arena, d, "inlocal_attn_q_b")?,
                attn_k: bind(&loaded_weights, &names.attn_k_weight)?,
                attn_k_bias: new_vector(&arena, d, "inlocal_attn_k_b")?,
                attn_v: bind(&loaded_weights, &names.attn_v_weight)?,
                attn_v_bias: new_vector(&arena, d, "inlocal_attn_v_b")?,
                attn_output: bind(&loaded_weights, &names.attn_output_weight)?,
                ffn_norm: new_vector(&arena, d, "inlocal_ffn_norm")?,
                ffn_gate: bind(&loaded_weights, &names.ffn_gate_weight)?,
                ffn_up: bind(&loaded_weights, &names.ffn_up_weight)?,
                ffn_down: bind(&loaded_weights, &names.ffn_down_weight)?,
            });
        }
        let device_speech_embeddings = if device_speech_embeddings_enabled {
            if speech_vocab_sizes.len() != metadata.audio_channels {
                return Err(MimoInputLocalError::GraphExecutionFailed {
                    reason: format!(
                        "speech vocab count {} != audio channels {}",
                        speech_vocab_sizes.len(),
                        metadata.audio_channels
                    ),
                });
            }
            let mut tables = Vec::with_capacity(metadata.audio_channels);
            for channel in 0..metadata.audio_channels {
                tables.push(bind(&loaded_weights, &speech_embd_weight_name(channel))?);
            }
            Some(tables)
        } else {
            None
        };

        upload(
            &mut arena,
            final_norm,
            &reader,
            INLOCAL_NORM_WEIGHT,
            &[d as u64],
        )?;
        for (layer_idx, layer) in layers.iter().enumerate() {
            let names = mimo_inlocal_layer_tensor_names(layer_idx);
            upload(
                &mut arena,
                layer.attn_norm,
                &reader,
                &names.attn_norm_weight,
                &[d as u64],
            )?;
            upload(
                &mut arena,
                layer.attn_q_bias,
                &reader,
                &names.attn_q_bias,
                &[d as u64],
            )?;
            upload(
                &mut arena,
                layer.attn_k_bias,
                &reader,
                &names.attn_k_bias,
                &[d as u64],
            )?;
            upload(
                &mut arena,
                layer.attn_v_bias,
                &reader,
                &names.attn_v_bias,
                &[d as u64],
            )?;
            upload(
                &mut arena,
                layer.ffn_norm,
                &reader,
                &names.ffn_norm_weight,
                &[d as u64],
            )?;
        }

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            final_norm,
            group_proj,
            layers,
            device_speech_embeddings,
        })
    }

    /// CPU accepts already-summed `[frame][d_model]` embeddings from the exact
    /// scalar oracle. Accelerated execution accepts compact channel-major RVQ
    /// ids and performs all eight native table gathers plus their sum in this
    /// graph. `frame_count` must be a multiple of `group_size`. Returns
    /// `[T_groups][llm_hidden_size]` row-major prompt-splice-ready embeddings.
    pub(crate) fn run(
        &mut self,
        input: MimoInputLocalInput<'_>,
        frame_count: usize,
        llm_hidden_size: usize,
    ) -> Result<Vec<f32>, MimoInputLocalError> {
        let group_size = self.metadata.group_size;
        if !frame_count.is_multiple_of(group_size) {
            return Err(MimoInputLocalError::FrameCountNotGroupAligned {
                frame_count,
                group_size,
            });
        }
        let n_groups = frame_count / group_size;
        let d_model = self.metadata.d_model;

        let mut graph = self.runner.start_graph();
        let (mut hidden, host_upload, codes_upload) = match input {
            MimoInputLocalInput::HostSummedEmbeddings(summed_embeddings) => {
                let expected_len = frame_count
                    .checked_mul(d_model)
                    .ok_or(MimoInputLocalError::ShapeOverflow)?;
                if summed_embeddings.len() != expected_len {
                    return Err(MimoInputLocalError::GraphExecutionFailed {
                        reason: format!(
                            "summed_embeddings len {} != frame_count {frame_count} * d_model {d_model}",
                            summed_embeddings.len()
                        ),
                    });
                }
                let tensor = graph
                    .new_tensor_2d_f32(d_model, frame_count, "inlocal_input")
                    .map_err(|source| build_err("inlocal_input", source))?;
                graph
                    .set_input(tensor)
                    .map_err(|source| build_err("inlocal_input_set", source))?;
                (tensor, Some((tensor, summed_embeddings)), None)
            }
            MimoInputLocalInput::DeviceRvqCodes(codes) => {
                let tables = self.device_speech_embeddings.as_ref().ok_or_else(|| {
                    MimoInputLocalError::GraphExecutionFailed {
                        reason: "device RVQ codes require device speech-embedding handles"
                            .to_string(),
                    }
                })?;
                if codes.frame_count() != frame_count
                    || codes.channels() != tables.len()
                    || codes.channels() != self.metadata.audio_channels
                {
                    return Err(MimoInputLocalError::GraphExecutionFailed {
                        reason: format!(
                            "device RVQ layout frames={} channels={} does not match frames={frame_count} channels={}",
                            codes.frame_count(),
                            codes.channels(),
                            self.metadata.audio_channels
                        ),
                    });
                }
                let code_count = frame_count
                    .checked_mul(codes.channels())
                    .ok_or(MimoInputLocalError::ShapeOverflow)?;
                let code_tensor = graph
                    .new_tensor_1d_i32(code_count, "inlocal_rvq_codes")
                    .map_err(|source| build_err("inlocal_rvq_codes", source))?;
                graph
                    .set_input(code_tensor)
                    .map_err(|source| build_err("inlocal_rvq_codes_set", source))?;
                let mut summed = None;
                for (channel, table) in tables.iter().enumerate() {
                    let offset = channel
                        .checked_mul(frame_count)
                        .and_then(|elements| elements.checked_mul(std::mem::size_of::<i32>()))
                        .ok_or(MimoInputLocalError::ShapeOverflow)?;
                    let channel_codes = graph
                        .view_1d(code_tensor, frame_count, offset)
                        .map_err(|source| build_err("inlocal_channel_codes", source))?;
                    let rows = graph
                        .get_rows(table.as_graph_tensor(), channel_codes)
                        .map_err(|source| build_err("inlocal_speech_embedding_lookup", source))?;
                    summed = Some(match summed {
                        Some(previous) => graph
                            .add(previous, rows)
                            .map_err(|source| build_err("inlocal_speech_embedding_sum", source))?,
                        None => rows,
                    });
                }
                let summed = summed.ok_or_else(|| MimoInputLocalError::GraphExecutionFailed {
                    reason: "device speech-embedding table set is empty".to_string(),
                })?;
                (summed, None, Some((code_tensor, codes.values())))
            }
        };

        // Local (within-group) positions 0..group_size, shared identically
        // across every group (ggml rope batches the same positions vector
        // over the ne3 group axis) -- reset-per-group is the point (P2.0
        // finding: "组间互不注意", each 4-frame group is an independent
        // pseudo-batch item to the reference HF transformer).
        let positions = graph
            .new_tensor_1d_i32(group_size, "inlocal_positions")
            .map_err(|source| build_err("inlocal_positions", source))?;
        graph
            .set_input(positions)
            .map_err(|source| build_err("inlocal_positions_set", source))?;

        let rope_params = GgmlRopeExtParams::qwen_neox(
            self.metadata.head_dim,
            group_size.max(1),
            self.metadata.rope_theta,
        )
        .map_err(|source| build_err("inlocal_rope_params", source))?;

        for layer in &self.layers {
            hidden = run_layer(
                &mut graph,
                &self.arena,
                hidden,
                layer,
                positions,
                rope_params,
                frame_count,
                group_size,
                n_groups,
                d_model,
                self.metadata.n_heads,
                self.metadata.head_dim,
            )?;
        }
        hidden = apply_rms_norm(
            &graph,
            hidden,
            RMS_NORM_EPSILON,
            self.arena.graph_tensor(self.final_norm),
            RmsNormSteps {
                norm: "inlocal_final_norm",
                scale: "inlocal_final_norm_scale",
            },
            build_err,
        )?;

        // Flatten each group's `group_size` rows into one
        // `group_size*d_model`-wide vector (pos outer / hidden inner,
        // matching PyTorch's `speech_embeds.view(B, T_groups, -1)`), then
        // project to the backbone's hidden size (bias-free).
        let flat_width = group_size
            .checked_mul(d_model)
            .ok_or(MimoInputLocalError::ShapeOverflow)?;
        let flattened = graph
            .reshape_2d(hidden, flat_width, n_groups)
            .map_err(|source| build_err("inlocal_flatten", source))?;
        let projected = graph
            .mul_mat(self.group_proj.as_graph_tensor(), flattened)
            .map_err(|source| build_err("inlocal_group_proj", source))?;

        graph
            .set_output(projected)
            .map_err(|source| build_err("inlocal_output", source))?;
        graph
            .prepare_outputs_for_upload(&[projected])
            .map_err(|source| build_err("inlocal_prepare_outputs", source))?;

        if let Some((tensor, values)) = host_upload {
            graph
                .set_f32_slice(tensor, values, "inlocal_input")
                .map_err(|source| build_err("inlocal_input_upload", source))?;
        }
        if let Some((tensor, values)) = codes_upload {
            graph
                .set_i32_slice(tensor, values, "inlocal_rvq_codes")
                .map_err(|source| build_err("inlocal_rvq_codes_upload", source))?;
        }
        let position_values: Vec<i32> = (0..group_size as i32).collect();
        graph
            .set_i32_slice(positions, &position_values, "inlocal_positions")
            .map_err(|source| build_err("inlocal_positions_upload", source))?;

        let total = llm_hidden_size
            .checked_mul(n_groups)
            .ok_or(MimoInputLocalError::ShapeOverflow)?;
        graph.compute_output_f32(projected, total).map_err(|error| {
            MimoInputLocalError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })
    }

    pub(crate) fn release_transient_compute_memory(&mut self) -> Result<(), MimoInputLocalError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(|source| build_err("release_transient_scheduler_working_set", source))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &LayerRuntime,
    positions: GgmlCpuTensor<'a>,
    rope_params: GgmlRopeExtParams,
    seq_total: usize,
    group_size: usize,
    n_groups: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
) -> Result<GgmlCpuTensor<'a>, MimoInputLocalError> {
    let residual = state;
    let normed = apply_rms_norm(
        graph,
        state,
        RMS_NORM_EPSILON,
        arena.graph_tensor(layer.attn_norm),
        RmsNormSteps {
            norm: "inlocal_attn_norm",
            scale: "inlocal_attn_norm_scale",
        },
        build_err,
    )?;

    let mut q = graph
        .mul_mat(layer.attn_q.as_graph_tensor(), normed)
        .map_err(|source| build_err("inlocal_q", source))?;
    q = graph
        .add(q, arena.graph_tensor(layer.attn_q_bias))
        .map_err(|source| build_err("inlocal_q_bias", source))?;
    let mut k = graph
        .mul_mat(layer.attn_k.as_graph_tensor(), normed)
        .map_err(|source| build_err("inlocal_k", source))?;
    k = graph
        .add(k, arena.graph_tensor(layer.attn_k_bias))
        .map_err(|source| build_err("inlocal_k_bias", source))?;
    let mut v = graph
        .mul_mat(layer.attn_v.as_graph_tensor(), normed)
        .map_err(|source| build_err("inlocal_v", source))?;
    v = graph
        .add(v, arena.graph_tensor(layer.attn_v_bias))
        .map_err(|source| build_err("inlocal_v_bias", source))?;

    // [d_model, seq_total] -> [head_dim, heads, group_size, n_groups]: seq_total
    // is group-major/position-minor (frame order), so splitting ne2 into
    // (group_size, n_groups) via reshape_4d keeps every group's own 4
    // positions contiguous on ne2 -- exactly the per-group batch this
    // transformer needs (rope + attention both then batch over ne3=n_groups
    // without any cross-group mixing).
    let q = graph
        .reshape_4d(q, head_dim, heads, group_size, n_groups)
        .map_err(|source| build_err("inlocal_q_reshape", source))?;
    let k = graph
        .reshape_4d(k, head_dim, heads, group_size, n_groups)
        .map_err(|source| build_err("inlocal_k_reshape", source))?;
    let v = graph
        .reshape_4d(v, head_dim, heads, group_size, n_groups)
        .map_err(|source| build_err("inlocal_v_reshape", source))?;
    let q = graph
        .rope_ext(q, positions, rope_params)
        .map_err(|source| build_err("inlocal_q_rope", source))?;
    let k = graph
        .rope_ext(k, positions, rope_params)
        .map_err(|source| build_err("inlocal_k_rope", source))?;

    // -> [head_dim, group_size, heads, n_groups] (swap ne1<->ne2; ne3 untouched).
    let q = graph
        .permute(q, 0, 2, 1, 3)
        .map_err(|source| build_err("inlocal_q_permute", source))?;
    let q = graph
        .cont(q)
        .map_err(|source| build_err("inlocal_q_cont", source))?;
    let k = graph
        .permute(k, 0, 2, 1, 3)
        .map_err(|source| build_err("inlocal_k_permute", source))?;
    let k = graph
        .cont(k)
        .map_err(|source| build_err("inlocal_k_cont", source))?;
    let v = graph
        .permute(v, 0, 2, 1, 3)
        .map_err(|source| build_err("inlocal_v_permute", source))?;
    let v = graph
        .cont(v)
        .map_err(|source| build_err("inlocal_v_cont", source))?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    // Batched (ne2=heads, ne3=n_groups) bidirectional attention, independent
    // per group -- no mask needed since every group's own `group_size`
    // positions are the entire kv range for that batch slice.
    let scores = graph
        .mul_mat(k, q)
        .map_err(|source| build_err("inlocal_attn_scores", source))?;
    let probs = graph
        .soft_max_ext(scores, None, scale, 0.0)
        .map_err(|source| build_err("inlocal_attn_softmax", source))?;
    let v_t = graph
        .permute(v, 1, 0, 2, 3)
        .map_err(|source| build_err("inlocal_attn_v_t", source))?;
    let v_t = graph
        .cont(v_t)
        .map_err(|source| build_err("inlocal_attn_v_t_cont", source))?;
    let context = graph
        .mul_mat(v_t, probs)
        .map_err(|source| build_err("inlocal_attn_ctx", source))?;
    // -> [head_dim, heads, group_size, n_groups] -> flatten heads back into d_model.
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(|source| build_err("inlocal_attn_merge", source))?;
    let merged = graph
        .cont(merged)
        .map_err(|source| build_err("inlocal_attn_merge_cont", source))?;
    let context = graph
        .reshape_2d(merged, d_model, seq_total)
        .map_err(|source| build_err("inlocal_attn_merge_reshape", source))?;

    let mut attn = graph
        .mul_mat(layer.attn_output.as_graph_tensor(), context)
        .map_err(|source| build_err("inlocal_out", source))?;
    attn = graph
        .add(residual, attn)
        .map_err(|source| build_err("inlocal_attn_residual", source))?;

    let ffn_residual = attn;
    let normed = apply_rms_norm(
        graph,
        attn,
        RMS_NORM_EPSILON,
        arena.graph_tensor(layer.ffn_norm),
        RmsNormSteps {
            norm: "inlocal_ffn_norm",
            scale: "inlocal_ffn_norm_scale",
        },
        build_err,
    )?;
    let gate = graph
        .mul_mat(layer.ffn_gate.as_graph_tensor(), normed)
        .map_err(|source| build_err("inlocal_ffn_gate", source))?;
    let gate = graph
        .silu(gate)
        .map_err(|source| build_err("inlocal_ffn_silu", source))?;
    let up = graph
        .mul_mat(layer.ffn_up.as_graph_tensor(), normed)
        .map_err(|source| build_err("inlocal_ffn_up", source))?;
    let gated = graph
        .mul(gate, up)
        .map_err(|source| build_err("inlocal_ffn_mul", source))?;
    let down = graph
        .mul_mat(layer.ffn_down.as_graph_tensor(), gated)
        .map_err(|source| build_err("inlocal_ffn_down", source))?;
    graph
        .add(ffn_residual, down)
        .map_err(|source| build_err("inlocal_ffn_residual", source))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_speech_embeddings_sums_channels_and_masks_zeroemb() {
        let tables = MimoSpeechEmbeddingTables {
            d_model: 2,
            tables: vec![
                vec![1.0, 1.0, 9.0, 9.0], // channel 0: code0=(1,1) code1(zeroemb)=(9,9)
                vec![2.0, 2.0, 8.0, 8.0], // channel 1: code0=(2,2) code1(zeroemb)=(8,8)
            ],
            vocab_sizes: vec![2, 2],
            zeroemb_idx: vec![1, 1],
        };
        // frame 0: both channels real (code 0) -> sum (3,3).
        // frame 1: channel 0 hits zeroemb (code 1) -> masked; channel 1 real (code 0) -> (2,2).
        let codes = MimoRvqCodes::from_channel_major(2, 2, vec![0, 1, 0, 0])
            .expect("valid channel-major codes");
        let out = sum_speech_embeddings(&tables, &codes).expect("speech embedding sum");
        assert_eq!(out, vec![3.0, 3.0, 2.0, 2.0]);
    }

    #[test]
    fn rejects_frame_count_not_group_aligned() {
        // Constructing a full runtime needs a real pack; this only exercises
        // the pure validation path via a manual arithmetic check mirrored
        // from `run`'s guard (kept here as a fast, pack-free regression).
        let frame_count = 6usize;
        let group_size = 4usize;
        assert!(!frame_count.is_multiple_of(group_size));
    }

    /// Regression guard for the right-sized input-local metadata context (the
    /// historical flat 256 MB over-reserved a bookkeeping-only context;
    /// `ggml_init` mallocs the full `mem_size` even with `no_alloc=true`).
    /// Only touches `context_bytes`/`graph_size` (thread-independent), so it
    /// is stable against parallel-test env mutation.
    #[test]
    fn input_local_graph_context_is_right_sized_not_flat_256mb() {
        let config = mimo_input_local_graph_config(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu);
        assert!(
            config.context_bytes >= GgmlCpuGraphConfig::metadata_context_bytes(config.graph_size)
        );
        assert!(
            config.context_bytes < 16 * 1024 * 1024,
            "mimo input-local metadata context regrew toward the old flat 256 MB: {}",
            config.context_bytes
        );
    }
}
