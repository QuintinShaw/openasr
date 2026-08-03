//! Load a sensevoice `.oasr` pack into host weights.
//!
//! Mirrors `parakeet_ctc::encoder_weights`: every tensor is read generically
//! (dims from the GGUF index, values dequantized to f32); the 2-D linear
//! projections (`attn.qkv/out`, `ffn.up/down`, `ctc.head.weight`) are bound
//! zero-copy from the mmap'd pack by the encoder graph, so their f32 host
//! payloads are dropped after shape validation (keep-quantized: the graph's
//! `mul_mat` consumes the native q8_0/q4_k blocks straight from the pack).
//! Norms, biases, and the FSMN depthwise kernels stay host-resident (arena
//! uploads); the CMVN vectors and the 16x560 prompt-embedding table are
//! consumed host-side by the frontend/prompt splice, never by the graph.

#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader, GgufTensorIndex};
use crate::models::runtime_memory::{
    ConstructionMemoryPlan, checked_sum, element_bytes, named_f32_tensor_quote_bytes,
    tensor_f32_bytes,
};
use crate::models::system_memory_owner::SystemMemoryOwnerError;

use super::runtime_contract::SenseVoiceExecutionMetadata;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SenseVoiceEncoderWeightsError {
    #[error("sensevoice encoder weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("sensevoice encoder tensor '{name}' has {got} elements, expected {expected}")]
    ElementCount {
        name: String,
        got: usize,
        expected: usize,
    },
}

/// A host weight: its stored dims (from the GGUF index) + dequantized f32 values.
#[derive(Debug, Clone)]
pub(crate) struct NamedTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

impl NamedTensor {
    fn element_count(&self) -> usize {
        self.values.len()
    }

    /// Drop the resident f32 host `values` (keeping name + dims) for a weight
    /// the encoder graph binds zero-copy from the mmap'd pack.
    fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

/// One SAN-M block's weights (`enc.blk.{i}.*` or `tp.blk.{i}.*`).
#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceLayerWeights {
    pub attn_norm_weight: NamedTensor,
    pub attn_norm_bias: NamedTensor,
    /// Fused `[in, 3*d_model]` QKV projection (bound zero-copy).
    pub attn_qkv_weight: NamedTensor,
    pub attn_qkv_bias: NamedTensor,
    pub attn_out_weight: NamedTensor,
    pub attn_out_bias: NamedTensor,
    /// FSMN depthwise conv kernel `[kernel, 1, d_model]` (f16 arena upload).
    pub attn_fsmn_weight: NamedTensor,
    pub ffn_norm_weight: NamedTensor,
    pub ffn_norm_bias: NamedTensor,
    pub ffn_up_weight: NamedTensor,
    pub ffn_up_bias: NamedTensor,
    pub ffn_down_weight: NamedTensor,
    pub ffn_down_bias: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceEncoderWeights {
    /// `enc.blk.0..n_layers-1` (block 0 consumes the 560-dim LFR+prompt input).
    pub enc_layers: Vec<SenseVoiceLayerWeights>,
    /// `tp.blk.0..tp_layers-1`, run after `enc_after_norm`.
    pub tp_layers: Vec<SenseVoiceLayerWeights>,
    pub enc_after_norm_weight: NamedTensor,
    pub enc_after_norm_bias: NamedTensor,
    pub tp_norm_weight: NamedTensor,
    pub tp_norm_bias: NamedTensor,
    pub ctc_head_weight: NamedTensor,
    pub ctc_head_bias: NamedTensor,
    /// 16x560 prompt-embedding table (host f32, spliced by the frontend).
    pub prompt_embed: NamedTensor,
    /// `am.mvn` CMVN vectors (host f32, applied by the frontend).
    pub cmvn_neg_mean: NamedTensor,
    pub cmvn_inv_stddev: NamedTensor,
}

/// Count-only host-memory topology for one SenseVoice runtime build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SenseVoiceSystemMemoryPlan {
    pub(crate) weights_peak_bytes: u64,
    pub(crate) weights_stable_bytes: u64,
    pub(crate) graph_retained_bytes: u64,
    pub(crate) prompt_embed_bytes: u64,
    pub(crate) cmvn_neg_mean_bytes: u64,
    pub(crate) cmvn_inv_stddev_bytes: u64,
}

fn named_tensor_lifetime_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
    retain_values: bool,
) -> Result<(u64, u64), SystemMemoryOwnerError> {
    Ok((
        named_f32_tensor_quote_bytes(tensor_index, name, true, "sensevoice")?,
        named_f32_tensor_quote_bytes(tensor_index, name, retain_values, "sensevoice")?,
    ))
}

fn tensor_batch_lifetime_bytes<I, S>(
    tensor_index: &GgufTensorIndex,
    tensors: I,
) -> Result<(u64, u64), SystemMemoryOwnerError>
where
    I: IntoIterator<Item = (S, bool)>,
    S: AsRef<str>,
{
    let mut materialized = 0_u64;
    let mut retained = 0_u64;
    for (name, retain_values) in tensors {
        let (tensor_materialized, tensor_retained) =
            named_tensor_lifetime_bytes(tensor_index, name.as_ref(), retain_values)?;
        materialized = checked_sum(
            [materialized, tensor_materialized],
            "sensevoice",
            "materialized tensor batch",
        )?;
        retained = checked_sum(
            [retained, tensor_retained],
            "sensevoice",
            "retained tensor batch",
        )?;
    }
    Ok((materialized, retained))
}

/// Mirrors every named tensor materialized by `load_sensevoice_encoder_weights`.
/// Four 2-D linears per SAN-M layer and the CTC head are mmap-bound after the
/// f32 shape check; every other tensor remains host-resident until graph build.
pub(crate) fn plan_sensevoice_system_memory(
    tensor_index: &GgufTensorIndex,
    metadata: SenseVoiceExecutionMetadata,
) -> Result<SenseVoiceSystemMemoryPlan, SystemMemoryOwnerError> {
    let mut lifetime = ConstructionMemoryPlan::new("sensevoice");

    let enc_descriptors = element_bytes::<SenseVoiceLayerWeights>(
        metadata.n_layers,
        "sensevoice",
        "encoder layer descriptors",
    )?;
    let tp_descriptors = element_bytes::<SenseVoiceLayerWeights>(
        metadata.tp_layers,
        "sensevoice",
        "transcription layer descriptors",
    )?;
    lifetime.retain(enc_descriptors, "encoder layer descriptor storage")?;

    for (scope, count) in [("enc.blk", metadata.n_layers)] {
        for layer in 0..count {
            let prefix = format!("{scope}.{layer}");
            let batch = [
                ("attn.norm.weight", true),
                ("attn.norm.bias", true),
                ("attn.qkv.weight", false),
                ("attn.qkv.bias", true),
                ("attn.out.weight", false),
                ("attn.out.bias", true),
                ("attn.fsmn.weight", true),
                ("ffn.norm.weight", true),
                ("ffn.norm.bias", true),
                ("ffn.up.weight", false),
                ("ffn.up.bias", true),
                ("ffn.down.weight", false),
                ("ffn.down.bias", true),
            ]
            .map(|(suffix, retain_values)| (format!("{prefix}.{suffix}"), retain_values));
            let (materialized, retained) = tensor_batch_lifetime_bytes(tensor_index, batch)?;
            lifetime.materialize_then_retain(
                materialized,
                retained,
                "encoder layer dequantization batch",
            )?;
        }
    }

    lifetime.retain(tp_descriptors, "transcription layer descriptor storage")?;
    for layer in 0..metadata.tp_layers {
        let prefix = format!("tp.blk.{layer}");
        let batch = [
            ("attn.norm.weight", true),
            ("attn.norm.bias", true),
            ("attn.qkv.weight", false),
            ("attn.qkv.bias", true),
            ("attn.out.weight", false),
            ("attn.out.bias", true),
            ("attn.fsmn.weight", true),
            ("ffn.norm.weight", true),
            ("ffn.norm.bias", true),
            ("ffn.up.weight", false),
            ("ffn.up.bias", true),
            ("ffn.down.weight", false),
            ("ffn.down.bias", true),
        ]
        .map(|(suffix, retain_values)| (format!("{prefix}.{suffix}"), retain_values));
        let (materialized, retained) = tensor_batch_lifetime_bytes(tensor_index, batch)?;
        lifetime.materialize_then_retain(
            materialized,
            retained,
            "transcription layer dequantization batch",
        )?;
    }

    // Mirror `load_sensevoice_encoder_weights` statement order. The CTC head
    // payload is dropped before the prompt, CMVN and trailing norm tensors are
    // loaded, so those later resident tensors cannot inflate its transient
    // peak.
    let (ctc_materialized, ctc_retained) = tensor_batch_lifetime_bytes(
        tensor_index,
        [("ctc.head.weight", false), ("ctc.head.bias", true)],
    )?;
    lifetime.materialize_then_retain(
        ctc_materialized,
        ctc_retained,
        "CTC head dequantization batch",
    )?;
    for name in [
        "embed.prompt.weight",
        "frontend.cmvn.neg_mean",
        "frontend.cmvn.inv_stddev",
        "enc.after_norm.weight",
        "enc.after_norm.bias",
        "tp.norm.weight",
        "tp.norm.bias",
    ] {
        let (materialized, retained) = named_tensor_lifetime_bytes(tensor_index, name, true)?;
        lifetime.materialize_then_retain(materialized, retained, "retained tail tensor")?;
    }

    // The graph constructs exactly one handle Vec per SAN-M stage.
    let graph_retained_bytes =
        crate::models::sensevoice::encoder_graph::quoted_graph_retained_bytes(
            metadata.n_layers,
            metadata.tp_layers,
        )?;
    let prompt_embed_bytes = tensor_f32_bytes(tensor_index, "embed.prompt.weight", "sensevoice")?;
    let cmvn_neg_mean_bytes =
        tensor_f32_bytes(tensor_index, "frontend.cmvn.neg_mean", "sensevoice")?;
    let cmvn_inv_stddev_bytes =
        tensor_f32_bytes(tensor_index, "frontend.cmvn.inv_stddev", "sensevoice")?;

    Ok(SenseVoiceSystemMemoryPlan {
        weights_peak_bytes: lifetime.peak_bytes(),
        weights_stable_bytes: lifetime.stable_bytes(),
        graph_retained_bytes,
        prompt_embed_bytes,
        cmvn_neg_mean_bytes,
        cmvn_inv_stddev_bytes,
    })
}

pub(crate) fn retained_weights_system_memory_bytes(
    weights: &SenseVoiceEncoderWeights,
) -> Result<u64, String> {
    let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
    bytes.add_vec(&weights.enc_layers, "sensevoice encoder layer descriptors")?;
    bytes.add_vec(
        &weights.tp_layers,
        "sensevoice transcription layer descriptors",
    )?;
    for layer in weights.enc_layers.iter().chain(&weights.tp_layers) {
        for tensor in sensevoice_layer_tensors(layer) {
            add_named_tensor_retained(tensor, &mut bytes, "sensevoice layer tensor")?;
        }
    }
    for tensor in [
        &weights.enc_after_norm_weight,
        &weights.enc_after_norm_bias,
        &weights.tp_norm_weight,
        &weights.tp_norm_bias,
        &weights.ctc_head_weight,
        &weights.ctc_head_bias,
        &weights.prompt_embed,
        &weights.cmvn_neg_mean,
        &weights.cmvn_inv_stddev,
    ] {
        add_named_tensor_retained(tensor, &mut bytes, "sensevoice encoder tensor")?;
    }
    Ok(bytes.finish())
}

fn sensevoice_layer_tensors(layer: &SenseVoiceLayerWeights) -> [&NamedTensor; 13] {
    [
        &layer.attn_norm_weight,
        &layer.attn_norm_bias,
        &layer.attn_qkv_weight,
        &layer.attn_qkv_bias,
        &layer.attn_out_weight,
        &layer.attn_out_bias,
        &layer.attn_fsmn_weight,
        &layer.ffn_norm_weight,
        &layer.ffn_norm_bias,
        &layer.ffn_up_weight,
        &layer.ffn_up_bias,
        &layer.ffn_down_weight,
        &layer.ffn_down_bias,
    ]
}

fn add_named_tensor_retained(
    tensor: &NamedTensor,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    bytes.add_string(&tensor.name, &format!("{label} name"))?;
    bytes.add_vec(&tensor.dims, &format!("{label} dims"))?;
    bytes.add_vec(&tensor.values, &format!("{label} values"))
}

fn load_named(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, SenseVoiceEncoderWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        SenseVoiceEncoderWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
            path: reader.tensor_index().path().to_path_buf(),
            tensor_name: name.to_string(),
        })
    })?;
    let values = reader.host_tensor_f32_copy_dequantized_by_name(name, &tensor.dims)?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    Ok(NamedTensor {
        name: name.to_string(),
        dims,
        values,
    })
}

fn load_layer(
    reader: &GgufTensorDataReader,
    scope: &str,
    layer: usize,
) -> Result<SenseVoiceLayerWeights, SenseVoiceEncoderWeightsError> {
    let n = |suffix: &str| format!("{scope}.{layer}.{suffix}");
    let mut weights = SenseVoiceLayerWeights {
        attn_norm_weight: load_named(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, &n("attn.norm.bias"))?,
        attn_qkv_weight: load_named(reader, &n("attn.qkv.weight"))?,
        attn_qkv_bias: load_named(reader, &n("attn.qkv.bias"))?,
        attn_out_weight: load_named(reader, &n("attn.out.weight"))?,
        attn_out_bias: load_named(reader, &n("attn.out.bias"))?,
        attn_fsmn_weight: load_named(reader, &n("attn.fsmn.weight"))?,
        ffn_norm_weight: load_named(reader, &n("ffn.norm.weight"))?,
        ffn_norm_bias: load_named(reader, &n("ffn.norm.bias"))?,
        ffn_up_weight: load_named(reader, &n("ffn.up.weight"))?,
        ffn_up_bias: load_named(reader, &n("ffn.up.bias"))?,
        ffn_down_weight: load_named(reader, &n("ffn.down.weight"))?,
        ffn_down_bias: load_named(reader, &n("ffn.down.bias"))?,
    };
    // Bound zero-copy by the graph: drop the dominant f32 host payloads.
    for w in [
        &mut weights.attn_qkv_weight,
        &mut weights.attn_out_weight,
        &mut weights.ffn_up_weight,
        &mut weights.ffn_down_weight,
    ] {
        w.drop_bound_payload();
    }
    Ok(weights)
}

pub(crate) fn load_sensevoice_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &SenseVoiceExecutionMetadata,
) -> Result<SenseVoiceEncoderWeights, SenseVoiceEncoderWeightsError> {
    let mut enc_layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        enc_layers.push(load_layer(reader, "enc.blk", layer)?);
    }
    let mut tp_layers = Vec::with_capacity(metadata.tp_layers);
    for layer in 0..metadata.tp_layers {
        tp_layers.push(load_layer(reader, "tp.blk", layer)?);
    }

    let mut ctc_head_weight = load_named(reader, "ctc.head.weight")?;
    let ctc_head_bias = load_named(reader, "ctc.head.bias")?;
    let expected_head = metadata.vocab_size * metadata.d_model;
    if ctc_head_weight.element_count() != expected_head {
        return Err(SenseVoiceEncoderWeightsError::ElementCount {
            name: ctc_head_weight.name.clone(),
            got: ctc_head_weight.element_count(),
            expected: expected_head,
        });
    }
    ctc_head_weight.drop_bound_payload();

    let prompt_embed = load_named(reader, "embed.prompt.weight")?;
    if !prompt_embed
        .element_count()
        .is_multiple_of(metadata.feature_dim)
    {
        return Err(SenseVoiceEncoderWeightsError::ElementCount {
            name: prompt_embed.name.clone(),
            got: prompt_embed.element_count(),
            expected: metadata.feature_dim,
        });
    }
    let cmvn_neg_mean = load_named(reader, "frontend.cmvn.neg_mean")?;
    let cmvn_inv_stddev = load_named(reader, "frontend.cmvn.inv_stddev")?;
    for cmvn in [&cmvn_neg_mean, &cmvn_inv_stddev] {
        if cmvn.element_count() != metadata.feature_dim {
            return Err(SenseVoiceEncoderWeightsError::ElementCount {
                name: cmvn.name.clone(),
                got: cmvn.element_count(),
                expected: metadata.feature_dim,
            });
        }
    }

    Ok(SenseVoiceEncoderWeights {
        enc_layers,
        tp_layers,
        enc_after_norm_weight: load_named(reader, "enc.after_norm.weight")?,
        enc_after_norm_bias: load_named(reader, "enc.after_norm.bias")?,
        tp_norm_weight: load_named(reader, "tp.norm.weight")?,
        tp_norm_bias: load_named(reader, "tp.norm.bias")?,
        ctc_head_weight,
        ctc_head_bias,
        prompt_embed,
        cmvn_neg_mean,
        cmvn_inv_stddev,
    })
}
