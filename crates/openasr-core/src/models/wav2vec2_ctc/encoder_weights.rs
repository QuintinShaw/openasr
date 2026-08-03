//! Load a wav2vec2-ctc `.oasr` pack into host weights.
//!
//! Every tensor is read generically (dims from the GGUF index, values
//! dequantized to f32). The 2-D linear projections (attn q/k/v/out, ffn up/down,
//! feature-projection, CTC head) are bound zero-copy from the mmap'd pack at
//! graph build (their f32 host copy is dropped after the shape check). The conv
//! kernels (feature extractor + folded pos-conv), group-norm gamma/beta, layer
//! norms and biases keep their `values` (arena-uploaded).

#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader, GgufTensorIndex};
use crate::models::runtime_memory::{
    ConstructionMemoryPlan, checked_sum, element_bytes, named_f32_tensor_quote_bytes,
};
use crate::models::system_memory_owner::SystemMemoryOwnerError;

use super::runtime_contract::{FEATURE_EXTRACTOR_CONV_DIM, Wav2Vec2CtcExecutionMetadata};

#[derive(Debug, thiserror::Error)]
pub(crate) enum Wav2Vec2EncoderWeightsError {
    #[error("wav2vec2-ctc encoder weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("wav2vec2-ctc encoder tensor '{name}' has {got} elements, expected {expected}")]
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

    fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2FeatureExtractorConv {
    /// Conv kernel `[K, in_channels, out_channels]` (ggml layout, stored f16).
    pub conv_weight: NamedTensor,
    /// Optional conv bias `[out_channels]` (hubert/lv60 `conv_bias=true`).
    pub conv_bias: Option<NamedTensor>,
    /// Channel-norm gamma/beta. For the base "group" model this is the layer-0
    /// GroupNorm; for the large "layer" model every conv layer carries a
    /// per-layer LayerNorm over channels. Absent layers carry `None`.
    pub norm_weight: Option<NamedTensor>,
    pub norm_bias: Option<NamedTensor>,
}

#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2EncoderLayerWeights {
    pub attn_q_weight: NamedTensor,
    pub attn_q_bias: NamedTensor,
    pub attn_k_weight: NamedTensor,
    pub attn_k_bias: NamedTensor,
    pub attn_v_weight: NamedTensor,
    pub attn_v_bias: NamedTensor,
    pub attn_out_weight: NamedTensor,
    pub attn_out_bias: NamedTensor,
    pub attn_norm_weight: NamedTensor,
    pub attn_norm_bias: NamedTensor,
    pub ffn_up_weight: NamedTensor,
    pub ffn_up_bias: NamedTensor,
    pub ffn_down_weight: NamedTensor,
    pub ffn_down_bias: NamedTensor,
    pub final_norm_weight: NamedTensor,
    pub final_norm_bias: NamedTensor,
}

/// One positional-conv layer: grouped conv kernel `[K, in/g, out]` (f16) + bias.
#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2PosConvLayer {
    pub weight: NamedTensor,
    pub bias: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2EncoderWeights {
    pub feature_extractor: Vec<Wav2Vec2FeatureExtractorConv>,
    pub fp_norm_weight: NamedTensor,
    pub fp_norm_bias: NamedTensor,
    pub fp_proj_weight: NamedTensor,
    pub fp_proj_bias: NamedTensor,
    /// Positional conv stack. wav2vec2/hubert: ONE folded weight-norm conv
    /// (`enc.posconv.weight`). data2vec: N plain grouped convs
    /// (`enc.posconv.{i}.weight`), each `[K, in/g, out]` f16 + bias, applied
    /// sequentially with gelu and added residually to hidden.
    pub pos_conv_layers: Vec<Wav2Vec2PosConvLayer>,
    pub encoder_norm_weight: NamedTensor,
    pub encoder_norm_bias: NamedTensor,
    pub layers: Vec<Wav2Vec2EncoderLayerWeights>,
    pub ctc_head_weight: NamedTensor,
    pub ctc_head_bias: NamedTensor,
}

/// Count-only host-memory topology for one wav2vec2-CTC runtime build.
///
/// The loader first materializes every named tensor as f32, then drops the
/// mmap-bound 2-D projection payloads before graph construction. The graph
/// retains only its Rust handle vectors; native ggml buffers are accounted for
/// by the backend memory owner and intentionally do not appear here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wav2Vec2SystemMemoryPlan {
    pub(crate) weights_peak_bytes: u64,
    pub(crate) weights_stable_bytes: u64,
    pub(crate) graph_retained_bytes: u64,
}

fn named_tensor_lifetime_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
    retain_values: bool,
) -> Result<(u64, u64), SystemMemoryOwnerError> {
    Ok((
        named_f32_tensor_quote_bytes(tensor_index, name, true, "wav2vec2-ctc")?,
        named_f32_tensor_quote_bytes(tensor_index, name, retain_values, "wav2vec2-ctc")?,
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
            "wav2vec2-ctc",
            "materialized tensor batch",
        )?;
        retained = checked_sum(
            [retained, tensor_retained],
            "wav2vec2-ctc",
            "retained tensor batch",
        )?;
    }
    Ok((materialized, retained))
}

/// Mirrors every `load_named`/`load_optional` call in this family. Missing
/// required tensors and every checked arithmetic failure are returned as a
/// capacity error so admission fails closed before materialization.
pub(crate) fn plan_wav2vec2_system_memory(
    tensor_index: &GgufTensorIndex,
    metadata: Wav2Vec2CtcExecutionMetadata,
) -> Result<Wav2Vec2SystemMemoryPlan, SystemMemoryOwnerError> {
    let mut lifetime = ConstructionMemoryPlan::new("wav2vec2-ctc");

    // `Vec::with_capacity(FEATURE_EXTRACTOR_CONV_DIM.len())`.
    let feature_count = FEATURE_EXTRACTOR_CONV_DIM.len();
    let feature_descriptors = element_bytes::<Wav2Vec2FeatureExtractorConv>(
        feature_count,
        "wav2vec2-ctc",
        "feature extractor descriptors",
    )?;
    lifetime.retain(feature_descriptors, "feature extractor descriptor storage")?;
    for layer in 0..feature_count {
        let mut names = vec![(format!("enc.fe.{layer}.conv.weight"), true)];
        for suffix in ["conv.bias", "gn.weight", "gn.bias"] {
            let name = format!("enc.fe.{layer}.{suffix}");
            if tensor_index.get(&name).is_some() {
                names.push((name, true));
            }
        }
        let (materialized, retained) = tensor_batch_lifetime_bytes(tensor_index, names)?;
        lifetime.materialize_then_retain(
            materialized,
            retained,
            "feature extractor layer materialization",
        )?;
    }

    for name in ["enc.fp.norm.weight", "enc.fp.norm.bias"] {
        let (materialized, retained) = named_tensor_lifetime_bytes(tensor_index, name, true)?;
        lifetime.materialize_then_retain(materialized, retained, "feature projection norm")?;
    }
    let (fp_materialized, fp_retained) = tensor_batch_lifetime_bytes(
        tensor_index,
        [("enc.fp.proj.weight", false), ("enc.fp.proj.bias", true)],
    )?;
    lifetime.materialize_then_retain(
        fp_materialized,
        fp_retained,
        "feature projection materialization",
    )?;

    let pos_conv_capacity = metadata.pos_conv_depth.max(1);
    let pos_conv_descriptors = element_bytes::<Wav2Vec2PosConvLayer>(
        pos_conv_capacity,
        "wav2vec2-ctc",
        "positional convolution descriptors",
    )?;
    lifetime.retain(
        pos_conv_descriptors,
        "positional convolution descriptor storage",
    )?;
    if metadata.pos_conv_depth <= 1 {
        let (materialized, retained) = tensor_batch_lifetime_bytes(
            tensor_index,
            [("enc.posconv.weight", true), ("enc.posconv.bias", true)],
        )?;
        lifetime.materialize_then_retain(
            materialized,
            retained,
            "positional convolution materialization",
        )?;
    } else {
        for layer in 0..metadata.pos_conv_depth {
            let (materialized, retained) = tensor_batch_lifetime_bytes(
                tensor_index,
                [
                    (format!("enc.posconv.{layer}.weight"), true),
                    (format!("enc.posconv.{layer}.bias"), true),
                ],
            )?;
            lifetime.materialize_then_retain(
                materialized,
                retained,
                "positional convolution layer materialization",
            )?;
        }
    }

    for name in ["enc.norm.weight", "enc.norm.bias"] {
        let (materialized, retained) = named_tensor_lifetime_bytes(tensor_index, name, true)?;
        lifetime.materialize_then_retain(materialized, retained, "encoder norm")?;
    }

    let layer_descriptors = element_bytes::<Wav2Vec2EncoderLayerWeights>(
        metadata.n_layers,
        "wav2vec2-ctc",
        "encoder layer descriptors",
    )?;
    lifetime.retain(layer_descriptors, "encoder layer descriptor storage")?;
    for layer in 0..metadata.n_layers {
        let prefix = format!("enc.blk.{layer}");
        let batch = [
            ("attn.q.weight", false),
            ("attn.q.bias", true),
            ("attn.k.weight", false),
            ("attn.k.bias", true),
            ("attn.v.weight", false),
            ("attn.v.bias", true),
            ("attn.out.weight", false),
            ("attn.out.bias", true),
            ("attn.norm.weight", true),
            ("attn.norm.bias", true),
            ("ffn.up.weight", false),
            ("ffn.up.bias", true),
            ("ffn.down.weight", false),
            ("ffn.down.bias", true),
            ("final.norm.weight", true),
            ("final.norm.bias", true),
        ]
        .map(|(suffix, retain_values)| (format!("{prefix}.{suffix}"), retain_values));
        let (materialized, retained) = tensor_batch_lifetime_bytes(tensor_index, batch)?;
        lifetime.materialize_then_retain(
            materialized,
            retained,
            "encoder layer dequantization batch",
        )?;
    }

    let (ctc_materialized, ctc_retained) = tensor_batch_lifetime_bytes(
        tensor_index,
        [("ctc.head.weight", false), ("ctc.head.bias", true)],
    )?;
    lifetime.materialize_then_retain(
        ctc_materialized,
        ctc_retained,
        "CTC head dequantization batch",
    )?;

    // The graph constructs these exact `with_capacity` vectors and keeps no
    // other Rust Vec handles in its retained topology.
    let graph_retained_bytes =
        crate::models::wav2vec2_ctc::encoder_graph::quoted_graph_retained_bytes(
            feature_count,
            pos_conv_capacity,
            metadata.n_layers,
        )?;

    Ok(Wav2Vec2SystemMemoryPlan {
        weights_peak_bytes: lifetime.peak_bytes(),
        weights_stable_bytes: lifetime.stable_bytes(),
        graph_retained_bytes,
    })
}

/// Post-build recursive count of the host containers that survive in a
/// materialized weight bundle. Kept here for tests and future callers that
/// inspect a loader result; the runtime itself drops the bundle after graph
/// construction.
pub(crate) fn retained_weights_system_memory_bytes(
    weights: &Wav2Vec2EncoderWeights,
) -> Result<u64, String> {
    let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
    bytes.add_vec(
        &weights.feature_extractor,
        "wav2vec2-ctc feature extractor descriptors",
    )?;
    for feature in &weights.feature_extractor {
        add_named_tensor_retained(
            &feature.conv_weight,
            &mut bytes,
            "wav2vec2-ctc feature tensor",
        )?;
        for tensor in [&feature.conv_bias, &feature.norm_weight, &feature.norm_bias]
            .into_iter()
            .flatten()
        {
            add_named_tensor_retained(tensor, &mut bytes, "wav2vec2-ctc feature tensor")?;
        }
    }
    for tensor in [
        &weights.fp_norm_weight,
        &weights.fp_norm_bias,
        &weights.fp_proj_weight,
        &weights.fp_proj_bias,
        &weights.encoder_norm_weight,
        &weights.encoder_norm_bias,
        &weights.ctc_head_weight,
        &weights.ctc_head_bias,
    ] {
        add_named_tensor_retained(tensor, &mut bytes, "wav2vec2-ctc encoder tensor")?;
    }
    bytes.add_vec(
        &weights.pos_conv_layers,
        "wav2vec2-ctc positional convolution descriptors",
    )?;
    for layer in &weights.pos_conv_layers {
        add_named_tensor_retained(&layer.weight, &mut bytes, "wav2vec2-ctc positional tensor")?;
        add_named_tensor_retained(&layer.bias, &mut bytes, "wav2vec2-ctc positional tensor")?;
    }
    bytes.add_vec(&weights.layers, "wav2vec2-ctc encoder layer descriptors")?;
    for layer in &weights.layers {
        for tensor in [
            &layer.attn_q_weight,
            &layer.attn_q_bias,
            &layer.attn_k_weight,
            &layer.attn_k_bias,
            &layer.attn_v_weight,
            &layer.attn_v_bias,
            &layer.attn_out_weight,
            &layer.attn_out_bias,
            &layer.attn_norm_weight,
            &layer.attn_norm_bias,
            &layer.ffn_up_weight,
            &layer.ffn_up_bias,
            &layer.ffn_down_weight,
            &layer.ffn_down_bias,
            &layer.final_norm_weight,
            &layer.final_norm_bias,
        ] {
            add_named_tensor_retained(tensor, &mut bytes, "wav2vec2-ctc layer tensor")?;
        }
    }
    Ok(bytes.finish())
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
) -> Result<NamedTensor, Wav2Vec2EncoderWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        Wav2Vec2EncoderWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
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

fn load_optional(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<Option<NamedTensor>, Wav2Vec2EncoderWeightsError> {
    if reader.tensor_index().get(name).is_some() {
        Ok(Some(load_named(reader, name)?))
    } else {
        Ok(None)
    }
}

fn load_layer(
    reader: &GgufTensorDataReader,
    layer: usize,
) -> Result<Wav2Vec2EncoderLayerWeights, Wav2Vec2EncoderWeightsError> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let mut weights = Wav2Vec2EncoderLayerWeights {
        attn_q_weight: load_named(reader, &n("attn.q.weight"))?,
        attn_q_bias: load_named(reader, &n("attn.q.bias"))?,
        attn_k_weight: load_named(reader, &n("attn.k.weight"))?,
        attn_k_bias: load_named(reader, &n("attn.k.bias"))?,
        attn_v_weight: load_named(reader, &n("attn.v.weight"))?,
        attn_v_bias: load_named(reader, &n("attn.v.bias"))?,
        attn_out_weight: load_named(reader, &n("attn.out.weight"))?,
        attn_out_bias: load_named(reader, &n("attn.out.bias"))?,
        attn_norm_weight: load_named(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, &n("attn.norm.bias"))?,
        ffn_up_weight: load_named(reader, &n("ffn.up.weight"))?,
        ffn_up_bias: load_named(reader, &n("ffn.up.bias"))?,
        ffn_down_weight: load_named(reader, &n("ffn.down.weight"))?,
        ffn_down_bias: load_named(reader, &n("ffn.down.bias"))?,
        final_norm_weight: load_named(reader, &n("final.norm.weight"))?,
        final_norm_bias: load_named(reader, &n("final.norm.bias"))?,
    };
    // Bind the 2-D linears zero-copy: drop their host f32 copy.
    for w in [
        &mut weights.attn_q_weight,
        &mut weights.attn_k_weight,
        &mut weights.attn_v_weight,
        &mut weights.attn_out_weight,
        &mut weights.ffn_up_weight,
        &mut weights.ffn_down_weight,
    ] {
        w.drop_bound_payload();
    }
    Ok(weights)
}

pub(crate) fn load_wav2vec2_ctc_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &Wav2Vec2CtcExecutionMetadata,
) -> Result<Wav2Vec2EncoderWeights, Wav2Vec2EncoderWeightsError> {
    let mut feature_extractor = Vec::with_capacity(FEATURE_EXTRACTOR_CONV_DIM.len());
    for layer in 0..FEATURE_EXTRACTOR_CONV_DIM.len() {
        feature_extractor.push(Wav2Vec2FeatureExtractorConv {
            conv_weight: load_named(reader, &format!("enc.fe.{layer}.conv.weight"))?,
            conv_bias: load_optional(reader, &format!("enc.fe.{layer}.conv.bias"))?,
            norm_weight: load_optional(reader, &format!("enc.fe.{layer}.gn.weight"))?,
            norm_bias: load_optional(reader, &format!("enc.fe.{layer}.gn.bias"))?,
        });
    }

    let fp_norm_weight = load_named(reader, "enc.fp.norm.weight")?;
    let fp_norm_bias = load_named(reader, "enc.fp.norm.bias")?;
    let mut fp_proj_weight = load_named(reader, "enc.fp.proj.weight")?;
    let fp_proj_bias = load_named(reader, "enc.fp.proj.bias")?;
    fp_proj_weight.drop_bound_payload();

    // Positional conv: a single folded conv (`enc.posconv.weight`, depth 1) or
    // data2vec's stacked plain convs (`enc.posconv.{i}.weight`, depth > 1).
    let mut pos_conv_layers = Vec::with_capacity(metadata.pos_conv_depth.max(1));
    if metadata.pos_conv_depth <= 1 {
        pos_conv_layers.push(Wav2Vec2PosConvLayer {
            weight: load_named(reader, "enc.posconv.weight")?,
            bias: load_named(reader, "enc.posconv.bias")?,
        });
    } else {
        for i in 0..metadata.pos_conv_depth {
            pos_conv_layers.push(Wav2Vec2PosConvLayer {
                weight: load_named(reader, &format!("enc.posconv.{i}.weight"))?,
                bias: load_named(reader, &format!("enc.posconv.{i}.bias"))?,
            });
        }
    }
    let encoder_norm_weight = load_named(reader, "enc.norm.weight")?;
    let encoder_norm_bias = load_named(reader, "enc.norm.bias")?;

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        layers.push(load_layer(reader, layer)?);
    }

    let mut ctc_head_weight = load_named(reader, "ctc.head.weight")?;
    let ctc_head_bias = load_named(reader, "ctc.head.bias")?;
    let expected_head = metadata.vocab_size * metadata.hidden_size;
    if ctc_head_weight.element_count() != expected_head {
        return Err(Wav2Vec2EncoderWeightsError::ElementCount {
            name: ctc_head_weight.name.clone(),
            got: ctc_head_weight.element_count(),
            expected: expected_head,
        });
    }
    // CTC head is bound zero-copy (the head is small f32 — keep it arena-bound
    // since it isn't reversed-stored f16 like parakeet's; drop only after the
    // check). It is loaded as a 2-D `[hidden, vocab]` weight; keep it bound.
    ctc_head_weight.drop_bound_payload();

    Ok(Wav2Vec2EncoderWeights {
        feature_extractor,
        fp_norm_weight,
        fp_norm_bias,
        fp_proj_weight,
        fp_proj_bias,
        pos_conv_layers,
        encoder_norm_weight,
        encoder_norm_bias,
        layers,
        ctc_head_weight,
        ctc_head_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::super::runtime_contract::parse_wav2vec2_ctc_execution_metadata;
    use super::*;
    use std::path::Path;

    fn pack_path() -> Option<std::path::PathBuf> {
        [Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/wav2vec2-base-960h-source/openasr/wav2vec2-base-960h-q4k.oasr")]
        .into_iter()
        .find(|p| p.exists())
    }

    #[test]
    fn loads_wav2vec2_encoder_weights_when_pack_present() {
        let Some(path) = pack_path() else {
            eprintln!("skipping: wav2vec2-base-960h pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_wav2vec2_ctc_execution_metadata(&gguf_metadata).expect("metadata");
        assert_eq!(metadata.n_layers, 12);

        let weights = load_wav2vec2_ctc_encoder_weights(&reader, &metadata).expect("weights");
        assert_eq!(weights.feature_extractor.len(), 7);
        // base-960h "group" variant: norm gamma/beta only on layer 0, no conv bias.
        assert!(weights.feature_extractor[0].norm_weight.is_some());
        assert!(weights.feature_extractor[1].norm_weight.is_none());
        assert!(weights.feature_extractor[0].conv_bias.is_none());
        assert_eq!(weights.layers.len(), 12);
        // wav2vec2 base: one folded pos-conv kernel [128, 48, 768] = 4_718_592.
        assert_eq!(weights.pos_conv_layers.len(), 1);
        assert_eq!(
            weights.pos_conv_layers[0]
                .weight
                .dims
                .iter()
                .product::<usize>(),
            128 * 48 * 768
        );
        assert_eq!(weights.ctc_head_bias.element_count(), metadata.vocab_size);
    }
}
