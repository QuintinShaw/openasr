//! Load a wav2vec2-ctc `.oasr` pack into host weights.
//!
//! Every tensor is read generically (dims from the GGUF index, values
//! dequantized to f32). The 2-D linear projections (attn q/k/v/out, ffn up/down,
//! feature-projection, CTC head) are bound zero-copy from the mmap'd pack at
//! graph build (their f32 host copy is dropped after the shape check). The conv
//! kernels (feature extractor + folded pos-conv), group-norm gamma/beta, layer
//! norms and biases keep their `values` (arena-uploaded).

#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

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
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let shape_u64: Vec<u64> = tensor.dims.clone();
    let values = reader.host_tensor_f32_copy_dequantized_by_name(name, &shape_u64)?;
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
