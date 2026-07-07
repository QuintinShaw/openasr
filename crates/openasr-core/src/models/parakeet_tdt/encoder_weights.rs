//! Load a parakeet-tdt `.oasr` pack into host weights: the FastConformer
//! encoder stack (mirroring `parakeet_ctc::encoder_weights`, including the
//! conv BatchNorm fold), the encoder joint projection, and the host-side
//! prediction-network / joint tensors.
//!
//! The v3 checkpoint has NO attention/conv/FFN biases (`attention_bias` /
//! `convolution_bias` false), so the loader synthesizes zero biases for the
//! shared `nn::encoder::conformer_block`, which is bias-shaped. Zero biases
//! are mathematically identity — nothing model-specific is fabricated.

// Consumed by the executor wired in the follow-up stage; tested meanwhile.
#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::ParakeetTdtExecutionMetadata;

const CONV_BN_EPSILON: f32 = 1.0e-5;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetTdtWeightsError {
    #[error("parakeet-tdt weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("parakeet-tdt tensor '{name}' has {got} elements, expected {expected}")]
    ElementCount {
        name: String,
        got: usize,
        expected: usize,
    },
    #[error("parakeet-tdt conv BatchNorm fold failed: {reason}")]
    BatchNormFold { reason: String },
}

/// A host weight: its stored dims (from the GGUF index) + dequantized f32
/// values. Bias slots the checkpoint does not ship are synthesized as zeros
/// (empty `name` marks them as synthetic for debugging only).
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
    /// the encoder graph binds zero-copy from the mmap'd pack (see
    /// `parakeet_ctc::encoder_weights::NamedTensor::drop_bound_payload`).
    fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtEncoderLayerWeights {
    pub ff1_norm_weight: NamedTensor,
    pub ff1_norm_bias: NamedTensor,
    pub ff1_up_weight: NamedTensor,
    pub ff1_up_bias: NamedTensor,
    pub ff1_down_weight: NamedTensor,
    pub ff1_down_bias: NamedTensor,
    pub attn_norm_weight: NamedTensor,
    pub attn_norm_bias: NamedTensor,
    pub attn_q_weight: NamedTensor,
    pub attn_q_bias: NamedTensor,
    pub attn_k_weight: NamedTensor,
    pub attn_k_bias: NamedTensor,
    pub attn_v_weight: NamedTensor,
    pub attn_v_bias: NamedTensor,
    pub attn_out_weight: NamedTensor,
    pub attn_out_bias: NamedTensor,
    pub attn_pos_weight: NamedTensor,
    pub attn_pos_bias_u: NamedTensor,
    pub attn_pos_bias_v: NamedTensor,
    pub conv_norm_weight: NamedTensor,
    pub conv_norm_bias: NamedTensor,
    pub conv_pw1_weight: NamedTensor,
    pub conv_pw1_bias: NamedTensor,
    /// BatchNorm folded into these two at load (the graph sees a plain dw conv).
    pub conv_dw_weight: NamedTensor,
    pub conv_dw_bias: NamedTensor,
    pub conv_pw2_weight: NamedTensor,
    pub conv_pw2_bias: NamedTensor,
    pub ff2_norm_weight: NamedTensor,
    pub ff2_norm_bias: NamedTensor,
    pub ff2_up_weight: NamedTensor,
    pub ff2_up_bias: NamedTensor,
    pub ff2_down_weight: NamedTensor,
    pub ff2_down_bias: NamedTensor,
    pub out_norm_weight: NamedTensor,
    pub out_norm_bias: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtEncoderWeights {
    /// dw-striding subsampling conv2d/linear tensors (`enc.sub.*`).
    pub subsampling: Vec<NamedTensor>,
    pub layers: Vec<ParakeetTdtEncoderLayerWeights>,
    /// Joint encoder projection `enc.proj.{weight,bias}` (d_model -> joint
    /// hidden), applied in-graph after the conformer stack. The weight is
    /// bound zero-copy (values dropped); the bias stays host f32.
    pub enc_proj_weight: NamedTensor,
    pub enc_proj_bias: NamedTensor,
}

/// Host-side prediction network + joint weights (consumed by the per-symbol
/// greedy loop on the CPU, mirroring the xasr decoder/joiner split: these are
/// per-step matvecs, not ggml graph matmuls).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtPredictorWeights {
    /// Token embedding, row-major `[vocab][pred_hidden]` (the blank row is
    /// NeMo's `padding_idx` and is all-zeros in the trained checkpoint).
    pub embedding: NamedTensor,
    /// Per-LSTM-layer packed gate weights, PyTorch order `[i|f|g|o]`:
    /// `w_ih[4*H][in]`, `w_hh[4*H][H]`, `b_ih[4*H]`, `b_hh[4*H]`.
    pub lstm_layers: Vec<ParakeetTdtLstmLayerWeights>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtLstmLayerWeights {
    pub w_ih: NamedTensor,
    pub w_hh: NamedTensor,
    pub b_ih: NamedTensor,
    pub b_hh: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtJointWeights {
    /// Predictor projection `joint.pred.{weight,bias}` (pred_hidden -> joint
    /// hidden), row-major `[joint_hidden][pred_hidden]`.
    pub pred_weight: NamedTensor,
    pub pred_bias: NamedTensor,
    /// Fused joint head `joint.out.{weight,bias}`: `[vocab + n_durations]`
    /// rows over the ReLU'd joint hidden.
    pub out_weight: NamedTensor,
    pub out_bias: NamedTensor,
}

fn load_named(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, ParakeetTdtWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        ParakeetTdtWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
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

/// Zero bias for a projection the checkpoint ships bias-free
/// (`attention_bias`/`convolution_bias`/FFN bias all false in v3). The shared
/// conformer block is bias-shaped; a zero bias is the mathematical identity.
fn zero_bias(name: String, len: usize) -> NamedTensor {
    NamedTensor {
        name,
        dims: vec![len],
        values: vec![0.0; len],
    }
}

fn load_layer(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
    layer: usize,
) -> Result<ParakeetTdtEncoderLayerWeights, ParakeetTdtWeightsError> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let d_model = metadata.hidden_size;
    let ffn = metadata.ffn_dim;
    let mut weights = ParakeetTdtEncoderLayerWeights {
        ff1_norm_weight: load_named(reader, &n("ff1.norm.weight"))?,
        ff1_norm_bias: load_named(reader, &n("ff1.norm.bias"))?,
        ff1_up_weight: load_named(reader, &n("ff1.up.weight"))?,
        ff1_up_bias: zero_bias(n("ff1.up.bias"), ffn),
        ff1_down_weight: load_named(reader, &n("ff1.down.weight"))?,
        ff1_down_bias: zero_bias(n("ff1.down.bias"), d_model),
        attn_norm_weight: load_named(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, &n("attn.norm.bias"))?,
        attn_q_weight: load_named(reader, &n("attn.q.weight"))?,
        attn_q_bias: zero_bias(n("attn.q.bias"), d_model),
        attn_k_weight: load_named(reader, &n("attn.k.weight"))?,
        attn_k_bias: zero_bias(n("attn.k.bias"), d_model),
        attn_v_weight: load_named(reader, &n("attn.v.weight"))?,
        attn_v_bias: zero_bias(n("attn.v.bias"), d_model),
        attn_out_weight: load_named(reader, &n("attn.out.weight"))?,
        attn_out_bias: zero_bias(n("attn.out.bias"), d_model),
        attn_pos_weight: load_named(reader, &n("attn.pos.weight"))?,
        attn_pos_bias_u: load_named(reader, &n("attn.pos_bias_u"))?,
        attn_pos_bias_v: load_named(reader, &n("attn.pos_bias_v"))?,
        conv_norm_weight: load_named(reader, &n("conv.norm.weight"))?,
        conv_norm_bias: load_named(reader, &n("conv.norm.bias"))?,
        conv_pw1_weight: load_named(reader, &n("conv.pw1.weight"))?,
        conv_pw1_bias: zero_bias(n("conv.pw1.bias"), 2 * d_model),
        conv_dw_weight: load_named(reader, &n("conv.dw.weight"))?,
        conv_dw_bias: zero_bias(n("conv.dw.bias"), d_model),
        conv_pw2_weight: load_named(reader, &n("conv.pw2.weight"))?,
        conv_pw2_bias: zero_bias(n("conv.pw2.bias"), d_model),
        ff2_norm_weight: load_named(reader, &n("ff2.norm.weight"))?,
        ff2_norm_bias: load_named(reader, &n("ff2.norm.bias"))?,
        ff2_up_weight: load_named(reader, &n("ff2.up.weight"))?,
        ff2_up_bias: zero_bias(n("ff2.up.bias"), ffn),
        ff2_down_weight: load_named(reader, &n("ff2.down.weight"))?,
        ff2_down_bias: zero_bias(n("ff2.down.bias"), d_model),
        out_norm_weight: load_named(reader, &n("out.norm.weight"))?,
        out_norm_bias: load_named(reader, &n("out.norm.bias"))?,
    };
    fold_batchnorm_into_depthwise(reader, layer, &mut weights)?;
    Ok(weights)
}

/// Fold the conv BatchNorm1d into the depthwise weight + (zero-synthesized)
/// bias so the graph runs a plain depthwise conv — identical math to
/// `parakeet_ctc::encoder_weights::fold_batchnorm_into_depthwise`.
fn fold_batchnorm_into_depthwise(
    reader: &GgufTensorDataReader,
    layer: usize,
    weights: &mut ParakeetTdtEncoderLayerWeights,
) -> Result<(), ParakeetTdtWeightsError> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let gamma = load_named(reader, &n("conv.bn.weight"))?;
    let beta = load_named(reader, &n("conv.bn.bias"))?;
    let mean = load_named(reader, &n("conv.bn.mean"))?;
    let var = load_named(reader, &n("conv.bn.var"))?;

    let channels = weights.conv_dw_bias.element_count();
    if gamma.element_count() != channels
        || beta.element_count() != channels
        || mean.element_count() != channels
        || var.element_count() != channels
    {
        return Err(ParakeetTdtWeightsError::BatchNormFold {
            reason: format!(
                "per-channel sizes disagree: dw_bias={channels} gamma={} beta={} mean={} var={}",
                gamma.element_count(),
                beta.element_count(),
                mean.element_count(),
                var.element_count()
            ),
        });
    }
    let dw_elems = weights.conv_dw_weight.element_count();
    if !dw_elems.is_multiple_of(channels) {
        return Err(ParakeetTdtWeightsError::BatchNormFold {
            reason: format!("depthwise weight {dw_elems} not divisible by channels {channels}"),
        });
    }
    let kernel = dw_elems / channels;
    let scale: Vec<f32> = (0..channels)
        .map(|c| gamma.values[c] / (var.values[c] + CONV_BN_EPSILON).sqrt())
        .collect();
    #[allow(clippy::needless_range_loop)]
    for c in 0..channels {
        for k in 0..kernel {
            weights.conv_dw_weight.values[c * kernel + k] *= scale[c];
        }
        weights.conv_dw_bias.values[c] =
            beta.values[c] + (weights.conv_dw_bias.values[c] - mean.values[c]) * scale[c];
    }
    Ok(())
}

/// Drop the f32 host payload of the 2-D linears the encoder graph binds
/// zero-copy (same set as parakeet-ctc plus the joint encoder projection).
fn drop_bound_linear_payloads(layer: &mut ParakeetTdtEncoderLayerWeights) {
    for w in [
        &mut layer.ff1_up_weight,
        &mut layer.ff1_down_weight,
        &mut layer.attn_q_weight,
        &mut layer.attn_k_weight,
        &mut layer.attn_v_weight,
        &mut layer.attn_out_weight,
        &mut layer.attn_pos_weight,
        &mut layer.conv_pw1_weight,
        &mut layer.conv_pw2_weight,
        &mut layer.ff2_up_weight,
        &mut layer.ff2_down_weight,
    ] {
        w.drop_bound_payload();
    }
}

fn drop_bound_subsampling_payloads(subsampling: &mut [NamedTensor]) {
    for weight in subsampling {
        if weight.name == "enc.sub.linear.weight" {
            weight.drop_bound_payload();
        }
    }
}

pub(crate) fn load_parakeet_tdt_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtEncoderWeights, ParakeetTdtWeightsError> {
    let mut subsampling = Vec::new();
    for sub_layer in [0usize, 2, 3, 5, 6] {
        for kind in ["weight", "bias"] {
            let name = format!("enc.sub.layers.{sub_layer}.{kind}");
            if reader.tensor_index().get(&name).is_some() {
                subsampling.push(load_named(reader, &name)?);
            }
        }
    }
    subsampling.push(load_named(reader, "enc.sub.linear.weight")?);
    subsampling.push(load_named(reader, "enc.sub.linear.bias")?);
    drop_bound_subsampling_payloads(&mut subsampling);

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        let mut layer_weights = load_layer(reader, metadata, layer)?;
        drop_bound_linear_payloads(&mut layer_weights);
        layers.push(layer_weights);
    }

    let mut enc_proj_weight = load_named(reader, "enc.proj.weight")?;
    let enc_proj_bias = load_named(reader, "enc.proj.bias")?;
    let expected_proj = metadata.joint_hidden * metadata.hidden_size;
    if enc_proj_weight.element_count() != expected_proj {
        return Err(ParakeetTdtWeightsError::ElementCount {
            name: enc_proj_weight.name.clone(),
            got: enc_proj_weight.element_count(),
            expected: expected_proj,
        });
    }
    if enc_proj_bias.element_count() != metadata.joint_hidden {
        return Err(ParakeetTdtWeightsError::ElementCount {
            name: enc_proj_bias.name.clone(),
            got: enc_proj_bias.element_count(),
            expected: metadata.joint_hidden,
        });
    }
    enc_proj_weight.drop_bound_payload();

    Ok(ParakeetTdtEncoderWeights {
        subsampling,
        layers,
        enc_proj_weight,
        enc_proj_bias,
    })
}

fn expect_elements(
    tensor: NamedTensor,
    expected: usize,
) -> Result<NamedTensor, ParakeetTdtWeightsError> {
    if tensor.element_count() != expected {
        return Err(ParakeetTdtWeightsError::ElementCount {
            name: tensor.name.clone(),
            got: tensor.element_count(),
            expected,
        });
    }
    Ok(tensor)
}

pub(crate) fn load_parakeet_tdt_predictor_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtPredictorWeights, ParakeetTdtWeightsError> {
    let hidden = metadata.pred_hidden;
    let embedding = expect_elements(
        load_named(reader, "dec.embed.weight")?,
        metadata.vocab_size * hidden,
    )?;
    let mut lstm_layers = Vec::with_capacity(metadata.pred_layers);
    for layer in 0..metadata.pred_layers {
        let n = |suffix: &str| format!("dec.lstm.{layer}.{suffix}");
        lstm_layers.push(ParakeetTdtLstmLayerWeights {
            w_ih: expect_elements(load_named(reader, &n("w_ih"))?, 4 * hidden * hidden)?,
            w_hh: expect_elements(load_named(reader, &n("w_hh"))?, 4 * hidden * hidden)?,
            b_ih: expect_elements(load_named(reader, &n("b_ih"))?, 4 * hidden)?,
            b_hh: expect_elements(load_named(reader, &n("b_hh"))?, 4 * hidden)?,
        });
    }
    Ok(ParakeetTdtPredictorWeights {
        embedding,
        lstm_layers,
    })
}

pub(crate) fn load_parakeet_tdt_joint_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtJointWeights, ParakeetTdtWeightsError> {
    let joint = metadata.joint_hidden;
    let out_rows = metadata.vocab_size + metadata.n_durations;
    Ok(ParakeetTdtJointWeights {
        pred_weight: expect_elements(
            load_named(reader, "joint.pred.weight")?,
            joint * metadata.pred_hidden,
        )?,
        pred_bias: expect_elements(load_named(reader, "joint.pred.bias")?, joint)?,
        out_weight: expect_elements(load_named(reader, "joint.out.weight")?, out_rows * joint)?,
        out_bias: expect_elements(load_named(reader, "joint.out.bias")?, out_rows)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::parakeet_tdt::runtime_contract::parse_parakeet_tdt_execution_metadata;
    use std::path::Path;

    fn pack_path() -> Option<std::path::PathBuf> {
        [Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr",
        )]
        .into_iter()
        .find(|p| p.exists())
    }

    #[test]
    fn loads_tdt_weights_with_synthesized_zero_biases_when_pack_present() {
        let Some(path) = pack_path() else {
            eprintln!("skipping: parakeet-tdt-0.6b-v3 pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_parakeet_tdt_execution_metadata(&gguf_metadata).expect("metadata");
        assert_eq!(metadata.n_layers, 24);
        assert_eq!(metadata.n_mels, 128);

        let weights = load_parakeet_tdt_encoder_weights(&reader, &metadata).expect("weights");
        assert_eq!(weights.layers.len(), 24);
        let l0 = &weights.layers[0];
        // Bias-free checkpoint: synthesized zero biases with the right widths.
        assert_eq!(l0.attn_q_bias.values, vec![0.0; 1024]);
        assert_eq!(l0.ff1_up_bias.values.len(), 4096);
        assert!(l0.ff1_up_bias.values.iter().all(|&v| v == 0.0));
        assert_eq!(l0.conv_pw1_bias.values.len(), 2048);
        // BN fold ran over the synthesized dw bias: beta - mean*scale is NOT
        // all-zero for a trained BN.
        assert!(l0.conv_dw_bias.values.iter().any(|&v| v != 0.0));
        // Bound linears dropped their payloads.
        assert!(l0.ff1_up_weight.values.is_empty());
        assert_eq!(l0.ff1_up_weight.dims.iter().product::<usize>(), 4096 * 1024);
        // Joint encoder projection present + bound.
        assert!(weights.enc_proj_weight.values.is_empty());
        assert_eq!(
            weights.enc_proj_weight.dims.iter().product::<usize>(),
            1024 * 640
        );
        assert_eq!(weights.enc_proj_bias.element_count(), 640);

        let predictor = load_parakeet_tdt_predictor_weights(&reader, &metadata).expect("pred");
        assert_eq!(predictor.embedding.element_count(), 8193 * 640);
        assert_eq!(predictor.lstm_layers.len(), 2);
        assert_eq!(predictor.lstm_layers[0].w_ih.element_count(), 4 * 640 * 640);
        // NeMo Embedding(padding_idx=blank): the blank row must be ~zero.
        let blank = metadata.blank_token_id as usize;
        let hidden = metadata.pred_hidden;
        let row = &predictor.embedding.values[blank * hidden..(blank + 1) * hidden];
        assert!(
            row.iter().all(|v| v.abs() < 1.0e-6),
            "blank embedding row must be zeros (padding_idx)"
        );

        let joint = load_parakeet_tdt_joint_weights(&reader, &metadata).expect("joint");
        assert_eq!(joint.out_bias.element_count(), 8193 + 5);
        assert_eq!(joint.out_weight.element_count(), (8193 + 5) * 640);
    }
}
