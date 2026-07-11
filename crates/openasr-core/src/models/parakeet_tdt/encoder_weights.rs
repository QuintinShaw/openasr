//! Load a parakeet-tdt `.oasr` pack into host weights: the FastConformer
//! encoder stack (the shared `models::fastconformer::weights` skeleton --
//! BatchNorm fold + zero-bias synthesis for the checkpoint's missing
//! attn/conv/FFN biases -- `parakeet_ctc::encoder_weights` also builds on),
//! the encoder joint projection, and the host-side prediction-network /
//! joint tensors (parakeet-tdt-only, no `parakeet_ctc` equivalent).
//!
//! The v3 checkpoint has NO attention/conv/FFN biases (`attention_bias` /
//! `convolution_bias` false), so the shared loader synthesizes zero biases
//! for the shared `nn::encoder::conformer_block`, which is bias-shaped. Zero
//! biases are mathematically identity -- nothing model-specific is fabricated.

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::fastconformer::{self, FastConformerLayerWeights, FastConformerWeightsError};
// Re-exported (not just imported) so `parakeet_tdt::greedy`/`predictor` --
// which construct `NamedTensor` values directly in their own tests -- can
// keep referring to it as `encoder_weights::NamedTensor`, unchanged by the
// type's move into the shared `fastconformer` module.
pub(crate) use crate::models::fastconformer::NamedTensor;

use super::runtime_contract::ParakeetTdtExecutionMetadata;

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

impl FastConformerWeightsError for ParakeetTdtWeightsError {
    fn batchnorm_fold(reason: String) -> Self {
        Self::BatchNormFold { reason }
    }
}

/// v3 ships no attn/conv/FFN bias tensors at all -- the shared loader
/// synthesizes zero biases of the right width for every layer.
pub(crate) type ParakeetTdtEncoderLayerWeights = FastConformerLayerWeights;

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

pub(crate) fn load_parakeet_tdt_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtEncoderWeights, ParakeetTdtWeightsError> {
    let subsampling =
        fastconformer::load_fastconformer_subsampling::<ParakeetTdtWeightsError>(reader)?;

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        // bias_present = false: v3 ships no attn/conv/FFN bias tensors; the
        // shared loader synthesizes zero biases of the right width instead.
        layers.push(fastconformer::load_fastconformer_layer::<
            ParakeetTdtWeightsError,
        >(
            reader,
            layer,
            metadata.hidden_size,
            metadata.ffn_dim,
            false,
        )?);
    }

    let mut enc_proj_weight: NamedTensor =
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "enc.proj.weight")?;
    let enc_proj_bias: NamedTensor =
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "enc.proj.bias")?;
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

pub(crate) fn load_parakeet_tdt_predictor_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtPredictorWeights, ParakeetTdtWeightsError> {
    let hidden = metadata.pred_hidden;
    let embedding = expect_elements(
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "dec.embed.weight")?,
        metadata.vocab_size * hidden,
    )?;
    let mut lstm_layers = Vec::with_capacity(metadata.pred_layers);
    for layer in 0..metadata.pred_layers {
        let n = |suffix: &str| format!("dec.lstm.{layer}.{suffix}");
        lstm_layers.push(ParakeetTdtLstmLayerWeights {
            w_ih: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &n("w_ih"))?,
                4 * hidden * hidden,
            )?,
            w_hh: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &n("w_hh"))?,
                4 * hidden * hidden,
            )?,
            b_ih: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &n("b_ih"))?,
                4 * hidden,
            )?,
            b_hh: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &n("b_hh"))?,
                4 * hidden,
            )?,
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
            fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "joint.pred.weight")?,
            joint * metadata.pred_hidden,
        )?,
        pred_bias: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "joint.pred.bias")?,
            joint,
        )?,
        out_weight: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "joint.out.weight")?,
            out_rows * joint,
        )?,
        out_bias: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(reader, "joint.out.bias")?,
            out_rows,
        )?,
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
