//! Load a parakeet-ctc `.oasr` pack into host weights + fold the conv BatchNorm.
//!
//! Every tensor is read generically (dims from the GGUF index, values
//! dequantized to f32) so we never hand-guess the stored dim convention — the
//! encoder graph (S3) reshapes each weight to what `nn::encoder::conformer_block`
//! expects from its element layout. The per-layer set mirrors `ConformerBlockWeights`
//! plus the dw-striding subsampling prelude + the CTC head; the BatchNorm1d
//! tensors are folded into the depthwise weight/bias at load (eps 1e-5, same as
//! cohere / HF default).

// Consumed by the encoder graph + executor wired in S3c/S4; tested meanwhile.
#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::ParakeetCtcExecutionMetadata;

const CONV_BN_EPSILON: f32 = 1.0e-5;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetEncoderWeightsError {
    #[error("parakeet-ctc encoder weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("parakeet-ctc encoder tensor '{name}' has {got} elements, expected {expected}")]
    ElementCount {
        name: String,
        got: usize,
        expected: usize,
    },
    #[error("parakeet-ctc encoder conv BatchNorm fold failed: {reason}")]
    BatchNormFold { reason: String },
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

    /// Drop the resident f32 host `values` (keeping name + dims) for a weight the
    /// encoder graph binds zero-copy from the mmap'd pack. Mirrors qwen's
    /// `dropped_projection_payload`: the dequantized f32 copy is dead weight for a
    /// bound tensor (its bytes come straight from the pack at `mul_mat`), and is
    /// the dominant peak-RSS term for the q4_K parakeet pack. Shape validators
    /// read `dims`, not `values`, and the encoder's `alloc_layer` binds these by
    /// name only — so the empty `values` is never observed.
    fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetEncoderLayerWeights {
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
    /// BatchNorm folded into these two at load (so the graph sees a plain dw conv).
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
pub(crate) struct ParakeetEncoderWeights {
    /// dw-striding subsampling conv2d/linear tensors, keyed by their `enc.sub.*`
    /// suffix (e.g. `layers.0.weight`, `linear.weight`).
    pub subsampling: Vec<NamedTensor>,
    pub layers: Vec<ParakeetEncoderLayerWeights>,
    pub ctc_head_weight: NamedTensor,
    pub ctc_head_bias: NamedTensor,
}

fn load_named(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, ParakeetEncoderWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        ParakeetEncoderWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
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

fn load_layer(
    reader: &GgufTensorDataReader,
    layer: usize,
) -> Result<ParakeetEncoderLayerWeights, ParakeetEncoderWeightsError> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let mut weights = ParakeetEncoderLayerWeights {
        ff1_norm_weight: load_named(reader, &n("ff1.norm.weight"))?,
        ff1_norm_bias: load_named(reader, &n("ff1.norm.bias"))?,
        ff1_up_weight: load_named(reader, &n("ff1.up.weight"))?,
        ff1_up_bias: load_named(reader, &n("ff1.up.bias"))?,
        ff1_down_weight: load_named(reader, &n("ff1.down.weight"))?,
        ff1_down_bias: load_named(reader, &n("ff1.down.bias"))?,
        attn_norm_weight: load_named(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, &n("attn.norm.bias"))?,
        attn_q_weight: load_named(reader, &n("attn.q.weight"))?,
        attn_q_bias: load_named(reader, &n("attn.q.bias"))?,
        attn_k_weight: load_named(reader, &n("attn.k.weight"))?,
        attn_k_bias: load_named(reader, &n("attn.k.bias"))?,
        attn_v_weight: load_named(reader, &n("attn.v.weight"))?,
        attn_v_bias: load_named(reader, &n("attn.v.bias"))?,
        attn_out_weight: load_named(reader, &n("attn.out.weight"))?,
        attn_out_bias: load_named(reader, &n("attn.out.bias"))?,
        attn_pos_weight: load_named(reader, &n("attn.pos.weight"))?,
        attn_pos_bias_u: load_named(reader, &n("attn.pos_bias_u"))?,
        attn_pos_bias_v: load_named(reader, &n("attn.pos_bias_v"))?,
        conv_norm_weight: load_named(reader, &n("conv.norm.weight"))?,
        conv_norm_bias: load_named(reader, &n("conv.norm.bias"))?,
        conv_pw1_weight: load_named(reader, &n("conv.pw1.weight"))?,
        conv_pw1_bias: load_named(reader, &n("conv.pw1.bias"))?,
        conv_dw_weight: load_named(reader, &n("conv.dw.weight"))?,
        conv_dw_bias: load_named(reader, &n("conv.dw.bias"))?,
        conv_pw2_weight: load_named(reader, &n("conv.pw2.weight"))?,
        conv_pw2_bias: load_named(reader, &n("conv.pw2.bias"))?,
        ff2_norm_weight: load_named(reader, &n("ff2.norm.weight"))?,
        ff2_norm_bias: load_named(reader, &n("ff2.norm.bias"))?,
        ff2_up_weight: load_named(reader, &n("ff2.up.weight"))?,
        ff2_up_bias: load_named(reader, &n("ff2.up.bias"))?,
        ff2_down_weight: load_named(reader, &n("ff2.down.weight"))?,
        ff2_down_bias: load_named(reader, &n("ff2.down.bias"))?,
        out_norm_weight: load_named(reader, &n("out.norm.weight"))?,
        out_norm_bias: load_named(reader, &n("out.norm.bias"))?,
    };
    fold_batchnorm_into_depthwise(reader, layer, &mut weights)?;
    Ok(weights)
}

/// Fold the conv BatchNorm1d (`conv.bn.{weight,bias,mean,var}`) into the depthwise
/// weight + bias so the graph runs a plain depthwise conv. Channel-major layout
/// (the depthwise weight is `[channels, 1, kernel]` C-order = `channel*kernel + k`).
fn fold_batchnorm_into_depthwise(
    reader: &GgufTensorDataReader,
    layer: usize,
    weights: &mut ParakeetEncoderLayerWeights,
) -> Result<(), ParakeetEncoderWeightsError> {
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
        return Err(ParakeetEncoderWeightsError::BatchNormFold {
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
        return Err(ParakeetEncoderWeightsError::BatchNormFold {
            reason: format!("depthwise weight {dw_elems} not divisible by channels {channels}"),
        });
    }
    let kernel = dw_elems / channels;
    let scale: Vec<f32> = (0..channels)
        .map(|c| gamma.values[c] / (var.values[c] + CONV_BN_EPSILON).sqrt())
        .collect();
    // `c` indexes several parallel per-channel arrays (scale, beta/mean, the dw
    // kernel rows + the dw bias), so a range loop is clearest here.
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

/// Drop the f32 host payload of the 2-D linears the encoder graph binds zero-copy
/// (the q4_K/f32 `ff*`, `attn.{q,k,v,out,pos}`, `conv.pw{1,2}` projections). Their
/// `dims` are retained for the binder + validators; only the dequantized values
/// are freed. Norms, biases, and the BN-folded depthwise conv are NOT dropped
/// (the graph still arena-uploads them from these `values`).
fn drop_bound_linear_payloads(layer: &mut ParakeetEncoderLayerWeights) {
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

pub(crate) fn load_parakeet_ctc_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetCtcExecutionMetadata,
) -> Result<ParakeetEncoderWeights, ParakeetEncoderWeightsError> {
    // dw-striding subsampling tensors: enc.sub.layers.{0,2,3,5,6}.{weight,bias}
    // + enc.sub.linear.{weight,bias}. Load whichever of these are present.
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
        let mut layer_weights = load_layer(reader, layer)?;
        // The encoder graph binds these 2-D linears zero-copy from the mmap'd
        // pack (the packer stored them `[in,out]`-native), so drop their f32 host
        // copy — the dominant peak-RSS term. Norms/biases + the BN-folded
        // depthwise conv keep their `values` (still arena-uploaded).
        drop_bound_linear_payloads(&mut layer_weights);
        layers.push(layer_weights);
    }

    let mut ctc_head_weight = load_named(reader, "ctc.head.weight")?;
    let ctc_head_bias = load_named(reader, "ctc.head.bias")?;
    let expected_head = metadata.vocab_size * metadata.hidden_size;
    if ctc_head_weight.element_count() != expected_head {
        return Err(ParakeetEncoderWeightsError::ElementCount {
            name: ctc_head_weight.name.clone(),
            got: ctc_head_weight.element_count(),
            expected: expected_head,
        });
    }
    // The CTC head is also bound zero-copy (f16 on disk); drop its f32 copy after
    // the element-count check (which reads `values`).
    ctc_head_weight.drop_bound_payload();

    Ok(ParakeetEncoderWeights {
        subsampling,
        layers,
        ctc_head_weight,
        ctc_head_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::parakeet_ctc::runtime_contract::parse_parakeet_ctc_execution_metadata;
    use std::path::Path;

    fn pack_path() -> Option<std::path::PathBuf> {
        // Resolve the worktree-relative pack; tmp/ is gitignored, so the test
        // skips when it is absent.
        [Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-ctc-0.6b/openasr/parakeet-ctc-0.6b-fp16.oasr")]
        .into_iter()
        .find(|p| p.exists())
    }

    #[test]
    fn loads_parakeet_encoder_weights_and_folds_batchnorm_when_pack_present() {
        let Some(path) = pack_path() else {
            eprintln!("skipping: parakeet-ctc-0.6b pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_parakeet_ctc_execution_metadata(&gguf_metadata).expect("metadata");
        assert_eq!(metadata.n_layers, 24);

        let weights = load_parakeet_ctc_encoder_weights(&reader, &metadata).expect("weights");
        assert_eq!(weights.layers.len(), 24);
        // The bound 2-D linears keep their `dims` but drop their f32 `values`
        // (bound zero-copy from the pack): assert via the dims product. ff1.up =
        // [in 1024, out 4096]; attn.q = [1024, 1024].
        let l0 = &weights.layers[0];
        let dims_product = |t: &NamedTensor| t.dims.iter().product::<usize>();
        assert_eq!(dims_product(&l0.ff1_up_weight), 4096 * 1024);
        assert!(
            l0.ff1_up_weight.values.is_empty(),
            "bound linear payload must be dropped"
        );
        assert_eq!(dims_product(&l0.attn_q_weight), 1024 * 1024);
        // Arena weights (kept): pos_bias + the BN-folded depthwise conv.
        assert_eq!(l0.attn_pos_bias_u.element_count(), 8 * 128);
        assert_eq!(
            l0.conv_dw_weight.element_count(),
            1024 * metadata.conv_kernel
        );
        // CTC head present + correctly sized (bound: dims kept, values dropped).
        assert_eq!(
            dims_product(&weights.ctc_head_weight),
            metadata.vocab_size * metadata.hidden_size
        );
        assert!(weights.ctc_head_weight.values.is_empty());
        assert_eq!(weights.ctc_head_bias.element_count(), metadata.vocab_size);
        // Subsampling: 3 conv stages (layers 0/2/3/5/6) + linear, all present.
        assert!(
            weights
                .subsampling
                .iter()
                .any(|t| t.name == "enc.sub.layers.0.weight")
        );
        let sub_linear = weights
            .subsampling
            .iter()
            .find(|t| t.name == "enc.sub.linear.weight")
            .expect("subsampling linear");
        assert!(sub_linear.values.is_empty());
    }

    #[test]
    fn drops_subsampling_linear_payload_only() {
        let mut subsampling = vec![
            NamedTensor {
                name: "enc.sub.linear.weight".to_string(),
                dims: vec![2, 3],
                values: vec![1.0; 6],
            },
            NamedTensor {
                name: "enc.sub.linear.bias".to_string(),
                dims: vec![3],
                values: vec![1.0; 3],
            },
            NamedTensor {
                name: "enc.sub.layers.0.weight".to_string(),
                dims: vec![3, 3, 1, 256],
                values: vec![1.0; 9],
            },
        ];

        drop_bound_subsampling_payloads(&mut subsampling);

        assert!(subsampling[0].values.is_empty());
        assert_eq!(subsampling[1].values.len(), 3);
        assert_eq!(subsampling[2].values.len(), 9);
    }
}
