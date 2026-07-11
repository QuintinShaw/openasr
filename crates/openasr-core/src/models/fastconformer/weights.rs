//! Shared FastConformer encoder weight loading: generic GGUF tensor reads,
//! the conv BatchNorm1d fold, zero-bias synthesis for bias-free checkpoints,
//! and the bound-linear payload drops that keep peak RSS down. Carried over
//! byte-for-byte from `parakeet_ctc::encoder_weights` /
//! `parakeet_tdt::encoder_weights` (which were identical modulo TDT's
//! zero-bias synthesis and error-type names).

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::FastConformerWeightsError;

const CONV_BN_EPSILON: f32 = 1.0e-5;

/// A host weight: its stored dims (from the GGUF index) + dequantized f32
/// values. Bias slots a bias-free checkpoint does not ship are synthesized
/// as zeros (see [`zero_bias`]); shape validators read `dims`, not `values`.
#[derive(Debug, Clone)]
pub(crate) struct NamedTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

impl NamedTensor {
    pub(crate) fn element_count(&self) -> usize {
        self.values.len()
    }

    /// Drop the resident f32 host `values` (keeping name + dims) for a weight
    /// the encoder graph binds zero-copy from the mmap'd pack. The
    /// dequantized f32 copy is dead weight for a bound tensor (its bytes come
    /// straight from the pack at `mul_mat`), and is the dominant peak-RSS
    /// term for a q4_K parakeet pack. The encoder's `alloc_layer` binds these
    /// by name only, so the empty `values` is never observed. `pub(crate)`
    /// so each family can drop its own tail tensor's payload (CTC head /
    /// joint encoder projection) the same way.
    pub(crate) fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

/// Per-layer FastConformer weights: the same field set both parakeet-ctc and
/// parakeet-tdt load, mirroring `nn::encoder::ConformerBlockWeights`. Norms,
/// biases, and the BN-folded depthwise conv stay plain arena tensors in the
/// graph; the 2-D linears (`ff*.{up,down}`, `attn.{q,k,v,out,pos}`,
/// `conv.pw{1,2}`) are bound zero-copy from the mmap'd pack.
#[derive(Debug, Clone)]
pub(crate) struct FastConformerLayerWeights {
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

pub(crate) fn load_named<E: FastConformerWeightsError>(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, E> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        E::from(GgufTensorDataReadError::TensorNotFound {
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

/// Zero bias for a projection a bias-free checkpoint does not ship (NeMo/HF
/// `attention_bias`/`convolution_bias`/FFN-bias all false, e.g.
/// parakeet-tdt-0.6b-v3). The shared conformer block is bias-shaped; a zero
/// bias is the mathematical identity, nothing model-specific is fabricated.
fn zero_bias(name: String, len: usize) -> NamedTensor {
    NamedTensor {
        name,
        dims: vec![len],
        values: vec![0.0; len],
    }
}

/// Load one FastConformer conformer-block layer's weights + fold its conv
/// BatchNorm. `bias_present` is the family's "does the checkpoint ship
/// attention/conv/FFN biases" knob: `true` loads every bias tensor from the
/// pack (parakeet-ctc); `false` synthesizes zero biases of the right width
/// (parakeet-tdt-0.6b-v3, whose HF conversion has no bias tensors at all).
pub(crate) fn load_fastconformer_layer<E: FastConformerWeightsError>(
    reader: &GgufTensorDataReader,
    layer: usize,
    d_model: usize,
    ffn_dim: usize,
    bias_present: bool,
) -> Result<FastConformerLayerWeights, E> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let bias = |suffix: &str, len: usize| -> Result<NamedTensor, E> {
        if bias_present {
            load_named::<E>(reader, &n(suffix))
        } else {
            Ok(zero_bias(n(suffix), len))
        }
    };
    let mut weights = FastConformerLayerWeights {
        ff1_norm_weight: load_named::<E>(reader, &n("ff1.norm.weight"))?,
        ff1_norm_bias: load_named::<E>(reader, &n("ff1.norm.bias"))?,
        ff1_up_weight: load_named::<E>(reader, &n("ff1.up.weight"))?,
        ff1_up_bias: bias("ff1.up.bias", ffn_dim)?,
        ff1_down_weight: load_named::<E>(reader, &n("ff1.down.weight"))?,
        ff1_down_bias: bias("ff1.down.bias", d_model)?,
        attn_norm_weight: load_named::<E>(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named::<E>(reader, &n("attn.norm.bias"))?,
        attn_q_weight: load_named::<E>(reader, &n("attn.q.weight"))?,
        attn_q_bias: bias("attn.q.bias", d_model)?,
        attn_k_weight: load_named::<E>(reader, &n("attn.k.weight"))?,
        attn_k_bias: bias("attn.k.bias", d_model)?,
        attn_v_weight: load_named::<E>(reader, &n("attn.v.weight"))?,
        attn_v_bias: bias("attn.v.bias", d_model)?,
        attn_out_weight: load_named::<E>(reader, &n("attn.out.weight"))?,
        attn_out_bias: bias("attn.out.bias", d_model)?,
        attn_pos_weight: load_named::<E>(reader, &n("attn.pos.weight"))?,
        attn_pos_bias_u: load_named::<E>(reader, &n("attn.pos_bias_u"))?,
        attn_pos_bias_v: load_named::<E>(reader, &n("attn.pos_bias_v"))?,
        conv_norm_weight: load_named::<E>(reader, &n("conv.norm.weight"))?,
        conv_norm_bias: load_named::<E>(reader, &n("conv.norm.bias"))?,
        conv_pw1_weight: load_named::<E>(reader, &n("conv.pw1.weight"))?,
        conv_pw1_bias: bias("conv.pw1.bias", 2 * d_model)?,
        conv_dw_weight: load_named::<E>(reader, &n("conv.dw.weight"))?,
        conv_dw_bias: bias("conv.dw.bias", d_model)?,
        conv_pw2_weight: load_named::<E>(reader, &n("conv.pw2.weight"))?,
        conv_pw2_bias: bias("conv.pw2.bias", d_model)?,
        ff2_norm_weight: load_named::<E>(reader, &n("ff2.norm.weight"))?,
        ff2_norm_bias: load_named::<E>(reader, &n("ff2.norm.bias"))?,
        ff2_up_weight: load_named::<E>(reader, &n("ff2.up.weight"))?,
        ff2_up_bias: bias("ff2.up.bias", ffn_dim)?,
        ff2_down_weight: load_named::<E>(reader, &n("ff2.down.weight"))?,
        ff2_down_bias: bias("ff2.down.bias", d_model)?,
        out_norm_weight: load_named::<E>(reader, &n("out.norm.weight"))?,
        out_norm_bias: load_named::<E>(reader, &n("out.norm.bias"))?,
    };
    fold_batchnorm_into_depthwise::<E>(reader, layer, &mut weights)?;
    drop_bound_layer_linear_payloads(&mut weights);
    Ok(weights)
}

/// Fold the conv BatchNorm1d (`conv.bn.{weight,bias,mean,var}`) into the
/// depthwise weight + bias so the graph runs a plain depthwise conv.
/// Channel-major layout (the depthwise weight is `[channels, 1, kernel]`
/// C-order = `channel*kernel + k`).
pub(crate) fn fold_batchnorm_into_depthwise<E: FastConformerWeightsError>(
    reader: &GgufTensorDataReader,
    layer: usize,
    weights: &mut FastConformerLayerWeights,
) -> Result<(), E> {
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let gamma: NamedTensor = load_named::<E>(reader, &n("conv.bn.weight"))?;
    let beta: NamedTensor = load_named::<E>(reader, &n("conv.bn.bias"))?;
    let mean: NamedTensor = load_named::<E>(reader, &n("conv.bn.mean"))?;
    let var: NamedTensor = load_named::<E>(reader, &n("conv.bn.var"))?;

    let channels = weights.conv_dw_bias.element_count();
    if gamma.element_count() != channels
        || beta.element_count() != channels
        || mean.element_count() != channels
        || var.element_count() != channels
    {
        return Err(E::batchnorm_fold(format!(
            "per-channel sizes disagree: dw_bias={channels} gamma={} beta={} mean={} var={}",
            gamma.element_count(),
            beta.element_count(),
            mean.element_count(),
            var.element_count()
        )));
    }
    let dw_elems = weights.conv_dw_weight.element_count();
    if !dw_elems.is_multiple_of(channels) {
        return Err(E::batchnorm_fold(format!(
            "depthwise weight {dw_elems} not divisible by channels {channels}"
        )));
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

/// Drop the f32 host payload of the 2-D linears the encoder graph binds
/// zero-copy (the q4_K/f32 `ff*`, `attn.{q,k,v,out,pos}`, `conv.pw{1,2}`
/// projections). Their `dims` are retained for the binder + validators; only
/// the dequantized values are freed. Norms, biases, and the BN-folded
/// depthwise conv are NOT dropped (the graph still arena-uploads them).
pub(crate) fn drop_bound_layer_linear_payloads(layer: &mut FastConformerLayerWeights) {
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

pub(crate) fn drop_bound_subsampling_payloads(subsampling: &mut [NamedTensor]) {
    for weight in subsampling {
        if weight.name == "enc.sub.linear.weight" {
            weight.drop_bound_payload();
        }
    }
}

/// Load the dw-striding subsampling prelude tensors: `enc.sub.layers.{0,2,3,5,6}.{weight,bias}`
/// (whichever are present) plus `enc.sub.linear.{weight,bias}`, then drop the
/// linear weight's f32 host payload (bound zero-copy from the mmap'd pack).
pub(crate) fn load_fastconformer_subsampling<E: FastConformerWeightsError>(
    reader: &GgufTensorDataReader,
) -> Result<Vec<NamedTensor>, E> {
    let mut subsampling = Vec::new();
    for sub_layer in [0usize, 2, 3, 5, 6] {
        for kind in ["weight", "bias"] {
            let name = format!("enc.sub.layers.{sub_layer}.{kind}");
            if reader.tensor_index().get(&name).is_some() {
                subsampling.push(load_named::<E>(reader, &name)?);
            }
        }
    }
    subsampling.push(load_named::<E>(reader, "enc.sub.linear.weight")?);
    subsampling.push(load_named::<E>(reader, "enc.sub.linear.bias")?);
    drop_bound_subsampling_payloads(&mut subsampling);
    Ok(subsampling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("read: {0}")]
        Read(#[from] GgufTensorDataReadError),
        #[error("batchnorm fold: {0}")]
        BatchNormFold(String),
    }

    impl FastConformerWeightsError for TestError {
        fn batchnorm_fold(reason: String) -> Self {
            Self::BatchNormFold(reason)
        }
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

    #[test]
    fn zero_bias_is_correct_width_and_zeroed() {
        let t = zero_bias("x".to_string(), 5);
        assert_eq!(t.dims, vec![5]);
        assert_eq!(t.values, vec![0.0; 5]);
    }

    // Compile-only check that the generic loaders are usable with a
    // family-shaped error type (the real integration coverage -- BN fold
    // correctness against a real pack -- stays in each family's own test
    // module, which already asserts on real tensor values).
    #[allow(dead_code)]
    fn _generic_signatures_compile(reader: &GgufTensorDataReader) {
        let _: Result<NamedTensor, TestError> = load_named(reader, "x");
        let _: Result<Vec<NamedTensor>, TestError> = load_fastconformer_subsampling(reader);
        let _: Result<FastConformerLayerWeights, TestError> =
            load_fastconformer_layer(reader, 0, 1024, 4096, true);
    }
}
