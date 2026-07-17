//! X-ASR GGUF tensor loading helpers.

use crate::ggml_runtime::{
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
};

use super::package_import::compact_xasr_name;
use super::runtime_contract::XasrZipformerExecutionMetadata;

#[derive(Debug, thiserror::Error)]
pub(crate) enum XasrWeightsError {
    #[error("xasr-zipformer tensor read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("xasr-zipformer tensor '{name}' has rank {rank}, expected {expected_rank}")]
    Rank {
        name: String,
        rank: usize,
        expected_rank: usize,
    },
    #[error("xasr-zipformer tensor '{name}' dims {dims:?} do not match {expected:?}")]
    Dims {
        name: String,
        dims: Vec<usize>,
        expected: Vec<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NamedTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredLinear {
    pub name: String,
    /// GGML `ne0`; storage is output-major with this many input values.
    pub input_dim: usize,
    /// GGML `ne1`; number of output rows.
    pub output_dim: usize,
    /// Dequantized row-major f32 weights, used by the per-symbol decoder/joiner
    /// `apply`/`apply_into` matvecs and by the f32 test graphs. Empty when the
    /// weight is served natively (`native` is `Some`) to the encoder ggml graph.
    pub values: Vec<f32>,
    /// Native (quantized / f16) ggml block payload for an encoder `mul_mat`
    /// weight operand. `Some` keeps the weight quantized end to end: the encoder
    /// graph binds it at its stored ggml type and uploads the raw blocks verbatim
    /// (mmap-backed, no dequant-to-f32 blow-up), so peak RSS orders q4 < q8 < fp16.
    /// `None` means the dequantized `values` path (decoder/joiner matvecs, tests).
    pub native: Option<GgufOwnedWeightTensorPayload>,
}

impl StoredLinear {
    pub(crate) fn apply(&self, input: &[f32], bias: Option<&[f32]>) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0_f32; self.output_dim];
        self.apply_into(input, bias, &mut output)?;
        Ok(output)
    }

    /// Matrix-vector product into a caller-owned buffer. The greedy RNN-T loop
    /// calls this thousands of times per utterance, so the kernel uses four
    /// independent accumulators (breaking the FP-add latency chain so the
    /// compiler can keep multiple SIMD FMAs in flight) and avoids allocating.
    pub(crate) fn apply_into(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) -> Result<(), String> {
        if input.len() != self.input_dim {
            return Err(format!(
                "linear '{}' expected input dim {}, got {}",
                self.name,
                self.input_dim,
                input.len()
            ));
        }
        if let Some(bias) = bias
            && bias.len() != self.output_dim
        {
            return Err(format!(
                "linear '{}' expected bias dim {}, got {}",
                self.name,
                self.output_dim,
                bias.len()
            ));
        }
        if output.len() != self.output_dim {
            return Err(format!(
                "linear '{}' expected output dim {}, got buffer of {}",
                self.name,
                self.output_dim,
                output.len()
            ));
        }
        if self.native.is_some() && self.values.is_empty() {
            return Err(format!(
                "linear '{}' is native-only; no dequantized values for CPU matvec",
                self.name
            ));
        }
        for (out_idx, out) in output.iter_mut().enumerate() {
            let row = &self.values[out_idx * self.input_dim..(out_idx + 1) * self.input_dim];
            *out = dot_f32(input, row) + bias.map_or(0.0, |bias| bias[out_idx]);
        }
        Ok(())
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut i = 0usize;
    while i + 16 <= n {
        let ca = &a[i..i + 16];
        let cb = &b[i..i + 16];
        s0 += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
        s1 += ca[4] * cb[4] + ca[5] * cb[5] + ca[6] * cb[6] + ca[7] * cb[7];
        s2 += ca[8] * cb[8] + ca[9] * cb[9] + ca[10] * cb[10] + ca[11] * cb[11];
        s3 += ca[12] * cb[12] + ca[13] * cb[13] + ca[14] * cb[14] + ca[15] * cb[15];
        i += 16;
    }
    let mut tail = 0.0_f32;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3) + tail
}

#[derive(Debug, Clone)]
pub(crate) struct XasrDecoderWeights {
    pub embedding: StoredLinear,
    pub conv_weight: NamedTensor,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct XasrJoinerWeights {
    pub encoder_proj_weight: StoredLinear,
    pub encoder_proj_bias: Vec<f32>,
    pub decoder_proj_weight: StoredLinear,
    pub decoder_proj_bias: Vec<f32>,
    pub output_linear_weight: StoredLinear,
    pub output_linear_bias: Vec<f32>,
}

pub(crate) fn load_xasr_decoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrDecoderWeights, XasrWeightsError> {
    let embedding = load_linear(
        reader,
        "decoder.embedding.weight",
        metadata.decoder_dim(),
        metadata.vocab_size,
    )?;
    let conv_weight = load_named(reader, "decoder.conv.weight")?;
    assert_rank(&conv_weight, 3)?;
    let expected = vec![
        metadata.decoder_context_size,
        metadata.decoder_dim() / 128,
        metadata.decoder_dim(),
    ];
    if conv_weight.dims != expected {
        return Err(XasrWeightsError::Dims {
            name: conv_weight.name,
            dims: conv_weight.dims,
            expected,
        });
    }
    Ok(XasrDecoderWeights {
        embedding,
        conv_weight,
        groups: 128,
    })
}

pub(crate) fn load_xasr_joiner_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrJoinerWeights, XasrWeightsError> {
    let encoder_output_dim = metadata.encoder_output_dim();
    let joiner_dim = metadata.joiner_dim;
    let vocab_size = metadata.vocab_size;
    Ok(XasrJoinerWeights {
        encoder_proj_weight: load_linear(
            reader,
            "joiner.encoder_proj.weight",
            encoder_output_dim,
            joiner_dim,
        )?,
        encoder_proj_bias: load_vector(reader, "joiner.encoder_proj.bias", joiner_dim)?,
        decoder_proj_weight: load_linear(
            reader,
            "joiner.decoder_proj.weight",
            joiner_dim,
            joiner_dim,
        )?,
        decoder_proj_bias: load_vector(reader, "joiner.decoder_proj.bias", joiner_dim)?,
        output_linear_weight: load_linear(
            reader,
            "joiner.output_linear.weight",
            joiner_dim,
            vocab_size,
        )?,
        output_linear_bias: load_vector(reader, "joiner.output_linear.bias", vocab_size)?,
    })
}

pub(crate) fn load_named(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
) -> Result<NamedTensor, XasrWeightsError> {
    let name = compact_xasr_name(upstream_name);
    let tensor = reader.tensor_index().get(&name).ok_or_else(|| {
        XasrWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
            path: reader.tensor_index().path().to_path_buf(),
            tensor_name: name.clone(),
        })
    })?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let shape_u64 = tensor.dims.clone();
    let values = reader.host_tensor_f32_copy_dequantized_by_name(&name, &shape_u64)?;
    Ok(NamedTensor { name, dims, values })
}

pub(crate) fn load_vector(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
    len: usize,
) -> Result<Vec<f32>, XasrWeightsError> {
    let tensor = load_named(reader, upstream_name)?;
    assert_rank(&tensor, 1)?;
    if tensor.dims != [len] {
        return Err(XasrWeightsError::Dims {
            name: tensor.name,
            dims: tensor.dims,
            expected: vec![len],
        });
    }
    Ok(tensor.values)
}

pub(crate) fn load_linear(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<StoredLinear, XasrWeightsError> {
    let tensor = load_named(reader, upstream_name)?;
    assert_rank(&tensor, 2)?;
    let expected = vec![input_dim, output_dim];
    if tensor.dims != expected {
        return Err(XasrWeightsError::Dims {
            name: tensor.name,
            dims: tensor.dims,
            expected,
        });
    }
    Ok(StoredLinear {
        name: tensor.name,
        input_dim,
        output_dim,
        values: tensor.values,
        native: None,
    })
}

/// Load a rank-2 `.weight` projection as its native (quantized / f16) ggml block
/// payload for an encoder `mul_mat` weight operand, instead of dequantizing to
/// f32. The payload carries the pack mmap (zero-copy) and its stored element type
/// (F16 / Q8_0 / Q4_K, decided per tensor by the importer); the encoder graph
/// binds it at that type and feeds `mul_mat` directly, so the weight stays
/// quantized in the backend buffer. Only the encoder linears (which run in the
/// ggml graph) use this; the per-symbol decoder/joiner matvecs keep the
/// dequantized f32 `values` path via [`load_linear`].
pub(crate) fn load_native_linear(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<StoredLinear, XasrWeightsError> {
    let stored = load_native_linear_by_actual_dims(reader, upstream_name)?;
    if stored.input_dim != input_dim || stored.output_dim != output_dim {
        return Err(XasrWeightsError::Dims {
            name: stored.name,
            dims: vec![stored.input_dim, stored.output_dim],
            expected: vec![input_dim, output_dim],
        });
    }
    Ok(stored)
}

/// Like [`load_native_linear`] but takes the rank-2 dims from the stored tensor
/// (`ne0` = input, `ne1` = output) rather than validating against expected dims --
/// used for `linear_pos`, whose output width is derived from the pack.
pub(crate) fn load_native_linear_by_actual_dims(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
) -> Result<StoredLinear, XasrWeightsError> {
    let name = compact_xasr_name(upstream_name);
    let payload = reader.owned_weight_tensor_payload_by_name(&name)?;
    let [input_dim, output_dim]: [usize; 2] =
        payload
            .dims
            .as_slice()
            .try_into()
            .map_err(|_| XasrWeightsError::Rank {
                name: name.clone(),
                rank: payload.dims.len(),
                expected_rank: 2,
            })?;
    Ok(StoredLinear {
        name,
        input_dim,
        output_dim,
        values: Vec::new(),
        native: Some(payload),
    })
}

fn assert_rank(tensor: &NamedTensor, expected_rank: usize) -> Result<(), XasrWeightsError> {
    if tensor.dims.len() == expected_rank {
        return Ok(());
    }
    Err(XasrWeightsError::Rank {
        name: tensor.name.clone(),
        rank: tensor.dims.len(),
        expected_rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_linear_applies_output_major_weight() {
        let weight = StoredLinear {
            name: "w".to_string(),
            input_dim: 3,
            output_dim: 2,
            values: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            native: None,
        };
        let output = weight.apply(&[1.0, 1.0, 1.0], Some(&[0.5, -1.0])).unwrap();
        assert_eq!(output, vec![6.5, 59.0]);
    }
}
