//! Load a parakeet-ctc `.oasr` pack into host weights + fold the conv BatchNorm.
//!
//! Every tensor is read generically (dims from the GGUF index, values
//! dequantized to f32) so we never hand-guess the stored dim convention — the
//! encoder graph (S3) reshapes each weight to what `nn::encoder::conformer_block`
//! expects from its element layout. The dw-striding subsampling + per-layer
//! conformer weights (mirroring `ConformerBlockWeights`) + BatchNorm fold are
//! the shared `models::fastconformer::weights` skeleton parakeet-tdt also
//! uses; this module adds only the CTC-head tail, which parakeet-tdt has no
//! equivalent of.

// Consumed by the encoder graph + executor wired in S3c/S4; tested meanwhile.
#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::fastconformer::{self, FastConformerLayerWeights, FastConformerWeightsError};
// Re-exported so other parakeet-ctc modules can keep referring to it as
// `encoder_weights::NamedTensor`, unchanged by the type's move into the
// shared `fastconformer` module.
pub(crate) use crate::models::fastconformer::NamedTensor;

use super::runtime_contract::ParakeetCtcExecutionMetadata;

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

impl FastConformerWeightsError for ParakeetEncoderWeightsError {
    fn batchnorm_fold(reason: String) -> Self {
        Self::BatchNormFold { reason }
    }
}

/// The parakeet-ctc checkpoint ships every conformer bias tensor (no
/// bias-free NeMo/HF conversion, unlike parakeet-tdt-0.6b-v3).
pub(crate) type ParakeetEncoderLayerWeights = FastConformerLayerWeights;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetEncoderWeights {
    /// dw-striding subsampling conv2d/linear tensors, keyed by their `enc.sub.*`
    /// suffix (e.g. `layers.0.weight`, `linear.weight`).
    pub subsampling: Vec<NamedTensor>,
    pub layers: Vec<ParakeetEncoderLayerWeights>,
    pub ctc_head_weight: NamedTensor,
    pub ctc_head_bias: NamedTensor,
}

pub(crate) fn load_parakeet_ctc_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetCtcExecutionMetadata,
) -> Result<ParakeetEncoderWeights, ParakeetEncoderWeightsError> {
    let subsampling =
        fastconformer::load_fastconformer_subsampling::<ParakeetEncoderWeightsError>(reader)?;

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        // bias_present = true: every attn/conv/FFN bias tensor is on disk.
        layers.push(fastconformer::load_fastconformer_layer::<
            ParakeetEncoderWeightsError,
        >(
            reader,
            layer,
            metadata.hidden_size,
            metadata.ffn_dim,
            true,
        )?);
    }

    let mut ctc_head_weight: NamedTensor =
        fastconformer::load_named::<ParakeetEncoderWeightsError>(reader, "ctc.head.weight")?;
    let ctc_head_bias: NamedTensor =
        fastconformer::load_named::<ParakeetEncoderWeightsError>(reader, "ctc.head.bias")?;
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
}
