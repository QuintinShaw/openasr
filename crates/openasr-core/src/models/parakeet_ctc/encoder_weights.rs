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
    #[error("parakeet-ctc tensor '{name}' is not part of the runtime tensor contract")]
    NotInContract { name: String },
    #[error("parakeet-ctc weight expectation overflowed: {reason}")]
    ExpectationOverflow { reason: String },
}

impl FastConformerWeightsError for ParakeetEncoderWeightsError {
    fn batchnorm_fold(reason: String) -> Self {
        Self::BatchNormFold { reason }
    }
    fn not_in_contract(name: String) -> Self {
        Self::NotInContract { name }
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

impl ParakeetEncoderWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        crate::models::parakeet_runtime_memory::fastconformer_weights_retained_bytes(
            &self.subsampling,
            &self.layers,
            &[&self.ctc_head_weight, &self.ctc_head_bias],
        )
    }
}

/// The read guard for one pack's full parakeet-ctc tensor contract (shared
/// FastConformer encoder plus the CTC head): every tensor the loader reads
/// must be enumerated here.
pub(crate) fn parakeet_ctc_read_guard(
    metadata: &ParakeetCtcExecutionMetadata,
) -> crate::models::tensor_binding::TensorReadGuard {
    crate::models::tensor_binding::TensorReadGuard::from_descriptors(
        &super::runtime_contract::parakeet_ctc_runtime_tensor_binding_descriptors(metadata),
    )
}

pub(crate) fn load_parakeet_ctc_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetCtcExecutionMetadata,
) -> Result<ParakeetEncoderWeights, ParakeetEncoderWeightsError> {
    let guard = parakeet_ctc_read_guard(metadata);
    let subsampling = fastconformer::load_fastconformer_subsampling::<ParakeetEncoderWeightsError>(
        reader, &guard,
    )?;

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        // bias_present = true: every attn/conv/FFN bias tensor is on disk.
        layers.push(fastconformer::load_fastconformer_layer::<
            ParakeetEncoderWeightsError,
        >(
            reader,
            &guard,
            layer,
            metadata.hidden_size,
            metadata.ffn_dim,
            true,
        )?);
    }

    let mut ctc_head_weight: NamedTensor = fastconformer::load_named::<ParakeetEncoderWeightsError>(
        reader,
        &guard,
        "ctc.head.weight",
    )?;
    let ctc_head_bias: NamedTensor =
        fastconformer::load_named::<ParakeetEncoderWeightsError>(reader, &guard, "ctc.head.bias")?;
    let expected_head = metadata
        .vocab_size
        .checked_mul(metadata.hidden_size)
        .ok_or_else(|| ParakeetEncoderWeightsError::ExpectationOverflow {
            reason: "vocab_size * hidden_size overflows".to_string(),
        })?;
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

    /// The equivalence evidence the count-plus-sampling pin used to fake: run
    /// the REAL encoder + CTC-head loader over a synthetic pack projected
    /// from the contract enumeration itself, with the tensor index's access
    /// trace enabled, and assert the traced read set equals the descriptor
    /// set name for name and shape for shape. Any drift -- a loader reading a
    /// tensor the contract does not list, a descriptor no loader reads, or a
    /// read violating the descriptor's shape -- fails here. Also exercises
    /// the read guard: every read is contract-listed.
    #[test]
    fn full_loader_read_trace_equals_the_descriptor_set() {
        use super::super::runtime_contract::{
            parakeet_ctc_runtime_tensor_binding_descriptors, parakeet_ctc_runtime_tensors,
        };
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

        let metadata = super::tests_support::tiny_execution_metadata();
        let shapes = parakeet_ctc_runtime_tensors(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("parakeet-ctc-trace.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in shapes {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write trace pack");

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        load_parakeet_ctc_encoder_weights(&reader, &metadata).expect("full encoder load");

        crate::models::tensor_binding::assert_trace_matches_descriptor_set(
            &reader.tensor_index().access_trace(),
            &parakeet_ctc_runtime_tensor_binding_descriptors(&metadata),
        );
    }

    /// The read guard fails closed on any tensor the contract does not
    /// enumerate, so a loader/name drift cannot read off-contract.
    #[test]
    fn read_guard_rejects_off_contract_tensors() {
        use crate::models::fastconformer::load_named;

        let metadata = super::tests_support::tiny_execution_metadata();
        let guard = parakeet_ctc_read_guard(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("parakeet-ctc-guard.oasr");
        let spec = crate::testing::TinyGgufFixtureSpec::new(std::collections::BTreeMap::new())
            .with_tensor_shape("off.contract.weight", vec![2, 2]);
        crate::testing::write_tiny_gguf_runtime_source(&path, &spec).expect("write pack");
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");

        let error =
            load_named::<ParakeetEncoderWeightsError>(&reader, &guard, "off.contract.weight")
                .expect_err("off-contract reads must fail closed");
        assert!(
            matches!(error, ParakeetEncoderWeightsError::NotInContract { ref name } if name == "off.contract.weight"),
            "unexpected error: {error}"
        );
    }
}

/// Test-only geometry support for this module's weight-free tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::super::runtime_contract::ParakeetCtcExecutionMetadata;

    pub(crate) fn tiny_execution_metadata() -> ParakeetCtcExecutionMetadata {
        ParakeetCtcExecutionMetadata {
            n_layers: 1,
            hidden_size: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            conv_kernel: 9,
            n_mels: 80,
            subsampling_factor: 8,
            subsampling_channels: 24,
            vocab_size: 12,
            blank_token_id: 11,
        }
    }
}
