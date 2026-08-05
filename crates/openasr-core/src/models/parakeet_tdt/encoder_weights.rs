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
    #[error("parakeet-tdt tensor '{name}' is not part of the runtime tensor contract")]
    NotInContract { name: String },
    #[error("parakeet-tdt weight expectation overflowed: {reason}")]
    ExpectationOverflow { reason: String },
}

impl FastConformerWeightsError for ParakeetTdtWeightsError {
    fn batchnorm_fold(reason: String) -> Self {
        Self::BatchNormFold { reason }
    }
    fn not_in_contract(name: String) -> Self {
        Self::NotInContract { name }
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

impl ParakeetTdtEncoderWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        crate::models::parakeet_runtime_memory::fastconformer_weights_retained_bytes(
            &self.subsampling,
            &self.layers,
            &[&self.enc_proj_weight, &self.enc_proj_bias],
        )
    }
}

impl ParakeetTdtPredictorWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        crate::models::parakeet_runtime_memory::add_named_tensor_capacity(
            &self.embedding,
            &mut bytes,
            "parakeet-tdt predictor embedding",
        )?;
        bytes.add_vec(
            &self.lstm_layers,
            "parakeet-tdt predictor layer descriptors",
        )?;
        for layer in &self.lstm_layers {
            for tensor in [&layer.w_ih, &layer.w_hh, &layer.b_ih, &layer.b_hh] {
                crate::models::parakeet_runtime_memory::add_named_tensor_capacity(
                    tensor,
                    &mut bytes,
                    "parakeet-tdt predictor tensor",
                )?;
            }
        }
        Ok(bytes.finish())
    }
}

impl ParakeetTdtJointWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        for tensor in [
            &self.pred_weight,
            &self.pred_bias,
            &self.out_weight,
            &self.out_bias,
        ] {
            crate::models::parakeet_runtime_memory::add_named_tensor_capacity(
                tensor,
                &mut bytes,
                "parakeet-tdt joint tensor",
            )?;
        }
        Ok(bytes.finish())
    }
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

/// Metadata-derived element-count expectation, fail-closed on overflow. The
/// parse-time architecture ceilings make overflow unreachable in practice;
/// the checked path keeps untrusted metadata from ever wrapping an
/// expectation into an admitting comparison.
fn checked_expectation(
    name: &str,
    value: Option<usize>,
    reason: &str,
) -> Result<usize, ParakeetTdtWeightsError> {
    value.ok_or_else(|| ParakeetTdtWeightsError::ExpectationOverflow {
        reason: format!("{name}: {reason}"),
    })
}

/// The read guard for one pack's full parakeet-tdt tensor contract (shared
/// FastConformer encoder plus the TDT tail): every tensor the loaders read
/// must be enumerated here.
pub(crate) fn parakeet_tdt_read_guard(
    metadata: &ParakeetTdtExecutionMetadata,
) -> crate::models::tensor_binding::TensorReadGuard {
    crate::models::tensor_binding::TensorReadGuard::from_descriptors(
        &super::runtime_contract::parakeet_tdt_runtime_tensor_binding_descriptors(metadata),
    )
}

pub(crate) fn load_parakeet_tdt_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<ParakeetTdtEncoderWeights, ParakeetTdtWeightsError> {
    let guard = parakeet_tdt_read_guard(metadata);
    let subsampling =
        fastconformer::load_fastconformer_subsampling::<ParakeetTdtWeightsError>(reader, &guard)?;

    let mut layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        // bias_present = false: v3 ships no attn/conv/FFN bias tensors; the
        // shared loader synthesizes zero biases of the right width instead.
        layers.push(fastconformer::load_fastconformer_layer::<
            ParakeetTdtWeightsError,
        >(
            reader,
            &guard,
            layer,
            metadata.hidden_size,
            metadata.ffn_dim,
            false,
        )?);
    }

    let mut enc_proj_weight: NamedTensor =
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, "enc.proj.weight")?;
    let enc_proj_bias: NamedTensor =
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, "enc.proj.bias")?;
    let expected_proj = checked_expectation(
        &enc_proj_weight.name,
        metadata.joint_hidden.checked_mul(metadata.hidden_size),
        "joint_hidden * hidden_size overflows",
    )?;
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
    let guard = parakeet_tdt_read_guard(metadata);
    let hidden = metadata.pred_hidden;
    let embedding = expect_elements(
        fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, "dec.embed.weight")?,
        checked_expectation(
            "dec.embed.weight",
            metadata.vocab_size.checked_mul(hidden),
            "vocab_size * pred_hidden overflows",
        )?,
    )?;
    let gate_weight_elems = checked_expectation(
        "dec.lstm gate weight",
        hidden
            .checked_mul(hidden)
            .and_then(|value| value.checked_mul(4)),
        "4 * pred_hidden * pred_hidden overflows",
    )?;
    let gate_bias_elems = checked_expectation(
        "dec.lstm gate bias",
        hidden.checked_mul(4),
        "4 * pred_hidden overflows",
    )?;
    let mut lstm_layers = Vec::with_capacity(metadata.pred_layers);
    for layer in 0..metadata.pred_layers {
        let n = |suffix: &str| format!("dec.lstm.{layer}.{suffix}");
        lstm_layers.push(ParakeetTdtLstmLayerWeights {
            w_ih: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, &n("w_ih"))?,
                gate_weight_elems,
            )?,
            w_hh: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, &n("w_hh"))?,
                gate_weight_elems,
            )?,
            b_ih: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, &n("b_ih"))?,
                gate_bias_elems,
            )?,
            b_hh: expect_elements(
                fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, &n("b_hh"))?,
                gate_bias_elems,
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
    let guard = parakeet_tdt_read_guard(metadata);
    let joint = metadata.joint_hidden;
    let out_rows = checked_expectation(
        "joint.out",
        metadata.vocab_size.checked_add(metadata.n_durations),
        "vocab_size + n_durations overflows",
    )?;
    Ok(ParakeetTdtJointWeights {
        pred_weight: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(
                reader,
                &guard,
                "joint.pred.weight",
            )?,
            checked_expectation(
                "joint.pred.weight",
                joint.checked_mul(metadata.pred_hidden),
                "joint_hidden * pred_hidden overflows",
            )?,
        )?,
        pred_bias: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(
                reader,
                &guard,
                "joint.pred.bias",
            )?,
            joint,
        )?,
        out_weight: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(
                reader,
                &guard,
                "joint.out.weight",
            )?,
            checked_expectation(
                "joint.out.weight",
                out_rows.checked_mul(joint),
                "(vocab_size + n_durations) * joint_hidden overflows",
            )?,
        )?,
        out_bias: expect_elements(
            fastconformer::load_named::<ParakeetTdtWeightsError>(reader, &guard, "joint.out.bias")?,
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

    /// The equivalence evidence the count-plus-sampling pin used to fake: run
    /// the REAL encoder + predictor + joint loaders over a synthetic pack
    /// projected from the contract enumeration itself, with the tensor
    /// index's access trace enabled, and assert the traced read set equals
    /// the descriptor set name for name and shape for shape. Any drift -- a
    /// loader reading a tensor the contract does not list, a descriptor no
    /// loader reads, or a read violating the descriptor's shape -- fails
    /// here. Also exercises the read guard: every read is contract-listed.
    #[test]
    fn full_loader_read_trace_equals_the_descriptor_set() {
        use super::super::runtime_contract::{
            parakeet_tdt_runtime_tensor_binding_descriptors, parakeet_tdt_runtime_tensors,
        };
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

        let metadata = super::tests_support::tiny_execution_metadata();
        let shapes = parakeet_tdt_runtime_tensors(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("parakeet-tdt-trace.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in shapes {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write trace pack");

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        load_parakeet_tdt_encoder_weights(&reader, &metadata).expect("full encoder load");
        load_parakeet_tdt_predictor_weights(&reader, &metadata).expect("full predictor load");
        load_parakeet_tdt_joint_weights(&reader, &metadata).expect("full joint load");

        crate::models::tensor_binding::assert_trace_matches_descriptor_set(
            &reader.tensor_index().access_trace(),
            &parakeet_tdt_runtime_tensor_binding_descriptors(&metadata),
        );
    }

    /// The read guard fails closed on any tensor the contract does not
    /// enumerate, so a loader/name drift cannot read off-contract.
    #[test]
    fn read_guard_rejects_off_contract_tensors() {
        use crate::models::fastconformer::load_named;

        let metadata = super::tests_support::tiny_execution_metadata();
        let guard = parakeet_tdt_read_guard(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("parakeet-tdt-guard.oasr");
        let spec = crate::testing::TinyGgufFixtureSpec::new(std::collections::BTreeMap::new())
            .with_tensor_shape("off.contract.weight", vec![2, 2]);
        crate::testing::write_tiny_gguf_runtime_source(&path, &spec).expect("write pack");
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");

        let error = load_named::<ParakeetTdtWeightsError>(&reader, &guard, "off.contract.weight")
            .expect_err("off-contract reads must fail closed");
        assert!(
            matches!(error, ParakeetTdtWeightsError::NotInContract { ref name } if name == "off.contract.weight"),
            "unexpected error: {error}"
        );
    }
}

/// Test-only geometry support shared by this module's tests and the runtime
/// contract's tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::super::runtime_contract::ParakeetTdtExecutionMetadata;

    pub(crate) fn tiny_execution_metadata() -> ParakeetTdtExecutionMetadata {
        ParakeetTdtExecutionMetadata {
            n_layers: 1,
            hidden_size: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            conv_kernel: 9,
            n_mels: 128,
            subsampling_factor: 8,
            subsampling_channels: 24,
            scale_input: false,
            vocab_size: 12,
            blank_token_id: 11,
            pred_hidden: 20,
            pred_layers: 2,
            joint_hidden: 24,
            n_durations: 5,
            max_symbols_per_step: 10,
        }
    }
}
