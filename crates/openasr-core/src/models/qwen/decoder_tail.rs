//! Contract-projected loader for the Qwen-shaped decoder tail.
//!
//! Admission and production materialization project final RMSNorm / token
//! embedding / optional logits weight from one bound [`QwenDecoderContract`].
//! Family callers cannot pass geometry or tensor names independently.
//!
//! When [`QwenDecoderTailTensorNames::output_weight`] is `None` (tied
//! embeddings, e.g. MOSS-Transcribe-Diarize), the logits head reuses
//! `token_embd` as its output-weight tensor. The descriptor set still lists the
//! embedding once; both consumers retain mmap views of the same payload.

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader};
use crate::models::tensor_binding::validate_tensor_binding_descriptors;

use super::decoder_contract::QwenDecoderContract;
use super::logits_head::{
    Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError,
    load_llm_logits_head_from_reader_with_tensor_names,
};
use crate::models::mapped_token_embedding::{
    MappedTokenEmbeddingError, MappedTokenEmbeddingTable,
    load_mapped_token_embedding_table_from_reader,
};

/// Final RMSNorm + logits head + token embedding loaded from one contract tail.
#[derive(Debug)]
pub(crate) struct QwenDecoderTail {
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub token_embedding: MappedTokenEmbeddingTable,
}

#[derive(Debug, Error)]
pub(crate) enum QwenDecoderTailLoadError {
    #[error("qwen decoder tail contract failed: {reason}")]
    Contract { reason: String },
    #[error(transparent)]
    LogitsHead(#[from] Qwen3AsrLlmLogitsHeadError),
    #[error(transparent)]
    TokenEmbedding(#[from] MappedTokenEmbeddingError),
}

/// Load the decoder tail from the same bound contract admission expands.
///
/// Shape authority is only
/// the contract's tail projection (`VectorLen` / `ExactDims`). The
/// pack cannot invent a second d_model/vocab geometry here. Transposed logits
/// or embedding matrices and a missing final norm fail closed before any host
/// materialization runs.
///
/// Contract tail names are `'static` because the resident logits head stores the output-weight
/// tensor name for diagnostics and fused-graph binding.
pub(crate) fn load_qwen_decoder_tail_from_contract(
    reader: &GgufTensorDataReader,
    contract: &QwenDecoderContract,
    rms_norm_epsilon: f32,
    backend: GgmlCpuGraphBackend,
) -> Result<QwenDecoderTail, QwenDecoderTailLoadError> {
    let geometry = contract.geometry();
    let (tail, descriptors) = contract
        .tail_projection()
        .map_err(|reason| QwenDecoderTailLoadError::Contract { reason })?;
    // Single shape gate: the same ExactDims/VectorLen admission validated.
    validate_tensor_binding_descriptors(
        reader.tensor_index(),
        &descriptors,
        |name| QwenDecoderTailLoadError::Contract {
            reason: format!("required tail tensor '{name}' is missing"),
        },
        |name, dims, reason| QwenDecoderTailLoadError::Contract {
            reason: format!("tail tensor '{name}' has invalid shape {dims:?}: {reason}"),
        },
    )?;

    // Tied embeddings: no separate logits weight in the pack / descriptor set.
    // The logits head still needs a weight tensor name; reuse token_embd.
    let output_weight_name = tail.output_weight.unwrap_or(tail.token_embd);

    let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
        reader,
        geometry.d_model,
        geometry.vocab_size,
        tail.output_norm,
        output_weight_name,
        rms_norm_epsilon,
        backend,
    )?;
    let token_embedding = load_mapped_token_embedding_table_from_reader(
        reader,
        tail.token_embd,
        geometry.d_model,
        geometry.vocab_size,
    )?;
    Ok(QwenDecoderTail {
        logits_head,
        token_embedding,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader};
    use crate::models::qwen::{
        QwenDecoderContractGeometry, QwenDecoderTailTensorNames, QwenDecoderVariant,
        QwenFamilyDecoderProfile, QwenFamilyLlmLayerTensorNames,
    };
    use crate::models::tensor_binding::{
        TensorBindingDescriptorRequirement, assert_trace_matches_descriptor_set,
        project_fixture_tensors,
    };
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    use super::*;

    fn tail_test_layer_names(layer: usize) -> QwenFamilyLlmLayerTensorNames {
        let prefix = format!("blk.{layer}");
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: format!("{prefix}.attn_norm.weight"),
            attn_q_name: format!("{prefix}.attn_q.weight"),
            attn_k_name: format!("{prefix}.attn_k.weight"),
            attn_v_name: format!("{prefix}.attn_v.weight"),
            attn_output_name: format!("{prefix}.attn_output.weight"),
            q_norm_name: Some(format!("{prefix}.attn_q_norm.weight")),
            k_norm_name: Some(format!("{prefix}.attn_k_norm.weight")),
            q_bias_name: None,
            k_bias_name: None,
            v_bias_name: None,
            ffn_norm_name: format!("{prefix}.ffn_norm.weight"),
            ffn_gate_name: format!("{prefix}.ffn_gate.weight"),
            ffn_up_name: format!("{prefix}.ffn_up.weight"),
            ffn_down_name: format!("{prefix}.ffn_down.weight"),
        }
    }

    fn bind_tail_only(
        geometry: QwenDecoderContractGeometry,
        tail: QwenDecoderTailTensorNames<'static>,
    ) -> QwenDecoderContract {
        QwenDecoderContract::bind(
            geometry,
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen3, tail_test_layer_names, tail),
        )
        .expect("tail-only bind")
    }

    fn tiny_geometry() -> QwenDecoderContractGeometry {
        QwenDecoderContractGeometry {
            n_layers: 1,
            d_model: 4,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            vocab_size: 6,
        }
    }

    fn untied_tail() -> QwenDecoderTailTensorNames<'static> {
        QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: Some("output.weight"),
            token_embd: "token_embd.weight",
        }
    }

    fn tied_tail() -> QwenDecoderTailTensorNames<'static> {
        QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: None,
            token_embd: "token_embd.weight",
        }
    }

    fn write_tail_fixture(
        path: &std::path::Path,
        contract: &QwenDecoderContract,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<u64>>),
    ) {
        let (_, descriptors) = contract.tail_projection().expect("tail descriptors");
        let mut shapes: BTreeMap<String, Vec<u64>> =
            project_fixture_tensors(&descriptors).into_iter().collect();
        mutate(&mut shapes);
        let mut spec = TinyGgufFixtureSpec::new(BTreeMap::new());
        for (name, dims) in shapes {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(path, &spec).expect("write tail fixture");
    }

    #[test]
    fn loader_read_trace_matches_tail_descriptors_untied() {
        let geometry = tiny_geometry();
        let tail = untied_tail();
        let contract = bind_tail_only(geometry, tail);
        let (_, descriptors) = contract.tail_projection().expect("descriptors");
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-untied.oasr");
        write_tail_fixture(&path, &contract, |_| {});

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        load_qwen_decoder_tail_from_contract(&reader, &contract, 1e-6, GgmlCpuGraphBackend::Cpu)
            .expect("load untied tail");
        assert_trace_matches_descriptor_set(&reader.tensor_index().access_trace(), &descriptors);
    }

    #[test]
    fn loader_read_trace_matches_tail_descriptors_tied() {
        let geometry = tiny_geometry();
        let tail = tied_tail();
        let contract = bind_tail_only(geometry, tail);
        let (_, descriptors) = contract.tail_projection().expect("descriptors");
        // Tied: descriptor set is norm + embd only (no separate output.weight).
        assert_eq!(descriptors.len(), 2);
        assert!(descriptors.iter().all(|d| d.tensor_name != "output.weight"));

        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-tied.oasr");
        write_tail_fixture(&path, &contract, |_| {});

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        let loaded = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            1e-6,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("load tied tail");
        assert_trace_matches_descriptor_set(&reader.tensor_index().access_trace(), &descriptors);

        // Logits head must have bound the tied embedding name.
        let emb = loaded.token_embedding.mapped_payload().expect("mapped emb");
        let logits = loaded
            .logits_head
            .mapped_output_weight_payload()
            .expect("mapped logits");
        assert!(
            emb.shares_backing_range(logits),
            "tied consumers must share one mmap range"
        );
    }

    #[test]
    fn rejects_transposed_logits_weight() {
        let geometry = tiny_geometry();
        let tail = untied_tail();
        let contract = bind_tail_only(geometry, tail);
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-transposed-logits.oasr");
        write_tail_fixture(&path, &contract, |shapes| {
            shapes.insert(
                "output.weight".to_string(),
                // Contract ExactDims is [d_model, vocab] = [4, 6]; force transpose.
                vec![6, 4],
            );
        });
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let err = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            1e-6,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect_err("transposed logits must fail closed");
        let message = err.to_string();
        assert!(
            message.contains("output.weight"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn rejects_missing_output_norm() {
        let geometry = tiny_geometry();
        let tail = untied_tail();
        let contract = bind_tail_only(geometry, tail);
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-missing-norm.oasr");
        write_tail_fixture(&path, &contract, |shapes| {
            shapes.remove("output_norm.weight");
        });
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let err = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            1e-6,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect_err("missing norm must fail closed");
        let message = err.to_string();
        assert!(
            message.contains("output_norm.weight") && message.contains("missing"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn descriptor_exact_dims_are_d_model_by_vocab() {
        let geometry = tiny_geometry();
        let tail = untied_tail();
        let contract = bind_tail_only(geometry, tail);
        let (_, descriptors) = contract.tail_projection().expect("descriptors");
        let emb = descriptors
            .iter()
            .find(|d| d.tensor_name == "token_embd.weight")
            .expect("embd");
        let out = descriptors
            .iter()
            .find(|d| d.tensor_name == "output.weight")
            .expect("output");
        let norm = descriptors
            .iter()
            .find(|d| d.tensor_name == "output_norm.weight")
            .expect("norm");
        assert!(matches!(
            &emb.requirement,
            TensorBindingDescriptorRequirement::ExactDims(dims)
                if dims == &[geometry.d_model, geometry.vocab_size]
        ));
        assert!(matches!(
            &out.requirement,
            TensorBindingDescriptorRequirement::ExactDims(dims)
                if dims == &[geometry.d_model, geometry.vocab_size]
        ));
        assert!(matches!(
            &norm.requirement,
            TensorBindingDescriptorRequirement::VectorLen(len) if *len == geometry.d_model
        ));
    }
}
