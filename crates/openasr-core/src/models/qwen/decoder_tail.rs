//! Contract-projected loader for the Qwen-shaped decoder tail.
//!
//! Admission expands final RMSNorm / token embedding / optional logits weight
//! through [`super::decoder_contract::qwen_decoder_tail_tensor_descriptors`].
//! Production materialization must use that same geometry +
//! [`QwenDecoderTailTensorNames`] pair so d_model/vocab shape authority cannot
//! drift into a second hand-written table at each family call site.
//!
//! When [`QwenDecoderTailTensorNames::output_weight`] is `None` (tied
//! embeddings, e.g. MOSS-Transcribe-Diarize), the logits head reuses
//! `token_embd` as its output-weight tensor. The descriptor set still lists the
//! embedding once; both consumers retain mmap views of the same payload.

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader};
use crate::models::tensor_binding::validate_tensor_binding_descriptors;

use super::decoder_contract::{
    QwenDecoderContractGeometry, QwenDecoderTailTensorNames, qwen_decoder_tail_tensor_descriptors,
};
use super::logits_head::{
    Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError,
    load_llm_logits_head_from_reader_with_tensor_names,
};
use super::token_embedding::{
    Qwen3AsrTokenEmbeddingError, Qwen3AsrTokenEmbeddingTable,
    load_token_embedding_table_from_reader_with_tensor_name,
};

/// Final RMSNorm + logits head + token embedding loaded from one contract tail.
#[derive(Debug)]
pub(crate) struct QwenDecoderTail {
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub token_embedding: Qwen3AsrTokenEmbeddingTable,
}

#[derive(Debug, Error)]
pub(crate) enum QwenDecoderTailLoadError {
    #[error("qwen decoder tail contract failed: {reason}")]
    Contract { reason: String },
    #[error(transparent)]
    LogitsHead(#[from] Qwen3AsrLlmLogitsHeadError),
    #[error(transparent)]
    TokenEmbedding(#[from] Qwen3AsrTokenEmbeddingError),
}

/// Load the decoder tail from the same geometry + tail names admission expands.
///
/// Shape authority is only
/// [`qwen_decoder_tail_tensor_descriptors`] (`VectorLen` / `ExactDims`). The
/// pack cannot invent a second d_model/vocab geometry here. Transposed logits
/// or embedding matrices and a missing final norm fail closed before any host
/// materialization runs.
///
/// `tail` is `'static` because the resident logits head stores the output-weight
/// tensor name for diagnostics and fused-graph binding.
pub(crate) fn load_qwen_decoder_tail_from_contract(
    reader: &GgufTensorDataReader,
    geometry: &QwenDecoderContractGeometry,
    tail: QwenDecoderTailTensorNames<'static>,
    rms_norm_epsilon: f32,
    backend: GgmlCpuGraphBackend,
) -> Result<QwenDecoderTail, QwenDecoderTailLoadError> {
    let descriptors = qwen_decoder_tail_tensor_descriptors(geometry, tail)
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
    let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
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
    use crate::models::tensor_binding::{
        TensorBindingDescriptorRequirement, assert_trace_matches_descriptor_set,
        project_fixture_tensors,
    };
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    use super::*;
    use crate::models::qwen::decoder_contract::qwen_decoder_tail_tensor_descriptors;

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
        geometry: &QwenDecoderContractGeometry,
        tail: QwenDecoderTailTensorNames<'static>,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<u64>>),
    ) {
        let descriptors =
            qwen_decoder_tail_tensor_descriptors(geometry, tail).expect("tail descriptors");
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
        let descriptors =
            qwen_decoder_tail_tensor_descriptors(&geometry, tail).expect("descriptors");
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-untied.oasr");
        write_tail_fixture(&path, &geometry, tail, |_| {});

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        load_qwen_decoder_tail_from_contract(
            &reader,
            &geometry,
            tail,
            1e-6,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("load untied tail");
        assert_trace_matches_descriptor_set(&reader.tensor_index().access_trace(), &descriptors);
    }

    #[test]
    fn loader_read_trace_matches_tail_descriptors_tied() {
        let geometry = tiny_geometry();
        let tail = tied_tail();
        let descriptors =
            qwen_decoder_tail_tensor_descriptors(&geometry, tail).expect("descriptors");
        // Tied: descriptor set is norm + embd only (no separate output.weight).
        assert_eq!(descriptors.len(), 2);
        assert!(descriptors.iter().all(|d| d.tensor_name != "output.weight"));

        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-tied.oasr");
        write_tail_fixture(&path, &geometry, tail, |_| {});

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        let loaded = load_qwen_decoder_tail_from_contract(
            &reader,
            &geometry,
            tail,
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
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-transposed-logits.oasr");
        write_tail_fixture(&path, &geometry, tail, |shapes| {
            shapes.insert(
                "output.weight".to_string(),
                // Contract ExactDims is [d_model, vocab] = [4, 6]; force transpose.
                vec![6, 4],
            );
        });
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let err = load_qwen_decoder_tail_from_contract(
            &reader,
            &geometry,
            tail,
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
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("tail-missing-norm.oasr");
        write_tail_fixture(&path, &geometry, tail, |shapes| {
            shapes.remove("output_norm.weight");
        });
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let err = load_qwen_decoder_tail_from_contract(
            &reader,
            &geometry,
            tail,
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
        let descriptors =
            qwen_decoder_tail_tensor_descriptors(&geometry, tail).expect("descriptors");
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
