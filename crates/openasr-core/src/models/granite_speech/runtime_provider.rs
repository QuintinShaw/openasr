//! GGUF-backed weight provider: reads a real `.oasr` pack (written by
//! `package_import.rs`) into the same `HashMap<String, Vec<f32>>` shape the
//! `encoder_graph`/`qformer`/`decoder_graph` weight providers already accept
//! (they were built and numerically validated against that exact interface
//! in the parity harness -- this module gives them a second, production data
//! source without touching their code at all).
//!
//! Not a zero-copy bind (every other family's prepared-runtime loader binds
//! its weight tensors directly against the mmap'd GGUF reader inside graph
//! construction); this materializes the requested tensors into a host
//! `HashMap` up front instead. That is a real, intentional tradeoff for this
//! pass -- it reuses the already-validated encoder/projector/decoder code
//! unchanged, at the cost of one extra host copy of the pack's tensors at
//! load time (this is a 2B model; `fp16` tensors dequantized to f32 is a few
//! GB, well within a 16GB dev machine) -- matching this family's other
//! explicitly-noted "correct now, optimize when the shared mechanism is
//! ready" tradeoffs (see `decode_executor.rs`'s O(n^2) prefill note).

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechRuntimeProviderError {
    #[error("granite-speech runtime provider failed to open pack '{path}': {source}")]
    Open {
        path: String,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("granite-speech runtime provider failed to read tensor '{name}': {source}")]
    Read {
        name: String,
        #[source]
        source: GgufTensorDataReadError,
    },
}

/// Reverses `package_import::remap_tensor_name`'s one forced rename (ggml's
/// 63-byte tensor-name cap forced the Q-Former's deeply nested names to
/// shorten on write; every other name round-trips unchanged).
fn unmap_tensor_name(packed_name: &str) -> String {
    const SHORT_PREFIX: &str = "projector.qf.";
    match packed_name.strip_prefix(SHORT_PREFIX) {
        Some(rest) => format!("projector.qformer.encoder.layer.{rest}"),
        None => packed_name.to_string(),
    }
}

/// Loads every tensor in `pack_path` whose *original* (pre-remap) name starts
/// with `prefix` into a `HashMap<original_name, f32 values>`, dequantizing
/// F16/quantized tensors to F32 on the way. `prefix` is one of
/// `"encoder."`/`"projector."`/`"language_model."`, matching the three
/// `*WeightProvider` traits' expected keys.
pub(crate) fn load_tensors_from_oasr_pack(
    pack_path: &Path,
    prefix: &str,
) -> Result<HashMap<String, Vec<f32>>, GraniteSpeechRuntimeProviderError> {
    let reader = GgufTensorDataReader::from_path(pack_path).map_err(|source| {
        GraniteSpeechRuntimeProviderError::Open {
            path: pack_path.display().to_string(),
            source,
        }
    })?;

    let mut out = HashMap::new();
    for tensor in reader.tensor_index().tensors() {
        let original_name = unmap_tensor_name(&tensor.name);
        if !original_name.starts_with(prefix) {
            continue;
        }
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(&tensor.name, &tensor.dims)
            .map_err(|source| GraniteSpeechRuntimeProviderError::Read {
                name: tensor.name.clone(),
                source,
            })?;
        out.insert(original_name, values);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmap_tensor_name_round_trips_qformer_prefix() {
        assert_eq!(
            unmap_tensor_name("projector.qf.0.crossattention.output.LayerNorm.weight"),
            "projector.qformer.encoder.layer.0.crossattention.output.LayerNorm.weight"
        );
    }

    #[test]
    fn unmap_tensor_name_leaves_other_names_unchanged() {
        assert_eq!(
            unmap_tensor_name("encoder.layers.7.attn.rel_pos_emb.weight"),
            "encoder.layers.7.attn.rel_pos_emb.weight"
        );
        assert_eq!(unmap_tensor_name("projector.query"), "projector.query");
        assert_eq!(
            unmap_tensor_name("language_model.model.embed_tokens.weight"),
            "language_model.model.embed_tokens.weight"
        );
    }
}
