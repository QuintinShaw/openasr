//! GGUF-backed weight provider: reads a real `.oasr` pack (written by
//! `package_import.rs`) into the same `HashMap<String, Vec<f32>>` shape the
//! `encoder_graph`/`qformer`/`decoder_graph` weight providers already accept
//! (they were built and numerically validated against that exact interface
//! in the parity harness -- this module gives them a second, production data
//! source without touching their code at all).
//!
//! Not a zero-copy bind (it materializes the requested tensors into a host
//! `HashMap`, dequantized to f32, up front). The **decoder** no longer uses this
//! path: its projection/norm/lm_head weights are bound zero-copy, keep-quantized,
//! from the mmap'd pack inside `decode_session::new_keep_quantized` (native
//! q8_0/q4_k/f16/f32, no host f32 copy), so a 2B decoder stays ~its packed size
//! resident instead of the ~8 GB an all-f32 dequant + upload cost. The executor
//! now calls this loader only for (a) the **encoder** and **projector** (still
//! the host-f32 arena path -- their keep-quantized migration, complicated by the
//! encoder's host-folded BatchNorm / rel-pos-emb, is a further follow-up), and
//! (b) the decoder's **token-embedding table alone** (used on the host by
//! `embed_token_row` for the prompt + per-step embedding lookup).

use std::collections::HashMap;

use crate::ggml_runtime::GgufTensorDataReadError;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechRuntimeProviderError {
    #[error("granite-speech runtime provider failed to build reader for pack '{path}': {reason}")]
    Preflight { path: String, reason: String },
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

pub(crate) fn load_tensors_from_preflight(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    prefix: &str,
) -> Result<HashMap<String, Vec<f32>>, GraniteSpeechRuntimeProviderError> {
    let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        GraniteSpeechRuntimeProviderError::Preflight {
            path: preflight.runtime_source.path().display().to_string(),
            reason: error.to_string(),
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
