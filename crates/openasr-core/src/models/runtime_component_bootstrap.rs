use thiserror::Error;

use crate::GgufTensorDataReader;

use super::frontend_component_registry::{
    BuiltinAudioFrontendComponent, BuiltinAudioFrontendComponentRegistryError,
    materialize_builtin_audio_frontend_for_architecture,
};
use super::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use super::runtime_asset_bootstrap::{
    BuiltinRuntimeAssetBootstrapError, build_builtin_runtime_asset_bootstrap,
};
use super::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;
use super::tokenizer_component_registry::{
    BuiltinTokenizerComponent, BuiltinTokenizerComponentRegistryError,
    materialize_builtin_tokenizer_for_architecture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinTokenizerMaterializationMode {
    Optional,
    Required,
}

#[derive(Debug)]
pub(crate) struct BuiltinRuntimeComponentBootstrap {
    pub metadata: RuntimeTensorContractMetadata,
    pub tensor_reader: GgufTensorDataReader,
    pub tokenizer: Option<BuiltinTokenizerComponent>,
    pub audio_frontend: BuiltinAudioFrontendComponent,
}

#[derive(Debug, Error)]
pub(crate) enum BuiltinRuntimeComponentBootstrapError {
    #[error("runtime asset bootstrap failed: {source}")]
    RuntimeAssetBootstrap {
        #[source]
        source: BuiltinRuntimeAssetBootstrapError,
    },
    #[error("tokenizer materialization failed: {source}")]
    TokenizerMaterialization {
        #[source]
        source: BuiltinTokenizerComponentRegistryError,
    },
    #[error("audio frontend materialization failed: {source}")]
    AudioFrontendMaterialization {
        #[source]
        source: BuiltinAudioFrontendComponentRegistryError,
    },
}

pub(crate) fn build_builtin_runtime_component_bootstrap(
    model_architecture: &str,
    preflight: &GgmlAsrRuntimeSourcePreflight,
    tokenizer_mode: BuiltinTokenizerMaterializationMode,
) -> Result<BuiltinRuntimeComponentBootstrap, BuiltinRuntimeComponentBootstrapError> {
    let asset_bootstrap = build_builtin_runtime_asset_bootstrap(model_architecture, preflight)
        .map_err(
            |source| BuiltinRuntimeComponentBootstrapError::RuntimeAssetBootstrap { source },
        )?;
    let metadata = asset_bootstrap.metadata;
    let tensor_reader = asset_bootstrap.tensor_reader;
    let tokenizer = match tokenizer_mode {
        BuiltinTokenizerMaterializationMode::Optional => {
            materialize_builtin_tokenizer_for_architecture(model_architecture, &preflight.metadata)
                .ok()
        }
        BuiltinTokenizerMaterializationMode::Required => Some(
            materialize_builtin_tokenizer_for_architecture(model_architecture, &preflight.metadata)
                .map_err(|source| {
                    BuiltinRuntimeComponentBootstrapError::TokenizerMaterialization { source }
                })?,
        ),
    };
    let audio_frontend = materialize_builtin_audio_frontend_for_architecture(
        model_architecture,
        &tensor_reader,
        metadata,
    )
    .map_err(
        |source| BuiltinRuntimeComponentBootstrapError::AudioFrontendMaterialization { source },
    )?;
    Ok(BuiltinRuntimeComponentBootstrap {
        metadata,
        tensor_reader,
        tokenizer,
        audio_frontend,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use tempfile::{NamedTempFile, TempPath};

    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::{
        read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
        validate_ggml_runtime_source_path,
    };

    fn write_cohere_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    fn qwen_frontend_fixture_spec() -> TinyGgufFixtureSpec {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        metadata.insert("qwen3-asr.sample_rate".to_string(), "16000".to_string());
        metadata.insert("qwen3-asr.n_mels".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.n_fft".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.win_length".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.hop_length".to_string(), "160".to_string());
        metadata.insert("qwen3-asr.audio.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.audio.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.audio.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.llm.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.n_kv_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.head_dim".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.llm.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.vocab_size".to_string(), "7".to_string());
        metadata.insert("qwen3-asr.llm.max_pos".to_string(), "256".to_string());
        metadata.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            "2".to_string(),
        );
        metadata.insert("qwen3-asr.audio_end_token_id".to_string(), "3".to_string());
        metadata.insert("qwen3-asr.audio_pad_token_id".to_string(), "4".to_string());
        metadata.insert("qwen3-asr.eos_token_id".to_string(), "0".to_string());
        metadata.insert("qwen3-asr.pad_token_id".to_string(), "6".to_string());
        TinyGgufFixtureSpec::new(metadata)
            .with_string_array_metadata(
                "tokenizer.ggml.tokens",
                [
                    "<|endoftext|>",
                    "hello",
                    "<|audio_start|>",
                    "<|audio_end|>",
                    "<|audio_pad|>",
                    "world",
                    "<|pad|>",
                ],
            )
            .with_string_array_metadata("tokenizer.ggml.merges", ["h e"])
            .with_metadata("tokenizer.ggml.model", "gpt2")
            .with_tensor_shape("audio.mel_filters", [8_u64, 201_u64])
            .with_tensor_shape("audio.mel_window", [400_u64])
            .with_tensor_shape("audio.conv.1.weight", [3_u64, 3_u64, 1_u64, 4_u64])
            .with_tensor_shape("audio.conv.1.bias", [4_u64])
            .with_tensor_shape("audio.conv.2.weight", [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape("audio.conv.2.bias", [4_u64])
            .with_tensor_shape("audio.conv.3.weight", [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape("audio.conv.3.bias", [4_u64])
            .with_tensor_shape("audio.conv_out.weight", [4_u64, 16_u64])
            .with_tensor_shape("audio.conv_out.bias", [16_u64])
            .with_tensor_shape("audio.ln_post.weight", [16_u64])
            .with_tensor_shape("audio.ln_post.bias", [16_u64])
            .with_tensor_shape("audio.proj1.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.proj1.bias", [16_u64])
            .with_tensor_shape("audio.proj2.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.proj2.bias", [16_u64])
            .with_tensor_shape("token_embd.weight", [16_u64, 7_u64])
            .with_tensor_shape("output.weight", [16_u64, 7_u64])
            .with_tensor_shape("output_norm.weight", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_norm.weight", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_norm.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.0.attn_q.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.0.attn_k.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.0.attn_v.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.attn_out.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.0.attn_out.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.ffn_norm.weight", [16_u64])
            .with_tensor_shape("audio.blk.0.ffn_norm.bias", [16_u64])
            .with_tensor_shape("audio.blk.0.ffn_up.weight", [16_u64, 32_u64])
            .with_tensor_shape("audio.blk.0.ffn_up.bias", [32_u64])
            .with_tensor_shape("audio.blk.0.ffn_down.weight", [32_u64, 16_u64])
            .with_tensor_shape("audio.blk.0.ffn_down.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_norm.weight", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_norm.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.1.attn_q.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.1.attn_k.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.1.attn_v.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.attn_out.weight", [16_u64, 16_u64])
            .with_tensor_shape("audio.blk.1.attn_out.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.ffn_norm.weight", [16_u64])
            .with_tensor_shape("audio.blk.1.ffn_norm.bias", [16_u64])
            .with_tensor_shape("audio.blk.1.ffn_up.weight", [16_u64, 32_u64])
            .with_tensor_shape("audio.blk.1.ffn_up.bias", [32_u64])
            .with_tensor_shape("audio.blk.1.ffn_down.weight", [32_u64, 16_u64])
            .with_tensor_shape("audio.blk.1.ffn_down.bias", [16_u64])
            .with_tensor_shape("blk.0.attn_norm.weight", [16_u64])
            .with_tensor_shape("blk.0.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_output.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_q_norm.weight", [8_u64])
            .with_tensor_shape("blk.0.attn_k_norm.weight", [8_u64])
            .with_tensor_shape("blk.0.ffn_norm.weight", [16_u64])
            .with_tensor_shape("blk.0.ffn_gate.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.0.ffn_up.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.0.ffn_down.weight", [16_u64, 32_u64])
            .with_tensor_shape("blk.1.attn_norm.weight", [16_u64])
            .with_tensor_shape("blk.1.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_output.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_q_norm.weight", [8_u64])
            .with_tensor_shape("blk.1.attn_k_norm.weight", [8_u64])
            .with_tensor_shape("blk.1.ffn_norm.weight", [16_u64])
            .with_tensor_shape("blk.1.ffn_gate.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.1.ffn_up.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.1.ffn_down.weight", [16_u64, 32_u64])
    }

    fn write_qwen_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = qwen_frontend_fixture_spec();
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn builds_required_cohere_runtime_components() {
        let (_runtime_path, preflight) = write_cohere_preflight();
        let components = build_builtin_runtime_component_bootstrap(
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            &preflight,
            BuiltinTokenizerMaterializationMode::Required,
        )
        .expect("components");

        assert!(components.tokenizer.is_some());
        assert!(components.audio_frontend.into_cohere_transcribe().is_some());
    }

    #[test]
    fn builds_optional_qwen_runtime_components() {
        let (_runtime_path, preflight) = write_qwen_preflight();
        let components = build_builtin_runtime_component_bootstrap(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &preflight,
            BuiltinTokenizerMaterializationMode::Optional,
        )
        .expect("components");

        assert!(components.tokenizer.is_some());
        assert!(components.audio_frontend.into_qwen3_asr().is_some());
    }

    #[test]
    fn optional_tokenizer_mode_tolerates_missing_tokenizer_metadata() {
        let (_runtime_path, mut preflight) = write_qwen_preflight();
        let mut values = preflight.metadata.values().clone();
        values.remove("tokenizer.ggml.tokens");
        preflight.metadata = std::sync::Arc::new(crate::GgufMetadata::from_values_for_test(values));

        let components = build_builtin_runtime_component_bootstrap(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &preflight,
            BuiltinTokenizerMaterializationMode::Optional,
        )
        .expect("components");

        assert!(components.tokenizer.is_none());
        assert!(components.audio_frontend.into_qwen3_asr().is_some());
    }
}
