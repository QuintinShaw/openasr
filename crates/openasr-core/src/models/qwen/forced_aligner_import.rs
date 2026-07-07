//! Local-source (safetensors) -> `.oasr` (GGUF) conversion for
//! Qwen3-ForcedAligner-0.6B.
//!
//! Qwen3-ForcedAligner shares its `thinker` (audio encoder + LM) tensor layout
//! byte-for-byte with Qwen3-ASR (same `Qwen3ASRForConditionalGeneration`
//! class in the reference `qwen_asr` package; see
//! `qwen_asr.core.transformers_backend.modeling_qwen3_asr`). The only
//! structural difference is the final head: when `model_type` contains
//! `"forced_aligner"`, the reference replaces the tied `lm_head` with an
//! independent `Linear(hidden_size, classify_num)` (5000 x 80ms timestamp
//! bins) instead of `Linear(hidden_size, vocab_size)`. Both live at the same
//! source tensor name `thinker.lm_head.weight`, so the generic tensor-name
//! remap/quantization pipeline in `package_import` is reused unmodified; this
//! module only differs in config parsing (top-level `timestamp_token_id`/
//! `timestamp_segment_time`, and `thinker_config.classify_num`) and in the
//! GGUF metadata it emits.
//!
//! Deliberately NOT wired into `ggml_family_registry`: this produces a valid
//! GGUF/.oasr pack for numeric-parity verification and Stage 2 ggml
//! execution, but the forced-aligner's NAR (single forward pass, argmax at
//! `<timestamp>` positions) decode policy is not the qwen3-asr autoregressive
//! greedy decoder, so it must not be dispatchable through the existing
//! qwen3-asr runtime path. Family registration + CLI wiring is out of scope
//! for this stage.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::ggml_runtime::{
    GgufWriteValue, read_gguf_metadata_from_runtime_source, read_gguf_tensor_index,
    read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
    write_gguf_file_v0,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, read_source_json_file, validate_error,
    validate_output_pack_extension,
};

use super::package_import::{
    Qwen3AsrRuntimeQuantizationMode, build_qwen_runtime_tensors, compose_model_id, insert_metadata,
    insert_metadata_string_array, insert_metadata_u32, load_merges, load_vocab_tokens,
    patch_added_tokens,
};

const SOURCE_CONFIG_JSON: &str = "config.json";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

/// GGUF `general.architecture` / `openasr.model.family` value for this
/// converter's output. Deliberately distinct from `qwen3-asr` so a bundled
/// pack can never be mistaken for (or dispatched through) the qwen3-asr
/// autoregressive runtime -- the decode policy is materially different (one
/// NAR forward pass + argmax at `<timestamp>` positions, not incremental
/// greedy generation).
pub const QWEN3_FORCED_ALIGNER_MODEL_FAMILY: &str = "qwen3-forced-aligner";
pub const QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID: &str = "qwen3-forced-aligner";

pub type Qwen3ForcedAlignerLocalSourceError = LocalSourceImportError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3ForcedAlignerLocalSourceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub package_id: String,
    pub package_variant: Option<String>,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: Qwen3AsrRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3ForcedAlignerLocalSourceImportRuntimeResult {
    pub output_path: PathBuf,
    pub model_id: String,
    pub tensor_count: usize,
}

#[derive(Debug, Deserialize)]
struct ForcedAlignerConfigJson {
    #[serde(default)]
    timestamp_token_id: Option<u32>,
    #[serde(default)]
    timestamp_segment_time: Option<u32>,
    thinker_config: ForcedAlignerThinkerConfigJson,
}

#[derive(Debug, Deserialize)]
struct ForcedAlignerThinkerConfigJson {
    audio_config: ForcedAlignerAudioConfigJson,
    text_config: ForcedAlignerTextConfigJson,
    #[serde(default)]
    classify_num: Option<usize>,
    #[serde(default)]
    audio_token_id: Option<u32>,
    #[serde(default)]
    audio_start_token_id: Option<u32>,
    #[serde(default)]
    audio_end_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ForcedAlignerAudioConfigJson {
    #[serde(default)]
    num_mel_bins: Option<usize>,
    #[serde(default)]
    encoder_layers: Option<usize>,
    #[serde(default)]
    d_model: Option<usize>,
    #[serde(default)]
    encoder_attention_heads: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ForcedAlignerTextConfigJson {
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone)]
struct ForcedAlignerMetadataFields {
    sample_rate_hz: u32,
    n_mels: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    audio_layers: usize,
    audio_d_model: usize,
    audio_heads: usize,
    llm_layers: usize,
    llm_d_model: usize,
    llm_heads: usize,
    llm_kv_heads: usize,
    llm_head_dim: usize,
    /// Token-embedding table size (`thinker.model.embed_tokens.weight` rows),
    /// distinct from `classify_num` -- unlike qwen3-asr, the forced aligner's
    /// output head is NOT tied to the embedding table.
    embed_vocab_size: usize,
    /// Classification head width (`thinker.lm_head.weight` rows): 5000
    /// 80ms-wide timestamp bins, independent of `embed_vocab_size`.
    classify_num: usize,
    llm_max_positions: usize,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_pad_token_id: u32,
    timestamp_token_id: u32,
    timestamp_segment_time_ms: u32,
}

pub fn convert_local_qwen_forced_aligner_source_to_runtime_pack(
    request: &Qwen3ForcedAlignerLocalSourceImportRequest,
) -> Result<Qwen3ForcedAlignerLocalSourceImportRuntimeResult, Qwen3ForcedAlignerLocalSourceError> {
    validate_request(request)?;
    let config: ForcedAlignerConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let mut tokens = load_vocab_tokens(&request.source_root)?;
    let merges = load_merges(&request.source_root)?;
    let fields = forced_aligner_metadata_fields(&config);
    if tokens.len() < fields.embed_vocab_size {
        tokens.resize_with(fields.embed_vocab_size, String::new);
    }
    patch_added_tokens(&request.source_root, &mut tokens)?;
    for (index, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{index}>");
        }
    }

    let safetensor_files = discover_safetensor_files(&request.source_root)?;
    let mut tensors = build_qwen_runtime_tensors(
        &safetensor_files,
        request.quantization,
        fields.n_mels,
        fields.n_fft,
        fields.sample_rate_hz,
        fields.win_length,
    )?;

    let model_id = compose_model_id(&request.package_id, request.package_variant.as_deref());
    let metadata = forced_aligner_gguf_metadata(request, &fields, &model_id, &tokens, &merges);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "Qwen3-ForcedAligner local-source GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    // Mechanical round-trip check only (tensor count / readability). This
    // converter is not registered with `ggml_family_registry`, so there is no
    // builtin runtime tensor contract to validate against yet -- that lands
    // with family registration in a later stage.
    let runtime_source =
        validate_ggml_runtime_source_path(&request.output_root).map_err(|error| {
            validate_error(format!(
                "Qwen3-ForcedAligner local-source import produced invalid runtime path '{}': {error}",
                request.output_root.display()
            ))
        })?;
    let _metadata_read =
        read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Qwen3-ForcedAligner import produced unreadable GGUF metadata: {error}"
            ))
        })?;
    let _tensor_index_read =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Qwen3-ForcedAligner import produced unreadable GGUF tensor index: {error}"
            ))
        })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Qwen3-ForcedAligner local-source GGUF writer produced unreadable tensor index: {error}"
        ))
    })?;
    tensors.clear();
    Ok(Qwen3ForcedAlignerLocalSourceImportRuntimeResult {
        output_path: request.output_root.clone(),
        model_id,
        tensor_count: index.tensors().len(),
    })
}

fn forced_aligner_metadata_fields(config: &ForcedAlignerConfigJson) -> ForcedAlignerMetadataFields {
    let audio = &config.thinker_config.audio_config;
    let text = &config.thinker_config.text_config;
    let llm_d_model = text.hidden_size.unwrap_or(1024);
    let llm_heads = text.num_attention_heads.unwrap_or(16);
    let llm_head_dim = text.head_dim.unwrap_or(llm_d_model / llm_heads.max(1));
    let embed_vocab_size = text.vocab_size.unwrap_or(152_064);
    ForcedAlignerMetadataFields {
        sample_rate_hz: 16_000,
        n_mels: audio.num_mel_bins.unwrap_or(128),
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        audio_layers: audio.encoder_layers.unwrap_or(24),
        audio_d_model: audio.d_model.unwrap_or(1024),
        audio_heads: audio.encoder_attention_heads.unwrap_or(16),
        llm_layers: text.num_hidden_layers.unwrap_or(28),
        llm_d_model,
        llm_heads,
        llm_kv_heads: text.num_key_value_heads.unwrap_or(8),
        llm_head_dim,
        embed_vocab_size,
        classify_num: config.thinker_config.classify_num.unwrap_or(5_000),
        llm_max_positions: text.max_position_embeddings.unwrap_or(8_192),
        audio_start_token_id: config
            .thinker_config
            .audio_start_token_id
            .unwrap_or(151_669),
        audio_end_token_id: config.thinker_config.audio_end_token_id.unwrap_or(151_670),
        audio_pad_token_id: config.thinker_config.audio_token_id.unwrap_or(151_676),
        timestamp_token_id: config.timestamp_token_id.unwrap_or(151_705),
        timestamp_segment_time_ms: config.timestamp_segment_time.unwrap_or(80),
    }
}

#[allow(clippy::too_many_arguments)]
fn forced_aligner_gguf_metadata(
    request: &Qwen3ForcedAlignerLocalSourceImportRequest,
    fields: &ForcedAlignerMetadataFields,
    model_id: &str,
    tokens: &[String],
    merges: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    insert_metadata(
        &mut metadata,
        crate::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION,
        crate::models::oasr_metadata::OASR_PACKAGE_VERSION_V1,
    );
    insert_metadata(
        &mut metadata,
        crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY,
        QWEN3_FORCED_ALIGNER_MODEL_FAMILY,
    );
    insert_metadata(
        &mut metadata,
        crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
    );
    insert_metadata(&mut metadata, OPENASR_MODEL_ID_KEY, model_id);
    insert_metadata(
        &mut metadata,
        GENERAL_ARCHITECTURE_KEY,
        QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
    );

    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.sample_rate_hz",
        fields.sample_rate_hz,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.n_mels",
        fields.n_mels as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.n_fft",
        fields.n_fft as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.win_length",
        fields.win_length as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.hop_length",
        fields.hop_length as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.n_layers",
        fields.audio_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.d_model",
        fields.audio_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio.n_heads",
        fields.audio_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.n_layers",
        fields.llm_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.d_model",
        fields.llm_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.n_heads",
        fields.llm_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.n_kv_heads",
        fields.llm_kv_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.head_dim",
        fields.llm_head_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.embed_vocab_size",
        fields.embed_vocab_size as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.classify_num",
        fields.classify_num as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.llm.max_positions",
        fields.llm_max_positions as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio_start_token_id",
        fields.audio_start_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio_end_token_id",
        fields.audio_end_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.audio_pad_token_id",
        fields.audio_pad_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.timestamp_token_id",
        fields.timestamp_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        "qwen3_forced_aligner.timestamp_segment_time_ms",
        fields.timestamp_segment_time_ms,
    );

    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2,
    );
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, tokens);
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_MERGES_KEY, merges);

    insert_metadata(&mut metadata, "openasr.source.name", &request.source_name);
    insert_metadata(
        &mut metadata,
        "openasr.source.revision",
        &request.source_revision,
    );
    insert_metadata(&mut metadata, "openasr.license.name", &request.license_name);
    insert_metadata(
        &mut metadata,
        "openasr.license.source",
        &request.license_source,
    );
    metadata
}

fn discover_safetensor_files(
    source_root: &Path,
) -> Result<Vec<SafetensorsFile>, LocalSourceImportError> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(source_root).map_err(|source| LocalSourceImportError::Read {
        path: source_root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| LocalSourceImportError::Read {
            path: source_root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(validate_error(format!(
            "Qwen3-ForcedAligner local-source converter could not find any *.safetensors in '{}'",
            source_root.display()
        )));
    }
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        files.push(SafetensorsFile::open(path)?);
    }
    Ok(files)
}

fn validate_request(
    request: &Qwen3ForcedAlignerLocalSourceImportRequest,
) -> Result<(), LocalSourceImportError> {
    if request.package_id.trim().is_empty() {
        return Err(validate_error(
            "Qwen3-ForcedAligner local-source converter requires non-empty package_id",
        ));
    }
    if request.source_name.trim().is_empty() {
        return Err(validate_error(
            "Qwen3-ForcedAligner local-source converter requires non-empty source_name",
        ));
    }
    if request.source_revision.trim().is_empty() {
        return Err(validate_error(
            "Qwen3-ForcedAligner local-source converter requires non-empty source_revision",
        ));
    }
    if request.license_name.trim().is_empty() {
        return Err(validate_error(
            "Qwen3-ForcedAligner local-source converter requires non-empty license_name",
        ));
    }
    if request.license_source.trim().is_empty() {
        return Err(validate_error(
            "Qwen3-ForcedAligner local-source converter requires non-empty license_source",
        ));
    }
    validate_output_pack_extension(&request.output_root)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Stage 1 gate: convert the real downloaded Qwen3-ForcedAligner-0.6B
    /// checkpoint (if present on disk -- this test is opt-in / dev-machine
    /// only, mirroring the firered `tmp/firered-out` precedent) and assert
    /// every mapped destination tensor's element count matches the source
    /// safetensors tensor 1:1, plus the two synthesized frontend tensors.
    #[test]
    fn forced_aligner_conversion_produces_tensor_parity_with_source_checkpoint() {
        let source_root =
            PathBuf::from("/Volumes/QuintinDocument/hf-cache/qwen3-forced-aligner-0.6b");
        if !source_root.exists() {
            eprintln!(
                "skipping: {} not present (Stage 0 reference download is dev-machine only)",
                source_root.display()
            );
            return;
        }
        let output_dir = std::env::temp_dir().join("openasr-forced-aligner-stage1-test");
        let _ = std::fs::create_dir_all(&output_dir);
        let output_root = output_dir.join("qwen3-forced-aligner-0.6b-fp16.oasr");

        let request = Qwen3ForcedAlignerLocalSourceImportRequest {
            source_root: source_root.clone(),
            output_root: output_root.clone(),
            package_id: "qwen3-forced-aligner-0.6b".to_string(),
            package_variant: Some("fp16".to_string()),
            source_name: "Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            source_revision: "test".to_string(),
            license_name: "Apache-2.0".to_string(),
            license_source: "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            quantization: Qwen3AsrRuntimeQuantizationMode::Fp16,
        };

        let result = convert_local_qwen_forced_aligner_source_to_runtime_pack(&request)
            .expect("forced-aligner conversion must succeed against the real checkpoint");

        // Cross-check against the raw safetensors header: every source tensor
        // must have a mapped destination (name known to `remap_qwen_tensor_name`
        // via the shared qwen3-asr tensor-name convention), and the two
        // synthesized frontend tensors (mel filters/window) are additional.
        let safetensor_files = discover_safetensor_files(&source_root).expect("safetensors");
        let mut source_tensor_names = BTreeSet::new();
        for file in &safetensor_files {
            for tensor in &file.header().tensors {
                source_tensor_names.insert(tensor.name.clone());
            }
        }
        // 2 synthesized frontend tensors (audio.mel_filters / audio.mel_window)
        // are not present in the source safetensors.
        assert_eq!(
            result.tensor_count,
            source_tensor_names.len() + 2,
            "expected every source tensor to map 1:1 to a destination tensor, plus 2 synthesized frontend tensors"
        );

        let index = read_gguf_tensor_index(&output_root).expect("tensor index");
        // Spot-check the classify head landed with its real (non-tied) shape:
        // [1024 (hidden), 5000 (classify_num)] after the qwen dim-reversal
        // convention (source safetensors shape is [5000, 1024]).
        let output_weight = index.get("output.weight").expect("output.weight present");
        assert_eq!(output_weight.dims, vec![1024, 5000]);

        let token_embd = index
            .get("token_embd.weight")
            .expect("token_embd.weight present");
        assert_eq!(token_embd.dims, vec![1024, 152_064]);

        let _ = std::fs::remove_file(&output_root);
    }
}
