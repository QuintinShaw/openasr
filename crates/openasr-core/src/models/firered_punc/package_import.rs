//! Convert a local FireRedPunc source (`chinese-lert-base` BERT encoder + a
//! 5-class token-classification head, upstream `FireRedTeam/FireRedPunc`) into a
//! punctuation `.oasr` (GGUF-v0) pack.
//!
//! Source layout: an F32 `.safetensors` produced from the upstream
//! `model.pth.tar` by `tooling/publish-model/scripts/pt_to_safetensors.py`
//! (keeps the HuggingFace `bert.*` / `classifier.*` `state_dict` names), plus
//! the upstream WordPiece `vocab.txt` (21128 lines). This module maps those
//! names onto the GGUF tensor contract the runtime binds by
//! (`super::tensor_names`), reversing the 2D linear/embedding dims to ggml `ne`
//! order (PyTorch `[out, in]` row-major -> ggml `[in, out]`, so `mul_mat`/
//! `get_rows` read the same bytes correctly), and writes the pack geometry +
//! tokenizer tokens as metadata. It runs only offline at publish time; the
//! numeric parity of the result is proven against the upstream PyTorch forward
//! by the env-gated golden test (see `runtime::tests`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    write_gguf_file_v0,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, SafetensorsTensorHeader,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, validate_error,
    validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::nn::half::f32_to_f16_bits;

use super::config::{
    FIRERED_PUNC_ARCHITECTURE_VALUE, FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY,
    FIRERED_PUNC_BLOCK_COUNT_KEY, FIRERED_PUNC_CONTEXT_LENGTH_KEY,
    FIRERED_PUNC_EMBEDDING_LENGTH_KEY, FIRERED_PUNC_EXPECTED_D_MODEL,
    FIRERED_PUNC_EXPECTED_FFN_DIM, FIRERED_PUNC_EXPECTED_HEADS, FIRERED_PUNC_EXPECTED_LAYERS,
    FIRERED_PUNC_EXPECTED_MAX_POSITIONS, FIRERED_PUNC_EXPECTED_VOCAB_SIZE,
    FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY, FIRERED_PUNC_LABEL_COUNT, FIRERED_PUNC_LABEL_COUNT_KEY,
    FIRERED_PUNC_VOCAB_SIZE_KEY, TOKENIZER_GGML_TOKENS_KEY,
};
use super::tensor_names::{
    EMBD_NORM_BIAS, EMBD_NORM_WEIGHT, POSITION_EMBD_WEIGHT, PUNC_HEAD_BIAS, PUNC_HEAD_WEIGHT,
    TOKEN_EMBD_WEIGHT, TOKEN_TYPE_EMBD_WEIGHT, firered_punc_layer_tensor_names,
};

pub(crate) const FIRERED_PUNC_MODEL_FAMILY: &str = "firered-punc";

/// Runtime tensor quantization for the GGUF-backed `.oasr` output. The runtime
/// loader dequantizes every tensor to f32 on read (BERT-base runs as an
/// occasional finalize-only pass), so quantization only trades pack size for a
/// small numeric perturbation; `Fp16` is the exact-parity default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum FireRedPuncQuantizationMode {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone)]
pub struct FireRedPuncImportRequest {
    /// F32 `.safetensors` with the upstream `bert.*` / `classifier.*` names.
    pub source_safetensors: PathBuf,
    /// Upstream WordPiece `vocab.txt` (one token per line, 21128 lines).
    pub vocab_txt: PathBuf,
    /// Output `.oasr` pack path (must end in `.oasr`).
    pub output_pack: PathBuf,
    /// Catalog model id recorded in the pack metadata.
    pub model_id: String,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: FireRedPuncQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireRedPuncImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub token_count: usize,
}

/// One tensor to emit: its GGUF (target) name, the upstream `state_dict` source
/// name, and whether it is a 2D weight whose dims must be reversed to ggml `ne`.
struct TensorMapping {
    target: String,
    source: String,
    reverse_dims: bool,
}

fn mapping(target: &str, source: &str, reverse_dims: bool) -> TensorMapping {
    TensorMapping {
        target: target.to_string(),
        source: source.to_string(),
        reverse_dims,
    }
}

/// The full target<-source tensor map, derived from the fixed chinese-lert-base
/// BERT structure so it cannot drift from the runtime's tensor-name contract
/// (`super::tensor_names`).
fn tensor_mappings() -> Vec<TensorMapping> {
    let mut mappings = vec![
        mapping(
            TOKEN_EMBD_WEIGHT,
            "bert.embeddings.word_embeddings.weight",
            true,
        ),
        mapping(
            POSITION_EMBD_WEIGHT,
            "bert.embeddings.position_embeddings.weight",
            true,
        ),
        mapping(
            TOKEN_TYPE_EMBD_WEIGHT,
            "bert.embeddings.token_type_embeddings.weight",
            true,
        ),
        mapping(EMBD_NORM_WEIGHT, "bert.embeddings.LayerNorm.weight", false),
        mapping(EMBD_NORM_BIAS, "bert.embeddings.LayerNorm.bias", false),
    ];
    for layer in 0..FIRERED_PUNC_EXPECTED_LAYERS {
        let names = firered_punc_layer_tensor_names(layer);
        let src = format!("bert.encoder.layer.{layer}");
        mappings.extend([
            mapping(
                &names.attn_q_weight,
                &format!("{src}.attention.self.query.weight"),
                true,
            ),
            mapping(
                &names.attn_q_bias,
                &format!("{src}.attention.self.query.bias"),
                false,
            ),
            mapping(
                &names.attn_k_weight,
                &format!("{src}.attention.self.key.weight"),
                true,
            ),
            mapping(
                &names.attn_k_bias,
                &format!("{src}.attention.self.key.bias"),
                false,
            ),
            mapping(
                &names.attn_v_weight,
                &format!("{src}.attention.self.value.weight"),
                true,
            ),
            mapping(
                &names.attn_v_bias,
                &format!("{src}.attention.self.value.bias"),
                false,
            ),
            mapping(
                &names.attn_output_weight,
                &format!("{src}.attention.output.dense.weight"),
                true,
            ),
            mapping(
                &names.attn_output_bias,
                &format!("{src}.attention.output.dense.bias"),
                false,
            ),
            mapping(
                &names.attn_norm_weight,
                &format!("{src}.attention.output.LayerNorm.weight"),
                false,
            ),
            mapping(
                &names.attn_norm_bias,
                &format!("{src}.attention.output.LayerNorm.bias"),
                false,
            ),
            mapping(
                &names.ffn_up_weight,
                &format!("{src}.intermediate.dense.weight"),
                true,
            ),
            mapping(
                &names.ffn_up_bias,
                &format!("{src}.intermediate.dense.bias"),
                false,
            ),
            mapping(
                &names.ffn_down_weight,
                &format!("{src}.output.dense.weight"),
                true,
            ),
            mapping(
                &names.ffn_down_bias,
                &format!("{src}.output.dense.bias"),
                false,
            ),
            mapping(
                &names.ffn_norm_weight,
                &format!("{src}.output.LayerNorm.weight"),
                false,
            ),
            mapping(
                &names.ffn_norm_bias,
                &format!("{src}.output.LayerNorm.bias"),
                false,
            ),
        ]);
    }
    mappings.push(mapping(PUNC_HEAD_WEIGHT, "classifier.weight", true));
    mappings.push(mapping(PUNC_HEAD_BIAS, "classifier.bias", false));
    mappings
}

/// ggml `ne` dims for a mapped tensor: 2D linear/embedding weights are stored
/// with reversed dims (PyTorch `[out, in]` -> `[in, out]`); everything else
/// keeps its logical shape (rank-0 scalars, which BERT has none of, would map
/// to `[1]`).
fn ggml_dims(header: &SafetensorsTensorHeader, reverse_dims: bool) -> Vec<u64> {
    if reverse_dims && header.shape.len() == 2 {
        vec![header.shape[1], header.shape[0]]
    } else if header.shape.is_empty() {
        vec![1]
    } else {
        header.shape.clone()
    }
}

/// Whether a 2D weight can be K-quantized (ggml K-quant superblock needs
/// `ne0 % 256 == 0`). BERT-base's `ne0` is always 768 or 3072, so all 2D
/// weights qualify; anything that does not falls back to F16.
fn kquant_eligible(dims: &[u64]) -> bool {
    dims.len() == 2 && dims[0].is_multiple_of(256)
}

fn build_tensor(
    safetensors: &SafetensorsFile,
    map: &TensorMapping,
    quantization: FireRedPuncQuantizationMode,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    let header = safetensors.tensor(&map.source).ok_or_else(|| {
        validate_error(format!(
            "FireRedPunc source is missing required tensor '{}'",
            map.source
        ))
    })?;
    let data = safetensors.tensor_data(header)?;
    let values = decode_safetensors_payload_as_f32(&header.name, &header.dtype, data)?;
    let dims = ggml_dims(header, map.reverse_dims);
    let expected: u64 = dims.iter().product();
    if expected as usize != values.len() {
        return Err(validate_error(format!(
            "FireRedPunc tensor '{}' decoded {} values but dims {dims:?} need {expected}",
            map.source,
            values.len(),
        )));
    }

    // Quantize the 2D weights per the requested mode; keep 1D tensors (biases,
    // LayerNorm affines) exact in F16. The loader dequantizes everything on
    // read either way.
    let quant_type = match quantization {
        FireRedPuncQuantizationMode::Fp16 => None,
        FireRedPuncQuantizationMode::Q8_0 if dims.len() == 2 => Some(GgufWriteTensorType::Q8_0),
        FireRedPuncQuantizationMode::Q4_K if kquant_eligible(&dims) => {
            Some(GgufWriteTensorType::Q4_K)
        }
        _ => None,
    };

    let (tensor_type, tensor_data) = match quant_type {
        Some(qtype) => {
            let quantized = quantize_f32_to_ggml_tensor_data(qtype, &dims, &values)
                .map_err(|error| validate_error(format!("FireRedPunc quantize failed: {error}")))?;
            (qtype, quantized)
        }
        None => (
            GgufWriteTensorType::F16,
            encode_f16_bits_le(values.into_iter().map(f32_to_f16_bits).collect()),
        ),
    };
    Ok(GgufWriteTensor {
        name: map.target.clone(),
        dims,
        tensor_type,
        data: tensor_data,
    })
}

/// Read the upstream WordPiece `vocab.txt` (one token per line) into the token
/// list stored as `tokenizer.ggml.tokens`. The trailing newline is dropped;
/// interior blank lines are preserved as tokens (BERT `[unusedNN]` slots), so
/// the count stays byte-faithful to the upstream vocabulary.
fn read_vocab_tokens(path: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|source| LocalSourceImportError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let tokens: Vec<String> = text
        .strip_suffix('\n')
        .unwrap_or(&text)
        .split('\n')
        .map(|line| line.to_string())
        .collect();
    Ok(tokens)
}

fn runtime_metadata(
    request: &FireRedPuncImportRequest,
    tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put("general.architecture", FIRERED_PUNC_ARCHITECTURE_VALUE);
    put(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put(OASR_METADATA_KEY_MODEL_FAMILY, FIRERED_PUNC_MODEL_FAMILY);
    put(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        FIRERED_PUNC_ARCHITECTURE_VALUE,
    );
    put("openasr.model.id", &request.model_id);
    put("openasr.source.name", &request.source_name);
    put("openasr.source.revision", &request.source_revision);
    put("openasr.license.name", &request.license_name);
    put("openasr.license.source", &request.license_source);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32(
        FIRERED_PUNC_BLOCK_COUNT_KEY,
        FIRERED_PUNC_EXPECTED_LAYERS as u32,
    );
    put_u32(
        FIRERED_PUNC_EMBEDDING_LENGTH_KEY,
        FIRERED_PUNC_EXPECTED_D_MODEL as u32,
    );
    put_u32(
        FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY,
        FIRERED_PUNC_EXPECTED_FFN_DIM as u32,
    );
    put_u32(
        FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY,
        FIRERED_PUNC_EXPECTED_HEADS as u32,
    );
    put_u32(
        FIRERED_PUNC_CONTEXT_LENGTH_KEY,
        FIRERED_PUNC_EXPECTED_MAX_POSITIONS as u32,
    );
    put_u32(
        FIRERED_PUNC_VOCAB_SIZE_KEY,
        FIRERED_PUNC_EXPECTED_VOCAB_SIZE as u32,
    );
    put_u32(
        FIRERED_PUNC_LABEL_COUNT_KEY,
        FIRERED_PUNC_LABEL_COUNT as u32,
    );

    metadata.insert(
        TOKENIZER_GGML_TOKENS_KEY.to_string(),
        GgufWriteValue::StringArray(tokens.to_vec()),
    );
    metadata
}

/// Convert a local FireRedPunc safetensors + vocab source into a punctuation
/// `.oasr` pack.
pub fn convert_local_firered_punc_source_to_runtime_pack(
    request: &FireRedPuncImportRequest,
) -> Result<FireRedPuncImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_pack)?;
    let safetensors = SafetensorsFile::open(&request.source_safetensors)?;
    let tokens = read_vocab_tokens(&request.vocab_txt)?;
    if tokens.len() != FIRERED_PUNC_EXPECTED_VOCAB_SIZE {
        return Err(validate_error(format!(
            "FireRedPunc vocab.txt has {} tokens, expected {FIRERED_PUNC_EXPECTED_VOCAB_SIZE}",
            tokens.len()
        )));
    }
    for special in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] {
        if !tokens.iter().any(|token| token == special) {
            return Err(validate_error(format!(
                "FireRedPunc vocab.txt is missing required special token '{special}'"
            )));
        }
    }

    let mappings = tensor_mappings();
    let mut tensors = Vec::with_capacity(mappings.len());
    for map in &mappings {
        tensors.push(build_tensor(&safetensors, map, request.quantization)?);
    }

    let metadata = runtime_metadata(request, &tokens);
    write_gguf_file_v0(&request.output_pack, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "FireRedPunc GGUF writer failed for '{}': {error}",
            request.output_pack.display()
        ))
    })?;
    Ok(FireRedPuncImportResult {
        output_path: request.output_pack.clone(),
        tensor_count: tensors.len(),
        token_count: tokens.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_map_covers_every_runtime_tensor_name_once() {
        let mappings = tensor_mappings();
        // 5 embedding/norm + 16 per layer * 12 + 2 head = 5 + 192 + 2 = 199.
        assert_eq!(mappings.len(), 199);
        let mut targets: Vec<&str> = mappings.iter().map(|m| m.target.as_str()).collect();
        targets.sort_unstable();
        let count = targets.len();
        targets.dedup();
        assert_eq!(targets.len(), count, "target tensor names must be unique");
    }

    #[test]
    fn ggml_dims_reverse_only_2d_weights() {
        let header = SafetensorsTensorHeader {
            name: "w".to_string(),
            dtype: "F32".to_string(),
            shape: vec![5, 768],
            data_offsets: [0, 0],
        };
        assert_eq!(ggml_dims(&header, true), vec![768, 5]);
        assert_eq!(ggml_dims(&header, false), vec![5, 768]);
        let bias = SafetensorsTensorHeader {
            name: "b".to_string(),
            dtype: "F32".to_string(),
            shape: vec![768],
            data_offsets: [0, 0],
        };
        assert_eq!(ggml_dims(&bias, true), vec![768]);
    }
}
