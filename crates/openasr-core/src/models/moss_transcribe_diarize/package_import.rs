//! Convert a local MOSS-Transcribe-Diarize (`OpenMOSS/MOSS-Transcribe-Diarize`,
//! 0.9B) source checkpoint into an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! MOSS-Transcribe-Diarize is an Encoder-Adaptor-LLM ASR+diarization model:
//! a Whisper-Medium-architecture audio encoder (80-mel, `d_model=1024`, 24
//! layers, 16 heads, `max_source_positions=1500`) feeds a pure-reshape 4x
//! time-merge (`(B,T,1024) -> (B,T/4,4096)`, no learned weights) into the
//! `VQAdaptor` bridge (`Linear(4096->1024) -> SiLU -> Linear(1024->1024) ->
//! LayerNorm`, the only weighted bridge layer despite the "VQ" name -- there
//! is no vector-quantization codebook in this checkpoint), whose output rows
//! get `masked_scatter`-spliced into a Qwen3-0.6B decoder's prompt embedding
//! stream at `<|audio_pad|>` (token id 151671) positions, bracketed by
//! `<|audio_start|>` (151669) / `<|audio_end|>` (151670). `[S01]`-style
//! speaker labels and inline timestamps are ordinary BPE tokens the Qwen3
//! decoder emits freely -- this importer treats them as opaque text, same as
//! every other token.
//!
//! This importer combines the checkpoint's three parameter groups (single
//! safetensors shard, `model.safetensors.index.json` names one shard) into
//! one GGUF:
//!
//!  1. **Encoder** (`model.whisper_encoder.*`, 367 tensors): standard HF
//!     `WhisperEncoder` naming (`conv1`/`conv2`/`embed_positions`/
//!     `layers.N.*`/`layer_norm`) -- verified against the checkpoint's own
//!     `config.json` `audio_config` (`model_type: "whisper"`) and a byte
//!     inspection of the safetensors header. This importer keeps its own
//!     `moss.enc.*` tensor namespace (see `tensor_names.rs`'s module doc for
//!     why) rather than whisper.cpp's `model.encoder.*` names, since this
//!     pack carries no whisper decoder branch for a shared binding contract
//!     to apply to.
//!  2. **VQAdaptor** (`model.vq_adaptor.layers.{0,2,3}.*`, 6 tensors):
//!     `layers.0` = Linear(4096->1024), `layers.1` = SiLU (no weights),
//!     `layers.2` = Linear(1024->1024), `layers.3` = LayerNorm(1024,
//!     `eps=1e-6`).
//!  3. **Qwen3-0.6B decoder** (`model.language_model.*`, 310 tensors):
//!     standard un-prefixed `Qwen3ForCausalLM` naming, QK-norm present
//!     (`self_attn.{q,k}_norm.weight`), no attention bias tensors (matches
//!     `config.json`'s `text_config.attention_bias: false`),
//!     `tie_word_embeddings: true` (no separate `lm_head.weight` tensor --
//!     verified absent from the safetensors header).
//!
//! Tokenizer: standard Qwen `vocab.json` + `merges.txt` GPT-2-style BPE, plus
//! `tokenizer.json`'s `added_tokens` array (29 entries, ids 151643..151671
//! contiguous, verified against the checkpoint's own file), which is where
//! the ChatML control tokens AND the three audio tokens
//! (`<|audio_start|>`=151669 / `<|audio_end|>`=151670 / `<|audio_pad|>`=151671)
//! live -- none of the three audio tokens are exposed via `vocab.json` or
//! `added_tokens.json` alone, only in `tokenizer.json`.
//!
//! **Stage status**: this importer produces a well-formed, self-describing
//! GGUF with every tensor this family needs and full tokenizer metadata. It
//! is NOT yet wired into `arch/mod.rs`'s family-descriptor table (no
//! `MOSS_TRANSCRIBE_DIARIZE_*` architecture/decode-policy/executor-component
//! registration exists yet), so a pack produced here is not yet runnable by
//! `openasr transcribe` -- the ggml execution graph (Whisper encoder reuse +
//! adaptor + Qwen3 decoder + decode-policy registration) is follow-up work.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, SafetensorsTensorHeader,
    decode_safetensors_payload_as_f16_bits, decode_safetensors_payload_as_f32, encode_f16_bits_le,
    read_source_file_bytes, read_source_json_file, tensor_element_count, validate_error,
    validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1, OasrMetadataBuilder,
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};

use super::tensor_names::{
    ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
    ADAPTOR_NORM_BIAS, ADAPTOR_NORM_WEIGHT, ENC_CONV1_BIAS, ENC_CONV1_WEIGHT, ENC_CONV2_BIAS,
    ENC_CONV2_WEIGHT, ENC_OUT_NORM_BIAS, ENC_OUT_NORM_WEIGHT, ENC_POS_EMBD_WEIGHT,
    LLM_OUTPUT_NORM_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, moss_encoder_layer_tensor_names,
    moss_llm_layer_tensor_names,
};

pub(crate) const MOSS_TD_MODEL_FAMILY: &str = "moss-transcribe-diarize";
pub(crate) const MOSS_TD_GGML_ARCHITECTURE_ID: &str = "moss-transcribe-diarize-whisper-qwen3";
/// Not yet registered in `arch/mod.rs` (see this module's doc comment) --
/// kept here as the value this importer stamps into the pack so the runtime
/// wiring stage can adopt it verbatim without an on-disk format change.
pub(crate) const MOSS_TD_AUDIO_FRONTEND_ID: &str = "moss-transcribe-diarize.fbank80.16khz.mono.v0";
pub(crate) const MOSS_TD_TOKENIZER_ID: &str = "moss-transcribe-diarize.qwen3-bpe.v0";
pub(crate) const MOSS_TD_DECODE_POLICY_ID: &str = "moss-transcribe-diarize.greedy.seq2seq.v0";

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_MERGES_TXT: &str = "merges.txt";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";

const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

const WHISPER_ENCODER_PREFIX: &str = "model.whisper_encoder.";
const VQ_ADAPTOR_PREFIX: &str = "model.vq_adaptor.";
const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";

/// `config.json`'s `audio_token_id` (the `<|audio_pad|>` id the model was
/// trained against) and `audio_merge_size` (the 4x reshape factor) are
/// checkpoint-declared, not hardcoded -- this importer cross-checks them
/// against the tokenizer's own `<|audio_pad|>` entry and fails closed on a
/// mismatch rather than silently trusting one source.
#[derive(Debug, Deserialize)]
struct MossConfigJson {
    audio_token_id: u32,
    audio_merge_size: u32,
    adaptor_input_dim: u32,
    tie_word_embeddings: bool,
    text_config: MossTextConfigJson,
    audio_config: MossAudioConfigJson,
}

#[derive(Debug, Deserialize)]
struct MossTextConfigJson {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    attention_bias: bool,
}

#[derive(Debug, Deserialize)]
struct MossAudioConfigJson {
    num_mel_bins: usize,
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    max_source_positions: usize,
}

#[derive(Debug, Deserialize)]
struct TokenizerJsonAddedToken {
    id: u32,
    content: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<TokenizerJsonAddedToken>,
}

pub type MossTdQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct MossTdImportRequest {
    /// Directory containing the checkpoint's `config.json`, safetensors
    /// shard(s), `vocab.json`, `merges.txt` and `tokenizer.json`.
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: MossTdQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossTdImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
}

pub fn convert_local_moss_transcribe_diarize_source_to_runtime_pack(
    request: &MossTdImportRequest,
) -> Result<MossTdImportResult, LocalSourceImportError> {
    validate_request(request)?;
    let config: MossConfigJson = read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    validate_moss_config(&config)?;

    let safetensors = open_safetensors_shard(&request.source_root)?;

    let mut tensors =
        build_encoder_runtime_tensors(&safetensors, &config.audio_config, request.quantization)?;
    tensors.extend(build_adaptor_runtime_tensors(
        &safetensors,
        &config,
        request.quantization,
    )?);
    tensors.extend(build_llm_runtime_tensors(
        &safetensors,
        &config.text_config,
        config.tie_word_embeddings,
        request.quantization,
    )?);

    let mut tokens = load_vocab_tokens(&request.source_root)?;
    let merges = load_merges(&request.source_root)?;
    let audio_token_ids =
        patch_added_tokens_and_find_audio_tokens(&request.source_root, &mut tokens)?;
    if audio_token_ids.pad != config.audio_token_id {
        return Err(validate_error(format!(
            "moss-transcribe-diarize tokenizer.json '<|audio_pad|>' id {} != \
             config.json audio_token_id {}",
            audio_token_ids.pad, config.audio_token_id
        )));
    }
    if tokens.len() < config.text_config.vocab_size {
        tokens.resize_with(config.text_config.vocab_size, String::new);
    }
    for (index, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{index}>");
        }
    }

    let metadata =
        moss_td_runtime_gguf_metadata(&config, request, &tokens, &merges, audio_token_ids);
    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "moss-transcribe-diarize GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "moss-transcribe-diarize import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(MossTdImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        vocab_size: tokens.len(),
    })
}

fn validate_request(request: &MossTdImportRequest) -> Result<(), LocalSourceImportError> {
    if request.model_id.trim().is_empty() {
        return Err(validate_error(
            "moss-transcribe-diarize local-source converter requires non-empty model_id",
        ));
    }
    validate_output_pack_extension(&request.output_root)
}

fn validate_moss_config(config: &MossConfigJson) -> Result<(), LocalSourceImportError> {
    let text = &config.text_config;
    if text.num_attention_heads == 0
        || !text
            .num_attention_heads
            .is_multiple_of(text.num_key_value_heads.max(1))
        || text.num_key_value_heads == 0
    {
        return Err(validate_error(format!(
            "moss-transcribe-diarize text_config num_attention_heads {} is not a multiple of \
             num_key_value_heads {}",
            text.num_attention_heads, text.num_key_value_heads
        )));
    }
    if text.attention_bias {
        return Err(validate_error(
            "moss-transcribe-diarize text_config.attention_bias=true is not the verified \
             Qwen3 (no-bias) parameterization this importer targets",
        ));
    }
    if !config.tie_word_embeddings {
        return Err(validate_error(
            "moss-transcribe-diarize config.tie_word_embeddings=false is unsupported: this \
             importer expects no separate lm_head.weight tensor",
        ));
    }
    let audio = &config.audio_config;
    if audio.d_model == 0
        || !audio
            .d_model
            .is_multiple_of(audio.encoder_attention_heads.max(1))
    {
        return Err(validate_error(format!(
            "moss-transcribe-diarize audio_config d_model {} is not a multiple of \
             encoder_attention_heads {}",
            audio.d_model, audio.encoder_attention_heads
        )));
    }
    if config.audio_merge_size == 0 {
        return Err(validate_error(
            "moss-transcribe-diarize config.audio_merge_size must be > 0",
        ));
    }
    let expected_adaptor_input = audio.d_model * config.audio_merge_size as usize;
    if config.adaptor_input_dim as usize != expected_adaptor_input {
        return Err(validate_error(format!(
            "moss-transcribe-diarize config.adaptor_input_dim {} != d_model {} * \
             audio_merge_size {} ({})",
            config.adaptor_input_dim,
            audio.d_model,
            config.audio_merge_size,
            expected_adaptor_input
        )));
    }
    Ok(())
}

fn open_safetensors_shard(source_root: &Path) -> Result<SafetensorsFile, LocalSourceImportError> {
    // The published checkpoint ships a single shard
    // (`model-00000-of-00001.safetensors`); rather than hardcode that exact
    // filename (a re-shard would silently break this), resolve it the same
    // way `model.safetensors.index.json` would: pick the sole `*.safetensors`
    // file in the source root.
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(source_root)
        .map_err(|error| {
            validate_error(format!(
                "moss-transcribe-diarize import cannot list '{}': {error}",
                source_root.display()
            ))
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    candidates.sort();
    match candidates.as_slice() {
        [single] => SafetensorsFile::open(single),
        [] => Err(validate_error(format!(
            "moss-transcribe-diarize import found no '*.safetensors' file under '{}'",
            source_root.display()
        ))),
        multiple => Err(validate_error(format!(
            "moss-transcribe-diarize import found {} '*.safetensors' shards under '{}'; \
             multi-shard checkpoints are not supported yet",
            multiple.len(),
            source_root.display()
        ))),
    }
}

// --- shared tensor materialization helpers ---------------------------------

fn f32_tensor(name: &str, dims: Vec<u64>, values: &[f32]) -> GgufWriteTensor {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GgufWriteTensor {
        name: name.to_string(),
        dims,
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

fn f16_tensor_from_source(
    tensor: &SafetensorsTensorHeader,
    data: &[u8],
    target_name: &str,
    target_dims: Vec<u64>,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    let bits = decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
    let expected = tensor_element_count(&tensor.name, &target_dims)?;
    if bits.len() != expected {
        return Err(validate_error(format!(
            "moss-transcribe-diarize tensor '{}' decoded {} values but expected {} for dims {:?}",
            tensor.name,
            bits.len(),
            expected,
            target_dims
        )));
    }
    Ok(GgufWriteTensor {
        name: target_name.to_string(),
        dims: target_dims,
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(bits),
    })
}

fn f32_tensor_from_source(
    tensor: &SafetensorsTensorHeader,
    data: &[u8],
    target_name: &str,
    target_dims: Vec<u64>,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
    let expected = tensor_element_count(&tensor.name, &target_dims)?;
    if values.len() != expected {
        return Err(validate_error(format!(
            "moss-transcribe-diarize tensor '{}' decoded {} values but expected {} for dims {:?}",
            tensor.name,
            values.len(),
            expected,
            target_dims
        )));
    }
    Ok(f32_tensor(target_name, target_dims, &values))
}

/// `[out, in]` safetensors row-major -> ggml's `[in, out]` (OutputByInput)
/// on-disk convention: reverse the dims, keep the flat byte layout (a
/// row-major `[out, in]` buffer IS a row-major `[in, out]`-shaped ggml tensor
/// read back with `ne0=in, ne1=out` -- this is the same "just relabel dims,
/// don't transpose bytes" convention every other importer in this crate
/// uses for 2D `Linear` weights).
fn reversed_dims(shape: &[u64]) -> Vec<u64> {
    let mut dims = shape.to_vec();
    dims.reverse();
    dims
}

fn quantized_linear_tensor_type(
    dims: &[u64],
    quantization: MossTdQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == MossTdQuantizationMode::Fp16 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn maybe_quantized_linear_tensor(
    tensor: &SafetensorsTensorHeader,
    data: &[u8],
    target_name: &str,
    target_dims: Vec<u64>,
    quantization: MossTdQuantizationMode,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    match quantized_linear_tensor_type(&target_dims, quantization) {
        Some(tensor_type) => {
            let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
            let expected = tensor_element_count(&tensor.name, &target_dims)?;
            if values.len() != expected {
                return Err(validate_error(format!(
                    "moss-transcribe-diarize tensor '{}' decoded {} values but expected {} for \
                     dims {:?}",
                    tensor.name,
                    values.len(),
                    expected,
                    target_dims
                )));
            }
            let quantized = quantize_f32_to_ggml_tensor_data(tensor_type, &target_dims, &values)
                .map_err(|error| {
                    validate_error(format!(
                        "moss-transcribe-diarize quantization failed for '{}' -> '{target_name}' \
                     ({tensor_type:?}): {error}",
                        tensor.name
                    ))
                })?;
            Ok(GgufWriteTensor {
                name: target_name.to_string(),
                dims: target_dims,
                tensor_type,
                data: quantized,
            })
        }
        None => f16_tensor_from_source(tensor, data, target_name, target_dims),
    }
}

fn tensor_by_name<'a>(
    safetensors: &'a SafetensorsFile,
    name: &str,
) -> Result<(&'a SafetensorsTensorHeader, &'a [u8]), LocalSourceImportError> {
    let tensor = safetensors.tensor(name).ok_or_else(|| {
        validate_error(format!(
            "moss-transcribe-diarize source is missing tensor '{name}'"
        ))
    })?;
    let data = safetensors.tensor_data(tensor)?;
    Ok((tensor, data))
}

// --- Whisper-Medium-style encoder ------------------------------------------

fn build_encoder_runtime_tensors(
    safetensors: &SafetensorsFile,
    audio: &MossAudioConfigJson,
    quantization: MossTdQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();

    let (conv1_w, conv1_w_data) = tensor_by_name(
        safetensors,
        &format!("{WHISPER_ENCODER_PREFIX}conv1.weight"),
    )?;
    out.push(f16_tensor_from_source(
        conv1_w,
        conv1_w_data,
        ENC_CONV1_WEIGHT,
        reversed_dims(&conv1_w.shape),
    )?);
    let (conv1_b, conv1_b_data) =
        tensor_by_name(safetensors, &format!("{WHISPER_ENCODER_PREFIX}conv1.bias"))?;
    out.push(f32_tensor_from_source(
        conv1_b,
        conv1_b_data,
        ENC_CONV1_BIAS,
        conv1_b.shape.clone(),
    )?);
    let (conv2_w, conv2_w_data) = tensor_by_name(
        safetensors,
        &format!("{WHISPER_ENCODER_PREFIX}conv2.weight"),
    )?;
    out.push(f16_tensor_from_source(
        conv2_w,
        conv2_w_data,
        ENC_CONV2_WEIGHT,
        reversed_dims(&conv2_w.shape),
    )?);
    let (conv2_b, conv2_b_data) =
        tensor_by_name(safetensors, &format!("{WHISPER_ENCODER_PREFIX}conv2.bias"))?;
    out.push(f32_tensor_from_source(
        conv2_b,
        conv2_b_data,
        ENC_CONV2_BIAS,
        conv2_b.shape.clone(),
    )?);
    let (pos_embd, pos_embd_data) = tensor_by_name(
        safetensors,
        &format!("{WHISPER_ENCODER_PREFIX}embed_positions.weight"),
    )?;
    out.push(f16_tensor_from_source(
        pos_embd,
        pos_embd_data,
        ENC_POS_EMBD_WEIGHT,
        pos_embd.shape.clone(),
    )?);
    let (norm_w, norm_w_data) = tensor_by_name(
        safetensors,
        &format!("{WHISPER_ENCODER_PREFIX}layer_norm.weight"),
    )?;
    out.push(f32_tensor_from_source(
        norm_w,
        norm_w_data,
        ENC_OUT_NORM_WEIGHT,
        norm_w.shape.clone(),
    )?);
    let (norm_b, norm_b_data) = tensor_by_name(
        safetensors,
        &format!("{WHISPER_ENCODER_PREFIX}layer_norm.bias"),
    )?;
    out.push(f32_tensor_from_source(
        norm_b,
        norm_b_data,
        ENC_OUT_NORM_BIAS,
        norm_b.shape.clone(),
    )?);

    for layer in 0..audio.encoder_layers {
        let names = moss_encoder_layer_tensor_names(layer);
        let src = |suffix: &str| format!("{WHISPER_ENCODER_PREFIX}layers.{layer}.{suffix}");

        let (t, d) = tensor_by_name(safetensors, &src("self_attn_layer_norm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_norm_weight,
            t.shape.clone(),
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn_layer_norm.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_norm_bias,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("self_attn.q_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_q_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.q_proj.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_q_bias,
            t.shape.clone(),
        )?);

        // Whisper's key projection has no bias (upstream `WhisperAttention`
        // only sets `bias=False` on `k_proj`, matching the safetensors
        // header this importer verified).
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.k_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_k_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("self_attn.v_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_v_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.v_proj.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_v_bias,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("self_attn.out_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_out_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.out_proj.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_out_bias,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("final_layer_norm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.ffn_norm_weight,
            t.shape.clone(),
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("final_layer_norm.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.ffn_norm_bias,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("fc1.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.ffn_up_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("fc1.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.ffn_up_bias,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("fc2.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.ffn_down_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("fc2.bias"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.ffn_down_bias,
            t.shape.clone(),
        )?);
    }

    Ok(out)
}

// --- VQAdaptor bridge -------------------------------------------------------

fn build_adaptor_runtime_tensors(
    safetensors: &SafetensorsFile,
    config: &MossConfigJson,
    quantization: MossTdQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();

    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.0.weight"))?;
    if t.shape.as_slice()
        != [
            config.text_config.hidden_size as u64,
            config.adaptor_input_dim as u64,
        ]
    {
        return Err(validate_error(format!(
            "moss-transcribe-diarize adaptor linear1 weight shape {:?} != expected [{}, {}]",
            t.shape, config.text_config.hidden_size, config.adaptor_input_dim
        )));
    }
    out.push(maybe_quantized_linear_tensor(
        t,
        d,
        ADAPTOR_LINEAR1_WEIGHT,
        reversed_dims(&t.shape),
        quantization,
    )?);
    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.0.bias"))?;
    out.push(f32_tensor_from_source(
        t,
        d,
        ADAPTOR_LINEAR1_BIAS,
        t.shape.clone(),
    )?);

    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.2.weight"))?;
    out.push(maybe_quantized_linear_tensor(
        t,
        d,
        ADAPTOR_LINEAR2_WEIGHT,
        reversed_dims(&t.shape),
        quantization,
    )?);
    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.2.bias"))?;
    out.push(f32_tensor_from_source(
        t,
        d,
        ADAPTOR_LINEAR2_BIAS,
        t.shape.clone(),
    )?);

    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.3.weight"))?;
    out.push(f32_tensor_from_source(
        t,
        d,
        ADAPTOR_NORM_WEIGHT,
        t.shape.clone(),
    )?);
    let (t, d) = tensor_by_name(safetensors, &format!("{VQ_ADAPTOR_PREFIX}layers.3.bias"))?;
    out.push(f32_tensor_from_source(
        t,
        d,
        ADAPTOR_NORM_BIAS,
        t.shape.clone(),
    )?);

    Ok(out)
}

// --- Qwen3-0.6B decoder ------------------------------------------------------

fn build_llm_runtime_tensors(
    safetensors: &SafetensorsFile,
    text: &MossTextConfigJson,
    tie_word_embeddings: bool,
    quantization: MossTdQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    let (t, d) = tensor_by_name(
        safetensors,
        &format!("{LANGUAGE_MODEL_PREFIX}embed_tokens.weight"),
    )?;
    if t.shape.as_slice() != [text.vocab_size as u64, text.hidden_size as u64] {
        return Err(validate_error(format!(
            "moss-transcribe-diarize embed_tokens shape {:?} != expected [{}, {}]",
            t.shape, text.vocab_size, text.hidden_size
        )));
    }
    // Embedding table: same "just relabel dims, gather is column-major
    // either way" convention as every other family's token-embedding tensor
    // in this crate (rows stay contiguous per-token; ggml reads it back as
    // `[hidden, vocab]`).
    out.push(f16_tensor_from_source(
        t,
        d,
        LLM_TOKEN_EMBD_WEIGHT,
        vec![text.hidden_size as u64, text.vocab_size as u64],
    )?);
    seen.insert(LLM_TOKEN_EMBD_WEIGHT.to_string());
    if !tie_word_embeddings {
        // Guarded by `validate_moss_config` above; defensive redundancy so a
        // future caller that skips validation still fails closed instead of
        // silently dropping the lm_head branch.
        return Err(validate_error(
            "moss-transcribe-diarize untied embeddings are not supported by this importer",
        ));
    }

    let (t, d) = tensor_by_name(safetensors, &format!("{LANGUAGE_MODEL_PREFIX}norm.weight"))?;
    out.push(f32_tensor_from_source(
        t,
        d,
        LLM_OUTPUT_NORM_WEIGHT,
        t.shape.clone(),
    )?);

    for layer in 0..text.num_hidden_layers {
        let names = moss_llm_layer_tensor_names(layer);
        let src = |suffix: &str| format!("{LANGUAGE_MODEL_PREFIX}layers.{layer}.{suffix}");

        let (t, d) = tensor_by_name(safetensors, &src("input_layernorm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_norm_weight,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("self_attn.q_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_q_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.k_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_k_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.v_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_v_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.o_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.attn_output_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.q_norm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_q_norm_weight,
            t.shape.clone(),
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("self_attn.k_norm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.attn_k_norm_weight,
            t.shape.clone(),
        )?);

        let (t, d) = tensor_by_name(safetensors, &src("post_attention_layernorm.weight"))?;
        out.push(f32_tensor_from_source(
            t,
            d,
            &names.ffn_norm_weight,
            t.shape.clone(),
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("mlp.gate_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.ffn_gate_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("mlp.up_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.ffn_up_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
        let (t, d) = tensor_by_name(safetensors, &src("mlp.down_proj.weight"))?;
        out.push(maybe_quantized_linear_tensor(
            t,
            d,
            &names.ffn_down_weight,
            reversed_dims(&t.shape),
            quantization,
        )?);
    }

    Ok(out)
}

// --- Qwen-style tokenizer (vocab.json + merges.txt + tokenizer.json added) -

fn load_vocab_tokens(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let vocab: BTreeMap<String, usize> = read_source_json_file(source_root, SOURCE_VOCAB_JSON)?;
    if vocab.is_empty() {
        return Err(validate_error(
            "moss-transcribe-diarize vocab.json cannot be empty",
        ));
    }
    let mut pairs = vocab.into_iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(_, token_id)| *token_id);
    let max_id = pairs.last().map(|(_, token_id)| *token_id).ok_or_else(|| {
        validate_error("moss-transcribe-diarize vocab.json cannot determine max token id")
    })?;
    let mut tokens = vec![String::new(); max_id + 1];
    for (token, token_id) in pairs {
        tokens[token_id] = token;
    }
    Ok(tokens)
}

fn load_merges(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let bytes = read_source_file_bytes(source_root, SOURCE_MERGES_TXT)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        validate_error(format!(
            "moss-transcribe-diarize merges.txt is not valid UTF-8: {error}"
        ))
    })?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdAudioTokenIds {
    pub start: u32,
    pub end: u32,
    pub pad: u32,
}

/// Reads `tokenizer.json`'s `added_tokens` array (ids 151643..151671 in the
/// verified checkpoint -- ChatML control tokens plus the three audio
/// tokens), patches them into `tokens`, and resolves the three audio token
/// ids by their literal content string (never assumed contiguous /
/// offset-derived from each other) so the caller can cross-check
/// `<|audio_pad|>` against `config.json`'s `audio_token_id`.
fn patch_added_tokens_and_find_audio_tokens(
    source_root: &Path,
    tokens: &mut Vec<String>,
) -> Result<MossTdAudioTokenIds, LocalSourceImportError> {
    let parsed: TokenizerJson = read_source_json_file(source_root, SOURCE_TOKENIZER_JSON)?;
    if parsed.added_tokens.is_empty() {
        return Err(validate_error(
            "moss-transcribe-diarize tokenizer.json has an empty 'added_tokens' array",
        ));
    }
    let mut start = None;
    let mut end = None;
    let mut pad = None;
    for entry in &parsed.added_tokens {
        let token_id = entry.id as usize;
        if token_id >= tokens.len() {
            tokens.resize_with(token_id + 1, String::new);
        }
        tokens[token_id] = entry.content.clone();
        match entry.content.as_str() {
            "<|audio_start|>" => start = Some(entry.id),
            "<|audio_end|>" => end = Some(entry.id),
            "<|audio_pad|>" => pad = Some(entry.id),
            _ => {}
        }
    }
    let missing = |name: &str| {
        validate_error(format!(
            "moss-transcribe-diarize tokenizer.json 'added_tokens' has no '{name}' entry"
        ))
    };
    Ok(MossTdAudioTokenIds {
        start: start.ok_or_else(|| missing("<|audio_start|>"))?,
        end: end.ok_or_else(|| missing("<|audio_end|>"))?,
        pad: pad.ok_or_else(|| missing("<|audio_pad|>"))?,
    })
}

// --- metadata ---------------------------------------------------------

fn moss_td_runtime_gguf_metadata(
    config: &MossConfigJson,
    request: &MossTdImportRequest,
    tokens: &[String],
    merges: &[String],
    audio_token_ids: MossTdAudioTokenIds,
) -> BTreeMap<String, GgufWriteValue> {
    let text = &config.text_config;
    let audio = &config.audio_config;
    OasrMetadataBuilder::new()
        .str(GENERAL_ARCHITECTURE_KEY, MOSS_TD_GGML_ARCHITECTURE_ID)
        .str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1)
        .str(OASR_METADATA_KEY_MODEL_FAMILY, MOSS_TD_MODEL_FAMILY)
        .str(
            OASR_METADATA_KEY_MODEL_ARCHITECTURE,
            MOSS_TD_GGML_ARCHITECTURE_ID,
        )
        .str(OASR_METADATA_KEY_AUDIO_FRONTEND, MOSS_TD_AUDIO_FRONTEND_ID)
        .str(OASR_METADATA_KEY_DECODE_POLICY, MOSS_TD_DECODE_POLICY_ID)
        .str(GGML_TOKENIZER_ID_KEY, MOSS_TD_TOKENIZER_ID)
        .str(OPENASR_MODEL_ID_KEY, &request.model_id)
        .str(TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2)
        .string_array(TOKENIZER_GGML_TOKENS_KEY, tokens)
        .string_array(TOKENIZER_GGML_MERGES_KEY, merges)
        .u32("moss_td.encoder.n_layers", audio.encoder_layers as u32)
        .u32("moss_td.encoder.d_model", audio.d_model as u32)
        .u32(
            "moss_td.encoder.n_heads",
            audio.encoder_attention_heads as u32,
        )
        .u32("moss_td.encoder.ffn_dim", audio.encoder_ffn_dim as u32)
        .u32("moss_td.encoder.n_mels", audio.num_mel_bins as u32)
        .u32(
            "moss_td.encoder.max_source_positions",
            audio.max_source_positions as u32,
        )
        .u32("moss_td.adaptor.merge_size", config.audio_merge_size)
        .u32("moss_td.adaptor.input_dim", config.adaptor_input_dim)
        .u32("moss_td.llm.n_layers", text.num_hidden_layers as u32)
        .u32("moss_td.llm.d_model", text.hidden_size as u32)
        .u32("moss_td.llm.ffn_dim", text.intermediate_size as u32)
        .u32("moss_td.llm.n_heads", text.num_attention_heads as u32)
        .u32("moss_td.llm.n_kv_heads", text.num_key_value_heads as u32)
        .u32("moss_td.llm.head_dim", text.head_dim as u32)
        .u32("moss_td.llm.vocab_size", tokens.len() as u32)
        .u32(
            "moss_td.llm.max_positions",
            text.max_position_embeddings as u32,
        )
        .u32("moss_td.llm.audio_start_token_id", audio_token_ids.start)
        .u32("moss_td.llm.audio_end_token_id", audio_token_ids.end)
        .u32("moss_td.llm.audio_pad_token_id", audio_token_ids.pad)
        .str("moss_td.llm.rms_norm_eps", text.rms_norm_eps)
        .str("moss_td.llm.rope_theta", text.rope_theta)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_json(dir: &Path, name: &str, value: &serde_json::Value) {
        let mut file = std::fs::File::create(dir.join(name)).expect("create json fixture");
        file.write_all(serde_json::to_string(value).unwrap().as_bytes())
            .expect("write json fixture");
    }

    #[test]
    fn validate_moss_config_accepts_the_real_checkpoint_shape() {
        let config = MossConfigJson {
            audio_token_id: 151_671,
            audio_merge_size: 4,
            adaptor_input_dim: 4096,
            tie_word_embeddings: true,
            text_config: MossTextConfigJson {
                vocab_size: 151_936,
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                max_position_embeddings: 131_072,
                attention_bias: false,
            },
            audio_config: MossAudioConfigJson {
                num_mel_bins: 80,
                d_model: 1024,
                encoder_layers: 24,
                encoder_attention_heads: 16,
                encoder_ffn_dim: 4096,
                max_source_positions: 1500,
            },
        };
        assert!(validate_moss_config(&config).is_ok());
    }

    #[test]
    fn validate_moss_config_rejects_attention_bias() {
        let mut config = base_config_for_test();
        config.text_config.attention_bias = true;
        assert!(validate_moss_config(&config).is_err());
    }

    #[test]
    fn validate_moss_config_rejects_untied_embeddings() {
        let mut config = base_config_for_test();
        config.tie_word_embeddings = false;
        assert!(validate_moss_config(&config).is_err());
    }

    #[test]
    fn validate_moss_config_rejects_adaptor_input_dim_mismatch() {
        let mut config = base_config_for_test();
        config.adaptor_input_dim = 999;
        assert!(validate_moss_config(&config).is_err());
    }

    fn base_config_for_test() -> MossConfigJson {
        MossConfigJson {
            audio_token_id: 151_671,
            audio_merge_size: 4,
            adaptor_input_dim: 4096,
            tie_word_embeddings: true,
            text_config: MossTextConfigJson {
                vocab_size: 151_936,
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                max_position_embeddings: 131_072,
                attention_bias: false,
            },
            audio_config: MossAudioConfigJson {
                num_mel_bins: 80,
                d_model: 1024,
                encoder_layers: 24,
                encoder_attention_heads: 16,
                encoder_ffn_dim: 4096,
                max_source_positions: 1500,
            },
        }
    }

    #[test]
    fn load_merges_skips_comments_and_blank_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("merges.txt"), "#version: 0.2\na b\n\nc d\n")
            .expect("write merges.txt");
        let merges = load_merges(dir.path()).expect("load merges");
        assert_eq!(merges, vec!["a b".to_string(), "c d".to_string()]);
    }

    #[test]
    fn patch_added_tokens_resolves_audio_pad_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_json(
            dir.path(),
            "tokenizer.json",
            &serde_json::json!({
                "added_tokens": [
                    {"id": 151643, "content": "<|endoftext|>"},
                    {"id": 151669, "content": "<|audio_start|>"},
                    {"id": 151670, "content": "<|audio_end|>"},
                    {"id": 151671, "content": "<|audio_pad|>"},
                ]
            }),
        );
        let mut tokens = vec![String::new(); 151_643];
        let audio_ids = patch_added_tokens_and_find_audio_tokens(dir.path(), &mut tokens)
            .expect("patch added tokens");
        assert_eq!(audio_ids.pad, 151_671);
        assert_eq!(audio_ids.start, 151_669);
        assert_eq!(audio_ids.end, 151_670);
        assert_eq!(tokens[151_669], "<|audio_start|>");
        assert_eq!(tokens[151_671], "<|audio_pad|>");
    }

    #[test]
    fn patch_added_tokens_rejects_missing_audio_pad() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_json(
            dir.path(),
            "tokenizer.json",
            &serde_json::json!({
                "added_tokens": [
                    {"id": 151643, "content": "<|endoftext|>"},
                ]
            }),
        );
        let mut tokens = vec![String::new(); 151_643];
        assert!(patch_added_tokens_and_find_audio_tokens(dir.path(), &mut tokens).is_err());
    }

    // --- Real-checkpoint conversion + tensor parity ------------------------
    //
    // Points at the private dev-only checkpoint download (~1.7GB bf16
    // safetensors, NOT committed -- same convention as every other family's
    // real-dev-pack test in this crate, e.g. `firered_llm::executor`'s
    // `dev_pack_path()`). Skips silently when absent so this stays runnable
    // without the multi-GB download.
    fn dev_checkpoint_root() -> PathBuf {
        PathBuf::from("/Volumes/QuintinDocument/openasr-dev/tmp/moss-td/model")
    }

    #[test]
    fn golden_diff_converted_pack_tensors_match_source_checkpoint_bit_for_bit() {
        let source_root = dev_checkpoint_root();
        if !source_root.exists() {
            eprintln!("skipping: {} not present", source_root.display());
            return;
        }
        let output_dir = tempfile::tempdir().expect("tempdir");
        let output_root = output_dir.path().join("moss-transcribe-diarize-fp16.oasr");
        let request = MossTdImportRequest {
            source_root: source_root.clone(),
            output_root: output_root.clone(),
            model_id: "moss-transcribe-diarize-test".to_string(),
            quantization: MossTdQuantizationMode::Fp16,
        };
        let result = convert_local_moss_transcribe_diarize_source_to_runtime_pack(&request)
            .expect("moss-transcribe-diarize conversion");
        // Encoder (367) + adaptor (6) + llm embed+norm (2) + llm 28 layers x
        // 11 tensors (308) = 683, matching the source safetensors tensor
        // count exactly (no tensor dropped, none duplicated).
        assert_eq!(result.tensor_count, 683);
        assert_eq!(result.vocab_size, 151_936);

        // Spot-check tensor-value parity on a representative sample from
        // each of the three branches: fp16 conversion is lossy relative to
        // the source bf16 (both are 16-bit but with different
        // mantissa/exponent splits), so this asserts closeness, not bit
        // equality -- the golden-diff test at the ggml-execution layer
        // (follow-up work, see this module's doc comment) is where
        // token-level parity against the reference PyTorch forward pass
        // gets checked.
        let safetensors = open_safetensors_shard(&source_root).expect("open safetensors");
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(&output_root)
            .expect("open converted pack for parity read");
        assert_tensor_close(
            &safetensors,
            &reader,
            "model.whisper_encoder.conv1.bias",
            ENC_CONV1_BIAS,
            &[1024],
        );
        assert_tensor_close(
            &safetensors,
            &reader,
            "model.vq_adaptor.layers.3.weight",
            ADAPTOR_NORM_WEIGHT,
            &[1024],
        );
        assert_tensor_close(
            &safetensors,
            &reader,
            "model.language_model.norm.weight",
            LLM_OUTPUT_NORM_WEIGHT,
            &[1024],
        );
    }

    fn assert_tensor_close(
        safetensors: &SafetensorsFile,
        reader: &crate::ggml_runtime::GgufTensorDataReader,
        source_name: &str,
        target_name: &str,
        dims: &[u64],
    ) {
        let (tensor, data) = tensor_by_name(safetensors, source_name).expect("source tensor");
        let expected = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)
            .expect("decode source f32");
        let actual = reader
            .host_tensor_f32_copy_dequantized_by_name(target_name, dims)
            .expect("read converted tensor");
        assert_eq!(
            expected.len(),
            actual.len(),
            "{target_name} length mismatch"
        );
        for (index, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (want - got).abs();
            assert!(
                diff <= 1e-2 * want.abs().max(1.0),
                "{target_name}[{index}] parity mismatch: source(bf16)={want} converted(f16)={got}"
            );
        }
    }
}
