//! Convert a local FireRedASR2-LLM source into an OpenASR `.oasr` (GGUF-v0)
//! runtime pack.
//!
//! FireRedASR2-LLM (`FireRedTeam/FireRedASR2-LLM`) is an Encoder-Adapter-LLM
//! ASR model: the `firered-aed` Conformer encoder (identical architecture,
//! independently-trained weights -- see the module-level note below) feeds a
//! 2x frame-stacking Adapter (2 `Linear` + `ReLU`), which splices its output
//! into a Qwen2-7B-Instruct decoder's prompt embedding stream. This importer
//! combines THREE upstream sources into one GGUF:
//!
//!  1. **Encoder + Adapter** (`encoder.*` / `encoder_projector.*`, 551 + 4
//!     tensors): `model.pth.tar`'s `model_state_dict`, normalized to F32
//!     safetensors by `pt_to_safetensors.py` (same tool the `firered_aed`
//!     importer's source uses). Maps 1:1 onto
//!     `firered_aed::package_import`'s encoder branch (this importer's
//!     encoder tensor map is a direct trim of that function down to its
//!     encoder cases -- no decoder exists in this family).
//!  2. **cmvn.txt**: `firered_llm_cmvn_ark_to_txt.py`'s output (a Kaldi
//!     text-matrix rendering of the upstream binary `cmvn.ark`), parsed with
//!     the same accumulator formula `firered_aed::package_import` uses.
//!  3. **LoRA-merged Qwen2** (`model.layers.*` / `model.embed_tokens.weight` /
//!     `model.norm.weight` / `lm_head.weight`, standard un-prefixed
//!     `Qwen2ForCausalLM.state_dict()` names, 339 tensors): the output of
//!     `firered_llm_merge_lora.py`, which folds the FireRedASR2-LLM PEFT LoRA
//!     adapter (`model.pth.tar`'s `llm.*` tensors) into the official
//!     `Qwen/Qwen2-7B-Instruct` base weights BEFORE this importer ever sees
//!     them. This importer has no LoRA awareness at all -- by the time it
//!     runs, the source is just a (fine-tuned) Qwen2.
//!
//! **Why the encoder weights cannot be shared with the published
//! `firered-aed-l-v2` pack**: despite byte-identical architecture (verified
//! against `asr_encoder.pth.tar`'s hparam stub and a numeric tensor-value
//! comparison), FireRedASR2-LLM trains with `freeze_encoder=0` -- the encoder
//! is jointly finetuned alongside the Adapter+LoRA, not frozen. See
//! `scratchpad/fr2/T1-findings.md` S3 for the full evidence (Pearson
//! correlation 0.89-0.9999 against `firered-aed-l-v2`'s encoder tensors --
//! "same initialization, diverged after joint training", not independently
//! trained, but also not identical). This importer always reads the
//! encoder's own weights from the LLM checkpoint's `model.pth.tar`; it never
//! links to an already-published firered-aed pack.
//!
//! **Stage status**: this importer produces a well-formed GGUF with every
//! tensor + hparam metadata the runtime needs, and the family is fully wired
//! for execution -- the Qwen2-parameterized `llm_transformer` (qwen3's
//! transformer has QK-norm and no qkv-bias; Qwen2 is the opposite -- see the
//! tensor map below, which carries `attn_{q,k,v}.bias` and omits any
//! `attn_{q,k}_norm`), the Adapter ggml graph, the dedicated executor
//! (`executor::FireRedLlmGgmlExecutor`), and the `firered-llm.greedy.seq2seq.v0`
//! decode-policy registration all exist. A pack produced by this importer is
//! runnable by `openasr transcribe`.

// Module-wide (not narrowed to individual items): matches every other model
// family's importer in this crate (e.g. `firered_aed::package_import`) --
// most of this module's public surface is exercised only from `#[cfg(test)]`
// round-trip tests and the CLI's `model-pack import` dispatch, both of which
// are invisible to `dead_code` analysis per-item. Narrowing this to a smaller
// item list here alone would diverge from the established per-family
// convention without a matching crate-wide pass.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::arch::{
    FIRERED_LLM_AUDIO_FRONTEND_ID, FIRERED_LLM_DECODE_POLICY_ID, FIRERED_LLM_GGML_ARCHITECTURE_ID,
    FIRERED_LLM_MODEL_FAMILY, FIRERED_LLM_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale, filterbank};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, read_source_file_bytes,
    read_source_json_file, tensor_element_count, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};

use super::tensor_names::{
    ADAPTER_LINEAR1_BIAS, ADAPTER_LINEAR1_WEIGHT, ADAPTER_LINEAR2_BIAS, ADAPTER_LINEAR2_WEIGHT,
    LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, qwen2_llm_layer_tensor_names,
};

const SOURCE_ENCODER_ADAPTER_SAFETENSORS: &str = "model.safetensors";
const SOURCE_CMVN_TXT: &str = "cmvn.txt";
const SOURCE_QWEN2_CONFIG_JSON: &str = "config.json";
const SOURCE_QWEN2_VOCAB_JSON: &str = "vocab.json";
const SOURCE_QWEN2_MERGES_TXT: &str = "merges.txt";
const SOURCE_QWEN2_TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

/// The `<speech>` placeholder token (`DEFAULT_SPEECH_TOKEN` upstream) is not
/// baked into any Qwen2 tokenizer file -- `fireredasr2/data/llm_tokenizer.py`
/// appends it at runtime via `add_special_tokens`, landing deterministically
/// at `len(tokenizer)` (151646) because the official Qwen2 tokenizer always
/// starts at exactly 151646 real+added tokens (151643 base vocab.json rows +
/// 3 ChatML added tokens). Verified by literally running this step with
/// `transformers` during stage-1 reconnaissance (`scratchpad/fr2/
/// T1-findings.md` S6) rather than assumed from source reading.
const SPEECH_TOKEN_ID: u32 = 151_646;
const SPEECH_TOKEN_TEXT: &str = "<speech>";
/// ChatML prompt boundary tokens used by `fireredasr_llm.py`'s inference path
/// (`<|im_start|>user\n<speech>{prompt}<|im_end|>\n<|im_start|>assistant\n`),
/// NOT `config.json`'s generic `bos_token_id` (which is 151643, the base
/// `<|endoftext|>` id Qwen2 ships as a fallback bos/eos/pad before any chat
/// template is applied). All three ids are fixed properties of the official
/// Qwen2 tokenizer, not derived from a checkpoint.
const CHATML_IM_START_TOKEN_ID: u32 = 151_644;
const CHATML_IM_END_TOKEN_ID: u32 = 151_645;
const ENDOFTEXT_TOKEN_ID: u32 = 151_643;

/// Kaldi CMVN variance floor (same constant `firered_aed::package_import`
/// uses -- both families' fbank frontends share this formula verbatim).
const CMVN_VARIANCE_FLOOR: f64 = 1e-20;

// fbank frontend contract, identical to `firered_aed`'s (verified against
// `fireredasr2/data/asr_feat.py`, and the two families' `cmvn.ark` files are
// byte-identical -- see T1-findings.md S4).
const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_SHIFT_MS: u32 = 10;
const FFT_SIZE: usize = 512;
const MEL_LOW_HZ: f32 = 20.0;

pub type FireRedLlmQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct FireRedLlmImportRequest {
    /// Directory containing `model.safetensors` (the `pt_to_safetensors.py`
    /// output over `model.pth.tar`'s `model_state_dict`, still holding the
    /// `llm.*` LoRA tensors alongside `encoder.*`/`encoder_projector.*` --
    /// this importer explicitly skips anything under `llm.*`, since the LLM
    /// branch comes from `qwen2_merged_safetensors_path` instead) and
    /// `cmvn.txt` (the `firered_llm_cmvn_ark_to_txt.py` output).
    pub encoder_adapter_source_root: PathBuf,
    /// The LoRA-merged Qwen2 safetensors file (`firered_llm_merge_lora.py`'s
    /// `--out`), standard un-prefixed `Qwen2ForCausalLM.state_dict()` names.
    pub qwen2_merged_safetensors_path: PathBuf,
    /// Directory containing the *official, unmodified* Qwen2-7B-Instruct
    /// `config.json` / `vocab.json` / `merges.txt` / `tokenizer_config.json`
    /// (does not need the safetensors weight shards themselves -- those are
    /// only read via `qwen2_merged_safetensors_path` above).
    pub qwen2_metadata_source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: FireRedLlmQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireRedLlmImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FireRedLlmEncoderHparams {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    conv_kernel: usize,
    subsample_channels: usize,
    subsample_out_dim: usize,
    feature_dim: usize,
    pe_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FireRedLlmAdapterHparams {
    llm_dim: usize,
    downsample_rate: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FireRedLlmDecoderHparams {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_positions: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen2ConfigJson {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: BTreeMap<String, AddedTokenEntry>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenEntry {
    content: String,
}

pub fn convert_local_firered_llm_source_to_runtime_pack(
    request: &FireRedLlmImportRequest,
) -> Result<FireRedLlmImportResult, LocalSourceImportError> {
    validate_request(request)?;
    let encoder_adapter_safetensors = SafetensorsFile::open(
        request
            .encoder_adapter_source_root
            .join(SOURCE_ENCODER_ADAPTER_SAFETENSORS),
    )?;
    let (cmvn_neg_mean, cmvn_inv_stddev) =
        parse_kaldi_cmvn_stats(&request.encoder_adapter_source_root.join(SOURCE_CMVN_TXT))?;

    let encoder_hparams =
        derive_and_validate_encoder_hparams(&encoder_adapter_safetensors, cmvn_neg_mean.len())?;
    let adapter_hparams =
        derive_and_validate_adapter_hparams(&encoder_adapter_safetensors, encoder_hparams.d_model)?;

    let mut tensors =
        build_encoder_adapter_runtime_tensors(&encoder_adapter_safetensors, request.quantization)?;
    tensors.push(f32_tensor(
        "frontend.cmvn.neg_mean",
        vec![encoder_hparams.feature_dim as u64],
        &cmvn_neg_mean,
    ));
    tensors.push(f32_tensor(
        "frontend.cmvn.inv_stddev",
        vec![encoder_hparams.feature_dim as u64],
        &cmvn_inv_stddev,
    ));
    tensors.push(build_mel_filterbank_tensor(encoder_hparams.feature_dim));

    let qwen2_config: Qwen2ConfigJson = read_source_json_file(
        &request.qwen2_metadata_source_root,
        SOURCE_QWEN2_CONFIG_JSON,
    )?;
    let qwen2_safetensors = SafetensorsFile::open(&request.qwen2_merged_safetensors_path)?;
    let decoder_hparams =
        derive_and_validate_decoder_hparams(&qwen2_safetensors, &qwen2_config, adapter_hparams)?;

    let llm_tensors = build_llm_runtime_tensors(&qwen2_safetensors, request.quantization)?;
    tensors.extend(llm_tensors);

    let mut tokens = load_vocab_tokens(&request.qwen2_metadata_source_root)?;
    let merges = load_merges(&request.qwen2_metadata_source_root)?;
    patch_added_tokens(&request.qwen2_metadata_source_root, &mut tokens)?;
    if tokens.len() != SPEECH_TOKEN_ID as usize {
        return Err(validate_error(format!(
            "firered-llm expected the official Qwen2 tokenizer to have exactly {} \
             base+added tokens before appending '<speech>', found {}",
            SPEECH_TOKEN_ID,
            tokens.len()
        )));
    }
    tokens.push(SPEECH_TOKEN_TEXT.to_string());
    if tokens.len() < decoder_hparams.vocab_size {
        tokens.resize_with(decoder_hparams.vocab_size, String::new);
    }
    for (index, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{index}>");
        }
    }

    let metadata = firered_llm_runtime_gguf_metadata(
        &encoder_hparams,
        &adapter_hparams,
        &decoder_hparams,
        request,
        &tokens,
        &merges,
    );
    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "firered-llm GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "firered-llm import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(FireRedLlmImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        vocab_size: tokens.len(),
    })
}

fn validate_request(request: &FireRedLlmImportRequest) -> Result<(), LocalSourceImportError> {
    if request.model_id.trim().is_empty() {
        return Err(validate_error(
            "firered-llm local-source converter requires non-empty model_id",
        ));
    }
    validate_output_pack_extension(&request.output_root)
}

// --- cmvn (identical formula to firered_aed::package_import) --------------

fn parse_kaldi_cmvn_stats(path: &Path) -> Result<(Vec<f32>, Vec<f32>), LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        validate_error(format!(
            "firered-llm import cannot read '{}': {error}",
            path.display()
        ))
    })?;
    let open = text
        .find('[')
        .ok_or_else(|| validate_error("firered-llm cmvn.txt has no '[' matrix opener"))?;
    let close = text
        .rfind(']')
        .ok_or_else(|| validate_error("firered-llm cmvn.txt has no ']' matrix closer"))?;
    if close < open {
        return Err(validate_error("firered-llm cmvn.txt has ']' before '['"));
    }
    let body = &text[open + 1..close];
    let rows: Vec<Vec<f64>> = body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| {
                    token.parse::<f64>().map_err(|error| {
                        validate_error(format!(
                            "firered-llm cmvn.txt has a non-numeric value '{token}': {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<f64>, _>>()
        })
        .collect::<Result<Vec<Vec<f64>>, _>>()?;
    if rows.len() != 2 {
        return Err(validate_error(format!(
            "firered-llm cmvn.txt must contain exactly 2 stat rows, found {}",
            rows.len()
        )));
    }
    let (sums, sum_squares) = (&rows[0], &rows[1]);
    if sums.len() != sum_squares.len() || sums.len() < 2 {
        return Err(validate_error(format!(
            "firered-llm cmvn.txt row lengths are inconsistent ({} vs {})",
            sums.len(),
            sum_squares.len()
        )));
    }
    let dim = sums.len() - 1;
    let count = sums[dim];
    if count < 1.0 {
        return Err(validate_error(format!(
            "firered-llm cmvn.txt frame count {count} must be >= 1"
        )));
    }
    let mut neg_mean = Vec::with_capacity(dim);
    let mut inv_stddev = Vec::with_capacity(dim);
    for d in 0..dim {
        let mean = sums[d] / count;
        let variance = (sum_squares[d] / count - mean * mean).max(CMVN_VARIANCE_FLOOR);
        neg_mean.push((-mean) as f32);
        inv_stddev.push((1.0 / variance.sqrt()) as f32);
    }
    Ok((neg_mean, inv_stddev))
}

fn build_mel_filterbank_tensor(n_mels: usize) -> GgufWriteTensor {
    let fft_bins = FFT_SIZE / 2 + 1;
    let high_hz = (SAMPLE_RATE_HZ as f32) / 2.0;
    let filters = filterbank(FilterbankConfig {
        scale: MelScale::Kaldi,
        sample_rate_hz: SAMPLE_RATE_HZ as f32,
        n_fft: FFT_SIZE,
        n_mels,
        fmin: MEL_LOW_HZ,
        fmax: high_hz,
        mel_point_order: MelPointOrder::SpanTimesIndexFirst,
    });
    let mut bytes = Vec::with_capacity(filters.len() * 4);
    for value in &filters {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GgufWriteTensor {
        name: "firered.mel_filters".to_string(),
        dims: vec![n_mels as u64, fft_bins as u64],
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

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

// --- encoder + adapter branch ----------------------------------------------

/// Mirrors `firered_aed::package_import::TensorClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorClass {
    F32Vector,
    PosBiasFlatten,
    Linear,
    PointwiseConvSqueeze,
    ConvKernel,
    F16Table,
}

/// A direct trim of `firered_aed::package_import::map_firered_tensor_name`
/// down to its `encoder.*` branch (this family has no decoder). Kept as an
/// independent copy rather than a shared helper: the two families' encoder
/// weights are NOT the same data (see this module's doc comment), so sharing
/// the map function would be sharing incidental naming code, not shared
/// architecture ownership -- and `firered_aed::package_import`'s function is
/// private to that module.
fn map_firered_encoder_tensor_name(source_name: &str) -> Option<(String, TensorClass)> {
    use TensorClass::*;
    match source_name {
        "encoder.input_preprocessor.conv.0.weight" => {
            return Some(("enc.subsample.conv1.weight".into(), ConvKernel));
        }
        "encoder.input_preprocessor.conv.0.bias" => {
            return Some(("enc.subsample.conv1.bias".into(), F32Vector));
        }
        "encoder.input_preprocessor.conv.2.weight" => {
            return Some(("enc.subsample.conv2.weight".into(), ConvKernel));
        }
        "encoder.input_preprocessor.conv.2.bias" => {
            return Some(("enc.subsample.conv2.bias".into(), F32Vector));
        }
        "encoder.input_preprocessor.out.weight" => {
            return Some(("enc.subsample.out.weight".into(), Linear));
        }
        "encoder.input_preprocessor.out.bias" => {
            return Some(("enc.subsample.out.bias".into(), F32Vector));
        }
        "encoder.positional_encoding.pe" => return Some(("enc.pos_enc.pe".into(), F16Table)),
        _ => {}
    }
    let rest = source_name.strip_prefix("encoder.layer_stack.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (suffix, class) = match tail {
        "ffn1.net.0.weight" => ("ffn1.norm.weight", F32Vector),
        "ffn1.net.0.bias" => ("ffn1.norm.bias", F32Vector),
        "ffn1.net.1.weight" => ("ffn1.up.weight", Linear),
        "ffn1.net.1.bias" => ("ffn1.up.bias", F32Vector),
        "ffn1.net.4.weight" => ("ffn1.down.weight", Linear),
        "ffn1.net.4.bias" => ("ffn1.down.bias", F32Vector),
        "ffn2.net.0.weight" => ("ffn2.norm.weight", F32Vector),
        "ffn2.net.0.bias" => ("ffn2.norm.bias", F32Vector),
        "ffn2.net.1.weight" => ("ffn2.up.weight", Linear),
        "ffn2.net.1.bias" => ("ffn2.up.bias", F32Vector),
        "ffn2.net.4.weight" => ("ffn2.down.weight", Linear),
        "ffn2.net.4.bias" => ("ffn2.down.bias", F32Vector),
        "mhsa.layer_norm_q.weight" => ("attn.norm_q.weight", F32Vector),
        "mhsa.layer_norm_q.bias" => ("attn.norm_q.bias", F32Vector),
        "mhsa.layer_norm_k.weight" => ("attn.norm_k.weight", F32Vector),
        "mhsa.layer_norm_k.bias" => ("attn.norm_k.bias", F32Vector),
        "mhsa.layer_norm_v.weight" => ("attn.norm_v.weight", F32Vector),
        "mhsa.layer_norm_v.bias" => ("attn.norm_v.bias", F32Vector),
        "mhsa.w_qs.weight" => ("attn.q.weight", Linear),
        "mhsa.w_ks.weight" => ("attn.k.weight", Linear),
        "mhsa.w_vs.weight" => ("attn.v.weight", Linear),
        "mhsa.fc.weight" => ("attn.out.weight", Linear),
        "mhsa.linear_pos.weight" => ("attn.pos.weight", Linear),
        "mhsa.pos_bias_u" => ("attn.pos_bias_u", PosBiasFlatten),
        "mhsa.pos_bias_v" => ("attn.pos_bias_v", PosBiasFlatten),
        "conv.pre_layer_norm.weight" => ("conv.norm.weight", F32Vector),
        "conv.pre_layer_norm.bias" => ("conv.norm.bias", F32Vector),
        "conv.pointwise_conv1.weight" => ("conv.pw1.weight", PointwiseConvSqueeze),
        "conv.depthwise_conv.weight" => ("conv.dw.weight", ConvKernel),
        "conv.batch_norm.weight" => ("conv.ln.weight", F32Vector),
        "conv.batch_norm.bias" => ("conv.ln.bias", F32Vector),
        "conv.pointwise_conv2.weight" => ("conv.pw2.weight", PointwiseConvSqueeze),
        "layer_norm.weight" => ("out_norm.weight", F32Vector),
        "layer_norm.bias" => ("out_norm.bias", F32Vector),
        _ => return None,
    };
    Some((format!("enc.blk.{layer}.{suffix}"), class))
}

fn map_adapter_tensor_name(source_name: &str) -> Option<(String, TensorClass)> {
    use TensorClass::*;
    match source_name {
        "encoder_projector.linear1.weight" => Some((ADAPTER_LINEAR1_WEIGHT.to_string(), Linear)),
        "encoder_projector.linear1.bias" => Some((ADAPTER_LINEAR1_BIAS.to_string(), F32Vector)),
        "encoder_projector.linear2.weight" => Some((ADAPTER_LINEAR2_WEIGHT.to_string(), Linear)),
        "encoder_projector.linear2.bias" => Some((ADAPTER_LINEAR2_BIAS.to_string(), F32Vector)),
        _ => None,
    }
}

fn target_dims_for_class(
    source_shape: &[u64],
    class: TensorClass,
) -> Result<Vec<u64>, LocalSourceImportError> {
    match class {
        TensorClass::F32Vector => {
            if source_shape.len() != 1 {
                return Err(validate_error(format!(
                    "firered-llm f32-vector tensor must be rank 1, got {source_shape:?}"
                )));
            }
            Ok(source_shape.to_vec())
        }
        TensorClass::PosBiasFlatten => {
            if source_shape.len() != 2 {
                return Err(validate_error(format!(
                    "firered-llm rel-pos bias must be rank 2, got {source_shape:?}"
                )));
            }
            Ok(vec![source_shape[0] * source_shape[1]])
        }
        TensorClass::Linear => {
            if source_shape.len() != 2 {
                return Err(validate_error(format!(
                    "firered-llm linear weight must be rank 2, got {source_shape:?}"
                )));
            }
            Ok(vec![source_shape[1], source_shape[0]])
        }
        TensorClass::PointwiseConvSqueeze => match source_shape {
            [out, input, 1] => Ok(vec![*input, *out]),
            _ => Err(validate_error(format!(
                "firered-llm pointwise conv must be [out, in, 1], got {source_shape:?}"
            ))),
        },
        TensorClass::ConvKernel | TensorClass::F16Table => {
            if source_shape.len() < 2 {
                return Err(validate_error(format!(
                    "firered-llm conv/table tensor must be rank >= 2, got {source_shape:?}"
                )));
            }
            let mut dims = source_shape.to_vec();
            dims.reverse();
            Ok(dims)
        }
    }
}

fn quantized_tensor_type_for_encoder_adapter_tensor(
    class: TensorClass,
    dims: &[u64],
    quantization: FireRedLlmQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == FireRedLlmQuantizationMode::Fp16 {
        return None;
    }
    if !matches!(
        class,
        TensorClass::Linear | TensorClass::PointwiseConvSqueeze
    ) {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn build_encoder_adapter_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: FireRedLlmQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        // The LLM branch is sourced from the separate LoRA-merged Qwen2
        // safetensors (see `build_llm_runtime_tensors`); this file's own
        // `llm.*` tensors are the raw LoRA increments, which this importer
        // never touches directly (they were already consumed and folded by
        // `firered_llm_merge_lora.py` upstream of this Rust import step).
        if tensor.name.starts_with("llm.") {
            continue;
        }
        let mapped = map_firered_encoder_tensor_name(tensor.name.as_str())
            .or_else(|| map_adapter_tensor_name(tensor.name.as_str()));
        let Some((target_name, class)) = mapped else {
            return Err(validate_error(format!(
                "firered-llm source has an unrecognized encoder/adapter tensor '{}'",
                tensor.name
            )));
        };
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "firered-llm import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let target_dims = target_dims_for_class(tensor.shape.as_slice(), class)?;
        let data = safetensors.tensor_data(tensor)?;
        let write_tensor = match quantized_tensor_type_for_encoder_adapter_tensor(
            class,
            &target_dims,
            quantization,
        ) {
            Some(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "firered-llm quantization failed for '{}' -> '{target_name}' ({qtype:?}): {error}",
                            tensor.name
                        ))
                    })?;
                GgufWriteTensor {
                    name: target_name,
                    dims: target_dims,
                    tensor_type: qtype,
                    data: quantized,
                }
            }
            None => match class {
                TensorClass::F32Vector | TensorClass::PosBiasFlatten => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                    let mut bytes = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: GgufWriteTensorType::F32,
                        data: bytes,
                    }
                }
                _ => {
                    let bits =
                        decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: GgufWriteTensorType::F16,
                        data: encode_f16_bits_le(bits),
                    }
                }
            },
        };
        out.push(write_tensor);
    }
    if out.is_empty() {
        return Err(validate_error(
            "firered-llm import found no encoder/adapter tensors in the source",
        ));
    }
    Ok(out)
}

fn derive_and_validate_encoder_hparams(
    safetensors: &SafetensorsFile,
    cmvn_dim: usize,
) -> Result<FireRedLlmEncoderHparams, LocalSourceImportError> {
    let mut shape_by_name: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in &safetensors.header().tensors {
        shape_by_name.insert(tensor.name.as_str(), tensor.shape.as_slice());
    }
    let shape = |name: &str| -> Result<&[u64], LocalSourceImportError> {
        shape_by_name
            .get(name)
            .copied()
            .ok_or_else(|| validate_error(format!("firered-llm source is missing tensor '{name}'")))
    };

    let count_layers = |prefix: &str, probe: &str| -> usize {
        let mut count = 0usize;
        while shape_by_name.contains_key(format!("{prefix}.{count}.{probe}").as_str()) {
            count += 1;
        }
        count
    };
    let n_layers = count_layers("encoder.layer_stack", "mhsa.w_qs.weight");
    if n_layers == 0 {
        return Err(validate_error(
            "firered-llm source has no 'encoder.layer_stack.N.*' tensors",
        ));
    }

    let pos_bias = shape("encoder.layer_stack.0.mhsa.pos_bias_u")?;
    let (n_heads, head_dim) = match pos_bias {
        [heads, head_dim] => (*heads as usize, *head_dim as usize),
        _ => {
            return Err(validate_error(format!(
                "firered-llm 'encoder.layer_stack.0.mhsa.pos_bias_u' has shape {pos_bias:?}, expected rank 2"
            )));
        }
    };
    let w_qs = shape("encoder.layer_stack.0.mhsa.w_qs.weight")?;
    if w_qs.len() != 2 {
        return Err(validate_error(format!(
            "firered-llm encoder attention q projection must be rank 2, got {w_qs:?}"
        )));
    }
    let d_model = w_qs[1] as usize;
    if n_heads == 0 || n_heads * head_dim != d_model {
        return Err(validate_error(format!(
            "firered-llm heads {n_heads} * head_dim {head_dim} != d_model {d_model}"
        )));
    }

    let ffn = shape("encoder.layer_stack.0.ffn1.net.1.weight")?;
    if ffn.len() != 2 || ffn[1] as usize != d_model {
        return Err(validate_error(
            "firered-llm encoder FFN up-projection must be rank 2 with d_model input",
        ));
    }
    let ffn_dim = ffn[0] as usize;

    let dw = shape("encoder.layer_stack.0.conv.depthwise_conv.weight")?;
    let conv_kernel = match dw {
        [channels, one, kernel] if *channels as usize == 2 * d_model && *one == 1 => {
            *kernel as usize
        }
        _ => {
            return Err(validate_error(format!(
                "firered-llm depthwise conv has unexpected shape {dw:?} (want [{}, 1, k])",
                2 * d_model
            )));
        }
    };
    if conv_kernel % 2 != 1 {
        return Err(validate_error(format!(
            "firered-llm conv kernel {conv_kernel} must be odd (symmetric padding)"
        )));
    }

    let conv1 = shape("encoder.input_preprocessor.conv.0.weight")?;
    let subsample_channels = match conv1 {
        [channels, 1, 3, 3] => *channels as usize,
        _ => {
            return Err(validate_error(format!(
                "firered-llm subsampling conv1 has unexpected shape {conv1:?} (want [C, 1, 3, 3])"
            )));
        }
    };
    let feature_dim = cmvn_dim;
    let expected_subsample_out = subsample_channels * (((feature_dim - 1) / 2 - 1) / 2);
    let out = shape("encoder.input_preprocessor.out.weight")?;
    if out.len() != 2 || out[0] as usize != d_model {
        return Err(validate_error(format!(
            "firered-llm subsampling out projection has unexpected shape {out:?}"
        )));
    }
    let subsample_out_dim = out[1] as usize;
    if subsample_out_dim != expected_subsample_out {
        return Err(validate_error(format!(
            "firered-llm subsampling out dim {subsample_out_dim} != {subsample_channels} channels x \
             subsampled {feature_dim}-mel width ({expected_subsample_out})"
        )));
    }

    let enc_pe = shape("encoder.positional_encoding.pe")?;
    let pe_len = match enc_pe {
        [1, len, dm] if *dm as usize == d_model => *len as usize,
        _ => {
            return Err(validate_error(format!(
                "firered-llm encoder pe has unexpected shape {enc_pe:?} (want [1, len, {d_model}])"
            )));
        }
    };
    if pe_len % 2 != 1 {
        return Err(validate_error(format!(
            "firered-llm encoder rel-pos table length {pe_len} must be odd (2*max-1)"
        )));
    }

    Ok(FireRedLlmEncoderHparams {
        n_layers,
        d_model,
        n_heads,
        head_dim,
        ffn_dim,
        conv_kernel,
        subsample_channels,
        subsample_out_dim,
        feature_dim,
        pe_len,
    })
}

fn derive_and_validate_adapter_hparams(
    safetensors: &SafetensorsFile,
    encoder_d_model: usize,
) -> Result<FireRedLlmAdapterHparams, LocalSourceImportError> {
    let mut shape_by_name: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in &safetensors.header().tensors {
        shape_by_name.insert(tensor.name.as_str(), tensor.shape.as_slice());
    }
    let shape = |name: &str| -> Result<&[u64], LocalSourceImportError> {
        shape_by_name
            .get(name)
            .copied()
            .ok_or_else(|| validate_error(format!("firered-llm source is missing tensor '{name}'")))
    };
    let linear1 = shape("encoder_projector.linear1.weight")?;
    let linear2 = shape("encoder_projector.linear2.weight")?;
    if linear1.len() != 2 || linear2.len() != 2 {
        return Err(validate_error(
            "firered-llm adapter linear projections must be rank 2",
        ));
    }
    let (llm_dim, stacked_in) = (linear1[0] as usize, linear1[1] as usize);
    if linear2[0] as usize != llm_dim || linear2[1] as usize != llm_dim {
        return Err(validate_error(format!(
            "firered-llm adapter linear2 shape {linear2:?} != expected [{llm_dim}, {llm_dim}]"
        )));
    }
    if encoder_d_model == 0 || !stacked_in.is_multiple_of(encoder_d_model) {
        return Err(validate_error(format!(
            "firered-llm adapter linear1 input dim {stacked_in} is not a multiple of the \
             encoder d_model {encoder_d_model}"
        )));
    }
    let downsample_rate = stacked_in / encoder_d_model;
    if downsample_rate == 0 {
        return Err(validate_error(
            "firered-llm adapter downsample_rate resolved to 0",
        ));
    }
    Ok(FireRedLlmAdapterHparams {
        llm_dim,
        downsample_rate,
    })
}

// --- LLM (LoRA-merged Qwen2) branch ----------------------------------------

fn remap_qwen2_tensor_name(source_name: &str) -> Result<Option<String>, LocalSourceImportError> {
    let direct = match source_name {
        "model.embed_tokens.weight" => Some(LLM_TOKEN_EMBD_WEIGHT),
        "model.norm.weight" => Some(LLM_OUTPUT_NORM_WEIGHT),
        "lm_head.weight" => Some(LLM_OUTPUT_WEIGHT),
        _ => None,
    };
    if let Some(mapped) = direct {
        return Ok(Some(mapped.to_string()));
    }
    let Some(rest) = source_name.strip_prefix("model.layers.") else {
        return Ok(None);
    };
    let (layer_str, tail) = rest.split_once('.').ok_or_else(|| {
        validate_error(format!("invalid firered-llm qwen2 tensor suffix '{rest}'"))
    })?;
    let layer_idx: usize = layer_str.parse().map_err(|error| {
        validate_error(format!(
            "invalid numeric layer index in firered-llm qwen2 tensor '{source_name}': {error}"
        ))
    })?;
    let names = qwen2_llm_layer_tensor_names(layer_idx);
    let mapped = match tail {
        "input_layernorm.weight" => names.attn_norm_weight,
        "self_attn.q_proj.weight" => names.attn_q_weight,
        "self_attn.q_proj.bias" => names.attn_q_bias,
        "self_attn.k_proj.weight" => names.attn_k_weight,
        "self_attn.k_proj.bias" => names.attn_k_bias,
        "self_attn.v_proj.weight" => names.attn_v_weight,
        "self_attn.v_proj.bias" => names.attn_v_bias,
        // Qwen2's o_proj has no bias (unlike q/k/v) -- matches the upstream
        // `Qwen2Attention` module, which only sets `attention_bias` on
        // q/k/v_proj (o_proj is always `bias=False`).
        "self_attn.o_proj.weight" => names.attn_out_weight,
        "post_attention_layernorm.weight" => names.ffn_norm_weight,
        "mlp.gate_proj.weight" => names.ffn_gate_weight,
        "mlp.up_proj.weight" => names.ffn_up_weight,
        "mlp.down_proj.weight" => names.ffn_down_weight,
        _ => return Ok(None),
    };
    Ok(Some(mapped))
}

fn is_qwen2_f32_tensor(name: &str, rank: usize) -> bool {
    rank <= 1 || name.ends_with(".bias") || name.contains("norm")
}

fn should_reverse_qwen2_tensor_dims(source_name: &str, source_dims: &[u64]) -> bool {
    source_dims.len() >= 2 && source_name.ends_with(".weight")
}

fn quantized_tensor_type_for_qwen2(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: FireRedLlmQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == FireRedLlmQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn build_llm_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: FireRedLlmQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for tensor in &safetensors.header().tensors {
        let Some(mapped_name) = remap_qwen2_tensor_name(&tensor.name)? else {
            return Err(validate_error(format!(
                "firered-llm merged-Qwen2 source has an unrecognized tensor '{}'",
                tensor.name
            )));
        };
        if !seen.insert(mapped_name.clone()) {
            return Err(validate_error(format!(
                "firered-llm import mapped duplicate destination tensor '{mapped_name}'"
            )));
        }
        let target_dims = tensor.shape.clone();
        let data = safetensors.tensor_data(tensor)?;
        let reverse =
            should_reverse_qwen2_tensor_dims(tensor.name.as_str(), target_dims.as_slice());
        let effective_dims = if reverse {
            target_dims.iter().copied().rev().collect()
        } else {
            target_dims.clone()
        };
        let force_f32 = is_qwen2_f32_tensor(&mapped_name, target_dims.len());
        let qtype =
            quantized_tensor_type_for_qwen2(&mapped_name, &effective_dims, force_f32, quantization);
        let write_tensor = match qtype {
            Some(tensor_type) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let expected = tensor_element_count(&tensor.name, &effective_dims)?;
                if values.len() != expected {
                    return Err(validate_error(format!(
                        "firered-llm qwen2 tensor '{}' decoded {} values but expected {} for dims {:?}",
                        tensor.name,
                        values.len(),
                        expected,
                        effective_dims
                    )));
                }
                let quantized =
                    quantize_f32_to_ggml_tensor_data(tensor_type, &effective_dims, &values)
                        .map_err(|error| {
                            validate_error(format!(
                                "firered-llm quantization failed for '{}' -> '{mapped_name}' ({tensor_type:?}): {error}",
                                tensor.name
                            ))
                        })?;
                GgufWriteTensor {
                    name: mapped_name,
                    dims: effective_dims,
                    tensor_type,
                    data: quantized,
                }
            }
            None if force_f32 => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let expected = tensor_element_count(&tensor.name, &effective_dims)?;
                if values.len() != expected {
                    return Err(validate_error(format!(
                        "firered-llm qwen2 tensor '{}' decoded {} values but expected {} for dims {:?}",
                        tensor.name,
                        values.len(),
                        expected,
                        effective_dims
                    )));
                }
                f32_tensor(&mapped_name, effective_dims, &values)
            }
            None => {
                let bits =
                    decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                GgufWriteTensor {
                    name: mapped_name,
                    dims: effective_dims,
                    tensor_type: GgufWriteTensorType::F16,
                    data: encode_f16_bits_le(bits),
                }
            }
        };
        out.push(write_tensor);
    }
    if out.is_empty() {
        return Err(validate_error(
            "firered-llm import found no tensors in the merged-Qwen2 source",
        ));
    }
    Ok(out)
}

fn derive_and_validate_decoder_hparams(
    safetensors: &SafetensorsFile,
    config: &Qwen2ConfigJson,
    adapter_hparams: FireRedLlmAdapterHparams,
) -> Result<FireRedLlmDecoderHparams, LocalSourceImportError> {
    let mut shape_by_name: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in &safetensors.header().tensors {
        shape_by_name.insert(tensor.name.as_str(), tensor.shape.as_slice());
    }
    let shape = |name: &str| -> Result<&[u64], LocalSourceImportError> {
        shape_by_name.get(name).copied().ok_or_else(|| {
            validate_error(format!(
                "firered-llm merged-Qwen2 source is missing tensor '{name}'"
            ))
        })
    };
    let count_layers = |probe: &str| -> usize {
        let mut count = 0usize;
        while shape_by_name.contains_key(format!("model.layers.{count}.{probe}").as_str()) {
            count += 1;
        }
        count
    };
    let n_layers = count_layers("self_attn.q_proj.weight");
    if n_layers == 0 {
        return Err(validate_error(
            "firered-llm merged-Qwen2 source has no 'model.layers.N.*' tensors",
        ));
    }
    if let Some(configured) = config.num_hidden_layers
        && configured != n_layers
    {
        return Err(validate_error(format!(
            "firered-llm qwen2 config.json num_hidden_layers {configured} != tensor-derived layer count {n_layers}"
        )));
    }

    let embed = shape("model.embed_tokens.weight")?;
    if embed.len() != 2 {
        return Err(validate_error(
            "firered-llm model.embed_tokens.weight must be rank 2",
        ));
    }
    let (embed_vocab, d_model) = (embed[0] as usize, embed[1] as usize);
    if let Some(configured) = config.hidden_size
        && configured != d_model
    {
        return Err(validate_error(format!(
            "firered-llm qwen2 config.json hidden_size {configured} != embed_tokens d_model {d_model}"
        )));
    }
    if d_model != adapter_hparams.llm_dim {
        return Err(validate_error(format!(
            "firered-llm qwen2 d_model {d_model} != adapter llm_dim {}",
            adapter_hparams.llm_dim
        )));
    }

    let q_proj = shape("model.layers.0.self_attn.q_proj.weight")?;
    let k_proj = shape("model.layers.0.self_attn.k_proj.weight")?;
    if q_proj.len() != 2 || k_proj.len() != 2 {
        return Err(validate_error(
            "firered-llm qwen2 attention projections must be rank 2",
        ));
    }
    let n_heads = config.num_attention_heads.ok_or_else(|| {
        validate_error("firered-llm qwen2 config.json is missing num_attention_heads")
    })?;
    let n_kv_heads = config.num_key_value_heads.ok_or_else(|| {
        validate_error("firered-llm qwen2 config.json is missing num_key_value_heads")
    })?;
    if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
        return Err(validate_error(format!(
            "firered-llm qwen2 d_model {d_model} is not a multiple of num_attention_heads {n_heads}"
        )));
    }
    let head_dim = d_model / n_heads;
    if q_proj[0] as usize != n_heads * head_dim {
        return Err(validate_error(format!(
            "firered-llm qwen2 q_proj out-dim {} != num_attention_heads {n_heads} * head_dim {head_dim}",
            q_proj[0]
        )));
    }
    if k_proj[0] as usize != n_kv_heads * head_dim {
        return Err(validate_error(format!(
            "firered-llm qwen2 k_proj out-dim {} != num_key_value_heads {n_kv_heads} * head_dim {head_dim}",
            k_proj[0]
        )));
    }

    let gate_proj = shape("model.layers.0.mlp.gate_proj.weight")?;
    if gate_proj.len() != 2 || gate_proj[1] as usize != d_model {
        return Err(validate_error(
            "firered-llm qwen2 mlp.gate_proj must be rank 2 with d_model input",
        ));
    }
    let ffn_dim = gate_proj[0] as usize;
    if let Some(configured) = config.intermediate_size
        && configured != ffn_dim
    {
        return Err(validate_error(format!(
            "firered-llm qwen2 config.json intermediate_size {configured} != tensor-derived ffn_dim {ffn_dim}"
        )));
    }

    let lm_head = shape("lm_head.weight")?;
    if lm_head.len() != 2 || lm_head[1] as usize != d_model || lm_head[0] as usize != embed_vocab {
        return Err(validate_error(format!(
            "firered-llm qwen2 lm_head shape {lm_head:?} != expected [{embed_vocab}, {d_model}]"
        )));
    }
    if let Some(configured) = config.vocab_size
        && configured != embed_vocab
    {
        return Err(validate_error(format!(
            "firered-llm qwen2 config.json vocab_size {configured} != embedding row count {embed_vocab}"
        )));
    }

    let max_positions = config.max_position_embeddings.unwrap_or(32_768);

    Ok(FireRedLlmDecoderHparams {
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_dim,
        vocab_size: embed_vocab,
        max_positions,
    })
}

// --- Qwen2 tokenizer (vocab.json + merges.txt + added tokens) -------------

fn load_vocab_tokens(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let vocab: BTreeMap<String, usize> =
        read_source_json_file(source_root, SOURCE_QWEN2_VOCAB_JSON)?;
    if vocab.is_empty() {
        return Err(validate_error(
            "firered-llm qwen2 vocab.json cannot be empty",
        ));
    }
    let mut pairs = vocab.into_iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(_, token_id)| *token_id);
    let max_id = pairs.last().map(|(_, token_id)| *token_id).ok_or_else(|| {
        validate_error("firered-llm qwen2 vocab.json cannot determine max token id")
    })?;
    let mut tokens = vec![String::new(); max_id + 1];
    for (token, token_id) in pairs {
        tokens[token_id] = token;
    }
    Ok(tokens)
}

fn load_merges(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let path = source_root.join(SOURCE_QWEN2_MERGES_TXT);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_source_file_bytes(source_root, SOURCE_QWEN2_MERGES_TXT)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        validate_error(format!(
            "firered-llm qwen2 merges.txt is not valid UTF-8 ({}): {error}",
            path.display()
        ))
    })?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

fn patch_added_tokens(
    source_root: &Path,
    tokens: &mut Vec<String>,
) -> Result<(), LocalSourceImportError> {
    let path = source_root.join(SOURCE_QWEN2_TOKENIZER_CONFIG_JSON);
    if !path.exists() {
        return Ok(());
    }
    let cfg: TokenizerConfigJson =
        read_source_json_file(source_root, SOURCE_QWEN2_TOKENIZER_CONFIG_JSON)?;
    for (token_id_str, entry) in cfg.added_tokens_decoder {
        let token_id = token_id_str.parse::<usize>().map_err(|error| {
            validate_error(format!(
                "invalid firered-llm qwen2 tokenizer added token id '{}' in {}: {error}",
                token_id_str,
                path.display()
            ))
        })?;
        if token_id >= tokens.len() {
            tokens.resize_with(token_id + 1, String::new);
        }
        tokens[token_id] = entry.content;
    }
    Ok(())
}

// --- metadata ---------------------------------------------------------

fn firered_llm_runtime_gguf_metadata(
    encoder: &FireRedLlmEncoderHparams,
    adapter: &FireRedLlmAdapterHparams,
    decoder: &FireRedLlmDecoderHparams,
    request: &FireRedLlmImportRequest,
    tokens: &[String],
    merges: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let put_str = |metadata: &mut BTreeMap<String, GgufWriteValue>, key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str(
        &mut metadata,
        GENERAL_ARCHITECTURE_KEY,
        FIRERED_LLM_GGML_ARCHITECTURE_ID,
    );
    put_str(
        &mut metadata,
        OASR_METADATA_KEY_PACKAGE_VERSION,
        OASR_PACKAGE_VERSION_V1,
    );
    put_str(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_FAMILY,
        FIRERED_LLM_MODEL_FAMILY,
    );
    put_str(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        FIRERED_LLM_GGML_ARCHITECTURE_ID,
    );
    put_str(
        &mut metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        FIRERED_LLM_AUDIO_FRONTEND_ID,
    );
    put_str(
        &mut metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        FIRERED_LLM_DECODE_POLICY_ID,
    );
    put_str(
        &mut metadata,
        GGML_TOKENIZER_ID_KEY,
        FIRERED_LLM_TOKENIZER_ID,
    );
    put_str(&mut metadata, OPENASR_MODEL_ID_KEY, &request.model_id);

    let put_u32 = |metadata: &mut BTreeMap<String, GgufWriteValue>, key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    // Encoder: same key namespace `firered_aed::package_import` uses (the
    // architecture is identical -- see this module's doc comment), so a
    // future consumer can reuse `firered_aed::runtime_contract`'s parser
    // verbatim on this family's encoder branch.
    put_u32(
        &mut metadata,
        "firered.encoder.n_layers",
        encoder.n_layers as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.d_model",
        encoder.d_model as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.n_heads",
        encoder.n_heads as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.head_dim",
        encoder.head_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.ffn_dim",
        encoder.ffn_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.conv_kernel",
        encoder.conv_kernel as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.subsample_channels",
        encoder.subsample_channels as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.subsample_out_dim",
        encoder.subsample_out_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.feature_dim",
        encoder.feature_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered.encoder.pe_len",
        encoder.pe_len as u32,
    );
    put_u32(&mut metadata, "firered.audio.sample_rate", SAMPLE_RATE_HZ);
    put_u32(&mut metadata, "firered.audio.n_fft", FFT_SIZE as u32);
    put_u32(
        &mut metadata,
        "firered.audio.frame_length_ms",
        FRAME_LENGTH_MS,
    );
    put_u32(
        &mut metadata,
        "firered.audio.frame_shift_ms",
        FRAME_SHIFT_MS,
    );
    put_u32(
        &mut metadata,
        "firered.audio.n_mels",
        encoder.feature_dim as u32,
    );

    // Adapter (2x frame-stacking projector: net new to this family).
    put_u32(
        &mut metadata,
        "firered_llm.adapter.downsample_rate",
        adapter.downsample_rate as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.adapter.llm_dim",
        adapter.llm_dim as u32,
    );

    // LLM (LoRA-merged Qwen2-7B-Instruct decoder).
    put_u32(
        &mut metadata,
        "firered_llm.llm.n_layers",
        decoder.n_layers as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.d_model",
        decoder.d_model as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.n_heads",
        decoder.n_heads as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.n_kv_heads",
        decoder.n_kv_heads as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.head_dim",
        decoder.head_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.ffn_dim",
        decoder.ffn_dim as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.vocab_size",
        decoder.vocab_size as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.max_positions",
        decoder.max_positions as u32,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.chatml_im_start_token_id",
        CHATML_IM_START_TOKEN_ID,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.chatml_im_end_token_id",
        CHATML_IM_END_TOKEN_ID,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.endoftext_token_id",
        ENDOFTEXT_TOKEN_ID,
    );
    put_u32(
        &mut metadata,
        "firered_llm.llm.speech_token_id",
        SPEECH_TOKEN_ID,
    );

    put_str(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2,
    );
    metadata.insert(
        TOKENIZER_GGML_TOKENS_KEY.to_string(),
        GgufWriteValue::StringArray(tokens.to_vec()),
    );
    metadata.insert(
        TOKENIZER_GGML_MERGES_KEY.to_string(),
        GgufWriteValue::StringArray(merges.to_vec()),
    );

    put_str(
        &mut metadata,
        "openasr.source.name",
        "FireRedTeam/FireRedASR2-LLM",
    );
    put_str(&mut metadata, "openasr.license.name", "Apache-2.0");
    put_str(
        &mut metadata,
        "openasr.license.source",
        "https://huggingface.co/FireRedTeam/FireRedASR2-LLM",
    );

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_encoder_block_tensors_and_drops_llm_prefix() {
        assert_eq!(
            map_firered_encoder_tensor_name("encoder.layer_stack.0.mhsa.w_qs.weight"),
            Some(("enc.blk.0.attn.q.weight".to_string(), TensorClass::Linear))
        );
        assert_eq!(
            map_firered_encoder_tensor_name("encoder.layer_stack.15.ffn2.net.4.weight"),
            Some((
                "enc.blk.15.ffn2.down.weight".to_string(),
                TensorClass::Linear
            ))
        );
        assert_eq!(
            map_firered_encoder_tensor_name("encoder.positional_encoding.pe"),
            Some(("enc.pos_enc.pe".to_string(), TensorClass::F16Table))
        );
        // There is no decoder in this family -- a decoder-shaped name (were
        // one ever accidentally present) must NOT map.
        assert_eq!(
            map_firered_encoder_tensor_name("decoder.layer_stack.0.self_attn.w_ks.weight"),
            None
        );
        assert_eq!(map_firered_encoder_tensor_name("llm.base_model.foo"), None);
    }

    #[test]
    fn maps_adapter_tensors() {
        assert_eq!(
            map_adapter_tensor_name("encoder_projector.linear1.weight"),
            Some((ADAPTER_LINEAR1_WEIGHT.to_string(), TensorClass::Linear))
        );
        assert_eq!(
            map_adapter_tensor_name("encoder_projector.linear1.bias"),
            Some((ADAPTER_LINEAR1_BIAS.to_string(), TensorClass::F32Vector))
        );
        assert_eq!(
            map_adapter_tensor_name("encoder_projector.linear2.weight"),
            Some((ADAPTER_LINEAR2_WEIGHT.to_string(), TensorClass::Linear))
        );
        assert_eq!(
            map_adapter_tensor_name("encoder_projector.linear3.weight"),
            None
        );
    }

    #[test]
    fn remaps_qwen2_llm_tensors_with_bias_and_no_qk_norm() {
        assert_eq!(
            remap_qwen2_tensor_name("model.layers.0.self_attn.q_proj.weight").unwrap(),
            Some("llm.blk.0.attn_q.weight".to_string())
        );
        assert_eq!(
            remap_qwen2_tensor_name("model.layers.0.self_attn.q_proj.bias").unwrap(),
            Some("llm.blk.0.attn_q.bias".to_string())
        );
        assert_eq!(
            remap_qwen2_tensor_name("model.layers.3.self_attn.o_proj.weight").unwrap(),
            Some("llm.blk.3.attn_out.weight".to_string())
        );
        assert_eq!(
            remap_qwen2_tensor_name("model.embed_tokens.weight").unwrap(),
            Some(LLM_TOKEN_EMBD_WEIGHT.to_string())
        );
        assert_eq!(
            remap_qwen2_tensor_name("lm_head.weight").unwrap(),
            Some(LLM_OUTPUT_WEIGHT.to_string())
        );
        // Qwen2 has no QK-norm (unlike qwen3-asr) -- a q_norm/k_norm-shaped
        // name (were one ever present, e.g. from a future Qwen3 source fed
        // to this importer by mistake) must fail closed as unrecognized.
        assert_eq!(
            remap_qwen2_tensor_name("model.layers.0.self_attn.q_norm.weight").unwrap(),
            None
        );
    }

    #[test]
    fn target_dims_reverse_squeeze_and_flatten() {
        assert_eq!(
            target_dims_for_class(&[5120, 1280], TensorClass::Linear).unwrap(),
            vec![1280, 5120]
        );
        assert_eq!(
            target_dims_for_class(&[3584, 2560], TensorClass::Linear).unwrap(),
            vec![2560, 3584]
        );
        assert_eq!(
            target_dims_for_class(&[20, 64], TensorClass::PosBiasFlatten).unwrap(),
            vec![1280]
        );
    }

    #[test]
    fn quantizes_only_linear_encoder_adapter_classes_with_aligned_ne0() {
        assert_eq!(
            quantized_tensor_type_for_encoder_adapter_tensor(
                TensorClass::Linear,
                &[1280, 5120],
                FireRedLlmQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            quantized_tensor_type_for_encoder_adapter_tensor(
                TensorClass::ConvKernel,
                &[33, 1, 2560],
                FireRedLlmQuantizationMode::Q4_K
            ),
            None
        );
        assert_eq!(
            quantized_tensor_type_for_encoder_adapter_tensor(
                TensorClass::Linear,
                &[2560, 3584],
                FireRedLlmQuantizationMode::Fp16
            ),
            None
        );
    }

    #[test]
    fn quantizes_qwen2_weight_matrices_but_not_bias_or_norm() {
        assert_eq!(
            quantized_tensor_type_for_qwen2(
                "llm.blk.0.attn_q.weight",
                &[3584, 3584],
                false,
                FireRedLlmQuantizationMode::Q8_0
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            quantized_tensor_type_for_qwen2(
                "llm.blk.0.attn_q.bias",
                &[3584],
                true,
                FireRedLlmQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            quantized_tensor_type_for_qwen2(
                "llm.blk.0.attn_norm.weight",
                &[3584],
                true,
                FireRedLlmQuantizationMode::Q8_0
            ),
            None
        );
    }

    #[test]
    fn parses_kaldi_cmvn_stats_with_upstream_formula() {
        let dir =
            std::env::temp_dir().join(format!("firered-llm-cmvn-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cmvn.txt");
        std::fs::write(&path, " [\n  8.0 4.0 4.0 \n  32.0 8.0 0.0 ]\n").unwrap();
        let (neg_mean, inv_stddev) = parse_kaldi_cmvn_stats(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(neg_mean, vec![-2.0, -1.0]);
        assert_eq!(inv_stddev, vec![0.5, 1.0]);
    }

    #[test]
    fn metadata_declares_family_and_speech_token_contract() {
        let encoder = FireRedLlmEncoderHparams {
            n_layers: 16,
            d_model: 1280,
            n_heads: 20,
            head_dim: 64,
            ffn_dim: 5120,
            conv_kernel: 33,
            subsample_channels: 32,
            subsample_out_dim: 608,
            feature_dim: 80,
            pe_len: 9999,
        };
        let adapter = FireRedLlmAdapterHparams {
            llm_dim: 3584,
            downsample_rate: 2,
        };
        let decoder = FireRedLlmDecoderHparams {
            n_layers: 28,
            d_model: 3584,
            n_heads: 28,
            n_kv_heads: 4,
            head_dim: 128,
            ffn_dim: 18944,
            vocab_size: 152064,
            max_positions: 32768,
        };
        let request = FireRedLlmImportRequest {
            encoder_adapter_source_root: PathBuf::from("/tmp/firered-llm-src"),
            qwen2_merged_safetensors_path: PathBuf::from("/tmp/qwen2-merged.safetensors"),
            qwen2_metadata_source_root: PathBuf::from("/tmp/qwen2-meta"),
            output_root: PathBuf::from("/tmp/firered-llm.oasr"),
            model_id: "firered2-llm".to_string(),
            quantization: FireRedLlmQuantizationMode::Fp16,
        };
        let tokens: Vec<String> = (0..152064).map(|i| format!("t{i}")).collect();
        let metadata =
            firered_llm_runtime_gguf_metadata(&encoder, &adapter, &decoder, &request, &tokens, &[]);
        assert!(matches!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(GgufWriteValue::String(family)) if family == FIRERED_LLM_MODEL_FAMILY
        ));
        assert!(matches!(
            metadata.get("firered_llm.llm.speech_token_id"),
            Some(GgufWriteValue::U32(151_646))
        ));
        assert!(matches!(
            metadata.get("firered_llm.llm.n_kv_heads"),
            Some(GgufWriteValue::U32(4))
        ));
        assert!(matches!(
            metadata.get("firered.encoder.d_model"),
            Some(GgufWriteValue::U32(1280))
        ));
    }
}
