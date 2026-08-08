//! The mimo-asr runtime pack contract validated at install/preflight time:
//! the `mimo.*` execution metadata baked by `tooling/mimo-asr/convert_mimo_asr.py`
//! (see `GGUF_MANIFEST.md` for the authoritative key/value list), the full
//! runtime tensor set (every tensor the three graph stages, the speech
//! embedding path, the baked mel front-end, and the RVQ codebooks bind, with
//! shapes derived from the parsed geometry), and the baked gpt2 tokenizer
//! contract the executor needs to build prompts and decode text. A pack that
//! cannot run fails closed here, at the shared `PackVerifier` seam, rather
//! than deep inside a graph build.
//!
//! Unlike `firered_llm::runtime_contract` (which treats `rope_theta`/
//! `rms_norm_epsilon` as family constants never written to the pack), the
//! mimo-asr converter DOES bake every hparam -- including the three P2.0
//! "blood lesson" corrections (`mimo.tok.encoder.skip_layer_id`,
//! `mimo.tok.conv{1,2}.stride`) -- as real metadata, so this module reads them
//! from the pack rather than re-asserting them as constants.

use thiserror::Error;

use crate::models::oasr_metadata::{required_metadata_u32, required_metadata_u32_array};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};
use crate::{GgufMetadata, GgufTensorIndex};

use super::audio_tokenizer_graph::{MIMO_AUDIOTOK_DOWN_SAMPLE_KERNEL, MIMO_AUDIOTOK_N_MELS};
use super::tensor_names::{
    AUDIOTOK_CONV1_BIAS, AUDIOTOK_CONV1_WEIGHT, AUDIOTOK_CONV2_BIAS, AUDIOTOK_CONV2_WEIGHT,
    AUDIOTOK_DOWN_SAMPLE_NORM_BIAS, AUDIOTOK_DOWN_SAMPLE_NORM_WEIGHT, AUDIOTOK_DOWN_SAMPLE_WEIGHT,
    AUDIOTOK_MEL_FILTERS, AUDIOTOK_MEL_WINDOW, AUDIOTOK_NORM_BIAS, AUDIOTOK_NORM_WEIGHT,
    INLOCAL_NORM_WEIGHT, OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT, SPEECH_GROUP_PROJ_WEIGHT,
    TOKEN_EMBD_WEIGHT, audiotok_codebook_name, mimo_audiotok_layer_tensor_names,
    mimo_inlocal_layer_tensor_names, mimo_llm_layer_tensor_names, speech_embd_weight_name,
};
use super::tokenizer::MimoAsrTokenizer;

#[derive(Debug, Error)]
pub(crate) enum MimoMetadataError {
    #[error("mimo-asr GGUF metadata is missing required key '{key}'")]
    MissingKey { key: &'static str },
    #[error("mimo-asr GGUF metadata key '{key}' is invalid: {reason}")]
    InvalidValue { key: &'static str, reason: String },
}

fn required_u32(metadata: &GgufMetadata, key: &'static str) -> Result<u32, MimoMetadataError> {
    required_metadata_u32(metadata, key, "mimo-asr")
        .map_err(|_| MimoMetadataError::MissingKey { key })
}

fn required_usize(metadata: &GgufMetadata, key: &'static str) -> Result<usize, MimoMetadataError> {
    Ok(required_u32(metadata, key)? as usize)
}

fn required_f32(metadata: &GgufMetadata, key: &'static str) -> Result<f32, MimoMetadataError> {
    metadata
        .get_f32(key)
        .ok_or(MimoMetadataError::MissingKey { key })
}

fn required_bool(metadata: &GgufMetadata, key: &'static str) -> Result<bool, MimoMetadataError> {
    metadata
        .get_bool(key)
        .ok_or(MimoMetadataError::MissingKey { key })
}

fn positive(value: usize, key: &'static str) -> Result<usize, MimoMetadataError> {
    if value == 0 {
        return Err(MimoMetadataError::InvalidValue {
            key,
            reason: "value must be greater than 0".to_string(),
        });
    }
    Ok(value)
}

fn bounded(value: usize, key: &'static str, max: usize) -> Result<usize, MimoMetadataError> {
    if value > max {
        return Err(MimoMetadataError::InvalidValue {
            key,
            reason: format!("value {value} exceeds the architecture ceiling {max}"),
        });
    }
    Ok(value)
}

/// Local ceiling for RoPE position tables; generous over production 8192.
const MIMO_LLM_MAX_POSITIONS: usize = 1_048_576;

/// Architecture ceilings for non-decoder geometry (inlocal / audiotok / mel).
/// Production MiMo-ASR is 6L inlocal d1024, 32L audiotok, 8 codebooks, mel
/// n_fft=640; ceilings match FunASR/Qwen headroom so malicious metadata cannot
/// force unbounded descriptor loops before tensor validation.
const MIMO_MAX_LAYERS: usize = 512;
const MIMO_MAX_D_MODEL: usize = 65_536;
const MIMO_MAX_N_HEADS: usize = 1_024;
const MIMO_MAX_HEAD_DIM: usize = 1_024;
const MIMO_MAX_FFN_DIM: usize = 262_144;
const MIMO_MAX_GROUP_SIZE: usize = 64;
const MIMO_MAX_AUDIO_CHANNELS: usize = 64;
const MIMO_MAX_CODEBOOKS: usize = 64;
const MIMO_MAX_CODEBOOK_SIZE: u32 = 1_048_576;
const MIMO_MAX_CONV_KERNEL: usize = 4_096;
const MIMO_MAX_CONV_STRIDE: usize = 1_024;
const MIMO_MAX_SAMPLE_RATE_HZ: usize = 384_000;
const MIMO_MAX_N_FFT: usize = 65_536;
const MIMO_MAX_HOP_LENGTH: usize = 65_536;
const MIMO_MAX_WIN_LENGTH: usize = 65_536;
const MIMO_MAX_N_MELS: usize = 4_096;
/// Global ceiling on tensor obligations one pack contract may construct.
const MIMO_MAX_TENSOR_OBLIGATIONS: usize = 1_000_000;
/// Per-inlocal-layer descriptors (attn norm, q/k/v w+b, out, ffn norm, gate/up/down).
const MIMO_INLOCAL_TENSORS_PER_LAYER: usize = 12;
/// Fixed inlocal tail (final norm + group proj) before speech embd channels.
const MIMO_INLOCAL_FIXED_TAIL: usize = 2;
/// Audiotok stem + fixed tables excluding per-layer and codebooks.
const MIMO_AUDIOTOK_FIXED_TENSOR_COUNT: usize = 12;
/// Per-audiotok-layer descriptors.
const MIMO_AUDIOTOK_TENSORS_PER_LAYER: usize = 12;

/// The 36L Qwen2 backbone: qkv bias on, no QK-norm (the same shape
/// `firered_llm`'s LLM branch already parameterizes into
/// `qwen::llm_transformer`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MimoLlmMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
}

pub(crate) fn parse_mimo_llm_metadata(
    metadata: &GgufMetadata,
) -> Result<MimoLlmMetadata, MimoMetadataError> {
    let n_layers = positive(
        required_usize(metadata, "mimo.llm.block_count")?,
        "mimo.llm.block_count",
    )?;
    let d_model = positive(
        required_usize(metadata, "mimo.llm.embedding_length")?,
        "mimo.llm.embedding_length",
    )?;
    let n_heads = positive(
        required_usize(metadata, "mimo.llm.attention.head_count")?,
        "mimo.llm.attention.head_count",
    )?;
    let n_kv_heads = positive(
        required_usize(metadata, "mimo.llm.attention.head_count_kv")?,
        "mimo.llm.attention.head_count_kv",
    )?;
    let head_dim = positive(
        required_usize(metadata, "mimo.llm.attention.key_length")?,
        "mimo.llm.attention.key_length",
    )?;
    let ffn_dim = positive(
        required_usize(metadata, "mimo.llm.feed_forward_length")?,
        "mimo.llm.feed_forward_length",
    )?;
    let vocab_size = positive(
        required_usize(metadata, "mimo.llm.vocab_size")?,
        "mimo.llm.vocab_size",
    )?;
    let max_positions = positive(
        required_usize(metadata, "mimo.llm.context_length")?,
        "mimo.llm.context_length",
    )?;
    let rms_norm_epsilon = required_f32(metadata, "mimo.llm.attention.layer_norm_rms_epsilon")?;
    let rope_theta = required_f32(metadata, "mimo.llm.rope.freq_base")?;
    let qkv_bias = required_bool(metadata, "mimo.llm.attention.qkv_bias")?;
    let qk_norm = required_bool(metadata, "mimo.llm.attention.qk_norm")?;
    if !qkv_bias || qk_norm {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.llm.attention.qkv_bias",
            reason: format!(
                "mimo-asr backbone requires qkv_bias=true, qk_norm=false; got qkv_bias={qkv_bias} qk_norm={qk_norm}"
            ),
        });
    }
    use crate::models::qwen::{
        QWEN_DECODER_MAX_D_MODEL, QWEN_DECODER_MAX_FFN_DIM, QWEN_DECODER_MAX_HEAD_DIM,
        QWEN_DECODER_MAX_LAYERS, QWEN_DECODER_MAX_N_HEADS, QWEN_DECODER_MAX_VOCAB_SIZE,
    };
    bounded(n_layers, "mimo.llm.block_count", QWEN_DECODER_MAX_LAYERS)?;
    bounded(
        d_model,
        "mimo.llm.embedding_length",
        QWEN_DECODER_MAX_D_MODEL,
    )?;
    bounded(
        n_heads,
        "mimo.llm.attention.head_count",
        QWEN_DECODER_MAX_N_HEADS,
    )?;
    bounded(
        n_kv_heads,
        "mimo.llm.attention.head_count_kv",
        QWEN_DECODER_MAX_N_HEADS,
    )?;
    bounded(
        head_dim,
        "mimo.llm.attention.key_length",
        QWEN_DECODER_MAX_HEAD_DIM,
    )?;
    bounded(
        ffn_dim,
        "mimo.llm.feed_forward_length",
        QWEN_DECODER_MAX_FFN_DIM,
    )?;
    bounded(
        vocab_size,
        "mimo.llm.vocab_size",
        QWEN_DECODER_MAX_VOCAB_SIZE,
    )?;
    bounded(
        max_positions,
        "mimo.llm.context_length",
        MIMO_LLM_MAX_POSITIONS,
    )?;
    if n_heads.checked_mul(head_dim) != Some(d_model) {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.llm.attention.key_length",
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.llm.attention.head_count_kv",
            reason: format!("n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"),
        });
    }
    Ok(MimoLlmMetadata {
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_dim,
        vocab_size,
        max_positions,
        rms_norm_epsilon,
        rope_theta,
    })
}

/// The 6L input-local transformer (audio-embedding sum -> bidirectional
/// per-4-frame-group Qwen2-shaped mini-transformer -> group downcast).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MimoInlocalMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    pub group_size: usize,
    pub audio_channels: usize,
}

pub(crate) fn parse_mimo_inlocal_metadata(
    metadata: &GgufMetadata,
) -> Result<MimoInlocalMetadata, MimoMetadataError> {
    let n_layers = positive(
        required_usize(metadata, "mimo.inlocal.block_count")?,
        "mimo.inlocal.block_count",
    )?;
    let d_model = positive(
        required_usize(metadata, "mimo.inlocal.embedding_length")?,
        "mimo.inlocal.embedding_length",
    )?;
    let n_heads = positive(
        required_usize(metadata, "mimo.inlocal.attention.head_count")?,
        "mimo.inlocal.attention.head_count",
    )?;
    let head_dim = positive(
        required_usize(metadata, "mimo.inlocal.attention.head_dim")?,
        "mimo.inlocal.attention.head_dim",
    )?;
    let ffn_dim = positive(
        required_usize(metadata, "mimo.inlocal.feed_forward_length")?,
        "mimo.inlocal.feed_forward_length",
    )?;
    let rope_theta = required_f32(metadata, "mimo.inlocal.rope.freq_base")?;
    let full_attention = required_bool(metadata, "mimo.inlocal.full_attention")?;
    if !full_attention {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.inlocal.full_attention",
            reason: "mimo-asr input-local transformer requires full (non-causal) attention"
                .to_string(),
        });
    }
    let group_size = positive(
        required_usize(metadata, "mimo.audio.group_size")?,
        "mimo.audio.group_size",
    )?;
    let audio_channels = positive(
        required_usize(metadata, "mimo.audio.channels")?,
        "mimo.audio.channels",
    )?;
    bounded(n_layers, "mimo.inlocal.block_count", MIMO_MAX_LAYERS)?;
    bounded(d_model, "mimo.inlocal.embedding_length", MIMO_MAX_D_MODEL)?;
    bounded(
        n_heads,
        "mimo.inlocal.attention.head_count",
        MIMO_MAX_N_HEADS,
    )?;
    bounded(
        head_dim,
        "mimo.inlocal.attention.head_dim",
        MIMO_MAX_HEAD_DIM,
    )?;
    bounded(
        ffn_dim,
        "mimo.inlocal.feed_forward_length",
        MIMO_MAX_FFN_DIM,
    )?;
    bounded(group_size, "mimo.audio.group_size", MIMO_MAX_GROUP_SIZE)?;
    bounded(
        audio_channels,
        "mimo.audio.channels",
        MIMO_MAX_AUDIO_CHANNELS,
    )?;
    if n_heads.checked_mul(head_dim) != Some(d_model) {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.inlocal.attention.head_dim",
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    Ok(MimoInlocalMetadata {
        n_layers,
        d_model,
        n_heads,
        head_dim,
        ffn_dim,
        rope_theta,
        group_size,
        audio_channels,
    })
}

/// The 32L audio-tokenizer encoder (conv stem -> rope transformer, skip@L3
/// -> final LayerNorm -> down-sample conv -> RVQ encode over the first 8
/// packed codebooks).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MimoAudiotokMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub skip_layer_id: usize,
    pub conv_kernel_size: usize,
    pub conv1_stride: usize,
    pub conv2_stride: usize,
    pub down_sample_stride: usize,
    pub rope_theta: f32,
    pub rvq_packed: usize,
    pub codebook_sizes: Vec<u32>,
}

impl MimoAudiotokMetadata {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(
            &self.codebook_sizes,
            "mimo-asr audio-tokenizer codebook sizes",
        )?;
        Ok(bytes.finish())
    }
}

pub(crate) fn parse_mimo_audiotok_metadata(
    metadata: &GgufMetadata,
) -> Result<MimoAudiotokMetadata, MimoMetadataError> {
    let n_layers = positive(
        required_usize(metadata, "mimo.tok.block_count")?,
        "mimo.tok.block_count",
    )?;
    let d_model = positive(
        required_usize(metadata, "mimo.tok.embedding_length")?,
        "mimo.tok.embedding_length",
    )?;
    let n_heads = positive(
        required_usize(metadata, "mimo.tok.attention.head_count")?,
        "mimo.tok.attention.head_count",
    )?;
    bounded(n_layers, "mimo.tok.block_count", MIMO_MAX_LAYERS)?;
    bounded(d_model, "mimo.tok.embedding_length", MIMO_MAX_D_MODEL)?;
    bounded(n_heads, "mimo.tok.attention.head_count", MIMO_MAX_N_HEADS)?;
    if !d_model.is_multiple_of(n_heads) {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.tok.attention.head_count",
            reason: format!("d_model {d_model} is not a multiple of n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    bounded(head_dim, "mimo.tok.attention.head_dim", MIMO_MAX_HEAD_DIM)?;
    let ffn_dim = positive(
        required_usize(metadata, "mimo.tok.feed_forward_length")?,
        "mimo.tok.feed_forward_length",
    )?;
    bounded(ffn_dim, "mimo.tok.feed_forward_length", MIMO_MAX_FFN_DIM)?;
    let skip_layer_id = required_usize(metadata, "mimo.tok.encoder.skip_layer_id")?;
    if skip_layer_id == 0 || skip_layer_id > n_layers {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.tok.encoder.skip_layer_id",
            reason: format!("skip_layer_id {skip_layer_id} out of range for {n_layers} layers"),
        });
    }
    let conv_kernel_size = positive(
        required_usize(metadata, "mimo.tok.conv.kernel_size")?,
        "mimo.tok.conv.kernel_size",
    )?;
    let conv1_stride = positive(
        required_usize(metadata, "mimo.tok.conv1.stride")?,
        "mimo.tok.conv1.stride",
    )?;
    let conv2_stride = positive(
        required_usize(metadata, "mimo.tok.conv2.stride")?,
        "mimo.tok.conv2.stride",
    )?;
    let down_sample_stride = positive(
        required_usize(metadata, "mimo.tok.down_sample.stride")?,
        "mimo.tok.down_sample.stride",
    )?;
    bounded(
        conv_kernel_size,
        "mimo.tok.conv.kernel_size",
        MIMO_MAX_CONV_KERNEL,
    )?;
    bounded(conv1_stride, "mimo.tok.conv1.stride", MIMO_MAX_CONV_STRIDE)?;
    bounded(conv2_stride, "mimo.tok.conv2.stride", MIMO_MAX_CONV_STRIDE)?;
    bounded(
        down_sample_stride,
        "mimo.tok.down_sample.stride",
        MIMO_MAX_CONV_STRIDE,
    )?;
    let rope_theta = required_f32(metadata, "mimo.tok.rope.freq_base")?;
    let rvq_packed = positive(
        required_usize(metadata, "mimo.tok.rvq.num_quantizers_packed")?,
        "mimo.tok.rvq.num_quantizers_packed",
    )?;
    bounded(
        rvq_packed,
        "mimo.tok.rvq.num_quantizers_packed",
        MIMO_MAX_CODEBOOKS,
    )?;
    let codebook_sizes =
        required_metadata_u32_array(metadata, "mimo.tok.rvq.codebook_sizes", "mimo-asr")
            .map_err(|_| MimoMetadataError::MissingKey {
                key: "mimo.tok.rvq.codebook_sizes",
            })?
            .to_vec();
    if codebook_sizes.len() != rvq_packed {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.tok.rvq.codebook_sizes",
            reason: format!(
                "codebook_sizes has {} entries, expected rvq_packed={rvq_packed}",
                codebook_sizes.len()
            ),
        });
    }
    for (idx, &size) in codebook_sizes.iter().enumerate() {
        if size == 0 || size > MIMO_MAX_CODEBOOK_SIZE {
            return Err(MimoMetadataError::InvalidValue {
                key: "mimo.tok.rvq.codebook_sizes",
                reason: format!(
                    "codebook_sizes[{idx}]={size} must be in 1..={MIMO_MAX_CODEBOOK_SIZE}"
                ),
            });
        }
    }
    Ok(MimoAudiotokMetadata {
        n_layers,
        d_model,
        n_heads,
        head_dim,
        ffn_dim,
        skip_layer_id,
        conv_kernel_size,
        conv1_stride,
        conv2_stride,
        down_sample_stride,
        rope_theta,
        rvq_packed,
        codebook_sizes,
    })
}

/// The baked-filter mel front-end spec (torchaudio `MelSpectrogram`-shaped:
/// htk scale, `norm=None`, `power=1` magnitude, natural-log with a clip
/// floor, `center=True` reflect padding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MimoMelMetadata {
    pub sample_rate_hz: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub log_clip: f32,
}

pub(crate) fn parse_mimo_mel_metadata(
    metadata: &GgufMetadata,
) -> Result<MimoMelMetadata, MimoMetadataError> {
    let sample_rate_hz = positive(
        required_usize(metadata, "mimo.mel.sample_rate")?,
        "mimo.mel.sample_rate",
    )?;
    let n_fft = positive(
        required_usize(metadata, "mimo.mel.n_fft")?,
        "mimo.mel.n_fft",
    )?;
    let hop_length = positive(
        required_usize(metadata, "mimo.mel.hop_length")?,
        "mimo.mel.hop_length",
    )?;
    let win_length = positive(
        required_usize(metadata, "mimo.mel.win_length")?,
        "mimo.mel.win_length",
    )?;
    let n_mels = positive(
        required_usize(metadata, "mimo.mel.n_mels")?,
        "mimo.mel.n_mels",
    )?;
    bounded(
        sample_rate_hz,
        "mimo.mel.sample_rate",
        MIMO_MAX_SAMPLE_RATE_HZ,
    )?;
    bounded(n_fft, "mimo.mel.n_fft", MIMO_MAX_N_FFT)?;
    bounded(hop_length, "mimo.mel.hop_length", MIMO_MAX_HOP_LENGTH)?;
    bounded(win_length, "mimo.mel.win_length", MIMO_MAX_WIN_LENGTH)?;
    bounded(n_mels, "mimo.mel.n_mels", MIMO_MAX_N_MELS)?;
    if win_length > n_fft {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.mel.win_length",
            reason: format!("win_length {win_length} must be <= n_fft {n_fft}"),
        });
    }
    if hop_length > n_fft {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.mel.hop_length",
            reason: format!("hop_length {hop_length} must be <= n_fft {n_fft}"),
        });
    }
    Ok(MimoMelMetadata {
        sample_rate_hz,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        log_clip: required_f32(metadata, "mimo.mel.log_clip")?,
    })
}

/// ChatML/audio boundary special-token ids (see `GGUF_MANIFEST.md`'s
/// `mimo.special.*` keys, pinned by the P2.0 modeling-code audit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MimoSpecialTokens {
    pub eos_id: u32,
    pub im_start_id: u32,
    pub im_end_id: u32,
    pub sosp_id: u32,
    pub eosp_id: u32,
    pub empty_id: u32,
    pub eot_id: u32,
    pub eostm_id: u32,
}

pub(crate) fn parse_mimo_special_tokens(
    metadata: &GgufMetadata,
) -> Result<MimoSpecialTokens, MimoMetadataError> {
    Ok(MimoSpecialTokens {
        eos_id: required_u32(metadata, "mimo.special.eos_id")?,
        im_start_id: required_u32(metadata, "mimo.special.im_start_id")?,
        im_end_id: required_u32(metadata, "mimo.special.im_end_id")?,
        sosp_id: required_u32(metadata, "mimo.special.sosp_id")?,
        eosp_id: required_u32(metadata, "mimo.special.eosp_id")?,
        empty_id: required_u32(metadata, "mimo.special.empty_id")?,
        eot_id: required_u32(metadata, "mimo.special.eot_id")?,
        eostm_id: required_u32(metadata, "mimo.special.eostm_id")?,
    })
}

/// Runtime tensor contract failures for a pack whose metadata already parsed:
/// a tensor the graphs/tokenizer need is absent or its shape contradicts the
/// parsed geometry. Surfaces at install/preflight time through the family
/// runtime validator, before any graph tries to bind the tensor.
#[derive(Debug, Error)]
pub(crate) enum MimoRuntimeTensorError {
    #[error("mimo-asr pack is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("mimo-asr tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
    #[error("mimo-asr backbone decoder geometry rejected by shared Qwen contract: {reason}")]
    InvalidDecoderGeometry { reason: String },
    #[error("mimo-asr geometry constructs {count} tensor obligations, exceeding the ceiling {max}")]
    TooManyTensorObligations { count: usize, max: usize },
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata = preflight.metadata();
    let llm_metadata = parse_mimo_llm_metadata(metadata).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error)
    })?;
    let inlocal_metadata = parse_mimo_inlocal_metadata(metadata).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error)
    })?;
    let audiotok_metadata = parse_mimo_audiotok_metadata(metadata).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error)
    })?;
    let mel_metadata = parse_mimo_mel_metadata(metadata).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error)
    })?;
    // The audio-tokenizer graph bakes a fixed 128 mel-band input channel count
    // into the conv1 kernel (`MIMO_AUDIOTOK_N_MELS`); a pack declaring any
    // other `n_mels` can never run, so fail closed at admission rather than
    // mid-graph.
    if mel_metadata.n_mels != MIMO_AUDIOTOK_N_MELS {
        return Err(
            crate::models::runtime_pack_contract::metadata_validation_error(
                "mimo-asr",
                MimoMetadataError::InvalidValue {
                    key: "mimo.mel.n_mels",
                    reason: format!(
                        "n_mels {} is unsupported: the audio-tokenizer conv1 input is fixed at {MIMO_AUDIOTOK_N_MELS} mel bands",
                        mel_metadata.n_mels
                    ),
                },
            ),
        );
    }
    let special_tokens = parse_mimo_special_tokens(metadata).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error)
    })?;
    validate_speech_channel_consistency(metadata, &inlocal_metadata, &audiotok_metadata).map_err(
        |error| crate::models::runtime_pack_contract::metadata_validation_error("mimo-asr", error),
    )?;
    validate_mimo_asr_runtime_tensors_with_index(
        preflight.tensor_index(),
        &llm_metadata,
        &inlocal_metadata,
        &audiotok_metadata,
        &mel_metadata,
    )
    .map_err(crate::models::runtime_pack_contract::tensor_validation_error)?;
    // The tokenizer contract is part of the runtime pack contract: a pack the
    // executor cannot build a prompt/text decoder from fails closed here, at
    // the same seam whisper's validator applies its own tokenizer check.
    MimoAsrTokenizer::from_gguf_metadata(metadata, special_tokens)
        .map(|_| ())
        .map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                "mimo-asr tokenizer",
                error,
            )
        })
}

/// The 8-codebook summation path iterates exactly the packed RVQ channels:
/// `mimo.audio.channels` must equal the codebook count, and the informational
/// `mimo.speech.vocab_size`/`mimo.speech.zeroemb_idx` arrays (the executor
/// reconstructs both from `codebook_sizes` rather than trusting them) must
/// agree with the packed codebooks when they are present.
fn validate_speech_channel_consistency(
    metadata: &GgufMetadata,
    inlocal: &MimoInlocalMetadata,
    audiotok: &MimoAudiotokMetadata,
) -> Result<(), MimoMetadataError> {
    if inlocal.audio_channels != audiotok.rvq_packed {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.audio.channels",
            reason: format!(
                "audio channels {} must equal the packed RVQ codebook count {}",
                inlocal.audio_channels, audiotok.rvq_packed
            ),
        });
    }
    if let Some(vocab_sizes) = metadata.get_u32_array("mimo.speech.vocab_size") {
        let expected: Vec<u32> = audiotok
            .codebook_sizes
            .iter()
            .map(|size| size + 1)
            .collect();
        if vocab_sizes != expected.as_slice() {
            return Err(MimoMetadataError::InvalidValue {
                key: "mimo.speech.vocab_size",
                reason: format!("expected codebook sizes + 1 {expected:?}, got {vocab_sizes:?}"),
            });
        }
    }
    if let Some(zeroemb_idx) = metadata.get_u32_array("mimo.speech.zeroemb_idx")
        && zeroemb_idx != audiotok.codebook_sizes.as_slice()
    {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.speech.zeroemb_idx",
            reason: format!(
                "expected the packed codebook sizes {:?}, got {zeroemb_idx:?}",
                audiotok.codebook_sizes
            ),
        });
    }
    Ok(())
}

/// Every tensor the runtime binds, derived from the parsed metadata. Rank-2
/// matmul weights use ordered ggml `[in, out]` ExactDims (GGUF stores torch
/// shapes reversed). Vectors, conv kernels, and the mel tables must match
/// exactly, mirroring the loaders' own shape assertions.
/// Map mimo-asr backbone LLM metadata onto the shared Qwen-shaped geometry.
pub(crate) fn mimo_asr_qwen_decoder_geometry(
    llm: &MimoLlmMetadata,
) -> crate::models::qwen::QwenDecoderContractGeometry {
    crate::models::qwen::QwenDecoderContractGeometry {
        n_layers: llm.n_layers,
        d_model: llm.d_model,
        n_heads: llm.n_heads,
        n_kv_heads: llm.n_kv_heads,
        head_dim: llm.head_dim,
        ffn_dim: llm.ffn_dim,
        vocab_size: llm.vocab_size,
    }
}

/// Layer name provider for the Qwen2 backbone (`blk.{i}.*`).
pub(crate) fn mimo_asr_qwen_family_layer_names(
    layer: usize,
) -> crate::models::qwen::QwenFamilyLlmLayerTensorNames {
    let names = mimo_llm_layer_tensor_names(layer);
    crate::models::qwen::QwenFamilyLlmLayerTensorNames {
        attn_norm_name: names.attn_norm_weight,
        attn_q_name: names.attn_q_weight,
        attn_k_name: names.attn_k_weight,
        attn_v_name: names.attn_v_weight,
        attn_output_name: names.attn_output_weight,
        q_norm_name: None,
        k_norm_name: None,
        q_bias_name: Some(names.attn_q_bias),
        k_bias_name: Some(names.attn_k_bias),
        v_bias_name: Some(names.attn_v_bias),
        ffn_norm_name: names.ffn_norm_weight,
        ffn_gate_name: names.ffn_gate_weight,
        ffn_up_name: names.ffn_up_weight,
        ffn_down_name: names.ffn_down_weight,
    }
}

/// Adapter-local Qwen2 backbone profile for MiMo-ASR: closed variant, layer
/// names, and tail. It is immediately geometry-bound into the contract
/// consumed by admission, planning, tail load, host quote, and compilation.
pub(crate) fn mimo_asr_qwen_decoder_profile() -> crate::models::qwen::QwenFamilyDecoderProfile {
    crate::models::qwen::QwenFamilyDecoderProfile::new(
        crate::models::qwen::QwenDecoderVariant::Qwen2,
        mimo_asr_qwen_family_layer_names,
        mimo_asr_qwen_decoder_tail_names(),
    )
}

/// The Qwen2 backbone decoder half: every `blk.*` layer plus token embd /
/// logits / final norm. Expanded from the shared Qwen decoder contract Module
/// so the per-layer tensor set (base 9 + Qwen2 qkv-bias 3 = 12) cannot drift
/// from FunASR-Nano / MOSS / FireRed2-LLM. Does not cover inlocal / audiotok /
/// speech embd tensors.
pub(crate) fn mimo_asr_qwen_decoder_contract(
    llm: &MimoLlmMetadata,
) -> Result<crate::models::qwen::QwenDecoderContract, MimoRuntimeTensorError> {
    crate::models::qwen::QwenDecoderContract::bind(
        mimo_asr_qwen_decoder_geometry(llm),
        mimo_asr_qwen_decoder_profile(),
    )
    .map_err(|reason| MimoRuntimeTensorError::InvalidDecoderGeometry { reason })
}

/// Static tail tensor names shared by admission descriptors and the contract-
/// projected tail loader. Keep this the single spelling source for MiMo-ASR.
pub(crate) fn mimo_asr_qwen_decoder_tail_names()
-> crate::models::qwen::QwenDecoderTailTensorNames<'static> {
    crate::models::qwen::QwenDecoderTailTensorNames {
        output_norm: OUTPUT_NORM_WEIGHT,
        output_weight: Some(OUTPUT_WEIGHT),
        token_embd: TOKEN_EMBD_WEIGHT,
    }
}

fn mimo_asr_runtime_tensor_bindings(
    llm: &MimoLlmMetadata,
    inlocal: &MimoInlocalMetadata,
    audiotok: &MimoAudiotokMetadata,
    mel: &MimoMelMetadata,
) -> Result<Vec<TensorBindingDescriptor>, MimoRuntimeTensorError> {
    // Obligation budget before expanding per-layer descriptor loops.
    let inlocal_layers = inlocal
        .n_layers
        .checked_mul(MIMO_INLOCAL_TENSORS_PER_LAYER)
        .ok_or(MimoRuntimeTensorError::TooManyTensorObligations {
            count: usize::MAX,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        })?;
    let audiotok_layers = audiotok
        .n_layers
        .checked_mul(MIMO_AUDIOTOK_TENSORS_PER_LAYER)
        .ok_or(MimoRuntimeTensorError::TooManyTensorObligations {
            count: usize::MAX,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        })?;
    // speech embd one matrix per codebook + one RVQ codebook tensor each.
    let codebook_tensors = audiotok.rvq_packed.checked_mul(2).ok_or(
        MimoRuntimeTensorError::TooManyTensorObligations {
            count: usize::MAX,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        },
    )?;
    let non_decoder = inlocal_layers
        .checked_add(MIMO_INLOCAL_FIXED_TAIL)
        .and_then(|n| n.checked_add(audiotok_layers))
        .and_then(|n| n.checked_add(MIMO_AUDIOTOK_FIXED_TENSOR_COUNT))
        .and_then(|n| n.checked_add(codebook_tensors))
        // mel filters + window
        .and_then(|n| n.checked_add(2))
        .ok_or(MimoRuntimeTensorError::TooManyTensorObligations {
            count: usize::MAX,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        })?;
    let decoder_contract = mimo_asr_qwen_decoder_contract(llm)?;
    let decoder_upper = decoder_contract
        .tensor_obligation_count()
        .map_err(|reason| MimoRuntimeTensorError::InvalidDecoderGeometry { reason })?;
    let total_upper = non_decoder.checked_add(decoder_upper).ok_or(
        MimoRuntimeTensorError::TooManyTensorObligations {
            count: usize::MAX,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        },
    )?;
    if total_upper > MIMO_MAX_TENSOR_OBLIGATIONS {
        return Err(MimoRuntimeTensorError::TooManyTensorObligations {
            count: total_upper,
            max: MIMO_MAX_TENSOR_OBLIGATIONS,
        });
    }

    let mut bindings = Vec::new();

    let vector = |name: String, len: usize, reason: &str| TensorBindingDescriptor {
        tensor_name: name,
        requirement: TensorBindingDescriptorRequirement::VectorLen(len),
        reason: reason.to_string(),
    };
    let rank2 =
        |name: String, rows_in: usize, cols_out: usize, reason: &str| TensorBindingDescriptor {
            tensor_name: name,
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![rows_in, cols_out]),
            reason: reason.to_string(),
        };

    // 36L Qwen2 backbone (qkv bias, no QK-norm) via the shared decoder contract.
    bindings.extend(
        decoder_contract
            .runtime_tensor_descriptors()
            .map_err(|reason| MimoRuntimeTensorError::InvalidDecoderGeometry { reason })?,
    );

    // 6L input-local transformer + the speech embedding sum path.
    let d_in = inlocal.d_model;
    for layer_idx in 0..inlocal.n_layers {
        let names = mimo_inlocal_layer_tensor_names(layer_idx);
        bindings.push(vector(
            names.attn_norm_weight,
            d_in,
            "expected input-local d_model attention norm vector",
        ));
        bindings.push(rank2(
            names.attn_q_weight,
            d_in,
            d_in,
            "expected input-local query projection",
        ));
        bindings.push(vector(
            names.attn_q_bias,
            d_in,
            "expected input-local query bias",
        ));
        bindings.push(rank2(
            names.attn_k_weight,
            d_in,
            d_in,
            "expected input-local key projection",
        ));
        bindings.push(vector(
            names.attn_k_bias,
            d_in,
            "expected input-local key bias",
        ));
        bindings.push(rank2(
            names.attn_v_weight,
            d_in,
            d_in,
            "expected input-local value projection",
        ));
        bindings.push(vector(
            names.attn_v_bias,
            d_in,
            "expected input-local value bias",
        ));
        bindings.push(rank2(
            names.attn_output_weight,
            d_in,
            d_in,
            "expected input-local output projection",
        ));
        bindings.push(vector(
            names.ffn_norm_weight,
            d_in,
            "expected input-local ffn norm vector",
        ));
        bindings.push(rank2(
            names.ffn_gate_weight,
            d_in,
            inlocal.ffn_dim,
            "expected input-local gate projection",
        ));
        bindings.push(rank2(
            names.ffn_up_weight,
            d_in,
            inlocal.ffn_dim,
            "expected input-local up projection",
        ));
        bindings.push(rank2(
            names.ffn_down_weight,
            inlocal.ffn_dim,
            d_in,
            "expected input-local down projection",
        ));
    }
    bindings.push(vector(
        INLOCAL_NORM_WEIGHT.to_string(),
        d_in,
        "expected input-local final norm vector",
    ));
    bindings.push(rank2(
        SPEECH_GROUP_PROJ_WEIGHT.to_string(),
        inlocal.group_size * d_in,
        llm.d_model,
        "expected group-concat x llm d_model group projection",
    ));
    for (channel, &codebook_size) in audiotok.codebook_sizes.iter().enumerate() {
        bindings.push(rank2(
            speech_embd_weight_name(channel),
            d_in,
            codebook_size as usize + 1,
            "expected input-local d_model x (codebook size + 1) speech embedding table",
        ));
    }

    // 32L audio-tokenizer encoder: conv stem, rope transformer, RVQ codebooks,
    // and the baked mel front-end tables.
    let d_tok = audiotok.d_model;
    bindings.push(TensorBindingDescriptor {
        tensor_name: AUDIOTOK_CONV1_WEIGHT.to_string(),
        requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
            audiotok.conv_kernel_size,
            MIMO_AUDIOTOK_N_MELS,
            d_tok,
        ]),
        reason: "expected [kernel, n_mels, d_model] conv1 kernel".to_string(),
    });
    bindings.push(vector(
        AUDIOTOK_CONV1_BIAS.to_string(),
        d_tok,
        "expected d_model conv1 bias",
    ));
    bindings.push(TensorBindingDescriptor {
        tensor_name: AUDIOTOK_CONV2_WEIGHT.to_string(),
        requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
            audiotok.conv_kernel_size,
            d_tok,
            d_tok,
        ]),
        reason: "expected [kernel, d_model, d_model] conv2 kernel".to_string(),
    });
    bindings.push(vector(
        AUDIOTOK_CONV2_BIAS.to_string(),
        d_tok,
        "expected d_model conv2 bias",
    ));
    bindings.push(TensorBindingDescriptor {
        tensor_name: AUDIOTOK_DOWN_SAMPLE_WEIGHT.to_string(),
        requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
            MIMO_AUDIOTOK_DOWN_SAMPLE_KERNEL,
            d_tok,
            d_tok,
        ]),
        reason:
            "expected [2, d_model, d_model] down-sample kernel (the graph bakes a fixed kernel-2 downsample; only the stride is metadata-driven)"
                .to_string(),
    });
    bindings.push(vector(
        AUDIOTOK_DOWN_SAMPLE_NORM_WEIGHT.to_string(),
        d_tok,
        "expected d_model down-sample norm weight",
    ));
    bindings.push(vector(
        AUDIOTOK_DOWN_SAMPLE_NORM_BIAS.to_string(),
        d_tok,
        "expected d_model down-sample norm bias",
    ));
    bindings.push(vector(
        AUDIOTOK_NORM_WEIGHT.to_string(),
        d_tok,
        "expected d_model encoder final norm weight",
    ));
    bindings.push(vector(
        AUDIOTOK_NORM_BIAS.to_string(),
        d_tok,
        "expected d_model encoder final norm bias",
    ));
    bindings.push(TensorBindingDescriptor {
        tensor_name: AUDIOTOK_MEL_FILTERS.to_string(),
        requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
            mel.n_mels,
            mel.n_fft / 2 + 1,
        ]),
        reason: "expected [n_mels, n_fft/2+1] baked mel filterbank".to_string(),
    });
    bindings.push(vector(
        AUDIOTOK_MEL_WINDOW.to_string(),
        mel.win_length,
        "expected win_length baked mel window",
    ));
    for (level, &codebook_size) in audiotok.codebook_sizes.iter().enumerate() {
        bindings.push(rank2(
            audiotok_codebook_name(level),
            d_tok,
            codebook_size as usize,
            "expected d_model x codebook size RVQ codebook",
        ));
    }
    for layer_idx in 0..audiotok.n_layers {
        let names = mimo_audiotok_layer_tensor_names(layer_idx);
        bindings.push(vector(
            names.attn_norm_weight,
            d_tok,
            "expected audio-tokenizer attention norm weight",
        ));
        bindings.push(vector(
            names.attn_norm_bias,
            d_tok,
            "expected audio-tokenizer attention norm bias",
        ));
        bindings.push(rank2(
            names.attn_q_weight,
            d_tok,
            d_tok,
            "expected audio-tokenizer query projection",
        ));
        bindings.push(vector(
            names.attn_q_bias,
            d_tok,
            "expected audio-tokenizer query bias",
        ));
        bindings.push(rank2(
            names.attn_k_weight,
            d_tok,
            d_tok,
            "expected audio-tokenizer key projection",
        ));
        bindings.push(rank2(
            names.attn_v_weight,
            d_tok,
            d_tok,
            "expected audio-tokenizer value projection",
        ));
        bindings.push(vector(
            names.attn_v_bias,
            d_tok,
            "expected audio-tokenizer value bias",
        ));
        bindings.push(rank2(
            names.attn_out_weight,
            d_tok,
            d_tok,
            "expected audio-tokenizer output projection",
        ));
        bindings.push(vector(
            names.attn_out_bias,
            d_tok,
            "expected audio-tokenizer output bias",
        ));
        bindings.push(vector(
            names.ffn_norm_weight,
            d_tok,
            "expected audio-tokenizer ffn norm weight",
        ));
        bindings.push(vector(
            names.ffn_norm_bias,
            d_tok,
            "expected audio-tokenizer ffn norm bias",
        ));
        bindings.push(rank2(
            names.ffn_up_weight,
            d_tok,
            audiotok.ffn_dim,
            "expected audio-tokenizer ffn up projection",
        ));
        bindings.push(vector(
            names.ffn_up_bias,
            audiotok.ffn_dim,
            "expected audio-tokenizer ffn up bias",
        ));
        bindings.push(rank2(
            names.ffn_down_weight,
            audiotok.ffn_dim,
            d_tok,
            "expected audio-tokenizer ffn down projection",
        ));
        bindings.push(vector(
            names.ffn_down_bias,
            d_tok,
            "expected audio-tokenizer ffn down bias",
        ));
    }

    Ok(bindings)
}

pub(crate) fn validate_mimo_asr_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    llm: &MimoLlmMetadata,
    inlocal: &MimoInlocalMetadata,
    audiotok: &MimoAudiotokMetadata,
    mel: &MimoMelMetadata,
) -> Result<(), MimoRuntimeTensorError> {
    let bindings = mimo_asr_runtime_tensor_bindings(llm, inlocal, audiotok, mel)?;
    validate_tensor_binding_descriptors(
        index,
        &bindings,
        |name| MimoRuntimeTensorError::MissingRequiredTensor {
            name: name.to_string(),
        },
        |name, dims, reason| MimoRuntimeTensorError::InvalidTensorShape {
            name: name.to_string(),
            shape: render_shape(dims),
            reason,
        },
    )
}

#[cfg(any(test, feature = "testing"))]
fn tiny_metadata_values()
-> std::collections::BTreeMap<String, crate::ggml_runtime::GgufMetadataValue> {
    let mut values = std::collections::BTreeMap::new();
    let u = |values: &mut std::collections::BTreeMap<
        String,
        crate::ggml_runtime::GgufMetadataValue,
    >,
             k: &str,
             v: u32| {
        values.insert(
            k.to_string(),
            crate::ggml_runtime::GgufMetadataValue::U32(v),
        );
    };
    let f = |values: &mut std::collections::BTreeMap<
        String,
        crate::ggml_runtime::GgufMetadataValue,
    >,
             k: &str,
             v: f32| {
        values.insert(
            k.to_string(),
            crate::ggml_runtime::GgufMetadataValue::F32(v),
        );
    };
    let b = |values: &mut std::collections::BTreeMap<
        String,
        crate::ggml_runtime::GgufMetadataValue,
    >,
             k: &str,
             v: bool| {
        values.insert(
            k.to_string(),
            crate::ggml_runtime::GgufMetadataValue::Bool(v),
        );
    };
    u(&mut values, "mimo.llm.block_count", 1);
    u(&mut values, "mimo.llm.embedding_length", 16);
    u(&mut values, "mimo.llm.feed_forward_length", 32);
    u(&mut values, "mimo.llm.attention.head_count", 2);
    u(&mut values, "mimo.llm.attention.head_count_kv", 1);
    u(&mut values, "mimo.llm.attention.key_length", 8);
    f(
        &mut values,
        "mimo.llm.attention.layer_norm_rms_epsilon",
        1e-6,
    );
    f(&mut values, "mimo.llm.rope.freq_base", 640000.0);
    u(&mut values, "mimo.llm.vocab_size", 32);
    u(&mut values, "mimo.llm.context_length", 64);
    b(&mut values, "mimo.llm.attention.qkv_bias", true);
    b(&mut values, "mimo.llm.attention.qk_norm", false);

    u(&mut values, "mimo.audio.channels", 1);
    u(&mut values, "mimo.audio.group_size", 4);
    u(&mut values, "mimo.inlocal.block_count", 1);
    u(&mut values, "mimo.inlocal.embedding_length", 8);
    u(&mut values, "mimo.inlocal.attention.head_count", 2);
    u(&mut values, "mimo.inlocal.attention.head_dim", 4);
    u(&mut values, "mimo.inlocal.feed_forward_length", 16);
    b(&mut values, "mimo.inlocal.full_attention", true);
    f(&mut values, "mimo.inlocal.rope.freq_base", 640000.0);

    u(&mut values, "mimo.tok.block_count", 1);
    u(&mut values, "mimo.tok.embedding_length", 8);
    u(&mut values, "mimo.tok.attention.head_count", 2);
    u(&mut values, "mimo.tok.feed_forward_length", 16);
    u(&mut values, "mimo.tok.encoder.skip_layer_id", 1);
    u(&mut values, "mimo.tok.conv.kernel_size", 3);
    u(&mut values, "mimo.tok.conv1.stride", 1);
    u(&mut values, "mimo.tok.conv2.stride", 2);
    u(&mut values, "mimo.tok.down_sample.stride", 2);
    f(&mut values, "mimo.tok.rope.freq_base", 10000.0);
    u(&mut values, "mimo.tok.rvq.num_quantizers_packed", 1);
    values.insert(
        "mimo.tok.rvq.codebook_sizes".to_string(),
        crate::ggml_runtime::GgufMetadataValue::U32Array(vec![16]),
    );

    u(&mut values, "mimo.mel.sample_rate", 24000);
    u(&mut values, "mimo.mel.n_fft", 8);
    u(&mut values, "mimo.mel.hop_length", 2);
    u(&mut values, "mimo.mel.win_length", 8);
    // The audio-tokenizer conv1 input is fixed at 128 mel bands
    // (`MIMO_AUDIOTOK_N_MELS`), so even the tiny skeleton must declare it.
    u(&mut values, "mimo.mel.n_mels", 128);
    f(&mut values, "mimo.mel.log_clip", 1e-7);

    u(&mut values, "mimo.special.eos_id", 1);
    u(&mut values, "mimo.special.im_start_id", 2);
    u(&mut values, "mimo.special.im_end_id", 3);
    u(&mut values, "mimo.special.sosp_id", 4);
    u(&mut values, "mimo.special.eosp_id", 5);
    u(&mut values, "mimo.special.empty_id", 6);
    u(&mut values, "mimo.special.eot_id", 7);
    u(&mut values, "mimo.special.eostm_id", 8);

    values
}

#[cfg(any(test, feature = "testing"))]
fn tiny_tensors() -> Vec<(String, Vec<u64>)> {
    let mut tensors = Vec::new();
    // Backbone projected from the shared Qwen decoder descriptors so positive
    // fixtures cannot drift from admission (blk.* + token_embd/output/output_norm).
    // Construct geometry directly: `GgufMetadata::from_values_for_test` is
    // `cfg(test)`-only, while this helper also builds under `feature = "testing"`.
    let llm = MimoLlmMetadata {
        n_layers: 1,
        d_model: 16,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        ffn_dim: 32,
        vocab_size: 32,
        max_positions: 64,
        rms_norm_epsilon: 1e-6,
        rope_theta: 640_000.0,
    };
    let decoder_contract = mimo_asr_qwen_decoder_contract(&llm).expect("backbone decoder contract");
    tensors.extend(crate::models::tensor_binding::project_fixture_tensors(
        &decoder_contract
            .runtime_tensor_descriptors()
            .expect("backbone descriptors"),
    ));
    // Input-local (1 layer, d=8, ffn=16) + speech path (group 4 x 8 -> 16).
    tensors.extend([
        ("inlocal.blk.0.attn_norm.weight".to_string(), vec![8]),
        ("inlocal.blk.0.attn_q.weight".to_string(), vec![8, 8]),
        ("inlocal.blk.0.attn_q.bias".to_string(), vec![8]),
        ("inlocal.blk.0.attn_k.weight".to_string(), vec![8, 8]),
        ("inlocal.blk.0.attn_k.bias".to_string(), vec![8]),
        ("inlocal.blk.0.attn_v.weight".to_string(), vec![8, 8]),
        ("inlocal.blk.0.attn_v.bias".to_string(), vec![8]),
        ("inlocal.blk.0.attn_output.weight".to_string(), vec![8, 8]),
        ("inlocal.blk.0.ffn_norm.weight".to_string(), vec![8]),
        ("inlocal.blk.0.ffn_gate.weight".to_string(), vec![8, 16]),
        ("inlocal.blk.0.ffn_up.weight".to_string(), vec![8, 16]),
        ("inlocal.blk.0.ffn_down.weight".to_string(), vec![16, 8]),
        ("inlocal.norm.weight".to_string(), vec![8]),
        ("speech_group_proj.weight".to_string(), vec![32, 16]),
        ("speech_embd.0.weight".to_string(), vec![8, 17]),
    ]);
    // Audio-tokenizer encoder (1 layer, d=8, ffn=16) + mel + 1 codebook.
    tensors.extend([
        ("audiotok.conv1.weight".to_string(), vec![3, 128, 8]),
        ("audiotok.conv1.bias".to_string(), vec![8]),
        ("audiotok.conv2.weight".to_string(), vec![3, 8, 8]),
        ("audiotok.conv2.bias".to_string(), vec![8]),
        ("audiotok.down_sample.weight".to_string(), vec![2, 8, 8]),
        ("audiotok.down_sample_norm.weight".to_string(), vec![8]),
        ("audiotok.down_sample_norm.bias".to_string(), vec![8]),
        ("audiotok.norm.weight".to_string(), vec![8]),
        ("audiotok.norm.bias".to_string(), vec![8]),
        ("audiotok.mel_filters".to_string(), vec![128, 5]),
        ("audiotok.mel_window".to_string(), vec![8]),
        ("audiotok.quant.0.codebook".to_string(), vec![8, 16]),
        ("audiotok.blk.0.attn_norm.weight".to_string(), vec![8]),
        ("audiotok.blk.0.attn_norm.bias".to_string(), vec![8]),
        ("audiotok.blk.0.attn_q.weight".to_string(), vec![8, 8]),
        ("audiotok.blk.0.attn_q.bias".to_string(), vec![8]),
        ("audiotok.blk.0.attn_k.weight".to_string(), vec![8, 8]),
        ("audiotok.blk.0.attn_v.weight".to_string(), vec![8, 8]),
        ("audiotok.blk.0.attn_v.bias".to_string(), vec![8]),
        ("audiotok.blk.0.attn_out.weight".to_string(), vec![8, 8]),
        ("audiotok.blk.0.attn_out.bias".to_string(), vec![8]),
        ("audiotok.blk.0.ffn_norm.weight".to_string(), vec![8]),
        ("audiotok.blk.0.ffn_norm.bias".to_string(), vec![8]),
        ("audiotok.blk.0.ffn_up.weight".to_string(), vec![8, 16]),
        ("audiotok.blk.0.ffn_up.bias".to_string(), vec![16]),
        ("audiotok.blk.0.ffn_down.weight".to_string(), vec![16, 8]),
        ("audiotok.blk.0.ffn_down.bias".to_string(), vec![8]),
    ]);
    tensors
}

#[cfg(any(test, feature = "testing"))]
/// Runtime-ready `TinyGgufFixtureSpec` for the mimo-asr production
/// PackVerifier skeleton gate: routing keys, the full tiny hparam set
/// (native u32/f32/bool), a minimal gpt2 tokenizer, and the complete tiny
/// tensor skeleton.
pub(crate) fn mimo_asr_oasr_v1_runtime_ready() -> crate::testing::TinyGgufFixtureSpec {
    let mut spec = crate::testing::TinyGgufFixtureSpec::new(std::collections::BTreeMap::new())
        .with_metadata("openasr.package.version", "1")
        .with_metadata("openasr.model.family", "mimo-asr")
        .with_metadata("openasr.model.architecture", "mimo-asr")
        .with_metadata("openasr.model.id", "mimo-tiny:q8")
        .with_metadata("openasr.audio.frontend", "mimo-tokenizer-rvq-v0")
        .with_metadata("openasr.decode.policy", "mimo-asr.greedy.seq2seq.v0")
        .with_metadata("openasr.tokenizer.id", "mimo-asr.gpt2-bpe.v0")
        .with_metadata("openasr.pack.quant", "q8_0")
        .with_metadata("tokenizer.ggml.model", "gpt2")
        .with_string_array_metadata(
            "tokenizer.ggml.tokens",
            // Must equal `mimo.llm.vocab_size` (32) from `tiny_metadata_values`.
            (0..32).map(|index| format!("fixture{index}")),
        )
        .with_string_array_metadata("tokenizer.ggml.merges", ["f i", "fix t", "fixt u"]);
    for (key, value) in tiny_metadata_values() {
        spec = match value {
            crate::ggml_runtime::GgufMetadataValue::U32(value) => {
                spec.with_u32_metadata(key, value)
            }
            crate::ggml_runtime::GgufMetadataValue::F32(value) => {
                spec.with_f32_metadata(key, value)
            }
            crate::ggml_runtime::GgufMetadataValue::Bool(value) => {
                spec.with_bool_metadata(key, value)
            }
            crate::ggml_runtime::GgufMetadataValue::U32Array(values) => {
                spec.with_u32_array_metadata(key, values)
            }
            other => panic!("unexpected tiny metadata value for {key}: {other:?}"),
        };
    }
    // TinyGgufFixtureSpec always carries a placeholder tensor; drop it so
    // the skeleton is exactly the runtime set.
    let mut spec = spec.without_tensor("fixture.tensor");
    for (name, dims) in tiny_tensors() {
        spec = spec.with_tensor_shape(name, dims);
    }
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufMetadataValue;
    use std::collections::BTreeMap;

    fn full_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        let u = |values: &mut BTreeMap<String, GgufMetadataValue>, k: &str, v: u32| {
            values.insert(k.to_string(), GgufMetadataValue::U32(v));
        };
        let f = |values: &mut BTreeMap<String, GgufMetadataValue>, k: &str, v: f32| {
            values.insert(k.to_string(), GgufMetadataValue::F32(v));
        };
        let b = |values: &mut BTreeMap<String, GgufMetadataValue>, k: &str, v: bool| {
            values.insert(k.to_string(), GgufMetadataValue::Bool(v));
        };
        u(&mut values, "mimo.llm.block_count", 36);
        u(&mut values, "mimo.llm.embedding_length", 4096);
        u(&mut values, "mimo.llm.feed_forward_length", 11008);
        u(&mut values, "mimo.llm.attention.head_count", 32);
        u(&mut values, "mimo.llm.attention.head_count_kv", 8);
        u(&mut values, "mimo.llm.attention.key_length", 128);
        f(
            &mut values,
            "mimo.llm.attention.layer_norm_rms_epsilon",
            1e-6,
        );
        f(&mut values, "mimo.llm.rope.freq_base", 640000.0);
        u(&mut values, "mimo.llm.vocab_size", 151680);
        u(&mut values, "mimo.llm.context_length", 8192);
        b(&mut values, "mimo.llm.attention.qkv_bias", true);
        b(&mut values, "mimo.llm.attention.qk_norm", false);

        u(&mut values, "mimo.audio.channels", 8);
        u(&mut values, "mimo.audio.group_size", 4);
        u(&mut values, "mimo.inlocal.block_count", 6);
        u(&mut values, "mimo.inlocal.embedding_length", 1024);
        u(&mut values, "mimo.inlocal.attention.head_count", 64);
        u(&mut values, "mimo.inlocal.attention.head_dim", 16);
        u(&mut values, "mimo.inlocal.feed_forward_length", 4096);
        b(&mut values, "mimo.inlocal.full_attention", true);
        f(&mut values, "mimo.inlocal.rope.freq_base", 640000.0);

        u(&mut values, "mimo.tok.block_count", 32);
        u(&mut values, "mimo.tok.embedding_length", 1280);
        u(&mut values, "mimo.tok.attention.head_count", 20);
        u(&mut values, "mimo.tok.feed_forward_length", 5120);
        u(&mut values, "mimo.tok.encoder.skip_layer_id", 3);
        u(&mut values, "mimo.tok.conv.kernel_size", 3);
        u(&mut values, "mimo.tok.conv1.stride", 1);
        u(&mut values, "mimo.tok.conv2.stride", 2);
        u(&mut values, "mimo.tok.down_sample.stride", 2);
        f(&mut values, "mimo.tok.rope.freq_base", 10000.0);
        u(&mut values, "mimo.tok.rvq.num_quantizers_packed", 8);
        values.insert(
            "mimo.tok.rvq.codebook_sizes".to_string(),
            GgufMetadataValue::U32Array(vec![1024, 1024, 128, 128, 128, 128, 128, 128]),
        );

        u(&mut values, "mimo.mel.sample_rate", 24000);
        u(&mut values, "mimo.mel.n_fft", 960);
        u(&mut values, "mimo.mel.hop_length", 240);
        u(&mut values, "mimo.mel.win_length", 960);
        u(&mut values, "mimo.mel.n_mels", 128);
        f(&mut values, "mimo.mel.log_clip", 1e-7);

        u(&mut values, "mimo.special.eos_id", 151643);
        u(&mut values, "mimo.special.im_start_id", 151644);
        u(&mut values, "mimo.special.im_end_id", 151645);
        u(&mut values, "mimo.special.sosp_id", 151665);
        u(&mut values, "mimo.special.eosp_id", 151666);
        u(&mut values, "mimo.special.empty_id", 151667);
        u(&mut values, "mimo.special.eot_id", 151672);
        u(&mut values, "mimo.special.eostm_id", 151671);

        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn parses_llm_metadata() {
        let parsed = parse_mimo_llm_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 36);
        assert_eq!(parsed.n_kv_heads, 8);
        assert_eq!(parsed.rope_theta, 640000.0);
    }

    #[test]
    fn rejects_inlocal_block_count_above_architecture_ceiling() {
        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.inlocal.block_count".to_string(),
            GgufMetadataValue::U32((MIMO_MAX_LAYERS as u32).saturating_add(1)),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(parse_mimo_inlocal_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_audiotok_block_count_above_architecture_ceiling() {
        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.tok.block_count".to_string(),
            GgufMetadataValue::U32((MIMO_MAX_LAYERS as u32).saturating_add(1)),
        );
        // skip_layer_id must stay in range of the inflated layer count or the
        // ceiling check is shadowed by the skip-layer range gate.
        values.insert(
            "mimo.tok.encoder.skip_layer_id".to_string(),
            GgufMetadataValue::U32(1),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(parse_mimo_audiotok_metadata(&metadata).is_err());
    }

    /// Capacity regression anchor: the shared KV byte derivation on this
    /// family's real-checkpoint Qwen2 backbone geometry (36 layers, 8 KV
    /// heads, head_dim 128 -- the fixture values above), split by storage
    /// copy. Runs the derivation golden for every `Derived` family, not just
    /// the one that consumes an integral window today.
    #[test]
    fn kv_bytes_per_position_matches_the_reference_decoder_geometry() {
        use crate::capacity::{KvGeometry, kv_bytes_per_position};
        use crate::nn::decoder::LlmKvCacheSpec;

        let geometry = KvGeometry {
            n_layers: 36,
            kv_heads: 8,
            head_dim: 128,
        };
        // 36 layers * 2 (K+V) * 8 kv-heads = 576 rows per position.
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 576 * 512); // f32 rows
        assert_eq!(default.resident, 576 * 256); // f16 rows
        let q8_0 = kv_bytes_per_position(&geometry, LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 576 * 136); // 128 / 32 * 34 B q8_0 rows
        assert_eq!(q8_0.resident, 576 * 136);
    }

    #[test]
    fn parses_inlocal_metadata() {
        let parsed = parse_mimo_inlocal_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 6);
        assert_eq!(parsed.n_heads, 64);
        assert_eq!(parsed.head_dim, 16);
        assert_eq!(parsed.group_size, 4);
    }

    #[test]
    fn parses_audiotok_metadata_with_blood_lesson_hparams() {
        let parsed = parse_mimo_audiotok_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.skip_layer_id, 3);
        assert_eq!(parsed.conv1_stride, 1);
        assert_eq!(parsed.conv2_stride, 2);
        assert_eq!(parsed.head_dim, 64);
        assert_eq!(
            parsed.codebook_sizes,
            vec![1024, 1024, 128, 128, 128, 128, 128, 128]
        );
    }

    #[test]
    fn parses_mel_metadata() {
        let parsed = parse_mimo_mel_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.sample_rate_hz, 24000);
        assert_eq!(parsed.n_fft, 960);
    }

    #[test]
    fn parses_special_tokens() {
        let parsed = parse_mimo_special_tokens(&full_metadata()).expect("parse");
        assert_eq!(parsed.sosp_id, 151665);
        assert_eq!(parsed.empty_id, 151667);
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.llm.attention.head_count_kv".to_string(),
            GgufMetadataValue::U32(3),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(parse_mimo_llm_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_qk_norm_true() {
        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.llm.attention.qk_norm".to_string(),
            GgufMetadataValue::Bool(true),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(parse_mimo_llm_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_audio_channels_diverging_from_packed_codebooks() {
        let inlocal = parse_mimo_inlocal_metadata(&full_metadata()).expect("parse");
        let mut audiotok = parse_mimo_audiotok_metadata(&full_metadata()).expect("parse");
        audiotok.rvq_packed = 7;
        audiotok.codebook_sizes.pop();
        let error = validate_speech_channel_consistency(&full_metadata(), &inlocal, &audiotok)
            .expect_err("channel/codebook divergence must fail closed");
        assert!(matches!(
            error,
            MimoMetadataError::InvalidValue {
                key: "mimo.audio.channels",
                ..
            }
        ));
    }

    #[test]
    fn rejects_speech_vocab_arrays_diverging_from_codebooks() {
        let inlocal = parse_mimo_inlocal_metadata(&full_metadata()).expect("parse");
        let audiotok = parse_mimo_audiotok_metadata(&full_metadata()).expect("parse");

        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.speech.vocab_size".to_string(),
            GgufMetadataValue::U32Array(vec![3, 3, 3, 3, 3, 3, 3, 3]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(matches!(
            validate_speech_channel_consistency(&metadata, &inlocal, &audiotok),
            Err(MimoMetadataError::InvalidValue {
                key: "mimo.speech.vocab_size",
                ..
            })
        ));

        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.speech.zeroemb_idx".to_string(),
            GgufMetadataValue::U32Array(vec![0, 0, 0, 0, 0, 0, 0, 0]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(matches!(
            validate_speech_channel_consistency(&metadata, &inlocal, &audiotok),
            Err(MimoMetadataError::InvalidValue {
                key: "mimo.speech.zeroemb_idx",
                ..
            })
        ));

        // The consistent arrays (codebook sizes + 1 / codebook sizes) pass.
        let mut values = full_metadata().values().clone();
        values.insert(
            "mimo.speech.vocab_size".to_string(),
            GgufMetadataValue::U32Array(vec![1025, 1025, 129, 129, 129, 129, 129, 129]),
        );
        values.insert(
            "mimo.speech.zeroemb_idx".to_string(),
            GgufMetadataValue::U32Array(vec![1024, 1024, 128, 128, 128, 128, 128, 128]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        validate_speech_channel_consistency(&metadata, &inlocal, &audiotok)
            .expect("consistent speech arrays must pass");
    }

    // --- Runtime tensor contract -----------------------------------------

    /// A tiny internally consistent geometry (1 layer per stage) so tensor
    /// contract tests exercise every binding family without real dimensions.
    fn tiny_llm() -> MimoLlmMetadata {
        parse_mimo_llm_metadata(&tiny_metadata()).expect("tiny llm metadata")
    }

    fn tiny_inlocal() -> MimoInlocalMetadata {
        parse_mimo_inlocal_metadata(&tiny_metadata()).expect("tiny inlocal metadata")
    }

    fn tiny_audiotok() -> MimoAudiotokMetadata {
        parse_mimo_audiotok_metadata(&tiny_metadata()).expect("tiny audiotok metadata")
    }

    fn tiny_mel() -> MimoMelMetadata {
        parse_mimo_mel_metadata(&tiny_metadata()).expect("tiny mel metadata")
    }

    fn tiny_metadata() -> GgufMetadata {
        GgufMetadata::from_values_for_test(tiny_metadata_values())
    }

    fn tensor_index_from(names_and_dims: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let snapshot = crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("mimo-asr-tiny.oasr"),
            data_section_offset_bytes: 0,
            tensors: names_and_dims
                .iter()
                .map(|(name, dims)| crate::GgufTensorMetadata {
                    name: name.clone(),
                    dims: dims.clone(),
                    ggml_type: 0,
                    type_name: "f32".to_string(),
                    size_bytes: 0,
                    offset_bytes: 0,
                })
                .collect(),
        };
        crate::GgufTensorIndex::from_snapshot(snapshot).expect("unique tensor names")
    }

    fn validate_tiny_tensors(tensors: &[(String, Vec<u64>)]) -> Result<(), MimoRuntimeTensorError> {
        validate_mimo_asr_runtime_tensors_with_index(
            &tensor_index_from(tensors),
            &tiny_llm(),
            &tiny_inlocal(),
            &tiny_audiotok(),
            &tiny_mel(),
        )
    }

    #[test]
    fn tensor_contract_accepts_the_complete_tiny_skeleton() {
        validate_tiny_tensors(&tiny_tensors()).expect("complete skeleton must validate");
    }

    #[test]
    fn tensor_contract_covers_every_binding_the_runtime_loads() {
        // The binding list is the validator's only source of tensor truth; it
        // must enumerate at least every tensor of the complete skeleton (a
        // missing entry would let a truncated pack pass validation and fail
        // later inside a graph build).
        let bindings = mimo_asr_runtime_tensor_bindings(
            &tiny_llm(),
            &tiny_inlocal(),
            &tiny_audiotok(),
            &tiny_mel(),
        )
        .expect("tiny geometry must expand");
        let mut bound_names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for binding in &bindings {
            assert!(
                bound_names.insert(&binding.tensor_name),
                "duplicate binding for {}",
                binding.tensor_name
            );
        }
        for (name, _) in tiny_tensors() {
            assert!(
                bound_names.contains(name.as_str()),
                "runtime tensor '{name}' has no contract binding"
            );
        }
    }

    #[test]
    fn tensor_contract_rejects_a_missing_tensor() {
        let tensors = tiny_tensors()
            .into_iter()
            .filter(|(name, _)| name != "audiotok.mel_filters")
            .collect::<Vec<_>>();
        let error = validate_tiny_tensors(&tensors).expect_err("missing tensor must fail");
        assert!(matches!(
            error,
            MimoRuntimeTensorError::MissingRequiredTensor { ref name }
                if name == "audiotok.mel_filters"
        ));
    }

    #[test]
    fn tensor_contract_rejects_a_shape_mismatch() {
        let tensors = tiny_tensors()
            .into_iter()
            .map(|(name, dims)| {
                if name == "blk.0.attn_k.bias" {
                    (name, vec![7])
                } else {
                    (name, dims)
                }
            })
            .collect::<Vec<_>>();
        let error = validate_tiny_tensors(&tensors).expect_err("bad shape must fail");
        assert!(matches!(
            error,
            MimoRuntimeTensorError::InvalidTensorShape { ref name, .. }
                if name == "blk.0.attn_k.bias"
        ));
    }

    #[test]
    fn rejects_transposed_inlocal_and_audiotok_projections() {
        for (tensor_name, transposed) in [
            ("inlocal.blk.0.ffn_gate.weight", vec![16_u64, 8]),
            ("inlocal.blk.0.ffn_down.weight", vec![8_u64, 16]),
            ("speech_group_proj.weight", vec![16_u64, 32]),
            ("speech_embd.0.weight", vec![17_u64, 8]),
            ("audiotok.blk.0.ffn_up.weight", vec![16_u64, 8]),
            ("audiotok.quant.0.codebook", vec![16_u64, 8]),
        ] {
            let tensors = tiny_tensors()
                .into_iter()
                .map(|(name, dims)| {
                    if name == tensor_name {
                        (name, transposed.clone())
                    } else {
                        (name, dims)
                    }
                })
                .collect::<Vec<_>>();
            let error =
                validate_tiny_tensors(&tensors).expect_err("transposed weight must fail closed");
            assert!(
                matches!(
                    error,
                    MimoRuntimeTensorError::InvalidTensorShape { ref name, .. }
                        if name == tensor_name
                ),
                "unexpected error for {tensor_name}: {error}"
            );
        }
    }

    // --- End-to-end: a tiny external-shaped pack through PackVerifier ----

    fn write_tiny_pack(spec: &crate::testing::TinyGgufFixtureSpec) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mimo-tiny.oasr");
        crate::testing::write_tiny_gguf_runtime_source(&path, spec).expect("write tiny pack");
        dir
    }

    #[test]
    fn tiny_pack_passes_the_production_verifier_end_to_end() {
        let spec = mimo_asr_oasr_v1_runtime_ready();
        let dir = write_tiny_pack(&spec);
        let verified = crate::models::pack_verifier::PackVerifier
            .verify_candidate(crate::models::pack_verifier::PackCandidate::new(
                dir.path().join("mimo-tiny.oasr"),
            ))
            .expect("a contract-complete tiny mimo pack must verify");
        assert!(verified.proves_asr_family(
            crate::arch::MIMO_ASR_MODEL_FAMILY,
            crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID
        ));
        assert_eq!(verified.catalog_family_id(), Some("mimo-asr"));
    }

    #[test]
    fn tiny_pack_missing_a_runtime_tensor_fails_the_verifier_closed() {
        let mut spec = mimo_asr_oasr_v1_runtime_ready();
        spec = spec.without_tensor("audiotok.quant.0.codebook");
        let dir = write_tiny_pack(&spec);
        let error = crate::models::pack_verifier::PackVerifier
            .verify_candidate(crate::models::pack_verifier::PackCandidate::new(
                dir.path().join("mimo-tiny.oasr"),
            ))
            .expect_err("a truncated pack must fail closed at verification");
        let message = error.to_string();
        assert!(
            message.contains("audiotok.quant.0.codebook"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn tiny_pack_without_a_tokenizer_fails_the_verifier_closed() {
        // Rebuild the tiny pack without the tokenizer arrays: the executor
        // cannot build a prompt or decode text without them, so verification
        // must reject the pack instead of admitting a runtime failure.
        let mut spec = crate::testing::TinyGgufFixtureSpec::new(BTreeMap::new())
            .with_metadata("openasr.package.version", "1")
            .with_metadata("openasr.model.family", "mimo-asr")
            .with_metadata("openasr.model.architecture", "mimo-asr")
            .with_metadata("openasr.audio.frontend", "mimo-tokenizer-rvq-v0")
            .with_metadata("openasr.decode.policy", "mimo-asr.greedy.seq2seq.v0")
            .with_metadata("tokenizer.ggml.model", "gpt2");
        for (key, value) in tiny_metadata_values() {
            spec = match value {
                GgufMetadataValue::U32(value) => spec.with_u32_metadata(key, value),
                GgufMetadataValue::F32(value) => spec.with_f32_metadata(key, value),
                GgufMetadataValue::Bool(value) => spec.with_bool_metadata(key, value),
                GgufMetadataValue::U32Array(values) => spec.with_u32_array_metadata(key, values),
                other => panic!("unexpected tiny metadata value: {other:?}"),
            };
        }
        let mut spec = spec.without_tensor("fixture.tensor");
        for (name, dims) in tiny_tensors() {
            spec = spec.with_tensor_shape(name, dims);
        }
        let dir = write_tiny_pack(&spec);
        let error = crate::models::pack_verifier::PackVerifier
            .verify_candidate(crate::models::pack_verifier::PackCandidate::new(
                dir.path().join("mimo-tiny.oasr"),
            ))
            .expect_err("a pack without tokenizer arrays must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("tokenizer.ggml.tokens"),
            "unexpected error: {message}"
        );
    }
}
