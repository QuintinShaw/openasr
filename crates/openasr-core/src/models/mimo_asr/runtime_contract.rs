//! mimo-asr execution metadata parsed from the `.oasr` GGUF header
//! (`mimo.*` keys baked by `tooling/mimo-asr/convert_mimo_asr.py`, see
//! `GGUF_MANIFEST.md` for the authoritative key/value list).
//!
//! Unlike `firered_llm::runtime_contract` (which treats `rope_theta`/
//! `rms_norm_epsilon` as family constants never written to the pack), the
//! mimo-asr converter DOES bake every hparam -- including the three P2.0
//! "blood lesson" corrections (`mimo.tok.encoder.skip_layer_id`,
//! `mimo.tok.conv{1,2}.stride`) -- as real metadata, so this module reads them
//! from the pack rather than re-asserting them as constants.

use thiserror::Error;

use crate::GgufMetadata;
use crate::models::oasr_metadata::{required_metadata_u32, required_metadata_u32_array};

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
    if n_heads * head_dim != d_model {
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
    if n_heads * head_dim != d_model {
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
    if !d_model.is_multiple_of(n_heads) {
        return Err(MimoMetadataError::InvalidValue {
            key: "mimo.tok.attention.head_count",
            reason: format!("d_model {d_model} is not a multiple of n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    let ffn_dim = positive(
        required_usize(metadata, "mimo.tok.feed_forward_length")?,
        "mimo.tok.feed_forward_length",
    )?;
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
    let rope_theta = required_f32(metadata, "mimo.tok.rope.freq_base")?;
    let rvq_packed = positive(
        required_usize(metadata, "mimo.tok.rvq.num_quantizers_packed")?,
        "mimo.tok.rvq.num_quantizers_packed",
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
    Ok(MimoMelMetadata {
        sample_rate_hz: positive(
            required_usize(metadata, "mimo.mel.sample_rate")?,
            "mimo.mel.sample_rate",
        )?,
        n_fft: positive(
            required_usize(metadata, "mimo.mel.n_fft")?,
            "mimo.mel.n_fft",
        )?,
        hop_length: positive(
            required_usize(metadata, "mimo.mel.hop_length")?,
            "mimo.mel.hop_length",
        )?,
        win_length: positive(
            required_usize(metadata, "mimo.mel.win_length")?,
            "mimo.mel.win_length",
        )?,
        n_mels: positive(
            required_usize(metadata, "mimo.mel.n_mels")?,
            "mimo.mel.n_mels",
        )?,
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
}
