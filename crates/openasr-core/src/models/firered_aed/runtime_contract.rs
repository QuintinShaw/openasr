//! firered-aed execution metadata + runtime pack contract parsed from the
//! `.oasr` GGUF header. The validator is depth-complete: a pack must satisfy
//! the metadata contract AND the frontend-audio contract AND the full runtime
//! tensor binding (every tensor the executor loads, with the shapes the
//! Conformer encoder / Transformer decoder graphs consume) AND the tokenizer
//! admission gate before it can be admitted, so a malformed pack fails closed
//! at verification instead of deep inside the weight loader or mid-decode.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::ggml_runtime::GgufMetadata;
use crate::models::oasr_metadata::TOKENIZER_GGML_TOKENS_KEY;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

use super::frontend::{FFT_SIZE, FRAME_LENGTH_MS, FRAME_SHIFT_MS, SAMPLE_RATE_HZ};
use super::tokenizer::FireRedTokenizer;

pub(crate) const FIRERED_ENCODER_N_LAYERS_KEY: &str = "firered.encoder.n_layers";
pub(crate) const FIRERED_ENCODER_D_MODEL_KEY: &str = "firered.encoder.d_model";
pub(crate) const FIRERED_ENCODER_N_HEADS_KEY: &str = "firered.encoder.n_heads";
pub(crate) const FIRERED_ENCODER_HEAD_DIM_KEY: &str = "firered.encoder.head_dim";
pub(crate) const FIRERED_ENCODER_FFN_DIM_KEY: &str = "firered.encoder.ffn_dim";
pub(crate) const FIRERED_ENCODER_CONV_KERNEL_KEY: &str = "firered.encoder.conv_kernel";
pub(crate) const FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY: &str =
    "firered.encoder.subsample_channels";
pub(crate) const FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY: &str = "firered.encoder.subsample_out_dim";
pub(crate) const FIRERED_ENCODER_FEATURE_DIM_KEY: &str = "firered.encoder.feature_dim";
pub(crate) const FIRERED_ENCODER_PE_LEN_KEY: &str = "firered.encoder.pe_len";
pub(crate) const FIRERED_DECODER_N_LAYERS_KEY: &str = "firered.decoder.n_layers";
pub(crate) const FIRERED_DECODER_FFN_DIM_KEY: &str = "firered.decoder.ffn_dim";
pub(crate) const FIRERED_DECODER_PE_LEN_KEY: &str = "firered.decoder.pe_len";
pub(crate) const FIRERED_VOCAB_SIZE_KEY: &str = "firered.vocab_size";
pub(crate) const FIRERED_SOS_TOKEN_ID_KEY: &str = "firered.sos_token_id";
pub(crate) const FIRERED_EOS_TOKEN_ID_KEY: &str = "firered.eos_token_id";
pub(crate) const FIRERED_PAD_TOKEN_ID_KEY: &str = "firered.pad_token_id";
pub(crate) const FIRERED_AUDIO_SAMPLE_RATE_KEY: &str = "firered.audio.sample_rate";
pub(crate) const FIRERED_AUDIO_N_FFT_KEY: &str = "firered.audio.n_fft";
pub(crate) const FIRERED_AUDIO_FRAME_LENGTH_MS_KEY: &str = "firered.audio.frame_length_ms";
pub(crate) const FIRERED_AUDIO_FRAME_SHIFT_MS_KEY: &str = "firered.audio.frame_shift_ms";
pub(crate) const FIRERED_AUDIO_N_MELS_KEY: &str = "firered.audio.n_mels";

/// FFT bin count of the packed provenance filterbank (`firered.mel_filters`):
/// `n_fft / 2 + 1` for the family-fixed 512-point FFT.
pub(crate) const FIRERED_MEL_FILTER_BANK_BINS: usize = FFT_SIZE / 2 + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FireRedAedExecutionMetadata {
    pub encoder_n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub encoder_ffn_dim: usize,
    pub conv_kernel: usize,
    pub subsample_channels: usize,
    pub subsample_out_dim: usize,
    pub feature_dim: usize,
    /// Relative-position table rows (`2 * max_frames - 1`, odd).
    pub encoder_pe_len: usize,
    pub decoder_n_layers: usize,
    pub decoder_ffn_dim: usize,
    /// Absolute sinusoidal position rows == decoder max context.
    pub decoder_pe_len: usize,
    pub vocab_size: usize,
    pub sos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl FireRedAedExecutionMetadata {
    /// Maximum encoder frame count the baked rel-pos table supports.
    pub(crate) fn encoder_max_frames(&self) -> usize {
        self.encoder_pe_len.div_ceil(2)
    }
}

/// Fail-closed tensor-contract errors, surfaced by the pack verifier before a
/// firered-aed pack can be admitted.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FireRedAedRuntimeContractError {
    #[error("firered-aed runtime tensor contract is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("firered-aed runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

fn missing_required_tensor(name: &str) -> FireRedAedRuntimeContractError {
    FireRedAedRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> FireRedAedRuntimeContractError {
    FireRedAedRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

pub(crate) fn parse_firered_aed_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedAedExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let encoder_n_layers = usize_key(FIRERED_ENCODER_N_LAYERS_KEY)?;
    let d_model = usize_key(FIRERED_ENCODER_D_MODEL_KEY)?;
    let n_heads = usize_key(FIRERED_ENCODER_N_HEADS_KEY)?;
    let head_dim = usize_key(FIRERED_ENCODER_HEAD_DIM_KEY)?;
    let encoder_ffn_dim = usize_key(FIRERED_ENCODER_FFN_DIM_KEY)?;
    let conv_kernel = usize_key(FIRERED_ENCODER_CONV_KERNEL_KEY)?;
    let subsample_channels = usize_key(FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY)?;
    let subsample_out_dim = usize_key(FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY)?;
    let feature_dim = usize_key(FIRERED_ENCODER_FEATURE_DIM_KEY)?;
    let encoder_pe_len = usize_key(FIRERED_ENCODER_PE_LEN_KEY)?;
    let decoder_n_layers = usize_key(FIRERED_DECODER_N_LAYERS_KEY)?;
    let decoder_ffn_dim = usize_key(FIRERED_DECODER_FFN_DIM_KEY)?;
    let decoder_pe_len = usize_key(FIRERED_DECODER_PE_LEN_KEY)?;
    let vocab_size = usize_key(FIRERED_VOCAB_SIZE_KEY)?;
    let sos_token_id = u32_key(FIRERED_SOS_TOKEN_ID_KEY)?;
    let eos_token_id = u32_key(FIRERED_EOS_TOKEN_ID_KEY)?;
    let pad_token_id = u32_key(FIRERED_PAD_TOKEN_ID_KEY)?;

    for (key, value) in [
        (FIRERED_ENCODER_N_LAYERS_KEY, encoder_n_layers),
        (FIRERED_ENCODER_D_MODEL_KEY, d_model),
        (FIRERED_ENCODER_N_HEADS_KEY, n_heads),
        (FIRERED_ENCODER_HEAD_DIM_KEY, head_dim),
        (FIRERED_ENCODER_FFN_DIM_KEY, encoder_ffn_dim),
        (FIRERED_ENCODER_CONV_KERNEL_KEY, conv_kernel),
        (FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, subsample_channels),
        (FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY, subsample_out_dim),
        (FIRERED_ENCODER_FEATURE_DIM_KEY, feature_dim),
        (FIRERED_ENCODER_PE_LEN_KEY, encoder_pe_len),
        (FIRERED_DECODER_N_LAYERS_KEY, decoder_n_layers),
        (FIRERED_DECODER_FFN_DIM_KEY, decoder_ffn_dim),
        (FIRERED_DECODER_PE_LEN_KEY, decoder_pe_len),
        (FIRERED_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    if n_heads * head_dim != d_model {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_HEAD_DIM_KEY,
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    if conv_kernel.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_CONV_KERNEL_KEY,
            reason: format!("conv kernel {conv_kernel} must be odd (symmetric padding)"),
        });
    }
    if encoder_pe_len.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_PE_LEN_KEY,
            reason: format!("rel-pos table length {encoder_pe_len} must be odd (2*max-1)"),
        });
    }
    let expected_subsample = subsample_channels * (((feature_dim - 1) / 2 - 1) / 2);
    if subsample_out_dim != expected_subsample {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
            reason: format!(
                "subsample_out_dim {subsample_out_dim} != channels {subsample_channels} x \
                 subsampled {feature_dim}-mel width ({expected_subsample})"
            ),
        });
    }
    for (key, id) in [
        (FIRERED_SOS_TOKEN_ID_KEY, sos_token_id),
        (FIRERED_EOS_TOKEN_ID_KEY, eos_token_id),
        (FIRERED_PAD_TOKEN_ID_KEY, pad_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(FireRedAedExecutionMetadata {
        encoder_n_layers,
        d_model,
        n_heads,
        head_dim,
        encoder_ffn_dim,
        conv_kernel,
        subsample_channels,
        subsample_out_dim,
        feature_dim,
        encoder_pe_len,
        decoder_n_layers,
        decoder_ffn_dim,
        decoder_pe_len,
        vocab_size,
        sos_token_id,
        eos_token_id,
        pad_token_id,
    })
}

/// The pack's declared fbank frontend contract (`firered.audio.*`, written by
/// [`super::package_import`]) must match the fixed frontend the executor
/// actually runs ([`super::frontend`]'s family constants): a pack claiming a
/// different sample rate, FFT, window, hop, or mel width is a mis-converted
/// artifact and fails closed at admission instead of silently running the
/// wrong feature pipeline.
pub(crate) fn validate_firered_aed_frontend_contract<M: ScalarMetadataView>(
    metadata: &M,
    feature_dim: usize,
) -> Result<(), MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let sample_rate_hz = usize_key(FIRERED_AUDIO_SAMPLE_RATE_KEY)?;
    let n_fft = usize_key(FIRERED_AUDIO_N_FFT_KEY)?;
    let frame_length_ms = usize_key(FIRERED_AUDIO_FRAME_LENGTH_MS_KEY)?;
    let frame_shift_ms = usize_key(FIRERED_AUDIO_FRAME_SHIFT_MS_KEY)?;
    let n_mels = usize_key(FIRERED_AUDIO_N_MELS_KEY)?;

    let expect =
        |key: &'static str, actual: usize, expected: usize| -> Result<(), MetadataContractError> {
            if actual == expected {
                return Ok(());
            }
            Err(MetadataContractError::InvalidValue {
                key,
                reason: format!(
                    "pack declares {actual} but the firered-aed frontend is fixed at {expected}"
                ),
            })
        };
    expect(
        FIRERED_AUDIO_SAMPLE_RATE_KEY,
        sample_rate_hz,
        SAMPLE_RATE_HZ as usize,
    )?;
    expect(FIRERED_AUDIO_N_FFT_KEY, n_fft, FFT_SIZE)?;
    expect(
        FIRERED_AUDIO_FRAME_LENGTH_MS_KEY,
        frame_length_ms,
        FRAME_LENGTH_MS as usize,
    )?;
    expect(
        FIRERED_AUDIO_FRAME_SHIFT_MS_KEY,
        frame_shift_ms,
        FRAME_SHIFT_MS as usize,
    )?;
    expect(FIRERED_AUDIO_N_MELS_KEY, n_mels, feature_dim)?;
    Ok(())
}

fn descriptor(
    tensor_name: String,
    requirement: TensorBindingDescriptorRequirement,
    reason: &str,
) -> TensorBindingDescriptor {
    TensorBindingDescriptor {
        tensor_name,
        requirement,
        reason: reason.to_string(),
    }
}

/// One Conformer encoder block's runtime tensor bindings (the 34 tensors
/// `encoder_weights::load` reads and `encoder_graph` consumes). All shapes are
/// fully determined by the parsed metadata; matmul weights are pinned in the
/// ggml `[in, out]` storage orientation the keep-quantized native binding
/// consumes without repack (the importer reverses the torch `[out, in]`
/// layout), so a transposed pack fails closed here instead of computing
/// garbage.
fn encoder_block_tensor_descriptors(
    metadata: &FireRedAedExecutionMetadata,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    let d_model = metadata.d_model;
    let ffn_dim = metadata.encoder_ffn_dim;
    let name = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let mut descriptors = Vec::new();
    for half in ["ffn1", "ffn2"] {
        descriptors.extend([
            descriptor(
                name(&format!("{half}.norm.weight")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "macaron FFN pre-norm gamma must span d_model",
            ),
            descriptor(
                name(&format!("{half}.norm.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "macaron FFN pre-norm beta must span d_model",
            ),
            descriptor(
                name(&format!("{half}.up.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![d_model, ffn_dim]),
                "macaron FFN up projection must be d_model x ffn_dim in [in, out] storage",
            ),
            descriptor(
                name(&format!("{half}.up.bias")),
                TensorBindingDescriptorRequirement::VectorLen(ffn_dim),
                "macaron FFN up bias must span ffn_dim",
            ),
            descriptor(
                name(&format!("{half}.down.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![ffn_dim, d_model]),
                "macaron FFN down projection must be ffn_dim x d_model in [in, out] storage",
            ),
            descriptor(
                name(&format!("{half}.down.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "macaron FFN down bias must span d_model",
            ),
        ]);
    }
    for projection in ["q", "k", "v"] {
        descriptors.extend([
            descriptor(
                name(&format!("attn.norm_{projection}.weight")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "per-projection attention LayerNorm gamma must span d_model",
            ),
            descriptor(
                name(&format!("attn.norm_{projection}.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "per-projection attention LayerNorm beta must span d_model",
            ),
        ]);
    }
    for projection in ["q", "k", "v", "out", "pos"] {
        descriptors.push(descriptor(
            name(&format!("attn.{projection}.weight")),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, d_model]),
            "attention projection must be d_model x d_model in [in, out] storage (upstream bias-free)",
        ));
    }
    descriptors.extend([
        descriptor(
            name("attn.pos_bias_u"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "Transformer-XL rel-pos bias u must be the flattened [heads, head_dim] vector",
        ),
        descriptor(
            name("attn.pos_bias_v"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "Transformer-XL rel-pos bias v must be the flattened [heads, head_dim] vector",
        ),
        descriptor(
            name("conv.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "conv-module pre-norm gamma must span d_model",
        ),
        descriptor(
            name("conv.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "conv-module pre-norm beta must span d_model",
        ),
        descriptor(
            name("conv.pw1.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, 2 * d_model * 2]),
            "GLU pointwise conv1 (kernel-1 squeezed) must be d_model x 4*d_model in [in, out] storage",
        ),
        descriptor(
            name("conv.dw.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![
                metadata.conv_kernel,
                1,
                2 * d_model,
            ]),
            "depthwise conv kernel must be [conv_kernel, 1, 2*d_model] for the im2col conv path",
        ),
        descriptor(
            name("conv.ln.weight"),
            TensorBindingDescriptorRequirement::VectorLen(2 * d_model),
            "conv mid-block LayerNorm gamma must span the 2*d_model GLU channels",
        ),
        descriptor(
            name("conv.ln.bias"),
            TensorBindingDescriptorRequirement::VectorLen(2 * d_model),
            "conv mid-block LayerNorm beta must span the 2*d_model GLU channels",
        ),
        descriptor(
            name("conv.pw2.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![2 * d_model, d_model]),
            "pointwise conv2 (kernel-1 squeezed) must be 2*d_model x d_model in [in, out] storage",
        ),
        descriptor(
            name("out_norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "block-tail affine LayerNorm gamma must span d_model",
        ),
        descriptor(
            name("out_norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "block-tail affine LayerNorm beta must span d_model",
        ),
    ]);
    descriptors
}

/// One pre-norm Transformer decoder block's runtime tensor bindings (the 24
/// tensors `decoder_weights::load` reads and `decoder_graph` consumes).
/// `self_attn.k` / `cross_attn.k` are upstream bias-free; the graph supplies
/// one shared zero bias for both, so no K-bias descriptor exists.
fn decoder_block_tensor_descriptors(
    metadata: &FireRedAedExecutionMetadata,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    let d_model = metadata.d_model;
    let ffn_dim = metadata.decoder_ffn_dim;
    let name = |suffix: &str| format!("dec.blk.{layer}.{suffix}");
    let mut descriptors = Vec::new();
    for scope in ["self_attn", "cross_attn"] {
        descriptors.extend([
            descriptor(
                name(&format!("{scope}.norm.weight")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "pre-attention LayerNorm gamma must span d_model",
            ),
            descriptor(
                name(&format!("{scope}.norm.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "pre-attention LayerNorm beta must span d_model",
            ),
            descriptor(
                name(&format!("{scope}.q.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![d_model, d_model]),
                "attention Q projection must be d_model x d_model in [in, out] storage",
            ),
            descriptor(
                name(&format!("{scope}.q.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "attention Q bias must span d_model",
            ),
            descriptor(
                name(&format!("{scope}.k.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![d_model, d_model]),
                "attention K projection must be d_model x d_model in [in, out] storage (upstream bias-free)",
            ),
            descriptor(
                name(&format!("{scope}.v.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![d_model, d_model]),
                "attention V projection must be d_model x d_model in [in, out] storage",
            ),
            descriptor(
                name(&format!("{scope}.v.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "attention V bias must span d_model",
            ),
            descriptor(
                name(&format!("{scope}.out.weight")),
                TensorBindingDescriptorRequirement::ExactDims(vec![d_model, d_model]),
                "attention output projection must be d_model x d_model in [in, out] storage",
            ),
            descriptor(
                name(&format!("{scope}.out.bias")),
                TensorBindingDescriptorRequirement::VectorLen(d_model),
                "attention output bias must span d_model",
            ),
        ]);
    }
    descriptors.extend([
        descriptor(
            name("ffn.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm gamma must span d_model",
        ),
        descriptor(
            name("ffn.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm beta must span d_model",
        ),
        descriptor(
            name("ffn.up.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, ffn_dim]),
            "FFN up projection must be d_model x ffn_dim in [in, out] storage",
        ),
        descriptor(
            name("ffn.up.bias"),
            TensorBindingDescriptorRequirement::VectorLen(ffn_dim),
            "FFN up bias must span ffn_dim",
        ),
        descriptor(
            name("ffn.down.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![ffn_dim, d_model]),
            "FFN down projection must be ffn_dim x d_model in [in, out] storage",
        ),
        descriptor(
            name("ffn.down.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "FFN down bias must span d_model",
        ),
    ]);
    descriptors
}

/// The runtime tensor contract for one firered-aed pack: every tensor the
/// executor materializes (`encoder_weights` + `decoder_weights` + the frontend
/// CMVN vectors), with the exact shapes the Conformer/Transformer graphs
/// consume, plus the two pack-contract provenance tables the importer always
/// bakes (`enc.pos_enc.pe`, `firered.mel_filters`). Derived from the parsed
/// metadata, so a checkpoint with different layer counts validates its own
/// geometry. Shared with the verifier-ready test fixture so both sides agree
/// on the tensor set through one contract.
pub(crate) fn firered_aed_runtime_tensor_binding_descriptors(
    metadata: &FireRedAedExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let d_model = metadata.d_model;
    let mut descriptors = Vec::new();
    for layer in 0..metadata.encoder_n_layers {
        descriptors.extend(encoder_block_tensor_descriptors(metadata, layer));
    }
    let channels = metadata.subsample_channels;
    descriptors.extend([
        descriptor(
            "enc.subsample.conv1.weight".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, 1, channels]),
            "subsampling conv1 kernel must be [3, 3, 1, channels] for the im2col conv path",
        ),
        descriptor(
            "enc.subsample.conv1.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(channels),
            "subsampling conv1 bias must span its channel count",
        ),
        descriptor(
            "enc.subsample.conv2.weight".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, channels, channels]),
            "subsampling conv2 kernel must be [3, 3, channels, channels] for the im2col conv path",
        ),
        descriptor(
            "enc.subsample.conv2.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(channels),
            "subsampling conv2 bias must span its channel count",
        ),
        descriptor(
            "enc.subsample.out.weight".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![
                metadata.subsample_out_dim,
                d_model,
            ]),
            "subsampling out projection must be subsample_out_dim x d_model in [in, out] storage",
        ),
        descriptor(
            "enc.subsample.out.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "subsampling out bias must span d_model",
        ),
        // Provenance table: the runtime synthesizes the rel-pos rows on demand
        // (`nn::encoder::build_relative_positional_encoding`), but every pack
        // this family's importer writes bakes the upstream table -- a missing
        // or mis-shaped one marks a truncated/mis-converted artifact.
        descriptor(
            "enc.pos_enc.pe".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, metadata.encoder_pe_len]),
            "encoder rel-pos provenance table must be d_model x pe_len (canonical rank-2 after trailing-axis trim)",
        ),
    ]);
    for layer in 0..metadata.decoder_n_layers {
        descriptors.extend(decoder_block_tensor_descriptors(metadata, layer));
    }
    descriptors.extend([
        descriptor(
            "dec.tok_emb.weight".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, metadata.vocab_size]),
            "token embedding table must be d_model x vocab_size (get_rows source)",
        ),
        descriptor(
            "dec.pos_enc.pe".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, metadata.decoder_pe_len]),
            "decoder absolute position table must be d_model x decoder_pe_len (canonical rank-2 after trailing-axis trim)",
        ),
        descriptor(
            "dec.out_norm.weight".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "decoder tail affine LayerNorm gamma must span d_model",
        ),
        descriptor(
            "dec.out_norm.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "decoder tail affine LayerNorm beta must span d_model",
        ),
        descriptor(
            "dec.out_proj.weight".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![d_model, metadata.vocab_size]),
            "untied output projection must be d_model x vocab_size in [in, out] storage (bias-free upstream)",
        ),
        descriptor(
            "frontend.cmvn.neg_mean".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(metadata.feature_dim),
            "CMVN neg-mean must span the feature dim",
        ),
        descriptor(
            "frontend.cmvn.inv_stddev".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(metadata.feature_dim),
            "CMVN inverse-stddev must span the feature dim",
        ),
        // Provenance filterbank: the runtime recomputes the bank through the
        // shared kaldi-fbank engine, but the importer always bakes the
        // upstream-equivalent table -- pin it so a truncated pack fails here.
        descriptor(
            "firered.mel_filters".to_string(),
            TensorBindingDescriptorRequirement::ExactDims(vec![
                metadata.feature_dim,
                FIRERED_MEL_FILTER_BANK_BINS,
            ]),
            "packed provenance mel filterbank must be [feature_dim, n_fft/2+1]",
        ),
    ]);
    descriptors
}

/// Validate the full runtime tensor set against the pack's tensor index.
pub(crate) fn validate_firered_aed_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &FireRedAedExecutionMetadata,
) -> Result<(), FireRedAedRuntimeContractError> {
    let descriptors = firered_aed_runtime_tensor_binding_descriptors(metadata);
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )
}

/// Admission-time tokenizer contract. The runtime materializes the tokenizer
/// from exactly `tokenizer.ggml.tokens` (`executor.rs` builds
/// [`FireRedTokenizer`] from that key), the decoder samples ids in
/// `[0, vocab_size)` (the `dec.out_proj` logits width) and detokenizes them
/// through the pack-carried vocab, so a missing/empty/shorter-than-vocab
/// tokens array fails closed at pack admission instead of mid-decode
/// (moonshine/sensevoice precedent).
pub(crate) fn validate_firered_aed_tokenizer_contract(
    metadata: &GgufMetadata,
    vocab_size: usize,
) -> Result<(), MetadataContractError> {
    let tokens = metadata.get_string_array(TOKENIZER_GGML_TOKENS_KEY).ok_or(
        MetadataContractError::MissingRequiredKey {
            key: TOKENIZER_GGML_TOKENS_KEY,
        },
    )?;
    if tokens.is_empty() {
        return Err(MetadataContractError::InvalidValue {
            key: TOKENIZER_GGML_TOKENS_KEY,
            reason: "tokenizer vocab is empty".to_string(),
        });
    }
    if tokens.len() < vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: TOKENIZER_GGML_TOKENS_KEY,
            reason: format!(
                "tokenizer vocab carries {} tokens but {FIRERED_VOCAB_SIZE_KEY}={vocab_size} \
                 requires coverage of every sampleable id",
                tokens.len()
            ),
        });
    }
    // Materialize the family tokenizer from the exact key the executor reads;
    // its vocab must agree with the coverage proof above.
    let tokenizer = FireRedTokenizer::new(tokens.to_vec());
    if tokenizer.vocab_size() < vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: TOKENIZER_GGML_TOKENS_KEY,
            reason: format!(
                "materialized firered tokenizer vocab {} < {FIRERED_VOCAB_SIZE_KEY}={vocab_size}",
                tokenizer.vocab_size()
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let execution_metadata =
        parse_firered_aed_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("firered-aed", error)
        })?;
    validate_firered_aed_frontend_contract(preflight.metadata(), execution_metadata.feature_dim)
        .map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                "firered-aed frontend",
                error,
            )
        })?;
    validate_firered_aed_runtime_tensors_with_index(preflight.tensor_index(), &execution_metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)?;
    validate_firered_aed_tokenizer_contract(preflight.metadata(), execution_metadata.vocab_size)
        .map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                "firered-aed tokenizer",
                error,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufMetadataValue;
    use crate::testing::TinyGgufFixtureSpec;
    use std::collections::BTreeMap;

    fn aed_l_metadata() -> BTreeMap<String, String> {
        [
            (FIRERED_ENCODER_N_LAYERS_KEY, "16"),
            (FIRERED_ENCODER_D_MODEL_KEY, "1280"),
            (FIRERED_ENCODER_N_HEADS_KEY, "20"),
            (FIRERED_ENCODER_HEAD_DIM_KEY, "64"),
            (FIRERED_ENCODER_FFN_DIM_KEY, "5120"),
            (FIRERED_ENCODER_CONV_KERNEL_KEY, "33"),
            (FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, "32"),
            (FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY, "608"),
            (FIRERED_ENCODER_FEATURE_DIM_KEY, "80"),
            (FIRERED_ENCODER_PE_LEN_KEY, "9999"),
            (FIRERED_DECODER_N_LAYERS_KEY, "16"),
            (FIRERED_DECODER_FFN_DIM_KEY, "5120"),
            (FIRERED_DECODER_PE_LEN_KEY, "5000"),
            (FIRERED_VOCAB_SIZE_KEY, "7832"),
            (FIRERED_SOS_TOKEN_ID_KEY, "3"),
            (FIRERED_EOS_TOKEN_ID_KEY, "4"),
            (FIRERED_PAD_TOKEN_ID_KEY, "2"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn aed_l_frontend_metadata() -> BTreeMap<String, String> {
        [
            (FIRERED_AUDIO_SAMPLE_RATE_KEY, "16000"),
            (FIRERED_AUDIO_N_FFT_KEY, "512"),
            (FIRERED_AUDIO_FRAME_LENGTH_MS_KEY, "25"),
            (FIRERED_AUDIO_FRAME_SHIFT_MS_KEY, "10"),
            (FIRERED_AUDIO_N_MELS_KEY, "80"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Rebuild the fixture's GGUF metadata view (scalars as strings + the
    /// string-array keys) so the tokenizer contract can be checked against it.
    fn gguf_metadata_from_spec(spec: &TinyGgufFixtureSpec) -> GgufMetadata {
        let mut values: BTreeMap<String, GgufMetadataValue> = spec
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), GgufMetadataValue::String(value.clone())))
            .collect();
        for (key, entries) in &spec.metadata_string_arrays {
            values.insert(key.clone(), GgufMetadataValue::StringArray(entries.clone()));
        }
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn parses_aed_l_metadata() {
        let parsed = parse_firered_aed_execution_metadata(&aed_l_metadata()).expect("parse");
        assert_eq!(parsed.encoder_n_layers, 16);
        assert_eq!(parsed.d_model, 1280);
        assert_eq!(parsed.head_dim, 64);
        assert_eq!(parsed.encoder_max_frames(), 5000);
        assert_eq!(parsed.sos_token_id, 3);
        assert_eq!(parsed.eos_token_id, 4);
    }

    #[test]
    fn rejects_head_geometry_mismatch() {
        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_ENCODER_HEAD_DIM_KEY.to_string(), "60".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_even_conv_kernel_and_pe_len() {
        let mut metadata = aed_l_metadata();
        metadata.insert(
            FIRERED_ENCODER_CONV_KERNEL_KEY.to_string(),
            "32".to_string(),
        );
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());

        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_ENCODER_PE_LEN_KEY.to_string(), "10000".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_subsample_out_dim_mismatch() {
        let mut metadata = aed_l_metadata();
        metadata.insert(
            FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY.to_string(),
            "600".to_string(),
        );
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_special_token_out_of_vocab() {
        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_EOS_TOKEN_ID_KEY.to_string(), "9000".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_missing_key() {
        let mut metadata = aed_l_metadata();
        metadata.remove(FIRERED_VOCAB_SIZE_KEY);
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn frontend_contract_accepts_the_fixed_fbank_configuration() {
        validate_firered_aed_frontend_contract(&aed_l_frontend_metadata(), 80)
            .expect("the importer-written frontend contract must validate");
    }

    #[test]
    fn frontend_contract_rejects_a_drifted_sample_rate() {
        let mut metadata = aed_l_frontend_metadata();
        metadata.insert(
            FIRERED_AUDIO_SAMPLE_RATE_KEY.to_string(),
            "8000".to_string(),
        );
        let error = validate_firered_aed_frontend_contract(&metadata, 80)
            .expect_err("a drifted sample rate must fail closed");
        assert!(matches!(
            error,
            MetadataContractError::InvalidValue {
                key: FIRERED_AUDIO_SAMPLE_RATE_KEY,
                ..
            }
        ));
    }

    #[test]
    fn frontend_contract_rejects_a_mel_width_mismatching_feature_dim() {
        let error = validate_firered_aed_frontend_contract(&aed_l_frontend_metadata(), 40)
            .expect_err("n_mels must equal the metadata feature_dim");
        assert!(matches!(
            error,
            MetadataContractError::InvalidValue {
                key: FIRERED_AUDIO_N_MELS_KEY,
                ..
            }
        ));
    }

    #[test]
    fn frontend_contract_rejects_a_missing_audio_key() {
        let mut metadata = aed_l_frontend_metadata();
        metadata.remove(FIRERED_AUDIO_N_FFT_KEY);
        assert!(matches!(
            validate_firered_aed_frontend_contract(&metadata, 80),
            Err(MetadataContractError::MissingRequiredKey {
                key: FIRERED_AUDIO_N_FFT_KEY
            })
        ));
    }

    #[test]
    fn binding_descriptors_cover_every_runtime_tensor_exactly_once() {
        let metadata = parse_firered_aed_execution_metadata(&aed_l_metadata()).expect("parse");
        let descriptors = firered_aed_runtime_tensor_binding_descriptors(&metadata);
        // 34 tensors per Conformer block (16) + 7 subsample/provenance tensors
        // + 24 tensors per decoder block (16) + 5 decoder-top tensors
        // + 3 frontend tensors.
        assert_eq!(descriptors.len(), 34 * 16 + 7 + 24 * 16 + 5 + 3);
        let mut names = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            descriptors.len(),
            "every runtime tensor must be bound exactly once"
        );
        for required in [
            "enc.blk.0.attn.q.weight",
            "enc.blk.15.conv.dw.weight",
            "enc.subsample.out.weight",
            "enc.pos_enc.pe",
            "dec.blk.0.self_attn.k.weight",
            "dec.blk.15.ffn.down.bias",
            "dec.tok_emb.weight",
            "dec.pos_enc.pe",
            "dec.out_proj.weight",
            "frontend.cmvn.neg_mean",
            "firered.mel_filters",
        ] {
            assert!(
                names.contains(&required),
                "binding list must cover {required}"
            );
        }
    }

    #[test]
    fn validates_runtime_ready_fixture_tensors() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::firered_aed_oasr_v1_runtime_ready("firered-aed-fixture");
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_firered_aed_execution_metadata(&spec.metadata).expect("metadata must parse");
        validate_firered_aed_frontend_contract(&spec.metadata, metadata.feature_dim)
            .expect("frontend contract must validate");
        validate_firered_aed_runtime_tensors_with_index(&index, &metadata)
            .expect("runtime-ready tensor fixture must validate");
        validate_firered_aed_tokenizer_contract(
            &gguf_metadata_from_spec(&spec),
            metadata.vocab_size,
        )
        .expect("tokenizer vocab must cover every sampleable id");
    }

    #[test]
    fn rejects_fixture_missing_required_tensor() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::firered_aed_oasr_v1_runtime_ready("firered-aed-fixture")
            .without_tensor("dec.blk.0.cross_attn.out.bias");
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_firered_aed_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_firered_aed_runtime_tensors_with_index(&index, &metadata)
            .expect_err("missing tensor must fail closed");
        assert!(
            matches!(error, FireRedAedRuntimeContractError::MissingRequiredTensor { ref name } if name == "dec.blk.0.cross_attn.out.bias"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_fixture_shape_mismatch() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::firered_aed_oasr_v1_runtime_ready("firered-aed-fixture")
            .with_tensor_shape("enc.blk.0.attn.q.weight", [16_u64, 12_u64]);
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_firered_aed_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_firered_aed_runtime_tensors_with_index(&index, &metadata)
            .expect_err("shape mismatch must fail closed");
        assert!(
            matches!(error, FireRedAedRuntimeContractError::InvalidTensorShape { ref name, .. } if name == "enc.blk.0.attn.q.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_transposed_matmul_weight_orientation() {
        // The keep-quantized native binding consumes [in, out] storage without
        // repack; a transposed projection must fail admission, not compute
        // garbage.
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::firered_aed_oasr_v1_runtime_ready("firered-aed-fixture")
            .with_tensor_shape("dec.blk.0.ffn.up.weight", [32_u64, 16_u64]);
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_firered_aed_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_firered_aed_runtime_tensors_with_index(&index, &metadata)
            .expect_err("transposed orientation must fail closed");
        assert!(
            matches!(error, FireRedAedRuntimeContractError::InvalidTensorShape { ref name, .. } if name == "dec.blk.0.ffn.up.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn tokenizer_contract_rejects_a_truncated_vocab() {
        let mut values: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(vec!["a".to_string(), "b".to_string()]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = validate_firered_aed_tokenizer_contract(&metadata, 3)
            .expect_err("short vocab must fail closed");
        assert!(matches!(
            error,
            MetadataContractError::InvalidValue {
                key: TOKENIZER_GGML_TOKENS_KEY,
                ..
            }
        ));
    }

    #[test]
    fn tokenizer_contract_rejects_missing_or_empty_vocab() {
        let metadata = GgufMetadata::from_values_for_test(BTreeMap::new());
        assert!(matches!(
            validate_firered_aed_tokenizer_contract(&metadata, 3),
            Err(MetadataContractError::MissingRequiredKey { .. })
        ));

        let mut values: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(Vec::new()),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(matches!(
            validate_firered_aed_tokenizer_contract(&metadata, 3),
            Err(MetadataContractError::InvalidValue { .. })
        ));
    }

    #[test]
    fn tokenizer_contract_accepts_full_coverage_and_oversized_vocabs() {
        for tokens in [
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "<extra>".to_string(),
            ],
        ] {
            let mut values: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
            values.insert(
                TOKENIZER_GGML_TOKENS_KEY.to_string(),
                GgufMetadataValue::StringArray(tokens),
            );
            let metadata = GgufMetadata::from_values_for_test(values);
            validate_firered_aed_tokenizer_contract(&metadata, 3)
                .expect("vocab covering every sampleable id must pass");
        }
    }

    /// The runtime contract's required scalar keys must be exactly the arch
    /// hparam schema (drift here would let a pack pass install but miss a key
    /// the executor needs) -- the dolphin precedent for this SSOT gate.
    #[test]
    fn required_keys_match_arch_hparam_schema() {
        let mut contract_keys = [
            FIRERED_ENCODER_N_LAYERS_KEY,
            FIRERED_ENCODER_D_MODEL_KEY,
            FIRERED_ENCODER_N_HEADS_KEY,
            FIRERED_ENCODER_HEAD_DIM_KEY,
            FIRERED_ENCODER_FFN_DIM_KEY,
            FIRERED_ENCODER_CONV_KERNEL_KEY,
            FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY,
            FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
            FIRERED_ENCODER_FEATURE_DIM_KEY,
            FIRERED_ENCODER_PE_LEN_KEY,
            FIRERED_DECODER_N_LAYERS_KEY,
            FIRERED_DECODER_FFN_DIM_KEY,
            FIRERED_DECODER_PE_LEN_KEY,
            FIRERED_VOCAB_SIZE_KEY,
            FIRERED_SOS_TOKEN_ID_KEY,
            FIRERED_EOS_TOKEN_ID_KEY,
            FIRERED_PAD_TOKEN_ID_KEY,
            FIRERED_AUDIO_SAMPLE_RATE_KEY,
            FIRERED_AUDIO_N_FFT_KEY,
            FIRERED_AUDIO_FRAME_LENGTH_MS_KEY,
            FIRERED_AUDIO_FRAME_SHIFT_MS_KEY,
            FIRERED_AUDIO_N_MELS_KEY,
        ]
        .to_vec();
        contract_keys.sort_unstable();
        let mut schema_keys = crate::arch::hparams::FIRERED_AED_HPARAM_SCHEMA.to_vec();
        schema_keys.sort_unstable();
        assert_eq!(contract_keys, schema_keys);
    }
}
