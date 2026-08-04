//! granite-speech execution contract against an admitted `.oasr` runtime
//! source: the family runtime validator parses every execution metadata key
//! `package_import` writes AND validates the complete three-stage tensor set
//! (Conformer encoder, Q-Former projector, Granite decoder) against the shapes
//! those metadata declare (metadata + tensor depth, the same contract shape
//! `moss_transcribe_diarize::runtime_contract` and `qwen::runtime_contract`
//! enforce). Key names mirror `arch::hparams::GRANITE_SPEECH_HPARAM_SCHEMA`
//! exactly, and the parsed values feed the same config structs
//! `encoder_graph`/`qformer`/`decoder_graph` already accept, so the install-
//! time pack check and the executor read the exact same parsed values -- no
//! second copy of the hparam list to drift.
//!
//! The tensor descriptors are derived from the parsed metadata -- never from a
//! hardcoded checkpoint shape -- so a future legitimately-shaped pack is
//! admitted by its own declared geometry while a truncated, reshaped, or
//! mis-converted pack fails closed at admission with the offending tensor
//! named, instead of passing a metadata-only check and failing later inside
//! the executor. Pack dim convention: every rank-2 matmul weight is stored in
//! ggml `[in, out]` order (the importer reverses the HF `[out, in]` extents),
//! so the contract pins that exact orientation; 1-D norm/bias/stat tensors and
//! the rank-3 depthwise conv kernel / learned query keep their source extent
//! order (see `package_import`'s module doc).

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::arch::{GENERAL_ARCHITECTURE_KEY, GRANITE_SPEECH_GGML_ARCHITECTURE_ID};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_f32_scalar, required_string_scalar,
    required_u64_scalar, u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

use super::decoder_graph::GraniteSpeechDecoderConfig;
use super::encoder_graph::GraniteSpeechEncoderConfig;
use super::executor::GRANITE_SPEECH_EOT_TOKEN_ID;
use super::prompt::GRANITE_SPEECH_AUDIO_TOKEN;
use super::qformer::GraniteSpeechProjectorConfig;

pub(crate) const GRANITE_SPEECH_CONTRACT_FAMILY: &str = "granite-speech";

// --- top-level family keys -------------------------------------------------
pub(crate) const AUDIO_TOKEN_INDEX_KEY: &str = "granite_speech.audio_token_index";
pub(crate) const DOWNSAMPLE_RATE_KEY: &str = "granite_speech.downsample_rate";
pub(crate) const WINDOW_SIZE_KEY: &str = "granite_speech.window_size";

// --- encoder stage keys ----------------------------------------------------
pub(crate) const ENCODER_INPUT_DIM_KEY: &str = "granite_speech.encoder.input_dim";
pub(crate) const ENCODER_HIDDEN_DIM_KEY: &str = "granite_speech.encoder.hidden_dim";
pub(crate) const ENCODER_NUM_LAYERS_KEY: &str = "granite_speech.encoder.num_layers";
pub(crate) const ENCODER_NUM_HEADS_KEY: &str = "granite_speech.encoder.num_heads";
pub(crate) const ENCODER_DIM_HEAD_KEY: &str = "granite_speech.encoder.dim_head";
pub(crate) const ENCODER_FEEDFORWARD_MULT_KEY: &str = "granite_speech.encoder.feedforward_mult";
pub(crate) const ENCODER_CONV_KERNEL_SIZE_KEY: &str = "granite_speech.encoder.conv_kernel_size";
pub(crate) const ENCODER_CONV_EXPANSION_FACTOR_KEY: &str =
    "granite_speech.encoder.conv_expansion_factor";
pub(crate) const ENCODER_CONTEXT_SIZE_KEY: &str = "granite_speech.encoder.context_size";
pub(crate) const ENCODER_MAX_POS_EMB_KEY: &str = "granite_speech.encoder.max_pos_emb";
pub(crate) const ENCODER_OUTPUT_DIM_KEY: &str = "granite_speech.encoder.output_dim";

// --- projector stage keys --------------------------------------------------
pub(crate) const PROJECTOR_ENCODER_HIDDEN_SIZE_KEY: &str =
    "granite_speech.projector.encoder_hidden_size";
pub(crate) const PROJECTOR_NUM_HIDDEN_LAYERS_KEY: &str =
    "granite_speech.projector.num_hidden_layers";
pub(crate) const PROJECTOR_NUM_ATTENTION_HEADS_KEY: &str =
    "granite_speech.projector.num_attention_heads";
pub(crate) const PROJECTOR_INTERMEDIATE_SIZE_KEY: &str =
    "granite_speech.projector.intermediate_size";

// --- decoder stage keys ----------------------------------------------------
pub(crate) const DECODER_HIDDEN_SIZE_KEY: &str = "granite_speech.decoder.hidden_size";
pub(crate) const DECODER_NUM_HIDDEN_LAYERS_KEY: &str = "granite_speech.decoder.num_hidden_layers";
pub(crate) const DECODER_NUM_ATTENTION_HEADS_KEY: &str =
    "granite_speech.decoder.num_attention_heads";
pub(crate) const DECODER_HEAD_DIM_KEY: &str = "granite_speech.decoder.head_dim";
pub(crate) const DECODER_NUM_KEY_VALUE_HEADS_KEY: &str =
    "granite_speech.decoder.num_key_value_heads";
pub(crate) const DECODER_INTERMEDIATE_SIZE_KEY: &str = "granite_speech.decoder.intermediate_size";
pub(crate) const DECODER_VOCAB_SIZE_KEY: &str = "granite_speech.decoder.vocab_size";
pub(crate) const DECODER_RMS_NORM_EPS_KEY: &str = "granite_speech.decoder.rms_norm_eps";
pub(crate) const DECODER_ROPE_THETA_KEY: &str = "granite_speech.decoder.rope_theta";
pub(crate) const DECODER_ATTENTION_MULTIPLIER_KEY: &str =
    "granite_speech.decoder.attention_multiplier";
pub(crate) const DECODER_EMBEDDING_MULTIPLIER_KEY: &str =
    "granite_speech.decoder.embedding_multiplier";
pub(crate) const DECODER_RESIDUAL_MULTIPLIER_KEY: &str =
    "granite_speech.decoder.residual_multiplier";
pub(crate) const DECODER_LOGITS_SCALING_KEY: &str = "granite_speech.decoder.logits_scaling";

/// Fixed architectural constants that never travel as pack metadata (they
/// mirror the shipped 4.1-2b checkpoint; same "family constant, not a GGUF
/// key" convention funasr-nano/moss use for their norm epsilons).
const GRANITE_SPEECH_ENCODER_LAYER_NORM_EPS: f32 = 1.0e-5;
const GRANITE_SPEECH_ENCODER_BATCH_NORM_EPS: f32 = 1.0e-5;
/// BLIP-2 Q-Former LayerNorm epsilon (fixed architectural constant).
const GRANITE_SPEECH_PROJECTOR_LAYER_NORM_EPS: f32 = 1.0e-12;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum GraniteSpeechRuntimeContractError {
    #[error("missing required granite-speech metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("granite-speech metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("granite-speech pack declares architecture '{found}', expected '{expected}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("granite-speech pack is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("granite-speech tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

impl From<MetadataContractError> for GraniteSpeechRuntimeContractError {
    fn from(error: MetadataContractError) -> Self {
        match error {
            MetadataContractError::MissingRequiredKey { key } => {
                Self::MissingRequiredMetadata { key }
            }
            MetadataContractError::InvalidValue { key, reason } => {
                Self::InvalidMetadataValue { key, reason }
            }
        }
    }
}

fn usize_key<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<usize, GraniteSpeechRuntimeContractError> {
    u64_to_usize(required_u64_scalar(metadata, key)?, key).map_err(Into::into)
}

fn u32_key<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<u32, GraniteSpeechRuntimeContractError> {
    u64_to_u32(required_u64_scalar(metadata, key)?, key).map_err(Into::into)
}

/// The Conformer encoder stage. The two norm epsilons are fixed architectural
/// constants, not pack metadata (see `GRANITE_SPEECH_ENCODER_LAYER_NORM_EPS`).
pub(crate) fn parse_encoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<GraniteSpeechEncoderConfig, GraniteSpeechRuntimeContractError> {
    let input_dim = usize_key(metadata, ENCODER_INPUT_DIM_KEY)?;
    let hidden_dim = usize_key(metadata, ENCODER_HIDDEN_DIM_KEY)?;
    let num_layers = usize_key(metadata, ENCODER_NUM_LAYERS_KEY)?;
    let num_heads = usize_key(metadata, ENCODER_NUM_HEADS_KEY)?;
    let dim_head = usize_key(metadata, ENCODER_DIM_HEAD_KEY)?;
    let feedforward_mult = usize_key(metadata, ENCODER_FEEDFORWARD_MULT_KEY)?;
    let conv_kernel_size = usize_key(metadata, ENCODER_CONV_KERNEL_SIZE_KEY)?;
    let conv_expansion_factor = usize_key(metadata, ENCODER_CONV_EXPANSION_FACTOR_KEY)?;
    let context_size = usize_key(metadata, ENCODER_CONTEXT_SIZE_KEY)?;
    let max_pos_emb = usize_key(metadata, ENCODER_MAX_POS_EMB_KEY)?;
    let output_dim = usize_key(metadata, ENCODER_OUTPUT_DIM_KEY)?;
    for (key, value) in [
        (ENCODER_INPUT_DIM_KEY, input_dim),
        (ENCODER_HIDDEN_DIM_KEY, hidden_dim),
        (ENCODER_NUM_LAYERS_KEY, num_layers),
        (ENCODER_NUM_HEADS_KEY, num_heads),
        (ENCODER_DIM_HEAD_KEY, dim_head),
        (ENCODER_FEEDFORWARD_MULT_KEY, feedforward_mult),
        (ENCODER_CONV_KERNEL_SIZE_KEY, conv_kernel_size),
        (ENCODER_CONV_EXPANSION_FACTOR_KEY, conv_expansion_factor),
        (ENCODER_CONTEXT_SIZE_KEY, context_size),
        (ENCODER_MAX_POS_EMB_KEY, max_pos_emb),
        (ENCODER_OUTPUT_DIM_KEY, output_dim),
    ] {
        validate_positive_usize(value, key)?;
    }
    // The depthwise temporal conv pads symmetrically with `kernel / 2` on both
    // sides (`conv_module`); only an odd kernel keeps the frame count
    // unchanged through the module, so an even kernel cannot be assembled.
    if conv_kernel_size.is_multiple_of(2) {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_CONV_KERNEL_SIZE_KEY,
            reason: format!("conv kernel {conv_kernel_size} must be odd (symmetric padding)"),
        });
    }
    // The Shaw relative-position table indexes `max_pos_emb +/- context_size`
    // (`attention_dists_table` clamps block-local distances to +/-context_size
    // before offsetting by max_pos_emb) into a `2 * max_pos_emb + 1` row
    // embedding; a context block wider than the positional span would read
    // outside the table.
    if max_pos_emb < context_size {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_MAX_POS_EMB_KEY,
            reason: format!(
                "max_pos_emb {max_pos_emb} must be >= context_size {context_size} or the relative-position table indexes out of bounds"
            ),
        });
    }
    Ok(GraniteSpeechEncoderConfig {
        input_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dim_head,
        feedforward_mult,
        conv_kernel_size,
        conv_expansion_factor,
        context_size,
        max_pos_emb,
        output_dim,
        layer_norm_eps: GRANITE_SPEECH_ENCODER_LAYER_NORM_EPS,
        batch_norm_eps: GRANITE_SPEECH_ENCODER_BATCH_NORM_EPS,
    })
}

/// The BLIP-2 Q-Former projector stage. Its LLM-side width is the decoder's
/// `hidden_size` key itself (the importer writes no independent projector LLM
/// key), so the cross-stage geometry this parse leaves to
/// [`parse_granite_speech_execution_metadata`] is the encoder->projector
/// bridge.
pub(crate) fn parse_projector_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<GraniteSpeechProjectorConfig, GraniteSpeechRuntimeContractError> {
    let encoder_hidden_size = usize_key(metadata, PROJECTOR_ENCODER_HIDDEN_SIZE_KEY)?;
    let llm_hidden_size = usize_key(metadata, DECODER_HIDDEN_SIZE_KEY)?;
    let window_size = usize_key(metadata, WINDOW_SIZE_KEY)?;
    let downsample_rate = usize_key(metadata, DOWNSAMPLE_RATE_KEY)?;
    let num_hidden_layers = usize_key(metadata, PROJECTOR_NUM_HIDDEN_LAYERS_KEY)?;
    let num_attention_heads = usize_key(metadata, PROJECTOR_NUM_ATTENTION_HEADS_KEY)?;
    let intermediate_size = usize_key(metadata, PROJECTOR_INTERMEDIATE_SIZE_KEY)?;
    for (key, value) in [
        (PROJECTOR_ENCODER_HIDDEN_SIZE_KEY, encoder_hidden_size),
        (DECODER_HIDDEN_SIZE_KEY, llm_hidden_size),
        (WINDOW_SIZE_KEY, window_size),
        (DOWNSAMPLE_RATE_KEY, downsample_rate),
        (PROJECTOR_NUM_HIDDEN_LAYERS_KEY, num_hidden_layers),
        (PROJECTOR_NUM_ATTENTION_HEADS_KEY, num_attention_heads),
        (PROJECTOR_INTERMEDIATE_SIZE_KEY, intermediate_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    // The learned-query count is `window_size / downsample_rate`
    // (`GraniteSpeechProjectorConfig::num_queries`); a fractional window would
    // truncate the query tensor the graph gathers.
    if !window_size.is_multiple_of(downsample_rate) {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: WINDOW_SIZE_KEY,
            reason: format!(
                "window_size {window_size} must be a multiple of downsample_rate {downsample_rate}"
            ),
        });
    }
    if !encoder_hidden_size.is_multiple_of(num_attention_heads) {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: PROJECTOR_NUM_ATTENTION_HEADS_KEY,
            reason: format!(
                "encoder_hidden_size {encoder_hidden_size} is not a multiple of num_attention_heads {num_attention_heads}"
            ),
        });
    }
    Ok(GraniteSpeechProjectorConfig {
        encoder_hidden_size,
        llm_hidden_size,
        window_size,
        downsample_rate,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        layer_norm_eps: GRANITE_SPEECH_PROJECTOR_LAYER_NORM_EPS,
    })
}

/// The Granite dense decoder stage (GQA + RoPE + SwiGLU with the four Granite
/// scaling scalars).
pub(crate) fn parse_decoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<GraniteSpeechDecoderConfig, GraniteSpeechRuntimeContractError> {
    let hidden_size = usize_key(metadata, DECODER_HIDDEN_SIZE_KEY)?;
    let num_layers = usize_key(metadata, DECODER_NUM_HIDDEN_LAYERS_KEY)?;
    let num_heads = usize_key(metadata, DECODER_NUM_ATTENTION_HEADS_KEY)?;
    let num_kv_heads = usize_key(metadata, DECODER_NUM_KEY_VALUE_HEADS_KEY)?;
    let head_dim = usize_key(metadata, DECODER_HEAD_DIM_KEY)?;
    let intermediate_size = usize_key(metadata, DECODER_INTERMEDIATE_SIZE_KEY)?;
    let vocab_size = usize_key(metadata, DECODER_VOCAB_SIZE_KEY)?;
    for (key, value) in [
        (DECODER_HIDDEN_SIZE_KEY, hidden_size),
        (DECODER_NUM_HIDDEN_LAYERS_KEY, num_layers),
        (DECODER_NUM_ATTENTION_HEADS_KEY, num_heads),
        (DECODER_NUM_KEY_VALUE_HEADS_KEY, num_kv_heads),
        (DECODER_HEAD_DIM_KEY, head_dim),
        (DECODER_INTERMEDIATE_SIZE_KEY, intermediate_size),
        (DECODER_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    // GQA: every kv head is shared across `num_heads / num_kv_heads` query
    // heads, so the kv head count must divide the query head count (same
    // invariant qwen/moss/funasr-nano enforce for their Qwen-class decoders).
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: DECODER_NUM_KEY_VALUE_HEADS_KEY,
            reason: format!(
                "num_attention_heads {num_heads} is not a multiple of num_key_value_heads {num_kv_heads}"
            ),
        });
    }
    // The decode driver stops on the EOT token id; a vocab that cannot
    // represent it would decode to the generation cap and truncate.
    if (GRANITE_SPEECH_EOT_TOKEN_ID as usize) >= vocab_size {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: DECODER_VOCAB_SIZE_KEY,
            reason: format!(
                "vocab_size {vocab_size} cannot represent the decode stop token id {GRANITE_SPEECH_EOT_TOKEN_ID}"
            ),
        });
    }
    let rms_norm_eps = required_f32_scalar(metadata, DECODER_RMS_NORM_EPS_KEY)?;
    let rope_theta = required_f32_scalar(metadata, DECODER_ROPE_THETA_KEY)?;
    let attention_multiplier = required_f32_scalar(metadata, DECODER_ATTENTION_MULTIPLIER_KEY)?;
    let embedding_multiplier = required_f32_scalar(metadata, DECODER_EMBEDDING_MULTIPLIER_KEY)?;
    let residual_multiplier = required_f32_scalar(metadata, DECODER_RESIDUAL_MULTIPLIER_KEY)?;
    let logits_scaling = required_f32_scalar(metadata, DECODER_LOGITS_SCALING_KEY)?;
    Ok(GraniteSpeechDecoderConfig {
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        attention_multiplier,
        embedding_multiplier,
        residual_multiplier,
        logits_scaling,
    })
}

/// The complete granite-speech execution metadata: route identity plus the
/// three stages and the cross-stage geometry bridges between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraniteSpeechExecutionMetadata {
    pub encoder: GraniteSpeechEncoderConfig,
    pub projector: GraniteSpeechProjectorConfig,
    pub decoder: GraniteSpeechDecoderConfig,
    pub audio_token_index: u32,
}

pub(crate) fn parse_granite_speech_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<GraniteSpeechExecutionMetadata, GraniteSpeechRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)?;
    if architecture != GRANITE_SPEECH_GGML_ARCHITECTURE_ID {
        return Err(GraniteSpeechRuntimeContractError::UnexpectedArchitecture {
            expected: GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            found: architecture.to_string(),
        });
    }
    let encoder = parse_encoder_metadata(metadata)?;
    let projector = parse_projector_metadata(metadata)?;
    let decoder = parse_decoder_metadata(metadata)?;
    let audio_token_index = u32_key(metadata, AUDIO_TOKEN_INDEX_KEY)?;

    // The Q-Former cross-attends to the encoder's final hidden state
    // (`encoder_out` carries `hidden_dim` rows into `project`), so its model
    // width must equal the encoder hidden size -- a hand-edited header cannot
    // admit a bridge the graphs will not construct.
    if projector.encoder_hidden_size != encoder.hidden_dim {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: PROJECTOR_ENCODER_HIDDEN_SIZE_KEY,
            reason: format!(
                "{PROJECTOR_ENCODER_HIDDEN_SIZE_KEY}={} must equal {ENCODER_HIDDEN_DIM_KEY}={}",
                projector.encoder_hidden_size, encoder.hidden_dim
            ),
        });
    }
    // The projector writes `llm_hidden_size`-wide audio rows into the decoder
    // prompt, and the pack declares the audio placeholder id the splice looks
    // up; that id must fall inside the vocab the decoder embeds.
    if (audio_token_index as usize) >= decoder.vocab_size {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: AUDIO_TOKEN_INDEX_KEY,
            reason: format!(
                "audio_token_index {audio_token_index} out of range for vocab_size {}",
                decoder.vocab_size
            ),
        });
    }
    Ok(GraniteSpeechExecutionMetadata {
        encoder,
        projector,
        decoder,
        audio_token_index,
    })
}

/// Metadata-derived tensor binding contract for the complete granite-speech
/// runtime tensor set: the Conformer CTC encoder (`encoder.*`), the Q-Former
/// projector (`projector.*`, packed under the `projector.qf.{i}.` shortening
/// of the 63-byte ggml name cap), and the Granite decoder
/// (`language_model.*`). Requirements reference the parsed metadata only, the
/// same "shapes the pack itself declares" policy
/// `moss_transcribe_diarize::runtime_contract` uses. Rank-2 weights pin the
/// ggml `[in, out]` orientation the importer writes and the zero-copy
/// keep-quantized bindings consume without a repack.
pub(crate) fn granite_speech_runtime_tensor_descriptors(
    metadata: GraniteSpeechExecutionMetadata,
) -> Result<Vec<TensorBindingDescriptor>, GraniteSpeechRuntimeContractError> {
    let encoder = metadata.encoder;
    let projector = metadata.projector;
    let decoder = metadata.decoder;

    let inner_attn_dim = encoder
        .num_heads
        .checked_mul(encoder.dim_head)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_DIM_HEAD_KEY,
            reason: format!(
                "num_heads {} * dim_head {} overflows while deriving the inner attention width",
                encoder.num_heads, encoder.dim_head
            ),
        })?;
    let ffn_dim = encoder
        .hidden_dim
        .checked_mul(encoder.feedforward_mult)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_FEEDFORWARD_MULT_KEY,
            reason: format!(
                "hidden_dim {} * feedforward_mult {} overflows while deriving the encoder FFN width",
                encoder.hidden_dim, encoder.feedforward_mult
            ),
        })?;
    let conv_inner_dim = encoder
        .hidden_dim
        .checked_mul(encoder.conv_expansion_factor)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_CONV_EXPANSION_FACTOR_KEY,
            reason: format!(
                "hidden_dim {} * conv_expansion_factor {} overflows while deriving the conv module width",
                encoder.hidden_dim, encoder.conv_expansion_factor
            ),
        })?;
    let rel_pos_rows = encoder
        .max_pos_emb
        .checked_mul(2)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: ENCODER_MAX_POS_EMB_KEY,
            reason: format!(
                "2 * max_pos_emb {} + 1 overflows while deriving the relative-position table rows",
                encoder.max_pos_emb
            ),
        })?;
    let q_width = decoder
        .num_heads
        .checked_mul(decoder.head_dim)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: DECODER_HEAD_DIM_KEY,
            reason: format!(
                "num_attention_heads {} * head_dim {} overflows while deriving the q projection width",
                decoder.num_heads, decoder.head_dim
            ),
        })?;
    let kv_width = decoder
        .num_kv_heads
        .checked_mul(decoder.head_dim)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: DECODER_HEAD_DIM_KEY,
            reason: format!(
                "num_key_value_heads {} * head_dim {} overflows while deriving the k/v projection width",
                decoder.num_kv_heads, decoder.head_dim
            ),
        })?;
    let num_queries = projector.window_size / projector.downsample_rate;

    fn exact(name: String, dims: Vec<usize>, reason: &str) -> TensorBindingDescriptor {
        TensorBindingDescriptor {
            tensor_name: name,
            requirement: TensorBindingDescriptorRequirement::ExactDims(dims),
            reason: reason.to_string(),
        }
    }
    fn vector(name: String, len: usize, reason: &str) -> TensorBindingDescriptor {
        TensorBindingDescriptor {
            tensor_name: name,
            requirement: TensorBindingDescriptorRequirement::VectorLen(len),
            reason: reason.to_string(),
        }
    }

    let mut descriptors = Vec::new();

    // --- Conformer encoder ---------------------------------------------------
    descriptors.extend([
        exact(
            "encoder.input_linear.weight".to_string(),
            vec![encoder.input_dim, encoder.hidden_dim],
            "expected the frame-stacking input projection from input_dim to the encoder hidden size",
        ),
        vector(
            "encoder.input_linear.bias".to_string(),
            encoder.hidden_dim,
            "expected the input projection bias with the encoder hidden size",
        ),
    ]);
    for layer in 0..encoder.num_layers {
        let p = |suffix: &str| format!("encoder.layers.{layer}.{suffix}");
        descriptors.extend([
            vector(p("ff1.pre_norm.weight"), encoder.hidden_dim, "expected an encoder hidden-size norm vector"),
            vector(p("ff1.pre_norm.bias"), encoder.hidden_dim, "expected an encoder hidden-size norm bias"),
            exact(p("ff1.up_proj.weight"), vec![encoder.hidden_dim, ffn_dim], "expected the FFN up projection from hidden to FFN width"),
            vector(p("ff1.up_proj.bias"), ffn_dim, "expected an FFN-width bias"),
            exact(p("ff1.down_proj.weight"), vec![ffn_dim, encoder.hidden_dim], "expected the FFN down projection from FFN to hidden width"),
            vector(p("ff1.down_proj.bias"), encoder.hidden_dim, "expected an encoder hidden-size bias"),
            vector(p("attn.pre_norm.weight"), encoder.hidden_dim, "expected an encoder hidden-size norm vector"),
            vector(p("attn.pre_norm.bias"), encoder.hidden_dim, "expected an encoder hidden-size norm bias"),
            exact(p("attn.to_q.weight"), vec![encoder.hidden_dim, inner_attn_dim], "expected the query projection from hidden to the inner attention width"),
            exact(p("attn.to_kv.weight"), vec![encoder.hidden_dim, 2 * inner_attn_dim], "expected the fused key/value projection from hidden to twice the inner attention width"),
            exact(p("attn.to_out.weight"), vec![inner_attn_dim, encoder.hidden_dim], "expected the attention output projection from inner attention to hidden width"),
            vector(p("attn.to_out.bias"), encoder.hidden_dim, "expected an encoder hidden-size attention output bias"),
            exact(p("attn.rel_pos_emb.weight"), vec![encoder.dim_head, rel_pos_rows], "expected the Shaw relative-position table with 2*max_pos_emb+1 rows of dim_head"),
            vector(p("conv.norm.weight"), encoder.hidden_dim, "expected an encoder hidden-size norm vector"),
            vector(p("conv.norm.bias"), encoder.hidden_dim, "expected an encoder hidden-size norm bias"),
            exact(p("conv.up_conv.weight"), vec![encoder.hidden_dim, 2 * conv_inner_dim], "expected the GLU up convolution from hidden to twice the conv module width"),
            vector(p("conv.up_conv.bias"), 2 * conv_inner_dim, "expected a GLU up convolution bias with twice the conv module width"),
            exact(p("conv.depth_conv.conv.weight"), vec![conv_inner_dim, 1, encoder.conv_kernel_size], "expected the depthwise temporal conv kernel in source [channels, 1, kernel] order"),
            vector(p("conv.batch_norm.weight"), conv_inner_dim, "expected the BatchNorm gamma over the conv module width"),
            vector(p("conv.batch_norm.bias"), conv_inner_dim, "expected the BatchNorm beta over the conv module width"),
            vector(p("conv.batch_norm.running_mean"), conv_inner_dim, "expected the BatchNorm running mean over the conv module width"),
            vector(p("conv.batch_norm.running_var"), conv_inner_dim, "expected the BatchNorm running variance over the conv module width"),
            exact(p("conv.down_conv.weight"), vec![conv_inner_dim, encoder.hidden_dim], "expected the conv down projection from conv module to hidden width"),
            vector(p("conv.down_conv.bias"), encoder.hidden_dim, "expected an encoder hidden-size conv down bias"),
            vector(p("ff2.pre_norm.weight"), encoder.hidden_dim, "expected an encoder hidden-size norm vector"),
            vector(p("ff2.pre_norm.bias"), encoder.hidden_dim, "expected an encoder hidden-size norm bias"),
            exact(p("ff2.up_proj.weight"), vec![encoder.hidden_dim, ffn_dim], "expected the FFN up projection from hidden to FFN width"),
            vector(p("ff2.up_proj.bias"), ffn_dim, "expected an FFN-width bias"),
            exact(p("ff2.down_proj.weight"), vec![ffn_dim, encoder.hidden_dim], "expected the FFN down projection from FFN to hidden width"),
            vector(p("ff2.down_proj.bias"), encoder.hidden_dim, "expected an encoder hidden-size bias"),
            vector(p("post_norm.weight"), encoder.hidden_dim, "expected an encoder hidden-size norm vector"),
            vector(p("post_norm.bias"), encoder.hidden_dim, "expected an encoder hidden-size norm bias"),
        ]);
    }
    descriptors.extend([
        exact(
            "encoder.out.weight".to_string(),
            vec![encoder.hidden_dim, encoder.output_dim],
            "expected the CTC output projection from hidden to the CTC vocabulary width",
        ),
        vector(
            "encoder.out.bias".to_string(),
            encoder.output_dim,
            "expected a CTC vocabulary-width bias",
        ),
        exact(
            "encoder.out_mid.weight".to_string(),
            vec![encoder.output_dim, encoder.hidden_dim],
            "expected the self-conditioned CTC tap projection from CTC vocabulary to hidden width",
        ),
        vector(
            "encoder.out_mid.bias".to_string(),
            encoder.hidden_dim,
            "expected an encoder hidden-size CTC tap bias",
        ),
    ]);

    // --- Q-Former projector --------------------------------------------------
    descriptors.extend([
        exact(
            "projector.query".to_string(),
            vec![1, num_queries, projector.encoder_hidden_size],
            "expected the learned Q-Former query in source [1, num_queries, encoder_hidden] order",
        ),
        vector(
            "projector.qformer.layernorm.weight".to_string(),
            projector.encoder_hidden_size,
            "expected the Q-Former input LayerNorm weight over the encoder hidden size",
        ),
        vector(
            "projector.qformer.layernorm.bias".to_string(),
            projector.encoder_hidden_size,
            "expected the Q-Former input LayerNorm bias over the encoder hidden size",
        ),
    ]);
    for layer in 0..projector.num_hidden_layers {
        // Packed under the shortened `projector.qf.{i}.` prefix: the full
        // `projector.qformer.encoder.layer.{i}.` names overflow ggml's
        // 63-byte tensor-name cap (see `package_import::remap_tensor_name`).
        let p = |suffix: &str| format!("projector.qf.{layer}.{suffix}");
        let d = projector.encoder_hidden_size;
        descriptors.extend([
            exact(p("attention.attention.query.weight"), vec![d, d], "expected a Q-Former self-attention query projection over the encoder hidden size"),
            vector(p("attention.attention.query.bias"), d, "expected a Q-Former self-attention query bias"),
            exact(p("attention.attention.key.weight"), vec![d, d], "expected a Q-Former self-attention key projection over the encoder hidden size"),
            vector(p("attention.attention.key.bias"), d, "expected a Q-Former self-attention key bias"),
            exact(p("attention.attention.value.weight"), vec![d, d], "expected a Q-Former self-attention value projection over the encoder hidden size"),
            vector(p("attention.attention.value.bias"), d, "expected a Q-Former self-attention value bias"),
            exact(p("attention.output.dense.weight"), vec![d, d], "expected a Q-Former self-attention output projection over the encoder hidden size"),
            vector(p("attention.output.dense.bias"), d, "expected a Q-Former self-attention output bias"),
            vector(p("attention.output.LayerNorm.weight"), d, "expected a Q-Former self-attention output LayerNorm weight"),
            vector(p("attention.output.LayerNorm.bias"), d, "expected a Q-Former self-attention output LayerNorm bias"),
            exact(p("crossattention.attention.query.weight"), vec![d, d], "expected a Q-Former cross-attention query projection over the encoder hidden size"),
            vector(p("crossattention.attention.query.bias"), d, "expected a Q-Former cross-attention query bias"),
            exact(p("crossattention.attention.key.weight"), vec![d, d], "expected a Q-Former cross-attention key projection over the encoder hidden size"),
            vector(p("crossattention.attention.key.bias"), d, "expected a Q-Former cross-attention key bias"),
            exact(p("crossattention.attention.value.weight"), vec![d, d], "expected a Q-Former cross-attention value projection over the encoder hidden size"),
            vector(p("crossattention.attention.value.bias"), d, "expected a Q-Former cross-attention value bias"),
            exact(p("crossattention.output.dense.weight"), vec![d, d], "expected a Q-Former cross-attention output projection over the encoder hidden size"),
            vector(p("crossattention.output.dense.bias"), d, "expected a Q-Former cross-attention output bias"),
            vector(p("crossattention.output.LayerNorm.weight"), d, "expected a Q-Former cross-attention output LayerNorm weight"),
            vector(p("crossattention.output.LayerNorm.bias"), d, "expected a Q-Former cross-attention output LayerNorm bias"),
            exact(p("intermediate_query.dense.weight"), vec![d, projector.intermediate_size], "expected the Q-Former FFN up projection to the intermediate size"),
            vector(p("intermediate_query.dense.bias"), projector.intermediate_size, "expected a Q-Former FFN intermediate-width bias"),
            exact(p("output_query.dense.weight"), vec![projector.intermediate_size, d], "expected the Q-Former FFN down projection from the intermediate size"),
            vector(p("output_query.dense.bias"), d, "expected a Q-Former FFN output bias over the encoder hidden size"),
            vector(p("output_query.LayerNorm.weight"), d, "expected a Q-Former FFN output LayerNorm weight"),
            vector(p("output_query.LayerNorm.bias"), d, "expected a Q-Former FFN output LayerNorm bias"),
        ]);
    }
    descriptors.extend([
        exact(
            "projector.linear.weight".to_string(),
            vec![projector.encoder_hidden_size, projector.llm_hidden_size],
            "expected the projector output projection from encoder hidden to LLM hidden width",
        ),
        vector(
            "projector.linear.bias".to_string(),
            projector.llm_hidden_size,
            "expected a projector output bias with the LLM hidden size",
        ),
    ]);

    // --- Granite decoder -----------------------------------------------------
    descriptors.extend([
        exact(
            "language_model.model.embed_tokens.weight".to_string(),
            vec![decoder.hidden_size, decoder.vocab_size],
            "expected the token embedding table with hidden size and vocab dimensions",
        ),
        vector(
            "language_model.model.norm.weight".to_string(),
            decoder.hidden_size,
            "expected the final RMSNorm weight with the decoder hidden size",
        ),
        exact(
            "language_model.lm_head.weight".to_string(),
            vec![decoder.hidden_size, decoder.vocab_size],
            "expected the logits head with hidden size and vocab dimensions",
        ),
    ]);
    for layer in 0..decoder.num_layers {
        let p = |suffix: &str| format!("language_model.model.layers.{layer}.{suffix}");
        descriptors.extend([
            vector(
                p("input_layernorm.weight"),
                decoder.hidden_size,
                "expected a decoder hidden-size RMSNorm weight",
            ),
            exact(
                p("self_attn.q_proj.weight"),
                vec![decoder.hidden_size, q_width],
                "expected the query projection from hidden to the q width",
            ),
            exact(
                p("self_attn.k_proj.weight"),
                vec![decoder.hidden_size, kv_width],
                "expected the key projection from hidden to the kv width",
            ),
            exact(
                p("self_attn.v_proj.weight"),
                vec![decoder.hidden_size, kv_width],
                "expected the value projection from hidden to the kv width",
            ),
            exact(
                p("self_attn.o_proj.weight"),
                vec![q_width, decoder.hidden_size],
                "expected the attention output projection from q width to hidden",
            ),
            vector(
                p("post_attention_layernorm.weight"),
                decoder.hidden_size,
                "expected a decoder hidden-size RMSNorm weight",
            ),
            exact(
                p("mlp.gate_proj.weight"),
                vec![decoder.hidden_size, decoder.intermediate_size],
                "expected the SwiGLU gate projection from hidden to intermediate width",
            ),
            exact(
                p("mlp.up_proj.weight"),
                vec![decoder.hidden_size, decoder.intermediate_size],
                "expected the SwiGLU up projection from hidden to intermediate width",
            ),
            exact(
                p("mlp.down_proj.weight"),
                vec![decoder.intermediate_size, decoder.hidden_size],
                "expected the SwiGLU down projection from intermediate to hidden width",
            ),
        ]);
    }

    Ok(descriptors)
}

/// Validate the pack's tensor set against the metadata-derived binding
/// contract. Runs after [`parse_granite_speech_execution_metadata`] succeeds;
/// a missing tensor or a shape the declared geometry cannot construct fails
/// closed with the offending tensor named.
pub(crate) fn validate_granite_speech_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: GraniteSpeechExecutionMetadata,
) -> Result<(), GraniteSpeechRuntimeContractError> {
    let descriptors = granite_speech_runtime_tensor_descriptors(metadata)?;
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )?;
    Ok(())
}

fn missing_required_tensor(name: &str) -> GraniteSpeechRuntimeContractError {
    GraniteSpeechRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> GraniteSpeechRuntimeContractError {
    GraniteSpeechRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

/// Validate the packed GPT-2 BPE tokenizer the executor materializes from
/// `tokenizer.ggml.*` metadata: it must parse through the production
/// constructor, and the dense token table must actually carry the audio
/// placeholder at the pack's declared index -- a drift between the declared
/// `audio_token_index` and the baked token table would break the prompt
/// splice at decode time, so it fails closed here at admission.
fn validate_granite_speech_packed_tokenizer(
    preflight: &crate::GgufRuntimeSourcePreflight,
    audio_token_index: u32,
) -> Result<(), GraniteSpeechRuntimeContractError> {
    let tokens_key = super::package_import::TOKENIZER_GGML_TOKENS_KEY;
    super::tokenizer::GraniteSpeechTokenizer::from_gguf_metadata(preflight.metadata()).map_err(
        |error| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: super::package_import::TOKENIZER_GGML_MODEL_KEY,
            reason: error.to_string(),
        },
    )?;
    let tokens = crate::models::oasr_metadata::required_metadata_string_array(
        preflight.metadata(),
        tokens_key,
        GRANITE_SPEECH_CONTRACT_FAMILY,
    )
    .map_err(
        |error| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: tokens_key,
            reason: error.to_string(),
        },
    )?;
    if (GRANITE_SPEECH_EOT_TOKEN_ID as usize) >= tokens.len() {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: tokens_key,
            reason: format!(
                "packed tokenizer carries {} tokens, too few to represent the decode stop token id {GRANITE_SPEECH_EOT_TOKEN_ID}",
                tokens.len()
            ),
        });
    }
    let audio_token = tokens
        .get(audio_token_index as usize)
        .ok_or_else(|| GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: AUDIO_TOKEN_INDEX_KEY,
            reason: format!(
                "audio_token_index {audio_token_index} is out of range for the {}-token packed table",
                tokens.len()
            ),
        })?;
    if audio_token != GRANITE_SPEECH_AUDIO_TOKEN {
        return Err(GraniteSpeechRuntimeContractError::InvalidMetadataValue {
            key: AUDIO_TOKEN_INDEX_KEY,
            reason: format!(
                "packed token at audio_token_index {audio_token_index} is '{audio_token}', expected '{GRANITE_SPEECH_AUDIO_TOKEN}'"
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata =
        parse_granite_speech_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                GRANITE_SPEECH_CONTRACT_FAMILY,
                error,
            )
        })?;
    validate_granite_speech_runtime_tensors_with_index(preflight.tensor_index(), metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)?;
    validate_granite_speech_packed_tokenizer(preflight, metadata.audio_token_index).map_err(
        |error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                "granite-speech tokenizer",
                error,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufTensorMetadata;
    use crate::ggml_runtime::GgufTensorIndexSnapshot;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Every execution metadata key the shipped
    /// `ibm-granite/granite-speech-4.1-2b` checkpoint writes (values mirror
    /// the `granite_speech_4_1_2b()` config constructors), as the stringified
    /// GGUF metadata view the importer produces.
    fn full_metadata() -> BTreeMap<String, String> {
        [
            (
                GENERAL_ARCHITECTURE_KEY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (AUDIO_TOKEN_INDEX_KEY, "100352"),
            (DOWNSAMPLE_RATE_KEY, "5"),
            (WINDOW_SIZE_KEY, "15"),
            (ENCODER_INPUT_DIM_KEY, "160"),
            (ENCODER_HIDDEN_DIM_KEY, "1024"),
            (ENCODER_NUM_LAYERS_KEY, "16"),
            (ENCODER_NUM_HEADS_KEY, "8"),
            (ENCODER_DIM_HEAD_KEY, "128"),
            (ENCODER_FEEDFORWARD_MULT_KEY, "4"),
            (ENCODER_CONV_KERNEL_SIZE_KEY, "15"),
            (ENCODER_CONV_EXPANSION_FACTOR_KEY, "2"),
            (ENCODER_CONTEXT_SIZE_KEY, "200"),
            (ENCODER_MAX_POS_EMB_KEY, "512"),
            (ENCODER_OUTPUT_DIM_KEY, "348"),
            (PROJECTOR_ENCODER_HIDDEN_SIZE_KEY, "1024"),
            (PROJECTOR_NUM_HIDDEN_LAYERS_KEY, "2"),
            (PROJECTOR_NUM_ATTENTION_HEADS_KEY, "16"),
            (PROJECTOR_INTERMEDIATE_SIZE_KEY, "4096"),
            (DECODER_HIDDEN_SIZE_KEY, "2048"),
            (DECODER_NUM_HIDDEN_LAYERS_KEY, "40"),
            (DECODER_NUM_ATTENTION_HEADS_KEY, "16"),
            (DECODER_HEAD_DIM_KEY, "128"),
            (DECODER_NUM_KEY_VALUE_HEADS_KEY, "4"),
            (DECODER_INTERMEDIATE_SIZE_KEY, "4096"),
            (DECODER_VOCAB_SIZE_KEY, "100353"),
            (DECODER_RMS_NORM_EPS_KEY, "0.00001"),
            (DECODER_ROPE_THETA_KEY, "10000"),
            (DECODER_ATTENTION_MULTIPLIER_KEY, "0.0078125"),
            (DECODER_EMBEDDING_MULTIPLIER_KEY, "12"),
            (DECODER_RESIDUAL_MULTIPLIER_KEY, "0.22"),
            (DECODER_LOGITS_SCALING_KEY, "8"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Tiny internally-consistent geometry for tensor-level tests: one
    /// encoder layer, one projector layer, one decoder layer, small widths.
    /// Every metadata invariant holds (odd conv kernel, max_pos_emb >=
    /// context_size, window_size % downsample_rate == 0, head divisibility,
    /// encoder hidden == projector encoder_hidden, token ids in vocab).
    fn tiny_metadata() -> BTreeMap<String, String> {
        [
            (
                GENERAL_ARCHITECTURE_KEY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (AUDIO_TOKEN_INDEX_KEY, "41"),
            (DOWNSAMPLE_RATE_KEY, "2"),
            (WINDOW_SIZE_KEY, "4"),
            (ENCODER_INPUT_DIM_KEY, "8"),
            (ENCODER_HIDDEN_DIM_KEY, "16"),
            (ENCODER_NUM_LAYERS_KEY, "1"),
            (ENCODER_NUM_HEADS_KEY, "2"),
            (ENCODER_DIM_HEAD_KEY, "8"),
            (ENCODER_FEEDFORWARD_MULT_KEY, "2"),
            (ENCODER_CONV_KERNEL_SIZE_KEY, "3"),
            (ENCODER_CONV_EXPANSION_FACTOR_KEY, "2"),
            (ENCODER_CONTEXT_SIZE_KEY, "4"),
            (ENCODER_MAX_POS_EMB_KEY, "8"),
            (ENCODER_OUTPUT_DIM_KEY, "12"),
            (PROJECTOR_ENCODER_HIDDEN_SIZE_KEY, "16"),
            (PROJECTOR_NUM_HIDDEN_LAYERS_KEY, "1"),
            (PROJECTOR_NUM_ATTENTION_HEADS_KEY, "2"),
            (PROJECTOR_INTERMEDIATE_SIZE_KEY, "32"),
            (DECODER_HIDDEN_SIZE_KEY, "16"),
            (DECODER_NUM_HIDDEN_LAYERS_KEY, "1"),
            (DECODER_NUM_ATTENTION_HEADS_KEY, "2"),
            (DECODER_HEAD_DIM_KEY, "8"),
            (DECODER_NUM_KEY_VALUE_HEADS_KEY, "1"),
            (DECODER_INTERMEDIATE_SIZE_KEY, "32"),
            // Must stay above the decode stop token id (100257) the contract
            // proves representable, even in the tiny geometry.
            (DECODER_VOCAB_SIZE_KEY, "100300"),
            (DECODER_RMS_NORM_EPS_KEY, "0.00001"),
            (DECODER_ROPE_THETA_KEY, "10000"),
            (DECODER_ATTENTION_MULTIPLIER_KEY, "1"),
            (DECODER_EMBEDDING_MULTIPLIER_KEY, "1"),
            (DECODER_RESIDUAL_MULTIPLIER_KEY, "1"),
            (DECODER_LOGITS_SCALING_KEY, "1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Every tensor the tiny geometry declares, with the shapes the importer
    /// writes for them (ggml `[in, out]` for rank-2 weights, source extent
    /// order for the rank-3 kernel/query).
    fn tiny_tensor_shapes() -> Vec<(String, Vec<u64>)> {
        let d: u64 = 16; // encoder + projector + decoder hidden size (tiny)
        let ffn: u64 = 32; // encoder FFN width (16 * 2)
        let inner: u64 = 16; // encoder inner attention (2 heads * 8 dim_head)
        let conv_inner: u64 = 32; // conv module width (16 * 2)
        let rel_pos_rows: u64 = 17; // 2 * max_pos_emb(8) + 1
        let q_width: u64 = 16; // decoder 2 heads * 8 head_dim
        let kv_width: u64 = 8; // decoder 1 kv head * 8 head_dim
        let inter: u64 = 32; // projector + decoder intermediate width
        let vocab: u64 = 100_300; // above the decode stop token id
        vec![
            ("encoder.input_linear.weight".to_string(), vec![8, d]),
            ("encoder.input_linear.bias".to_string(), vec![d]),
            ("encoder.layers.0.ff1.pre_norm.weight".to_string(), vec![d]),
            ("encoder.layers.0.ff1.pre_norm.bias".to_string(), vec![d]),
            (
                "encoder.layers.0.ff1.up_proj.weight".to_string(),
                vec![d, ffn],
            ),
            ("encoder.layers.0.ff1.up_proj.bias".to_string(), vec![ffn]),
            (
                "encoder.layers.0.ff1.down_proj.weight".to_string(),
                vec![ffn, d],
            ),
            ("encoder.layers.0.ff1.down_proj.bias".to_string(), vec![d]),
            ("encoder.layers.0.attn.pre_norm.weight".to_string(), vec![d]),
            ("encoder.layers.0.attn.pre_norm.bias".to_string(), vec![d]),
            (
                "encoder.layers.0.attn.to_q.weight".to_string(),
                vec![d, inner],
            ),
            (
                "encoder.layers.0.attn.to_kv.weight".to_string(),
                vec![d, 2 * inner],
            ),
            (
                "encoder.layers.0.attn.to_out.weight".to_string(),
                vec![inner, d],
            ),
            ("encoder.layers.0.attn.to_out.bias".to_string(), vec![d]),
            (
                "encoder.layers.0.attn.rel_pos_emb.weight".to_string(),
                vec![8, rel_pos_rows],
            ),
            ("encoder.layers.0.conv.norm.weight".to_string(), vec![d]),
            ("encoder.layers.0.conv.norm.bias".to_string(), vec![d]),
            (
                "encoder.layers.0.conv.up_conv.weight".to_string(),
                vec![d, 2 * conv_inner],
            ),
            (
                "encoder.layers.0.conv.up_conv.bias".to_string(),
                vec![2 * conv_inner],
            ),
            (
                "encoder.layers.0.conv.depth_conv.conv.weight".to_string(),
                vec![conv_inner, 1, 3],
            ),
            (
                "encoder.layers.0.conv.batch_norm.weight".to_string(),
                vec![conv_inner],
            ),
            (
                "encoder.layers.0.conv.batch_norm.bias".to_string(),
                vec![conv_inner],
            ),
            (
                "encoder.layers.0.conv.batch_norm.running_mean".to_string(),
                vec![conv_inner],
            ),
            (
                "encoder.layers.0.conv.batch_norm.running_var".to_string(),
                vec![conv_inner],
            ),
            (
                "encoder.layers.0.conv.down_conv.weight".to_string(),
                vec![conv_inner, d],
            ),
            ("encoder.layers.0.conv.down_conv.bias".to_string(), vec![d]),
            ("encoder.layers.0.ff2.pre_norm.weight".to_string(), vec![d]),
            ("encoder.layers.0.ff2.pre_norm.bias".to_string(), vec![d]),
            (
                "encoder.layers.0.ff2.up_proj.weight".to_string(),
                vec![d, ffn],
            ),
            ("encoder.layers.0.ff2.up_proj.bias".to_string(), vec![ffn]),
            (
                "encoder.layers.0.ff2.down_proj.weight".to_string(),
                vec![ffn, d],
            ),
            ("encoder.layers.0.ff2.down_proj.bias".to_string(), vec![d]),
            ("encoder.layers.0.post_norm.weight".to_string(), vec![d]),
            ("encoder.layers.0.post_norm.bias".to_string(), vec![d]),
            ("encoder.out.weight".to_string(), vec![d, 12]),
            ("encoder.out.bias".to_string(), vec![12]),
            ("encoder.out_mid.weight".to_string(), vec![12, d]),
            ("encoder.out_mid.bias".to_string(), vec![d]),
            ("projector.query".to_string(), vec![1, 2, d]),
            ("projector.qformer.layernorm.weight".to_string(), vec![d]),
            ("projector.qformer.layernorm.bias".to_string(), vec![d]),
            (
                "projector.qf.0.attention.attention.query.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.attention.attention.query.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.attention.attention.key.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.attention.attention.key.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.attention.attention.value.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.attention.attention.value.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.attention.output.dense.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.attention.output.dense.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.attention.output.LayerNorm.weight".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.attention.output.LayerNorm.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.attention.query.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.crossattention.attention.query.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.attention.key.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.crossattention.attention.key.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.attention.value.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.crossattention.attention.value.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.output.dense.weight".to_string(),
                vec![d, d],
            ),
            (
                "projector.qf.0.crossattention.output.dense.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.output.LayerNorm.weight".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.crossattention.output.LayerNorm.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.intermediate_query.dense.weight".to_string(),
                vec![d, inter],
            ),
            (
                "projector.qf.0.intermediate_query.dense.bias".to_string(),
                vec![inter],
            ),
            (
                "projector.qf.0.output_query.dense.weight".to_string(),
                vec![inter, d],
            ),
            (
                "projector.qf.0.output_query.dense.bias".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.output_query.LayerNorm.weight".to_string(),
                vec![d],
            ),
            (
                "projector.qf.0.output_query.LayerNorm.bias".to_string(),
                vec![d],
            ),
            ("projector.linear.weight".to_string(), vec![d, d]),
            ("projector.linear.bias".to_string(), vec![d]),
            (
                "language_model.model.embed_tokens.weight".to_string(),
                vec![d, vocab],
            ),
            ("language_model.model.norm.weight".to_string(), vec![d]),
            ("language_model.lm_head.weight".to_string(), vec![d, vocab]),
            (
                "language_model.model.layers.0.input_layernorm.weight".to_string(),
                vec![d],
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![d, q_width],
            ),
            (
                "language_model.model.layers.0.self_attn.k_proj.weight".to_string(),
                vec![d, kv_width],
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.weight".to_string(),
                vec![d, kv_width],
            ),
            (
                "language_model.model.layers.0.self_attn.o_proj.weight".to_string(),
                vec![q_width, d],
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight".to_string(),
                vec![d],
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight".to_string(),
                vec![d, inter],
            ),
            (
                "language_model.model.layers.0.mlp.up_proj.weight".to_string(),
                vec![d, inter],
            ),
            (
                "language_model.model.layers.0.mlp.down_proj.weight".to_string(),
                vec![inter, d],
            ),
        ]
    }

    fn tensor_index_from_shapes(shapes: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let tensors = shapes
            .iter()
            .enumerate()
            .map(|(index, (name, dims))| GgufTensorMetadata {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(GgufTensorIndexSnapshot {
            path: PathBuf::from("/tmp/granite-speech-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    #[test]
    fn parses_encoder_metadata_matching_real_checkpoint() {
        let parsed = parse_encoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.hidden_dim, 1024);
        assert_eq!(parsed.num_layers, 16);
        assert_eq!(parsed.context_size, 200);
        assert_eq!(parsed.output_dim, 348);
    }

    #[test]
    fn parses_projector_metadata_matching_real_checkpoint() {
        let parsed = parse_projector_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.encoder_hidden_size, 1024);
        assert_eq!(parsed.llm_hidden_size, 2048);
        assert_eq!(parsed.num_hidden_layers, 2);
    }

    #[test]
    fn parses_decoder_metadata_matching_real_checkpoint() {
        let parsed = parse_decoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.num_layers, 40);
        assert_eq!(parsed.num_kv_heads, 4);
        assert_eq!(parsed.vocab_size, 100_353);
        assert_eq!(parsed.logits_scaling, 8.0);
    }

    #[test]
    fn parses_execution_metadata_with_route_identity_and_bridges() {
        let parsed = parse_granite_speech_execution_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.encoder.num_layers, 16);
        assert_eq!(parsed.projector.window_size, 15);
        assert_eq!(parsed.decoder.hidden_size, 2048);
        assert_eq!(parsed.audio_token_index, 100_352);
    }

    #[test]
    fn rejects_a_foreign_general_architecture() {
        let mut metadata = full_metadata();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            "some-other-architecture".to_string(),
        );
        let error = parse_granite_speech_execution_metadata(&metadata)
            .expect_err("foreign architecture must fail closed");
        assert!(matches!(
            error,
            GraniteSpeechRuntimeContractError::UnexpectedArchitecture { .. }
        ));
    }

    #[test]
    fn rejects_an_encoder_projector_bridge_the_hidden_sizes_cannot_feed() {
        let mut metadata = full_metadata();
        metadata.insert(
            PROJECTOR_ENCODER_HIDDEN_SIZE_KEY.to_string(),
            "768".to_string(),
        );
        let error = parse_granite_speech_execution_metadata(&metadata)
            .expect_err("encoder->projector bridge mismatch must fail closed");
        assert!(matches!(
            error,
            GraniteSpeechRuntimeContractError::InvalidMetadataValue {
                key: PROJECTOR_ENCODER_HIDDEN_SIZE_KEY,
                ..
            }
        ));
    }

    #[test]
    fn rejects_even_conv_kernel() {
        let mut metadata = full_metadata();
        metadata.insert(ENCODER_CONV_KERNEL_SIZE_KEY.to_string(), "14".to_string());
        assert!(parse_encoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_context_blocks_wider_than_the_rel_pos_span() {
        let mut metadata = full_metadata();
        metadata.insert(ENCODER_MAX_POS_EMB_KEY.to_string(), "100".to_string());
        let error = parse_encoder_metadata(&metadata)
            .expect_err("max_pos_emb < context_size must fail closed");
        assert!(matches!(
            error,
            GraniteSpeechRuntimeContractError::InvalidMetadataValue {
                key: ENCODER_MAX_POS_EMB_KEY,
                ..
            }
        ));
    }

    #[test]
    fn rejects_a_window_the_downsample_rate_does_not_divide() {
        let mut metadata = full_metadata();
        metadata.insert(WINDOW_SIZE_KEY.to_string(), "14".to_string());
        assert!(parse_projector_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut metadata = full_metadata();
        metadata.insert(DECODER_NUM_KEY_VALUE_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_a_vocab_that_cannot_express_the_stop_token() {
        let mut metadata = full_metadata();
        metadata.insert(DECODER_VOCAB_SIZE_KEY.to_string(), "50000".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_an_audio_token_index_out_of_vocab() {
        let mut metadata = full_metadata();
        metadata.insert(AUDIO_TOKEN_INDEX_KEY.to_string(), "100999".to_string());
        let error = parse_granite_speech_execution_metadata(&metadata)
            .expect_err("audio token id out of vocab must fail closed");
        assert!(matches!(
            error,
            GraniteSpeechRuntimeContractError::InvalidMetadataValue {
                key: AUDIO_TOKEN_INDEX_KEY,
                ..
            }
        ));
    }

    /// The importer writes 938 runtime-required tensors for the real
    /// checkpoint geometry (518 encoder + 57 projector + 363 decoder). The
    /// metadata-derived descriptor set must name exactly that set: one
    /// descriptor per tensor, no duplicates, no omissions -- a drift in either
    /// direction means the validator either under-checks the pack or demands a
    /// tensor the importer never writes. (The importer additionally carries a
    /// top-level `lm_head.weight` when a source checkpoint exposes one outside
    /// `language_model.*`; the runtime never reads that defensive cargo, so it
    /// stays outside the required set.)
    #[test]
    fn tensor_contract_covers_every_runtime_required_tensor_exactly_once() {
        let metadata = parse_granite_speech_execution_metadata(&full_metadata()).expect("parse");
        let descriptors = granite_speech_runtime_tensor_descriptors(metadata).expect("descriptors");
        let mut names: Vec<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(names.len(), 938, "descriptor count");
        names.dedup();
        assert_eq!(names.len(), 938, "duplicate descriptor names");
        assert!(names.contains(&"encoder.input_linear.weight"));
        assert!(names.contains(&"encoder.layers.15.conv.batch_norm.running_var"));
        assert!(names.contains(&"projector.qf.1.output_query.LayerNorm.bias"));
        assert!(names.contains(&"language_model.model.embed_tokens.weight"));
        assert!(names.contains(&"language_model.model.layers.39.mlp.down_proj.weight"));
        assert!(names.contains(&"language_model.lm_head.weight"));
    }

    #[test]
    fn validates_the_tiny_reference_tensor_set() {
        let metadata = parse_granite_speech_execution_metadata(&tiny_metadata()).expect("parse");
        let index = tensor_index_from_shapes(&tiny_tensor_shapes());
        validate_granite_speech_runtime_tensors_with_index(&index, metadata)
            .expect("tiny tensor set");
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let metadata = parse_granite_speech_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        shapes.retain(|(name, _)| name != "projector.linear.weight");
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_granite_speech_runtime_tensors_with_index(&index, metadata)
            .expect_err("missing projector tensor must fail closed");
        match error {
            GraniteSpeechRuntimeContractError::MissingRequiredTensor { name } => {
                assert_eq!(name, "projector.linear.weight");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_a_decoder_projection_with_the_wrong_kv_width() {
        let metadata = parse_granite_speech_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        for (name, dims) in shapes.iter_mut() {
            if name == "language_model.model.layers.0.self_attn.k_proj.weight" {
                *dims = vec![16, 99];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_granite_speech_runtime_tensors_with_index(&index, metadata)
            .expect_err("k projection width mismatch must fail closed");
        match error {
            GraniteSpeechRuntimeContractError::InvalidTensorShape { name, .. } => {
                assert_eq!(
                    name,
                    "language_model.model.layers.0.self_attn.k_proj.weight"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_a_depthwise_conv_kernel_with_the_wrong_channel_band() {
        let metadata = parse_granite_speech_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        for (name, dims) in shapes.iter_mut() {
            if name == "encoder.layers.0.conv.depth_conv.conv.weight" {
                *dims = vec![31, 1, 3];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_granite_speech_runtime_tensors_with_index(&index, metadata)
            .expect_err("conv channel mismatch must fail closed");
        match error {
            GraniteSpeechRuntimeContractError::InvalidTensorShape { name, .. } => {
                assert_eq!(name, "encoder.layers.0.conv.depth_conv.conv.weight");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
