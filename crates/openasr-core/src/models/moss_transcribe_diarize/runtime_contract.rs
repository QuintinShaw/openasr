//! moss-transcribe-diarize execution contract against an admitted `.oasr`
//! runtime source: the family runtime validator parses every execution
//! metadata key the importer writes AND validates the complete tensor set
//! against the shapes those metadata declare (metadata + tensor depth, the
//! same contract shape `qwen::runtime_contract` and
//! `moonshine::runtime_contract` enforce). Key names match exactly what
//! `package_import` writes.
//!
//! The tensor descriptors are derived from the parsed metadata -- never from
//! a hardcoded checkpoint shape -- so a future legitimately-shaped pack is
//! admitted by its own declared geometry while a truncated, reshaped, or
//! mis-converted pack fails closed at admission with the offending tensor
//! named.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::arch::{GENERAL_ARCHITECTURE_KEY, MOSS_TD_GGML_ARCHITECTURE_ID};
use crate::capacity::decode_schedule::greedy_self_kv_positions;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

use super::encoder_graph::MOSS_ENCODER_FFN_EXPANSION;
use super::tensor_names::{
    ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
    ADAPTOR_NORM_BIAS, ADAPTOR_NORM_WEIGHT, ENC_CONV1_BIAS, ENC_CONV1_WEIGHT, ENC_CONV2_BIAS,
    ENC_CONV2_WEIGHT, ENC_OUT_NORM_BIAS, ENC_OUT_NORM_WEIGHT, ENC_POS_EMBD_WEIGHT,
    LLM_OUTPUT_NORM_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, moss_encoder_layer_tensor_names,
    moss_llm_layer_tensor_names,
};

pub(crate) const ENCODER_N_LAYERS_KEY: &str = "moss_td.encoder.n_layers";
pub(crate) const ENCODER_D_MODEL_KEY: &str = "moss_td.encoder.d_model";
pub(crate) const ENCODER_N_HEADS_KEY: &str = "moss_td.encoder.n_heads";
pub(crate) const ENCODER_FFN_DIM_KEY: &str = "moss_td.encoder.ffn_dim";
pub(crate) const ENCODER_N_MELS_KEY: &str = "moss_td.encoder.n_mels";
pub(crate) const ENCODER_MAX_SOURCE_POSITIONS_KEY: &str = "moss_td.encoder.max_source_positions";
pub(crate) const ADAPTOR_MERGE_SIZE_KEY: &str = "moss_td.adaptor.merge_size";
pub(crate) const ADAPTOR_INPUT_DIM_KEY: &str = "moss_td.adaptor.input_dim";
pub(crate) const LLM_N_LAYERS_KEY: &str = "moss_td.llm.n_layers";
pub(crate) const LLM_D_MODEL_KEY: &str = "moss_td.llm.d_model";
pub(crate) const LLM_FFN_DIM_KEY: &str = "moss_td.llm.ffn_dim";
pub(crate) const LLM_N_HEADS_KEY: &str = "moss_td.llm.n_heads";
pub(crate) const LLM_N_KV_HEADS_KEY: &str = "moss_td.llm.n_kv_heads";
pub(crate) const LLM_HEAD_DIM_KEY: &str = "moss_td.llm.head_dim";
pub(crate) const LLM_VOCAB_SIZE_KEY: &str = "moss_td.llm.vocab_size";
pub(crate) const LLM_MAX_POSITIONS_KEY: &str = "moss_td.llm.max_positions";
pub(crate) const LLM_AUDIO_START_TOKEN_ID_KEY: &str = "moss_td.llm.audio_start_token_id";
pub(crate) const LLM_AUDIO_END_TOKEN_ID_KEY: &str = "moss_td.llm.audio_end_token_id";
pub(crate) const LLM_AUDIO_PAD_TOKEN_ID_KEY: &str = "moss_td.llm.audio_pad_token_id";

/// The Whisper conv stem's kernel size. Both conv layers are kernel-3 (conv1
/// stride 1, conv2 stride 2), verified against upstream
/// `transformers.models.whisper.modeling_whisper.WhisperEncoder`; the importer
/// writes the HF `[out, in, kernel]` shapes reversed into ggml's
/// `[kernel, in, out]` order, which is exactly what the tensor contract below
/// pins.
const MOSS_TD_ENCODER_CONV_KERNEL: usize = 3;

/// `rope_theta` (1e6) and RMSNorm epsilon (1e-6) are fixed properties of the
/// checkpoint's Qwen3-0.6B decoder (`config.json`'s `text_config.rope_theta`
/// / `rms_norm_eps`, verified against the real checkpoint), not per-pack
/// metadata -- the same "family constant, not a GGUF key" convention
/// `firered_llm::runtime_contract`'s `FIRERED_LLM_ROPE_THETA` uses.
pub(crate) const MOSS_TD_ROPE_THETA: f32 = 1_000_000.0;
pub(crate) const MOSS_TD_RMS_NORM_EPSILON: f32 = 1e-6;
/// `nn.LayerNorm`'s `eps` in `VQAdaptor.__init__` (`config.py`:
/// `norm_eps=config.text_config.rms_norm_eps`) -- same value as the decoder's
/// RMSNorm epsilon, verified against the real checkpoint's `config.json`.
pub(crate) const MOSS_TD_ADAPTOR_NORM_EPSILON: f32 = 1e-6;

/// Final position safety ceiling for this family's Qwen3 decoder.
///
/// The checkpoint's `text_config.max_position_embeddings` is 131072 -- the
/// decoder's *RoPE context limit*, NOT a sane KV-cache capacity. But a decode
/// allocates TWO KV copies per position -- the host f32 copy
/// (`Qwen3AsrLayerKvCacheState`, eager on first write) and the device-resident
/// copy (`allocate_zeroed_llm_resident_kv_arena`) -- each shaped 28 layers x
/// 2 (K+V) x 8 kv-heads x 128 head_dim = 448 rows per position, so feeding
/// 131072 straight through reserves:
///
/// - host f32 copy: 131072 positions x 448 rows x 128 values x 4 B = 28 GiB
///   (the old "~30 GB" estimate counted this copy ALONE -- 30.06 decimal GB
///   -- and under-counted the policy by the resident half)
/// - resident f16 copy: 131072 x 448 rows x 128 values x 2 B = 14 GiB
/// - worst-case `DEFAULT` policy total: 336 KiB/position, 42 GiB
///
/// The host reservation is lazy-zeroed and harmless on the CPU backend (only
/// the touched prefix is resident), but the Metal backend physically wires
/// the resident buffers and exhausts a 16 GB machine many times over. Bytes
/// per position is not even a pack constant: the runtime may resolve the
/// `Q8_0` policy (both copies q8_0, 136 B per 128-value row, 119 KiB/position
/// total) or fall back to `DEFAULT` (discrete GPU, no native GQA, no flash
/// attention, wrong head_dim, or `OPENASR_QWEN_KV_CACHE_F32=1`), so static
/// reasoning must take the worst-case DEFAULT figure -- `crate::capacity`
/// pins both policies' numbers.
///
/// This limit is only a final validation guard above the unified topology
/// planner's request and session-envelope spans. It is never substituted as a
/// resident allocation size and it is not a VRAM-admission guarantee. Even a
/// legal KV span can be infeasible beside a particular weight layout,
/// allocator block geometry, encoder workspace, driver reserve, or external
/// process; those physical facts are quoted by the selected backend and
/// admitted transactionally by the device-memory broker. A rejected candidate
/// is handed back to execution policy without changing the semantic window.
///
/// Lesson, recorded so it does not recur: `max_position_embeddings` is an
/// attention/positional-encoding ceiling, not a working-set size; the two must
/// not be conflated when sizing runtime buffers -- and KV byte figures are
/// always computed for BOTH copies (host + resident), never one.
pub(crate) const MOSS_TD_MAX_KV_CACHE_POSITIONS: usize = 8192;

/// Intersect the pack's mathematical position ceiling with the family's final
/// safety ceiling. The result validates a topology demand; it does not request
/// an allocation of that size.
pub(crate) fn moss_td_kv_cache_positions(max_positions: usize) -> usize {
    max_positions.min(MOSS_TD_MAX_KV_CACHE_POSITIONS)
}

/// Return the exact request-sized KV allocation when its complete decode budget
/// fits both the imported pack's advertised ceiling and the family-wide cap.
/// `None` is a fail-closed result: callers must reject the request before
/// constructing a decoder cache, rather than clamp it and hit a KV bounds error
/// partway through generation.
pub(crate) fn moss_td_request_kv_cache_positions(
    pack_max_positions: usize,
    prompt_tokens: usize,
    max_generated_tokens: usize,
) -> Option<usize> {
    // Context legality is a semantic token bound: every returned token counts
    // even though the final sampled token is never fed back into the decoder.
    // Keep that proof separate from the physical greedy-cache write count.
    let semantic_positions = prompt_tokens.checked_add(max_generated_tokens)?;
    if semantic_positions > moss_td_kv_cache_positions(pack_max_positions) {
        return None;
    }
    let request_positions = greedy_self_kv_positions(prompt_tokens, max_generated_tokens).ok()?;
    Some(request_positions)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdEncoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub n_mels: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdAdaptorMetadata {
    pub merge_size: usize,
    pub input_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdDecoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
}

/// The complete parsed execution metadata for one moss-transcribe-diarize
/// pack: encoder + adaptor + decoder stages plus the cross-stage invariants
/// that only hold once all three are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdExecutionMetadata {
    pub encoder: MossTdEncoderMetadata,
    pub adaptor: MossTdAdaptorMetadata,
    pub decoder: MossTdDecoderMetadata,
}

#[derive(Debug, Error)]
pub(crate) enum MossTdRuntimeContractError {
    #[error("moss-transcribe-diarize missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("moss-transcribe-diarize GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("moss-transcribe-diarize expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("moss-transcribe-diarize missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("moss-transcribe-diarize GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn parse_encoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdEncoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(ENCODER_N_LAYERS_KEY)?;
    let d_model = usize_key(ENCODER_D_MODEL_KEY)?;
    let n_heads = usize_key(ENCODER_N_HEADS_KEY)?;
    let ffn_dim = usize_key(ENCODER_FFN_DIM_KEY)?;
    let n_mels = usize_key(ENCODER_N_MELS_KEY)?;
    let max_source_positions = usize_key(ENCODER_MAX_SOURCE_POSITIONS_KEY)?;
    for (key, value) in [
        (ENCODER_N_LAYERS_KEY, n_layers),
        (ENCODER_D_MODEL_KEY, d_model),
        (ENCODER_N_HEADS_KEY, n_heads),
        (ENCODER_FFN_DIM_KEY, ffn_dim),
        (ENCODER_N_MELS_KEY, n_mels),
        (ENCODER_MAX_SOURCE_POSITIONS_KEY, max_source_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: ENCODER_N_HEADS_KEY,
            reason: format!("d_model {d_model} is not a multiple of n_heads {n_heads}"),
        });
    }
    // The encoder graph bakes the FFN width as `MOSS_ENCODER_FFN_EXPANSION *
    // d_model` (it loads `ffn_up_bias` and binds the FFN projections at that
    // width), so a pack declaring any other `ffn_dim` can never run; fail
    // closed at admission rather than mid-graph.
    if ffn_dim != MOSS_ENCODER_FFN_EXPANSION * d_model {
        return Err(MetadataContractError::InvalidValue {
            key: ENCODER_FFN_DIM_KEY,
            reason: format!(
                "ffn_dim {ffn_dim} is unsupported: the encoder FFN width is fixed at {MOSS_ENCODER_FFN_EXPANSION} * d_model {}",
                MOSS_ENCODER_FFN_EXPANSION * d_model
            ),
        });
    }
    Ok(MossTdEncoderMetadata {
        n_layers,
        d_model,
        n_heads,
        ffn_dim,
        n_mels,
        max_source_positions,
    })
}

pub(crate) fn parse_adaptor_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdAdaptorMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let merge_size = usize_key(ADAPTOR_MERGE_SIZE_KEY)?;
    let input_dim = usize_key(ADAPTOR_INPUT_DIM_KEY)?;
    validate_positive_usize(merge_size, ADAPTOR_MERGE_SIZE_KEY)?;
    validate_positive_usize(input_dim, ADAPTOR_INPUT_DIM_KEY)?;
    Ok(MossTdAdaptorMetadata {
        merge_size,
        input_dim,
    })
}

pub(crate) fn parse_decoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdDecoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(LLM_N_LAYERS_KEY)?;
    let d_model = usize_key(LLM_D_MODEL_KEY)?;
    let ffn_dim = usize_key(LLM_FFN_DIM_KEY)?;
    let n_heads = usize_key(LLM_N_HEADS_KEY)?;
    let n_kv_heads = usize_key(LLM_N_KV_HEADS_KEY)?;
    let head_dim = usize_key(LLM_HEAD_DIM_KEY)?;
    let vocab_size = usize_key(LLM_VOCAB_SIZE_KEY)?;
    let max_positions = usize_key(LLM_MAX_POSITIONS_KEY)?;
    let audio_start_token_id = u32_key(LLM_AUDIO_START_TOKEN_ID_KEY)?;
    let audio_end_token_id = u32_key(LLM_AUDIO_END_TOKEN_ID_KEY)?;
    let audio_pad_token_id = u32_key(LLM_AUDIO_PAD_TOKEN_ID_KEY)?;

    for (key, value) in [
        (LLM_N_LAYERS_KEY, n_layers),
        (LLM_D_MODEL_KEY, d_model),
        (LLM_FFN_DIM_KEY, ffn_dim),
        (LLM_N_HEADS_KEY, n_heads),
        (LLM_N_KV_HEADS_KEY, n_kv_heads),
        (LLM_HEAD_DIM_KEY, head_dim),
        (LLM_VOCAB_SIZE_KEY, vocab_size),
        (LLM_MAX_POSITIONS_KEY, max_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    // Unlike Qwen2/firered-llm, Qwen3 decouples the per-head projection width
    // from `d_model / n_heads`: the real checkpoint's `head_dim` (128) times
    // `n_heads` (16) is 2048, not `d_model`'s 1024 -- `q_proj`/`k_proj`/
    // `v_proj` project to `n_heads * head_dim` and `attn_output` projects
    // back down to `d_model` (verified against the real checkpoint's
    // `config.json`). So there is no `n_heads * head_dim == d_model`
    // invariant to enforce here (matches `qwen::runtime_contract`, which
    // never asserts one either).
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: LLM_N_KV_HEADS_KEY,
            reason: format!("n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"),
        });
    }
    for (key, id) in [
        (LLM_AUDIO_START_TOKEN_ID_KEY, audio_start_token_id),
        (LLM_AUDIO_END_TOKEN_ID_KEY, audio_end_token_id),
        (LLM_AUDIO_PAD_TOKEN_ID_KEY, audio_pad_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(MossTdDecoderMetadata {
        n_layers,
        d_model,
        ffn_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_positions,
        audio_start_token_id,
        audio_end_token_id,
        audio_pad_token_id,
    })
}

/// Parse the complete moss-transcribe-diarize execution metadata, including
/// the facts that only become checkable once all three stages are present:
/// the `general.architecture` route identity and the encoder->adaptor
/// geometry bridge.
pub(crate) fn parse_moss_td_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdExecutionMetadata, MossTdRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)
        .map_err(map_metadata_contract_error)?;
    if architecture != MOSS_TD_GGML_ARCHITECTURE_ID {
        return Err(MossTdRuntimeContractError::UnexpectedArchitecture {
            expected: MOSS_TD_GGML_ARCHITECTURE_ID,
            found: architecture.to_string(),
        });
    }

    let encoder = parse_encoder_metadata(metadata).map_err(map_metadata_contract_error)?;
    let adaptor = parse_adaptor_metadata(metadata).map_err(map_metadata_contract_error)?;
    let decoder = parse_decoder_metadata(metadata).map_err(map_metadata_contract_error)?;

    // The adaptor's first linear consumes `merge_size` consecutive encoder
    // rows at once (`(B,T,E) -> (B,T/G,G*E)`), so its input width is exactly
    // `encoder.d_model * merge_size`. The importer cross-checks this against
    // `config.json`'s `adaptor_input_dim`; the runtime contract re-proves it
    // from the pack's own declared geometry so a hand-edited header cannot
    // admit a bridge the graph will not construct.
    let expected_input_dim = encoder
        .d_model
        .checked_mul(adaptor.merge_size)
        .ok_or_else(|| MossTdRuntimeContractError::InvalidMetadataValue {
            key: ADAPTOR_MERGE_SIZE_KEY,
            reason: format!(
                "encoder d_model {} * merge_size {} overflows while deriving the adaptor input width",
                encoder.d_model, adaptor.merge_size
            ),
        })?;
    if adaptor.input_dim != expected_input_dim {
        return Err(MossTdRuntimeContractError::InvalidMetadataValue {
            key: ADAPTOR_INPUT_DIM_KEY,
            reason: format!(
                "{ADAPTOR_INPUT_DIM_KEY}={} must equal {ENCODER_D_MODEL_KEY}={} * {ADAPTOR_MERGE_SIZE_KEY}={} ({})",
                adaptor.input_dim, encoder.d_model, adaptor.merge_size, expected_input_dim
            ),
        });
    }

    Ok(MossTdExecutionMetadata {
        encoder,
        adaptor,
        decoder,
    })
}

/// The decoder's attention projection widths. Qwen3's q/k/v project to
/// `heads * head_dim` (NOT `d_model`) -- see `parse_decoder_metadata`'s
/// divisibility note -- so these are derived products, checked for overflow.
fn decoder_projection_widths(
    decoder: MossTdDecoderMetadata,
) -> Result<(usize, usize), MossTdRuntimeContractError> {
    let q_width = decoder
        .n_heads
        .checked_mul(decoder.head_dim)
        .ok_or_else(|| MossTdRuntimeContractError::InvalidMetadataValue {
            key: LLM_HEAD_DIM_KEY,
            reason: format!(
                "n_heads {} * head_dim {} overflows while deriving the q projection width",
                decoder.n_heads, decoder.head_dim
            ),
        })?;
    let kv_width = decoder
        .n_kv_heads
        .checked_mul(decoder.head_dim)
        .ok_or_else(|| MossTdRuntimeContractError::InvalidMetadataValue {
            key: LLM_HEAD_DIM_KEY,
            reason: format!(
                "n_kv_heads {} * head_dim {} overflows while deriving the k/v projection width",
                decoder.n_kv_heads, decoder.head_dim
            ),
        })?;
    Ok((q_width, kv_width))
}

/// Metadata-derived tensor binding contract for the complete
/// moss-transcribe-diarize runtime tensor set: the Whisper-style encoder
/// (`moss.enc.*`), the VQAdaptor bridge (`moss.adaptor.*`), and the
/// Qwen3-parameterized decoder (`moss.llm.*`). Requirements reference the
/// parsed metadata only, the same "shapes the pack itself declares" policy
/// `qwen::runtime_contract::qwen3_runtime_tensor_descriptors` uses.
pub(crate) fn moss_td_runtime_tensor_descriptors(
    metadata: MossTdExecutionMetadata,
) -> Result<Vec<TensorBindingDescriptor>, MossTdRuntimeContractError> {
    let encoder = metadata.encoder;
    let adaptor = metadata.adaptor;
    let decoder = metadata.decoder;
    let (q_width, kv_width) = decoder_projection_widths(decoder)?;

    let mut descriptors = Vec::new();

    // --- encoder conv stem + fixed tables ----------------------------------
    descriptors.extend([
        TensorBindingDescriptor {
            tensor_name: ENC_CONV1_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                MOSS_TD_ENCODER_CONV_KERNEL,
                encoder.n_mels,
                encoder.d_model,
            ]),
            reason: "expected kernel-3 conv1 over the mel band into the encoder hidden size"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_CONV1_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
            reason: "expected conv1 bias with the encoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_CONV2_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                MOSS_TD_ENCODER_CONV_KERNEL,
                encoder.d_model,
                encoder.d_model,
            ]),
            reason: "expected kernel-3 stride-2 conv2 over the encoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_CONV2_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
            reason: "expected conv2 bias with the encoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_POS_EMBD_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                encoder.max_source_positions,
                encoder.d_model,
            ]),
            reason: "expected the fixed positional embedding table for max_source_positions"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_OUT_NORM_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
            reason: "expected encoder output norm weight".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ENC_OUT_NORM_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
            reason: "expected encoder output norm bias".to_string(),
        },
    ]);

    // --- encoder transformer layers -----------------------------------------
    for layer_idx in 0..encoder.n_layers {
        let names = moss_encoder_layer_tensor_names(layer_idx);
        descriptors.extend([
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.d_model,
                    encoder.d_model,
                ),
                reason: "expected rank-2 encoder attention q matrix over the hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.d_model,
                    encoder.d_model,
                ),
                reason: "expected rank-2 encoder attention k matrix over the hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.d_model,
                    encoder.d_model,
                ),
                reason: "expected rank-2 encoder attention v matrix over the hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_out_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.d_model,
                    encoder.d_model,
                ),
                reason: "expected rank-2 encoder attention output matrix over the hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_out_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.d_model,
                    encoder.ffn_dim,
                ),
                reason: "expected rank-2 encoder FFN up matrix between hidden and FFN sizes"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.ffn_dim),
                reason: "expected encoder FFN up bias with the FFN size".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    encoder.ffn_dim,
                    encoder.d_model,
                ),
                reason: "expected rank-2 encoder FFN down matrix between FFN and hidden sizes"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(encoder.d_model),
                reason: "expected encoder FFN down bias with the hidden size".to_string(),
            },
        ]);
    }

    // --- VQAdaptor bridge -----------------------------------------------------
    descriptors.extend([
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_LINEAR1_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                adaptor.input_dim,
                decoder.d_model,
            ),
            reason: "expected rank-2 adaptor linear1 between the merged encoder width and the decoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_LINEAR1_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            reason: "expected adaptor linear1 bias with the decoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_LINEAR2_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                decoder.d_model,
                decoder.d_model,
            ),
            reason: "expected rank-2 adaptor linear2 over the decoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_LINEAR2_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            reason: "expected adaptor linear2 bias with the decoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_NORM_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            reason: "expected adaptor norm weight with the decoder hidden size".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: ADAPTOR_NORM_BIAS.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            reason: "expected adaptor norm bias with the decoder hidden size".to_string(),
        },
    ]);

    // --- Qwen3 decoder ---------------------------------------------------------
    descriptors.extend([
        TensorBindingDescriptor {
            tensor_name: LLM_TOKEN_EMBD_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                decoder.d_model,
                decoder.vocab_size,
            ),
            reason: "expected token embedding matrix with decoder hidden size and vocab dimensions"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: LLM_OUTPUT_NORM_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            reason: "expected output norm vector with the decoder hidden size length".to_string(),
        },
    ]);
    for layer_idx in 0..decoder.n_layers {
        let names = moss_llm_layer_tensor_names(layer_idx);
        descriptors.extend([
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
                reason: "expected decoder hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.d_model,
                    q_width,
                ),
                reason:
                    "expected rank-2 decoder attn_q matrix between hidden size and n_heads*head_dim"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.d_model,
                    kv_width,
                ),
                reason: "expected rank-2 decoder attn_k matrix between hidden size and n_kv_heads*head_dim".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.d_model,
                    kv_width,
                ),
                reason: "expected rank-2 decoder attn_v matrix between hidden size and n_kv_heads*head_dim".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_output_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    q_width,
                    decoder.d_model,
                ),
                reason:
                    "expected rank-2 decoder attn_output matrix between n_heads*head_dim and hidden size"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.head_dim),
                reason: "expected QK-norm q vector with the per-head dimension".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.head_dim),
                reason: "expected QK-norm k vector with the per-head dimension".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
                reason: "expected decoder hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_gate_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.d_model,
                    decoder.ffn_dim,
                ),
                reason: "expected rank-2 decoder FFN gate matrix between hidden and FFN sizes"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.d_model,
                    decoder.ffn_dim,
                ),
                reason: "expected rank-2 decoder FFN up matrix between hidden and FFN sizes"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                    decoder.ffn_dim,
                    decoder.d_model,
                ),
                reason: "expected rank-2 decoder FFN down matrix between FFN and hidden sizes"
                    .to_string(),
            },
        ]);
    }

    Ok(descriptors)
}

/// Validate the pack's tensor set against the metadata-derived binding
/// contract. Runs after [`parse_moss_td_execution_metadata`] succeeds; a
/// missing tensor or a shape the declared geometry cannot construct fails
/// closed with the offending tensor named.
pub(crate) fn validate_moss_td_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: MossTdExecutionMetadata,
) -> Result<(), MossTdRuntimeContractError> {
    let descriptors = moss_td_runtime_tensor_descriptors(metadata)?;
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )?;
    Ok(())
}

fn missing_required_tensor(name: &str) -> MossTdRuntimeContractError {
    MossTdRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(name: &str, shape: &[u64], reason: String) -> MossTdRuntimeContractError {
    MossTdRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn map_metadata_contract_error(error: MetadataContractError) -> MossTdRuntimeContractError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            MossTdRuntimeContractError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            MossTdRuntimeContractError::InvalidMetadataValue { key, reason }
        }
    }
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata = parse_moss_td_execution_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error(
            "moss-transcribe-diarize",
            error,
        )
    })?;
    validate_moss_td_runtime_tensors_with_index(preflight.tensor_index(), metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufTensorMetadata;
    use crate::ggml_runtime::GgufTensorIndexSnapshot;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn full_metadata() -> BTreeMap<String, String> {
        [
            (GENERAL_ARCHITECTURE_KEY, MOSS_TD_GGML_ARCHITECTURE_ID),
            (ENCODER_N_LAYERS_KEY, "24"),
            (ENCODER_D_MODEL_KEY, "1024"),
            (ENCODER_N_HEADS_KEY, "16"),
            (ENCODER_FFN_DIM_KEY, "4096"),
            (ENCODER_N_MELS_KEY, "80"),
            (ENCODER_MAX_SOURCE_POSITIONS_KEY, "1500"),
            (ADAPTOR_MERGE_SIZE_KEY, "4"),
            (ADAPTOR_INPUT_DIM_KEY, "4096"),
            (LLM_N_LAYERS_KEY, "28"),
            (LLM_D_MODEL_KEY, "1024"),
            (LLM_FFN_DIM_KEY, "3072"),
            (LLM_N_HEADS_KEY, "16"),
            (LLM_N_KV_HEADS_KEY, "8"),
            (LLM_HEAD_DIM_KEY, "128"),
            (LLM_VOCAB_SIZE_KEY, "151936"),
            (LLM_MAX_POSITIONS_KEY, "131072"),
            (LLM_AUDIO_START_TOKEN_ID_KEY, "151669"),
            (LLM_AUDIO_END_TOKEN_ID_KEY, "151670"),
            (LLM_AUDIO_PAD_TOKEN_ID_KEY, "151671"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Tiny internally-consistent geometry for tensor-level tests: one
    /// encoder layer, one decoder layer, small widths. Every metadata
    /// invariant holds (d_model % n_heads, n_heads % n_kv_heads, token ids in
    /// vocab, input_dim == d_model * merge_size).
    fn tiny_metadata() -> BTreeMap<String, String> {
        [
            (GENERAL_ARCHITECTURE_KEY, MOSS_TD_GGML_ARCHITECTURE_ID),
            (ENCODER_N_LAYERS_KEY, "1"),
            (ENCODER_D_MODEL_KEY, "16"),
            (ENCODER_N_HEADS_KEY, "2"),
            // The encoder graph bakes the FFN width as 4 * d_model
            // (`MOSS_ENCODER_FFN_EXPANSION`), so the tiny geometry declares 64.
            (ENCODER_FFN_DIM_KEY, "64"),
            (ENCODER_N_MELS_KEY, "8"),
            (ENCODER_MAX_SOURCE_POSITIONS_KEY, "20"),
            (ADAPTOR_MERGE_SIZE_KEY, "2"),
            (ADAPTOR_INPUT_DIM_KEY, "32"),
            (LLM_N_LAYERS_KEY, "1"),
            (LLM_D_MODEL_KEY, "16"),
            (LLM_FFN_DIM_KEY, "32"),
            (LLM_N_HEADS_KEY, "2"),
            (LLM_N_KV_HEADS_KEY, "1"),
            (LLM_HEAD_DIM_KEY, "8"),
            (LLM_VOCAB_SIZE_KEY, "64"),
            (LLM_MAX_POSITIONS_KEY, "128"),
            (LLM_AUDIO_START_TOKEN_ID_KEY, "5"),
            (LLM_AUDIO_END_TOKEN_ID_KEY, "6"),
            (LLM_AUDIO_PAD_TOKEN_ID_KEY, "7"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Every tensor the tiny geometry declares, with the shapes the importer
    /// writes for them (the production contract's own reference orientation).
    fn tiny_tensor_shapes() -> Vec<(String, Vec<u64>)> {
        let mut tensors: Vec<(String, Vec<u64>)> = vec![
            (ENC_CONV1_WEIGHT.to_string(), vec![3, 8, 16]),
            (ENC_CONV1_BIAS.to_string(), vec![16]),
            (ENC_CONV2_WEIGHT.to_string(), vec![3, 16, 16]),
            (ENC_CONV2_BIAS.to_string(), vec![16]),
            (ENC_POS_EMBD_WEIGHT.to_string(), vec![20, 16]),
            (ENC_OUT_NORM_WEIGHT.to_string(), vec![16]),
            (ENC_OUT_NORM_BIAS.to_string(), vec![16]),
            (ADAPTOR_LINEAR1_WEIGHT.to_string(), vec![32, 16]),
            (ADAPTOR_LINEAR1_BIAS.to_string(), vec![16]),
            (ADAPTOR_LINEAR2_WEIGHT.to_string(), vec![16, 16]),
            (ADAPTOR_LINEAR2_BIAS.to_string(), vec![16]),
            (ADAPTOR_NORM_WEIGHT.to_string(), vec![16]),
            (ADAPTOR_NORM_BIAS.to_string(), vec![16]),
            (LLM_TOKEN_EMBD_WEIGHT.to_string(), vec![16, 64]),
            (LLM_OUTPUT_NORM_WEIGHT.to_string(), vec![16]),
        ];
        let enc = moss_encoder_layer_tensor_names(0);
        tensors.extend([
            (enc.attn_norm_weight, vec![16]),
            (enc.attn_norm_bias, vec![16]),
            (enc.attn_q_weight, vec![16, 16]),
            (enc.attn_q_bias, vec![16]),
            (enc.attn_k_weight, vec![16, 16]),
            (enc.attn_v_weight, vec![16, 16]),
            (enc.attn_v_bias, vec![16]),
            (enc.attn_out_weight, vec![16, 16]),
            (enc.attn_out_bias, vec![16]),
            (enc.ffn_norm_weight, vec![16]),
            (enc.ffn_norm_bias, vec![16]),
            (enc.ffn_up_weight, vec![16, 64]),
            (enc.ffn_up_bias, vec![64]),
            (enc.ffn_down_weight, vec![64, 16]),
            (enc.ffn_down_bias, vec![16]),
        ]);
        let llm = moss_llm_layer_tensor_names(0);
        tensors.extend([
            (llm.attn_norm_weight, vec![16]),
            (llm.attn_q_weight, vec![16, 16]),
            (llm.attn_k_weight, vec![16, 8]),
            (llm.attn_v_weight, vec![16, 8]),
            (llm.attn_output_weight, vec![16, 16]),
            (llm.attn_q_norm_weight, vec![8]),
            (llm.attn_k_norm_weight, vec![8]),
            (llm.ffn_norm_weight, vec![16]),
            (llm.ffn_gate_weight, vec![16, 32]),
            (llm.ffn_up_weight, vec![16, 32]),
            (llm.ffn_down_weight, vec![32, 16]),
        ]);
        tensors
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
            path: PathBuf::from("/tmp/moss-td-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    #[test]
    fn parses_encoder_metadata_matching_real_checkpoint() {
        let parsed = parse_encoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 24);
        assert_eq!(parsed.d_model, 1024);
        assert_eq!(parsed.max_source_positions, 1500);
    }

    #[test]
    fn parses_adaptor_metadata_matching_real_checkpoint() {
        let parsed = parse_adaptor_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.merge_size, 4);
        assert_eq!(parsed.input_dim, 4096);
    }

    #[test]
    fn parses_decoder_metadata_matching_real_checkpoint() {
        let parsed = parse_decoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_kv_heads, 8);
        assert_eq!(parsed.audio_pad_token_id, 151_671);
    }

    #[test]
    fn parses_execution_metadata_with_route_identity_and_bridge() {
        let parsed = parse_moss_td_execution_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.encoder.n_layers, 24);
        assert_eq!(parsed.adaptor.merge_size, 4);
        assert_eq!(parsed.decoder.audio_pad_token_id, 151_671);
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut metadata = full_metadata();
        metadata.insert(LLM_N_KV_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_audio_token_id_out_of_vocab() {
        let mut metadata = full_metadata();
        metadata.insert(LLM_AUDIO_PAD_TOKEN_ID_KEY.to_string(), "999999".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_a_foreign_general_architecture() {
        let mut metadata = full_metadata();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            "some-other-architecture".to_string(),
        );
        let error = parse_moss_td_execution_metadata(&metadata)
            .expect_err("foreign architecture must fail closed");
        assert!(matches!(
            error,
            MossTdRuntimeContractError::UnexpectedArchitecture { .. }
        ));
    }

    #[test]
    fn rejects_an_adaptor_bridge_the_encoder_geometry_cannot_feed() {
        let mut metadata = full_metadata();
        metadata.insert(ADAPTOR_INPUT_DIM_KEY.to_string(), "999".to_string());
        let error = parse_moss_td_execution_metadata(&metadata)
            .expect_err("input_dim != d_model * merge_size must fail closed");
        match error {
            MossTdRuntimeContractError::InvalidMetadataValue { key, reason } => {
                assert_eq!(key, ADAPTOR_INPUT_DIM_KEY);
                assert!(reason.contains(ADAPTOR_INPUT_DIM_KEY), "{reason}");
                assert!(reason.contains(ENCODER_D_MODEL_KEY), "{reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn tensor_contract_covers_every_imported_tensor_exactly_once() {
        // The importer writes 683 tensors for the real checkpoint geometry
        // (367 encoder + 6 adaptor + 310 decoder, see `package_import`'s
        // golden parity test). The metadata-derived descriptor set must name
        // exactly that set: one descriptor per tensor, no duplicates, no
        // omissions -- a drift in either direction means the validator either
        // under-checks the pack or demands a tensor the importer never
        // writes.
        let metadata = parse_moss_td_execution_metadata(&full_metadata()).expect("parse");
        let descriptors = moss_td_runtime_tensor_descriptors(metadata).expect("descriptors");
        let mut names: Vec<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(names.len(), 683, "descriptor count");
        names.dedup();
        assert_eq!(names.len(), 683, "duplicate descriptor names");
        assert!(names.contains(&ENC_CONV1_WEIGHT));
        assert!(names.contains(&"moss.enc.blk.23.ffn_down.bias"));
        assert!(names.contains(&ADAPTOR_NORM_BIAS));
        assert!(names.contains(&LLM_TOKEN_EMBD_WEIGHT));
        assert!(names.contains(&"moss.llm.blk.27.ffn_down.weight"));
    }

    #[test]
    fn validates_the_tiny_reference_tensor_set() {
        let metadata = parse_moss_td_execution_metadata(&tiny_metadata()).expect("parse");
        let index = tensor_index_from_shapes(&tiny_tensor_shapes());
        validate_moss_td_runtime_tensors_with_index(&index, metadata).expect("tiny tensor set");
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let metadata = parse_moss_td_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        shapes.retain(|(name, _)| *name != ADAPTOR_LINEAR1_WEIGHT);
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_moss_td_runtime_tensors_with_index(&index, metadata)
            .expect_err("missing adaptor tensor must fail closed");
        match error {
            MossTdRuntimeContractError::MissingRequiredTensor { name } => {
                assert_eq!(name, ADAPTOR_LINEAR1_WEIGHT);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_a_conv_kernel_with_the_wrong_mel_band() {
        let metadata = parse_moss_td_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        for (name, dims) in shapes.iter_mut() {
            if *name == ENC_CONV1_WEIGHT {
                *dims = vec![3, 4, 16];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_moss_td_runtime_tensors_with_index(&index, metadata)
            .expect_err("conv1 mel-band mismatch must fail closed");
        match error {
            MossTdRuntimeContractError::InvalidTensorShape { name, .. } => {
                assert_eq!(name, ENC_CONV1_WEIGHT);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_a_decoder_projection_with_the_wrong_kv_width() {
        let metadata = parse_moss_td_execution_metadata(&tiny_metadata()).expect("parse");
        let mut shapes = tiny_tensor_shapes();
        let llm = moss_llm_layer_tensor_names(0);
        for (name, dims) in shapes.iter_mut() {
            if *name == llm.attn_k_weight.as_str() {
                *dims = vec![16, 99];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_moss_td_runtime_tensors_with_index(&index, metadata)
            .expect_err("k projection width mismatch must fail closed");
        match error {
            MossTdRuntimeContractError::InvalidTensorShape { name, .. } => {
                assert_eq!(name, llm.attn_k_weight);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn kv_cache_positions_caps_the_rope_context_limit() {
        // A pack with the raw RoPE ceiling (131072) baked in clamps down to the
        // final safety ceiling. The topology planner still requests only its
        // proven invocation/session span; neither 8192 nor 131072 is an arena
        // allocation request.
        assert_eq!(
            moss_td_kv_cache_positions(131_072),
            MOSS_TD_MAX_KV_CACHE_POSITIONS
        );
        assert_eq!(moss_td_kv_cache_positions(8_192), 8_192);
        // A short-enough value passes through untouched.
        assert_eq!(moss_td_kv_cache_positions(300), 300);
    }

    #[test]
    fn request_kv_cache_capacity_respects_pack_ceiling_and_decode_budget() {
        // A legacy pack retains its raw RoPE metadata, but a short request only
        // allocates the prompt plus its configured generation budget.
        assert_eq!(
            moss_td_request_kv_cache_positions(131_072, 300, 4_096),
            Some(4_395)
        );
        // A freshly imported pack advertises the same 8192-position ceiling.
        assert_eq!(
            moss_td_request_kv_cache_positions(8_192, 4_096, 4_096),
            Some(8_191)
        );
        // Never let an imported lower ceiling be silently expanded.
        assert_eq!(moss_td_request_kv_cache_positions(4_096, 1, 4_096), None);
        // Arithmetic overflow is also fail-closed rather than saturating into
        // an undersized cache.
        assert_eq!(
            moss_td_request_kv_cache_positions(8_192, usize::MAX, 1),
            None
        );
    }
}
