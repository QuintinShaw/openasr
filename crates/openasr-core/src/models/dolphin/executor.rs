//! Dolphin `small.cn` dedicated executor: the full end-to-end transcribe path.
//!
//! Pipeline (all from the `.oasr` pack): kaldi-fbank [`frontend`] + the checkpoint's
//! global CMVN -> the parity-verified E-Branchformer [`encoder_graph`] ->
//! CTC/attention [`joint_decode`] (CTC prefix-beam over the CTC head, rescored by
//! the Transformer [`decoder_graph`]) -> char detokenize. The executor fails closed
//! with typed errors on a bad pack and never fabricates a transcript.
//!
//! [`frontend`]: super::frontend
//! [`joint_decode`]: super::joint_decode
//! [`encoder_graph`]: super::encoder_graph
//! [`decoder_graph`]: super::decoder_graph

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::NativeAsrSession;
use crate::PhraseBiasConfig;
use crate::api::backend::{Segment, Transcription};
use crate::arch::DOLPHIN_GGML_ADAPTER_ID;
use crate::ggml_runtime::{
    GGML_TYPE_F32, GgmlCpuGraphBackend, GgufMetadata, GgufOwnedWeightTensorPayload,
    GgufTensorDataReadError, GgufTensorDataReader, GgufWeightTensorElementType,
};
use crate::models::admitted_host_object_cache::{
    AdmittedHostObjectCache, AdmittedHostObjectCacheLimits,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::system_memory_owner::{
    AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use super::decoder_graph::{DolphinDecoderConfig, DolphinDecoderRescoreRuntime};
use super::encoder_graph::{
    DolphinEncoderConfig, DolphinEncoderOutput, DolphinEncoderRuntime, DolphinNativeWeight,
    DolphinWeightProvider, encode, minimum_subsample_input_frames,
};
use super::frontend::{
    DolphinEspnetFrontend, DolphinFbankFrontend, apply_global_cmvn, espnet_min_samples_for_frames,
    kaldi_min_samples_for_frames,
};
use super::hotword_context::{
    apply_hotword_deep_biasing, encode_hotword_context_embeddings, tokenize_hotword_phrase,
};
use super::joint_decode::{
    DolphinCtcHeadRuntime, DolphinJointDecodeConfig, detokenize_char_tokens, joint_decode,
};
use super::language::{build_dolphin_decode_prefix, build_dolphin_multilingual_decode_prefix};
use super::package_import::DolphinLanguageScheme;
use super::runtime_contract::parse_dolphin_execution_metadata;

/// Encoder weight namespace baked into the pack under exact WeNet names.
const ENCODER_TENSOR_PREFIX: &str = "encoder.";
/// Sentinels proving the pack baked the encoder + CTC head namespaces (cheap
/// index probe, no dequantization), common to both language schemes.
const ENCODER_SENTINEL_TENSORS: [&str; 2] = ["encoder.after_norm.weight", "ctc.ctc_lo.weight"];
/// CnDialect-only sentinel: the multilingual scheme's encoder attention never
/// bakes this table (its `rel_pos_v1` table is computed fresh per request
/// instead -- see `encoder_graph::dolphin_relative_positional_table`), so
/// requiring it there would fail closed on every valid multilingual pack.
const ENCODER_CN_DIALECT_SENTINEL_TENSOR: &str = "encoder.embed.pos_enc.pe";

/// Global CMVN vectors baked in the pack (checkpoint's own `encoder.global_cmvn`).
const CMVN_MEAN_TENSOR: &str = "encoder.global_cmvn.mean";
const CMVN_ISTD_TENSOR: &str = "encoder.global_cmvn.istd";

/// Pack metadata keys the decode reads (mirrors the importer's writes). The
/// decode prefix is no longer read from the pack: it is built per request from the
/// vocab + the requested language code (see [`build_dolphin_decode_prefix`]), so a
/// single pack can honor any advertised dialect region rather than one baked one.
const EOS_TOKEN_ID_KEY: &str = "dolphin.eos_token_id";
/// Selects the decode-prefix builder (see `run_dolphin_pipeline`); absent on
/// a pre-existing pack, which defaults to the cn-dialect scheme.
const LANGUAGE_SCHEME_KEY: &str = "dolphin.language.scheme";
const BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// CTC prefix-beam width used for joint decode (WeNet default).
const DOLPHIN_BEAM_SIZE: usize = 10;

/// Rescoring combination weight. The reference `attention_rescoring` decode selects
/// purely by attention score over the CTC n-best (`ctc_weight = 0.0`); the model's
/// `0.3` is the *training* loss weight (`model_conf.ctc_weight`), a different knob.
/// Kept `0.0` so the runtime reproduces the golden reference decode.
pub(crate) const DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT: f32 = 0.0;

/// Whether a pack tensor is a rank-2 `.weight` matmul operand that should be bound
/// in its native (quantized / f16) ggml layout rather than dequantized to f32.
///
/// The quantized packs block-quantize exactly these rank-2 `.weight` matrices, and
/// `mul_mat` runs the quantized/f16 lhs directly -- so keeping them native lets the
/// backend buffer hold the small quantized weights (q4 < q8 < fp16 in RAM) instead
/// of a dequantized-to-f32 blow-up. Three tensors are deliberately excluded because
/// they are consumed outside `mul_mat`: `decoder.embed.0.weight` and
/// `context_module.context_extractor.word_embedding.weight` are `ggml_get_rows`
/// (row lookup) / plain-Rust row-lookup operands, and
/// `context_module.context_encoder.0.weight` is a `Linear` consumed by the
/// pure-Rust hotword context encoder (`models::dolphin::hotword_context`), not the
/// ggml graph -- all three only accept f32 (or f32/f16 for get_rows), so they stay
/// dequantized to f32 -- and only rank-2 `.weight` *matmul* operands actually fed
/// to a ggml `mul_mat` go native by design.
const PURE_RUST_MATMUL_WEIGHT_EXCLUSIONS: [&str; 3] = [
    "decoder.embed.0.weight",
    "context_module.context_extractor.word_embedding.weight",
    "context_module.context_encoder.0.weight",
];

fn is_native_matmul_weight(name: &str, dims: &[u64]) -> bool {
    name.ends_with(".weight")
        && dims.len() == 2
        && !PURE_RUST_MATMUL_WEIGHT_EXCLUSIONS.contains(&name)
}

/// Materialize one pack tensor into the pool: rank-2 `.weight` matmul operands are
/// kept as their native (quantized / f16) mmap-backed block payload (zero-copy, no
/// dequant); everything else (1-D norms/biases, convs, position tables, the CMVN
/// vectors, the mel filterbank, and the decoder token embedding) is dequantized to
/// f32. Reading each tensor at its own stored dims keeps this layout-agnostic --
/// every graph re-declares its own ggml shapes and consumes the payload only by
/// element/byte count.
fn insert_pool_tensor(
    pool: &mut DolphinRuntimeWeights,
    reader: &GgufTensorDataReader,
    name: &str,
    dims: &[u64],
) -> Result<(), GgufTensorDataReadError> {
    if is_native_matmul_weight(name, dims) {
        let payload = reader.owned_weight_tensor_payload_by_name(name)?;
        if !matches!(payload.element_type, GgufWeightTensorElementType::F32) {
            pool.native_weights.insert(name.to_string(), payload);
            return Ok(());
        }
        // An f32-stored rank-2 `.weight` (not produced by the fp16/quant packs)
        // has nothing to keep-quantize; fall through to the f32 path.
    }
    let values = reader.host_tensor_f32_copy_dequantized_by_name(name, dims)?;
    pool.f32_tensors.insert(name.to_string(), values);
    Ok(())
}

/// Load every tensor in the pack into the runtime pool (rank-2 `.weight` matmul
/// operands kept native/quantized, the rest dequantized to f32) keyed by its exact
/// WeNet name -- the provider shape the encoder/decoder/CTC graphs consume.
pub(crate) fn load_dolphin_runtime_weights_from_pack(
    reader: &GgufTensorDataReader,
) -> Result<DolphinRuntimeWeights, GgufTensorDataReadError> {
    let mut weights = DolphinRuntimeWeights::default();
    for tensor in reader.tensor_index().tensors() {
        insert_pool_tensor(&mut weights, reader, &tensor.name, &tensor.dims)?;
    }
    Ok(weights)
}

/// Load only the `encoder.*` tensors from the pack (the encoder-from-pack parity
/// path; the full transcribe path uses [`load_dolphin_runtime_weights_from_pack`]).
pub(crate) fn load_dolphin_encoder_weights_from_pack(
    reader: &GgufTensorDataReader,
) -> Result<DolphinRuntimeWeights, GgufTensorDataReadError> {
    let mut weights = DolphinRuntimeWeights::default();
    for tensor in reader.tensor_index().tensors() {
        if !tensor.name.starts_with(ENCODER_TENSOR_PREFIX) {
            continue;
        }
        insert_pool_tensor(&mut weights, reader, &tensor.name, &tensor.dims)?;
    }
    Ok(weights)
}

/// Run the verified E-Branchformer encoder graph on weights loaded from the pack.
/// `features` is the CMVN'd `[frames_in, feature_dim]` log-mel input (frame-major,
/// mel bin innermost), matching the golden `logmel_feats_cmvn` fixture the raw
/// safetensors parity harness uses.
pub(crate) fn encode_dolphin_encoder_from_pack(
    reader: &GgufTensorDataReader,
    features: &[f32],
    frames_in: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinEncoderOutput, String> {
    let weights = load_dolphin_encoder_weights_from_pack(reader)
        .map_err(|error| format!("dolphin encoder weight load failed: {error}"))?;
    let config = DolphinEncoderConfig::small_cn();
    // Every caller of this pack-from-disk entry point (the production pipeline
    // plus the fp16/quant-rung regression test) only reads `encoder_out`, so
    // taps stay off here (P6): see `encode`'s doc comment.
    encode(&config, &weights, features, frames_in, backend, false)
        .map_err(|error| format!("dolphin encoder graph failed: {error}"))
}

/// A rescored joint-decode hypothesis, detokenized for reporting.
#[derive(Debug, Clone)]
pub(crate) struct DolphinScoredText {
    pub text: String,
    pub ctc_score: f32,
    pub attention_score: f32,
    pub combined_score: f32,
}

/// End-to-end transcription output plus the diagnostics the harness reports.
#[derive(Debug, Clone)]
pub(crate) struct DolphinPipelineOutput {
    /// Best (rescored) transcript.
    pub text: String,
    pub best_token_ids: Vec<u32>,
    /// CTC greedy transcript (pre-rescoring), for comparison.
    pub ctc_greedy_text: String,
    /// Rescored n-best, best-first.
    pub scored_nbest: Vec<DolphinScoredText>,
    /// Normalized recognition code the decode prefix selected (`zh`, `zh-sichuan`,
    /// ...), surfaced so the executor reports the language it actually decoded.
    pub resolved_language: String,
}

/// Runtime weights for one pack, shared behind an `Arc` so the process-level pool
/// can hand the same immutable table to every call. Rank-2 `.weight` matmul
/// operands are held as their native (quantized / f16) mmap-backed block payload --
/// zero-copy over the pack mmap, so the pool itself adds no per-tensor host copy
/// for them and the quantized weight lands in the ggml backend buffer at run time.
/// Everything else (1-D vectors, convs, position tables, CMVN, mel, the decoder
/// token embedding) is dequantized to f32.
#[derive(Default)]
pub(crate) struct DolphinRuntimeWeights {
    f32_tensors: HashMap<String, Vec<f32>>,
    native_weights: HashMap<String, GgufOwnedWeightTensorPayload>,
}

impl DolphinRuntimeWeights {
    fn quoted_system_memory_bytes(reader: &GgufTensorDataReader) -> Result<(u64, u64), String> {
        let mut retained = 0u64;
        let mut peak = 0u64;
        for tensor in reader.tensor_index().tensors() {
            let key_bytes = u64::try_from(tensor.name.len())
                .map_err(|_| "dolphin tensor name length exceeds u64".to_string())?;
            let native_candidate = is_native_matmul_weight(&tensor.name, &tensor.dims);
            if native_candidate && tensor.ggml_type != GGML_TYPE_F32 {
                let payload =
                    GgufOwnedWeightTensorPayload::quoted_retained_system_memory_bytes(tensor)?;
                let entry =
                    u64::try_from(std::mem::size_of::<(String, GgufOwnedWeightTensorPayload)>())
                        .map_err(|_| "dolphin native map entry size exceeds u64".to_string())?;
                let component = key_bytes
                    .checked_add(payload)
                    .and_then(|bytes| bytes.checked_add(entry))
                    .ok_or_else(|| "dolphin native weight quote overflowed".to_string())?;
                retained = retained
                    .checked_add(component)
                    .ok_or_else(|| "dolphin retained weight quote overflowed".to_string())?;
                peak = peak.max(retained);
                continue;
            }

            let elements = tensor.num_elements().ok_or_else(|| {
                format!("dolphin tensor '{}' element count overflowed", tensor.name)
            })?;
            let values = elements
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| format!("dolphin tensor '{}' f32 quote overflowed", tensor.name))?;
            let entry = u64::try_from(std::mem::size_of::<(String, Vec<f32>)>())
                .map_err(|_| "dolphin f32 map entry size exceeds u64".to_string())?;
            let component = key_bytes
                .checked_add(values)
                .and_then(|bytes| bytes.checked_add(entry))
                .ok_or_else(|| "dolphin f32 weight quote overflowed".to_string())?;
            let temporary_payload = if native_candidate {
                GgufOwnedWeightTensorPayload::quoted_retained_system_memory_bytes(tensor)?
            } else {
                0
            };
            let payload_phase = retained
                .checked_add(temporary_payload)
                .ok_or_else(|| "dolphin temporary weight quote overflowed".to_string())?;
            let materialized_phase = retained
                .checked_add(component)
                .ok_or_else(|| "dolphin weight construction peak overflowed".to_string())?;
            peak = peak.max(payload_phase).max(materialized_phase);
            retained = retained
                .checked_add(component)
                .ok_or_else(|| "dolphin retained weight quote overflowed".to_string())?;
        }
        Ok((peak.max(retained), retained))
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_usize(
            self.f32_tensors
                .len()
                .checked_mul(std::mem::size_of::<(String, Vec<f32>)>())
                .ok_or_else(|| "dolphin f32 map entry bytes overflowed".to_string())?,
            "dolphin f32 weight map entries",
        )?;
        for (name, values) in &self.f32_tensors {
            bytes.add_string(name, "dolphin f32 tensor name")?;
            bytes.add_vec(values, "dolphin f32 tensor values")?;
        }
        bytes.add_usize(
            self.native_weights
                .len()
                .checked_mul(std::mem::size_of::<(String, GgufOwnedWeightTensorPayload)>())
                .ok_or_else(|| "dolphin native map entry bytes overflowed".to_string())?,
            "dolphin native weight map entries",
        )?;
        for (name, payload) in &self.native_weights {
            bytes.add_string(name, "dolphin native tensor name")?;
            bytes.add(
                payload.retained_system_memory_bytes()?,
                "dolphin native tensor payload metadata",
            )?;
        }
        Ok(bytes.finish())
    }
}

impl DolphinWeightProvider for DolphinRuntimeWeights {
    fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.f32_tensors.get(name).map(Vec::as_slice)
    }

    fn native_weight(&self, name: &str) -> Option<DolphinNativeWeight<'_>> {
        self.native_weights
            .get(name)
            .map(|payload| DolphinNativeWeight {
                ggml_type: payload.element_type.ggml_type(),
                bytes: payload.bytes(),
            })
    }
}

// Lets the serving path (already-loaded weights) resolve
// `dolphin.{encoder,decoder}.max_ctx` from the baked position-table tensor's
// own element count, mirroring the `GgufTensorIndex`-based probe the install
// gate uses before any weight is dequantized. See
// `runtime_contract::resolve_position_table_max_ctx`.
impl super::runtime_contract::DolphinPositionTableSource for DolphinRuntimeWeights {
    fn tensor_element_count(&self, name: &str) -> Option<usize> {
        DolphinWeightProvider::tensor(self, name).map(<[f32]>::len)
    }
}

/// Prepared per-`(pack, backend)` graph runtimes: the encoder, CTC head, and
/// rescore decoder each keep their weights resident in a persistent
/// WEIGHTS-usage arena (see the respective runtime types). Cached thread-local
/// by `(PackContentKey, backend)` -- the sensevoice/moonshine prepared-runtime
/// pattern -- so a warm request (or a streaming tick, which re-enters
/// `execute()` per snapshot) pays zero weight re-upload: only the per-call
/// audio features / encoder memory / token ids travel to the backend.
/// Residency changes no computed value; every output stays golden-identical.
pub(crate) struct DolphinPreparedRuntime {
    backend: GgmlCpuGraphBackend,
    encoder: DolphinEncoderRuntime,
    ctc_head: DolphinCtcHeadRuntime,
    rescore: DolphinDecoderRescoreRuntime,
}

impl DolphinPreparedRuntime {
    fn quoted_system_memory_bytes(
        weights: &DolphinRuntimeWeights,
        metadata: &GgufMetadata,
    ) -> Result<(u64, u64), String> {
        let language_scheme = parse_dolphin_language_scheme(metadata)?;
        let execution_metadata = parse_dolphin_execution_metadata(metadata, weights)
            .map_err(|error| format!("dolphin runtime metadata contract failed: {error}"))?;
        let encoder_config =
            DolphinEncoderConfig::from_execution_metadata(&execution_metadata, language_scheme);
        let decoder_config = DolphinDecoderConfig::from_execution_metadata(&execution_metadata);
        let (encoder_peak, encoder_retained) = DolphinEncoderRuntime::quoted_system_memory_bytes(
            &encoder_config,
            encoder_config.max_positions,
        )?;
        let (decoder_peak, decoder_retained) =
            DolphinDecoderRescoreRuntime::quoted_system_memory_bytes(&decoder_config)?;
        let retained = encoder_retained
            .checked_add(decoder_retained)
            .ok_or_else(|| "dolphin prepared retained quote overflowed".to_string())?;
        let decoder_phase = encoder_retained
            .checked_add(decoder_peak)
            .ok_or_else(|| "dolphin decoder construction peak overflowed".to_string())?;
        Ok((encoder_peak.max(decoder_phase).max(retained), retained))
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        self.encoder
            .retained_system_memory_bytes()?
            .checked_add(self.rescore.retained_system_memory_bytes()?)
            .ok_or_else(|| "dolphin prepared retained measurement overflowed".to_string())
    }
}

type DolphinPreparedRuntimeCacheKey = (
    crate::models::runtime_cache_coordinator::PackContentKey,
    crate::models::native_execution_services::ExecutionLaneKey,
);

struct DolphinPreparedRuntimeActorState {
    runtime: DolphinPreparedRuntime,
    _weights: AdmittedHostObject<DolphinRuntimeWeights>,
}

impl std::fmt::Debug for DolphinPreparedRuntimeActorState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DolphinPreparedRuntimeActorState")
            .finish_non_exhaustive()
    }
}

type DolphinRuntimeWeightsCache = AdmittedHostObjectCache<
    crate::models::runtime_cache_coordinator::PackContentKey,
    DolphinRuntimeWeights,
>;
type DolphinPreparedRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    DolphinPreparedRuntimeCacheKey,
    DolphinPreparedRuntimeActorState,
>;
type DolphinPreparedRuntimeActor =
    PinnedRuntimeActorCheckout<DolphinPreparedRuntimeCacheKey, DolphinPreparedRuntimeActorState>;

/// Build the three prepared graph runtimes for one pack + backend. The
/// encoder's resident position-table capacity is sized to the pack's own
/// `encoder.max_ctx`; a longer utterance (multilingual scheme only, whose
/// table is computed rather than baked) falls back to a one-shot runtime in
/// [`run_dolphin_pipeline`] instead of failing.
pub(crate) fn build_dolphin_prepared_runtime(
    weights: &DolphinRuntimeWeights,
    metadata: &GgufMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinPreparedRuntime, String> {
    let language_scheme = parse_dolphin_language_scheme(metadata)?;
    let execution_metadata = parse_dolphin_execution_metadata(metadata, weights)
        .map_err(|error| format!("dolphin runtime metadata contract failed: {error}"))?;
    let encoder_config =
        DolphinEncoderConfig::from_execution_metadata(&execution_metadata, language_scheme);
    let decoder_config = DolphinDecoderConfig::from_execution_metadata(&execution_metadata);
    let encoder = DolphinEncoderRuntime::new(
        &encoder_config,
        weights,
        backend,
        encoder_config.max_positions,
    )
    .map_err(|error| format!("dolphin encoder runtime build failed: {error}"))?;
    let ctc_head = DolphinCtcHeadRuntime::new(
        weights,
        decoder_config.d_model,
        decoder_config.vocab_size,
        backend,
    )
    .map_err(|error| format!("dolphin ctc head runtime build failed: {error}"))?;
    let rescore = DolphinDecoderRescoreRuntime::new(&decoder_config, weights, backend)
        .map_err(|error| format!("dolphin rescore runtime build failed: {error}"))?;
    Ok(DolphinPreparedRuntime {
        backend,
        encoder,
        ctc_head,
        rescore,
    })
}

/// The complete Dolphin transcribe pipeline over 16 kHz mono PCM (`samples` in
/// `[-1, 1]`): fbank + CMVN -> encoder -> CTC/attention joint decode -> detokenize.
/// Loads the pack's weights from `reader` and builds fresh prepared runtimes
/// each call (the uncached path the parity harness drives); the executor keeps
/// admitted weights and owner-thread [`DolphinPreparedRuntime`] actors to reuse
/// both across requests.
pub(crate) fn transcribe_dolphin_pcm(
    reader: &GgufTensorDataReader,
    metadata: &GgufMetadata,
    samples: &[f32],
    ctc_weight: f32,
    backend: GgmlCpuGraphBackend,
    language: Option<&str>,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<DolphinPipelineOutput, String> {
    let weights = load_dolphin_runtime_weights_from_pack(reader)
        .map_err(|error| format!("dolphin runtime weight load failed: {error}"))?;
    let mut prepared = build_dolphin_prepared_runtime(&weights, metadata, backend)?;
    run_dolphin_pipeline(
        &mut prepared,
        &weights,
        metadata,
        samples,
        ctc_weight,
        language,
        phrase_bias,
    )
}

/// Parse the pack's `dolphin.language.scheme` metadata (see [`LANGUAGE_SCHEME_KEY`])
/// into the typed [`DolphinLanguageScheme`] that dispatches the decode-prefix
/// builder, audio frontend, and encoder rel-pos-attention scheme. A **missing**
/// key is an intentional backward-compat default to `CnDialect` -- every pack
/// baked before this key existed (both originally published dolphin packs) is
/// cn-dialect. A key that IS **present** but holds anything other than the two
/// recognized values fails closed with a typed error instead of silently
/// falling back: a corrupt or future-versioned pack must never be silently
/// misdispatched to the wrong frontend/attention scheme.
fn parse_dolphin_language_scheme(metadata: &GgufMetadata) -> Result<DolphinLanguageScheme, String> {
    parse_dolphin_language_scheme_value(metadata.get_string(LANGUAGE_SCHEME_KEY))
}

/// The string-level half of [`parse_dolphin_language_scheme`], split out so a
/// test can pin it against [`DolphinLanguageScheme::label`] (the importer's
/// writer) without needing to construct a [`GgufMetadata`] -- the two literal
/// sets (writer labels here, reader match arms in `package_import.rs`) must
/// never drift out of sync.
fn parse_dolphin_language_scheme_value(
    value: Option<&str>,
) -> Result<DolphinLanguageScheme, String> {
    match value {
        None => Ok(DolphinLanguageScheme::CnDialect),
        Some("cn_dialect") => Ok(DolphinLanguageScheme::CnDialect),
        Some("multilingual") => Ok(DolphinLanguageScheme::Multilingual),
        Some(other) => Err(format!(
            "dolphin pack has unrecognized '{LANGUAGE_SCHEME_KEY}' value {other:?} \
             (expected 'cn_dialect' or 'multilingual')"
        )),
    }
}

/// Run the fbank+CMVN -> encoder -> joint-decode -> detokenize pipeline over
/// already-loaded `weights` and already-`prepared` graph runtimes. Split out
/// from [`transcribe_dolphin_pcm`] so the executor can reuse both across
/// requests without re-dequantizing or re-uploading.
pub(crate) fn run_dolphin_pipeline(
    prepared: &mut DolphinPreparedRuntime,
    weights: &DolphinRuntimeWeights,
    metadata: &GgufMetadata,
    samples: &[f32],
    ctc_weight: f32,
    language: Option<&str>,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<DolphinPipelineOutput, String> {
    let backend = prepared.backend;
    let tokens = metadata
        .get_string_array(TOKENIZER_TOKENS_KEY)
        .ok_or_else(|| format!("dolphin pack is missing the '{TOKENIZER_TOKENS_KEY}' vocab"))?;
    // The pack's own `dolphin.language.scheme` (absent on every pack predating
    // this key, which defaults to the cn-dialect family -- both originally
    // published dolphin packs are cn-dialect). This single signal now
    // dispatches three things that all trace back to which of the two
    // DataoceanAI training pipelines produced the checkpoint: the decode
    // prefix builder (below), the audio frontend, and the encoder's
    // relative-position-attention scheme (`DolphinEncoderConfig`). See
    // `parse_dolphin_language_scheme` for the fail-closed handling of a
    // present-but-unrecognized value.
    let language_scheme = parse_dolphin_language_scheme(metadata)?;
    // Build the `<sos> <lang> <region> <asr> <notimestamp>` prefix per request
    // from the pack vocab; fail closed (typed) on an unknown code or a missing
    // language/region token.
    let prefix = match language_scheme {
        DolphinLanguageScheme::Multilingual => {
            build_dolphin_multilingual_decode_prefix(tokens, language).map_err(|error| {
                format!("dolphin multilingual decode prefix build failed: {error}")
            })?
        }
        DolphinLanguageScheme::CnDialect => build_dolphin_decode_prefix(tokens, language)
            .map_err(|error| format!("dolphin decode prefix build failed: {error}"))?,
    };
    let eos_token_id = metadata
        .get_u32(EOS_TOKEN_ID_KEY)
        .ok_or_else(|| format!("dolphin pack is missing '{EOS_TOKEN_ID_KEY}'"))?;
    let blank_token_id = metadata
        .get_u32(BLANK_TOKEN_ID_KEY)
        .ok_or_else(|| format!("dolphin pack is missing '{BLANK_TOKEN_ID_KEY}'"))?;
    // Structural hparams (d_model/heads/FFN/layer counts/...) come from the
    // pack's own runtime contract, never a fixed `small.cn` shape -- this is
    // what lets base/small/multilingual checkpoints of any width share this
    // one pipeline. `execute()`'s Gate-0 already fail-closed-validated this
    // parses; re-parsing here (instead of threading the result through) keeps
    // this function's signature stable for its other caller
    // (`encode_dolphin_encoder_from_pack`'s parity test, which intentionally
    // stays pinned to `small_cn()`).
    let execution_metadata = parse_dolphin_execution_metadata(metadata, weights)
        .map_err(|error| format!("dolphin runtime metadata contract failed: {error}"))?;

    // Frontend: kaldi fbank (cn-dialect) or the ESPnet DefaultFrontend
    // (multilingual) -> global CMVN (the exact tensor the encoder consumes).
    // See `frontend::DolphinEspnetFrontend`'s doc comment for why these two
    // checkpoints need materially different feature pipelines.
    let mut features = match language_scheme {
        DolphinLanguageScheme::CnDialect => DolphinFbankFrontend::new().compute(samples),
        DolphinLanguageScheme::Multilingual => DolphinEspnetFrontend::new().compute(samples),
    }
    .map_err(|error| format!("dolphin frontend failed: {error}"))?;
    let cmvn_mean = weights
        .tensor(CMVN_MEAN_TENSOR)
        .ok_or_else(|| format!("dolphin pack is missing '{CMVN_MEAN_TENSOR}'"))?;
    let cmvn_istd = weights
        .tensor(CMVN_ISTD_TENSOR)
        .ok_or_else(|| format!("dolphin pack is missing '{CMVN_ISTD_TENSOR}'"))?;
    apply_global_cmvn(&mut features.data, features.n_mels, cmvn_mean, cmvn_istd)
        .map_err(|error| format!("dolphin global CMVN failed: {error}"))?;

    // Encoder (parity-verified for small.cn; shape-derived for every size;
    // `language_scheme` picks the rel-pos-attention flavor -- see
    // `DolphinEncoderConfig`'s doc comment). The prepared runtime's weights
    // are already backend-resident; only when an utterance outgrows its
    // resident position table (multilingual scheme only, whose table is
    // computed rather than baked -- see `DolphinEncoderRuntime`) does this
    // fall back to a one-shot runtime sized for exactly this call, matching
    // the pre-runtime per-call behavior. Production transcription only ever
    // reads `encoder.encoder_out` below; `after_subsample`/per-block taps
    // exist solely for `#[cfg(test)]` parity, so they stay off here (P6).
    let encoder = if prepared.encoder.supports_input_frames(features.n_frames) {
        prepared
            .encoder
            .encode(&features.data, features.n_frames, false)
    } else {
        let encoder_config =
            DolphinEncoderConfig::from_execution_metadata(&execution_metadata, language_scheme);
        encode(
            &encoder_config,
            weights,
            &features.data,
            features.n_frames,
            backend,
            false,
        )
    }
    .map_err(|error| format!("dolphin encoder graph failed: {error}"))?;

    // Hotword deep-biasing (native `context_module.*` fusion). Upstream's
    // `decode()` computes `ctc_logprobs` from the *unbiased* encoder output
    // before `apply_deep_biasing` replaces it, so only the decoder's
    // `attention_rescoring` input is biased -- the CTC prefix-beam n-best below
    // always reads `encoder.encoder_out` unchanged. When no hotwords are
    // requested this borrows the same buffer, so the no-hotword path pays no
    // extra graph build/copy (byte-identical to before this feature).
    let rescoring_encoder_out: std::borrow::Cow<'_, [f32]> = match phrase_bias {
        Some(config) if !config.is_empty() => {
            let hotword_token_ids: Vec<Vec<u32>> = config
                .entries()
                .iter()
                .map(|entry| tokenize_hotword_phrase(tokens, entry.phrase()))
                .collect();
            let context_emb = encode_hotword_context_embeddings(weights, &hotword_token_ids)
                .map_err(|error| format!("dolphin hotword context embedding failed: {error}"))?;
            let biased = apply_hotword_deep_biasing(
                weights,
                &encoder.encoder_out,
                encoder.frames,
                &context_emb,
                backend,
            )
            .map_err(|error| format!("dolphin hotword biasing fusion failed: {error}"))?;
            std::borrow::Cow::Owned(biased)
        }
        _ => std::borrow::Cow::Borrowed(encoder.encoder_out.as_slice()),
    };

    // CTC/attention joint decode over the prepared (weight-resident) CTC head
    // and rescore runtimes.
    let decode_config = DolphinJointDecodeConfig {
        beam_size: DOLPHIN_BEAM_SIZE,
        ctc_weight,
        prompt_prefix: prefix.token_ids,
        eos_token_id,
        blank_token_id,
    };
    let decoded = joint_decode(
        &mut prepared.ctc_head,
        &mut prepared.rescore,
        &encoder.encoder_out,
        &rescoring_encoder_out,
        encoder.frames,
        &decode_config,
    )
    .map_err(|error| format!("dolphin joint decode failed: {error}"))?;

    let text = detokenize_char_tokens(&decoded.best_token_ids, tokens);
    let ctc_greedy_text = detokenize_char_tokens(&decoded.ctc_greedy_token_ids, tokens);
    let scored_nbest = decoded
        .scored_nbest
        .iter()
        .map(|hyp| DolphinScoredText {
            text: detokenize_char_tokens(&hyp.token_ids, tokens),
            ctc_score: hyp.ctc_score,
            attention_score: hyp.attention_score,
            combined_score: hyp.combined_score,
        })
        .collect();

    Ok(DolphinPipelineOutput {
        text,
        best_token_ids: decoded.best_token_ids,
        ctc_greedy_text,
        scored_nbest,
        resolved_language: prefix.resolved_language,
    })
}

const DOLPHIN_RUNTIME_MAX_IDLE_ENTRIES: usize = 4;
const DOLPHIN_RUNTIME_MAX_INSTANCES_PER_KEY: usize = 2;

/// Dedicated `GgmlAsrViewExecutor` for the Dolphin family (DedicatedRuntimeExecutorV1).
/// All resident host weights and backend-bound graph runtimes are owned by this
/// service-scoped executor; no process-global pool or worker TLS can outlive its
/// memory admission lease.
#[derive(Clone)]
pub(crate) struct DolphinGgmlExecutor {
    weights: Arc<DolphinRuntimeWeightsCache>,
    prepared_runtimes: Arc<DolphinPreparedRuntimePool>,
}

impl std::fmt::Debug for DolphinGgmlExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DolphinGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for DolphinGgmlExecutor {
    fn default() -> Self {
        let host_budget = crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        Self {
            weights: Arc::new(AdmittedHostObjectCache::new(
                AdmittedHostObjectCacheLimits::new(DOLPHIN_RUNTIME_MAX_IDLE_ENTRIES, host_budget),
            )),
            prepared_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-dolphin-runtime-owner",
                AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                    DOLPHIN_RUNTIME_MAX_IDLE_ENTRIES,
                    host_budget,
                    DOLPHIN_RUNTIME_MAX_INSTANCES_PER_KEY,
                ),
            )),
        }
    }
}

impl DolphinGgmlExecutor {
    fn cached_runtime_weights(
        &self,
        runtime_source: &crate::GgmlRuntimeSource,
        reader: &GgufTensorDataReader,
    ) -> Result<AdmittedHostObject<DolphinRuntimeWeights>, String> {
        let key = crate::models::runtime_cache_coordinator::PackContentKey::for_runtime_source(
            runtime_source,
        );
        let content_id = runtime_source.content_id().to_string();
        self.weights.get_or_try_insert_with(
            key,
            || {
                let (peak_bytes, retained_bytes) =
                    DolphinRuntimeWeights::quoted_system_memory_bytes(reader)?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("dolphin-runtime-weights:{content_id}"),
                    peak_bytes,
                    retained_bytes,
                )
                .map_err(|error| error.to_string())?;
                Ok((retained_bytes, quote))
            },
            |quote| match SystemMemoryOwner::try_allocate_transaction(quote, || {
                let weights = load_dolphin_runtime_weights_from_pack(reader)
                    .map_err(|error| format!("dolphin runtime weight load failed: {error}"))?;
                let retained = weights.retained_system_memory_bytes()?;
                Ok(SystemMemoryAllocationOutcome::new(
                    weights, retained, retained,
                ))
            }) {
                Ok(owner) => Ok(Arc::new(owner)),
                Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(error.to_string())
                }
            },
            || "dolphin runtime weight cache lock is poisoned".to_string(),
        )
    }

    fn checkout_prepared_runtime(
        &self,
        runtime_source: &crate::GgmlRuntimeSource,
        metadata: Arc<GgufMetadata>,
        weights: AdmittedHostObject<DolphinRuntimeWeights>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<DolphinPreparedRuntimeActor, String> {
        let key = (
            crate::models::runtime_cache_coordinator::PackContentKey::for_runtime_source(
                runtime_source,
            ),
            crate::models::native_execution_services::current_execution_lane_key(backend),
        );
        let quote_weights = Arc::clone(&weights);
        let quote_metadata = Arc::clone(&metadata);
        let content_id = runtime_source.content_id().to_string();
        self.prepared_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let (peak_bytes, retained_bytes) =
                    DolphinPreparedRuntime::quoted_system_memory_bytes(
                        &quote_weights,
                        &quote_metadata,
                    )?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("dolphin-prepared-runtime:{content_id}"),
                    peak_bytes,
                    retained_bytes,
                )
                .map_err(|error| error.to_string())?;
                Ok((retained_bytes, (quote, weights, metadata)))
            },
            move |(quote, weights, metadata)| match SystemMemoryOwner::try_allocate_transaction(
                quote,
                || {
                    let runtime = build_dolphin_prepared_runtime(&weights, &metadata, backend)?;
                    let retained = runtime.retained_system_memory_bytes()?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        DolphinPreparedRuntimeActorState {
                            runtime,
                            _weights: weights,
                        },
                        retained,
                        retained,
                    ))
                },
            ) {
                Ok(owner) => Ok(owner),
                Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(error.to_string())
                }
            },
            |error: PinnedRuntimeActorError| error.to_string(),
        )
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.prepared_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.weights
            .evict_where(|key| key.pack_content_id == pack_content_id);
    }

    fn clear_runtime_state(&self) {
        self.prepared_runtimes.clear();
        self.weights.clear();
    }
}

impl GgmlAsrViewExecutor for DolphinGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::DOLPHIN_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // Native deep-biasing over the `context_module.*` tensors (see
        // `models::dolphin::hotword_context`); the phrase list feeds the trained
        // context extractor + biasing attention fusion. Per-phrase `boost` has no
        // upstream semantics here (see that module's docs) and is ignored.
        true
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::DOLPHIN_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Gate-0: validate the runtime source and load its metadata + tensor index.
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        // Fail closed on an incomplete pack (missing runtime scalar keys).
        parse_dolphin_execution_metadata(&preflight.metadata, preflight.tensor_index.as_ref())
            .map_err(|error| fail(format!("dolphin runtime metadata contract failed: {error}")))?;
        // Resolve the language scheme once here (fail closed on an unrecognized
        // value at Gate-0, before any decode work); `run_dolphin_pipeline` below
        // re-derives the same result from the same metadata key rather than
        // threading it through, per its own doc comment.
        let language_scheme = parse_dolphin_language_scheme(&preflight.metadata).map_err(fail)?;
        // Confirm the encoder + CTC namespaces are actually baked before decoding.
        let mut sentinels = ENCODER_SENTINEL_TENSORS.to_vec();
        if language_scheme == DolphinLanguageScheme::CnDialect {
            sentinels.push(ENCODER_CN_DIALECT_SENTINEL_TENSOR);
        }
        for sentinel in sentinels {
            if preflight.tensor_index.get(sentinel).is_none() {
                return Err(fail(format!(
                    "dolphin pack is missing required tensor '{sentinel}'"
                )));
            }
        }

        // Resolved once by whoever built this request (this architecture's
        // `auto_gpu_policy = AllBackends`), carried as an explicit field --
        // never re-derived from a thread-local here.
        let backend = request.resolved_runtime.backend();
        let reader = GgufTensorDataReader::from_runtime_source(&preflight.runtime_source)
            .map_err(|error| fail(format!("dolphin pack tensor reader failed: {error}")))?;
        // The immutable dequantized/native-mapped table is admitted once per
        // content id. Backend-bound mutable runtimes are then checked out from
        // owner-thread actors keyed by the exact content and execution lane.
        let weights = self
            .cached_runtime_weights(&preflight.runtime_source, &reader)
            .map_err(fail)?;
        let actor = self
            .checkout_prepared_runtime(
                &preflight.runtime_source,
                Arc::clone(&preflight.metadata),
                weights,
                backend,
            )
            .map_err(fail)?;
        let metadata = Arc::clone(&preflight.metadata);
        let samples = request.prepared_audio.samples_f32.to_owned_pcm_slice();
        let language = request.request_options.language.clone();
        let phrase_bias = request.request_options.phrase_bias.clone();
        // Thread the request language into the decode prefix builder; an
        // unsupported code / missing region token fails closed here (typed).
        let output = actor
            .call_mut(move |state| {
                run_dolphin_pipeline(
                    &mut state.runtime,
                    &state._weights,
                    &metadata,
                    &samples,
                    DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
                    language.as_deref(),
                    phrase_bias.as_ref(),
                )
            })
            .map_err(|error| fail(error.to_string()))?
            .map_err(fail)?;

        let duration = request.prepared_audio.samples_f32.len() as f32
            / request.prepared_audio.sample_rate_hz.max(1) as f32;
        let segments = if output.text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: duration,
                text: output.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: output.text,
                segments,
                longform: None,
                // Surface the region/language the prefix actually selected (the
                // model does not detect it, but the selection is a genuine input);
                // the transcribe layer prefers this per the SpecifyOnly mode.
                language: Some(output.resolved_language),
            },
            carry_context: None,
            decode_truncation: None,
        })
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_state();
    }
}

const DOLPHIN_STREAMING_EXECUTOR_ID: &str = "dolphin-ggml-snapshot-streaming-executor-v1";

impl GgmlAsrStreamingExecutor for DolphinGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        DOLPHIN_STREAMING_EXECUTOR_ID
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_state();
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: DOLPHIN_STREAMING_EXECUTOR_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Resolve the pack's language scheme once here so the streaming driver
        // can be told the minimum raw-sample count its frontend + encoder can
        // turn into output (see `minimum_encodable_samples` below): a trailing
        // window shorter than the Conv2dSubsampling4 receptive field (7 mel
        // frames, no padding) reaches `ggml_conv_2d`'s `im2col` precondition
        // and aborts the whole process instead of returning a Rust error (the
        // idle_unload short-press crash), so the driver must skip that decode
        // call entirely rather than rely on catching an error afterward.
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let language_scheme = parse_dolphin_language_scheme(&preflight.metadata).map_err(fail)?;
        let min_frames = minimum_subsample_input_frames();
        let min_samples = match language_scheme {
            DolphinLanguageScheme::CnDialect => kaldi_min_samples_for_frames(min_frames),
            DolphinLanguageScheme::Multilingual => espnet_min_samples_for_frames(min_frames),
        };
        // Dolphin has no cheap CTC-greedy partial surface (the pipeline output
        // exposes only the rescored transcript), so partials re-decode the
        // trailing window through the same offline joint decode as the FINAL.
        // The shared re-decode session (used by every non-frame-sync family)
        // keeps the FINAL byte-identical to `execute()`; only the partial cadence
        // differs. Its adaptive throttle absorbs the heavier per-partial cost.
        build_seq2seq_streaming_session(
            self.clone(),
            DOLPHIN_STREAMING_EXECUTOR_ID,
            DOLPHIN_GGML_ADAPTER_ID,
            "dolphin",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT.with_minimum_encodable_samples(min_samples),
            <DolphinGgmlExecutor as GgmlAsrViewExecutor>::execute_view,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::dolphin::package_import::{
        DolphinImportRequest, DolphinQuantizationMode,
        convert_local_dolphin_wenet_source_to_runtime_pack,
    };
    use crate::models::runtime_cache_coordinator::PackContentKey;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    fn cached_placeholder(
        executor: &DolphinGgmlExecutor,
        key: PackContentKey,
    ) -> AdmittedHostObject<DolphinRuntimeWeights> {
        executor
            .weights
            .get_or_try_insert_with(
                key,
                || Ok::<_, String>((0, ())),
                |()| {
                    Ok(Arc::new(
                        SystemMemoryOwner::with_committed_requested_bytes_for_test(
                            DolphinRuntimeWeights::default(),
                            0,
                        ),
                    ))
                },
                || "poisoned".to_string(),
            )
            .expect("cache placeholder")
    }

    /// Writes a minimal valid GGUF-magic fixture: `PackContentKey` now only
    /// resolves from an already-open `GgmlRuntimeSource`, which only ever
    /// admits GGUF-magic files.
    fn write_gguf_fixture(path: &Path, payload: &[u8]) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(payload);
        std::fs::write(path, bytes).expect("write fixture");
    }

    fn key_for(path: &Path) -> PackContentKey {
        let source = crate::validate_ggml_runtime_source_path(path).expect("validate source");
        PackContentKey::for_runtime_source(&source)
    }

    /// Same-path A/B content swap must not hit the previous weight table. Uses a
    /// minimal GGUF so the pool insert path is exercised without a full Dolphin
    /// checkpoint (load may fail on missing tensors; we only need the content-key
    /// lookup/miss behavior, so we seed the pool map directly with Arc placeholders).
    #[test]
    fn dolphin_weights_pool_keys_by_content_id_not_path() {
        let executor = DolphinGgmlExecutor::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.oasr");
        write_gguf_fixture(&path, b"dolphin-content-a");
        let key_a = key_for(&path);
        write_gguf_fixture(&path, b"dolphin-content-b-different");
        let key_b = key_for(&path);
        assert_ne!(key_a.pack_content_id, key_b.pack_content_id);
        assert_ne!(key_a, key_b);

        let cached_a = cached_placeholder(&executor, key_a.clone());
        let cached_a_again = cached_placeholder(&executor, key_a.clone());
        let cached_b = cached_placeholder(&executor, key_b);
        assert!(Arc::ptr_eq(&cached_a, &cached_a_again));
        assert!(!Arc::ptr_eq(&cached_a, &cached_b));

        // Idle unload must drop the pool entirely.
        executor.clear_runtime_state();
        assert_eq!(executor.weights.usage_for_test(), (0, 0));
    }

    /// No global invalidation: evicting one pack's content id (the pull
    /// install/replace path) must not disturb a *different*, unrelated
    /// pack's resident entry. This is the direct regression test for the
    /// audited bug -- a single shared-epoch bump used to invalidate every
    /// resident content identity in the process at once.
    #[test]
    fn evicting_one_content_id_leaves_a_different_pack_resident() {
        let executor = DolphinGgmlExecutor::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path_one = dir.path().join("pack-one.oasr");
        let path_two = dir.path().join("pack-two.oasr");
        write_gguf_fixture(&path_one, b"dolphin-pack-one-bytes");
        write_gguf_fixture(&path_two, b"dolphin-pack-two-different-bytes");

        let key_one = key_for(&path_one);
        let key_two = key_for(&path_two);
        assert_ne!(key_one, key_two);

        let cached_one = cached_placeholder(&executor, key_one.clone());
        let cached_two = cached_placeholder(&executor, key_two.clone());

        // The invalidation action: evict pack one's content id only (what
        // `pull` does with the old content id after an install/replace).
        executor.evict_prepared_runtime_content_id(&key_one.pack_content_id);

        let rebuilt_one = cached_placeholder(&executor, key_one);
        let cached_two_again = cached_placeholder(&executor, key_two);
        assert!(!Arc::ptr_eq(&cached_one, &rebuilt_one));
        assert!(Arc::ptr_eq(&cached_two, &cached_two_again));
    }

    #[test]
    fn dolphin_unload_idle_state_clears_weights_pool() {
        let executor = DolphinGgmlExecutor::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pool.oasr");
        write_gguf_fixture(&path, b"dolphin-pool-seed");
        let key = key_for(&path);
        let _cached = cached_placeholder(&executor, key);
        assert_eq!(executor.weights.usage_for_test(), (1, 0));

        <DolphinGgmlExecutor as GgmlAsrViewExecutor>::unload_idle_state(&executor);
        assert_eq!(executor.weights.usage_for_test(), (0, 0));
    }

    fn root() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_DOLPHIN_PARITY_ROOT",
            "Dolphin parity fixture directory",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    #[test]
    fn language_scheme_label_round_trips_through_the_executor_parser() {
        for scheme in [
            DolphinLanguageScheme::CnDialect,
            DolphinLanguageScheme::Multilingual,
        ] {
            assert_eq!(
                parse_dolphin_language_scheme_value(Some(scheme.label())),
                Ok(scheme),
                "label {:?} must parse back to {scheme:?}",
                scheme.label()
            );
        }
        assert_eq!(
            parse_dolphin_language_scheme_value(None),
            Ok(DolphinLanguageScheme::CnDialect),
            "a missing scheme key must keep defaulting to cn-dialect for pre-existing packs"
        );
        assert!(
            parse_dolphin_language_scheme_value(Some("bogus"))
                .unwrap_err()
                .contains("bogus"),
            "an unrecognized scheme value must fail closed rather than silently default"
        );
    }
    /// Golden `attention_rescoring` transcript (manifest `text_nospecial`): the
    /// model's own joint-decode output for the Sichuan clip. This is the parity
    /// target -- the human ground-truth WSC transcript differs by one homophone
    /// (河 vs 和), a model-accuracy gap, not an implementation gap.
    const REFERENCE_RESCORING_TEXT: &str = "学校和底下好多那种野生枸杞";
    /// Human ground-truth transcript (manifest `reference_transcript_wsc`).
    const REFERENCE_WSC_TEXT: &str = "学校河底下好多那种野生枸杞";
    /// Reference CTC greedy transcript (manifest `ctc_greedy_search.text`).
    const REFERENCE_CTC_GREEDY_TEXT: &str = "学校火底下好多那种野生枸杞";

    // --- minimal little-endian f32 .npy reader (mirrors parity.rs) -------------
    fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect("read npy");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
            .expect("npy header utf8");
        assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
        assert!(
            header.contains("'fortran_order': False"),
            "expected C order"
        );
        let shape_start = header.find("'shape':").expect("shape key");
        let paren = header[shape_start..].find('(').unwrap() + shape_start;
        let close = header[paren..].find(')').unwrap() + paren;
        let shape: Vec<usize> = header[paren + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = header_start + header_len;
        let values: Vec<f32> = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, values)
    }

    fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        actual
            .iter()
            .zip(expected)
            .fold(0.0f32, |m, (a, e)| m.max((a - e).abs()))
    }

    fn relative_max_diff(actual: &[f32], expected: &[f32]) -> f32 {
        let max = max_abs_diff(actual, expected);
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if scale > 0.0 { max / scale } else { max }
    }

    /// Char-level edit distance (Levenshtein) over Unicode scalar values.
    fn char_edit_distance(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for (i, &ca) in a.iter().enumerate() {
            cur[0] = i + 1;
            for (j, &cb) in b.iter().enumerate() {
                let cost = usize::from(ca != cb);
                cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    fn char_error_rate(hypothesis: &str, reference: &str) -> f32 {
        let ref_len = reference.chars().count();
        if ref_len == 0 {
            return if hypothesis.is_empty() { 0.0 } else { 1.0 };
        }
        char_edit_distance(hypothesis, reference) as f32 / ref_len as f32
    }

    /// The three quantization rungs the producer/consumer tests exercise, in the
    /// order they are reported. fp16 is the parity/CER-0 golden; q8_0 and q4_k are
    /// the size-shrunk rungs held to a documented dequant tolerance.
    const DOLPHIN_QUANT_RUNGS: [DolphinQuantizationMode; 3] = [
        DolphinQuantizationMode::Fp16,
        DolphinQuantizationMode::Q8_0,
        DolphinQuantizationMode::Q4_K,
    ];

    /// Produce-if-absent the `.oasr` pack for `quant` at its stable per-quant path,
    /// exactly once per process, and hand every caller the same path. The heavy
    /// `#[ignore]` tests below (producers + consumers over the three rungs) share
    /// this so a fresh checkout converts each rung exactly once and later callers
    /// reuse the result.
    ///
    /// The write is atomic: the pack is built into a uniquely-named temp file in
    /// the packs dir and then `rename`d into place. Same-directory rename is
    /// atomic on the local fs, so a reader that opens the stable path never
    /// observes a half-written or missing pack -- the path always resolves to a
    /// complete pack (the previous one, or the freshly renamed one), and a reader
    /// holding an fd keeps reading its complete inode across the swap. This is
    /// what removes the earlier producer/consumer race (the old producer did
    /// `remove_file` + in-place rewrite, opening a window where the consumer read
    /// an absent/torn pack); the `dolphin-pack` nextest test-group additionally
    /// serializes the tests so they never even overlap. Returns `None` when the
    /// local checkpoint is absent (the tests skip).
    fn ensure_dolphin_pack(root: &Path, quant: DolphinQuantizationMode) -> Option<PathBuf> {
        static PACKS: OnceLock<Mutex<HashMap<DolphinQuantizationMode, Option<PathBuf>>>> =
            OnceLock::new();
        let mut memo = PACKS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        memo.entry(quant)
            .or_insert_with(|| produce_dolphin_pack_atomic(root, quant))
            .clone()
    }

    fn produce_dolphin_pack_atomic(root: &Path, quant: DolphinQuantizationMode) -> Option<PathBuf> {
        let pack = root.join(format!(
            "packs/dolphin-cn-dialect-small-{}.oasr",
            quant.label()
        ));
        if pack.exists() {
            return Some(pack);
        }
        let safetensors = root.join("weights/full.safetensors");
        let units = root.join("src/units.txt");
        if !safetensors.exists() || !units.exists() {
            return None;
        }
        let packs_dir = pack.parent().expect("pack has a parent dir");
        std::fs::create_dir_all(packs_dir).expect("create packs dir");
        // Reserve a uniquely-named temp `.oasr` path in the same dir (the `.oasr`
        // suffix keeps the importer's output-extension gate happy; the unique name
        // means two concurrent producers in distinct processes never collide).
        // The GGUF writer refuses to clobber an existing file, so drop the empty
        // reservation file and let the writer create it fresh; `TempPath` still
        // cleans it up on an early return, and `persist` publishes it with an
        // atomic same-dir rename.
        let temp_path = tempfile::Builder::new()
            .prefix(".dolphin-pack-")
            .suffix(".oasr")
            .tempfile_in(packs_dir)
            .expect("create temp pack")
            .into_temp_path();
        std::fs::remove_file(&temp_path).expect("clear temp reservation");
        convert_local_dolphin_wenet_source_to_runtime_pack(&DolphinImportRequest {
            safetensors_path: safetensors,
            units_path: units,
            output_path: temp_path.to_path_buf(),
            model_id: "dolphin-cn-dialect-small".to_string(),
            quantization: quant,
            language_scheme:
                crate::models::dolphin::package_import::DolphinLanguageScheme::CnDialect,
        })
        .expect("dolphin import");
        temp_path.persist(&pack).expect("publish dolphin pack");
        Some(pack)
    }

    /// Per-quant encoder-from-pack tolerance on the scale-invariant relative max
    /// diff of `encoder_out` vs the golden. fp16 now binds its rank-2 `.weight`
    /// operands as GGML_TYPE_F16 in-graph (keep-quantized), so the matmuls round
    /// activations through f16 -- a small, deliberate lossy step above the raw-f32
    /// bit-exact gate that stays in `parity::dolphin_encoder_parity`. q8_0/q4_k add
    /// per-block weight quantization on those same rank-2 matrices. The bounds sit a
    /// few x above the measured relative max diff on the committed golden (see the
    /// eprintln), enough headroom for thread-order jitter while an algorithmic/layout
    /// bug -- which blows the diff up by orders of magnitude -- still trips the gate.
    fn encoder_from_pack_rel_tolerance(quant: DolphinQuantizationMode) -> f32 {
        match quant {
            DolphinQuantizationMode::Fp16 => 3.0e-3,
            DolphinQuantizationMode::Q8_0 => 5.0e-2,
            DolphinQuantizationMode::Q4_K => 2.5e-1,
            // Dolphin's importer only ever produces fp16/q8_0/q4_k (see
            // `DOLPHIN_QUANT_RUNGS`); q3_k is unreachable for this family even
            // though the shared `PackQuant` enum also carries it for qwen.
            DolphinQuantizationMode::Q3_K => unreachable!("dolphin never produces a q3_k pack"),
        }
    }

    /// Per-quant tolerance on the scale-invariant relative max diff of a quant
    /// rung's `encoder_out` vs the **fp16-from-pack** reference (not the golden):
    /// this is the "per-logit tolerance vs fp16" gate for the keep-quantized rungs.
    /// q8_0 sits ~1e-2, q4_k looser; both are a few x above the measured spread.
    fn encoder_vs_fp16_rel_tolerance(quant: DolphinQuantizationMode) -> f32 {
        match quant {
            DolphinQuantizationMode::Fp16 => 0.0,
            DolphinQuantizationMode::Q8_0 => 5.0e-2,
            DolphinQuantizationMode::Q4_K => 2.5e-1,
            DolphinQuantizationMode::Q3_K => unreachable!("dolphin never produces a q3_k pack"),
        }
    }

    /// Produce each `.oasr` rung (fp16/q8_0/q4_k) from the local WeNet checkpoint and
    /// assert the encoder-from-pack matches the golden `encoder_out` within that
    /// rung's tolerance. This is the convert+load gate: every rung loads, clears the
    /// fail-closed install gate, and its encoder rank-2 `.weight` operands bind
    /// **natively** under their WeNet names -- fp16 as GGML_TYPE_F16, q8_0/q4_k as
    /// their reversed-dim block-quant types fed straight to `mul_mat` -- and the
    /// verified encoder graph reproduces the golden output. Additionally gates each
    /// quant rung's `encoder_out` against the fp16-from-pack reference (per-logit
    /// tolerance vs fp16). The f32-exact bit-level gate stays in
    /// `parity::dolphin_encoder_parity` (raw safetensors, f32xf32).
    ///
    /// `#[ignore]`: needs the 1.7 GB checkpoint under `tmp/publish` (never
    /// committed). Run with:
    /// `cargo test -p openasr-core dolphin_encoder_from_pack_parity -- --ignored --nocapture`
    #[test]
    #[ignore = "requires local Dolphin checkpoint + golden under tmp/publish (not committed)"]
    fn dolphin_encoder_from_pack_parity() {
        let Some(root) = root() else {
            return;
        };
        if ensure_dolphin_pack(&root, DolphinQuantizationMode::Fp16).is_none() {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        }

        let (in_shape, features) = load_npy_f32(&root.join("golden/logmel_feats_cmvn.npy"));
        assert_eq!(in_shape.len(), 3, "expected (1,T,80), got {in_shape:?}");
        let frames_in = in_shape[1];
        let (_, golden_out) = load_npy_f32(&root.join("golden/encoder_out.npy"));

        // fp16-from-pack reference `encoder_out`, captured for the per-logit
        // tolerance the quant rungs are held to.
        let mut fp16_encoder_out: Vec<f32> = Vec::new();
        for quant in DOLPHIN_QUANT_RUNGS {
            let pack =
                ensure_dolphin_pack(&root, quant).expect("pack builds when checkpoint present");

            // The produced pack must clear the fail-closed install gate (adapter
            // selection + the dolphin runtime-metadata contract) exactly as
            // `openasr pull` would enforce it.
            crate::validate_native_runtime_model_pack_contract(&pack)
                .expect("dolphin pack must pass the native install gate");

            // Vocab is a property of the produced pack (the char tokenizer table the
            // importer baked from `units.txt`) -- unchanged across quant rungs.
            let pack_metadata =
                crate::ggml_runtime::read_gguf_metadata(&pack).expect("pack metadata");
            let vocab_size = pack_metadata
                .get_string_array(TOKENIZER_TOKENS_KEY)
                .expect("pack carries the tokenizer vocab")
                .len();
            assert_eq!(vocab_size, 18173);

            let pack_bytes = std::fs::metadata(&pack).expect("pack stat").len();
            let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
            let output = encode_dolphin_encoder_from_pack(
                &reader,
                &features,
                frames_in,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("encode");

            let max = max_abs_diff(&output.encoder_out, &golden_out);
            let rel = relative_max_diff(&output.encoder_out, &golden_out);
            let tolerance = encoder_from_pack_rel_tolerance(quant);
            let vs_fp16 = if quant == DolphinQuantizationMode::Fp16 {
                0.0
            } else {
                relative_max_diff(&output.encoder_out, &fp16_encoder_out)
            };
            eprintln!(
                "dolphin encoder-from-pack ({}): size {:.1}MB  max abs {max:.3e}  rel-vs-golden {rel:.3e}  (gate {tolerance:.1e})  rel-vs-fp16 {vs_fp16:.3e}  (gate {:.1e})",
                quant.label(),
                pack_bytes as f64 / 1.0e6,
                encoder_vs_fp16_rel_tolerance(quant),
            );
            assert!(
                rel < tolerance,
                "encoder-from-pack relative max diff {rel:.3e} exceeds the {} tolerance {tolerance:.1e}",
                quant.label()
            );
            if quant == DolphinQuantizationMode::Fp16 {
                fp16_encoder_out = output.encoder_out.clone();
            } else {
                let vs_fp16_tolerance = encoder_vs_fp16_rel_tolerance(quant);
                assert!(
                    vs_fp16 < vs_fp16_tolerance,
                    "encoder-from-pack ({}) relative max diff vs fp16 {vs_fp16:.3e} exceeds {vs_fp16_tolerance:.1e}",
                    quant.label()
                );
            }
        }
    }

    /// M1 CPU-vs-Metal x with/without weight-reuse AB harness. One config per
    /// invocation (selected by env) so `peak_rss_bytes` (process-global
    /// `ru_maxrss` high-water) is isolated per process; the driver script runs it
    /// 4x. Prints a machine-greppable `DOLPHIN_AB ...` line with best-of-N RTF and
    /// peak RSS. Never asserts a timing number (host-dependent); it only measures.
    ///
    /// Env: `OPENASR_DOLPHIN_AB_BACKEND=cpu|metal` (default cpu),
    /// `OPENASR_DOLPHIN_AB_QUANT=fp16|q8_0|q4_k` (default fp16),
    /// `OPENASR_DOLPHIN_AB_REUSE=0|1` (default 0 = rebuild pool each run),
    /// `OPENASR_DOLPHIN_AB_RUNS=<n>` (default 3).
    #[test]
    #[ignore = "perf AB harness: requires local Dolphin checkpoint + golden clip under tmp/publish"]
    fn dolphin_perf_ab() {
        use std::time::{Duration, Instant};
        let Some(root) = root() else {
            return;
        };
        let clip = root.join("golden/clip_sichuan.wav");
        // Quant rung under test (fp16 golden by default); the driver script sweeps
        // fp16/q8_0/q4_k so each is measured in its own process (isolated peak RSS).
        let quant = match std::env::var("OPENASR_DOLPHIN_AB_QUANT").as_deref() {
            Ok("q8_0") => DolphinQuantizationMode::Q8_0,
            Ok("q4_k") => DolphinQuantizationMode::Q4_K,
            _ => DolphinQuantizationMode::Fp16,
        };
        let Some(pack) = ensure_dolphin_pack(&root, quant) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };
        if !clip.exists() {
            eprintln!("skip: golden clip not present at {clip:?}");
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &clip,
            "dolphin AB",
            "clip_sichuan.wav",
        )
        .expect("load clip");
        let audio_s = samples.len() as f64 / 16_000.0;
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        let backend = match std::env::var("OPENASR_DOLPHIN_AB_BACKEND").as_deref() {
            Ok("metal") | Ok("gpu") => GgmlCpuGraphBackend::Metal,
            _ => GgmlCpuGraphBackend::Cpu,
        };
        let reuse = matches!(
            std::env::var("OPENASR_DOLPHIN_AB_REUSE").as_deref(),
            Ok("1")
        );
        let runs: usize = std::env::var("OPENASR_DOLPHIN_AB_RUNS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3)
            .max(1);
        let ctc_weight = DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT;

        // Reuse == build the weight pool + prepared graph runtimes once, reuse
        // across runs (the cached executor path). No-reuse == rebuild both
        // every run (the cold per-request cost). Best-of-N wall time isolates
        // the reuse delta.
        let mut preloaded = reuse.then(|| {
            let weights = load_dolphin_runtime_weights_from_pack(&reader).expect("weights");
            let prepared =
                build_dolphin_prepared_runtime(&weights, &metadata, backend).expect("prepared");
            (weights, prepared)
        });
        let mut best = Duration::MAX;
        let mut text = String::new();
        for _ in 0..runs {
            let started = Instant::now();
            let output = if let Some((weights, prepared)) = preloaded.as_mut() {
                run_dolphin_pipeline(
                    prepared,
                    weights,
                    &metadata,
                    &samples,
                    ctc_weight,
                    Some("zh-sichuan"),
                    None,
                )
            } else {
                let weights =
                    load_dolphin_runtime_weights_from_pack(&reader).expect("weights reload");
                let mut prepared = build_dolphin_prepared_runtime(&weights, &metadata, backend)
                    .expect("prepared reload");
                run_dolphin_pipeline(
                    &mut prepared,
                    &weights,
                    &metadata,
                    &samples,
                    ctc_weight,
                    Some("zh-sichuan"),
                    None,
                )
            }
            .expect("dolphin pipeline");
            best = best.min(started.elapsed());
            text = output.text;
        }
        let rtf = best.as_secs_f64() / audio_s;
        let peak_rss_mb = crate::metrics::peak_rss_bytes()
            .map(|bytes| bytes as f64 / 1.0e6)
            .unwrap_or(0.0);
        eprintln!(
            "DOLPHIN_AB quant={} backend={backend:?} reuse={reuse} runs={runs} audio={audio_s:.2}s \
             best={best:?} RTF={rtf:.3} peak_rss={peak_rss_mb:.0}MB text={text}",
            quant.label()
        );
    }

    /// Full end-to-end joint-decode harness: read the Sichuan clip, run
    /// fbank+CMVN -> encoder -> CTC/attention rescoring from the produced `.oasr`
    /// pack, print the transcript + CER, and assert the rescored transcript
    /// reproduces the golden `attention_rescoring` output exactly (CER 0).
    ///
    /// `#[ignore]`: needs the checkpoint/golden under `tmp/publish` (not committed).
    /// Run with:
    /// `cargo test -p openasr-core dolphin_joint_decode_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore = "requires local Dolphin checkpoint + golden clip under tmp/publish (not committed)"]
    fn dolphin_joint_decode_end_to_end() {
        let Some(root) = root() else {
            return;
        };
        let clip = root.join("golden/clip_sichuan.wav");
        let Some(pack) = ensure_dolphin_pack(&root, DolphinQuantizationMode::Fp16) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };
        if !clip.exists() {
            eprintln!("skip: golden clip not present at {clip:?}");
            return;
        }

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &clip,
            "dolphin end-to-end harness",
            "clip_sichuan.wav",
        )
        .expect("load clip");

        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        // Reference-faithful decode: attention-only selection over the CTC n-best
        // (ctc_weight 0.0), the WeNet `attention_rescoring` default. The Sichuan
        // clip is decoded under the `zh-sichuan` prefix -- the same
        // `<sos> <zh> <SICHUAN> <asr> <notimestamp>` ids the pack used to bake --
        // so the golden transcript stays bit-exact through the per-code builder.
        let output = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
            None,
        )
        .expect("dolphin transcribe");

        let cer_vs_rescoring = char_error_rate(&output.text, REFERENCE_RESCORING_TEXT);
        let cer_vs_wsc = char_error_rate(&output.text, REFERENCE_WSC_TEXT);

        eprintln!("== Dolphin CTC/attention joint decode (end-to-end) ==");
        eprintln!("transcript (rescored) : {}", output.text);
        eprintln!("reference (rescoring) : {REFERENCE_RESCORING_TEXT}");
        eprintln!("reference (human WSC) : {REFERENCE_WSC_TEXT}");
        eprintln!("ctc greedy            : {}", output.ctc_greedy_text);
        eprintln!("ctc greedy (reference): {REFERENCE_CTC_GREEDY_TEXT}");
        eprintln!(
            "CER vs rescoring ref  : {:.4}  ({} edits / {} chars)",
            cer_vs_rescoring,
            char_edit_distance(&output.text, REFERENCE_RESCORING_TEXT),
            REFERENCE_RESCORING_TEXT.chars().count()
        );
        eprintln!(
            "CER vs human WSC ref  : {:.4}  ({} edits / {} chars)",
            cer_vs_wsc,
            char_edit_distance(&output.text, REFERENCE_WSC_TEXT),
            REFERENCE_WSC_TEXT.chars().count()
        );
        eprintln!("rescored n-best (best-first):");
        for hyp in &output.scored_nbest {
            eprintln!(
                "  combined {:8.3}  attn {:8.3}  ctc {:8.3}  {}",
                hyp.combined_score, hyp.attention_score, hyp.ctc_score, hyp.text
            );
        }

        // Also report what the task-mentioned 0.3 rescoring weight would pick, to
        // show the training-vs-decode ctc_weight distinction concretely.
        let output_03 = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            0.3,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
            None,
        )
        .expect("dolphin transcribe (ctc_weight 0.3)");
        eprintln!(
            "with ctc_weight 0.3   : {}  (CER vs rescoring ref {:.4})",
            output_03.text,
            char_error_rate(&output_03.text, REFERENCE_RESCORING_TEXT)
        );

        // Sanity: the CTC greedy path reproduces the reference greedy transcript.
        assert_eq!(
            output.ctc_greedy_text, REFERENCE_CTC_GREEDY_TEXT,
            "CTC greedy transcript diverged from the reference"
        );
        // Parity: the rescored transcript reproduces the golden attention_rescoring
        // output exactly (the 河/和 homophone gap to the human WSC transcript is a
        // model-accuracy artifact the reference decode shares).
        assert_eq!(
            output.text, REFERENCE_RESCORING_TEXT,
            "rescored transcript diverged from the golden attention_rescoring output"
        );
        assert_eq!(
            cer_vs_rescoring, 0.0,
            "CER against the rescoring reference must be 0"
        );
        // The decoded region/language is surfaced honestly (not None).
        assert_eq!(output.resolved_language, "zh-sichuan");

        // Spot-check the quantized rungs transcribe the clip sensibly. fp16 is the
        // CER-0 golden above; q8_0/q4_k keep their rank-2 `.weight` operands
        // quantized in-graph (fed straight to `mul_mat`), trading a documented
        // quantization error for size, so they are held to a loose CER bound against
        // the rescoring reference rather than exact equality. This proves the
        // keep-quantized native bind produces a usable transcript end-to-end (not
        // just close encoder activations), and pins the size ordering
        // fp16 > q8_0 > q4_k.
        let fp16_bytes = std::fs::metadata(&pack).expect("fp16 stat").len();
        for (quant, cer_bound) in [
            (DolphinQuantizationMode::Q8_0, 0.10_f32),
            (DolphinQuantizationMode::Q4_K, 0.35_f32),
        ] {
            let qpack = ensure_dolphin_pack(&root, quant).expect("quant pack builds");
            let qbytes = std::fs::metadata(&qpack).expect("quant stat").len();
            let qreader = GgufTensorDataReader::from_path(&qpack).expect("quant reader");
            let qmetadata =
                crate::ggml_runtime::read_gguf_metadata(&qpack).expect("quant metadata");
            let qoutput = transcribe_dolphin_pcm(
                &qreader,
                &qmetadata,
                &samples,
                DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
                GgmlCpuGraphBackend::Cpu,
                Some("zh-sichuan"),
                None,
            )
            .expect("dolphin transcribe (quant)");
            let qcer = char_error_rate(&qoutput.text, REFERENCE_RESCORING_TEXT);
            eprintln!(
                "quant {:5}: size {:.1}MB ({:.2}x fp16)  CER vs rescoring {qcer:.4}  text {}",
                quant.label(),
                qbytes as f64 / 1.0e6,
                qbytes as f64 / fp16_bytes as f64,
                qoutput.text,
            );
            assert!(
                qbytes < fp16_bytes,
                "{} pack ({qbytes} B) must be smaller than fp16 ({fp16_bytes} B)",
                quant.label()
            );
            assert!(
                !qoutput.text.is_empty(),
                "{} transcript must not be empty",
                quant.label()
            );
            assert!(
                qcer <= cer_bound,
                "{} CER {qcer:.4} exceeds the spot-check bound {cer_bound:.2}",
                quant.label()
            );
            assert_eq!(qoutput.resolved_language, "zh-sichuan");
        }
    }

    /// End-to-end hotword demo: the un-biased `attention_rescoring` decode gets the
    /// 和/河 homophone wrong (see `REFERENCE_RESCORING_TEXT` above); native
    /// `context_module.*` deep-biasing with the hotword "河" flips it to the correct
    /// human transcript. Mirrors `work/hotword_parity.py`'s PyTorch reference demo
    /// exactly (same clip, same hotword, same "no hotword" vs "with hotword" pair).
    ///
    /// `#[ignore]`: needs the checkpoint/golden under `tmp/publish` (not committed).
    /// Run with:
    /// `cargo test -p openasr-core dolphin_hotword_flips_recognition_error -- --ignored --nocapture`
    #[test]
    #[ignore = "requires local Dolphin checkpoint + golden clip under tmp/publish (not committed)"]
    fn dolphin_hotword_flips_recognition_error() {
        let Some(root) = root() else {
            return;
        };
        let clip = root.join("golden/clip_sichuan.wav");
        let Some(pack) = ensure_dolphin_pack(&root, DolphinQuantizationMode::Fp16) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };
        if !clip.exists() {
            eprintln!("skip: golden clip not present at {clip:?}");
            return;
        }

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &clip,
            "dolphin hotword demo",
            "clip_sichuan.wav",
        )
        .expect("load clip");
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        let no_hotword = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
            None,
        )
        .expect("dolphin transcribe (no hotword)");

        let phrase_bias = crate::PhraseBiasConfig::from_phrases_with_default_boost(["河"], None)
            .expect("hotword phrase config");
        let with_hotword = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
            Some(&phrase_bias),
        )
        .expect("dolphin transcribe (with hotword)");

        eprintln!("== Dolphin hotword deep-biasing demo (河) ==");
        eprintln!("no hotword   : {}", no_hotword.text);
        eprintln!("with hotword : {}", with_hotword.text);

        assert_eq!(
            no_hotword.text, REFERENCE_RESCORING_TEXT,
            "no-hotword baseline diverged from the golden un-biased transcript"
        );
        assert_eq!(
            with_hotword.text, REFERENCE_WSC_TEXT,
            "hotword-biased transcript did not flip to the human WSC reference"
        );
        assert_ne!(
            no_hotword.text, with_hotword.text,
            "the hotword must change the rescored transcript on this clip"
        );

        // The quantized rungs must also load + transcribe with the hotword: this
        // proves the keep-quantized biasing_layer/combiner weights (bound native,
        // like every other family matmul weight) still drive a usable fused
        // decode, not just the fp16 rung above.
        for quant in [DolphinQuantizationMode::Q8_0, DolphinQuantizationMode::Q4_K] {
            let qpack = ensure_dolphin_pack(&root, quant).expect("quant pack builds");
            let qreader = GgufTensorDataReader::from_path(&qpack).expect("quant reader");
            let qmetadata =
                crate::ggml_runtime::read_gguf_metadata(&qpack).expect("quant metadata");
            let qoutput = transcribe_dolphin_pcm(
                &qreader,
                &qmetadata,
                &samples,
                DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
                GgmlCpuGraphBackend::Cpu,
                Some("zh-sichuan"),
                Some(&phrase_bias),
            )
            .expect("dolphin transcribe (quant, with hotword)");
            eprintln!("with hotword ({}) : {}", quant.label(), qoutput.text);
            assert!(
                !qoutput.text.is_empty(),
                "{} hotword transcript must not be empty",
                quant.label()
            );
        }
    }
}
