use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::Arc,
};

use thiserror::Error;

use crate::PhraseBiasConfig;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext,
    GgmlRopeExtParams, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyTokenDecoder,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, canonical_runtime_cache_path,
    with_thread_local_cached_mut_by_key,
};
use crate::nn::decoder::{
    LlmResidentKvArena, Seq2SeqReusableDecodeGraph, allocate_zeroed_llm_resident_kv_arena,
    build_fixed_kv_attention_mask_bits, build_fixed_kv_attention_mask_bits_for_sequences,
    reusable_decode_graph_supported_for_runner as shared_reusable_decode_graph_supported_for_runner,
};
use crate::{Segment, Transcription};

use super::encoder_graph::MoonshineEncoderOutput;
use super::graph_config::moonshine_decoder_graph_config;
use super::lora::{
    LoraSlot, MoonshineLoraAdapter, moonshine_adapter_cache_fingerprint, new_lora_slot_tensors,
};
use super::runtime_contract::MoonshineExecutionMetadata;
use super::tokenizer::MoonshineTokenizer;
use super::weights::{MoonshineDecoderLayerWeights, MoonshineDecoderWeights, MoonshineWeight};

const MOONSHINE_LAYER_NORM_EPSILON: f32 = 1.0e-5;
/// Floor for the decoder's `no_alloc` metadata context node/tensor budget:
/// covers the per-step decode cgraph, the weight+cross-KV arena, and the
/// resident self-KV arena (all metadata-only -- see
/// `GgmlStaticTensorArena`/`allocate_zeroed_llm_resident_kv_arena`: real
/// tensor bytes land in a backend buffer sized from actual shapes,
/// independent of this context's size). Mirrors the encoder's proven
/// `16_384` headroom (`moonshine_encoder_graph_config`).
const MOONSHINE_DECODER_GRAPH_SIZE_FLOOR: usize = 16_384;

/// (canonical pack path, backend, cross frame count, adapter fingerprint).
/// The adapter fingerprint MUST stay in this key: the runtime owns prepared
/// cgraphs with the adapter tensors baked in, so reuse keyed only on the base
/// pack would serve stale adapter graphs (correctness bug).
type MoonshineDecoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend, usize, String);

thread_local! {
    static MOONSHINE_DECODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<MoonshineDecoderRuntimeCacheKey, MoonshineDecoderGraphRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MoonshineDecodeOutput {
    pub transcription: Transcription,
    pub generated_tokens: Vec<u32>,
}

#[derive(Debug, Error)]
pub(crate) enum MoonshineDecoderGraphError {
    #[error("moonshine decoder graph input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[error("moonshine decoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("moonshine decoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("moonshine decoder graph shape overflowed")]
    ShapeOverflow,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_moonshine_decoder_short_form(
    decoder_weights: &MoonshineDecoderWeights,
    tokenizer: &MoonshineTokenizer,
    metadata: MoonshineExecutionMetadata,
    encoder_output: &MoonshineEncoderOutput,
    phrase_bias: Option<&PhraseBiasConfig>,
    prefer_cpu_backend: bool,
    runtime_path: Option<&Path>,
    word_timestamps: bool,
    audio_duration_seconds: f32,
    adapter: Option<&MoonshineLoraAdapter>,
) -> Result<MoonshineDecodeOutput, MoonshineDecoderGraphError> {
    if let Some(runtime_path) = runtime_path {
        let key = moonshine_decoder_runtime_cache_key(
            runtime_path,
            encoder_output.frame_count,
            prefer_cpu_backend,
            adapter,
        );
        return with_thread_local_cached_mut_by_key(
            &MOONSHINE_DECODER_RUNTIME_BY_KEY,
            key,
            DEFAULT_RUNTIME_CACHE_CAPACITY,
            || {
                MoonshineDecoderGraphRuntime::new(
                    decoder_weights,
                    metadata,
                    encoder_output.frame_count,
                    prefer_cpu_backend,
                    Some(runtime_path),
                    adapter,
                )
            },
            |runtime| {
                run_moonshine_decoder_short_form_with_runtime(
                    runtime,
                    tokenizer,
                    metadata,
                    encoder_output,
                    phrase_bias,
                    word_timestamps,
                    audio_duration_seconds,
                )
            },
        );
    }

    let mut runtime = MoonshineDecoderGraphRuntime::new(
        decoder_weights,
        metadata,
        encoder_output.frame_count,
        prefer_cpu_backend,
        runtime_path,
        adapter,
    )?;
    run_moonshine_decoder_short_form_with_runtime(
        &mut runtime,
        tokenizer,
        metadata,
        encoder_output,
        phrase_bias,
        word_timestamps,
        audio_duration_seconds,
    )
}

fn moonshine_decoder_runtime_cache_key(
    runtime_path: &Path,
    cross_frame_count: usize,
    prefer_cpu_backend: bool,
    adapter: Option<&MoonshineLoraAdapter>,
) -> MoonshineDecoderRuntimeCacheKey {
    (
        canonical_runtime_cache_path(runtime_path),
        moonshine_decoder_graph_config(prefer_cpu_backend).backend,
        cross_frame_count,
        moonshine_adapter_cache_fingerprint(adapter),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_moonshine_decoder_short_form_with_runtime(
    runtime: &mut MoonshineDecoderGraphRuntime,
    tokenizer: &MoonshineTokenizer,
    metadata: MoonshineExecutionMetadata,
    encoder_output: &MoonshineEncoderOutput,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    audio_duration_seconds: f32,
) -> Result<MoonshineDecodeOutput, MoonshineDecoderGraphError> {
    runtime.populate_cross_attention_cache(encoder_output)?;
    let mut step_executor = MoonshineDecoderStepExecutor { runtime };
    let token_decoder = MoonshineGreedyTokenDecoder { tokenizer };

    let prompt_tokens = vec![metadata.bos_token_id];
    let max_generated_tokens = metadata
        .decoder_max_context
        .saturating_sub(prompt_tokens.len())
        .max(1);
    // moonshine routes through the shared decode-policy registry (same path as
    // whisper/cohere/qwen) instead of hand-building a config: the descriptor
    // declares no suppression, no extra stop tokens and Identity postprocess, so
    // the resolved config is byte-identical to the previous inline one, and the
    // registry owns phrase-bias tokenization against the tokenizer token source.
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: prompt_tokens,
        eot_token_id: metadata.eos_token_id,
        vocab_size: metadata.vocab_size,
        max_generated_tokens,
    };
    let decode_text_token_ids = |token_ids: &[u32]| token_decoder.decode_text_token_ids(token_ids);
    let decode = match run_builtin_seq2seq_decode_policy::<Seq2SeqGreedyDecodeError>(
        crate::MOONSHINE_DECODE_POLICY_ID,
        &config,
        tokenizer,
        phrase_bias,
        &mut step_executor,
        &decode_text_token_ids,
        |error| error,
        |error| error,
        |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
            reason: error.to_string(),
        },
    ) {
        Ok(output) => output,
        Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens,
            generated_probabilities,
            ..
        }) => Seq2SeqGreedyDecodeResult {
            text: token_decoder
                .decode_text_token_ids(&generated_tokens)
                .map_err(|error| MoonshineDecoderGraphError::InvalidInput {
                    reason: error.to_string(),
                })?,
            generated_tokens,
            generated_probabilities,
        },
        Err(error) => {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: error.to_string(),
            });
        }
    };

    let text = decode.text.trim().to_string();
    let words = if word_timestamps {
        seq2seq_word_timestamps_from_generated_tokens(
            &decode.generated_tokens,
            &decode.generated_probabilities,
            0.0,
            audio_duration_seconds,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &|token_ids| token_decoder.decode_text_token_ids(token_ids),
        )
        .map_err(|error| MoonshineDecoderGraphError::InvalidInput {
            reason: error.to_string(),
        })?
    } else {
        Vec::new()
    };
    let segments = if words.is_empty() || text.is_empty() {
        Vec::new()
    } else {
        vec![Segment {
            start: 0.0,
            end: audio_duration_seconds.max(0.0),
            text: text.clone(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words,
        }]
    };

    Ok(MoonshineDecodeOutput {
        transcription: Transcription {
            text,
            segments,
            longform: None,
            language: None,
        },
        generated_tokens: decode.generated_tokens,
    })
}

struct MoonshineGreedyTokenDecoder<'a> {
    tokenizer: &'a MoonshineTokenizer,
}

impl Seq2SeqGreedyTokenDecoder for MoonshineGreedyTokenDecoder<'_> {
    fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, Seq2SeqGreedyDecodeError> {
        self.tokenizer
            .decode_text_token_ids(token_ids)
            .map_err(|error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                reason: error.to_string(),
            })
    }
}

/// moonshine has no whisper-style special control tokens (no stop tokens beyond
/// `<eos>`, no suppression list), so the decode-policy token source is the empty
/// default over the tokenizer, which supplies phrase-bias encoding via the
/// [`PhraseBiasTokenEncoder`] supertrait it already implements.
impl BuiltinSeq2SeqDecodePolicyTokenSource for MoonshineTokenizer {}

/// A 2-D linear: bound zero-copy from the mmap'd pack (native q8_0 `[in,out]`)
/// or, as a fallback, an arena-resident f32 tensor. moonshine loads its bindable
/// linears meta-only, so binding is mandatory and `Arena` is never constructed —
/// retained for parity with the wav2vec2/cohere pattern + a future non-mmap path.
#[derive(Clone, Copy)]
enum WeightSlot {
    #[allow(dead_code)]
    Arena(GgmlStaticTensor),
    Loaded(GgmlLoadedTensor),
}

impl WeightSlot {
    fn graph<'a>(self, arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(handle) => arena.graph_tensor(handle),
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
        }
    }
}

fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, MoonshineDecoderGraphError> {
    match loaded.and_then(|ctx| ctx.tensor(name)) {
        Some(tensor) => Ok(WeightSlot::Loaded(tensor)),
        None => Err(MoonshineDecoderGraphError::GraphExecutionFailed {
            reason: format!(
                "2-D linear '{name}' could not be bound zero-copy from the runtime pack \
                 (loaded weight context missing or tensor absent); host payload was meta-only"
            ),
        }),
    }
}

#[derive(Default, Clone, Copy)]
struct MoonshineDecoderLayerLora {
    attn_q: Option<LoraSlot>,
    attn_k: Option<LoraSlot>,
    attn_v: Option<LoraSlot>,
    attn_o: Option<LoraSlot>,
    cross_q: Option<LoraSlot>,
    cross_k: Option<LoraSlot>,
    cross_v: Option<LoraSlot>,
    cross_o: Option<LoraSlot>,
    ffn_up: Option<LoraSlot>,
    ffn_down: Option<LoraSlot>,
}

struct MoonshineDecoderLayerRuntime {
    attn_norm: GgmlStaticTensor,
    attn_q: WeightSlot,
    attn_k: WeightSlot,
    attn_v: WeightSlot,
    attn_o: WeightSlot,
    cross_norm: GgmlStaticTensor,
    cross_q: WeightSlot,
    cross_k: WeightSlot,
    cross_v: WeightSlot,
    cross_o: WeightSlot,
    ffn_norm: GgmlStaticTensor,
    ffn_up: WeightSlot,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down: WeightSlot,
    ffn_down_bias: GgmlStaticTensor,
    lora: MoonshineDecoderLayerLora,
}

/// Allocate (but do not yet upload) arena tensors for one optional LoRA
/// target; payload uploads are deferred until all arena tensors exist.
fn new_lora_slot<'adapter>(
    arena: &GgmlStaticTensorArena,
    adapter: Option<&'adapter MoonshineLoraAdapter>,
    base_tensor_name: &str,
    pending_uploads: &mut Vec<(GgmlStaticTensor, &'adapter [f32], &'static str)>,
) -> Result<Option<LoraSlot>, MoonshineDecoderGraphError> {
    let Some(target) = adapter.and_then(|adapter| adapter.target(base_tensor_name)) else {
        return Ok(None);
    };
    let slot =
        new_lora_slot_tensors(arena, target, "dec_lora_a", "dec_lora_b").map_err(|source| {
            MoonshineDecoderGraphError::GraphBuildFailed {
                step: "dec_lora_alloc",
                source,
            }
        })?;
    pending_uploads.push((slot.a, target.a_values.as_slice(), "dec_lora_a"));
    pending_uploads.push((
        slot.b_scaled,
        target.b_scaled_values.as_slice(),
        "dec_lora_b",
    ));
    Ok(Some(slot))
}

struct MoonshineCrossLayerRuntime {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
}

pub(crate) struct MoonshineDecoderGraphRuntime {
    metadata: MoonshineExecutionMetadata,
    // `reuse` holds raw pointers into `runner`, `loaded_weights`, `arena`,
    // `resident_kv`, and the cross-KV tensors, so it must be declared first
    // and dropped first.
    reuse: Option<Seq2SeqReusableDecodeGraph>,
    resident_kv: Option<LlmResidentKvArena>,
    // Owns the mmap'd pack backing every zero-copy WeightSlot::Loaded handle.
    #[allow(dead_code)]
    loaded_weights: Option<GgmlLoadedWeightContext>,
    runner: GgmlCpuGraphRunner,
    /// The `no_alloc` metadata context size used for `runner`'s own graph
    /// context and `arena`; reused for the resident self-KV arena
    /// ([`Self::ensure_resident_self_kv_arena`]) and for
    /// `start_persistent_graph_session` in
    /// [`Self::build_reusable_decode_graph`] instead of a hardcoded constant.
    persistent_graph_context_bytes: usize,
    arena: GgmlStaticTensorArena,
    embedding: GgmlStaticTensor,
    out_norm: GgmlStaticTensor,
    layers: Vec<MoonshineDecoderLayerRuntime>,
    cross_layers: Vec<MoonshineCrossLayerRuntime>,
    cross_frame_count: usize,
    n_seq: usize,
}

struct MoonshineDecoderStepExecutor<'a> {
    runtime: &'a mut MoonshineDecoderGraphRuntime,
}

impl Seq2SeqGreedyDecodeStepExecutor for MoonshineDecoderStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        let logits = if input.initial_prompt_tokens.len() == 1
            && self.runtime.supports_reusable_decode_graph()
        {
            let current_token = input
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(input.initial_prompt_tokens[0]);
            let position = input.generated_tokens.len();
            self.runtime
                .compute_incremental_step_logits(current_token, position)
        } else {
            let prefix = input
                .initial_prompt_tokens
                .iter()
                .copied()
                .chain(input.generated_tokens.iter().copied())
                .collect::<Vec<_>>();
            self.runtime.compute_full_prefix_step_logits(&prefix)
        }
        .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
            reason: error.to_string(),
        })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

impl MoonshineDecoderGraphRuntime {
    fn supports_reusable_decode_graph(&self) -> bool {
        reusable_decode_graph_supported_for_runner(&self.runner)
    }

    fn ensure_resident_self_kv_arena(&mut self) -> Result<(), MoonshineDecoderGraphError> {
        if self.resident_kv.is_some() {
            return Ok(());
        }
        self.resident_kv = Some(
            allocate_zeroed_llm_resident_kv_arena(
                &self.runner,
                self.persistent_graph_context_bytes,
                self.layers.len(),
                self.metadata.head_dim,
                self.metadata.decoder_max_context,
                self.metadata.n_heads,
                self.n_seq,
                "moonshine_decoder_resident_kv",
            )
            .map_err(build_err("moonshine_resident_self_kv"))?,
        );
        Ok(())
    }

    pub(crate) fn new(
        decoder_weights: &MoonshineDecoderWeights,
        metadata: MoonshineExecutionMetadata,
        cross_frame_count: usize,
        prefer_cpu_backend: bool,
        runtime_path: Option<&Path>,
        adapter: Option<&MoonshineLoraAdapter>,
    ) -> Result<Self, MoonshineDecoderGraphError> {
        Self::new_with_n_seq(
            decoder_weights,
            metadata,
            cross_frame_count,
            prefer_cpu_backend,
            runtime_path,
            1,
            adapter,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_n_seq(
        decoder_weights: &MoonshineDecoderWeights,
        metadata: MoonshineExecutionMetadata,
        cross_frame_count: usize,
        prefer_cpu_backend: bool,
        runtime_path: Option<&Path>,
        n_seq: usize,
        adapter: Option<&MoonshineLoraAdapter>,
    ) -> Result<Self, MoonshineDecoderGraphError> {
        if decoder_weights.layers.len() != metadata.decoder_layers {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder layer count mismatch: weights={}, metadata={}",
                    decoder_weights.layers.len(),
                    metadata.decoder_layers
                ),
            });
        }
        if cross_frame_count == 0 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "encoder frame_count must be > 0".to_string(),
            });
        }
        if n_seq == 0 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "moonshine decoder n_seq must be positive".to_string(),
            });
        }

        let mut config = moonshine_decoder_graph_config(prefer_cpu_backend);
        config.graph_size = config.graph_size.max(MOONSHINE_DECODER_GRAPH_SIZE_FLOOR);
        config.context_bytes =
            config
                .context_bytes
                .max(GgmlCpuGraphConfig::metadata_context_bytes(
                    config.graph_size,
                ));
        let persistent_graph_context_bytes = config.context_bytes;
        let runner = GgmlCpuGraphRunner::new(config).map_err(build_err("runner_init"))?;
        // Bind the per-layer 2-D linears (self/cross attn + ffn) zero-copy from the
        // mmap'd pack (native q8_0 [in,out]); the loader supplies them meta-only, so
        // binding is mandatory. The tied embedding stays arena-resident (it feeds
        // get_rows + tied-logits mul_mat and is loaded full).
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .map_err(build_err("static_tensor_arena"))?;

        let embedding = new_matrix(&arena, &decoder_weights.embedding, "dec_emb")?;
        let out_norm = new_vector(&arena, decoder_weights.out_norm.len(), "dec_out_norm")?;
        let d_model = metadata.d_model;

        let mut layers = Vec::with_capacity(decoder_weights.layers.len());
        let mut cross_layers = Vec::with_capacity(decoder_weights.layers.len());
        let mut pending_lora_uploads = Vec::new();
        for layer in &decoder_weights.layers {
            let lora = MoonshineDecoderLayerLora {
                attn_q: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.attn_q.name,
                    &mut pending_lora_uploads,
                )?,
                attn_k: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.attn_k.name,
                    &mut pending_lora_uploads,
                )?,
                attn_v: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.attn_v.name,
                    &mut pending_lora_uploads,
                )?,
                attn_o: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.attn_o.name,
                    &mut pending_lora_uploads,
                )?,
                cross_q: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.cross_q.name,
                    &mut pending_lora_uploads,
                )?,
                cross_k: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.cross_k.name,
                    &mut pending_lora_uploads,
                )?,
                cross_v: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.cross_v.name,
                    &mut pending_lora_uploads,
                )?,
                cross_o: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.cross_o.name,
                    &mut pending_lora_uploads,
                )?,
                ffn_up: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.ffn_up.name,
                    &mut pending_lora_uploads,
                )?,
                ffn_down: new_lora_slot(
                    &arena,
                    adapter,
                    &layer.ffn_down.name,
                    &mut pending_lora_uploads,
                )?,
            };
            layers.push(MoonshineDecoderLayerRuntime {
                attn_norm: new_vector(&arena, layer.attn_norm.len(), "dec_attn_norm")?,
                attn_q: bind_loaded(loaded, &layer.attn_q.name)?,
                attn_k: bind_loaded(loaded, &layer.attn_k.name)?,
                attn_v: bind_loaded(loaded, &layer.attn_v.name)?,
                attn_o: bind_loaded(loaded, &layer.attn_o.name)?,
                cross_norm: new_vector(&arena, layer.cross_norm.len(), "dec_cross_norm")?,
                cross_q: bind_loaded(loaded, &layer.cross_q.name)?,
                cross_k: bind_loaded(loaded, &layer.cross_k.name)?,
                cross_v: bind_loaded(loaded, &layer.cross_v.name)?,
                cross_o: bind_loaded(loaded, &layer.cross_o.name)?,
                ffn_norm: new_vector(&arena, layer.ffn_norm.len(), "dec_ffn_norm")?,
                ffn_up: bind_loaded(loaded, &layer.ffn_up.name)?,
                ffn_up_bias: new_vector(&arena, layer.ffn_up_bias.len(), "dec_ffn_up_b")?,
                ffn_down: bind_loaded(loaded, &layer.ffn_down.name)?,
                ffn_down_bias: new_vector(&arena, layer.ffn_down_bias.len(), "dec_ffn_down_b")?,
                lora,
            });
            cross_layers.push(MoonshineCrossLayerRuntime {
                key: new_cross_cache(
                    &arena,
                    d_model,
                    cross_frame_count,
                    n_seq,
                    "dec_cross_k_cache",
                )?,
                value: new_cross_cache(
                    &arena,
                    d_model,
                    cross_frame_count,
                    n_seq,
                    "dec_cross_v_cache",
                )?,
            });
        }

        upload(&mut arena, embedding, &decoder_weights.embedding, "dec_emb")?;
        upload(
            &mut arena,
            out_norm,
            &decoder_weights.out_norm,
            "dec_out_norm",
        )?;
        for (runtime, layer) in layers.iter().zip(&decoder_weights.layers) {
            upload_layer(&mut arena, runtime, layer)?;
        }
        for (tensor, values, name) in pending_lora_uploads {
            arena
                .set_f32_slice(tensor, values, name)
                .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed {
                    step: name,
                    source,
                })?;
        }

        Ok(Self {
            metadata,
            reuse: None,
            resident_kv: None,
            loaded_weights,
            runner,
            persistent_graph_context_bytes,
            arena,
            embedding,
            out_norm,
            layers,
            cross_layers,
            cross_frame_count,
            n_seq,
        })
    }

    /// Precompute per-layer cross-attention K/V from the encoder output (once per utterance).
    pub(super) fn populate_cross_attention_cache(
        &mut self,
        encoder_output: &MoonshineEncoderOutput,
    ) -> Result<(), MoonshineDecoderGraphError> {
        self.populate_cross_attention_cache_slot(0, encoder_output)
    }

    pub(super) fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        encoder_output: &MoonshineEncoderOutput,
    ) -> Result<(), MoonshineDecoderGraphError> {
        if slot_index >= self.n_seq {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "moonshine cross-cache slot {slot_index} exceeds n_seq {}",
                    self.n_seq
                ),
            });
        }
        let d_model = self.metadata.d_model;
        let expected = encoder_output
            .frame_count
            .checked_mul(d_model)
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        if encoder_output.rows.len() != expected {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "encoder rows length mismatch: got {}, expected {}",
                    encoder_output.rows.len(),
                    expected
                ),
            });
        }

        // Project encoder rows through each layer's cross_k / cross_v on the host (no RoPE).
        // The encoder runs once and the cross-KV is reused across all decode steps.
        for layer_idx in 0..self.cross_layers.len() {
            self.populate_cross_attention_cache_slot_layer(slot_index, layer_idx, encoder_output)
                .map(|_| ())?;
        }
        Ok(())
    }

    /// One layer of the cross-KV precompute. Returns the computed cross-V
    /// projection rows (`[frame][d_model]` f32) — production callers discard
    /// them; the LoRA oracle tests compare them against host math.
    fn populate_cross_attention_cache_slot_layer(
        &mut self,
        slot_index: usize,
        layer_idx: usize,
        encoder_output: &MoonshineEncoderOutput,
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        let d_model = self.metadata.d_model;
        let expected = encoder_output
            .frame_count
            .checked_mul(d_model)
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        let frame_count = encoder_output.frame_count;
        let cross = &self.cross_layers[layer_idx];
        {
            let mut graph = self.runner.start_graph();
            let enc = graph
                .new_tensor_2d_f32(d_model, frame_count, "moonshine_enc_rows")
                .map_err(build_err("ggml_new_tensor_2d(enc_rows)"))?;
            graph
                .set_input(enc)
                .map_err(build_err("ggml_set_input(enc_rows)"))?;
            // Reload the per-layer cross projections via stored layer runtimes.
            // CRITICAL: this per-utterance precompute must route through the
            // same LoRA side-path as the decode graphs — otherwise a decode
            // with cross_k/cross_v targeted would mix base cross-KV caches
            // with adapter projections elsewhere.
            let layer = &self.layers[layer_idx];
            let key = matmul(
                &graph,
                &self.arena,
                layer.cross_k_proj(),
                layer.lora.cross_k,
                enc,
                "ggml_mul_mat(cross_k)",
            )?;
            let value = matmul(
                &graph,
                &self.arena,
                layer.cross_v_proj(),
                layer.lora.cross_v,
                enc,
                "ggml_mul_mat(cross_v)",
            )?;
            let key_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross.key),
                d_model,
                frame_count,
                self.n_seq,
                slot_index,
                "moonshine_cross_k_slot",
            )?;
            let value_target = cross_cache_slot_target(
                &graph,
                self.arena.graph_tensor(cross.value),
                d_model,
                frame_count,
                self.n_seq,
                slot_index,
                "moonshine_cross_v_slot",
            )?;
            let write_key = graph
                .cpy(key, key_target)
                .map_err(build_err("ggml_cpy(cross_k_cache)"))?;
            graph
                .add_side_effect_root(write_key)
                .map_err(build_err("side_effect(cross_k)"))?;
            let write_value = graph
                .cpy(value, value_target)
                .map_err(build_err("ggml_cpy(cross_v_cache)"))?;
            graph
                .add_side_effect_root(write_value)
                .map_err(build_err("side_effect(cross_v)"))?;
            graph
                .set_output(value)
                .map_err(build_err("ggml_set_output(cross)"))?;
            graph
                .set_f32_slice(enc, &encoder_output.rows, "moonshine_enc_rows")
                .map_err(build_err("ggml_set_f32_slice(enc_rows)"))?;
            graph.compute_output_f32(value, expected).map_err(|error| {
                MoonshineDecoderGraphError::GraphExecutionFailed {
                    reason: error.to_string(),
                }
            })
        }
    }

    /// LoRA oracle hook: cross-V projection rows for one layer of slot 0,
    /// computed by the same graph the production precompute runs.
    #[cfg(test)]
    pub(super) fn cross_value_projection_rows_for_test(
        &mut self,
        layer_idx: usize,
        encoder_output: &MoonshineEncoderOutput,
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        self.populate_cross_attention_cache_slot_layer(0, layer_idx, encoder_output)
    }

    pub(super) fn compute_incremental_step_logits(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        if self.n_seq != 1 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "single moonshine decode step requires n_seq=1".to_string(),
            });
        }
        if position >= self.metadata.decoder_max_context {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder position {position} exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }
        let token_id =
            i32::try_from(token_id).map_err(|_| MoonshineDecoderGraphError::InvalidInput {
                reason: format!("token id {token_id} does not fit i32"),
            })?;
        let position_i32 =
            i32::try_from(position).map_err(|_| MoonshineDecoderGraphError::InvalidInput {
                reason: format!("decoder position {position} does not fit i32"),
            })?;
        let total_tokens = position
            .checked_add(1)
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.max_positions != self.metadata.decoder_max_context || reuse.n_seq != 1
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
        }

        let reuse = self
            .reuse
            .as_mut()
            .expect("moonshine reusable decode graph built above");
        let token_tensor = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let attention_mask = reuse.attention_mask;
        let logits = reuse.logits;
        let max_positions = reuse.max_positions;
        let graph = reuse.builder();

        graph
            .set_i32_slice(token_tensor, &[token_id], "moonshine_reuse_token")
            .map_err(build_err("ggml_set_i32_slice(reuse_token)"))?;
        graph
            .set_i32_slice(row_index, &[position_i32], "moonshine_reuse_row")
            .map_err(build_err("ggml_set_i32_slice(reuse_row)"))?;
        graph
            .set_i32_slice(position_tensor, &[position_i32], "moonshine_reuse_position")
            .map_err(build_err("ggml_set_i32_slice(reuse_position)"))?;
        let mask_bits = build_fixed_kv_attention_mask_bits(max_positions, total_tokens)
            .map_err(build_err("moonshine_reuse_self_mask"))?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "moonshine_reuse_self_mask")
            .map_err(build_err("ggml_set_f16_bits_slice(reuse_mask)"))?;

        graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| MoonshineDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })
    }

    // Wired by the moonshine serve-batch owner thread in the follow-up step.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        total_tokens_by_sequence: &[usize],
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        if self.n_seq == 1 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "batched moonshine decode step requires n_seq > 1".to_string(),
            });
        }
        if token_ids.len() != self.n_seq
            || positions.len() != self.n_seq
            || total_tokens_by_sequence.len() != self.n_seq
        {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched moonshine decode inputs must have n_seq={} entries",
                    self.n_seq
                ),
            });
        }
        if positions
            .iter()
            .any(|&position| position >= self.metadata.decoder_max_context)
        {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched moonshine decoder position exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }
        if total_tokens_by_sequence.iter().any(|&total_tokens| {
            total_tokens == 0 || total_tokens > self.metadata.decoder_max_context
        }) {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched moonshine total token count must be in 1..={}",
                    self.metadata.decoder_max_context
                ),
            });
        }

        let token_ids = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| MoonshineDecoderGraphError::InvalidInput {
                    reason: format!("token id {token_id} does not fit i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positions = positions
            .iter()
            .map(|&position| {
                i32::try_from(position).map_err(|_| MoonshineDecoderGraphError::InvalidInput {
                    reason: format!("decoder position {position} does not fit i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.max_positions != self.metadata.decoder_max_context
                    || reuse.n_seq != self.n_seq
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
        }

        let reuse = self
            .reuse
            .as_mut()
            .expect("moonshine batched reusable decode graph built above");
        let token_tensor = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let attention_mask = reuse.attention_mask;
        let logits = reuse.logits;
        let max_positions = reuse.max_positions;
        let graph = reuse.builder();

        graph
            .set_i32_slice(token_tensor, &token_ids, "moonshine_reuse_batch_token")
            .map_err(build_err("ggml_set_i32_slice(reuse_batch_token)"))?;
        graph
            .set_i32_slice(row_index, &positions, "moonshine_reuse_batch_row")
            .map_err(build_err("ggml_set_i32_slice(reuse_batch_row)"))?;
        graph
            .set_i32_slice(
                position_tensor,
                &positions,
                "moonshine_reuse_batch_position",
            )
            .map_err(build_err("ggml_set_i32_slice(reuse_batch_position)"))?;
        let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
            max_positions,
            total_tokens_by_sequence,
        )
        .map_err(build_err("moonshine_reuse_batch_self_mask"))?;
        graph
            .set_f16_bits_slice(
                attention_mask,
                &mask_bits,
                "moonshine_reuse_batch_self_mask",
            )
            .map_err(build_err("ggml_set_f16_bits_slice(reuse_batch_mask)"))?;

        graph
            .compute_output_f32(logits, self.metadata.vocab_size * self.n_seq)
            .map_err(|error| MoonshineDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })
    }

    pub(super) fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        if self.n_seq == 1 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "batched moonshine prefill requires n_seq > 1".to_string(),
            });
        }
        let token_count = prompt_tokens.len();
        if token_count == 0 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "batched moonshine prefill token_count must be > 0".to_string(),
            });
        }
        if token_count > self.metadata.decoder_max_context {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "batched moonshine prefill token_count {token_count} exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }

        self.reuse = None;
        self.ensure_resident_self_kv_arena()?;
        let resident_layers = self
            .resident_kv
            .as_ref()
            .expect("moonshine resident self-KV arena initialized above")
            .graph_tensors();

        let output_tokens = token_count
            .checked_mul(self.n_seq)
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        let prompt_tokens_i32 = tokens_as_i32(prompt_tokens)?;
        let mut token_ids = Vec::with_capacity(output_tokens);
        let mut positions = Vec::with_capacity(output_tokens);
        let mut row_indices = Vec::with_capacity(output_tokens);
        for _ in 0..self.n_seq {
            for (position, &token_id) in prompt_tokens_i32.iter().enumerate() {
                let position_i32 = i32::try_from(position).map_err(|_| {
                    MoonshineDecoderGraphError::InvalidInput {
                        reason: format!("decoder position {position} does not fit i32"),
                    }
                })?;
                token_ids.push(token_id);
                positions.push(position_i32);
                row_indices.push(position_i32);
            }
        }

        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;
        let frame_count = self.cross_frame_count;

        let mut graph = self.runner.start_graph();
        let token_ids_tensor = graph
            .new_tensor_1d_i32(output_tokens, "moonshine_prefill_token")
            .map_err(build_err("ggml_new_tensor_1d(prefill_token)"))?;
        let positions_tensor = graph
            .new_tensor_1d_i32(output_tokens, "moonshine_prefill_position")
            .map_err(build_err("ggml_new_tensor_1d(prefill_position)"))?;
        let row_index_tensor = graph
            .new_tensor_4d_i32(token_count, 1, self.n_seq, 1, "moonshine_prefill_row")
            .map_err(build_err("ggml_new_tensor_4d(prefill_row)"))?;
        let attention_mask = graph
            .new_tensor_4d_f16(
                token_count,
                token_count,
                1,
                self.n_seq,
                "moonshine_prefill_self_mask",
            )
            .map_err(build_err("ggml_new_tensor_4d(prefill_mask)"))?;
        graph
            .set_input(token_ids_tensor)
            .map_err(build_err("ggml_set_input(prefill_token)"))?;
        graph
            .set_input(positions_tensor)
            .map_err(build_err("ggml_set_input(prefill_position)"))?;
        graph
            .set_input(row_index_tensor)
            .map_err(build_err("ggml_set_input(prefill_row)"))?;
        graph
            .set_input(attention_mask)
            .map_err(build_err("ggml_set_input(prefill_mask)"))?;

        let mut state = graph
            .get_rows(self.arena.graph_tensor(self.embedding), token_ids_tensor)
            .map_err(build_err("ggml_get_rows(prefill_emb)"))?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cross = &self.cross_layers[layer_idx];
            let (self_k, self_v) = resident_layers[layer_idx];
            state = run_prefill_decoder_layer(
                &mut graph,
                &self.arena,
                state,
                layer,
                cross,
                self_k,
                self_v,
                row_index_tensor,
                positions_tensor,
                attention_mask,
                token_count,
                frame_count,
                d_model,
                heads,
                head_dim,
                self.metadata.decoder_ffn_dim,
                self.metadata.rotary_dim,
                self.metadata.decoder_max_context,
                self.metadata.rope_theta,
                self.n_seq,
            )?;
        }

        state = apply_weighted_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.out_norm),
            "prefill_dec_out_norm",
        )?;
        let last = view_batched_last_token(&graph, state, d_model, token_count, self.n_seq)?;
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.embedding), last)
            .map_err(build_err("ggml_mul_mat(prefill_logits)"))?;
        graph
            .set_output(logits)
            .map_err(build_err("ggml_set_output(prefill_logits)"))?;
        // Allocate the batched prefill graph through the scheduler's gallocr
        // before uploading inputs, mirroring the single-step reuse graph above
        // and the sibling cohere/firered decoders.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(build_err("ggml_prepare_outputs(prefill_logits)"))?;

        graph
            .set_i32_slice(token_ids_tensor, &token_ids, "moonshine_prefill_token")
            .map_err(build_err("ggml_set_i32_slice(prefill_token)"))?;
        graph
            .set_i32_slice(positions_tensor, &positions, "moonshine_prefill_position")
            .map_err(build_err("ggml_set_i32_slice(prefill_position)"))?;
        graph
            .set_i32_slice(row_index_tensor, &row_indices, "moonshine_prefill_row")
            .map_err(build_err("ggml_set_i32_slice(prefill_row)"))?;
        let mask_bits = build_batched_causal_mask_f16_bits(token_count, self.n_seq)?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "moonshine_prefill_self_mask")
            .map_err(build_err("ggml_set_f16_bits_slice(prefill_mask)"))?;

        graph
            .compute_output_f32(logits, self.metadata.vocab_size * self.n_seq)
            .map_err(|error| MoonshineDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })
    }

    fn build_reusable_decode_graph(&mut self) -> Result<(), MoonshineDecoderGraphError> {
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;
        let frame_count = self.cross_frame_count;
        let max_context = self.metadata.decoder_max_context;
        let n_seq = self.n_seq;

        self.ensure_resident_self_kv_arena()?;

        let mut session = self
            .runner
            .start_persistent_graph_session(self.persistent_graph_context_bytes)
            .map_err(build_err("moonshine_reuse_session"))?;
        let graph = session.builder();
        let token_id = graph
            .new_tensor_1d_i32(n_seq, "moonshine_reuse_token")
            .map_err(build_err("ggml_new_tensor_1d(reuse_token)"))?;
        let row_index = if n_seq == 1 {
            graph
                .new_tensor_1d_i32(1, "moonshine_reuse_row")
                .map_err(build_err("ggml_new_tensor_1d(reuse_row)"))?
        } else {
            graph
                .new_tensor_4d_i32(1, 1, n_seq, 1, "moonshine_reuse_row")
                .map_err(build_err("ggml_new_tensor_4d(reuse_row)"))?
        };
        let position = graph
            .new_tensor_1d_i32(n_seq, "moonshine_reuse_position")
            .map_err(build_err("ggml_new_tensor_1d(reuse_position)"))?;
        let attention_mask = if n_seq == 1 {
            graph
                .new_tensor_3d_f16(max_context, 1, 1, "moonshine_reuse_self_mask")
                .map_err(build_err("ggml_new_tensor_3d(reuse_mask)"))?
        } else {
            graph
                .new_tensor_4d_f16(max_context, 1, 1, n_seq, "moonshine_reuse_self_mask")
                .map_err(build_err("ggml_new_tensor_4d(reuse_mask)"))?
        };
        graph
            .set_input(token_id)
            .map_err(build_err("ggml_set_input(reuse_token)"))?;
        graph
            .set_input(row_index)
            .map_err(build_err("ggml_set_input(reuse_row)"))?;
        graph
            .set_input(position)
            .map_err(build_err("ggml_set_input(reuse_position)"))?;
        graph
            .set_input(attention_mask)
            .map_err(build_err("ggml_set_input(reuse_mask)"))?;

        let mut state = graph
            .get_rows(self.arena.graph_tensor(self.embedding), token_id)
            .map_err(build_err("ggml_get_rows(reuse_emb)"))?;
        let resident_layers = self
            .resident_kv
            .as_ref()
            .expect("moonshine resident self-KV arena initialized above")
            .graph_tensors();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cross = &self.cross_layers[layer_idx];
            let (self_k, self_v) = resident_layers[layer_idx];
            state = run_incremental_decoder_layer(
                graph,
                &self.arena,
                state,
                layer,
                cross,
                self_k,
                self_v,
                row_index,
                position,
                attention_mask,
                frame_count,
                d_model,
                heads,
                head_dim,
                self.metadata.decoder_ffn_dim,
                self.metadata.rotary_dim,
                self.metadata.decoder_max_context,
                self.metadata.rope_theta,
                n_seq,
            )?;
        }

        state = apply_weighted_norm(
            graph,
            state,
            self.arena.graph_tensor(self.out_norm),
            "dec_out_norm",
        )?;
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.embedding), state)
            .map_err(build_err("ggml_mul_mat(reuse_logits)"))?;
        graph
            .set_output(logits)
            .map_err(build_err("ggml_set_output(reuse_logits)"))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(build_err("ggml_prepare_outputs(reuse_logits)"))?;

        self.reuse = Some(Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena(
            session,
            max_context,
            n_seq,
            token_id,
            row_index,
            position,
            attention_mask,
            logits,
        ));
        Ok(())
    }

    pub(super) fn compute_full_prefix_step_logits(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, MoonshineDecoderGraphError> {
        let token_count = tokens.len();
        if token_count == 0 {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: "decoder token_count must be > 0".to_string(),
            });
        }
        if token_count > self.metadata.decoder_max_context {
            return Err(MoonshineDecoderGraphError::InvalidInput {
                reason: format!(
                    "decoder token_count {token_count} exceeds max context {}",
                    self.metadata.decoder_max_context
                ),
            });
        }

        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;
        let frame_count = self.cross_frame_count;

        let mut graph = self.runner.start_graph();

        let token_ids = graph
            .new_tensor_1d_i32(token_count, "moonshine_dec_tokens")
            .map_err(build_err("ggml_new_tensor_1d(tokens)"))?;
        let positions = graph
            .new_tensor_1d_i32(token_count, "moonshine_dec_positions")
            .map_err(build_err("ggml_new_tensor_1d(positions)"))?;
        graph
            .set_input(token_ids)
            .map_err(build_err("ggml_set_input(tokens)"))?;
        graph
            .set_input(positions)
            .map_err(build_err("ggml_set_input(positions)"))?;

        // Shared causal self-attention mask (declared up-front; one for all layers).
        let self_mask = if token_count == 1 {
            None
        } else {
            let mask = graph
                .new_tensor_3d_f16(token_count, token_count, 1, "dec_self_mask")
                .map_err(build_err("ggml_new_tensor_3d(self_mask)"))?;
            graph
                .set_input(mask)
                .map_err(build_err("ggml_set_input(self_mask)"))?;
            Some(mask)
        };

        let mut state = graph
            .get_rows(self.arena.graph_tensor(self.embedding), token_ids)
            .map_err(build_err("ggml_get_rows(emb)"))?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let cross = &self.cross_layers[layer_idx];
            state = run_decoder_layer(
                &mut graph,
                &self.arena,
                state,
                layer,
                cross,
                positions,
                self_mask,
                token_count,
                frame_count,
                d_model,
                heads,
                head_dim,
                self.metadata.decoder_ffn_dim,
                self.metadata.rotary_dim,
                self.metadata.decoder_max_context,
                self.metadata.rope_theta,
            )?;
        }

        state = apply_weighted_norm(
            &graph,
            state,
            self.arena.graph_tensor(self.out_norm),
            "dec_out_norm",
        )?;
        let last = view_last_token(&graph, state, d_model, token_count)?;
        // Tied logits: embedding is [d_model, vocab]; mul_mat gives [vocab, 1].
        let logits = graph
            .mul_mat(self.arena.graph_tensor(self.embedding), last)
            .map_err(build_err("ggml_mul_mat(logits)"))?;
        graph
            .set_output(logits)
            .map_err(build_err("ggml_set_output(logits)"))?;
        // Allocate the full-prefix step graph through the scheduler's gallocr
        // before uploading inputs, mirroring the single-step reuse graph and
        // batched prefill above.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(build_err("ggml_prepare_outputs(logits)"))?;

        let token_values = tokens_as_i32(tokens)?;
        let position_values: Vec<i32> = (0..token_count as i32).collect();
        graph
            .set_i32_slice(token_ids, &token_values, "moonshine_dec_tokens")
            .map_err(build_err("ggml_set_i32_slice(tokens)"))?;
        graph
            .set_i32_slice(positions, &position_values, "moonshine_dec_positions")
            .map_err(build_err("ggml_set_i32_slice(positions)"))?;
        if let Some(mask) = self_mask {
            let mask_bits = build_causal_mask_f16_bits(token_count)?;
            graph
                .set_f16_bits_slice(mask, &mask_bits, "dec_self_mask")
                .map_err(build_err("ggml_set_f16_bits_slice(self_mask)"))?;
        }

        graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| MoonshineDecoderGraphError::GraphExecutionFailed {
                reason: error.to_string(),
            })
    }
}

fn reusable_decode_graph_supported_for_runner(runner: &GgmlCpuGraphRunner) -> bool {
    shared_reusable_decode_graph_supported_for_runner(runner)
}

impl MoonshineDecoderLayerRuntime {
    fn cross_k_proj(&self) -> WeightSlot {
        self.cross_k
    }
    fn cross_v_proj(&self) -> WeightSlot {
        self.cross_v
    }
}

#[allow(clippy::too_many_arguments)]
fn run_incremental_decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &MoonshineDecoderLayerRuntime,
    cross: &MoonshineCrossLayerRuntime,
    self_k_cache: GgmlCpuTensor<'a>,
    self_v_cache: GgmlCpuTensor<'a>,
    row_index: GgmlCpuTensor<'a>,
    position: GgmlCpuTensor<'a>,
    self_mask: GgmlCpuTensor<'a>,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rotary_dim: usize,
    rope_max_context: usize,
    rope_theta: f32,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // --- Self-attention (single token, resident incremental KV) ---
    let attn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.attn_norm),
        "dec_attn_norm",
    )?;
    let q = matmul(
        graph,
        arena,
        layer.attn_q,
        layer.lora.attn_q,
        normed,
        "dec_q",
    )?;
    let k = matmul(
        graph,
        arena,
        layer.attn_k,
        layer.lora.attn_k,
        normed,
        "dec_k",
    )?;
    let v = matmul(
        graph,
        arena,
        layer.attn_v,
        layer.lora.attn_v,
        normed,
        "dec_v",
    )?;

    let q = rope_incremental_heads_for_attn(
        graph,
        q,
        head_dim,
        heads,
        position,
        rotary_dim,
        rope_max_context,
        rope_theta,
        n_seq,
        "dec_q_rope",
    )?;
    let k = rope_incremental_heads_for_attn(
        graph,
        k,
        head_dim,
        heads,
        position,
        rotary_dim,
        rope_max_context,
        rope_theta,
        n_seq,
        "dec_k_rope",
    )?;
    let v = reshape_incremental_for_attn(graph, v, head_dim, heads, n_seq, "dec_v_attn")?;
    let k = graph
        .set_rows(self_k_cache, k, row_index)
        .map_err(build_err("ggml_set_rows(dec_self_k_cache)"))?;
    let v = graph
        .set_rows(self_v_cache, v, row_index)
        .map_err(build_err("ggml_set_rows(dec_self_v_cache)"))?;

    let context = scaled_dot_product_attention(
        graph,
        q,
        k,
        v,
        Some(self_mask),
        scale,
        head_dim,
        1,
        heads,
        d_model,
        n_seq,
        "dec_self",
    )?;
    let attn = matmul(
        graph,
        arena,
        layer.attn_o,
        layer.lora.attn_o,
        context,
        "dec_self_o",
    )?;
    let state = graph
        .add(attn_input, attn)
        .map_err(build_err("dec_self_residual"))?;

    run_cross_attention_and_ffn(
        graph,
        arena,
        state,
        layer,
        cross,
        1,
        frame_count,
        d_model,
        heads,
        head_dim,
        ffn_dim,
        n_seq,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &MoonshineDecoderLayerRuntime,
    cross: &MoonshineCrossLayerRuntime,
    self_k_cache: GgmlCpuTensor<'a>,
    self_v_cache: GgmlCpuTensor<'a>,
    row_index: GgmlCpuTensor<'a>,
    positions: GgmlCpuTensor<'a>,
    self_mask: GgmlCpuTensor<'a>,
    token_count: usize,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rotary_dim: usize,
    max_context: usize,
    rope_theta: f32,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // --- Self-attention (whole prompt, resident self-KV seed) ---
    let attn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.attn_norm),
        "prefill_dec_attn_norm",
    )?;
    let q = matmul(
        graph,
        arena,
        layer.attn_q,
        layer.lora.attn_q,
        normed,
        "prefill_dec_q",
    )?;
    let k = matmul(
        graph,
        arena,
        layer.attn_k,
        layer.lora.attn_k,
        normed,
        "prefill_dec_k",
    )?;
    let v = matmul(
        graph,
        arena,
        layer.attn_v,
        layer.lora.attn_v,
        normed,
        "prefill_dec_v",
    )?;

    let q = rope_heads_for_attn_with_n_seq(
        graph,
        q,
        head_dim,
        heads,
        token_count,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        n_seq,
        "prefill_dec_q_rope",
    )?;
    let k = rope_heads_for_attn_with_n_seq(
        graph,
        k,
        head_dim,
        heads,
        token_count,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        n_seq,
        "prefill_dec_k_rope",
    )?;
    let v = reshape_for_attn_with_n_seq(
        graph,
        v,
        head_dim,
        heads,
        token_count,
        n_seq,
        "prefill_dec_v_attn",
    )?;
    let k = graph
        .set_rows(self_k_cache, k, row_index)
        .map_err(build_err("ggml_set_rows(prefill_dec_self_k_cache)"))?;
    let v = graph
        .set_rows(self_v_cache, v, row_index)
        .map_err(build_err("ggml_set_rows(prefill_dec_self_v_cache)"))?;
    let k = view_self_kv_prefix(
        graph,
        k,
        head_dim,
        token_count,
        heads,
        max_context,
        n_seq,
        "prefill_dec_self_k_view",
    )?;
    let v = view_self_kv_prefix(
        graph,
        v,
        head_dim,
        token_count,
        heads,
        max_context,
        n_seq,
        "prefill_dec_self_v_view",
    )?;

    let context = scaled_dot_product_attention(
        graph,
        q,
        k,
        v,
        Some(self_mask),
        scale,
        head_dim,
        token_count,
        heads,
        d_model,
        n_seq,
        "prefill_dec_self",
    )?;
    let attn = matmul(
        graph,
        arena,
        layer.attn_o,
        layer.lora.attn_o,
        context,
        "prefill_dec_self_o",
    )?;
    let state = graph
        .add(attn_input, attn)
        .map_err(build_err("prefill_dec_self_residual"))?;

    run_cross_attention_and_ffn(
        graph,
        arena,
        state,
        layer,
        cross,
        token_count,
        frame_count,
        d_model,
        heads,
        head_dim,
        ffn_dim,
        n_seq,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &MoonshineDecoderLayerRuntime,
    cross: &MoonshineCrossLayerRuntime,
    positions: GgmlCpuTensor<'a>,
    self_mask: Option<GgmlCpuTensor<'a>>,
    token_count: usize,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rotary_dim: usize,
    max_context: usize,
    rope_theta: f32,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // --- Self-attention (causal, partial RoPE) ---
    let attn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.attn_norm),
        "dec_attn_norm",
    )?;
    let q = matmul(
        graph,
        arena,
        layer.attn_q,
        layer.lora.attn_q,
        normed,
        "dec_q",
    )?;
    let k = matmul(
        graph,
        arena,
        layer.attn_k,
        layer.lora.attn_k,
        normed,
        "dec_k",
    )?;
    let v = matmul(
        graph,
        arena,
        layer.attn_v,
        layer.lora.attn_v,
        normed,
        "dec_v",
    )?;

    let q = rope_heads(
        graph,
        q,
        head_dim,
        heads,
        token_count,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        "dec_q_rope",
    )?;
    let k = rope_heads(
        graph,
        k,
        head_dim,
        heads,
        token_count,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        "dec_k_rope",
    )?;
    let q = roped_for_attn(graph, q, "dec_q_attn")?;
    let k = roped_for_attn(graph, k, "dec_k_attn")?;
    let v = reshape_for_attn(graph, v, head_dim, heads, token_count, "dec_v_attn")?;

    let context = scaled_dot_product_attention(
        graph,
        q,
        k,
        v,
        self_mask,
        scale,
        head_dim,
        token_count,
        heads,
        d_model,
        1,
        "dec_self",
    )?;
    let attn = matmul(
        graph,
        arena,
        layer.attn_o,
        layer.lora.attn_o,
        context,
        "dec_self_o",
    )?;
    let state = graph
        .add(attn_input, attn)
        .map_err(build_err("dec_self_residual"))?;

    run_cross_attention_and_ffn(
        graph,
        arena,
        state,
        layer,
        cross,
        token_count,
        frame_count,
        d_model,
        heads,
        head_dim,
        ffn_dim,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_cross_attention_and_ffn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &MoonshineDecoderLayerRuntime,
    cross: &MoonshineCrossLayerRuntime,
    token_count: usize,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if n_seq == 0 {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: "moonshine attention n_seq must be positive".to_string(),
        });
    }
    let scale = 1.0 / (head_dim as f32).sqrt();

    // --- Cross-attention (no RoPE, precomputed cross-KV) ---
    let cross_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.cross_norm),
        "dec_cross_norm",
    )?;
    let q = matmul(
        graph,
        arena,
        layer.cross_q,
        layer.lora.cross_q,
        normed,
        "dec_cross_q",
    )?;
    let q = reshape_for_attn_with_n_seq(
        graph,
        q,
        head_dim,
        heads,
        token_count,
        n_seq,
        "dec_cross_q_attn",
    )?;
    let cross_k = view_cross_kv(
        graph,
        arena.graph_tensor(cross.key),
        d_model,
        head_dim,
        frame_count,
        heads,
        n_seq,
        "dec_cross_k_view",
    )?;
    let cross_v = view_cross_kv(
        graph,
        arena.graph_tensor(cross.value),
        d_model,
        head_dim,
        frame_count,
        heads,
        n_seq,
        "dec_cross_v_view",
    )?;
    let context = scaled_dot_product_attention(
        graph,
        q,
        cross_k,
        cross_v,
        None,
        scale,
        head_dim,
        token_count,
        heads,
        d_model,
        n_seq,
        "dec_cross",
    )?;
    let attn = matmul(
        graph,
        arena,
        layer.cross_o,
        layer.lora.cross_o,
        context,
        "dec_cross_o",
    )?;
    let state = graph
        .add(cross_input, attn)
        .map_err(build_err("dec_cross_residual"))?;

    // --- Gated SwiGLU FFN ---
    // fc1 -> [2*ffn_dim]; HF chunk(2): hidden = first half, gate = second half;
    // out = silu(gate) * hidden.
    let ffn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.ffn_norm),
        "dec_ffn_norm",
    )?;
    let mut fc1 = matmul(
        graph,
        arena,
        layer.ffn_up,
        layer.lora.ffn_up,
        normed,
        "dec_ffn_up",
    )?;
    fc1 = graph
        .add(fc1, arena.graph_tensor(layer.ffn_up_bias))
        .map_err(build_err("dec_ffn_up_bias"))?;
    // fc1 is [2*ffn_dim, token] column-major. The chunk(2) halves are strided along ne0,
    // so express each half as a strided 3D view (row stride = full 2*ffn_dim) then make it
    // contiguous and reshape back to [ffn_dim, token].
    let element_size = std::mem::size_of::<f32>();
    let row_stride = ffn_dim
        .checked_mul(2)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let column_count = token_count
        .checked_mul(n_seq)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let plane_stride = row_stride
        .checked_mul(column_count)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let hidden = graph
        .view_3d(fc1, ffn_dim, column_count, 1, row_stride, plane_stride, 0)
        .map_err(build_err("ggml_view_3d(ffn_hidden)"))?;
    let gate = graph
        .view_3d(
            fc1,
            ffn_dim,
            column_count,
            1,
            row_stride,
            plane_stride,
            ffn_dim * element_size,
        )
        .map_err(build_err("ggml_view_3d(ffn_gate)"))?;
    let hidden = graph
        .cont(hidden)
        .map_err(build_err("ggml_cont(ffn_hidden)"))?;
    let gate = graph.cont(gate).map_err(build_err("ggml_cont(ffn_gate)"))?;
    let hidden = graph
        .reshape_2d(hidden, ffn_dim, column_count)
        .map_err(build_err("ggml_reshape_2d(ffn_hidden)"))?;
    let gate = graph
        .reshape_2d(gate, ffn_dim, column_count)
        .map_err(build_err("ggml_reshape_2d(ffn_gate)"))?;
    let activated = graph.silu(gate).map_err(build_err("ggml_silu(ffn_gate)"))?;
    let fused = graph
        .mul(activated, hidden)
        .map_err(build_err("ggml_mul(ffn_swiglu)"))?;
    let mut down = matmul(
        graph,
        arena,
        layer.ffn_down,
        layer.lora.ffn_down,
        fused,
        "dec_ffn_down",
    )?;
    down = graph
        .add(down, arena.graph_tensor(layer.ffn_down_bias))
        .map_err(build_err("dec_ffn_down_bias"))?;
    let state = graph
        .add(ffn_input, down)
        .map_err(build_err("dec_ffn_residual"))?;

    Ok(state)
}

/// Manual scaled dot-product attention over q,k,v laid out as [head_dim, seq, heads].
/// Returns merged context [d_model, q_len]. Avoids flash_attn_ext (head_dim=36 is not a
/// Metal flash-attention supported size). Optional additive mask is [kv_len, q_len, 1].
#[allow(clippy::too_many_arguments)]
fn scaled_dot_product_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    q: GgmlCpuTensor<'a>,
    k: GgmlCpuTensor<'a>,
    v: GgmlCpuTensor<'a>,
    mask: Option<GgmlCpuTensor<'a>>,
    scale: f32,
    _head_dim: usize,
    q_len: usize,
    _heads: usize,
    d_model: usize,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let _ = step;
    if n_seq == 0 {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: "moonshine attention n_seq must be positive".to_string(),
        });
    }
    let scores = graph
        .mul_mat(k, q)
        .map_err(build_err("ggml_mul_mat(attn_scores)"))?;
    let probs = graph
        .soft_max_ext(scores, mask, scale, 0.0)
        .map_err(build_err("ggml_soft_max_ext(attn_scores)"))?;
    let v_t = graph
        .permute(v, 1, 0, 2, 3)
        .map_err(build_err("ggml_permute(attn_v_t)"))?;
    let v_t = graph.cont(v_t).map_err(build_err("ggml_cont(attn_v_t)"))?;
    let context = graph
        .mul_mat(v_t, probs)
        .map_err(build_err("ggml_mul_mat(attn_ctx)"))?;
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(attn_merge)"))?;
    let merged = graph
        .cont(merged)
        .map_err(build_err("ggml_cont(attn_merge)"))?;
    let output_columns = q_len
        .checked_mul(n_seq)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    graph
        .reshape_2d(merged, d_model, output_columns)
        .map_err(build_err("ggml_reshape_2d(attn_merge)"))
}

#[allow(clippy::too_many_arguments)]
fn rope_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    positions: GgmlCpuTensor<'a>,
    rotary_dim: usize,
    max_context: usize,
    rope_theta: f32,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(build_err("ggml_reshape_3d(rope)"))?;
    let params = GgmlRopeExtParams::moonshine_gptj(rotary_dim, max_context, rope_theta)
        .map_err(build_err("rope_params"))?;
    graph
        .rope_ext(reshaped, positions, params)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn roped_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    roped: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let permuted = graph
        .permute(roped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(rope_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

#[allow(clippy::too_many_arguments)]
fn rope_incremental_heads_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    positions: GgmlCpuTensor<'a>,
    rotary_dim: usize,
    max_context: usize,
    rope_theta: f32,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let roped = rope_heads(
        graph,
        projection,
        head_dim,
        heads,
        n_seq,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        step,
    )?;
    if n_seq == 1 {
        return roped_for_attn(graph, roped, step);
    }
    let roped = graph
        .reshape_4d(roped, head_dim, heads, 1, n_seq)
        .map_err(build_err("ggml_reshape_4d(rope_batch_attn)"))?;
    let permuted = graph
        .permute(roped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(rope_batch_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

#[allow(clippy::too_many_arguments)]
fn rope_heads_for_attn_with_n_seq<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    token_count: usize,
    positions: GgmlCpuTensor<'a>,
    rotary_dim: usize,
    max_context: usize,
    rope_theta: f32,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if n_seq == 1 {
        let roped = rope_heads(
            graph,
            projection,
            head_dim,
            heads,
            token_count,
            positions,
            rotary_dim,
            max_context,
            rope_theta,
            step,
        )?;
        return roped_for_attn(graph, roped, step);
    }
    let output_tokens = token_count
        .checked_mul(n_seq)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let roped = rope_heads(
        graph,
        projection,
        head_dim,
        heads,
        output_tokens,
        positions,
        rotary_dim,
        max_context,
        rope_theta,
        step,
    )?;
    let roped = graph
        .reshape_4d(roped, head_dim, heads, token_count, n_seq)
        .map_err(build_err("ggml_reshape_4d(rope_prefill_attn)"))?;
    let permuted = graph
        .permute(roped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(rope_prefill_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn reshape_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(build_err("ggml_reshape_3d(attn)"))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn reshape_incremental_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if n_seq == 1 {
        return reshape_for_attn(graph, projection, head_dim, heads, 1, step);
    }
    let reshaped = graph
        .reshape_4d(projection, head_dim, heads, 1, n_seq)
        .map_err(build_err("ggml_reshape_4d(batch_attn)"))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(batch_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn reshape_for_attn_with_n_seq<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    token_count: usize,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if n_seq == 1 {
        return reshape_for_attn(graph, projection, head_dim, heads, token_count, step);
    }
    if token_count == 1 {
        return reshape_incremental_for_attn(graph, projection, head_dim, heads, n_seq, step);
    }
    let reshaped = graph
        .reshape_4d(projection, head_dim, heads, token_count, n_seq)
        .map_err(build_err("ggml_reshape_4d(prefill_attn)"))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(prefill_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

/// View a cross cache as attention K/V heads [head_dim, frame, heads, n_seq].
fn view_cross_kv<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    d_model: usize,
    head_dim: usize,
    frame_count: usize,
    heads: usize,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let element_size = std::mem::size_of::<f32>();
    let nb1 = d_model
        .checked_mul(element_size)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let nb2 = head_dim
        .checked_mul(element_size)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    if n_seq > 1 {
        let nb3 = d_model
            .checked_mul(frame_count)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        return graph
            .view_4d(
                tensor,
                head_dim,
                frame_count,
                heads,
                n_seq,
                nb1,
                nb2,
                nb3,
                0,
            )
            .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source });
    }
    graph
        .view_3d(tensor, head_dim, frame_count, heads, nb1, nb2, 0)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

/// View the resident self cache prefix as attention K/V heads
/// `[head_dim, token_count, heads, n_seq]`.
#[allow(clippy::too_many_arguments)]
fn view_self_kv_prefix<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    head_dim: usize,
    token_count: usize,
    heads: usize,
    max_context: usize,
    n_seq: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if n_seq == 0 {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: "moonshine self-KV prefix view n_seq must be positive".to_string(),
        });
    }
    if token_count > max_context {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: format!(
                "moonshine self-KV prefix token_count {token_count} exceeds max context {max_context}"
            ),
        });
    }
    // The resident self-KV cache is f16 (see `ensure_resident_self_kv_arena`),
    // so the view strides are in f16 (2-byte) elements.
    let element_size = std::mem::size_of::<u16>();
    let nb1 = head_dim
        .checked_mul(element_size)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let nb2 = head_dim
        .checked_mul(max_context)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    if n_seq > 1 {
        let nb3 = head_dim
            .checked_mul(max_context)
            .and_then(|value| value.checked_mul(heads))
            .and_then(|value| value.checked_mul(element_size))
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        return graph
            .view_4d(
                tensor,
                head_dim,
                token_count,
                heads,
                n_seq,
                nb1,
                nb2,
                nb3,
                0,
            )
            .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source });
    }
    graph
        .view_3d(tensor, head_dim, token_count, heads, nb1, nb2, 0)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn cross_cache_slot_target<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    cache: GgmlCpuTensor<'a>,
    d_model: usize,
    frame_count: usize,
    n_seq: usize,
    slot_index: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if slot_index >= n_seq {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: format!("moonshine cross-cache slot {slot_index} exceeds n_seq {n_seq}"),
        });
    }
    if n_seq == 1 {
        return Ok(cache);
    }
    let element_size = std::mem::size_of::<f32>();
    let row_stride = d_model
        .checked_mul(element_size)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let slot_stride = d_model
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let offset = slot_stride
        .checked_mul(slot_index)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(cache, d_model, frame_count, row_stride, offset)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn view_last_token<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    d_model: usize,
    token_count: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let contiguous = graph
        .cont(state)
        .map_err(build_err("ggml_cont(last_token)"))?;
    let row_stride = d_model
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let offset = token_count
        .checked_sub(1)
        .and_then(|index| index.checked_mul(row_stride))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(contiguous, d_model, 1, row_stride, offset)
        .map_err(build_err("ggml_view_2d(last_token)"))
}

fn view_batched_last_token<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    d_model: usize,
    token_count: usize,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    if token_count == 0 || n_seq == 0 {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: "batched moonshine last-token view requires positive token_count and n_seq"
                .to_string(),
        });
    }
    if n_seq == 1 {
        return view_last_token(graph, state, d_model, token_count);
    }
    let contiguous = graph
        .cont(state)
        .map_err(build_err("ggml_cont(batched_last_token)"))?;
    let element_size = std::mem::size_of::<f32>();
    let column_stride = d_model
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let offset = token_count
        .checked_sub(1)
        .and_then(|index| index.checked_mul(d_model))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    graph
        .view_2d(contiguous, d_model, n_seq, column_stride, offset)
        .map_err(build_err("ggml_view_2d(batched_last_token)"))
}

/// `y = W@x`, optionally with the dynamic LoRA side branch
/// `y = W@x + B_scaled@(A@x)` when this linear is an adapter target.
fn matmul<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    weight: WeightSlot,
    lora: Option<LoraSlot>,
    input: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let build_failed = |source| MoonshineDecoderGraphError::GraphBuildFailed { step, source };
    let base = graph
        .mul_mat(weight.graph(arena), input)
        .map_err(build_failed)?;
    let Some(lora) = lora else {
        return Ok(base);
    };
    let ax = graph
        .mul_mat(arena.graph_tensor(lora.a), input)
        .map_err(build_failed)?;
    let delta = graph
        .mul_mat(arena.graph_tensor(lora.b_scaled), ax)
        .map_err(build_failed)?;
    graph.add(base, delta).map_err(build_failed)
}

fn apply_weighted_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineDecoderGraphError> {
    let normed = graph
        .norm(input, MOONSHINE_LAYER_NORM_EPSILON)
        .map_err(build_err("ggml_norm(weighted_ln)"))?;
    graph
        .mul(normed, weight)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step, source })
}

fn build_causal_mask_f16_bits(
    token_count: usize,
) -> Result<Arc<[u16]>, MoonshineDecoderGraphError> {
    let total = token_count
        .checked_mul(token_count)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let zero = f32_to_f16_bits(0.0);
    let neg_inf = f32_to_f16_bits(-f32::INFINITY);
    let mut values = vec![zero; total];
    for query_idx in 0..token_count {
        let row = query_idx
            .checked_mul(token_count)
            .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
        for key_idx in 0..token_count {
            if key_idx > query_idx {
                values[row + key_idx] = neg_inf;
            }
        }
    }
    Ok(Arc::<[u16]>::from(values.into_boxed_slice()))
}

fn build_batched_causal_mask_f16_bits(
    token_count: usize,
    n_seq: usize,
) -> Result<Vec<u16>, MoonshineDecoderGraphError> {
    if n_seq == 0 {
        return Err(MoonshineDecoderGraphError::InvalidInput {
            reason: "batched moonshine causal mask requires n_seq > 0".to_string(),
        });
    }
    let single = build_causal_mask_f16_bits(token_count)?;
    let total = single
        .len()
        .checked_mul(n_seq)
        .ok_or(MoonshineDecoderGraphError::ShapeOverflow)?;
    let mut values = Vec::with_capacity(total);
    for _ in 0..n_seq {
        values.extend_from_slice(single.as_ref());
    }
    Ok(values)
}

fn tokens_as_i32(tokens: &[u32]) -> Result<Vec<i32>, MoonshineDecoderGraphError> {
    tokens
        .iter()
        .copied()
        .map(|token| {
            i32::try_from(token).map_err(|_| MoonshineDecoderGraphError::InvalidInput {
                reason: format!("token id {token} does not fit i32"),
            })
        })
        .collect()
}

fn new_vector(
    arena: &GgmlStaticTensorArena,
    len: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MoonshineDecoderGraphError> {
    arena
        .new_tensor_1d_f32(len, name)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step: name, source })
}

fn new_matrix(
    arena: &GgmlStaticTensorArena,
    weight: &MoonshineWeight,
    name: &'static str,
) -> Result<GgmlStaticTensor, MoonshineDecoderGraphError> {
    arena
        .new_tensor_2d_f32(weight.dims[0], weight.dims[1], name)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step: name, source })
}

fn new_cross_cache(
    arena: &GgmlStaticTensorArena,
    d_model: usize,
    frame_count: usize,
    n_seq: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MoonshineDecoderGraphError> {
    if n_seq > 1 {
        return arena
            .new_tensor_3d_f32(d_model, frame_count, n_seq, name)
            .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step: name, source });
    }
    arena
        .new_tensor_2d_f32(d_model, frame_count, name)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step: name, source })
}

fn upload(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &MoonshineWeight,
    name: &'static str,
) -> Result<(), MoonshineDecoderGraphError> {
    arena
        .set_f32_slice(tensor, &weight.values, name)
        .map_err(|source| MoonshineDecoderGraphError::GraphBuildFailed { step: name, source })
}

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    runtime: &MoonshineDecoderLayerRuntime,
    layer: &MoonshineDecoderLayerWeights,
) -> Result<(), MoonshineDecoderGraphError> {
    // The 2-D linears (self/cross attn q/k/v/o, ffn up/down) are bound zero-copy
    // from the mmap'd pack (WeightSlot::Loaded, meta-only host) — only the
    // arena-resident norms + biases are uploaded here.
    upload(arena, runtime.attn_norm, &layer.attn_norm, "dec_attn_norm")?;
    upload(
        arena,
        runtime.cross_norm,
        &layer.cross_norm,
        "dec_cross_norm",
    )?;
    upload(arena, runtime.ffn_norm, &layer.ffn_norm, "dec_ffn_norm")?;
    upload(
        arena,
        runtime.ffn_up_bias,
        &layer.ffn_up_bias,
        "dec_ffn_up_b",
    )?;
    upload(
        arena,
        runtime.ffn_down_bias,
        &layer.ffn_down_bias,
        "dec_ffn_down_b",
    )
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa_with_hidden >> shift) as u16;
        let round_bit = 1_u32 << shift.saturating_sub(1);
        if shift > 0
            && (mantissa_with_hidden & round_bit) != 0
            && ((mantissa_with_hidden & (round_bit - 1)) != 0 || (half_mantissa & 1) != 0)
        {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }
    let mut half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x1000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

fn build_err(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> MoonshineDecoderGraphError {
    move |source| MoonshineDecoderGraphError::GraphBuildFailed { step, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
    };

    const MOONSHINE_BATCH_REAL_PACK_ENV: &str = "OPENASR_MOONSHINE_BATCH_REAL_PACK";

    fn read_runtime_source_preflight(runtime_path: &Path) -> GgmlAsrRuntimeSourcePreflight {
        let runtime_source =
            validate_ggml_runtime_source_path(runtime_path).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        }
    }

    fn sample_encoder_output(
        metadata: MoonshineExecutionMetadata,
        phase: f32,
        frame_count: usize,
    ) -> MoonshineEncoderOutput {
        let mut rows = Vec::with_capacity(frame_count * metadata.d_model);
        for frame_idx in 0..frame_count {
            for hidden_idx in 0..metadata.d_model {
                rows.push(
                    (((frame_idx * metadata.d_model + hidden_idx) as f32 * 0.03125) + phase).sin(),
                );
            }
        }
        MoonshineEncoderOutput {
            frame_count,
            hidden_size: metadata.d_model,
            rows,
        }
    }

    fn assert_argmax_matches(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .inspect(|(_, value)| assert!(value.is_finite()))
                .max_by(|(_, left), (_, right)| {
                    left.partial_cmp(right)
                        .expect("finite logits are comparable")
                })
                .map(|(index, _)| index)
                .expect("logits must be non-empty")
        };
        assert_eq!(argmax(left), argmax(right));
    }

    #[test]
    fn reusable_decode_graph_is_disabled_on_cpu_direct_runner() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
        let runner = GgmlCpuGraphRunner::new(config).expect("direct runner should initialize");

        assert!(!reusable_decode_graph_supported_for_runner(&runner));
    }

    #[test]
    fn reusable_decode_graph_is_disabled_on_scheduler_runner() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = true;
        let runner = GgmlCpuGraphRunner::new(config).expect("scheduler runner should initialize");

        assert!(!reusable_decode_graph_supported_for_runner(&runner));
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_decoder_runtime_batched_real_pack_selected_backend_logits_match_serial_argmax() {
        let runtime_path = std::env::var_os(MOONSHINE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{MOONSHINE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack")
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared = super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
            .expect("prepared runtime");
        let metadata = prepared.metadata;
        let encoder_output_0 = sample_encoder_output(metadata, 0.0, 32);
        let encoder_output_1 = sample_encoder_output(metadata, 0.25, 32);

        let mut serial_runtime_0 = MoonshineDecoderGraphRuntime::new(
            &prepared.decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            false,
            Some(runtime_path.as_path()),
            None,
        )
        .expect("serial runtime 0");
        serial_runtime_0
            .populate_cross_attention_cache(&encoder_output_0)
            .expect("serial cross cache 0");
        let serial_logits_0 = serial_runtime_0
            .compute_incremental_step_logits(metadata.bos_token_id, 0)
            .expect("serial logits 0");

        let mut serial_runtime_1 = MoonshineDecoderGraphRuntime::new(
            &prepared.decoder_weights,
            metadata,
            encoder_output_1.frame_count,
            false,
            Some(runtime_path.as_path()),
            None,
        )
        .expect("serial runtime 1");
        serial_runtime_1
            .populate_cross_attention_cache(&encoder_output_1)
            .expect("serial cross cache 1");
        let serial_logits_1 = serial_runtime_1
            .compute_incremental_step_logits(metadata.bos_token_id, 0)
            .expect("serial logits 1");

        let mut batched_runtime = MoonshineDecoderGraphRuntime::new_with_n_seq(
            &prepared.decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            false,
            Some(runtime_path.as_path()),
            2,
            None,
        )
        .expect("batched runtime");
        batched_runtime
            .populate_cross_attention_cache_slot(0, &encoder_output_0)
            .expect("batched cross cache 0");
        batched_runtime
            .populate_cross_attention_cache_slot(1, &encoder_output_1)
            .expect("batched cross cache 1");
        let batched_logits = batched_runtime
            .compute_reused_batched_step_logits(
                &[metadata.bos_token_id, metadata.bos_token_id],
                &[0, 0],
                &[1, 1],
            )
            .expect("batched logits");

        assert_eq!(batched_logits.len(), metadata.vocab_size * 2);
        assert_argmax_matches(&batched_logits[0..metadata.vocab_size], &serial_logits_0);
        assert_argmax_matches(&batched_logits[metadata.vocab_size..], &serial_logits_1);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_MOONSHINE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn moonshine_decoder_runtime_batched_prefill_real_pack_logits_match_serial_argmax() {
        let runtime_path = std::env::var_os(MOONSHINE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{MOONSHINE_BATCH_REAL_PACK_ENV} must point to a moonshine .oasr model pack")
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let prepared = super::super::prepared_runtime::build_moonshine_prepared_runtime(&preflight)
            .expect("prepared runtime");
        let metadata = prepared.metadata;
        let encoder_output_0 = sample_encoder_output(metadata, 0.0, 32);
        let encoder_output_1 = sample_encoder_output(metadata, 0.25, 32);
        let prompt = vec![metadata.bos_token_id, metadata.bos_token_id];
        let followup = metadata.bos_token_id;

        let mut serial_runtime_0 = MoonshineDecoderGraphRuntime::new(
            &prepared.decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            false,
            Some(runtime_path.as_path()),
            None,
        )
        .expect("serial runtime 0");
        serial_runtime_0
            .populate_cross_attention_cache(&encoder_output_0)
            .expect("serial cross cache 0");
        let serial_prefill_logits_0 = serial_runtime_0
            .compute_full_prefix_step_logits(&prompt)
            .expect("serial prefill logits 0");
        let mut serial_followup_0 = prompt.clone();
        serial_followup_0.push(followup);
        let serial_followup_logits_0 = serial_runtime_0
            .compute_full_prefix_step_logits(&serial_followup_0)
            .expect("serial followup logits 0");

        let mut serial_runtime_1 = MoonshineDecoderGraphRuntime::new(
            &prepared.decoder_weights,
            metadata,
            encoder_output_1.frame_count,
            false,
            Some(runtime_path.as_path()),
            None,
        )
        .expect("serial runtime 1");
        serial_runtime_1
            .populate_cross_attention_cache(&encoder_output_1)
            .expect("serial cross cache 1");
        let serial_prefill_logits_1 = serial_runtime_1
            .compute_full_prefix_step_logits(&prompt)
            .expect("serial prefill logits 1");
        let mut serial_followup_1 = prompt.clone();
        serial_followup_1.push(followup);
        let serial_followup_logits_1 = serial_runtime_1
            .compute_full_prefix_step_logits(&serial_followup_1)
            .expect("serial followup logits 1");

        let mut batched_runtime = MoonshineDecoderGraphRuntime::new_with_n_seq(
            &prepared.decoder_weights,
            metadata,
            encoder_output_0.frame_count,
            false,
            Some(runtime_path.as_path()),
            2,
            None,
        )
        .expect("batched runtime");
        batched_runtime
            .populate_cross_attention_cache_slot(0, &encoder_output_0)
            .expect("batched cross cache 0");
        batched_runtime
            .populate_cross_attention_cache_slot(1, &encoder_output_1)
            .expect("batched cross cache 1");
        let batched_prefill_logits = batched_runtime
            .compute_batched_prefill_logits(&prompt)
            .expect("batched prefill logits");
        assert_eq!(batched_prefill_logits.len(), metadata.vocab_size * 2);
        assert_argmax_matches(
            &batched_prefill_logits[0..metadata.vocab_size],
            &serial_prefill_logits_0,
        );
        assert_argmax_matches(
            &batched_prefill_logits[metadata.vocab_size..],
            &serial_prefill_logits_1,
        );

        let batched_followup_logits = batched_runtime
            .compute_reused_batched_step_logits(&[followup, followup], &[2, 2], &[3, 3])
            .expect("batched followup logits");
        assert_eq!(batched_followup_logits.len(), metadata.vocab_size * 2);
        assert_argmax_matches(
            &batched_followup_logits[0..metadata.vocab_size],
            &serial_followup_logits_0,
        );
        assert_argmax_matches(
            &batched_followup_logits[metadata.vocab_size..],
            &serial_followup_logits_1,
        );
    }
}
