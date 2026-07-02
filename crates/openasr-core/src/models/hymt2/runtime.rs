use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgufMetadataReadError, GgufTensorDataReadError, GgufTensorDataReader,
    GgufTensorIndexReadError,
};
use crate::models::hymt2::prompt::{
    build_hymt2_subtitle_prompt_token_parts, build_hymt2_user_chat_prompt_tokens,
    build_subtitle_translation_prompt, max_output_tokens_for_source_tokens,
};
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmFusedLogitsHeadSpec, Qwen3AsrLlmLayerAttentionProjection,
    Qwen3AsrLlmLogitsHead, Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrLlmWholeStepOutput,
    Qwen3AsrLlmWholeStepTop1Output, Qwen3AsrTokenEmbeddingTable, even_prefill_chunk_len,
    load_qwen3_llm_attention_projections_from_reader_with_materialized_qkv,
    load_qwen3_llm_logits_head_from_reader_with_output_tensor,
    load_qwen3_token_embedding_table_from_reader,
};
use crate::{
    GgmlRuntimeSourcePathError, NativeAsrError, read_gguf_metadata_from_runtime_source,
    read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
};

use super::config::{
    Hymt2ConfigError, Hymt2ExecutionMetadata, parse_hymt2_execution_metadata,
    validate_hymt2_runtime_tensors_with_index,
};
use super::tensor_names::TOKEN_EMBD_WEIGHT;
use super::tokenizer::Hymt2Tokenizer;

const HYMT2_PROFILE_ENV: &str = "OPENASR_HYMT2_PROFILE";
const HYMT2_PREFILL_CHUNK_TOKENS_ENV: &str = "OPENASR_HYMT2_PREFILL_CHUNK_TOKENS";
const HYMT2_METAL_PREFILL_QUERY_TOKENS: usize = 64;

#[derive(Debug)]
pub struct Hymt2Runtime {
    metadata: Hymt2ExecutionMetadata,
    tokenizer: Hymt2Tokenizer,
    token_embedding_table: Qwen3AsrTokenEmbeddingTable,
    logits_head: Qwen3AsrLlmLogitsHead,
    session: Mutex<Hymt2RuntimeSession>,
}

#[derive(Debug)]
struct Hymt2RuntimeSession {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hymt2DecodeResult {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub text: String,
    pub first_step_logits: Vec<f32>,
    pub timings: Hymt2DecodeTimings,
}

impl Hymt2DecodeResult {
    pub fn prefix_reuse_report(&self) -> Hymt2PrefixReuseReport {
        Hymt2PrefixReuseReport {
            prompt_tokens: self.timings.prompt_tokens,
            source_prefix_tokens: self.prompt_tokens.len().saturating_sub(1),
            reused_prefix_tokens: self.timings.reused_prefix_tokens,
            prefilled_tokens: self.timings.prefilled_tokens,
            cache_backoff_tokens: self.timings.cache_backoff_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hymt2DecodeTimings {
    pub prefill: Duration,
    pub decode: Duration,
    pub total: Duration,
    pub prompt_tokens: usize,
    pub prefilled_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub cache_backoff_tokens: usize,
    pub generated_tokens: usize,
}

impl Hymt2DecodeTimings {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.prefilled_tokens, self.prefill)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.generated_tokens, self.decode)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hymt2PrefixCacheConfig {
    pub unstable_tail_backoff_tokens: usize,
}

impl Default for Hymt2PrefixCacheConfig {
    fn default() -> Self {
        Self {
            unstable_tail_backoff_tokens: 2,
        }
    }
}

#[derive(Debug, Default)]
pub struct Hymt2TranslationSessionCache {
    config: Hymt2PrefixCacheConfig,
    active: Option<Hymt2ActivePrefixCache>,
}

#[derive(Debug)]
struct Hymt2ActivePrefixCache {
    clause_id: String,
    static_context_tokens: Vec<u32>,
    source_prefix_tokens: Vec<u32>,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    max_positions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hymt2PrefixReuseReport {
    pub prompt_tokens: usize,
    pub source_prefix_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub prefilled_tokens: usize,
    pub cache_backoff_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Hymt2PrefixReusePlan {
    reused_prefix_tokens: usize,
    cache_backoff_tokens: usize,
}

impl Hymt2TranslationSessionCache {
    pub fn new(config: Hymt2PrefixCacheConfig) -> Self {
        Self {
            config,
            active: None,
        }
    }

    pub fn invalidate(&mut self) {
        self.active = None;
    }

    pub fn active_prefix_token_count(&self) -> usize {
        self.active
            .as_ref()
            .map(|active| active.source_prefix_tokens.len())
            .unwrap_or(0)
    }
}

#[derive(Debug, Error)]
pub enum Hymt2RuntimeError {
    #[error("hymt2 runtime source path is invalid: {source}")]
    RuntimeSourcePath {
        #[source]
        source: GgmlRuntimeSourcePathError,
    },
    #[error("hymt2 GGUF metadata read failed: {source}")]
    MetadataRead {
        #[source]
        source: GgufMetadataReadError,
    },
    #[error("hymt2 GGUF tensor index read failed: {source}")]
    TensorIndexRead {
        #[source]
        source: GgufTensorIndexReadError,
    },
    #[error("hymt2 config invalid: {source}")]
    Config {
        #[source]
        source: Hymt2ConfigError,
    },
    #[error("hymt2 tokenizer failed: {source}")]
    Tokenizer {
        #[source]
        source: NativeAsrError,
    },
    #[error("hymt2 tensor reader failed: {source}")]
    TensorReader {
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("hymt2 weight materialization failed: {reason}")]
    WeightMaterialization { reason: String },
    #[error("hymt2 prompt is empty")]
    EmptyPrompt,
    #[error("hymt2 source clause is empty")]
    EmptySource,
    #[error(
        "hymt2 prompt/generation length exceeds runtime context: prompt_tokens={prompt_tokens}, max_output_tokens={max_output_tokens}, runtime_context={runtime_context}"
    )]
    ContextExceeded {
        prompt_tokens: usize,
        max_output_tokens: usize,
        runtime_context: usize,
    },
    #[error("hymt2 graph failed: {source}")]
    Graph {
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("hymt2 decode failed: {reason}")]
    Decode { reason: String },
}

impl Hymt2RuntimeSession {
    fn new(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source_path: &Path,
        metadata: Hymt2ExecutionMetadata,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
    ) -> Result<Self, GgmlCpuGraphError> {
        let whole_decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head(
                projections,
                Some(runtime_source_path),
                metadata.rms_norm_epsilon,
                fused_logits_head,
            )?;
        Ok(Self {
            whole_decoder,
            layer_kv_caches: Vec::new(),
        })
    }

    fn reset_layer_kv_caches(
        &mut self,
        metadata: Hymt2ExecutionMetadata,
        max_positions: usize,
    ) -> Result<(), Hymt2RuntimeError> {
        if self.layer_kv_caches.len() != metadata.layers {
            self.layer_kv_caches = (0..metadata.layers)
                .map(|_| {
                    Qwen3AsrLayerKvCacheState::new(
                        max_positions,
                        metadata.kv_heads,
                        metadata.head_dim,
                    )
                })
                .collect();
            return Ok(());
        }

        for cache in &mut self.layer_kv_caches {
            cache.clear_written_positions();
            cache
                .resize_max_positions(max_positions)
                .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
        }
        Ok(())
    }
}

impl Hymt2Runtime {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Hymt2RuntimeError> {
        let runtime_source = validate_ggml_runtime_source_path(path.as_ref())
            .map_err(|source| Hymt2RuntimeError::RuntimeSourcePath { source })?;
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .map_err(|source| Hymt2RuntimeError::MetadataRead { source })?;
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .map_err(|source| Hymt2RuntimeError::TensorIndexRead { source })?;
        let hymt2_metadata = parse_hymt2_execution_metadata(&metadata)
            .map_err(|source| Hymt2RuntimeError::Config { source })?;
        validate_hymt2_runtime_tensors_with_index(&tensor_index, hymt2_metadata)
            .map_err(|source| Hymt2RuntimeError::Config { source })?;

        let tokenizer = Hymt2Tokenizer::from_gguf_metadata(&metadata)
            .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?;
        let reader = GgufTensorDataReader::from_tensor_index_shared(Arc::new(tensor_index))
            .map_err(|source| Hymt2RuntimeError::TensorReader { source })?;
        let qwen_metadata = hymt2_metadata.qwen_llm_metadata();
        let token_embedding_table =
            load_qwen3_token_embedding_table_from_reader(&reader, qwen_metadata).map_err(
                |error| Hymt2RuntimeError::WeightMaterialization {
                    reason: error.to_string(),
                },
            )?;
        let logits_head = load_qwen3_llm_logits_head_from_reader_with_output_tensor(
            &reader,
            qwen_metadata,
            TOKEN_EMBD_WEIGHT,
            hymt2_metadata.rms_norm_epsilon,
        )
        .map_err(|error| Hymt2RuntimeError::WeightMaterialization {
            reason: error.to_string(),
        })?;
        let layer_attention_projections =
            load_qwen3_llm_attention_projections_from_reader_with_materialized_qkv(
                &reader,
                qwen_metadata,
            )
            .map_err(|error| Hymt2RuntimeError::WeightMaterialization {
                reason: error.to_string(),
            })?;
        if layer_attention_projections.len() != hymt2_metadata.layers {
            return Err(Hymt2RuntimeError::WeightMaterialization {
                reason: format!(
                    "loaded {} LLM layers, expected {}",
                    layer_attention_projections.len(),
                    hymt2_metadata.layers
                ),
            });
        }
        let session = Hymt2RuntimeSession::new(
            &layer_attention_projections,
            runtime_source.path(),
            hymt2_metadata,
            logits_head.fused_top1_spec(),
        )
        .map_err(|source| Hymt2RuntimeError::Graph { source })?;
        if hymt2_profile_enabled() {
            eprintln!(
                "openasr_hymt2_profile: stage=runtime_backend backend={}",
                session.whole_decoder.backend_label()
            );
        }

        Ok(Self {
            metadata: hymt2_metadata,
            tokenizer,
            token_embedding_table,
            logits_head,
            session: Mutex::new(session),
        })
    }

    pub fn probe_path(path: impl AsRef<Path>) -> Result<Hymt2ExecutionMetadata, Hymt2RuntimeError> {
        let runtime_source = validate_ggml_runtime_source_path(path.as_ref())
            .map_err(|source| Hymt2RuntimeError::RuntimeSourcePath { source })?;
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .map_err(|source| Hymt2RuntimeError::MetadataRead { source })?;
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .map_err(|source| Hymt2RuntimeError::TensorIndexRead { source })?;
        let hymt2_metadata = parse_hymt2_execution_metadata(&metadata)
            .map_err(|source| Hymt2RuntimeError::Config { source })?;
        validate_hymt2_runtime_tensors_with_index(&tensor_index, hymt2_metadata)
            .map_err(|source| Hymt2RuntimeError::Config { source })?;
        Ok(hymt2_metadata)
    }

    pub fn metadata(&self) -> Hymt2ExecutionMetadata {
        self.metadata
    }

    pub fn tokenizer(&self) -> &Hymt2Tokenizer {
        &self.tokenizer
    }

    /// Returns `Hymt2RuntimeError::EmptySource` for empty or whitespace-only clauses.
    pub fn translate_clause(
        &self,
        source_clause: &str,
        finalized_context: &[(&str, &str)],
    ) -> Result<Hymt2DecodeResult, Hymt2RuntimeError> {
        validate_non_empty_source_clause(source_clause)?;
        let prompt = build_subtitle_translation_prompt(source_clause, finalized_context);
        let prompt_tokens = build_hymt2_user_chat_prompt_tokens(&self.tokenizer, &prompt)
            .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?;
        let source_tokens = self
            .tokenizer
            .encode_content_text(source_clause)
            .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?;
        let max_output_tokens = max_output_tokens_for_source_tokens(source_tokens.len());
        self.decode_prompt_tokens(prompt_tokens, max_output_tokens)
    }

    pub fn translate_clause_with_cache(
        &self,
        cache: &mut Hymt2TranslationSessionCache,
        clause_id: impl AsRef<str>,
        source_clause: &str,
        finalized_context: &[(&str, &str)],
        finalized: bool,
    ) -> Result<Hymt2DecodeResult, Hymt2RuntimeError> {
        validate_non_empty_source_clause(source_clause)?;
        let clause_id = clause_id.as_ref();
        let parts = build_hymt2_subtitle_prompt_token_parts(
            &self.tokenizer,
            source_clause,
            finalized_context,
        )
        .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?;
        if parts.prompt_tokens.is_empty() || parts.source_prefix_tokens.is_empty() {
            return Err(Hymt2RuntimeError::EmptyPrompt);
        }
        let source_prefix_len = parts.source_prefix_tokens.len();
        let prompt_len = parts.prompt_tokens.len();
        let source_tokens = parts.source_tokens.len();
        let max_output_tokens = max_output_tokens_for_source_tokens(source_tokens);
        let Some(context_remaining) = self.metadata.runtime_context_length.checked_sub(prompt_len)
        else {
            return Err(Hymt2RuntimeError::ContextExceeded {
                prompt_tokens: prompt_len,
                max_output_tokens,
                runtime_context: self.metadata.runtime_context_length,
            });
        };
        let max_output_tokens = max_output_tokens.min(context_remaining);
        if max_output_tokens == 0 {
            return Err(Hymt2RuntimeError::ContextExceeded {
                prompt_tokens: prompt_len,
                max_output_tokens,
                runtime_context: self.metadata.runtime_context_length,
            });
        }
        let max_positions =
            prompt_len
                .checked_add(max_output_tokens)
                .ok_or_else(|| Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 cached decode max positions overflowed".to_string(),
                })?;

        let total_started_at = Instant::now();
        let prefill_started_at = Instant::now();
        let (mut source_prefilled_tokens, reuse_plan) = self.update_source_prefix_cache(
            cache,
            clause_id,
            &parts.source_prefix_tokens,
            parts.static_context_token_count,
            finalized,
            max_positions,
        )?;

        let active = cache
            .active
            .as_ref()
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefix cache missing after source prefill".to_string(),
            })?;
        let decode_layer_kv_caches = active
            .layer_kv_caches
            .iter()
            .map(|layer| layer.fork_prefix(source_prefix_len, max_positions))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
        let marker_embeddings = self
            .token_embedding_table
            .gather_rows(&parts.generation_marker_tokens)
            .map_err(|error| Hymt2RuntimeError::Decode {
                reason: error.to_string(),
            })?;
        source_prefilled_tokens = source_prefilled_tokens
            .checked_add(parts.generation_marker_tokens.len())
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 cached prefill token count overflowed".to_string(),
            })?;
        let stop_token_ids = self.tokenizer.stop_token_ids();
        let (first_step_logits, generated_tokens, prefill, decode) = {
            let session_started_at = hymt2_profile_start();
            let mut session = self.session.lock().map_err(|_| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 runtime session lock poisoned".to_string(),
            })?;
            session.layer_kv_caches = decode_layer_kv_caches;
            hymt2_profile_log_opt("session_lock_cached_decode", session_started_at);
            let Hymt2RuntimeSession {
                whole_decoder,
                layer_kv_caches,
            } = &mut *session;
            let mut stepper = Hymt2GreedyStepper {
                metadata: self.metadata,
                token_embedding_table: &self.token_embedding_table,
                logits_head: &self.logits_head,
                whole_decoder,
                layer_kv_caches: layer_kv_caches.as_mut_slice(),
                prompt_token_count: prompt_len,
                decode_graph_initialized: false,
            };
            let logits = stepper.prefill_tokens_at_offset_and_compute_last_logits(
                &marker_embeddings,
                source_prefix_len,
                parts.generation_marker_tokens.len(),
            )?;
            let prefill = prefill_started_at.elapsed();
            let first_step_logits = logits.clone();

            let mut generated_tokens = Vec::with_capacity(max_output_tokens);
            let decode_started_at = Instant::now();
            let mut token_id = greedy_argmax(&logits)?;
            for _ in 0..max_output_tokens {
                if stop_token_ids.contains(&token_id) {
                    break;
                }
                generated_tokens.push(token_id);
                if generated_tokens.len() == max_output_tokens {
                    break;
                }
                token_id = stepper.decode_next_token_id(&generated_tokens)?;
            }
            let decode = decode_started_at.elapsed();
            (first_step_logits, generated_tokens, prefill, decode)
        };
        let text = self
            .tokenizer
            .decode_text_token_ids(&generated_tokens)
            .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?
            .trim()
            .to_string();
        let generated_token_count = generated_tokens.len();
        if finalized {
            cache.invalidate();
        }
        Ok(Hymt2DecodeResult {
            prompt_tokens: parts.prompt_tokens,
            generated_tokens,
            text,
            first_step_logits,
            timings: Hymt2DecodeTimings {
                prefill,
                decode,
                total: total_started_at.elapsed(),
                prompt_tokens: prompt_len,
                prefilled_tokens: source_prefilled_tokens,
                reused_prefix_tokens: reuse_plan.reused_prefix_tokens,
                cache_backoff_tokens: reuse_plan.cache_backoff_tokens,
                generated_tokens: generated_token_count,
            },
        })
    }

    pub fn translate_request_with_cache(
        &self,
        cache: &mut Hymt2TranslationSessionCache,
        request: &crate::translation::TranslationRequest,
    ) -> Result<crate::translation::TranslationWorkerOutput, Hymt2RuntimeError> {
        let context = request
            .finalized_context
            .iter()
            .map(|entry| (entry.source_text.as_str(), entry.target_text.as_str()))
            .collect::<Vec<_>>();
        let result = self.translate_clause_with_cache(
            cache,
            request.clause_id.to_string(),
            &request.source_text,
            &context,
            request.finalized,
        )?;
        let text = crate::translation::align_translation_terminal_punctuation(
            &request.source_text,
            &result.text,
        )
        .unwrap_or(result.text);
        Ok(crate::translation::TranslationWorkerOutput {
            text,
            timings: crate::translation::TranslationTimings {
                prefill: result.timings.prefill,
                decode: result.timings.decode,
                total: result.timings.total,
                prompt_tokens: result.timings.prompt_tokens,
                prefilled_tokens: result.timings.prefilled_tokens,
                reused_prefix_tokens: result.timings.reused_prefix_tokens,
                cache_backoff_tokens: result.timings.cache_backoff_tokens,
                generated_tokens: result.timings.generated_tokens,
                ..crate::translation::TranslationTimings::default()
            },
        })
    }

    pub fn decode_prompt_tokens(
        &self,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
    ) -> Result<Hymt2DecodeResult, Hymt2RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(Hymt2RuntimeError::EmptyPrompt);
        }
        let Some(context_remaining) = self
            .metadata
            .runtime_context_length
            .checked_sub(prompt_tokens.len())
        else {
            return Err(Hymt2RuntimeError::ContextExceeded {
                prompt_tokens: prompt_tokens.len(),
                max_output_tokens,
                runtime_context: self.metadata.runtime_context_length,
            });
        };
        let max_output_tokens = max_output_tokens.min(context_remaining);
        if max_output_tokens == 0 {
            return Err(Hymt2RuntimeError::ContextExceeded {
                prompt_tokens: prompt_tokens.len(),
                max_output_tokens,
                runtime_context: self.metadata.runtime_context_length,
            });
        }

        let total_started_at = Instant::now();
        let gather_started_at = hymt2_profile_start();
        let prompt_embeddings = self
            .token_embedding_table
            .gather_rows(&prompt_tokens)
            .map_err(|error| Hymt2RuntimeError::Decode {
                reason: error.to_string(),
            })?;
        hymt2_profile_log_opt("token_embedding_gather", gather_started_at);
        let session_started_at = hymt2_profile_start();
        let mut session = self.session.lock().map_err(|_| Hymt2RuntimeError::Decode {
            reason: "Hy-MT2 runtime session lock poisoned".to_string(),
        })?;
        session.reset_layer_kv_caches(
            self.metadata,
            prompt_tokens.len().saturating_add(max_output_tokens),
        )?;
        hymt2_profile_log_opt("session_lock_reset", session_started_at);
        let Hymt2RuntimeSession {
            whole_decoder,
            layer_kv_caches,
        } = &mut *session;
        let mut stepper = Hymt2GreedyStepper {
            metadata: self.metadata,
            token_embedding_table: &self.token_embedding_table,
            logits_head: &self.logits_head,
            whole_decoder,
            layer_kv_caches: layer_kv_caches.as_mut_slice(),
            prompt_token_count: prompt_tokens.len(),
            decode_graph_initialized: false,
        };
        let prefill_started_at = Instant::now();
        let logits = stepper.prefill_prompt_and_compute_last_logits(&prompt_embeddings)?;
        let prefill = prefill_started_at.elapsed();
        let first_step_logits = logits.clone();
        let stop_token_ids = self.tokenizer.stop_token_ids();
        let mut generated_tokens = Vec::with_capacity(max_output_tokens);
        let decode_started_at = Instant::now();
        let mut token_id = greedy_argmax(&logits)?;
        for _ in 0..max_output_tokens {
            if stop_token_ids.contains(&token_id) {
                break;
            }
            generated_tokens.push(token_id);
            if generated_tokens.len() == max_output_tokens {
                break;
            }
            token_id = stepper.decode_next_token_id(&generated_tokens)?;
        }
        let decode = decode_started_at.elapsed();
        let text = self
            .tokenizer
            .decode_text_token_ids(&generated_tokens)
            .map_err(|source| Hymt2RuntimeError::Tokenizer { source })?
            .trim()
            .to_string();
        let generated_token_count = generated_tokens.len();
        Ok(Hymt2DecodeResult {
            prompt_tokens: prompt_tokens.clone(),
            generated_tokens,
            text,
            first_step_logits,
            timings: Hymt2DecodeTimings {
                prefill,
                decode,
                total: total_started_at.elapsed(),
                prompt_tokens: prompt_tokens.len(),
                prefilled_tokens: prompt_tokens.len(),
                reused_prefix_tokens: 0,
                cache_backoff_tokens: 0,
                generated_tokens: generated_token_count,
            },
        })
    }

    fn update_source_prefix_cache(
        &self,
        cache: &mut Hymt2TranslationSessionCache,
        clause_id: &str,
        source_prefix_tokens: &[u32],
        static_context_token_count: usize,
        finalized: bool,
        max_positions: usize,
    ) -> Result<(usize, Hymt2PrefixReusePlan), Hymt2RuntimeError> {
        update_hymt2_source_prefix_cache(
            cache,
            clause_id,
            source_prefix_tokens,
            static_context_token_count,
            finalized,
            max_positions,
            |max_positions| self.empty_layer_kv_caches(max_positions),
            |mut layer_kv_caches, suffix_tokens, reused_prefix_tokens| {
                let suffix_embeddings = self
                    .token_embedding_table
                    .gather_rows(suffix_tokens)
                    .map_err(|error| Hymt2RuntimeError::Decode {
                        reason: error.to_string(),
                    })?;
                let session_started_at = hymt2_profile_start();
                let mut session = self.session.lock().map_err(|_| Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 runtime session lock poisoned".to_string(),
                })?;
                hymt2_profile_log_opt("session_lock_prefix_prefill", session_started_at);
                let mut stepper = Hymt2GreedyStepper {
                    metadata: self.metadata,
                    token_embedding_table: &self.token_embedding_table,
                    logits_head: &self.logits_head,
                    whole_decoder: &mut session.whole_decoder,
                    layer_kv_caches: layer_kv_caches.as_mut_slice(),
                    prompt_token_count: source_prefix_tokens.len(),
                    decode_graph_initialized: false,
                };
                stepper.prefill_tokens_at_offset_and_compute_last_logits(
                    &suffix_embeddings,
                    reused_prefix_tokens,
                    suffix_tokens.len(),
                )?;
                Ok(layer_kv_caches)
            },
        )
    }

    fn empty_layer_kv_caches(&self, max_positions: usize) -> Vec<Qwen3AsrLayerKvCacheState> {
        (0..self.metadata.layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    max_positions,
                    self.metadata.kv_heads,
                    self.metadata.head_dim,
                )
            })
            .collect()
    }
}

fn update_hymt2_source_prefix_cache(
    cache: &mut Hymt2TranslationSessionCache,
    clause_id: &str,
    source_prefix_tokens: &[u32],
    static_context_token_count: usize,
    finalized: bool,
    max_positions: usize,
    empty_layer_kv_caches: impl FnOnce(usize) -> Vec<Qwen3AsrLayerKvCacheState>,
    prefill_suffix: impl FnOnce(
        Vec<Qwen3AsrLayerKvCacheState>,
        &[u32],
        usize,
    ) -> Result<Vec<Qwen3AsrLayerKvCacheState>, Hymt2RuntimeError>,
) -> Result<(usize, Hymt2PrefixReusePlan), Hymt2RuntimeError> {
    let result = (|| {
        let static_context_tokens = source_prefix_tokens
            .get(..static_context_token_count)
            .unwrap_or(&[]);
        let should_reset = match cache.active.as_ref() {
            Some(active) => {
                active.clause_id != clause_id
                    || active.static_context_tokens != static_context_tokens
            }
            None => true,
        };
        if should_reset {
            cache.active = Some(Hymt2ActivePrefixCache {
                clause_id: clause_id.to_string(),
                static_context_tokens: static_context_tokens.to_vec(),
                source_prefix_tokens: Vec::new(),
                layer_kv_caches: empty_layer_kv_caches(max_positions),
                max_positions,
            });
        }

        let active = cache
            .active
            .as_mut()
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefix cache missing after reset".to_string(),
            })?;
        if active.max_positions < max_positions {
            for layer in &mut active.layer_kv_caches {
                layer
                    .resize_max_positions(max_positions)
                    .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
            }
            active.max_positions = max_positions;
        }

        let reuse_plan = plan_hymt2_prefix_reuse(
            &active.source_prefix_tokens,
            source_prefix_tokens,
            static_context_token_count,
            cache.config.unstable_tail_backoff_tokens,
            finalized,
        );
        let mut staged_layer_kv_caches = active.layer_kv_caches.clone();
        for layer in &mut staged_layer_kv_caches {
            layer
                .truncate_written_positions(reuse_plan.reused_prefix_tokens)
                .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
        }

        let suffix_tokens = source_prefix_tokens
            .get(reuse_plan.reused_prefix_tokens..)
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 source prefix reuse span is out of bounds".to_string(),
            })?;
        let suffix_len = suffix_tokens.len();
        if !suffix_tokens.is_empty() {
            staged_layer_kv_caches = prefill_suffix(
                staged_layer_kv_caches,
                suffix_tokens,
                reuse_plan.reused_prefix_tokens,
            )?;
        }
        active.layer_kv_caches = staged_layer_kv_caches;
        active.source_prefix_tokens = source_prefix_tokens.to_vec();
        Ok((suffix_len, reuse_plan))
    })();
    if result.is_err() {
        cache.invalidate();
    }
    result
}

fn validate_non_empty_source_clause(source_clause: &str) -> Result<(), Hymt2RuntimeError> {
    if source_clause.trim().is_empty() {
        return Err(Hymt2RuntimeError::EmptySource);
    }
    Ok(())
}

struct Hymt2GreedyStepper<'a> {
    metadata: Hymt2ExecutionMetadata,
    token_embedding_table: &'a Qwen3AsrTokenEmbeddingTable,
    logits_head: &'a Qwen3AsrLlmLogitsHead,
    whole_decoder: &'a mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    layer_kv_caches: &'a mut [Qwen3AsrLayerKvCacheState],
    prompt_token_count: usize,
    decode_graph_initialized: bool,
}

impl Hymt2GreedyStepper<'_> {
    fn prefill_prompt_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let token_count = self.prompt_token_count;
        if token_count == 0 {
            return Err(Hymt2RuntimeError::EmptyPrompt);
        }
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Hymt2RuntimeError::Decode {
                reason: format!(
                    "Hy-MT2 layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        let Some(chunk_size) = hymt2_prefill_chunk_size_for(self.whole_decoder, token_count) else {
            return self.prefill_prompt_serial_and_compute_last_logits(token_major_embeddings);
        };
        self.prefill_prompt_chunked_and_compute_last_logits(token_major_embeddings, chunk_size)
    }

    fn prefill_tokens_at_offset_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
        position_offset: usize,
        token_count: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        if token_count == 0 {
            return Err(Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 offset prefill token count is zero".to_string(),
            });
        }
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Hymt2RuntimeError::Decode {
                reason: format!(
                    "Hy-MT2 layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        let Some(chunk_size) = self
            .whole_decoder
            .safe_host_cache_prefill_chunk_size_for(token_count)
        else {
            return self.prefill_tokens_at_offset_serial_and_compute_last_logits(
                token_major_embeddings,
                position_offset,
                token_count,
            );
        };
        self.prefill_tokens_at_offset_chunked_and_compute_last_logits(
            token_major_embeddings,
            position_offset,
            token_count,
            chunk_size,
        )
    }

    fn prefill_prompt_chunked_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
        chunk_size: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        if chunk_size == 0 {
            return Err(Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill chunk size is zero".to_string(),
            });
        }
        let token_count = self.prompt_token_count;
        if token_count <= chunk_size {
            if let Some(max_positions) = self.reusable_decode_max_positions() {
                let step = self
                    .whole_decoder
                    .run_prefill_into_reused_batched(
                        token_major_embeddings,
                        token_count,
                        1,
                        max_positions,
                        self.metadata.rope_freq_base,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })?;
                hymt2_profile_log_step("prefill_resident_full", None, token_count, &step);
                self.decode_graph_initialized = true;
                let final_hidden =
                    self.final_hidden_from_token_major_output(token_count, &step.hidden)?;
                return self.compute_logits_for_last_hidden(&final_hidden, "prefill_logits");
            }
            let step = self
                .whole_decoder
                .run_prefill(
                    token_major_embeddings,
                    token_count,
                    self.metadata.rope_freq_base,
                )
                .map_err(|source| Hymt2RuntimeError::Graph { source })?;
            hymt2_profile_log_step("prefill_full", None, token_count, &step);
            return self.write_prefill_step_outputs_and_compute_last_logits(token_count, step);
        }
        let hidden_size = self.metadata.d_model;
        let require_even_chunks = self.whole_decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden = None;
        while position_offset < token_count {
            let remaining = token_count - position_offset;
            let chunk_len = if require_even_chunks {
                even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = position_offset.checked_mul(hidden_size).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 prefill hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len =
                chunk_len
                    .checked_mul(hidden_size)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill hidden width overflowed".to_string(),
                    })?;
            let hidden_end =
                hidden_start
                    .checked_add(hidden_len)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill hidden end overflowed".to_string(),
                    })?;
            let total_token_count = position_offset.checked_add(chunk_len).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 prefill span overflowed".to_string(),
                }
            })?;
            let step = self
                .whole_decoder
                .run_prefill_chunk(
                    &token_major_embeddings[hidden_start..hidden_end],
                    chunk_len,
                    position_offset,
                    total_token_count,
                    &*self.layer_kv_caches,
                    self.metadata.rope_freq_base,
                )
                .map_err(|source| Hymt2RuntimeError::Graph { source })?;
            hymt2_profile_log_step("prefill_chunk", Some(position_offset), chunk_len, &step);
            final_hidden =
                Some(self.write_prefill_chunk_outputs(position_offset, chunk_len, step)?);
            position_offset = total_token_count;
        }
        self.compute_logits_for_last_hidden(
            &final_hidden.ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill produced no final hidden state".to_string(),
            })?,
            "prefill_logits",
        )
    }

    fn prefill_tokens_at_offset_chunked_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
        position_offset: usize,
        token_count: usize,
        chunk_size: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        if chunk_size == 0 {
            return Err(Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 offset prefill chunk size is zero".to_string(),
            });
        }
        let hidden_size = self.metadata.d_model;
        let require_even_chunks = self.whole_decoder.prefill_chunks_require_even_width();
        let mut relative_offset = 0usize;
        let mut final_hidden = None;
        while relative_offset < token_count {
            let remaining = token_count - relative_offset;
            let chunk_len = if require_even_chunks {
                even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = relative_offset.checked_mul(hidden_size).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 offset prefill hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len =
                chunk_len
                    .checked_mul(hidden_size)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 offset prefill hidden width overflowed".to_string(),
                    })?;
            let hidden_end =
                hidden_start
                    .checked_add(hidden_len)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 offset prefill hidden end overflowed".to_string(),
                    })?;
            let absolute_offset =
                position_offset
                    .checked_add(relative_offset)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 offset prefill absolute offset overflowed".to_string(),
                    })?;
            let total_token_count = absolute_offset.checked_add(chunk_len).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 offset prefill total token count overflowed".to_string(),
                }
            })?;
            let step = self
                .whole_decoder
                .run_prefill_chunk(
                    &token_major_embeddings[hidden_start..hidden_end],
                    chunk_len,
                    absolute_offset,
                    total_token_count,
                    &*self.layer_kv_caches,
                    self.metadata.rope_freq_base,
                )
                .map_err(|source| Hymt2RuntimeError::Graph { source })?;
            hymt2_profile_log_step(
                "prefill_offset_chunk",
                Some(absolute_offset),
                chunk_len,
                &step,
            );
            final_hidden =
                Some(self.write_prefill_chunk_outputs(absolute_offset, chunk_len, step)?);
            relative_offset = relative_offset.checked_add(chunk_len).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 offset prefill relative offset overflowed".to_string(),
                }
            })?;
        }
        self.compute_logits_for_last_hidden(
            &final_hidden.ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 offset prefill produced no final hidden state".to_string(),
            })?,
            "prefill_offset_logits",
        )
    }

    fn prefill_prompt_serial_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let mut final_hidden = None;
        for token_position in 0..self.prompt_token_count {
            let hidden = self.prefill_prompt_hidden_at(token_major_embeddings, token_position)?;
            let hidden = self.run_llm_layers_with_host_kv(hidden, token_position)?;
            final_hidden = Some(hidden);
        }
        self.compute_logits_for_last_hidden(
            &final_hidden.ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill produced no final hidden state".to_string(),
            })?,
            "prefill_logits",
        )
    }

    fn prefill_tokens_at_offset_serial_and_compute_last_logits(
        &mut self,
        token_major_embeddings: &[f32],
        position_offset: usize,
        token_count: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let mut final_hidden = None;
        for relative_position in 0..token_count {
            let absolute_position =
                position_offset
                    .checked_add(relative_position)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 offset serial prefill position overflowed".to_string(),
                    })?;
            let hidden =
                self.prefill_prompt_hidden_at(token_major_embeddings, relative_position)?;
            let hidden = self.run_llm_layers_with_host_kv(hidden, absolute_position)?;
            final_hidden = Some(hidden);
        }
        self.compute_logits_for_last_hidden(
            &final_hidden.ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 offset serial prefill produced no final hidden state".to_string(),
            })?,
            "prefill_offset_logits",
        )
    }

    fn write_prefill_step_outputs_and_compute_last_logits(
        &mut self,
        token_count: usize,
        step: Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let final_hidden = self.write_prefill_chunk_outputs(0, token_count, step)?;
        self.compute_logits_for_last_hidden(&final_hidden, "prefill_logits")
    }

    fn write_prefill_chunk_outputs(
        &mut self,
        position_offset: usize,
        token_count: usize,
        step: Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        if step.layer_kv.len() != self.layer_kv_caches.len() {
            return Err(Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill layer-KV count mismatch".to_string(),
            });
        }
        let kv_row_width = self
            .metadata
            .kv_heads
            .checked_mul(self.metadata.head_dim)
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill KV row width overflowed".to_string(),
            })?;
        for token_position in 0..token_count {
            let absolute_position =
                position_offset.checked_add(token_position).ok_or_else(|| {
                    Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill absolute row overflowed".to_string(),
                    }
                })?;
            let row_start = token_position.checked_mul(kv_row_width).ok_or_else(|| {
                Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 prefill KV row offset overflowed".to_string(),
                }
            })?;
            let row_end =
                row_start
                    .checked_add(kv_row_width)
                    .ok_or_else(|| Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill KV row end overflowed".to_string(),
                    })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                    Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill K row out of bounds".to_string(),
                    }
                })?;
                let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                    Hymt2RuntimeError::Decode {
                        reason: "Hy-MT2 prefill V row out of bounds".to_string(),
                    }
                })?;
                self.layer_kv_caches[layer_index]
                    .write(absolute_position, key_row, value_row)
                    .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
            }
        }
        self.final_hidden_from_token_major_output(token_count, &step.hidden)
    }

    fn final_hidden_from_token_major_output(
        &self,
        token_count: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let hidden_size = self.metadata.d_model;
        if hidden.len() == hidden_size {
            return Ok(hidden.to_vec());
        }
        let final_hidden_start = token_count
            .checked_sub(1)
            .and_then(|position| position.checked_mul(hidden_size))
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill final-hidden offset overflowed".to_string(),
            })?;
        let final_hidden_end = final_hidden_start.checked_add(hidden_size).ok_or_else(|| {
            Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill final-hidden end overflowed".to_string(),
            }
        })?;
        hidden
            .get(final_hidden_start..final_hidden_end)
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill final hidden out of bounds".to_string(),
            })
            .map(<[f32]>::to_vec)
    }

    fn prefill_prompt_hidden_at(
        &self,
        token_major_embeddings: &[f32],
        token_position: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let hidden_size = self.metadata.d_model;
        let start =
            token_position
                .checked_mul(hidden_size)
                .ok_or_else(|| Hymt2RuntimeError::Decode {
                    reason: "Hy-MT2 prefill hidden-state indexing overflowed".to_string(),
                })?;
        let end = start
            .checked_add(hidden_size)
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill hidden-state indexing overflowed".to_string(),
            })?;
        token_major_embeddings
            .get(start..end)
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 prefill hidden-state slice is out of bounds".to_string(),
            })
            .map(<[f32]>::to_vec)
    }

    fn decode_next_token_id(&mut self, generated_tokens: &[u32]) -> Result<u32, Hymt2RuntimeError> {
        let cache_position = self
            .prompt_token_count
            .checked_add(generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 decode cache position underflowed".to_string(),
            })?;
        let hidden = self.gather_last_generated_token_hidden(generated_tokens)?;
        self.run_llm_layers_with_kv_top1(hidden, cache_position)
    }

    fn compute_logits_for_last_hidden(
        &self,
        hidden: &[f32],
        stage: &'static str,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let started_at = hymt2_profile_start();
        let logits = self
            .logits_head
            .compute_logits_for_last_hidden(hidden)
            .map_err(|error| Hymt2RuntimeError::Decode {
                reason: error.to_string(),
            })?;
        hymt2_profile_log_opt(stage, started_at);
        Ok(logits)
    }

    fn compute_top1_token_for_last_hidden(
        &self,
        hidden: &[f32],
        stage: &'static str,
    ) -> Result<u32, Hymt2RuntimeError> {
        let started_at = hymt2_profile_start();
        let token_id = self
            .logits_head
            .compute_top1_token_for_last_hidden(hidden)
            .map_err(|error| Hymt2RuntimeError::Decode {
                reason: error.to_string(),
            })?;
        hymt2_profile_log_opt(stage, started_at);
        Ok(token_id)
    }

    fn run_llm_layers_with_kv_top1(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
    ) -> Result<u32, Hymt2RuntimeError> {
        let reuse_max_positions = self.reusable_decode_max_positions();
        if let Some(max_positions) = reuse_max_positions {
            return self.run_llm_layers_with_seeded_reuse_top1(
                hidden,
                cache_position,
                max_positions,
            );
        }

        self.run_llm_layers_with_host_kv_top1(hidden, cache_position)
    }

    fn reusable_decode_max_positions(&self) -> Option<usize> {
        self.layer_kv_caches
            .first()
            .map(|cache| cache.max_positions())
            .filter(|_| self.whole_decoder.supports_graph_reuse())
    }

    fn run_llm_layers_with_host_kv(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let started_at = hymt2_profile_start();
        let step = self
            .whole_decoder
            .run_step(
                &hidden,
                cache_position,
                &*self.layer_kv_caches,
                self.metadata.rope_freq_base,
            )
            .map_err(|source| Hymt2RuntimeError::Graph { source })?;
        hymt2_profile_log_step("decode_host_step", None, 1, &step);
        hymt2_profile_log_opt("decode_host_step_total", started_at);
        for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
            self.layer_kv_caches[layer_index]
                .write(cache_position, projected_k, projected_v)
                .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
        }
        Ok(step.hidden)
    }

    fn run_llm_layers_with_host_kv_top1(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
    ) -> Result<u32, Hymt2RuntimeError> {
        if !self.whole_decoder.supports_fused_top1() {
            let hidden = self.run_llm_layers_with_host_kv(hidden, cache_position)?;
            return self.compute_top1_token_for_last_hidden(&hidden, "decode_top1");
        }

        let started_at = hymt2_profile_start();
        let step = self
            .whole_decoder
            .run_step_top1(
                &hidden,
                cache_position,
                &*self.layer_kv_caches,
                self.metadata.rope_freq_base,
            )
            .map_err(|source| Hymt2RuntimeError::Graph { source })?;
        hymt2_profile_log_top1_step("decode_host_fused_top1_step", None, 1, &step);
        hymt2_profile_log_opt("decode_host_fused_top1_step_total", started_at);
        for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
            self.layer_kv_caches[layer_index]
                .write(cache_position, projected_k, projected_v)
                .map_err(|reason| Hymt2RuntimeError::Decode { reason })?;
        }
        Ok(step.token_id)
    }

    fn run_llm_layers_with_seeded_reuse(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
        max_positions: usize,
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let cache_positions = [cache_position];
        let step = if self.decode_graph_initialized {
            let started_at = hymt2_profile_start();
            self.whole_decoder
                .run_step_reused_batched(
                    &hidden,
                    &cache_positions,
                    self.metadata.rope_freq_base,
                    max_positions,
                )
                .map_err(|source| Hymt2RuntimeError::Graph { source })
                .inspect(|step| {
                    hymt2_profile_log_step("decode_reused_step", None, 1, step);
                    hymt2_profile_log_opt("decode_reused_step_total", started_at);
                })?
        } else {
            self.decode_graph_initialized = true;
            if self.whole_decoder.reused_graph_matches(1, max_positions) {
                let seed_started_at = hymt2_profile_start();
                self.whole_decoder
                    .seed_reused_batched_slot(
                        0,
                        cache_position,
                        &*self.layer_kv_caches,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })?;
                hymt2_profile_log_opt("decode_reuse_seed_slot", seed_started_at);
                let started_at = hymt2_profile_start();
                self.whole_decoder
                    .run_step_reused_batched(
                        &hidden,
                        &cache_positions,
                        self.metadata.rope_freq_base,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })
                    .inspect(|step| {
                        hymt2_profile_log_step("decode_reused_seeded_existing_step", None, 1, step);
                        hymt2_profile_log_opt(
                            "decode_reused_seeded_existing_step_total",
                            started_at,
                        );
                    })?
            } else {
                let seed_layers: [&[Qwen3AsrLayerKvCacheState]; 1] = [&*self.layer_kv_caches];
                let started_at = hymt2_profile_start();
                self.whole_decoder
                    .run_step_reused_batched_seeded(
                        &hidden,
                        &cache_positions,
                        &seed_layers,
                        self.metadata.rope_freq_base,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })
                    .inspect(|step| {
                        hymt2_profile_log_step("decode_reused_seeded_build_step", None, 1, step);
                        hymt2_profile_log_opt("decode_reused_seeded_build_step_total", started_at);
                    })?
            }
        };
        Ok(step.hidden)
    }

    fn run_llm_layers_with_seeded_reuse_top1(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
        max_positions: usize,
    ) -> Result<u32, Hymt2RuntimeError> {
        if !self.whole_decoder.supports_fused_top1() {
            let hidden =
                self.run_llm_layers_with_seeded_reuse(hidden, cache_position, max_positions)?;
            return self.compute_top1_token_for_last_hidden(&hidden, "decode_top1");
        }

        let cache_positions = [cache_position];
        let step = if self.decode_graph_initialized {
            let started_at = hymt2_profile_start();
            self.whole_decoder
                .run_step_reused_batched_top1(
                    &hidden,
                    &cache_positions,
                    self.metadata.rope_freq_base,
                    max_positions,
                )
                .map_err(|source| Hymt2RuntimeError::Graph { source })
                .inspect(|step| {
                    hymt2_profile_log_top1_step("decode_reused_fused_top1_step", None, 1, step);
                    hymt2_profile_log_opt("decode_reused_fused_top1_step_total", started_at);
                })?
        } else {
            self.decode_graph_initialized = true;
            if self.whole_decoder.reused_graph_matches(1, max_positions) {
                let seed_started_at = hymt2_profile_start();
                self.whole_decoder
                    .seed_reused_batched_slot(
                        0,
                        cache_position,
                        &*self.layer_kv_caches,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })?;
                hymt2_profile_log_opt("decode_reuse_seed_slot", seed_started_at);
                let started_at = hymt2_profile_start();
                self.whole_decoder
                    .run_step_reused_batched_top1(
                        &hidden,
                        &cache_positions,
                        self.metadata.rope_freq_base,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })
                    .inspect(|step| {
                        hymt2_profile_log_top1_step(
                            "decode_reused_seeded_existing_fused_top1_step",
                            None,
                            1,
                            step,
                        );
                        hymt2_profile_log_opt(
                            "decode_reused_seeded_existing_fused_top1_step_total",
                            started_at,
                        );
                    })?
            } else {
                let seed_layers: [&[Qwen3AsrLayerKvCacheState]; 1] = [&*self.layer_kv_caches];
                let started_at = hymt2_profile_start();
                self.whole_decoder
                    .run_step_reused_batched_seeded_top1(
                        &hidden,
                        &cache_positions,
                        &seed_layers,
                        self.metadata.rope_freq_base,
                        max_positions,
                    )
                    .map_err(|source| Hymt2RuntimeError::Graph { source })
                    .inspect(|step| {
                        hymt2_profile_log_top1_step(
                            "decode_reused_seeded_build_fused_top1_step",
                            None,
                            1,
                            step,
                        );
                        hymt2_profile_log_opt(
                            "decode_reused_seeded_build_fused_top1_step_total",
                            started_at,
                        );
                    })?
            }
        };
        Ok(step.token_id)
    }

    fn gather_last_generated_token_hidden(
        &self,
        generated_tokens: &[u32],
    ) -> Result<Vec<f32>, Hymt2RuntimeError> {
        let last_token = *generated_tokens
            .last()
            .ok_or_else(|| Hymt2RuntimeError::Decode {
                reason: "Hy-MT2 generated token history is unexpectedly empty".to_string(),
            })?;
        self.token_embedding_table
            .gather_rows(&[last_token])
            .map_err(|error| Hymt2RuntimeError::Decode {
                reason: error.to_string(),
            })
    }
}

fn greedy_argmax(logits: &[f32]) -> Result<u32, Hymt2RuntimeError> {
    if logits.is_empty() {
        return Err(Hymt2RuntimeError::Decode {
            reason: "Hy-MT2 decode step produced empty logits".to_string(),
        });
    }
    let mut best_index = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Hymt2RuntimeError::Decode {
                reason: format!("Hy-MT2 decode logits contain non-finite value at {index}"),
            });
        }
        if value > best {
            best = value;
            best_index = index;
        }
    }
    u32::try_from(best_index).map_err(|_| Hymt2RuntimeError::Decode {
        reason: format!("Hy-MT2 greedy token index {best_index} does not fit u32"),
    })
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return 0.0;
    }
    tokens as f64 / seconds
}

fn hymt2_prefill_chunk_size_for(
    decoder: &Qwen3AsrLlmWholeDecoderGraphExecutor,
    token_count: usize,
) -> Option<usize> {
    if let Some(chunk_size) = std::env::var(HYMT2_PREFILL_CHUNK_TOKENS_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&chunk_size| chunk_size > 0)
    {
        return Some(chunk_size.min(token_count));
    }
    if decoder.backend_is_metal() {
        return Some(HYMT2_METAL_PREFILL_QUERY_TOKENS.min(token_count));
    }
    decoder.safe_multi_query_prefill_chunk_size_for(token_count)
}

fn hymt2_profile_enabled() -> bool {
    std::env::var(HYMT2_PROFILE_ENV)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn hymt2_profile_start() -> Option<Instant> {
    hymt2_profile_enabled().then(Instant::now)
}

fn hymt2_profile_log_opt(stage: &str, started_at: Option<Instant>) {
    if let Some(started_at) = started_at {
        eprintln!(
            "openasr_hymt2_profile: stage={} total_us={}",
            stage,
            started_at.elapsed().as_micros()
        );
    }
}

fn hymt2_profile_log_step(
    stage: &str,
    position_offset: Option<usize>,
    token_count: usize,
    step: &Qwen3AsrLlmWholeStepOutput,
) {
    if hymt2_profile_enabled() {
        eprintln!(
            "openasr_hymt2_profile: stage={} position_offset={:?} token_count={} build_us={} compute_us={}",
            stage, position_offset, token_count, step.build_micros, step.compute_micros
        );
    }
}

fn hymt2_profile_log_top1_step(
    stage: &str,
    position_offset: Option<usize>,
    token_count: usize,
    step: &Qwen3AsrLlmWholeStepTop1Output,
) {
    if hymt2_profile_enabled() {
        eprintln!(
            "openasr_hymt2_profile: stage={} position_offset={:?} token_count={} token_id={} build_us={} compute_us={}",
            stage,
            position_offset,
            token_count,
            step.token_id,
            step.build_micros,
            step.compute_micros
        );
    }
}

fn plan_hymt2_prefix_reuse(
    cached_prefix_tokens: &[u32],
    next_prefix_tokens: &[u32],
    static_context_token_count: usize,
    unstable_tail_backoff_tokens: usize,
    finalized: bool,
) -> Hymt2PrefixReusePlan {
    let common_prefix = longest_common_token_prefix(cached_prefix_tokens, next_prefix_tokens);
    let reusable_floor = if common_prefix >= static_context_token_count {
        static_context_token_count
    } else {
        0
    };
    let backoff = if finalized || common_prefix <= reusable_floor {
        0
    } else {
        unstable_tail_backoff_tokens.min(common_prefix - reusable_floor)
    };
    Hymt2PrefixReusePlan {
        reused_prefix_tokens: common_prefix.saturating_sub(backoff),
        cache_backoff_tokens: backoff,
    }
}

fn longest_common_token_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::super::prompt::build_subtitle_translation_prompt;
    use super::super::tokenizer::{HYMT2_ASSISTANT_TOKEN, HYMT2_BOS_TOKEN, HYMT2_USER_TOKEN};
    use super::*;
    use sha2::{Digest, Sha256};

    // gguf_sha256 pins the published hymt2-1.8b-q4_k_m.oasr pack: the PR6
    // importer splices openasr.* KV metadata in front of the upstream KV
    // section while preserving the upstream GGUF tensor data byte-for-byte
    // (source GGUF sha256 dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699).
    const HYMT2_LLAMA_CPP_ORACLE: Hymt2LlamaCppOracle = Hymt2LlamaCppOracle {
        source_clause: "我们需要保持流式路径很快。",
        gguf_sha256: "56eae4c672e6d0fc1cacb210719ef494025c98c03753cd6b5a77b2fefa1557f6",
        llama_cpp_build_hash: "d2e22ed97",
    };

    #[derive(Clone, Copy)]
    struct Hymt2LlamaCppOracle {
        source_clause: &'static str,
        gguf_sha256: &'static str,
        llama_cpp_build_hash: &'static str,
    }

    impl Hymt2LlamaCppOracle {
        fn assert_runtime_pack(self, runtime_path: &std::path::Path) {
            let actual = file_sha256(runtime_path);
            assert_eq!(
                actual,
                self.gguf_sha256,
                "llama.cpp oracle mismatch\nruntime pack path: {}\nexpected gguf sha256: {}\nobserved gguf sha256: {actual}",
                runtime_path.display(),
                self.gguf_sha256,
            );
        }

        fn assert_llama_cpp_build(self, llama_completion: &std::path::Path) {
            let output = std::process::Command::new(llama_completion)
                .arg("--version")
                .output()
                .unwrap_or_else(|error| {
                    panic!(
                        "run llama-completion --version at {}: {error}",
                        llama_completion.display()
                    )
                });
            assert!(
                output.status.success(),
                "llama.cpp oracle mismatch\nllama-completion path: {}\nexpected build hash: {}\nversion command failed: status={} stdout={} stderr={}",
                llama_completion.display(),
                self.llama_cpp_build_hash,
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            let version_output = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let expected_fragment = format!("({})", self.llama_cpp_build_hash);
            assert!(
                version_output.contains(&expected_fragment),
                "llama.cpp oracle mismatch\nllama-completion path: {}\nexpected build hash: {}\nobserved version output:\n{}",
                llama_completion.display(),
                self.llama_cpp_build_hash,
                version_output,
            );
        }
    }

    #[test]
    fn greedy_argmax_selects_first_max() {
        assert_eq!(greedy_argmax(&[0.1, 0.3, 0.3]).expect("argmax"), 1);
    }

    #[test]
    fn timing_helpers_return_zero_for_zero_duration() {
        let timings = Hymt2DecodeTimings {
            prefill: Duration::ZERO,
            decode: Duration::from_millis(10),
            total: Duration::from_millis(10),
            prompt_tokens: 10,
            prefilled_tokens: 10,
            reused_prefix_tokens: 0,
            cache_backoff_tokens: 0,
            generated_tokens: 2,
        };
        assert_eq!(timings.prefill_tokens_per_second(), 0.0);
        assert!(timings.decode_tokens_per_second() > 0.0);
    }

    #[test]
    fn translate_clause_rejects_empty_source_before_prompt_decode() {
        assert!(matches!(
            validate_non_empty_source_clause("   \n\t").expect_err("empty source"),
            Hymt2RuntimeError::EmptySource
        ));
        validate_non_empty_source_clause("字幕").expect("non-empty source");
    }

    #[test]
    fn prefix_reuse_plan_backs_off_unstable_tail() {
        let cached = [1, 2, 3, 4, 5, 6, 7];
        let growing = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let plan = plan_hymt2_prefix_reuse(&cached, &growing, 4, 2, false);

        assert_eq!(
            plan,
            Hymt2PrefixReusePlan {
                reused_prefix_tokens: 5,
                cache_backoff_tokens: 2,
            }
        );
        let cached_prefill = growing.len() - plan.reused_prefix_tokens + 1;
        let uncached_prefill = growing.len() + 1;
        assert!(cached_prefill < uncached_prefill);
    }

    #[test]
    fn prefix_reuse_plan_keeps_final_clause_tail() {
        let cached = [1, 2, 3, 4, 5, 6, 7];
        let growing = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let plan = plan_hymt2_prefix_reuse(&cached, &growing, 4, 2, true);

        assert_eq!(
            plan,
            Hymt2PrefixReusePlan {
                reused_prefix_tokens: 7,
                cache_backoff_tokens: 0,
            }
        );
    }

    #[test]
    fn prefix_cache_prefill_failure_invalidates_active_cache() {
        let mut seeded_layer = Qwen3AsrLayerKvCacheState::new(8, 1, 1);
        for position in 0..4 {
            let value = position as f32 + 1.0;
            seeded_layer
                .write(position, &[value], &[value + 10.0])
                .expect("seed row");
        }
        let mut cache = Hymt2TranslationSessionCache::new(Hymt2PrefixCacheConfig {
            unstable_tail_backoff_tokens: 0,
        });
        cache.active = Some(Hymt2ActivePrefixCache {
            clause_id: "c-cache".to_string(),
            static_context_tokens: vec![1, 2],
            source_prefix_tokens: vec![1, 2, 3, 4],
            layer_kv_caches: vec![seeded_layer],
            max_positions: 8,
        });

        let injected = update_hymt2_source_prefix_cache(
            &mut cache,
            "c-cache",
            &[1, 2, 3, 4, 5],
            2,
            true,
            8,
            single_layer_caches,
            |mut staged, suffix_tokens, reused_prefix_tokens| {
                assert_eq!(reused_prefix_tokens, 4);
                assert_eq!(suffix_tokens, &[5]);
                staged[0]
                    .write(4, &[99.0], &[199.0])
                    .expect("inject contaminated staged row");
                Err(Hymt2RuntimeError::Decode {
                    reason: "injected prefill failure".to_string(),
                })
            },
        );

        assert!(injected.is_err());
        assert_eq!(cache.active_prefix_token_count(), 0);

        let (prefilled_tokens, reuse_plan) = update_hymt2_source_prefix_cache(
            &mut cache,
            "c-cache",
            &[1, 2, 3, 4, 5],
            2,
            true,
            8,
            single_layer_caches,
            |mut staged, suffix_tokens, reused_prefix_tokens| {
                assert_eq!(reused_prefix_tokens, 0);
                assert_eq!(suffix_tokens, &[1, 2, 3, 4, 5]);
                for (position, token) in suffix_tokens.iter().copied().enumerate() {
                    let value = token as f32;
                    staged[0]
                        .write(position, &[value], &[value + 10.0])
                        .expect("write rebuilt row");
                }
                Ok(staged)
            },
        )
        .expect("rebuild after injected failure");

        assert_eq!(prefilled_tokens, 5);
        assert_eq!(
            reuse_plan,
            Hymt2PrefixReusePlan {
                reused_prefix_tokens: 0,
                cache_backoff_tokens: 0,
            }
        );
        let active = cache.active.as_ref().expect("rebuilt active cache");
        assert_eq!(active.source_prefix_tokens, [1, 2, 3, 4, 5]);
        assert_eq!(active.layer_kv_caches[0].written_positions(), 5);
    }

    #[test]
    #[ignore = "manual real-pack harness: set OPENASR_HYMT2_REAL_PACK to Hy-MT2-1.8B-Q4_K_M.gguf or local .oasr"]
    fn hymt2_real_pack_decodes_and_reports_perf() {
        let source_clause = hymt2_real_pack_perf_source_clause("我们需要保持流式路径很快。");
        let report = run_hymt2_real_pack_decode(&source_clause);
        eprintln!(
            "hymt2 real-pack decode text={:?} prompt_tokens={} generated_tokens={} prefill_tps={:.2} decode_tps={:.2} total_ms={:.2}",
            report.result.text,
            report.result.prompt_tokens.len(),
            report.result.generated_tokens.len(),
            report.result.timings.prefill_tokens_per_second(),
            report.result.timings.decode_tokens_per_second(),
            report.result.timings.total.as_secs_f64() * 1000.0
        );
        assert!(
            report
                .result
                .first_step_logits
                .iter()
                .all(|value| value.is_finite()),
            "Hy-MT2 first-step logits must be finite"
        );
    }

    fn single_layer_caches(max_positions: usize) -> Vec<Qwen3AsrLayerKvCacheState> {
        vec![Qwen3AsrLayerKvCacheState::new(max_positions, 1, 1)]
    }

    #[test]
    #[ignore = "manual real-pack hot-session harness: set OPENASR_HYMT2_REAL_PACK to Hy-MT2-1.8B-Q4_K_M.gguf or local .oasr"]
    fn hymt2_real_pack_hot_session_reports_perf() {
        let runtime_path = hymt2_real_pack_path();
        let load_started = std::time::Instant::now();
        let runtime = Hymt2Runtime::from_path(&runtime_path).expect("load Hy-MT2 runtime");
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("hymt2 cold model load_ms={load_ms:.2}");
        let source_clause = hymt2_real_pack_perf_source_clause("我们需要保持流式路径很快。");
        let cold = run_hymt2_real_pack_decode_with_runtime(&runtime_path, &runtime, &source_clause);
        let hot = run_hymt2_real_pack_decode_with_runtime(&runtime_path, &runtime, &source_clause);
        eprintln!(
            "hymt2 hot-session decode cold_prefill_tps={:.2} cold_decode_tps={:.2} cold_total_ms={:.2} hot_prefill_tps={:.2} hot_decode_tps={:.2} hot_total_ms={:.2} hot_text={:?}",
            cold.result.timings.prefill_tokens_per_second(),
            cold.result.timings.decode_tokens_per_second(),
            cold.result.timings.total.as_secs_f64() * 1000.0,
            hot.result.timings.prefill_tokens_per_second(),
            hot.result.timings.decode_tokens_per_second(),
            hot.result.timings.total.as_secs_f64() * 1000.0,
            hot.result.text,
        );
        assert_eq!(cold.result.generated_tokens, hot.result.generated_tokens);
    }

    /// Line-by-line subtitle translation eval mirroring the realtime worker
    /// contract: each line is one finalized clause translated through
    /// `translate_clause_with_cache` with the rolling last-2 finalized-context
    /// window. Prints `SRC\tTGT` pairs for offline quality comparison.
    #[test]
    #[ignore = "manual eval harness: set OPENASR_HYMT2_REAL_PACK and OPENASR_HYMT2_EVAL_LINES (UTF-8 file, one source clause per line)"]
    fn hymt2_real_pack_translates_eval_lines_with_rolling_context() {
        let runtime_path = hymt2_real_pack_path();
        let lines_path = std::env::var("OPENASR_HYMT2_EVAL_LINES")
            .expect("set OPENASR_HYMT2_EVAL_LINES to a UTF-8 file with one source clause per line");
        let raw = std::fs::read_to_string(&lines_path).expect("read eval lines file");
        let runtime = Hymt2Runtime::from_path(&runtime_path).expect("load Hy-MT2 runtime");
        let mut cache = Hymt2TranslationSessionCache::default();
        let mut context: Vec<(String, String)> = Vec::new();
        let mut total_ms = 0.0_f64;
        let mut clause_count = 0_usize;
        for (index, line) in raw.lines().enumerate() {
            let source = line.trim();
            if source.is_empty() {
                continue;
            }
            let context_refs = context
                .iter()
                .rev()
                .take(2)
                .rev()
                .map(|(source, target)| (source.as_str(), target.as_str()))
                .collect::<Vec<_>>();
            let clause_id = format!("eval-{index}");
            let result = runtime
                .translate_clause_with_cache(&mut cache, &clause_id, source, &context_refs, true)
                .expect("translate eval clause");
            total_ms += result.timings.total.as_secs_f64() * 1000.0;
            clause_count += 1;
            let text =
                crate::translation::align_translation_terminal_punctuation(source, &result.text)
                    .unwrap_or_else(|| result.text.clone());
            println!("{source}\t{text}");
            context.push((source.to_string(), text));
        }
        eprintln!(
            "hymt2 eval lines={clause_count} total_ms={total_ms:.0} avg_ms={:.0}",
            if clause_count > 0 {
                total_ms / clause_count as f64
            } else {
                0.0
            }
        );
    }

    #[test]
    #[ignore = "manual llama.cpp parity harness: set OPENASR_HYMT2_REAL_PACK, OPENASR_HYMT2_LLAMA_COMPLETION, and OPENASR_HYMT2_LLAMA_TOKENIZE; oracle pins pack sha256 and llama.cpp build hash"]
    fn hymt2_real_pack_matches_llama_cpp_greedy_tokens() {
        let oracle = HYMT2_LLAMA_CPP_ORACLE;
        let report = run_hymt2_real_pack_decode(oracle.source_clause);
        oracle.assert_runtime_pack(&report.runtime_path);
        let llama_completion = hymt2_llama_completion_path();
        oracle.assert_llama_cpp_build(&llama_completion);
        let llama_output = run_llama_completion_greedy(&report, &llama_completion);
        let llama_tokens = tokenize_llama_output_text(&report.runtime_path, &llama_output.text);
        eprintln!(
            "hymt2 llama parity gguf_sha256={} llama_cpp_build_hash={} rust_text={:?} llama_text={:?} rust_tokens={:?} llama_tokens={:?} llama_prompt_tps={:?} llama_decode_tps={:?}",
            oracle.gguf_sha256,
            oracle.llama_cpp_build_hash,
            report.result.text,
            llama_output.text,
            report.result.generated_tokens,
            llama_tokens,
            llama_output.prompt_tps,
            llama_output.decode_tps
        );
        if report.result.generated_tokens != llama_tokens {
            maybe_report_first_step_logits_cosine();
        }
        assert_eq!(
            report.result.generated_tokens, llama_tokens,
            "Hy-MT2 greedy generated token IDs must match llama.cpp for the same prompt"
        );
    }

    #[test]
    #[ignore = "manual real-pack replay: set OPENASR_HYMT2_REAL_PACK or keep a local tmp/hymt2-local copy"]
    fn hymt2_prefix_cache_replay_reuses_prefill_and_matches_uncached_outputs() {
        let runtime_path = hymt2_real_pack_path();
        let runtime = Hymt2Runtime::from_path(&runtime_path).expect("load Hy-MT2 runtime");
        let mut cache = Hymt2TranslationSessionCache::default();
        let sequence = [
            "我们需要保持",
            "我们需要保持流式路径",
            "我们需要保持流式路径很快",
            "我们需要保持流式路径很快并且",
            "我们需要保持流式路径很快并且翻译稳定。",
        ];
        let mut uncached_prefill = 0usize;
        let mut cached_prefill = 0usize;
        let mut cached_total_ms = 0.0f64;
        let mut observed_backoff = false;
        for source in sequence {
            let uncached = runtime
                .translate_clause(source, &[])
                .expect("uncached translation");
            let cached = runtime
                .translate_clause_with_cache(&mut cache, "c-replay", source, &[], false)
                .expect("cached translation");
            eprintln!(
                "hymt2 prefix replay source={source:?} text={:?} uncached_prefill_tokens={} cached_prefill_tokens={} reused_prefix_tokens={} backoff_tokens={} total_ms={:.2}",
                cached.text,
                uncached.timings.prefilled_tokens,
                cached.timings.prefilled_tokens,
                cached.timings.reused_prefix_tokens,
                cached.timings.cache_backoff_tokens,
                cached.timings.total.as_secs_f64() * 1000.0
            );
            assert_eq!(
                cached.generated_tokens, uncached.generated_tokens,
                "cached decode must preserve greedy output tokens"
            );
            assert_eq!(cached.text, uncached.text);
            observed_backoff |= cached.timings.cache_backoff_tokens > 0;
            uncached_prefill += uncached.timings.prefilled_tokens;
            cached_prefill += cached.timings.prefilled_tokens;
            cached_total_ms += cached.timings.total.as_secs_f64() * 1000.0;
        }
        assert!(
            observed_backoff,
            "provisional replay should exercise unstable-tail backoff"
        );
        let changed_context = [("上一句已经完成。", "The previous sentence is done.")];
        let source = "我们需要保持流式路径很快并且翻译稳定。";
        let uncached_context = runtime
            .translate_clause(source, &changed_context)
            .expect("uncached context-change translation");
        let cached_context = runtime
            .translate_clause_with_cache(&mut cache, "c-replay", source, &changed_context, false)
            .expect("cached context-change translation");
        assert_eq!(
            cached_context.generated_tokens, uncached_context.generated_tokens,
            "context-change replay must preserve greedy output tokens"
        );
        assert_eq!(cached_context.text, uncached_context.text);
        assert_eq!(
            cached_context.timings.reused_prefix_tokens, 0,
            "static context changes must invalidate source-prefix reuse"
        );
        eprintln!(
            "hymt2 prefix replay summary uncached_prefill_tokens={uncached_prefill} cached_prefill_tokens={cached_prefill} cached_total_ms={cached_total_ms:.2}"
        );
        assert!(
            cached_prefill < uncached_prefill,
            "cached replay should prefill fewer tokens"
        );
    }

    #[test]
    #[ignore = "manual real-pack latency harness: set OPENASR_HYMT2_REAL_PACK; prints hot per-clause translate p50/p90"]
    fn hymt2_real_pack_hot_clause_latency_distribution() {
        let runtime_path = hymt2_real_pack_path();
        let runtime = Hymt2Runtime::from_path(&runtime_path).expect("load Hy-MT2 runtime");
        let mut cache = Hymt2TranslationSessionCache::default();
        // Realistic clause-retranslation traffic: each clause grows in steps
        // (provisional retranslations) and ends with its stable form, capped
        // around the 24-char clause segmentation budget from the design memo.
        let clause_steps: &[(&str, &[&str])] = &[
            (
                "c1",
                &["大家好", "大家好，欢迎来到", "大家好，欢迎来到今天的会议。"],
            ),
            (
                "c2",
                &[
                    "今天我们讨论",
                    "今天我们讨论实时翻译",
                    "今天我们讨论实时翻译的发布计划。",
                ],
            ),
            ("c3", &["首先看延迟", "首先看延迟指标。"]),
            ("c4", &["模型已经上线", "模型已经上线了。"]),
            (
                "c5",
                &[
                    "请大家注意",
                    "请大家注意内存占用",
                    "请大家注意内存占用和稳定性。",
                ],
            ),
            ("c6", &["谢谢大家。"]),
        ];
        // Warm the session so the distribution reflects hot operation.
        runtime
            .translate_clause_with_cache(&mut cache, "warmup", "预热。", &[], false)
            .expect("warmup translation");
        let mut latencies_ms = Vec::new();
        for (clause_id, steps) in clause_steps {
            for (index, source) in steps.iter().enumerate() {
                let is_final = index + 1 == steps.len();
                let result = runtime
                    .translate_clause_with_cache(&mut cache, clause_id, source, &[], is_final)
                    .expect("hot clause translation");
                let total_ms = result.timings.total.as_secs_f64() * 1000.0;
                latencies_ms.push(total_ms);
                eprintln!(
                    "hymt2 hot clause clause_id={clause_id} chars={} generated_tokens={} reused_prefix_tokens={} total_ms={total_ms:.2} text={:?}",
                    source.chars().count(),
                    result.generated_tokens.len(),
                    result.timings.reused_prefix_tokens,
                    result.text,
                );
            }
        }
        let mut sorted = latencies_ms.clone();
        sorted.sort_by(|left, right| left.total_cmp(right));
        let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        eprintln!(
            "hymt2 hot clause latency summary n={} p50_ms={:.2} p90_ms={:.2} max_ms={:.2}",
            sorted.len(),
            pick(0.5),
            pick(0.9),
            sorted[sorted.len() - 1]
        );
        assert!(
            latencies_ms.iter().all(|value| value.is_finite()),
            "latencies must be finite"
        );
    }

    struct Hymt2RealPackDecodeReport {
        runtime_path: std::path::PathBuf,
        prompt_text: String,
        max_output_tokens: usize,
        result: Hymt2DecodeResult,
    }

    struct LlamaCliOutput {
        text: String,
        prompt_tps: Option<f64>,
        decode_tps: Option<f64>,
    }

    fn run_hymt2_real_pack_decode(source_clause: &str) -> Hymt2RealPackDecodeReport {
        let runtime_path = hymt2_real_pack_path();
        let runtime = Hymt2Runtime::from_path(&runtime_path).expect("load Hy-MT2 runtime");
        run_hymt2_real_pack_decode_with_runtime(&runtime_path, &runtime, source_clause)
    }

    fn hymt2_real_pack_perf_source_clause(default: &str) -> String {
        std::env::var("OPENASR_HYMT2_SOURCE_CLAUSE").unwrap_or_else(|_| default.to_string())
    }

    fn run_hymt2_real_pack_decode_with_runtime(
        runtime_path: &std::path::Path,
        runtime: &Hymt2Runtime,
        source_clause: &str,
    ) -> Hymt2RealPackDecodeReport {
        let prompt_text = build_subtitle_translation_prompt(source_clause, &[]);
        let prompt_tokens = runtime
            .tokenizer()
            .encode_user_chat_prompt(&prompt_text)
            .expect("Hy-MT2 prompt tokens");
        let source_tokens = runtime
            .tokenizer()
            .encode_content_text(source_clause)
            .expect("Hy-MT2 source tokens");
        let max_output_tokens =
            crate::models::hymt2::prompt::max_output_tokens_for_source_tokens(source_tokens.len());
        let result = runtime
            .decode_prompt_tokens(prompt_tokens, max_output_tokens)
            .expect("Hy-MT2 greedy decode");
        Hymt2RealPackDecodeReport {
            runtime_path: runtime_path.to_path_buf(),
            prompt_text,
            max_output_tokens,
            result,
        }
    }

    fn run_llama_completion_greedy(
        report: &Hymt2RealPackDecodeReport,
        llama_completion: &std::path::Path,
    ) -> LlamaCliOutput {
        let prompt = hymt2_chat_prompt_text(&report.prompt_text);
        let output = std::process::Command::new(llama_completion)
            .args([
                "--no-display-prompt",
                "--no-conversation",
                "--ctx-size",
                "4096",
                "--gpu-layers",
                "99",
                "--predict",
                &report.max_output_tokens.to_string(),
                "--temp",
                "0",
                "--repeat-penalty",
                "1",
                "--repeat-last-n",
                "0",
                "--model",
            ])
            .arg(&report.runtime_path)
            .arg("--prompt")
            .arg(prompt)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "run llama-completion at {}: {error}",
                    llama_completion.display()
                )
            });
        assert!(
            output.status.success(),
            "llama-completion failed: status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        LlamaCliOutput {
            text: strip_llama_end_marker(stdout.trim()).to_string(),
            prompt_tps: parse_llama_perf_tps(&stderr, "prompt eval time"),
            decode_tps: parse_llama_perf_tps(&stderr, "eval time"),
        }
    }

    fn hymt2_llama_completion_path() -> std::path::PathBuf {
        if let Some(path) = std::env::var_os("OPENASR_HYMT2_LLAMA_COMPLETION") {
            return std::path::PathBuf::from(path);
        }
        if let Some(path) = std::env::var_os("OPENASR_HYMT2_LLAMA_CLI") {
            let mut path = std::path::PathBuf::from(path);
            path.set_file_name("llama-completion");
            return path;
        }
        panic!(
            "OPENASR_HYMT2_LLAMA_COMPLETION must point to llama-completion (or set OPENASR_HYMT2_LLAMA_CLI beside it)"
        );
    }

    fn tokenize_llama_output_text(model_path: &std::path::Path, text: &str) -> Vec<u32> {
        let llama_tokenize = std::env::var_os("OPENASR_HYMT2_LLAMA_TOKENIZE")
            .map(std::path::PathBuf::from)
            .expect("OPENASR_HYMT2_LLAMA_TOKENIZE must point to llama-tokenize");
        let output = std::process::Command::new(&llama_tokenize)
            .args(["--log-disable", "--ids", "--no-bos", "--model"])
            .arg(model_path)
            .arg("--prompt")
            .arg(text)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "run llama-tokenize at {}: {error}",
                    llama_tokenize.display()
                )
            });
        assert!(
            output.status.success(),
            "llama-tokenize failed: status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let value: serde_json::Value =
            serde_json::from_str(stdout.trim()).expect("llama-tokenize JSON token id array");
        value
            .as_array()
            .expect("llama-tokenize token id array")
            .iter()
            .map(|value| {
                let token_id = value.as_u64().expect("llama token id");
                u32::try_from(token_id).expect("llama token id fits u32")
            })
            .collect()
    }

    fn hymt2_chat_prompt_text(user_content: &str) -> String {
        format!("{HYMT2_BOS_TOKEN}{HYMT2_USER_TOKEN}{user_content}{HYMT2_ASSISTANT_TOKEN}")
    }

    fn strip_llama_end_marker(text: &str) -> &str {
        text.strip_suffix("[end of text]")
            .map(str::trim_end)
            .unwrap_or(text)
    }

    fn parse_llama_perf_tps(logs: &str, label: &str) -> Option<f64> {
        logs.lines()
            .find(|line| {
                line.contains(label)
                    && line.contains("tokens per second")
                    && (label != "eval time" || !line.contains("prompt eval time"))
            })
            .and_then(|line| line.split("tokens per second").next())
            .and_then(|head| head.split_whitespace().last())
            .and_then(|number| number.parse::<f64>().ok())
    }

    fn maybe_report_first_step_logits_cosine() {
        let Some(path) = std::env::var_os("OPENASR_HYMT2_LLAMA_FIRST_LOGITS_JSON") else {
            eprintln!(
                "OPENASR_HYMT2_LLAMA_FIRST_LOGITS_JSON not set; token divergence has no llama.cpp first-step logits cosine oracle"
            );
            return;
        };
        let bytes = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "read OPENASR_HYMT2_LLAMA_FIRST_LOGITS_JSON at {}: {error}",
                std::path::PathBuf::from(path).display()
            )
        });
        let values: Vec<f32> =
            serde_json::from_slice(&bytes).expect("llama first-step logits JSON array");
        eprintln!(
            "llama.cpp first-step logits oracle provided with {} values; compare externally against Hymt2DecodeResult::first_step_logits",
            values.len()
        );
    }

    fn hymt2_real_pack_path() -> std::path::PathBuf {
        if let Some(path) = std::env::var_os("OPENASR_HYMT2_REAL_PACK") {
            return std::path::PathBuf::from(path);
        }
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(std::path::Path::parent)
            .expect("openasr-core lives under crates/openasr-core");
        let candidate = repo_root.join("tmp/hymt2-local/Hy-MT2-1.8B-Q4_K_M.oasr");
        if candidate.exists() {
            return candidate;
        }
        panic!(
            "OPENASR_HYMT2_REAL_PACK must point to Hy-MT2-1.8B-Q4_K_M.gguf or a local .oasr hard-link/copy"
        );
    }

    fn file_sha256(path: &std::path::Path) -> String {
        let mut file = std::fs::File::open(path)
            .unwrap_or_else(|error| panic!("open {} for sha256: {error}", path.display()));
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let bytes_read = std::io::Read::read(&mut file, &mut buffer)
                .unwrap_or_else(|error| panic!("read {} for sha256: {error}", path.display()));
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        format!("{:x}", hasher.finalize())
    }
}
