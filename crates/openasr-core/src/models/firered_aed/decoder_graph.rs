//! firered-aed Transformer decoder ggml graph (Stage 3).
//!
//! Faithfully reproduces `fireredasr/models/module/transformer_decoder.py`: a
//! standard ESPnet/WeNet-lineage pre-norm Transformer decoder --
//! `decoder.tgt_word_emb` scaled by `sqrt(d_model)` plus the baked absolute
//! sinusoidal `decoder.positional_encoding.pe` -> 16 x DecoderLayer -> final
//! affine LayerNorm (`decoder.layer_norm_out`) -> untied `decoder.tgt_word_prj`
//! (bias-free). Each DecoderLayer is pre-norm: `norm -> causal self-attn ->
//! residual`, `norm -> cross-attn on the encoder output -> residual`, `norm ->
//! GELU FFN -> residual`. `self_attn.w_ks` and `cross_attn.w_ks` are upstream
//! bias-free linears (see [`super::decoder_weights`]); this graph supplies one
//! shared zero bias tensor for both.
//!
//! Built on the shared incremental seq2seq decoder block
//! ([`crate::nn::decoder::seq2seq_layer`]): pre-norm causal self-attention
//! with an f16 KV cache, pre-norm cross-attention over cross-KV precomputed
//! once from the encoder output, and a GELU feed-forward. Runs CPU-only,
//! rebuilding a fresh graph every decode step (matches the Stage 2 encoder's
//! CPU-only staging -- GPU reuse can follow once parity is established).

#![allow(dead_code)]

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::nn::decoder::{
    CrossKvHandle, SelfKvHandle, Seq2SeqLayerConfig, Seq2SeqLayerWeights,
    build_causal_mask_f16_bits, seq2seq_layer,
};
use crate::nn::ffn::FeedForwardActivation;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::decoder_weights::{FireRedDecoderWeights, FireRedDecoderWeightsError};
use super::runtime_contract::FireRedAedExecutionMetadata;

const FIRERED_DECODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const FIRERED_DECODER_GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;
const FIRERED_DECODER_CACHE_ARENA_BYTES: usize = 256 * 1024 * 1024;
const FIRERED_DECODER_GRAPH_SIZE: usize = 8192;

#[derive(Debug, Error)]
pub(crate) enum FireRedDecoderError {
    #[error("firered-aed decoder weights: {0}")]
    Weights(#[from] FireRedDecoderWeightsError),
    #[error("firered-aed decoder input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[error("firered-aed decoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("firered-aed decoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("firered-aed decoder shape overflowed")]
    ShapeOverflow,
}

fn map_err(step: &'static str, source: GgmlCpuGraphError) -> FireRedDecoderError {
    FireRedDecoderError::GraphBuildFailed { step, source }
}

pub(crate) fn firered_decoder_graph_config() -> GgmlCpuGraphConfig {
    GgmlCpuGraphConfig {
        context_bytes: FIRERED_DECODER_GRAPH_CONTEXT_BYTES,
        graph_size: FIRERED_DECODER_GRAPH_SIZE,
        n_threads: None,
        backend: GgmlCpuGraphBackend::Cpu,
        use_scheduler: false,
    }
}

#[derive(Clone, Copy)]
struct FireRedDecoderCrossCacheLayer {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
}

#[derive(Clone, Copy)]
struct FireRedDecoderSelfKvLayer {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
}

/// Owns the decoder's mmap'd weight context plus the persistent runtime state
/// (cross-KV cache, incremental self-KV cache) for one transcription. Each
/// transcription gets a fresh runtime sized to that request's encoder frame
/// count -- the cross-KV cache arena is not resizable.
pub(crate) struct FireRedDecoderGraphRuntime {
    runner: GgmlCpuGraphRunner,
    _loaded: GgmlLoadedWeightContext,
    weights: FireRedDecoderWeights,
    metadata: FireRedAedExecutionMetadata,
    arena: GgmlStaticTensorArena,
    /// Shared zero bias for the two bias-free K projections (self-attn and
    /// cross-attn `w_ks`), length `d_model`.
    zero_bias: GgmlStaticTensor,
    cross_layers: Vec<FireRedDecoderCrossCacheLayer>,
    self_kv_layers: Vec<FireRedDecoderSelfKvLayer>,
    cross_frame_count: usize,
    cached_positions: usize,
}

impl FireRedDecoderGraphRuntime {
    pub(crate) fn new(
        runtime_path: &Path,
        metadata: FireRedAedExecutionMetadata,
        cross_frame_count: usize,
    ) -> Result<Self, FireRedDecoderError> {
        if cross_frame_count == 0 {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "cross_frame_count must be > 0".to_string(),
            });
        }
        let runner = GgmlCpuGraphRunner::new(firered_decoder_graph_config())
            .map_err(|source| map_err("runner_init", source))?;
        let loaded = runner
            .load_gguf_weight_context(runtime_path)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let weights = FireRedDecoderWeights::load(&loaded, metadata.decoder_n_layers)?;

        let arena = runner
            .start_static_tensor_arena(FIRERED_DECODER_CACHE_ARENA_BYTES)
            .map_err(|source| map_err("static_tensor_arena", source))?;
        let zero_bias = arena
            .new_tensor_1d_f32(metadata.d_model, "firered_dec_zero_bias")
            .map_err(|source| map_err("zero_bias_alloc", source))?;
        let mut cross_layers = Vec::with_capacity(metadata.decoder_n_layers);
        let mut self_kv_layers = Vec::with_capacity(metadata.decoder_n_layers);
        for _ in 0..metadata.decoder_n_layers {
            cross_layers.push(FireRedDecoderCrossCacheLayer {
                key: arena
                    .new_tensor_2d_f32(metadata.d_model, cross_frame_count, "firered_dec_cross_k")
                    .map_err(|source| map_err("cross_k_alloc", source))?,
                value: arena
                    .new_tensor_2d_f32(metadata.d_model, cross_frame_count, "firered_dec_cross_v")
                    .map_err(|source| map_err("cross_v_alloc", source))?,
            });
            self_kv_layers.push(FireRedDecoderSelfKvLayer {
                key: arena
                    .new_tensor_3d_f16(
                        metadata.head_dim,
                        metadata.decoder_pe_len,
                        metadata.n_heads,
                        "firered_dec_self_k",
                    )
                    .map_err(|source| map_err("self_k_alloc", source))?,
                value: arena
                    .new_tensor_3d_f16(
                        metadata.head_dim,
                        metadata.decoder_pe_len,
                        metadata.n_heads,
                        "firered_dec_self_v",
                    )
                    .map_err(|source| map_err("self_v_alloc", source))?,
            });
        }
        let mut arena = arena;
        arena
            .set_f32_slice(
                zero_bias,
                &vec![0.0f32; metadata.d_model],
                "firered_dec_zero_bias",
            )
            .map_err(|source| map_err("zero_bias_upload", source))?;

        Ok(Self {
            runner,
            _loaded: loaded,
            weights,
            metadata,
            arena,
            zero_bias,
            cross_layers,
            self_kv_layers,
            cross_frame_count,
            cached_positions: 0,
        })
    }

    /// Precompute cross-attention K/V for every layer from the encoder output
    /// and write them into the persistent cross-KV cache. Must be called once
    /// before the first [`Self::compute_step_logits`].
    pub(crate) fn populate_cross_attention_cache(
        &mut self,
        encoder_rows: &[f32],
    ) -> Result<(), FireRedDecoderError> {
        let d_model = self.metadata.d_model;
        let frame_count = self.cross_frame_count;
        let expected = frame_count
            .checked_mul(d_model)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        if encoder_rows.len() != expected {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "encoder rows length mismatch: got {}, expected {expected}",
                    encoder_rows.len()
                ),
            });
        }
        self.cached_positions = 0;

        let mut graph = self.runner.start_graph();
        let encoder_tensor = graph
            .new_tensor_2d_f32(d_model, frame_count, "firered_dec_encoder_rows")
            .map_err(|source| map_err("encoder_rows_alloc", source))?;
        graph
            .set_input(encoder_tensor)
            .map_err(|source| map_err("encoder_rows_input", source))?;

        let zero_bias_tensor = self.arena.graph_tensor(self.zero_bias);
        let mut last_value_rows = None;
        for (layer, cross) in self.weights.layers.iter().zip(&self.cross_layers) {
            let key_rows = apply_linear_with_bias(
                &mut graph,
                encoder_tensor,
                layer.cross_attn_k_weight.as_graph_tensor(),
                zero_bias_tensor,
                "cross_cache_k",
            )?;
            let key_target = self.arena.graph_tensor(cross.key);
            let write_key = graph
                .cpy(key_rows, key_target)
                .map_err(|source| map_err("cross_cache_k_write", source))?;
            graph
                .add_side_effect_root(write_key)
                .map_err(|source| map_err("cross_cache_k_root", source))?;

            let value_rows = apply_linear_with_bias(
                &mut graph,
                encoder_tensor,
                layer.cross_attn_v_weight.as_graph_tensor(),
                layer.cross_attn_v_bias.as_graph_tensor(),
                "cross_cache_v",
            )?;
            let value_target = self.arena.graph_tensor(cross.value);
            let write_value = graph
                .cpy(value_rows, value_target)
                .map_err(|source| map_err("cross_cache_v_write", source))?;
            graph
                .add_side_effect_root(write_value)
                .map_err(|source| map_err("cross_cache_v_root", source))?;
            last_value_rows = Some(value_rows);
        }
        let output_root = last_value_rows.ok_or(FireRedDecoderError::InvalidInput {
            reason: "decoder must have at least one layer".to_string(),
        })?;
        graph
            .set_output(output_root)
            .map_err(|source| map_err("cross_cache_set_output", source))?;
        graph
            .set_f32_slice(encoder_tensor, encoder_rows, "firered_dec_encoder_rows")
            .map_err(|source| map_err("encoder_rows_upload", source))?;
        let expected_len = frame_count
            .checked_mul(d_model)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        graph
            .compute_output_f32(output_root, expected_len)
            .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        Ok(())
    }

    /// Compute logits for the next token given the full token prefix so far
    /// (prompt + already-generated tokens). Incremental: after the first call
    /// (which may prefill more than one token), every subsequent call must
    /// append exactly one new token.
    pub(crate) fn compute_step_logits(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        let total_prefix_tokens = decoder_tokens.len();
        if total_prefix_tokens == 0 {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "decoder token_count must be > 0".to_string(),
            });
        }
        if total_prefix_tokens > self.metadata.decoder_pe_len {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "decoder token_count {total_prefix_tokens} exceeds max context {}",
                    self.metadata.decoder_pe_len
                ),
            });
        }
        let position_offset = self.cached_positions;
        let single_token;
        let decode_tokens: &[u32] = if position_offset == 0 {
            decoder_tokens
        } else {
            if total_prefix_tokens != position_offset.saturating_add(1) {
                return Err(FireRedDecoderError::InvalidInput {
                    reason: format!(
                        "incremental decoder prefix mismatch: got {total_prefix_tokens} tokens, \
                         expected {position_offset} cached + 1"
                    ),
                });
            }
            single_token = [*decoder_tokens.last().expect("checked non-empty above")];
            &single_token
        };
        let token_count = decode_tokens.len();
        let total_token_count = position_offset
            .checked_add(token_count)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;

        let mut graph = self.runner.start_graph();
        let token_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "firered_dec_tokens")
            .map_err(|source| map_err("tokens_alloc", source))?;
        graph
            .set_input(token_ids_tensor)
            .map_err(|source| map_err("tokens_input", source))?;
        let position_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "firered_dec_positions")
            .map_err(|source| map_err("positions_alloc", source))?;
        graph
            .set_input(position_ids_tensor)
            .map_err(|source| map_err("positions_input", source))?;

        let self_attention_mask = if token_count > 1 {
            let mask = graph
                .new_tensor_3d_f16(token_count, token_count, 1, "firered_dec_self_mask")
                .map_err(|source| map_err("self_mask_alloc", source))?;
            graph
                .set_input(mask)
                .map_err(|source| map_err("self_mask_input", source))?;
            Some(mask)
        } else {
            None
        };

        let token_ids_i32 = tokens_as_i32(decode_tokens)?;
        let position_ids_i32 = position_ids_i32_with_offset(position_offset, token_count)?;

        let token_state = graph
            .get_rows(
                self.weights.token_embedding.as_graph_tensor(),
                token_ids_tensor,
            )
            .map_err(|source| map_err("embed_get_rows", source))?;
        let scaled_token_state = graph
            .scale(token_state, (d_model as f32).sqrt())
            .map_err(|source| map_err("embed_xscale", source))?;
        let position_state = graph
            .get_rows(
                self.weights.positional_encoding.as_graph_tensor(),
                position_ids_tensor,
            )
            .map_err(|source| map_err("position_get_rows", source))?;
        let mut state = graph
            .add(scaled_token_state, position_state)
            .map_err(|source| map_err("embed_add_pos", source))?;

        let zero_bias_tensor = self.arena.graph_tensor(self.zero_bias);
        for (layer, (cross, self_kv)) in self
            .weights
            .layers
            .iter()
            .zip(self.cross_layers.iter().zip(&self.self_kv_layers))
        {
            let config = Seq2SeqLayerConfig {
                hidden: d_model,
                attention_heads: heads,
                head_dim,
                token_count,
                n_seq: 1,
                total_token_count,
                position_offset,
                layer_norm_epsilon: FIRERED_DECODER_LAYER_NORM_EPSILON,
                ffn_activation: FeedForwardActivation::Gelu,
                self_kv_max_positions: self.metadata.decoder_pe_len,
                cross_frame_count: self.cross_frame_count,
                cross_hidden_size: d_model,
            };
            let weights = Seq2SeqLayerWeights {
                self_attn_norm_weight: layer.self_attn_norm_weight.as_graph_tensor(),
                self_attn_norm_bias: layer.self_attn_norm_bias.as_graph_tensor(),
                self_attn_q_weight: layer.self_attn_q_weight.as_graph_tensor(),
                self_attn_q_bias: layer.self_attn_q_bias.as_graph_tensor(),
                self_attn_k_weight: layer.self_attn_k_weight.as_graph_tensor(),
                self_attn_k_bias: zero_bias_tensor,
                self_attn_v_weight: layer.self_attn_v_weight.as_graph_tensor(),
                self_attn_v_bias: layer.self_attn_v_bias.as_graph_tensor(),
                self_attn_o_weight: layer.self_attn_out_weight.as_graph_tensor(),
                self_attn_o_bias: layer.self_attn_out_bias.as_graph_tensor(),
                cross_attn_norm_weight: layer.cross_attn_norm_weight.as_graph_tensor(),
                cross_attn_norm_bias: layer.cross_attn_norm_bias.as_graph_tensor(),
                cross_attn_q_weight: layer.cross_attn_q_weight.as_graph_tensor(),
                cross_attn_q_bias: layer.cross_attn_q_bias.as_graph_tensor(),
                cross_attn_o_weight: layer.cross_attn_out_weight.as_graph_tensor(),
                cross_attn_o_bias: layer.cross_attn_out_bias.as_graph_tensor(),
                ffn_norm_weight: layer.ffn_norm_weight.as_graph_tensor(),
                ffn_norm_bias: layer.ffn_norm_bias.as_graph_tensor(),
                ffn_up_weight: layer.ffn_up_weight.as_graph_tensor(),
                ffn_up_bias: layer.ffn_up_bias.as_graph_tensor(),
                ffn_down_weight: layer.ffn_down_weight.as_graph_tensor(),
                ffn_down_bias: layer.ffn_down_bias.as_graph_tensor(),
            };
            let self_kv_handle = SelfKvHandle {
                key: self.arena.graph_tensor(self_kv.key),
                value: self.arena.graph_tensor(self_kv.value),
                row_indices: None,
                attention_mask: self_attention_mask,
            };
            let cross_kv_handle = CrossKvHandle {
                key: self.arena.graph_tensor(cross.key),
                value: self.arena.graph_tensor(cross.value),
            };
            let block = seq2seq_layer(
                &mut graph,
                state,
                config,
                weights,
                self_kv_handle,
                cross_kv_handle,
                map_err,
            )?;
            if let Some((mask_tensor, bits)) = block.deferred_self_mask {
                graph
                    .set_f16_bits_slice(mask_tensor, &bits, "firered_dec_layer_self_mask")
                    .map_err(|source| map_err("layer_self_mask_upload", source))?;
            }
            state = block.output;
        }

        state = apply_affine_layer_norm(
            &graph,
            state,
            FIRERED_DECODER_LAYER_NORM_EPSILON,
            self.weights.out_norm_weight.as_graph_tensor(),
            self.weights.out_norm_bias.as_graph_tensor(),
            AffineLayerNormSteps {
                norm: "decoder_out_norm",
                scale: "decoder_out_norm",
                bias: "decoder_out_norm",
            },
            map_err,
        )?;
        let last_state = view_last_token_state(&graph, state, d_model, token_count)?;
        let logits = graph
            .mul_mat(self.weights.out_proj_weight.as_graph_tensor(), last_state)
            .map_err(|source| map_err("output_proj", source))?;
        graph
            .set_output(logits)
            .map_err(|source| map_err("set_output", source))?;

        graph
            .set_i32_slice(token_ids_tensor, &token_ids_i32, "firered_dec_tokens")
            .map_err(|source| map_err("tokens_upload", source))?;
        graph
            .set_i32_slice(
                position_ids_tensor,
                &position_ids_i32,
                "firered_dec_positions",
            )
            .map_err(|source| map_err("positions_upload", source))?;
        if let Some(mask) = self_attention_mask {
            let bits = build_causal_mask_f16_bits(token_count, "firered_dec_self_mask", map_err)?;
            graph
                .set_f16_bits_slice(mask, &bits, "firered_dec_self_mask")
                .map_err(|source| map_err("self_mask_upload", source))?;
        }

        let output = graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        self.cached_positions = total_token_count;
        Ok(output)
    }
}

fn apply_linear_with_bias<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, FireRedDecoderError> {
    let projected = graph
        .mul_mat(weight, input)
        .map_err(|source| map_err(step, source))?;
    graph
        .add(projected, bias)
        .map_err(|source| map_err(step, source))
}

fn view_last_token_state<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    hidden: usize,
    prefix_len: usize,
) -> Result<GgmlCpuTensor<'a>, FireRedDecoderError> {
    let contiguous_state = graph
        .cont(state)
        .map_err(|source| map_err("last_token_cont", source))?;
    let row_stride = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FireRedDecoderError::ShapeOverflow)?;
    let offset = prefix_len
        .checked_sub(1)
        .and_then(|index| index.checked_mul(row_stride))
        .ok_or(FireRedDecoderError::ShapeOverflow)?;
    graph
        .view_2d(contiguous_state, hidden, 1, row_stride, offset)
        .map_err(|source| map_err("last_token_view", source))
}

fn tokens_as_i32(tokens: &[u32]) -> Result<Vec<i32>, FireRedDecoderError> {
    tokens
        .iter()
        .map(|&token| {
            i32::try_from(token).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("token id {token} does not fit i32"),
            })
        })
        .collect()
}

fn position_ids_i32_with_offset(
    position_offset: usize,
    token_count: usize,
) -> Result<Vec<i32>, FireRedDecoderError> {
    (0..token_count)
        .map(|index| {
            let position = position_offset
                .checked_add(index)
                .ok_or(FireRedDecoderError::ShapeOverflow)?;
            i32::try_from(position).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("decoder position {position} does not fit i32"),
            })
        })
        .collect()
}

/// Greedy-decode result: the detokenized text and the raw generated ids
/// (excluding the leading `<sos>` prompt token, excluding the trailing
/// `<eos>`).
#[derive(Debug, Clone)]
pub(crate) struct FireRedAedGreedyDecodeOutput {
    pub text: String,
    pub generated_tokens: Vec<u32>,
}

/// Run the full attention-based greedy decode for one utterance: build a
/// fresh decoder runtime sized to `encoder_frame_count`, precompute the
/// cross-KV cache from `encoder_rows`, then autoregress from `<sos>` one
/// token at a time (argmax logits) until `<eos>` or the generation budget is
/// exhausted.
pub(crate) fn run_firered_aed_decoder_greedy(
    runtime_path: &Path,
    metadata: FireRedAedExecutionMetadata,
    encoder_rows: &[f32],
    encoder_frame_count: usize,
    decode_text: impl Fn(&[u32]) -> Result<String, String>,
) -> Result<FireRedAedGreedyDecodeOutput, FireRedDecoderError> {
    let mut runtime = FireRedDecoderGraphRuntime::new(runtime_path, metadata, encoder_frame_count)?;
    run_firered_aed_decoder_greedy_with_runtime(&mut runtime, metadata, encoder_rows, decode_text)
}

/// Same greedy decode loop as [`run_firered_aed_decoder_greedy`], but against
/// an already-built (and possibly cached/reused across transcriptions)
/// [`FireRedDecoderGraphRuntime`]. Resets the runtime's cross-KV cache and
/// incremental self-KV position for this utterance via
/// [`FireRedDecoderGraphRuntime::populate_cross_attention_cache`] before
/// decoding, so a cache hit never leaks state from a prior utterance.
pub(crate) fn run_firered_aed_decoder_greedy_with_runtime(
    runtime: &mut FireRedDecoderGraphRuntime,
    metadata: FireRedAedExecutionMetadata,
    encoder_rows: &[f32],
    decode_text: impl Fn(&[u32]) -> Result<String, String>,
) -> Result<FireRedAedGreedyDecodeOutput, FireRedDecoderError> {
    runtime.populate_cross_attention_cache(encoder_rows)?;

    let max_new_tokens =
        crate::models::decode_token_history::context_window_budget(metadata.decoder_pe_len, 1)
            .unwrap_or(0);
    let mut prefix = vec![metadata.sos_token_id];
    let mut generated_tokens = Vec::new();
    for _ in 0..max_new_tokens {
        let logits = runtime.compute_step_logits(&prefix)?;
        let next_token = argmax_token_id(&logits)?;
        if next_token == metadata.eos_token_id {
            break;
        }
        prefix.push(next_token);
        generated_tokens.push(next_token);
    }
    let text =
        decode_text(&generated_tokens).map_err(|reason| FireRedDecoderError::InvalidInput {
            reason: format!("tokenizer decode failed: {reason}"),
        })?;
    Ok(FireRedAedGreedyDecodeOutput {
        text,
        generated_tokens,
    })
}

fn argmax_token_id(logits: &[f32]) -> Result<u32, FireRedDecoderError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .ok_or(FireRedDecoderError::InvalidInput {
            reason: "decoder step produced empty logits".to_string(),
        })
}
