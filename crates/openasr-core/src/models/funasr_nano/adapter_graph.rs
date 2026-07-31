//! The Fun-ASR-Nano audio adaptor: `linear1 (512->2048) -> ReLU -> linear2
//! (2048->1024)` then 2 standard pre-norm Transformer blocks (`D=1024`, `H=8`,
//! bidirectional self-attention, ReLU FFN `1024->256->1024`), run as one ggml
//! graph over the full utterance's encoder hidden states. `downsample_rate` is
//! 1 (no frame stacking): the adaptor preserves the encoder's frame count and
//! the executor keeps only the leading `n_aud` frames afterward.
//!
//! Weight residency mirrors `firered_llm::adapter_graph`: every tensor (the two
//! MLP linears and each block's q/k/v/out/up/down projections plus their
//! biases and LayerNorm affines) is bound zero-copy from the same already-open
//! `.oasr` mmap via [`GgmlLoadedWeightContext`], and the per-block transformer
//! math reuses the shared `nn::encoder::transformer_layer` primitive with a
//! zero (bidirectional) attention mask.

use thiserror::Error;

use crate::GgmlRuntimeSource;
use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedTensor, GgmlLoadedWeightContext,
};
use crate::nn::encoder::{
    TransformerEncoderConfig, TransformerEncoderLayerWeights, transformer_layer,
};
use crate::nn::ffn::FeedForwardActivation;

use crate::models::sensevoice::graph_config::sensevoice_encoder_graph_config;

use super::runtime_contract::{FUNASR_NANO_ADAPTOR_LAYER_NORM_EPSILON, FunasrNanoAdapterMetadata};
use super::tensor_names::{
    ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
};

const ADAPTER_ENC_ROWS_TENSOR_NAME: &str = "funasr_nano_adapter_enc_rows";
const ADAPTER_MASK_TENSOR_NAME: &str = "funasr_nano_adapter_mask";

#[derive(Debug, Error)]
pub(crate) enum FunasrNanoAdapterError {
    #[error("funasr-nano adapter graph failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("funasr-nano adapter is missing tensor '{name}'")]
    MissingTensor { name: String },
    #[error(
        "funasr-nano adapter encoder rows shape is invalid: frame_count={frame_count} \
         encoder_d_model={encoder_d_model} values_len={values_len}"
    )]
    InvalidEncoderRowsShape {
        frame_count: usize,
        encoder_d_model: usize,
        values_len: usize,
    },
    #[error("funasr-nano adapter graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("funasr-nano adapter output contains non-finite values")]
    NonFiniteValues,
    #[error("funasr-nano adapter shape overflowed")]
    ShapeOverflow,
}

fn map_err(step: &'static str, source: GgmlCpuGraphError) -> FunasrNanoAdapterError {
    FunasrNanoAdapterError::GraphBuildFailed { step, source }
}

fn tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, FunasrNanoAdapterError> {
    loaded
        .tensor(name)
        .ok_or_else(|| FunasrNanoAdapterError::MissingTensor {
            name: name.to_string(),
        })
}

/// One transformer block's bound weight handles (`adaptor.blk.{i}.*`).
struct AdapterBlock {
    attn_norm_weight: GgmlLoadedTensor,
    attn_norm_bias: GgmlLoadedTensor,
    attn_q_weight: GgmlLoadedTensor,
    attn_q_bias: GgmlLoadedTensor,
    attn_k_weight: GgmlLoadedTensor,
    attn_k_bias: GgmlLoadedTensor,
    attn_v_weight: GgmlLoadedTensor,
    attn_v_bias: GgmlLoadedTensor,
    attn_out_weight: GgmlLoadedTensor,
    attn_out_bias: GgmlLoadedTensor,
    ffn_norm_weight: GgmlLoadedTensor,
    ffn_norm_bias: GgmlLoadedTensor,
    ffn_up_weight: GgmlLoadedTensor,
    ffn_up_bias: GgmlLoadedTensor,
    ffn_down_weight: GgmlLoadedTensor,
    ffn_down_bias: GgmlLoadedTensor,
}

fn load_block(
    loaded: &GgmlLoadedWeightContext,
    index: usize,
) -> Result<AdapterBlock, FunasrNanoAdapterError> {
    let n = |suffix: &str| format!("adaptor.blk.{index}.{suffix}");
    Ok(AdapterBlock {
        attn_norm_weight: tensor(loaded, &n("attn.norm.weight"))?,
        attn_norm_bias: tensor(loaded, &n("attn.norm.bias"))?,
        attn_q_weight: tensor(loaded, &n("attn.q.weight"))?,
        attn_q_bias: tensor(loaded, &n("attn.q.bias"))?,
        attn_k_weight: tensor(loaded, &n("attn.k.weight"))?,
        attn_k_bias: tensor(loaded, &n("attn.k.bias"))?,
        attn_v_weight: tensor(loaded, &n("attn.v.weight"))?,
        attn_v_bias: tensor(loaded, &n("attn.v.bias"))?,
        attn_out_weight: tensor(loaded, &n("attn.out.weight"))?,
        attn_out_bias: tensor(loaded, &n("attn.out.bias"))?,
        ffn_norm_weight: tensor(loaded, &n("ffn.norm.weight"))?,
        ffn_norm_bias: tensor(loaded, &n("ffn.norm.bias"))?,
        ffn_up_weight: tensor(loaded, &n("ffn.up.weight"))?,
        ffn_up_bias: tensor(loaded, &n("ffn.up.bias"))?,
        ffn_down_weight: tensor(loaded, &n("ffn.down.weight"))?,
        ffn_down_bias: tensor(loaded, &n("ffn.down.bias"))?,
    })
}

pub(crate) struct FunasrNanoAdapterGraph {
    runner: GgmlCpuGraphRunner,
    _loaded: GgmlLoadedWeightContext,
    metadata: FunasrNanoAdapterMetadata,
    linear1_weight: GgmlLoadedTensor,
    linear1_bias: GgmlLoadedTensor,
    linear2_weight: GgmlLoadedTensor,
    linear2_bias: GgmlLoadedTensor,
    blocks: Vec<AdapterBlock>,
}

impl FunasrNanoAdapterGraph {
    pub(crate) fn new(
        runtime_source: &GgmlRuntimeSource,
        metadata: FunasrNanoAdapterMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FunasrNanoAdapterError> {
        let runner = GgmlCpuGraphRunner::new(sensevoice_encoder_graph_config(backend))
            .map_err(|source| map_err("runner_init", source))?;
        let loaded = runner
            .load_gguf_weight_context(runtime_source)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let linear1_weight = tensor(&loaded, ADAPTOR_LINEAR1_WEIGHT)?;
        let linear1_bias = tensor(&loaded, ADAPTOR_LINEAR1_BIAS)?;
        let linear2_weight = tensor(&loaded, ADAPTOR_LINEAR2_WEIGHT)?;
        let linear2_bias = tensor(&loaded, ADAPTOR_LINEAR2_BIAS)?;
        let mut blocks = Vec::with_capacity(metadata.n_layers);
        for index in 0..metadata.n_layers {
            blocks.push(load_block(&loaded, index)?);
        }
        Ok(Self {
            runner,
            _loaded: loaded,
            metadata,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            blocks,
        })
    }

    /// Run the adaptor over a full utterance's encoder hidden states
    /// (token-major `[frame][encoder_d_model]`). Returns (token-major output
    /// rows `[frame][llm_dim]`, frame_count) -- the frame count is unchanged
    /// (downsample_rate 1).
    pub(crate) fn run(
        &mut self,
        encoder_rows: &[f32],
        frame_count: usize,
        encoder_d_model: usize,
    ) -> Result<(Vec<f32>, usize), FunasrNanoAdapterError> {
        let llm_dim = self.metadata.llm_dim;
        let head_dim = llm_dim / self.metadata.n_heads;
        let expected_len = frame_count.checked_mul(encoder_d_model).ok_or(
            FunasrNanoAdapterError::InvalidEncoderRowsShape {
                frame_count,
                encoder_d_model,
                values_len: encoder_rows.len(),
            },
        )?;
        if encoder_rows.len() != expected_len || frame_count == 0 {
            return Err(FunasrNanoAdapterError::InvalidEncoderRowsShape {
                frame_count,
                encoder_d_model,
                values_len: encoder_rows.len(),
            });
        }
        if encoder_rows.iter().any(|value| !value.is_finite()) {
            return Err(FunasrNanoAdapterError::NonFiniteValues);
        }

        let mut graph = self.runner.start_graph();
        let enc_rows = graph
            .new_tensor_2d_f32(encoder_d_model, frame_count, ADAPTER_ENC_ROWS_TENSOR_NAME)
            .map_err(|source| map_err("ggml_new_tensor_2d(enc_rows)", source))?;
        graph
            .set_input(enc_rows)
            .map_err(|source| map_err("ggml_set_input(enc_rows)", source))?;
        // Zero (bidirectional) additive attention mask [frames, frames]: the
        // adaptor attends over the whole utterance with no causal masking.
        let mask = graph
            .new_tensor_2d_f32(frame_count, frame_count, ADAPTER_MASK_TENSOR_NAME)
            .map_err(|source| map_err("ggml_new_tensor_2d(mask)", source))?;
        graph
            .set_input(mask)
            .map_err(|source| map_err("ggml_set_input(mask)", source))?;

        // linear1 -> ReLU -> linear2.
        let mut hidden = graph
            .mul_mat(self.linear1_weight.as_graph_tensor(), enc_rows)
            .map_err(|source| map_err("ggml_mul_mat(linear1)", source))?;
        hidden = graph
            .add(hidden, self.linear1_bias.as_graph_tensor())
            .map_err(|source| map_err("ggml_add(linear1_bias)", source))?;
        hidden = graph
            .relu(hidden)
            .map_err(|source| map_err("ggml_relu(adapter)", source))?;
        let mut state = graph
            .mul_mat(self.linear2_weight.as_graph_tensor(), hidden)
            .map_err(|source| map_err("ggml_mul_mat(linear2)", source))?;
        state = graph
            .add(state, self.linear2_bias.as_graph_tensor())
            .map_err(|source| map_err("ggml_add(linear2_bias)", source))?;

        // 2 standard pre-norm transformer blocks (bidirectional, ReLU FFN).
        let config = TransformerEncoderConfig {
            head_dim,
            attention_heads: self.metadata.n_heads,
            token_count: frame_count,
            layer_norm_epsilon: FUNASR_NANO_ADAPTOR_LAYER_NORM_EPSILON,
            ffn_activation: FeedForwardActivation::Relu,
            use_flash_attention: false,
        };
        let map =
            |step: &'static str, source| FunasrNanoAdapterError::GraphBuildFailed { step, source };
        for block in &self.blocks {
            state = transformer_layer(
                &mut graph,
                state,
                mask,
                config,
                TransformerEncoderLayerWeights {
                    attn_norm_weight: block.attn_norm_weight.as_graph_tensor(),
                    attn_norm_bias: block.attn_norm_bias.as_graph_tensor(),
                    attn_q_weight: block.attn_q_weight.as_graph_tensor(),
                    attn_q_bias: block.attn_q_bias.as_graph_tensor(),
                    attn_k_weight: block.attn_k_weight.as_graph_tensor(),
                    attn_k_bias: block.attn_k_bias.as_graph_tensor(),
                    attn_v_weight: block.attn_v_weight.as_graph_tensor(),
                    attn_v_bias: block.attn_v_bias.as_graph_tensor(),
                    attn_out_weight: block.attn_out_weight.as_graph_tensor(),
                    attn_out_bias: block.attn_out_bias.as_graph_tensor(),
                    ffn_norm_weight: block.ffn_norm_weight.as_graph_tensor(),
                    ffn_norm_bias: block.ffn_norm_bias.as_graph_tensor(),
                    ffn_up_weight: block.ffn_up_weight.as_graph_tensor(),
                    ffn_up_bias: block.ffn_up_bias.as_graph_tensor(),
                    ffn_down_weight: block.ffn_down_weight.as_graph_tensor(),
                    ffn_down_bias: block.ffn_down_bias.as_graph_tensor(),
                },
                map,
            )?;
        }

        graph
            .set_output(state)
            .map_err(|source| map_err("ggml_set_output(adapter)", source))?;
        graph
            .prepare_outputs_for_upload(&[state])
            .map_err(|source| map_err("ggml_prepare_outputs(adapter)", source))?;
        graph
            .set_f32_slice(enc_rows, encoder_rows, ADAPTER_ENC_ROWS_TENSOR_NAME)
            .map_err(|source| map_err("ggml_set_f32_slice(enc_rows)", source))?;
        let zero_mask = vec![0.0f32; frame_count * frame_count];
        graph
            .set_f32_slice(mask, &zero_mask, ADAPTER_MASK_TENSOR_NAME)
            .map_err(|source| map_err("ggml_set_f32_slice(mask)", source))?;

        let expected_output_len = frame_count
            .checked_mul(llm_dim)
            .ok_or(FunasrNanoAdapterError::ShapeOverflow)?;
        let rows = graph
            .compute_output_f32(state, expected_output_len)
            .map_err(|error| FunasrNanoAdapterError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        if rows.iter().any(|value| !value.is_finite()) {
            return Err(FunasrNanoAdapterError::NonFiniteValues);
        }
        Ok((rows, frame_count))
    }
}
