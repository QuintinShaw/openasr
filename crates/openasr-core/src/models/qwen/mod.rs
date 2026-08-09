mod audio_encoder;
mod batched_decode;
pub(crate) mod capacity;
mod decode_budget;
mod decode_prompt;
mod decoder_contract;
mod decoder_tail;
mod forced_aligner_align_text;
mod forced_aligner_import;
pub(crate) mod forced_aligner_pack;
mod forced_aligner_runtime;
mod frontend;
mod ggml_executor;
mod graph_config;
mod greedy_decode;
mod kv_cache;
mod llm_prefill;
mod llm_transformer;
mod logits_head;
pub(crate) mod lora;
mod package_import;
mod prepared_runtime;
mod prompt_embedding;
pub(crate) mod runtime_contract;
mod tensor_names;
mod token_embedding;
mod tokenizer;

pub(crate) use audio_encoder::{
    Qwen3AsrAudioEncoderWeights, load_qwen3_audio_encoder_weights_from_reader,
};
pub(crate) use decode_prompt::Qwen3AsrDecodePrompt;
pub(crate) use decoder_contract::{
    QWEN_DECODER_MAX_D_MODEL, QWEN_DECODER_MAX_FFN_DIM, QWEN_DECODER_MAX_HEAD_DIM,
    QWEN_DECODER_MAX_LAYERS, QWEN_DECODER_MAX_N_HEADS, QWEN_DECODER_MAX_VOCAB_SIZE,
    QwenDecoderContract, QwenDecoderContractGeometry, QwenDecoderTailTensorNames,
    QwenDecoderVariant, QwenFamilyDecoderProfile,
};
pub(crate) use decoder_tail::{
    QwenDecoderTail, QwenDecoderTailLoadError, load_qwen_decoder_tail_from_contract,
};
pub(crate) use forced_aligner_import::forced_aligner_tensor_role;
pub use forced_aligner_import::{
    QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID, QWEN3_FORCED_ALIGNER_MODEL_FAMILY,
    Qwen3ForcedAlignerLocalSourceError, Qwen3ForcedAlignerLocalSourceImportRequest,
    Qwen3ForcedAlignerLocalSourceImportRuntimeResult,
    convert_local_qwen_forced_aligner_source_to_runtime_pack,
};
pub(crate) use forced_aligner_runtime::{
    ForcedAlignItem, ForcedAlignerProgressEvent, Qwen3ForcedAlignerSession,
    validate_forced_aligner_runtime_pack_contract,
};
pub(crate) use frontend::{Qwen3AsrMelFrontendPlan, load_qwen3_mel_frontend_plan_from_reader};
pub(crate) use ggml_executor::Qwen3AsrGgmlExecutor;
pub(crate) use kv_cache::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrKvCacheCapacityError, Qwen3AsrLayerKvCacheState, qwen_host_kv_quoted_bytes,
};
#[cfg(test)]
pub(crate) use llm_transformer::{
    Qwen3AsrLlmLayerAttentionProjection, load_qwen_family_llm_layer_attention_projection_generic,
    load_qwen3_llm_attention_projections_from_reader,
};
pub(crate) use llm_transformer::{
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrLlmWholeStepOutput,
    Qwen3AsrLlmWholeStepTop1Output, QwenFamilyLlmLayerTensorNames,
    QwenPreparedDecoderGraphCompileRequest, QwenWholeDecoderPlan,
    add_qwen_decoder_prepared_runtime_quote, compile_qwen_whole_decoder_graph_from_prepared_plan,
    even_prefill_chunk_len, quoted_qwen_decoder_system_memory_bytes,
    resolve_qwen_family_production_kv_cache_policy,
};
#[cfg(test)]
pub(crate) use logits_head::load_qwen3_llm_logits_head_from_reader;
pub(crate) use logits_head::{
    DEFAULT_RMS_NORM_EPSILON, Qwen3AsrLlmFusedLogitsHeadSpec, Qwen3AsrLlmLogitsHead,
    Qwen3AsrLlmLogitsHeadRuntime, load_qwen3_llm_logits_head_from_reader_with_output_tensor,
    logits_head_ggml_enabled,
};
pub(crate) use package_import::TENSOR_QUANTIZATION_CONTRACT;
pub use package_import::{
    Qwen3AsrLocalSourceError, Qwen3AsrLocalSourceImportRequest,
    Qwen3AsrLocalSourceImportRuntimeResult, Qwen3AsrRuntimeQuantizationMode,
    convert_local_qwen_source_to_runtime_pack,
};
pub(crate) use prepared_runtime::{
    Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError, build_qwen_prepared_runtime,
};
pub(crate) use prompt_embedding::{
    Qwen3AsrPromptEmbeddings, build_qwen3_prompt_embeddings_with_audio_splice,
};
pub(crate) use token_embedding::load_qwen3_token_embedding_table_from_reader;
pub(crate) use tokenizer::Qwen3AsrTokenizer;

pub const QWEN3_ASR_MODEL_FAMILY: &str = "qwen3-asr";
