mod audio_encoder;
mod batched_decode;
mod decode_prompt;
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
pub use forced_aligner_import::{
    QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID, QWEN3_FORCED_ALIGNER_MODEL_FAMILY,
    Qwen3ForcedAlignerLocalSourceError, Qwen3ForcedAlignerLocalSourceImportRequest,
    Qwen3ForcedAlignerLocalSourceImportRuntimeResult,
    convert_local_qwen_forced_aligner_source_to_runtime_pack,
};
pub(crate) use forced_aligner_runtime::{
    ForcedAlignItem, refine_word_timestamps_with_forced_aligner,
};
pub(crate) use frontend::{Qwen3AsrMelFrontendPlan, load_qwen3_mel_frontend_plan_from_reader};
pub(crate) use ggml_executor::Qwen3AsrGgmlExecutor;
pub(crate) use kv_cache::Qwen3AsrLayerKvCacheState;
pub(crate) use llm_transformer::{
    Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmWholeDecoderGraphExecutor,
    Qwen3AsrLlmWholeStepOutput, Qwen3AsrLlmWholeStepTop1Output, even_prefill_chunk_len,
    load_qwen3_llm_attention_projections_from_reader,
    load_qwen3_llm_attention_projections_from_reader_with_materialized_qkv,
};
pub(crate) use logits_head::{
    Qwen3AsrLlmFusedLogitsHeadSpec, Qwen3AsrLlmLogitsHead, load_qwen3_llm_logits_head_from_reader,
    load_qwen3_llm_logits_head_from_reader_with_output_tensor,
};
pub use package_import::{
    Qwen3AsrLocalSourceError, Qwen3AsrLocalSourceImportRequest,
    Qwen3AsrLocalSourceImportRuntimeResult, Qwen3AsrRuntimeQuantizationMode,
    convert_local_qwen_source_to_runtime_pack,
};
pub(crate) use prepared_runtime::{
    Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError, build_qwen_prepared_runtime,
};
pub(crate) use token_embedding::{
    Qwen3AsrTokenEmbeddingTable, load_qwen3_token_embedding_table_from_reader,
};
pub(crate) use tokenizer::Qwen3AsrTokenizer;

pub const QWEN3_ASR_MODEL_FAMILY: &str = "qwen3-asr";
