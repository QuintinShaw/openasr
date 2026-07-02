mod batched_decode;
mod execution_policy;
mod execution_trace;
mod frontend;
mod ggml_decoder_graph;
mod ggml_decoder_weights;
mod ggml_encoder_graph;
mod ggml_encoder_prelude;
mod ggml_encoder_weights;
mod ggml_executor;
mod ggml_tensor_binding;
mod graph_config;
mod greedy_decode;
mod lid;
mod local_source;
mod mel;
mod package_import;
mod runtime_contract;
mod tokenizer;

pub use frontend::whisper_log_mel_spectrogram_16khz_mono_v0;
pub(crate) use ggml_executor::WhisperGgmlExecutor;
pub use local_source::WhisperLocalSourceError;
pub use package_import::{
    WhisperLocalSourceImportRequest, WhisperLocalSourceImportRuntimeResult,
    WhisperRuntimeQuantizationMode, convert_local_whisper_hf_source_to_runtime_pack,
};
pub use tokenizer::WhisperTokenizer;
pub(crate) use tokenizer::whisper_metadata_is_multilingual;

pub const WHISPER_MODEL_FAMILY: &str = "whisper";

const WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT: usize = 32;
