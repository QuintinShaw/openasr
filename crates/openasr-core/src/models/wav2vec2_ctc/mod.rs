//! wav2vec2-ctc (facebook/wav2vec2-base-960h) — a second `Ctc`-shape onboarding.
//!
//! Unlike parakeet (log-mel + FastConformer), wav2vec2 consumes RAW 16 kHz
//! waveform: a 7-layer strided Conv1d feature extractor → feature projection →
//! a grouped positional-conv embedding (weight-norm folded at import) → 12
//! post-norm transformer encoder layers (`do_stable_layer_norm=False`) → CTC
//! head. Reuses the shared `nn::attn`/`nn::norm` sub-builders, the new
//! `nn::wav2vec2` blocks (grouped conv + post-norm layer), and the
//! non-autoregressive `ctc_greedy_decode` path.

pub mod package_import;
pub use package_import::{
    Wav2Vec2CtcImportRequest, Wav2Vec2CtcImportResult, Wav2Vec2CtcQuantizationMode,
    convert_local_wav2vec2_ctc_source_to_runtime_pack,
};
pub(crate) mod encoder_graph;
pub(crate) mod encoder_weights;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod graph_config;
pub(crate) mod runtime_contract;
pub(crate) mod tokenizer;

/// Crate-internal model-family + architecture ids for wav2vec2-ctc.
pub(crate) const WAV2VEC2_CTC_MODEL_FAMILY: &str = "wav2vec2-ctc";
pub(crate) const WAV2VEC2_CTC_GGML_ARCHITECTURE_ID: &str = "wav2vec2-ctc";
