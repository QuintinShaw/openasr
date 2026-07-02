//! `GilgameshWind/X-ASR-zh-en` — a k2/icefall **streaming Zipformer2 transducer**
//! (offline-streaming unified): a 6-stack Zipformer2 encoder + a *stateless*
//! RNN-T decoder (embedding + context-2 conv) + a tanh joiner. This is NOT the
//! FastConformer/`parakeet_ctc` family — the encoder math is genuinely different
//! (downsampled stacks, NonlinAttention, BiasNorm, bypass), so it gets its own
//! module rather than reusing the shared `nn::encoder::conformer_block`.
//!
//! Status: executor onboarding in progress. The importer is **name-preserving**
//! at the semantic level and uses `compact_xasr_name()` as the shared pack ↔
//! runtime tensor naming contract. Runtime code must resolve upstream icefall
//! names through the same compaction function rather than hard-coding GGUF names.
#![allow(dead_code)]

pub(crate) mod decoder;
pub(crate) mod encoder_graph;
pub(crate) mod encoder_ops;
pub(crate) mod encoder_reference;
pub(crate) mod encoder_weights;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod graph_config;
pub(crate) mod greedy;
pub(crate) mod joiner;
pub mod package_import;
pub use package_import::{
    XasrZipformerImportRequest, XasrZipformerImportResult, XasrZipformerQuantizationMode,
    convert_local_xasr_zipformer_source_to_runtime_pack,
};
pub(crate) mod runtime;
pub(crate) mod runtime_contract;
pub(crate) mod streaming_decoder;
pub(crate) mod tokenizer;
pub(crate) mod weights;
