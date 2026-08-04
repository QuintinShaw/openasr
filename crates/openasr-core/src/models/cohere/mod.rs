//! Cohere Transcribe (`CohereLabs/cohere-transcribe`, 2B) model family: a
//! Conformer audio encoder (pre-encode conv stem + `enc.blk` self-attention /
//! conv / dual-FFN blocks) feeding a stock Transformer seq2seq decoder
//! (`dec.blk` self-attention + cross-attention layers) over an SPM-BPE vocab.
//! It is a data-driven composer family: the encoder/decoder stacks assemble
//! over the shared `nn/` blocks and validate against the descriptor's
//! block-stack facets at construction.
//!
//! The family's complete lifecycle row -- identity, pack, execution,
//! topology, optimization, quantization, and conformance facets -- is declared
//! in the canonical architecture inventory (`arch/mod.rs`); offline/streaming
//! dispatch, executor materialization, runtime-validator routing, and
//! content-id eviction are generated projections of that row, not
//! family-specific central wiring. The checkpoint-to-`.oasr` importer
//! ([`package_import`]) is the row's pack-import surface: it reads the
//! HF-style source through the shared `local_source_import` track, writes
//! through the shared `PackEnvelope`/`OasrPackWriter` seam, and returns the
//! `VerifiedPack` proof every install/runtime path consumes -- the exposed
//! `output_path` is diagnostic, not an execution capability.
//!
//! Family-varying semantics stay in typed descriptor facets plus narrow
//! adapters: phrase bias is `OpenAsrPhraseBiasStrategy::Always` with the SPM
//! encoder riding the shared [`crate::models::phrase_bias_decode`] seam;
//! eligible batched serving goes through the shared seq2seq serve-batch owner
//! ([`batched_decode`] is the family adapter, not a second batch loop); and
//! source-language steering is the descriptor's `SelectsViaPrompt` hint,
//! whose conditioned default is read from the facet in [`prompt`].

mod batched_decode;
pub(crate) mod capacity;
mod decode_budget;
mod decoder_graph;
mod decoder_weights;
pub(crate) mod encoder_graph;
mod encoder_weights;
mod frontend;
mod ggml_executor;
mod graph_config;
mod greedy_decode;
mod package_import;
mod prepared_runtime;
mod prompt;
pub(crate) mod runtime_contract;
mod tensor_names;
mod tokenizer;
mod weights;

pub const COHERE_TRANSCRIBE_MODEL_FAMILY: &str = "cohere-transcribe";

pub(crate) use decoder_weights::{
    CohereTranscribeDecoderWeights, load_cohere_transcribe_decoder_weights_for_runtime_from_reader,
};
pub(crate) use encoder_weights::{
    CohereTranscribeEncoderWeights, load_cohere_transcribe_encoder_weights_from_reader,
};
pub(crate) use frontend::{
    CohereTranscribeFrontendPlan, load_cohere_transcribe_frontend_plan_from_reader,
};
pub(crate) use ggml_executor::CohereTranscribeGgmlExecutor;
pub(crate) use package_import::TENSOR_QUANTIZATION_CONTRACT;
pub use package_import::{
    CohereLocalSourceError, CohereLocalSourceImportRequest, CohereLocalSourceImportRuntimeResult,
    CohereRuntimeQuantizationMode, convert_local_cohere_source_to_runtime_pack,
};
pub(crate) use prepared_runtime::{
    CoherePreparedRuntime, CoherePreparedRuntimeError, build_cohere_prepared_runtime,
};
pub(crate) use tokenizer::CohereTranscribeTokenizer;
