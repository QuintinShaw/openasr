//! parakeet-ctc (NVIDIA FastConformer-CTC) — the goal-1 `Ctc`-shape onboarding.
//!
//! Reuses the shared `nn::encoder::conformer_block` verbatim (the FastConformer
//! encoder layer is the same Transformer-XL two-bias rel-pos + macaron-FFN +
//! GLU/depthwise-conv block cohere already runs), the cohere
//! `enc.blk.{i}.*` tensor-name convention, and the non-autoregressive
//! `ctc_greedy_decode` path. The only genuinely-new pieces are the dw-striding
//! subsampling prelude, the CTC head, and the per-family loader/executor.

pub mod package_import;
pub use package_import::{
    ParakeetCtcImportRequest, ParakeetCtcImportResult, ParakeetCtcQuantizationMode,
    convert_local_parakeet_ctc_source_to_runtime_pack,
};
pub(crate) mod encoder_graph;
pub(crate) mod encoder_weights;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod graph_config;
pub(crate) mod runtime_contract;
pub(crate) mod tokenizer;

/// Crate-internal model-family + architecture ids for parakeet-ctc. The full
/// architecture descriptor + component-id wiring lands in S4; the importer (S2)
/// only needs these as pack metadata strings.
pub(crate) const PARAKEET_CTC_MODEL_FAMILY: &str = "parakeet-ctc";
pub(crate) const PARAKEET_CTC_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-ctc";
