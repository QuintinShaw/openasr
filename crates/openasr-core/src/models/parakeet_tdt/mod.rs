//! parakeet-tdt (NVIDIA Parakeet TDT, FastConformer + Token-and-Duration
//! Transducer) — the multilingual transducer sibling of `parakeet_ctc`.
//!
//! The encoder is the SAME FastConformer stack parakeet-ctc already runs
//! (dw-striding subsampling prelude + shared `nn::encoder::conformer_block`,
//! same `enc.blk.{i}.*` / `enc.sub.*` tensor conventions), with two data-level
//! differences: the v3 checkpoint ships NO projection/conv biases
//! (`attention_bias`/`convolution_bias` false — the loader synthesizes zero
//! biases so the shared block applies unchanged) and `scale_input` is false.
//! The genuinely-new pieces are the encoder output projection (`enc.proj`, the
//! joint's encoder branch), the 2-layer LSTM prediction network, the joint
//! head with its fused `[vocab+blank | durations]` output, and the TDT greedy
//! decode loop (duration-driven frame skipping).

pub mod package_import;
pub use package_import::{
    ParakeetTdtImportRequest, ParakeetTdtImportResult, ParakeetTdtQuantizationMode,
    convert_local_parakeet_tdt_source_to_runtime_pack,
};
// Encoder graph / predictor / greedy decode / executor land in the follow-up
// stages; the importer + runtime contract are the S1 surface.
pub(crate) mod runtime_contract;

pub(crate) const PARAKEET_TDT_MODEL_FAMILY: &str = "parakeet-tdt";
pub(crate) const PARAKEET_TDT_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-tdt";
