//! Dolphin `small.cn` dialect model family (WeNet-format E-Branchformer encoder
//! + Transformer decoder + CTC head, char tokenizer, CTC/attention joint decode).
//!
//! Convert + load phase: the WeNet->GGUF importer ([`package_import`]) writes the
//! fp16 `.oasr` runtime pack, and the dedicated executor ([`executor`]) loads the
//! encoder weights from that pack and runs the parity-verified encoder graph. The
//! fbank frontend and CTC-prefix-beam + attention-rescoring joint decode land in
//! a later phase.

pub(crate) mod decoder_graph;
pub(crate) mod encoder_graph;
pub(crate) mod executor;
pub mod package_import;
pub(crate) mod runtime_contract;

pub use package_import::{
    DolphinImportRequest, DolphinImportResult, DolphinQuantizationMode,
    convert_local_dolphin_wenet_source_to_runtime_pack,
};

#[cfg(test)]
mod parity;
