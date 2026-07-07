//! Dolphin `small.cn` dialect model family (WeNet-format E-Branchformer encoder
//! + Transformer decoder + CTC head, char tokenizer, CTC/attention joint decode).
//!
//! The WeNet->GGUF importer ([`package_import`]) writes the fp16 `.oasr` runtime
//! pack; the dedicated executor ([`executor`]) runs the full end-to-end pipeline
//! from that pack: the kaldi-fbank [`frontend`] + global CMVN, the parity-verified
//! E-Branchformer [`encoder_graph`], and the CTC/attention [`joint_decode`]
//! (CTC prefix-beam over the CTC head, rescored by the Transformer
//! [`decoder_graph`]).

pub(crate) mod decoder_graph;
pub(crate) mod encoder_graph;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod hotword_context;
pub(crate) mod joint_decode;
pub(crate) mod language;
pub mod package_import;
pub(crate) mod runtime_contract;

pub use package_import::{
    DolphinImportRequest, DolphinImportResult, DolphinLanguageScheme, DolphinQuantizationMode,
    convert_local_dolphin_wenet_source_to_runtime_pack,
};

#[cfg(test)]
mod parity;
