//! FireRedASR-AED-L (`FireRedTeam/FireRedASR-AED-L`) model family.
//!
//! Attention-based encoder-decoder: a 16-layer Conformer encoder (macaron FFN,
//! rel-pos MHSA with per-projection q/k/v LayerNorms, GLU + depthwise conv with
//! a LayerNorm mid-block) over a Conv2d 4x subsampling stem, plus a 16-layer
//! pre-norm Transformer decoder (causal self-attention + cross-attention +
//! GELU FFN, absolute sinusoidal positions). No CTC branch: decoding is pure
//! autoregressive attention. Char + SentencePiece hybrid vocab (`dict.txt`),
//! Mandarin/Chinese-dialect + English. Apache-2.0.
//!
//! Stage status:
//! - The checkpoint-to-GGUF importer lives in [`package_import`].
//! - The fbank+CMVN frontend ([`frontend`]), the detokenizer ([`tokenizer`]),
//!   and the pack-metadata contract ([`runtime_contract`]) are implemented and
//!   unit-tested here.
//! - The Conformer encoder graph, KV-cached decoder, and executor land in the
//!   executor stage.

pub(crate) mod encoder_graph;
pub(crate) mod encoder_weights;
pub(crate) mod frontend;
pub mod package_import;
pub(crate) mod runtime_contract;
pub(crate) mod tokenizer;

pub use package_import::{
    FireRedAedImportRequest, FireRedAedImportResult, FireRedAedQuantizationMode,
    convert_local_firered_aed_source_to_runtime_pack,
};
