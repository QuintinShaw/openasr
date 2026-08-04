//! FireRedASR-AED-L (`FireRedTeam/FireRedASR-AED-L`) model family.
//!
//! Attention-based encoder-decoder: a 16-layer Conformer encoder (macaron FFN,
//! rel-pos MHSA with per-projection q/k/v LayerNorms, GLU + depthwise conv with
//! a LayerNorm mid-block) over a Conv2d 4x subsampling stem, plus a 16-layer
//! pre-norm Transformer decoder (causal self-attention + cross-attention +
//! GELU FFN, absolute sinusoidal positions). No CTC branch: decoding is pure
//! autoregressive attention from the `<sos>` prompt. Char + SentencePiece
//! hybrid vocab (`dict.txt`), Mandarin/Chinese-dialect + English. Apache-2.0.
//!
//! The family's complete lifecycle row -- identity, pack, execution,
//! topology, optimization, quantization, and conformance facets -- is declared
//! in the canonical architecture inventory (`arch/mod.rs`); offline/streaming
//! dispatch, executor materialization, runtime-validator routing, and
//! content-id eviction are generated projections of that row, not
//! family-specific central wiring. The checkpoint-to-`.oasr` importer
//! ([`package_import`]) is the row's pack-import surface: it reads the
//! safetensors source through the shared `local_source_import` track, writes
//! through the shared `PackEnvelope`/`OasrPackWriter` seam, and returns the
//! `VerifiedPack` proof every install/runtime path consumes -- the exposed
//! `output_path` is diagnostic, not an execution capability.
//!
//! Family-varying semantics stay in typed descriptor facets plus narrow
//! adapters: greedy decode rides the shared seq2seq driver
//! (`run_builtin_seq2seq_decode_policy` in [`decoder_graph`]) under the row's
//! `SharedSeq2SeqGreedy` strategy -- the structural fix for issue #60's
//! long-audio repetition, never a hand-written argmax loop -- with the
//! `ConservativeSeq2SeqV1` longform profile carried by the family decode
//! policy component; the Conformer encoder and attention decoder are an
//! `ArchitectureGraph` (the three-independent-LayerNorm / bias-free topology
//! is not a composer block kind, see [`encoder_graph`]); and the runtime
//! pack contract ([`runtime_contract`]) is depth-complete -- metadata,
//! frontend-audio, tensor binding, and tokenizer admission all fail closed at
//! pack verification.

pub(crate) mod capacity;
mod decode_budget;
pub(crate) mod decoder_graph;
pub(crate) mod decoder_weights;
pub(crate) mod encoder_graph;
pub(crate) mod encoder_weights;
pub(crate) mod executor;
pub(crate) mod frontend;
pub(crate) mod graph_config;
pub mod package_import;
pub(crate) mod runtime_contract;
pub(crate) mod tokenizer;

pub use package_import::{
    FireRedAedImportRequest, FireRedAedImportResult, FireRedAedQuantizationMode,
    convert_local_firered_aed_source_to_runtime_pack,
};
