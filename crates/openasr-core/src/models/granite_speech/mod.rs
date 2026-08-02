//! IBM Granite Speech 4.1 (`ibm-granite/granite-speech-4.1-2b`) model family.
//!
//! Architecture (verified against the HF `GraniteSpeechForConditionalGeneration`
//! source and cross-checked against upstream llama.cpp's `granite-speech.cpp`
//! ggml graph, since llama.cpp is a reference implementation only, not an
//! OpenASR upstream):
//!   16-layer Conformer CTC encoder (Shaw relative-position self-attention,
//!   4-second block-local attention windows, GLU + depthwise-conv module,
//!   self-conditioned CTC tap after layer 8) -> BLIP-2 Q-Former projector
//!   (window_size=15, downsample_rate=5, 2 cross-attention layers) -> Granite
//!   dense decoder-only LLM (GQA + RoPE + SwiGLU, with the four Granite scaling
//!   scalars: attention/embedding/residual multipliers + logits scaling).
//!
//! This pass ships the encoder + projector numeric core only (validated by the
//! `parity` dev harness against an HF `transformers` fp32 reference) and the
//! safetensors -> `.oasr` converter for all three weight segments (encoder,
//! projector, decoder). The decoder ggml graph, shared greedy-decode-driver
//! registration, and end-to-end golden are a separate follow-up pass -- see
//! `docs` note in `encoder_graph.rs` on the long-audio context-window bound.

pub(crate) mod capacity;
pub(crate) mod decode_executor;
pub(crate) mod decode_session;
pub(crate) mod decoder_graph;
pub(crate) mod encoder_graph;
pub(crate) mod executor;
pub(crate) mod frontend;
pub mod package_import;
pub(crate) mod prompt;
pub(crate) mod qformer;
pub(crate) mod runtime_contract;
pub(crate) mod runtime_provider;
pub(crate) mod tokenizer;

#[cfg(test)]
mod parity;

// Force-linked pack-import surface: the architecture integration descriptor
// names this convert symbol, and `models::pack_import_surface` proves it stays
// linked. Re-exported at the module root to match every other family's shape.
pub use package_import::convert_local_granite_speech_source_to_runtime_pack;

// Model-family + architecture ids live in `crate::arch` (see
// `GRANITE_SPEECH_MODEL_FAMILY`/`GRANITE_SPEECH_GGML_ARCHITECTURE_ID` there),
// the single source of truth every builtin registry resolves against.
