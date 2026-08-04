//! IBM Granite Speech 4.1 (`ibm-granite/granite-speech-4.1-2b`) model family.
//!
//! Architecture (verified against the HF `GraniteSpeechForConditionalGeneration`
//! source; llama.cpp's `granite-speech.cpp` ggml graph was a cross-check
//! reference only, never an OpenASR upstream):
//!   16-layer Conformer CTC encoder (Shaw relative-position self-attention,
//!   4-second block-local attention windows, GLU + depthwise-conv module,
//!   self-conditioned CTC tap after layer 8) -> BLIP-2 Q-Former projector
//!   (window_size=15, downsample_rate=5, 2 cross-attention layers) -> Granite
//!   dense decoder-only LLM (GQA + RoPE + SwiGLU, with the four Granite scaling
//!   scalars: attention/embedding/residual multipliers + logits scaling).
//!
//! Current status: the safetensors -> `.oasr` converter ([`package_import`])
//! carries all three weight segments, and the full execution path is
//! implemented: [`encoder_graph`] + [`qformer`] numeric cores (validated by
//! the `parity` dev harness against an HF `transformers` fp32 reference), the
//! keep-quantized resident decoder session ([`decode_session`]), and the
//! dedicated executor in [`executor`]. Greedy decode rides the one shared
//! seq2seq driver through the `GRANITE_SPEECH_DECODE_POLICY_ID` policy
//! descriptor (no family-local argmax loop), and cancellation stays on the
//! shared request control end to end: the driver polls it at every token
//! boundary, and the pinned-runtime actor pool republishes the request's
//! graph-abort flag on the owner thread so the encoder, projector, prefill,
//! and step graphs observe it under the shared L2 fence. The family's
//! complete lifecycle row -- identity, pack, execution, topology,
//! optimization, quantization, and conformance facets -- is declared in the
//! canonical architecture inventory (`arch/mod.rs`); offline/streaming
//! dispatch, executor materialization, runtime-validator routing, and
//! content-id eviction are generated projections of that row, not
//! family-specific central wiring. The runtime validator
//! ([`runtime_contract`]) proves the three-stage metadata, the full 938-tensor
//! set, and the packed tokenizer at pack admission.
//!
//! Honest gaps, tracked outside weight-free CI: the importer is fp16-only
//! and is reachable only through its force-linked CoreConvert symbol (no
//! `openasr model-pack import granite-speech` subcommand yet -- the publish
//! tooling records this), the encoder/projector still bind through the
//! host-f32 loader in [`runtime_provider`] (their keep-quantized migration is
//! a follow-up; the decoder already binds zero-copy), streaming re-runs the
//! whole offline pipeline per partial (correctness-only cadence), and the
//! end-to-end golden tests need a local real pack, so they stay `#[ignore]`d.
//! This module makes no performance claim.

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

// Force-linked pack-import surface: the architecture registry descriptor
// names this convert symbol, and `models::pack_import_surface` proves it stays
// linked. Re-exported at the module root to match every other family's shape.
pub use package_import::convert_local_granite_speech_source_to_runtime_pack;

// Model-family + architecture ids live in `crate::arch` (see
// `GRANITE_SPEECH_MODEL_FAMILY`/`GRANITE_SPEECH_GGML_ARCHITECTURE_ID` there),
// the single source of truth every builtin registry resolves against.
