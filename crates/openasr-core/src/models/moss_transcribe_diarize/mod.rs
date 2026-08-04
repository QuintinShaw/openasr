//! MOSS-Transcribe-Diarize (`OpenMOSS/MOSS-Transcribe-Diarize`, 0.9B) model
//! family: a joint transcription + speaker-diarization ASR model built from
//! a Whisper-Medium-architecture audio encoder, a small `VQAdaptor` bridge
//! (a plain 3-layer MLP+LayerNorm despite the "VQ" name -- there is no
//! vector-quantization codebook in this checkpoint), and a Qwen3-0.6B
//! decoder. `[S01]`-style speaker labels and inline timestamps are ordinary
//! BPE tokens the Qwen3 decoder emits freely as part of its transcript text;
//! [`speaker_segments`] parses that markup, fail-closed, into the shared
//! `Segment` speaker-turn shape so `verbose_json`/SRT/VTT get real per-speaker
//! segments (see its module doc for the grammar and the fail-closed policy).
//! The executor's top-level `Transcription::text` stays the raw, tag-included
//! decode -- unlike cohere, whose diarization markers are non-printing
//! special tokens, moss-td's tags are literal characters, so stripping them
//! from the plain/CLI text output would rewrite what the model actually said.
//!
//! Current status: the checkpoint-to-GGUF importer ([`package_import`]) and
//! the full ggml execution graph (Whisper encoder reuse via [`encoder_graph`],
//! the [`adaptor_graph`] bridge, Qwen3 decoder reuse via [`llm_decoder`], and
//! the dedicated executor in [`executor`]) are implemented. The family's
//! complete lifecycle row -- identity, pack, execution, topology,
//! optimization, quantization, and conformance facets -- is declared in the
//! canonical architecture inventory (`arch/mod.rs`); offline/streaming
//! dispatch, executor materialization, runtime-validator routing, and
//! content-id eviction are generated projections of that row, not
//! family-specific central wiring. A pack produced by this importer runs
//! through `openasr transcribe --model-pack <pack>`, and `openasr model-pack
//! import moss` dispatches the importer. Public catalog coverage is live in
//! `model-registry/catalog.public.json` with the three published quantization
//! tiers: `fp16`, `q8_0`, and `q4_k`.
//!
//! Auto backend selection follows `AutoGpuPolicy::AllBackends`, so Auto may
//! resolve to Metal on Apple Silicon and explicit accelerated requests use the
//! same registered family path. End-to-end text/speaker/timestamp checks still
//! require a local real pack and reference fixtures, so those `#[ignore]` tests
//! remain outside weight-free CI; this module makes no performance claim.

mod adaptor_graph;
pub(crate) mod capacity;
mod decode_budget;
mod decode_prompt;
mod encoder_graph;
pub(crate) mod executor;
mod graph_config;
mod llm_decoder;
pub(crate) mod package_import;
mod prepared_runtime;
mod prompt_embedding;
pub(crate) mod runtime_contract;
pub(crate) mod speaker_segments;
pub(crate) mod tensor_names;
mod tokenizer;

// Re-exported for the `openasr model-pack import moss` dispatch and crate-level
// consumers. The converter is also covered by fixture-gated unit tests, while
// tests that require a local real pack remain outside weight-free CI.
#[allow(unused_imports)]
pub use package_import::{
    MossTdImportRequest, MossTdImportResult, MossTdQuantizationMode,
    convert_local_moss_transcribe_diarize_source_to_runtime_pack,
};
